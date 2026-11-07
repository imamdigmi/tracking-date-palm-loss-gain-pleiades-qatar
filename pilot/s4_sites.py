"""S4 — pembentukan site + matriks observasi (site x epoch) dari consensus deteksi run_002.

Semua tier masuk (tier & score tetap kolom, tanpa filter). Koordinat dikoreksi ke frame epoch referensi dengan
hasil S1 sebelum clustering: x_ref = x - dx_px*0,5 ; y_ref = y + dy_px*0,5 (dx>0 = isi epoch bergeser timur,
dy>0 = bergeser selatan; y proyeksi naik ke utara). Epoch referensi (tanpa baris di summary S1) = tanpa koreksi, reg_ok=True;
tak terukur -> tanpa koreksi, reg_ok=False; flag merah -> tetap dikoreksi, reg_ok=False.
Site = clustering leader per farm (r dari gates.yaml `radius_site_m`): titik diproses dari skor tertinggi; titik masuk
ke site yang PUSATNYA (= titik pertama/skor tertinggi, tetap) paling dekat dalam r dan belum punya titik di epoch itu,
kalau tidak ada membuka site baru. Jadi tidak ada rantai (komponen terhubung/DBSCAN min_samples=1 merantai pohon
bertetangga sampai 208 m, lihat vault 015) dan maksimal 1 titik per epoch per site (`n_dup_epochs` selalu 0, cek
invarian). Lalu pecahan digabung: algoritma yang sama di level site (urut jumlah epoch terbanyak, radius MERGE_M=4 m
~ setengah jarak tanam 6,7 m), hanya jika himpunan epoch tidak bertabrakan -- 16-21% "absen" site stabil ternyata
titiknya jatuh ke site pecahan di sebelahnya (vault 015 bagian 7). Posisi = median anggota; `extent_m` = jarak
anggota terjauh ke posisi itu.
obs = 1 baris per (site, epoch) untuk semua epoch farm di manifest. `observable` = pusat site di dalam kotak window 0
DAN di dalam batas clip epoch DAN mask==1 DAN piksel clip bukan nodata (semua band == nodata_used). Clip disampel di
frame epoch (pusat site digeser balik dengan shift S1); mask = poligon farm (geografis) disampel di frame referensi.
Baris tak-observable TIDAK berarti absen: score/detected tetap apa adanya, hanya flag. Window 1 tidak pernah dijalankan -> diabaikan.
q = metadata saja (tidak melihat isi citra dekat pohon): gsd, bands, nodata_pct, reg_resid_m, reg_ok, dt_days.
`fe_n_det` (jumlah titik consensus farm-epoch) BUKAN q (bergantung deteksi): hanya diagnostik untuk membedakan
"pohon absen" dari "detektor buta di epoch itu" (fe_n_det = 0 padahal citra ada).

Paket: numpy, pandas, scipy, rasterio, pyarrow, pyyaml (sudah ada di .venv proyek).
Jalankan (cwd = repo, bukan Drive): python pilot/s4_sites.py [--data-root "<.../Thesis/datasets>"] [--run "<.../pilot/run_002>"] \\
          [--radius 2.5] [--out pilot/out/s4]
"""
import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data/datasets")
DEFAULT_RUN = str(Path(DEFAULT_DATA_ROOT).parent / "pilot" / "run_002")
PX_M = 0.5                      # satuan px S1 (grid pengukuran 0,5 m)
RADII = [1.5, 2.0, 2.5, 3.0, 4.0]
MERGE_M = 4.0                   # radius gabung pecahan (pusat ke pusat)
TIER_RANK = {"low": 0, "med": 1, "high": 2}
TIER_NAME = {v: k for k, v in TIER_RANK.items()}
KEY = ["farm", "site"]


# ---------- koreksi & clustering ----------
def correct(x, y, dx_px, dy_px):
    """Titik di frame epoch -> frame referensi."""
    return x - dx_px * PX_M, y + dy_px * PX_M


def cluster(xy, ep, score, r):
    """Leader clustering: label site per item. Pusat site = item pembuka (skor tertinggi), tidak bergeser.
    Item -> site terdekat (jarak pusat <= r) yang belum berisi epoch-nya; kalau tidak ada -> site baru.
    ep: epoch per item (titik) atau set epoch per item (site, untuk gabung pecahan)."""
    xs, ys = xy[:, 0].tolist(), xy[:, 1].tolist()
    ep = [e if isinstance(e, (set, frozenset)) else {e} for e in ep]
    lab = np.empty(len(xs), int)
    cx, cy, eps, grid = [], [], [], {}      # grid sel r x r -> cukup cek 3x3 sel tetangga
    for i in np.argsort(-np.asarray(score), kind="stable").tolist():
        x, y, e = xs[i], ys[i], ep[i]
        gx, gy = int(x // r), int(y // r)
        best, bd = -1, r
        for k in ((gx + a, gy + b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
            for s in grid.get(k, ()):
                d = ((x - cx[s]) ** 2 + (y - cy[s]) ** 2) ** 0.5
                if d <= bd and eps[s].isdisjoint(e):
                    best, bd = s, d
        if best < 0:
            best = len(cx)
            cx.append(x), cy.append(y), eps.append(set())
            grid.setdefault((gx, gy), []).append(best)
        eps[best] |= e
        lab[i] = best
    return lab


def build_sites(pts, r, merge=MERGE_M):
    """pts: farm,x,y,epoch,score,tier (x,y sudah dikoreksi). Return (pts + kolom site, sites)."""
    pts = pts.reset_index(drop=True)
    lab = np.empty(len(pts), int)
    for idx in pts.groupby("farm").indices.values():
        g = pts.iloc[idx]
        lab0 = cluster(g[["x", "y"]].to_numpy(), g.epoch.to_numpy(), g.score.to_numpy(), r)
        s = g.groupby(lab0).agg(x=("x", "median"), y=("y", "median"), ep=("epoch", set))
        s["n"] = s.ep.map(len)
        lab[idx] = cluster(s[["x", "y"]].to_numpy(), s.ep.tolist(), s.n.to_numpy(), merge)[s.index.get_indexer(lab0)]
    pts["site"] = lab
    pts["tier_i"] = pts.tier.map(TIER_RANK)
    sites = pts.groupby(KEY).agg(x=("x", "median"), y=("y", "median"), n_pts=("x", "size"),
                                 n_epochs_det=("epoch", "nunique"), best=("tier_i", "max"),
                                 first_epoch=("epoch", "min"), last_epoch=("epoch", "max"))
    c = pts.join(sites[["x", "y"]], on=KEY, rsuffix="_c")
    sites["extent_m"] = np.hypot(c.x - c.x_c, c.y - c.y_c).groupby([c.farm, c.site]).max()
    dup = pts.groupby(KEY + ["epoch"]).size().gt(1).groupby(KEY).sum()
    sites["n_dup_epochs"] = dup.reindex(sites.index).astype(int)
    sites["best_tier"] = sites.pop("best").map(TIER_NAME)
    sites = sites.reset_index()
    sites.insert(1, "site_id", sites.farm + "_" + sites.site.astype(str))
    return pts, sites


# ---------- meta farm-epoch (q) ----------
def epoch_meta(man, summ, counts):
    """man: manifest farm-target; summ: registration_summary; counts: Series (farm, epoch) -> fe_n_det."""
    m = man.merge(summ, on=["farm", "epoch"], how="left", indicator=True)
    ref = m._merge == "left_only"        # epoch referensi tidak punya baris S1
    bad = m.red_flag.eq(True) | m.unmeasured.eq(True)
    m["dx_px"] = m.median_dx_px.fillna(0.0)
    m["dy_px"] = m.median_dy_px.fillna(0.0)
    m["reg_resid_m"] = np.where(ref, 0.0, m.median_resid_m)
    m["reg_ok"] = ref | ~bad
    m["is_ref"] = ref
    d = pd.to_datetime(m.epoch + "_15", format="%Y_%m_%d")      # presisi bulan -> tanggal 15
    m = m.assign(_d=d).sort_values(["farm", "_d"])
    m["dt_days"] = m.groupby("farm")._d.diff().dt.days.astype(float)
    m["gsd"] = m.gsd_x
    m["fe_n_det"] = [int(counts.get((f, e), 0)) for f, e in zip(m.farm, m.epoch)]
    return m.drop(columns=["_d", "_merge"]).reset_index(drop=True)


# ---------- obs ----------
def make_obs(pts, sites, meta):
    """Satu baris (site, epoch) untuk semua epoch farm. observable diisi terpisah (add_observable)."""
    per = pts.sort_values("score").groupby(KEY + ["epoch"]).agg(
        n_pts=("score", "size"), score=("score", "max"), tier=("tier", "last"))   # tier = tier titik skor maks
    q = ["farm", "epoch", "gsd", "bands", "nodata_pct", "reg_resid_m", "reg_ok", "dt_days", "fe_n_det"]
    o = sites[KEY + ["site_id"]].merge(meta[q], on="farm").join(per, on=KEY + ["epoch"])
    o["detected"] = o.n_pts.notna()
    o["n_pts"] = o.n_pts.fillna(0).astype(int)
    o["score"] = o.score.fillna(0.0)
    return o


def sample(src, x, y):
    """Nilai raster (n, band) di (x,y) dunia + flag di dalam raster. Satu pembacaan windowed."""
    t = src.transform
    col = np.floor((x - t.c) / t.a).astype(int)
    row = np.floor((t.f - y) / -t.e).astype(int)
    ins = (col >= 0) & (col < src.width) & (row >= 0) & (row < src.height)
    out = np.zeros((len(x), src.count), src.dtypes[0])
    if ins.any():
        r0, r1, c0, c1 = row[ins].min(), row[ins].max(), col[ins].min(), col[ins].max()
        d = src.read(window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
        out[ins] = d[:, row[ins] - r0, col[ins] - c0].T
    return out, ins


def observable_flags(clip, msk, nodata, x, y, dx_px, dy_px, box):
    """x,y: pusat site (frame referensi). clip/msk: dataset terbuka (msk None = tanpa mask)."""
    ok = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
    if not ok.any():
        return ok
    xi, yi = x[ok], y[ok]
    vals, ins = sample(clip, xi + dx_px * PX_M, yi - dy_px * PX_M)     # balik koreksi: frame epoch
    good = ins & ~np.all(vals == nodata, axis=1)
    if msk is not None:
        mv, mins = sample(msk, xi, yi)
        good &= mins & (mv[:, 0] == 1)
    ok[np.flatnonzero(ok)] = good
    return ok


def add_observable(obs, meta, windows, root):
    flag = np.zeros(len(obs), bool)
    X, Y = obs.x.to_numpy(), obs.y.to_numpy()
    for (farm, epoch), idx in obs.groupby(["farm", "epoch"]).indices.items():
        m = meta[(meta.farm == farm) & (meta.epoch == epoch)].iloc[0]
        if not m.has_clip:
            continue
        d = Path(root) / m.service / farm
        clip = rasterio.open(d / f"{farm}_{m.service}.tif")
        msk = rasterio.open(d / f"{farm}_{m.service}_mask.tif") if m.has_mask else None
        try:
            flag[idx] = observable_flags(clip, msk, m.nodata_used, X[idx], Y[idx], m.dx_px, m.dy_px, windows[farm][0])
        finally:
            clip.close()
            if msk is not None:
                msk.close()
    obs["observable"] = flag
    return obs


# ---------- sensitivitas & laporan ----------
def sensitivity(pts, radii, r0, merges=(3.0, 5.0)):
    """Variasi r (MERGE_M tetap) + variasi MERGE_M (r = r0): setelah gabung, MERGE_M yang dominan."""
    rows = []
    for r, mg in [(r, MERGE_M) for r in radii] + [(r0, m) for m in merges]:
        s = build_sites(pts, r, mg)[1]
        rows.append(dict(r=r, merge=mg, n_sites=len(s), n_ge10=int((s.n_epochs_det >= 10).sum()), only_1_epoch=int((s.n_epochs_det == 1).sum()),
                         median_epochs=float(s.n_epochs_det.median()), dup_share=float((s.n_dup_epochs > 0).mean()),
                         extent_med=float(s.extent_m.median()), extent_p95=float(s.extent_m.quantile(0.95))))
    return pd.DataFrame(rows)


def make_report(sites, obs, meta, sens, skipped, elapsed, args, r):
    ob = obs[obs.observable]
    L = ["# S4 site + matriks observasi", "",
         f"Waktu jalan: {elapsed:.1f} s; radius site r = {r:g} m (gates.yaml `radius_site_m`, bisa di-override `--radius`).", "",
         "## Input",
         f"- consensus: `{Path(args.run) / 'consensus.parquet'}`; windows: `{Path(args.run) / 'windows.json'}` (hanya window 0)",
         f"- registrasi S1: `{args.summary}`; manifest S0: `{args.manifest}`; data-root: `{args.data_root}`", "",
         "## Jumlah",
         f"- Farm dipakai: {sites.farm.nunique()}; **site: {len(sites)}**; **baris obs: {len(obs)}**; "
         f"titik consensus masuk: {int(sites.n_pts.sum())}.",
         f"- Observable: {obs.observable.mean():.4f} dari baris obs; terdeteksi di antara observable: {ob.detected.mean():.3f}.",
         f"- Farm dilewati (tanpa window): {', '.join(skipped) if skipped else '-'}.",
         f"- Site: median epoch terdeteksi {sites.n_epochs_det.median():g}; extent_m median {sites.extent_m.median():.2f}, "
         f"p95 {sites.extent_m.quantile(0.95):.2f}, maks {sites.extent_m.max():.2f}; "
         f"site dengan n_dup_epochs>0: {(sites.n_dup_epochs > 0).mean():.3f}.", "",
         "## Per epoch", "| epoch | observable | terdeteksi/observable | farm-epoch fe_n_det=0 |", "|---|---|---|---|"]
    for e, g in obs.groupby("epoch"):
        z = meta[(meta.epoch == e) & (meta.fe_n_det == 0)]
        L.append(f"| {e} | {g.observable.mean():.4f} | {g[g.observable].detected.mean():.3f} | {len(z)} |")
    fe = obs.groupby(["farm", "epoch"]).observable.mean()
    low = fe[fe < 0.5]
    L += ["", f"## Farm-epoch dengan observable < 0,5: {len(low)} dari {len(fe)}"]
    L += [f"- {f} {e}: {v:.3f}" for (f, e), v in low.sort_values().head(30).items()]
    L += ["", "## Sensitivitas radius site (r) dan radius gabung (MERGE_M)",
          "| r (m) | MERGE_M (m) | n_sites | site >=10 epoch | terdeteksi 1 epoch | median epoch terdeteksi | share n_dup_epochs>0 | extent median | extent p95 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in sens.itertuples():
        L.append(f"| {t.r:g} | {t.merge:g} | {t.n_sites} | {t.n_ge10} | {t.only_1_epoch} | {t.median_epochs:g} | {t.dup_share:.3f} | {t.extent_med:.2f} | {t.extent_p95:.2f} |")
    L += ["", "## Extent terbesar (indikasi rantai)", "| site | extent m | n_pts | n_epochs_det | n_dup_epochs |", "|---|---|---|---|---|"]
    for t in sites.sort_values("extent_m", ascending=False).head(5).itertuples():
        L.append(f"| {t.site_id} | {t.extent_m:.2f} | {t.n_pts} | {t.n_epochs_det} | {t.n_dup_epochs} |")
    L += ["", "## Catatan",
          "- Epoch 2024_10 (0,3 m): tier hanya dari native 0,3 m tanpa varian upsample -> komposisi tier tidak sebanding dengan epoch lain.",
          "- Site = lokasi kandidat, bukan pohon stabil terverifikasi. Tier & score tidak difilter.",
          "- `observable=False` = bukan absen (di luar window/clip/mask atau piksel nodata); baris tetap ada sebagai flag.",
          "- `fe_n_det=0` pada farm-epoch yang punya citra = detektor buta (atau farm tanpa pohon), bukan bukti pohon absen. Bukan bagian q.",
          "- Window 1 (6 farm) tidak pernah dijalankan: tidak ada site/obs di sana.",
          "- Mask disampel di frame referensi (poligon geografis); clip di frame epoch (shift S1 dibalik)."]
    return "\n".join(L) + "\n"


# ---------- driver ----------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--run", default=DEFAULT_RUN, help="folder run berisi consensus.parquet & windows.json")
    ap.add_argument("--summary", default=str(HERE / "out" / "s1" / "registration_summary.parquet"))
    ap.add_argument("--manifest", default=str(HERE / "out" / "s0" / "manifest.parquet"))
    ap.add_argument("--out", default=str(HERE / "out" / "s4"))
    ap.add_argument("--radius", type=float, help="default: radius_site_m di gates.yaml")
    a = ap.parse_args(argv)
    if rasterio is None:
        sys.exit("ERROR: paket 'rasterio' belum terpasang. Install: pip install rasterio")
    for p in (a.data_root, a.run):
        if not Path(p).is_dir():
            sys.exit(f"ERROR: folder tidak ada: {p}")
    r = a.radius or float(yaml.safe_load((HERE / "gates.yaml").read_text())["radius_site_m"])
    t0 = time.time()
    windows = json.loads((Path(a.run) / "windows.json").read_text())
    skipped = sorted(f for f, w in windows.items() if not w)
    windows = {f: w for f, w in windows.items() if w}
    man = pd.read_parquet(a.manifest).astype({"farm": str})
    man = man[man.farm.isin(windows)]
    cons = pd.read_parquet(Path(a.run) / "consensus.parquet").astype({"farm": str})
    cons = cons[cons.farm.isin(windows)]
    summ = pd.read_parquet(a.summary).astype({"farm": str})
    if set(windows) - set(man.farm) or cons.empty:
        sys.exit("ERROR: farm ber-window tidak ada di manifest / consensus kosong")
    if (man.groupby("farm").epoch.apply(lambda e: len(set(e) - set(summ[summ.farm == e.name].epoch)))
            != 1).any():
        sys.exit("ERROR: tiap farm harus punya tepat 1 epoch referensi (tanpa baris S1)")
    meta = epoch_meta(man, summ, cons.groupby(["farm", "epoch"]).size())
    # koreksi titik ke frame referensi
    k = cons.merge(meta[["farm", "epoch", "dx_px", "dy_px"]], on=["farm", "epoch"], how="left")
    if k.dx_px.isna().any():
        sys.exit("ERROR: titik consensus pada farm-epoch yang tidak ada di manifest")
    k["x"], k["y"] = correct(k.x, k.y, k.dx_px, k.dy_px)
    pts, sites = build_sites(k[["farm", "x", "y", "epoch", "score", "tier"]], r)
    print(f"{len(pts)} titik -> {len(sites)} site (r={r:g})", file=sys.stderr)
    obs = make_obs(pts, sites, meta)
    obs = obs.merge(sites[KEY + ["x", "y"]], on=KEY)
    obs = add_observable(obs, meta, windows, a.data_root)
    print(f"obs {len(obs)} baris, observable {obs.observable.mean():.3f}", file=sys.stderr)
    sens = sensitivity(pts, sorted(set(RADII + [r])), r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites[["farm", "site_id", "x", "y", "n_pts", "n_epochs_det", "n_dup_epochs", "extent_m", "best_tier",
           "first_epoch", "last_epoch"]].to_parquet(out / "sites.parquet", index=False)
    obs[["farm", "site_id", "epoch", "detected", "score", "tier", "n_pts", "observable", "gsd", "bands",
         "nodata_pct", "reg_resid_m", "reg_ok", "dt_days", "fe_n_det"]].to_parquet(out / "obs.parquet", index=False)
    elapsed = time.time() - t0
    (out / "report.md").write_text(make_report(sites, obs, meta, sens, skipped, elapsed, a, r))
    print(f"S4 selesai {elapsed:.1f}s: {len(sites)} site, {len(obs)} obs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
