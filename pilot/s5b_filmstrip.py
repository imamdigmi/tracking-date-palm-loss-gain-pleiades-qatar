"""S5b — paket review manusia BUTA (filmstrip) dari kandidat S5a: 70 item x 20 epoch.

Sampel (seed 0) dari `pilot/out/s5/candidates.parquet`: 40 loss + 20 gain, hanya `persistent` dan `nb_rate >= 0,6`
(tetangga stabil tetap terlihat -> absen lebih kredibel); maks 3 per farm dan maks 6 per interval (t_before, t_after),
acak dalam pool. Bila cap membuat satu tipe kurang: cap farm dilonggarkan dulu, lalu cap interval (dicatat di report.md).
Ditambah 10 negatif (5 stable + 5 flicker) tanpa cap. Urutan diacak (seed 0), id review R001..R070.
Crop 48 x 48 m berpusat di pusat site (frame referensi) digeser ke frame tiap epoch dengan shift S1 seperti
`s4_sites.observable_flags`: x + dx_px*0,5 ; y - dy_px*0,5. Semua epoch di-resample ke 192 x 192 px (0,25 m/px) supaya
epoch 0,3 m dan 0,5 m tampil di skala tanah yang sama. Kontras: persentil 2-98 per crop (bersama untuk 3 band);
cincin tipis radius 4 m menandai lokasi tanpa menutup pusat tajuk. Piksel di luar raster/nodata = hitam.
HTML hanya memuat id review, farm, dan epoch: TANPA tipe kandidat, skor, atau state detektor (review buta).
`key.parquet` (kunci jawaban, JANGAN dibuka pemilik sebelum selesai) = review_id + semua kolom kandidat.

Jalankan (cwd = repo): python pilot/s5b_filmstrip.py [--data-root "<.../Thesis/datasets>"] [--out pilot/out/s5b]
"""
import argparse
import html
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import s4_sites as s4

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
except ImportError:
    rasterio = None

HERE = Path(__file__).resolve().parent
SEED = 0
N_PER_TYPE = {"loss": 40, "gain": 20}
N_NEG = 5
NB_MIN = 0.6
CAPS = [(3, 6), (np.inf, 6), (np.inf, np.inf)]       # (maks per farm, maks per interval); longgarkan dari kiri ke kanan
CAP_NAMES = ["tidak dilonggarkan", "cap farm dilonggarkan", "cap farm dan interval dilonggarkan"]
SIZE_M, PX = 48.0, 192
RING_M = 4.0
COLS = 5


# ---------- sampling ----------
def pick(pool, n, seed=SEED):
    """n baris acak dari pool dengan cap farm/interval; cap dilonggarkan bertahap. Return (baris, level CAPS)."""
    if len(pool) < n:
        sys.exit(f"ERROR: pool {len(pool)} < {n} yang diminta")
    pool = pool.sample(frac=1, random_state=seed)
    take, nf, ni = [], Counter(), Counter()
    for lvl, (fc, ic) in enumerate(CAPS):
        for t in pool.itertuples():
            if len(take) == n:
                break
            iv = (t.t_before, t.t_after)
            if t.Index not in take and nf[t.farm] < fc and ni[iv] < ic:
                take.append(t.Index)
                nf[t.farm] += 1
                ni[iv] += 1
        if len(take) == n:
            break
    return pool.loc[take], lvl


def sample(cands, seed=SEED, n_per_type=N_PER_TYPE, tier=None, exclude=None, prefix="R"):
    """Return (sampel dengan review_id acak R001.., {tipe: level pelonggaran}).
    tier: hanya best_tier ini (vault 017: hanya high layak dipercaya); exclude: key lama, site+tipe yang sudah direview dibuang."""
    if tier:
        cands = cands[cands.best_tier == tier]
    if exclude is not None:
        cands = cands[~cands.set_index(["farm", "site_id", "type"]).index.isin(exclude.set_index(["farm", "site_id", "type"]).index)]
    ok = cands[cands.persistent.fillna(False).astype(bool) & (cands.nb_rate >= NB_MIN)]
    parts, relax = [], {}
    for t, n in n_per_type.items():
        p, relax[t] = pick(ok[ok.type == t], n, seed)
        parts.append(p)
    for t in ("stable", "flicker"):
        neg = cands[cands.type == t]
        if len(neg) < N_NEG:
            sys.exit(f"ERROR: negatif {t} {len(neg)} < {N_NEG}")
        parts.append(neg.sample(N_NEG, random_state=seed))
    s = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    s.insert(0, "review_id", [f"{prefix}{i + 1:03d}" for i in range(len(s))])
    return s, relax


# ---------- crop ----------
def crop(src, cx, cy, nodata):
    """Crop SIZE_M x SIZE_M berpusat (cx, cy) frame raster -> uint8 (3, PX, PX); kontras 2-98, cincin RING_M."""
    win = from_bounds(cx - SIZE_M / 2, cy - SIZE_M / 2, cx + SIZE_M / 2, cy + SIZE_M / 2, src.transform)
    v = src.read([1, 2, 3], window=win, out_shape=(3, PX, PX), boundless=True, fill_value=nodata,
                 resampling=Resampling.bilinear).astype(float)
    valid = ~np.all(v == nodata, axis=0)
    img = np.zeros(v.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(v[:, valid], [2, 98])
        img = np.where(valid, np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1) * 255, 0).astype(np.uint8)
    d = np.hypot(*(np.indices((PX, PX)) + 0.5 - PX / 2))
    img[:, np.abs(d - RING_M / SIZE_M * PX) < 0.75] = np.array([[255], [255], [0]], np.uint8)
    return img


def write_png(path, img):
    with rasterio.open(path, "w", driver="PNG", height=PX, width=PX, count=3, dtype="uint8") as d:
        d.write(img)


# ---------- HTML ----------
def make_html(items, epochs):
    """items: sampel (review_id, farm); epochs: {farm: [epoch]}. Tanpa tipe/skor/state."""
    h = ["<!doctype html><html lang=id><meta charset=utf-8><title>Review pohon kurma</title>",
         "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}"
         f".g{{display:grid;grid-template-columns:repeat({COLS},1fr);gap:4px}}"
         "figure{margin:0}img{width:100%;display:block}figcaption{font-size:12px;text-align:center}</style>",
         "<h1>Review filmstrip pohon kurma</h1>",
         "<p>Tiap bagian = satu lokasi; 20 gambar = 20 epoch (pemotretan satelit) berurutan waktu. Nilai HANYA tajuk kurma "
         "yang pusatnya paling dekat ke pusat cincin kuning; tajuk tetangga di tepi cincin diabaikan. Per gambar putuskan: "
         "ada / tidak ada / tidak jelas (buram, kuning, gelap -> lewati). Dari urutan ada/tidak ada pilih verdict. "
         "Pusat cincin jatuh di antara dua tajuk -> ambigu, note 'meleset'. "
         "Isi <code>review.csv</code> (kolom <code>verdict</code>, <code>first_change_epoch</code>, <code>second_change_epoch</code>, <code>note</code>) "
         "per review_id. Abaikan epoch yang citranya tidak jelas.</p><ul>"
         "<li><b>ada</b> = pohon ada di semua epoch yang jelas.</li>"
         "<li><b>hilang</b> = ada lalu hilang; isi first_change_epoch = epoch pertama pohon tidak ada.</li>"
         "<li><b>muncul</b> = tidak ada lalu muncul; isi first_change_epoch = epoch pertama pohon ada.</li>"
         "<li><b>muncul_hilang</b> = tidak ada, lalu ada, lalu hilang lagi; first_change_epoch = epoch pertama ada, "
         "second_change_epoch = epoch pertama hilang lagi.</li>"
         "<li><b>tidak_ada</b> = tidak pernah ada pohon kurma.</li>"
         "<li><b>ambigu</b> = tidak bisa diputuskan.</li></ul>"
         "<p>Aturan tambahan (vault 017): nilai tiap epoch dari gambar saja, jangan menebak dari epoch lain. "
         "Pohon dihitung <b>ada</b> bila tajuk + bayangan jelas dan diameter tajuk &ge; 2,5 m (&asymp; 10 px di gambar ini, "
         "cincin kuning berdiameter 8 m). <b>muncul</b> hanya bila ada di &ge; 2 epoch jelas berurutan. Tajuk 1,5&ndash;2,5 m = "
         "<b>ambigu</b>. Cincin di celah antar pohon: ambigu, note 'celah'. Greenhouse/bukan kurma: tidak_ada, note.</p>"]
    for r in items.itertuples():
        h.append(f'<h2 id="{r.review_id}">{r.review_id} &mdash; farm {html.escape(str(r.farm))}</h2><div class=g>')
        for e in epochs[r.farm]:
            h.append(f'<figure><img src="img/{r.review_id}_{e}.png" loading=lazy alt="{e}"><figcaption>{e}</figcaption></figure>')
        h.append("</div>")
    return "\n".join(h) + "\n"


# ---------- laporan ----------
def make_report(s, relax, fails, elapsed, n_img):
    L = ["# S5b filmstrip review buta", "",
         f"Waktu jalan: {elapsed:.1f} s; {len(s)} item, {n_img} PNG, epoch gagal dibaca: {len(fails)}.", "",
         f"Pool: persistent dan nb_rate >= {NB_MIN}; seed {SEED}; cap awal {CAPS[0][0]}/farm, {CAPS[0][1]}/interval.", "",
         "## Pelonggaran cap"]
    L += [f"- {t}: {CAP_NAMES[lvl]}" for t, lvl in relax.items()]
    L += ["", "## Jumlah per tipe", "| tipe | n | farm |", "|---|---|---|"]
    L += [f"| {t} | {len(g)} | {g.farm.nunique()} |" for t, g in s.groupby("type")]
    for name, keys in (("farm", ["farm"]), ("interval (t_before -> t_after)", ["t_before", "t_after"])):
        L += ["", f"## Per {name}", "| " + " | ".join(keys) + " | loss | gain | stable | flicker |", "|" + "---|" * (len(keys) + 4)]
        c = s.groupby(keys + ["type"]).size().unstack(fill_value=0).reindex(columns=["loss", "gain", "stable", "flicker"], fill_value=0)
        L += ["| " + " | ".join(map(str, np.atleast_1d(k))) + " | " + " | ".join(map(str, r)) + " |" for k, r in zip(c.index, c.to_numpy())]
    L += ["", "## Epoch gagal"] + ([f"- {a} farm {f} {e}: {m}" for a, f, e, m in fails] or ["- tidak ada"])
    return "\n".join(L) + "\n"


# ---------- driver ----------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=s4.DEFAULT_DATA_ROOT)
    ap.add_argument("--cands", default=str(HERE / "out" / "s5" / "candidates.parquet"))
    ap.add_argument("--summary", default=str(HERE / "out" / "s1" / "registration_summary.parquet"))
    ap.add_argument("--manifest", default=str(HERE / "out" / "s0" / "manifest.parquet"))
    ap.add_argument("--out", default=str(HERE / "out" / "s5b"))
    ap.add_argument("--tier", default=None, help="hanya best_tier ini, mis. high (batch 2)")
    ap.add_argument("--n-loss", type=int, default=N_PER_TYPE["loss"])
    ap.add_argument("--n-gain", type=int, default=N_PER_TYPE["gain"])
    ap.add_argument("--exclude", nargs="*", default=[], help="key.parquet batch lama; site+tipe di sana tidak diambil lagi")
    ap.add_argument("--prefix", default="R", help="awalan review_id, mis. Q untuk batch 2 (hindari bentrok R001)")
    a = ap.parse_args(argv)
    if rasterio is None:
        sys.exit("ERROR: paket 'rasterio' belum terpasang. Install: pip install rasterio")
    for p in (a.cands, a.summary, a.manifest):
        if not Path(p).is_file():
            sys.exit(f"ERROR: tidak ada {p}")
    if not Path(a.data_root).is_dir():
        sys.exit(f"ERROR: folder tidak ada: {a.data_root}")
    t0 = time.time()
    cands = pd.read_parquet(a.cands)
    excl = pd.concat([pd.read_parquet(p) for p in a.exclude]) if a.exclude else None
    s, relax = sample(cands, n_per_type={"loss": a.n_loss, "gain": a.n_gain}, tier=a.tier, exclude=excl, prefix=a.prefix)
    man = pd.read_parquet(a.manifest).astype({"farm": str})
    man = man[man.farm.isin(s.farm)]
    meta = s4.epoch_meta(man, pd.read_parquet(a.summary).astype({"farm": str}), pd.Series(dtype=int)).set_index(["farm", "epoch"])
    epochs = {f: sorted(g.epoch) for f, g in man.groupby("farm")}
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    fails, n_img = [], 0
    for (f, e), m in meta.iterrows():
        d = Path(a.data_root) / m.service / f
        try:
            src = rasterio.open(d / f"{f}_{m.service}.tif")
        except Exception as ex:                 # clip hilang/rusak: catat per item, lanjut
            fails += [("buka", f, e, str(ex)) for _ in range(len(s[s.farm == f]))]
            continue
        with src:
            for r in s[s.farm == f].itertuples():
                try:
                    write_png(out / "img" / f"{r.review_id}_{e}.png",
                              crop(src, r.x + m.dx_px * s4.PX_M, r.y - m.dy_px * s4.PX_M, m.nodata_used))
                    n_img += 1
                except Exception as ex:
                    fails.append((r.review_id, f, e, str(ex)))
        print(f"farm {f} {e} {time.time() - t0:.0f}s", file=sys.stderr)
    (out / "index.html").write_text(make_html(s, epochs))
    pd.DataFrame({"review_id": s.review_id, "farm": s.farm, "verdict": "", "first_change_epoch": "", "second_change_epoch": "",
                  "note": ""}).to_csv(out / "review.csv", index=False)
    s.to_parquet(out / "key.parquet", index=False)
    elapsed = time.time() - t0
    (out / "report.md").write_text(make_report(s, relax, fails, elapsed, n_img))
    print(f"S5b selesai {elapsed:.1f}s: {len(s)} item, {n_img} PNG, {len(fails)} gagal -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
