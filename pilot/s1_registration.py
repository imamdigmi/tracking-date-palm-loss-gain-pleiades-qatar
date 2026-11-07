"""S1 — registrasi antar-epoch: offset tiap epoch terhadap epoch referensi per farm (phase correlation).

Tidak ada citra yang di-warp/ditulis. Pengukuran dilakukan pada grid 0,5 m (MEASURE_GSD) yang berjangkar di
pojok kiri-atas clip referensi; clip 0,3 m dibaca dengan resampling `average` HANYA untuk pengukuran.
gray = rata-rata band 1-3 (R,G,B; NIR diabaikan). Satuan "px" = piksel grid pengukuran 0,5 m.
Tanda: dx>0 = isi epoch tampak bergeser ke kanan (timur) relatif ke referensi; dy>0 = ke bawah (selatan).
Referensi: epoch 0,3 m bila ada, kalau tidak epoch dengan varian Laplacian (dalam mask) tertinggi.
Mask: dipakai hanya bila grid_match & biner (manifest S0); selain itu fallback ke piksel nodata raster
(semua band 1-3 == nodata). Subpiksel: upsampling matrix-DFT (numpy); skimage tidak dipakai.

Paket: numpy, pandas, rasterio, pyarrow  (tanpa pip install tambahan di .venv proyek).
Jalankan: python pilot/s1_registration.py --data-root "<.../Thesis/datasets>" --manifest pilot/out/s0/manifest.parquet \
          [--out pilot/out/s1] [--farms F003,F037]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
except ImportError:
    rasterio = None

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data/datasets")
MEASURE_GSD = 0.5
FINE_GSD = 0.4          # gsd < ini dianggap epoch 0,3 m
MIN_INSIDE = 0.8        # fraksi minimum piksel valid & di dalam mask per blok
RED_PX = 1.0            # flag merah bila median residual > 1 px
COLS = ["service", "epoch", "farm", "block_row", "block_col", "dx_px", "dy_px", "dx_m", "dy_m", "resp", "n_blocks"]


# ---------- phase correlation (numpy) ----------
def phase_corr(ref, mov, upsample=20):
    """Return (dx, dy, resp): mov(x) ~ ref(x - d). resp = puncak korelasi ternormalisasi (1 = sempurna)."""
    H, W = ref.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    a, b = (x - x.mean() for x in (ref, mov))
    R = np.fft.fft2(b * win) * np.conj(np.fft.fft2(a * win))
    R /= np.abs(R) + 1e-2 * np.abs(R).mean()  # regulariser: frekuensi tanpa energi (noise) tidak ikut memutih
    corr = np.fft.ifft2(R).real
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    resp = float(corr[iy, ix])
    sy, sx = (iy - H if iy > H // 2 else iy), (ix - W if ix > W // 2 else ix)
    # refine: DFT terupsample di sekitar puncak kasar (+-1,5 px)
    off = np.linspace(-1.5, 1.5, int(3 * upsample) + 1)
    ky, kx = np.fft.fftfreq(H) * H, np.fft.fftfreq(W) * W
    Ey = np.exp(2j * np.pi * (sy + off)[:, None] * ky[None, :] / H)
    Ex = np.exp(2j * np.pi * kx[:, None] * (sx + off)[None, :] / W)
    with np.errstate(all="ignore"):  # BLAS macOS memunculkan warning FP palsu pada matmul kompleks; hasil finite
        up = (Ey @ R @ Ex).real
    jy, jx = np.unravel_index(np.argmax(up), up.shape)
    return float(sx + off[jx]), float(sy + off[jy]), resp


# ---------- baca blok ----------
def read_block(src, x0, y0, size, nodata_default=0.0):
    """Blok size x size piksel 0,5 m dengan pojok kiri-atas (x0,y0) dunia. None bila keluar dari raster.
    return (gray float32, valid bool)."""
    gsd, span = abs(src.transform.a), size * MEASURE_GSD
    col, row = (x0 - src.transform.c) / gsd, (src.transform.f - y0) / abs(src.transform.e)
    w, h = span / gsd, span / abs(src.transform.e)
    eps = 1e-6
    if col < -eps or row < -eps or col + w > src.width + eps or row + h > src.height + eps:
        return None
    rs = Resampling.average if abs(gsd - MEASURE_GSD) > 1e-6 else Resampling.bilinear
    d = src.read([1, 2, 3], window=Window(max(col, 0), max(row, 0), w, h), out_shape=(3, size, size),
                 resampling=rs).astype(np.float32)
    nd = src.nodata if src.nodata is not None else nodata_default
    return d.mean(0), ~np.all(d == nd, axis=0)


def read_mask(msk, x0, y0, size):
    gsd = abs(msk.transform.a)
    col, row = (x0 - msk.transform.c) / gsd, (msk.transform.f - y0) / abs(msk.transform.e)
    w, h = size * MEASURE_GSD / gsd, size * MEASURE_GSD / abs(msk.transform.e)
    return msk.read(1, window=Window(max(col, 0), max(row, 0), w, h), out_shape=(size, size),
                    resampling=Resampling.nearest) == 1


def block_origin(src, r, c, size):
    """Pojok kiri-atas blok (r,c) pada grid 0,5 m berjangkar di pojok kiri-atas src."""
    return src.transform.c + c * size * MEASURE_GSD, src.transform.f - r * size * MEASURE_GSD


def grid_shape(src, size):
    return (int(src.height * abs(src.transform.e) / MEASURE_GSD) // size,
            int(src.width * abs(src.transform.a) / MEASURE_GSD) // size)


# ---------- metrik ----------
def grad_score(g, inside):
    gy, gx = np.gradient(g)
    return float(np.hypot(gx, gy)[inside].mean())


def lap_var(g, inside):
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    m = inside[1:-1, 1:-1]
    return float(lap[m].var()) if m.sum() > 1 else float("nan")


def sharpness(src, msk, size, n_samples=12):
    """Rerata varian Laplacian pada <=n_samples blok tersebar merata (grid sendiri) yang >=80% dalam mask."""
    nr, nc = grid_shape(src, size)
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    cells = [cells[i] for i in np.linspace(0, len(cells) - 1, min(len(cells), n_samples * 3)).astype(int)] if cells else []
    vals = []
    for r, c in cells:
        x0, y0 = block_origin(src, r, c, size)
        b = read_block(src, x0, y0, size)
        if b is None:
            continue
        inside = b[1] & (read_mask(msk, x0, y0, size) if msk is not None else True)
        if inside.mean() >= MIN_INSIDE:
            vals.append(lap_var(b[0], inside))
    return float(np.nanmean(vals)) if vals else float("nan")


def choose_reference(srcs, msks, gsds, size):
    """srcs/msks/gsds: dict epoch -> ...; return (epoch_ref, alasan, {epoch: sharpness})."""
    fine = sorted(e for e, g in gsds.items() if g < FINE_GSD)
    if fine:
        return fine[0], "epoch 0,3 m", {}
    sh = {e: sharpness(srcs[e], msks[e], size) for e in srcs}
    ok = {e: s for e, s in sh.items() if s == s}
    if not ok:
        return sorted(srcs)[0], "tidak ada skor sharpness; epoch pertama", sh
    return max(ok, key=ok.get), "sharpness (varian Laplacian) tertinggi", sh


def select_blocks(ref, rmsk, size, max_blocks, grad_min):
    """Blok referensi >=80% valid/dalam mask dan padat tepi; top-N berdasarkan gradien. -> [(score, r, c, inside)]"""
    nr, nc = grid_shape(ref, size)
    out = []
    for r in range(nr):
        for c in range(nc):
            x0, y0 = block_origin(ref, r, c, size)
            b = read_block(ref, x0, y0, size)
            if b is None:
                continue
            inside = b[1] & (read_mask(rmsk, x0, y0, size) if rmsk is not None else True)
            if inside.mean() < MIN_INSIDE:
                continue
            s = grad_score(b[0], inside)
            if s >= grad_min:
                out.append((s, r, c, inside))
    out.sort(key=lambda t: -t[0])
    return out[:max_blocks]


def register_farm(ref, rmsk, eps, size, max_blocks, grad_min, upsample):
    """ref: dataset referensi; eps: {epoch: dataset}. Return ({epoch: [(r,c,dx,dy,resp)]}, n_blok_terpilih)."""
    blocks = select_blocks(ref, rmsk, size, max_blocks, grad_min)
    res = {e: [] for e in eps}
    for _, r, c, inside in blocks:
        x0, y0 = block_origin(ref, r, c, size)
        g_ref = read_block(ref, x0, y0, size)[0]
        for e, src in eps.items():
            b = read_block(src, x0, y0, size)
            if b is None:
                continue
            common = inside & b[1]
            if common.mean() < MIN_INSIDE:
                continue
            mean_ref, mean_mov = g_ref[common].mean(), b[0][common].mean()
            a = np.where(common, g_ref, mean_ref)  # piksel tak valid -> rata-rata, bukan tepi palsu
            m = np.where(common, b[0], mean_mov)
            dx, dy, resp = phase_corr(a, m, upsample)
            res[e].append((r, c, dx, dy, resp))
    return res, len(blocks)


# ---------- ringkasan ----------
def summarize(df, resp_min):
    """df: baris blok. Per (farm, epoch): median residual (hypot) memakai blok resp>=resp_min."""
    rows = []
    for (farm, epoch), g in df.groupby(["farm", "epoch"], sort=True):
        ok = g[g.resp >= resp_min]
        mag = np.hypot(ok.dx_px, ok.dy_px)
        med = float(mag.median()) if len(ok) else float("nan")
        rows.append(dict(farm=farm, epoch=epoch, n_blocks=len(g), n_reliable=len(ok), median_dx_px=ok.dx_px.median(),
                         median_dy_px=ok.dy_px.median(), median_resid_px=med, median_resid_m=med * MEASURE_GSD,
                         red_flag=bool(len(ok) and med > RED_PX), unmeasured=not len(ok)))
    return pd.DataFrame(rows)


def make_report(summ, refs, elapsed, args, notes):
    red = summ[summ.red_flag]
    L = ["# S1 registrasi antar-epoch", "",
         f"Waktu jalan: {elapsed:.1f} s; farm: {summ.farm.nunique() if len(summ) else 0}; pasangan (farm, epoch) terukur: {len(summ)}.", "",
         f"- **Flag merah (median residual > {RED_PX:g} px = {RED_PX * MEASURE_GSD:g} m): {len(red)} epoch di {red.farm.nunique()} farm** "
         f"dari {len(summ)} pasangan.",
         f"- Tidak terukur (tanpa blok dengan resp ≥ {args.resp_min:g}): {int(summ.unmeasured.sum())}",
         f"- Parameter: blok {args.block}px, maks {args.max_blocks} blok/farm, grad_min {args.grad_min:g}, upsample {args.upsample}x.",
         "- Pengukuran di grid 0,5 m; clip 0,3 m (QATAR_2024_10_C030_PLE) di-resample (average) hanya saat membaca blok, tidak ada citra ditulis/di-warp.",
         "- Residual = median dari hypot(dx, dy) per blok; px = piksel 0,5 m. Tanda: dx>0 timur, dy>0 selatan.",
         "- Jumlah pasangan per kelas bergantung pada kekuatan puncak: korelasi antar-musim bisa lemah; lihat `resp` di parquet."]
    L += [f"- {n}" for n in notes]
    L += ["", "## Per farm", "| farm | referensi | alasan | epoch terukur | flag merah | tak terukur |", "|---|---|---|---|---|---|"]
    for farm, (ref, why) in sorted(refs.items()):
        s = summ[summ.farm == farm]
        L.append(f"| {farm} | {ref} | {why} | {len(s)} | {int(s.red_flag.sum())} | {int(s.unmeasured.sum())} |")
    L += ["", "## Epoch berflag merah", "| farm | epoch | median px | median m | dx | dy | blok andal |", "|---|---|---|---|---|---|---|"]
    for r in red.sort_values("median_resid_px", ascending=False).itertuples():
        L.append(f"| {r.farm} | {r.epoch} | {r.median_resid_px:.2f} | {r.median_resid_m:.2f} | "
                 f"{r.median_dx_px:.2f} | {r.median_dy_px:.2f} | {r.n_reliable}/{r.n_blocks} |")
    return "\n".join(L) + "\n"


# ---------- driver ----------
def process_farm(farm, g, root, args):
    """g: baris manifest satu farm. Return (rows, ref_info, notes)."""
    ds, msks, gsds, notes, rows = {}, {}, {}, [], []
    try:
        for r in g.itertuples():
            d = Path(root) / r.service / str(farm)
            ds[r.epoch] = rasterio.open(d / f"{farm}_{r.service}.tif")
            gsds[r.epoch] = float(r.gsd_x)
            use_mask = bool(r.has_mask) and bool(r.grid_match) and bool(r.mask_binary)
            msks[r.epoch] = rasterio.open(d / f"{farm}_{r.service}_mask.tif") if use_mask else None
            if not use_mask:
                notes.append(f"farm {farm} epoch {r.epoch}: mask tidak dipercaya -> fallback nodata raster")
        svc = dict(zip(g.epoch, g.service))
        if len(ds) < 2:
            return rows, None, notes + [f"farm {farm}: <2 epoch, dilewati"]
        ref_e, why, _ = choose_reference(ds, msks, gsds, args.block)
        others = {e: s for e, s in ds.items() if e != ref_e}
        res, nsel = register_farm(ds[ref_e], msks[ref_e], others, args.block, args.max_blocks, args.grad_min, args.upsample)
        if nsel == 0:
            notes.append(f"farm {farm}: tidak ada blok padat tepi di referensi {ref_e}")
        for e, lst in res.items():
            for r, c, dx, dy, resp in lst:
                rows.append(dict(service=svc[e], epoch=e, farm=farm, block_row=r, block_col=c, dx_px=dx, dy_px=dy,
                                 dx_m=dx * MEASURE_GSD, dy_m=dy * MEASURE_GSD, resp=resp, n_blocks=len(lst)))
            if not lst:  # epoch tanpa blok terukur tetap tercatat (n_blocks=0) agar muncul sebagai tak terukur
                rows.append(dict(service=svc[e], epoch=e, farm=farm, block_row=-1, block_col=-1, dx_px=np.nan,
                                 dy_px=np.nan, dx_m=np.nan, dy_m=np.nan, resp=np.nan, n_blocks=0))
        return rows, (ref_e, why), notes
    finally:
        for x in list(ds.values()) + [m for m in msks.values() if m is not None]:
            x.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", default=str(HERE / "out" / "s0" / "manifest.parquet"))
    ap.add_argument("--out", default=str(HERE / "out" / "s1"))
    ap.add_argument("--farms", help="subset farm, dipisah koma")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--grad-min", type=float, default=3.0, help="gradien rerata minimum (DN uint8) agar blok tidak dianggap datar")
    ap.add_argument("--resp-min", type=float, default=0.03, help="resp minimum agar blok masuk median")
    ap.add_argument("--upsample", type=int, default=20)
    a = ap.parse_args(argv)
    if rasterio is None:
        sys.exit("ERROR: paket 'rasterio' belum terpasang. Install: pip install rasterio")
    root = Path(a.data_root)
    if not root.is_dir():
        sys.exit(f"ERROR: --data-root tidak ada: {root}")
    man = pd.read_parquet(a.manifest)
    man = man[man.has_clip.astype(bool) & man.gsd_x.notna()]
    man["farm"] = man.farm.astype(str)
    if a.farms:
        man = man[man.farm.isin(a.farms.split(","))]
    if man.empty:
        sys.exit("ERROR: manifest kosong setelah filter")
    t0 = time.time()
    rows, refs, notes = [], {}, []
    for farm, g in man.groupby("farm", sort=True):
        t1 = time.time()
        r, ref, n = process_farm(farm, g, root, a)
        rows += r
        notes += n
        if ref:
            refs[farm] = ref
        print(f"farm {farm}: ref={ref[0] if ref else '-'} {time.time() - t1:.1f}s", file=sys.stderr)
    if not rows:
        sys.exit("ERROR: tidak ada hasil pengukuran")
    df = pd.DataFrame(rows, columns=COLS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "registration.parquet", index=False)
    summ = summarize(df, a.resp_min)
    summ.to_parquet(out / "registration_summary.parquet", index=False)
    elapsed = time.time() - t0
    (out / "report.md").write_text(make_report(summ, refs, elapsed, a, notes))
    print(f"S1 selesai {elapsed:.1f}s: {int(summ.red_flag.sum())}/{len(summ)} epoch flag merah -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
