"""Tes S1 dengan data sintetis. Jalankan: python pilot/tests/test_s1_registration.py (atau pytest)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s1_registration as s1  # noqa: E402

TOL = 0.2  # px (0,5 m)
rng = np.random.default_rng(0)
# medan kontinu: blob gaussian acak (sigma 1,5-3 m) -> bisa disampel di GSD apa pun dengan offset dunia
BLOBS = [(rng.uniform(0, 200), rng.uniform(0, 200), rng.uniform(1.5, 3), rng.uniform(-1, 1)) for _ in range(2500)]
_B = np.array(BLOBS)


def field(xs, ys):
    """xs, ys: koordinat dunia 1D -> citra uint8 (len(ys), len(xs)); hanya blob dekat yang dihitung."""
    X, Y = np.meshgrid(xs, ys)
    img = np.zeros(X.shape)
    for bx, by, s, a in _B[(_B[:, 0] > xs.min() - 12) & (_B[:, 0] < xs.max() + 12)
                           & (_B[:, 1] > ys.min() - 12) & (_B[:, 1] < ys.max() + 12)]:
        img += a * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * s ** 2))
    return np.clip(128 + 60 * img, 0, 255).astype(np.uint8)


def write_tif(path, gsd, size_m, shift=(0.0, 0.0), origin=(1000.0, 5000.0), bands=3, hole=False):
    import rasterio
    from rasterio.transform import from_origin
    n = int(size_m / gsd)
    # isi epoch(x) = ref(x - shift) -> citra sampel di (x - shift_m); dx>0 = ke timur
    xs = origin[0] - 1000 + (np.arange(n) + 0.5) * gsd - shift[0]
    ys = origin[1] - 5000 + (np.arange(n) + 0.5) * gsd - shift[1]  # y piksel ke bawah = y medan naik
    band = field(xs, ys)
    if hole:
        band[: n // 3] = 0  # sepertiga atas nodata
    data = np.stack([band] * bands)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=bands, dtype="uint8",
                       crs="EPSG:2932", transform=from_origin(*origin, gsd, gsd), nodata=0) as dst:
        dst.write(data)


def test_phase_corr_recovers_subpixel_shift():
    img = rng.normal(size=(128, 128))
    img = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-(np.fft.fftfreq(128)[:, None] ** 2 + np.fft.fftfreq(128)[None] ** 2) * 200)))
    for dx, dy in [(0, 0), (3, -2), (1.4, 2.6), (-5.25, 0.5)]:
        ky, kx = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(128), indexing="ij")
        mov = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (kx * dx + ky * dy))))
        gx, gy, resp = s1.phase_corr(img, mov)
        assert abs(gx - dx) < 0.15 and abs(gy - dy) < 0.15 and resp > 0.03, (dx, dy, gx, gy, resp)


def test_flat_block_has_low_gradient():
    flat = np.full((64, 64), 100.0)
    ok = np.ones((64, 64), bool)
    assert s1.grad_score(flat, ok) == 0.0
    assert s1.lap_var(flat, ok) == 0.0
    assert s1.lap_var(rng.normal(size=(64, 64)), ok) > 1


def test_farm_end_to_end_mixed_gsd_and_nodata():
    shifts = {"E1": (0.0, 0.0), "E2": (1.0, -0.5), "E3": (-0.75, 1.25), "E4": (2.0, 1.0)}  # meter
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        man = []
        for e, sh in shifts.items():
            gsd = 0.3 if e == "E3" else 0.5  # E3 = 0,3 m -> referensi, jadi geser relatif ke E3
            svc, farm = f"QATAR_{e}_C050_PLE", "9"
            d = t / svc / farm
            d.mkdir(parents=True)
            write_tif(d / f"{farm}_{svc}.tif", gsd, 200, shift=sh, bands=4 if e == "E1" else 3, hole=(e == "E4"))
            man.append(dict(service=svc, epoch=e, farm=farm, has_clip=True, has_mask=False, gsd_x=gsd,
                            grid_match=False, mask_binary=False))
        mp = t / "manifest.parquet"
        pd.DataFrame(man).to_parquet(mp)
        out = t / "out"
        assert s1.main(["--data-root", str(t), "--manifest", str(mp), "--out", str(out), "--block", "128",
                        "--grad-min", "0.5", "--max-blocks", "3"]) == 0
        df = pd.read_parquet(out / "registration.parquet")
        assert list(df.columns) == s1.COLS
        assert set(df.epoch) == {"E1", "E2", "E4"}  # E3 (0,3 m) = referensi, tidak diukur
        summ = pd.read_parquet(out / "registration_summary.parquet").set_index("epoch")
        for e in ("E1", "E2", "E4"):
            exp = (np.array(shifts[e]) - np.array(shifts["E3"])) / 0.5  # px 0,5 m relatif ke E3
            g = df[(df.epoch == e) & (df.resp >= 0.03)]
            assert len(g) >= 1, e
            assert abs(g.dx_px.median() - exp[0]) < TOL and abs(g.dy_px.median() - exp[1]) < TOL, (e, exp, g)
            assert abs(g.dx_m.median() - exp[0] * 0.5) < TOL * 0.5
        # semua epoch bergeser >1 px dari E3 -> merah; E1 dgn 1,5 px, E2 3,5 px, E4 5,5 px
        assert summ.red_flag.all()
        assert "E3" in (out / "report.md").read_text()


def test_summarize_flag_boundary_and_unmeasured():
    rows = [dict(farm="f", epoch="a", dx_px=0.5, dy_px=0.0, resp=0.5),
            dict(farm="f", epoch="b", dx_px=1.5, dy_px=0.0, resp=0.5),
            dict(farm="f", epoch="c", dx_px=9.0, dy_px=0.0, resp=0.001)]
    s = s1.summarize(pd.DataFrame(rows), 0.03).set_index("epoch")
    assert not s.loc["a", "red_flag"] and s.loc["b", "red_flag"]
    assert s.loc["c", "unmeasured"] and not s.loc["c", "red_flag"]
    assert s.loc["b", "median_resid_m"] == 0.75


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
