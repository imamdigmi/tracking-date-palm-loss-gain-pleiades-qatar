"""Tes S5b dengan data sintetis. Jalankan: python pilot/tests/test_s5b_filmstrip.py (atau pytest)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s5b_filmstrip as s5b  # noqa: E402


def cands(rows):
    """rows: (type, farm, interval_idx, nb_rate, persistent)."""
    return pd.DataFrame([dict(farm=f, site_id=f"{f}_{i}", x=0.0, y=0.0, type=t, t_before=f"2020_{iv:02d}", t_after=f"2021_{iv:02d}",
                              nb_rate=nb, persistent=p) for i, (t, f, iv, nb, p) in enumerate(rows)])


def test_pick_caps_and_relax():
    rows = [("loss", f"f{i % 10}", i % 10, 0.9, True) for i in range(100)]          # 10 farm x 10 baris, 10 interval
    p, lvl = s5b.pick(cands(rows), 20)
    assert lvl == 0 and len(p) == 20 and max(Counter(p.farm).values()) <= 3
    assert max(Counter(zip(p.t_before, p.t_after)).values()) <= 6
    # hanya 2 farm: cap farm 3 -> maks 6; minta 10 -> cap farm dilonggarkan (level 1), cap interval 6 tetap
    p, lvl = s5b.pick(cands([("loss", f"f{i % 2}", i % 4, 0.9, True) for i in range(40)]), 10)
    assert lvl == 1 and len(p) == 10 and max(Counter(zip(p.t_before, p.t_after)).values()) <= 6
    # satu interval saja: kedua cap longgar
    p, lvl = s5b.pick(cands([("loss", f"f{i % 2}", 0, 0.9, True) for i in range(40)]), 10)
    assert lvl == 2 and len(p) == 10
    assert s5b.pick(cands(rows), 20)[0].index.tolist() == s5b.pick(cands(rows), 20)[0].index.tolist()   # deterministik


def test_sample_filters_and_ids():
    rows = [(t, f"f{i % 12}", i % 9, 0.9, True) for t in ("loss", "gain") for i in range(120)]
    rows += [("loss", "f0", 0, 0.5, True), ("loss", "f0", 0, 0.9, False)] * 5           # nb rendah / non-persistent: tersaring
    rows += [(t, f"f{i}", 0, np.nan, None) for t in ("stable", "flicker") for i in range(20)]
    c = cands(rows)
    c["persistent"] = c.persistent.astype("boolean")
    s, relax = s5b.sample(c)
    assert len(s) == 70 and s.type.value_counts().to_dict() == {"loss": 40, "gain": 20, "stable": 5, "flicker": 5}
    assert s.review_id.tolist() == [f"R{i:03d}" for i in range(1, 71)] and set(relax) == {"loss", "gain"}
    ev = s[s.type.isin(["loss", "gain"])]
    assert (ev.nb_rate >= 0.6).all() and ev.persistent.all()
    assert s.type.iloc[:10].nunique() > 1                                                  # diacak, bukan terurut per tipe
    # batch 2: tier high saja, site lama dibuang, awalan id baru
    c["best_tier"] = np.where(np.arange(len(c)) % 4 == 0, "med", "high")
    s2, _ = s5b.sample(c, n_per_type={"loss": 30, "gain": 10}, tier="high", exclude=s, prefix="Q")
    assert (s2.best_tier == "high").all() and len(s2) == 50 and s2.review_id.iloc[0].startswith("Q")
    assert not s2.set_index(["farm", "site_id", "type"]).index.isin(s.set_index(["farm", "site_id", "type"]).index).any()


def _tif(path, res, blob, size=100.0):
    """Raster RGB res m/px, origin (1000, 5000+size); latar 50, kotak putih 10 x 10 m berpusat di blob=(x, y)."""
    n = int(size / res)
    t = from_origin(1000.0, 5000.0 + size, res, res)
    xs = 1000.0 + (np.arange(n) + 0.5) * res
    ys = 5000.0 + size - (np.arange(n) + 0.5) * res
    inb = (np.abs(xs[None, :] - blob[0]) <= 5) & (np.abs(ys[:, None] - blob[1]) <= 5)
    data = np.where(inb, 255, 50).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=3, dtype="uint8", crs="EPSG:2932",
                       transform=t, nodata=0) as d:
        d.write(np.stack([data] * 3))


def centroid(img):
    m = img[2] == 255                                  # kanal biru: cincin kuning bernilai 0 di sini
    yy, xx = np.nonzero(m)
    return xx.mean() + 0.5, yy.mean() + 0.5


def test_crop_centre_shift_and_scale(tmp_path):
    _tif(tmp_path / "a.tif", 0.5, (1052.0, 5052.0))
    _tif(tmp_path / "b.tif", 0.3, (1052.0, 5052.0))
    for f in "ab":
        with rasterio.open(tmp_path / f"{f}.tif") as src:
            # situs referensi (1050, 5053), shift S1 dx=+4 px, dy=+2 px -> pusat frame epoch (1052, 5052) = pusat kotak
            cx, cy = 1050.0 + 4 * 0.5, 5053.0 - 2 * 0.5
            img = s5b.crop(src, cx, cy, 0)
            assert img.shape == (3, 192, 192) and img.dtype == np.uint8
            assert all(abs(v - 96) < 1.5 for v in centroid(img))
            bad = s5b.crop(src, 1050.0 - 2.0, 5053.0 + 1.0, 0)                            # tanda salah -> geser 4 m = 16 px
            assert abs(centroid(bad)[0] - 96) > 10
    d = np.hypot(*(np.indices((192, 192)) + 0.5 - 96))
    ring = (img[0] == 255) & (img[1] == 255) & (img[2] == 0)
    assert abs(d[ring].mean() - 16) < 1 and not ring[90:102, 90:102].any()              # radius 4 m = 16 px; pusat bebas
    with rasterio.open(tmp_path / "a.tif") as src:                                      # di luar raster -> hitam
        out = s5b.crop(src, 900.0, 4900.0, 0)
        assert out[:, :10, :10].max() == 0
    s5b.write_png(tmp_path / "x.png", img)
    with rasterio.open(tmp_path / "x.png") as p:
        assert (p.count, p.width, p.height) == (3, 192, 192) and (p.read() == img).all()


def test_html_blind_and_main(tmp_path):
    s = pd.DataFrame(dict(review_id=["R001", "R002"], farm=["f1", "f2"], type=["loss", "gain"], score_ref=[0.9, 0.8]))
    page = s5b.make_html(s, {"f1": ["2012_08", "2013_01"], "f2": ["2012_08"]}).lower()
    for w in ("loss", "gain", "score", "stable", "flicker", "persistent", "nb_rate", "tier", "detect"):
        assert w not in page
    assert 'src="img/r001_2013_01.png"' in page and "farm f1" in page and "hilang" in page and "muncul" in page

    root = tmp_path / "data"
    farm, svc = "f1", "SVC_A"
    (root / svc / farm).mkdir(parents=True)
    _tif(root / svc / farm / f"{farm}_{svc}.tif", 0.5, (1052.0, 5052.0))
    man = pd.DataFrame([dict(service=svc, farm=farm, epoch=e, has_clip=True, has_mask=False, gsd_x=0.5, bands=3, nodata_pct=0.0, nodata_used=0.0)
                        for e in ("2020_01", "2020_02")])
    man.to_parquet(tmp_path / "man.parquet")
    summ = pd.DataFrame([dict(farm=farm, epoch="2020_02", median_dx_px=4.0, median_dy_px=2.0, median_resid_m=0.2, red_flag=False, unmeasured=False)])
    summ.to_parquet(tmp_path / "summ.parquet")
    rows = [("loss", farm, i % 7, 0.9, True) for i in range(70)] + [("gain", farm, i % 7, 0.9, True) for i in range(70)]
    rows += [(t, farm, 0, np.nan, None) for t in ("stable", "flicker") for _ in range(6)]
    c = cands(rows).assign(x=1050.0, y=5053.0)
    c["persistent"] = c.persistent.astype("boolean")
    c.to_parquet(tmp_path / "c.parquet")
    o = tmp_path / "o"
    assert s5b.main(["--data-root", str(root), "--cands", str(tmp_path / "c.parquet"), "--summary", str(tmp_path / "summ.parquet"),
                     "--manifest", str(tmp_path / "man.parquet"), "--out", str(o)]) == 0
    assert len(list((o / "img").glob("*.png"))) == 70 * 2
    rv, key = pd.read_csv(o / "review.csv"), pd.read_parquet(o / "key.parquet")
    assert rv.columns.tolist() == ["review_id", "farm", "verdict", "first_change_epoch", "second_change_epoch", "note"] and len(rv) == 70
    assert rv[["verdict", "first_change_epoch", "second_change_epoch", "note"]].isna().all().all() and "type" in key and len(key) == 70
    assert "cap farm dilonggarkan" in (o / "report.md").read_text()
    body = (o / "index.html").read_text().lower()
    assert "loss" not in body and "gain" not in body and body.count("<h2") == 70


if __name__ == "__main__":
    import tempfile
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            if "tmp_path" in v.__code__.co_varnames[:1]:
                with tempfile.TemporaryDirectory() as t:
                    v(Path(t))
            else:
                v()
            print("ok", k)
