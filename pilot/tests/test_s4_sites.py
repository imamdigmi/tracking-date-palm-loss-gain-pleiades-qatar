"""Tes S4 dengan data sintetis. Jalankan: python pilot/tests/test_s4_sites.py (atau pytest)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s4_sites as s4  # noqa: E402


def pts(rows):
    return pd.DataFrame(rows, columns=["farm", "x", "y", "epoch", "score", "tier"])


def test_correction_sign_collapses_shifted_point():
    # isi epoch bergeser dx=+4 px (timur 2 m), dy=+2 px (selatan 1 m) -> titik kembali ke posisi referensi
    x, y = s4.correct(1002.0, 4999.0, 4, 2)
    assert (x, y) == (1000.0, 5000.0)


def test_cluster_radius():
    sc = np.array([0.9, 0.5])
    assert len(set(s4.cluster(np.array([[0, 0], [2.0, 0]]), ["A", "B"], sc, 2.5))) == 1
    assert len(set(s4.cluster(np.array([[0, 0], [3.0, 0]]), ["A", "B"], sc, 2.5))) == 2
    assert len(set(s4.cluster(np.array([[0, 0], [2.0, 0]]), ["A", "A"], sc, 2.5))) == 2   # 1 titik/epoch/site


def test_cluster_no_chaining():
    # barisan pohon tiap 2 m di epoch berbeda: komponen terhubung = 1 site 6 m; leader = pusat tetap, tidak merantai
    xy = np.array([[0, 0], [2.0, 0], [4.0, 0], [6.0, 0]])
    lab = s4.cluster(xy, ["A", "B", "C", "D"], np.array([0.9, 0.8, 0.7, 0.6]), 2.5)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[1] != lab[2]


def test_merge_fragment_only_if_epochs_disjoint():
    # pecahan 3,5 m (> r 2,5, < MERGE_M 4) digabung bila epoch-nya belum ada di site besar
    p = pts([("f", 0, 0, "A", 0.9, "high"), ("f", 0, 0, "B", 0.8, "high"), ("f", 3.5, 0, "C", 0.7, "low")])
    assert len(s4.build_sites(p, 2.5)[1]) == 1
    p = pts([("f", 0, 0, "A", 0.9, "high"), ("f", 0, 0, "B", 0.8, "high"), ("f", 3.5, 0, "A", 0.7, "low")])
    assert len(s4.build_sites(p, 2.5)[1]) == 2
    p = pts([("f", 0, 0, "A", 0.9, "high"), ("f", 0, 0, "B", 0.8, "high"), ("f", 4.5, 0, "C", 0.7, "low")])
    assert len(s4.build_sites(p, 2.5)[1]) == 2


def test_sites_obs_pick_score_tier_and_dup():
    p = pts([("f", 0, 0, "A", 0.9, "high"), ("f", 1, 0, "A", 0.5, "low"),   # titik ke-2 epoch A -> site sendiri
             ("f", 0.5, 0, "B", 0.6, "med"), ("f", 50, 0, "A", 0.3, "low")])
    pp, sites = s4.build_sites(p, 2.5)
    s = sites.set_index("x").loc[0.25]
    assert len(sites) == 3 and s.n_pts == 2 and s.n_epochs_det == 2 and (sites.n_dup_epochs == 0).all()
    assert s.best_tier == "high" and s.first_epoch == "A" and s.last_epoch == "B" and abs(s.extent_m - 0.25) < 1e-9
    meta = pd.DataFrame([dict(farm="f", epoch=e, gsd=0.5, bands=3, nodata_pct=0.0, reg_resid_m=0.0, reg_ok=True,
                              dt_days=np.nan, fe_n_det=2) for e in "ABC"])
    o = s4.make_obs(pp, sites, meta).set_index(["site_id", "epoch"])
    assert len(o) == 9
    a = o.loc[(s.site_id, "A")]
    assert a.detected and a.score == 0.9 and a.tier == "high" and a.n_pts == 1
    c = o.loc[(s.site_id, "C")]
    assert not c.detected and c.score == 0.0 and c.n_pts == 0 and pd.isna(c.tier)


def test_epoch_meta_reg_flags_and_dt():
    man = pd.DataFrame([dict(farm="f", epoch=e, gsd_x=0.5, bands=3, nodata_pct=0.0) for e in ("2024_10", "2024_04", "2024_06", "2024_07")])
    summ = pd.DataFrame([dict(farm="f", epoch="2024_04", median_dx_px=2.0, median_dy_px=-1.0, median_resid_m=1.1, red_flag=True, unmeasured=False),
                         dict(farm="f", epoch="2024_06", median_dx_px=np.nan, median_dy_px=np.nan, median_resid_m=np.nan, red_flag=False, unmeasured=True),
                         dict(farm="f", epoch="2024_07", median_dx_px=0.5, median_dy_px=0.0, median_resid_m=0.25, red_flag=False, unmeasured=False)])
    m = s4.epoch_meta(man, summ, pd.Series({("f", "2024_10"): 5})).set_index("epoch")
    assert m.loc["2024_10", "reg_ok"] and m.loc["2024_10", "reg_resid_m"] == 0.0 and m.loc["2024_10", "dx_px"] == 0.0
    assert not m.loc["2024_06", "reg_ok"] and np.isnan(m.loc["2024_06", "reg_resid_m"]) and m.loc["2024_06", "dx_px"] == 0.0
    assert not m.loc["2024_04", "reg_ok"] and m.loc["2024_04", "dx_px"] == 2.0   # merah: tetap dikoreksi
    assert m.loc["2024_07", "reg_ok"]
    assert np.isnan(m.loc["2024_04", "dt_days"]) and m.loc["2024_06", "dt_days"] == 61 and m.loc["2024_10", "fe_n_det"] == 5
    assert m.loc["2024_04", "fe_n_det"] == 0


def _tif(path, data, nodata=0):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                       dtype="uint8", crs="EPSG:2932", transform=from_origin(1000.0, 5020.0, 1.0, 1.0), nodata=nodata) as d:
        d.write(data)


def test_observable_window_nodata_mask_and_shift():
    import rasterio
    clip = np.full((3, 20, 20), 100, np.uint8)
    clip[:, :, 0:5] = 0                      # nodata kolom x 1000..1005
    mask = np.ones((1, 20, 20), np.uint8)
    mask[:, :, 15:] = 0                      # di luar farm x >= 1015
    with tempfile.TemporaryDirectory() as t:
        _tif(Path(t) / "c.tif", clip)
        _tif(Path(t) / "m.tif", mask)
        x = np.array([1010.5, 1002.5, 1017.5, 1010.5, 1010.5, 1030.5, 1006.5])
        y = np.array([5010.5, 5010.5, 5010.5, 5010.5, 5030.5, 5010.5, 5010.5])
        box = (1000, 5000, 1020, 5020)       # y=5030 di luar box; x=1030 di luar box & clip
        with rasterio.open(Path(t) / "c.tif") as c, rasterio.open(Path(t) / "m.tif") as m:
            f = s4.observable_flags(c, m, 0, x, y, 0.0, 0.0, box)
            assert f.tolist() == [True, False, False, True, False, False, True]
            # shift S1 dx=+4 px (2 m timur): sampel clip di x+2 -> 1004.5 nodata, 1008.5 valid
            f = s4.observable_flags(c, m, 0, np.array([1002.5, 1006.5]), np.array([5010.5] * 2), 4.0, 0.0, box)
            assert f.tolist() == [False, True]
            assert s4.observable_flags(c, None, 0, np.array([1017.5]), np.array([5010.5]), 0.0, 0.0, box).tolist() == [True]


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
