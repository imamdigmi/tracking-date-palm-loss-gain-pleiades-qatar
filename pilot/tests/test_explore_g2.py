"""Tes kecil explore_g2: metrik numpy dan seleksi silver E1 deterministik."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import explore_g2 as ex  # noqa: E402
import explore_plots as pl  # noqa: E402


def test_metrics_selfcheck_and_ap_ties():
    ex.selfcheck()
    y = np.array([1, 0, 1, 0], bool)
    assert ex.auc(np.array([.9, .1, .8, .2]), y) == 1 and ex.ap(np.array([.9, .1, .8, .2]), y) == 1
    assert abs(ex.ap(np.array([.5, .5, .5, .5]), y) - 0.5) < 1e-12   # tie penuh -> AP = prevalensi
    assert np.isnan(ex.auc(np.ones(3), np.ones(3, bool)))


def test_silver_sample_deterministic_and_filters():
    eps = [f"e{i}" for i in range(14)]
    sites = pd.DataFrame([dict(farm=f, site_id=f"{f}_{i}", best_tier="high") for f in ("1", "2") for i in range(25)])
    obs = pd.DataFrame([dict(site_id=s, observable=True, detected=(i % 25 != 0 or e != "e0")) for i, s in enumerate(sites.site_id)
                        for e in eps])
    cand = pd.DataFrame(dict(site_id=["1_3", "1_4"]))
    U, s1 = ex.silver_sample(sites, obs, cand, {"1", "2"})
    _, s2 = ex.silver_sample(sites, obs, cand, {"1", "2"})
    assert s1.equals(s2) and s1.groupby("farm").size().tolist() == [10, 10]
    assert not s1.site_id.isin(["1_3", "1_4"]).any() and (s1.type == "stable").all()


def test_plot_import_guard():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            pl.plot_e1(".", ".")
