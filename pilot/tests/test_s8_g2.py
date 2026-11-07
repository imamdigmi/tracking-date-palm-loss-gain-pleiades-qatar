"""Tes S8 dengan label & obs mini sintetis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s7_g1 as s7  # noqa: E402
import s8_g2 as s8  # noqa: E402


def test_real_truth_classes():
    L = pd.DataFrame(dict(farm="1", site_id=list("abcdef"), label=["loss", "gain", "none", "none", "tolak", "ambigu"],
                          verdict=["hilang", "muncul", "ada", "tidak_ada", "ada", "ambigu"],
                          change_epoch=["2024_04", "2024_06", "", "2024_06", "", ""]))
    t = s8.real_truth(L)  # 'tolak' (mis. note meleset) bukan negatif walau verdict 'ada'
    assert t.type.tolist() == ["loss", "gain", "stable", "absent"] and t.epoch.tolist() == ["2024_04", "2024_06", "", ""]


def test_load_runs_absent_and_end_to_end():
    rng = np.random.default_rng(0)
    eps = [f"2020_{i:02d}" for i in range(1, 9)]
    sites = [(f"f{i % 4}", f"f{i % 4}_{i}", ty) for i, ty in enumerate(["loss", "gain", "stable", "absent"] * 6)]
    truth = pd.DataFrame([dict(farm=f, site_id=s, type=ty, epoch="2020_05" if ty in ("loss", "gain") else "", run="real")
                          for f, s, ty in sites])
    col = np.arange(8)
    rows = []
    for f, s, ty in sites:
        pres = {"loss": col < 4, "gain": col >= 4, "stable": col >= 0, "absent": col < 0}[ty]
        det = pres & (rng.random(8) < 0.9)
        for j, e in enumerate(eps):
            rows.append(dict(farm=f, site_id=s, epoch=e, detected=bool(det[j]), score=0.8 * det[j], observable=True, gsd=0.5,
                             nodata_pct=0.0, reg_resid_m=np.nan, reg_ok=False, dt_days=np.nan if j == 0 else 60.0,
                             fe_n_det=int(det[j]) * 5, run="real", fe_nobs=20))
    obs = pd.DataFrame(rows)
    R, e, _ = s7.load_runs(obs, truth, {f"f{i}": i % 2 for i in range(4)})["real"]
    assert not R["present"][truth.type.eq("absent").to_numpy()].any()
    assert (R["nobs"] == 20).all()
    P = s7.fit(R, np.ones(len(truth), bool), 0.05, 1)
    p = s8.predict(R, e, P, 1)
    fm = s8.farm_table(p, (0.05, 0.01, 0.1))
    assert set(p.method) == set(s7.METHODS) and (p[p.method == "B0"].p_event == 0).all()
    assert fm[fm.method == "A"].n_events.sum() == 12 and p.p_event.between(0, 1).all()
