"""Tes S6 dengan obs S4 palsu. Jalankan: python pilot/tests/test_s6_synth.py (atau pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s5_events as s5  # noqa: E402
import s6_synth as s6  # noqa: E402

EPS = [f"2030_{i + 1:02d}" for i in range(20)]
DT = [np.nan] + [100.0 + 10 * i for i in range(19)]


def fake_s4(unobs=0.0, seed=0):
    """3 farm x 40 site x 20 epoch; epoch 0 punya porsi tier high rendah; q bervariasi."""
    rng = np.random.default_rng(seed)
    rows, srows = [], []
    for f in "abc":
        for i in range(40):
            sid = f"{f}_{i}"
            srows.append(dict(farm=f, site_id=sid, x=100.0 * i, y=0.0, best_tier="high", n_epochs_det=15))
            for e, ep in enumerate(EPS):
                det = bool(rng.random() < 0.5)
                tier = ("high" if rng.random() < (0.02 if e == 0 else 0.5) else "low") if det else None
                ok = bool(rng.random() > 0.2)
                rows.append(dict(farm=f, site_id=sid, epoch=ep, detected=det, score=0.66 + 0.1 * rng.random() if det else 0.0,
                                 tier=tier, n_pts=int(det), observable=bool(rng.random() >= unobs), gsd=0.5, bands=3,
                                 nodata_pct=0.0, reg_resid_m=rng.random() if ok else np.nan, reg_ok=ok, dt_days=DT[e],
                                 fe_n_det=5))
    return pd.DataFrame(rows), pd.DataFrame(srows)


@pytest.fixture(scope="module")
def base():
    return s6.make_base(*fake_s4(unobs=0.1), 10)


@pytest.fixture
def perfect(monkeypatch):
    """Family tanpa noise: detected == present & observable persis."""
    monkeypatch.setitem(s6.EMIT, "perfect", lambda b, fe, rng, ctx: (np.ones_like(b), np.zeros_like(b)))


def present_from_truth(truth, eps):
    col = np.arange(len(eps))[None, :]
    t = np.array([eps.index(e) if isinstance(e, str) else -1 for e in truth.epoch])[:, None]
    ty = truth.type.to_numpy()[:, None]
    return np.select([ty == "loss", ty == "gain", ty == "flicker"], [col < t, col >= t, col != t], True)


def test_truth_consistent_with_obs_perfect_emission(base, perfect):
    for ev in (0.0, 0.5):
        obs, truth, present = s6.simulate(base, "perfect", 1, 20, 1.0, ev, 0, 40, "none")
        eps = list(obs.epoch.unique())
        assert (present_from_truth(truth, eps) == present).all()                 # truth merekonstruksi trajektori
        det = obs.detected.to_numpy().reshape(present.shape)
        assert (det == (present & obs.observable.to_numpy().reshape(present.shape))).all()
        assert not obs[~obs.observable].detected.any()                           # tak-observable bukan absen
        assert set(zip(truth.farm, truth.site_id)) == set(zip(obs.farm, obs.site_id))
        assert list(obs.columns) == s6.OBS_COLS and (obs.groupby("run").size() == len(truth) * 20).all()
    assert truth.type.isin(["loss", "gain"]).mean() > 0.4 and (truth.type == "flicker").any()


def test_event_margin_and_interval(base):
    obs, truth, _ = s6.simulate(base, "mismatched", 1, 20, 1.0, 0.5, 1, 40, "real")
    ev = truth[truth.type != "stable"]
    idx = lambda c: ev[c].map(EPS.index)  # noqa: E731
    assert ((idx("epoch") >= 3) & (idx("epoch") <= 20 - 3)).all()
    assert ((idx("t_before") == idx("epoch") - 1) & (idx("t_after") == idx("epoch") + (ev.type == "flicker"))).all()
    assert truth[truth.type == "stable"].epoch.isna().all()


def test_s5_recovers_truth_events_when_detection_perfect(perfect):
    obs, sites = fake_s4()
    b = s6.make_base(obs, sites, 10)
    o, truth, _ = s6.simulate(b, "perfect", 1, 20, 1.0, 0.5, 2, 40, "none")
    o = o[o.observable]                                                           # fake: semua observable
    _, _, ev, _ = s5.mine(o, sites, 3, 2, False)
    got = {(r.site_id, r.type): (r.t_before, r.t_after) for r in ev.itertuples()}
    exp = {(r.site_id, r.type): (r.t_before, r.t_after) for r in truth.itertuples() if r.type in ("loss", "gain")}
    assert got == exp                                                            # flicker/stable: tidak ada kandidat


def test_gap_and_skipped_cells(base):
    obs, _, _ = s6.simulate(base, "matched", 2, 10, 1.0, 0.25, 0, 5, "none")
    d = obs[obs.site_id == obs.site_id.iloc[0]].dt_days.to_numpy()
    assert np.isnan(d[0]) and d[1] == DT[1] + DT[2] and d[2] == DT[3] + DT[4]    # jumlah gap nyata yang dilompati
    assert obs.epoch.iloc[:10].tolist() == EPS[::2]
    assert s6.simulate(base, "matched", 2, 20, 1.0, 0.25, 0, 5, "none") is None


def test_families_axes_and_skew(base):
    rate = lambda *a: s6.simulate(base, *a)[0].pipe(lambda o: o.detected[o.observable].mean())  # noqa: E731
    assert rate("mismatched", 1, 20, 0.5, 0.1, 0, 40, "none") > rate("mismatched", 1, 20, 3.0, 0.1, 0, 40, "none")
    assert rate("matched", 1, 20, 0.5, 0.1, 0, 40, "none") > rate("matched", 1, 20, 3.0, 0.1, 0, 40, "none")
    # epoch 0 (high-share 2%) ditekan oleh skew real, tidak oleh none
    e0 = lambda sk: s6.simulate(base, "mismatched", 1, 20, 0.5, 0.0, 0, 40, sk)[0].pipe(  # noqa: E731
        lambda o: o[o.epoch == EPS[0]].detected.mean())
    assert e0("real") < 0.5 * e0("none") and e0("extreme") < 0.1
    _, tn, _ = s6.simulate(base, "null", 1, 20, 0.5, 0.5, 0, 40, "real")
    assert not tn.type.isin(["loss", "gain"]).any() and (tn.type == "flicker").any()   # null: tanpa event, flicker tetap
    a, b = (s6.simulate(base, "stress", 1, 20, 1.0, 0.25, s, 40, "real")[0] for s in (0, 0))
    assert a.equals(b)                                                           # deterministik per seed
    assert not a.detected.equals(s6.simulate(base, "stress", 1, 20, 1.0, 0.25, 1, 40, "real")[0].detected)


def test_cli_end_to_end(tmp_path):
    obs, sites = fake_s4(unobs=0.05)
    obs.to_parquet(tmp_path / "obs.parquet")
    sites.to_parquet(tmp_path / "sites.parquet")
    out = tmp_path / "out"
    args = ["--s4", str(tmp_path), "--out", str(out), "--per-farm", "20", "--events", "0.25", "--qscales", "1"]
    with pytest.raises(SystemExit):
        s6.main(args + ["--seeds", "2"])                                         # minimal 3 seed
    assert s6.main(args) == 0
    o, t = pd.read_parquet(out / "obs.parquet"), pd.read_csv(out / "truth_events.csv", keep_default_na=False)
    assert set(o.run) == set(t.run) and set(t.family) == set(s6.FAMILIES)
    assert (t.groupby("run").size() * 0 + 60).eq(t.groupby("run").size()).all()  # 3 farm x 20 site
    assert t.groupby(["family", "seed"]).ngroups == 4 * 3
    rep = (out / "report.md").read_text()
    assert "stride 2 x n_epochs 20" in rep and "| mismatched | 0 |" in rep


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
