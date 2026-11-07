"""Tes S7 dengan data palsu kecil. Jalankan: python pilot/tests/test_s7_g1.py (atau pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s6_synth as s6  # noqa: E402
import s7_g1 as s7  # noqa: E402
from test_s6_synth import fake_s4  # noqa: E402


def test_hmm_perfect_detector_recovers_type_and_epoch():
    n = 12
    col = np.arange(n)[None, :]
    t = np.array([5, 7, 4])[:, None]
    present = np.vstack([(col < t[0]), (col >= t[1]), np.ones((1, n), bool)])   # loss@5, gain@7, stable
    one = np.ones(present.shape)
    dt = np.full(present.shape, 10.0)
    pl, pg, el, eg = s7.hmm(present, np.ones_like(present), one, 0 * one, dt, 1e-3, 1e-3)
    assert pl[0] > 0.99 and pg[1] > 0.99 and pl[2] + pg[2] < 0.01
    assert el[0] == 5 and eg[1] == 7
    assert (pl + pg <= 1 + 1e-9).all()


def test_hmm_posterior_sums_to_one_and_unobservable_ignored():
    rng = np.random.default_rng(0)
    det, obs = rng.random((50, 10)) < 0.5, rng.random((50, 10)) < 0.7
    p = rng.uniform(0.3, 0.9, det.shape)
    dt = np.full(det.shape, 50.0)
    a = s7.hmm(det & obs, obs, p, 0.05 + 0 * p, dt, 1e-3, 1e-3)
    b = s7.hmm(~det | (det & obs), obs, p, 0.05 + 0 * p, dt, 1e-3, 1e-3)   # nilai di baris tak-observable diganti
    b2 = s7.hmm(np.where(obs, det, True), obs, p, 0.05 + 0 * p, dt, 1e-3, 1e-3)
    for x, y in zip(a, b2):
        assert np.allclose(x, y)
    assert ((a[0] + a[1]) <= 1 + 1e-9).all() and b[0].shape == (50,)


def test_b3_run_length_rule():
    det = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1]], bool)
    sc, ty, ep = s7.b3(det, np.ones_like(det), det * 0.9, 2, 2)
    assert (ty[:2] == ["loss", "gain"]).all() and (ep[:2] == [3, 3]).all() and sc[0] > 0 and sc[1] > 0
    assert sc[2] == 0                                    # redeteksi setelah gap -> bukan event


def test_recall_at_fa_constant_and_perfect():
    ev = np.array([1] * 20 + [0] * 80, bool)
    assert s7.recall_at_fa(np.full(100, 0.3), ev.copy(), ev, 0.05) == 0
    assert s7.recall_at_fa(np.where(ev, 1.0, 0.0), ev.copy(), ev, 0.05) == 1
    assert s7.recall_at_fa(np.where(ev, 1.0, 0.0), np.zeros(100, bool), ev, 0.05) == 0   # jenis/epoch salah


def test_verdict_win_tie_and_b0():
    rng = np.random.default_rng(1)
    dB, dR = -0.05 + 0.01 * rng.standard_normal(18), 0.2 + 0.01 * rng.standard_normal(18)
    assert s7.verdict(dB, dR, reps=500)["win"]
    assert not s7.verdict(0 * dB, 0 * dR, reps=500)["win"]
    fm = pd.DataFrame(dict(family="mismatched", farm=list("abcdef") * 2, method=["B0"] * 6 + ["B4"] * 6,
                           brier=[0.3] * 6 + [0.1] * 6, **{"recall_0.05": [0.0] * 6 + [0.7] * 6}))
    assert not s7.compare(fm, "mismatched", "B0", "B4", reps=200)["win"]


def test_end_to_end_A_beats_B4_when_emission_depends_on_q(monkeypatch):
    monkeypatch.setitem(s6.EMIT, "mismatched", lambda b, fe, rng, ctx: (np.where(b > 1.1, 0.1, 0.95), np.full_like(b, 0.02)))
    base = s6.make_base(*fake_s4(), 10)
    sims = [s6.simulate(base, "mismatched", 1, 20, 1.0, 0.5, seed, 40, "none") for seed in range(4)]
    obs = pd.concat([s[0] for s in sims], ignore_index=True)
    truth = pd.concat([s[1] for s in sims], ignore_index=True)
    preds, fits = s7.evaluate(obs, truth, 0.05, 1)
    assert set(preds.method) == set(s7.METHODS) and (preds[preds.method == "B0"].p_event == 0).all()
    assert len(preds) == 5 * len(truth)                                   # tiap site tepat sekali per metode
    fm = s7.farm_table(preds)
    b = fm.groupby("method").brier.mean()
    assert b["A"] < b["B4"]
    assert fm[fm.method == "B0"]["recall_0.05"].max() == 0


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
