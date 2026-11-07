"""Tes S5a dengan data sintetis. Jalankan: python pilot/tests/test_s5_events.py (atau pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s5_events as s5  # noqa: E402


def world(seqs, xy=None, fe0=(), reg_bad=()):
    """seqs: {site: 'DDUDAA'}; D=terdeteksi, U=observable False, A=absen valid. Epoch '2030_01'.. (urut leksikal,
    di luar WEAK).
    fe0: epoch (indeks) dengan fe_n_det=0; reg_bad: (site, indeks) dengan reg_ok False."""
    L = max(map(len, seqs.values()))
    seqs = {n: v.ljust(L, "U") for n, v in seqs.items()}      # sama panjang: sisanya tak-observable
    names = list(seqs)
    xy = xy or {n: (100.0 * i, 0.0) for i, n in enumerate(names)}
    sites = pd.DataFrame([dict(farm="f", site_id=n, x=xy[n][0], y=xy[n][1], best_tier="high", n_epochs_det=seqs[n].count("D"))
                          for n in names])
    rows = [dict(farm="f", site_id=n, epoch=f"2030_{e + 1:02d}", detected=c == "D", score=0.5 + 0.01 * e,
                 observable=c != "U", reg_ok=(n, e) not in reg_bad, fe_n_det=0 if e in fe0 else 5)
            for n in names for e, c in enumerate(seqs[n])]
    return pd.DataFrame(rows), sites


def mine(seqs, k=3, m=2, reg_unknown=False, **kw):
    obs, sites = world(seqs, **kw)
    return s5.mine(obs, sites, k, m, reg_unknown)


def ev(seqs, **kw):
    e = mine(seqs, **kw)[2]
    return list(zip(e.type, e.t_before, e.t_after, e.k_run, e.m_run))


def test_states_neighbour_guard_fe0_and_unobservable():
    det = np.array([[1, 1, 0], [0, 0, 0]], bool)
    xy = np.array([[0, 0], [2.0, 0]])
    S = s5.farm_states(det, np.zeros_like(det), xy)
    assert S[0].tolist() == [1, 1, 0] and S[1].tolist() == [-1, -1, 0]      # tetangga 2 m terdeteksi -> U bukan A
    S = s5.farm_states(det, np.zeros_like(det), np.array([[0, 0], [3.5, 0]]))
    assert S[1].tolist() == [0, 0, 0]                                      # > 3 m: tidak ada penjaga
    S = s5.farm_states(det, np.array([[0, 1, 1], [0, 0, 0]], bool), np.array([[0, 0], [50.0, 0]]))
    assert S[0].tolist() == [1, 1, -1]                                     # D menang atas U; absen yang unk -> U
    # lewat mine: fe_n_det == 0 dan observable False -> U, bukan A
    T = mine({"s": "DDDADU"}, fe0=(3,))[0].iloc[0]
    assert (T.n_D, T.n_A, T.n_U) == (4, 0, 2)


def test_u_dropped_before_runs():
    assert ev({"s": "DDUDAA"}) == [("loss", "2030_04", "2030_05", 3, 2)]  # D D U D A A -> loss dengan k=3
    assert ev({"s": "DDUDAA"}, k=4) == []
    assert ev({"s": "DDDUAA"}) == [("loss", "2030_03", "2030_05", 3, 2)]  # interval melompati U


def test_loss_gain_thresholds():
    assert ev({"s": "DDDAA"}) == [("loss", "2030_03", "2030_04", 3, 2)]   # tepat k dan m
    assert ev({"s": "DDAA"}) == []                                          # k-1
    assert ev({"s": "DDDA"}) == []                                          # m-1
    assert ev({"s": "DDDA"}, m=1) == [("loss", "2030_03", "2030_04", 3, 1)]
    assert ev({"s": "AADDD"}) == [("gain", "2030_02", "2030_03", 3, 2)]
    assert ev({"s": "ADDD"}) == [] and ev({"s": "AADD"}) == []
    assert ev({"s": "DDDAADDD"}) == [("loss", "2030_03", "2030_04", 3, 2), ("gain", "2030_05", "2030_06", 3, 2)]
    assert ev({"s": "DDDAAA"}) == [("loss", "2030_03", "2030_04", 3, 3)]    # panjang run sebenarnya


def test_multiple_transitions_one_site():
    e = ev({"s": "DDDAADDDAA"})
    assert [t[0] for t in e] == ["loss", "gain", "loss"] and e[2][1:3] == ("2030_08", "2030_09")


def test_reg_unknown_flag():
    bad = {("s", 3), ("s", 4)}
    assert len(ev({"s": "DDDAAA"}, reg_bad=bad)) == 1
    assert ev({"s": "DDDAAA"}, reg_bad=bad, reg_unknown=True) == []        # tersisa 1 A < m=2


def test_weak_epoch_flag():
    obs, sites = world({"s": "DDDAA"})
    obs["epoch"] = obs.epoch.map(dict(zip(sorted(obs.epoch.unique()), ["2012_06", "2012_07", "2012_08", "2012_09", "2012_10"])))
    e = s5.mine(obs, sites, 3, 2, False)[2]
    assert e.touches_weak_epoch.tolist() == [True]                         # interval 2012_08 -> 2012_09
    assert not mine({"s": "DDDAA"})[2].touches_weak_epoch.iloc[0]


def test_stable_and_flicker():
    seqs = {"st": "D" * 20, "st2": "D" * 14, "st3": "D" * 17 + "AAA", "fl": "DADADADADADA", "fl2": "DADADADADA"[:9],
            "dead": "D" * 5 + "A" * 7}
    neg = mine(seqs)[3].set_index("site_id").type.to_dict()
    assert neg == {"st": "stable", "fl": "flicker"}      # st2 <15 valid; st3 loss; fl2 <10 valid; dead loss
    assert mine(seqs)[2].site_id.tolist() == ["st3", "dead"]
    # epoch U tidak dihitung sebagai valid
    assert "st" not in mine({"st": "D" * 10 + "U" * 10})[3].site_id.tolist()


def test_sample_neg_cap_and_main(tmp_path):
    seqs = {f"s{i}": "D" * 20 for i in range(30)}
    seqs["l"] = "DDD" + "A" * 17
    xy = {n: (10.0 * i, 0.0) for i, n in enumerate(seqs)}
    obs, sites = world(seqs, xy=xy)
    s4 = tmp_path / "s4"
    s4.mkdir()
    sites.to_parquet(s4 / "sites.parquet")
    obs.to_parquet(s4 / "obs.parquet")
    assert s5.main(["--s4", str(s4), "--out", str(tmp_path / "o"), "--neg-per-farm", "5"]) == 0
    c = pd.read_parquet(tmp_path / "o" / "candidates.parquet")
    assert (c.type == "stable").sum() == 5 and (c.type == "loss").sum() == 1
    assert (tmp_path / "o" / "report.md").read_text().startswith("# S5a")


def test_persistent_and_weak_absence_is_u():
    e = mine({"a": "DDDAAA", "b": "DDDAAD", "c": "AAADDD", "d": "AAADDDAA"}, m=2)[2]
    p = dict(zip(e.site_id + e.type, e.persistent))
    assert p["aloss"] and not p["bloss"] and p["cgain"] and not p["dgain"]
    # absen di epoch WEAK = U: DDDD A A(2020_06) hanya punya 1 A valid -> tidak ada loss
    obs, sites = world({"a": "DDDDAA"})
    obs["epoch"] = obs.epoch.str.replace("2030", "2020")                  # indeks 5 -> 2020_06
    assert s5.mine(obs, sites, 3, 2, False)[2].empty


def test_nb_rate_stable_neighbours_only():
    seqs = {"c": "DDDAAAAAAAAA", "n1": "D" * 12, "n5": "D" * 10 + "AA", "wk": "D" * 6 + "A" * 6, "far": "D" * 12,
            "g": "AAADDDDDDDDD", "m1": "AA" + "D" * 10, "lone": "DDDAAAAAAAAA"}
    xy = {"c": (0, 0), "n1": (10, 0), "n5": (5, 0), "wk": (20, 0), "far": (50, 0), "g": (100, 0), "m1": (110, 0), "lone": (500, 0)}
    obs, sites = world(seqs, xy=xy)
    e = s5.mine(obs, sites, 3, 2, False)[2]
    r = s5.nb_rate(e, obs, sites)
    got = {(a, b): round(v, 4) if v == v else v for a, b, v in zip(e.site_id, e.type, r)}
    assert got[("c", "loss")] == round(16 / 18, 4)          # n1 + n5 pada epoch >= t_after; wk (tak stabil) & far diabaikan
    assert got[("g", "gain")] == round(1 / 3, 4)            # m1 pada epoch <= t_before
    assert np.isnan(got[("lone", "loss")])                  # tanpa tetangga
    neg = s5.negatives(*s5.mine(obs, sites, 3, 2, False)[:1], e)
    assert s5.nb_rate(neg, obs, sites).isna().all()         # stable/flicker -> NaN
