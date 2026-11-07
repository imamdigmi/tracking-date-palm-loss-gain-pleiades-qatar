"""Eksperimen eksploratori pasca-G2 (E3, E2, E1-auto) — pre-registrasi: vault Process/025 bagian 5.

EKSPLORATORI, bukan pengganti G2 (verdict FAIL / NO-GO tidak berubah). Kalimat wajib bila dilaporkan:
"analisis eksploratori pasca-G2, tidak dipre-registrasi sebelum G2".
Jalankan: python pilot/explore_g2.py {e3,e2,e1,all} [--root pilot/out] [--out pilot/out/explore] [--boot 2000]
Input di --root: s4/{obs,sites}.parquet, s5/candidates.parquet, s6/synth, g2/{labels,preds}.parquet. Tidak menulis ke g2/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7_g1 as s7  # noqa: E402
import s8_g2 as s8  # noqa: E402
from s6_synth import md  # noqa: E402

HERE = Path(__file__).resolve().parent
MS = ("B3", "B4", "B5", "A")
NOTE = "_Analisis eksploratori pasca-G2, tidak dipre-registrasi sebelum G2. Verdict G2 (FAIL) dan NO-GO tidak berubah._\n"


# ---------- metrik numpy (tanpa sklearn) ----------
def auc(s, y):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def roc_points(s, y):
    y = np.asarray(y, bool)
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    last = np.r_[np.diff(s) != 0, True]   # satu titik per skor berbeda (tie ditangani)
    tp, fp = np.cumsum(y)[last], np.cumsum(~y)[last]
    return np.r_[0, fp / (~y).sum()], np.r_[0, tp / y.sum()]


def ap(s, y):
    """Average precision (gaya sklearn: sum (R_k - R_k-1) P_k pada skor berbeda)."""
    y = np.asarray(y, bool)
    if not y.any():
        return np.nan
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    last = np.r_[np.diff(s) != 0, True]
    tp, k = np.cumsum(y)[last], (np.arange(len(y)) + 1)[last]
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * tp / k))


def murphy(p, y, K=10):
    """Brier = reliability - resolution + uncertainty (bin quantile; eksak bila p konstan di dalam bin)."""
    y = np.asarray(y, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, K + 1)))
    b = np.searchsorted(edges[1:-1], p, side="right")
    rel = res = 0.0
    ob = y.mean()
    for k in np.unique(b):
        m = b == k
        rel += m.mean() * (p[m].mean() - y[m].mean()) ** 2
        res += m.mean() * (y[m].mean() - ob) ** 2
    return dict(rel=rel, res=res, unc=ob * (1 - ob), brier=np.mean((p - y) ** 2))


def reliability_bins(p, y, K=10):
    y = np.asarray(y, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, K + 1)))
    b = np.searchsorted(edges[1:-1], p, side="right")
    return [dict(bin=int(k), n=int((b == k).sum()), mean_p=p[b == k].mean(), obs_rate=y[b == k].mean()) for k in np.unique(b)]


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def selfcheck():
    r = np.random.default_rng(1)
    y = r.random(4000) < 0.3
    assert auc(y.astype(float), y) == 1 and ap(y.astype(float), y) == 1
    assert abs(auc(r.random(4000), y) - 0.5) < 0.03
    assert auc(np.zeros(10), np.arange(10) < 5) == 0.5   # skor konstan -> 0.5 (tie)
    p = r.random(4000)
    m = murphy(p, r.random(4000) < p)   # identitas hanya eksak bila p konstan per bin; selisih = varians dalam-bin (kecil)
    assert abs(m["brier"] - (m["rel"] - m["res"] + m["unc"])) < 5e-3 and m["rel"] < 0.01


# ---------- E3 ----------
def metric_row(g, idx):
    """g: dict array per metode (p, y, tm, ok). Metrik satu metode pada indeks idx."""
    p, y = g["p"][idx], g["y"][idx]
    mu = murphy(p, y)
    e = y
    return dict(auc=auc(p, y), ap_neg=ap(1 - p, ~y), ap_event=ap(p, y), brier=mu["brier"], reliability=mu["rel"],
                resolution=mu["res"], uncertainty=mu["unc"], logloss=logloss(p, y),
                acc_type=g["tm"][idx][e].mean() if e.any() else np.nan, acc_type_epoch=g["ok"][idx][e].mean() if e.any() else np.nan)


def e3(root, out, boot, seed=0):
    selfcheck()
    out.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(root / "g2" / "preds.parquet")
    G, ref = {}, None
    for m in MS:
        d = pr[pr.method == m].sort_values("site_id").reset_index(drop=True)
        ref = d if ref is None else ref
        assert (d.site_id == ref.site_id).all() and (d.true_type == ref.true_type).all()
        G[m] = dict(p=d.p_event.to_numpy(), y=d.true_type.isin(["loss", "gain"]).to_numpy(), tm=(d.pred_type == d.true_type).to_numpy(),
                    ok=d.ok.to_numpy())
    farms = ref.farm.to_numpy()
    uf = sorted(set(farms))
    fidx = [np.flatnonzero(farms == f) for f in uf]
    y = G["A"]["y"]
    nev, nneg = int(y.sum()), int((~y).sum())
    allidx = np.arange(len(y))
    pt = {m: metric_row(G[m], allidx) for m in MS}
    pd.DataFrame(pt).T.rename_axis("method").round(4).to_csv(out / "metrics.csv")
    # bootstrap berpasangan A - {B5,B4,B3}
    base = ("B5", "B4", "B3")
    acc = []
    rng = np.random.default_rng(seed)
    for _ in range(boot):
        idx = np.concatenate([fidx[k] for k in rng.integers(0, len(uf), len(uf))])
        r = {m: metric_row(G[m], idx) for m in ("A",) + base}
        acc.append({b: {k: r["A"][k] - r[b][k] for k in r["A"]} for b in base})
    rows, drops = [], {}
    for b in base:
        for k in pt["A"]:
            v = np.array([a[b][k] for a in acc])
            ok = np.isfinite(v)
            drops[k] = int((~ok).sum())
            lo, hi = np.percentile(v[ok], [2.5, 97.5])
            rows.append(dict(vs=b, metric=k, point=pt["A"][k] - pt[b][k], ci_lo=lo, ci_hi=hi, n_boot_dropped=drops[k]))
    diff = pd.DataFrame(rows)
    diff.round(4).to_csv(out / "diff_A_minus_baseline.csv", index=False)
    # data grafik
    rb = pd.DataFrame([dict(method=m, **r) for m in MS for r in reliability_bins(G[m]["p"], y)])
    rb.round(4).to_csv(out / "reliability_bins.csv", index=False)
    rp = []
    for m in MS:
        fpr, tpr = roc_points(G[m]["p"], y)
        rp += [dict(method=m, fpr=a, tpr=b) for a, b in zip(fpr, tpr)]
    pd.DataFrame(rp).round(4).to_csv(out / "roc_points.csv", index=False)
    # LOFO dBrier(A - B5), Brier gabungan
    lf = []
    for f, ix in zip(uf, fidx):
        keep = np.setdiff1d(allidx, ix)
        lf.append(dict(farm=f, n_sites=len(ix), dBrier=np.mean((G["A"]["p"][keep] - y[keep]) ** 2) - np.mean((G["B5"]["p"][keep] - y[keep]) ** 2)))
    lofo = pd.DataFrame(lf)
    lofo.round(5).to_csv(out / "lofo_dbrier.csv", index=False)
    # aturan interpretasi pre-registrasi
    d5 = diff[diff.vs == "B5"].set_index("metric")
    dB, dAUC = d5.loc["brier"], d5.loc["auc"]
    cal_only = bool(dB.point < 0 and pt["A"]["reliability"] < pt["B5"]["reliability"]
                    and (abs(dAUC.point) < 0.03 or dAUC.ci_lo <= 0 <= dAUC.ci_hi))
    discr = bool(dAUC.ci_lo > 0)
    timing = bool(d5.loc["acc_type_epoch"].ci_lo > 0)
    fragile = bool((np.sign(lofo.dBrier) != np.sign(dB.point)).any())
    low_auc = bool(all(pt[m]["auc"] < 0.65 for m in MS))
    label = "kalibrasi saja" if cal_only else "bukan (hanya) kalibrasi"
    s = dict(experiment="E3", n_sites=len(y), n_event=nev, n_negatif=nneg, n_farm=len(uf), boot=boot, seed=seed,
             base_rate_ap_neg=nneg / len(y), base_rate_ap_event=nev / len(y), n_boot_dropped=drops,
             dBrier_A_B5=dict(point=dB.point, ci=[dB.ci_lo, dB.ci_hi]), dAUC_A_B5=dict(point=dAUC.point, ci=[dAUC.ci_lo, dAUC.ci_hi]),
             reliability_A=pt["A"]["reliability"], reliability_B5=pt["B5"]["reliability"],
             auc={m: pt[m]["auc"] for m in MS},
             lofo_sign_changes=int((np.sign(lofo.dBrier) != np.sign(dB.point)).sum()),
             labels=dict(keunggulan_A=label, A_lebih_diskriminatif=discr, A_lebih_baik_timing=timing, rapuh_LOFO=fragile,
                         semua_AUC_lt_065=low_auc))
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    (out / "report.md").write_text(f"""# E3 — metrik bebas ambang pada preds G2
{NOTE}
Data: {len(y)} site ({nev} event, {nneg} negatif, {len(uf)} farm). Bootstrap farm {boot}x seed {seed}; baseline AP-negatif = {nneg / len(y):.3f}, AP-event = {nev / len(y):.3f}.

## Metrik per metode
{md(pd.DataFrame(pt).T.round(4).rename_axis("method").reset_index())}

## Selisih A - baseline (CI bootstrap farm 95%; n_boot_dropped = replikasi tanpa negatif/event untuk metrik itu)
{md(diff.round(4))}

## LOFO dBrier(A-B5) (Brier gabungan)
min {lofo.dBrier.min():+.4f}, maks {lofo.dBrier.max():+.4f}; berganti tanda: {s['lofo_sign_changes']} dari {len(uf)} farm.

## Kesimpulan otomatis (aturan pre-registrasi)
- **Keunggulan A atas B5: {label}** (dBrier {dB.point:+.4f}; reliability A {pt['A']['reliability']:.4f} vs B5 {pt['B5']['reliability']:.4f}; dAUC {dAUC.point:+.4f} CI [{dAUC.ci_lo:+.4f}, {dAUC.ci_hi:+.4f}]).
- A lebih diskriminatif: {'ya' if discr else 'tidak'}. A lebih baik dalam timing (acc jenis+epoch): {'ya' if timing else 'tidak'}.
- Rapuh (dBrier berganti tanda pada >= 1 farm LOFO): {'ya' if fragile else 'tidak'}.
- Semua AUC < 0,65 ('skor tidak memisahkan event dari hard negative'; tidak membedakan masalah model vs desain): {'ya' if low_auc else 'tidak'}.
""")
    return s


# ---------- util run S6 + prediksi ----------
class Ctx:
    def __init__(self, root, boot):
        self.root, self.boot = root, boot
        gt = yaml.safe_load(open(HERE / "gates.yaml"))
        g1 = gt["g1_synthetic"]
        self.fa, self.tol = g1["fixed_fa"], g1["event_time_tol_epochs"]
        self.fas = (self.fa, 0.01, 0.10)
        self.obs = pd.read_parquet(root / "s4" / "obs.parquet")
        self.labels = pd.read_parquet(root / "g2" / "labels.parquet")
        self.truth = s8.real_truth(self.labels)

    def params(self, src):
        s6 = self.root / "s6" / "synth"
        o6 = pd.read_parquet(s6 / "obs.parquet", columns=s7.OBS_COLS)
        t6 = pd.read_csv(s6 / "truth_events.csv", keep_default_na=False, dtype=str)
        o6, t6 = o6[o6.run == src], t6[t6.run == src]
        R6, _, _ = s7.load_runs(o6, t6, {f: 0 for f in t6.farm.unique()})[src]
        return s7.fit(R6, np.ones(len(R6["farm"]), bool), self.fa, self.tol)

    def preds(self, truth, P):
        """Pipeline s8 pada truth sembarang (fold kalibrasi = paritas urutan farm di truth itu)."""
        obs = s8.real_obs(self.obs, truth)
        farms = sorted(truth.farm.unique())
        R, eps, _ = s7.load_runs(obs[s7.OBS_COLS + ["fe_nobs"]], truth, {f: i % 2 for i, f in enumerate(farms)})["real"]
        return s8.predict(R, eps, P, self.tol)

    def summarize(self, preds):
        """Per metode: brier, recall@fas gabungan, AUC. Plus tabel farm untuk perbandingan A-B5."""
        fm = s8.farm_table(preds, self.fas)
        rows = []
        for m, g in preds.groupby("method", sort=False):
            p, ev, ok = g.p_event.to_numpy(), g.true_type.isin(["loss", "gain"]).to_numpy(), g.ok.to_numpy()
            r = dict(method=m, n=len(g), n_event=int(ev.sum()), brier=np.mean((p - ev) ** 2), auc=auc(p, ev))
            r.update({f"recall_{fa:g}": s7.recall_at_fa(p, ok, ev, fa) for fa in self.fas})
            rows.append(r)
        return pd.DataFrame(rows), fm

    def cmp(self, fm, b="B5"):
        fm = fm.copy()
        v = s7.compare(fm, "real", "A", b, f"recall_{self.fa:g}", reps=self.boot, seed=0)
        return v


# ---------- E2 ----------
def e2_runs(root):
    t6 = pd.read_csv(root / "s6" / "synth" / "truth_events.csv", keep_default_na=False, dtype=str, usecols=["run"]).run.unique()
    pre = [f"mismatched|st1|n20|q{q}|ev{e}|seed{s}" for q in (0.5, 1.5) for e in (0.1, 0.25, 0.5) for s in (0, 1, 2)]
    miss = [r for r in pre if r not in t6]
    assert not miss, f"run E2 tidak ada: {miss}"
    sec = [r.replace("mismatched", "matched", 1) for r in pre if r.replace("mismatched", "matched", 1) in t6]
    return pre, sec


def e2_one(C, run, out):
    slug = run.replace("|", "_")
    d = out / "runs" / slug
    s8.main(["--s6", str(C.root / "s6" / "synth"), "--obs", str(C.root / "s4" / "obs.parquet"), "--labels", str(C.root / "g2" / "labels.parquet"),
             "--out", str(d), "--src", run, "--boot", str(C.boot)])
    pr, fm = pd.read_parquet(d / "preds.parquet"), pd.read_csv(d / "farm_metrics.csv", dtype={"farm": str})  # str: urutan farm = s8 (bootstrap identik)
    summ, _ = C.summarize(pr)
    s = summ.set_index("method")
    v = C.cmp(fm)
    return dict(run=run, **{f"brier_{m}": s.loc[m, "brier"] for m in MS}, **{f"recall05_{m}": s.loc[m, "recall_0.05"] for m in MS},
                **{f"auc_{m}": s.loc[m, "auc"] for m in MS}, dBrier_pooled=s.loc["A", "brier"] - s.loc["B5", "brier"],
                dRecall_pooled=s.loc["A", "recall_0.05"] - s.loc["B5", "recall_0.05"], dBrier_farmmean=v["dB"], ciB_lo=v["ciB"][0],
                ciB_hi=v["ciB"][1], dRecall_farmmean=v["dR"], ciR_lo=v["ciR"][0], ciR_hi=v["ciR"][1],
                A_menang=("ya" if v["win"] else "tidak"))


def e2(root, out, boot, C=None):
    out.mkdir(parents=True, exist_ok=True)
    C = C or Ctx(root, boot)
    pre, sec = e2_runs(root)
    df = pd.DataFrame([e2_one(C, r, out) for r in pre])
    df.round(4).to_csv(out / "runs.csv", index=False)
    if sec:
        pd.DataFrame([e2_one(C, r, out) for r in sec]).round(4).to_csv(out / "runs_secondary_matched.csv", index=False)
    orig = df[df.run == s8.SRC].iloc[0]
    # sanity: run asli harus identik dengan preds G2 yang ada
    a = pd.read_parquet(root / "g2" / "preds.parquet").sort_values(["method", "site_id"]).reset_index(drop=True)
    b = pd.read_parquet(out / "runs" / s8.SRC.replace("|", "_") / "preds.parquet").sort_values(["method", "site_id"]).reset_index(drop=True)
    repro = bool(np.allclose(a.p_event, b.p_event) and (a.ok == b.ok).all())
    n_win = int((df.dBrier_pooled < 0).sum())
    med = float(df.dBrier_pooled.median())
    rng_ = float(df.dBrier_pooled.max() - df.dBrier_pooled.min())
    width = float(orig.ciB_hi - orig.ciB_lo)
    if n_win >= 15 and med < 0:
        label = "arah robust"
    elif n_win <= 12:
        label = "sensitif terhadap parameter"
    else:
        label = "tidak meyakinkan"
    s = dict(experiment="E2", n_runs=len(df), runs_A_brier_lt_B5=n_win, median_dBrier_pooled=med, range_dBrier_pooled=rng_,
             ci_width_g2_original=width, range_gt_ci_width=bool(rng_ > width), runs_verdict_A_menang=int((df.A_menang == "ya").sum()),
             reproduces_g2_preds=repro, label=label, secondary_matched_n=len(sec))
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    cols = ["run", "brier_B5", "brier_A", "dBrier_pooled", "dBrier_farmmean", "ciB_lo", "ciB_hi", "dRecall_pooled", "A_menang"]
    (out / "report.md").write_text(f"""# E2 — sensitivitas parameter terhadap run sumber S6
{NOTE}
18 run mismatched|st1|n20 (q x ev x seed). Run asli G2 = `{s8.SRC}`; reproduksi preds G2 asli: {'ya' if repro else '**TIDAK**'}. CI = bootstrap farm {boot}x seed 0 (rata-rata per farm, seperti G2).

{md(df[cols].round(4))}

- Run dengan Brier gabungan A < B5: **{n_win}/{len(df)}**; median dBrier gabungan {med:+.4f}.
- Rentang dBrier (maks - min) = {rng_:.4f} vs lebar CI G2 asli {width:.4f}: {'ketidakpastian parameter LEBIH BESAR dari ketidakpastian sampling' if rng_ > width else 'tidak lebih besar dari ketidakpastian sampling'}.
- Run dengan verdict 'A menang' (dilaporkan apa adanya, bukan verdict G2): {int((df.A_menang == 'ya').sum())}/{len(df)}.
- Sekunder (matched, hanya dilaporkan): {len(sec)} run di `runs_secondary_matched.csv`.

## Kesimpulan otomatis
**{label}** (aturan: robust >= 15/18 dan median < 0; sensitif <= 12/18; 13-14 tidak meyakinkan). Memilih run terbaik post-hoc dilarang.
""")
    return s


# ---------- E1-auto ----------
def silver_sample(sites, obs, cand, g2farms, seed=20261107):
    """Populasi non-kandidat tier high, observable>=12, frac deteksi>=0,9; 10 site acak per farm (deterministik)."""
    o = obs[obs.site_id.isin(sites.site_id)]
    c = o.groupby("site_id").agg(n_obs=("observable", "sum"), n_det=("detected", "sum"))
    c["frac"] = np.minimum(1, c.n_det / c.n_obs.where(c.n_obs > 0))
    s = sites[sites.farm.isin(g2farms) & ~sites.site_id.isin(cand.site_id) & (sites.best_tier == "high")].merge(
        c, left_on="site_id", right_index=True)
    U = s[(s.n_obs >= 12) & (s.frac >= 0.9)].sort_values("site_id")
    rng = np.random.default_rng(seed)
    pick = []
    for f in sorted(U.farm.unique()):
        ids = U[U.farm == f].site_id.to_numpy()
        pick += list(rng.choice(ids, 10, replace=False)) if len(ids) > 10 else list(ids)
    return U, pd.DataFrame(dict(farm=U.set_index("site_id").loc[pick, "farm"].to_numpy(), site_id=pick, epoch="", type="stable", run="real"))


def e1(root, out, boot, C=None):
    out.mkdir(parents=True, exist_ok=True)
    C = C or Ctx(root, boot)
    sites = pd.read_parquet(root / "s4" / "sites.parquet")
    cand = pd.read_parquet(root / "s5" / "candidates.parquet", columns=["site_id"])
    U, silver = silver_sample(sites, C.obs[["site_id", "observable", "detected"]], cand, set(C.truth.farm))
    silver.to_csv(out / "silver_sample.csv", index=False)
    P = C.params(s8.SRC)
    ev = C.truth[C.truth.type.isin(["loss", "gain"])]
    sets = {"a_G2_asli": C.truth, "b_event+silver": pd.concat([ev, silver], ignore_index=True),
            "c_event+hard+silver": pd.concat([C.truth, silver], ignore_index=True)}
    rows, cmps = [], []
    for name, t in sets.items():
        assert t.site_id.is_unique
        summ, fm = C.summarize(C.preds(t, P))
        rows.append(summ.assign(set=name))
        for b in ("B5", "B4", "B3"):
            v = C.cmp(fm, b)
            cmps.append(dict(set=name, vs=b, dBrier=v["dB"], ciB_lo=v["ciB"][0], ciB_hi=v["ciB"][1], dRecall=v["dR"], ciR_lo=v["ciR"][0],
                             ciR_hi=v["ciR"][1], n_farm_boot=int(fm[fm.method == "A"].n_events.gt(0).sum())))
    res = pd.concat(rows, ignore_index=True)
    res = res[["set"] + [c for c in res.columns if c != "set"]]
    res.round(4).to_csv(out / "metrics_by_set.csv", index=False)
    cm = pd.DataFrame(cmps)
    cm.round(4).to_csv(out / "diff_A_minus_baseline.csv", index=False)
    b = res[res.set == "b_event+silver"].set_index("method")
    hmm = ("B4", "B5", "A")
    supported = bool(all(b.loc[m, "recall_0.05"] >= 0.30 and b.loc[m, "auc"] >= 0.80 for m in hmm))
    rejected = bool(b.loc["A", "auc"] < 0.70 and b.loc["B5", "auc"] < 0.70)
    diag = "didukung" if supported else ("ditolak" if rejected else "tidak meyakinkan")
    c5 = cm[cm.vs == "B5"].set_index("set")
    consistent = bool(all(c5.loc[k, "dBrier"] < 0 and c5.loc[k, "dRecall"] > 0 for k in ("b_event+silver", "c_event+hard+silver")))
    s = dict(experiment="E1-auto", n_population=len(U), n_population_farms=int(U.farm.nunique()), n_silver=len(silver),
             n_silver_farms=int(silver.farm.nunique()), diagnosis_hard_negative=diag, A_vs_B5_konsisten=consistent,
             set_b=dict(recall05={m: b.loc[m, "recall_0.05"] for m in hmm}, auc={m: b.loc[m, "auc"] for m in hmm}))
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    (out / "report.md").write_text(f"""# E1-auto — negatif silver dari site non-kandidat S5
{NOTE}
**PERINGATAN SIRKULARITAS:** negatif silver dipilih dari pola deteksi yang juga menjadi input metode (fraksi deteksi >= 0,9), dan tidak diverifikasi manusia. Set (b) pasti mudah; AUC tinggi di (b) sebagian tautologis. E1-auto hanya menguji alarm palsu pada pohon yang terdeteksi stabil; BUKAN estimasi FPR populasi.

Populasi U: {len(U)} site di {U.farm.nunique()} farm; sampel silver: {len(silver)} site di {silver.farm.nunique()} farm (seed 20261107). Kalibrasi isotonic dihitung ulang tiap set. Bootstrap farm {boot}x seed 0 (farm dengan event).

## Metrik per set (recall: FPR gabungan)
{md(res.round(4))}

## A - baseline per set (CI bootstrap farm)
{md(cm.round(4))}

## Kesimpulan otomatis
- Diagnosis 'hard negative menyebabkan keruntuhan recall': **{diag}** (didukung bila recall@0,05 B4,B5,A di (b) >= 0,30 dan AUC >= 0,80; ditolak bila AUC (b) < 0,70 untuk A dan B5).
- A vs B5 konsisten (dBrier < 0 dan dRecall@0,05 > 0 di (b) dan (c)): **{'ya' if consistent else 'tidak'}**. Signifikansi tetap eksploratori.
""")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("which", choices=["e3", "e2", "e1", "all"])
    ap.add_argument("--root", default=str(HERE / "out"))
    ap.add_argument("--out", default=str(HERE / "out" / "explore"))
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    assert out.resolve() != (root / "g2").resolve(), "jangan menimpa g2/"
    C = None
    for k in ("e3", "e2", "e1"):
        if a.which in (k, "all"):
            t = time.time()
            if k != "e3":
                C = C or Ctx(root, a.boot)
            s = {"e3": lambda: e3(root, out / "E3", a.boot), "e2": lambda: e2(root, out / "E2", a.boot, C),
                 "e1": lambda: e1(root, out / "E1", a.boot, C)}[k]()
            print(f"== {k.upper()} ({time.time() - t:.0f}s)\n{json.dumps(s, indent=1, default=float)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
