"""S7 — evaluasi G1: model observabilitas A vs baseline B3 (run-length baseline, terinspirasi Reiner, bukan implementasi setia), B4, B5 (HMM) pada benchmark sintetis S6.

Semua metode keluarkan per site p_loss, p_gain, p_event, jenis & epoch perubahan. B0 = 'selalu tidak ada perubahan' (hanya
untuk membuktikan gate gagal). B3 = aturan run-length (K,M dituning di train farm). B4/B5/A = HMM 2-state Present/Absent
dengan paling banyak 1 perubahan; beda hanya di p_det (B4 konstan, B5 shrinkage per farm-epoch, A = sigmoid(w^T q) dan
eps = sigmoid(v^T q), dilatih dari presence TRUE sintetis). Baris tak-observable tidak masuk likelihood.
Protokol: 2-fold farm held-out (paritas urutan farm). Parameter + kalibrasi isotonic p_event dilatih di run sumber pada
farm fold lain: mismatched/stress/null -> run mismatched yang sama (null: event terkecil); matched -> dirinya sendiri.
Metrik per farm (pool semua run family): Brier, recall@FPR (jenis benar & |epoch| <= toleransi). Verdict = bootstrap farm.
Aturan dari pilot/gates.yaml (g1_synthetic). Awas: matched hanya pembanding; stress held-out; G1 hanya bisa MENOLAK.
Jalankan: python pilot/s7_g1.py [--s6 pilot/out/s6/synth] [--out pilot/out/s7]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import isotonic_regression, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s6_synth import md, sigmoid  # noqa: E402

HERE = Path(__file__).resolve().parent
METHODS = ("B0", "B3", "B4", "B5", "A")
PSEUDO = 20          # pseudo-count shrinkage B5
P_STABLE_ABSENT = 0.01   # porsi sisa massa prior untuk 'stable-absent' (penyerap site sampah)
CLIP = 1e-4
OBS_COLS = ["farm", "site_id", "epoch", "detected", "score", "observable", "gsd", "nodata_pct", "reg_resid_m",
            "reg_ok", "dt_days", "fe_n_det", "run"]


# ---------- data -> struktur per run ----------
def load_runs(obs, truth, farm_fold):
    """Return {run: (R, eps, family)}; R = dict array per-site (axis 0 = site), obs/truth sama urutan (grid penuh)."""
    runs = {}
    tix = truth.groupby("run", sort=False).indices
    for name, ix in obs.groupby("run", sort=False).indices.items():
        o, t = obs.iloc[ix], truth.iloc[tix[name]]
        ns = len(t)
        n = len(o) // ns
        assert len(o) == ns * n and (o.site_id.to_numpy()[::n] == t.site_id.to_numpy()).all(), name
        eps = o.epoch.to_numpy()[:n]
        pos = {e: i for i, e in enumerate(eps)}
        tt = np.array([pos.get(e, -1) for e in t.epoch])
        ty = t.type.to_numpy()
        col = np.arange(n)[None, :]
        present = np.select([ty[:, None] == "loss", ty[:, None] == "gain", ty[:, None] == "flicker", ty[:, None] == "absent"],
                            [col < tt[:, None], col >= tt[:, None], col != tt[:, None], False], True)
        g = lambda c: o[c].to_numpy().reshape(ns, n)  # noqa: E731
        farm = t.farm.to_numpy()
        code = pd.factorize(farm)[0]
        nobs = np.zeros((code.max() + 1, n))
        np.add.at(nobs, code, g("observable").astype(float))
        nobs = nobs[code] if "fe_nobs" not in o else g("fe_nobs").astype(float)   # fe_nobs: subset site (G2) tapi laju farm-epoch penuh
        resid = np.minimum(np.where(np.isnan(g("reg_resid_m")), np.where(g("reg_ok"), 0.0, 0.8), g("reg_resid_m")), 1.5)
        qb = np.stack([np.ones((ns, n)), g("gsd"), resid, g("reg_ok").astype(float), np.nan_to_num(g("nodata_pct")) / 10], -1)
        R = dict(farm=farm, fold=np.array([farm_fold[f] for f in farm]), det=g("detected"), obs=g("observable"),
                 score=g("score"), dt=g("dt_days"), fe_n=g("fe_n_det").astype(float), nobs=nobs, qb=qb,
                 ttype=ty, tidx=tt, present=present, site_id=t.site_id.to_numpy())
        runs[name] = (R, eps, name.split("|")[0])
    return runs


def sub(R, m):
    return {k: v[m] for k, v in R.items()}


def source_name(name, fam, ev_null):
    if fam == "matched":
        return name
    p = name.split("|")
    p[0] = "mismatched"
    if fam == "null":
        p[4] = f"ev{ev_null:g}"
    return "|".join(p)


# ---------- B3: aturan run-length pada epoch observable ----------
def b3(det, obs, score, K, M):
    """Return (skor, tipe, epoch-grid). Redeteksi setelah gap = misdetection -> bukan event (sisi absent harus bersih),
    jadi fraksi non-deteksi sisi absent selalu 1 dan skor = rata-rata score sisi present."""
    ns, n = det.shape
    order = np.argsort(~obs, axis=1, kind="stable")           # epoch observable dulu, urutan terjaga
    d = np.take_along_axis(det & obs, order, 1).astype(float)
    s = np.take_along_axis(score, order, 1)
    m = obs.sum(1)[:, None]
    D = np.pad(np.cumsum(d, 1), ((0, 0), (1, 0)))
    S = np.pad(np.cumsum(s, 1), ((0, 0), (1, 0)))
    r = np.arange(1, n)[None, :]
    at = lambda A, i: np.take_along_axis(A, np.broadcast_to(np.clip(i, 0, n), (ns, n - 1)), 1)  # noqa: E731
    Dr, Dm, Sr, Sm = at(D, r), np.take_along_axis(D, m, 1), at(S, r), np.take_along_axis(S, m, 1)
    loss_ok = (r >= K) & (Dr - at(D, r - K) == K) & (m - r >= M) & (Dm - Dr == 0)
    gain_ok = (r >= M) & (Dr == 0) & (m - r >= K) & (at(D, r + K) - Dr == K)
    sl = np.where(loss_ok, Sr / r, -1.0)
    sg = np.where(gain_ok, (Sm - Sr) / np.maximum(m - r, 1), -1.0)
    kl, kg = sl.argmax(1), sg.argmax(1)
    bl, bg = sl.max(1), sg.max(1)
    is_gain = bg > bl
    rr = np.where(is_gain, kg, kl) + 1                        # boundary terkompres
    sc = np.maximum(bl, bg)
    ep = np.take_along_axis(order, np.minimum(rr, n - 1)[:, None], 1)[:, 0]
    return np.maximum(sc, 0.0), np.where(is_gain, "gain", "loss"), np.where(sc > 0, ep, 0)


# ---------- inti HMM (shortcut: max 1 perubahan) ----------
def hmm(det, obs, pdet, eps, dt, lam_l, lam_g):
    """Enumerasi eksak: stable-present, stable-absent, loss di batas k, gain di batas k (k=1..n-1).
    shortcut: max 1 perubahan per site, ganti ke forward-backward penuh bila data nyata punya >1 perubahan
    Return p_loss, p_gain, epoch MAP loss, epoch MAP gain (indeks grid)."""
    ns, n = det.shape
    pdet, eps = np.clip(pdet, CLIP, 1 - CLIP), np.clip(eps, CLIP, 1 - CLIP)
    c = lambda p: np.pad(np.cumsum(np.where(obs, np.where(det, np.log(p), np.log1p(-p)), 0.0), 1), ((0, 0), (1, 0)))  # noqa: E731
    C1, C0 = c(pdet), c(eps)
    k = np.arange(1, n)
    pl, pg = 1 - np.exp(-lam_l * np.nan_to_num(dt[:, 1:])), 1 - np.exp(-lam_g * np.nan_to_num(dt[:, 1:]))
    rest = np.maximum(1 - pl.sum(1) - pg.sum(1), 1e-6)
    ll = np.concatenate([
        (C1[:, n] + np.log(rest * (1 - P_STABLE_ABSENT)))[:, None], (C0[:, n] + np.log(rest * P_STABLE_ABSENT))[:, None],
        C1[:, k] + C0[:, [n]] - C0[:, k] + np.log(np.maximum(pl, 1e-300)),
        C0[:, k] + C1[:, [n]] - C1[:, k] + np.log(np.maximum(pg, 1e-300))], 1)
    post = np.exp(ll - ll.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    L, G = post[:, 2:n + 1], post[:, n + 1:]
    return L.sum(1), G.sum(1), 1 + L.argmax(1), 1 + G.argmax(1)


def fit_logit(X, y, ridge=1e-3):
    def f(w):
        z = X @ w
        g = X.T @ (sigmoid(z) - y) / len(y)
        g[1:] += 2 * ridge * w[1:]
        return np.mean(np.logaddexp(0, z) - y * z) + ridge * w[1:] @ w[1:], g
    with np.errstate(all="ignore"):          # line search L-BFGS sesekali melompat jauh; hasil akhir tetap finit
        return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def shrunk_rate(R, p0):
    """B5: laju deteksi farm-epoch dishrink ke p0 (konstanta B4)."""
    return (R["fe_n"] + PSEUDO * p0) / (R["nobs"] + PSEUDO)


def qmat(R, p0):
    r = np.clip(shrunk_rate(R, p0), CLIP, 1 - CLIP)
    return np.concatenate([R["qb"], np.log(r / (1 - r))[..., None]], -1)


# ---------- prediksi mentah & metrik ----------
def lam_of(R):
    span = np.nansum(R["dt"][:, 1:])
    return (R["ttype"] == "loss").sum() / span, (R["ttype"] == "gain").sum() / span


def raw_preds(R, P):
    """Return {metode: (p_loss, p_gain, epoch_loss, epoch_gain)} untuk B3, B4, B5, A (belum dikalibrasi)."""
    out = {}
    sc, ty, ep = b3(R["det"], R["obs"], R["score"], P["K"], P["M"])
    out["B3"] = (sc * (ty == "loss"), sc * (ty == "gain"), ep, ep)
    args = (R["det"], R["obs"])
    one = np.ones_like(R["dt"])
    out["B4"] = hmm(*args, P["pdet"] * one, P["eps"] * one, R["dt"], *P["lam"])
    out["B5"] = hmm(*args, shrunk_rate(R, P["pdet"]), P["eps"] * one, R["dt"], *P["lam"])
    Q = qmat(R, P["pdet"])
    out["A"] = hmm(*args, sigmoid(Q @ P["w"]), sigmoid(Q @ P["v"]), R["dt"], *P["lam"])
    return out


def final_type_epoch(pl, pg, el, eg):
    loss = pl >= pg
    return np.where(loss, "loss", "gain"), np.where(loss, el, eg)


def is_ok(R, ptype, pep, tol):
    ev = np.isin(R["ttype"], ["loss", "gain"])
    return ev & (ptype == R["ttype"]) & (np.abs(pep - R["tidx"]) <= tol)


def fa_thr(p, is_ev, fa):
    """Threshold = skor negatif ke-(floor(fa*n_neg)+1) terbesar; flag = p > thr (ketat, skor konstan -> tanpa flag)."""
    neg = np.sort(p[~is_ev])[::-1]
    a = int(np.floor(fa * len(neg) + 1e-9))
    return neg[a] if a < len(neg) else -np.inf


def recall_at_fa(p, ok, is_ev, fa):
    if not is_ev.any():
        return np.nan
    return (ok & (p > fa_thr(p, is_ev, fa))).sum() / is_ev.sum()


def iso_fit(p, y):
    u, inv = np.unique(p, return_inverse=True)
    w = np.bincount(inv).astype(float)
    return u, isotonic_regression(np.bincount(inv, y) / w, weights=w).x


def fit(R, tr, fa, tol):
    """Parameter semua metode + kalibrasi isotonic, dari farm train (run sumber)."""
    S = sub(R, tr)
    pm, am = S["present"] & S["obs"], ~S["present"] & S["obs"]
    P = dict(pdet=np.clip(S["det"][pm].mean(), CLIP, 1 - CLIP), eps=np.clip(S["det"][am].mean(), CLIP, 1 - CLIP), lam=lam_of(S))
    X = qmat(S, P["pdet"])
    P["w"], P["v"] = fit_logit(X[pm], S["det"][pm]), fit_logit(X[am], S["det"][am])
    ev = np.isin(S["ttype"], ["loss", "gain"])

    def rec(K, M):                      # recall@fa rata-rata per farm pada train, untuk tuning B3
        sc, ty, ep = b3(S["det"], S["obs"], S["score"], K, M)
        ok = is_ok(S, ty, ep, tol)
        return np.nanmean([recall_at_fa(sc[g], ok[g], ev[g], fa) for g in (S["farm"] == f for f in np.unique(S["farm"]))])
    P["K"], P["M"] = max(((K, M) for K in (1, 2, 3) for M in (1, 2, 3)), key=lambda km: rec(*km))
    P["iso"] = {m: iso_fit(pl + pg, ev.astype(float)) for m, (pl, pg, _, _) in raw_preds(S, P).items()}
    return P


# ---------- evaluasi ----------
def evaluate(obs, truth, fa, tol, log=lambda *a: None):
    """Return (preds DataFrame, fit_records)."""
    farms = sorted(truth.farm.unique())
    runs = load_runs(obs, truth, {f: i % 2 for i, f in enumerate(farms)})
    evs = [float(n.split("|")[4][2:]) for n, (_, _, fam) in runs.items() if fam == "mismatched"]
    ev_null = min(evs) if evs else 0.0
    cache, rows = {}, []
    for i, (name, (R, eps, fam)) in enumerate(runs.items()):
        src = source_name(name, fam, ev_null)
        if src not in runs:
            sys.exit(f"ERROR: run sumber {src} untuk {name} tidak ada")
        for f in (0, 1):
            te = R["fold"] == f
            if not te.any():
                continue
            if (src, f) not in cache:
                cache[src, f] = fit(runs[src][0], runs[src][0]["fold"] != f, fa, tol)
            P, S = cache[src, f], sub(R, te)
            raw = raw_preds(S, P)
            for m in METHODS:
                if m == "B0":
                    z = np.zeros(len(S["farm"]))
                    pl, pg, el, eg = z, z, z.astype(int), z.astype(int)
                else:
                    pl, pg, el, eg = raw[m]
                    pe = pl + pg
                    cal = np.interp(pe, *P["iso"][m])
                    sc = np.divide(cal, pe, out=np.zeros_like(pe), where=pe > 0)
                    pl, pg = pl * sc, pg * sc
                ty, ep = final_type_epoch(pl, pg, el, eg)
                rows.append(pd.DataFrame(dict(run=name, family=fam, farm=S["farm"], site_id=S["site_id"], method=m, p_loss=pl,
                                              p_gain=pg, p_event=pl + pg, pred_type=ty, pred_epoch=eps[ep],
                                              true_type=S["ttype"], ok=is_ok(S, ty, ep, tol))))
        log(f"[{i + 1}/{len(runs)}] {name}")
    fits = [dict(src_family=src.split("|")[0], pdet=P["pdet"], eps=P["eps"], K=P["K"], M=P["M"],
                 **{f"w{j}": x for j, x in enumerate(P["w"])}, **{f"v{j}": x for j, x in enumerate(P["v"])})
            for (src, _), P in cache.items()]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fits)


def farm_table(preds, fas=(0.05, 0.01, 0.10)):
    rows = []
    for (fam, m, farm), g in preds.groupby(["family", "method", "farm"], sort=False):
        p, y = g.p_event.to_numpy(), g.true_type.isin(["loss", "gain"]).to_numpy()
        r = dict(family=fam, method=m, farm=farm, brier=np.mean((p - y) ** 2), mean_p_event=p.mean(), n_events=int(y.sum()))
        for fa in fas:
            r[f"recall_{fa:g}"] = recall_at_fa(p, g.ok.to_numpy(), y, fa)
        rows.append(r)
    return pd.DataFrame(rows)


def verdict(dB, dR, reps=2000, seed=0):
    """Bootstrap farm (cluster) atas rata-rata selisih. Menang iff CI(dBrier) < 0 dan CI(dRecall) > 0 seluruhnya."""
    dB, dR = np.asarray(dB, float), np.asarray(dR, float)
    ok = np.isfinite(dB) & (np.isfinite(dR) | np.isnan(dR).all())
    dB, dR = dB[ok], dR[ok]
    ix = np.random.default_rng(seed).integers(0, len(dB), (reps, len(dB)))
    ci = lambda x: np.percentile(x[ix].mean(1), [2.5, 97.5])  # noqa: E731
    cb = ci(dB)
    if np.isnan(dR).all():       # null: tanpa event -> recall tidak ada
        return dict(dB=dB.mean(), ciB=cb, dR=np.nan, ciR=(np.nan, np.nan), win=None)
    cr = ci(dR)
    return dict(dB=dB.mean(), ciB=cb, dR=dR.mean(), ciR=cr, win=bool(cb[1] < 0 and cr[0] > 0))


def compare(fm, fam, a, b, col="recall_0.05", **kw):
    t = fm[fm.family == fam].pivot(index="farm", columns="method", values=["brier", col])   # pivot: null punya recall NaN semua
    return verdict(t["brier"][a] - t["brier"][b], t[col][a] - t[col][b], **kw)


def fmt(v):
    ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"  # noqa: E731
    return {"dBrier": f"{v['dB']:+.4f}", "CI_dBrier": ci(v["ciB"]), "dRecall": f"{v['dR']:+.4f}", "CI_dRecall": ci(v["ciR"]),
            "A_menang": "n/a" if v["win"] is None else ("ya" if v["win"] else "tidak")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s6", default=str(HERE / "out" / "s6" / "synth"))
    ap.add_argument("--out", default=str(HERE / "out" / "s7"))
    ap.add_argument("--gates", default=str(HERE / "gates.yaml"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s6 = Path(a.s6)
    for f in ("obs.parquet", "truth_events.csv"):
        if not (s6 / f).is_file():
            sys.exit(f"ERROR: tidak ada {s6 / f} (jalankan s6_synth.py dulu)")
    g = yaml.safe_load(open(a.gates))["g1_synthetic"]
    fa, tol, prim, must = g["fixed_fa"], g["event_time_tol_epochs"], g["primary_family"], g["must_beat"]
    t0 = time.time()
    obs = pd.read_parquet(s6 / "obs.parquet", columns=OBS_COLS)
    truth = pd.read_csv(s6 / "truth_events.csv", keep_default_na=False, dtype=str)   # 'null' bukan NaN
    log = lambda s: print(s, file=sys.stderr)  # noqa: E731
    preds, fits = evaluate(obs, truth, fa, tol, log)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "preds.parquet", index=False)
    fm = farm_table(preds, (fa, 0.01, 0.10))
    fm.to_csv(out / "farm_metrics.csv", index=False)

    rc = f"recall_{fa:g}"
    fams = [f for f in ("null", "matched", "mismatched", "stress") if f in set(fm.family)]
    summ = fm.groupby(["family", "method"])[["brier", rc]].mean().round(4).reset_index()
    summ = summ.pivot(index="method", columns="family", values=["brier", rc]).reindex(list(METHODS))
    summ.columns = [f"{c}/{f}" for c, f in summ.columns]   # "|" merusak tabel markdown
    vrows = [dict(family=f, vs=b, **fmt(compare(fm, f, "A", b, rc, reps=a.boot, seed=a.seed))) for f in fams for b in must]
    wins = [r["A_menang"] == "ya" for r in vrows if r["family"] == prim]
    g1 = "PASS" if wins and all(wins) else "FAIL"
    b0 = compare(fm, prim, "B0", "B4", rc, reps=a.boot, seed=a.seed)
    b0line = ("B0 vs B4 GAGAL (benar): recall B0 = 0" if not b0["win"] else
              "PERINGATAN: B0 LOLOS -> EVALUASI RUSAK, jangan percaya hasil apa pun di laporan ini")
    sens = fm.groupby(["family", "method"])[["recall_0.01", "recall_0.05", "recall_0.1"]].mean().round(4).reset_index()
    fcols = [c for c in fits.columns if c not in ("src_family",)]
    fit_tbl = fits.groupby("src_family")[fcols].mean().round(4).reset_index()
    elapsed = time.time() - t0
    (out / "g1_report.md").write_text(f"""# S7 — evaluasi G1 (A vs B3/B4/B5)

Run {elapsed:.1f}s. Args: {vars(a)}. Gate: fixed_fa={fa}, tol={tol} epoch, primary={prim}, must_beat={must}.
Protokol: 2-fold farm held-out; {preds.run.nunique()} run, {preds.farm.nunique()} farm, prediksi out-of-fold sekali per run.

## Rata-rata per-farm: Brier dan recall@{fa:g} (metode x family)
{md(summ.reset_index())}

## Verdict (A vs baseline; selisih = A - baseline, CI bootstrap farm {a.boot}x, 95%)
Menang iff CI dBrier < 0 dan CI dRecall > 0. `matched` hanya pembanding, `stress` held-out, `null` tanpa recall.
{md(pd.DataFrame(vrows))}

**G1 {g1}** (family utama {prim}: A menang atas {', '.join(must)} = {wins})

Sanity B0: {b0line} (dBrier {b0['dB']:+.4f}, dRecall {b0['dR']:+.4f}, win={b0['win']})

## Sensitivitas FPR (recall rata-rata per farm)
{md(sens)}

## Parameter hasil fit (rata-rata fold x run sumber, per family sumber)
pdet/eps = B4; w0..w5/v0..v5 = A untuk q=[1, gsd, resid, reg_ok, nodata/10, logit laju farm-epoch]; K,M = B3 terpilih.
{md(fit_tbl)}

## Batasan
- Data sintetis; emisi tidak dikalibrasi ke data nyata. G1 hanya bisa MENOLAK A, lolos tidak membuktikan apa pun di data nyata.
- HMM dibatasi paling banyak 1 perubahan per site (sesuai struktur truth); data nyata dengan >1 perubahan butuh forward-backward penuh.
- Label A (presence) berasal dari truth sintetis; di data nyata label itu harus datang dari anotasi (G2).
- `matched` memberi A keuntungan bentuk model; verdict penentu hanya `{prim}`.
- Kalibrasi isotonic menghasilkan plateau (ties) yang menurunkan recall pada FPR ketat; aturan sama untuk semua metode.
""")
    print(f"S7 selesai {elapsed:.1f}s: G1 {g1}; {b0line} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
