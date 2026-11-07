"""S8 — evaluasi G2: A vs B3/B4/B5 (+B0) pada event nyata berlabel manusia (pilot/out/g2/labels.parquet).

Kebenaran: label loss/gain -> event di change_epoch; verdict 'ada' -> stable present, 'tidak_ada' -> stable absent
(negatif); sisanya (ambigu, arah terbalik) dikeluarkan. Parameter B3/B4/B5/A dilatih HANYA di satu run S6 mismatched
(semua farm sintetis; tanpa label nyata). Kalibrasi isotonic p_event memakai label nyata dengan 2-fold farm held-out
(dilatih di fold farm lain). Threshold FPR digabung lintas farm (negatif ~2/farm); recall per farm memakai threshold gabungan.
Verdict = bootstrap farm (s7.verdict) atas farm yang punya event.
Jalankan: python pilot/s8_g2.py [--src RUN_S6] [--out pilot/out/g2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7_g1 as s7  # noqa: E402
from s6_synth import md  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = "mismatched|st1|n20|q1.5|ev0.25|seed0"   # n20 = jumlah epoch data nyata; pilihan tunggal, sensitivitas via --src


def real_truth(labels):
    """Label review -> truth (type loss/gain/stable/absent, epoch). Kelas lain dikeluarkan."""
    L = labels.copy()
    L["type"] = np.select([L.label.eq("loss"), L.label.eq("gain"), L.label.eq("none") & L.verdict.eq("ada"), L.label.eq("none") & L.verdict.eq("tidak_ada")],
                          ["loss", "gain", "stable", "absent"], "")
    L = L[L.type != ""].copy()
    L.loc[L.type.isin(["stable", "absent"]), "change_epoch"] = ""
    L["epoch"] = L.change_epoch.fillna("")
    assert L.site_id.is_unique, "site ganda di label"
    return L[["farm", "site_id", "epoch", "type"]].assign(run="real").reset_index(drop=True)


def real_obs(obs, truth):
    """Subset obs ke site berlabel (urutan = truth), tambah fe_nobs = jumlah site observable per farm-epoch (farm penuh)."""
    obs = obs.merge(obs.groupby(["farm", "epoch"]).observable.sum().rename("fe_nobs").reset_index(), on=["farm", "epoch"])
    o = obs[obs.site_id.isin(truth.site_id)].copy()
    o["site_id"] = pd.Categorical(o.site_id, truth.site_id)
    o = o.sort_values(["site_id", "epoch"]).assign(run="real")
    o["site_id"] = o.site_id.astype(str)
    return o


def predict(R, eps, P, tol):
    """Prediksi mentah semua metode + kalibrasi isotonic 2-fold farm (iso dari fold lain). Return DataFrame."""
    raw = s7.raw_preds(R, P)
    ev = np.isin(R["ttype"], ["loss", "gain"])
    z = np.zeros(len(ev))
    raw["B0"] = (z, z, z.astype(int), z.astype(int))
    rows = []
    for m in s7.METHODS:
        pl, pg, el, eg = raw[m]
        if m != "B0":
            pe, cal = pl + pg, np.zeros(len(ev))
            for f in (0, 1):
                te = R["fold"] == f
                cal[te] = np.interp(pe[te], *s7.iso_fit(pe[~te], ev[~te].astype(float)))
            sc = np.divide(cal, pe, out=np.zeros_like(pe), where=pe > 0)
            pl, pg = pl * sc, pg * sc
        ty, ep = s7.final_type_epoch(pl, pg, el, eg)
        rows.append(pd.DataFrame(dict(farm=R["farm"], site_id=R["site_id"], method=m, p_loss=pl, p_gain=pg, p_event=pl + pg,
                                      pred_type=ty, pred_epoch=eps[ep], true_type=R["ttype"], ok=s7.is_ok(R, ty, ep, tol))))
    return pd.concat(rows, ignore_index=True)


def farm_table(preds, fas):
    """Seperti s7.farm_table tapi threshold FPR digabung lintas farm; farm tanpa event -> recall NaN."""
    rows = []
    for m, g in preds.groupby("method", sort=False):
        p, y, ok = g.p_event.to_numpy(), g.true_type.isin(["loss", "gain"]).to_numpy(), g.ok.to_numpy()
        thr = {fa: s7.fa_thr(p, y, fa) for fa in fas}
        for farm, ix in g.groupby("farm").indices.items():
            r = dict(family="real", method=m, farm=farm, brier=np.mean((p[ix] - y[ix]) ** 2), n_events=int(y[ix].sum()),
                     n_neg=int((~y[ix]).sum()))
            for fa in fas:
                r[f"recall_{fa:g}"] = (ok[ix] & (p[ix] > thr[fa])).sum() / y[ix].sum() if y[ix].any() else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def pooled(preds, fas):
    rows = []
    for m, g in preds.groupby("method", sort=False):
        p, y, ok = g.p_event.to_numpy(), g.true_type.isin(["loss", "gain"]).to_numpy(), g.ok.to_numpy()
        r = dict(method=m, brier=np.mean((p - y) ** 2))
        r.update({f"recall_{fa:g}": s7.recall_at_fa(p, ok, y, fa) for fa in fas})
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s6", default=str(HERE / "out" / "s6" / "synth"))
    ap.add_argument("--obs", default=str(HERE / "out" / "s4" / "obs.parquet"))
    ap.add_argument("--labels", default=str(HERE / "out" / "g2" / "labels.parquet"))
    ap.add_argument("--out", default=str(HERE / "out" / "g2"))
    ap.add_argument("--gates", default=str(HERE / "gates.yaml"))
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    gt = yaml.safe_load(open(a.gates))
    g1, g2 = gt["g1_synthetic"], gt["g2_real_events"]
    fa, tol, must = g1["fixed_fa"], g1["event_time_tol_epochs"], g1["must_beat"]
    fas = (fa, 0.01, 0.10)

    truth = real_truth(pd.read_parquet(a.labels))
    obs = real_obs(pd.read_parquet(a.obs), truth)
    farms = sorted(truth.farm.unique())
    R, eps, _ = s7.load_runs(obs[s7.OBS_COLS + ["fe_nobs"]], truth, {f: i % 2 for i, f in enumerate(farms)})["real"]

    s6 = Path(a.s6)
    obs6 = pd.read_parquet(s6 / "obs.parquet", columns=s7.OBS_COLS)
    truth6 = pd.read_csv(s6 / "truth_events.csv", keep_default_na=False, dtype=str)
    obs6, truth6 = obs6[obs6.run == a.src], truth6[truth6.run == a.src]
    if truth6.empty:
        sys.exit(f"ERROR: run sumber {a.src} tidak ada di {s6}")
    R6, _, _ = s7.load_runs(obs6, truth6, {f: 0 for f in truth6.farm.unique()})[a.src]
    P = s7.fit(R6, np.ones(len(R6["farm"]), bool), fa, tol)

    preds = predict(R, eps, P, tol)
    fm = farm_table(preds, fas)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "preds.parquet", index=False)
    fm.to_csv(out / "farm_metrics.csv", index=False)

    rc = f"recall_{fa:g}"
    cls = truth.assign(kelas=truth.type.map({"loss": "loss", "gain": "gain"}).fillna("negatif")).pivot_table(
        index="farm", columns="kelas", values="site_id", aggfunc="count", fill_value=0)
    cls = cls.reindex(columns=["loss", "gain", "negatif"], fill_value=0)
    cls.loc["TOTAL"] = cls.sum()
    ev_farms = int((fm[fm.method == "A"].n_events > 0).sum())
    vrows = []
    for b in must:
        v = s7.compare(fm, "real", "A", b, rc, reps=a.boot, seed=a.seed)
        vrows.append(dict(vs=b, **s7.fmt(v)))
    wins = [r["A_menang"] == "ya" for r in vrows]
    nl, ng, nf = int((truth.type == "loss").sum()), int((truth.type == "gain").sum()), len(farms)
    min_ok = nl >= g2["min_loss"] and ng >= g2["min_gain"] and nf >= g2["min_farms"]
    verd = "PASS" if all(wins) else "FAIL"
    b0 = s7.compare(fm, "real", "B0", "B4", rc, reps=a.boot, seed=a.seed)
    b0line = "B0 vs B4 gagal (benar)" if not b0["win"] else "PERINGATAN: B0 LOLOS -> EVALUASI RUSAK"
    fm_tbl = fm[fm.method == "A"][["farm", "n_events", "n_neg"]]
    (out / "g2_report.md").write_text(f"""# S8 — evaluasi G2 (data nyata): A vs B3/B4/B5

Run {time.time() - t0:.1f}s. Args: {vars(a)}. Aturan: sama dengan G1 (fixed_fa={fa}, tol={tol} epoch, must_beat={must}, CI bootstrap farm {a.boot}x seed {a.seed}).

## Kelas & farm (label tier high hasil review manusia)
{md(cls.reset_index())}

Minimum gates.yaml (loss>={g2['min_loss']}, gain>={g2['min_gain']}, farm>={g2['min_farms']}, kappa>={g2['min_kappa']}): loss {nl}, gain {ng}, farm {nf}, kappa BELUM ADA. Syarat jumlah terpenuhi: {min_ok}. Farm dengan event (dipakai bootstrap): {ev_farms}.

## Metrik gabungan (semua site; FPR gabungan lintas farm)
{md(pooled(preds, fas))}

## Metrik rata-rata per farm (farm dengan event)
{md(fm[fm.n_events > 0].groupby("method")[["brier", "recall_0.05", "recall_0.01", "recall_0.1"]].mean().round(4).reindex(list(s7.METHODS)).reset_index())}

## Verdict (A - baseline, CI bootstrap farm 95%)
Menang iff CI dBrier < 0 dan CI dRecall@{fa:g} > 0.
{md(pd.DataFrame(vrows))}

**G2 {verd}** (A menang atas {', '.join(must)} = {wins}). Sanity: {b0line} (dBrier {b0['dB']:+.4f}, dRecall {b0['dR']:+.4f}).
Catatan: PASS/FAIL ini bersyarat pada keterbatasan di bawah; kappa belum ada sehingga gate G2 formal belum bisa ditutup.

## Per farm (jumlah)
{md(fm_tbl)}

## Metode
- Truth: loss -> epoch pertama pohon tidak ada; gain -> epoch pertama ada; verdict 'ada' = stable present, 'tidak_ada' = stable absent (negatif); ambigu & arah terbalik dikeluarkan.
- Parameter B3 (K,M), B4 (pdet,eps), B5, A (w,v), laju loss/gain: dilatih di S6 run `{a.src}` (semua farm sintetis), tanpa label nyata. K,M={P['K']},{P['M']}; pdet={P['pdet']:.3f}, eps={P['eps']:.3f}.
- Kalibrasi isotonic p_event: label nyata, 2-fold farm (paritas urutan farm), tiap site dikalibrasi dari fold farm lain. B0 tidak dikalibrasi.
- laju farm-epoch (fe_n_det/fe_nobs) dari seluruh farm di s4, bukan hanya site berlabel.

## Penyimpangan & keterbatasan
- FPR dihitung GABUNGAN lintas farm (bukan per farm): negatif ~2/farm. Dengan 30 negatif, FPR 0.05 = threshold negatif tertinggi ke-2; 0.01 = negatif tertinggi; sangat kasar.
- loss {nl} < 80 (min gates.yaml). Kappa belum ada (penilai tunggal).
- n kecil, negatif sedikit (30; sebagian dari tipe kandidat berbeda dari event, jadi distribusi negatif tidak sama dengan sampel acak site).
- Event berasal dari kandidat detektor (selection bias terhadap pola yang dilihat detektor); recall tidak mengukur event yang tak pernah jadi kandidat.
- Bootstrap hanya atas farm ber-event; Brier dan recall per farm yang negatif-saja dibuang.
- Parameter dilatih di satu run S6 (pilihan tunggal); emisi sintetis tak identik dengan data nyata. HMM max 1 perubahan/site.
- Fold kalibrasi 2-fold dengan ~73 site per fold; isotonic bisa tidak stabil.
""")
    print(f"S8 selesai: G2 {verd} {wins}; {b0line} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
