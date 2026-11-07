"""S6 — benchmark semi-sintetis G1: trajektori present/absent + observasi disimulasikan di level obs (tanpa edit citra).

Site stabil nyata S4 (n_epochs_det >= --min-det) dipakai sebagai lokasi; kovariat kualitas q per site-epoch (gsd,
reg_resid_m, reg_ok, nodata_pct, dt_days) diambil dari obs S4 nyata. Hanya ground truth dan `detected/score/tier`
yang disimulasikan. `observable` tetap dari S4 (baris tak-observable tidak pernah detected dan tidak dihitung absen).
Trajektori: stable | loss (present sampai t-1, absent mulai t) | gain (kebalikannya) | flicker (absent tepat 1 epoch);
t dalam [3, n-3] jadi >= 3 epoch di tiap sisi. Truth per site ada di truth_events.csv (semua site, bukan hanya event).

Family (vault 013 asumsi 1, 017): null (tanpa loss/gain, flicker tetap), matched (sigmoid(w^T q), SAMA bentuknya dengan
model A: hanya pembanding berlabel), mismatched (UTAMA; Weibull non-logistik + noise per farm-epoch + efek per site),
stress (HELD-OUT; piecewise + site 'sulit' + miss berurutan + site clutter). Tuning HANYA pada mismatched; stress tidak
boleh dipakai tuning. --epoch-skew mengalikan peluang deteksi per epoch dengan faktor dari porsi tier high nyata
(012: 2012_08 ~1%, 2020_06 ~6%, 2023_08 ~52%; dihitung ulang dari obs S4): none | real | extreme.
Sumbu grid: stride epoch (gap waktu), n epoch, qscale (laju miss karena kualitas), proporsi event, seed (>= 3).
Unit statistik = farm (site disampel --per-farm per farm).

Keluaran (--out, default pilot/out/s6/synth): obs.parquet (kolom S4 + `run`), truth_events.csv, report.md.
Awas: family "null" dibaca NaN oleh pandas.read_csv default -> pakai keep_default_na=False.
Jalankan: python pilot/s6_synth.py [--s4 pilot/out/s4] [--out ...] [--seeds 3] [--per-farm 60] [--epoch-skew real]
"""
import argparse
import itertools
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
MARGIN = 3                      # epoch minimum di tiap sisi event (k=3 di S5)
REF_HIGH_SHARE = 0.52           # porsi tier high epoch kontrol 2023_08 (vault 012)
FAMILIES = ("null", "matched", "mismatched", "stress")
Q_COLS = ["gsd", "bands", "nodata_pct", "reg_resid_m", "reg_ok", "dt_days", "observable"]
OBS_COLS = ["farm", "site_id", "epoch", "detected", "score", "tier", "n_pts", "observable", "gsd", "bands",
            "nodata_pct", "reg_resid_m", "reg_ok", "dt_days", "fe_n_det", "run"]
SCHEMA = pa.schema([("farm", pa.string()), ("site_id", pa.string()), ("epoch", pa.string()), ("detected", pa.bool_()),
                    ("score", pa.float64()), ("tier", pa.string()), ("n_pts", pa.int64()), ("observable", pa.bool_()),
                    ("gsd", pa.float64()), ("bands", pa.int64()), ("nodata_pct", pa.float64()),
                    ("reg_resid_m", pa.float64()), ("reg_ok", pa.bool_()), ("dt_days", pa.float64()),
                    ("fe_n_det", pa.int64()), ("run", pa.string())])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------- data nyata -> matriks (site x epoch) ----------
def make_base(obs, sites, min_det):
    """Site stabil nyata -> matriks q (n_site x n_epoch), faktor epoch, pool skor nyata."""
    d = obs[obs.detected]
    share = (d.tier == "high").groupby(d.epoch).mean()
    keep = sites[sites.n_epochs_det >= min_det][["farm", "site_id", "x", "y"]]
    o = obs.merge(keep[["farm", "site_id"]]).sort_values(["farm", "site_id", "epoch"], ignore_index=True)
    eps = sorted(obs.epoch.unique())
    ne = len(eps)
    if o.empty or len(o) % ne or (o.epoch.to_numpy().reshape(-1, ne) != eps).any():
        sys.exit("ERROR: obs bukan grid penuh site x epoch")
    M = {c: o[c].to_numpy().reshape(-1, ne) for c in Q_COLS}
    resid = np.where(np.isnan(M["reg_resid_m"]), np.where(M["reg_ok"], 0.0, 0.8), M["reg_resid_m"])
    # badness b: galat registrasi + registrasi tak terukur + GSD kasar + nodata (q metadata saja, sama dengan S4)
    M["b"] = np.maximum(0, np.minimum(resid, 1.5) + 0.5 * ~M["reg_ok"] + 2 * (M["gsd"] - 0.3) + np.nan_to_num(M["nodata_pct"]) / 10)
    sk = sites.merge(keep).set_index(["farm", "site_id"]).loc[list(zip(o.farm[::ne], o.site_id[::ne]))]
    return SimpleNamespace(M=M, eps=eps, farm=o.farm.to_numpy()[::ne], site_id=o.site_id.to_numpy()[::ne],
                           x=sk.x.to_numpy(), y=sk.y.to_numpy(), best_tier=sk.best_tier.to_numpy(),
                           share=share.reindex(eps).fillna(REF_HIGH_SHARE).to_numpy(),
                           pool=np.sort(d.score.to_numpy()))


def epoch_factor(share, mode):
    f = np.clip(share / REF_HIGH_SHARE, 0.05, 1.0)
    return {"none": np.ones_like(f), "real": f, "extreme": np.where(f < 0.5, 0.02, 1.0)}[mode]


# ---------- model emisi: (p_det, p_false_alarm) per site-epoch ----------
def fam_matched(b, fe, rng, ctx):
    return sigmoid(3.0 - 2.0 * ctx.q * b) * fe, sigmoid(-4.2 + 0.8 * ctx.q * b)


def fam_mismatched(b, fe, rng, ctx):
    x = ctx.q * b
    p = 0.93 * np.exp(-(x / 1.6) ** 2.5)                                    # Weibull: datar lalu jatuh tajam
    p = p * fe * np.exp(rng.normal(0, 0.3, (ctx.nfarm, b.shape[1]))[ctx.fi])  # noise farm-epoch (tidak ada di A)
    p = p * (rng.beta(6, 1.5, (len(b), 1)) / 0.8)                           # efek detektabilitas per site
    fa = (0.015 + 0.03 * np.minimum(x, 2.5) ** 1.5) * np.exp(rng.normal(0, 0.3, (ctx.nfarm, b.shape[1]))[ctx.fi])
    return np.clip(p, 0, 0.98), np.clip(fa, 0, 0.5)


def fam_stress(b, fe, rng, ctx):
    x = ctx.q * b
    p = np.interp(x, [0.5, 1.5, 3.0], [0.9, 0.35, 0.15]) * fe               # piecewise-linear, bukan Weibull
    p = p * np.where(rng.random((len(b), 1)) < 0.15, 0.3, 1.0)              # 15% site 'sulit'
    fa = np.where(rng.random((len(b), 1)) < 0.05, 0.25, 0.03) * np.ones_like(b)   # 5% site clutter
    return np.clip(p, 0, 0.98), fa


EMIT = {"null": fam_mismatched, "matched": fam_matched, "mismatched": fam_mismatched, "stress": fam_stress}
PERSIST = {"stress"}           # setelah miss, peluang deteksi epoch berikutnya x0,6


# ---------- satu run ----------
def simulate(base, family, stride, n_epochs, q, ev, seed, per_farm, skew, flick=0.1):
    """Return (obs, truth, present) atau None bila stride*n_epochs tidak muat. present: matriks bool (site x epoch)."""
    ne = len(base.eps)
    sel = np.arange(ne)[::stride][:n_epochs]
    if len(sel) < n_epochs:
        return None
    ev = 0.0 if family == "null" else ev
    run = f"{family}|st{stride}|n{n_epochs}|q{q:g}|ev{ev:g}|seed{seed}"
    rng = np.random.default_rng([seed, zlib.crc32(run.encode())])
    idx = np.sort(np.concatenate([rng.choice(g, min(len(g), per_farm), replace=False)
                                  for g in pd.Series(np.arange(len(base.farm))).groupby(base.farm).agg(list)]))
    ns, n = len(idx), len(sel)
    eps = np.array(base.eps)[sel]
    farm, sid = base.farm[idx], base.site_id[idx]
    # trajektori
    u, t = rng.random(ns), rng.integers(MARGIN, n - MARGIN + 1, ns)
    typ = np.select([u < ev / 2, u < ev, u < ev + flick], ["loss", "gain", "flicker"], "stable")
    col = np.arange(n)[None, :]
    present = np.select([typ[:, None] == "loss", typ[:, None] == "gain", typ[:, None] == "flicker"],
                        [col < t[:, None], col >= t[:, None], col != t[:, None]], True)
    # observasi
    M = {k: v[idx][:, sel] for k, v in base.M.items()}
    fi = pd.factorize(farm)[0]
    ctx = SimpleNamespace(q=q, nfarm=fi.max() + 1, fi=fi)
    fe = epoch_factor(base.share, skew)[sel][None, :]
    p, fa = EMIT[family](M["b"], fe, rng, ctx)
    obs_ok = M["observable"]
    hit = np.zeros((ns, n), bool)
    for j in range(n):
        pj = p[:, j] * (np.where(hit[:, j - 1] | ~present[:, j - 1], 1.0, 0.6) if family in PERSIST and j else 1.0)
        hit[:, j] = rng.random(ns) < np.where(present[:, j], pj, fa[:, j])
    hit &= obs_ok
    # score / tier: kuantil pool skor nyata; hit pada p tinggi cenderung skor tinggi, false alarm skor rendah
    uq = np.where(present, 0.5 * rng.random((ns, n)) + 0.5 * np.clip(p, 0, 1), 0.6 * rng.random((ns, n)))
    pool = base.pool
    score = np.where(hit, pool[(uq * (len(pool) - 1)).astype(int)], 0.0)
    lo, hi = np.quantile(pool, [0.33, 0.60])
    tier = np.where(~hit, None, np.where(score >= hi, "high", np.where(score >= lo, "med", "low")))
    cum = np.cumsum(np.nan_to_num(base.M["dt_days"][idx]), axis=1)[:, sel]           # gap = jumlah gap nyata yang dilompati
    dt = np.full((ns, n), np.nan)
    dt[:, 1:] = np.diff(cum, axis=1)
    fe_n = pd.DataFrame(hit).groupby(farm).transform("sum").to_numpy()
    flat = lambda a: a.ravel()           # noqa: E731
    df = pd.DataFrame(dict(farm=np.repeat(farm, n), site_id=np.repeat(sid, n), epoch=np.tile(eps, ns), detected=flat(hit),
                           score=flat(score), tier=flat(tier), n_pts=flat(hit).astype(int), observable=flat(obs_ok),
                           gsd=flat(M["gsd"]), bands=flat(M["bands"]), nodata_pct=flat(M["nodata_pct"]),
                           reg_resid_m=flat(M["reg_resid_m"]), reg_ok=flat(M["reg_ok"]), dt_days=flat(dt),
                           fe_n_det=flat(fe_n), run=run))
    ch = np.isin(typ, ["loss", "gain", "flicker"])
    tr = pd.DataFrame(dict(run=run, family=family, stride=stride, n_epochs=n_epochs, qscale=q, event_frac=ev, seed=seed,
                           farm=farm, site_id=sid, type=typ,
                           epoch=np.where(ch, eps[t], None), t_before=np.where(ch, eps[t - 1], None),
                           t_after=np.where(typ == "flicker", eps[np.minimum(t + 1, n - 1)], np.where(ch, eps[t], None))))
    return df, tr, present


def check(obs, truth, present):
    """Invarian internal: truth merekonstruksi present, dan obs tidak melanggarnya."""
    assert not (obs.detected & ~obs.observable).any() and (obs.score[~obs.detected] == 0).all()
    assert obs.groupby(["site_id", "epoch"]).size().max() == 1
    ev = truth[truth.type.isin(["loss", "gain"])]
    assert ((ev.t_before < ev.t_after) & (ev.t_after == ev.epoch)).all()
    assert present.shape[0] == len(truth)


# ---------- laporan ----------
def md(df):
    df = df.astype(str)
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(r) + " |" for r in df.to_numpy()])


def make_report(truth, stats, skipped, elapsed, a):
    cnt = (truth.groupby(["family", "seed", "type"]).size().unstack(fill_value=0)
           .reindex(columns=["loss", "gain", "flicker", "stable"], fill_value=0).reset_index())
    nf = truth.groupby(["family", "seed"]).farm.nunique().rename("farms").reset_index()
    cnt = cnt.merge(nf)
    cnt["runs"] = cnt.merge(stats.groupby(["family", "seed"]).size().rename("r").reset_index()).r
    ev = truth[truth.type.isin(["loss", "gain"])].groupby(["family", "farm"]).size()
    farm_ev = ev.groupby("family").agg(min="min", median="median", n_farm="size").reset_index()
    emit = (stats.groupby("family")[["det_present", "det_absent", "det_present_high"]].mean().round(3).reset_index())
    cell = (stats[stats.family == "mismatched"].groupby(["stride", "n_epochs", "qscale", "event_frac"])
            .det_present.mean().round(3).reset_index())
    return f"""# S6 — benchmark semi-sintetis G1 (synth)

Run {elapsed:.1f}s. Args: {vars(a)}.
Aturan pakai: tuning HANYA pada `mismatched`; `matched` = pembanding berlabel (bentuk sama dengan A, kemenangan A di sini
tidak berarti apa-apa); `stress` = held-out, jangan dipakai tuning; `null` = false alarm saat tanpa event.
Event = loss+gain; flicker dan stable = negatif. Unit statistik = farm (site disampel {a.per_farm}/farm).

## Jumlah site per family/seed (dijumlah semua sel grid)
{md(cnt)}

## Event (loss+gain) per farm, per family
{md(farm_ev)}

## Sanity emisi (rata-rata antar run; baris observable)
det_present = P(detected | present), det_absent = P(detected | absent) = false alarm, det_present_high = P(tier high | present).
{md(emit)}

## det_present family mismatched per sel
{md(cell)}

## Sel dilewati (stride x n_epochs melebihi 20 epoch)
{skipped or 'tidak ada'}

## Batasan
- q hanya metadata S4 (gsd, galat registrasi, reg_ok, nodata); sharpness/bayangan tidak ada di S4 sehingga tidak disimulasikan.
- Faktor epoch dari porsi tier high S4 (mengikuti 012) adalah pendekatan kasar; recall nyata tidak pernah diukur.
- Semua parameter emisi adalah pilihan desain, bukan hasil kalibrasi ke data nyata. Menang di G1 hanya bisa MENOLAK.
"""


def csv_list(s, typ):
    return [typ(v) for v in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s4", default=str(HERE / "out" / "s4"), help="folder berisi sites.parquet & obs.parquet")
    ap.add_argument("--out", default=str(HERE / "out" / "s6" / "synth"))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--strides", default="1,2", help="gap waktu: ambil tiap stride-epoch")
    ap.add_argument("--n-epochs", default="10,20")
    ap.add_argument("--qscales", default="0.5,1.5", help="laju miss karena kualitas")
    ap.add_argument("--events", default="0.1,0.25,0.5", help="proporsi site loss+gain: rendah,sedang,tinggi")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-farm", type=int, default=60)
    ap.add_argument("--min-det", type=int, default=10, help="site stabil: terdeteksi >= N epoch di S4")
    ap.add_argument("--flicker", type=float, default=0.1)
    ap.add_argument("--epoch-skew", choices=["none", "real", "extreme"], default="real")
    a = ap.parse_args(argv)
    if a.seeds < 3:
        sys.exit("ERROR: minimal 3 seed per sel (vault 013)")
    fams = csv_list(a.families, str)
    if set(fams) - set(FAMILIES):
        sys.exit(f"ERROR: family tidak dikenal: {set(fams) - set(FAMILIES)}")
    s4 = Path(a.s4)
    for f in ("sites.parquet", "obs.parquet"):
        if not (s4 / f).is_file():
            sys.exit(f"ERROR: tidak ada {s4 / f} (jalankan s4_sites.py dulu)")
    t0 = time.time()
    base = make_base(pd.read_parquet(s4 / "obs.parquet"), pd.read_parquet(s4 / "sites.parquet"), a.min_det)
    print(f"{len(base.farm)} site stabil dari {len(set(base.farm))} farm", file=sys.stderr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truths, stats, skipped = [], [], set()
    with pq.ParquetWriter(out / "obs.parquet", SCHEMA) as w:
        for fam, st, n, q, ev, seed in itertools.product(fams, csv_list(a.strides, int), csv_list(a.n_epochs, int),
                                                         csv_list(a.qscales, float), csv_list(a.events, float),
                                                         range(a.seeds)):
            if fam == "null" and ev != csv_list(a.events, float)[0]:
                continue                       # null tidak punya sumbu event
            r = simulate(base, fam, st, n, q, ev, seed, a.per_farm, a.epoch_skew, a.flicker)
            if r is None:
                skipped.add(f"stride {st} x n_epochs {n}")
                continue
            df, tr, present = r
            check(df, tr, present)
            w.write_table(pa.Table.from_pandas(df[OBS_COLS], schema=SCHEMA, preserve_index=False))
            truths.append(tr)
            o = df.observable.to_numpy().reshape(present.shape)
            hit, hi = df.detected.to_numpy().reshape(present.shape), (df.tier == "high").to_numpy().reshape(present.shape)
            pm, am = present & o, ~present & o
            stats.append(dict(family=fam, seed=seed, stride=st, n_epochs=n, qscale=q, event_frac=tr.event_frac[0],
                              det_present=hit[pm].mean(), det_absent=hit[am].mean() if am.any() else np.nan,
                              det_present_high=hi[pm].mean()))
    truth = pd.concat(truths, ignore_index=True)
    truth.to_csv(out / "truth_events.csv", index=False)
    elapsed = time.time() - t0
    (out / "report.md").write_text(make_report(truth, pd.DataFrame(stats), ", ".join(sorted(skipped)), elapsed, a))
    print(f"S6 selesai {elapsed:.1f}s: {len(truths)} run, {len(truth)} site-run -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
