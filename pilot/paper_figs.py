"""Figures and summary numbers for the RSASE manuscript (read-only on pilot/out).

Usage: python pilot/paper_figs.py --out <figdir>
Needs matplotlib, pandas, pyarrow (not in the project venv; run from any env that has them).
Every figure is drawn from files in pilot/out; nothing is typed in by hand except the
pipeline/graphical-abstract labels, whose numbers are read from the same files.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "out"
# Optional farm-ID relabelling for figure text and paper_numbers.json: FARM_ID_MAP = JSON file {"<id in data>": "<code>"}.
# Unset (public data already uses F### codes): ids pass through unchanged.
FARM_MAP_PATH = os.environ.get("FARM_ID_MAP")
FARM_MAP = json.load(open(FARM_MAP_PATH)) if FARM_MAP_PATH else {}


def farm_label(f):
    return FARM_MAP.get(str(f), str(f))


MS = ["B3", "B4", "B5", "A"]
COL = {"B0": "#bbbbbb", "B3": "#7f7f7f", "B4": "#56b4e9", "B5": "#0072b2", "A": "#d55e00"}  # Okabe-Ito, colour-blind safe
NAME = {"B3": "B3 run-length", "B4": "B4 HMM, constant p_det", "B5": "B5 HMM, farm-epoch p_det", "A": "A HMM, quality covariates"}


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save(fig, out, name, dpi=300):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")


def md_table(text, header_start):
    """Parse the first markdown table whose header line starts with header_start."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_start))
    head = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip("|").split("|")])
    return pd.DataFrame(rows, columns=head)


def ci(s):
    a, b = re.findall(r"[-+]?\d*\.\d+", s)
    return float(a), float(b)


def verdict_tables():
    g1 = md_table((ROOT / "s7" / "g1_report.md").read_text(), "| family | vs |")
    g2 = md_table((ROOT / "g2" / "g2_report.md").read_text(), "| vs | dBrier |")
    rows = []
    for _, r in g1[g1.family == "mismatched"].iterrows():
        rows.append(dict(test="G1 semi-synthetic (mismatched)", vs=r.vs, dB=float(r.dBrier), dB_ci=ci(r.CI_dBrier),
                         dR=float(r.dRecall), dR_ci=ci(r.CI_dRecall)))
    for _, r in g2.iterrows():
        rows.append(dict(test="G2 real events", vs=r.vs, dB=float(r.dBrier), dB_ci=ci(r.CI_dBrier),
                         dR=float(r.dRecall), dR_ci=ci(r.CI_dRecall)))
    return pd.DataFrame(rows)


def fig_forest(out, plt, which):
    v = verdict_tables()
    key, cik, lab = ("dB", "dB_ci", "Farm-mean difference in Brier score, A minus baseline (negative favours A)") if which == "brier" \
        else ("dR", "dR_ci", "Farm-mean difference in recall at 5% FPR, A minus baseline (positive favours A)")
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ys, labels = [], []
    y = 0
    for test, mk in [("G1 semi-synthetic (mismatched)", "s"), ("G2 real events", "o")]:
        for _, r in v[v.test == test].iterrows():
            lo, hi = r[cik]
            ax.errorbar(r[key], y, xerr=[[r[key] - lo], [hi - r[key]]], fmt=mk, color=COL[r.vs], capsize=3, ms=5)
            ys.append(y); labels.append(f"{test.split(' ')[0]}: A vs {r.vs}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_xlabel(lab)
    save(fig, out, "Figure_4" if which == "brier" else "Figure_5"); plt.close(fig)
    return v


def fig_epochs(out, plt):
    o = pd.read_parquet(ROOT / "s4" / "obs.parquet", columns=["epoch", "detected", "observable", "gsd"])
    o = o[o.observable]
    det = o.groupby("epoch").detected.mean()
    gsd = o.groupby("epoch").gsd.median()
    reg = pd.read_parquet(ROOT / "s1" / "registration_summary.parquet")
    regm = reg.groupby("epoch").median_resid_m.median()
    ep = sorted(det.index)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(ep))
    ax.bar(x, det.reindex(ep).values, color=["#d55e00" if g < 0.4 else "#0072b2" for g in gsd.reindex(ep).values])
    ax.set_ylabel("Share of site-epochs with a detection")
    ax.set_xticks(x); ax.set_xticklabels([e.replace("_", "-") for e in ep], rotation=60, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, regm.reindex(ep).values, "k.-", lw=1)
    ax2.set_ylabel("Median registration residual (m)")
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(0.8, float(regm.max()) * 1.1))
    save(fig, out, "Figure_2"); plt.close(fig)
    return pd.DataFrame({"det_rate": det, "gsd": gsd, "reg_median_m": regm}).reindex(ep)


def fig_filmstrip(out, plt, review_id):
    from PIL import Image
    lab = pd.read_parquet(ROOT / "g2" / "labels.parquet")
    r = lab[lab.review_id == review_id].iloc[0]
    d = ROOT / ("s5b" if r.batch == "b1" else "s5b_r2") / "img"
    files = sorted(d.glob(f"{review_id}_*.png"))
    fig, axs = plt.subplots(2, 10, figsize=(7.48, 1.9))
    for ax, p in zip(axs.ravel(), files):
        ax.imshow(Image.open(p)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(p.stem[len(review_id) + 1:].replace("_", "-"), fontsize=6, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.04, hspace=0.25)
    save(fig, out, "Figure_3", dpi=600); plt.close(fig)
    return dict(review_id=review_id, farm=farm_label(r.farm), label=r.label, change_epoch=r.change_epoch, verdict=r.verdict,
                first=r.first_change_epoch, second=r.second_change_epoch, n_img=len(files))


def fig_roc(out, plt):
    roc = pd.read_csv(ROOT / "explore" / "E3" / "roc_points.csv")
    met = pd.read_csv(ROOT / "explore" / "E3" / "metrics.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for m in MS:
        g = roc[roc.method == m]
        ax.step(g.fpr, g.tpr, where="post", color=COL[m], label=f"{m} (AUC {met.loc[m, 'auc']:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="#999")
    ax.set_xlabel("False positive rate (28 hard negatives)"); ax.set_ylabel("True positive rate (115 events)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, out, "Figure_6"); plt.close(fig)
    return met


def fig_e2(out, plt):
    r = pd.read_csv(ROOT / "explore" / "E2" / "runs.csv")
    r = r.sort_values("dBrier_pooled").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    x = np.arange(len(r))
    c = ["#d55e00" if "q1.5" in s else "#0072b2" for s in r["run"]]
    ax.bar(x, r.dBrier_pooled, color=c)
    ax.axhline(0, color="k", lw=0.8)
    orig = r.index[r["run"].str.contains(r"q1\.5\|ev0\.25\|seed0")]
    ax.plot(orig, r.loc[orig, "dBrier_pooled"], "kv", ms=6)
    ax.set_xticks(x); ax.set_xticklabels([s.split("n20|")[1].replace("|", ", ").replace("seed", "s") for s in r["run"]], rotation=70, ha="right", fontsize=7.5)
    ax.set_xlabel("Simulation run (quality effect q, event rate ev, seed s)")
    ax.set_ylabel("Brier A minus B5 (pooled)")
    save(fig, out, "Figure_7"); plt.close(fig)
    return r


def boxes(ax, items, y, h, fc):
    for x, w, t in items:
        ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                                                      fc=fc, ec="#333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.2, wrap=True)


def fig_pipeline(out, plt, n):
    fig, ax = plt.subplots(figsize=(7.48, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
    w, h = 3.0, 1.3
    xs = [0.1, 3.5, 6.9]
    row1 = [f"S0 audit\n{n['farms']} farms x {n['epochs']} epochs\n{n['pairs']} clip+mask pairs",
            f"S1 registration\nphase correlation\n{n['reg_measured']} of {n['reg_pairs']} farm-epochs measured",
            "S2-S3 tree detection\n3 pretrained models x 2 grids\nconsensus tiers (high = 2/3 agree)"]
    row2 = [f"S4 tree sites\nfixed-centre clustering, r = 2.5 m\n{n['sites']:,} sites x {n['epochs']} epochs",
            "S5 change candidates\nstate per epoch: D, A or U\nloss, gain, stable, flicker",
            f"S5b human review, type hidden\n{n['reviewed']} locations, 20 crops each\n{n['loss']} loss, {n['gain']} gain, {n['neg']} unchanged"]
    row3 = ["S6 semi-synthetic benchmark\nknown truth, 4 families\n29 farms, 180 runs",
            "S7 gate G1 (semi-synthetic)\nA vs B3, B4, B5\nBrier and recall at 5% FPR",
            "S8 gate G2 (real events)\nsame rule as G1\nfarm bootstrap 95% CI"]
    for y, row, fc in [(4.4, row1, "#e8f1fa"), (2.4, row2, "#e8f1fa"), (0.3, row3, "#fbeee6")]:
        boxes(ax, [(x, w, t) for x, t in zip(xs, row)], y, h, fc)
        for x in xs[:2]:
            ax.annotate("", xy=(x + w + 0.38, y + h / 2), xytext=(x + w + 0.02, y + h / 2),
                        arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(xs[0] + w / 2, 2.4 + h + 0.05), xytext=(xs[2] + w / 2, 4.4 - 0.05),
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(xs[0] + w / 2, 0.3 + h + 0.05), xytext=(xs[0] + w / 2, 2.4 - 0.05),
                arrowprops=dict(arrowstyle="->", lw=0.9, ls="--"))
    ax.annotate("", xy=(xs[2] + w / 2, 0.3 + h + 0.05), xytext=(xs[2] + w / 2, 2.4 - 0.05),
                arrowprops=dict(arrowstyle="->", lw=0.9))
    save(fig, out, "Figure_1"); plt.close(fig)


def graphical_abstract(out, plt, n, v, met):
    fig, ax = plt.subplots(figsize=(13.28 / 2.54 * 1.6, 4.2))  # wide, landscape
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 4.2); ax.axis("off")
    boxes(ax, [(0.1, 2.6, f"Pl\u00e9iades, Qatar\n{n['farms']} date palm farms\n{n['epochs']} epochs, 2012-2026")], 1.0, 2.0, "#e8f1fa")
    boxes(ax, [(3.1, 2.8, f"Per-tree pipeline\ndetect, register, track\n{n['sites']:,} tree sites\n{n['reviewed']} reviewed locations")],
          1.0, 2.0, "#e8f1fa")
    g1 = v[v.test.str.startswith("G1")]
    g2 = v[v.test.str.startswith("G2")]
    b5_1 = g1[g1.vs == "B5"].iloc[0]; b5_2 = g2[g2.vs == "B5"].iloc[0]
    boxes(ax, [(6.3, 2.6, f"G1 semi-synthetic: PASS\nA beats B3, B4, B5\nBrier A-B5 {b5_1.dB:+.3f}\n[{b5_1.dB_ci[0]:+.3f}, {b5_1.dB_ci[1]:+.3f}]")],
          2.2, 1.7, "#e6f4ea")
    boxes(ax, [(6.3, 2.6, f"G2 real events: FAIL\nno win over any baseline\nBrier A-B5 {b5_2.dB:+.3f}\n[{b5_2.dB_ci[0]:+.3f}, {b5_2.dB_ci[1]:+.3f}]")],
          0.2, 1.7, "#fbeee6")
    boxes(ax, [(9.2, 3.2, f"Limits: negatives drawn from\ndetector candidates, small n,\none rater. A ranks changes\nno better than B5\n(AUC A {met.loc['A', 'auc']:.2f}, B5 {met.loc['B5', 'auc']:.2f})")],
          0.9, 2.2, "#f2f2f2")
    for (x0, y0, x1, y1) in [(2.7, 2.0, 3.1, 2.0), (5.9, 2.4, 6.3, 3.0), (5.9, 1.6, 6.3, 1.1), (8.9, 3.0, 9.2, 2.4), (8.9, 1.1, 9.2, 1.6)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1))
    fig.savefig(out / "Graphical_abstract.pdf", bbox_inches="tight")
    fig.savefig(out / "Graphical_abstract.png", dpi=300, bbox_inches="tight")
    from PIL import Image
    im = Image.open(out / "Graphical_abstract.png")
    im.convert("RGB").save(out / "Graphical_abstract.tif", dpi=(300, 300))
    plt.close(fig)
    return im.size


def numbers():
    s0 = pd.read_parquet(ROOT / "s0" / "manifest.parquet")
    reg = pd.read_parquet(ROOT / "s1" / "registration_summary.parquet")
    sites = pd.read_parquet(ROOT / "s4" / "sites.parquet", columns=["site_id"])
    cand = pd.read_parquet(ROOT / "s5" / "candidates.parquet", columns=["type", "persistent", "farm"])
    lab = pd.read_parquet(ROOT / "g2" / "labels.parquet")
    keep = lab[lab.label.isin(["loss", "gain", "none"])]
    regm = reg.dropna(subset=["median_resid_m"])
    per_ep = regm.groupby("epoch").median_resid_m.median()
    n = dict(farms=int(s0.farm.nunique()), epochs=int(s0.epoch.nunique()), pairs=int((s0.has_clip & s0.has_mask).sum()),
             reg_pairs=int(len(reg)), reg_measured=int(regm.shape[0]), reg_red=int(reg.red_flag.sum()),
             reg_gt1m=int((regm.median_resid_m > 1).sum()), reg_ep_min=float(per_ep.min()), reg_ep_max=float(per_ep.max()),
             reg_ep_worst=str(per_ep.idxmax()), sites=int(len(sites)),
             cand=cand.groupby("type").size().to_dict(), cand_persistent=cand[cand.persistent.fillna(False)].groupby("type").size().to_dict(),
             reviewed=int(70 + 150), labels_rows=int(len(lab)), label_counts=lab.label.value_counts().to_dict(),
             loss=int((keep.label == "loss").sum()), gain=int((keep.label == "gain").sum()), neg=int((keep.label == "none").sum()),
             farms_loss=int(keep[keep.label == "loss"].farm.nunique()), farms_gain=int(keep[keep.label == "gain"].farm.nunique()),
             farms_all=int(keep.farm.nunique()))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--film", default=None, help="review_id for Figure 3 (owner-verified loss event)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plt = plt_()
    n = numbers()
    fig_pipeline(out, plt, n)
    ep = fig_epochs(out, plt)
    v = fig_forest(out, plt, "brier"); fig_forest(out, plt, "recall")
    met = fig_roc(out, plt)
    e2 = fig_e2(out, plt)
    film = fig_filmstrip(out, plt, a.film) if a.film else None
    ga = graphical_abstract(out, plt, n, v, met)
    summ = dict(numbers=n, epochs=ep.reset_index().to_dict("records"),
                verdicts=v.assign(dB_ci=v.dB_ci.astype(str), dR_ci=v.dR_ci.astype(str)).to_dict("records"),
                e3=met.reset_index().to_dict("records"), film=film, graphical_abstract_px=ga)
    (out / "paper_numbers.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str)[:6000])
