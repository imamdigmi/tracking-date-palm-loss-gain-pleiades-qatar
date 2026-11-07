"""Grafik eksploratori pasca-G2 (judul & label Bahasa Indonesia). Dipanggil dari notebook 03; matplotlib diimpor malas.

Semua fungsi: plot_xxx(root, figdir) -> Path PNG. root = pilot/out (s7/, g2/, explore/E1..E3). Tiap grafik membaca CSV saja.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
MS = ["B3", "B4", "B5", "A"]
COL = {"B3": "#999999", "B4": "#6baed6", "B5": "#3182bd", "A": "#e6550d", "B0": "#cccccc"}


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib tidak tersedia (dipakai di Colab, bukan di venv lokal)") from e
    return plt


def _save(fig, plt, figdir, name):
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _bars(ax, labels, vals, title, ylabel):
    ax.bar(labels, vals, color=[COL.get(m, "#888") for m in labels])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)


def plot_g1(root, figdir):
    """[G1] Brier & recall@5% per metode, data sintetis (keluarga mismatched), rata-rata per farm."""
    plt = _plt()
    fm = pd.read_csv(Path(root) / "s7" / "farm_metrics.csv", dtype={"farm": str})
    fm = fm[fm.family == "mismatched"]
    g = fm.groupby("method")[["brier", "recall_0.05"]].mean().reindex(["B0"] + MS).dropna(how="all")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    _bars(ax[0], list(g.index), g["brier"].tolist(), "Brier (lebih kecil = lebih baik)", "Brier")
    _bars(ax[1], list(g.index), g["recall_0.05"].tolist(), "Recall pada alarm palsu 5% (lebih besar = lebih baik)", "Recall@5%")
    fig.suptitle("[G1] Data sintetis: perbandingan metode")
    return _save(fig, plt, figdir, "G1_metode.png")


def g2_diffs(fm):
    """Selisih A - baseline (rata-rata per farm) + CI bootstrap farm, dari farm_metrics G2. DataFrame per baseline."""
    import s7_g1 as s7
    rows = []
    for b in ("B3", "B4", "B5"):
        v = s7.compare(fm, "real", "A", b, "recall_0.05", reps=2000, seed=0)
        rows.append(dict(vs=b, dBrier=v["dB"], B_lo=v["ciB"][0], B_hi=v["ciB"][1], dRecall=v["dR"], R_lo=v["ciR"][0], R_hi=v["ciR"][1]))
    return pd.DataFrame(rows)


def plot_g2(root, figdir):
    """[G2] Brier & recall@5% per metode (data nyata) dan selisih A - baseline dengan CI."""
    plt = _plt()
    fm = pd.read_csv(Path(root) / "g2" / "farm_metrics.csv", dtype={"farm": str})
    g = fm[fm.n_events > 0].groupby("method")[["brier", "recall_0.05"]].mean().reindex(["B0"] + MS)
    d = g2_diffs(fm)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    _bars(ax[0, 0], list(g.index), g["brier"].tolist(), "Brier per metode (lebih kecil = lebih baik)", "Brier")
    _bars(ax[0, 1], list(g.index), g["recall_0.05"].tolist(), "Recall@5% per metode (lebih besar = lebih baik)", "Recall@5%")
    for a, c, lo, hi, t in ((ax[1, 0], "dBrier", "B_lo", "B_hi", "Selisih Brier: A dikurangi baseline\n(di bawah 0 = A lebih baik)"),
                            (ax[1, 1], "dRecall", "R_lo", "R_hi", "Selisih Recall@5%: A dikurangi baseline\n(di atas 0 = A lebih baik)")):
        a.bar(d.vs, d[c], yerr=[d[c] - d[lo], d[hi] - d[c]], capsize=5, color=[COL[b] for b in d.vs])
        a.axhline(0, color="k", lw=1)
        a.set_title(t)
        a.set_ylabel("selisih (garis = CI 95%)")
    fig.suptitle("[G2] Data nyata: A vs baseline (CI mencakup 0 = tidak signifikan)")
    return _save(fig, plt, figdir, "G2_metode_dan_selisih.png")


def plot_e3(root, figdir):
    """[E3] Kurva ROC per metode + diagram reliabilitas A vs B5."""
    plt = _plt()
    e = Path(root) / "explore" / "E3"
    roc, rb = pd.read_csv(e / "roc_points.csv"), pd.read_csv(e / "reliability_bins.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for m in MS:
        r = roc[roc.method == m]
        ax[0].plot(r.fpr, r.tpr, label=m, color=COL[m], lw=2 if m == "A" else 1.2)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="acak")
    ax[0].set(title="Kurva ROC: seberapa baik skor memisahkan\nevent dari negatif", xlabel="Alarm palsu (FPR)", ylabel="Event terdeteksi (TPR)")
    ax[0].legend()
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="kalibrasi sempurna")
    for m in ("B5", "A"):
        r = rb[rb.method == m]
        ax[1].plot(r.mean_p, r.obs_rate, "o-", label=m, color=COL[m])
    ax[1].set(title="Diagram reliabilitas: probabilitas prediksi\nvs frekuensi event sebenarnya", xlabel="Probabilitas prediksi (rata-rata per bin)",
              ylabel="Proporsi event sebenarnya")
    ax[1].legend()
    fig.suptitle("[E3] Data G2: diskriminasi (ROC) dan kalibrasi")
    return _save(fig, plt, figdir, "E3_roc_reliabilitas.png")


def plot_e2(root, figdir):
    """[E2] dBrier(A-B5) per run sumber S6, dengan pita CI G2 asli."""
    plt = _plt()
    e = Path(root) / "explore" / "E2"
    df = pd.read_csv(e / "runs.csv")
    orig = df[df.run == "mismatched|st1|n20|q1.5|ev0.25|seed0"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(orig.ciB_lo, orig.ciB_hi, color="#fdd0a2", alpha=0.6, label="CI 95% G2 asli (ketidakpastian sampling)")
    ax.axhline(0, color="k", lw=1)
    x = range(len(df))
    ax.scatter(x, df.dBrier_farmmean, color=[COL["A"] if v < 0 else "#3182bd" for v in df.dBrier_farmmean], zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r.split("|", 3)[3].replace("|", " ") for r in df.run], rotation=75, fontsize=7)
    ax.set(title="[E2] Selisih Brier A - B5 untuk 18 pilihan parameter sumber (di bawah 0 = A lebih baik)",
           ylabel="dBrier (rata-rata per farm)", xlabel="Run sumber S6: q, ev, seed")
    ax.legend()
    return _save(fig, plt, figdir, "E2_dbrier_per_run.png")


def plot_e1(root, figdir):
    """[E1] recall@5% dan AUC per metode untuk set (a) G2 asli, (b) event+silver, (c) event+hard+silver."""
    plt = _plt()
    df = pd.read_csv(Path(root) / "explore" / "E1" / "metrics_by_set.csv")
    sets = list(dict.fromkeys(df.set))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    w = 0.2
    for a, c, t in ((ax[0], "recall_0.05", "Recall@5% (lebih besar = lebih baik)"), (ax[1], "auc", "AUC (0,5 = acak, 1 = sempurna)")):
        for j, m in enumerate(MS):
            v = [df[(df.set == s) & (df.method == m)][c].iloc[0] for s in sets]
            a.bar([i + (j - 1.5) * w for i in range(len(sets))], v, w, label=m, color=COL[m])
        a.set_xticks(range(len(sets)))
        a.set_xticklabels(["(a) G2 asli", "(b) event+\nsilver", "(c) event+hard+\nsilver"][:len(sets)])
        a.set_title(t)
        a.legend()
    fig.suptitle("[E1] Efek komposisi negatif (silver = tidak diverifikasi manusia, sebagian tautologis)")
    return _save(fig, plt, figdir, "E1_set_negatif.png")


def plot_all(root, figdir):
    return [f(root, figdir) for f in (plot_g1, plot_g2, plot_e3, plot_e2, plot_e1)]
