"""S5a — penambangan kandidat event (loss/gain) otomatis dari matriks observasi S4 + hard negative.

Hanya KANDIDAT untuk diverifikasi manusia lewat filmstrip (S5b); bukan event sampai diverifikasi, jadi aturan sengaja
sederhana dan transparan.
State per (site, epoch): D = terdeteksi; U = tak diketahui; A = absen valid. U bila SALAH SATU: observable False;
fe_n_det == 0 (detektor buta / farm-epoch kosong); penjaga tetangga: site lain dalam 3,0 m (pusat ke pusat, farm sama)
terdeteksi di epoch itu (S4 meninggalkan site duplikat untuk satu pohon; ~10-14% "absen" site stabil jatuh di sini).
D selalu menang atas U (deteksi tetap deteksi). reg_ok False BUKAN U secara default (27% baris; radius gabung 4 m >>
residual tipikal); `--reg-unknown` menjadikannya U. Laporan memuat hitungan kedua varian.
Aturan pada deret per site SETELAH epoch U dibuang (label epoch dipertahankan; urutan leksikal = urutan waktu):
- loss: >= k D berturut-turut langsung disusul >= m A berturut-turut; interval = (D terakhir, A pertama).
- gain: >= m A berturut-turut langsung disusul >= k D berturut-turut; interval = (A terakhir, D pertama).
Satu site boleh punya beberapa transisi; semuanya dikeluarkan. Default k=3, m=2.
Hard negative (tanpa kandidat loss/gain): `stable` = >= 15 epoch valid dan D >= 90% dari epoch valid;
`flicker` = >= 10 epoch valid dan >= 4 pergantian D<->A. Disampel maks `--neg-per-farm` per farm per tipe
(random_state=0); total tetap dilaporkan. `touches_weak_epoch` = interval memuat 2012_08 atau 2020_06 (share deteksi ~4%).
Absen di epoch lemah (WEAK) = U: dengan recall ~4% absen di sana bukan bukti (vault 016). Deteksi di sana tetap D.
`persistent` = loss yang tidak pernah terdeteksi lagi (run A terakhir) / gain yang absen sejak awal lalu tidak pernah
absen lagi (deret = A..A D..D). Recall detektor naik-turun antar epoch, jadi kandidat non-persistent didominasi artefak.
`score_ref` = skor di D terakhir (loss) / D pertama (gain); NaN untuk stable/flicker. k_run/m_run = panjang run D/A
yang sebenarnya.
`nb_rate` (loss/gain; NaN bila tanpa tetangga) = rata-rata `detected` site STABIL (sites.n_epochs_det >= 10) farm yang sama\
dalam NB_M = 30 m dari pusat site (tanpa site itu sendiri) pada epoch >= t_after (loss) / <= t_before (gain). Tetangga tetap\
terlihat saat site ini absen -> citra tidak \"redup\" -> absen lebih kredibel.\
Target pra-registrasi gates.yaml `g2_real_events`: >= 80 loss dan >= 40 gain TERVERIFIKASI di >= 6 farm.

Jalankan (cwd = repo): python pilot/s5_events.py [--s4 pilot/out/s4] [--out pilot/out/s5] [--k 3] [--m 2] \\
          [--neg-per-farm 10] [--reg-unknown]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
GUARD_M = 3.0
NB_M, NB_MIN_EPOCHS = 30.0, 10
WEAK = ("2012_08", "2020_06")
D, A, U = 1, 0, -1
GRID_K, GRID_M = (2, 3, 4), (1, 2, 3)
COLS = ["farm", "site_id", "x", "y", "type", "t_before", "t_after", "k_run", "m_run", "touches_weak_epoch",
        "best_tier", "n_valid", "n_D", "n_A", "n_U", "score_ref", "persistent"]


# ---------- state & run ----------
def farm_states(det, unk, xy):
    """det, unk: (n_site, n_epoch) bool; xy: (n_site, 2). Return int8 D=1 / A=0 / U=-1 dengan penjaga tetangga."""
    nb = np.zeros(det.shape, int)
    pairs = cKDTree(xy).query_pairs(GUARD_M, output_type="ndarray")
    if len(pairs):
        np.add.at(nb, pairs[:, 0], det[pairs[:, 1]])
        np.add.at(nb, pairs[:, 1], det[pairs[:, 0]])
    return np.where(det, D, np.where(unk | (nb > 0), U, A)).astype(np.int8)


def run_list(row):
    """RLE deret state setelah U dibuang -> [(state, idx_pertama, idx_terakhir, panjang)] (idx = kolom epoch asli)."""
    idx = np.flatnonzero(row != U)
    runs = []
    for i in idx:
        s = int(row[i])
        if runs and runs[-1][0] == s:
            runs[-1] = (s, runs[-1][1], int(i), runs[-1][3] + 1)
        else:
            runs.append((s, int(i), int(i), 1))
    return runs


def loss_gain(runs, k, m):
    """Pasangan run bersebelahan -> [(type, run_D, run_A)]; run bersebelahan pasti beda state."""
    out = []
    for a, b in zip(runs, runs[1:]):
        if a[0] == D and a[3] >= k and b[3] >= m:
            out.append(("loss", a, b))
        elif a[0] == A and a[3] >= m and b[3] >= k:
            out.append(("gain", b, a))
    return out


# ---------- tabel site ----------
def prepare(obs, sites, reg_unknown):
    """Return (T, aux): T = 1 baris per site (+ runs, hitungan); aux[farm] = (epochs, matriks skor)."""
    if obs.duplicated(["site_id", "epoch"]).any():
        sys.exit("ERROR: obs punya baris ganda (site_id, epoch)")
    unk = ~obs.observable | (obs.fe_n_det == 0) | obs.epoch.isin(WEAK)
    if reg_unknown:
        unk |= ~obs.reg_ok
    obs = obs.assign(unk=unk)
    parts, aux = [], {}
    for farm, s in sites.groupby("farm", sort=True):
        o = obs[obs.farm == farm]
        eps = sorted(o.epoch.unique())
        if len(o) != len(s) * len(eps) or set(o.site_id) != set(s.site_id):
            sys.exit(f"ERROR: obs farm {farm} bukan grid penuh site x epoch")
        si = pd.Index(s.site_id).get_indexer(o.site_id)
        ei = pd.Index(eps).get_indexer(o.epoch)
        det, un, sc = (np.zeros((len(s), len(eps)), t) for t in (bool, bool, float))
        det[si, ei], un[si, ei], sc[si, ei] = o.detected, o.unk, o.score
        S = farm_states(det, un, s[["x", "y"]].to_numpy())
        t = s[["farm", "site_id", "x", "y", "best_tier"]].copy()
        t["row"] = np.arange(len(s))
        t["n_D"], t["n_A"], t["n_U"] = (S == D).sum(1), (S == A).sum(1), (S == U).sum(1)
        t["n_valid"] = t.n_D + t.n_A
        t["runs"] = [run_list(r) for r in S]
        parts.append(t)
        aux[farm] = (eps, sc)
    return pd.concat(parts, ignore_index=True), aux


def events(T, aux, k, m):
    rows = []
    for t in T.itertuples():
        eps, sc = aux[t.farm]
        for typ, d, a in loss_gain(t.runs, k, m):
            b, c = (d[2], a[1]) if typ == "loss" else (a[2], d[1])       # indeks epoch sebelum / sesudah
            ref = d[2] if typ == "loss" else d[1]
            rows.append((t.farm, t.site_id, t.x, t.y, typ, eps[b], eps[c], d[3], a[3],
                         any(eps[b] <= w <= eps[c] for w in WEAK), t.best_tier,
                         t.n_valid, t.n_D, t.n_A, t.n_U, sc[t.row, ref],
                         a == t.runs[-1] if typ == "loss" else (a == t.runs[0] and d == t.runs[-1])))
    return pd.DataFrame(rows, columns=COLS)


def negatives(T, ev):
    """stable / flicker dari site tanpa kandidat; tipe lain kolom event dibiarkan NaN."""
    n = T[~T.site_id.isin(ev.site_id)]
    switches = n.runs.map(len) - 1
    st = n[(n.n_valid >= 15) & (n.n_D >= 0.9 * n.n_valid)].assign(type="stable")
    fl = n[(n.n_valid >= 10) & (switches >= 4)].assign(type="flicker")
    neg = pd.concat([st, fl], ignore_index=True)
    for c in COLS:
        if c not in neg:
            neg[c] = np.nan
    return neg[COLS].astype({"touches_weak_epoch": object, "persistent": object})


def mine(obs, sites, k, m, reg_unknown):
    T, aux = prepare(obs, sites, reg_unknown)
    ev = events(T, aux, k, m)
    return T, aux, ev, negatives(T, ev)


def nb_rate(ev, obs, sites):
    """Series (indeks ev): lihat docstring modul. Epoch sufiks (loss) / prefiks (gain) -> irisan kolom."""
    out = pd.Series(np.nan, index=ev.index)
    stable = sites[sites.n_epochs_det >= NB_MIN_EPOCHS].set_index("site_id")
    for farm, g in ev[ev.type.isin(["loss", "gain"])].groupby("farm"):
        st = stable[stable.farm == farm]
        if st.empty:
            continue
        det = obs[(obs.farm == farm) & obs.site_id.isin(st.index)].pivot(index="site_id", columns="epoch", values="detected")
        eps, M = list(det.columns), det.loc[st.index].to_numpy(float)
        nbs = cKDTree(st[["x", "y"]].to_numpy()).query_ball_point(g[["x", "y"]].to_numpy(), NB_M)
        for (i, r), nb in zip(g.iterrows(), nbs):
            nb = [j for j in nb if st.index[j] != r.site_id]
            if nb:
                cols = slice(eps.index(r.t_after), None) if r.type == "loss" else slice(0, eps.index(r.t_before) + 1)
                out[i] = M[nb][:, cols].mean()
    return out


def sample_neg(neg, n):
    return pd.concat([g.sample(min(n, len(g)), random_state=0) for _, g in neg.groupby(["farm", "type"])] or [neg])


# ---------- laporan ----------
def make_report(ev, neg, neg_s, T, aux, alt, grid, elapsed, args):
    nloss, ngain = (ev.type == t for t in ("loss", "gain"))
    L = ["# S5a kandidat event (loss/gain) + hard negative", "",
         f"Waktu jalan: {elapsed:.1f} s; k={args.k}, m={args.m}, `--reg-unknown`={args.reg_unknown}.", "",
         "**Semua baris di sini KANDIDAT, bukan event, sampai diverifikasi manusia (S5b).** Target pra-registrasi "
         "`gates.yaml` `g2_real_events`: >= 80 loss dan >= 40 gain TERVERIFIKASI di >= 6 farm.", "",
         "## Input", f"- `{args.s4}/sites.parquet` ({T.farm.nunique()} farm, {len(T)} site), `{args.s4}/obs.parquet`.",
         f"- Epoch valid (non-U) {int(T.n_valid.sum())}, U {int(T.n_U.sum())} "
         f"({T.n_U.sum() / (T.n_valid.sum() + T.n_U.sum()):.3f} dari baris obs).", "",
         "## Aturan",
         "- U: observable False, atau fe_n_det == 0, atau tetangga <= 3,0 m terdeteksi di epoch itu; D menang atas U; "
         "reg_ok False hanya U bila `--reg-unknown`.",
         "- loss: >= k D lalu >= m A (interval = D terakhir -> A pertama); gain: >= m A lalu >= k D "
         "(A terakhir -> D pertama); epoch U dibuang dulu.",
         "- stable: >= 15 epoch valid, D >= 90%, tanpa kandidat; flicker: >= 10 epoch valid, tanpa kandidat, >= 4 pergantian D<->A.",
         f"- Negatif disampel maks {args.neg_per_farm} per farm per tipe (random_state=0).", "",
         "## Jumlah per tipe", "| tipe | total | di output | farm |", "|---|---|---|---|"]
    for t in ("loss", "gain"):
        L.append(f"| {t} | {(ev.type == t).sum()} | {(ev.type == t).sum()} | {ev[ev.type == t].farm.nunique()} |")
    for t in ("stable", "flicker"):
        L.append(f"| {t} | {(neg.type == t).sum()} | {(neg_s.type == t).sum()} | {neg[neg.type == t].farm.nunique()} |")
    L += ["", "## Per farm (kandidat)", "| farm | loss | gain |", "|---|---|---|"]
    pf = ev.groupby(["farm", "type"]).size().unstack(fill_value=0).reindex(sorted(T.farm.unique()), fill_value=0)
    pf = pf.reindex(columns=["loss", "gain"], fill_value=0)
    L += [f"| {f} | {r.loss} | {r.gain} |" for f, r in pf.iterrows()]
    L += ["", "## Histogram interval (t_before -> t_after), 20 teratas per tipe"]
    for t in ("loss", "gain"):
        h = ev[ev.type == t].groupby(["t_before", "t_after"]).size().sort_values(ascending=False)
        L += ["", f"### {t}", "| interval | n | share |", "|---|---|---|"]
        L += [f"| {a} -> {b} | {n} | {n / max(len(ev[ev.type == t]), 1):.3f} |" for (a, b), n in h.head(20).items()]
    L += ["", "## Persistent (loss tak pernah kembali / gain tak pernah absen lagi)", "| tipe | n | farm | interval teratas |", "|---|---|---|---|"]
    for t in ("loss", "gain"):
        p = ev[(ev.type == t) & ev.persistent.astype(bool)]
        top = ", ".join(f"{a}->{b}: {n}" for (a, b), n in p.groupby(["t_before", "t_after"]).size().nlargest(4).items())
        L.append(f"| {t} | {len(p)} | {p.farm.nunique()} | {top} |")
    L += ["", "## nb_rate >= 0,6 (tetangga stabil <= 30 m tetap terlihat; absen lebih kredibel)", "| tipe | semua | persistent |", "|---|---|---|"]
    for t in ("loss", "gain"):
        e = ev[ev.type == t]
        ok = e.nb_rate >= 0.6
        L.append(f"| {t} | {int(ok.sum())} dari {len(e)} | {int((ok & e.persistent.astype(bool)).sum())} dari {int(e.persistent.astype(bool).sum())} |")
    L += ["", "## Menyentuh epoch lemah (2012_08 / 2020_06)"]
    for t in ("loss", "gain"):
        s = ev[ev.type == t].touches_weak_epoch
        L.append(f"- {t}: {int(s.sum())} dari {len(s)} ({s.mean() if len(s) else 0:.3f}).")
    L += ["", "## Varian reg_ok (default k, m)", "| reg_ok False | loss | gain | farm loss>=1 | stable | flicker |", "|---|---|---|---|---|---|"]
    for name, (e, n) in alt.items():
        L.append(f"| {name} | {(e.type == 'loss').sum()} | {(e.type == 'gain').sum()} | "
                 f"{e[e.type == 'loss'].farm.nunique()} | {(n.type == 'stable').sum()} | {(n.type == 'flicker').sum()} |")
    L += ["", "## Sensitivitas k x m (reg_ok False tidak U)", "| k | m | loss | gain | farm loss>=1 |", "|---|---|---|---|---|"]
    L += [f"| {k} | {m} | {nl} | {ng} | {nf} |" for k, m, nl, ng, nf in grid]
    L += ["", "## Catatan",
          "- Kandidat non-persistent didominasi naik-turun recall detektor antar epoch; prioritaskan `persistent` untuk S5b.",
          "- Site = lokasi kandidat; absen bisa tetap artefak deteksi (tier/skor tidak difilter, epoch lemah, citra berbeda).",
          "- Satu site bisa punya beberapa kandidat; satu run A bisa jadi akhir sebuah loss dan awal sebuah gain."]
    return "\n".join(L) + "\n"


# ---------- driver ----------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s4", default=str(HERE / "out" / "s4"), help="folder berisi sites.parquet & obs.parquet")
    ap.add_argument("--out", default=str(HERE / "out" / "s5"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--neg-per-farm", type=int, default=10)
    ap.add_argument("--reg-unknown", action="store_true", help="reg_ok False dianggap U")
    a = ap.parse_args(argv)
    if a.k < 1 or a.m < 1 or a.neg_per_farm < 0:
        sys.exit("ERROR: k, m >= 1 dan neg-per-farm >= 0")
    for f in ("sites", "obs"):
        if not (Path(a.s4) / f"{f}.parquet").is_file():
            sys.exit(f"ERROR: tidak ada {Path(a.s4) / f'{f}.parquet'}")
    t0 = time.time()
    sites = pd.read_parquet(Path(a.s4) / "sites.parquet")
    obs = pd.read_parquet(Path(a.s4) / "obs.parquet")
    res = {}
    for ru in (False, True):
        res[ru] = mine(obs, sites, a.k, a.m, ru)
        print(f"reg_unknown={ru}: {res[ru][2].type.value_counts().to_dict()} {time.time() - t0:.0f}s", file=sys.stderr)
    T, aux, ev, neg = res[a.reg_unknown]
    ev = ev.assign(nb_rate=nb_rate(ev, obs, sites))
    alt = {"tidak U": res[False][2:], "U": res[True][2:]}
    nl = lambda e: e[e.type == "loss"]  # noqa: E731
    grid = []
    for k in GRID_K:
        for m in GRID_M:
            e = events(res[False][0], res[False][1], k, m)
            grid.append((k, m, (e.type == "loss").sum(), (e.type == "gain").sum(), nl(e).farm.nunique()))
    neg_s = sample_neg(neg, a.neg_per_farm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = pd.concat([ev, neg_s], ignore_index=True)[COLS + ["nb_rate"]].astype({"k_run": "Int64", "m_run": "Int64", "touches_weak_epoch": "boolean", "persistent": "boolean"})
    c.to_parquet(out / "candidates.parquet", index=False)
    elapsed = time.time() - t0
    (out / "report.md").write_text(make_report(ev, neg, neg_s, T, aux, alt, grid, elapsed, a))
    print(f"S5a selesai {elapsed:.1f}s: {len(ev)} kandidat, {len(neg_s)}/{len(neg)} negatif -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
