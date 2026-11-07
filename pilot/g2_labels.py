"""G2 langkah 1: gabungkan vonis review S5b (batch 1 + 2, tier high saja) jadi tabel label event nyata.

label: loss | gain | none (stable/tidak ada/ada terus) | ambigu | tolak (salah arah / bukan pohon).
change_epoch: loss -> epoch hilang terakhir (second_change bila muncul_hilang); gain -> epoch muncul pertama.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
BATCHES = {"b1": OUT / "s5b", "b2": OUT / "s5b_r2"}


# note owner = alasan keputusan: cincin tidak di atas tajuk / bukan target -> lokasi tidak sah, apa pun verdict-nya
BAD_NOTE = ("meleset", "tidak tepat", "gap", "celah", "bukan pohon kurma", "greenhouse", "tengah cincin")


def label(typ, verdict, note=""):
    if verdict == "ambigu":
        return "ambigu"
    if any(w in note.lower() for w in BAD_NOTE):
        return "tolak"
    if typ == "loss" and verdict in ("hilang", "muncul_hilang"):
        return "loss"
    if typ == "gain" and verdict == "muncul":
        return "gain"
    if verdict in ("ada", "tidak_ada"):
        return "none"  # manusia: tidak berubah (negatif), apa pun tipe kandidatnya
    return "tolak"  # arah terbalik, dst.


def load(batch, d):
    k = pd.read_parquet(d / "key.parquet")
    r = pd.read_csv(d / "review.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    df = k.merge(r.drop(columns="farm"), on="review_id", validate="1:1")
    df["batch"] = batch
    rr = OUT / "s5b_rerev" / "review.csv"
    if rr.exists():  # note dari re-review buta (K01-K30) ikut berlaku untuk item aslinya
        kk = pd.read_parquet(OUT / "g2" / "rerev_key.parquet")[["new_id", "review_id"]]
        n = pd.read_csv(rr, sep=";", encoding="utf-8-sig", dtype=str).fillna("")[["review_id", "note"]]
        extra = kk.merge(n, left_on="new_id", right_on="review_id", suffixes=("", "_k")).set_index("review_id")["note"]
        df["note"] = (df["note"] + " " + df["review_id"].map(extra).fillna("")).str.strip()
    df["label"] = [label(t, v, n) for t, v, n in zip(df["type"], df["verdict"], df["note"])]
    df["change_epoch"] = [b if v == "muncul_hilang" else a
                          for v, a, b in zip(df["verdict"], df["first_change_epoch"], df["second_change_epoch"])]
    return df


if __name__ == "__main__":
    df = pd.concat([load(b, d) for b, d in BATCHES.items()], ignore_index=True)
    assert len(df) == 220 and df["review_id"].is_unique, "review_id harus unik lintas batch"
    df = df[df["best_tier"] == "high"].copy()
    (OUT / "g2").mkdir(exist_ok=True)
    df.to_parquet(OUT / "g2" / "labels.parquet", index=False)
    print(df.groupby(["type", "label"]).size().unstack(fill_value=0))
    for t in ("loss", "gain"):
        s = df[df["label"] == t]
        print(t, "n =", len(s), "farm =", s["farm"].nunique())
