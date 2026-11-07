"""G2 kappa intra-rater: 30 item acak (dari label G2 tier high) diberi id baru K01-K30, urutan baru, gambar disalin.

Kunci (K -> review_id lama) disimpan di out/g2/, BUKAN di folder paket, supaya penilai tetap buta.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2_labels import BATCHES, OUT  # noqa: E402
from s5b_filmstrip import make_html  # noqa: E402

N, SEED = 30, 7
PKG = OUT / "s5b_rerev"

if __name__ == "__main__":
    lab = pd.read_parquet(OUT / "g2" / "labels.parquet")
    lab["grp"] = lab["label"].replace({"none": "neg", "tolak": "neg"})
    # strata kira-kira proporsional, tiap kelas terwakili
    quota = {"loss": 10, "gain": 8, "neg": 7, "ambigu": 5}
    s = pd.concat([g.sample(quota[k], random_state=SEED) for k, g in lab.groupby("grp")])
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["new_id"] = [f"K{i + 1:02d}" for i in range(len(s))]
    assert len(s) == N
    (PKG / "img").mkdir(parents=True, exist_ok=True)
    epochs = {}
    for r in s.itertuples():
        src = sorted((BATCHES[r.batch] / "img").glob(f"{r.review_id}_*.png"))
        assert len(src) == 20, (r.review_id, len(src))
        eps = [p.stem[len(r.review_id) + 1:] for p in src]
        epochs[r.new_id] = eps
        for p, e in zip(src, eps):
            shutil.copy(p, PKG / "img" / f"{r.new_id}_{e}.png")
    items = s[["new_id"]].rename(columns={"new_id": "review_id"}).assign(farm=s["new_id"])
    html = make_html(items, epochs).replace("&mdash; farm ", "&mdash; ")  # sembunyikan farm (bantu ingatan)
    (PKG / "index.html").write_text(html, encoding="utf-8")
    pd.DataFrame({"review_id": s["new_id"], "verdict": "", "first_change_epoch": "", "second_change_epoch": "",
                  "note": ""}).to_csv(PKG / "review.csv", sep=";", index=False, encoding="utf-8-sig")
    s[["new_id", "review_id", "batch", "farm", "verdict", "label"]].to_parquet(OUT / "g2" / "rerev_key.parquet")
    print(s["grp"].value_counts().to_dict(), "->", PKG)
