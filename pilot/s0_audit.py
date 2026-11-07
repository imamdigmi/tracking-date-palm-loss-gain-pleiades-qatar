"""S0 — audit data multi-temporal + manifest + gate G0 (pilot/gates.yaml: g0_data).

Layout: <data-root>/<SERVICE>/<FARM>/<FARM>_<SERVICE>{.tif,_mask.tif,.json,...}
Mask: 1 di dalam farm / 0 di luar (acquire_farm_rasters.py SPEC 5.2), grid sama dengan clip.
Nodata dihitung dari RASTER (semua band == nodata) hanya pada piksel mask==1, karena mask
hanya batas farm, bukan mask validitas.

Paket yang perlu diinstal: rasterio, PyYAML (opsional: pyarrow untuk parquet; tanpa itu CSV).
Jalankan: python pilot/s0_audit.py --data-root "<path/to/datasets>" [--out pilot/out/s0]
"""
import argparse
import os
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    import rasterio
except ImportError:
    rasterio = None
try:
    import yaml
except ImportError:
    yaml = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data/datasets")
EXPECTED_FARMS, EXPECTED_EPOCHS = 36, 20
SERVICE_RE = re.compile(r"^QATAR_(\d{4})_(\d{2})_")
# Band order dipaksa (field band_order di JSON diabaikan sesuai instruksi owner).
# NB: komentar di acquire_farm_rasters.py (diukur 2026-09-15) bilang R-G-B-NIR; konflik ini ditandai di laporan.
FORCED_BAND_ORDER = {4: "R-G-B-NIR", 3: "R-G-B"}


def need(mod, pkg, why):
    if mod is None:
        sys.exit(f"ERROR: paket '{pkg}' belum terpasang ({why}). Install: pip install {pkg}")
    return mod


def parse_epoch(service):
    """'QATAR_2025_04_C050_PLE' -> '2025_04'; None bila nama tak cocok."""
    m = SERVICE_RE.match(service)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def is_junk(name):
    return name.startswith(".azDownload-") or name == ".DS_Store"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha_match(actual, expected):
    """True hanya bila keduanya ada dan sama (case-insensitive)."""
    return bool(actual) and bool(expected) and str(actual).lower() == str(expected).strip().lower()


def read_sidecar(path):
    try:
        d = json.loads(Path(path).read_text())
        return (d, None) if isinstance(d, dict) else ({}, "json bukan object")
    except (OSError, ValueError) as e:
        return {}, f"{type(e).__name__}: {e}"


def scan(data_root):
    """Yield (service, farm, dir) untuk service QATAR_YYYY_MM_* dan sub-dir farm; junk diabaikan."""
    for sdir in sorted(Path(data_root).iterdir()):
        if is_junk(sdir.name) or not sdir.is_dir() or not parse_epoch(sdir.name):
            continue
        for fdir in sorted(sdir.iterdir()):
            if not is_junk(fdir.name) and fdir.is_dir():
                yield sdir.name, fdir.name, fdir


def raster_stats(tif, mask, nodata_fallback):
    """Baca per blok (clip bisa ~GB). nodata_pct = % piksel mask==1 yang semua band == nodata."""
    import numpy as np
    from rasterio.windows import Window
    rio = need(rasterio, "rasterio", "membaca raster")
    with rio.open(tif) as src, rio.open(mask) as msk:
        nd = src.nodata if src.nodata is not None else nodata_fallback
        out = dict(bands=src.count, width=src.width, height=src.height,
                   gsd_x=abs(src.transform.a), gsd_y=abs(src.transform.e),
                   crs=str(src.crs) if src.crs else None, nodata_used=nd,
                   nodata_from="raster" if src.nodata is not None else "sidecar/0",
                   dtype=src.dtypes[0])
        out["area_m2"] = src.width * src.height * out["gsd_x"] * out["gsd_y"]
        out["grid_match"] = (src.width, src.height, src.transform) == (msk.width, msk.height, msk.transform)
        out["mask_values"] = None
        out["mask_inside_px"] = out["nodata_inside_px"] = None
        out["nodata_pct"] = out["mask_coverage"] = None
        if not out["grid_match"]:  # tanpa grid sama, nodata dalam mask tak terdefinisi
            return out
        vals, inside, bad = set(), 0, 0
        for _, win in src.block_windows(1):
            m = msk.read(1, window=win)
            vals.update(np.unique(m).tolist())
            sel = m == 1
            n = int(sel.sum())
            if n:
                inside += n
                bad += int((np.all(src.read(window=win) == nd, axis=0) & sel).sum())
        out.update(mask_values=sorted(vals), mask_inside_px=inside, nodata_inside_px=bad,
                   mask_coverage=inside / (src.width * src.height),
                   nodata_pct=100.0 * bad / inside if inside else None)
        return out


def audit_pair(service, farm, d):
    tif, mask, js = (d / f"{farm}_{service}{s}" for s in (".tif", "_mask.tif", ".json"))
    row = dict(service=service, epoch=parse_epoch(service), farm=farm,
               has_clip=tif.is_file(), has_mask=mask.is_file(), has_json=js.is_file(),
               acquired_utc=None, acquired_utc_is_image_date=False,  # waktu unduh
               service_url=None, sidecar_error=None, clip_sha_ok=False, mask_sha_ok=False,
               error=None)
    sc, row["sidecar_error"] = read_sidecar(js) if row["has_json"] else ({}, None)
    row["acquired_utc"], row["service_url"] = sc.get("acquired_utc"), sc.get("service_url")
    if row["has_clip"]:
        row["clip_sha_ok"] = sha_match(sha256_file(tif), sc.get("raster_sha256"))
    if row["has_mask"]:
        row["mask_sha_ok"] = sha_match(sha256_file(mask), sc.get("mask_sha256"))
    if row["has_clip"] and row["has_mask"]:
        try:
            fb = float(sc.get("nodata_value"))
        except (TypeError, ValueError):
            fb = 0.0
        try:
            st = raster_stats(tif, mask, fb)
            st["mask_binary"] = st["mask_values"] is not None and set(st["mask_values"]) <= {0, 1}
            st["band_order_forced"] = FORCED_BAND_ORDER.get(st["bands"], "unknown")
            st["mask_values"] = json.dumps(st["mask_values"])
            row.update(st)
        except Exception as e:  # satu file rusak tidak boleh menggagalkan audit; dicatat dan dianggap tidak valid
            row["error"] = f"{type(e).__name__}: {e}"
    return row


def is_valid(row, max_nodata_pct):
    p = row.get("nodata_pct")
    return bool(row["has_clip"] and row["has_mask"] and row["clip_sha_ok"] and row["mask_sha_ok"]
                and p is not None and not math.isnan(p) and p <= max_nodata_pct)


def evaluate_gate(rows, g0):
    """rows: dict-baris manifest. Epoch valid = ada >=1 pasangan valid untuk (farm, epoch)."""
    valid = {}
    for r in rows:
        valid.setdefault(r["farm"], set())
        if is_valid(r, g0["max_nodata_pct"]):
            valid[r["farm"]].add(r["epoch"])
    n_valid = {f: len(e) for f, e in valid.items()}
    eligible = sorted(f for f, n in n_valid.items() if n >= g0["min_valid_epochs"])
    return dict(valid_epochs_per_farm=n_valid, eligible_farms=eligible,
                n_eligible=len(eligible), min_farms=g0["min_farms"],
                verdict="PASS" if len(eligible) >= g0["min_farms"] else "FAIL")


def load_g0(gates_path):
    y = need(yaml, "PyYAML", "membaca gates.yaml")
    g0 = y.safe_load(Path(gates_path).read_text())["g0_data"]
    for k in ("min_farms", "min_valid_epochs", "max_nodata_pct"):
        if k not in g0:
            sys.exit(f"ERROR: g0_data.{k} tidak ada di {gates_path}")
    return g0


def write_manifest(rows, out):
    if pa is not None:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        pq.write_table(pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in rows]),
                       out / "manifest.parquet")
        return "manifest.parquet"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return "manifest.csv (pyarrow tidak ada)"


def make_report(rows, g0, gate, manifest_name):
    pairs = [r for r in rows if r["has_clip"] and r["has_mask"]]
    farms = sorted({r["farm"] for r in rows})
    epochs = sorted({r["epoch"] for r in rows})
    sha_bad = [r for r in pairs if not (r["clip_sha_ok"] and r["mask_sha_ok"])]
    nonbin = [r for r in pairs if r.get("mask_binary") is False]
    gridbad = [r for r in pairs if r.get("grid_match") is False]
    errs = [r for r in pairs if r["error"]]
    gsd = sorted({round(r["gsd_x"], 4) for r in pairs if r.get("gsd_x")})
    bands = sorted({r["bands"] for r in pairs if r.get("bands")})
    nd = [r["nodata_pct"] for r in pairs if r.get("nodata_pct") is not None]
    cov = [r["mask_coverage"] for r in pairs if r.get("mask_coverage") is not None]
    rng = lambda v, fmt: f"{fmt % min(v)} – {fmt % max(v)}" if v else "n/a"
    L = [f"# S0 audit — G0 data: **{gate['verdict']}**", "",
         f"Farm layak {gate['n_eligible']} / min {g0['min_farms']} "
         f"(epoch valid ≥ {g0['min_valid_epochs']}, nodata ≤ {g0['max_nodata_pct']}% dalam mask, sha cocok).", "",
         "## Kelengkapan",
         f"- Service/epoch ditemukan: {len(epochs)} (target {EXPECTED_EPOCHS}); farm: {len(farms)} (target {EXPECTED_FARMS})",
         f"- Pasangan clip+mask: {len(pairs)} / {EXPECTED_FARMS * EXPECTED_EPOCHS}; "
         f"**defisit {EXPECTED_FARMS * EXPECTED_EPOCHS - len(pairs)}**",
         f"- Entri tanpa clip/mask: {len(rows) - len(pairs)}; tanpa JSON: {sum(not r['has_json'] for r in rows)}",
         "## Kualitas",
         f"- GSD (m): {gsd or 'n/a'}; band count: {bands or 'n/a'}; band order R-G-B-NIR (3 band: R-G-B), diverifikasi manual di QGIS+ArcGIS pada farm F003 epoch 2025_04 saja",
         f"- sha256 tidak cocok/tidak ada di JSON: {len(sha_bad)} pasangan",
         f"- nodata dalam mask (%): {rng(nd, '%.2f')}; coverage mask (frac clip): {rng(cov, '%.3f')}",
         f"- Isi mask non-biner (nilai di luar {{0,1}}): {len(nonbin)}; grid mask≠raster: {len(gridbad)}; "
         f"error baca: {len(errs)}",
         "- Mask hanya batas farm (1 dalam / 0 luar), BUKAN mask validitas piksel → nodata diukur dari raster.",
         "- `acquired_utc` di JSON = waktu unduh, **bukan tanggal citra**; tanggal citra tidak ada di data ini.",
         "- Catatan: urutan band R-G-B-NIR cocok dengan acquire_farm_rasters.py; belum diverifikasi di epoch lain.",
         "", "## Epoch valid per farm", "| farm | valid | layak |", "|---|---|---|"]
    for f in farms:
        n = gate["valid_epochs_per_farm"].get(f, 0)
        L.append(f"| {f} | {n} | {'ya' if f in gate['eligible_farms'] else 'tidak'} |")
    L += ["", f"Manifest: `{manifest_name}` ({len(rows)} baris)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=str(HERE / "out" / "s0"))
    ap.add_argument("--gates", default=str(HERE / "gates.yaml"))
    a = ap.parse_args(argv)
    need(rasterio, "rasterio", "membaca raster")
    g0 = load_g0(a.gates)
    root = Path(a.data_root)
    if not root.is_dir():
        sys.exit(f"ERROR: --data-root tidak ada: {root}")
    rows = []
    for service, farm, d in scan(root):
        rows.append(audit_pair(service, farm, d))
        print(f"{service}/{farm}", file=sys.stderr)
    if not rows:
        sys.exit(f"ERROR: tidak ada service/farm di {root}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = evaluate_gate(rows, g0)
    name = write_manifest(rows, out)
    (out / "report.md").write_text(make_report(rows, g0, gate, name))
    print(f"G0 {gate['verdict']}: {gate['n_eligible']}/{g0['min_farms']} farm layak -> {out}")
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
