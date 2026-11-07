"""Tes S0. Jalankan: python pilot/tests/test_s0_audit.py  (atau pytest). Bagian raster di-skip tanpa rasterio."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s0_audit as s0  # noqa: E402

G0 = dict(min_farms=30, min_valid_epochs=12, max_nodata_pct=10)


def row(farm, epoch, nodata=0.0, sha=True, mask=True):
    return dict(farm=farm, epoch=epoch, has_clip=True, has_mask=mask, clip_sha_ok=sha,
                mask_sha_ok=sha, nodata_pct=nodata)


def test_parse_and_junk():
    assert s0.parse_epoch("QATAR_2025_04_C050_PLE") == "2025_04"
    assert s0.parse_epoch("foo") is None and s0.parse_epoch("QATAR_25_4_x") is None
    assert s0.is_junk(".azDownload-abc-1.tif") and s0.is_junk(".DS_Store") and not s0.is_junk("F037")


def test_sha():
    assert s0.sha_match("AB", "ab") and not s0.sha_match("ab", "cd")
    assert not s0.sha_match("ab", None) and not s0.sha_match(None, "ab")


def test_is_valid_boundaries():
    assert s0.is_valid(row("f", "e", nodata=10.0), 10)
    assert not s0.is_valid(row("f", "e", nodata=10.01), 10)
    assert not s0.is_valid(row("f", "e", nodata=None), 10)
    assert not s0.is_valid(row("f", "e", nodata=float("nan")), 10)
    assert not s0.is_valid(row("f", "e", sha=False), 10)
    assert not s0.is_valid(row("f", "e", mask=False), 10)


def gate_rows(n_farms, n_epochs):
    return [row(f"F{i}", f"2020_{e:02d}") for i in range(n_farms) for e in range(1, n_epochs + 1)]


def test_gate_thresholds():
    g = s0.evaluate_gate(gate_rows(30, 12), G0)  # tepat di ambang -> PASS
    assert g["verdict"] == "PASS" and g["n_eligible"] == 30
    assert s0.evaluate_gate(gate_rows(29, 20), G0)["verdict"] == "FAIL"
    assert s0.evaluate_gate(gate_rows(36, 11), G0)["verdict"] == "FAIL"
    # duplikat epoch (dua service sama YYYY_MM) dihitung sekali
    assert s0.evaluate_gate(gate_rows(30, 11) + [row("F0", "2020_01")], G0)["n_eligible"] == 0


def make_tree(root):
    """Farm 1: pasangan lengkap tapi sha clip salah; farm 2: mask hilang. Plus junk."""
    svc = "QATAR_2025_04_C050_PLE"
    for farm in ("1", "2"):
        d = root / svc / farm
        d.mkdir(parents=True)
        (d / f"{farm}_{svc}.tif").write_bytes(b"clip" + farm.encode())
        (d / ".azDownload-x-1.tif").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"x")
    (root / svc / "1" / f"1_{svc}_mask.tif").write_bytes(b"mask")
    sha = lambda b: hashlib.sha256(b).hexdigest()
    (root / svc / "1" / f"1_{svc}.json").write_text(json.dumps(
        dict(raster_sha256="0" * 64, mask_sha256=sha(b"mask"), acquired_utc="2026-09-20T00:00:00Z")))
    return svc


def test_scan_and_audit_without_raster():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        svc = make_tree(root)
        found = [(s, f) for s, f, _ in s0.scan(root)]
        assert found == [(svc, "1"), (svc, "2")]
        r1 = s0.audit_pair(svc, "1", root / svc / "1") if s0.rasterio else None
        d2 = s0.audit_pair(svc, "2", root / svc / "2")
        assert d2["has_clip"] and not d2["has_mask"] and not d2["has_json"] and not d2["clip_sha_ok"]
        assert d2["acquired_utc_is_image_date"] is False
        if r1:  # raster palsu -> error baca tercatat, bukan crash
            assert r1["mask_sha_ok"] and not r1["clip_sha_ok"] and r1["error"]
        rows = [d2]
        out = root / "out"
        out.mkdir()
        name = s0.write_manifest(rows, out)
        assert (out / name.split()[0]).exists()
        rep = s0.make_report(rows, G0, s0.evaluate_gate(rows, G0), name)
        assert "FAIL" in rep and "bukan tanggal citra" in rep and "defisit 720" in rep


def test_raster_stats_synthetic():
    if s0.rasterio is None:
        print("SKIP test_raster_stats_synthetic: rasterio tidak terpasang")
        return
    import numpy as np
    from rasterio.transform import from_origin
    with tempfile.TemporaryDirectory() as t:
        tif, msk = Path(t) / "c.tif", Path(t) / "m.tif"
        kw = dict(driver="GTiff", height=10, width=10, transform=from_origin(0, 10, 0.5, 0.5),
                  crs="EPSG:2932")
        data = np.ones((4, 10, 10), np.uint8)
        data[:, :2, :] = 0  # 2 baris nodata (20 px)
        m = np.zeros((10, 10), np.uint8)
        m[:4, :] = 1  # 40 px dalam farm; 20 di antaranya nodata
        with s0.rasterio.open(tif, "w", count=4, dtype="uint8", **kw) as d:
            d.write(data)
        with s0.rasterio.open(msk, "w", count=1, dtype="uint8", **kw) as d:
            d.write(m, 1)
        st = s0.raster_stats(tif, msk, 0.0)
        assert st["bands"] == 4 and st["gsd_x"] == 0.5 and abs(st["area_m2"] - 25.0) < 1e-9
        assert st["mask_inside_px"] == 40 and st["nodata_inside_px"] == 20
        assert abs(st["nodata_pct"] - 50.0) < 1e-9 and abs(st["mask_coverage"] - 0.4) < 1e-9
        assert st["mask_values"] == [0, 1] and st["grid_match"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
