"""Input-format tests on tiny synthetic scans (scripts/make_synthetic_volume.py).

No MATLAB is needed at test time: the legacy reference outputs were produced once with
the UNMODIFIED myOCT code and stored in tests/data/legacy_formats_ref.npz by

    python tests/test_formats.py --build-reference      (needs `matlab` on PATH, ~3 min)

which regenerates the synthetic inputs (deterministic, seeded; a checksum of every raw
input is stored and checked) and runs validation/matlab/dump_synthetic_plane.m (serial copy
of the yOCTProcessTiledScan plane loop) and validation/matlab/dump_legacy_loader.m
(yOCTLoadInterfFromFile / header functions / WhatOCTSystemIsIt).

Run:  pytest tests/test_formats.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_volume import make_volume  # noqa: E402

from octrecon.core.spectral import legacy_interferogram  # noqa: E402
from octrecon.core.stitching import MIN_WEIGHT  # noqa: E402
from octrecon.io.detect import what_oct_system_is_it  # noqa: E402
from octrecon.io.scaninfo import ScanInfo  # noqa: E402
from octrecon.io.volume import TileReader, extract_volume, scan_volume_layout, tile_layout  # noqa: E402
from octrecon.pipeline import DATA_DIR, ReconConfig, Reconstructor  # noqa: E402

REF = Path(__file__).parent / "data" / "legacy_formats_ref.npz"
SIM = Path(__file__).parent / "data" / "simulated_scan"   # written by legacy yOCTSimulateTileScan
DISPERSION = 8.949e7

# ---- tiled volumes: full pipeline vs legacy yOCTProcessTiledScan plane (y = 2) ----------
_TILED = dict(nx=8, ny=2, tiles_x=2, tiles_y=1, depths=(0.0, 0.01), n_lambda=2048, seed=1)
PIPELINE_CASES = {
    "base": dict(),
    "bavg2": dict(bscan_avg=2),
    "aavg2": dict(ascan_avg=2),
    "sbin2": dict(spectra_avg=2),
    "sbin3": dict(spectra_avg=3),
    "combo": dict(bscan_avg=2, ascan_avg=2, spectra_avg=2),
    "ganymede": dict(system="ganymede"),
    "telesto": dict(system="telesto"),
}
REF_PLANE_Y = 2  # 1-based output plane compared

# ---- single tiles: reader vs yOCTLoadInterfFromFile ------------------------------------
_TILE = dict(nx=6, ny=3, tiles_x=1, tiles_y=1, depths=(0.0,), n_lambda=256, apod=5, seed=2)
LOADER_CASES = {  # name: (make_volume kwargs, legacy kind, octSystem passed, y frame)
    "thor": (dict(bscan_avg=2, ascan_avg=2, spectra_avg=3), "thorlabs", "gan632", 2),
    "thor_tel": (dict(system="telesto", spectra_avg=2), "thorlabs", "", 1),
    "srr": (dict(fmt="srr", bscan_avg=2), "srr", "Ganymede_SRR", 2),
    # legacy SRR header takes the system from the file name ('Telesto', without _SRR) when
    # none is given, which then selects the non-SRR Thorlabs data reader -> pass it explicitly
    "srr_tel": (dict(fmt="srr", system="telesto"), "srr", "Telesto_SRR", 3),
    "was": (dict(fmt="wasatch", bscan_avg=2), "wasatch", "Wasatch", 2),
    "was_tif": (dict(fmt="wasatch_tif", ny=1, bscan_avg=3, apod=4), "wasatch", "Wasatch", 1),
    "was_tif_nobg": (dict(fmt="wasatch_tif", ny=1, bscan_avg=2, apod=0), "wasatch", "Wasatch", 1),
}
DETECT_CASES = {  # name: (loader case providing the folder, expected legacy result or None=error)
    "det_thor": ("thor", ("Ganymede", "Thorlabs")),     # GAN632 header says Series=Ganymede
    "det_thor_tel": ("thor_tel", ("Telesto", "Thorlabs")),
    "det_srr": ("srr", ("Ganymede_SRR", "Thorlabs_SRR")),
    "det_srr_tel": ("srr_tel", ("Telesto_SRR", "Thorlabs_SRR")),
    "det_was": ("was", ("Wasatch", "Wasatch")),
    "det_was_tif": ("was_tif", None),                   # legacy textscan quirk drops all .tif
}


def _dir_checksum(folder: Path) -> str:
    h = hashlib.sha1()
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix not in (".json", ".mat"):
            h.update(p.relative_to(folder).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _make_tiled(root: Path, name: str, **extra) -> Path:
    kw = dict(_TILED)
    kw.update(PIPELINE_CASES.get(name, {}))
    kw.update(extra)
    make_volume(root / name, **kw)
    return root / name


def _make_tile(root: Path, name: str) -> Path:
    kw = dict(_TILE)
    kw.update(LOADER_CASES[name][0])
    make_volume(root / name, **kw)
    return root / name / "Data01"


def _recon(vol: Path, precision="float64", **kw):
    info = json.loads((vol / "synthetic_info.json").read_text())
    cfg = ReconConfig(fep_removal=False, volume_folder=str(vol), device="cpu", precision=precision,
                      dispersion_quadratic_term=info.get("dispersion", DISPERSION),
                      focus_positions=float(info["focus_pix"]),
                      focus_sigma=10, crop_z_range_mm="none", output_pixel_size_um=info.get("pixel_um", 2),
                      interp_method="sinc5", batch_frames=4, io_threads=2, **kw)
    return Reconstructor(cfg, log=lambda *a: None)


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("reference file missing; run python tests/test_formats.py --build-reference")
    return dict(np.load(REF, allow_pickle=False))


@pytest.fixture(scope="module")
def work():
    d = Path(tempfile.mkdtemp(prefix="octrecon_formats_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =========================================================================== layouts
def test_unzipped_vs_oct_inplace_vs_extracted(work):
    """Same data stored unzipped, as .oct read in place, and .oct extracted -> identical."""
    a = _make_tiled(work, "lay_unzipped", layout="unzipped", bscan_avg=2)
    b = _make_tiled(work, "lay_oct", layout="oct", bscan_avg=2)
    c = _make_tiled(work, "lay_oct_x", layout="oct", bscan_avg=2)
    assert scan_volume_layout(b, [t.folder for t in ScanInfo(b).tiles])["layout"] == "oct"
    # entry names use Windows separators like real Thorlabs files
    import zipfile
    names = zipfile.ZipFile(b / "Data01" / "VolumeGanymedeOCTFile.oct").namelist()
    assert "data\\Spectral0.data" in names
    ra, rb = _recon(a), _recon(b)
    assert type(rb.reader("Data01")).__name__ == "TileReader" and rb.layout["layout"] == "oct"
    rc = _recon(c, raw_input="extract")
    assert rc.layout["layout"] == "unzipped" and (c / "Data01" / "data" / "Spectral3.data").exists()
    pa, pb, pc = (r.process_planes([0, 1]) for r in (ra, rb, rc))
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(pa, pc)
    # extract_volume with deletion leaves a plain unzipped tile
    d = _make_tiled(work, "lay_oct_del", layout="oct")
    extract_volume(d, [t.folder for t in ScanInfo(d).tiles], delete_archives=True)
    assert tile_layout(d / "Data01") == "unzipped"
    assert not (d / "Data01" / "VolumeGanymedeOCTFile.oct").exists()


def test_full_run_writes_tiff(work):
    vol = _make_tiled(work, "run_oct", layout="oct", ascan_avg=2)
    r = _recon(vol, output_root=str(work / "out"), output_name="run", legacy_double_quantization=True)
    s = r.run()
    assert Path(s["tiff"]).exists() and s["output_shape_yzx"][0] == 2


# =========================================================================== legacy parity
@pytest.mark.parametrize("name", list(PIPELINE_CASES))
def test_pipeline_matches_legacy(name, ref, work):
    vol = _make_tiled(work, name)
    assert _dir_checksum(vol) == str(ref[f"{name}__checksum"]), "synthetic generator output changed"
    p = _recon(vol).process_planes([REF_PLANE_Y - 1])[0].astype(float)
    m = ref[f"{name}__planeDb"].astype(float)
    tie = ref[f"{name}__tie"]          # legacy total weight == exp(-4.5) (threshold ties)
    assert p.shape == m.shape
    fin = np.isfinite(p) & np.isfinite(m)
    assert np.abs(p - m)[fin].max() < 1e-4                    # float32 output rounding ~2e-6 dB
    # NaN masks must agree exactly off the threshold; pixels whose legacy total weight equals
    # exp(-4.5) to 1e-12 are decided by the last bit in MATLAB itself. With
    # core.stitching.min_weight_threshold at most ~1% of those may differ (1 of 146 here,
    # at the z edge of an uncropped volume; was ~95 before the fix).
    mism = np.isfinite(p) != np.isfinite(m)
    assert not np.any(mism & ~tie)
    assert mism.sum() <= max(1, 0.01 * tie.sum()), f"{int(mism.sum())} tie mismatches of {int(tie.sum())}"


@pytest.mark.parametrize("name", [n for n in LOADER_CASES])
def test_reader_matches_legacy_loader(name, ref, work):
    kw, kind, system, yframe = LOADER_CASES[name]
    tile = _make_tile(work, name)
    assert _dir_checksum(tile) == str(ref[f"{name}__checksum"])
    r = TileReader(tile)
    h = r.header
    nb, na = max(1, h.bscan_avg), max(1, h.ascan_avg)
    files = [(yframe - 1) * nb + b for b in range(nb)]
    buf = np.empty((len(files), h.interf_size, h.n_lambda), np.dtype(h.raw_dtype))
    r.read_bscans(files, buf)
    it = legacy_interferogram(buf, h.apod_size, h.spectra_avg, h.apod_mode, nb)    # (B, X*A, N)
    it = it.reshape(nb, h.size_x, na, h.n_lambda).transpose(3, 1, 2, 0)            # (N, X, A, B)
    m = ref[f"{name}__interf"]
    np.testing.assert_allclose(np.squeeze(it), m, rtol=0, atol=1e-9)
    # wavelengths
    sysname = system or what_oct_system_is_it(tile)[0]
    np.testing.assert_allclose(r.lambda_nm(sysname, DATA_DIR), ref[f"{name}__lambda"].ravel(), rtol=1e-12)
    # raw apodization lines (Thorlabs / SRR return them unaveraged)
    if h.apod_mode == "lines" and h.manufacturer != "Wasatch":
        ap = buf[:, :h.apod_size, :].astype(float).transpose(2, 1, 0)
        np.testing.assert_array_equal(np.squeeze(ap), ref[f"{name}__apod"])


@pytest.mark.parametrize("name", list(DETECT_CASES))
def test_system_detection_matches_legacy(name, ref, work):
    src, expected = DETECT_CASES[name]
    tile = _make_tile(work, src)
    legacy_ok = bool(ref[f"{name}__ok"])
    assert legacy_ok == (expected is not None)
    if expected is None:
        with pytest.raises(ValueError):
            what_oct_system_is_it(tile)
    else:
        assert what_oct_system_is_it(tile) == (str(ref[f"{name}__octSystem"]), str(ref[f"{name}__manufacturer"]))
        assert what_oct_system_is_it(tile) == expected


def test_simulated_scan_matches_legacy(ref, work):
    """'Simulated Ganymede' tiles (data.mat) made by the legacy simulator. The simulation is
    noiseless: its background (~140 dB below the peak) is floating-point round-off, and legacy
    computes partly in single precision, so compare pixels within 60 dB of the peak."""
    if not SIM.exists():
        pytest.skip("simulated scan fixture missing")
    vol = work / "simulated"
    shutil.copytree(SIM, vol)
    r = _recon(vol)
    assert r.hdr.manufacturer == "Simulated" and r.si.oct_system == "Simulated Ganymede"
    p = r.process_planes([REF_PLANE_Y - 1])[0].astype(float)
    m = ref["simulated__planeDb"].astype(float)
    assert p.shape == m.shape
    sel = m > np.nanmax(m) - 60
    assert np.abs(p - m)[sel].max() < 1e-3
    assert not np.any((np.isfinite(p) != np.isfinite(m)) & ~ref["simulated__tie"])


# =========================================================================== extensions
def test_autodetect_when_scaninfo_lacks_octsystem(work):
    vol = _make_tiled(work, "noinfo", write_octsystem=False)
    si = ScanInfo(vol)
    assert si.oct_system == "Ganymede" and si.oct_system_source.startswith("auto-detected")
    ref_vol = work / "withinfo"                                   # same data, explicit name
    shutil.copytree(vol, ref_vol)
    j = json.loads((ref_vol / "ScanInfo.json").read_text())
    j["octSystem"] = "Ganymede"
    (ref_vol / "ScanInfo.json").write_text(json.dumps(j))
    np.testing.assert_array_equal(_recon(vol).process_planes([0]), _recon(ref_vol).process_planes([0]))


def test_oct_archive_detection(work):
    vol = _make_tiled(work, "det_oct", layout="oct", system="telesto", write_octsystem=False)
    assert what_oct_system_is_it(vol / "Data01") == ("Telesto", "Thorlabs")
    assert ScanInfo(vol).oct_system == "Telesto"


@pytest.mark.parametrize("fmt,extra", [("srr", dict(bscan_avg=2)), ("wasatch", dict(bscan_avg=2)),
                                       ("srr", dict(system="telesto"))])
def test_tiled_srr_and_wasatch_volumes(fmt, extra, work):
    """Legacy yOCTProcessTiledScan cannot process these (docs/06); the port can."""
    vol = _make_tiled(work, f"tiled_{fmt}_{extra.get('system', '')}{extra.get('bscan_avg', 1)}", fmt=fmt, **extra)
    r = _recon(vol)
    assert r.hdr.manufacturer == {"srr": "Thorlabs_SRR", "wasatch": "Wasatch"}[fmt]
    p = r.process_planes([0, 1])
    fin = p[np.isfinite(p)]
    assert fin.size > 0.9 * p.size and fin.max() > 10      # reflectors reconstructed
    # detection without octSystem gives the same result
    vol2 = _make_tiled(work, f"tiled_{fmt}_auto{extra.get('system', '')}", fmt=fmt, write_octsystem=False, **extra)
    np.testing.assert_array_equal(_recon(vol2).process_planes([0, 1]), p)


def test_srr_upper_bits_are_masked(work):
    tile = _make_tile(work, "srr")
    r = TileReader(tile)
    buf = np.empty((1, r.header.interf_size, r.header.n_lambda), np.int16)
    r.read_bscans([0], buf)
    assert buf.min() >= 0 and buf.max() < 4096


def test_system_name_mismatch_is_rejected(work):
    vol = _make_tiled(work, "mismatch", fmt="srr")
    si = json.loads((vol / "ScanInfo.json").read_text())
    si["octSystem"] = "Ganymede"
    (vol / "ScanInfo.json").write_text(json.dumps(si))
    with pytest.raises(ValueError, match="does not match"):
        _recon(vol)


def test_binning_columns_match_filter2():
    """filter2(ones(1,B)/B, 1:10) then (:, max(1,floor(B/2)):B:end), values from MATLAB R2024a."""
    from octrecon.core.spectral import bin_ascans
    x = np.arange(1, 11, dtype=float)[None, :, None]
    full = {2: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 5.0],
            3: [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 19 / 3],
            4: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 6.75, 4.75]}
    for B, f in full.items():
        want = np.asarray(f)[max(1, B // 2) - 1::B]
        np.testing.assert_allclose(bin_ascans(x, B)[0, :, 0], want, rtol=1e-15)


# =========================================================================== reference builder
def build_reference(matlab="matlab"):
    work = Path(tempfile.mkdtemp(prefix="octrecon_ref_"))
    out = {}
    try:
        vols = {n: _make_tiled(work, n) for n in PIPELINE_CASES}
        tiles = {n: _make_tile(work, n) for n in LOADER_CASES}
        for n, v in vols.items():
            out[f"{n}__checksum"] = np.asarray(_dir_checksum(v))
        for n, t in tiles.items():
            out[f"{n}__checksum"] = np.asarray(_dir_checksum(t))
        # --- tiled planes
        # legacy simulator output (kept in tests/data, it cannot be regenerated without MATLAB)
        sim = work / "simulated"
        subprocess.run([matlab, "-batch", f"addpath('{ROOT / 'validation' / 'matlab'}'); "
                        f"make_legacy_simulated_scan('{sim}', 8, 2)"], check=True)
        if SIM.exists():
            shutil.rmtree(SIM)
        shutil.copytree(sim, SIM)
        vols["simulated"] = sim
        cmds = [f"dump_synthetic_plane('{v}', {REF_PLANE_Y}, '{work}/{n}_plane.mat');" for n, v in vols.items()]
        # --- loader cases
        cases = [(n, str(tiles[n]), k, s, y) for n, (_, k, s, y) in LOADER_CASES.items()]
        cases += [(n, str(tiles[src]), "detect", "", 1) for n, (src, _) in DETECT_CASES.items()]
        cs = ",".join("struct('name','%s','folder','%s','kind','%s','system','%s','yframe',%d)" % c for c in cases)
        cmds.append(f"dump_legacy_loader([{cs}], '{work}/loader.mat');")
        script = f"addpath('{ROOT / 'validation' / 'matlab'}'); " + " ".join(cmds)
        subprocess.run([matlab, "-batch", script], check=True)
        import scipy.io as sio
        for n in vols:
            M = sio.loadmat(work / f"{n}_plane.mat", squeeze_me=True)
            out[f"{n}__planeDb"] = M["planeDb"].astype(np.float32)
            out[f"{n}__tie"] = np.abs(M["totalWeightsRaw"] - MIN_WEIGHT) < 1e-12
        L = sio.loadmat(work / "loader.mat", squeeze_me=True, struct_as_record=False)
        for n, *_ in cases:
            r = L[n]
            out[f"{n}__ok"] = np.asarray(bool(r.ok))
            print(n, "ok" if r.ok else f"legacy error: {r.err}",
                  getattr(r, "regularPath", "") and f"| regular path: {r.regularPath}")
            for f in ("interf", "apod", "lambda", "octSystem", "manufacturer"):
                if hasattr(r, f):
                    out[f"{n}__{f}"] = np.asarray(getattr(r, f))
        REF.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(REF, **out)
        print(f"wrote {REF} ({REF.stat().st_size / 1e6:.2f} MB)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    if "--build-reference" in sys.argv:
        build_reference(os.environ.get("MATLAB", "matlab"))
    else:
        sys.exit(pytest.main([__file__, "-q"]))
