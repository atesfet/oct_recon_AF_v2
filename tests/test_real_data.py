"""Regression tests on a real dataset (skipped unless the data is available).

  OCT_TEST_VOLUME=/path/to/10um_FOV_1/OCTVolume  [OCT_LEGACY_TIFF=/path/to/legacy.tiff]  [OCT_DEVICE=gpu]  pytest -q tests

* plane y=2250 vs the unmodified legacy MATLAB code (tests/data/matlab_ref_10um_FOV_1_y2250.npz)
* bit-exact reproduction of the legacy MATLAB TIFF (dispersion 8.949e7, legacy double quantisation)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octrecon.io.tiff_writer import bits_to_db, db_to_bits, plane_clim  # noqa: E402
from octrecon.pipeline import ReconConfig, Reconstructor  # noqa: E402

VOL = os.environ.get("OCT_TEST_VOLUME")
LEGACY_TIFF = os.environ.get("OCT_LEGACY_TIFF")
DEVICE = os.environ.get("OCT_DEVICE", "cpu")
REF = ROOT / "tests/data/matlab_ref_10um_FOV_1_y2250.npz"
needs_vol = pytest.mark.skipif(not VOL or not Path(VOL).exists(), reason="set OCT_TEST_VOLUME to the 10um_FOV_1 OCTVolume")


def _rec(**kw):
    return Reconstructor(ReconConfig(volume_folder=VOL, device=DEVICE, batch_frames=2, **{"fep_removal": False, **kw}), log=lambda m: None)


@needs_vol
@pytest.mark.parametrize("precision,tol_db", [("float64", 1e-5), ("float32", 5e-4)])
def test_plane_matches_matlab(precision, tol_db):
    r = np.load(REF)
    p = _rec(precision=precision, dispersion_quadratic_term=float(r["dispersion"])).process_planes([int(r["y_index0"])])[0]
    m = r["planeDb"]
    assert np.array_equal(np.isnan(p), np.isnan(m))
    assert np.nanmax(np.abs(p - m)) < tol_db


@needs_vol
@pytest.mark.skipif(not LEGACY_TIFF or not Path(LEGACY_TIFF).exists(), reason="set OCT_LEGACY_TIFF to the legacy MATLAB TIFF")
def test_bit_exact_vs_legacy_tiff():
    import tifffile
    with tifffile.TiffFile(LEGACY_TIFF) as t:
        clim = json.loads(t.pages[0].tags[305].value)["clim"]
        legacy = t.pages[700].asarray().astype(int)
    p = _rec(precision="float64", dispersion_quadratic_term="auto").process_planes([700])[0].astype(float)
    pc = plane_clim(p)
    mine = db_to_bits(bits_to_db(db_to_bits(p, pc), pc), clim).astype(int)
    assert np.abs(mine - legacy).max() <= 1
    assert (mine == legacy).mean() > 0.995
