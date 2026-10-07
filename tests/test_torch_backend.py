"""The PyTorch backend (Apple-silicon GPU / MPS) against the NumPy backend and legacy MATLAB.

Runs on the PyTorch *CPU* device ("torch-cpu"), which exercises exactly the code used on the
Apple GPU (only the device differs). On an Apple-silicon Mac, `OCT_TORCH_DEVICE=mps pytest`
runs the same tests on the GPU. Skipped when PyTorch is not installed.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_formats as TF  # noqa: E402
from octrecon.backend import get_xp  # noqa: E402
from octrecon.core.torch_backend import TorchXP, selftest  # noqa: E402
from octrecon.pipeline import ReconConfig, Reconstructor  # noqa: E402

TORCH_DEVICE = os.environ.get("OCT_TORCH_DEVICE", "torch-cpu")


@pytest.fixture(scope="module")
def work():
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="octrecon_torch_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _recon(vol: Path, device: str):
    info = json.loads((vol / "synthetic_info.json").read_text())
    cfg = ReconConfig(fep_removal=False, volume_folder=str(vol), device=device, precision="float32",
                      dispersion_quadratic_term=info.get("dispersion", TF.DISPERSION),
                      focus_positions=float(info["focus_pix"]), focus_sigma=10, crop_z_range_mm="none",
                      output_pixel_size_um=info.get("pixel_um", 2), interp_method="sinc5",
                      batch_frames=4, io_threads=2)
    return Reconstructor(cfg, log=lambda *a: None)


def test_selftest_and_device():
    xp, name = get_xp(TORCH_DEVICE)
    assert name in ("torch-cpu", "mps")
    ok, msg = selftest(xp if name == "mps" else TorchXP("cpu"))
    assert ok, msg


@pytest.mark.parametrize("name", list(TF.PIPELINE_CASES))
def test_torch_matches_numpy_and_legacy(name, work):
    ref = np.load(TF.REF)
    vol = TF._make_tiled(work, name)
    y = TF.REF_PLANE_Y - 1
    p_t = _recon(vol, TORCH_DEVICE).process_planes([y])[0].astype(float)
    p_n = _recon(vol, "cpu").process_planes([y])[0].astype(float)
    m = ref[f"{name}__planeDb"].astype(float)
    tie = ref[f"{name}__tie"]
    # Both backends compute in float32. Deep in the noise floor (~75 dB below the peak)
    # float32 rounding reaches a few 1e-3 dB for either backend (float64 is exact), so the
    # criterion is: the torch path is no less accurate than the validated numpy float32 path.
    assert np.array_equal(np.isfinite(p_t), np.isfinite(p_n))
    both = np.isfinite(p_t) & np.isfinite(m) & np.isfinite(p_n)
    err_t = np.abs(p_t - m)[both].max()
    err_n = np.abs(p_n - m)[both].max()
    assert err_t <= 1.5 * err_n + 1e-4, (err_t, err_n)
    assert err_t < 1e-2
    mism = np.isfinite(p_t) != np.isfinite(m)
    assert not np.any(mism & ~tie)
