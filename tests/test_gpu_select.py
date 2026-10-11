"""Choosing between several NVIDIA GPUs (device 'gpu:N' / ReconConfig.gpu_id)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from octrecon.backend import _norm_bus, choose_gpu, get_xp, list_gpus, parse_device  # noqa: E402

HAVE_GPU = bool(list_gpus())


def test_parse_device():
    assert parse_device("gpu:1") == ("gpu", 1)
    assert parse_device("CUDA:2") == ("gpu", 2)
    assert parse_device("gpu", 3) == ("gpu", 3)
    assert parse_device("auto", "auto") == ("auto", "auto")
    assert parse_device("cpu") == ("cpu", None)


def test_choose_gpu_most_free_memory():
    gpus = [{"id": 0, "free_GB": 2.0, "total_GB": 16.0}, {"id": 1, "free_GB": 20.0, "total_GB": 24.0},
            {"id": 2, "free_GB": 20.0, "total_GB": 24.0}]
    assert choose_gpu(gpus) == 1                                     # most free, lowest id on a tie
    assert choose_gpu([{"id": 0, "free_GB": None, "total_GB": 8.0},
                       {"id": 1, "free_GB": None, "total_GB": 24.0}]) == 1   # total if free unknown
    with pytest.raises(RuntimeError):
        choose_gpu([])


def test_pci_bus_matching():
    # nvidia-smi prints an 8-digit domain, CUDA a 4-digit one
    assert _norm_bus("00000000:01:00.0") == _norm_bus("0000:01:00.0") == "0000:01:00.0"
    assert _norm_bus("00000001:3B:00.0") == "0001:3b:00.0"


def test_list_gpus_is_safe():
    g = list_gpus()                                                  # never raises; [] without CUDA
    assert isinstance(g, list)
    for x in g:
        assert {"id", "name", "total_GB", "pci"} <= set(x)


@pytest.mark.skipif(not HAVE_GPU, reason="no NVIDIA GPU")
def test_select_existing_and_invalid_gpu():
    from octrecon.backend import current_gpu
    xp, name = get_xp("gpu:0")
    assert name == "gpu" and current_gpu()["id"] == 0
    xp, name = get_xp("gpu", gpu_id="auto")
    assert name == "gpu" and current_gpu()["id"] == choose_gpu(list_gpus())
    with pytest.raises(ValueError, match="does not exist"):
        get_xp("gpu", gpu_id=len(list_gpus()) + 5)


@pytest.mark.skipif(not HAVE_GPU, reason="no NVIDIA GPU")
def test_reconstruction_records_the_gpu(tmp_path):
    import test_formats as TF
    from octrecon.pipeline import ReconConfig, Reconstructor
    vol = TF._make_tiled(tmp_path, "base")
    s = Reconstructor(ReconConfig(volume_folder=str(vol), output_root=str(tmp_path / "out"), output_name="g",
                                  device="gpu", gpu_id=0, focus_positions=40, crop_z_range_mm="none",
                                  fep_removal=False, flatfield_correction=False, xy_projection=False),
                      log=lambda m: None).run()
    assert s["device"] == "gpu" and s["gpu"]["id"] == 0
    summ = json.loads((Path(s["output_dir"]) / "g_run_summary.json").read_text())
    assert summ["gpu"]["name"] == list_gpus()[0]["name"]
    with pytest.raises(ValueError):
        Reconstructor(ReconConfig(volume_folder=str(vol), device="gpu", gpu_id=99), log=lambda m: None)
