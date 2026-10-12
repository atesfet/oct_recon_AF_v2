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
    chk = summ["gpu_check"]
    if chk.get("checked"):                                   # nvidia-smi reports this process
        assert chk["new_contexts_on_other_gpus"] == [] and 0 in chk["process_gpus_after"]
    with pytest.raises(ValueError):
        Reconstructor(ReconConfig(volume_folder=str(vol), device="gpu", gpu_id=99), log=lambda m: None)


def test_auto_keeps_the_gpu_selected_in_the_thread():
    """A helper asking for plain 'gpu'/'auto' in the middle of a run must not move it to another GPU."""
    from octrecon.backend import resolve_gpu_index
    gpus = [{"id": 0, "free_GB": 1.0, "total_GB": 16.0, "name": "a"},
            {"id": 1, "free_GB": 20.0, "total_GB": 24.0, "name": "b"}]
    assert resolve_gpu_index("auto", 2, gpus, current=None) == 1        # nothing selected yet: most free memory
    assert resolve_gpu_index("auto", 2, gpus, current=0) == 0           # already on GPU 0: stay there
    assert resolve_gpu_index(None, 2, gpus, current=0) == 0
    assert resolve_gpu_index(1, 2, gpus, current=0) == 1                # an explicit index always wins
    with pytest.raises(ValueError, match="does not exist"):
        resolve_gpu_index(2, 2, gpus, current=0)


@pytest.mark.skipif(not HAVE_GPU, reason="no NVIDIA GPU")
def test_listing_gpus_creates_no_cuda_context():
    """gpu_available()/list_gpus() must not reserve memory on GPU 0 (no CUDA context)."""
    import subprocess
    code = ("import os, subprocess, sys; sys.path.insert(0, %r)\n"
            "from octrecon.backend import gpu_available, list_gpus\n"
            "assert gpu_available() and list_gpus()\n"
            "out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],"
            " capture_output=True, text=True).stdout\n"
            "print('CONTEXT' if str(os.getpid()) in out.split() else 'NO_CONTEXT')\n") % str(ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "NO_CONTEXT" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not HAVE_GPU, reason="no NVIDIA GPU")
def test_estimators_inside_a_run_get_the_run_gpu(tmp_path, monkeypatch):
    """Focus detection during set-up is called with 'gpu:N' (it used to get plain 'gpu' and re-ran the
    auto choice, which moved the rest of the run to another GPU)."""
    import test_formats as TF
    import octrecon.estimation.focus as F
    from octrecon.backend import current_gpu
    from octrecon.pipeline import ReconConfig, Reconstructor
    seen = {}
    real = F.detect_focus

    def spy(*a, **kw):
        seen["device"] = kw.get("device")
        return real(*a, **kw)
    monkeypatch.setattr(F, "detect_focus", spy)
    vol = TF._make_tiled(tmp_path, "base")
    R = Reconstructor(ReconConfig(volume_folder=str(vol), device="gpu", gpu_id=0, focus_positions="estimate",
                                  crop_z_range_mm="none", fep_removal=False), log=lambda m: None)
    assert seen.get("device") == "gpu:0"
    assert current_gpu()["id"] == 0 and R.gpu_info["id"] == 0
