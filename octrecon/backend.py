"""Array backend selection.

  cpu        NumPy + Numba + multithreaded pocketfft
  gpu        NVIDIA GPU via CuPy (fused CUDA kernel, cuFFT)                 -> name "gpu"
  gpu:N      the N-th NVIDIA GPU (same numbering as nvidia-smi: PCI bus order); with plain
             "gpu"/"auto" and several GPUs, `gpu_id` selects one ("auto" = most free memory)
             or, on Apple silicon, the GPU via PyTorch Metal/MPS            -> name "mps"
  mps        force the Apple GPU (PyTorch MPS)
  torch-cpu  the PyTorch backend on the CPU (used to validate the MPS code path anywhere)
  auto       CUDA if usable, else MPS if usable (and its self-test passes), else CPU

Every numerical routine takes an ``xp`` so the same pipeline runs on all backends.
"""
from __future__ import annotations

import os

import numpy as np

# let PyTorch run ops that MPS does not implement on the CPU instead of failing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# number CUDA GPUs like nvidia-smi does (PCI bus order) so that "GPU N" means the same everywhere
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_MPS_STATUS: tuple[bool, str] | None = None


def gpu_available() -> bool:
    """NVIDIA GPU usable through CuPy. Does not create a CUDA context (that would reserve memory on
    GPU 0 even if another GPU is selected); a broken driver already fails the device count."""
    return bool(list_gpus())


def mps_status() -> tuple[bool, str]:
    """(usable, message) for the Apple-silicon GPU via PyTorch MPS (cached)."""
    global _MPS_STATUS
    if _MPS_STATUS is None:
        try:
            import torch
            if not torch.backends.mps.is_available():
                built = torch.backends.mps.is_built()
                _MPS_STATUS = (False, "PyTorch MPS not available on this machine" if built
                               else "PyTorch was built without MPS support")
            else:
                from .core.torch_backend import TorchXP, selftest
                ok, msg = selftest(TorchXP("mps"))
                _MPS_STATUS = (ok, "Apple GPU (Metal/MPS) self-test passed" if ok
                               else f"MPS self-test failed ({msg}); using CPU")
        except ImportError:
            _MPS_STATUS = (False, "PyTorch is not installed")
        except Exception as e:  # noqa: BLE001
            _MPS_STATUS = (False, f"MPS unavailable: {type(e).__name__}: {e}")
    return _MPS_STATUS


def mps_available() -> bool:
    return mps_status()[0]


# --------------------------------------------------------------------------- multiple NVIDIA GPUs
def _smi_stats() -> dict:
    """{pci bus id (lower case, 8-digit domain): {'free_GB', 'used_GB', 'util_pct'}} from nvidia-smi
    (no CUDA context is created); {} if nvidia-smi is unavailable."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.free,memory.used,utilization.gpu,index",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    stats = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        free, used, util = num(parts[1]), num(parts[2]), num(parts[3])
        smi_idx = num(parts[4]) if len(parts) > 4 else None
        stats[_norm_bus(parts[0])] = {"free_GB": round(free / 1024, 1) if free is not None else None,
                                      "used_GB": round(used / 1024, 1) if used is not None else None,
                                      "util_pct": util, "smi_index": int(smi_idx) if smi_idx is not None else None}
    return stats


def _norm_bus(bus: str) -> str:
    """'00000000:01:00.0' / '0000:01:00.0' -> '0000:01:00.0' (lower case)."""
    bus = bus.strip().lower()
    dom, rest = bus.split(":", 1)
    return f"{int(dom, 16):04x}:{rest}"


def list_gpus() -> list[dict]:
    """The NVIDIA GPUs CuPy can use, in CUDA numbering (= nvidia-smi order, see CUDA_DEVICE_ORDER;
    CUDA_VISIBLE_DEVICES is respected). Each: id, name, total_GB, free_GB, util_pct, compute, pci.
    Does not create CUDA contexts. [] if CUDA is unusable."""
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return []
    smi = _smi_stats()
    gpus = []
    for i in range(n):
        try:
            p = cp.cuda.runtime.getDeviceProperties(i)
        except Exception:
            continue
        name = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
        pci = f"{p['pciDomainID']:04x}:{p['pciBusID']:02x}:{p['pciDeviceID']:02x}.0"
        st = smi.get(pci, {})
        gpus.append({"id": i, "name": name, "total_GB": round(p["totalGlobalMem"] / 2**30, 1),
                     "free_GB": st.get("free_GB"), "used_GB": st.get("used_GB"), "util_pct": st.get("util_pct"),
                     "smi_index": st.get("smi_index"),
                     "compute": f"{p['major']}.{p['minor']}", "pci": pci})
    return gpus


def choose_gpu(gpus: list[dict]) -> int:
    """'auto' choice: the GPU with the most free memory (total memory if unknown); lowest id on ties."""
    if not gpus:
        raise RuntimeError("no NVIDIA GPU available")
    return min(gpus, key=lambda g: (-(g["free_GB"] if g.get("free_GB") is not None else g["total_GB"]), g["id"]))["id"]


def parse_device(device: str, gpu_id=None):
    """'gpu:1' / 'cuda:1' -> ('gpu', 1); other device strings pass through with gpu_id."""
    d = str(device).strip().lower()
    for pre in ("gpu:", "cuda:"):
        if d.startswith(pre):
            return "gpu", int(d[len(pre):])
    return d, gpu_id


_CURRENT_GPU = {}


def resolve_gpu_index(gpu_id, n: int, gpus: list[dict], current=None) -> int:
    """Which CUDA device to use. An explicit index wins; None/'auto' keeps the GPU already selected in
    this thread (so a helper that asks for plain 'gpu' in the middle of a run cannot move the run to
    another GPU), else the GPU with the most free memory."""
    if gpu_id in (None, "", "auto"):
        if current is not None and 0 <= current < n:
            return current
        return choose_gpu(gpus) if (n > 1 and gpus) else 0
    idx = int(gpu_id)
    if not 0 <= idx < n:
        avail = ", ".join(f"{g['id']}: {g['name']}" for g in gpus) or "none"
        raise ValueError(f"GPU {idx} does not exist; available GPUs: {avail}")
    return idx


def select_gpu(gpu_id=None) -> int:
    """Make GPU `gpu_id` (int, or None/'auto') CuPy's current device for the calling thread and check
    that it works. Returns the CUDA index. Raises ValueError for an unknown index."""
    import threading
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    tid = threading.get_ident()
    idx = resolve_gpu_index(gpu_id, n, list_gpus() if n > 1 else [], _CURRENT_GPU.get(tid))
    use_gpu(idx)
    cp.zeros(1).sum()                     # context on that device only; fails on a broken driver
    return idx


def use_gpu(idx: int):
    """(Re-)make CUDA device `idx` current for the calling thread (CuPy's current device is per thread)."""
    import threading
    import cupy as cp
    cp.cuda.Device(int(idx)).use()
    _CURRENT_GPU[threading.get_ident()] = int(idx)


def gpus_used_by_this_process() -> list[int] | None:
    """CUDA indices of the GPUs on which this process holds a CUDA context, from nvidia-smi
    (None if it cannot be determined, e.g. no nvidia-smi or a container that hides PIDs)."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    by_pci = {g["pci"]: g["id"] for g in list_gpus()}
    used = set()
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(os.getpid()):
            idx = by_pci.get(_norm_bus(parts[1]))
            if idx is not None:
                used.add(idx)
    return sorted(used)


def current_gpu() -> dict | None:
    """Info of the GPU selected in this thread (None if no CUDA GPU is in use)."""
    import threading
    idx = _CURRENT_GPU.get(threading.get_ident())
    if idx is None:
        return None
    for g in list_gpus():
        if g["id"] == idx:
            return g
    return {"id": idx}


def get_xp(device: str = "auto", gpu_id=None):
    """Return (xp, device_name) with device_name in {'gpu', 'mps', 'torch-cpu', 'cpu'}.
    device may be 'gpu:N'; gpu_id (int or 'auto') selects the NVIDIA GPU for 'gpu'/'auto'."""
    device, gpu_id = parse_device(device, gpu_id)
    if device not in ("auto", "gpu", "cpu", "mps", "torch-cpu"):
        raise ValueError(f"device must be auto|gpu|gpu:N|cpu|mps, got {device!r}")
    if device == "cpu":
        return np, "cpu"
    if device == "torch-cpu":
        from .core.torch_backend import TorchXP
        return TorchXP("cpu"), "torch-cpu"
    if device in ("auto", "gpu") and list_gpus():
        import cupy as cp
        try:
            select_gpu(gpu_id)
            return cp, "gpu"
        except ValueError:
            raise
        except Exception:
            if gpu_id not in (None, "", "auto"):
                raise RuntimeError(f"GPU {gpu_id} could not be initialised (CUDA error); after a laptop suspend "
                                   "run `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` or reboot")
    if device in ("auto", "gpu", "mps") and mps_available():
        from .core.torch_backend import TorchXP
        return TorchXP("mps"), "mps"
    if device == "auto":
        return np, "cpu"
    if device == "mps":
        raise RuntimeError(f"Apple GPU requested but not usable: {mps_status()[1]}")
    raise RuntimeError(
        "GPU requested but no usable GPU was found. NVIDIA: CUDA could not be initialised "
        "(after a laptop suspend this usually needs a reboot or "
        "`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`). Apple silicon: " + mps_status()[1])


def is_torch(xp) -> bool:
    return hasattr(xp, "torch") and hasattr(xp, "device")


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "detach"):          # torch tensor
        return a.detach().cpu().numpy()
    return a.get()                    # cupy


def synchronize(xp):
    if xp is np:
        return
    if is_torch(xp):
        xp.synchronize()
    else:
        xp.cuda.Stream.null.synchronize()
