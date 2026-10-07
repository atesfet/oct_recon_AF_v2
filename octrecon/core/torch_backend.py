"""PyTorch backend: Apple-silicon GPU (Metal / MPS) acceleration.

Used when the device resolves to "mps" (macOS). The same code also runs on PyTorch's CPU
device ("torch-cpu"), which is how it is validated on machines without an Apple GPU.
NVIDIA GPUs keep using the CuPy backend (fused CUDA kernel), which is faster there.

Numerics mirror SpectralProcessor / TileStitcher exactly, with two portability choices:
  * complex arithmetic is limited to the FFT itself (window applied as real/imag parts,
    magnitude via view_as_real), because MPS complex-op coverage varies across versions;
  * MPS has no float64, so the optical-path validity test uses integer row bounds
    precomputed in float64 on the host (equivalent to 0 <= r + P/dz <= nZ-1).
Ops that MPS does not implement fall back to the CPU automatically
(PYTORCH_ENABLE_MPS_FALLBACK=1, set in octrecon.backend before torch is imported).
"""
from __future__ import annotations

import numpy as np

from .geometry import SpectralGeometry
from .spectral import prepare_raw

_NP2T = {np.dtype(np.float32): "float32", np.dtype(np.float64): "float64",
         np.dtype(np.int16): "int16", np.dtype(np.int32): "int32", np.dtype(np.int64): "int64",
         np.dtype(np.bool_): "bool", np.dtype(np.uint8): "uint8"}


class TorchXP:
    """The small numpy-like namespace the pipeline's accumulation code needs."""
    nan = float("nan")

    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.name = "mps" if self.device.type == "mps" else f"torch-{self.device.type}"

    def tdtype(self, dtype):
        return getattr(self.torch, _NP2T[np.dtype(dtype)])

    def zeros(self, shape, dtype=np.float32):
        return self.torch.zeros(shape, dtype=self.tdtype(dtype), device=self.device)

    def asarray(self, a, dtype=None):
        torch = self.torch
        if isinstance(a, torch.Tensor):
            t = a.to(self.device)
        else:
            arr = np.asarray(a)
            if arr.dtype == np.uint16:            # limited uint16 op support in torch
                arr = arr.astype(np.int32)
            elif arr.dtype.kind in "iu" and arr.dtype not in _NP2T:
                arr = arr.astype(np.int64)
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        return t if dtype is None else t.to(self.tdtype(dtype))

    def log10(self, x):
        return self.torch.log10(x)

    def synchronize(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()
        elif self.device.type == "cuda":
            self.torch.cuda.synchronize()


class TorchSpectralProcessor:
    """Same interface/results as SpectralProcessor.magnitude, on a torch device (float32)."""

    def __init__(self, sg: SpectralGeometry, apod_size: int, xpt: TorchXP, ascan_binning: int = 1,
                 apod_mode: str = "lines", apod_group: int = 1):
        torch = xpt.torch
        self.xpt = xpt
        self.dtype = np.dtype(np.float32)
        self.raw_apod_size = apod_size
        self.apod_mode = apod_mode
        self.apod_group = max(1, int(apod_group))
        self.ascan_binning = max(1, int(ascan_binning))
        self.apod_size = 1 if apod_mode == "frame_mean" else apod_size
        self.n = len(sg.lambda_nm)
        self.nz = self.n // 2
        dev = xpt.device
        self.idx = torch.from_numpy(np.ascontiguousarray(sg.sinc_idx.T, dtype=np.int64)).to(dev)     # (T, N)
        self.w = torch.from_numpy(np.ascontiguousarray(sg.sinc_w.T, dtype=np.float32)).to(dev)       # (T, N)
        self.win_re = torch.from_numpy(np.ascontiguousarray(sg.window.real, dtype=np.float32)).to(dev)
        self.win_im = torch.from_numpy(np.ascontiguousarray(sg.window.imag, dtype=np.float32)).to(dev)

    def _prepared(self, raw):
        """Binning / frame-mean background use the host implementation (rare formats)."""
        torch = self.xpt.torch
        if self.ascan_binning > 1 or self.apod_mode == "frame_mean":
            host = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
            host = prepare_raw(host, self.raw_apod_size, self.ascan_binning, self.apod_mode,
                               self.apod_group, np, np.float32)
            return self.xpt.asarray(np.asarray(host, dtype=np.float32))
        return self.xpt.asarray(raw)

    def magnitude(self, raw):
        """raw (B, interfSize, N) -> |scan| (B, nX, N/2) float32 tensor on the device.

        The spectral axis is moved to the front so every sinc tap is a gather of whole
        contiguous rows (5x faster than gathering along the innermost axis), and the FFT
        runs along that axis. Results are identical to the row-major formulation."""
        torch = self.xpt.torch
        x = self._prepared(raw).to(torch.float32)
        a = self.apod_size
        apod = x[:, :a, :].mean(dim=1, keepdim=True)
        interf = x[:, a:, :] - apod
        B, nX, N = interf.shape
        mv = interf.mean(dim=-1).reshape(1, -1)                     # (1, M) per-A-line mean
        dT = interf.reshape(-1, N).T.contiguous() - mv             # (N, M), spectral axis first
        eq = torch.zeros_like(dT)
        for t in range(self.idx.shape[0]):                         # banded sinc5 operator (<= 14 taps)
            eq += self.w[t][:, None] * dT.index_select(0, self.idx[t])
        eq += mv
        spec = torch.complex(eq * self.win_re[:, None], eq * self.win_im[:, None])
        ft = torch.fft.ifft(spec, dim=0)[: self.nz]                 # (nZ, M)
        if getattr(self, "_want_complex", False):
            return ft.T.reshape(B, nX, self.nz)
        ri = torch.view_as_real(ft)
        mag = torch.sqrt(ri[..., 0] * ri[..., 0] + ri[..., 1] * ri[..., 1])
        return mag.T.reshape(B, nX, self.nz)

    def scan(self, raw):
        """complex scan (B, nX, N/2) as a torch tensor (for FEP removal)."""
        self._want_complex = True
        try:
            return self.magnitude(raw)
        finally:
            self._want_complex = False


class TorchTileStitcher:
    """Wraps a host-built TileStitcher (xp=numpy) and applies it on a torch device."""

    def __init__(self, base, oc, xpt: TorchXP):
        torch = xpt.torch
        self.xpt = xpt
        self.empty = base.empty
        if self.empty:
            return
        self.r0, self.r1, self.c0, self.c1 = base.r0, base.r1, base.c0, base.c1
        dev = xpt.device

        def f32(a):
            return torch.from_numpy(np.ascontiguousarray(np.asarray(a), dtype=np.float32)).to(dev)
        self.WzT = f32(np.asarray(base.Wz).T)              # (R, nOutZ)
        self.WxT = f32(base.WxT)                           # (nTileX, C)
        self.fz = f32(base.fz)                             # (R,)
        self.n_rows = base.n_rows
        self.rr = torch.arange(base.r0, base.r1, device=dev, dtype=torch.int64)
        if oc is not None:
            # exact integer form of 0 <= r + pdz <= nZ-1 (evaluated in float64 on the host)
            lo = np.ceil(-oc.pdz).astype(np.int64)
            hi = np.floor(oc.n_rows - 1 - oc.pdz).astype(np.int64)
            self.shift = torch.from_numpy(np.ascontiguousarray(oc.shift, dtype=np.int64)).to(dev)
            self.lo = torch.from_numpy(lo).to(dev)
            self.hi = torch.from_numpy(hi).to(dev)
        else:
            self.shift = None

    def contribute(self, mag, frame_idx):
        torch = self.xpt.torch
        fi = frame_idx.to(torch.int64)
        if self.shift is not None:
            src = (self.rr + self.shift[fi][..., None]).clamp_(0, self.n_rows - 1)        # (B, nX, R)
            valid = (self.rr >= self.lo[fi][..., None]) & (self.rr <= self.hi[fi][..., None])
        else:
            src = self.rr.expand(len(fi), mag.shape[1], len(self.rr))
            valid = torch.ones(src.shape, dtype=torch.bool, device=mag.device)
        a = torch.take_along_dim(mag, src, dim=-1)
        f = valid.to(torch.float32) * self.fz
        a = a * f
        num = torch.matmul(torch.matmul(a, self.WzT).permute(0, 2, 1), self.WxT)
        den = torch.matmul(torch.matmul(f, self.WzT).permute(0, 2, 1), self.WxT)
        return num, den


def selftest(xpt: TorchXP) -> tuple[bool, str]:
    """Check the ops this backend relies on against NumPy on a small random problem."""
    try:
        torch = xpt.torch
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 7, 64)).astype(np.float32)
        idx = rng.integers(0, 64, size=64)
        t = xpt.asarray(x)
        if not np.allclose(torch.index_select(t, -1, xpt.asarray(idx)).cpu().numpy(), x[..., idx]):
            return False, "index_select mismatch"
        ri = torch.view_as_real(torch.fft.ifft(torch.complex(t, t * 0.5), dim=-1)).cpu().numpy()
        ref = np.fft.ifft(x + 0.5j * x, axis=-1)
        if not (np.allclose(ri[..., 0], ref.real, atol=1e-5) and np.allclose(ri[..., 1], ref.imag, atol=1e-5)):
            return False, "complex FFT mismatch"
        src = rng.integers(0, 64, size=(3, 7, 5))
        if not np.allclose(torch.take_along_dim(t, xpt.asarray(src), dim=-1).cpu().numpy(),
                           np.take_along_axis(x, src, axis=-1)):
            return False, "take_along_dim mismatch"
        if not np.allclose(torch.matmul(t, t.transpose(1, 2)).cpu().numpy(), x @ x.transpose(0, 2, 1),
                           rtol=1e-4, atol=1e-3):
            return False, "matmul mismatch"
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
