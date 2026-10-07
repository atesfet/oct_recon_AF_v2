"""Raw spectral B-scans -> depth-resolved magnitude (batched, CPU or GPU).

Per A-line this reproduces, in order:
  yOCTLoadInterfFromFile.m      apodization subtraction (mean of apod A-lines)
  yOCTEquispaceInterf.m         sincN k-linearisation (mean removed, then re-added)
  yOCTInterfToScanCpx.m         * Hann/rms * exp(-i*beta*(k-k0)^2), ifft, keep N/2
  yOCTProcessTiledScan.m        abs()

The legacy code loops over 2048 output samples in MATLAB for the sinc step
(~77% of total runtime). Here the sinc kernel is a precomputed banded operator
(<=14 taps/row) applied as a handful of vectorised gathers.
"""
from __future__ import annotations

import numpy as np

from .geometry import SpectralGeometry


def binned_columns(n_lines: int, binning: int) -> np.ndarray:
    """0-based A-line indices kept after AScanBinning decimation
    (yOCTLoadInterfFromFile_ThorlabsData.m l.124: interfAvg(:, max(1,floor(B/2)):B:end))."""
    s0 = max(1, binning // 2) - 1
    return np.arange(s0, n_lines, binning)


def bin_ascans(x, binning: int, xp=np):
    """AScanBinning of yOCTLoadInterfFromFile_ThorlabsData.m l.120-127:
        interfAvg = filter2(ones(1,B)/B, interf);                 % 'same', zero padded
        interfAvg = interfAvg(:, max(1,floor(B/2)):B:end);
    applied along the A-line axis. x: (..., nLines, N) float -> (..., nKept, N).
    For kernel length B, filter2 'same' output i averages lines
    [i-ceil(B/2)+1, i+floor(B/2)] (lines outside the file count as 0)."""
    n = x.shape[-2]
    cols = binned_columns(n, binning)
    left = (binning + 1) // 2 - 1
    right = binning // 2
    pad = [(0, 0)] * x.ndim
    pad[-2] = (left, right)
    xpad = xp.pad(x, pad)
    h = 1.0 / binning
    acc = None
    for t in range(binning):          # window of output col c = padded lines c .. c+B-1
        sl = xpad[..., cols[0] + t: cols[0] + t + binning * (len(cols) - 1) + 1: binning, :]
        acc = sl * h if acc is None else acc + sl * h
    return acc


def prepare_raw(raw, apod_size: int, ascan_binning: int = 1, apod_mode: str = "lines",
                apod_group: int = 1, xp=np, dtype=np.float64):
    """Raw files (B, lines, N) -> array whose first `a` A-lines are the apodization
    (a = apod_size, or 1 for apod_mode 'frame_mean') followed by the scan A-lines after
    AScanBinning. Returns `raw` itself when there is nothing to do."""
    if apod_mode == "frame_mean":
        x = raw.astype(dtype)
        B, nL, N = x.shape
        g = max(1, int(apod_group))
        if B % g:
            raise ValueError(f"batch of {B} files is not a whole number of {g}-file frames")
        # WasatchData.m l.126: squeeze(mean(mean(interf,3),2)) -> mean over x per file,
        # then yOCTLoadInterfFromFile.m l.211 mean(apodization,2) -> mean over repeats
        apod = x.reshape(B // g, g, nL, N).mean(axis=2).mean(axis=1)          # (B/g, N)
        apod = xp.repeat(apod, g, axis=0)[:, None, :]
        if ascan_binning > 1:
            x = bin_ascans(x, ascan_binning, xp)
        return xp.concatenate([apod, x], axis=1)
    if ascan_binning > 1:
        x = raw.astype(dtype)
        a = apod_size
        return xp.concatenate([x[:, :a, :], bin_ascans(x[:, a:, :], ascan_binning, xp)], axis=1)
    return raw


def legacy_interferogram(raw, apod_size: int, ascan_binning: int = 1, apod_mode: str = "lines",
                         apod_group: int = 1, xp=np, dtype=np.float64):
    """Apodization-corrected interferogram as yOCTLoadInterfFromFile returns it, in file
    layout (B, nLines, N): prepare_raw, then minus the mean of the apodization lines
    (yOCTLoadInterfFromFile.m l.201-216)."""
    x = prepare_raw(raw, apod_size, ascan_binning, apod_mode, apod_group, xp, dtype).astype(dtype, copy=False)
    a = 1 if apod_mode == "frame_mean" else apod_size
    return x[:, a:, :] - x[:, :a, :].mean(axis=1, keepdims=True)


class SpectralProcessor:
    """ascan_binning: Thorlabs AScanBinning (Header IntensityAveraging/Spectra), >1 averages
    consecutive raw A-lines before processing (legacy filter2 + decimation).
    apod_mode: 'lines' (apodization = first apod_size A-lines of every raw file) or
    'frame_mean' (no apodization lines; background = mean over all A-lines of each group of
    apod_group consecutive files, i.e. of one y frame incl. its B-scan repeats - Wasatch)."""

    def __init__(self, sg: SpectralGeometry, apod_size: int, xp=np, dtype=np.float32, use_numba=True,
                 use_gpu_kernel=False, ascan_binning: int = 1, apod_mode: str = "lines", apod_group: int = 1):
        self.xp = xp
        self.use_gpu_kernel = use_gpu_kernel and np.dtype(dtype) == np.float32
        self.use_numba = use_numba
        self.dtype = np.dtype(dtype)
        cdtype = np.complex64 if self.dtype == np.float32 else np.complex128
        if apod_mode not in ("lines", "frame_mean"):
            raise ValueError(apod_mode)
        self.apod_mode = apod_mode
        self.apod_group = max(1, int(apod_group))
        self.ascan_binning = max(1, int(ascan_binning))
        self.raw_apod_size = apod_size
        # apodization lines seen by the kernels (frame_mean prepends one synthetic line)
        self.apod_size = 1 if apod_mode == "frame_mean" else apod_size
        self.n = len(sg.lambda_nm)
        self.nz = self.n // 2
        self.idx = xp.asarray(sg.sinc_idx.T.copy())                         # (T, N)
        self.w = xp.asarray(sg.sinc_w.T.copy().astype(self.dtype))          # (T, N)
        self.window = xp.asarray(sg.window.astype(cdtype))                  # (N,)

    def prepare(self, raw):
        """Apply AScanBinning / synthetic frame-mean apodization line. Returns raw unchanged
        (any integer dtype) when neither applies, else a float array (B, apod + nX, N)."""
        return prepare_raw(raw, self.raw_apod_size, self.ascan_binning, self.apod_mode, self.apod_group,
                           self.xp, self.dtype)

    def linearise(self, raw):
        """raw: (B, interfSize, N) int16 -> equispaced interferogram (B, nX, N)."""
        xp = self.xp
        return self._linearise_prepared(self.prepare(raw))

    def _linearise_prepared(self, raw):
        xp = self.xp
        x = raw.astype(self.dtype, copy=False)
        apod = x[:, : self.apod_size, :].mean(axis=1, keepdims=True)
        interf = x[:, self.apod_size:, :] - apod
        mv = interf.mean(axis=-1, keepdims=True)
        d = interf - mv
        out = xp.zeros_like(d)
        for t in range(self.idx.shape[0]):
            out += self.w[t] * xp.take(d, self.idx[t], axis=-1)
        out += mv
        return out

    def magnitude(self, raw):
        """raw (B, interfSize, N) int16/uint16/float -> |scan| (B, nX, N/2), float.
        nX = A-lines after the apodization lines and after AScanBinning."""
        return self.xp.abs(self.scan(raw)).astype(self.dtype, copy=False)

    def scan(self, raw):
        """raw (B, interfSize, N) -> complex scan (B, nX, N/2) (same fast path as magnitude)."""
        xp = self.xp
        raw = self.prepare(raw)
        if xp is np and self.use_numba:
            from .cpu_kernels import fused_pre_fft
            spec = np.empty((raw.shape[0], raw.shape[1] - self.apod_size, self.n), self.window.dtype)
            fused_pre_fft(raw, self.apod_size, self.idx, self.w, self.window, spec)
        elif xp is not np and self.use_gpu_kernel:
            from .gpu_kernels import SUPPORTED_RAW_DTYPES, fused_pre_fft as gpu_fused
            if raw.dtype not in SUPPORTED_RAW_DTYPES:
                raw = raw.astype(self.dtype)
            spec = gpu_fused(raw, self.apod_size, self.idx, self.w, self.window)
        else:
            spec = self._linearise_prepared(raw) * self.window
        ft = self._ifft(spec)
        return ft[..., : self.nz]

    def complex_scan(self, raw):
        xp = self.xp
        eq = self.linearise(raw)
        return self._ifft(eq * self.window)[..., : self.nz]

    def _ifft(self, a):
        if self.xp is np:  # multithreaded pocketfft on CPU (same 1/N normalisation as MATLAB)
            import os
            import scipy.fft
            return scipy.fft.ifft(a, axis=-1, workers=os.cpu_count(), overwrite_x=True)
        return self.xp.fft.ifft(a, axis=-1)


def average_repeats(mag, n_frames: int, hdr):
    """Legacy yOCTProcessTiledScan.m l.288-291: |scan| is averaged over the trailing dims of
    (z, x, AScanAvg, BScanAvg) -- B-scan repeats first, then A-scan repeats.
    mag: (n_frames*bscan_avg, size_x*ascan_avg, nZ) -> (n_frames, size_x, nZ)."""
    nb, na = max(1, int(hdr.bscan_avg)), max(1, int(hdr.ascan_avg))
    if nb == 1 and na == 1:
        return mag
    m5 = mag.reshape(n_frames, nb, hdr.size_x, na, mag.shape[-1])
    m4 = m5.mean(1) if nb > 1 else m5[:, 0]          # positional axis: numpy, cupy and torch
    return m4.mean(2) if na > 1 else m4[:, :, 0]
