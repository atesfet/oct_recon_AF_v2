"""Removal of specular reflections from FEP films (or any flat, mirror-like interface).

Problem
-------
The tissue is sandwiched between FEP films. Their flat surfaces act as mirrors: each gives a
sharp, very strong reflection (~20 dB above tissue) exactly at the tissue surface (and
sometimes at its bottom). In the stitched volume this appears as a bright band at the tissue
surface and as round blobs (strongest where the beam hits the film at normal incidence) in
every tile outside the tissue.

What distinguishes the reflection from tissue (measured on 10um_FOV_1)
-----------------------------------------------------------------------
* axially it is a single point-spread function (PSF): the same complex depth profile in every
  A-line, only scaled/phase-shifted and shifted by a fraction of a pixel. Tissue speckle is
  random in depth and spreads over all depth profiles;
* laterally it is a smooth surface (after the optical-path / field-curvature correction it is
  flat to +-1 px), whereas tissue structure varies from A-line to A-line.

Method (per raw B-scan, on the complex A-scans, before |.| and stitching)
----------------------------------------------------------------------------
1. detect surfaces: sharp, prominent peaks of the laterally averaged, curvature-flattened
   magnitude profile (only in the depth range that reaches the output);
2. per A-line and surface: expected position from the surface row + the optical-path shift,
   refined by a local peak search and a lateral median (a tissue speckle cannot hijack it);
3. project the complex segment (+-R samples) onto a rank-r PSF basis (sub-pixel shifts of the
   system PSF), learnt from the data itself (bootstrapped from the theoretical PSF);
4. subtract the projection where the reflection is present: the energy fraction captured by the
   basis ("specular-likeness", median over neighbouring A-lines) controls a soft weight, and the
   PSF component is shrunk down to the local background level (not to zero: the background /
   tissue speckle also has energy along the basis, removing it all would leave a dark trench).
Tissue outside the +-R window is untouched; inside it only the components along the PSF basis
(r of 2R+1 dimensions) are affected. Ground-truth tests: see tests/test_fep.py, docs/07_fep_removal.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class FEPConfig:
    enabled: bool = True
    rank: int = 3                 # PSF basis size (1 = fixed shape; 2-3 also absorb sub-pixel shifts)
    half_window: int = 8          # R: depth samples on each side of the surface peak
    search: int = 4               # +- samples searched around the expected surface row
    lateral_median: int = 15      # A-lines: median for surface position and specular decision
    frac_lo: float = 0.35         # specular-likeness below which nothing is removed
    frac_hi: float = 0.60         # ... above which the full reflection is removed
    detect_min_db: float = 6.0    # surface must stand this far above the local depth baseline
    max_surfaces: int = 4         # per B-scan batch (top / bottom FEP, film back side, ...)
    bg_rows: int = 10             # background rows on each side for the shrinkage
    shrink: bool = True
    keep_level: float = 0.7       # fraction of the local background energy kept along the basis
    bg_mode: str = "both"         # background power for the shrinkage: both | max | inside
    coherence_limit: bool = True  # cap the removal at the laterally coherent (film) energy
    coh_lag: int = 4              # A-line lag of the coherence product (beyond the speckle size)
    coh_window: int = 31          # A-lines averaged for the coherent film-energy estimate
    coh_margin: float = 1.0       # allowed excess over the estimate (film amplitude jitter)
    use_specular_weight: bool = True  # keep the specular-likeness decision weight w
    learn_basis: bool = True      # learn the PSF basis from the scan (else theoretical)


# --------------------------------------------------------------------------- basis
def theory_basis(window: np.ndarray, R: int, rank: int) -> np.ndarray:
    """Complex depth profiles of an ideal reflector at sub-pixel positions, through the actual
    spectral window (magnitude), compressed to `rank` components (2R+1, rank)."""
    N = len(window)
    aw = np.abs(window)
    m = N // 4
    prof = []
    for d in np.linspace(-0.5, 0.5, 41):
        a = np.fft.ifft(np.exp(-2j * np.pi * np.arange(N) * (m + d) / N) * aw)
        s = a[m - R:m + R + 1]
        prof.append(s / np.linalg.norm(s))
    U, _, _ = np.linalg.svd(np.array(prof).T, full_matrices=False)
    return U[:, :rank]


def basis_from_segments(segs: np.ndarray, rank: int):
    """segs (n, 2R+1) complex, each centred on its own peak -> (U (2R+1, rank), captured energy)."""
    peak = segs[:, segs.shape[1] // 2]
    S = segs * np.exp(-1j * np.angle(peak))[:, None]
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    U, sv, _ = np.linalg.svd(S.T, full_matrices=False)
    e = sv ** 2
    return U[:, :rank], float(e[:rank].sum() / e.sum())


# --------------------------------------------------------------------------- helpers
def _np(xp, a):
    if xp is np:
        return a
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return a.get()


def flatten_mag(mag, shift_rows):
    """Curvature-flatten |scan| like the optical-path correction: out[b,x,r] = mag[b,x,r+shift[b,x]]
    (rows outside -> 0). mag (B, nX, nZ) numpy; shift_rows (B, nX) int."""
    B, nX, nZ = mag.shape
    r = np.arange(nZ)
    src = r[None, None, :] + shift_rows[:, :, None]
    ok = (src >= 0) & (src < nZ)
    return np.take_along_axis(mag, np.clip(src, 0, nZ - 1), axis=-1) * ok


def _detect_from_profile(mean_mag: np.ndarray, row_lo: int, row_hi: int, cfg: FEPConfig):
    """mean_mag: laterally averaged flattened |scan| (nZ, NaN outside the analysed band).
    Returns rows of sharp peaks standing >= detect_min_db above the local (median) baseline."""
    nZ = len(mean_mag)
    prof = 20 * np.log10(np.where(np.isfinite(mean_mag), mean_mag, np.nan) + 1e-30)
    ok = np.isfinite(prof)
    filled = np.where(ok, prof, np.nanmedian(prof[ok]) if ok.any() else 0.0)
    base = median_filter(filled, size=51, mode="nearest")
    lo, hi = max(row_lo, 3), min(row_hi, nZ - 4)
    cand = []
    for r in range(lo, hi + 1):
        if ok[r] and filled[r] == filled[r - 3:r + 4].max() and filled[r] - base[r] >= cfg.detect_min_db:
            cand.append((filled[r] - base[r], r))
    cand.sort(reverse=True)
    rows = []
    for _, r in cand:                       # strongest first, at least R apart
        if all(abs(r - q) > cfg.half_window for q in rows):
            rows.append(r)
        if len(rows) >= cfg.max_surfaces:
            break
    return sorted(rows)


def detect_surfaces(flat_mag: np.ndarray, row_lo: int, row_hi: int, cfg: FEPConfig, x_frac=(0.15, 0.85)):
    """Rows (flattened) of sharp, prominent peaks in the laterally averaged profile (host arrays)."""
    B, nX, nZ = flat_mag.shape
    xs = slice(int(nX * x_frac[0]), int(nX * x_frac[1]))
    m = np.mean(flat_mag[:, xs, :], axis=(0, 1))
    m2 = np.full(nZ, np.nan)
    a, b = max(row_lo, 0), min(row_hi, nZ - 1)
    m2[a:b + 1] = m[a:b + 1]
    return _detect_from_profile(m2, row_lo, row_hi, cfg), m


# --------------------------------------------------------------------------- removal
class FEPRemover:
    """Applies the removal to batches of complex B-scans of one tile.

    shift / pdz: optical-path correction maps (nY, nX) (OpticalPathCorrection.shift), used to
    convert flattened rows to native rows per A-line."""

    def __init__(self, U: np.ndarray, shift: np.ndarray, cfg: FEPConfig, xp=np):
        self.cfg = cfg
        self.xp = xp
        self.U = U.astype(np.complex64)
        self.shift = np.asarray(shift, np.int64)
        self.R = cfg.half_window
        self.stats = {"batches": 0, "surfaces": 0, "removed_energy_db": []}

    def _background_power(self, wxp, work, pk, bidx_off, bg, es, ec, n_win, rank):
        """Per-sample power of the signal *under* the reflection (tissue or noise) that the
        shrinkage keeps. both: mean over bg rows above and below the window; max: the brighter
        side (the tissue side at a film/tissue interface); inside: energy of the window segment
        orthogonal to the PSF basis, per dimension (the level right at the surface)."""
        mode = self.cfg.bg_mode
        if mode == "inside":
            return wxp.maximum(es - ec, 0) / (n_win - rank)
        p = wxp.abs(wxp.take_along_axis(work, wxp.asarray(pk[..., None] + bidx_off), axis=-1)) ** 2
        if mode == "max":
            return wxp.maximum(wxp.mean(p[..., :bg], axis=-1), wxp.mean(p[..., bg:], axis=-1))
        return wxp.mean(p, axis=-1)

    def _coherence_gain(self, wxp, work, idx, ec, B, nX, nZ):
        """Cap the subtracted energy at the film energy, estimated from the lateral coherence:
        E_f(x) = | mean_{x' in window} <s(x'), s(x'+L)> |, where both segments are taken on the
        rows of A-line x'. The film is a continuous mirror, so its products add up coherently;
        tissue speckle decorrelates beyond the speckle size (lag L) and averages out.
        Returns g_coh = min(1, (1 + margin) * sqrt(E_f / ||c||^2))."""
        cfg = self.cfg
        L = int(cfg.coh_lag)
        xl = np.arange(nX) + L
        xl = np.where(xl < nX, xl, np.arange(nX) - L)                    # mirror at the right edge
        flat = ((np.arange(B)[:, None, None] * nX + xl[None, :, None]) * nZ + idx)
        work_flat = work.reshape(-1)
        seg = wxp.take_along_axis(work, wxp.asarray(idx), axis=-1)
        seg_l = work_flat[wxp.asarray(flat.reshape(-1))].reshape(seg.shape)
        prod = _np(wxp, wxp.sum(seg * wxp.conj(seg_l), axis=-1))          # (B, nX) complex, host
        from scipy.ndimage import uniform_filter1d
        m = (uniform_filter1d(prod.real, cfg.coh_window, axis=-1, mode="nearest")
             + 1j * uniform_filter1d(prod.imag, cfg.coh_window, axis=-1, mode="nearest"))
        # remove the bias of incoherent (tissue) products: E|mean|^2 = mean|q|^2 / n for random phases
        v = uniform_filter1d(np.abs(prod) ** 2, cfg.coh_window, axis=-1, mode="nearest") / cfg.coh_window
        ef = np.sqrt(np.maximum(np.abs(m) ** 2 - v, 0.0))
        ec_h = np.maximum(_np(wxp, ec), 1e-30)
        g = np.minimum(1.0, (1.0 + cfg.coh_margin) * np.sqrt(ef / ec_h)).astype(np.float32)
        self.stats.setdefault("coh_gain_median", []).append(float(np.median(g)))
        return wxp.asarray(g)

    def process(self, cpx, frames, row_lo: int, row_hi: int):
        """cpx: (B, nX, nZ) complex scans on the backend (numpy / cupy) or a torch tensor.
        frames: tile-local frame indices of the batch (host ints). row_lo/row_hi: flattened rows
        (stitcher rows) that reach the output. Heavy work stays on the device and only touches
        the depth band that matters; only small (B, nX) arrays go to the host."""
        cfg, xp, R, bg, S = self.cfg, self.xp, self.R, self.cfg.bg_rows, self.cfg.search
        torch_like = hasattr(cpx, "detach")
        work = cpx.detach().cpu().numpy() if torch_like else cpx
        wxp = np if torch_like else xp
        B, nX, nZ = work.shape
        sh = self.shift[np.asarray(frames)]                          # (B, nX) host
        self.stats["batches"] += 1
        # ---- detection on the curvature-flattened profile (device gather, host 1-D profile)
        f0, f1 = max(row_lo - R - S, 0), min(row_hi + R + S, nZ - 1)
        frows = np.arange(f0, f1 + 1)
        src = np.clip(frows[None, None, :] + sh[:, :, None], 0, nZ - 1)
        xs = slice(int(nX * 0.15), int(nX * 0.85))
        flat = wxp.abs(wxp.take_along_axis(work[:, xs, :], wxp.asarray(src[:, xs, :]), axis=-1))
        prof = _np(wxp, wxp.mean(flat, axis=(0, 1)))
        full = np.full(nZ, np.nan)
        full[f0:f1 + 1] = prof
        rows = _detect_from_profile(full, f0, f1, cfg)
        if not rows:
            return cpx
        self.stats["surfaces"] += len(rows)
        U = wxp.asarray(self.U)
        Uc = wxp.conj(U)
        k = np.arange(-R, R + 1)
        lo_lim, hi_lim = R + bg, nZ - R - bg - 1
        bidx_off = np.concatenate([-R - 1 - np.arange(bg), R + 1 + np.arange(bg)])
        for r in rows:
            # per A-line peak near the expected native row, then lateral median (host, small)
            cand = np.clip((r + sh)[:, :, None] + np.arange(-S, S + 1), 0, nZ - 1)
            cmag = wxp.abs(wxp.take_along_axis(work, wxp.asarray(cand), axis=-1))
            am = _np(wxp, wxp.argmax(cmag, axis=-1))
            pk = np.take_along_axis(cand, am[..., None], -1)[..., 0]
            pk = median_filter(pk.astype(float), size=(1, cfg.lateral_median), mode="nearest").round().astype(np.int64)
            pk = np.clip(pk, lo_lim, hi_lim)
            idx = pk[..., None] + k                                  # (B, nX, 2R+1)
            idx_d = wxp.asarray(idx)
            seg = wxp.take_along_axis(work, idx_d, axis=-1)
            c = seg @ Uc                                             # (B, nX, rank)
            ec = wxp.sum(wxp.abs(c) ** 2, axis=-1)
            es = wxp.sum(wxp.abs(seg) ** 2, axis=-1) + 1e-30
            frac = _np(wxp, ec / es)
            fmed = median_filter(frac, size=(1, cfg.lateral_median), mode="nearest")
            w = np.clip((fmed - cfg.frac_lo) / (cfg.frac_hi - cfg.frac_lo), 0, 1).astype(np.float32)
            if not cfg.use_specular_weight:
                w = np.ones_like(w)
            gain = wxp.asarray(w)
            if cfg.coherence_limit:
                gain = gain * self._coherence_gain(wxp, work, idx, ec, B, nX, nZ)
            if cfg.shrink:
                bpow = self._background_power(wxp, work, pk, bidx_off, bg, es, ec, 2 * R + 1, U.shape[1])
                keep = cfg.keep_level * U.shape[1] * bpow                # expected background energy in the subspace
                gain = gain * wxp.maximum(0.0, 1.0 - wxp.sqrt(keep / wxp.maximum(ec, 1e-30))).astype(np.float32)
            delta = gain[..., None] * (c @ U.T)                      # (B, nX, 2R+1)
            flat_idx = (wxp.arange(B)[:, None, None] * nX + wxp.arange(nX)[None, :, None]) * nZ + idx_d
            work = work.reshape(-1)
            work[flat_idx.reshape(-1)] = (seg - delta).reshape(-1)
            work = work.reshape(B, nX, nZ)
            removed = float(_np(wxp, wxp.sum(wxp.abs(delta) ** 2) / wxp.sum(es)))
            self.stats["removed_energy_db"].append(10 * np.log10(max(removed, 1e-12)))
        if torch_like:
            import torch
            return torch.from_numpy(work).to(cpx.device)
        return work

    def summary(self):
        e = self.stats["removed_energy_db"]
        return {"batches": self.stats["batches"], "surfaces_processed": self.stats["surfaces"],
                "median_removed_fraction_db": float(np.median(e)) if e else None, "config": asdict(self.cfg)}


# --------------------------------------------------------------------------- learning
def learn_basis_from_volume(get_cpx, tiles, shift, window, rows_for_zi, cfg: FEPConfig, log=print, max_segments=6000):
    """Learn the PSF basis from strongly specular segments of a few tiles.

    get_cpx(folder, frames) -> (B, nX, nZ) complex numpy scans; tiles: list of (folder, zi);
    rows_for_zi(zi) -> (row_lo, row_hi) flattened rows that reach the output.
    Returns (U, info)."""
    R = cfg.half_window
    U0 = theory_basis(window, R, cfg.rank)
    segs = []
    for folder, zi in tiles:
        nfr = len(shift)                                   # B-scans per tile
        f0 = max(0, nfr // 2 - 10)
        frames = list(range(f0, min(nfr, f0 + 20), 4)) or [0]   # a few from the tile centre
        c = get_cpx(folder, frames)
        B, nX, nZ = c.shape
        sh = np.asarray(shift)[frames]
        mag = np.abs(c).astype(np.float32)
        lo, hi = rows_for_zi(zi)
        rows, _ = detect_surfaces(flatten_mag(mag, sh), lo - R, hi + R, cfg)
        for r in rows:
            exp_row = r + sh
            cand = np.clip(exp_row[:, :, None] + np.arange(-cfg.search, cfg.search + 1), R, nZ - R - 1)
            pk = np.take_along_axis(cand, np.argmax(np.take_along_axis(mag, cand, axis=-1), -1)[..., None], -1)[..., 0]
            idx = pk[..., None] + np.arange(-R, R + 1)
            s = np.take_along_axis(c, idx, axis=-1).reshape(-1, 2 * R + 1)
            f = np.sum(np.abs(s @ U0.conj()) ** 2, -1) / (np.sum(np.abs(s) ** 2, -1) + 1e-30)
            amp = np.abs(s[:, R])
            keep = (f >= 0.75) & (amp >= np.percentile(amp, 50))
            segs.append(s[keep])
    segs = np.concatenate(segs) if segs else np.zeros((0, 2 * R + 1), complex)
    if len(segs) < 50 or not cfg.learn_basis:
        log(f"FEP: using the theoretical PSF basis ({len(segs)} clean specular segments found)")
        return U0, {"source": "theory", "segments": int(len(segs))}
    if len(segs) > max_segments:
        segs = segs[np.random.default_rng(0).choice(len(segs), max_segments, replace=False)]
    U, cap = basis_from_segments(segs, cfg.rank)
    f_theory = float(np.median(np.sum(np.abs(segs @ U0.conj()) ** 2, -1) / np.sum(np.abs(segs) ** 2, -1)))
    f_data = float(np.median(np.sum(np.abs(segs @ U.conj()) ** 2, -1) / np.sum(np.abs(segs) ** 2, -1)))
    log(f"FEP: PSF basis learnt from {len(segs)} specular segments in {len(tiles)} tiles "
        f"(captures {100*f_data:.1f}% of a reflection vs {100*f_theory:.1f}% for the theoretical PSF)")
    return U, {"source": "data", "segments": int(len(segs)), "captured_data": f_data, "captured_theory": f_theory}
