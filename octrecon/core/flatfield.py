"""Tile flat-field correction (new in v2): removes the tile-periodic brightness falloff (vignetting)
that makes the patch boundaries visible in the stitched volume and its projections.

Every tile is acquired with the same scan, so the signal falls off towards the tile edges in the same
way in every tile (objective vignetting / collection efficiency and field curvature moving the focus
away from the reconstructed depth). The falloff depends on depth z. Model per output voxel:

    A(z, y, x) = N(z) + G(z, u, v) * S(z, y, x)

with A the stitched amplitude (10^(dB/20)), N(z) the noise floor (no-tissue columns), S the tissue
signal and (u, v) the position inside the tile. G is estimated from the data, streamed tile row by
tile row (memory: one tile row):

    G(z, u, v) = sum_t X_t(z, u, v) / sum_t m_t(z),   X = max(A - N, 0),  m_t(z) = mean of X in tile t

over the tiles that are fully covered by tissue (normalising each tile by its own level, so tissue
heterogeneity between tiles averages out), smoothed laterally (Gaussian, `smooth_px`) and along z,
and normalised to mean 1 over the tile. Planes with too little signal get G = 1, and the gain is
capped at `max_gain_db` (24 dB) where the falloff is extreme. The correction
divides only the part above the noise floor, so background noise at the tile edges is not amplified:

    A' = A + max(A - N, 0) * (1 / G - 1)

Tiles do not overlap in this acquisition, so the correction cannot register or blend them; it removes
the brightness steps at the seams.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _amp(db):
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(db), np.power(10.0, db / 20.0, dtype=np.float32), np.nan).astype(np.float32)


def tile_layout(ny, nx, tile_y, tile_x, col_ranges=None):
    """Lists of (start, stop) output rows / columns of the tiles. col_ranges: optional explicit
    column ranges (from the stitcher); default: consecutive blocks of tile_x."""
    rows = [(a, min(a + tile_y, ny)) for a in range(0, ny, tile_y)]
    cols = col_ranges or [(a, min(a + tile_x, nx)) for a in range(0, nx, tile_x)]
    return rows, cols


def estimate_flatfield(vol, footprint, rows, cols, smooth_px: float = 15.0, smooth_z: float = 1.0,
                       min_coverage: float = 0.9, min_signal_frac: float = 0.05, max_gain_db: float = 24.0,
                       log=print) -> dict:
    """vol: (ny, nz, nx) dB (memmap ok). footprint: (ny, nx) bool tissue columns.
    Returns {"G": (nz, ty, tx) float32, "N": (nz,) noise floor amplitude, "n_tiles": int, ...}."""
    ny, nz, nx = vol.shape
    ty = max(b - a for a, b in rows)
    tx = max(b - a for a, b in cols)
    # noise floor per depth: median amplitude of the no-tissue columns (subsampled)
    nf = np.full(nz, np.nan, np.float64)
    outside = ~footprint
    if outside.mean() > 0.01:
        sel_rows = np.arange(0, ny, max(1, ny // 300))
        vals = [[] for _ in range(nz)]
        for y in sel_rows:
            a = _amp(np.asarray(vol[y]))[:, outside[y]]
            for z in range(nz):
                v = a[z]
                vals[z].append(v[np.isfinite(v)])
        nf = np.array([np.median(np.concatenate(v)) if sum(len(x) for x in v) else np.nan for v in vals])
    if not np.isfinite(nf).any():
        # field fully covered by tissue: the noise floor is taken from the emptiest depth plane
        # (lowest plane median), assuming it is flat in depth
        med = []
        for z in range(nz):
            a = _amp(np.asarray(vol[::max(1, ny // 100), z, ::max(1, nx // 300)]))
            med.append(np.nanmedian(a))
        nf = np.full(nz, np.nanmin(med) if np.isfinite(med).any() else 0.0)
    nf = np.where(np.isfinite(nf), nf, np.nanmin(nf)).astype(np.float32)
    num = np.zeros((nz, ty, tx), np.float64)
    den = np.zeros((nz, 1, 1), np.float64)
    n_t = 0
    for (y0, y1) in rows:
        if y1 - y0 != ty:
            continue
        chunk = _amp(np.asarray(vol[y0:y1]))                         # (ty, nz, nx)
        for (x0, x1) in cols:
            if x1 - x0 != tx or footprint[y0:y1, x0:x1].mean() < min_coverage:
                continue
            X = np.maximum(chunk[:, :, x0:x1] - nf[None, :, None], 0)  # (ty, nz, tx)
            X = np.nan_to_num(X).transpose(1, 0, 2)                     # (nz, ty, tx)
            m = X.mean(axis=(1, 2), keepdims=True)                      # tile level per depth
            num += X
            den += m
            n_t += 1
    if n_t == 0:
        log("flat-field: no tile fully covered by tissue - correction skipped")
        return {"G": np.ones((nz, ty, tx), np.float32), "N": nf, "n_tiles": 0, "applied": False}
    with np.errstate(invalid="ignore", divide="ignore"):
        G = num / np.maximum(den, 1e-30)
    sig = den[:, 0, 0] / n_t                                          # mean signal level per depth
    weak = sig < min_signal_frac * sig.max()
    G = gaussian_filter(np.nan_to_num(G, nan=1.0), sigma=(0, smooth_px, smooth_px), mode="nearest")
    G = G / np.maximum(G.mean(axis=(1, 2), keepdims=True), 1e-12)
    # along z: signal-weighted smoothing, so planes without tissue do not pull the gain towards 1
    if smooth_z > 0:
        from scipy.ndimage import gaussian_filter1d
        w = np.where(weak, 0.0, sig)[:, None, None]
        G = gaussian_filter1d(G * w, smooth_z, axis=0, mode="nearest") / np.maximum(
            gaussian_filter1d(np.broadcast_to(w, G.shape), smooth_z, axis=0, mode="nearest"), 1e-30)
    G[weak] = 1.0
    # cap the correction: where the falloff is extreme (tile corner at the scan start) there is little
    # signal left and a larger gain would mostly amplify noise
    G = np.clip(G, 10 ** (-max_gain_db / 20), 10 ** (max_gain_db / 20)).astype(np.float32)
    span = 20 * np.log10(G.max(axis=(1, 2)) / G.min(axis=(1, 2)))
    log(f"flat-field: estimated from {n_t} tissue tiles; within-tile falloff up to {span[~weak].max() if (~weak).any() else 0:.1f} dB "
        f"(depths with signal: {(~weak).sum()}/{nz})")
    return {"G": G, "N": nf, "n_tiles": n_t, "applied": True, "falloff_db_per_z": span.tolist(),
            "signal_per_z": sig.tolist()}


def apply_flatfield(vol, ff: dict, rows, cols, progress=None):
    """In place on vol (ny, nz, nx) dB: A' = A + max(A - N, 0) (1/G - 1). Returns per-plane clims."""
    G, N = ff["G"], ff["N"]
    ny, nz, nx = vol.shape
    clims = np.full((ny, 2), np.nan)
    inv = (1.0 / G - 1.0).astype(np.float32)                          # (nz, ty, tx)
    for k, (y0, y1) in enumerate(rows):
        chunk = np.asarray(vol[y0:y1])                                # (h, nz, nx) dB
        A = _amp(chunk)
        h = y1 - y0
        for (x0, x1) in cols:
            w = x1 - x0
            f = inv[:, :h, :w].transpose(1, 0, 2)                    # (h, nz, w)
            sub = A[:, :, x0:x1]
            A[:, :, x0:x1] = sub + np.maximum(sub - N[None, :, None], 0) * f
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.isfinite(chunk), 20 * np.log10(np.maximum(A, 1e-30)), np.nan).astype(np.float32)
        vol[y0:y1] = out
        for i in range(h):
            fin = out[i][np.isfinite(out[i])]
            clims[y0 + i] = (fin.min(), fin.max()) if fin.size else (np.nan, np.nan)
        if progress:
            progress(stage="flatfield", done=k + 1, total=len(rows))
    return clims
