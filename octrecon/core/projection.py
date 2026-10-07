"""Tissue-only xy (en-face) projection of a reconstructed volume (new in v2).

The projection is restricted to the tissue: air / film / background voxels and columns without
tissue do not contribute (they are NaN in the output).

Tissue segmentation (streamed over the (y, z, x) dB volume in y-chunks, so it works on a
memory-mapped volume larger than RAM):
1. lateral smoothing per depth plane: the amplitude A = 10^(I/20) is averaged over a
   `smooth_um` x `smooth_um` window in (y, x) (normalised convolution, NaN-aware);
2. threshold: Otsu's threshold of the smoothed dB values (global histogram), or a fixed value;
   voxel mask M = smoothed > threshold;
3. per column (y, x): tissue footprint = at least `min_voxels` masked voxels; small enclosed
   holes (dark lumens, vignetted tile corners; < `max_hole_mm2`) are filled, specks (< `min_area_mm2`)
   removed; the top / bottom tissue surfaces (first / last masked voxel) are smoothed laterally
   and interpolated into the filled holes -> a tissue slab [top, bottom] per column;
4. projections over the slab (only finite voxels):
       mean :  P(x,y) = 20 log10( mean_{z in slab} 10^(I/20) )   (mean amplitude, like the stitching)
       max  :  P(x,y) = max_{z in slab} I
   plus the tissue thickness map (um) and the slab surfaces.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BINS = np.arange(-120.0, 80.0, 0.05)


def _smooth_db(chunk_db: np.ndarray, size: int) -> np.ndarray:
    """NaN-aware lateral mean of the amplitude per depth plane -> dB. chunk (y, z, x)."""
    ok = np.isfinite(chunk_db)
    a = np.power(10.0, np.where(ok, chunk_db, -np.inf) / 20.0).astype(np.float32)
    sa = ndi.uniform_filter(a, size=(size, 1, size), mode="nearest")
    so = ndi.uniform_filter(ok.astype(np.float32), size=(size, 1, size), mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(so > 0.5, 20 * np.log10(sa / so), np.nan).astype(np.float32)


def otsu_from_hist(h: np.ndarray, centers: np.ndarray) -> float:
    w0 = np.cumsum(h).astype(float)
    w1 = w0[-1] - w0
    s = np.cumsum(h * centers)
    m0 = s / np.maximum(w0, 1)
    m1 = (s[-1] - s) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _chunks(ny, step, pad):
    for a in range(0, ny, step):
        b = min(ny, a + step)
        yield a, b, max(0, a - pad), min(ny, b + pad)


def _fill_nan_smooth(v: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """Normalised-convolution smoothing of v over `valid`, then nearest-valid fill of the rest."""
    num = ndi.uniform_filter(np.where(valid, v, 0).astype(np.float32), size, mode="nearest")
    den = ndi.uniform_filter(valid.astype(np.float32), size, mode="nearest")
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
    bad = ~np.isfinite(out)
    if bad.all() or not bad.any():
        return out
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return out[tuple(idx)]


def _components_area(mask):
    lab, n = ndi.label(mask)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    return lab, n, area


def tissue_projection(vol, px_um: float, dz_um: float, smooth_um: float = 30.0, threshold_db="auto",
                      min_voxels: int = 2, max_hole_mm2: float = 0.5, min_area_mm2: float = 0.005,
                      surface_smooth_um: float = 50.0, chunk: int = 256, log=print, progress=None) -> dict:
    """vol: (y, z, x) float32 dB (NaN = no data), ndarray or memmap. px_um: lateral pixel (um).
    Returns dict of 2D float32 arrays (y, x): mean, max (dB, NaN outside tissue), thickness_um,
    top_um / bottom_um (slab surfaces, um below the first z plane), footprint (bool), + info."""
    ny, nz, nx = vol.shape
    size = max(1, int(round(smooth_um / px_um)) | 1)
    pad = size
    nchunks = (ny + chunk - 1) // chunk
    # ---- pass 1: threshold
    if threshold_db in (None, "auto"):
        h = np.zeros(len(BINS) - 1, np.int64)
        for k, (a, b, pa, pb) in enumerate(_chunks(ny, chunk, pad)):
            s = _smooth_db(np.asarray(vol[pa:pb]), size)[a - pa:b - pa]
            h += np.histogram(s[np.isfinite(s)], BINS)[0]
            if progress:
                progress(stage="projection", done=k + 1, total=3 * nchunks)
        thr = otsu_from_hist(h, (BINS[:-1] + BINS[1:]) / 2)
        src = "otsu"
    else:
        thr, src = float(threshold_db), "user"
    log(f"tissue mask: lateral smoothing {size} px ({size * px_um:.0f} um), threshold {thr:.2f} dB ({src})")
    # ---- pass 2: per-column mask statistics
    count = np.zeros((ny, nx), np.int16)
    top = np.full((ny, nx), -1, np.int16)
    bot = np.full((ny, nx), -1, np.int16)
    for k, (a, b, pa, pb) in enumerate(_chunks(ny, chunk, pad)):
        m = _smooth_db(np.asarray(vol[pa:pb]), size)[a - pa:b - pa] > thr
        anym = m.any(1)
        count[a:b] = m.sum(1)
        top[a:b] = np.where(anym, np.argmax(m, 1), -1)
        bot[a:b] = np.where(anym, nz - 1 - np.argmax(m[:, ::-1], 1), -1)
        if progress:
            progress(stage="projection", done=nchunks + k + 1, total=3 * nchunks)
    # ---- 2D footprint (specks removed, small enclosed holes filled) and slab surfaces
    px_mm2 = (px_um * 1e-3) ** 2
    measured = count >= min_voxels
    lab, n, area = _components_area(measured)
    keep = area * px_mm2 >= min_area_mm2
    keep[0] = False
    fp = keep[lab]
    lab, n, area = _components_area(~fp)
    small = area * px_mm2 < max_hole_mm2
    small[0] = False
    small[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = False   # open to the border: outside
    fp = fp | small[lab]
    ssz = max(1, int(round(surface_smooth_um / px_um)) | 1)
    t_s = _fill_nan_smooth(top.astype(np.float32), measured, ssz)
    b_s = _fill_nan_smooth(bot.astype(np.float32), measured, ssz)
    t_i = np.where(fp, np.floor(np.nan_to_num(t_s, nan=0)), 0).astype(np.int16)
    b_i = np.where(fp, np.ceil(np.nan_to_num(b_s, nan=-1)), -1).astype(np.int16)
    # ---- pass 3: projections over the slab
    pmean = np.full((ny, nx), np.nan, np.float32)
    pmax = np.full((ny, nx), np.nan, np.float32)
    zz = np.arange(nz)[None, :, None]
    for k, (a, b, _, _) in enumerate(_chunks(ny, chunk, 0)):
        c = np.asarray(vol[a:b])
        inside = (zz >= t_i[a:b, None, :]) & (zz <= b_i[a:b, None, :]) & np.isfinite(c)
        amp = np.power(10.0, np.where(inside, c, -np.inf) / 20.0).astype(np.float32)
        n_in = inside.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmean[a:b] = np.where(n_in > 0, 20 * np.log10(amp.sum(1) / np.maximum(n_in, 1)), np.nan)
        pmax[a:b] = np.where(n_in > 0, np.max(np.where(inside, c, -np.inf), 1), np.nan)
        if progress:
            progress(stage="projection", done=2 * nchunks + k + 1, total=3 * nchunks)
    thick = np.where(fp, (b_i - t_i + 1) * dz_um, 0).astype(np.float32)
    info = {"threshold_db": thr, "threshold_source": src, "smooth_px": size,
            "footprint_fraction": float(fp.mean()),
            "median_thickness_um": float(np.median(thick[fp])) if fp.any() else 0.0}
    log(f"tissue footprint {100 * info['footprint_fraction']:.1f}% of the field, median slab "
        f"{info['median_thickness_um']:.1f} um")
    return {"mean": pmean, "max": pmax, "thickness_um": thick,
            "top_um": np.where(fp, t_i * dz_um, np.nan).astype(np.float32),
            "bottom_um": np.where(fp, b_i * dz_um, np.nan).astype(np.float32),
            "footprint": fp, "info": info}


def all_z_projection(vol, chunk: int = 256) -> np.ndarray:
    """Mean-amplitude projection over all finite z (used for the removed-signal volume)."""
    ny, nz, nx = vol.shape
    out = np.full((ny, nx), np.nan, np.float32)
    for a, b, _, _ in _chunks(ny, chunk, 0):
        c = np.asarray(vol[a:b])
        ok = np.isfinite(c)
        amp = np.power(10.0, np.where(ok, c, -np.inf) / 20.0).astype(np.float32)
        n = ok.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[a:b] = np.where(n > 0, 20 * np.log10(amp.sum(1) / np.maximum(n, 1)), np.nan)
    return out
