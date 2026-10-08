"""v2 outputs written after the reconstruction: tissue-only xy projection, removed FEP signal
(volume + projection) and an overview figure. Called by Reconstructor.run()."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core.projection import all_z_projection, tissue_projection
from .io import tiff_writer
from .io.metadata import voxel_size_um


def write_2d_tiff(path: Path, img: np.ndarray, md: dict, acq: dict | None, info: dict | None = None):
    """Calibrated float32 2D TIFF (pixel size from the output x / y axes, unit micron, acquisition
    metadata, value meaning; .json sidecar). NaN = no data."""
    return tiff_writer.write_2d_calibrated(path, img, md, acq, info)


def _clim(img, lo=1.0, hi=99.7):
    fin = img[np.isfinite(img)]
    if not fin.size:
        return 0.0, 1.0
    a, b = np.percentile(fin, [lo, hi])
    return float(a), float(b) if b > a else float(a) + 1.0


def cfg_enh_sigma(px_um: float, scale_um: float = 80.0) -> float:
    return max(2.0, scale_um / px_um)


def local_contrast(img_db: np.ndarray, sigma_px: float) -> np.ndarray:
    """Display-only local contrast normalisation of a dB image (NaN = no tissue): subtract the local
    mean and divide by the local standard deviation, both NaN-aware Gaussian (normalised convolution)
    over sigma_px. Features become equally visible in bright and dim regions; values are z-scores."""
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(img_db)
    v = np.where(ok, img_db, 0.0).astype(np.float32)
    w = gaussian_filter(ok.astype(np.float32), sigma_px)
    mu = gaussian_filter(v, sigma_px) / np.maximum(w, 1e-6)
    var = gaussian_filter(np.where(ok, (v - mu) ** 2, 0.0).astype(np.float32), sigma_px) / np.maximum(w, 1e-6)
    out = (v - mu) / np.sqrt(np.maximum(var, 1e-6))
    return np.where(ok, out, np.nan).astype(np.float32)


def write_png(path: Path, img: np.ndarray, clim=None, cmap="gray", px_um: float | None = None, text: dict | None = None):
    """8-bit preview; with px_um the physical pixel size is stored (PNG pHYs, pixels per metre)."""
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image, PngImagePlugin
    lo, hi = clim or _clim(img)
    rgba = matplotlib.colormaps[cmap]((np.clip(np.where(np.isfinite(img), img, lo), lo, hi) - lo) / max(hi - lo, 1e-12))
    im = Image.fromarray((rgba[..., :3] * 255).round().astype(np.uint8))
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(str(k), str(v))
    kw = {"pnginfo": info}
    if px_um:
        kw["dpi"] = (25400.0 / px_um, 25400.0 / px_um)
    im.save(str(path), **kw)
    return lo, hi


def _cm(name):
    import matplotlib
    return matplotlib.colormaps[name].with_extremes(bad="black")   # NaN / nothing removed / no data


def _overview(path, name, proj, rm_proj, vol, vol_rm, md, clim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.asarray(md["x"]["values"]); y = np.asarray(md["y"]["values"]); z = np.asarray(md["z"]["values"]) * 1e3
    ny = vol.shape[0]
    yi = ny // 2
    st = max(1, int(np.ceil(max(vol.shape[0], vol.shape[2]) / 2500)))
    panels = []
    if proj is not None:
        panels.append(("xy projection (tissue only, strongest layer per pixel)", proj, "gray"))
    if rm_proj is not None:
        panels.append(("removed FEP signal, xy projection (mean over z)", rm_proj, "magma"))
    nb = 2 if vol_rm is not None else 1
    npn = max(1, len(panels))
    aspect = (y[-1] - y[0]) / max(x[-1] - x[0], 1e-9)
    hp = 7.5 * aspect + 1.2
    fig = plt.figure(figsize=(7.5 * npn, hp + 2.4 * nb), constrained_layout=True)
    gs = fig.add_gridspec(1 + nb, npn, height_ratios=[hp] + [2.4] * nb)
    ext_xy = [x[0], x[-1], y[-1], y[0]]
    for i, (t, im, cm) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        lo, hi = _clim(im)
        h = ax.imshow(im[::st, ::st], cmap=_cm(cm), vmin=lo, vmax=hi, extent=ext_xy, aspect="equal", interpolation="nearest")
        ax.axhline(y[yi], color="c", lw=0.6, ls="--")
        ax.set_title(t, fontsize=10); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        fig.colorbar(h, ax=ax, label="dB", shrink=0.8)
    ext_b = [x[0], x[-1], z[-1], z[0]]
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(vol[yi][:, ::st], cmap="gray", vmin=clim[0], vmax=clim[1], extent=ext_b, aspect="auto", interpolation="nearest")
    ax.set_title(f"B-scan at y = {y[yi]:.3f} mm (FEP removed)" if vol_rm is not None else f"B-scan at y = {y[yi]:.3f} mm")
    ax.set_ylabel("z [µm]")
    if vol_rm is not None:
        ax = fig.add_subplot(gs[2, :])
        ax.imshow(vol_rm[yi][:, ::st], cmap=_cm("magma"), vmin=clim[0], vmax=clim[1], extent=ext_b, aspect="auto", interpolation="nearest")
        ax.set_title("removed (subtracted) FEP signal, same B-scan, same dB scale"); ax.set_ylabel("z [µm]"); ax.set_xlabel("x [mm]")
    fig.suptitle(name)
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


def write_v2_outputs(rec, out_dir: Path, name: str, vol, vol_rm, clim, rm_clims, md: dict, acq: dict,
                     write_removed_volume: bool = True) -> dict:
    cfg, log = rec.cfg, rec.log
    vox = voxel_size_um(md)
    px, dz = float(vox["x"]), float(vox["z"])
    res = {}
    proj = None
    if cfg.xy_projection:
        rec.progress(stage="projection", done=0, total=1)
        if cfg.xy_projection_tissue_only:
            P = tissue_projection(vol, px, dz, smooth_um=cfg.tissue_smooth_um, threshold_db=cfg.tissue_threshold_db,
                                  max_hole_mm2=cfg.tissue_max_hole_mm2, min_area_mm2=cfg.tissue_min_area_mm2,
                                  log=log, progress=rec.progress)
            info = dict(P["info"], mode="tissue only", fep_removal=rec.fep is not None)
            if rec.fep is None:
                log("note: FEP removal is off - film reflections can be segmented as tissue in the projection")
        else:
            m = all_z_projection(vol)
            mx = np.full(m.shape, np.nan, np.float32)
            for a in range(0, vol.shape[0], 256):
                c = np.asarray(vol[a:a + 256])
                mx[a:a + 256] = np.where(np.isfinite(c).any(1), np.nanmax(np.where(np.isfinite(c), c, -np.inf), 1), np.nan)
            P = {"mean": m, "max": mx}
            info = {"mode": "all z", "fep_removal": rec.fep is not None}
        proj = P.get("topk", P["mean"])
        files = {}
        for key in [k for k in ("topk", "mean", "max") if k in P]:
            tif = out_dir / f"{name}_xy_{key}.tif"
            write_2d_tiff(tif, P[key], md, acq, dict(info, projection=key, value_unit="dB", nan="no tissue",
                                                     definition={"mean": "20 log10 mean amplitude over the tissue slab",
                                                                 "max": "max dB over the tissue slab",
                                                                 "topk": "20 log10 mean of the k strongest background-subtracted slices of the tissue slab"}[key]))
            lo, hi = write_png(out_dir / f"{name}_xy_{key}.png", P[key], px_um=px,
                               text={"pixel_size_um": px, "value": f"dB {key} projection, display {P[key].dtype}"})
            files[key] = {"tif": str(tif), "png": str(out_dir / f"{name}_xy_{key}.png"), "png_clim_dB": [lo, hi]}
            enh = out_dir / f"{name}_xy_{key}_enhanced.png"
            write_png(enh, local_contrast(P[key], cfg_enh_sigma(px)), clim=(-2.5, 2.5), px_um=px,
                      text={"pixel_size_um": px, "display": "local contrast normalised (display only, not quantitative)"})
            files[key]["png_enhanced"] = str(enh)
        if "thickness_um" in P:
            tif = out_dir / f"{name}_tissue_thickness_um.tif"
            write_2d_tiff(tif, P["thickness_um"], md, acq, {"value_unit": "um", "definition": "tissue slab thickness", "zero": "no tissue"})
            write_png(out_dir / f"{name}_tissue_thickness_um.png", np.where(P["footprint"], P["thickness_um"], np.nan),
                      cmap="viridis", px_um=px)
            files["thickness"] = str(tif)
        res["xy_projection"] = dict(info, files=files, pixel_um=px)
        log(f"wrote xy projections ({info['mode']}): {files['mean']['tif']}")
    rm_proj = None
    if vol_rm is not None:
        tif = out_dir / f"{name}_fep_removed.tiff"
        if write_removed_volume:
            hi = float(np.nanmax(rm_clims[:, 1])) if np.isfinite(rm_clims[:, 1]).any() else clim[1]
            clim_rm = (float(clim[0]), max(hi, float(clim[0]) + 1.0))
            tiff_writer.write_legacy_tiff(tif, vol_rm, clim_rm, md, False, None, acquisition=acq)
        else:
            clim_rm = tuple(tiff_writer.read_meta(tif)["clim"]) if tif.exists() else (float(clim[0]), float(clim[1]))
        rm_proj = all_z_projection(vol_rm)
        tif2 = out_dir / f"{name}_fep_removed_xy.tif"
        write_2d_tiff(tif2, rm_proj, md, acq, {"value_unit": "dB", "projection": "removed FEP signal",
                                               "definition": "20 log10 mean amplitude over all z", "nan": "no data"})
        write_png(out_dir / f"{name}_fep_removed_xy.png", rm_proj, cmap="magma", px_um=px)
        res["fep_removed"] = {"tiff": str(tif), "clim_dB": list(clim_rm), "xy_tif": str(tif2)}
        log(f"wrote removed FEP signal: {tif}")
    if proj is not None or rm_proj is not None:
        ov = out_dir / f"{name}_fep_overview.png" if vol_rm is not None else out_dir / f"{name}_overview.png"
        _overview(ov, name, proj, rm_proj, vol, vol_rm, md, clim)
        res["overview_png"] = str(ov)
    return res


def reproject(out_dir, name: str | None = None, tissue_only: bool = True, smooth_um: float = 30.0,
              threshold_db="auto", max_hole_mm2: float = 0.5, min_area_mm2: float = 0.005, log=print) -> dict:
    """Recompute the xy projection (and removed-signal projection / overview) of an existing
    reconstruction from its kept float32 volume(s) (`keep_float_volume=True`), without
    reconstructing again."""
    from types import SimpleNamespace
    out_dir = Path(out_dir)
    if name is None:
        cands = sorted(out_dir.glob("*_dB_float32.npy"))
        cands = [c for c in cands if not c.name.endswith("_fep_removed_dB_float32.npy")]
        if not cands:
            raise FileNotFoundError(f"no <name>_dB_float32.npy in {out_dir} (reconstruct with keep_float_volume=true)")
        name = cands[0].name[:-len("_dB_float32.npy")]
    vol = np.load(out_dir / f"{name}_dB_float32.npy", mmap_mode="r")
    rmp = out_dir / f"{name}_fep_removed_dB_float32.npy"
    vol_rm = np.load(rmp, mmap_mode="r") if rmp.exists() else None
    meta = tiff_writer.read_meta(out_dir / f"{name}.tiff")
    cfg = SimpleNamespace(xy_projection=True, xy_projection_tissue_only=tissue_only, tissue_smooth_um=smooth_um,
                          tissue_threshold_db=threshold_db, tissue_max_hole_mm2=max_hole_mm2,
                          tissue_min_area_mm2=min_area_mm2)
    rec = SimpleNamespace(cfg=cfg, log=log, progress=lambda **kw: None,
                          fep=True if vol_rm is not None else None)
    res = write_v2_outputs(rec, out_dir, name, vol, vol_rm, tuple(meta["clim"]), None, meta["metadata"],
                           meta.get("acquisition"), write_removed_volume=False)
    sp = out_dir / f"{name}_run_summary.json"
    if sp.exists():
        summ = json.loads(sp.read_text())
        summ.update(res)
        sp.write_text(json.dumps(summ, indent=2, default=str))
    return res


def flatfield_correct(vol, md: dict, rows, cols, out_dir: Path, name: str, cfg, log=print, progress=None,
                      footprint=None) -> tuple[np.ndarray, dict]:
    """Tile flat-field correction of the stitched dB volume in place (see core/flatfield.py).
    Writes the correction map (<name>_flatfield_gain_dB.tif, z x v x u). Returns (per-plane clims, info)."""
    from .core.flatfield import apply_flatfield, estimate_flatfield
    vox = voxel_size_um(md)
    if footprint is None:
        footprint = np.isfinite(tissue_projection(vol, float(vox["x"]), float(vox["z"]), smooth_um=cfg.tissue_smooth_um,
                                                  threshold_db=cfg.tissue_threshold_db, max_hole_mm2=cfg.tissue_max_hole_mm2,
                                                  min_area_mm2=cfg.tissue_min_area_mm2, log=lambda m: None)["mean"])
    ff = estimate_flatfield(vol, footprint, rows, cols, smooth_px=cfg.flatfield_smooth_px,
                            max_gain_db=cfg.flatfield_max_gain_db, log=log)
    info = {k: v for k, v in ff.items() if k not in ("G", "N")}
    if not ff["applied"]:
        return None, info
    clims = apply_flatfield(vol, ff, rows, cols, progress=progress)
    import tifffile
    gdb = (20 * np.log10(ff["G"])).astype(np.float32)
    path = out_dir / f"{name}_flatfield_gain_dB.tif"
    tifffile.imwrite(path, gdb, imagej=True, resolution=(1.0 / float(vox["x"]), 1.0 / float(vox["y"])),
                     metadata={"unit": "micron", "spacing": float(vox["z"]), "axes": "ZYX",
                               "Info": "Tile flat-field gain G (dB) per output depth (slices) and position in the tile "
                                       "(rows: y, columns: x). Corrected amplitude A' = A + max(A - N, 0) (1/G - 1)."})
    np.save(out_dir / f"{name}_flatfield_noise_floor.npy", ff["N"])
    info.update(gain_map=str(path), noise_floor_db=(20 * np.log10(np.maximum(ff["N"], 1e-30))).tolist(),
                smooth_px=cfg.flatfield_smooth_px, max_gain_db=cfg.flatfield_max_gain_db)
    log(f"flat-field applied; gain map: {path}")
    return clims, info


def flatfield_existing(out_dir, name: str | None = None, smooth_px: float = 8.0, max_gain_db: float = 24.0,
                       log=print) -> dict:
    """Apply the tile flat-field to an existing reconstruction that kept its float volume
    (keep_float_volume=true): corrects the float volume in place, rewrites <name>.tiff and redoes the
    projections / overview."""
    from types import SimpleNamespace
    out_dir = Path(out_dir)
    if name is None:
        cands = [c for c in sorted(out_dir.glob("*_dB_float32.npy")) if not c.name.endswith("_fep_removed_dB_float32.npy")]
        if not cands:
            raise FileNotFoundError(f"no <name>_dB_float32.npy in {out_dir} (reconstruct with keep_float_volume=true)")
        name = cands[0].name[:-len("_dB_float32.npy")]
    sp = out_dir / f"{name}_run_summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    if (summ.get("flatfield") or {}).get("applied"):
        raise RuntimeError("flat-field was already applied to this volume")
    vol = np.load(out_dir / f"{name}_dB_float32.npy", mmap_mode="r+")
    meta = tiff_writer.read_meta(out_dir / f"{name}.tiff")
    md, acq = meta["metadata"], meta.get("acquisition")
    p = (acq or {}).get("patches", {})
    ty, tx = (p.get("patch_size_px") or [500, 500])[1], (p.get("patch_size_px") or [500, 500])[0]
    from .core.flatfield import tile_layout
    rows, cols = tile_layout(vol.shape[0], vol.shape[2], ty, tx)
    cfg = SimpleNamespace(tissue_smooth_um=30.0, tissue_threshold_db="auto", tissue_max_hole_mm2=0.5,
                          tissue_min_area_mm2=0.005, flatfield_smooth_px=smooth_px, flatfield_max_gain_db=max_gain_db)
    clims, info = flatfield_correct(vol, md, rows, cols, out_dir, name, cfg, log=log)
    vol.flush()
    if clims is not None:
        clim = (float(np.nanmin(clims[:, 0])), float(np.nanmax(clims[:, 1])))
        tiff_writer.write_legacy_tiff(out_dir / f"{name}.tiff", vol, clim, md, acquisition=acq)
        summ["clim_dB"] = clim
        log(f"rewrote {out_dir / (name + '.tiff')} (clim {clim[0]:.2f} .. {clim[1]:.2f} dB)")
    summ["flatfield"] = info
    sp.write_text(json.dumps(summ, indent=2, default=str))
    reproject(out_dir, name, log=log)
    return info
