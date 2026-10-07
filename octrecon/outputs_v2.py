"""v2 outputs written after the reconstruction: tissue-only xy projection, removed FEP signal
(volume + projection) and an overview figure. Called by Reconstructor.run()."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core.projection import all_z_projection, tissue_projection
from .io import tiff_writer
from .io.metadata import voxel_size_um


def write_2d_tiff(path: Path, img: np.ndarray, px_um: float, info: dict | None = None):
    """float32 2D TIFF with ImageJ calibration (um / pixel). NaN = no data."""
    import tifffile
    desc = json.dumps(info or {}, default=str)
    tifffile.imwrite(str(path), np.asarray(img, np.float32), imagej=True,
                     resolution=(1.0 / px_um, 1.0 / px_um),
                     metadata={"unit": "micron", "spacing": px_um, "Info": desc})


def _clim(img, lo=1.0, hi=99.7):
    fin = img[np.isfinite(img)]
    if not fin.size:
        return 0.0, 1.0
    a, b = np.percentile(fin, [lo, hi])
    return float(a), float(b) if b > a else float(a) + 1.0


def write_png(path: Path, img: np.ndarray, clim=None, cmap="gray"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = clim or _clim(img)
    plt.imsave(str(path), np.where(np.isfinite(img), img, lo), cmap=cmap, vmin=lo, vmax=hi)
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
        panels.append(("xy projection (tissue only, mean amplitude)", proj, "gray"))
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
        proj = P["mean"]
        files = {}
        for key in ("mean", "max"):
            tif = out_dir / f"{name}_xy_{key}.tif"
            write_2d_tiff(tif, P[key], px, dict(info, projection=key, units="dB", pixel_um=px))
            lo, hi = write_png(out_dir / f"{name}_xy_{key}.png", P[key])
            files[key] = {"tif": str(tif), "png": str(out_dir / f"{name}_xy_{key}.png"), "png_clim_dB": [lo, hi]}
        if "thickness_um" in P:
            tif = out_dir / f"{name}_tissue_thickness_um.tif"
            write_2d_tiff(tif, P["thickness_um"], px, {"units": "um", "pixel_um": px})
            write_png(out_dir / f"{name}_tissue_thickness_um.png", np.where(P["footprint"], P["thickness_um"], np.nan),
                      cmap="viridis")
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
        write_2d_tiff(tif2, rm_proj, px, {"units": "dB", "projection": "mean amplitude over all z", "pixel_um": px})
        write_png(out_dir / f"{name}_fep_removed_xy.png", rm_proj, cmap="magma")
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
