"""Calibration (voxel size) and acquisition metadata for reconstruction outputs.

Voxel sizes are *measured* from the output axes (not assumed) and cross-checked against the
raw-data geometry in ScanInfo.json:

  TIFF page = one y plane, rows = z (depth), columns = x.  In ImageJ/Fiji terms:
      pixel width  = dx  (x, fast-scan / B-scan axis)
      pixel height = dz  (depth, after resampling to the output grid)
      voxel depth  = dy  (spacing between pages = slow-scan axis)

Legacy reference: LoadSave/yOCT2Tif.m buildTiffFrameTags (XResolution = 1/dx,
YResolution = 1/dz in pixels/cm; ImageDescription "ImageJ=1.53 unit=um spacing=dy images=N").
"""
from __future__ import annotations

import numpy as np


def _axis_step_um(axis: dict | None) -> float | None:
    """Mean spacing of a dimension-struct axis in micrometres (units mm or um)."""
    if not axis or "values" not in axis or len(axis["values"]) < 2:
        return None
    v = np.asarray(axis["values"], float)
    units = str(axis.get("units", "mm")).lower()
    f = 1e3 if units.startswith("mm") or "milli" in units else (1.0 if "um" in units or "micron" in units else 1e3)
    step = abs(v[-1] - v[0]) / (len(v) - 1) * f
    dev = np.max(np.abs(np.abs(np.diff(v)) * f - step)) if len(v) > 2 else 0.0
    if dev > 1e-3 * step + 1e-9:
        raise ValueError(f"non-uniform axis spacing (max deviation {dev:.3g} um from {step:.6g} um)")
    return float(round(step, 6))   # um, rounded to the picometre (removes float noise)


def voxel_size_um(metadata: dict) -> dict:
    """{'x','y','z'} voxel size in um measured from the output axes of a dim struct."""
    dx = _axis_step_um(metadata.get("x"))
    dz = _axis_step_um(metadata.get("z"))
    dy = _axis_step_um(metadata.get("y"))
    if dy is None:          # single y plane: legacy uses dx
        dy = dx
    return {"x": dx, "y": dy, "z": dz}


def acquisition_summary(si, hdr, z_um_native, n_medium: float, focus_pix, out_shape_yzx, voxel: dict) -> dict:
    """Raw-data / stitching facts derived from ScanInfo.json + tile header."""
    j = si.json
    nxc, nyc, nzd = len(si.x_centers_mm), len(si.y_centers_mm), len(si.z_depths_mm)

    def from_range(key, tile):
        r = j.get(key)
        return int(round((r[1] - r[0]) / tile)) if r and len(r) == 2 and tile else None

    def step(c, tile):
        return float(np.median(np.diff(np.sort(c)))) if len(c) > 1 else float(tile)

    sx, sy = step(si.x_centers_mm, si.tile_range_x_mm), step(si.y_centers_mm, si.tile_range_y_mm)
    tile_px_um = [si.tile_range_x_mm / si.n_x_px * 1e3, si.tile_range_y_mm / si.n_y_px * 1e3]
    zn = np.asarray(z_um_native, float)
    native_dz = float((zn[-1] - zn[0]) / (len(zn) - 1)) if len(zn) > 1 else None
    zd = np.sort(si.z_depths_mm)
    out = {
        "patches": {
            "n_x": nxc, "n_y": nyc, "n_total_xy": nxc * nyc,
            "n_x_from_range": from_range("xRange_mm", si.tile_range_x_mm),
            "n_y_from_range": from_range("yRange_mm", si.tile_range_y_mm),
            "n_focus_depths": nzd, "n_tiles_total": len(si.tiles),
            "patch_fov_mm": [si.tile_range_x_mm, si.tile_range_y_mm],
            "patch_size_px": [si.n_x_px, si.n_y_px],
            "patch_step_mm": [sx, sy],
            "patch_overlap_mm": [si.tile_range_x_mm - sx, si.tile_range_y_mm - sy],
            "x_centers_mm": si.x_centers_mm.tolist(), "y_centers_mm": si.y_centers_mm.tolist(),
            "scan_range_mm": {"x": j.get("xRange_mm"), "y": j.get("yRange_mm")},
        },
        "focus_stack": {
            "z_depths_mm": si.z_depths_mm.tolist(),
            "z_depth_step_um": float(np.median(np.diff(zd)) * 1e3) if nzd > 1 else None,
            "focus_position_px": [None if (f is None or np.isnan(f)) else float(f) for f in focus_pix],
        },
        "raw": {
            "oct_system": si.oct_system, "probe": (si.probe or {}).get("ObjectiveName"),
            "scan_pixel_size_um": si.pixel_size_um, "patch_pixel_size_um": tile_px_um,
            "spectral_pixels": hdr.n_lambda, "alines_per_raw_bscan": hdr.interf_size,
            "apodization_lines": hdr.apod_size, "bscan_avg": hdr.bscan_avg, "ascan_avg": hdr.ascan_avg,
            "spectral_binning": hdr.spectra_avg,
            "native_depth_pixel_um": native_dz, "native_depth_pixels": len(zn),
            "tissue_refractive_index": float(n_medium),
        },
        "output": {
            "shape_yzx": list(map(int, out_shape_yzx)),
            "voxel_size_um": voxel,
            "extent_mm": {"x": out_shape_yzx[2] * voxel["x"] / 1e3, "y": out_shape_yzx[0] * voxel["y"] / 1e3,
                          "z": out_shape_yzx[1] * voxel["z"] / 1e3},
            "axis_order_in_tiff": "pages = y (slow scan), rows = z (depth), columns = x (fast scan)",
        },
        "notes": [
            "Voxel sizes are measured from the output axes; x/y equal the raw scan pixel size, "
            "z is the output depth grid (native depth sampling resampled to the output pixel size).",
            "Depth (z) is in tissue units (optical path / refractive index) and depends on the "
            "spectrometer wavelength range used for this OCT system.",
        ],
    }
    checks = []
    if out["patches"]["n_x_from_range"] not in (None, nxc):
        checks.append(f"x: {nxc} patch centres but xRange/tile FOV gives {out['patches']['n_x_from_range']}")
    if out["patches"]["n_y_from_range"] not in (None, nyc):
        checks.append(f"y: {nyc} patch centres but yRange/tile FOV gives {out['patches']['n_y_from_range']}")
    for ax, i in (("x", 0), ("y", 1)):
        if voxel.get(ax) and abs(voxel[ax] - tile_px_um[i]) > 1e-3 * tile_px_um[i]:
            checks.append(f"{ax} voxel {voxel[ax]:.6g} um != patch FOV / patch px {tile_px_um[i]:.6g} um")
    exp_x = nxc * si.n_x_px if abs(sx - si.tile_range_x_mm) < 1e-9 else None
    if exp_x and exp_x != out_shape_yzx[2]:
        checks.append(f"x: {nxc} patches x {si.n_x_px} px = {exp_x} != output width {out_shape_yzx[2]}")
    # y: a full reconstruction has n_y * patch px planes; a row subset a whole number of patch rows
    if out_shape_yzx[0] % si.n_y_px or out_shape_yzx[0] > nyc * si.n_y_px:
        checks.append(f"y: {out_shape_yzx[0]} output planes is not a whole number (<= {nyc}) "
                      f"of {si.n_y_px}-px patch rows")
    out["patches"]["n_y_reconstructed"] = out_shape_yzx[0] // si.n_y_px
    out["consistency_checks"] = checks or ["ok"]
    return out


def description_fields(meta: dict) -> list[tuple[str, object]]:
    """key=value pairs appended to the ImageJ ImageDescription. Fiji lists them under
    Image > Show Info with both of its TIFF readers (ImageJ native and Bio-Formats, which is
    used for BigTIFF); ImageJ ignores keys it does not know."""
    a = meta.get("acquisition") or {}
    v = meta.get("voxel_size_um") or {}
    p, r, fs, o = a.get("patches", {}), a.get("raw", {}), a.get("focus_stack", {}), a.get("output", {})
    c1, c2 = meta.get("clim", [None, None])
    f = [("oct_voxel_x_um", v.get("x")), ("oct_voxel_z_um", v.get("z")), ("oct_voxel_y_um", v.get("y")),
         ("oct_axes", "pages_y_rows_z_columns_x"),
         ("oct_db_clim_min", c1), ("oct_db_clim_max", c2)]
    if p:
        fov, px, stp, ov = (p.get(k) or [None, None] for k in
                            ("patch_fov_mm", "patch_size_px", "patch_step_mm", "patch_overlap_mm"))
        f += [("oct_patches_x", p.get("n_x")), ("oct_patches_y", p.get("n_y")),
              ("oct_patches_y_reconstructed", p.get("n_y_reconstructed")),
              ("oct_focus_depths", p.get("n_focus_depths")), ("oct_raw_tiles", p.get("n_tiles_total")),
              ("oct_patch_fov_x_mm", fov[0]), ("oct_patch_fov_y_mm", fov[1]),
              ("oct_patch_px_x", px[0]), ("oct_patch_px_y", px[1]),
              ("oct_patch_step_x_mm", stp[0]), ("oct_patch_step_y_mm", stp[1]),
              ("oct_patch_overlap_x_mm", ov[0]), ("oct_patch_overlap_y_mm", ov[1]),
              ("oct_focus_depth_step_um", fs.get("z_depth_step_um")),
              ("oct_native_depth_pixel_um", r.get("native_depth_pixel_um")),
              ("oct_refractive_index", r.get("tissue_refractive_index")),
              ("oct_system", r.get("oct_system")), ("oct_probe", r.get("probe")),
              ("oct_consistency", "ok" if a.get("consistency_checks") == ["ok"] else "see_json")]
    return [(k, val) for k, val in f if val is not None]


def imagej_info_text(meta: dict) -> str:
    """Human-readable block shown by Fiji under Image > Show Info."""
    a = meta.get("acquisition") or {}
    v = meta.get("voxel_size_um") or {}
    p, r, fs, o = a.get("patches", {}), a.get("raw", {}), a.get("focus_stack", {}), a.get("output", {})
    c1, c2 = meta.get("clim", [None, None])

    def pair(xy):
        return f"{xy[0]} x {xy[1]}" if xy else "?"
    # one "key: value" fact per line and no '=' (Bio-Formats splits Info lines at '=')
    lines = [
        "OCT reconstruction: octrecon (port of myOCT yOCTProcessTiledScan)",
        f"Voxel size X, columns, fast scan [um]: {v.get('x')}",
        f"Voxel size Z, rows, depth [um]: {v.get('z')}",
        f"Voxel size Y, slice spacing, slow scan [um]: {v.get('y')}",
        "Axes: pages are y (slow scan), rows are z (depth), columns are x (fast scan)",
        f"Output shape y, z, x [voxels]: {o.get('shape_yzx')}",
        f"Intensity: dB is (value - 1) * (c2 - c1) / 65534 + c1 with c1 {c1}, c2 {c2}; value 0 means no data",
    ]
    if p:
        lines += [
            f"Patches stitched in x: {p.get('n_x')}",
            f"Patches stitched in y: {p.get('n_y')} (reconstructed: {p.get('n_y_reconstructed', p.get('n_y'))})",
            f"Focus depths per patch: {p.get('n_focus_depths')}",
            f"Raw tiles total: {p.get('n_tiles_total')}",
            f"Patch FOV x, y [mm]: {pair(p.get('patch_fov_mm'))}",
            f"Patch size x, y [px]: {pair(p.get('patch_size_px'))}",
            f"Patch step x, y [mm]: {pair(p.get('patch_step_mm'))}",
            f"Patch overlap x, y [mm]: {pair(p.get('patch_overlap_mm'))}",
            f"Focus depths [mm]: {fs.get('z_depths_mm')}",
            f"Focus pixel per depth: {fs.get('focus_position_px')}",
            f"OCT system / probe: {r.get('oct_system')} / {r.get('probe')}",
            f"Scan pixel size [um]: {r.get('scan_pixel_size_um')}",
            f"Native depth pixel in tissue [um]: {r.get('native_depth_pixel_um')} (n {r.get('tissue_refractive_index')})",
            f"Consistency checks: {'; '.join(a.get('consistency_checks', []))}",
        ]
    return "\n".join(str(x) for x in lines)
