"""Output writers compatible with myOCT's yOCT2Tif / yOCTFromTif.

Legacy format (yOCT2Tif.m + yOCT2Tif_ConvertBitsData.m):
  * BigTIFF, one page per output y plane, page = (nZ rows, nX cols) uint16
  * PackBits compression
  * bits = uint16(round((dB - c1) / (c2 - c1) * 65534)) + 1, NaN -> 0
  * TIFF tag 305 "Software" = JSON {"metadata": dim-struct, "clim": [c1 c2], "version": 3}
    (yOCTFromTif reads it from the first page only)
  * calibration (buildTiffFrameTags): XResolution = 1/dx, YResolution = 1/dz [pixels/cm],
    ResolutionUnit = cm, ImageDescription "ImageJ=1.53 unit=um spacing=dy images=N" so
    Fiji/ImageJ shows the true voxel size (dx, dz, dy) -- here measured from the output axes.
  * additions (ignored by legacy readers): JSON keys "voxel_size_um" and "acquisition"
    (patches stitched in x/y, patch FOV/size, focus stack, raw parameters), and the same
    summary as ImageJ "Info" (Fiji: Image > Show Info).
  * legacy clim = [min, max] of all finite dB values in the volume

Legacy also quantises twice (per-plane clim, then re-quantised to the global
clim in partialFileMode 3). `legacy_double_quantization=True` reproduces that
for bit-level comparisons; default is a single (more accurate) quantisation.
"""
from __future__ import annotations

import json
from pathlib import Path

from .metadata import description_fields, imagej_info_text, voxel_size_um

import numpy as np
import tifffile

MAXBIT = 2 ** 16 - 1


def _matlab_uint16(v):
    """MATLAB uint16(): round half away from zero, saturate, NaN -> 0."""
    out = np.floor(np.abs(v) + 0.5) * np.sign(v)
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(out, 0, MAXBIT).astype(np.uint16)


def db_to_bits(db: np.ndarray, clim) -> np.ndarray:
    c1, c2 = sorted(clim)
    bits = _matlab_uint16((db - c1) / (c2 - c1) * (MAXBIT - 1)).astype(np.uint32) + 1
    bits = np.minimum(bits, MAXBIT).astype(np.uint16)
    bits[np.isnan(db)] = 0
    return bits


def bits_to_db(bits: np.ndarray, clim) -> np.ndarray:
    c1, c2 = sorted(clim)
    out = (bits.astype(np.float64) - 1) * (c2 - c1) / (MAXBIT - 1) + c1
    out[bits == 0] = np.nan
    return out


def plane_clim(db: np.ndarray):
    fin = db[np.isfinite(db)]
    return (float(fin.min()), float(fin.max())) if fin.size else (np.nan, np.nan)


def _tiff_tags(meta: dict, n_pages: int):
    """(resolution, description, extratags) for the calibrated legacy/ImageJ layout."""
    v = meta.get("voxel_size_um") or {}
    dx, dy, dz = v.get("x"), v.get("y"), v.get("z")
    if not dx or not dz:
        return (1.0, 1.0), f"ImageJ=1.53\nimages={n_pages}\nslices={n_pages}\nspacing=1.0\n", []
    res = (1.0 / (dx * 1e-4), 1.0 / (dz * 1e-4))          # pixels per cm (legacy)
    desc = (f"ImageJ=1.53\nimages={n_pages}\nslices={n_pages}\nunit=micron\n"
            f"spacing={dy:.10g}\nloop=false\n")
    for k, val in description_fields(meta):
        val = f"{val:.10g}" if isinstance(val, float) else str(val).replace("\n", " ").replace("=", ":")
        desc += f"{k}={val}\n"
    extratags = list(tifffile.imagej_metadata_tag({"Info": imagej_info_text(meta)}, "<"))
    return res, desc, extratags


def build_meta(metadata: dict, clim, acquisition: dict | None = None) -> dict:
    meta = {"metadata": metadata, "clim": [float(clim[0]), float(clim[1])], "version": 3,
            "voxel_size_um": voxel_size_um(metadata)}
    if acquisition:
        meta["acquisition"] = acquisition
    return meta


def write_legacy_tiff(path: str | Path, db_volume, clim, metadata: dict,
                      legacy_double_quantization: bool = False, plane_clims=None, progress=None,
                      acquisition: dict | None = None):
    """db_volume: array-like (nY, nZ, nX) float32 (may be a np.memmap)."""
    path = Path(path)
    meta = build_meta(metadata, clim, acquisition)
    n = db_volume.shape[0]

    def planes():
        for yi in range(n):
            plane = np.asarray(db_volume[yi], dtype=np.float64)
            if legacy_double_quantization:
                pc = plane_clims[yi] if plane_clims is not None else plane_clim(plane)
                plane = bits_to_db(db_to_bits(plane, pc), pc)
            if progress:
                progress(yi)
            yield db_to_bits(plane, clim)

    _write_pages(path, planes(), n, meta)
    return meta


def _write_pages(path: Path, pages, n_pages: int, meta: dict):
    res, desc, extratags = _tiff_tags(meta, n_pages)
    meta_json = json.dumps(meta, separators=(",", ":"))
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for yi, bits in enumerate(pages):
            first = yi == 0
            tw.write(bits, compression="packbits", photometric="minisblack",
                     resolution=res, resolutionunit="CENTIMETER",
                     description=desc if first else None, software=meta_json if first else None,
                     extratags=extratags if first else [], metadata=None, contiguous=False)
    with open(path.with_suffix(path.suffix + ".json"), "w") as f:
        json.dump(meta, f)


def read_meta(path: str | Path) -> dict:
    with tifffile.TiffFile(path) as t:
        tag = t.pages[0].tags.get(305) or t.pages[0].tags.get(270)
        return json.loads(tag.value)


def retag_tiff(src: str | Path, dst: str | Path | None = None, acquisition: dict | None = None, progress=None):
    """Rewrite an existing reconstruction TIFF with correct calibration/metadata.
    Pixel data are copied unchanged (verified page by page). dst=None rewrites src in place
    (via a temporary file). Returns the new metadata."""
    import os
    src = Path(src)
    old = read_meta(src)
    meta = build_meta(old["metadata"], old["clim"], acquisition or old.get("acquisition"))
    out = Path(dst) if dst else src.with_name(src.name + ".retag.tmp")
    with tifffile.TiffFile(src) as t:
        n = len(t.pages)

        def pages():
            for i in range(n):
                if progress:
                    progress(i, n)
                yield t.pages[i].asarray()
        _write_pages(out, pages(), n, meta)
    with tifffile.TiffFile(src) as a, tifffile.TiffFile(out) as b:       # verify pixels unchanged
        if len(a.pages) != len(b.pages):
            raise RuntimeError("page count changed while retagging")
        for i in range(len(a.pages)):
            if not np.array_equal(a.pages[i].asarray(), b.pages[i].asarray()):
                raise RuntimeError(f"pixel data changed on page {i} while retagging")
    if dst is None:
        os.replace(out, src)
        tmp_json = out.with_suffix(out.suffix + ".json")
        if tmp_json.exists():
            os.replace(tmp_json, src.with_suffix(src.suffix + ".json"))
    return meta
