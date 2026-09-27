"""Fix the calibration / metadata of existing reconstruction TIFFs without re-reconstructing.

    python -m octrecon retag <reconstruction.tiff> [--volume <OCTVolume>] [--output new.tiff]

Writes the correct voxel size (measured from the output axes stored in the TIFF), the
ImageJ calibration and, when the raw volume is given (or found via <name>_config.json),
the acquisition summary (patches in x/y, patch FOV/size, focus stack, raw parameters).
Pixel values are copied unchanged and verified page by page.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import params as P
from .core.geometry import build_spectral_geometry
from .io.metadata import acquisition_summary, voxel_size_um
from .io.scaninfo import ScanInfo
from .io.tiff_writer import read_meta, retag_tiff
from .io.volume import TileReader

DATA_DIR = Path(__file__).parent / "data"


def acquisition_for_volume(volume, metadata: dict, shape_yzx, n_medium=None, focus=None) -> dict:
    vol = Path(volume)
    if not (vol / "ScanInfo.json").exists() and (vol / "OCTVolume" / "ScanInfo.json").exists():
        vol = vol / "OCTVolume"
    si = ScanInfo(vol)
    tr = TileReader(vol / si.tiles[0].folder)
    try:
        hdr = tr.header
        lam = tr.lambda_nm(si.oct_system, DATA_DIR)
    finally:
        tr.close()
    n = float(n_medium) if n_medium is not None else P.resolve_n(si)[0]
    z_um = build_spectral_geometry(lam, 1.0, n, "sinc5").z_um       # depth axis does not depend on dispersion
    if focus is None:
        f, _ = P.resolve_focus(si, vol, "auto")
        focus = f if f is not None else np.full(len(si.z_depths_mm), np.nan)
    focus = [np.nan if v is None else float(v) for v in np.atleast_1d(focus)]
    return acquisition_summary(si, hdr, z_um, n, focus, shape_yzx, voxel_size_um(metadata))


def retag(tiff, volume=None, output=None, log=print):
    tiff = Path(tiff)
    old = read_meta(tiff)
    md = old["metadata"]
    shape = [len(md["y"]["values"]), len(md["z"]["values"]), len(md["x"]["values"])]
    n_medium = focus = None
    cfg_path = tiff.with_name(tiff.stem + "_config.json")
    if cfg_path.exists():                       # the run's own resolved parameters
        cfg = json.loads(cfg_path.read_text())
        res = cfg.get("resolved", {})
        n_medium, focus = res.get("n_medium"), res.get("focus_positions")
        volume = volume or cfg.get("volume_folder")
    acq = None
    if volume:
        acq = acquisition_for_volume(volume, md, shape, n_medium, focus)
        log(f"acquisition metadata from {volume}: patches {acq['patches']['n_x']} x {acq['patches']['n_y']}, "
            f"checks: {'; '.join(acq['consistency_checks'])}")
    else:
        log("no raw volume given/found: writing calibration only (no patch information)")
    meta = retag_tiff(tiff, output, acquisition=acq,
                      progress=lambda i, n: log(f"  page {i}/{n}") if i % 1000 == 0 else None)
    log(f"voxel size [um]: {meta['voxel_size_um']}  -> {output or tiff}")
    return meta
