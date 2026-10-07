"""Command line interface.

  python -m octrecon web [--port 8765] [--no-browser]
  python -m octrecon inspect  <OCTVolume>
  python -m octrecon reconstruct <OCTVolume> --output-root DIR --output-name NAME [--device auto|gpu|mps|cpu] [--config cfg.json|yaml] [--set key=value ...]
  python -m octrecon estimate-dispersion <OCTVolume> [--device ...]
  python -m octrecon detect-focus <OCTVolume> [--dispersion B] [--device ...]
  python -m octrecon extract <OCTVolume> [--delete-archives]      # legacy yOCTUnzipTiledScan
  python -m octrecon retag <reconstruction.tiff> [--volume <OCTVolume>] [--output new.tiff]   # fix calibration/metadata
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _val(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="octrecon", description="Tiled OCT reconstruction (GPU/CPU port of myOCT)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("web", help="start the web UI"); w.add_argument("--port", type=int, default=8765); w.add_argument("--no-browser", action="store_true")
    i = sub.add_parser("inspect", help="print scan summary and automatic parameters"); i.add_argument("volume")
    r = sub.add_parser("reconstruct", help="run a reconstruction")
    r.add_argument("volume"); r.add_argument("--output-root", default="outputs"); r.add_argument("--output-name")
    r.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu", "mps"]); r.add_argument("--config")
    r.add_argument("--rows", help="comma separated y-tile rows (0-based)")
    r.add_argument("--no-fep", action="store_true", help="disable FEP-film reflection removal (legacy-identical output)")
    r.add_argument("--no-projection", action="store_true", help="do not save the tissue-only xy projection")
    r.add_argument("--v1", action="store_true", help="v1 / legacy-identical reconstruction (= --no-fep --no-projection)")
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="any ReconConfig field, JSON values allowed")
    d = sub.add_parser("estimate-dispersion"); d.add_argument("volume"); d.add_argument("--device", default="auto")
    f = sub.add_parser("detect-focus"); f.add_argument("volume"); f.add_argument("--dispersion", type=float); f.add_argument("--device", default="auto")
    x = sub.add_parser("extract", help="extract .oct tile archives (legacy yOCTUnzipTiledScan)"); x.add_argument("volume"); x.add_argument("--delete-archives", action="store_true")
    pj = sub.add_parser("project", help="recompute the tissue-only xy projection from a kept float volume")
    pj.add_argument("output_dir"); pj.add_argument("--name"); pj.add_argument("--all-z", action="store_true")
    pj.add_argument("--smooth-um", type=float, default=30.0); pj.add_argument("--threshold-db", default="auto")
    pj.add_argument("--max-hole-mm2", type=float, default=0.5); pj.add_argument("--min-area-mm2", type=float, default=0.005)
    t = sub.add_parser("retag", help="rewrite calibration/metadata of an existing reconstruction TIFF")
    t.add_argument("tiff"); t.add_argument("--volume"); t.add_argument("--output")
    a = ap.parse_args(argv)

    if a.cmd == "web":
        from .webapp.server import serve
        return serve(port=a.port, open_browser=not a.no_browser)
    if a.cmd == "inspect":
        from .params import inspect_volume
        print(json.dumps(inspect_volume(a.volume), indent=2, default=str)); return
    if a.cmd == "reconstruct":
        from .pipeline import ReconConfig, Reconstructor
        base = {}
        if a.config:
            p = Path(a.config)
            base = (__import__("yaml").safe_load(p.read_text()) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text()))
            base = base.get("config", base)
        base.update(volume_folder=a.volume, output_root=a.output_root, device=a.device)
        base["output_name"] = a.output_name or base.get("output_name") or f"{Path(a.volume).resolve().parent.name}_recon"
        if a.rows:
            base["rows"] = [int(v) for v in a.rows.split(",")]
        if a.no_fep or a.v1:
            base["fep_removal"] = False
        if a.no_projection or a.v1:
            base["xy_projection"] = False
        for kv in a.set:
            k, v = kv.split("=", 1); base[k] = _val(v)
        s = Reconstructor(ReconConfig.from_dict(base), log=lambda m: print(m, flush=True)).run()
        print(json.dumps({k: s[k] for k in ("output_dir", "reconstruction_seconds", "clim_dB", "output_shape_yzx")}, indent=2, default=str)); return
    if a.cmd == "estimate-dispersion":
        from .backend import get_xp
        from .estimation.dispersion import estimate_dispersion
        r = estimate_dispersion(a.volume, device=get_xp(a.device)[1])
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, (bytes, bytearray))}, indent=2, default=str)); return
    if a.cmd == "detect-focus":
        from .backend import get_xp
        from .estimation.focus import detect_focus
        r = detect_focus(a.volume, dispersion_quadratic_term=a.dispersion, device=get_xp(a.device)[1])
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, (bytes, bytearray))}, indent=2, default=str)); return
    if a.cmd == "project":
        from .outputs_v2 import reproject
        thr = a.threshold_db if a.threshold_db == "auto" else float(a.threshold_db)
        r = reproject(a.output_dir, a.name, not a.all_z, a.smooth_um, thr, a.max_hole_mm2, a.min_area_mm2)
        print(json.dumps(r, indent=2, default=str)); return
    if a.cmd == "retag":
        from .retag import retag
        retag(a.tiff, a.volume, a.output); return
    if a.cmd == "extract":
        from .io.scaninfo import ScanInfo
        from .io.volume import extract_volume
        si = ScanInfo(a.volume)
        print(extract_volume(a.volume, [t.folder for t in si.tiles], a.delete_archives,
                             progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True))); return


if __name__ == "__main__":
    sys.exit(main())
