"""End-to-end tiled-scan reconstruction (Python/GPU port of myOCT yOCTProcessTiledScan.m).

Data flow (per y-tile row, i.e. nYPixelsInEachTile output y planes):
  for each of the nX*nZ tiles in the row (x-major, z-minor like legacy):
      read raw B-scans in batches (threaded prefetch, overlaps with compute)
      -> SpectralProcessor.magnitude      (apod, sinc5, window+dispersion, ifft, abs)
      -> TileStitcher.contribute          (optical path corr., focus weight, interp)
      -> accumulate numerator/denominator of the weighted mean on device
  finalise: weight < exp(-4.5) -> NaN, mean, 20*log10  -> float32 dB rows
After all rows: global clim -> uint16 BigTIFF identical in layout to legacy.

Output: <output_root>/<output_name>/
    <output_name>.tiff                 uint16 BigTIFF (legacy yOCT2Tif format)
    <output_name>.tiff.json            metadata + clim (dB = (v-1)*(c2-c1)/65534 + c1, 0 = NaN)
    <output_name>_dB_float32.npy       optional float32 dB volume (y, z, x)
    <output_name>_config.json          fully resolved parameters + their sources
    <output_name>_run_summary.json     timings, clim, device
  v2 (optional, see ReconConfig):
    <output_name>_xy_mean.tif / _xy_max.tif     tissue-only en-face projections, float32 dB (NaN = no tissue)
    <output_name>_xy_mean.png / _xy_max.png     8-bit previews
    <output_name>_tissue_thickness_um.tif       tissue slab thickness used for the projection
    <output_name>_fep_removed.tiff (+ .json)    removed FEP signal, uint16 volume (legacy format)
    <output_name>_fep_removed_xy.tif / .png     its mean projection over all z
    <output_name>_fep_overview.png              projection / removed signal / B-scans at a glance
    <output_name>.log                  run log
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import params as P
from .backend import get_xp, is_torch, synchronize, to_numpy
from .core.geometry import build_spectral_geometry, build_tiled_geometry
from .core.spectral import SpectralProcessor, average_repeats, binned_columns
from .core.stitching import OpticalPathCorrection, TileStitcher, min_weight_threshold
from .io import tiff_writer
from .io.detect import manufacturer_of
from .io.scaninfo import ScanInfo
from .io.volume import TileReader, extract_volume, scan_volume_layout

DATA_DIR = Path(__file__).parent / "data"


def _abs(xp, c, dtype):
    """|complex| for numpy / cupy arrays and torch tensors (MPS-safe via view_as_real)."""
    if hasattr(c, "detach"):
        import torch
        ri = torch.view_as_real(c)
        return torch.sqrt(ri[..., 0] * ri[..., 0] + ri[..., 1] * ri[..., 1])
    return xp.abs(c).astype(dtype, copy=False)


class Cancelled(Exception):
    pass


@dataclass
class ReconConfig:
    volume_folder: str
    output_root: str = "outputs"
    output_name: str = "reconstruction"
    # --- reconstruction parameters ("auto" = resolved from the scan / probe presets) ---
    dispersion_quadratic_term: object = "auto"   # auto | estimate | number  (nm^2/rad)
    focus_positions: object = "auto"             # auto | estimate | none | path.mat | number | list
    focus_sigma: object = "auto"                 # auto | number (pixels)
    crop_z_range_mm: object = "auto"             # auto | none | [zmin, zmax]
    output_pixel_size_um: object = "auto"        # auto | number (must equal scan pixel size)
    n_medium: object = "auto"                    # auto | number
    interp_method: str = "sinc5"
    apply_path_length_correction: bool = True
    raw_input: str = "auto"                      # auto (read .oct in place if not unzipped) | extract
    delete_archives_after_extract: bool = False
    # --- execution ---
    device: str = "auto"                         # auto | gpu | cpu
    precision: str = "float32"                   # float32 | float64
    gpu_fused_kernel: bool = True
    batch_frames: int = 50
    io_threads: int = 8
    prefetch_batches: int = 16
    rows: list | None = None                     # subset of y-tile rows (0-based); None = all
    write_tiff: bool = True
    keep_float_volume: bool = False
    legacy_double_quantization: bool = False
    # --- v2: removal of specular reflections from the FEP films (see octrecon/core/fep.py) ---
    fep_removal: bool = True
    fep_rank: int = 3
    fep_half_window: int = 8
    fep_search: int = 4
    fep_lateral_median: int = 15
    fep_frac_lo: float = 0.35
    fep_frac_hi: float = 0.60
    fep_detect_min_db: float = 6.0
    fep_max_surfaces: int = 4
    fep_shrink: bool = True
    fep_keep_level: float = 0.7
    fep_coherence_limit: bool = True           # cap removal at the laterally coherent (film) energy
    fep_coh_lag: int = 4
    fep_coh_window: int = 31
    fep_coh_margin: float = 1.0
    fep_bg_mode: str = "both"                  # both | max | inside
    fep_y_smooth: int = 5                      # B-scans averaged for the removal (avoids line artefacts)
    fep_learn_basis: bool = True
    fep_basis_file: str | None = None          # .npy (2R+1, rank) complex; overrides learning
    fep_save_removed: bool = True              # also save the removed (subtracted) signal volume + projection
    # --- v2: tissue-only xy (en-face) projection (see octrecon/core/projection.py) ---
    xy_projection: bool = True
    xy_projection_tissue_only: bool = True     # False: project over all z
    tissue_smooth_um: float = 30.0             # lateral smoothing for the tissue mask
    tissue_threshold_db: object = "auto"       # auto (Otsu) | number (dB)
    tissue_max_hole_mm2: float = 0.5           # enclosed non-tissue holes smaller than this are filled
    tissue_min_area_mm2: float = 0.005         # tissue specks smaller than this are dropped
    # --- v2: tile flat-field (vignetting) correction (see octrecon/core/flatfield.py) ---
    flatfield_correction: bool = True
    flatfield_smooth_px: float = 8.0
    flatfield_max_gain_db: float = 24.0

    @classmethod
    def from_dict(cls, d: dict):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_file(cls, path, **overrides):
        path = Path(path)
        txt = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            d = yaml.safe_load(txt)
        else:
            d = json.loads(txt)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(d)


class Reconstructor:
    def __init__(self, cfg: ReconConfig, log=print, progress=None, cancel: threading.Event | None = None):
        self.cfg = cfg
        self.log = log
        self.progress = progress or (lambda **kw: None)
        self.cancel = cancel or threading.Event()
        self.xp, self.device = get_xp(cfg.device)
        self.dtype = np.float32 if cfg.precision == "float32" else np.float64
        if is_torch(self.xp) and self.dtype != np.float32:
            self.log("note: the Apple GPU (MPS) supports float32 only -> precision set to float32")
            self.dtype = np.float32

        vol = Path(cfg.volume_folder)
        if not (vol / "ScanInfo.json").exists() and (vol / "OCTVolume" / "ScanInfo.json").exists():
            vol = vol / "OCTVolume"
        self.volume = vol
        self.si = si = ScanInfo(vol)
        folders = [t.folder for t in si.tiles]
        layout = scan_volume_layout(vol, folders)
        if layout["n_missing"]:
            raise FileNotFoundError(f"{layout['n_missing']} tile folders missing/unreadable, e.g. {layout['missing'][:3]}")
        if cfg.raw_input == "extract" and layout["counts"]["oct"]:
            self.log(f"extracting {layout['counts']['oct']} .oct tile archives (legacy yOCTUnzipTiledScan)")
            extract_volume(vol, folders, cfg.delete_archives_after_extract,
                           progress=lambda i, n: self.progress(stage="extract", done=i, total=n))
            layout = scan_volume_layout(vol, folders)
        self.layout = layout
        self._readers: dict[str, TileReader] = {}
        first = self.reader(folders[0])
        self.hdr = hdr = first.header
        # octSystem name -> manufacturer (yOCTLoadInterfFromFile.m l.104-117) must match the
        # raw files found (Thorlabs OCITY / SRR / Wasatch)
        man = manufacturer_of(si.oct_system)
        if man != hdr.manufacturer:
            raise ValueError(f"octSystem '{si.oct_system}' ({man}) does not match the raw files in "
                             f"{folders[0]} ({hdr.manufacturer})")
        # all tiles use the first tile's header + wavelengths (legacy dimOneTile)
        self.lambda_nm = first.lambda_nm(si.oct_system, DATA_DIR)
        if len(self.lambda_nm) != hdr.n_lambda:
            raise ValueError(f"chirp length {len(self.lambda_nm)} != spectral pixels {hdr.n_lambda}")
        if hdr.size_x != si.n_x_px:
            raise ValueError(f"tile header has {hdr.size_x} A-scans per B-scan but ScanInfo "
                             f"nXPixelsInEachTile={si.n_x_px} (legacy interp2 would fail)")
        n_lines = len(binned_columns(hdr.n_scan_lines, hdr.spectra_avg)) if hdr.spectra_avg > 1 else hdr.n_scan_lines
        if n_lines != hdr.size_x * max(1, hdr.ascan_avg):
            raise ValueError(f"{n_lines} A-lines per file after AScanBinning={hdr.spectra_avg} != "
                             f"SizeX {hdr.size_x} x AScanAvg {hdr.ascan_avg}")

        # ---- resolve every parameter (value + provenance) ----
        src = {}
        disp, src["dispersion_quadratic_term"] = P.resolve_dispersion(si, cfg.dispersion_quadratic_term)
        n_med, src["n_medium"] = P.resolve_n(si, cfg.n_medium)
        sigma, src["focus_sigma"] = P.resolve_focus_sigma(si, cfg.focus_sigma)
        crop, src["crop_z_range_mm"] = P.resolve_crop(si, cfg.crop_z_range_mm)
        px, src["output_pixel_size_um"] = P.resolve_pixel_size(si, cfg.output_pixel_size_um)
        if cfg.dispersion_quadratic_term == "estimate":
            from .estimation.dispersion import estimate_dispersion
            self.log("estimating dispersion automatically ...")
            est = estimate_dispersion(vol, device=self.device, initial=disp, log=self.log)
            disp, src["dispersion_quadratic_term"] = float(est["value"]), f"automatic estimate ({est.get('method', '')})"
        self.sg = build_spectral_geometry(self.lambda_nm, disp, n_med, cfg.interp_method)
        focus, src["focus_positions"] = P.resolve_focus(si, vol, cfg.focus_positions)
        if focus is None:
            from .estimation.focus import detect_focus
            self.log("detecting focus position automatically ...")
            det = detect_focus(vol, dispersion_quadratic_term=disp, device=self.device, log=self.log)
            focus = np.asarray(det["focus_positions"], float)
            src["focus_positions"] = f"automatic detection ({det.get('method', '')})"
        self.focus = focus
        self.resolved = {"dispersion_quadratic_term": disp, "n_medium": n_med, "focus_sigma": sigma,
                         "crop_z_range_mm": crop, "output_pixel_size_um": px,
                         "focus_positions": [None if np.isnan(f) else float(f) for f in focus],
                         "interp_method": cfg.interp_method}
        self.sources = src

        self.tg = build_tiled_geometry(si, self.sg.z_um, self.focus, px, crop)
        poly = si.optical_path_polynomial() if cfg.apply_path_length_correction else None
        self.oc = (OpticalPathCorrection.build(poly, self.tg.tile_x_mm, self.tg.tile_y_mm, self.tg.tile_z_mm)
                   if poly is not None else None)
        torch_backend = is_torch(self.xp)
        if torch_backend:   # Apple GPU (MPS) / torch-cpu: same maths on PyTorch tensors
            from .core.torch_backend import TorchSpectralProcessor, TorchTileStitcher
            self.sp = TorchSpectralProcessor(self.sg, hdr.apod_size, self.xp, ascan_binning=hdr.spectra_avg,
                                             apod_mode=hdr.apod_mode, apod_group=max(1, hdr.bscan_avg))
        else:
            self.sp = SpectralProcessor(self.sg, hdr.apod_size, xp=self.xp, dtype=self.dtype,
                                        use_gpu_kernel=cfg.gpu_fused_kernel,
                                        ascan_binning=hdr.spectra_avg, apod_mode=hdr.apod_mode,
                                        apod_group=max(1, hdr.bscan_avg))
        self.stitchers = {}
        for xi, xc in enumerate(si.x_centers_mm):
            for zi, zd in enumerate(si.z_depths_mm):
                st = TileStitcher(self.tg, self.oc, xc, zd, self.focus[zi], sigma,
                                  xp=np if torch_backend else self.xp, dtype=self.dtype)
                self.stitchers[(xi, zi)] = TorchTileStitcher(st, self.oc, self.xp) if torch_backend else st
        self.n_out = (len(self.tg.out_y_mm), len(self.tg.out_z_mm), len(self.tg.out_x_mm))
        self.prefetch = self._memory_safe_prefetch()
        self.fep = self._build_fep() if cfg.fep_removal else None
        self.timers = {"io_wait": 0.0, "h2d": 0.0, "spectral": 0.0, "stitch": 0.0, "finalise": 0.0}

    def _memory_safe_prefetch(self) -> int:
        """Limit read-ahead so buffered raw batches use <= 25 % of the currently free RAM."""
        want = max(1, int(self.cfg.prefetch_batches))
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            return want
        h = self.hdr
        batch_bytes = self.cfg.batch_frames * max(1, h.bscan_avg) * h.bytes_per_file
        fit = max(2, int(0.25 * avail / max(batch_bytes, 1)))
        if fit < want:
            self.log(f"note: only {avail / 1e9:.1f} GB RAM free -> read-ahead reduced from {want} to {fit} batches")
            return fit
        return want

    # ------------------------------------------------------------------ I/O
    def reader(self, folder: str) -> TileReader:
        r = self._readers.get(folder)
        if r is None:
            r = self._readers[folder] = TileReader(self.volume / folder)
        return r

    def _fep_pdz(self):
        """Fractional optical-path depth shift (rows) per (frame, A-line), or None."""
        if self.oc is None:
            return None
        return np.repeat(np.asarray(self.oc.pdz, np.float64), max(1, self.hdr.ascan_avg), axis=1)

    def _carrier(self):
        """Film phase per depth row / 2 pi: a reflector at depth m has phase -2 pi c m with
        c = k_0 / (N dk) for the equispaced k grid of the spectral processing."""
        k = 2 * np.pi / np.asarray(self.sg.lambda_eq_nm, np.float64)
        n = len(k)
        dk = abs(k[0] - k[-1]) / (n - 1)
        return float(max(k[0], k[-1]) / (n * dk))

    def _fep_shift(self):
        """Optical-path row shift per (frame, A-line); A-scan repeats share their position."""
        h = self.hdr
        if self.oc is None:
            sh = np.zeros((h.size_y, h.size_x), np.int64)
        else:
            sh = np.asarray(self.oc.shift, np.int64)
        return np.repeat(sh, max(1, h.ascan_avg), axis=1)

    def _scan_numpy(self, folder, frames):
        files = [f for fr_ in frames for f in self.frame_files(int(fr_))]
        h = self.hdr
        buf = np.empty((len(files), h.interf_size, h.n_lambda), np.dtype(h.raw_dtype))
        self.reader(folder).read_bscans(files, buf)
        return to_numpy(self.sp.scan(self.xp.asarray(buf)))

    def _build_fep(self):
        from .core.fep import FEPConfig, FEPRemover, learn_basis_from_volume
        cfg = self.cfg
        fc = FEPConfig(enabled=True, rank=cfg.fep_rank, half_window=cfg.fep_half_window, search=cfg.fep_search,
                       lateral_median=cfg.fep_lateral_median, frac_lo=cfg.fep_frac_lo, frac_hi=cfg.fep_frac_hi,
                       detect_min_db=cfg.fep_detect_min_db, max_surfaces=cfg.fep_max_surfaces,
                       shrink=cfg.fep_shrink, keep_level=cfg.fep_keep_level, learn_basis=cfg.fep_learn_basis,
                       bg_mode=cfg.fep_bg_mode, coherence_limit=cfg.fep_coherence_limit, coh_lag=cfg.fep_coh_lag,
                       coh_window=cfg.fep_coh_window, coh_margin=cfg.fep_coh_margin,
                       y_smooth=cfg.fep_y_smooth)
        shift = self._fep_shift()
        if cfg.fep_basis_file:
            U = np.load(cfg.fep_basis_file)
            info = {"source": f"file {cfg.fep_basis_file}"}
        else:
            si = self.si
            zi0 = int(np.argmin(np.abs(si.z_depths_mm)))          # depth with the surface at focus
            pick = []
            for xi in np.unique(np.linspace(0, len(si.x_centers_mm) - 1, 4).round().astype(int)):
                for yi in np.unique(np.linspace(0, len(si.y_centers_mm) - 1, 3).round().astype(int)):
                    t = next((t for t in si.tiles if (t.xi, t.yi, t.zi) == (xi, yi, zi0)), None)
                    if t is not None:
                        pick.append((t.folder, t.zi))

            def rows_for_zi(zi):
                st = next(v for (x_, z_), v in self.stitchers.items() if z_ == zi and not v.empty)
                return st.r0, st.r1
            U, info = learn_basis_from_volume(lambda f, fr: self._scan_numpy(f, fr), pick, shift,
                                              self.sg.window, rows_for_zi, fc, log=self.log)
        self.fep_basis_info = info
        self.log(f"FEP removal on: rank {fc.rank}, +-{fc.half_window} samples, basis from {info.get('source')}")
        return FEPRemover(U, shift, fc, xp=self.xp, pdz=self._fep_pdz(), carrier=self._carrier())

    def frame_files(self, frame: int):
        """Raw file indices holding tile-local y frame `frame` (B-scan repeats consecutive)."""
        n = max(1, self.hdr.bscan_avg)
        return [frame * n + a for a in range(n)]

    # ------------------------------------------------------------------ core
    def process_row(self, yi: int, frames=None, pool=None, on_batch=None, return_removed=False):
        """Reconstruct output planes of y-tile row `yi`.
        frames: tile-local frame indices (0-based) to produce; default all.
        Returns float32 dB array (nFrames, nOutZ, nOutX) on host; with return_removed (and FEP
        removal on) also the removed signal |s - s'| stitched with the same weights (dB)."""
        xp, cfg, hdr = self.xp, self.cfg, self.hdr
        frames = np.arange(self.si.n_y_px) if frames is None else np.asarray(frames)
        nF = len(frames)
        _, nz_out, nx_out = self.n_out
        num = xp.zeros((nF, nz_out, nx_out), self.dtype)
        den = xp.zeros((nF, nz_out, nx_out), self.dtype)
        want_rm = bool(return_removed and self.fep is not None)
        num_rm = xp.zeros((nF, nz_out, nx_out), self.dtype) if want_rm else None

        tiles = self.si.tiles_in_row(yi)
        B = cfg.batch_frames
        jobs = []
        for t in tiles:
            if self.stitchers[(t.xi, t.zi)].empty:
                continue
            for s in range(0, nF, B):
                jobs.append((t, s, frames[s:s + B]))

        def read_job(job):
            t, s, fr = job
            files = [f for fr_ in fr for f in self.frame_files(int(fr_))]
            buf = np.empty((len(files), hdr.interf_size, hdr.n_lambda), np.dtype(hdr.raw_dtype))
            self.reader(t.folder).read_bscans(files, buf)
            return buf

        own_pool = pool is None
        pool = pool or ThreadPoolExecutor(cfg.io_threads)
        pending = deque()
        it = iter(jobs)
        for _ in range(self.prefetch):
            j = next(it, None)
            if j is None:
                break
            pending.append((j, pool.submit(read_job, j)))
        n_done = 0
        try:
            while pending:
                if self.cancel.is_set():
                    raise Cancelled()
                job, fut = pending.popleft()
                t0 = time.perf_counter()
                raw = fut.result()
                t1 = time.perf_counter()
                self.timers["io_wait"] += t1 - t0
                nxt = next(it, None)
                if nxt is not None:
                    pending.append((nxt, pool.submit(read_job, nxt)))

                t, s, fr = job
                st = self.stitchers[(t.xi, t.zi)]
                raw_d = xp.asarray(raw)
                synchronize(xp); t2 = time.perf_counter(); self.timers["h2d"] += t2 - t1
                if self.fep is not None:
                    c = self.sp.scan(raw_d)
                    c_in = c.clone() if (want_rm and hasattr(c, "clone")) else (c.copy() if want_rm else None)
                    fr_scans = [int(f_) for f_ in fr for _ in range(max(1, hdr.bscan_avg))]
                    c = self.fep.process(c, fr_scans, st.r0, st.r1)
                    mag = _abs(xp, c, self.dtype)
                    if want_rm:
                        rm = average_repeats(_abs(xp, c_in - c, self.dtype), len(fr), hdr)
                        del c_in
                else:
                    mag = self.sp.magnitude(raw_d)
                # legacy yOCTProcessTiledScan.m l.288-291: abs, then mean over the trailing
                # dims of (z, x, AScanAvg, BScanAvg): B-scan repeats first, then A-scan repeats
                mag = average_repeats(mag, len(fr), hdr)
                synchronize(xp); t3 = time.perf_counter(); self.timers["spectral"] += t3 - t2
                n_, d_ = st.contribute(mag, xp.asarray(fr))
                num[s:s + len(fr), :, st.c0:st.c1] += n_
                den[s:s + len(fr), :, st.c0:st.c1] += d_
                if want_rm:
                    num_rm[s:s + len(fr), :, st.c0:st.c1] += st.contribute(rm, xp.asarray(fr))[0]
                synchronize(xp); self.timers["stitch"] += time.perf_counter() - t3
                n_done += 1
                if on_batch:
                    on_batch(n_done, len(jobs))
        finally:
            if own_pool:
                pool.shutdown(wait=True, cancel_futures=True)
            else:
                for _, f in pending:
                    f.cancel()

        t0 = time.perf_counter()
        den[den < min_weight_threshold(self.dtype)] = xp.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20 * xp.log10(num / den)
            db_rm = 20 * xp.log10(num_rm / den) if want_rm else None
        out = to_numpy(db).astype(np.float32, copy=False)
        self.timers["finalise"] += time.perf_counter() - t0
        if return_removed:
            return out, (to_numpy(db_rm).astype(np.float32, copy=False) if want_rm else None)
        return out

    def tile_blocks(self, n_rows_done: int):
        """Output (row, column) ranges of the tiles: one block of n_y_px rows per reconstructed y-tile
        row, and per x-tile the output columns inside [x_centre +- tile range / 2]."""
        ny = self.si.n_y_px
        rows = [(k * ny, (k + 1) * ny) for k in range(n_rows_done)]
        x = self.tg.out_x_mm
        h = self.si.tile_range_x_mm / 2
        cols = []
        for xc in self.si.x_centers_mm:
            c0 = int(np.searchsorted(x, xc - h - 1e-9))
            c1 = int(np.searchsorted(x, xc + h - 1e-9))
            if c1 > c0:
                cols.append((c0, c1))
        return rows, cols

    def global_y_to_row(self, y_index0: int):
        return divmod(int(y_index0), self.si.n_y_px)

    def process_planes(self, y_indices0):
        """Reconstruct arbitrary output y planes (0-based). Returns (n, nZ, nX)."""
        out, by_row = {}, {}
        for y in y_indices0:
            r, f = self.global_y_to_row(y)
            by_row.setdefault(r, []).append((y, f))
        for r, lst in by_row.items():
            planes = self.process_row(r, [f for _, f in lst])
            for (y, _), p in zip(lst, planes):
                out[y] = p
        return np.stack([out[y] for y in y_indices0])

    # --------------------------------------------------------------- full run
    def metadata(self, y_sel=None):
        tg = self.tg
        yv = tg.out_y_mm if y_sel is None else tg.out_y_mm[y_sel]

        def axis(values, order, units, origin, extra=None):
            d = {"order": order, "values": list(map(float, values)), "units": units,
                 "index": list(range(1, len(values) + 1)), "origin": origin}
            if extra:
                d.update(extra)
            return d
        return {
            "lambda": {"order": 1, "values": list(map(float, self.lambda_nm * 1e-6)), "units": "mm"},
            "z": axis(tg.out_z_mm, 1, "mm", "z=0 is tissue interface as specified by user"),
            "x": axis(tg.out_x_mm, 2, "mm", "x=0 is OCT scanner origin (under objective's principal) when xCenters=0 scan was taken"),
            "y": axis(yv, 3, "mm", "y=0 is OCT scanner origin (under objective's principal) when yCenters=0 scan was taken",
                      {"indexMax": self.si.n_y_px}),
            "aux": {"interfSize": self.hdr.interf_size, "apodSize": self.hdr.apod_size,
                    "AScanBinning": self.hdr.spectra_avg, "octSystem": self.si.oct_system},
            "scanZDepths_mm": list(map(float, self.si.z_depths_mm)),
        }

    def acquisition(self, md: dict, shape) -> dict:
        """Patches / raw-data / calibration summary written into the TIFF and JSON outputs."""
        from .io.metadata import acquisition_summary, voxel_size_um
        return acquisition_summary(self.si, self.hdr, self.sg.z_um, self.resolved["n_medium"],
                                   self.focus, shape, voxel_size_um(md))

    def run(self):
        cfg, log = self.cfg, self.log
        out_dir = Path(cfg.output_root) / cfg.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        name = cfg.output_name
        n_rows_total = len(self.si.y_centers_mm)
        rows = list(range(n_rows_total)) if not cfg.rows else sorted(int(r) for r in cfg.rows)
        ny = self.si.n_y_px
        y_sel = np.concatenate([np.arange(r * ny, (r + 1) * ny) for r in rows])
        shape = (len(y_sel), self.n_out[1], self.n_out[2])
        with open(out_dir / f"{name}_config.json", "w") as f:
            json.dump({"config": asdict(cfg), "resolved": self.resolved, "sources": self.sources,
                       "device": self.device, "volume_folder": str(self.volume),
                       "storage": self.layout}, f, indent=2, default=str)
        log(f"device={self.device}  output (y,z,x)={shape}  rows={len(rows)}/{n_rows_total}  -> {out_dir}")
        for k, v in self.resolved.items():
            log(f"  {k} = {v}   [{self.sources.get(k, 'config')}]")

        npy = out_dir / f"{name}_dB_float32.npy"
        vol = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=shape)
        plane_clims = np.full((shape[0], 2), np.nan)
        save_rm = bool(cfg.fep_save_removed and self.fep is not None)
        npy_rm = out_dir / f"{name}_fep_removed_dB_float32.npy"
        vol_rm = np.lib.format.open_memmap(npy_rm, mode="w+", dtype=np.float32, shape=shape) if save_rm else None
        rm_clims = np.full((shape[0], 2), np.nan)
        t_start = time.perf_counter()
        try:
            with ThreadPoolExecutor(cfg.io_threads) as pool:
                for k, r in enumerate(rows):
                    tr = time.perf_counter()

                    def on_batch(i, n, k=k):
                        el = time.perf_counter() - t_start
                        frac = (k + i / n) / len(rows)
                        self.progress(stage="reconstruct", done=round(frac * 100, 2), total=100,
                                      row=k + 1, rows=len(rows), elapsed_s=el,
                                      eta_s=el / frac * (1 - frac) if frac > 0 else None)

                    db, db_rm = self.process_row(r, pool=pool, on_batch=on_batch, return_removed=True)
                    vol[k * ny:(k + 1) * ny] = db
                    if save_rm:
                        vol_rm[k * ny:(k + 1) * ny] = db_rm
                        for i, p in enumerate(db_rm):
                            rm_clims[k * ny + i] = tiff_writer.plane_clim(p)
                    for i, p in enumerate(db):
                        plane_clims[k * ny + i] = tiff_writer.plane_clim(p)
                    el = time.perf_counter() - t_start
                    log(f"[{self.device}] row {k + 1}/{len(rows)} (y-row {r}) done in {time.perf_counter() - tr:.1f}s "
                        f"(elapsed {el / 60:.1f} min, ETA {el / (k + 1) * (len(rows) - k - 1) / 60:.1f} min)")
        finally:
            for r_ in self._readers.values():
                r_.close()
        vol.flush()
        if save_rm:
            vol_rm.flush()
        recon_s = time.perf_counter() - t_start

        md = self.metadata(y_sel)
        acq = self.acquisition(md, shape)
        ff_info = None
        if cfg.flatfield_correction:
            from .outputs_v2 import flatfield_correct
            t0 = time.perf_counter()
            ff_clims, ff_info = flatfield_correct(vol, md, *self.tile_blocks(len(rows)), out_dir, name, cfg,
                                                  log=log, progress=self.progress)
            if ff_clims is not None:
                plane_clims = ff_clims
                vol.flush()
            ff_info["seconds"] = time.perf_counter() - t0
        clim = (float(np.nanmin(plane_clims[:, 0])), float(np.nanmax(plane_clims[:, 1])))
        summary = {"device": self.device, "rows": rows, "output_dir": str(out_dir),
                   "fep_removal": (dict(self.fep.summary(), basis=getattr(self, "fep_basis_info", None))
                                   if self.fep is not None else None),
                   "voxel_size_um": acq["output"]["voxel_size_um"], "acquisition": acq,
                   "output_shape_yzx": list(shape), "clim_dB": clim,
                   "reconstruction_seconds": recon_s, "timers": self.timers, "flatfield": ff_info,
                   "resolved": self.resolved, "sources": self.sources}
        if cfg.write_tiff:
            self.progress(stage="write_tiff", done=0, total=shape[0])
            t0 = time.perf_counter()
            tif = out_dir / f"{name}.tiff"
            tiff_writer.write_legacy_tiff(tif, vol, clim, md,
                                          cfg.legacy_double_quantization,
                                          plane_clims if cfg.legacy_double_quantization else None,
                                          acquisition=acq,
                                          progress=lambda i: self.progress(stage="write_tiff", done=i + 1, total=shape[0])
                                          if i % 100 == 0 or i == shape[0] - 1 else None)
            summary["tiff_seconds"] = time.perf_counter() - t0
            summary["tiff"] = str(tif)
            log(f"wrote {tif} in {summary['tiff_seconds']:.0f}s")
        if cfg.xy_projection or save_rm:
            from .outputs_v2 import write_v2_outputs
            t0 = time.perf_counter()
            summary.update(write_v2_outputs(self, out_dir, name, vol, vol_rm, clim, rm_clims, md, acq))
            summary["v2_outputs_seconds"] = time.perf_counter() - t0
        del vol
        if not cfg.keep_float_volume:
            npy.unlink()
        else:
            summary["float_volume"] = str(npy)
        if save_rm:
            del vol_rm
            if not cfg.keep_float_volume:
                npy_rm.unlink()
            else:
                summary["fep_removed_float_volume"] = str(npy_rm)
        with open(out_dir / f"{name}_run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.progress(stage="done", done=1, total=1)
        return summary
