"""Engines behind the two interactive legacy tools, ported for the web app.

1. Manual dispersion tool -- Demo_DispersionCorrectionManual.m
   * one B-scan: Y frame `frame` (legacy: 1) and the first B-scan repeat of a chosen DataX
     folder, apodization-subtracted (yOCTLoadInterfFromFile)
   * k-linearised ONCE with pchip (yOCTEquispaceInterf default), so the slider only changes
     the dispersion phase
   * slider value v in [-10, 10] (log10 units, initial log10(100)); beta = sign(v) * 10^|v|
   * shown: ln|ifft(interf_eq * hann/rms * exp(-i*beta*(k-mean k)^2))|[first N/2 bins],
     gray colormap, caxis [-5, 6], title "dispersionQuadraticTerm=..."

2. "Choose Focus Positions" window -- yOCTMeasureFocusDrift.m (measureFocus & helpers)
   * geometry from yOCTProcessTiledScan_createDimStructure(volume, NaN): tile z from the
     zero-delay (no focus shift), tile x/y in mm
   * depths offered: selectDepthsToMeasure (first 50 um inside tissue, then every 50 um;
     shallow scans fall back to z >= 0)
   * start: central X tile, central Y tile row, B-scan round(nY/2); X tile and B-scan carry
     over between depths
   * B-scan: reconstructCenterBScan = loader + yOCTInterfToScanCpx with only the dispersion
     term (=> pchip equispacing) + |.| + mean over repeats + yOCTOpticalPathCorrection;
     displayed as mag2db(|scan| + eps), base range 5th..99.8th percentile, brightness /
     contrast sliders mapped exactly as onAdjustDisplay
   * click -> nearest z pixel (yellow line); blue dashed guide = robust drift fit of the
     accepted clicks (yOCTMeasureFocusDrift_fitDrift); live drift readout
   * finish: fit for every z-depth, save zChosenFocusPositions.mat (+ .png drift figure)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..core.geometry import build_tiled_geometry, matlab_linspace
from ..core.spectral import legacy_interferogram
from ..core.stitching import OpticalPathCorrection
from ..io.scaninfo import ScanInfo
from ..io.volume import TileReader
from .drift_fit import fit_drift
from .focus import select_depths_to_measure

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DISPERSION_CLIM = (-5.0, 6.0)          # Demo_DispersionCorrectionManual: caxis([-5 6])


def resolve_volume(volume) -> Path:
    v = Path(volume)
    if not (v / "ScanInfo.json").exists() and (v / "OCTVolume" / "ScanInfo.json").exists():
        v = v / "OCTVolume"
    if not (v / "ScanInfo.json").exists():
        raise FileNotFoundError(f"No ScanInfo.json in {v}")
    return v


# ------------------------------------------------------------------ shared maths
def pchip_equispace(interf: np.ndarray, lambda_nm: np.ndarray):
    """yOCTEquispaceInterf(interf, dim) with its default 'pchip'.
    interf: (nLines, N) spectra (lambda along the last axis). Returns (interf_eq, lambda_eq)."""
    lam = np.asarray(lambda_nm, float)
    k = 2 * np.pi / lam
    k_lin = matlab_linspace(k.max(), k.min(), len(k))
    order = np.argsort(k)
    f = PchipInterpolator(k[order], np.asarray(interf, float)[..., order], axis=-1, extrapolate=True)
    lam_eq = PchipInterpolator(k[order], lam[order], extrapolate=True)(k_lin)
    return f(k_lin), lam_eq


def _is_equispaced(lam):
    k = 2 * np.pi / np.asarray(lam, float)
    dk = np.diff(k)
    return abs((dk.max() - dk.min()) / k.max()) <= 1e-10


def scan_window(lambda_eq_nm: np.ndarray, beta: float) -> np.ndarray:
    """yOCTInterfToScanCpx filter on an already-equispaced axis: hann/rms * exp(-i beta (k-k0)^2)."""
    N = len(lambda_eq_nm)
    k = 2 * np.pi / np.asarray(lambda_eq_nm, float)
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    hann = hann / np.sqrt(np.mean(hann ** 2))
    return hann * np.exp(1j * (-float(beta) * (k - k.mean()) ** 2))


def scan_cpx(interf_eq: np.ndarray, lambda_eq_nm: np.ndarray, beta: float) -> np.ndarray:
    """(nLines, N) equispaced -> complex scan (nLines, N/2) (MATLAB ifft incl. 1/N)."""
    ft = np.fft.ifft(interf_eq * scan_window(lambda_eq_nm, beta), axis=-1)
    return ft[..., : interf_eq.shape[-1] // 2]


def _equispace_like_legacy(interf, lambda_nm):
    """yOCTInterfToScanCpx: equispace (pchip, default interpMethod) unless already equispaced;
    the returned lambda is then tested again exactly like the legacy code would."""
    if _is_equispaced(lambda_nm):
        return np.asarray(interf, float), np.asarray(lambda_nm, float)
    return pchip_equispace(interf, lambda_nm)


class VolumeInfo:
    """Small, cached description of a volume for the tools (tiles, positions, geometry)."""

    def __init__(self, volume):
        self.volume = resolve_volume(volume)
        self.si = si = ScanInfo(self.volume)
        r = TileReader(self.volume / si.tiles[0].folder)
        try:
            self.hdr = r.header
            self.lambda_nm = r.lambda_nm(si.oct_system, DATA_DIR)
        finally:
            r.close()
        self.by_pos = {(t.xi, t.yi, t.zi): t for t in si.tiles}
        self.by_folder = {t.folder: t for t in si.tiles}

    def tile_list(self):
        return [{"folder": t.folder, "xi": t.xi, "yi": t.yi, "zi": t.zi, "x_mm": t.x_center_mm,
                 "y_mm": t.y_center_mm, "z_mm": t.z_depth_mm} for t in self.si.tiles]

    def summary(self):
        si = self.si
        return {"volume": str(self.volume), "x_centers_mm": si.x_centers_mm.tolist(),
                "y_centers_mm": si.y_centers_mm.tolist(), "z_depths_mm": si.z_depths_mm.tolist(),
                "n_frames": int(self.hdr.size_y), "n_lines": int(self.hdr.size_x), "tiles": self.tile_list()}

    def read_frame_interferogram(self, folder: str, frame0: int, repeat0: int = 0):
        """Apodization-corrected interferogram (nLines, N) of tile-local y frame (0-based)."""
        h = self.hdr
        r = TileReader(self.volume / folder)
        try:
            nb = max(1, h.bscan_avg)
            fi = frame0 * nb + repeat0
            buf = np.empty((1, h.interf_size, h.n_lambda), np.dtype(h.raw_dtype))
            r.read_bscans([fi], buf)
        finally:
            r.close()
        return legacy_interferogram(buf, h.apod_size, h.spectra_avg, h.apod_mode, 1, np, np.float64)[0]


# ------------------------------------------------------------------ 1. dispersion tool
def slider_to_beta(v: float) -> float:
    """Demo_DispersionCorrectionManual SliderCallback: beta = sign(v) * 10^|v|."""
    return float(np.sign(v) * 10 ** abs(v))


def beta_to_slider(beta: float) -> float:
    return float(np.sign(beta) * np.log10(abs(beta))) if beta else 0.0


class DispersionTool:
    """One loaded B-scan; render(beta) recomputes only the dispersion phase (as legacy)."""

    def __init__(self, vi: VolumeInfo, folder: str, frame0: int = 0, legacy_axis: bool = False):
        """legacy_axis=True reproduces the MATLAB demo exactly: it calls the loader without
        octSystem, so the system is auto-detected from Header.xml (a GAN632 header reads as
        'Ganymede' -> lambda range 796.23-1010.02 nm instead of 796-1010 nm). Default False
        uses the scan's octSystem, i.e. the same wavelength axis as the reconstruction."""
        self.vi, self.folder, self.frame0 = vi, folder, int(frame0)
        self.legacy_axis = bool(legacy_axis)
        lam = vi.lambda_nm
        if self.legacy_axis:
            from ..io.thorlabs import detect_thorlabs_system
            r = TileReader(vi.volume / folder)
            try:
                sysname = detect_thorlabs_system(r.read_bytes("Header.xml"))
                lam = r.lambda_nm(sysname, DATA_DIR)
            finally:
                r.close()
            self.system_used = sysname
        else:
            self.system_used = vi.si.oct_system
        interf = vi.read_frame_interferogram(folder, self.frame0, 0)       # (nLines, N)
        if vi.hdr.ascan_avg > 1:   # keep all lines like legacy squeeze; average repeats for display
            n = vi.hdr.size_x
            interf = interf.reshape(n, vi.hdr.ascan_avg, -1).mean(axis=1)
        ie, le = pchip_equispace(interf, lam)                       # yOCTEquispaceInterf (demo)
        self.interf_eq, self.lambda_eq = _equispace_like_legacy(ie, le)   # check inside yOCTInterfToScanCpx

    def ln_image(self, beta: float) -> np.ndarray:
        """ln|scan| as (N/2 depth, nLines) like imagesc(lg) in the legacy figure."""
        a = np.abs(scan_cpx(self.interf_eq, self.lambda_eq, beta))
        with np.errstate(divide="ignore"):
            return np.log(a).T

    @staticmethod
    def sharpness(lg: np.ndarray) -> float:
        """Guide value (not in legacy): mean of the brightest 1 % of |scan|^2, normalised."""
        a = np.exp(lg.astype(np.float64))
        p = a ** 2
        top = np.sort(p.ravel())[-max(1, p.size // 100):]
        return float(top.mean() / p.mean())


def matlab_prctile(a, q):
    """MATLAB prctile (midpoint definition) == numpy 'hazen'."""
    return np.percentile(np.asarray(a).ravel(), q, method="hazen")


def gray_png(img: np.ndarray, lo: float, hi: float) -> bytes:
    """Pixel-exact grayscale PNG (one image pixel per data value) with fixed limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.clip((np.nan_to_num(img, nan=lo, neginf=lo) - lo) / max(hi - lo, 1e-12), 0, 1)
    buf = io.BytesIO()
    plt.imsave(buf, g, cmap="gray", vmin=0, vmax=1, format="png")
    return buf.getvalue()


# ------------------------------------------------------------------ 2. focus tool
class FocusTool:
    """State-free helpers for the 'Choose Focus Positions' window (the browser keeps state)."""

    def __init__(self, vi: VolumeInfo):
        self.vi = vi
        si = vi.si
        # dimOneTile_mm of createDimStructure(volume, NaN): z from zero delay, no focus shift
        lam_eq = pchip_equispace(np.zeros((1, len(vi.lambda_nm))), vi.lambda_nm)[1]
        l = np.sort([lam_eq[0], lam_eq[-1]])
        n = si.tissue_ri
        dz_air = 0.5 * (l.mean() / 1e3) ** 2 / (np.diff(l)[0] / 1e3)
        N = len(vi.lambda_nm)
        z_um = matlab_linspace(0.0, dz_air / n * N / 2, N // 2)
        self.tg = build_tiled_geometry(si, z_um, [np.nan], None, None)
        self.z_mm = self.tg.tile_z_mm
        self.x_mm = self.tg.tile_x_mm
        poly = si.optical_path_polynomial()
        self.oc = (OpticalPathCorrection.build(poly, self.tg.tile_x_mm, self.tg.tile_y_mm, self.tg.tile_z_mm)
                   if poly is not None else None)
        self.z_pixel_um = float(np.median(np.diff(self.z_mm)) * 1e3)
        self.n_z = len(self.z_mm)
        self.n_medium = float(si.tissue_ri)
        self.xi0 = int(np.argmin(np.abs(si.x_centers_mm)))
        self.yi0 = int(np.argmin(np.abs(si.y_centers_mm)))
        self.n_frames = int(vi.hdr.size_y)
        self.frame_center = max(1, int(np.floor(self.n_frames / 2 + 0.5)))   # MATLAB round(nY/2), 1-based
        depths, fallback = select_depths_to_measure(si.z_depths_mm)
        self.depths = depths
        self.shallow_fallback = fallback

    def setup(self):
        si = self.vi.si
        return {"depths": [{"zi": int(z), "z_mm": float(si.z_depths_mm[z])} for z in self.depths],
                "shallow_fallback": self.shallow_fallback, "xi0": self.xi0, "yi0": self.yi0,
                "y_center_mm": float(si.y_centers_mm[self.yi0]), "frame_center": self.frame_center,
                "n_frames": self.n_frames, "x_centers_mm": si.x_centers_mm.tolist(),
                "z_depths_mm": si.z_depths_mm.tolist(), "z_pixel_um": self.z_pixel_um, "n_z": self.n_z,
                "z_mm_first": float(self.z_mm[0]), "z_mm_last": float(self.z_mm[-1]),
                "x_mm_first": float(self.x_mm[0]), "x_mm_last": float(self.x_mm[-1]),
                "tissue_refractive_index": self.n_medium}

    def folder_for(self, zi: int, xi: int):
        t = self.vi.by_pos.get((int(xi), self.yi0, int(zi)))
        return t.folder if t else None

    def bscan_db(self, folder: str, frame1: int, beta: float) -> np.ndarray:
        """reconstructCenterBScan + mag2db(|scan|+eps): (nZ, nX) dB image."""
        vi, h = self.vi, self.vi.hdr
        nb = max(1, h.bscan_avg)
        mags = []
        for rep in range(nb):
            interf = vi.read_frame_interferogram(folder, frame1 - 1, rep)
            ie, le = _equispace_like_legacy(interf, vi.lambda_nm)
            mags.append(np.abs(scan_cpx(ie, le, beta)))
        mag = np.mean(mags, axis=0)                                     # (lines, nZ)
        if h.ascan_avg > 1:
            mag = mag.reshape(h.size_x, h.ascan_avg, -1).mean(axis=1)
        mag = mag.T                                                     # (nZ, nX)
        if self.oc is not None:
            j = frame1 - 1
            r = np.arange(self.n_z)
            src = np.clip(r[:, None] + self.oc.shift[j][None, :], 0, self.n_z - 1)
            t = r[:, None] + self.oc.pdz[j][None, :]
            valid = (t >= 0) & (t <= self.n_z - 1)
            mag = np.take_along_axis(mag, src, axis=0) * valid
        return 20 * np.log10(mag + np.finfo(float).eps)

    def fit(self, z_stage_mm, focus_pix, z_query_mm):
        return fit_drift(z_stage_mm, focus_pix, z_query_mm, self.z_pixel_um, self.n_z,
                         tissue_refractive_index=self.n_medium)


def drift_readout(diag: dict, n: int) -> str:
    """updateDriftReadout text."""
    regime = diag.get("driftRegime")
    if regime == "no-measurable-drift":
        ns = f"~{diag['tissueRI']:.2f} (no measurable drift)"
    elif regime == "suspicious-structure-clicks":
        ns = "not physical (clicks on a structure?)"
    else:
        ns = f"{diag['tissueRI']:.3f}"
    return (f"Clicked points: {n} ({len(diag.get('rejectedIdx', []))} rejected)\n"
            f"Drift slope: {diag['driftSlope']:.3f} um/um\nTissue RI: {ns}")


def finish_focus(ft: FocusTool, measurements, out_dir: Path, also_volume=False):
    """getFocusForAllDepths + generateFocusDiagnostic: fit all depths, save .mat + .png.
    measurements: list of {zi, pix, z_mm} (accepted clicks, in measuring order)."""
    import scipy.io as sio
    si = ft.vi.si
    if not measurements:
        raise ValueError("No focus measurements were recorded. The focus was not visible/clickable on any tile.")
    idx = np.array([int(m["zi"]) for m in measurements])
    pix = np.array([float(m["pix"]) for m in measurements])
    zmm = np.array([float(m["z_mm"]) for m in measurements])
    z_stage = si.z_depths_mm[idx]
    focus, diag = ft.fit(z_stage, pix, si.z_depths_mm)
    drift_um = (zmm - zmm[0]) * 1e3
    d_stage = (z_stage - z_stage[0]) * 1e3
    with np.errstate(invalid="ignore", divide="ignore"):
        per_stage = drift_um / d_stage
    rejected = np.isin(np.arange(len(idx)), np.asarray(diag.get("rejectedIdx", []), int))
    table = {"tileIndex": (idx + 1).astype(float), "zStage_mm": z_stage, "focusMeasured_pix": pix,
             "focusMeasured_z_mm": zmm, "drift_um": drift_um, "dStageFromRef_um": d_stage,
             "driftPerStage": per_stage, "rejected": rejected}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir]
    if also_volume:
        targets.append(ft.vi.volume)
    def _mat(v):        # MATLAB-friendly: numeric lists -> double arrays, text lists -> cell arrays
        if v is None:
            return np.zeros((0, 0))
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=object).ravel()
            if all(isinstance(x, str) for x in arr):
                return arr.reshape(-1, 1)
            return np.asarray(v, float)
        return v
    diag_mat = {k: _mat(v) for k, v in diag.items()}
    png = drift_figure_png(table, diag)
    saved = []
    for d in targets:
        mat = Path(d) / "zChosenFocusPositions.mat"
        if mat.exists() and d == ft.vi.volume:
            bak = mat.with_name("zChosenFocusPositions_backup.mat")
            if not bak.exists():
                mat.rename(bak)
        sio.savemat(mat, {"focusPositionInImageZpix": np.asarray(focus, float).reshape(1, -1),
                          "zDepths_mm": si.z_depths_mm.reshape(1, -1),
                          "focusTable": table, "fitDiagnostics": diag_mat})
        (Path(d) / "zChosenFocusPositions.png").write_bytes(png)
        saved.append(str(mat))
    return {"focus_positions": [float(f) for f in focus], "diagnostics": diag, "table": {k: np.asarray(v).tolist() for k, v in table.items()},
            "saved": saved, "figure_png": png}


def drift_figure_png(table: dict, diag: dict) -> bytes:
    """plotFocusDrift: drift vs stage depth, rejected clicks, physical fit line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.asarray(table["zStage_mm"]) * 1e3
    d = np.asarray(table["drift_um"])
    rej = np.asarray(table["rejected"], bool)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    ax.plot(z[~rej], d[~rej], "o", ms=7, color=(0.2, 0.4, 0.9), label="Identified focus")
    if rej.any():
        ax.plot(z[rej], d[rej], "x", ms=10, mew=2, color=(0.85, 0.2, 0.2), label="Rejected clicks")
    ax.grid(True)
    ax.set_xlabel("Stage depth [µm]")
    ax.set_ylabel("Focus drift in image [µm]")
    ax.set_title("Focus drift vs stage depth")
    if len(z) >= 2:
        zl = np.linspace(z.min(), z.max(), 100)
        fp = diag["slopeUsed_pixPerUm"] * np.maximum(zl, 0) + diag["intercept_pix"]
        ax.plot(zl, (fp - np.asarray(table["focusMeasured_pix"])[0]) * diag["zPixelSize_um"], "-",
                color=(0.85, 0.2, 0.2), lw=1.5, label="Physical fit (outlier clicks excluded)")
        regime = diag.get("driftRegime")
        ns = (f"~{diag['tissueRI']:.2f} (no measurable drift)" if regime == "no-measurable-drift" else
              "not physical (structure clicks?)" if regime == "suspicious-structure-clicks" else f"{diag['tissueRI']:.3f}")
        sub = f"Drift slope = {diag['driftSlope']:.4f} um/um, Tissue RI = {ns}"
        if diag.get("r2") is not None and not np.isnan(diag.get("r2", np.nan)):
            sub += f",  R² = {diag['r2']:.3f}"
        if diag.get("wasSlopeClamped"):
            sub += "  [slope clamped to physical range]"
        ax.set_title("Focus drift vs stage depth\n" + sub, fontsize=10)
        ax.legend(loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
