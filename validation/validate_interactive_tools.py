"""Compare the web-app interactive tools with the legacy MATLAB figures.

  matlab -batch "addpath('validation/matlab'); make_interactive_reference('<OCTVolume>', 'ref.mat', '<myOCT>')"
  python validation/validate_interactive_tools.py <OCTVolume> ref.mat

(1) Demo_DispersionCorrectionManual ln|scan| images, (2) the default B-scan of the
"Choose Focus Positions" window (yOCTMeasureFocusDrift), (3) yOCTMeasureFocusDrift_fitDrift.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octrecon.estimation.interactive import DispersionTool, FocusTool, VolumeInfo, slider_to_beta  # noqa: E402


def main(volume, ref_path):
    f = h5py.File(ref_path, "r")
    vi = VolumeInfo(volume)
    ok = True

    # (1) manual dispersion tool
    dt = DispersionTool(vi, "Data200", 0, legacy_axis=True)     # the demo auto-detects the system
    print(f"[dispersion] lambda_eq max|d| = {np.abs(dt.lambda_eq - f['D/lambdaEq'][()].ravel()).max():.2e} nm")
    lg_ref = f["D/lg"][()]
    for i, v in enumerate(f["D/sliderVals"][()].ravel()):
        b = slider_to_beta(v)
        lg, r = dt.ln_image(b), lg_ref[i].T
        fin = np.isfinite(lg) & np.isfinite(r)
        err = np.abs(lg - r)[fin].max()
        ok &= err < 1e-6 and lg.shape == r.shape
        print(f"[dispersion] slider {v:+.4f} (beta {b:+.4e}): max |d ln| = {err:.2e}  shape {lg.shape}")

    # (2) focus window default B-scan
    ft = FocusTool(vi)
    folder = "".join(chr(c) for c in f["F/folder"][()].ravel())
    y = int(f["F/yIInFile"][()].ravel()[0])
    s = ft.setup()
    print(f"[focus] window opens on depths {[d['zi'] + 1 for d in s['depths']]} (1-based), X tile {s['xi0'] + 1}, "
          f"B-scan {s['frame_center']}, folder {ft.folder_for(s['depths'][0]['zi'], s['xi0'])} (MATLAB default: {folder}, {y})")
    img, ref = ft.bscan_db(folder, y, 8.949e7), f["F/imLog"][()].T
    err = np.abs(img - ref).max()
    ok &= err < 1e-6
    print(f"[focus] dB image {img.shape}: max |d| = {err:.2e} dB")
    print(f"[focus] z axis max|d| = {np.abs(ft.z_mm - f['F/z_mm'][()].ravel()).max():.2e} mm, "
          f"x axis max|d| = {np.abs(ft.x_mm - f['F/x_mm'][()].ravel()).max():.2e} mm")
    from octrecon.estimation.interactive import matlab_prctile
    p = matlab_prctile(img, [5, 99.8])
    print(f"[focus] display base range py {p[0]:.4f}..{p[1]:.4f}  MATLAB "
          f"{f['F/baseLo'][()].ravel()[0]:.4f}..{f['F/baseHi'][()].ravel()[0]:.4f} ")
    ok &= abs(p[1] - f['F/baseHi'][()].ravel()[0]) < 1e-6 and abs(p[0] - f['F/baseLo'][()].ravel()[0]) < 1e-6

    # (3) drift fit
    cl = f["G/clicks"][()].T
    foc, d = ft.fit(cl[:, 0], cl[:, 1], f["G/zAll"][()].ravel())
    same = np.array_equal(np.asarray(foc, float), f["G/focus"][()].ravel())
    ok &= same
    print(f"[fit] focus vector identical: {same}; slope {d['driftSlope']:.6f} vs {f['G/driftSlope'][()].ravel()[0]:.6f}; "
          f"RI {d['tissueRI']:.6f} vs {f['G/tissueRI'][()].ravel()[0]:.6f}; rejected {list(d['rejectedIdx'])} vs "
          f"{(f['G/rejectedIdx'][()].ravel() - 1).astype(int).tolist() if f['G/rejectedIdx'].size > 1 or f['G/rejectedIdx'][()].ravel()[0] > 0 else []}")
    foc1, _ = ft.fit([0.0], [433.0], vi.si.z_depths_mm)
    print(f"[fit] single click 433 -> {list(map(int, foc1))}  MATLAB {f['G/focusOne'][()].ravel().astype(int).tolist()}")
    print("ALL MATCH" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
