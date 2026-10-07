# Validation against the legacy MATLAB code

These tools reproduce the validation described in `docs/oct_recon_AF_v1.pdf`. They need MATLAB
and the original myOCT library (not included, GPL-3.0: https://github.com/MyYo/myOCT).
Put myOCT at `../Reconstruction_code_legacy/myOCT` relative to the repository, or edit
the `addpath` lines.

| File | Purpose |
|---|---|
| `matlab/make_reference_plane.m` | Runs the **unmodified** legacy per-plane code for one output y plane, times each stage and dumps every intermediate (`ref_y2250.mat`). |
| `validate_stages.py` | Compares the Python stages with that dump (apodization, sinc5, complex scan, \|scan\|, optical path correction). |
| `validate_full_volume.py` | Compares two reconstruction TIFFs plane by plane (e.g. port vs legacy): NaN mask, ΔdB, LSB histogram, comparison images. |
| `matlab/run_legacy_full.m` | Full-volume timing run of legacy `yOCTProcessTiledScan`. Warning: the default 8-worker pool needs more than 14 GB RAM. |
| `matlab/bench_legacy_parfor.m` | Legacy parfor throughput micro-benchmark. |
| `matlab/make_interactive_reference.m` + `validate_interactive_tools.py` | Reference images of the two interactive legacy figures (Demo_DispersionCorrectionManual, the "Choose Focus Positions" window) and the drift fit; the Python script compares the web-app engines with them. |
| `matlab/run_legacy_synthetic.m`, `dump_synthetic_plane.m`, `dump_legacy_loader.m`, `make_legacy_simulated_scan.m` | Build the MATLAB references for `tests/test_formats.py` (`python tests/test_formats.py --build-reference`). |

Results on the reference dataset (10um_FOV_1, 956 GB):
* Stage errors are ≤ 2e-12 (float64); a full stitched plane is within 1e-6 dB of MATLAB.
* The legacy TIFF is reproduced bit-exactly with dispersion 8.949e7 and legacy double
  quantisation.
* Across the whole volume, 945 M voxels are within 1 LSB with 0 NaN mismatches.
