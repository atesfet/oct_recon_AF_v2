# 05 · Automatic parameter estimation (dispersion, focus, focus drift)

Package: `octrecon/estimation/`

| module | public API | legacy source |
|---|---|---|
| `dispersion.py` | `estimate_dispersion(volume_folder, device='cpu', initial=None, bounds=None, n_bscans=8, tiles=None, metric='auto', log=print)` | `Applications/yOCTScanGlassSlideToFindFocusAndDispersionQuadraticTerm.m:146-155`, `Demo_DispersionCorrectionManual.m` (manual slider) |
| `focus.py` | `detect_focus(volume_folder, dispersion_quadratic_term=None, device='cpu', mode='auto', tile_selection='center', confocal_check=True, log=print)` | `Processing/yOCTFindFocusTilledScan.m`, `yOCTMeasureFocusDrift.m` |
| `drift_fit.py` | `fit_drift(z_clicked_stage_mm, focus_clicked_pix, z_all_depths_mm, z_pixel_size_um, n_z_pixels, tissue_refractive_index=1.4, immersion_refractive_index=1.33, max_tissue_refractive_index=1.55)` → `(focus_per_depth, diagnostics)` | `yOCTMeasureFocusDrift_fitDrift.m` (whole file) |
| `preview.py` | `preview_bscan(...)`, `render_png(...)`, `dispersion_curve_png(result)`, `focus_profile_png(result)` | `yOCTMeasureFocusDrift.m:601-615` (`reconstructCenterBScan`) |
| `_common.py` | `VolumeContext` (tile readers, spectral geometry, per-B-scan magnitude) | – |

Everything runs on CPU (NumPy/numba) or GPU (CuPy) through `octrecon.backend.get_xp`;
results are plain JSON-friendly dicts. The pipeline calls both estimators when
`dispersion_quadratic_term: estimate` / `focus_positions: estimate` (or `auto` with no
`zChosenFocusPositions.mat`), and so do the CLI (`estimate-dispersion`, `detect-focus`) and the web UI.

All B-scan processing reuses the validated reconstruction code (`core/spectral.py`):
apodization subtraction, sinc5 k-linearisation, Hann/rms window,
`exp(-i·β·(k−k0)²)`, `ifft`, `|·|`, mean over B-scan repeats.

---

## 1. Dispersion (`estimate_dispersion`)

### Legacy
The only automatic legacy method needs hardware: it scans a glass slide and runs
`fminsearch(@(d) -max(mean(log|scan(d)|, x)), initialGuess)`
(`yOCTScanGlassSlideToFindFocusAndDispersionQuadraticTerm.m:146-155`). Otherwise the value
was set by eye with the slider in `Demo_DispersionCorrectionManual.m`.

### Algorithm (offline, on an existing tiled scan)
1. **B-scan choice** – `select_bscans`: tiles with zDepth > 0 (inside tissue; else ≥ 0; else all)
   in the central ±30 % of the xy grid, frames spread over 10–90 % of the tile,
   2·`n_bscans` candidates (seeded RNG, deterministic). Each candidate is scored by
   bright-pixel dynamic range (99.5th percentile / median of |A|², dB) and the best
   `n_bscans` (default 8) are kept, which drops tiles with no tissue in them. `tiles=` overrides the choice
   (folder names or `(folder, frame)` pairs).
2. **Linearise once** – each B-scan is k-linearised and multiplied by the Hann/rms window
   once (`X`). A candidate β then costs one multiply and one FFT: `A(β) = |ifft(X·exp(-iβ(k−k0)²))|`,
   with `k = 2π/λ_eq` exactly as in `build_spectral_geometry` (unit test checks the window
   is identical).
3. **Score** – per B-scan metric, ignoring the first 20 depth pixels (DC/autocorrelation):
   * `top_percentile` (default, `metric='auto'`): log of the mean of the brightest 1 % of |A|² pixels
   * `kurtosis`: Σ I² / (Σ I)²
   * `entropy`: Σ p log p, p = I/ΣI
   * `legacy_peak`: max_z mean_x log|A| (the glass-slide metric)
4. **Search** – coarse linear grid (41 points) over `bounds`. The default bounds are
   `sign(initial)·|initial|·[0.5, 1.5]`, and `initial` defaults to `params.resolve_dispersion`
   (probe preset, then the probe `.ini` default, then the legacy default 7.943e7). Each B-scan's curve is
   normalised to [0, 1] and the curves are averaged. The maximum of the average is then refined with
   `scipy.optimize.minimize_scalar(method='bounded')` inside the neighbouring grid
   cells (xatol = 2e-5·|β|). Per-B-scan optima are refined the same way and returned
   (`per_bscan_values`, `median_per_bscan`, `spread_mad`). `at_bound=True` means the optimum is at the edge of the search range, so widen `bounds`.

Returned keys: `value, method, metric, initial, initial_source, bounds, candidates,
score_curve, score_curves_per_bscan, per_bscan_values, median_per_bscan, spread_mad,
at_bound, tiles_used[{folder, frame, z_depth_mm, signal_dB}], n_evaluations, device, elapsed_s`.

### Validation on 10um_FOV_1 (CPU, 16 threads)
The reference is 8.949e7, which reproduces the legacy TIFF bit-exactly. To avoid
starting at the answer, the search starts from the probe `.ini` default 8.6883e7.

| run | estimate | error vs 8.949e7 |
|---|---|---|
| **default (`top_percentile`, 8 B-scans, initial 8.6883e7)** | **8.903e7** | **−0.51 %** |
| initial 7.943e7 (legacy default, bounds 3.97–11.9e7) | 8.902e7 | −0.53 % |
| initial = preset 8.949e7 (what the pipeline uses) | 8.911e7 | −0.42 % |
| n_bscans = 4 / 16 | 8.905e7 / 8.927e7 | −0.49 % / −0.25 % |
| other B-scan draws (RNG seeds 1, 2, 3) | 8.934e7, 8.942e7, 8.936e7 | −0.16 %, −0.08 %, −0.15 % |
| metric `kurtosis` | 8.790e7 | −1.78 % |
| metric `entropy` | 8.796e7 | −1.71 % |
| metric `legacy_peak` | lower bound (4.34e7) | fails on tissue |

* Per-B-scan optima (default run): 8.83–8.99e7 (robust spread, 1.4826·MAD = 0.75e6 ≈ 0.8 %).
* Timing: 13.6 s with a cold disk cache (USB SSD), 5.2 s warm; 148 score evaluations.
* Image effect: mean |ΔdB| of a single reconstructed B-scan (Data430/428/431), compared with the same B-scan at 8.949e7:
  **0.27 dB** at the estimate 8.903e7, both over the full frame and within ±30 px of the focus. That is about speckle level; the image is visually identical.
  For comparison, the `.ini` default 8.6883e7 gives 1.43 dB and 7.943e7 gives 4.1 dB.
* The `top_percentile` metric was chosen empirically on this dataset. Over 40 random B-scans at all depths
  the per-B-scan argmax medians were: top 0.1 % 8.76e7, top 0.3 % 8.82e7, **top 1 % 8.91e7 (8.93e7 on z>0 tiles)**,
  top 3 % 8.87e7, kurtosis 8.78e7, entropy 8.75e7, legacy −. The sharpness optimum rises
  slightly with depth (gel tiles ~8.7e7, deepest tissue tiles ~8.8–8.9e7), which is consistent with added
  tissue dispersion. This is why tissue tiles (z > 0) are used.
* The optimum is broad (see `dispersion_curve_png`): about ±1 % of β changes the score by less than 1 %.
  An accuracy of about ±0.5 % is realistic, and results closer than that depend on the metric.

### Why the legacy glass-slide metric fails here
`max_z mean_x log|A|` is a geometric-mean amplitude. On a glass slide one specular
reflector dominates, and compensating the dispersion raises its peak. In tissue, a wrong β spreads
energy from bright scatterers into dark pixels, which *raises* the mean log amplitude, so the
metric runs to the lower bound. Use `legacy_peak` only for glass-slide-like data.

---

## 2. Focus (`detect_focus`)

### Legacy ports
**`mode='legacy'`**: a non-interactive port of `yOCTFindFocusTilledScan.m` (`manualRefinment=false`):
* step 1 (l.57-59, 78-127): the tile at the zDepth closest to 0 and 5 y-frames `round(linspace(1,nY,5))`.
  For each frame it takes `meanAbs`, computes the median over x, blanks the top `round(nZ/3)` pixels and takes the first argmax; the
  result is `round(median)`.
* step 2 (l.61-71, 134-149): the tile one depth above (in the gel). It computes `mean_y(median_x(meanAbs))`, applies
  `imgaussfilt(σ=4)` (17 taps, replicate padding; ported as `imgaussfilt_1d`), then `findpeaks`
  (MATLAB semantics: flat peaks at their first sample, no endpoints; `findpeaks_matlab`), and takes the peak
  closest to step 1.
* **Processing:** `yOCTProcessScan` does *not* apply optical-path correction (the call is
  commented out at `yOCTProcessScan.m:270-274`). `n` only scales z. Dispersion comes from
  `reconstructConfig`, and **without it the legacy code uses `dispersionQuadraticTerm=100`**, i.e. no
  compensation (`yOCTInterfToScanCpx.m:43`). The port uses `params.resolve_dispersion` unless
  a value is given. This matters: with 100, step 1 moves from 418 to 404.
* **Quirk:** the legacy code selects the tile `octFolders{zDepthIndex}` (l.59/71), which is the first xy tile
  when z is the innermost scan loop. `tile_selection='legacy'` reproduces this, and `'center'` (the default) uses
  the central xy tile at the same depth.

**`mode='measure'`** (the default, `'auto'`, alias `'drift'`) automates `yOCTMeasureFocusDrift.m`. This is the
interactive tool that wrote `zChosenFocusPositions.mat` for 10um_FOV_1: its PNG shows one click at stage z = 0.
* It uses the same depth selection, `selectDepthsToMeasure` (l.126-162): from 50 µm, every 50 µm. For shallow scans it falls back to
  z ≥ 0 with the same stride, which gives only z = 0 here.
* It uses the central tile and B-scans around the central one (±5 % of nY, 5 frames), with optical-path correction as in
  `reconstructCenterBScan` (l.601-615).
* The user's click is replaced as follows. At z ≈ 0, where by protocol the gel/tissue interface is placed at the focus, the port uses the
  step-1 rule on the OPC-corrected frames. Deeper, it uses the confocal peak of the detrended depth profile
  (dB, smoothed with σ = 4 minus smoothed with σ = 30) nearest to the running drift prediction (±60 px, with a prominence of at least 0.3 dB).
* The measurements then go to `fit_drift`. A single click or no measurable drift gives a constant focus. Otherwise the result is a
  per-depth hinge line (flat for z ≤ 0).

**Diagnostic** (`confocal_check=True`): `stationary_confocal_peak` relies on the fact that the focus is the only depth
feature that does not move when the stage moves. It takes the pointwise median over all zDepths of the dB depth
profiles (central tile, OPC), which removes the moving reflectors and the tissue surface, and returns the detrended maximum.

Returned keys: `focus_positions` (1-based, one per zDepth), `method, mode, step1, step2,
legacy{step1{tile, frames, per_frame_pix, profile, value}, step2{..., profile_smoothed, peaks}},
per_depth[{zi, z_depth_mm, focus_pix, se_pix, source}], drift{measurements, fit}, confocal_peak,
dispersion_quadratic_term, z_pixel_size_um, n_z, elapsed_s`.

### Validation on 10um_FOV_1 (reference 433 for all 8 depths)
The dispersion used is 8.949e7; the timings are with a warm cache.

| method | result | Δ vs 433 |
|---|---|---|
| **default `measure` (auto)** | **435 ×8** (constant; frames 225–275 all 435) | **+2 px (2.9 µm)** |
| `measure` with dispersion 8.6883e7 / 7.943e7 | 435 / 434 | +2 / +1 |
| `measure` with dispersion 100 (legacy no-config default) | 421 | −12 |
| `legacy`, central tile (step 1 = 418, step 2 = 427) | 427 | −6 |
| `legacy`, `tile_selection='legacy'` (Data04 / Data03, corner tile) | 416 | −17 |
| stationary confocal peak (diagnostic) | 427 (also 424–427 on 5 other xy tiles) | −6 |

* Step 1 per frame (central tile, z = 0, no OPC): 380, 418, **433** (central frame), 423, 389. The
  interface is tilted by about 50 px (70 µm) across the tile in y, so only the central B-scans, where the operator
  focused, carry the reference.
* The stationary confocal maximum is 427 in every tile tested, about 6 px (8.6 µm) above the interface
  that the operator clicked. Legacy step 2 in the gel also finds it. It is reported as a
  diagnostic because the reference (and the protocol) define the focus as the z=0 interface.
* The result is constant for every depth. The positive depth range is 40 µm (< 50 µm), so the drift
  cannot be measured; the fit reports a single click and returns slope 0.
* Time: 1.5–1.7 s including the confocal check (about 30 B-scans read).

---

## 3. Drift fit (`fit_drift`)
This is a line-by-line port of `yOCTMeasureFocusDrift_fitDrift.m`:
* It drops NaN clicks and clicks at stage z < 0 (l.72-95).
* It uses a Theil-Sen fit over all pairs at distinct z (l.343-367), with up to 6 rounds of residual rejection. The
  threshold is `max(3·1.4826·MAD, 15 px)`, followed by a final refit (l.97-139).
* It computes the slope SE `σ_res/√(0.91·Sxx)` and R² (l.141-168).
* It clamps the slope to `[0, (n_s,max² − n_i²)/(n_i·n_a)/dz]` (l.98-100, 170-172).
* It assigns a regime (l.174-219): `normal`, `no-measurable-drift` (|slope/SE| < 2), `clamped-to-max`,
  or `suspicious-structure-clicks` (slope < −3 SE and < −0.1·(n_i/n_a)/dz).
* It computes the tissue RI `n_s = √(n_i² + m·dz·n_i·n_a)`, floored at n_i, with its SE (l.221-238).
* It evaluates a hinge at z = 0, `round(m·max(z,0) + b)`, clamped to [1, nZ] (l.255-268), with a per-depth SE that includes the
  π/2 and 0.91 corrections (l.270-288), and an extrapolation warning for more than 100 µm (l.290-300).

Diagnostics use the MATLAB field names. Indices are 0-based.
Unit tests cover synthetic clicks: normal drift with a rejected structure click (recovers slope, RI and hinge),
no drift with gel clicks ignored, a single click, clamped-to-max, and structure clicks.

---

## 4. Preview helpers (web UI)
* `preview_bscan(volume, zi, xi=None, yi=None, frame=None, dispersion_quadratic_term=None, device, apply_opc=True)`
  returns `image_db` (float32, nZ × ≤500 columns, block-averaged in x), `z_px_count`, `tile`, `frame`,
  `xi, yi, zi, z_depth_mm`, `z_um`, `x_mm`, and `focus_hint` (from `zChosenFocusPositions.mat` if present).
  The defaults are the central tile and the central B-scan, with OPC (like `reconstructCenterBScan`).
* `render_png(image_db, focus_pix=None, clim=None)` renders grayscale with a 5–99.8 percentile window (legacy drift tool) and
  dashed focus line(s).
* `dispersion_curve_png(result)` plots per-B-scan and mean normalised score curves, the estimate, the initial value and the per-B-scan optima.
* `focus_profile_png(result)` plots the interface, gel and measured profiles with the focus, legacy step 2 and confocal peak.

## 5. Tests
`tests/test_estimation.py` contains 7 unit tests that always run and 3 real-data tests that run when `OCT_TEST_VOLUME` points to an OCTVolume folder.
The real-data tests assert:
* the dispersion estimate is within 1 % of 8.949e7 (measured −0.51 %), not at a bound, and takes < 60 s;
* the focus is constant and within ±3 px of 433 (measured +2).

```
OCT_TEST_VOLUME=/media/atesfet/TS801/Q3_OCT_reconstruction/10um_FOV_1/OCTVolume \
  python -m pytest -q tests/test_estimation.py        # 10 passed in 8.7 s (CPU, warm cache)
```

## 6. Limitations
* **Dispersion:** only the quadratic term is estimated, and the optimum is metric-dependent at the
  ±1–2 % level (a broad optimum). `top_percentile` was selected on one dataset (OCTG 20x probe,
  positive β). For OCTP900 probes (β ≈ −1.5e8) pass `initial` with the right sign, or explicit `bounds`.
  The search never crosses zero unless `bounds` do. Scans with no tissue signal in the central tiles
  would need `tiles=`.
* **Focus, shallow scans:** the answer is the z = 0 interface on the central B-scans. It relies on the protocol
  (the operator places the interface at the focus at the scan centre). If the interface was not placed at the focus, the
  `confocal_peak` diagnostic (the true stationary maximum) is the better value; here it differs by 6 px.
* **Focus, deep scans (≥ 50 µm of positive depths):** per-depth confocal-peak detection in tissue replaces
  the human click. It is **not validated on real data** (10um_FOV_1 only reaches 40 µm). Check
  `drift.measurements` and `drift.fit.driftRegime` and the focus plot before trusting a non-constant result.
* `mode='legacy'` is faithful, including its weaknesses: step 2 picks the nearest local maximum of a noisy
  profile, and with `tile_selection='legacy'` the corner tile may contain no tissue.
* The GPU path uses the same code through CuPy (`xp.partition`, `xp.fft`). It was not exercised during
  validation because the GPU driver was unavailable; all numbers above are CPU numbers.


## Interactive tools (manual counterparts, web app)

`octrecon/estimation/interactive.py` + `octrecon/webapp/static/tools.js` port the two
interactive MATLAB figures of the legacy workflow.

| Legacy | Web app | What is reproduced |
|---|---|---|
| `Demo_DispersionCorrectionManual.m` | *Tune manually…* (dispersion) | Y frame of a Data folder, pchip `yOCTEquispaceInterf` once, slider v ∈ [−10, 10] → β = sign(v)·10^|v| (steps 0.01 / 0.1, as `SliderStep [0.0005 0.1]` × range 20), `log(abs(scanCpx))` with `caxis([-5 6])`. Additions: tile map / depth / x / y / B-scan navigation, numeric β entry, sharpness guide value. |
| `yOCTMeasureFocusDrift.m` (`measureFocus`, `renderTileBScan`, `onAdjustDisplay`, `onClickFocus`, navigation callbacks, `updateDriftReadout`, `getFocusForAllDepths`, `plotFocusDrift`) | *Choose focus positions…* | depth selection, start tile/B-scan, sticky X tile and B-scan across depths, `reconstructCenterBScan` (pchip + optical-path correction), `mag2db` display with the 5 / 99.8 % (`prctile`) base range and the brightness/contrast mapping, nearest-pixel click, blue predicted line, Accept/Skip/Stop/prev/next semantics, the fit for all depths, `.mat` (focusPositionInImageZpix, zDepths_mm, focusTable as a struct, fitDiagnostics) and the drift figure. |

Validation against MATLAB R2024a on 10um_FOV_1 (`validation/matlab/make_interactive_reference.m`):

| Item | Max difference |
|---|---|
| Dispersion tool ln-image, β = 100 / 8.949e7 / −1.2e8 (legacy wavelength axis) | 4.5e-13 / 2.0e-11 / 3.6e-11 |
| Focus window default B-scan (Data428, B-scan 250), dB | 5.5e-9 |
| z axis / x axis | 2.2e-16 / 5.6e-17 mm |
| Display base range (5 / 99.8 %) | identical (MATLAB `prctile` = numpy `hazen`) |
| Drift fit on 5 example clicks (focus vector, slope, RI, rejected click) | identical |
| Your original single click (433 px at z = 0) | `[433 × 8]`, identical to the legacy `zChosenFocusPositions.mat` |

Notes:
* The demo calls the loader without `octSystem`, so a GAN632 is auto-detected as
  "Ganymede". This gives a 0.23 nm different wavelength axis and a ~0.06 % different
  optimal β. The web tool uses the scan's system by default; the *Legacy wavelength axis*
  checkbox reproduces the demo exactly.
* Closing the focus window with ✕ discards the session. *Stop measuring here* finishes
  with the depths accepted so far, as closing the MATLAB window did.
* The `.mat` stores `focusTable` as a struct of columns: a MATLAB `table` cannot be
  written from Python. The reconstruction only reads `focusPositionInImageZpix`.
