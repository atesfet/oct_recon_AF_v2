# 07 — Removing the FEP-film reflections (new in v2)

The tissue is imaged sandwiched in FEP film. The film surfaces are smooth, flat dielectric
interfaces, so they give **specular reflections** that are 10–30 dB brighter than tissue: a
bright sheet at the tissue surface (z ≈ 0) and, when it is inside the reconstructed depth
range, at the bottom surface too. In the stitched volume they show up as a bright band in every
B-scan and as round, tile-periodic blobs in en-face planes (the specular return depends on the
scan angle, so it is bright in the tile centre and fades towards the edges).

v2 removes these reflections **before** the magnitude is taken (in the complex depth profile of
every A-line), so that the reconstructed volume only contains tissue signal. It is on by default
(`fep_removal=True`); `fep_removal=False` (`--no-fep`, or untick the box in the web app) gives
the v1 / legacy-identical output.

![en-face before/after](figures/fep_enface.png)
![B-scan before/after](figures/fep_bscan.png)

*Data set `10um_FOV_1`, y-tile row 4 (500 output planes), left: original, right/bottom: FEP
removed. Display range is the same for both.*

## Why it is possible

A specular reflector at depth `m` produces exactly the system point-spread function (PSF) in the
complex A-scan: `a(z) = A·e^{iφ}·psf(z − m)`. The PSF is fixed by the spectral window and the
residual dispersion; it is only 2–3 pixels wide. Tissue is a volume scatterer: its complex A-scan
is speckle, which has energy in all directions of the local (2R+1)-sample space. So the
reflection lives in a tiny subspace (the PSF at sub-pixel shifts: rank ≈ 3 out of 17 dimensions),
and tissue only has ≈ 3/17 of its energy there.

Things that do **not** work well on this data (measured, see the investigation notes in the PR):

* averaging / subtracting a lateral (x or y) mean A-line — the film amplitude jitters by ±20 %
  from A-line to A-line, the B-scan-to-B-scan phase is random, so lateral methods saturate at
  10–17 dB of suppression;
* low-rank (SVD) background removal in a B-scan window — tissue is locally low-rank, too;
* zeroing / notching the band — destroys the tissue at the surface.

## Algorithm (`octrecon/core/fep.py`)

Per tile and batch of B-scans (complex scans, before the magnitude):

1. **Detect surfaces.** Curvature-flatten |scan| with the optical-path (field-curvature) shift,
   average laterally (central 70 % of x) and find sharp peaks ≥ `fep_detect_min_db` (6 dB) above
   the 51-row median baseline. Only the depth band that reaches the output is analysed.
   Up to `fep_max_surfaces` (4) are handled — top / bottom film surfaces, film back sides.
2. **Locate the surface in every A-line.** Expected native row = surface row + curvature shift;
   refined by an arg-max within ±`fep_search` (4) px and a lateral median over
   `fep_lateral_median` (15) A-lines, so a bright tissue speckle cannot hijack it.
3. **Project onto the PSF basis.** The complex segment of ±`fep_half_window` (R = 8) samples
   around the peak is projected onto a rank-`fep_rank` (3) basis. The basis is learnt from the scan
   itself at start-up (≈ 6 000 strong, PSF-like surface segments from 12 tiles spread over the
   grid; bootstrapped from the theoretical PSF of the window). On this data it captures 96.8 % of
   the film-reflection energy (theoretical PSF: 85.5 %), because it also absorbs the residual
   dispersion.
4. **Decide where a film is present.** The fraction of segment energy captured by the basis
   ("specular-likeness", lateral median) gives a soft weight: 0 below `fep_frac_lo` (0.35),
   1 above `fep_frac_hi` (0.6). Tissue without a film reflection is left untouched.
5. **Subtract down to the background level, not to zero.** The background / tissue speckle also
   has energy along the basis (≈ `rank` × local background power, measured on ±10 rows outside
   the window). Removing the whole projection leaves a dark trench (≈ −6 dB) where the film was.
   The coefficient is therefore shrunk to the local background energy:
   `c ← c · sqrt(keep · rank · P_bg / |c|²)` (`fep_keep_level` = keep = 0.7). With `keep = 0` the
   whole projection is removed.

Everything outside the ±R window (+ search range) is bit-identical to the input; inside it only
the `rank` PSF components are changed. Heavy work runs on the selected device (CuPy / PyTorch);
only small (B, nX) arrays go to the host.

## Results

Synthetic ground truth (`tests/test_fep.py`; reflection = exact PSF with sub-pixel tilt, random
phase, ±20 % amplitude jitter, on speckle tissue):

| case | before | after |
|---|---|---|
| reflection +10/+20/+30 dB on tissue | residual error ≥ amp − 12 dB rel. tissue | < −6 dB rel. tissue |
| reflection on empty background | — | within ±3 dB of the background (no trench) |
| tissue without reflection | — | change < −20 dB rel. tissue |

Real data (`10um_FOV_1`, y-tile row 4, stitched output, dB):

| output depth | median before → after | 99th pct before → after |
|---|---|---|
| z = −5.5 µm (film above tissue) | −10.7 → −14.2 | 7.9 → −3.0 |
| z = 0.5 µm (tissue surface) | −5.1 → −12.9 | 10.2 → 0.4 |
| z = 4.5 µm | −5.1 → −12.3 | 10.7 → 2.7 |
| z = 8.5 µm | −7.4 → −13.0 | 9.5 → 3.7 |
| z = 14.5 µm | −14.7 → −15.9 | 6.0 → 3.3 |
| z = 30.5 µm (deep tissue) | −18.2 → −18.2 | −9.4 → −9.8 |

Cost: ≈ +40 % reconstruction time on the GPU (row 4: 106 s → 150–190 s), plus ≈ 30 s once for
learning the basis.

## Limitations

* Tissue signal that is itself *specular-like* at the film interface (a flat, sharp tissue
  boundary exactly on the film) is attenuated together with the film, because it is
  indistinguishable from the film in a single A-line.
* Inside the film window, the reconstructed tissue keeps its intensity level, but the
  `rank` PSF components are partly replaced by the background-level remainder of the film
  coefficient: the fine axial speckle pattern within ±1–2 px of the surface is not original.
* Tile seams / tile-periodic illumination (vignetting) are a different effect and are not
  addressed by this step.

## Parameters

| ReconConfig field | default | meaning |
|---|---|---|
| `fep_removal` | `True` | enable the step |
| `fep_keep_level` | 0.7 | background energy kept along the basis (0 = remove all) |
| `fep_rank` | 3 | PSF basis size |
| `fep_half_window` | 8 | R, samples on each side of the surface |
| `fep_search` | 4 | ± samples searched for the per-A-line peak |
| `fep_lateral_median` | 15 | A-lines for the position / decision median |
| `fep_frac_lo`, `fep_frac_hi` | 0.35, 0.60 | specular-likeness ramp |
| `fep_detect_min_db` | 6 | surface detection threshold |
| `fep_max_surfaces` | 4 | surfaces per B-scan batch |
| `fep_shrink` | `True` | background-level shrinkage (False = subtract the full projection) |
| `fep_learn_basis` | `True` | learn the basis from the scan (False = theoretical PSF) |
| `fep_basis_file` | `None` | `.npy` (2R+1, rank) complex basis; overrides learning |
| `fep_save_removed` | `True` | also save the removed signal: volume `<name>_fep_removed.tiff`, its xy projection and `<name>_fep_overview.png` (see doc 08) |

The run summary (`<output_name>_run_summary.json`) records the learnt basis (captured energy,
number of segments) and the removal statistics under `fep_removal`.
