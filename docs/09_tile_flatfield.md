# 09 — Tile seam (flat-field) correction (new in v2)

The volume is stitched from 12 × 9 patches of 1 × 1 mm (500 × 500 pixels), acquired with **no
overlap** (patch step = patch FOV = 1 mm). Without correction the patch boundaries are clearly
visible as a dark grid in en-face planes and projections.

## What causes the seams (measured on `10um_FOV_1`)

* **Brightness falloff inside every patch (vignetting), the same in every patch.** At the tissue
  depth the signal falls by ≈ 10 dB towards the left patch edge (start of the fast scan), ≈ 6 dB
  towards the right and top edges and ≈ 4 dB towards the bottom; it depends on depth (largest at the
  tissue, absent in the background below it). Folded over all patches the pattern spans 5.2 dB
  (5th–95th percentile).
* Patch-to-patch gain differences are negligible (0.3 dB standard deviation).
* Neighbouring patches share no content (no overlap; the texture correlation across a seam is ≈ 0,
  versus ≈ 0.5 between neighbouring columns inside a patch). Registration / blending is therefore not
  possible for this acquisition; acquiring with ≈ 10 % overlap would allow it.

The correction therefore removes the shared falloff.

## Method (`octrecon/core/flatfield.py`)

Model per output voxel: `A(z,y,x) = N(z) + G(z,u,v) · S(z,y,x)` with `A = 10^(dB/20)` the stitched
amplitude, `N(z)` the noise floor (median of the no-tissue columns; for a field fully covered by
tissue, the emptiest depth plane), `S` the tissue signal and `(u,v)` the position inside the patch.

1. Tissue footprint from the tissue mask (doc 08).
2. `G(z,u,v) = Σ_t X_t / Σ_t m_t(z)` over the patches fully covered by tissue, with
   `X = max(A − N, 0)` and `m_t(z)` the mean of `X` in patch `t` (each patch normalised by its own
   level, so tissue differences between patches average out). Streamed one patch row at a time.
3. Smoothed laterally (Gaussian, `flatfield_smooth_px` = 8 px) and along depth with signal weights
   (planes without tissue do not pull the gain to 1), normalised to mean 1 per plane; planes with
   < 5 % of the peak signal get `G = 1`; the gain is capped at `flatfield_max_gain_db` = 24 dB.
4. Correction, applied in place before the TIFF and the projections are written:
   `A' = A + max(A − N, 0) · (1/G − 1)` — only the signal above the noise floor is rescaled, so the
   background at the patch edges is not amplified.

The gain map is saved as `<name>_flatfield_gain_dB.tif` (slices = output depth, 500 × 500 per patch)
and the noise floor as `<name>_flatfield_noise_floor.npy`; the run summary has `flatfield`.

## Results (`10um_FOV_1`)

Brightness step across the seams in the tissue-only projection (median |difference| of 5-pixel bands
on both sides, 50-pixel blocks along the seam): x seams 3.6 → 1.7 dB, y seams 2.7 → 1.1 dB (the same
measure between two columns inside a patch: 0.6 dB); folded patch modulation 5.2 → 1.5 dB. The
remaining step at the x seams is mostly the texture discontinuity between non-overlapping patches.

## Use

On by default (`flatfield_correction`); `--no-flatfield` or the web-app checkbox turns it off; `--v1`
turns it off together with FEP removal and the projection (legacy-identical output). For a
reconstruction that kept its float volume:
`python -m octrecon flatfield <output_dir>` (corrects the volume in place, rewrites the TIFF and the
projections).

## Limitations

* Corrects brightness only; the texture break between non-overlapping patches remains.
* Where the falloff exceeds the cap (patch corner at the start of the fast scan) the correction is
  partial.
* Assumes the same scan pattern in every patch (true for a tiled scan with one objective/scan setting).
