# 08 — Tissue-only xy (en-face) projection (new in v2)

Besides the 3D volume, v2 saves **one 2D xy image of the tissue**: every (x, y) column of the
reconstructed volume is collapsed over depth, using **only the voxels that belong to the
tissue**. Air above the tissue, the FEP film, the background below it and columns without any
tissue do not contribute; columns without tissue are NaN (shown black in the PNG).

Code: `octrecon/core/projection.py` (`tissue_projection`), outputs written by
`octrecon/outputs_v2.py`. On by default (`xy_projection=True`); `--no-projection` or unticking
*Save the xy projection* in the web app turns it off. With FEP removal also off (`--v1`) the
output is identical to v1.

## Outputs

| File | Content |
|---|---|
| `<name>_xy_mean.tif` | mean-amplitude projection over the tissue slab (float32 dB, NaN = no tissue) |
| `<name>_xy_max.tif` | maximum-intensity projection over the tissue slab |
| `<name>_xy_mean.png`, `_xy_max.png` | 8-bit previews (1st–99.7th percentile) |
| `<name>_tissue_thickness_um.tif` (+ `.png`) | thickness of the slab used per column (µm, 0 = no tissue) |

The TIFFs are calibrated (ImageJ: µm per pixel = output x/y pixel size); the description holds
the mask threshold and settings. The run summary has them under `xy_projection`.

Re-run only the projection (e.g. with other mask settings) from a reconstruction that kept its
float volume (`keep_float_volume=true`), without reconstructing again:
`python -m octrecon project <output_dir> [--threshold-db -12 --max-hole-mm2 0.5 --smooth-um 30 --all-z]`.

## Tissue mask

The volume `I(y, z, x)` (dB, as in the TIFF) is processed in y-chunks straight from the
memory-mapped result, so it works for volumes larger than RAM (full 4500 × 35 × 6000 volume:
3 streaming passes).

1. **Lateral smoothing per depth plane.** The amplitude `A = 10^(I/20)` is averaged over a
   `tissue_smooth_um` × `tissue_smooth_um` (30 µm) window in (y, x) — NaN-aware normalised
   convolution — and converted back to dB: `S(y, z, x)`. This averages the speckle so tissue
   (volume scatterer) and background (noise) separate.
2. **Threshold.** Otsu's threshold of the histogram of all `S` values (tissue vs background are
   the two classes), or `tissue_threshold_db` if given. Voxel mask `M = S > threshold`.
3. **Footprint.** A column is tissue if it has ≥ 2 masked voxels. Connected specks smaller than
   `tissue_min_area_mm2` (0.005 mm²) are dropped; enclosed holes smaller than
   `tissue_max_hole_mm2` (0.5 mm²) — dark lumens, ducts, the darker (vignetted) tile corners — are filled
   (they are inside the tissue section); larger holes and regions open to the edge of the field
   stay outside.
4. **Slab.** Per column the top / bottom tissue surfaces are the first / last masked voxel; they
   are smoothed laterally (50 µm, normalised convolution) and interpolated into the filled holes.
   The slab is `[top, bottom]`, so dark structures *inside* the tissue (between its surfaces)
   are included in the projection.

## Projections

Over the voxels `z ∈ slab(x, y)` with finite values (`N` of them):

* **mean** (default image): `P_mean(x, y) = 20·log10( (1/N) Σ_z 10^(I(x,y,z)/20) )` — the mean
  amplitude, the same averaging the stitching uses;
* **max**: `P_max(x, y) = max_z I(x, y, z)`.

## Why FEP removal matters for it

Without FEP removal, the film reflection is the brightest thing in the volume: in tiles without
tissue it is segmented as "tissue" (round blobs in the footprint) and it dominates the top of
every slab. With the reflections removed the mask and the projection only contain tissue. The
run log warns when the projection runs with FEP removal off.

## Removed-signal visualisation (`fep_save_removed`)

To make the FEP correction inspectable, v2 also stitches the **removed signal** — the magnitude
of the subtracted complex component, `|s − s'|`, with exactly the stitching weights of the main
volume — into `<name>_fep_removed.tiff` (same grid and format as `<name>.tiff`, lower clim = the
main volume's), its mean projection over all z (`<name>_fep_removed_xy.tif/png`) and an overview
figure `<name>_fep_overview.png` (tissue projection, removed-signal projection, one B-scan of
each on the same dB scale).
