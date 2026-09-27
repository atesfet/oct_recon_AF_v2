# 02 — Raw OCT data: structure and storage

Example dataset: `10um_FOV_1/OCTVolume` (acquired 2026-09-10/11, Thorlabs GAN632,
Olympus 20x OCTG WINTER probe, Python SDK `yOCTScan3DVolume()`).

## 1. Folder layout

```
10um_FOV_1/
├── OCTVolume/                       # raw tiled scan (956 GB)
│   ├── ScanInfo.json                # scan description (written once, after all tiles)
│   ├── zChosenFocusPositions.mat    # focus pixel per depth (yOCTMeasureFocusDrift)
│   ├── zChosenFocusPositions.png    # its diagnostic plot
│   ├── Data01/ … Data864/           # one folder per tile (x, y, z-depth)
│   │   ├── Header.xml               # Thorlabs OCITY header (~130 KB)
│   │   └── data/
│   │       ├── Chirp.data           # float32[2048]
│   │       ├── OffsetErrors.data    # float32[2048] (unused by reconstruction)
│   │       └── Spectral0.data … Spectral499.data   # raw B-scans
├── 10um_H&E_1mmFOV.tiff             # LEGACY RECONSTRUCTION of OCTVolume (uint16, 3.2 GB)
└── Reslice of 10um_H&E_1mmFOV.tif   # ImageJ reslice of the above
```

## 2. `ScanInfo.json` (key fields, this dataset)

| Field | Value | Meaning |
|---|---|---|
| `version`, `units` | 1.1, mm | |
| `octSystem` | `gan632` | selects λ range / loader |
| `octProbePath`, `octProbe` | 20x OCTG WINTER ini (full content embedded) | probe calibration (§5) |
| `pixelSize_um` | 2 | lateral pixel |
| `tissueRefractiveIndex` | 1.33 | scales depth axis |
| `xRange_mm`, `yRange_mm` | [-6, 6], [-4.5, 4.5] | overall area 12 × 9 mm |
| `octProbeFOV_mm`, `tileRangeX/Y_mm` | 1 | tile = 1 × 1 mm |
| `nX/nYPixelsInEachTile` | 500 × 500 | |
| `xCenters_mm` | −5.5 … 5.5 (12) | |
| `yCenters_mm` | −4 … 4 (9) | |
| `zDepths` | −0.03 … 0.04 step 0.01 (8) | stage z offsets; 0 = focus at tissue/coverslip interface |
| `gridXcc/gridYcc/gridZcc` | 864 entries | tile centre of each folder; order = z fastest, then x, then y (`meshgrid(x, z, y)(:)`) |
| `octFolders` | `Data01`…`Data864` | folder of each grid entry (`%02d`, so 3 digits beyond 99) |
| `scanOrder` | 1…864 | acquisition order |
| `galvoPhaseDelayXOffsetCorrection_mm` | 0.01915 | galvo lag already compensated at acquisition |
| `oct2stageXYAngleDeg` | −0.77 | OCT→stage rotation, used only for stage moves |
| `nBScanAvg` | 1 | |

Tile index formula: `folder index (1-based) = zi + 8·xi + 96·yi + 1` (0-based xi, yi, zi).

## 3. `Header.xml` (per tile, Thorlabs OCITY)

* `DataFiles/DataFile Type="Raw"` one per B-scan: `SizeZ=2048` (spectral px), `SizeX=525`
  (A-lines incl. apodization), `SizeY=500`, `BytesPerPixel=2`, `ApoRegion 0–25`, `ScanRegion 25–525`.
* `Image`: SizePixel 1024 × 500 × 500, SizeReal 1.656 × 1 × 1 mm, `CenterX=0.01735` (dynamic offset + galvo correction), axis order ZXYT.
* `Acquisition`: all averaging = 1, ScanTime 5.56 s per tile, `ApodizationType=EachBScan`, 25 apodization A-scans, FrameByFrame.
* `Instrument`: GAN632 (Ganymede series), 2048 px, 12-bit camera, unsigned raw, central λ 880 nm, bandwidth 239 nm, 100 kHz line rate.
* `Processing` block describes Thorlabs' own processing settings — **not used** by myOCT.

## 4. Binary files

| File | dtype / shape | Notes |
|---|---|---|
| `Spectral{i}.data` | int16 little-endian, 525 × 2048 (C order: A-line major, 2 150 400 bytes) | B-scan i = tile-local y frame i (0-based). A-lines 0–24 = apodization (galvo parked, reference only), 25–524 = scan A-lines x=0…499. Values are 12-bit (0…4095) so int16 is safe. MATLAB reads `fread('short')` then `reshape([2048 525])`. |
| `Chirp.data` | float32[2048] | spectrometer pixel → k mapping, λ = 1/(chirp·(1/1010−1/796)/2047 + 1/796) nm |
| `OffsetErrors.data` | float32[2048] | camera offset calibration — not used |

Sizes: one tile = 500 × 2.15 MB = 1.075 GB; 864 tiles = **929 GB of spectra** (956 GB on disk).
Raw values per tile: 500 (y) × 525 (A-lines) × 2048 (spectral) samples.

## 5. Probe calibration used by reconstruction (`octProbe` in ScanInfo)

* `OpticalPathCorrectionPolynomial` [p1…p5]: field-curvature z shift (µm) = p1·x + p2·y + p3·x² + p4·y² + p5·x·y, tile-local x,y in µm.
* `DefaultDispersionQuadraticTerm` 8.6883e7: used only if reconstruction is called with `dispersionQuadraticTerm=[]`.
* `GalvoPhaseDelay_Asamples` 19.15: only used by reconstruction for old scans lacking `galvoPhaseDelayXOffsetCorrection_mm`.
* Everything else (Factor/Offset voltages, RangeMax, camera calibration, stage angle) is acquisition-only.

## 6. Focus file `zChosenFocusPositions.mat`

`focusPositionInImageZpix = [433 ×8]`, `zDepths_mm`, `focusTable` (MATLAB table: one click on
tile depth z=0 at pixel 433), `fitDiagnostics` (slope 0, "Only one focus click: assuming no drift").
The scan is shallower than `firstMeasureDepth_um=50`, so only z=0 was offered — by design.
Python reads it with `scipy.io.loadmat(...)["focusPositionInImageZpix"]`.

## 7. Reconstructed output format (legacy and Python port — identical)

BigTIFF, 4500 pages (one per y, −4.5 → 4.498 mm), each page 35 (z) × 6000 (x) uint16, PackBits.
`dB = (bits−1)·(c2−c1)/65534 + c1`, `bits = 0` ⇒ NaN (no data). Tag 305 (`Software`) holds JSON
`{"metadata": {lambda, z, x, y, aux, scanZDepths_mm}, "clim": [c1, c2], "version": 3}`.
The port writes that JSON on page 0 (all the legacy reader uses) plus a `<file>.tiff.json`
sidecar, and optionally keeps the float32 dB volume as `<name>_dB_float32.npy` (y, z, x).

Calibration (as legacy `yOCT2Tif` buildTiffFrameTags, with voxel sizes measured from the
output axes): `XResolution` = 1/dx and `YResolution` = 1/dz (pixels/cm, `ResolutionUnit` cm),
and an ImageJ `ImageDescription` with `unit=micron` and `spacing=dy`. Fiji therefore shows
dx × dz × dy µm, e.g. 2 × 2 × 2 µm for 10um_FOV_1. The same description carries `oct_*`
key=value entries, and the JSON carries `voxel_size_um` and `acquisition`:
* patches stitched in x/y (12 × 9);
* patch FOV (1 × 1 mm) and patch size (500 × 500 px);
* patch step and overlap (0);
* focus depths (8, 10 µm apart);
* native depth pixel (1.434 µm in tissue) and refractive index;
* consistency checks.

## 8. Reading raw data in Python

```python
from octrecon.io.scaninfo import ScanInfo
from octrecon.io.thorlabs import read_header, read_chirp, read_bscans_into
si  = ScanInfo("10um_FOV_1/OCTVolume")
t   = si.tiles[0]                                   # Tile(folder='Data01', xi, yi, zi, centres…)
hdr = read_header(si.folder / t.folder)
buf = np.empty((10, hdr.interf_size, hdr.n_lambda), np.int16)
read_bscans_into(si.folder / t.folder, range(10), buf, hdr)   # frames 0-9
```

## 9. Storage / I/O characteristics

* Disk: Crucial X10 8 TB USB SSD, exFAT. Measured sequential read ≈ **710–800 MB/s** per tile (cold).
* Full raw read ≈ 956 GB / 0.75 GB/s ≈ **21–22 min** — the hard lower bound for any reconstruction of this dataset from this drive.
* 435 k small files (2.15 MB each) on exFAT; per-file open overhead is small relative to 2 MB reads.

## 10. Storage convention — what is Thorlabs-standard and what is myOCT-specific

**Unit of storage = one raw B-scan per file (not per A-scan, not per tile).**
Each `Spectral{i}.data` is exactly one frame as delivered by the Thorlabs SpectralRadar
SDK: 525 consecutive camera lines × 2048 pixels, int16. The first 25 lines are the
*apodization* lines (galvo parked at `ApoVoltage` so the beam sees no sample ⇒ reference
spectrum, recorded before **each** B-scan: `ApodizationType=EachBScan`), the next 500 are the
scan A-lines of that B-scan. A-lines are contiguous 2048-sample spectra (camera line order),
so one file = `raw[525][2048]`. No processing has been applied (no dechirp, no FFT, no
background subtraction) — these are the camera counts (12-bit in 16-bit words).

**Per-tile container = Thorlabs OCITY (`.oct`) format, unzipped.** How a tile folder is made
(`Scanning code/.../ThorlabsImagerPython/thorlabs_imager_oct.py`, `yOCTScan3DVolume`, l.167-312):
1. Scan pattern: `create_volume_pattern(rangeX, 500, rangeY, 500, EACH_BSCAN apodization, FRAME_BY_FRAME)`,
   shifted to the tile centre (the probe's `DynamicOffsetX` + galvo-delay correction ⇒ `CenterX=0.01735`).
2. Acquisition `ASYNC_FINITE`: the SDK delivers B-scans one at a time; each is added to an
   `OCTFile(OCITY)` as `data\Spectral{bscan_idx}.data` in arrival order
   (`bscan_idx = (y-1)·nBScanAvg + (avg-1)`, so B-scan repeats would be consecutive). A lost frame aborts the tile.
3. `save_calibration` adds `data/Chirp.data` + `data/OffsetErrors.data`; `set_metadata` fills the XML
   (device, probe, pattern, processing defaults); scan time and comment are set.
4. The `.oct` is written (`scan.oct`, which **is a ZIP archive** — the standard ThorImage OCT file),
   then **extracted in place and deleted** so MATLAB can read the parts directly.
5. `_fix_header_xml_for_matlab` rewrites parts of `Header.xml` (Raw DataFile `SizeZ/SizeX/SizeY`,
   `ApoRegion`/`ScanRegion`, `Image/SizePixel/SizeY`, `SpeckleAveraging/SlowAxis=nBScanAvg`)
   because the SDK header does not match what `yOCTLoadInterfFromFile` expects.

So: **file format of a tile = standard Thorlabs OCITY content** (Header.xml + `data/*.data`
binary blobs, readable by ThorImage OCT if re-zipped), with a patched header; per-B-scan
`Spectral{i}.data` naming is the SDK's own raw-frame naming inside OCITY files.

**Everything above the tile is myOCT's convention, not Thorlabs'** (`ThorlabsImager/yOCTScanTile.m`):
* One OCITY folder per tile named `Data%02d` (Data01…Data864), in scan order
  z fastest → x → y (`meshgrid(xCenters, zDepths, yCenters)(:)`); the stage moves between tiles
  (rotated by `Oct2StageXYAngleDeg`), the galvos only scan within a 1×1 mm tile.
* `ScanInfo.json` (myOCT, written after all tiles) records the grid, per-folder tile centres
  (`gridXcc/gridYcc/gridZcc`), the full probe .ini (`octProbe`), pixel size, refractive index, etc.
* `zChosenFocusPositions.mat/.png` (myOCT post-processing, not acquisition).
* Tiles do **not** overlap in x/y (tile step = tile size = 1 mm); in z the 8 depths overlap
  heavily and are blended by the focus-weighted stitching.

**Consequences for reconstruction speed:** 435 k files of 2.15 MB; the natural read unit is a
tile (500 sequential files, 1.07 GB). The legacy stitcher reads per *output* plane (one file from
each of 96 tile folders per plane); the port reads tile-major and streams at the drive's limit.
Raw data are uncompressed (`CompressionLevel 0`, `unzipOCTFile=false` for Gan632), so the 956 GB
on disk is essentially the raw camera stream (100 kHz × 2048 px × 2 B ≈ 410 MB/s during acquisition).
