# OCT Reconstruction (oct_recon_AF_v1)

GPU/CPU reconstruction of **tiled Thorlabs OCT volumes** with a local **web app**.
It runs on NVIDIA GPUs (CUDA, Linux/Windows), on **Apple-silicon GPUs (Metal/MPS, macOS)** and on any CPU.
This is a Python port of the MATLAB **myOCT** tiled-scan reconstruction
(`yOCTProcessTiledScan`). Its output matches the legacy MATLAB output to within 1 LSB
(bit-exact with the legacy quantisation), and it runs **~80× faster on an NVIDIA GPU**.
The CPU version is ~15× faster than legacy.

| Full volume (12×9 tiles × 8 depths, 956 GB raw, 4500×35×6000 output) | Time |
|---|---|
| Legacy MATLAB (measured plane × 4500) | ~25.5 h |
| This repo, CPU (Ryzen 9 6900HX, 16 threads) | 1.6 h |
| This repo, GPU (RTX 3080 Ti Laptop; limited by the USB-SSD read speed) | **18.4 min** |

---

## 1. Quick start

1. Install **Miniforge** (recommended, all platforms): <https://conda-forge.org/download/>.
   Any Anaconda/Miniconda also works. Without conda the launchers fall back to a Python
   `venv` + `pip`, which needs Python ≥ 3.10.
2. Download or clone this repository.
3. Start the app for your platform:

| Platform | How to start |
|---|---|
| **Ubuntu / Linux** | `./start_linux.sh` from a terminal, or double-click it and choose *Run* |
| **macOS** | Double-click `start_mac.command`. The first time, macOS may block it: right-click → *Open* → *Open*. If needed, run `chmod +x start_mac.command start_linux.sh` once. On Apple silicon it also installs PyTorch, so the **Apple GPU (Metal/MPS)** is used. |
| **Windows** | Double-click `start_windows.bat` |

The **first start** creates the conda environment `oct_reconstruction`, which takes a few
minutes. It also adds GPU support automatically:
* **NVIDIA driver found:** CuPy, matched to your driver's CUDA version;
* **macOS:** PyTorch, for the Apple GPU (Metal/MPS).

After that, the app opens in your browser at `http://127.0.0.1:8765/`. Keep the terminal window open while you use the app, and close it
(or press Ctrl+C) to stop the server.

> The server listens on `127.0.0.1` only. Nothing is exposed to the network, and no internet
> connection is needed after installation.

Manual start (any platform, inside the environment):
```bash
conda env create -f environment.yml          # once
conda activate oct_reconstruction
python tools/setup_gpu.py --conda "$(which conda)" --env oct_reconstruction   # optional, NVIDIA only
python launch.py                              # --port 8765 --no-browser
```

### GPU requirements
| GPU | Backend | Requirements |
|---|---|---|
| NVIDIA (Linux/Windows) | CuPy: fused CUDA kernel + cuFFT. The fastest option, fully validated. | A recent NVIDIA driver. The CUDA toolkit is installed inside the environment. |
| Apple silicon M1–M4 (macOS) | PyTorch Metal/**MPS**: the same maths on PyTorch tensors, float32. | macOS 13 or newer, PyTorch ≥ 2.3 (installed by the launcher). |
| none | CPU: Numba + multithreaded FFT. | Any 64-bit CPU. |

* In the web page, **GPU** means "whichever GPU this machine has": CUDA first, then Apple MPS.
* The Apple-GPU backend runs a numerical **self-test** against the CPU at start-up. If the
  self-test fails (e.g. an old macOS without MPS FFT support), the app says so and uses the
  CPU.
* Ops that MPS does not implement fall back to the CPU automatically
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
* Estimators and previews run on the CPU on Macs; they take a few seconds.
* The MPS code path is validated on every machine through PyTorch's CPU device
  (`tests/test_torch_backend.py`): it matches legacy MATLAB to 9e-5 dB on the reference
  dataset. However, it has **not yet been timed on real Apple hardware**; please report
  throughput with `python scripts/benchmark.py <OCTVolume> --device mps --planes 20`.
* If the GPU is shown as *not available*: on laptops, CUDA sometimes breaks after
  **suspend/resume** (`cuInit` error 999). Reboot, or run
  `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`. To prevent it permanently on Ubuntu,
  enable NVIDIA's suspend services:
  `sudo systemctl enable nvidia-suspend nvidia-resume nvidia-hibernate` and add
  `options nvidia NVreg_PreserveVideoMemoryAllocations=1` to `/etc/modprobe.d/nvidia-power.conf`.

---

## 2. Using the web app

1. **Raw OCT volume.** Browse to the `OCTVolume` folder, the one containing `ScanInfo.json`
   and the tile folders `Data01 … DataNN`. You can also select the sample folder that
   contains `OCTVolume`. **Inspect** reads only metadata and shows:
   * the scan geometry;
   * the raw size;
   * the storage layout (unzipped or compressed `.oct` tiles);
   * the expected output size.
2. **Output.** Choose an output folder and a reconstruction name. Everything is written to
   `<output folder>/<name>/` (see §5). The defaults are
   `<sample folder>/reconstructions/<sample>_recon`.
3. **Reconstruction inputs.** All values are filled in automatically, and the source of each
   one is shown. The *focus position* can be:
   * **Automatic**: the `zChosenFocusPositions.mat` in the volume folder if present,
     otherwise automatic detection;
   * detected now with **Detect focus**;
   * picked by clicking the focus band in **Preview B-scan…**;
   * typed in manually, loaded from another `.mat` file, or disabled.

   **Advanced options** hold the constants of the OCT system and probe. They are
   pre-filled, but you can override them or re-estimate them:
   * dispersion β (**Estimate** runs an automatic sharpness search);
   * focus σ, refractive index, z-crop, output pixel size, k-linearisation method;
   * optical-path correction;
   * how compressed tiles are handled.
4. **Compute.** Pick **GPU** or **CPU**. The compute parameters default to tuned values; see
   §4.3.
5. **Run.** Watch the progress, the estimated time remaining and the live log. Cancel at any
   time. When the run finishes, **Open output folder** shows the results.
   **Save config… / Load config…** store and restore every setting as JSON, which is also
   usable from the CLI.

View the result in Fiji/ImageJ (*File → Import → TIFF Virtual Stack* for large volumes), in
napari, or in MATLAB with myOCT's `yOCTFromTif`.

---

## 3. Accepted raw inputs

A volume is always a folder with `ScanInfo.json` and one sub-folder per tile (`Data01…`). Every
input the legacy myOCT tiled pipeline accepts is supported. The format is detected per tile,
and each one was validated against the unmodified MATLAB code (`docs/06_input_formats.md`):

| Tile content | System (`octSystem`) | Notes |
|---|---|---|
| **Unzipped Thorlabs OCITY**: `Header.xml` + `data/Spectral{i}.data` (+ `Chirp.data`). This is what the GAN632 Python SDK acquisition writes. | `gan632`, `ganymede`, `telesto` | Reference format. |
| **Compressed `.oct`**: `VolumeGanymedeOCTFile.oct`, or a single `*.oct` (a Thorlabs OCITY ZIP) | same | **Read in place** by default (no extraction, no extra disk space). Optionally extracted like legacy `yOCTUnzipTiledScan` (Advanced → *Extract*, optional *delete archive*; CLI `python -m octrecon extract`). Mixed unzipped/compressed volumes work. |
| Thorlabs **SRR** (`Data_Y…_B….srr` + `Chirp.dat`) | `ganymede_srr`, `telesto_srr` | The legacy tiled code fails on these (header bug); the port reads them correctly. |
| **Wasatch** (`*_raw_us_*.bin`, or 2D `raw_*.tif` + background) | `wasatch` | The legacy tiled code fails on these (no header branch); the port reads them correctly. |
| myOCT **simulated** tiles (`data.mat`) | `Simulated Ganymede` | For testing. |

All formats also support B-scan averaging (`nBScanAvg`), A-scan averaging and spectral
binning (AScanBinning), exactly as legacy does: the mean of |scan| over the repeats. If
`ScanInfo.json` has no `octSystem`, it is detected from the files using the legacy rules.
Not supported: AWS `s3://` paths, Thorlabs 1D point scans, and mixing manufacturers in one
volume.

Raw data layout (one file per raw B-scan: apodization + scan A-lines × spectral px;
Thorlabs versus myOCT conventions): `docs/02_raw_data_format.md`.

---|---|---|
| **Unzipped tiles** | `DataNN/Header.xml` + `DataNN/data/Spectral{i}.data` (+ `Chirp.data`). This is what the Thorlabs GAN632 Python SDK acquisition writes. | Read directly. |
| **Compressed tiles** | `DataNN/VolumeGanymedeOCTFile.oct` (or a single `*.oct`), i.e. a Thorlabs OCITY ZIP archive. | **Read in place** by default: no extraction and no extra disk space. Optionally extracted first like legacy `yOCTUnzipTiledScan` (Advanced → *Extract*, optional *delete archive*). CLI: `python -m octrecon extract <OCTVolume>`. |
| Mixed layouts | Some tiles unzipped, some compressed. | Both are handled transparently. |
| OCT systems | `gan632`, `ganymede`, `telesto` (λ ranges from myOCT). SRR / Wasatch variants: see `docs/06_input_formats.md`. | From `ScanInfo.json` `octSystem`. |
| Averaging | B-scan repeats (`nBScanAvg`), A-scan averaging, spectral binning. | As in legacy: the mean of \|scan\| over the repeats. |

Raw data format details (one file per raw B-scan: 25 apodization + 500 A-lines ×
2048 spectral px, int16; Thorlabs versus myOCT conventions): `docs/02_raw_data_format.md`.

---

## 4. Parameters

Every parameter can be set in the web UI, in a JSON/YAML config (`--config`), or with
`--set key=value` on the CLI. The value `"auto"` means *resolve automatically*. Each run
records the resolved value **and its source** in `<name>_config.json`.

### 4.1 Required
| Key | Meaning |
|---|---|
| `volume_folder` | The `OCTVolume` folder, or its parent. |
| `output_root` | The folder in which the output folder is created. |
| `output_name` | The name of the reconstruction; it names the output folder and the files. |

### 4.2 Reconstruction parameters (auto-filled)
| Key | Default → source | Notes (legacy name) |
|---|---|---|
| `focus_positions` | `auto` → `zChosenFocusPositions.mat` in the volume folder, else automatic detection | `focusPositionInImageZpix`. Also accepts a number, a list (one per depth), a path to a `.mat`, `estimate` or `none`. |
| `dispersion_quadratic_term` | `auto` → probe preset (`octrecon/presets.py`), else `ScanInfo.octProbe.DefaultDispersionQuadraticTerm`, else 7.943e7 | `dispersionQuadraticTerm` [nm²/rad]. `estimate` runs the automatic search. **A system/probe constant.** |
| `focus_sigma` | `auto` → probe preset, else the objective rule (10x: 20, 20x: 10, 40x: 10), else 20 | `focusSigma` [px]. The effective Gaussian std is √2·σ, as in legacy. |
| `n_medium` | `auto` → `ScanInfo.tissueRefractiveIndex` | Scales the z axis. |
| `crop_z_range_mm` | `auto` → [min, max] of `ScanInfo.zDepths` | `cropZRange_mm`; `none` = no crop. |
| `output_pixel_size_um` | `auto` → `ScanInfo.pixelSize_um` | `outputFilePixelSize_um`. Must equal the scan pixel size (a legacy restriction). |
| `interp_method` | `sinc5` | `interpMethod` (k-linearisation). |
| `apply_path_length_correction` | `true` | Optical-path (field-curvature) correction from the probe polynomial. |
| `raw_input` | `auto` | `auto` = read `.oct` in place; `extract` = extract first. |
| `delete_archives_after_extract` | `false` | The legacy default is `true`. |

**Validated preset:** GAN632 + Olympus 20x OCTG (WINTER): β = **8.949e7**, σ = 10. This
reproduces the legacy reconstruction of the reference dataset bit-exactly. To register a
new probe, add a line to `PROBE_PRESETS` in `octrecon/presets.py` (use **Estimate** to find
its β).

### 4.3 Compute parameters
| Key | Default | Notes |
|---|---|---|
| `device` | `auto` | `auto` (CUDA, else Apple MPS, else CPU), `gpu` (CUDA or MPS), `mps`, `cpu`. |
| `precision` | `float32` | Within 1e-4 dB of `float64`; the uint16 step is 1e-3 dB. MPS supports float32 only. |
| `gpu_fused_kernel` | `true` | Fused CUDA pre-FFT kernel (6× faster spectral stage). |
| `batch_frames` | 50 | B-scans per batch. Lower it for GPUs with little memory; 50 uses about 4 GB of VRAM. |
| `io_threads` | 8 | Parallel raw-file readers. |
| `prefetch_batches` | 16 | Batches read ahead of compute (≈ 16 × 107 MB of RAM). |
| `rows` | all | A subset of y-tile rows (0-based), e.g. `[4]`, useful for quick tests. |
| `write_tiff` | `true` | Write the uint16 BigTIFF. |
| `keep_float_volume` | `false` | Also keep the float32 dB volume (`.npy`, y×z×x). |
| `legacy_double_quantization` | `false` | Reproduce legacy's two-step uint16 quantisation (bit-exact comparison with legacy TIFFs). |

---

## 5. Outputs

`<output_root>/<output_name>/`:

| File | Content |
|---|---|
| `<name>.tiff` | uint16 BigTIFF, one page per y plane (z × x), PackBits. dB = (v−1)·(c2−c1)/65534 + c1, v=0 → no data. Same layout and metadata as legacy `yOCT2Tif`, so it can be read by `yOCTFromTif`. **Calibrated**: see below. |
| `<name>.tiff.json` | Metadata: axes in mm and `clim` [c1, c2] in dB. |
| `<name>_config.json` | All settings plus the resolved values and their sources, and the storage layout. |
| `<name>_run_summary.json` | Device, timings, clim, output shape. |
| `<name>.log` | The run log. |
| `<name>_dB_float32.npy` | Only if `keep_float_volume`. |

**TIFF calibration and metadata.** The voxel size is measured from the output axes (not
assumed) and written the way legacy `yOCT2Tif` does, so Fiji/ImageJ shows real units:
* **pixel width = x** (fast scan), **pixel height = z** (depth), **voxel depth = y**
  (slow scan, the page spacing). All three are in µm.
* The tags are `XResolution`/`YResolution` (pixels/cm) plus the ImageJ description
  (`unit=micron`, `spacing`).

The acquisition facts are added too. Fiji shows them under *Image → Show Info*
(`oct_*` keys), and they are also in the JSON (`voxel_size_um`, `acquisition`):
* **patches stitched in x and y** (e.g. 12 × 9), cross-checked between the ScanInfo scan
  range ÷ patch FOV and the list of patch centres;
* patch FOV (mm), patch size (px), patch step and overlap, and the number of focus depths;
* the native depth pixel before resampling (e.g. 1.434 µm in tissue at n = 1.33), the
  refractive index, and the OCT system and probe;
* a consistency check that the voxel size equals the patch FOV ÷ patch pixels, and that
  the output width equals the number of patches × the patch pixels.

Files written before this was added can be fixed without reconstructing again:
`python -m octrecon retag <file.tiff> [--volume <OCTVolume>]`. It rewrites only the
metadata and verifies that the pixels are unchanged.

---

## 6. Command line

```bash
python -m octrecon web                                   # the web app (same as launch.py)
python -m octrecon inspect  /data/sample/OCTVolume       # summary + automatic parameters (JSON)
python -m octrecon reconstruct /data/sample/OCTVolume --output-root /data/recon --output-name sample_recon --device gpu
python -m octrecon reconstruct /data/sample/OCTVolume --config my_config.json --rows 4
python -m octrecon reconstruct /data/sample/OCTVolume --set dispersion_quadratic_term=8.9e7 focus_positions=433
python -m octrecon estimate-dispersion /data/sample/OCTVolume
python -m octrecon detect-focus /data/sample/OCTVolume
python -m octrecon extract /data/sample/OCTVolume [--delete-archives]
python -m octrecon retag /data/recon/sample_recon/sample_recon.tiff --volume /data/sample/OCTVolume   # fix calibration/metadata of an existing TIFF
```
`configs/example_10um_FOV_1.yaml` is an example config.

---

## 7. How it works and why it is fast

For each raw B-scan the pipeline does, in order:
1. apodization subtraction;
2. sinc5 k-linearisation;
3. Hann window × dispersion phase;
4. inverse FFT;
5. magnitude;
6. optical-path correction;
7. focus-weighted interpolation into the output grid;
8. the weighted mean across the z-stack, then conversion to dB.

What changed relative to legacy:
* **GPU (NVIDIA):** the per-voxel maths runs on the GPU. A fused CUDA kernel does the apodization
  subtraction, the banded sinc5 operator and the window. Then come batched cuFFT, the
  magnitude, the gather-based optical-path correction, and separable interpolation as
  batched matrix products, with the accumulators kept in VRAM.
* **GPU (Apple silicon):** the same stages run as PyTorch MPS tensor ops. The sinc5 gather
  is done with the spectral axis first (contiguous row gathers), then the FFT, the
  optical-path gather and the interpolation matmuls.
* **CPU:** the same maths runs as a Numba-parallel kernel with multithreaded FFT.
* **I/O:** tiles are read tile-major and sequentially, by threaded readers that prefetch
  and overlap with compute. With a GPU the run is limited by disk read speed, so faster
  storage (e.g. an internal NVMe) makes it faster still.

Details, benchmarks and validation are in `docs/oct_recon_AF_v1.pdf` and `docs/0*.md`:

| Doc | Content |
|---|---|
| `docs/01_legacy_reconstruction.md` | How the MATLAB pipeline works, parameters, a per-script reference. |
| `docs/02_raw_data_format.md` | Raw data layout, Thorlabs versus myOCT storage conventions, output format. |
| `docs/04_reconstruction_parameters.md` | Every input, where it comes from, what can be estimated. |
| `docs/05_parameter_estimation.md` | Automatic dispersion / focus / drift estimation (Python ports). |
| `docs/06_input_formats.md` | Supported raw formats and their validation against MATLAB. |
| `docs/oct_recon_AF_v1.pdf` | Technical report: acceleration analysis, benchmarks, validation. |

---

## 8. Repository layout

```
octrecon/                 Python package
  pipeline.py             ReconConfig + Reconstructor (end-to-end driver, progress/cancel)
  params.py               volume inspection + automatic parameter resolution
  presets.py              system/probe constants (dispersion, focus sigma)
  io/                     ScanInfo, Thorlabs headers/chirp, tile readers (.oct / unzipped / ...), TIFF writer
  core/                   geometry, spectral processing (CPU/CUDA kernels), stitching,
                          torch_backend.py (Apple GPU / MPS)
  estimation/             automatic dispersion / focus detection, drift fit, B-scan previews
  webapp/                 local web server + static UI (no internet needed)
  cli.py                  command line (python -m octrecon ...)
launch.py                 starts the web app and opens the browser
start_linux.sh / start_mac.command / start_windows.bat   one-click launchers
tools/setup_gpu.py        adds GPU support: CuPy matched to the NVIDIA driver, or PyTorch (MPS) on macOS
environment.yml, requirements*.txt, pyproject.toml
configs/                  example configs
scripts/                  benchmark, resource monitor, synthetic test-volume generator
tests/                    pytest (synthetic data; real-data tests if OCT_TEST_VOLUME is set)
validation/               comparison against the legacy MATLAB code (MATLAB harness scripts)
docs/                     documentation + technical report (PDF)
```

## 9. Testing

```bash
conda activate oct_reconstruction
pytest -q tests          # ~10 s, CPU: synthetic volumes in every supported format, compared with
                         # stored outputs of the unmodified legacy MATLAB code (tests/data)
# real-data regression (optional):
OCT_TEST_VOLUME=/data/10um_FOV_1/OCTVolume OCT_LEGACY_TIFF=/data/10um_FOV_1/legacy.tiff pytest -q tests
OCT_DEVICE=gpu OCT_TEST_VOLUME=... pytest -q tests          # the same on the GPU
OCT_TORCH_DEVICE=mps pytest -q tests/test_torch_backend.py  # Apple GPU (default: PyTorch CPU device)
python scripts/make_synthetic_volume.py --help              # generate small test volumes
python tests/test_formats.py --build-reference              # rebuild MATLAB references (needs MATLAB + myOCT)
```

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| The browser does not open | Open `http://127.0.0.1:8765/` manually. The port may be different if 8765 is in use; the terminal prints the URL. |
| GPU card greyed out | See *GPU requirements* above. The UI shows the reason. CPU mode always works. |
| "Out of memory" on GPU | Lower `batch_frames` (e.g. 20). |
| The system runs out of RAM | Lower `prefetch_batches` / `io_threads`, and close large viewers such as Fiji stacks. |
| Slow on GPU | Check the disk read speed. The GPU pipeline reads the raw data at the drive's limit, so copying the raw data to an internal SSD helps. |
| macOS: GPU card greyed out | The UI shows the reason. Re-run `start_mac.command` (it installs PyTorch), and check macOS ≥ 13 on Apple silicon. |
| macOS: "cannot be opened" | Right-click `start_mac.command` → *Open*, or `xattr -d com.apple.quarantine start_mac.command`. |

## License

GPL-3.0 (derivative of myOCT). See `LICENSE` and `NOTICE.md`.
