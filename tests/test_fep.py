"""FEP-surface removal (octrecon/core/fep.py) against synthetic ground truth.

A specular reflection is the system PSF at a (sub-pixel) depth; it is injected into complex
scans of speckle "tissue" / noise and must be removed without touching anything else."""
import numpy as np
import pytest

from octrecon.core.fep import FEPConfig, FEPRemover, basis_from_segments, theory_basis

N = 2048            # spectral samples
NZ = N // 2         # depth samples of the scan
B, NX = 4, 200
ROW = 400           # surface row
R = 8


def _window():
    return np.hanning(N)


def _reflection(rng, amp_db, ref=1.0):
    """(B, NX, NZ) complex: one specular surface (peak amp_db above `ref`), smooth sub-pixel tilt, a
    laterally coherent phase (the film is one continuous mirror: smooth phase along x, random per
    B-scan) and +-20% amplitude jitter per A-line, as measured on the real FEP film."""
    aw = _window()
    n = np.arange(N)
    out = np.zeros((B, NX, NZ), np.complex64)
    for b in range(B):
        phi0 = 2 * np.pi * rng.uniform()
        for x in range(NX):
            m = ROW + 0.8 * np.sin(2 * np.pi * x / NX)
            a = np.fft.ifft(np.exp(-2j * np.pi * n * m / N) * aw)[:NZ]
            a /= np.abs(a).max()
            amp = ref * 10 ** (amp_db / 20) * (1 + 0.2 * rng.uniform(-1, 1))
            out[b, x] = amp * np.exp(1j * (phi0 + 0.01 * x)) * a
    return out


def _speckle(rng, rows, level=1.0):
    s = (rng.normal(size=(B, NX, NZ)) + 1j * rng.normal(size=(B, NX, NZ))) / np.sqrt(2)
    # axial correlation like a real (windowed) scan
    k = np.hanning(5)
    s = np.apply_along_axis(lambda v: np.convolve(v, k / np.linalg.norm(k), mode="same"), -1, s)
    m = np.zeros(NZ)
    m[rows] = level
    return (s * m).astype(np.complex64)


def _remover(**kw):
    cfg = FEPConfig(**kw)
    U = theory_basis(_window(), cfg.half_window, cfg.rank)
    return FEPRemover(U, np.zeros((B, NX), np.int64), cfg)


def _p(a, rows):
    return 10 * np.log10(np.mean(np.abs(a[:, 20:-20, rows]) ** 2))


@pytest.mark.parametrize("amp_db", [10, 20, 30])
def test_reflection_on_tissue_removed(amp_db):
    rng = np.random.default_rng(amp_db)
    noise = _speckle(rng, slice(0, NZ), 0.03)
    tissue = _speckle(rng, slice(ROW - 3, 700)) + noise
    refl = _reflection(rng, amp_db)
    out = _remover().process(tissue + refl, list(range(B)), 300, 700)
    band = slice(ROW - R, ROW + R + 1)
    before = _p(refl, band) - _p(tissue, band)
    after = _p(out - tissue, band) - _p(tissue, band)
    assert before > amp_db - 12          # band average over 2R+1 rows < peak
    assert after < -6, f"residual {after:.1f} dB rel. tissue (before {before:.1f})"
    # tissue outside the window (+ peak search range) is untouched (bit-exact)
    np.testing.assert_array_equal(out[..., ROW + R + 5:], (tissue + refl)[..., ROW + R + 5:])


def test_no_trench_in_empty_background():
    """Removal must leave the background level, not a dark trench."""
    rng = np.random.default_rng(1)
    noise = _speckle(rng, slice(0, NZ), 0.03)
    out = _remover().process(noise + _reflection(rng, 25, ref=0.03), list(range(B)), 300, 700)
    d = _p(out, slice(ROW - 3, ROW + 4)) - _p(noise, slice(ROW - 3, ROW + 4))
    assert abs(d) < 3, f"surface region {d:+.1f} dB vs background"


def test_tissue_without_reflection_unchanged():
    rng = np.random.default_rng(2)
    tissue = _speckle(rng, slice(ROW - 50, 700)) + _speckle(rng, slice(0, NZ), 0.03)
    out = _remover().process(tissue.copy(), list(range(B)), 300, 700)
    damage = _p(out - tissue, slice(300, 700)) - _p(tissue, slice(300, 700))
    assert damage < -20


def test_basis_from_segments_recovers_psf():
    rng = np.random.default_rng(3)
    refl = _reflection(rng, 30)
    pk = np.argmax(np.abs(refl[0]), axis=-1)
    segs = np.array([refl[0, x, p - R:p + R + 1] for x, p in enumerate(pk)])
    U, captured = basis_from_segments(segs, 3)
    assert U.shape == (2 * R + 1, 3) and captured > 0.99


# --------------------------------------------------------------------------- pipeline
def _pipeline(vol, device, fep):
    import json
    import test_formats as TF
    from octrecon.pipeline import ReconConfig, Reconstructor
    info = json.loads((vol / "synthetic_info.json").read_text())
    cfg = ReconConfig(fep_removal=fep, volume_folder=str(vol), device=device, precision="float32",
                      dispersion_quadratic_term=info.get("dispersion", TF.DISPERSION),
                      focus_positions=float(info["focus_pix"]), focus_sigma=10, crop_z_range_mm="none",
                      output_pixel_size_um=info.get("pixel_um", 2), batch_frames=4, io_threads=2)
    return Reconstructor(cfg, log=lambda *a: None)


@pytest.fixture(scope="module")
def synth_volume(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import test_formats as TF
    return TF._make_tiled(tmp_path_factory.mktemp("fep"), list(TF.PIPELINE_CASES)[0]), TF.REF_PLANE_Y - 1


def _devices():
    devs = ["cpu"]
    try:
        import torch  # noqa: F401
        devs.append("torch-cpu")
    except ImportError:
        pass
    try:
        from octrecon.backend import get_xp
        if get_xp("gpu")[1] == "gpu":                    # CuPy (CUDA)
            devs.append("gpu")
    except Exception:
        pass
    return devs


@pytest.mark.parametrize("device", _devices())
def test_pipeline_with_fep_runs_and_backends_agree(synth_volume, device):
    vol, y = synth_volume
    ref = _pipeline(vol, "cpu", True).process_planes([y])[0].astype(float)
    r = _pipeline(vol, device, True)
    p = r.process_planes([y])[0].astype(float)
    assert r.fep is not None and np.isfinite(p).any()
    ok = np.isfinite(ref) & np.isfinite(p)
    assert np.nanmax(np.abs(p[ok] - ref[ok])) < 0.05      # dB


def test_incoherent_bright_peaks_are_not_removed():
    """Bright PSF-shaped peaks with a random phase per A-line (speckle-like, e.g. a bright tissue
    layer) are not a film: the coherence cap must leave them (and the tissue) in place."""
    rng = np.random.default_rng(5)
    tissue = _speckle(rng, slice(ROW - 3, 700)) + _speckle(rng, slice(0, NZ), 0.03)
    peaks = _reflection(rng, 10)
    peaks *= np.exp(2j * np.pi * rng.uniform(size=(B, NX, 1)))          # destroy lateral coherence
    obs = tissue + peaks
    out = _remover().process(obs.copy(), list(range(B)), 300, 700)
    band = slice(ROW - R, ROW + R + 1)
    changed = _p(out - obs, band) - _p(obs, band)
    assert changed < -10, f"incoherent peaks were removed ({changed:+.1f} dB of the signal changed)"


def test_weak_coherent_film_on_tissue_keeps_tissue():
    rng = np.random.default_rng(6)
    noise = _speckle(rng, slice(0, NZ), 0.03)
    tissue = _speckle(rng, slice(ROW - 3, 700)) + noise
    refl = _reflection(rng, -6)                                         # film weaker than tissue (tile corner)
    out = _remover().process(tissue + refl, list(range(B)), 300, 700)
    band = slice(ROW - R, ROW + R + 1)
    err = _p(out - tissue, band) - _p(tissue, band)
    before = _p(refl, band) - _p(tissue, band)
    assert err < before, f"error {err:+.1f} dB not below the film itself ({before:+.1f} dB)"
    assert err < -10
