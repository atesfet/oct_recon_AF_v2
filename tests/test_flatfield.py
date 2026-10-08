"""Tile flat-field (seam) correction and calibrated 2-D outputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octrecon.core.flatfield import apply_flatfield, estimate_flatfield, tile_layout  # noqa: E402
from octrecon.io.tiff_writer import write_2d_calibrated  # noqa: E402

T, NTY, NTX, NZ = 40, 3, 4, 6


def _volume(rng, gain_db_edge=-8.0):
    """Tiles of tissue speckle (planes 2-3) with the same within-tile falloff G0 in every tile,
    on a flat noise floor."""
    ny, nx = NTY * T, NTX * T
    u = (np.arange(T) - (T - 1) / 2) / (T / 2)
    g_line = 10 ** (gain_db_edge / 20 * np.abs(u) ** 2)                 # 1 in the centre, edge -8 dB
    G0 = np.outer(g_line, np.minimum(1, 10 ** (gain_db_edge / 20 * np.maximum(-u, 0) ** 2)) * g_line)
    tissue = np.zeros((ny, NZ, nx))
    tissue[:, 2:4, :] = rng.exponential(1.0, size=(ny, 2, nx))          # speckle-like amplitude
    gain = np.tile(G0, (NTY, NTX))[:, None, :]
    noise = 0.05 * (1 + 0.2 * rng.standard_normal((ny, NZ, nx)))
    amp = noise + gain * tissue
    return (20 * np.log10(amp)).astype(np.float32), tissue, noise, G0


def _tile_mod(vol_db, rows, cols):
    a = 10 ** (vol_db[:, 2:4, :] / 20)
    st = np.stack([a[y0:y1, :, x0:x1].mean(1) for y0, y1 in rows for x0, x1 in cols])
    f = st.mean(0)
    b = f.reshape(T // 8, 8, T // 8, 8).mean((1, 3))           # 8x8 blocks: average the speckle out
    return 20 * np.log10(b.max() / b.min())


def test_flatfield_recovers_and_removes_vignetting():
    rng = np.random.default_rng(0)
    vol, tissue, noise, G0 = _volume(rng)
    rows, cols = tile_layout(vol.shape[0], vol.shape[2], T, T)
    fp = np.ones((vol.shape[0], vol.shape[2]), bool)
    ff = estimate_flatfield(vol, fp, rows, cols, smooth_px=2, smooth_z=0.5, log=lambda m: None)
    G = ff["G"][2]
    r = np.corrcoef(np.log(G).ravel(), np.log(G0 / G0.mean()).ravel())[0, 1]
    assert r > 0.95, f"estimated gain pattern does not match the true one (r = {r:.2f})"
    before = _tile_mod(vol, rows, cols)
    out = vol.copy()
    apply_flatfield(out, ff, rows, cols)
    after = _tile_mod(out, rows, cols)
    assert before > 6 and after < 2.0, f"tile modulation {before:.1f} -> {after:.1f} dB"
    # planes without tissue (noise only) are not amplified at the tile edges
    d = np.abs(out[:, 5, :] - vol[:, 5, :])
    assert np.median(d) < 0.5


def test_flatfield_skips_without_tissue_tiles():
    rng = np.random.default_rng(1)
    vol, *_ = _volume(rng)
    rows, cols = tile_layout(vol.shape[0], vol.shape[2], T, T)
    ff = estimate_flatfield(vol, np.zeros((vol.shape[0], vol.shape[2]), bool), rows, cols, log=lambda m: None)
    assert not ff["applied"] and np.all(ff["G"] == 1)


def test_2d_tiff_calibration(tmp_path):
    md = {"x": {"values": list(np.arange(30) * 0.002 - 0.5), "units": "mm"},
          "y": {"values": list(np.arange(20) * 0.003 + 1.0), "units": "mm"}}      # 2 um x, 3 um y
    img = np.random.default_rng(2).normal(size=(20, 30)).astype(np.float32)
    img[0, 0] = np.nan
    acq = {"patches": {"n_x": 12, "n_y": 9, "patch_fov_mm": [1, 1], "patch_size_px": [500, 500],
                       "patch_step_mm": [1, 1], "patch_overlap_mm": [0, 0]}, "raw": {"oct_system": "gan632"}}
    p = tmp_path / "p.tif"
    write_2d_calibrated(p, img, md, acq, {"value_unit": "dB", "nan": "no tissue"})
    with tifffile.TiffFile(p) as t:
        pg = t.pages[0]
        xr, yr = pg.tags["XResolution"].value, pg.tags["YResolution"].value
        assert pg.tags["ResolutionUnit"].value == 3                      # centimetre
        assert 1e4 * xr[1] / xr[0] == pytest.approx(2.0, rel=1e-6)       # um per pixel
        assert 1e4 * yr[1] / yr[0] == pytest.approx(3.0, rel=1e-6)
        assert t.imagej_metadata["unit"] == "micron"
        assert t.imagej_metadata["oct_patches_x"] == 12 and t.imagej_metadata["oct_value_unit"] == "dB"
        np.testing.assert_array_equal(np.isnan(t.asarray()), np.isnan(img))
    side = json.loads((tmp_path / "p.tif.json").read_text())
    assert side["pixel_size_um"] == {"x": 2.0, "y": 3.0} and side["origin_mm"]["y"] == pytest.approx(1.0)
