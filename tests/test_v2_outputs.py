"""v2 outputs: tissue-only xy projection, removed FEP signal volume, and the v1 fallback."""
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_formats as TF  # noqa: E402
from octrecon.core.projection import tissue_projection  # noqa: E402
from octrecon.pipeline import ReconConfig, Reconstructor  # noqa: E402


def _run(w, vol, name, **kw):
    cfg = ReconConfig(volume_folder=str(vol), output_root=str(w / "out"), output_name=name, device="cpu",
                      focus_positions=40, crop_z_range_mm="none", **kw)
    return Reconstructor(cfg, log=lambda m: None).run()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    w = tmp_path_factory.mktemp("v2out")
    vol = TF._make_tiled(w, "base")
    return w, _run(w, vol, "v2"), _run(w, vol, "v1", fep_removal=False, xy_projection=False)


def test_v2_files(runs):
    w, s2, _ = runs
    d = Path(s2["output_dir"])
    for f in ("v2.tiff", "v2_xy_mean.tif", "v2_xy_max.tif", "v2_xy_mean.png", "v2_tissue_thickness_um.tif",
              "v2_fep_removed.tiff", "v2_fep_removed_xy.tif", "v2_fep_overview.png"):
        assert (d / f).exists(), f
    ny, nz, nx = s2["output_shape_yzx"]
    with tifffile.TiffFile(d / "v2_xy_mean.tif") as t:
        img = t.asarray()
        assert img.shape == (ny, nx) and img.dtype == np.float32
        xr = t.pages[0].tags["XResolution"].value
        assert xr[1] / xr[0] == pytest.approx(s2["voxel_size_um"]["x"], rel=1e-4)
    with tifffile.TiffFile(d / "v2_fep_removed.tiff") as t:
        assert len(t.pages) == ny


def test_both_off_is_v1(runs):
    """fep_removal=False + xy_projection=False writes exactly the v1 file set."""
    _, _, s1 = runs
    names = sorted(p.name for p in Path(s1["output_dir"]).iterdir())
    assert names == sorted(["v1.tiff", "v1.tiff.json", "v1_config.json", "v1_run_summary.json"]) or \
        set(names) <= {"v1.tiff", "v1.tiff.json", "v1_config.json", "v1_run_summary.json", "v1.log"}
    assert s1["fep_removal"] is None and "xy_projection" not in s1


def test_tissue_projection_synthetic():
    """A bright slab (tissue) at known depths inside a noise volume: footprint, slab and projection."""
    rng = np.random.default_rng(0)
    ny, nz, nx = 120, 30, 160
    vol = (-30 + 3 * rng.standard_normal((ny, nz, nx))).astype(np.float32)
    yy, xx = np.mgrid[:ny, :nx]
    disk = (yy - 60) ** 2 + (xx - 80) ** 2 < 45 ** 2
    tissue = np.zeros((ny, nz, nx), bool)
    tissue[:, 10:20, :] = disk[:, None, :]
    vol[tissue] = -5 + 3 * rng.standard_normal(tissue.sum())
    P = tissue_projection(vol, 2.0, 2.0, smooth_um=6, min_area_mm2=1e-4, log=lambda m: None)
    fp = P["footprint"]
    assert (fp & disk).sum() / disk.sum() > 0.95 and (fp & ~disk).sum() / (~disk).sum() < 0.05
    assert np.nanmedian(P["thickness_um"][fp]) == pytest.approx(20, abs=4)
    assert np.isnan(P["mean"][~fp]).all()
    inner = disk & fp
    assert np.nanmedian(P["mean"][inner]) > -10       # projection averages tissue only, not the noise


def test_reproject_from_kept_volume(runs, tmp_path):
    """`python -m octrecon project`: same projection as the run, from the kept float volume."""
    from octrecon.outputs_v2 import reproject
    w, _, _ = runs
    vol = TF._make_tiled(tmp_path, "base")
    s = _run(tmp_path, vol, "k", keep_float_volume=True)
    d = Path(s["output_dir"])
    before = tifffile.imread(d / "k_xy_mean.tif")
    r = reproject(d, log=lambda m: None)
    after = tifffile.imread(d / "k_xy_mean.tif")
    np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
    np.testing.assert_allclose(before[np.isfinite(before)], after[np.isfinite(after)])
    assert "xy_projection" in r and (d / "k_fep_overview.png").exists()
