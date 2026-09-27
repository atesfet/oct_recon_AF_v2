"""TIFF calibration + acquisition metadata (voxel sizes, patches) and the retag tool."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_formats as TF  # noqa: E402
from octrecon.io.metadata import voxel_size_um  # noqa: E402
from octrecon.pipeline import ReconConfig, Reconstructor  # noqa: E402
from octrecon.retag import retag  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    w = tmp_path_factory.mktemp("tiffmeta")
    vol = TF._make_tiled(w, "base")
    s = Reconstructor(ReconConfig(volume_folder=str(vol), output_root=str(w / "out"), output_name="t",
                                  device="cpu", focus_positions=40, crop_z_range_mm="none"),
                      log=lambda m: None).run()
    return vol, Path(s["tiff"]), s


def _desc_fields(desc):
    return dict(l.split("=", 1) for l in desc.strip().split("\n") if "=" in l)


def test_calibration_tags(run):
    vol, tif, s = run
    with tifffile.TiffFile(tif) as t:
        meta = json.loads(t.pages[0].tags[305].value)
        v = meta["voxel_size_um"]
        # measured from the output axes
        md = meta["metadata"]
        assert v["x"] == pytest.approx((md["x"]["values"][-1] - md["x"]["values"][0]) / (len(md["x"]["values"]) - 1) * 1e3)
        assert v["z"] == pytest.approx((md["z"]["values"][-1] - md["z"]["values"][0]) / (len(md["z"]["values"]) - 1) * 1e3)
        for page in (t.pages[0], t.pages[-1]):          # every page carries the resolution (legacy)
            xr, yr = page.tags[282].value, page.tags[283].value
            assert xr[0] / xr[1] == pytest.approx(1e4 / v["x"])      # pixels per cm
            assert yr[0] / yr[1] == pytest.approx(1e4 / v["z"])
            assert page.tags[296].value == 3                          # centimetre
        f = _desc_fields(t.pages[0].tags[270].value)
        assert f["ImageJ"] and f["unit"] == "micron"
        assert float(f["spacing"]) == pytest.approx(v["y"])
        assert int(f["images"]) == len(t.pages)
        assert int(f["oct_patches_x"]) == 2 and int(f["oct_patches_y"]) == 1
        assert int(f["oct_focus_depths"]) == 2
        assert f["oct_consistency"] == "ok"
        assert "Patches stitched in x: 2" in t.imagej_metadata["Info"]
    acq = s["acquisition"]
    assert acq["patches"]["n_x"] == acq["patches"]["n_x_from_range"] == 2
    assert acq["patches"]["patch_overlap_mm"] == [0.0, 0.0]
    assert acq["consistency_checks"] == ["ok"]
    assert acq["raw"]["native_depth_pixel_um"] == pytest.approx(1.4338, rel=1e-3)
    # legacy readers only need these keys
    assert {"metadata", "clim", "version"} <= set(meta)


def test_retag_legacy_style_file(run, tmp_path):
    vol, tif, _ = run
    # an uncalibrated file like the ones written before this fix
    with tifffile.TiffFile(tif) as t:
        meta = json.loads(t.pages[0].tags[305].value)
        pages = [p.asarray() for p in t.pages]
    old = {"metadata": meta["metadata"], "clim": meta["clim"], "version": 3}
    src = tmp_path / "old.tiff"
    with tifffile.TiffWriter(src, bigtiff=True) as tw:
        for i, p in enumerate(pages):
            tw.write(p, compression="packbits", software=json.dumps(old) if i == 0 else None, metadata=None)
    with tifffile.TiffFile(src) as t:
        assert t.pages[0].tags[296].value == 1                        # no calibration before
    new = retag(src, volume=vol, log=lambda m: None)
    assert new["voxel_size_um"] == voxel_size_um(meta["metadata"])
    with tifffile.TiffFile(src) as t:
        assert t.pages[0].tags[296].value == 3
        f = _desc_fields(t.pages[0].tags[270].value)
        assert int(f["oct_patches_x"]) == 2
        for a, b in zip(pages, t.pages):
            assert np.array_equal(a, b.asarray())                     # pixels untouched
