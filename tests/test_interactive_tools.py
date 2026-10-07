"""Interactive tools: manual dispersion tuner (Demo_DispersionCorrectionManual) and
Choose Focus Positions (yOCTMeasureFocusDrift).

Synthetic tests always run; the comparison with the legacy MATLAB figures runs when
OCT_TEST_VOLUME points to the 10um_FOV_1 OCTVolume (reference: tests/data/matlab_ref_interactive_10um_FOV_1.npz,
made by validation/matlab/make_interactive_reference.m).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_formats as TF  # noqa: E402
from octrecon.estimation import interactive as I  # noqa: E402
from octrecon.params import load_focus_file  # noqa: E402

VOL = os.environ.get("OCT_TEST_VOLUME")
REF = ROOT / "tests/data/matlab_ref_interactive_10um_FOV_1.npz"


def test_slider_mapping():
    # Demo_DispersionCorrectionManual: beta = sign(v) * 10^|v|, initial v = log10(100)
    assert I.slider_to_beta(2.0) == pytest.approx(100.0)
    assert I.slider_to_beta(-8.0) == pytest.approx(-1e8)
    assert I.slider_to_beta(0.0) == 0.0
    for b in (8.949e7, -1.5e8, 3.2):
        assert I.slider_to_beta(I.beta_to_slider(b)) == pytest.approx(b)


def test_pchip_equispace_is_identity_on_equispaced_k():
    k = np.linspace(2 * np.pi / 800, 2 * np.pi / 1000, 64)
    lam = 2 * np.pi / k
    x = np.random.default_rng(1).normal(size=(3, 64))
    xe, le = I.pchip_equispace(x, lam)
    assert np.allclose(xe, x, atol=1e-12) and np.allclose(le, lam, rtol=1e-12)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return TF._make_tiled(tmp_path_factory.mktemp("itools"), "base")


def test_tools_on_synthetic_volume(synth, tmp_path):
    vi = I.VolumeInfo(synth)
    s = vi.summary()
    assert len(s["tiles"]) == len(vi.si.tiles)
    dt = I.DispersionTool(vi, vi.si.tiles[0].folder, 0)
    lg = dt.ln_image(8.6e7)
    assert lg.shape == (vi.hdr.n_lambda // 2, vi.hdr.size_x)
    png = I.gray_png(lg, *I.DISPERSION_CLIM)
    assert png[:4] == b"\x89PNG"
    ft = I.FocusTool(vi)
    st = ft.setup()
    zi = st["depths"][0]["zi"]
    img = ft.bscan_db(ft.folder_for(zi, st["xi0"]), st["frame_center"], 8.6e7)
    assert img.shape == (ft.n_z, vi.hdr.size_x) and np.isfinite(img).all()
    res = I.finish_focus(ft, [{"zi": zi, "pix": 40, "z_mm": float(ft.z_mm[39])}], tmp_path)
    assert res["focus_positions"] == [40.0] * len(vi.si.z_depths_mm)
    # the saved file is directly usable by the reconstruction
    f = load_focus_file(res["saved"][0])
    assert np.array_equal(f, np.full(len(vi.si.z_depths_mm), 40.0))
    assert (tmp_path / "zChosenFocusPositions.png").exists()
    with pytest.raises(ValueError):
        I.finish_focus(ft, [], tmp_path)


@pytest.mark.skipif(not VOL or not Path(VOL).exists() or not REF.exists(), reason="set OCT_TEST_VOLUME (10um_FOV_1)")
def test_matches_legacy_matlab_figures():
    r = np.load(REF)
    vi = I.VolumeInfo(VOL)
    # (1) manual dispersion tool, exactly as the MATLAB demo (legacy wavelength axis)
    dt = I.DispersionTool(vi, "Data200", 0, legacy_axis=True)
    assert np.abs(dt.lambda_eq - r["disp_lambda_eq"]).max() < 1e-9
    for i, v in enumerate(r["disp_slider"]):
        lg = dt.ln_image(I.slider_to_beta(float(v)))[r["disp_iz"], r["disp_ix"]]
        ref = r["disp_lg"][i]
        fin = np.isfinite(lg) & np.isfinite(ref)
        assert np.abs(lg - ref)[fin].max() < 1e-8
    # (2) default B-scan of the Choose Focus Positions window
    ft = I.FocusTool(vi)
    st = ft.setup()
    assert ft.folder_for(st["depths"][0]["zi"], st["xi0"]) == str(r["focus_folder"])
    assert st["frame_center"] == int(r["focus_frame"])
    img = ft.bscan_db(str(r["focus_folder"]), int(r["focus_frame"]), 8.949e7)
    assert np.abs(img[r["focus_iz"], r["focus_ix"]] - r["focus_db"]).max() < 1e-6
    assert np.allclose(I.matlab_prctile(img, [5, 99.8]), r["focus_base"], atol=1e-6)
    assert np.abs(ft.z_mm - r["focus_z_mm"]).max() < 1e-12
    # (3) drift fit
    foc, d = ft.fit(r["fit_clicks"][:, 0], r["fit_clicks"][:, 1], r["fit_zall"])
    assert np.array_equal(np.asarray(foc, float), r["fit_focus"])
    assert d["driftSlope"] == pytest.approx(float(r["fit_slope"])) and d["tissueRI"] == pytest.approx(float(r["fit_ri"]))
