"""Direct tests for SWG native-C-grid coarse-graining."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity import swg_coarse_graining as coarse
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
)


def test_masked_gaussian_preserves_constant_and_handles_walls(monkeypatch):
    calls = 0
    real_filter = coarse.gaussian_filter

    def counted_filter(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(coarse, "gaussian_filter", counted_filter)
    mask = np.zeros((41, 41), dtype=bool)
    mask[5:36, 7:34] = True
    field = np.where(mask, 3.25, 1.0e20)
    filtered = coarse.masked_gaussian_filter(field, mask, sigma_cells=4.0)
    assert calls == 2
    assert np.allclose(filtered[mask], 3.25, rtol=0.0, atol=2.0e-15)
    assert np.all(filtered[~mask] == 0.0)
    # The planted unnormalised wall treatment cannot preserve this constant.
    planted = coarse.gaussian_filter(
        np.where(mask, 3.25, 0.0), 4.0, mode="constant", cval=0.0)
    assert np.max(np.abs(planted[mask] - 3.25)) > 0.1


def test_gaussian_attenuates_plane_wave_by_analytic_factor():
    n = 512
    sigma = 8.0
    mode = 4
    x = np.arange(n, dtype=np.float64)
    wave = np.cos(2.0 * np.pi * mode * x / n)[None, :] * np.ones((n, 1))
    mask = np.ones((n, n), dtype=bool)
    filtered = coarse.masked_gaussian_filter(wave, mask, sigma_cells=sigma)
    interior = np.s_[48:-48, 48:-48]
    measured = np.sum(filtered[interior] * wave[interior]) / np.sum(
        wave[interior] ** 2)
    expected = np.exp(-0.5 * (sigma * 2.0 * np.pi * mode / n) ** 2)
    assert np.isclose(measured, expected, rtol=2.0e-5, atol=2.0e-5)


def test_round1_diagnostic_is_explicitly_distinct_from_conservative_target():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    rng = np.random.default_rng(2)
    constants = {kind: np.full(source["mask"][kind].shape, 7.0)
                 for kind in ("T", "U", "V")}
    constant_out = coarse.coarse_grain_cgrid(
        constants, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    for kind in ("T", "U", "V"):
        assert np.allclose(constant_out[kind][target["mask"][kind]], 7.0,
                           rtol=0.0, atol=2.0e-15)
    fields = {kind: rng.normal(size=source["mask"][kind].shape)
              for kind in ("T", "U", "V")}
    diagnostic = coarse.coarse_grain_cgrid(
        fields, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    conservative = coarse.conservative_restrict_cgrid(fields, source, target)
    for kind in ("T", "U", "V"):
        source_mean = np.mean(fields[kind][source["mask"][kind]], dtype=np.float64)
        target_mean = np.mean(
            diagnostic[kind][target["mask"][kind]], dtype=np.float64)
        assert np.isclose(source_mean, target_mean, rtol=0.0, atol=5.0e-16)
        assert np.all(diagnostic[kind][~target["mask"][kind]] == 0.0)
    assert any(not np.array_equal(diagnostic[kind], conservative[kind])
               for kind in ("T", "U", "V"))


def _synthetic_state(source, seed=8):
    rng = np.random.default_rng(seed)
    return {
        kind: rng.normal(scale=0.01, size=source["mask"][kind].shape)
        for kind in ("T", "U", "V")
    }


def test_conservative_restriction_preserves_volume_and_continuity_at_walls():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(source)
    output = coarse.conservative_restrict_cgrid(fields, source, target)
    source_wet = source["mask"]["T"]
    target_wet = target["mask"]["T"]
    source_volume = np.sum(
        source["area"]["T"][source_wet]
        * (source["reference_thickness"]["T"][source_wet]
           + fields["T"][source_wet]), dtype=np.float64)
    target_volume = np.sum(
        target["area"]["T"][target_wet]
        * (target["reference_thickness"]["T"][target_wet]
           + output["T"][target_wet]), dtype=np.float64)
    assert np.isclose(source_volume, target_volume, rtol=2.0e-16, atol=0.0)
    result = coarse.require_continuity_commutes(
        fields, source, target, atol_m_s=1.0e-15)
    assert result["interior"]["max_abs_m_s"] <= 1.0e-15
    assert result["outer_four_cells"]["max_abs_m_s"] <= 1.0e-15


def test_analytic_cgrid_face_staggering_selects_tiling_faces():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    rows, columns = np.indices(source["mask"]["T"].shape)
    fields = {
        "T": np.zeros_like(rows, dtype=np.float64),
        "U": rows.astype(np.float64) + 0.01 * columns,
        "V": 2.0 * columns.astype(np.float64) + 0.01 * rows,
    }
    output = coarse.conservative_restrict_cgrid(fields, source, target)
    # First coarse east face tiles fine native-U column 9 over T rows 2..9.
    assert np.isclose(output["U"][1, 1], np.mean(fields["U"][2:10, 9]),
                      rtol=0.0, atol=2.0e-15)
    # First coarse north face tiles fine native-V row 9 over T columns 2..9.
    assert np.isclose(output["V"][1, 1], np.mean(fields["V"][9, 2:10]),
                      rtol=0.0, atol=2.0e-15)
    # Adjacent-face plants are deliberately distinguishable.
    assert output["U"][1, 1] != np.mean(fields["U"][2:10, 8])
    assert output["V"][1, 1] != np.mean(fields["V"][8, 2:10])


def test_gaussian_vorticity_path_spreads_impulse_without_mean_patch():
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    wet = target["mask"]["T"]
    field = np.zeros(wet.shape, dtype=np.float64)
    at = tuple(np.argwhere(wet)[0])
    field[at] = 1.0
    smoothed = coarse.gaussian_smooth_restricted_scalar(
        field, wet, sigma_cells=1.0)
    assert 0.0 < smoothed[at] < 1.0
    assert np.count_nonzero(smoothed[wet]) > 1
    assert np.isclose(np.sum(smoothed[wet]), 1.0, rtol=0.0, atol=2.0e-15)
    fields = {"T": field, "U": np.zeros_like(field), "V": np.zeros_like(field)}
    u_at = tuple(np.argwhere(target["mask"]["U"])[100])
    fields["U"][u_at] = 1.0
    raw = coarse.restricted_relative_vorticity(fields, target)
    filtered = coarse.restricted_relative_vorticity(
        fields, target, gaussian_sigma_cells=1.0)
    assert raw.shape == filtered.shape == wet.shape
    assert not np.array_equal(raw, filtered)


def test_planted_round1_nonconservative_state_is_refused():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(source, seed=9)
    planted = coarse.coarse_grain_cgrid(
        fields, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    with pytest.raises(ValueError, match="non-conservative restriction"):
        coarse.require_continuity_commutes(
            fields, source, target, atol_m_s=1.0e-15,
            coarse_fields=planted)


_REAL_FINE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/"
    "legoesm_runs/dg_gpu_1095d_daily_mean")


@pytest.mark.skipif(not (_REAL_FINE_ROOT / "manifest.json").is_file(),
                    reason="complete three-year fine run not mounted")
def test_real_day1095_continuity_commutes_interior_and_walls():
    manifest_path = _REAL_FINE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = "day1095.npz"
    path = _REAL_FINE_ROOT / name
    assert manifest["status"] == "complete"
    assert manifest["daily_sha256"][name] == coarse.sha256(path)
    report = coarse.audit_snapshot(
        path, manifest_path, commit="a" * 40)
    assert report["status"] == "PASS"
    assert report["regions"]["interior"]["max_abs_m_s"] <= 1.0e-15
    assert report["regions"]["outer_four_cells"]["max_abs_m_s"] <= 1.0e-15
