"""Direct tests for SWG native-C-grid coarse-graining."""
from __future__ import annotations

import numpy as np

from legoesm.ocean.fidelity import swg_coarse_graining as coarse
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
)


def test_masked_gaussian_preserves_constant_and_handles_walls():
    mask = np.zeros((41, 41), dtype=bool)
    mask[5:36, 7:34] = True
    field = np.where(mask, 3.25, 1.0e20)
    filtered = coarse.masked_gaussian_filter(field, mask, sigma_cells=4.0)
    assert np.array_equal(filtered[mask], np.full(np.count_nonzero(mask), 3.25))
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


def test_actual_cgrid_constant_and_basin_means_are_preserved():
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
    output = coarse.coarse_grain_cgrid(
        fields, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    for kind in ("T", "U", "V"):
        source_mean = np.mean(fields[kind][source["mask"][kind]], dtype=np.float64)
        target_mean = np.mean(output[kind][target["mask"][kind]], dtype=np.float64)
        assert np.isclose(source_mean, target_mean, rtol=0.0, atol=5.0e-16)
        assert np.all(output[kind][~target["mask"][kind]] == 0.0)
