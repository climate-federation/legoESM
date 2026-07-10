"""Tests for the E_LUC land-use-change carbon bookkeeping.

The correctness anchor is CARBON CONSERVATION: the bookkeeping never creates or
destroys carbon, so the exact accounting identity

    cum_eluc + (product pools) - regrow_debt == (1 - slash) * cleared_C - gained_C

holds at every step (not just at equilibrium), to ~1e-9.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.land.surface_params import N_PFT_CLM5
from legoesm.land.land_use_change import (
    ELUCState,
    LandUseChangeConfig,
    annual_eluc_series,
    default_veg_carbon_density,
    eluc_step,
    init_eluc_state,
    validate_luc_config,
)

_FOREST, _CROP = 1, 15          # two PFT slots with very different biomass
_CFG = LandUseChangeConfig(scheme="bookkeeping")


def _frac(idx):
    f = np.zeros((1, N_PFT_CLM5)); f[0, idx] = 1.0
    return jnp.asarray(f)


def _veg(dens):
    v = np.zeros(N_PFT_CLM5)
    for i, d in dens.items():
        v[i] = d
    return jnp.asarray(v)


def _pool_sum(st):
    return float(st.prod_1yr[0] + st.prod_10yr[0] + st.prod_100yr[0])


def _run(cfg, frac0, frac1, veg_ref, n):
    """Apply a one-step cover pulse (frac0->frac1) then n-1 quiet steps."""
    st = init_eluc_state(1)
    st, _ = eluc_step(st, frac0, frac1, veg_ref, cfg, 1.0)
    for _ in range(n - 1):
        st, _ = eluc_step(st, frac1, frac1, veg_ref, cfg, 1.0)
    return st


def test_exact_carbon_conservation_identity():
    """cum_eluc + pools - regrow_debt == (1-slash)*cleared - gained, every step."""
    Df, Dc = 10000.0, 400.0
    veg = _veg({_FOREST: Df, _CROP: Dc})
    slash = _CFG.clear_slash_frac
    expected = (1.0 - slash) * Df - Dc            # forest cleared, crop gained
    for n in (1, 3, 25, 400):
        st = _run(_CFG, _frac(_FOREST), _frac(_CROP), veg, n)
        lhs = float(st.cum_eluc[0]) + _pool_sum(st) - float(st.regrow_debt[0])
        np.testing.assert_allclose(lhs, expected, rtol=1e-9, atol=1e-6)


def test_deforestation_drains_to_committed_emission():
    """Forest -> bare: cumulative E_LUC -> (1-slash)*forest_C, pools -> 0."""
    Df = 10000.0
    veg = _veg({_FOREST: Df})                     # bare (idx0) density 0
    st = _run(_CFG, _frac(_FOREST), _frac(0), veg, 2000)   # >> 10*tau_100
    np.testing.assert_allclose(float(st.cum_eluc[0]),
                               (1.0 - _CFG.clear_slash_frac) * Df, rtol=1e-4)
    assert _pool_sum(st) < 1e-3                    # products drained


def test_afforestation_is_a_sink():
    """Bare -> forest: E_LUC is negative (uptake); regrow debt drains to 0."""
    Df = 10000.0
    veg = _veg({_FOREST: Df})
    st = _run(_CFG, _frac(0), _frac(_FOREST), veg, 2000)
    np.testing.assert_allclose(float(st.cum_eluc[0]), -Df, rtol=1e-4)
    assert float(st.regrow_debt[0]) < 1e-3
    assert float(st.cum_eluc[0]) < 0.0             # a sink


def test_deforestation_is_a_source_first_step():
    veg = _veg({_FOREST: 10000.0})
    _, eluc = eluc_step(init_eluc_state(1), _frac(_FOREST), _frac(0), veg, _CFG, 1.0)
    assert float(eluc[0]) > 0.0                     # source to atmosphere


def test_no_cover_change_no_flux():
    veg = default_veg_carbon_density()
    st0 = init_eluc_state(1)
    st1, eluc = eluc_step(st0, _frac(_FOREST), _frac(_FOREST), veg, _CFG, 1.0)
    np.testing.assert_allclose(float(eluc[0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st1.cum_eluc), 0.0, atol=1e-12)


def test_product_pools_never_negative():
    veg = _veg({_FOREST: 10000.0})
    st = _run(_CFG, _frac(_FOREST), _frac(0), veg, 500)
    for p in (st.prod_1yr, st.prod_10yr, st.prod_100yr, st.regrow_debt):
        assert float(p[0]) >= 0.0


# --- config validation (dispatch discipline) ---
def test_validate_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown land-use-change scheme"):
        validate_luc_config(LandUseChangeConfig(scheme="bogus"))


def test_validate_scheme_none_ok():
    validate_luc_config(LandUseChangeConfig(scheme="none"))     # no raise


def test_validate_burn_plus_slash_over_one_raises():
    with pytest.raises(ValueError, match="clear_burn_frac \\+ clear_slash_frac"):
        validate_luc_config(LandUseChangeConfig(
            scheme="bookkeeping", clear_burn_frac=0.7, clear_slash_frac=0.5))


def test_validate_product_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_luc_config(LandUseChangeConfig(
            scheme="bookkeeping", prod_frac_1yr=0.5, prod_frac_10yr=0.5, prod_frac_100yr=0.5))


def test_validate_negative_product_fraction_raises():
    # Sums to 1 but a component is negative -> would drive a pool negative.
    with pytest.raises(ValueError, match="prod_frac_1yr must be >= 0"):
        validate_luc_config(LandUseChangeConfig(
            scheme="bookkeeping", prod_frac_1yr=-1.0, prod_frac_10yr=2.0, prod_frac_100yr=0.0))


def test_validate_nonpositive_tau_raises():
    with pytest.raises(ValueError, match="tau_10yr_years must be > 0"):
        validate_luc_config(LandUseChangeConfig(scheme="bookkeeping", tau_10yr_years=0.0))


def test_annual_eluc_series_deforestation_pulse():
    """3-year cover [forest, forest, bare]: E_LUC is 0, 0, then a source (PgC/yr)."""
    Df = 10000.0
    veg = _veg({_FOREST: Df})
    f_forest, f_bare = np.asarray(_frac(_FOREST)), np.asarray(_frac(0))
    pft = np.stack([f_forest, f_forest, f_bare], axis=0)          # (3, 1, npft)
    area = jnp.asarray([1.0e10])
    eluc_pgc, st = annual_eluc_series(pft, area, _CFG, veg_ref=veg)

    assert eluc_pgc.shape == (3,)
    assert float(eluc_pgc[0]) == 0.0                              # no prior year
    np.testing.assert_allclose(float(eluc_pgc[1]), 0.0, atol=1e-12)  # no cover change
    assert float(eluc_pgc[2]) > 0.0                              # deforestation = source


def test_annual_eluc_series_no_change_is_zero():
    veg = default_veg_carbon_density()
    f = np.asarray(_frac(_FOREST))
    pft = np.stack([f, f, f], axis=0)
    eluc_pgc, _ = annual_eluc_series(pft, jnp.asarray([1.0e10]), _CFG, veg_ref=veg)
    np.testing.assert_allclose(np.asarray(eluc_pgc), 0.0, atol=1e-12)


def test_annual_eluc_series_ocean_nan_area_ignored():
    """A NaN-area (ocean) cell contributes nothing to the global total."""
    Df = 10000.0
    veg = _veg({_FOREST: Df})
    f_forest = np.asarray(_frac(_FOREST))[0]
    f_bare = np.asarray(_frac(0))[0]
    # 2 cells: col0 deforests, col1 is ocean (NaN cover + NaN area).
    land = np.stack([f_forest, f_bare], axis=0)                   # (2, npft) forest->bare
    nan = np.full_like(f_forest, np.nan)
    pft = np.stack([np.stack([f_forest, nan]),                    # year0
                    np.stack([f_bare, nan])], axis=0)             # year1 -> (2,2,npft)
    area = jnp.asarray([1.0e10, np.nan])
    eluc_pgc, _ = annual_eluc_series(pft, area, _CFG, veg_ref=veg)
    # Only the land cell's deforestation flux survives (ocean NaN -> 0), finite.
    assert np.isfinite(float(eluc_pgc[1])) and float(eluc_pgc[1]) > 0.0


def test_default_veg_density_shape_and_ordering():
    v = np.asarray(default_veg_carbon_density())
    assert v.shape == (N_PFT_CLM5,)
    assert v[0] == 0.0                              # bare soil
    assert v[_FOREST] > v[_CROP] > 0.0              # forest >> crop biomass
