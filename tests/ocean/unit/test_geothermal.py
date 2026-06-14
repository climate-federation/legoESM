"""Unit tests for the geothermal bottom heat-flux BC (NEMO ln_trabbc).

Exercise the leaf tendency ``geothermal_bottom_heating_tendency`` directly:
deepest-wet-cell-only heating, the boundary heat-deposit identity
(rho_0 c_sw dz_bottom dT == flux dt), land/dry -> zero, per-column flux fields,
and JIT + autodiff safety.  Plus the ``apply_geothermal_step`` wrapper's
enabled/disabled behaviour.
"""
from __future__ import annotations

import collections

import jax

jax.config.update("jax_enable_x64", True)  # sci test: float64 deposit identities

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.coupler.geothermal_apply import apply_geothermal_step
from legoesm.ocean.physics.geothermal import (
    GeothermalConfig,
    _GEOTHERMAL_FLUX_MEAN_WM2,
    geothermal_bottom_heating_tendency,
)

RHO = constants.rho_ocean
CP = constants.c_sw


def _column(nlev=5, wet_levels=3, dz=100.0):
    """One column: ``wet_levels`` wet cells of thickness ``dz`` then dry rock."""
    dz_live = np.where(np.arange(nlev) < wet_levels, dz, 0.0)
    wet = (np.arange(nlev) < wet_levels).astype(float)
    return dz_live[None, :], wet[None, :]   # (1, nlev)


def test_heats_only_deepest_wet_cell():
    dz, wet = _column(nlev=5, wet_levels=3)
    dTdt = np.asarray(geothermal_bottom_heating_tendency(
        dz, wet, flux_wm2=0.0864))
    # Non-zero only at level 2 (deepest wet); zero above and in dry rock below.
    assert dTdt[0, 2] > 0
    assert np.allclose(dTdt[0, [0, 1, 3, 4]], 0.0)


def test_boundary_heat_deposit_identity():
    # The column heat gained equals flux*dt exactly (single-cell deposit):
    #   rho_0 * c_sw * dz_bottom * (dTdt*dt) == flux * dt.
    flux = 0.12
    dz_bottom = 80.0
    dz, wet = _column(nlev=4, wet_levels=2, dz=dz_bottom)
    dTdt = np.asarray(geothermal_bottom_heating_tendency(dz, wet, flux_wm2=flux))
    deposited_per_s = RHO * CP * dz_bottom * dTdt[0, 1]      # W/m^2
    np.testing.assert_allclose(deposited_per_s, flux, rtol=1e-12)


def test_partial_bottom_cell_warms_more():
    # A THIN partial bottom cell warms faster than a thick one (same flux).
    dz_thin = np.array([[100.0, 10.0, 0.0]])
    dz_thick = np.array([[100.0, 200.0, 0.0]])
    wet = np.array([[1.0, 1.0, 0.0]])
    t_thin = np.asarray(geothermal_bottom_heating_tendency(dz_thin, wet, flux_wm2=0.1))
    t_thick = np.asarray(geothermal_bottom_heating_tendency(dz_thick, wet, flux_wm2=0.1))
    assert t_thin[0, 1] > t_thick[0, 1]


def test_interior_dry_gap_heats_only_single_deepest_wet():
    # Pathological column with an interior dry gap wet=[1,0,1]: ONLY the deepest
    # wet cell (level 2) is heated -- never two deposits (codex one-hot bug).
    dz = np.array([[100.0, 0.0, 60.0]])
    wet = np.array([[1.0, 0.0, 1.0]])
    dTdt = np.asarray(geothermal_bottom_heating_tendency(dz, wet, flux_wm2=0.1))
    assert dTdt[0, 2] > 0
    assert dTdt[0, 0] == 0.0 and dTdt[0, 1] == 0.0
    # Exactly one cell heated -> total deposit == flux (not 2x).
    np.testing.assert_allclose(RHO * CP * 60.0 * dTdt[0, 2], 0.1, rtol=1e-12)


def test_negative_constant_flux_rejected():
    import collections
    _State = collections.namedtuple("_State", ["T"])
    state = _State(T=Field(jnp.asarray(np.array([[[10.0, 8.0]]])), name="T",
                           dims=("y", "x", "z"), units="degC"))
    with pytest.raises(ValueError):
        apply_geothermal_step(
            state, dz_live=np.array([[[100.0, 100.0]]]),
            wet_cell=np.array([[[1.0, 1.0]]]), dt=600.0,
            config=GeothermalConfig(enabled=True, flux_wm2=-0.1))


def test_dry_and_land_columns_get_zero():
    # A fully dry column (no wet cells) receives no heat anywhere.
    dz = np.zeros((1, 4))
    wet = np.zeros((1, 4))
    dTdt = np.asarray(geothermal_bottom_heating_tendency(dz, wet, flux_wm2=0.0864))
    assert np.allclose(dTdt, 0.0)


def test_per_column_flux_field():
    # Two columns with different per-column flux; bottom-cell heating scales.
    dz = np.array([[100.0, 100.0], [100.0, 100.0]])
    wet = np.array([[1.0, 1.0], [1.0, 1.0]])
    flux = np.array([0.05, 0.20])                 # (ncells,)
    dTdt = np.asarray(geothermal_bottom_heating_tendency(dz, wet, flux_wm2=flux))
    ratio = dTdt[1, 1] / dTdt[0, 1]
    np.testing.assert_allclose(ratio, 0.20 / 0.05, rtol=1e-12)


def test_bad_flux_shape_raises():
    dz = np.ones((3, 4)); wet = np.ones((3, 4))
    with pytest.raises(ValueError):
        geothermal_bottom_heating_tendency(dz, wet, flux_wm2=np.ones((2, 2, 2)))


def test_jit_and_grad_safe():
    dz, wet = _column(nlev=5, wet_levels=3)
    f = jax.jit(lambda flux: jnp.sum(
        geothermal_bottom_heating_tendency(dz, wet, flux_wm2=flux)))
    val = float(f(0.0864))
    assert np.isfinite(val) and val > 0
    # d(sum dTdt)/d(flux) = 1/(rho cp dz_bottom) (only the bottom cell depends
    # on flux), a clean nonzero gradient -> differentiable wrt the flux.
    g = float(jax.grad(f)(0.0864))
    np.testing.assert_allclose(g, 1.0 / (RHO * CP * 100.0), rtol=1e-9)


def test_default_flux_is_nemo_constant():
    assert _GEOTHERMAL_FLUX_MEAN_WM2 == pytest.approx(0.0864)
    assert GeothermalConfig().flux_wm2 == pytest.approx(0.0864)


def test_apply_step_disabled_is_noop():
    _State = collections.namedtuple("_State", ["T"])
    T0 = np.array([[[10.0, 8.0, 6.0]]])
    state = _State(T=Field(jnp.asarray(T0), name="T", dims=("y", "x", "z"),
                           units="degC"))
    dz = np.array([[[100.0, 100.0, 100.0]]])
    wet = np.array([[[1.0, 1.0, 1.0]]])
    out = apply_geothermal_step(state, dz_live=dz, wet_cell=wet, dt=600.0,
                                config=GeothermalConfig(enabled=False))
    np.testing.assert_array_equal(np.asarray(out.T.data), T0)


def test_apply_step_enabled_warms_bottom_only():
    _State = collections.namedtuple("_State", ["T"])
    T0 = np.array([[[10.0, 8.0, 6.0]]])
    state = _State(T=Field(jnp.asarray(T0), name="T", dims=("y", "x", "z"),
                           units="degC"))
    dz = np.array([[[100.0, 100.0, 50.0]]])
    wet = np.array([[[1.0, 1.0, 1.0]]])
    dt = 3600.0
    out = apply_geothermal_step(
        state, dz_live=dz, wet_cell=wet, dt=dt,
        config=GeothermalConfig(enabled=True, flux_wm2=0.0864))
    Tn = np.asarray(out.T.data)
    assert Tn[0, 0, 2] > T0[0, 0, 2]                       # bottom warmed
    np.testing.assert_array_equal(Tn[0, 0, :2], T0[0, 0, :2])  # above unchanged
    dT = 0.0864 * dt / (RHO * CP * 50.0)
    np.testing.assert_allclose(Tn[0, 0, 2], 6.0 + dT, rtol=1e-12)
