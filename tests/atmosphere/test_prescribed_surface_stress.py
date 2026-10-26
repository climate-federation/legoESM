"""
Tests for prescribed surface stress / flux overrides in the surface layer.

Covers:
- compute_surface_fluxes: tau/ustar override semantics
- _resolve_prescribed_surface_fluxes: verbatim W/m^2 / Pa transfer, kinematic
  conversion for surface_wth_override, and the conflict ValueError
- PhysicsState: carrying overrides through update_physics_state,
  PHYSSTATE_INPUT_FIELDS membership
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.combined import PhysicsConfig
from legoesm.atmosphere.physics.physics_state import (
    PHYSSTATE_INPUT_FIELDS,
    init_physics_state,
    update_physics_state,
)
from legoesm.atmosphere.physics.turbulence import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.integration import (
    _resolve_prescribed_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.louis import LouisConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes

from legoesm import constants

NCOL = 5


def _bulk_inputs():
    rng = np.random.default_rng(0)
    u = 1.0 + rng.random(NCOL)
    v = -0.5 + 0.3 * rng.random(NCOL)
    T = np.full(NCOL, 290.0)
    q_v = np.full(NCOL, 0.005)
    T_sfc = np.full(NCOL, 295.0)
    q_sfc = np.full(NCOL, 0.008)
    rho = np.full(NCOL, 1.2)
    return u, v, T, q_v, T_sfc, q_sfc, rho


def test_stress_override_replaces_tau_and_sets_ustar():
    u, v, T, q_v, T_sfc, q_sfc, rho = _bulk_inputs()
    cfg = SurfaceLayerConfig(prescribed_tau_x_pa=0.07, prescribed_tau_y_pa=-0.01)
    tau_x, tau_y, _, _, ustar = compute_surface_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, cfg
    )
    np.testing.assert_allclose(tau_x, 0.07)
    np.testing.assert_allclose(tau_y, -0.01)
    expected = np.sqrt(np.sqrt(0.07**2 + 0.01**2) / rho)
    np.testing.assert_allclose(ustar, expected, rtol=1e-10)


def test_stress_override_accepts_column_vector():
    u, v, T, q_v, T_sfc, q_sfc, rho = _bulk_inputs()
    tx = np.linspace(0.01, 0.09, NCOL)
    ty = np.linspace(-0.02, -0.06, NCOL)
    cfg = SurfaceLayerConfig(prescribed_tau_x_pa=tx, prescribed_tau_y_pa=ty)
    tau_x, tau_y, _, _, ustar = compute_surface_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, cfg
    )
    np.testing.assert_allclose(tau_x, tx)
    np.testing.assert_allclose(tau_y, ty)
    np.testing.assert_allclose(ustar, np.sqrt(np.sqrt(tx**2 + ty**2) / rho), rtol=1e-10)


def test_heat_flux_prescription_alone_leaves_momentum_alone():
    u, v, T, q_v, T_sfc, q_sfc, rho = _bulk_inputs()
    cfg_sh = SurfaceLayerConfig(prescribed_shflx_w_m2=25.0)
    tx_ref, ty_ref, _, _, us_ref = compute_surface_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, SurfaceLayerConfig()
    )
    tx, ty, _, _, us = compute_surface_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, cfg_sh
    )
    np.testing.assert_allclose(tx, tx_ref, rtol=1e-12)
    np.testing.assert_allclose(ty, ty_ref, rtol=1e-12)
    np.testing.assert_allclose(us, us_ref, rtol=1e-12)


def _make_phys_state(**kwargs):
    ps = init_physics_state(NCOL, 4, PhysicsConfig())
    return ps._replace(**kwargs)


def test_resolve_puts_overrides_verbatim_in_surface_config():
    nlev = 4
    rho = np.full((NCOL, nlev), 1.15)
    ps = _make_phys_state(
        surface_shflx_override_w_m2=31.0,
        surface_lhflx_override_w_m2=-12.5,
        surface_tau_x_override_pa=0.03,
        surface_tau_y_override_pa=-0.04,
    )
    cfg = _resolve_prescribed_surface_fluxes(LouisConfig(), ps, rho)
    assert cfg.surface.prescribed_shflx_w_m2 == 31.0
    assert cfg.surface.prescribed_lhflx_w_m2 == -12.5
    assert cfg.surface.prescribed_tau_x_pa == 0.03
    assert cfg.surface.prescribed_tau_y_pa == -0.04


def test_resolve_converts_kinematic_wth_with_surface_density():
    nlev = 4
    rho = np.tile(np.linspace(1.0, 1.3, nlev), (NCOL, 1))
    wth = np.full(NCOL, 0.002)  # K m/s
    ps = _make_phys_state(surface_wth_override=wth)
    cfg = _resolve_prescribed_surface_fluxes(LouisConfig(), ps, rho)
    expected = rho[:, -1] * constants.c_pd * wth
    np.testing.assert_allclose(cfg.surface.prescribed_shflx_w_m2, expected, rtol=1e-12)


def test_resolve_raises_on_kinematic_and_watt_conflict():
    nlev = 4
    rho = np.full((NCOL, nlev), 1.1)
    ps = _make_phys_state(
        surface_wth_override=np.full(NCOL, 0.001),
        surface_shflx_override_w_m2=10.0,
    )
    with pytest.raises(ValueError):
        _resolve_prescribed_surface_fluxes(LouisConfig(), ps, rho)


def test_update_physics_state_carries_flux_overrides():
    ps = _make_phys_state(
        surface_shflx_override_w_m2=17.0,
        surface_tau_y_override_pa=-0.02,
    )
    updated = update_physics_state(ps, {"tke": ps.tke})
    assert updated.surface_shflx_override_w_m2 == 17.0
    assert updated.surface_tau_y_override_pa == -0.02
    assert updated.surface_lhflx_override_w_m2 is None
    assert updated.surface_tau_x_override_pa is None


def test_update_physics_state_allows_clearing_override_via_update():
    ps = _make_phys_state(surface_tau_x_override_pa=0.05)
    updated = update_physics_state(ps, {"surface_tau_x_override_pa": 0.09})
    assert updated.surface_tau_x_override_pa == 0.09


def test_flux_anchor_slots_are_declared_inputs_not_physics_memory():
    names = {
        "surface_shflx_override_w_m2",
        "surface_lhflx_override_w_m2",
        "surface_tau_x_override_pa",
        "surface_tau_y_override_pa",
    }
    assert names <= set(PHYSSTATE_INPUT_FIELDS)


def test_zero_prescribed_stress_has_a_finite_ustar_gradient():
    """A calm ERA5 column (|tau| = 0) must not poison the adjoint: the nested
    roots in ustar are 0/0 there without the floor."""
    import jax

    u, v, T, q_v, T_sfc, q_sfc, rho = _bulk_inputs()
    cfg = SurfaceLayerConfig(prescribed_tau_x_pa=0.0, prescribed_tau_y_pa=0.0)

    def ustar_sum(rho_):
        return jnp.sum(compute_surface_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho_, cfg)[4])

    g = jax.grad(ustar_sum)(jnp.asarray(rho))
    assert bool(jnp.all(jnp.isfinite(g)))
