"""Unit tests for the canonical Haney surface-tracer relaxation kernel
(``legoesm.ocean.forcing.surface_relaxation``) used by the OMIP restoring
path and the coupled 3D-ocean WOA spin-up anchor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.forcing.surface_relaxation import (
    relax_surface_tracers,
    apply_surface_relaxation_step,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def test_relax_moves_toward_target_by_alpha():
    """One step relaxes by exactly fraction alpha toward the target."""
    T = jnp.array([[10.0, 10.0]])
    S = jnp.array([[34.0, 34.0]])
    T_tgt = jnp.array([[20.0, 20.0]])
    S_tgt = jnp.array([[35.0, 35.0]])
    mask = jnp.ones_like(T)
    T_new, S_new = relax_surface_tracers(T, S, T_tgt, S_tgt, 0.25, 0.5, mask)
    # T: 10 - 0.25*(10-20) = 12.5 ; S: 34 - 0.5*(34-35) = 34.5
    np.testing.assert_allclose(np.asarray(T_new), 12.5)
    np.testing.assert_allclose(np.asarray(S_new), 34.5)


def test_relax_land_masked_untouched():
    """Dry cells (mask=0) are left exactly unchanged."""
    T = jnp.array([[10.0, 10.0]])
    S = jnp.array([[34.0, 34.0]])
    T_tgt = jnp.array([[20.0, 20.0]])
    S_tgt = jnp.array([[35.0, 35.0]])
    mask = jnp.array([[1.0, 0.0]])
    T_new, S_new = relax_surface_tracers(T, S, T_tgt, S_tgt, 0.5, 0.5, mask)
    assert float(T_new[0, 1]) == 10.0   # land untouched
    assert float(S_new[0, 1]) == 34.0
    assert float(T_new[0, 0]) == 15.0   # ocean relaxed


def test_relax_alpha_zero_is_identity():
    T = jnp.array([[10.0]])
    S = jnp.array([[34.0]])
    T_new, S_new = relax_surface_tracers(
        T, S, jnp.array([[20.0]]), jnp.array([[35.0]]), 0.0, 0.0, jnp.ones_like(T))
    assert float(T_new[0, 0]) == 10.0 and float(S_new[0, 0]) == 34.0


def _state():
    grid = create_latlon_grid(n_lat=12, n_lon=24)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=2.0, S_uniform=34.0)
    return state


def test_apply_step_only_touches_surface_layer():
    """apply_surface_relaxation_step changes only the top level; subsurface and
    the salinity/temperature deep columns are bit-unchanged."""
    state = _state()
    T0 = np.asarray(state.T.data).copy()
    S0 = np.asarray(state.S.data).copy()
    T_tgt = jnp.full(T0.shape[:2], 25.0)     # warm surface target
    S_tgt = jnp.full(S0.shape[:2], 36.0)
    new = apply_surface_relaxation_step(
        state, T_target=T_tgt, S_target=S_tgt,
        dt=3600.0, tau_T_s=30.0 * 86400.0, tau_S_s=0.0,
    )
    Tn = np.asarray(new.T.data)
    Sn = np.asarray(new.S.data)
    # Subsurface temperature unchanged; surface moved toward the warm target.
    np.testing.assert_array_equal(Tn[..., 1:], T0[..., 1:])
    assert np.all(Tn[..., 0] >= T0[..., 0])          # warmed toward 25C
    assert np.any(Tn[..., 0] > T0[..., 0])
    # tau_S_s=0 => salinity untouched everywhere.
    np.testing.assert_array_equal(Sn, S0)


def test_apply_step_disabled_is_byte_identical():
    """Both timescales 0 => the state object is returned unchanged."""
    state = _state()
    T_tgt = jnp.full(state.T.data.shape[:2], 25.0)
    S_tgt = jnp.full(state.S.data.shape[:2], 36.0)
    new = apply_surface_relaxation_step(
        state, T_target=T_tgt, S_target=S_tgt,
        dt=3600.0, tau_T_s=0.0, tau_S_s=0.0,
    )
    assert new is state


def test_apply_step_is_differentiable():
    """The relaxation is AD-safe: grad of a surface-T loss wrt the target is
    finite (the coupled training path must stay differentiable)."""
    state = _state()
    S_tgt = jnp.full(state.S.data.shape[:2], 36.0)

    def loss(T_tgt):
        new = apply_surface_relaxation_step(
            state, T_target=T_tgt, S_target=S_tgt,
            dt=3600.0, tau_T_s=30.0 * 86400.0, tau_S_s=30.0 * 86400.0,
        )
        return jnp.sum(new.T.data[..., 0] ** 2)

    T_tgt0 = jnp.full(state.T.data.shape[:2], 25.0)
    g = jax.grad(loss)(T_tgt0)
    assert np.all(np.isfinite(np.asarray(g)))
