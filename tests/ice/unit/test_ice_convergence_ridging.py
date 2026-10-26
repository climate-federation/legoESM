"""Selectable convergence-only ridging: analytic signs, live fold and real step.

This tests a subset, not SI3 equivalence. Run with JAX_PLATFORMS=cpu and
JAX_ENABLE_X64=1. No duplicated divergence or redistribution implementation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import FoldDescriptor, create_latlon_geometry, create_latlon_grid
from legoesm.ice import (
    SeaIceConfig, distribute_dynamic_state_to_categories,
    init_dynamic_ice_state, step_sea_ice,
)
from legoesm.ice.config import BrineConfig, RidgingConfig
from legoesm.ice.sea_ice import _closing_rate_from_velocity
from tests.unit.test_run_omip_core2_ice_categories import _min_forcing


@pytest.fixture
def geometry():
    return create_latlon_geometry(8, 12)


def _closing(u, v, grid, cap=1.0):
    return _closing_rate_from_velocity(
        u, v, grid, cap=cap, cs_shear=RidgingConfig().cs_shear_ridging,
        closing_scheme="convergence")


@pytest.mark.parametrize("rotated", [False, True])
def test_zonal_analytic_sign_and_cap(geometry, rotated):
    """Centered face average of sin(lon) has known discrete derivative.

    A 90-degree grid rotation makes geographic north drive grid-i flow;
    omitting rotation, negating divergence, or dropping either clip fails.
    """
    g = geometry
    wave = jnp.broadcast_to(jnp.sin(g.lon), (8, 12))
    zero = jnp.zeros_like(wave)
    if rotated:
        g = g._replace(cos_alpha_u=jnp.zeros_like(g.cos_alpha_u),
                       sin_alpha_u=jnp.ones_like(g.sin_alpha_u),
                       cos_alpha_v=jnp.zeros_like(g.cos_alpha_v),
                       sin_alpha_v=jnp.ones_like(g.sin_alpha_v))
        u, v = zero, wave
    else:
        u, v = wave, zero
    # The face-averaged sinusoid has east-minus-west value cos(lon)*sin(dlon).
    # Multiply by the meridional edge length and divide by the EXACT cell
    # area; the continuum R*cos(lat)*dlon divisor is not this FV quadrature.
    expected_signed = (-np.cos(np.asarray(g.lon))[None, :] * np.sin(g.dlon)
                       * np.asarray(g.dy)[:, None] / 2 / np.asarray(g.area_T))
    cap = float(np.max(expected_signed)) / 2
    expected = np.clip(expected_signed, 0, cap)
    assert np.any(expected_signed > cap) and np.any(expected_signed < 0)
    np.testing.assert_allclose(_closing(u, v, g, cap), expected, rtol=2e-6, atol=1e-13)


def test_live_fold_closing_matches_boundary_flux(geometry):
    """Nonzero seam width and unequal partners make the fold test able to fail."""
    g = geometry
    perm = jnp.arange(11, -1, -1)
    seam_width = float(jnp.mean(g.dx_v[1:-1]))
    g = g._replace(
        fold=FoldDescriptor(is_active=True, fold_j=7, cap_j=7,
                            perm_T=perm, perm_v=perm,
                            vector_sign_u=-1.0, vector_sign_v=-1.0),
        dx_v=g.dx_v.at[-1].set(seam_width))
    top = np.linspace(-0.4, 0.7, 12)
    v = jnp.zeros((8, 12)).at[-1].set(top)
    u = jnp.zeros_like(v)
    # South face averages the top row with a zero row. Shared seam flow is
    # half the difference of partner projections (normal reversal).
    south_flux = top / 2 * np.asarray(g.dx_v[-2])
    north_flux = (top - top[::-1]) / 2 * seam_width
    signed = (south_flux - north_flux) / np.asarray(g.area_T[-1])
    assert np.max(np.abs(north_flux)) > 0
    assert np.any(signed > 0) and np.any(signed < 0)
    np.testing.assert_allclose(_closing(u, v, g)[-1], np.maximum(signed, 0),
                               rtol=2e-6, atol=1e-13)
    np.testing.assert_allclose(jax.jit(lambda vel: _closing(u, vel, g))(v)[-1],
                               np.maximum(signed, 0), rtol=2e-6, atol=1e-13)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_gradient_is_nonzero_and_matches_finite_difference(geometry, dtype):
    g = geometry
    wave = jnp.broadcast_to(jnp.sin(g.lon), (8, 12)).astype(dtype)
    z = jnp.zeros_like(wave)
    def loss(scale):
        return jnp.sum(_closing(scale * wave, z, g))
    scale = jnp.asarray(0.4, dtype=dtype)
    np.testing.assert_allclose(jax.jit(loss)(scale), loss(scale), rtol=2e-6)
    deriv = jax.jit(jax.grad(loss))(scale)
    delta = jnp.asarray(0.01, dtype=dtype)
    finite_difference = (loss(scale + delta) - loss(scale - delta)) / (2 * delta)
    assert float(deriv) > 1e-7
    np.testing.assert_allclose(deriv, finite_difference, rtol=2e-4)
    # Quiescent input must not produce the sqrt-at-zero NaN adjoint.
    assert np.all(np.isfinite(jax.grad(lambda u: jnp.sum(_closing(u, z, g)))(z)))


def test_regular_agrid_convergence_omits_shear():
    g = create_latlon_grid(8, 12)
    u = jnp.broadcast_to(jnp.sin(g.lat)[:, None], (8, 12))
    v = jnp.zeros_like(u)
    conv = _closing(u, v, g)
    full = _closing_rate_from_velocity(u, v, g, cap=1.0,
                                      cs_shear=RidgingConfig().cs_shear_ridging)
    np.testing.assert_allclose(conv, 0, atol=1e-15)
    assert float(jnp.max(full)) > 1e-10


@pytest.mark.parametrize("enabled", [False, True])
def test_unknown_closing_scheme_raises_at_entry(geometry, enabled):
    cfg = SeaIceConfig(ridging=RidgingConfig(enabled=enabled, closing_scheme="typo"))
    with pytest.raises(ValueError, match="Unknown ridging closing scheme"):
        step_sea_ice(None, None, None, None, None, cfg, U_min=0, dt=1, grid=geometry)


def test_unknown_closing_helper_scheme_raises(geometry):
    z = jnp.zeros((8, 12))
    with pytest.raises(ValueError, match="Unknown ridging closing scheme"):
        _closing_rate_from_velocity(z, z, geometry, cap=1, closing_scheme="typo")


@pytest.mark.parametrize("dynamics,closing_scheme", [
    ("evp", "convergence"), ("mevp", "convergence"), ("free_drift", "strain")])
def test_cgrid_still_rejects_unimplemented_mechanics(geometry, dynamics, closing_scheme):
    cfg = SeaIceConfig(dynamics=dynamics,
                       ridging=RidgingConfig(enabled=True, closing_scheme=closing_scheme))
    with pytest.raises(ValueError, match="strain-rate"):
        step_sea_ice(None, None, None, None, None, cfg, U_min=0, dt=1, grid=geometry)


@pytest.mark.parametrize("grid", [None, object()])
def test_convergence_requires_supported_grid(grid):
    cfg = SeaIceConfig(ridging=RidgingConfig(enabled=True, closing_scheme="convergence"))
    with pytest.raises(ValueError, match="requires a grid|requires a grid argument"):
        step_sea_ice(None, None, None, None, None, cfg, U_min=0, dt=1, grid=grid)


def test_convergence_ridging_rejects_single_category_noop(geometry):
    cfg = SeaIceConfig(ridging=RidgingConfig(enabled=True, closing_scheme="convergence"))
    with pytest.raises(ValueError, match="n_categories >= 2"):
        step_sea_ice(None, None, None, None, None, cfg, U_min=0, dt=1, grid=geometry)


def test_full_step_ridges_and_conserves_column_ice_volume_and_salt(geometry):
    g = geometry
    shape = (8, 12)
    st = init_dynamic_ice_state(shape, S_ice_init=3.0)
    st = st._replace(h_ice=st.h_ice.replace(data=jnp.full(shape, 0.8)),
                     concentration=st.concentration.replace(data=jnp.full(shape, 0.7)),
                     T_ice=st.T_ice.replace(data=jnp.full(shape, 260.0)))
    st = distribute_dynamic_state_to_categories(st, 5)
    cfg = SeaIceConfig(dynamics="free_drift", transport="none", n_categories=5,
                       itd_remap="lipscomb2001", brine=BrineConfig(enabled=True),
                       ridging=RidgingConfig(enabled=True, closing_scheme="convergence"))
    z = jnp.zeros(shape)
    ocean_u = jnp.broadcast_to(0.5 * jnp.sin(g.lon), shape)
    atm = _min_forcing(shape)
    def advance(config):
        return step_sea_ice(st, atm, jnp.full(shape, constants.T_freeze_ocean),
                            ocean_u, z, config, U_min=0, dt=1800, grid=g)[0]
    on = advance(cfg)
    compiled = jax.jit(lambda: advance(cfg))()
    for eager, traced in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(compiled)):
        np.testing.assert_allclose(eager, traced, rtol=2e-6, atol=1e-10)
    off = advance(cfg._replace(ridging=cfg.ridging._replace(enabled=False)))
    a_on, a_off = on.concentration.data, off.concentration.data
    assert float(jnp.max(jnp.sum(a_off - a_on, axis=-1))) > 1e-5
    assert float(jnp.max(jnp.abs(on.h_ice.data - off.h_ice.data))) > 1e-3
    for state in (on, off):
        for leaf in jax.tree_util.tree_leaves(state):
            assert np.all(np.isfinite(leaf))
    volume_on = a_on * on.h_ice.data
    volume_off = a_off * off.h_ice.data
    np.testing.assert_allclose(jnp.sum(volume_on, -1), jnp.sum(volume_off, -1),
                               rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(jnp.sum(volume_on * on.S_ice.data, -1),
                               jnp.sum(volume_off * off.S_ice.data, -1),
                               rtol=2e-6, atol=1e-10)
