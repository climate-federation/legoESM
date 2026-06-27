"""Direct unit tests for the implicit-solver K/A profile composer
``ocean/physics/vertical_mixing/k_profiles.compute_vertical_K_profiles``.

The momentum-viscosity composition (``constant`` + ``enhanced_diffusion`` and
the KPP-suppression guard) is already pinned in
``test_enhanced_diffusion_momentum.py``; the TKE / CATKE prognostic paths are in
``test_tke_integration.py`` / ``test_catke_integration.py``.  This file fills the
remaining leaf branches that no other test exercises directly:

  * the ``vertical_mixing.scheme == "none"`` short-circuit — the output is
    exactly the supplied background floors, at the interior-interface shape;
  * the ``constant`` scheme — K/A equal the configured constants (plus floor);
  * background floors are additive.

The KPP / TKE / CATKE composition paths need cell-centred velocity input and
are exercised through the model-level factories in ``test_mpas_physics.py`` /
``test_tke_integration.py`` / the enhanced-diffusion suite; here we pin the two
analytic branches (``none`` / ``constant``) whose output is exactly predictable.

All on a small lat-lon C-grid rest state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    ConstantVerticalMixingConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def grid_z_state():
    grid = create_latlon_grid(n_lat=8, n_lon=12)
    z = create_ocean_z_star(n_levels=5, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    return grid, z, state


def _base(**over):
    return dict(
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(scheme="none"),
        **over,
    )


class TestNoneScheme:
    def test_returns_background_floors_at_interface_shape(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"), **_base())
        K, A = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=1e-4, K_v_background=2e-5)
        nlev = state.T.data.shape[-1]
        assert K.shape == state.T.data.shape[:-1] + (nlev - 1,)
        # With no scheme + no convection, output IS the supplied floors.
        assert jnp.allclose(K, 2e-5)
        assert jnp.allclose(A, 1e-4)

    def test_zero_floors_give_zero(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"), **_base())
        K, A = compute_vertical_K_profiles(state, z, None, cfg)
        assert jnp.allclose(K, 0.0)
        assert jnp.allclose(A, 0.0)


class TestConstantScheme:
    def test_constant_values(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=3e-3, K_v=7e-4),
            ),
            **_base())
        K, A = compute_vertical_K_profiles(state, z, None, cfg)
        assert jnp.allclose(K, 7e-4)
        assert jnp.allclose(A, 3e-3)
        assert bool(jnp.all(jnp.isfinite(K)))

    def test_background_floor_is_additive(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
            ),
            **_base())
        K0, A0 = compute_vertical_K_profiles(state, z, None, cfg)
        K1, A1 = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=5e-4, K_v_background=2e-4)
        assert jnp.allclose(K1 - K0, 2e-4)
        assert jnp.allclose(A1 - A0, 5e-4)


class TestUnknownSchemeRaises:
    """Finding #3: an unknown vertical_mixing scheme must RAISE, not silently
    fall through to a zero K_v/A_v ('be defensive') return that disables vertical
    mixing on a typo (dispatch hardening; CLAUDE.md 'Dispatch')."""

    def test_unknown_scheme_raises_value_error(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="kpp_typo"), **_base())
        with pytest.raises(ValueError, match="unknown vertical_mixing.scheme"):
            compute_vertical_K_profiles(state, z, None, cfg)

    def test_none_scheme_still_returns_zeros(self, grid_z_state):
        """The explicit 'none' gate must keep returning background floors (here
        zero) — the raise must not catch the legitimate 'none' value."""
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"), **_base())
        K, A = compute_vertical_K_profiles(state, z, None, cfg)
        assert jnp.allclose(K, 0.0)
        assert jnp.allclose(A, 0.0)
