"""Symmetry and invariance tests for shallow water dynamics (Category 5).

Tests:
  5a) Axisymmetric preservation on lat-lon grid
  5b) Hemispheric symmetry on lat-lon grid
  5d) Time-reversal symmetry on lat-lon grid (inviscid)
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
    FVShallowWaterLatLonModel,
    FVShallowWaterLatLonConfig,
)
from legoesm.core.state import ShallowWaterState
from legoesm.core.field import Field
from legoesm.grids.latlon import create_latlon_grid
from legoesm import constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIMS = ("lat", "lon")
_H0 = constants.H_MEAN  # 10 000 m


def _make_latlon_state(grid, h_data, u_data, v_data):
    """Build a ShallowWaterState from raw arrays on a lat-lon grid."""
    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=_DIMS, units="m"),
        u=Field(data=u_data, name="u", dims=_DIMS, units="m/s"),
        v=Field(data=v_data, name="v", dims=_DIMS, units="m/s"),
        h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=_DIMS, units="m"),
    )


def _step_n(model, state, dt, n_steps):
    """Step forward *n_steps* times."""
    for _ in range(n_steps):
        state = model.step(state, dt)
    return state


# ---------------------------------------------------------------------------
# 5a  Axisymmetric preservation
# ---------------------------------------------------------------------------

class TestAxiSymmetricPreservation:
    """A zonally-uniform state should remain zonally uniform."""

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        h_data = _H0 + 100.0 * jnp.cos(lat2d)
        u_data = 10.0 * jnp.cos(lat2d)
        v_data = jnp.zeros_like(lat2d)

        state0 = _make_latlon_state(grid, h_data, u_data, v_data)
        return model, state0

    def test_zonal_std_small(self, setup):
        model, state0 = setup
        state = _step_n(model, state0, 60.0, 20)

        h = state.h.data
        zonal_std = jnp.std(h, axis=1)         # std over longitudes
        zonal_mean = jnp.mean(jnp.abs(h), axis=1)
        ratio = zonal_std / jnp.maximum(zonal_mean, 1e-30)
        assert float(jnp.max(ratio)) < 0.01, (
            f"Zonal std/mean = {float(jnp.max(ratio)):.4e}, expected < 0.01"
        )


# ---------------------------------------------------------------------------
# 5b  Hemispheric symmetry
# ---------------------------------------------------------------------------

class TestHemisphericSymmetry:
    """A state symmetric about the equator should stay symmetric."""

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        h_data = _H0 + 100.0 * jnp.cos(2.0 * lat2d)
        u_data = 10.0 * jnp.cos(lat2d)
        v_data = jnp.zeros_like(lat2d)

        state0 = _make_latlon_state(grid, h_data, u_data, v_data)
        return model, state0, grid

    def test_h_symmetric(self, setup):
        model, state0, grid = setup
        state = _step_n(model, state0, 60.0, 20)

        h = state.h.data  # (n_lat, n_lon)
        n_lat = h.shape[0]
        h_south = h[:n_lat // 2, :]       # southern hemisphere
        h_north = h[n_lat // 2:, :][::-1]  # northern hemisphere, flipped

        # Check that the max relative difference is < 1%
        diff = jnp.abs(h_south - h_north)
        scale = jnp.maximum(jnp.abs(h_south) + jnp.abs(h_north), 1e-30) / 2.0
        rel_diff = diff / scale
        assert float(jnp.max(rel_diff)) < 0.01, (
            f"Max hemispheric asymmetry = {float(jnp.max(rel_diff)):.4e}, expected < 0.01"
        )


# ---------------------------------------------------------------------------
# 5d  Time-reversal symmetry
# ---------------------------------------------------------------------------

class TestTimeReversal:
    """Run N steps forward, negate velocities, run N steps back."""

    @pytest.fixture
    def setup(self):
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        # Inviscid, no conservation fixer, no polar filter for cleanest reversal
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=False,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        h_data = _H0 + 100.0 * jnp.cos(lat2d)
        u_data = 10.0 * jnp.cos(lat2d)
        v_data = jnp.zeros_like(lat2d)

        state0 = _make_latlon_state(grid, h_data, u_data, v_data)
        return model, state0

    def test_h_recovers(self, setup):
        model, state0 = setup
        n_steps = 20
        dt = 60.0

        # Forward
        state_fwd = _step_n(model, state0, dt, n_steps)

        # Negate velocities
        state_rev = state_fwd._replace(
            u=state_fwd.u.replace(data=-state_fwd.u.data),
            v=state_fwd.v.replace(data=-state_fwd.v.data),
        )

        # Backward
        state_back = _step_n(model, state_rev, dt, n_steps)

        h0 = state0.h.data
        h_back = state_back.h.data
        norm_h0 = jnp.sqrt(jnp.mean(h0 ** 2))
        err = jnp.sqrt(jnp.mean((h_back - h0) ** 2)) / norm_h0
        assert float(err) < 0.1, (
            f"Time-reversal error = {float(err):.4e}, expected < 0.1"
        )
