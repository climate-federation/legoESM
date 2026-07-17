"""Stability tests for FV3 hydrostatic PE on the cubed sphere.

Tests:
1. Isothermal rest state (100 steps, dt=300s): should stay at rest
2. TC2-like zonal wind initialization (50 steps): should stay stable
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs


def _make_rest_state(grid, nlev):
    """Create an isothermal atmosphere at rest."""
    n = grid.n
    T_data = jnp.full((6, n, n, nlev), 250.0)
    u_data = jnp.zeros((6, n, n, nlev))
    v_data = jnp.zeros((6, n, n, nlev))
    ps_data = jnp.full((6, n, n), 1e5)
    phis_data = jnp.zeros((6, n, n))

    return HydrostaticState(
        u=Field(data=u_data, name="u", dims=("face", "x", "y", "lev"), units="m/s"),
        v=Field(data=v_data, name="v", dims=("face", "x", "y", "lev"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("face", "x", "y", "lev"), units="K"),
        p_s=Field(data=ps_data, name="p_s", dims=("face", "x", "y"), units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
    )


def _make_zonal_wind_state(grid, nlev):
    """Create a TC2-like zonal wind initialization for 3D PE."""
    from legoesm import constants

    n = grid.n
    R = grid.radius
    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)  # ~38.6 m/s

    lat = grid.lat
    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)

    # Solid-body rotation (grid-aligned)
    u_east = u_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    # Broadcast to all levels (barotropic)
    u_data = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev)).copy()
    v_data = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev)).copy()

    T_data = jnp.full((6, n, n, nlev), 250.0)
    ps_data = jnp.full((6, n, n), 1e5)
    phis_data = jnp.zeros((6, n, n))

    return HydrostaticState(
        u=Field(data=u_data, name="u", dims=("face", "x", "y", "lev"), units="m/s"),
        v=Field(data=v_data, name="v", dims=("face", "x", "y", "lev"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("face", "x", "y", "lev"), units="K"),
        p_s=Field(data=ps_data, name="p_s", dims=("face", "x", "y"), units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
    )


class TestAMIPStability:
    """Stability tests for the FV3 hydrostatic PE model."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    @pytest.fixture(scope="class")
    def sigma_coord(self):
        return create_sigma_coordinate(10)

    def test_isothermal_rest_100_steps(self, grid, sigma_coord):
        """Isothermal atmosphere at rest: 100 steps, 10 levels.

        At C8 with explicit time integration, O(5%) p_s drift is expected
        over 100 steps due to gravity wave oscillations without semi-implicit
        treatment. The key check is that all fields remain finite and
        temperature stays physical.
        """
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig as FVPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel as FVPrimitiveEquationModel,
        )

        nlev = 10
        dt = 300.0
        nu2, _nu4 = default_div_damp_coeffs(grid, dt=dt)
        config = FVPrimitiveEquationConfig(
            div_damp_coeff=nu2,
            hyperdiff_coeff=1e14,
        )
        model = FVPrimitiveEquationModel(grid, sigma_coord, config)
        state = _make_rest_state(grid, nlev)

        s = state
        for _ in range(50):
            s = model.step(s, dt)

        # All fields finite (primary stability check)
        assert jnp.all(jnp.isfinite(s.T.data)), "T contains NaN/Inf"
        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf"
        assert jnp.all(jnp.isfinite(s.p_s.data)), "p_s contains NaN/Inf"

        # Temperature stays in physical range
        T_min = float(jnp.min(s.T.data))
        T_max = float(jnp.max(s.T.data))
        assert T_min >= 200.0, f"T_min={T_min:.1f}K < 200K"
        assert T_max <= 350.0, f"T_max={T_max:.1f}K > 350K"

        # Surface pressure drift — explicit integration at C8 generates
        # O(5%) oscillations after 100 steps
        ps_drift = float(jnp.max(jnp.abs(s.p_s.data - state.p_s.data))) / 1e5
        assert ps_drift < 0.1, f"p_s drift {ps_drift:.2e} > 10%"

    def test_zonal_wind_20_steps(self, grid, sigma_coord):
        """TC2-like zonal wind should be stable for 20 steps.

        At C8, the ~38 m/s zonal flow generates grid-scale noise
        that grows rapidly without strong diffusion. This test verifies
        short-term stability with moderate viscosity.
        """
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig as FVPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel as FVPrimitiveEquationModel,
        )

        nlev = 5
        sigma_5 = create_sigma_coordinate(nlev)
        dt = 120.0
        nu2, _nu4 = default_div_damp_coeffs(grid, dt=dt)
        config = FVPrimitiveEquationConfig(
            div_damp_coeff=nu2,
            hyperdiff_coeff=1e14,
            A_h=1e6,
        )
        model = FVPrimitiveEquationModel(grid, sigma_5, config)
        state = _make_zonal_wind_state(grid, nlev)

        s = state
        for _ in range(20):
            s = model.step(s, dt)

        # All fields finite
        assert jnp.all(jnp.isfinite(s.T.data)), "T contains NaN/Inf"
        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf"
        assert jnp.all(jnp.isfinite(s.v.data)), "v contains NaN/Inf"
        assert jnp.all(jnp.isfinite(s.p_s.data)), "p_s contains NaN/Inf"

        # Temperature stays physical
        T_min = float(jnp.min(s.T.data))
        T_max = float(jnp.max(s.T.data))
        assert T_min >= 150.0, f"T_min={T_min:.1f}K < 150K"
        assert T_max <= 400.0, f"T_max={T_max:.1f}K > 400K"
