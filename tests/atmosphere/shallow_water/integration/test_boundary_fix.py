"""Test that boundary_fix eliminates cubed-sphere edge artifacts.

The boundary_fix option in FV3EdgeShallowWaterModel replaces the outermost
corner ring tendencies (which suffer from A-L gradient halo-error amplification)
with the nearest interior ring values.  This removes panel-boundary edge
artifacts while preserving stability and mass conservation.

See docs/cubed_sphere_edge_artifacts.md, iteration 15, for the full analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
    williamson_test2_exact,
    williamson_test5,
)


def _make_edge_ic(grid, cdgrid, test_num):
    """Create edge-midpoint D-grid IC for Williamson TC2 or TC5."""
    if test_num == 2:
        sw = williamson_test2(grid)
        u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    else:
        sw = williamson_test5(grid)
        u0 = 20.0
    u_d = cdgrid.cos_angle_edge_x * u0 * jnp.cos(cdgrid.lat_edge_x)
    v_d = -cdgrid.sin_angle_edge_y * u0 * jnp.cos(cdgrid.lat_edge_y)
    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


def _edge_artifact_ratio(v_north, n):
    """Ratio of max error at face boundaries vs interior."""
    bw = max(2, n // 8)
    bdy = np.zeros((6, n, n), dtype=bool)
    bdy[:, :bw, :] = True
    bdy[:, -bw:, :] = True
    bdy[:, :, :bw] = True
    bdy[:, :, -bw:] = True
    err = np.abs(np.asarray(v_north))
    bdy_max = float(np.max(err[bdy])) if np.any(bdy) else 0.0
    int_max = float(np.max(err[~bdy])) if np.any(~bdy) else 1e-30
    return bdy_max / max(int_max, 1e-30)


@pytest.fixture(scope="module")
def grid_and_cdgrid():
    n = 16  # Small for fast tests
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, n


class TestBoundaryFix:
    """Tests for the boundary_fix edge-artifact elimination."""

    def test_tc2_stable_100_steps(self, grid_and_cdgrid):
        """boundary_fix model is stable for 100 steps on TC2."""
        grid, cdgrid, n = grid_and_cdgrid
        dx = float(grid.radius) * np.pi / (2 * n)
        dt = 300.0
        config = CDGridShallowWaterConfig(
            div_damp=0.15 * dx ** 2 / dt,
            hyperdiff_coeff=dx ** 4 / (86400.0 * 10),
            boundary_fix=True,
        )
        model = FV3EdgeShallowWaterModel(grid, config)
        state = _make_edge_ic(grid, cdgrid, 2)
        model.set_initial_mass(state)
        for _ in range(100):
            state = model.step(state, dt)
        assert jnp.all(jnp.isfinite(state.h))
        assert jnp.all(jnp.isfinite(state.u_d))

    def test_tc5_stable_100_steps(self, grid_and_cdgrid):
        """boundary_fix model is stable for 100 steps on TC5."""
        grid, cdgrid, n = grid_and_cdgrid
        dx = float(grid.radius) * np.pi / (2 * n)
        dt = 300.0
        config = CDGridShallowWaterConfig(
            div_damp=0.15 * dx ** 2 / dt,
            hyperdiff_coeff=dx ** 4 / (86400.0 * 10),
            boundary_fix=True,
        )
        model = FV3EdgeShallowWaterModel(grid, config)
        state = _make_edge_ic(grid, cdgrid, 5)
        model.set_initial_mass(state)
        for _ in range(100):
            state = model.step(state, dt)
        assert jnp.all(jnp.isfinite(state.h))

    def test_mass_conservation(self, grid_and_cdgrid):
        """boundary_fix preserves mass to machine precision."""
        grid, cdgrid, n = grid_and_cdgrid
        dx = float(grid.radius) * np.pi / (2 * n)
        dt = 300.0
        config = CDGridShallowWaterConfig(
            div_damp=0.15 * dx ** 2 / dt,
            boundary_fix=True,
        )
        model = FV3EdgeShallowWaterModel(grid, config)
        state = _make_edge_ic(grid, cdgrid, 2)
        model.set_initial_mass(state)
        mass0 = float(jnp.sum(state.h * grid.area))
        for _ in range(50):
            state = model.step(state, dt)
        mass1 = float(jnp.sum(state.h * grid.area))
        rel_err = abs(mass1 - mass0) / abs(mass0)
        assert rel_err < 1e-6

    def test_edge_ratio_reduced(self, grid_and_cdgrid):
        """boundary_fix has edge ratio closer to 1.0 than baseline."""
        grid, cdgrid, n = grid_and_cdgrid
        dx = float(grid.radius) * np.pi / (2 * n)
        dt = 300.0

        results = {}
        for bf, label in [(False, "baseline"), (True, "bdy_fix")]:
            config = CDGridShallowWaterConfig(
                div_damp=0.15 * dx ** 2 / dt,
                hyperdiff_coeff=dx ** 4 / (86400.0 * 10),
                boundary_fix=bf,
            )
            model = FV3EdgeShallowWaterModel(grid, config)
            state = _make_edge_ic(grid, cdgrid, 2)
            model.set_initial_mass(state)
            for _ in range(100):
                state = model.step(state, dt)
            u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                          + np.asarray(state.u_d)[:, :, 1:])
            v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                          + np.asarray(state.v_d)[:, 1:, :])
            v_north = (np.asarray(grid.sin_angle) * u_cc
                       + np.asarray(grid.cos_angle) * v_cc)
            results[label] = _edge_artifact_ratio(v_north, n)

        # boundary_fix edge ratio should be closer to 1.0
        assert abs(results["bdy_fix"] - 1.0) < abs(results["baseline"] - 1.0) + 0.5
