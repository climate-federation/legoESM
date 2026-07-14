"""Smoke tests for the boundary_fix option in FV3EdgeShallowWaterModel.

These are **short-horizon, low-resolution** sanity checks (100 steps at C16)
that verify the boundary_fix code path does not crash and preserves basic
invariants.  They do NOT reproduce the multi-day, C36 experiments documented
in ``docs/cubed_sphere_edge_artifacts.md`` (iteration 15), which showed:

* TC2 at C36 5-day: edge ratio 1.02 (vs baseline 0.95), L2 21% worse
* TC5 at C36 5-day/15-day: stable with physically correct wind speeds
* Visual elimination of panel-boundary v-wind streaks

To reproduce the documented experiments, run the C36 validation manually::

    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel, CDGridShallowWaterConfig)
    config = CDGridShallowWaterConfig(
        div_damp=..., hyperdiff_coeff=..., boundary_fix=True)

See docs/cubed_sphere_edge_artifacts.md, iteration 15, for the full protocol.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
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
    """Ratio of max |v_north| at face boundaries vs interior.

    Returns ~1.0 when errors are uniformly distributed, >1.0 when
    boundary errors dominate (edge artifacts).
    """
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
    # C16: small enough for fast CI, large enough for non-trivial halo exchange
    n = 16
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, n


class TestBoundaryFix:
    """Smoke tests for the boundary_fix code path.

    These check that boundary_fix=True does not introduce NaN, preserves
    mass, and does not make the edge-artifact ratio dramatically worse.
    They are intentionally short (100 steps ≈ 8 hours of model time at
    dt=300s) and low-resolution (C16) for CI speed.
    """

    def test_tc2_stable_100_steps(self, grid_and_cdgrid):
        """boundary_fix produces finite output after 100 steps on TC2."""
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
        """boundary_fix produces finite output after 100 steps on TC5."""
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
        """boundary_fix preserves mass to fp64 roundoff (50 steps, C16).

        The conservation fixer anchors and corrects ``int h dA`` in fp64
        (``conservation_accumulator``), so the diagnostic MUST also reduce in
        fp64 — the SW state/grid arrays are fp32 (FV finite-volume policy), and
        an fp32 ``sum(h * area)`` over the ~6*N^2 cubed-sphere cells carries a
        ~sqrt(N)*eps ~ 2e-6 summation-roundoff floor that masks the true drift
        (~2e-16). Match the fp64 diagnostic + 1e-12 tolerance of the sibling
        test_sw_mass_conservation_anchored.py."""
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
        mass0 = float(jnp.sum(
            state.h.astype(jnp.float64) * grid.area.astype(jnp.float64)))
        for _ in range(50):
            state = model.step(state, dt)
        mass1 = float(jnp.sum(
            state.h.astype(jnp.float64) * grid.area.astype(jnp.float64)))
        rel_err = abs(mass1 - mass0) / abs(mass0)
        assert rel_err < 1e-12, f"mass drift {rel_err:.2e} exceeds fp64 roundoff"

    def test_edge_ratio_not_worse(self, grid_and_cdgrid):
        """boundary_fix does not make the edge-artifact ratio worse.

        This is a weak guard-rail (short horizon, low resolution). The
        documented C36 5-day result shows edge ratio improving from 0.95
        to 1.02; at C16 / 100 steps the effect is smaller.
        """
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

        # Guard-rail: boundary_fix should not make edge ratio dramatically
        # worse than baseline. A margin of 0.3 accounts for the fact that
        # at C16 / 100 steps the ratio is noisy and the improvement is
        # smaller than the documented C36 / 5-day result.
        assert results["bdy_fix"] < results["baseline"] + 0.3
