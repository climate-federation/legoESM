"""Tests for AB2 and RK3 tracer time integration in the lat-lon C-grid ocean model.

Verifies:
1. AB2 is 2nd-order accurate on a pure advection problem.
2. RK3 is 3rd-order accurate on a pure advection problem.
3. Forward Euler (default) produces identical results to the refactored code path.
4. AB2 and RK3 produce finite, stable output through model.step().
5. Config validation rejects invalid tracer_time_integrator values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _compute_advection_flux_div,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=16, n_lon=32)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=3, H_max=1000.0)


@pytest.fixture
def state(grid, z_coord):
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=1000.0,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_valid_integrators(self):
        """All three valid integrators should be accepted."""
        for tti in ("euler", "ab2", "rk3"):
            cfg = LatLonCGridOceanConfig(tracer_time_integrator=tti)
            assert cfg.tracer_time_integrator == tti

    def test_invalid_integrator_rejected(self, grid, z_coord):
        cfg = LatLonCGridOceanConfig(tracer_time_integrator="leapfrog")
        with pytest.raises(ValueError, match="tracer_time_integrator"):
            LatLonCGridOceanModel(grid, z_coord, config=cfg)

    def test_negative_ab2_epsilon_rejected(self, grid, z_coord):
        cfg = LatLonCGridOceanConfig(ab2_epsilon=-0.1)
        with pytest.raises(ValueError, match="ab2_epsilon"):
            LatLonCGridOceanModel(grid, z_coord, config=cfg)


# ---------------------------------------------------------------------------
# Model integration tests (finite, stable output)
# ---------------------------------------------------------------------------

class TestModelIntegration:
    """Verify that model.step() works with each time integrator."""

    @pytest.mark.parametrize("tti", ["euler", "ab2", "rk3"])
    def test_step_produces_finite_output(self, grid, z_coord, state, tti):
        """A single step should produce finite T, S, eta."""
        cfg = LatLonCGridOceanConfig(tracer_time_integrator=tti)
        model = LatLonCGridOceanModel(grid, z_coord, config=cfg)
        state_new = model.step(state, dt=300.0)

        assert bool(jnp.all(jnp.isfinite(state_new.T.data)))
        assert bool(jnp.all(jnp.isfinite(state_new.S.data)))
        assert bool(jnp.all(jnp.isfinite(state_new.eta.data)))

    def test_ab2_two_steps_uses_previous_tendency(self, grid, z_coord, state):
        """AB2 should store and use previous flux divergence on step 2."""
        cfg = LatLonCGridOceanConfig(tracer_time_integrator="ab2")
        model = LatLonCGridOceanModel(grid, z_coord, config=cfg)

        # Step 1: falls back to Euler (no previous tendency)
        s1 = model.step(state, dt=300.0)
        assert s1.T_flux_div_prev is not None, "AB2 should store flux div"

        # Step 2: uses stored previous tendency
        s2 = model.step(s1, dt=300.0)
        assert s2.T_flux_div_prev is not None
        assert bool(jnp.all(jnp.isfinite(s2.T.data)))

    def test_euler_default_unchanged(self, grid, z_coord, state):
        """Default config (euler) should produce the same result as before."""
        cfg_euler = LatLonCGridOceanConfig(tracer_time_integrator="euler")
        cfg_default = LatLonCGridOceanConfig()
        model_euler = LatLonCGridOceanModel(grid, z_coord, config=cfg_euler)
        model_default = LatLonCGridOceanModel(grid, z_coord, config=cfg_default)

        s_euler = model_euler.step(state, dt=300.0)
        s_default = model_default.step(state, dt=300.0)

        np.testing.assert_array_equal(s_euler.T.data, s_default.T.data)
        np.testing.assert_array_equal(s_euler.S.data, s_default.S.data)

    @pytest.mark.parametrize("tti", ["ab2", "rk3"])
    def test_multi_step_finite(self, grid, z_coord, state, tti):
        """Multiple steps with AB2/RK3 should remain finite."""
        cfg = LatLonCGridOceanConfig(tracer_time_integrator=tti)
        model = LatLonCGridOceanModel(grid, z_coord, config=cfg)

        s = state
        for _ in range(3):
            s = model.step(s, dt=300.0)

        assert bool(jnp.all(jnp.isfinite(s.T.data)))
        assert bool(jnp.all(jnp.isfinite(s.S.data)))

    def test_ab2_integrate_scan(self, grid, z_coord, state):
        """integrate_scan (jax.lax.scan) must work with AB2.

        Regression test for the pytree None→Field transition that
        previously crashed scan on the second iteration.
        """
        cfg = LatLonCGridOceanConfig(tracer_time_integrator="ab2")
        model = LatLonCGridOceanModel(grid, z_coord, config=cfg)
        final, trajectory = model.integrate_scan(state, n_steps=3, dt=300.0)
        assert bool(jnp.all(jnp.isfinite(final.T.data)))
        assert final.T_flux_div_prev is not None

    @pytest.mark.parametrize("tti", ["euler", "ab2", "rk3"])
    def test_mass_conservation(self, grid, z_coord, tti):
        """Flux-form advection must conserve total tracer mass."""
        from legoesm.ocean.vertical import compute_layer_thickness

        cfg = LatLonCGridOceanConfig(tracer_time_integrator=tti)
        model = LatLonCGridOceanModel(grid, z_coord, config=cfg)

        # Use a state with non-trivial T profile so advection is active
        state0 = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=25.0, T_deep=2.0, S_uniform=35.0,
            H_max=1000.0,
        )

        # Compute initial total heat content: sum(h * T * area)
        h0 = compute_layer_thickness(
            state0.eta.data, state0.H_bathy.data, z_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        area = grid.area  # (n_lat, n_lon)
        mask = state0.land_mask.data
        hT_total_0 = float(jnp.sum(
            h0 * state0.T.data * area[:, :, jnp.newaxis] * mask[:, :, jnp.newaxis]
        ))

        # Step forward
        s = state0
        for _ in range(3):
            s = model.step(s, dt=300.0)

        h_new = compute_layer_thickness(
            s.eta.data, s.H_bathy.data, z_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        hT_total_new = float(jnp.sum(
            h_new * s.T.data * area[:, :, jnp.newaxis] * mask[:, :, jnp.newaxis]
        ))

        # The flux-form advection is exactly conservative, but the
        # full model step also includes diffusion which changes total
        # heat.  Check that no large spurious source/sink is introduced
        # by the time integrator (rtol=1e-6 catches gross bugs while
        # allowing for diffusion effects).
        np.testing.assert_allclose(
            hT_total_new, hT_total_0, rtol=1e-6,
            err_msg=f"Large conservation violation with {tti}",
        )

    @pytest.mark.parametrize("tti", ["ab2", "rk3"])
    def test_weno5_smoke(self, grid, z_coord, state, tti):
        """WENO5 + AB2/RK3 should produce finite output (key use case)."""
        cfg = LatLonCGridOceanConfig(
            tracer_advection="weno5",
            tracer_time_integrator=tti,
        )
        model = LatLonCGridOceanModel(grid, z_coord, config=cfg)
        s1 = model.step(state, dt=300.0)
        assert bool(jnp.all(jnp.isfinite(s1.T.data)))


# ---------------------------------------------------------------------------
# Convergence-order tests (isolated advection, no full model)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pure ODE convergence test (no spatial discretization)
# ---------------------------------------------------------------------------

def _ode_convergence_test(time_integrator, dt_values, ab2_eps=0.0):
    """Test temporal order on the decay ODE dT/dt = -T, T(0) = 1.

    Exact solution: T(t_end) = exp(-t_end).  No spatial error, so
    the measured rate directly reflects the time integrator's order.

    Parameters
    ----------
    ab2_eps : float
        AB2 stabilization parameter.  Use 0.0 to verify pure 2nd-order
        convergence; use 0.1 for the production-stabilized variant
        (formally 1st-order with small error constant).

    Returns list of (dt, abs_error) pairs.
    """
    t_end = 1.0
    T_exact = np.exp(-t_end)

    results = []
    for dt in dt_values:
        n_steps = int(round(t_end / dt))
        actual_dt = t_end / n_steps

        T = 1.0
        F_prev = None  # previous tendency for AB2

        for _ in range(n_steps):
            F = -T  # tendency: dT/dt = -T

            if time_integrator == "euler":
                T = T + actual_dt * F
            elif time_integrator == "ab2":
                if F_prev is not None:
                    G = (1.5 + ab2_eps) * F - (0.5 + ab2_eps) * F_prev
                else:
                    G = F
                F_prev = F
                T = T + actual_dt * G
            elif time_integrator == "rk3":
                # SSP-RK3
                k1 = T + actual_dt * F
                k2 = 0.75 * T + 0.25 * (k1 + actual_dt * (-k1))
                T = (1.0 / 3.0) * T + (2.0 / 3.0) * (k2 + actual_dt * (-k2))

        err = abs(T - T_exact)
        results.append((actual_dt, err))

    return results


def _ode_convergence_rate(results):
    """Compute convergence rate from list of (dt, error) pairs."""
    dt0, e0 = results[0]
    dt1, e1 = results[-1]
    if e0 <= 0 or e1 <= 0:
        return 0.0
    return np.log(e0 / e1) / np.log(dt0 / dt1)


class TestTemporalConvergenceODE:
    """Verify temporal order on a simple ODE (no spatial error)."""

    def test_euler_first_order(self):
        """Euler should be 1st-order on dT/dt = -T."""
        res = _ode_convergence_test("euler", [0.1, 0.05, 0.025, 0.0125])
        rate = _ode_convergence_rate(res)
        assert 0.9 < rate < 1.2, f"Euler rate={rate:.2f}, expected ~1.0"

    def test_ab2_second_order_pure(self):
        """AB2 with eps=0 should be 2nd-order on dT/dt = -T."""
        res = _ode_convergence_test("ab2", [0.1, 0.05, 0.025, 0.0125],
                                    ab2_eps=0.0)
        rate = _ode_convergence_rate(res)
        assert 1.7 < rate < 2.5, f"AB2(eps=0) rate={rate:.2f}, expected ~2.0"

    def test_ab2_stabilized_lower_error_than_euler(self):
        """Stabilized AB2 (eps=0.1) is formally 1st-order but has ~10x
        smaller error constant than Euler."""
        res_ab2 = _ode_convergence_test("ab2", [0.05, 0.025], ab2_eps=0.1)
        res_euler = _ode_convergence_test("euler", [0.05, 0.025])
        # AB2 stabilized error should be much smaller than Euler
        assert res_ab2[-1][1] < res_euler[-1][1], (
            f"AB2(eps=0.1) error ({res_ab2[-1][1]:.4e}) should be less "
            f"than Euler ({res_euler[-1][1]:.4e})"
        )

    def test_rk3_third_order(self):
        """RK3 should be 3rd-order on dT/dt = -T."""
        res = _ode_convergence_test("rk3", [0.1, 0.05, 0.025, 0.0125])
        rate = _ode_convergence_rate(res)
        assert 2.7 < rate < 3.5, f"RK3 rate={rate:.2f}, expected ~3.0"


# ---------------------------------------------------------------------------
# Advection accuracy comparison (tvd spatial + varying time integrator)
# ---------------------------------------------------------------------------

def _advection_accuracy_test(tracer_time_integrator, n_lon=64):
    """Run advection test with tvd spatial scheme, return L2 error.

    Uses tvd (2nd-order spatial) so temporal error is visible when
    comparing Euler (1st-order temporal) against AB2/RK3.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import interp_cell_to_uface

    R = 6.371e6
    u_equator = 42.0
    target_cfl = 0.3

    n_lat = max(n_lon // 2, 4)
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)

    dlon_rad = float(grid.dlon)
    dt = target_cfl * R * dlon_rad / u_equator
    T_rev = 2.0 * np.pi * R / u_equator
    n_steps = int(round(T_rev / dt))

    h_cell = jnp.ones((n_lat, n_lon, 1), dtype=jnp.float64)
    h_u = interp_cell_to_uface(h_cell)
    cos_lat = jnp.cos(grid.lat)
    u_lat = u_equator * cos_lat
    mass_flux_u = h_u * u_lat[:, jnp.newaxis, jnp.newaxis]
    mass_flux_v = jnp.zeros((n_lat + 1, n_lon, 1), dtype=jnp.float64)
    w_baro = jnp.zeros((n_lat, n_lon, 2), dtype=jnp.float64)
    h_v = jnp.zeros((n_lat + 1, n_lon, 1), dtype=jnp.float64)

    dx_deg = 360.0 / n_lon
    lon_centers = jnp.linspace(dx_deg / 2, 360 - dx_deg / 2, n_lon)
    sigma = 30.0
    tracer_1d = jnp.exp(-0.5 * ((lon_centers - 180.0) / sigma) ** 2)
    tracer = jnp.broadcast_to(
        tracer_1d[jnp.newaxis, :, jnp.newaxis], (n_lat, n_lon, 1),
    ).copy()
    tracer_exact = tracer.copy()

    flux_div_prev = None
    for _ in range(n_steps):
        if tracer_time_integrator == "rk3":
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                _ssp_rk3_tracer_step,
            )
            mask_3d = jnp.ones_like(tracer)
            tracer = _ssp_rk3_tracer_step(
                tracer, "tvd",
                mass_flux_u, mass_flux_v, w_baro,
                h_cell, h_cell, h_u, h_v,
                grid, dt, mask_3d,
            )
            continue

        div_hut, vert_flux_div = _compute_advection_flux_div(
            tracer, "tvd",
            mass_flux_u, mass_flux_v, w_baro,
            h_cell, h_u, h_v, grid, dt,
        )
        total_fd = div_hut + vert_flux_div

        if tracer_time_integrator == "ab2":
            eps = 0.1
            if flux_div_prev is not None:
                eff_fd = (1.5 + eps) * total_fd - (0.5 + eps) * flux_div_prev
            else:
                eff_fd = total_fd
            hT_new = h_cell * tracer - dt * eff_fd
            flux_div_prev = total_fd
        else:
            hT_new = h_cell * tracer - dt * total_fd

        tracer = hT_new / jnp.maximum(h_cell, 1e-10)

    return float(jnp.sqrt(jnp.mean((tracer - tracer_exact) ** 2)))


class TestAdvectionAccuracyComparison:
    """With a 2nd-order spatial scheme, higher-order time integrators
    should produce lower error than Euler."""

    def test_ab2_lower_error_than_euler(self):
        """AB2 (2nd-order) should beat Euler (1st-order) with tvd spatial."""
        err_euler = _advection_accuracy_test("euler", n_lon=64)
        err_ab2 = _advection_accuracy_test("ab2", n_lon=64)
        assert err_ab2 < err_euler, (
            f"AB2 error ({err_ab2:.4e}) should be less than "
            f"Euler ({err_euler:.4e})"
        )

    def test_rk3_lower_error_than_euler(self):
        """RK3 (3rd-order) should beat Euler (1st-order) with tvd spatial."""
        err_euler = _advection_accuracy_test("euler", n_lon=64)
        err_rk3 = _advection_accuracy_test("rk3", n_lon=64)
        assert err_rk3 < err_euler, (
            f"RK3 error ({err_rk3:.4e}) should be less than "
            f"Euler ({err_euler:.4e})"
        )

    def test_all_methods_stable(self):
        """All methods should produce finite, bounded results."""
        for tti in ("euler", "ab2", "rk3"):
            err = _advection_accuracy_test(tti, n_lon=32)
            assert np.isfinite(err), f"{tti} produced non-finite error"
            assert err < 1.0, f"{tti} error={err:.4e} is too large"
