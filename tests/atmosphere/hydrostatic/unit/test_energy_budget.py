"""Unit tests for energy budget closure diagnostics (Task 11).

Tests cover:
- Column moist static energy computation
- Column dry static energy computation
- TOA net radiation
- Surface net radiation and energy flux
- Energy budget tracker integration
- JAX differentiability
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from legoesm import constants
from legoesm.diagnostics.energy_budget import (
    EnergyBudget,
    EnergyBudgetTracker,
    column_dry_static_energy,
    column_moist_static_energy,
    surface_energy_flux,
    surface_net_radiation,
    toa_net_radiation,
)


def _make_sigma(nlev: int):
    """Create simple evenly-spaced sigma coordinate arrays."""
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = jnp.diff(sigma_half)
    return sigma_full, dsigma


class TestColumnMoistStaticEnergy:
    """Tests for column_moist_static_energy."""

    def test_shape_1d(self):
        """Output shape should match spatial dimensions (no level axis)."""
        nlev = 10
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.full((nlev,), 0.01)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        E = column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
        assert E.shape == ()

    def test_shape_3d(self):
        """Should work with cubed-sphere (6, n, n, nlev) arrays."""
        nlev = 5
        sigma_full, dsigma = _make_sigma(nlev)
        shape = (6, 4, 4)
        T = jnp.full(shape + (nlev,), 280.0)
        q_v = jnp.full(shape + (nlev,), 0.005)
        u = jnp.zeros(shape + (nlev,))
        v = jnp.zeros(shape + (nlev,))
        phis = jnp.zeros(shape)
        p_s = jnp.full(shape, 1e5)

        E = column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
        assert E.shape == shape

    def test_positive_energy(self):
        """Column energy should be positive for reasonable atmosphere."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 260.0)
        q_v = jnp.full((nlev,), 0.001)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        E = column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
        assert float(E) > 0.0

    def test_energy_increases_with_temperature(self):
        """Warmer atmosphere should have higher column energy."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        q_v = jnp.full((nlev,), 0.005)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        T_cold = jnp.full((nlev,), 250.0)
        T_warm = jnp.full((nlev,), 300.0)
        E_cold = column_moist_static_energy(T_cold, q_v, u, v, phis, p_s, dsigma, sigma_full)
        E_warm = column_moist_static_energy(T_warm, q_v, u, v, phis, p_s, dsigma, sigma_full)
        assert float(E_warm) > float(E_cold)

    def test_energy_increases_with_moisture(self):
        """Moister atmosphere should have higher column energy (L_v * q term)."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        q_dry = jnp.full((nlev,), 0.001)
        q_moist = jnp.full((nlev,), 0.015)
        E_dry = column_moist_static_energy(T, q_dry, u, v, phis, p_s, dsigma, sigma_full)
        E_moist = column_moist_static_energy(T, q_moist, u, v, phis, p_s, dsigma, sigma_full)
        assert float(E_moist) > float(E_dry)

    def test_energy_increases_with_wind(self):
        """Adding kinetic energy should increase column energy."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.full((nlev,), 0.005)
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        u_calm = jnp.zeros((nlev,))
        u_wind = jnp.full((nlev,), 30.0)
        E_calm = column_moist_static_energy(T, q_v, u_calm, v, phis, p_s, dsigma, sigma_full)
        E_wind = column_moist_static_energy(T, q_v, u_wind, v, phis, p_s, dsigma, sigma_full)
        assert float(E_wind) > float(E_calm)

    def test_energy_increases_with_surface_elevation(self):
        """Higher topography (larger phis) should increase column energy."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.full((nlev,), 0.005)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        p_s = jnp.array(1e5)

        phis_low = jnp.array(0.0)
        phis_high = jnp.array(5000.0 * constants.g)  # 5 km elevation
        E_low = column_moist_static_energy(T, q_v, u, v, phis_low, p_s, dsigma, sigma_full)
        E_high = column_moist_static_energy(T, q_v, u, v, phis_high, p_s, dsigma, sigma_full)
        assert float(E_high) > float(E_low)

    def test_scales_with_pressure(self):
        """Column energy should scale roughly with p_s (more mass = more energy)."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.full((nlev,), 0.005)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)

        E_low = column_moist_static_energy(T, q_v, u, v, phis, jnp.array(5e4), dsigma, sigma_full)
        E_high = column_moist_static_energy(T, q_v, u, v, phis, jnp.array(1e5), dsigma, sigma_full)
        # Should be approximately 2x
        ratio = float(E_high) / float(E_low)
        npt.assert_allclose(ratio, 2.0, atol=0.1)

    def test_finite(self):
        """Output should be finite."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.full((nlev,), 0.005)
        u = jnp.full((nlev,), 10.0)
        v = jnp.full((nlev,), -5.0)
        phis = jnp.array(100.0)
        p_s = jnp.array(1e5)

        E = column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
        assert jnp.isfinite(E)

    def test_differentiable(self):
        """jax.grad should work through column_moist_static_energy."""
        nlev = 10
        sigma_full, dsigma = _make_sigma(nlev)
        q_v = jnp.full((nlev,), 0.005)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        def loss(T):
            return column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)

        grad = jax.grad(loss)(jnp.full((nlev,), 280.0))
        assert jnp.all(jnp.isfinite(grad))
        # Gradient w.r.t. T should be positive (more T = more energy)
        assert jnp.all(grad > 0.0)


class TestColumnDryStaticEnergy:
    """Tests for column_dry_static_energy."""

    def test_less_than_moist(self):
        """Dry static energy should be less than moist static energy."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.full((nlev,), 0.01)
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        E_moist = column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
        E_dry = column_dry_static_energy(T, phis, p_s, dsigma, sigma_full)
        assert float(E_dry) < float(E_moist)

    def test_equals_moist_for_dry_atm(self):
        """With no moisture or KE, dry and moist should be close."""
        nlev = 20
        sigma_full, dsigma = _make_sigma(nlev)
        T = jnp.full((nlev,), 280.0)
        q_v = jnp.zeros((nlev,))
        u = jnp.zeros((nlev,))
        v = jnp.zeros((nlev,))
        phis = jnp.array(0.0)
        p_s = jnp.array(1e5)

        E_moist = column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
        E_dry = column_dry_static_energy(T, phis, p_s, dsigma, sigma_full)
        npt.assert_allclose(float(E_dry), float(E_moist), rtol=1e-10)


class TestTOANetRadiation:
    """Tests for toa_net_radiation."""

    def test_balanced_planet(self):
        """If SW absorbed = LW emitted, net = 0."""
        sw_down = jnp.array(340.0)  # S_0/4
        sw_up = jnp.array(100.0)    # reflected
        lw_up = jnp.array(240.0)    # emitted
        net = toa_net_radiation(sw_down, sw_up, lw_up)
        npt.assert_allclose(float(net), 0.0, atol=1e-10)

    def test_warming_planet(self):
        """Positive imbalance means planet is warming."""
        sw_down = jnp.array(340.0)
        sw_up = jnp.array(100.0)
        lw_up = jnp.array(230.0)
        net = toa_net_radiation(sw_down, sw_up, lw_up)
        assert float(net) > 0.0  # 10 W/m² imbalance

    def test_cooling_planet(self):
        """Negative imbalance means planet is cooling."""
        sw_down = jnp.array(340.0)
        sw_up = jnp.array(100.0)
        lw_up = jnp.array(250.0)
        net = toa_net_radiation(sw_down, sw_up, lw_up)
        assert float(net) < 0.0

    def test_array_input(self):
        """Should work with arrays."""
        sw_down = jnp.array([340.0, 0.0])  # day, night
        sw_up = jnp.array([100.0, 0.0])
        lw_up = jnp.array([240.0, 200.0])
        net = toa_net_radiation(sw_down, sw_up, lw_up)
        assert net.shape == (2,)
        npt.assert_allclose(float(net[0]), 0.0, atol=1e-10)
        npt.assert_allclose(float(net[1]), -200.0, atol=1e-10)


class TestSurfaceRadiation:
    """Tests for surface_net_radiation and surface_energy_flux."""

    def test_surface_net(self):
        """Surface net = SW + LW."""
        sw = jnp.array(200.0)
        lw = jnp.array(-60.0)
        net = surface_net_radiation(sw, lw)
        npt.assert_allclose(float(net), 140.0, atol=1e-10)

    def test_surface_energy_flux(self):
        """Surface energy flux into atmosphere."""
        sw = jnp.array(200.0)
        lw = jnp.array(-60.0)
        sh = jnp.array(20.0)
        lh = jnp.array(80.0)
        F = surface_energy_flux(sw, lw, sh, lh)
        # F = 200 + (-60) - 20 - 80 = 40
        npt.assert_allclose(float(F), 40.0, atol=1e-10)


class TestEnergyBudgetTracker:
    """Tests for EnergyBudgetTracker."""

    def _make_state(self, nlev=10, T_val=280.0):
        """Create a minimal atmospheric state for testing."""
        sigma_full, dsigma = _make_sigma(nlev)
        shape = (6, 4, 4)
        T = jnp.full(shape + (nlev,), T_val)
        q_v = jnp.full(shape + (nlev,), 0.005)
        u = jnp.zeros(shape + (nlev,))
        v = jnp.zeros(shape + (nlev,))
        phis = jnp.zeros(shape)
        p_s = jnp.full(shape, 1e5)
        return T, q_v, u, v, phis, p_s, dsigma, sigma_full

    def _make_fluxes(self, shape=(6, 4, 4)):
        """Create balanced TOA/surface fluxes."""
        sw_down_toa = jnp.full(shape, 340.0)
        sw_up_toa = jnp.full(shape, 100.0)
        lw_up_toa = jnp.full(shape, 240.0)
        sw_net_sfc = jnp.full(shape, 200.0)
        lw_net_sfc = jnp.full(shape, -60.0)
        return sw_down_toa, sw_up_toa, lw_up_toa, sw_net_sfc, lw_net_sfc

    def test_first_update_zero_residual(self):
        """First update should have zero dE/dt and residual (no previous)."""
        tracker = EnergyBudgetTracker()
        T, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state()
        sw_down_toa, sw_up_toa, lw_up_toa, sw_sfc, lw_sfc = self._make_fluxes()

        budget = tracker.update(
            T, q_v, u, v, phis, p_s, dsigma, sigma_full,
            sw_down_toa, sw_up_toa, lw_up_toa, sw_sfc, lw_sfc,
            elapsed_seconds=0.0,
        )
        assert budget.dE_dt == 0.0
        assert budget.residual == 0.0

    def test_two_updates_computes_tendency(self):
        """After two updates with different T, dE/dt should be nonzero."""
        tracker = EnergyBudgetTracker()
        nlev = 10
        T1, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state(nlev=nlev, T_val=280.0)
        fluxes = self._make_fluxes()

        tracker.update(T1, q_v, u, v, phis, p_s, dsigma, sigma_full,
                       *fluxes, elapsed_seconds=0.0)

        T2 = jnp.full_like(T1, 281.0)  # 1K warming
        budget = tracker.update(T2, q_v, u, v, phis, p_s, dsigma, sigma_full,
                                *fluxes, elapsed_seconds=3600.0)

        assert budget.dE_dt != 0.0
        # Warming → positive tendency
        assert budget.dE_dt > 0.0

    def test_steady_state_small_residual(self):
        """If state doesn't change, dE/dt=0 and residual = R_TOA."""
        tracker = EnergyBudgetTracker()
        T, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state()
        # Balanced TOA: R_TOA = 0
        sw_down_toa = jnp.full((6, 4, 4), 340.0)
        sw_up_toa = jnp.full((6, 4, 4), 100.0)
        lw_up_toa = jnp.full((6, 4, 4), 240.0)
        sw_sfc = jnp.full((6, 4, 4), 200.0)
        lw_sfc = jnp.full((6, 4, 4), -60.0)

        tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                       sw_down_toa, sw_up_toa, lw_up_toa, sw_sfc, lw_sfc,
                       elapsed_seconds=0.0)
        budget = tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                                sw_down_toa, sw_up_toa, lw_up_toa, sw_sfc, lw_sfc,
                                elapsed_seconds=3600.0)

        # No change in energy → dE/dt = 0, residual = R_TOA = 0
        npt.assert_allclose(budget.dE_dt, 0.0, atol=1e-10)
        npt.assert_allclose(budget.residual, 0.0, atol=1e-10)

    def test_toa_net_correct(self):
        """TOA net should match SW_down - SW_up - LW_up."""
        tracker = EnergyBudgetTracker()
        T, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state()
        sw_down_toa = jnp.full((6, 4, 4), 340.0)
        sw_up_toa = jnp.full((6, 4, 4), 100.0)
        lw_up_toa = jnp.full((6, 4, 4), 235.0)
        sw_sfc = jnp.full((6, 4, 4), 200.0)
        lw_sfc = jnp.full((6, 4, 4), -60.0)

        budget = tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                                sw_down_toa, sw_up_toa, lw_up_toa, sw_sfc, lw_sfc,
                                elapsed_seconds=0.0)
        npt.assert_allclose(budget.toa_net, 5.0, atol=1e-10)

    def test_summary_string(self):
        """summary() should return a non-empty string after two updates."""
        tracker = EnergyBudgetTracker()
        T, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state()
        fluxes = self._make_fluxes()

        tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                       *fluxes, elapsed_seconds=0.0)
        tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                       *fluxes, elapsed_seconds=3600.0)

        s = tracker.summary()
        assert "Energy Budget Summary" in s
        assert "Residual" in s

    def test_lists_accumulate(self):
        """Tracker lists should grow with each update."""
        tracker = EnergyBudgetTracker()
        T, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state()
        fluxes = self._make_fluxes()

        for i in range(5):
            tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                           *fluxes, elapsed_seconds=float(i * 3600))

        assert len(tracker.times) == 5
        assert len(tracker.toa_net) == 5
        assert len(tracker.residual) == 5

    def test_column_energy_realistic_magnitude(self):
        """Column energy should be O(10^9) J/m² for a realistic atmosphere."""
        tracker = EnergyBudgetTracker()
        T, q_v, u, v, phis, p_s, dsigma, sigma_full = self._make_state()
        fluxes = self._make_fluxes()

        budget = tracker.update(T, q_v, u, v, phis, p_s, dsigma, sigma_full,
                                *fluxes, elapsed_seconds=0.0)
        # c_p * T * p_s / g ≈ 1005 * 280 * 1e5 / 9.81 ≈ 2.87e9 J/m²
        assert 1e9 < budget.column_energy < 5e9
