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
    area_weighted_mean,
    area_weighted_profile,
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


class TestAreaWeightedMean:
    """Tests for the area-weighted global-mean helper.

    The bug it fixes: a plain ``jnp.mean`` over a lat-lon grid weights each
    cell equally, so the polar rows (spanning ~cos(lat) less area) over-count
    and bias every global mean toward the cold high latitudes — on a coupled
    run this faked a -36 W/m² TOA "cold drift" where the area-weighted budget
    is near balance.
    """

    def test_none_area_is_plain_mean(self):
        f = jnp.arange(12.0).reshape(3, 4)
        npt.assert_allclose(float(area_weighted_mean(f, None)), float(jnp.mean(f)))

    def test_uniform_area_equals_plain_mean(self):
        # Cube cells are ~equal area: a uniform weight must reproduce jnp.mean.
        f = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
        npt.assert_allclose(
            float(area_weighted_mean(f, jnp.ones_like(f))), float(jnp.mean(f)),
        )

    def test_unnormalized_weight_ok(self):
        # Weight need not sum to 1; only ratios matter.
        f = jnp.asarray([[1.0, 3.0]])
        w = jnp.asarray([[2.0, 6.0]])  # 8x the canonical weight
        npt.assert_allclose(
            float(area_weighted_mean(f, w)), (1.0 * 2 + 3.0 * 6) / 8.0,
        )

    def test_latlon_cos_lat_recovers_s0_over_4(self):
        """Synthetic equinox TOA insolation (S_0/pi)*cos(lat) area-averages to S_0/4.

        This is the exact failure mode that produced <rsdt>~281: the
        unweighted mean lands at S_0*2/pi^2 ~ 276; the cos(lat) area-weighted
        mean recovers the energy-conserving S_0/4 = 340.
        """
        lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 90))
        S0 = constants.S_0
        rsdt = (S0 / jnp.pi) * jnp.cos(lat)[:, None] * jnp.ones((1, 6))
        area = jnp.cos(lat)[:, None] * jnp.ones((1, 6))
        unweighted = float(jnp.mean(rsdt))
        weighted = float(area_weighted_mean(rsdt, area))
        # Unweighted under-reads; area-weighted nails S_0/4 (uniform-lat
        # quadrature of cos^2 vs cos is exact to <1 W/m² at 90 bands).
        assert unweighted < S0 / 4 - 50.0
        npt.assert_allclose(weighted, S0 / 4.0, atol=1.0)

    def test_trailing_vertical_axis_averaged_uniformly(self):
        # A (H..., nlev) field: vertical averaged uniformly, horizontal
        # area-weighted. A constant field returns the constant.
        f = jnp.ones((3, 4, 7)) * 5.0
        area = jnp.cos(jnp.deg2rad(jnp.linspace(-80, 80, 3)))[:, None] * jnp.ones((1, 4))
        npt.assert_allclose(float(area_weighted_mean(f, area)), 5.0)

    def test_shape_mismatch_falls_back_to_plain_mean(self):
        # An ocean field on a different grid than the supplied area must not
        # broadcast silently — it falls back to an unweighted mean.
        f = jnp.arange(20.0).reshape(5, 4)
        area = jnp.ones((3, 4))  # wrong horizontal shape
        npt.assert_allclose(float(area_weighted_mean(f, area)), float(jnp.mean(f)))

    def test_profile_retains_vertical_and_weights_horizontal(self):
        lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 90))
        area = jnp.cos(lat)[:, None] * jnp.ones((1, 6))
        # Per-level constant -> profile equals that constant at every level.
        field = jnp.ones((90, 6, 3)) * jnp.asarray([2.0, 4.0, 6.0])
        prof = area_weighted_profile(field, area)
        assert prof.shape == (3,)
        npt.assert_allclose(prof, jnp.asarray([2.0, 4.0, 6.0]), atol=1e-10)

    def test_profile_none_area_is_plain_horizontal_mean(self):
        field = jnp.arange(2 * 3 * 4.0).reshape(2, 3, 4)
        npt.assert_allclose(
            area_weighted_profile(field, None), jnp.mean(field, axis=(0, 1)),
        )

    def test_profile_shape_mismatch_falls_back(self):
        # Mismatched horizontal area must not broadcast: fall back to a plain
        # horizontal mean (same guard as the scalar helper).
        field = jnp.arange(5 * 4 * 3.0).reshape(5, 4, 3)
        area = jnp.ones((3, 4))  # wrong horizontal shape
        npt.assert_allclose(
            area_weighted_profile(field, area), jnp.mean(field, axis=(0, 1)),
        )


class TestTrackerAreaWeighting:
    """The trackers must thread ``area_weights`` into their global means."""

    def test_energy_tracker_area_weights_change_toa(self):
        nlat, nlon = 16, 8
        lat = jnp.deg2rad(jnp.linspace(-84.0, 84.0, nlat))
        nlev = 6
        sigma_full, dsigma = _make_sigma(nlev)
        shp = (nlat, nlon, nlev)
        T = jnp.ones(shp) * 280.0
        q_v = jnp.ones(shp) * 1e-3
        u = jnp.zeros(shp)
        v = jnp.zeros(shp)
        phis = jnp.zeros((nlat, nlon))
        p_s = jnp.ones((nlat, nlon)) * 1e5
        # Latitudinally varying TOA SW down so weighting actually matters.
        sw_down = (jnp.cos(lat)[:, None] * jnp.ones((1, nlon))) * 400.0
        sw_up = jnp.ones((nlat, nlon)) * 80.0
        lw_up = jnp.ones((nlat, nlon)) * 240.0
        sw_sfc = jnp.ones((nlat, nlon)) * 150.0
        lw_sfc = jnp.ones((nlat, nlon)) * -50.0
        area = jnp.cos(lat)[:, None] * jnp.ones((1, nlon))

        plain = EnergyBudgetTracker().update(
            T, q_v, u, v, phis, p_s, dsigma, sigma_full,
            sw_down, sw_up, lw_up, sw_sfc, lw_sfc, elapsed_seconds=0.0,
        )
        wtd = EnergyBudgetTracker().update(
            T, q_v, u, v, phis, p_s, dsigma, sigma_full,
            sw_down, sw_up, lw_up, sw_sfc, lw_sfc, elapsed_seconds=0.0,
            area_weights=area,
        )
        # cos(lat) weight emphasises the bright equator -> larger <SW_down>.
        assert wtd.toa_sw_down > plain.toa_sw_down + 5.0
        # The flat fields are unchanged by weighting.
        npt.assert_allclose(wtd.toa_sw_up, plain.toa_sw_up, atol=1e-6)

    def test_moisture_tracker_accepts_area_weights(self):
        nlat, nlon, nlev = 8, 4, 5
        _, dsigma = _make_sigma(nlev)
        q_v = jnp.ones((nlat, nlon, nlev)) * 5e-3
        p_s = jnp.ones((nlat, nlon)) * 1e5
        precip = jnp.ones((nlat, nlon)) * 1e-5
        lat = jnp.deg2rad(jnp.linspace(-80, 80, nlat))
        area = jnp.cos(lat)[:, None] * jnp.ones((1, nlon))
        from legoesm.diagnostics.energy_budget import MoistureBudgetTracker
        b = MoistureBudgetTracker().update(
            q_v, p_s, dsigma, precip, jnp.zeros((nlat, nlon)),
            elapsed_seconds=0.0, area_weights=area,
        )
        # Uniform fields: weighting leaves the means physically sensible.
        assert b.precip_rate > 0.0
        assert b.column_water > 0.0
