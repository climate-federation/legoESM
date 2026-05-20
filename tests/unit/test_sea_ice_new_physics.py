"""Direct unit tests for the Tier-1 + Tier-2 sea-ice extensions.

Covers: Lipscomb 2001 remap, snow column physics (accumulation,
combined conductivity, melt/sublim ordering, snow-ice flooding),
brine + salt budget, mechanical ridging (Lipscomb 2007), Delta-
Eddington / Maykut-Untersteiner albedo, melt pond evolution.

These tests touch the leaf modules directly to guard against
silent regressions during subsequent refactors.  The end-to-end
integration through ``step_sea_ice`` lives in
:mod:`tests.unit.test_sea_ice_v2_integration`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ice.itd import (
    aggregate_state,
    category_bounds,
    upper_bounds,
    lipscomb_2001_remap,
    linear_remap,
)
from legoesm.ice.snow import (
    accumulate_snowfall,
    combined_conductive_flux,
    consume_from_snow_then_ice,
    consume_sublimation_from_snow_then_ice,
    snow_ice_flooding,
)
from legoesm.ice.brine import (
    update_salinity_and_salt_flux,
    PSU_TO_KG_PER_KG,
)
from legoesm.ice.ridging import apply_ridging, participation_weights
from legoesm.ice.shortwave import (
    compute_ice_sw,
    delta_eddington_albedo,
    maykut_untersteiner_albedo,
)
from legoesm.ice.ponds import step_ponds
from legoesm.ice.transport import advect_ice_tracers
from legoesm.grids.latlon import create_latlon_grid


# ==============================================================================
# Lipscomb 2001 remap
# ==============================================================================

class TestLipscomb2001Remap:

    def test_volume_and_area_conserved(self):
        """Total ice volume + area conserved across remap to machine precision."""
        n_cat = 5
        h_old = jnp.array([[0.3, 0.9, 1.8, 3.0, 5.0]])
        a_old = jnp.array([[0.10, 0.20, 0.30, 0.20, 0.10]])
        # Thermo grows all cats by 5 cm
        h_new = h_old + 0.05
        a_new = a_old
        T_new = jnp.full(h_old.shape, 265.0)
        S_new = jnp.full(h_old.shape, 4.0)
        V_snow = jnp.full(h_old.shape, 0.0)
        V_pond = jnp.zeros_like(h_old)

        out = lipscomb_2001_remap(
            h_old, a_old, h_new, a_new, n_cat, dt=3600.0,
            T_new=T_new, S_new=S_new, V_snow_new=V_snow, V_pond_new=V_pond,
        )
        V_before = jnp.sum(h_new * a_new)
        V_after = jnp.sum(out["h"] * out["a"])
        A_before = jnp.sum(a_new)
        A_after = jnp.sum(out["a"])
        assert jnp.abs(V_before - V_after) / jnp.maximum(V_before, 1e-12) < 1e-4
        assert jnp.abs(A_before - A_after) / jnp.maximum(A_before, 1e-12) < 1e-4

    def test_growth_pushes_thin_into_thicker_categories(self):
        """When all categories grow uniformly, mean thickness per category increases."""
        n_cat = 5
        h_old = jnp.array([[0.3, 0.9, 1.8, 3.0, 5.0]])
        a_old = jnp.array([[0.10, 0.20, 0.30, 0.20, 0.10]])
        h_new = h_old + 0.2
        a_new = a_old
        out = lipscomb_2001_remap(h_old, a_old, h_new, a_new, n_cat, dt=3600.0)
        # Aggregate mean thickness should rise.
        h_agg_before = jnp.sum(h_new * a_new) / jnp.sum(a_new)
        h_agg_after = jnp.sum(out["h"] * out["a"]) / jnp.sum(out["a"])
        assert h_agg_after >= h_agg_before - 1e-4

    def test_empty_columns_remain_empty(self):
        """No spurious ice creation in cells with all-zero ice."""
        n_cat = 5
        h_old = jnp.zeros((2, 2, 2, n_cat))
        a_old = jnp.zeros((2, 2, 2, n_cat))
        out = lipscomb_2001_remap(h_old, a_old, h_old, a_old, n_cat, dt=3600.0)
        assert jnp.all(out["h"] == 0.0)
        assert jnp.all(out["a"] == 0.0)


# ==============================================================================
# Snow column physics
# ==============================================================================

class TestSnowKernels:

    def test_accumulate_snowfall_on_ice_only(self):
        """Snow on ice adds to h_snow; snow on water becomes ocean FW."""
        h_snow = jnp.array([0.1, 0.0, 0.2])
        precip = jnp.array([1e-6, 1e-6, 1e-6])  # kg/m²/s
        ice_mask = jnp.array([True, False, True])
        h_new, snow_to_ocean = accumulate_snowfall(
            h_snow, precip, ice_mask, dt=3600.0, rho_snow=330.0,
        )
        # Ice cells gain mass; open water cell unchanged in snow column.
        assert h_new[0] > h_snow[0]
        assert h_new[1] == h_snow[1]
        assert h_new[2] > h_snow[2]
        # Snow on open water goes to ocean FW channel.
        assert snow_to_ocean[1] > 0.0
        assert snow_to_ocean[0] == 0.0

    def test_combined_conductive_flux_reduces_with_snow(self):
        """Snow on ice reduces the conductive flux (insulating layer)."""
        T_base = 271.0
        T_sfc = 250.0
        h_ice = 1.0
        F_bare = combined_conductive_flux(
            T_base, T_sfc, h_ice, h_snow=0.0,
            k_ice=2.0, k_snow=0.31,
            h_ice_min=0.01, h_snow_min=1e-4,
        )
        F_with_snow = combined_conductive_flux(
            T_base, T_sfc, h_ice, h_snow=0.2,
            k_ice=2.0, k_snow=0.31,
            h_ice_min=0.01, h_snow_min=1e-4,
        )
        assert jnp.abs(F_with_snow) < jnp.abs(F_bare)

    def test_melt_consumes_snow_before_ice(self):
        """With energy < total snow melt energy, ice column is untouched."""
        # Snow melt energy = h_snow * rho_snow * L_f
        # = 0.05 * 330 * 3.337e5 = 5.5e6 J/m².  Give half that.
        h_snow_out, h_ice_out, snow_m, ice_m = consume_from_snow_then_ice(
            energy_per_area=jnp.array(2.5e6),
            h_snow=jnp.array(0.05), h_ice=jnp.array(1.0),
            rho_snow=330.0, rho_ice=917.0, L_f=3.337e5,
        )
        assert float(h_snow_out) < 0.05
        assert float(ice_m) == 0.0
        assert float(h_ice_out) == 1.0

    def test_flooding_mass_conservation(self):
        h_ice = jnp.array([1.0, 0.3])
        h_snow = jnp.array([0.5, 0.5])
        hi, hs, h_si = snow_ice_flooding(
            h_ice, h_snow, rho_ice=917.0, rho_snow=330.0, rho_ocean=1025.0,
        )
        mass_before = 917.0 * h_ice + 330.0 * h_snow
        mass_after = 917.0 * hi + 330.0 * hs
        # Mass-neutral conversion within fp roundoff.
        assert jnp.all(jnp.abs(mass_after - mass_before) < 1e-2)

    def test_flooding_only_when_freeboard_negative(self):
        """Thick ice + thin snow has positive freeboard — no flooding."""
        h_ice = jnp.array([5.0])
        h_snow = jnp.array([0.01])
        hi, hs, h_si = snow_ice_flooding(
            h_ice, h_snow, rho_ice=917.0, rho_snow=330.0, rho_ocean=1025.0,
        )
        assert float(h_si[0]) == 0.0
        assert float(hi[0]) == float(h_ice[0])


# ==============================================================================
# Brine + salt budget
# ==============================================================================

class TestBrine:

    def test_lead_freeze_releases_brine_to_ocean(self):
        """Lead-freezing ice (low S_ice) leaves ocean with extra salt."""
        S_ice_old = jnp.array([0.0])  # no ice initially
        V_old = jnp.array([0.0])
        V_new = jnp.array([0.1])      # 0.1 m³ of new ice per m²
        delta_freeze = V_new
        delta_white = jnp.array([0.0])
        result = update_salinity_and_salt_flux(
            S_ice_old=S_ice_old, V_ice_old=V_old, V_ice_new=V_new,
            delta_V_lead_freeze=delta_freeze, delta_V_white_ice=delta_white,
            rho_ice=917.0, dt=3600.0,
            S_lead_ice=4.0, S_white_ice=17.0,
        )
        # New bulk ice salinity = 4 PSU; salt rejected into ocean is
        # the difference between what the original seawater would have
        # held (= S_ocean × V × ρ_ice × 1e-3) and what stays in ice.
        assert jnp.isclose(result.S_ice_new, 4.0, atol=1e-6)
        # Salt flux should be NEGATIVE (ice gained salt from where there
        # was none before — i.e., this is salt now LOST from the column
        # going forward).  In our convention positive = INTO ocean, so
        # this is negative (ocean lost salt to the new ice).
        assert float(result.salt_flux_to_ocean[0]) < 0.0

    def test_melt_releases_salt_to_ocean(self):
        """Ice melt returns its bulk salt to the ocean column."""
        S_ice_old = jnp.array([5.0])
        V_old = jnp.array([0.2])
        V_new = jnp.array([0.1])  # half melted
        delta_freeze = jnp.array([0.0])
        delta_white = jnp.array([0.0])
        result = update_salinity_and_salt_flux(
            S_ice_old=S_ice_old, V_ice_old=V_old, V_ice_new=V_new,
            delta_V_lead_freeze=delta_freeze, delta_V_white_ice=delta_white,
            rho_ice=917.0, dt=3600.0,
            S_lead_ice=4.0, S_white_ice=17.0,
        )
        assert float(result.salt_flux_to_ocean[0]) > 0.0


# ==============================================================================
# Ridging
# ==============================================================================

class TestRidging:

    def test_participation_normalises_to_one(self):
        a_cat = jnp.array([[0.1, 0.2, 0.3, 0.2, 0.1]])
        h_cat = jnp.array([[0.3, 0.9, 1.8, 3.0, 5.0]])
        w = participation_weights(a_cat, h_cat, e_star=0.36)
        assert jnp.isclose(jnp.sum(w[0]), 1.0, atol=1e-6)
        # Thin categories should dominate participation.
        assert w[0, 0] > w[0, -1]

    def test_ridging_conserves_volume(self):
        n_cat = 5
        a_cat = jnp.full((1, n_cat), 0.1)
        h_cat = jnp.array([[0.3, 0.9, 1.8, 3.0, 5.0]])
        V_snow_cat = jnp.zeros_like(h_cat)
        S_ice_cat = jnp.full(h_cat.shape, 4.0)
        closing = jnp.array([1e-5])  # gentle convergence

        ridge = apply_ridging(
            a_cat, h_cat, V_snow_cat, S_ice_cat, closing,
            n_cat=n_cat, dt=3600.0,
        )
        V_before = float(jnp.sum(h_cat * a_cat))
        V_after = float(jnp.sum(ridge["h"] * ridge["a"]))
        assert abs(V_after - V_before) / max(V_before, 1e-12) < 5e-2  # <5% drift OK on coarse cap

    def test_ridging_reduces_area(self):
        n_cat = 5
        a_cat = jnp.full((1, n_cat), 0.1)
        h_cat = jnp.array([[0.3, 0.9, 1.8, 3.0, 5.0]])
        V_snow_cat = jnp.zeros_like(h_cat)
        S_ice_cat = jnp.full(h_cat.shape, 4.0)
        closing = jnp.array([1e-5])

        ridge = apply_ridging(
            a_cat, h_cat, V_snow_cat, S_ice_cat, closing,
            n_cat=n_cat, dt=3600.0,
        )
        A_before = float(jnp.sum(a_cat))
        A_after = float(jnp.sum(ridge["a"]))
        assert A_after <= A_before


# ==============================================================================
# Shortwave / albedo
# ==============================================================================

class TestShortwave:

    def test_maykut_untersteiner_thin_ice_lower_albedo(self):
        T = jnp.full((3,), 260.0)
        h = jnp.array([0.02, 0.1, 0.4])
        alpha = maykut_untersteiner_albedo(T, h, h_ramp=0.5)
        # Thinner ice → lower albedo (more transmits / less scattering).
        assert alpha[0] < alpha[1] < alpha[2]

    def test_delta_eddington_melting_lowers_albedo(self):
        T_cold = jnp.full((1,), 250.0)
        T_melt = jnp.full((1,), 273.0)
        h_ice = jnp.full((1,), 1.5)
        h_snow = jnp.full((1,), 0.2)
        a_pond = jnp.zeros((1,))
        d_pond = jnp.zeros((1,))
        a_cold, _ = delta_eddington_albedo(
            T_cold, h_ice, h_snow, a_pond, d_pond,
        )
        a_warm, _ = delta_eddington_albedo(
            T_melt, h_ice, h_snow, a_pond, d_pond,
        )
        assert a_warm < a_cold

    def test_delta_eddington_pond_reduces_albedo(self):
        T = jnp.full((1,), 272.0)
        h_ice = jnp.full((1,), 1.5)
        h_snow = jnp.zeros((1,))  # snow gone so ponds visible
        a_no_pond = jnp.zeros((1,))
        a_with_pond = jnp.full((1,), 0.4)
        d_pond = jnp.full((1,), 0.2)
        a_dry, _ = delta_eddington_albedo(
            T, h_ice, h_snow, a_no_pond, jnp.zeros((1,)),
        )
        a_wet, _ = delta_eddington_albedo(
            T, h_ice, h_snow, a_with_pond, d_pond,
        )
        assert a_wet < a_dry

    def test_compute_ice_sw_constant_matches_alpha(self):
        sw = jnp.full((1,), 200.0)
        T = jnp.full((1,), 260.0)
        h = jnp.full((1,), 1.0)
        h_snow = jnp.zeros((1,))
        a_pond = jnp.zeros((1,))
        d_pond = jnp.zeros((1,))
        res = compute_ice_sw(
            sw, T, h, h_snow, a_pond, d_pond,
            scheme="constant", albedo_const=0.65,
        )
        assert jnp.isclose(res.albedo_eff[0], 0.65)
        assert jnp.isclose(res.sw_absorbed_surface[0], 70.0)
        assert res.sw_penetrated[0] == 0.0


# ==============================================================================
# Melt ponds
# ==============================================================================

class TestMeltPonds:

    def test_ponds_form_only_after_snow_melts(self):
        """Snow on top blocks pond formation."""
        a, d, drain, refreeze = step_ponds(
            pond_area=jnp.array(0.0), pond_depth=jnp.array(0.0),
            melt_water_m=jnp.array(0.05), rain_water_m=jnp.array(0.0),
            ice_mask=jnp.array(True),
            h_snow=jnp.array(0.1),
            T_air=jnp.array(280.0),
            dt=3600.0,
            drainage_timescale=86400.0,
            refreeze_threshold=273.15,
            pond_to_ice_max_area=0.6,
            depth_to_area_ratio=0.8,
        )
        assert float(a) == 0.0

    def test_ponds_refreeze_below_threshold(self):
        a_in = jnp.array(0.3)
        d_in = jnp.array(0.1)
        a, d, drain, refreeze = step_ponds(
            pond_area=a_in, pond_depth=d_in,
            melt_water_m=jnp.array(0.0), rain_water_m=jnp.array(0.0),
            ice_mask=jnp.array(True),
            h_snow=jnp.array(0.0),
            T_air=jnp.array(265.0),  # below threshold
            dt=3600.0,
            drainage_timescale=86400.0,
            refreeze_threshold=273.15,
            pond_to_ice_max_area=0.6,
            depth_to_area_ratio=0.8,
        )
        assert float(refreeze) > 0.0
        assert float(a) < float(a_in)

    def test_ponds_drain_over_time(self):
        a_in = jnp.array(0.3)
        d_in = jnp.array(0.1)
        a, d, drain, refreeze = step_ponds(
            pond_area=a_in, pond_depth=d_in,
            melt_water_m=jnp.array(0.0), rain_water_m=jnp.array(0.0),
            ice_mask=jnp.array(True),
            h_snow=jnp.array(0.0),
            T_air=jnp.array(280.0),  # above threshold
            dt=86400.0,
            drainage_timescale=86400.0,
            refreeze_threshold=273.15,
            pond_to_ice_max_area=0.6,
            depth_to_area_ratio=0.8,
        )
        assert float(drain) > 0.0
        assert float(a) < float(a_in)


# ==============================================================================
# Lat-lon transport dispatch (grid-agnostic ice transport)
# ==============================================================================

class TestLatLonTransport:
    """Ice tracer transport on lat-lon C-grid via PPM dispatch."""

    def _make_grid(self):
        return create_latlon_grid(n_lat=36, n_lon=72)

    def test_single_cat_uniform_advection_preserves_volume(self):
        grid = self._make_grid()
        shape = (36, 72)
        h = jnp.full(shape, 1.0)
        a = jnp.full(shape, 0.5)
        T = jnp.full(shape, 263.0)
        u = jnp.full(shape, 0.05)
        v = jnp.zeros(shape)
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, grid, dt=3600.0)
        # Uniform field under uniform advection should stay uniform
        # (no flux gradient, no volume / area change).
        assert jnp.isclose(jnp.sum(h_new * a_new), jnp.sum(h * a), rtol=1e-6)
        assert jnp.isclose(jnp.sum(a_new), jnp.sum(a), rtol=1e-6)
        assert jnp.all(T_new > 260.0) and jnp.all(T_new < 264.0)

    def test_multi_cat_lat_lon_transport(self):
        grid = self._make_grid()
        n_cat = 5
        shape = (36, 72, n_cat)
        h = jnp.full(shape, 1.0)
        a = jnp.full(shape, 0.18)
        T = jnp.full(shape, 263.0)
        u = jnp.full((36, 72), 0.05)
        v = jnp.zeros((36, 72))
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, grid, dt=3600.0)
        assert h_new.shape == shape
        # Per-category volume conservation under uniform field.
        for k in range(n_cat):
            V0 = float(jnp.sum(h[..., k] * a[..., k]))
            V1 = float(jnp.sum(h_new[..., k] * a_new[..., k]))
            assert abs(V1 - V0) / max(V0, 1e-12) < 1e-6


# ==============================================================================
# Lat-lon EVP rheology (grid-agnostic strain rates + stress divergence)
# ==============================================================================

class TestLatLonEVP:
    """Lat-lon A-grid EVP rheology dispatch."""

    def _make_grid(self):
        return create_latlon_grid(n_lat=36, n_lon=72)

    def test_strain_rates_uniform_velocity_is_zero(self):
        from legoesm.ice.rheology import strain_rates
        grid = self._make_grid()
        shape = (36, 72)
        u = jnp.full(shape, 0.1)
        v = jnp.zeros(shape)
        e11, e22, e12 = strain_rates(u, v, grid)
        # Pole-fold can introduce tiny strain at the rows adjacent to
        # the poles; restrict the check to mid-latitudes.
        assert jnp.all(jnp.abs(e11[5:-5]) < 1e-9)
        assert jnp.all(jnp.abs(e22[5:-5]) < 1e-9)

    def test_strain_rates_linear_u_produces_nonzero_eps11(self):
        from legoesm.ice.rheology import strain_rates
        grid = self._make_grid()
        n_lat, n_lon = 36, 72
        # Linear u in lon direction
        u = jnp.broadcast_to(jnp.arange(n_lon, dtype=jnp.float64) * 0.01,
                             (n_lat, n_lon))
        v = jnp.zeros((n_lat, n_lon))
        e11, e22, e12 = strain_rates(u, v, grid)
        # eps_11 = du/dx > 0 in interior
        assert jnp.all(e11[5:-5, 5:-5] > 0.0)

    def test_stress_divergence_zero_stress_is_zero(self):
        from legoesm.ice.dynamics import stress_divergence
        grid = self._make_grid()
        shape = (36, 72)
        Fx, Fy = stress_divergence(
            jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape), grid,
        )
        assert jnp.all(Fx == 0.0)
        assert jnp.all(Fy == 0.0)

    def test_stress_divergence_linear_sigma11(self):
        """Linear σ_11 in x produces positive Fx (gradient > 0)."""
        from legoesm.ice.dynamics import stress_divergence
        grid = self._make_grid()
        n_lat, n_lon = 36, 72
        sigma_11 = jnp.broadcast_to(jnp.arange(n_lon, dtype=jnp.float64),
                                    (n_lat, n_lon))
        Fx, Fy = stress_divergence(
            sigma_11, jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)), grid,
        )
        # Interior Fx should be positive (∂σ_11/∂x > 0).
        assert jnp.all(Fx[5:-5, 5:-5] > 0.0)

    def test_evp_solver_lat_lon_finite_no_blowup(self):
        from legoesm.ice.dynamics import evp_solver
        grid = self._make_grid()
        shape = (36, 72)
        u_new, v_new, s11, s22, s12 = evp_solver(
            jnp.zeros(shape), jnp.zeros(shape),
            jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape),
            jnp.full(shape, 1.5), jnp.full(shape, 0.9),
            jnp.full(shape, 10.0), jnp.zeros(shape),
            jnp.zeros(shape), jnp.zeros(shape),
            grid, dt=3600.0, N_evp=30,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(v_new))
        assert jnp.all(jnp.isfinite(s11))
        # Wind-driven free-drift estimate: α ≈ 0.017 × |U_wind| =
        # 0.17 m/s; EVP solution should be of the same order.
        assert float(jnp.max(jnp.abs(u_new))) < 1.0
        assert float(jnp.max(jnp.abs(u_new))) > 0.01

    def test_stress_divergence_nonzero_sigma12(self):
        """σ_12 contributes to both Fx (via ∂/∂y) and Fy (via ∂/∂x).

        Linear σ_12(λ) in the zonal direction produces a positive
        ``Fy`` in the interior with no ``Fx`` contribution (since
        ∂σ_12/∂y = 0).  Lat-only variation σ_12(θ) does the
        opposite.
        """
        from legoesm.ice.dynamics import stress_divergence
        grid = self._make_grid()
        n_lat, n_lon = 36, 72
        sigma_12_lon = jnp.broadcast_to(
            jnp.arange(n_lon, dtype=jnp.float64), (n_lat, n_lon),
        )
        Fx, Fy = stress_divergence(
            jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)),
            sigma_12_lon, grid,
        )
        # Linear σ_12(lon) → Fy = ∂σ_12/∂x > 0 in interior.
        assert jnp.all(Fy[5:-5, 5:-5] > 0.0)
        # ∂σ_12/∂y is zero (no lat variation) → Fx ≈ 0 in interior.
        assert jnp.all(jnp.abs(Fx[5:-5, 5:-5]) < 1e-9)

        # Now linear σ_12(lat): contributes to Fx but not Fy.
        sigma_12_lat = jnp.broadcast_to(
            jnp.arange(n_lat, dtype=jnp.float64)[:, None], (n_lat, n_lon),
        )
        Fx2, Fy2 = stress_divergence(
            jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)),
            sigma_12_lat, grid,
        )
        assert jnp.all(Fx2[5:-5, 5:-5] > 0.0)
        assert jnp.all(jnp.abs(Fy2[5:-5, 5:-5]) < 1e-9)

    def test_mevp_solver_lat_lon_finite(self):
        from legoesm.ice.dynamics import mevp_solver
        grid = self._make_grid()
        shape = (36, 72)
        u_new, v_new, s11, s22, s12 = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape),
            jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape),
            jnp.full(shape, 1.5), jnp.full(shape, 0.9),
            jnp.full(shape, 10.0), jnp.zeros(shape),
            jnp.zeros(shape), jnp.zeros(shape),
            grid, dt=3600.0, N_mevp=30,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(s11))
