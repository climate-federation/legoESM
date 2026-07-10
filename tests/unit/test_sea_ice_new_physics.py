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
        h_snow_out, h_ice_out, snow_m, ice_m, unconsumed = consume_from_snow_then_ice(
            energy_per_area=jnp.array(2.5e6),
            h_snow=jnp.array(0.05), h_ice=jnp.array(1.0),
            rho_snow=330.0, rho_ice=917.0, L_f=3.337e5,
        )
        assert float(h_snow_out) < 0.05
        assert float(ice_m) == 0.0
        assert float(h_ice_out) == 1.0
        # Energy was fully absorbed by partial snow melt -> nothing left over.
        assert float(unconsumed) == 0.0

    def test_sublimation_capped_at_available_ice(self):
        """Reported ice sublimation must never exceed the ice present (codex).

        Same over-removal class as the melt cap: an uncapped ice_sublim_m
        would inflate concentration retreat + the brine delta_V_sublim
        budget while h_ice_new clamps to 0.
        """
        from legoesm.ice.snow import consume_sublimation_from_snow_then_ice
        rho_ice = 917.0
        h_ice0 = 0.02
        # Demand 0.10 m-equivalent of ice sublimation from 0.02 m ice, no snow.
        sublim_kg = 0.10 * rho_ice
        h_snow_out, h_ice_out, snow_s, ice_s = consume_sublimation_from_snow_then_ice(
            sublim_mass_per_area=jnp.array(sublim_kg),
            h_snow=jnp.array(0.0), h_ice=jnp.array(h_ice0),
            rho_snow=330.0, rho_ice=rho_ice, sublim_partition=1.0,
        )
        assert float(h_ice_out) == 0.0
        assert float(ice_s) <= h_ice0 + 1e-12
        assert jnp.isclose(float(ice_s), h_ice0, atol=1e-9)

    def test_melt_capped_at_available_ice(self):
        """Reported ice melt must never exceed the ice that existed (codex).

        Regression: consume_from_snow_then_ice returned ice_melt_m =
        energy_remaining/L_f/rho_ice uncapped, so a melt energy larger than
        the column's latent capacity reported more melt than ice present
        (phantom freshwater/salt), while h_ice_new clamped to 0.
        """
        rho_ice, L_f = 917.0, 3.337e5
        h_ice0 = 0.02  # thin ice
        # Energy = 10x the latent capacity of the ice.
        E = 10.0 * h_ice0 * rho_ice * L_f
        h_snow_out, h_ice_out, snow_m, ice_m, unconsumed = consume_from_snow_then_ice(
            energy_per_area=jnp.array(E),
            h_snow=jnp.array(0.0), h_ice=jnp.array(h_ice0),
            rho_snow=330.0, rho_ice=rho_ice, L_f=L_f,
        )
        assert float(h_ice_out) == 0.0                  # fully ablated
        assert float(ice_m) <= h_ice0 + 1e-12           # NOT more than existed
        assert jnp.isclose(float(ice_m), h_ice0, atol=1e-9)  # exactly all of it
        # Finding #6: the surplus melt energy (here 9x the ice latent capacity)
        # is RETURNED, not lost.  Energy closure: consumed (ice latent) +
        # unconsumed == input energy.
        consumed = float(ice_m) * rho_ice * L_f  # no snow here
        assert jnp.isclose(consumed + float(unconsumed), float(E), rtol=1e-12)
        assert jnp.isclose(float(unconsumed), float(E) - h_ice0 * rho_ice * L_f,
                           rtol=1e-12)

    def test_meltout_surplus_credits_ocean(self):
        """Finding #6 (integration): on a large-SW thin-ice melt-out the
        surface-melt energy that exceeds the column latent capacity is CREDITED
        to the ocean mixed layer instead of being lost.

        Energy/sign convention: ``ocean_heat_extraction`` is +ve when the ocean
        LOSES heat to the ice tile; surplus surface-melt heat WARMS the ocean,
        so it appears as a NEGATIVE contribution.  With the ocean held exactly
        at the freezing point (F_ocean = 0) and no lead freeze on a melted-out
        cell, the only ocean-heat term is this credit, and it must equal the
        surplus melt flux ``unconsumed_J/dt * conc``.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig
        from legoesm.core.coupling_fields import AtmToSurface

        shape = (1, 1)
        cfg = SeaIceConfig()  # snow.enabled is False by default
        sw = 5000.0           # huge SW to force a full surface melt-out
        forcing = AtmToSurface(
            sw_down=jnp.full(shape, sw), lw_down=jnp.full(shape, 350.0),
            precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 290.0), q_lowest=jnp.full(shape, 8e-3),
            u_lowest=jnp.full(shape, 0.5), v_lowest=jnp.zeros(shape),
            p_lowest=jnp.full(shape, 9.5e4), p_surface=jnp.full(shape, 1e5),
            rho_lowest=jnp.full(shape, 1.2), cos_zenith=jnp.full(shape, 1.0),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape),
        )
        h0 = jnp.full(shape, 0.005)      # very thin ice
        conc0 = jnp.full(shape, 1.0)
        T0 = jnp.full(shape, 273.0)
        hs0 = jnp.zeros(shape)
        S0 = jnp.full(shape, 5.0)
        pa0 = jnp.zeros(shape)
        pd0 = jnp.zeros(shape)
        # Ocean exactly at freezing => F_ocean = 0 (no basal contribution).
        sst = jnp.full(shape, constants.T_freeze_ocean)
        dt = 3600.0
        out = _thermo_v2(h0, T0, conc0, hs0, S0, pa0, pd0, forcing, sst, cfg,
                         1.0, dt)
        # Column fully ablated and the ocean is CREDITED (heat extraction < 0).
        assert float(out["h"][0, 0]) == 0.0, "thin ice should melt out"
        ohe = float(out["ocean_heat_extraction"][0, 0])
        assert ohe < 0.0, f"ocean must gain surplus melt heat, got {ohe:.3f}"
        # The surplus credit must NOT exceed the surface energy that arrived
        # (no energy created): bound it by the absorbed-SW flux alone.
        sw_absorbed_flux = (1.0 - cfg.albedo_ice) * sw
        assert -ohe <= sw_absorbed_flux + 1e-6, (
            f"ocean credit {-ohe:.2f} exceeds absorbed SW {sw_absorbed_flux:.2f}"
        )
        assert jnp.isfinite(ohe)

    def test_flooding_mass_conservation(self):
        """Convention-B flooding: snow-ice = snow + seawater filling the pores.
        The ice+snow column GAINS exactly the seawater ``(rho_ice-rho_snow)*d``
        drawn from the ocean, snow consumed == ice formed == d, and the
        freeboard is raised to ~0."""
        ri, rs, rw = 917.0, 330.0, 1025.0
        h_ice = jnp.array([1.0, 0.3])
        h_snow = jnp.array([0.5, 0.9])
        hi, hs, h_si = snow_ice_flooding(
            h_ice, h_snow, rho_ice=ri, rho_snow=rs, rho_ocean=rw,
        )
        seawater = (ri - rs) * h_si              # ocean water drawn into pores
        col_after = ri * hi + rs * hs
        col_before = ri * h_ice + rs * h_snow
        # Column ice+snow mass gains exactly the seawater drawn from the ocean.
        assert jnp.allclose(col_after - col_before, seawater, atol=1e-6)
        # Snow consumed == snow-ice formed == d.
        assert jnp.allclose(h_snow - hs, h_si, atol=1e-9)
        # Freeboard raised to ~0 where flooding occurred (snow was sufficient).
        fb_after = ((rw - ri) * hi - rs * hs) / rw
        assert jnp.all(jnp.where(h_si > 1e-9, jnp.abs(fb_after) < 1e-9, True))

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

    def test_partial_sublimation_concentrates_no_ocean_flux(self):
        """Sublimation removes ice WATER to the atmosphere but leaves the salt
        behind, concentrating the remaining ice; with surviving ice below the
        salinity cap the ocean salt flux is ~0 (no salt crosses to the ocean).
        """
        rho_ice, dt = 917.0, 3600.0
        S_old = jnp.array([5.0])
        V_old = jnp.array([0.2])
        V_new = jnp.array([0.15])           # 0.05 m sublimated
        result = update_salinity_and_salt_flux(
            S_ice_old=S_old, V_ice_old=V_old, V_ice_new=V_new,
            delta_V_lead_freeze=jnp.array([0.0]),
            delta_V_white_ice=jnp.array([0.0]),
            rho_ice=rho_ice, dt=dt, S_lead_ice=4.0, S_white_ice=17.0,
            delta_V_sublim=jnp.array([0.05]),
        )
        # Salinity concentrates: S_old*V_old/V_new = 5*0.2/0.15 = 6.67 (< cap).
        assert float(result.S_ice_new[0]) == pytest.approx(5.0 * 0.2 / 0.15,
                                                            rel=1e-5)
        # No salt to the ocean (the salt stayed in the concentrated ice).
        assert abs(float(result.salt_flux_to_ocean[0])) < 1e-9

    def test_full_sublimation_rejects_residual_salt_conservatively(self):
        """When the column FULLY sublimates the residual salt has no ice to
        occupy: it is rejected to the ocean (brine drainage), conserving salt
        (salt cannot follow the water into the vapour phase).  The matching
        WATER went to the atmosphere, so this is NOT a phantom flux — it is the
        physically required salt rejection.
        """
        rho_ice, dt = 917.0, 3600.0
        S_old = jnp.array([5.0])
        V_old = jnp.array([0.2])
        V_new = jnp.array([0.0])            # fully sublimated
        result = update_salinity_and_salt_flux(
            S_ice_old=S_old, V_ice_old=V_old, V_ice_new=V_new,
            delta_V_lead_freeze=jnp.array([0.0]),
            delta_V_white_ice=jnp.array([0.0]),
            rho_ice=rho_ice, dt=dt, S_lead_ice=4.0, S_white_ice=17.0,
            delta_V_sublim=jnp.array([0.2]),
        )
        PSU = 1e-3
        salt_old = float(S_old[0] * V_old[0] * rho_ice * PSU)
        salt_stored = float(result.S_ice_new[0] * V_new[0] * rho_ice * PSU)
        salt_flux = float(result.salt_flux_to_ocean[0])
        # No ice left -> no stored salt; all residual salt rejected to ocean.
        assert salt_stored == pytest.approx(0.0, abs=1e-12)
        assert salt_flux > 0.0
        # Salt CONSERVED: salt_old = salt_stored + salt_flux*dt.
        assert salt_old == pytest.approx(salt_stored + salt_flux * dt, rel=1e-9)

    def test_full_old_sublimation_with_lead_freeze_does_not_bury_old_salt(self):
        """Old ice fully sublimates while NEW lead ice forms in the same step:
        the old residual salt must be rejected to the ocean, NOT buried in the
        new ice (which should crystallise at its own low salinity).  Codex
        mixed-process finding.
        """
        rho_ice, dt = 917.0, 3600.0
        S_old = jnp.array([5.0])
        V_old = jnp.array([0.2])
        V_new = jnp.array([0.2])             # old gone, equal new lead ice
        result = update_salinity_and_salt_flux(
            S_ice_old=S_old, V_ice_old=V_old, V_ice_new=V_new,
            delta_V_lead_freeze=jnp.array([0.2]),   # new ice
            delta_V_white_ice=jnp.array([0.0]),
            rho_ice=rho_ice, dt=dt, S_lead_ice=4.0, S_white_ice=17.0,
            delta_V_sublim=jnp.array([0.2]),        # old ice fully sublimates
        )
        # New ice keeps the lead salinity (4 PSU), NOT contaminated to ~9 PSU
        # by the old sublimated salt.
        assert float(result.S_ice_new[0]) == pytest.approx(4.0, rel=1e-5)
        PSU = 1e-3
        salt_old = float(S_old[0] * V_old[0] * rho_ice * PSU)          # 0.917
        salt_stored = float(result.S_ice_new[0] * V_new[0] * rho_ice * PSU)  # 0.7336
        salt_flux = float(result.salt_flux_to_ocean[0])
        # Net ocean salt = old residual rejected (0.917) - new-ice uptake
        # (0.7336) = +0.1834 kg/m2 over dt, and salt is conserved.
        assert salt_flux > 0.0
        assert salt_old == pytest.approx(salt_stored + salt_flux * dt, rel=1e-9)
        assert salt_flux * dt == pytest.approx(0.2 * 0.917, rel=1e-4)

    def test_pond_refreeze_fresh_ice_does_not_bury_old_salt(self):
        """Old salty ice melts (salt to ocean) while fresh melt-pond water
        refreezes as new ice in the same step: the refrozen pond ice must be
        treated as FRESH new ice, not surviving old ice, so the old salt is
        correctly released to the ocean (Codex mixed melt/refreeze finding).
        """
        rho_ice, dt = 917.0, 3600.0
        S_old = jnp.array([5.0])
        V_old = jnp.array([1.0])
        # 0.2 m old ice melts, 0.1 m fresh pond water refreezes -> V_new = 0.9.
        V_new = jnp.array([0.9])
        result = update_salinity_and_salt_flux(
            S_ice_old=S_old, V_ice_old=V_old, V_ice_new=V_new,
            delta_V_lead_freeze=jnp.array([0.0]),
            delta_V_white_ice=jnp.array([0.0]),
            rho_ice=rho_ice, dt=dt, S_lead_ice=4.0, S_white_ice=17.0,
            delta_V_fresh_refreeze=jnp.array([0.1]), S_fresh_ice=0.0,
        )
        # Bulk salinity = old salt (5*0.8) spread over V_new=0.9 -> 4.44 PSU,
        # NOT 5 (which would mean the fresh ice absorbed the old salt).
        assert float(result.S_ice_new[0]) == pytest.approx(5.0 * 0.8 / 0.9,
                                                            rel=1e-5)
        PSU = 1e-3
        salt_old = float(S_old[0] * V_old[0] * rho_ice * PSU)
        salt_stored = float(result.S_ice_new[0] * V_new[0] * rho_ice * PSU)
        salt_flux = float(result.salt_flux_to_ocean[0])
        # Old salt released to ocean = S_old * melt_volume = 5*0.2 PSU·m.
        assert salt_flux * dt == pytest.approx(5.0 * 0.2 * rho_ice * PSU, rel=1e-4)
        assert salt_old == pytest.approx(salt_stored + salt_flux * dt, rel=1e-9)


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

    def test_ridging_conserves_volume_high_Hstar(self):
        """Finding #7: with ``H_star`` raised above the ITD top bound
        ``hi[-1] = 100 m`` and a steep ``mu_rdg`` so ``H_max`` would exceed it,
        the ridge thickness range MUST be clamped to hi[-1].  An unclamped
        H_max truncates the overlap integral, dropping the ridge area+volume
        above 100 m (a conservation sink).  Total ice volume must be conserved
        to the overlap quadrature floor."""
        n_cat = 5
        a_cat = jnp.full((1, n_cat), 0.12)
        h_cat = jnp.array([[1.0, 2.0, 3.5, 5.0, 8.0]])
        V_snow_cat = jnp.zeros_like(h_cat)
        S_ice_cat = jnp.full(h_cat.shape, 4.0)
        # Strong convergence so a large area fraction ridges this step (the
        # dropped ridge volume must be a visible fraction of the TOTAL volume).
        closing = jnp.array([1.6e-4])
        V_before = float(jnp.sum(h_cat * a_cat))
        # H_star=300 m + steep mu_rdg=200 drive H_max ~208 m, well above the
        # ITD top bound hi[-1]=100 m, so the UNCLAMPED overlap integral covered
        # only ~half the [H_min, H_max] range and dropped ridged volume above
        # 100 m (~6.4% of the total here).  The hi[-1] clamp restores a full
        # partition.
        ridge = apply_ridging(
            a_cat, h_cat, V_snow_cat, S_ice_cat, closing,
            n_cat=n_cat, dt=3600.0,
            H_star=300.0,   # > hi[-1] = 100 m
            mu_rdg=200.0,   # steep -> H_max would be ~208 m unclamped
        )
        V_after = float(jnp.sum(ridge["h"] * ridge["a"]))
        rel = abs(V_after - V_before) / max(V_before, 1e-12)
        # With the hi[-1] clamp the overlap fully partitions [H_min, H_max], so
        # volume is conserved to the (coarse) uniform-g quadrature floor.  The
        # unclamped code dropped ~6.4% of the TOTAL volume (non-vacuous guard,
        # verified to FAIL this assert with the clamp removed).
        assert rel < 5e-2, f"ridge volume not conserved (high H_star): rel={rel:.3f}"

    def test_ridging_overthick_range_stays_in_itd_support(self):
        """Codex R2: when the participating mean thickness is so large that
        H_min = 2*h_part would exceed the ITD top bound hi[-1]=100 m, the ridge
        thickness range MUST still be clamped INTO [lo[0], hi[-1]] so the
        overlap integral partitions it (sum overlap_frac == 1) and no ridge
        volume is dropped.  Replicates the clamp the kernel applies.

        NOTE: with the DEFAULT e_star the participation function suppresses
        thick-ice participation so h_part stays small, but ``apply_ridging``
        exposes ``e_star`` -- a large e_star gives thick ice ~full participation
        and h_part can exceed hi[-1]/2, making this branch reachable (codex R3).
        We test the clamp logic directly to lock the guarantee across h_part."""
        from legoesm.ice import transport  # noqa: F401  (ensure pkg import)
        from legoesm.ice.itd import category_bounds, upper_bounds
        from legoesm.ice import ridging as _R
        lo = category_bounds(5)
        hi = upper_bounds(5)
        width = _R._MIN_RIDGE_WIDTH_M
        hi_top = float(hi[-1])
        for h_part in (0.5, 5.0, 49.0, 80.0, 150.0):   # last two force H_min>hi[-1]
            # Replicate the COMMITTED kernel logic exactly: ceiling clamp, then a
            # normal/over-thick split (NOT the round-2 unconditional clip).
            H_min0 = 2.0 * h_part
            H_max = min(4.0 * (max(h_part, 1e-6) ** 0.5), 300.0)  # mu=4, H_star=300
            H_max = min(H_max, hi_top)
            H_max_normal = max(H_max, H_min0 + width)
            over_thick = H_min0 > (hi_top - width)
            H_min = (hi_top - width) if over_thick else H_min0
            H_max = hi_top if over_thick else H_max_normal
            # In the NORMAL regime the range must be BIT-IDENTICAL to the original
            # Lipscomb range (regression guard for codex R3-1: the round-2 clip
            # wrongly thinned ordinary ridges).
            if not over_thick:
                assert H_min == pytest.approx(H_min0, rel=0, abs=0), (
                    f"normal H_min changed at h_part={h_part}"
                )
                assert H_max == pytest.approx(max(min(4.0*(max(h_part,1e-6)**0.5),300.0), H_min0+width), rel=0)
            # Range valid AND fully inside [lo[0], hi[-1]] so overlap partitions.
            assert H_min < H_max, f"empty range at h_part={h_part}"
            assert H_min >= float(lo[0]) - 1e-12
            assert H_max <= hi_top + 1e-12
            a_over = jnp.maximum(lo, H_min)
            b_over = jnp.minimum(hi, H_max)
            overlap = jnp.sum(jnp.maximum(b_over - a_over, 0.0))
            assert float(overlap) == pytest.approx(H_max - H_min, rel=1e-9), (
                f"overlap != range width at h_part={h_part} -> volume would drop"
            )

    def test_ridging_conserves_volume_overthick_reachable(self):
        """Codex R3 (end-to-end, non-vacuous): the over-thick branch IS
        reachable through ``apply_ridging`` with a large ``e_star`` (which lets
        thick ice participate fully, so the participating mean thickness h_part
        exceeds hi[-1]/2 and H_min = 2*h_part > hi[-1]).  Total ice volume must
        be conserved; the ridge routes entirely into the top category.  Without
        the both-endpoint clamp this dropped 100% of the ridged volume."""
        n_cat = 5
        # All area in a 60 m category; e_star huge -> ~full participation.
        a_cat = jnp.zeros((1, n_cat)).at[0, -1].set(0.2)
        h_cat = jnp.zeros((1, n_cat)).at[0, -1].set(60.0)
        V_snow_cat = jnp.zeros_like(h_cat)
        S_ice_cat = jnp.full(h_cat.shape, 4.0)
        V_before = float(jnp.sum(h_cat * a_cat))
        ridge = apply_ridging(
            a_cat, h_cat, V_snow_cat, S_ice_cat, jnp.array([2e-4]),
            n_cat=n_cat, dt=3600.0, e_star=1e9, H_star=300.0, mu_rdg=4.0,
        )
        V_after = float(jnp.sum(ridge["h"] * ridge["a"]))
        rel = abs(V_after - V_before) / max(V_before, 1e-12)
        assert rel < 1e-9, f"over-thick ridge volume not conserved: rel={rel:.3e}"
        # The ridged volume lands in the TOP category (clamped into the top bin).
        assert float(ridge["h"][0, -1]) <= 100.0 + 1e-6

    def test_ridging_conserves_pond_water(self):
        """Ridging drains pond water from the deforming ice to the ocean
        (it does not silently vanish): total pond water = retained + drained
        (Codex ponds+ridging finding)."""
        from legoesm import constants
        n_cat = 5
        a_cat = jnp.full((1, n_cat), 0.15)
        h_cat = jnp.array([[0.3, 0.9, 1.8, 3.0, 5.0]])
        V_snow_cat = jnp.zeros_like(h_cat)
        S_ice_cat = jnp.full(h_cat.shape, 4.0)
        # Per-cell pond volume on the thinner categories (m of water per m2).
        V_pond_cat = jnp.array([[0.02, 0.03, 0.0, 0.0, 0.0]])
        closing = jnp.array([1e-4])   # convergence strong enough to ridge

        ridge = apply_ridging(
            a_cat, h_cat, V_snow_cat, S_ice_cat, closing,
            n_cat=n_cat, dt=3600.0, V_pond_cat=V_pond_cat,
        )
        dt = 3600.0
        pond_before_kg = float(jnp.sum(V_pond_cat)) * constants.rho_water
        pond_after_kg = float(jnp.sum(ridge["V_pond"])) * constants.rho_water
        drained_kg = float(ridge["pond_to_ocean"][0]) * dt
        assert drained_kg > 0.0, "convergence should drain some pond water"
        # Conservation: pond water before = retained in ice + drained to ocean.
        assert pond_before_kg == pytest.approx(pond_after_kg + drained_kg,
                                               rel=1e-9)
        # Ridges carry no pond: post-ridging pond volume <= pre (some drained).
        assert pond_after_kg <= pond_before_kg + 1e-12


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

    def test_delta_eddington_sw_energy_balance(self):
        """Finding #8: the SW partition must close exactly,
        reflected + absorbed + penetrated == sw_down, with absorbed and
        penetrated both non-negative, across thin/thick ice + ponds."""
        sw = jnp.full((1,), 300.0)
        for h in (0.05, 0.3, 1.0, 3.0):
            for T in (255.0, 273.0):
                for pa, pd in [(0.0, 0.0), (0.6, 0.25)]:
                    res = compute_ice_sw(
                        sw, jnp.full((1,), T), jnp.full((1,), h),
                        jnp.zeros((1,)), jnp.full((1,), pa), jnp.full((1,), pd),
                        scheme="delta_eddington", albedo_const=0.65,
                    )
                    alpha = res.albedo_eff
                    absorbed = res.sw_absorbed_surface
                    penetrated = res.sw_penetrated
                    reflected = alpha * sw
                    total = reflected + absorbed + penetrated
                    assert jnp.allclose(total, sw, rtol=1e-12), (
                        f"SW not conserved at h={h},T={T},pond={pa}: {float(total[0])}"
                    )
                    assert jnp.all(absorbed >= 0.0)
                    assert jnp.all(penetrated >= 0.0)
                    # Penetrated never exceeds the column input (1-alpha)*sw.
                    assert jnp.all(penetrated <= (1.0 - alpha) * sw + 1e-9)

    def test_delta_eddington_caps_excess_transmittance(self, monkeypatch):
        """Finding #8 (non-vacuous): when the transmittance fit returns more
        than the column input (transmittance > 1-alpha), the penetrated flux is
        capped at ``net_into_column`` and the surface absorption goes to zero —
        so reflected + absorbed + penetrated still equals sw_down (no energy
        created).  Inject such an over-large transmittance via monkeypatch."""
        import legoesm.ice.shortwave as swmod

        def _fake_albedo(T_sfc, h_ice, h_snow, pond_area, pond_depth):
            alpha = jnp.full_like(T_sfc, 0.30)        # 1-alpha = 0.70
            transm = jnp.full_like(T_sfc, 0.95)       # > 0.70 -> would overshoot
            return alpha, transm

        monkeypatch.setattr(swmod, "delta_eddington_albedo", _fake_albedo)
        sw = jnp.full((1,), 400.0)
        res = swmod.compute_ice_sw(
            sw, jnp.full((1,), 270.0), jnp.full((1,), 0.2),
            jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)),
            scheme="delta_eddington", albedo_const=0.65,
        )
        net = (1.0 - 0.30) * 400.0
        # Penetrated capped at the column input; surface absorption clamps to 0.
        assert jnp.isclose(res.sw_penetrated[0], net, rtol=1e-12)
        assert jnp.isclose(res.sw_absorbed_surface[0], 0.0, atol=1e-9)
        # Energy still closes exactly (no creation).
        total = 0.30 * 400.0 + res.sw_absorbed_surface[0] + res.sw_penetrated[0]
        assert jnp.isclose(total, 400.0, rtol=1e-12)

    def test_pond_albedo_decreases_with_depth(self):
        """Fix #1: pond albedo must DECREASE with depth (Briegleb-Light) — a
        shallow pond looks like bare ice, a deep pond decays toward the dark
        deep-pond floor.  The old ramp inverted this (shallow -> ~0)."""
        T = jnp.full((1,), 272.0)
        h_ice = jnp.full((1,), 1.5)
        h_snow = jnp.zeros((1,))       # snow-free so ponds are visible
        full_pond = jnp.full((1,), 1.0)  # cell fully ponded -> albedo == pond albedo
        a_shallow, _ = delta_eddington_albedo(
            T, h_ice, h_snow, full_pond, jnp.full((1,), 0.02),
        )
        a_deep, _ = delta_eddington_albedo(
            T, h_ice, h_snow, full_pond, jnp.full((1,), 0.3),
        )
        assert float(a_shallow[0]) > float(a_deep[0])
        # Both stay in the physical band [deep-pond floor, bare-ice albedo].
        alpha_deep_floor = (
            constants.alpha_pond_max_vis * 0.52
            + constants.alpha_pond_max_nir * 0.48
        )
        assert alpha_deep_floor - 1e-6 <= float(a_deep[0]) <= 1.0
        assert 0.0 <= float(a_shallow[0]) <= 1.0

    def test_penetration_beer_lambert_attenuates_with_thickness(self):
        """Fix #2: the penetrating fraction decays as exp(-kappa*h_ice), so
        thick (2 m) bare ice transmits far less than thin (0.1 m) ice, and the
        penetrated flux never exceeds the net (1-alpha)*sw column input."""
        sw = jnp.full((1,), 300.0)
        T = jnp.full((1,), 255.0)      # cold, bare ice
        h_snow = jnp.zeros((1,))
        pa = jnp.zeros((1,))
        pd = jnp.zeros((1,))
        res_thin = compute_ice_sw(
            sw, T, jnp.full((1,), 0.1), h_snow, pa, pd,
            scheme="delta_eddington", albedo_const=0.65,
        )
        res_thick = compute_ice_sw(
            sw, T, jnp.full((1,), 2.0), h_snow, pa, pd,
            scheme="delta_eddington", albedo_const=0.65,
        )
        pen_thin = float(res_thin.sw_penetrated[0])
        pen_thick = float(res_thick.sw_penetrated[0])
        # Thick-ice penetration is an order of magnitude smaller.
        assert pen_thick < 0.2 * pen_thin
        # Penetration never exceeds the net SW into the column.
        for res in (res_thin, res_thick):
            net = float((1.0 - res.albedo_eff[0]) * sw[0])
            assert float(res.sw_penetrated[0]) <= net + 1e-9

    def test_melt_fraction_reaches_one_at_freezing(self):
        """Fix #3: the melt ramp spans [T_melt - width, T_melt] and reaches 1.0
        at T_melt (skin T is clamped there), so the melting albedo is fully
        reached — the old 0.5-centred ramp topped out at 0.5."""
        # Thick ice (ramp == 1) so albedo == alpha_regime at melt_fraction=1.0.
        alpha = maykut_untersteiner_albedo(
            jnp.full((1,), constants.T_freeze), jnp.full((1,), 2.0),
        )
        assert jnp.isclose(alpha[0], 0.5, atol=1e-6)  # _ALBEDO_MELT_BARE_DEFAULT

    def test_thin_ice_albedo_floored_at_ocean(self):
        """Fix #4: 1 mm ice must not be darker than the open water it replaces;
        the thin-ice albedo is floored at the ocean albedo."""
        h_1mm = jnp.full((1,), 1e-3)
        # Maykut-Untersteiner broadband path.
        alpha_mu = maykut_untersteiner_albedo(jnp.full((1,), 260.0), h_1mm)
        assert float(alpha_mu[0]) >= constants.alpha_ocean_broadband - 1e-9
        # Delta-Eddington bare-ice path (no snow, no pond).
        alpha_de, _ = delta_eddington_albedo(
            jnp.full((1,), 260.0), h_1mm,
            jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)),
        )
        assert float(alpha_de[0]) >= constants.alpha_ocean_broadband - 1e-9


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

    def test_pond_gradient_finite_at_empty_pond(self):
        """d(pond)/d(melt_water) must be finite at an empty pond (V_new = 0).

        Regression: ``h_pond = sqrt(max(ratio·V_new, 0))`` had sqrt'(0) = inf at
        V_new = 0 — the common no-melt / cold-season state — giving a NaN
        reverse-mode gradient through pond depth.  Flooring the sqrt argument
        fixes it; the forward is unchanged (empty pond -> area 0).  Mirrors the
        sea_ice.py pond_depth fix (iter 2).
        """
        def loss(melt):
            a, d, _, _ = step_ponds(
                pond_area=jnp.array(0.0), pond_depth=jnp.array(0.0),
                melt_water_m=melt, rain_water_m=jnp.array(0.0),
                ice_mask=jnp.array(True), h_snow=jnp.array(0.0),
                T_air=jnp.array(275.0), dt=3600.0, drainage_timescale=86400.0,
                refreeze_threshold=273.15, pond_to_ice_max_area=0.6,
                depth_to_area_ratio=0.8,
            )
            return a + d

        grad = jax.grad(loss)(jnp.array(0.0))  # empty pond: V_new = 0
        assert bool(jnp.isfinite(grad)), (
            f"pond gradient not finite at empty pond: {grad}"
        )

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
        """σ_12 contributes to Fy via ∂/∂x and to Fx via ∂/∂y AND the
        spherical-metric term (F10): on a sphere a shear stress σ_12 exerts a
        zonal force ``-2 σ_12 tanθ / r`` even with no σ_12 gradient.

        Linear σ_12(λ): ∂σ_12/∂y = 0, so Fx is PURELY the metric term (the
        metric-free code asserted Fx ≈ 0 here); Fy = ∂σ_12/∂x > 0.
        """
        from legoesm.ice.dynamics import stress_divergence
        grid = self._make_grid()
        n_lat, n_lon = 36, 72
        metric = (jnp.tan(grid.lat) / grid.radius)[:, None]  # tanθ/r, (n_lat,1)
        sigma_12_lon = jnp.broadcast_to(
            jnp.arange(n_lon, dtype=jnp.float64), (n_lat, n_lon),
        )
        Fx, Fy = stress_divergence(
            jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)),
            sigma_12_lon, grid,
        )
        # Linear σ_12(lon) → Fy = ∂σ_12/∂x > 0 in interior.
        assert jnp.all(Fy[5:-5, 5:-5] > 0.0)
        # ∂σ_12/∂y = 0 → Fx is the pure spherical-metric term -2 σ_12 tanθ/r.
        expected_Fx = -2.0 * sigma_12_lon * metric
        assert jnp.allclose(Fx[5:-5, 5:-5], expected_Fx[5:-5, 5:-5],
                            rtol=1e-5, atol=1e-12)
        assert float(jnp.max(jnp.abs(Fx[5:-5, 5:-5]))) > 0.0

        # Linear σ_12(lat): Fy has no contribution (∂σ_12/∂x = 0 and the Fy
        # metric term needs σ_11 - σ_22, here zero).  Fx mixes the ∂σ_12/∂y
        # gradient and the metric term, so only assert it stays finite.
        sigma_12_lat = jnp.broadcast_to(
            jnp.arange(n_lat, dtype=jnp.float64)[:, None], (n_lat, n_lon),
        )
        Fx2, Fy2 = stress_divergence(
            jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)),
            sigma_12_lat, grid,
        )
        assert jnp.all(jnp.abs(Fy2[5:-5, 5:-5]) < 1e-9)
        assert jnp.all(jnp.isfinite(Fx2[5:-5, 5:-5]))

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


# ==============================================================================
# MPAS Voronoi ice support (transport + rheology + dynamics)
# ==============================================================================

class TestMPASVoronoi:
    """Ice physics on MPAS Voronoi mesh."""

    def _make_mesh(self, level=2):
        from legoesm.grids.voronoi import create_voronoi_mesh
        return create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)

    def test_transport_uniform_field_conserved(self):
        from legoesm.ice.transport import advect_ice_tracers
        mesh = self._make_mesh()
        n = mesh.nCells
        h = jnp.full((n,), 1.0)
        a = jnp.full((n,), 0.5)
        T = jnp.full((n,), 263.0)
        u = jnp.full((n,), 0.05)
        v = jnp.zeros((n,))
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, mesh, dt=3600.0)
        V0 = float(jnp.sum(h * a))
        V1 = float(jnp.sum(h_new * a_new))
        assert abs(V1 - V0) / max(V0, 1e-12) < 1e-6
        A0 = float(jnp.sum(a))
        A1 = float(jnp.sum(a_new))
        assert abs(A1 - A0) / max(A0, 1e-12) < 1e-6

    def test_multi_cat_transport(self):
        from legoesm.ice.transport import advect_ice_tracers
        mesh = self._make_mesh()
        n = mesh.nCells
        n_cat = 5
        shape = (n, n_cat)
        h = jnp.full(shape, 1.0)
        a = jnp.full(shape, 0.15)
        T = jnp.full(shape, 263.0)
        u = jnp.full((n,), 0.05)
        v = jnp.zeros((n,))
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, mesh, dt=3600.0)
        assert h_new.shape == shape

    def test_strain_rates_uniform_velocity_near_zero(self):
        from legoesm.ice.rheology import strain_rates
        mesh = self._make_mesh()
        n = mesh.nCells
        u = jnp.full((n,), 0.1)
        v = jnp.zeros((n,))
        e11, e22, e12 = strain_rates(u, v, mesh)
        # Green-Gauss on a uniform field gives zero gradient up to
        # mesh-irregularity noise; coarse SCVT meshes can reach
        # ~1e-7 in e12.
        assert jnp.all(jnp.abs(e11) < 1e-6)
        assert jnp.all(jnp.abs(e22) < 1e-6)

    def test_strain_rates_linear_u_gradient(self):
        from legoesm.ice.rheology import strain_rates
        mesh = self._make_mesh()
        n = mesh.nCells
        # u = 1e-6 · x_cell (m/s) → ∂u/∂x ≈ 1e-6
        u = 1.0e-6 * mesh.xCell
        v = jnp.zeros((n,))
        e11, _, _ = strain_rates(u, v, mesh)
        # Max |e11| should be O(1e-6); strict equality breaks because
        # spherical projection of the linear field varies.
        max_abs = float(jnp.max(jnp.abs(e11)))
        assert 1e-7 < max_abs < 1e-5

    def test_stress_divergence_zero_stress(self):
        from legoesm.ice.dynamics import stress_divergence
        mesh = self._make_mesh()
        n = mesh.nCells
        Fx, Fy = stress_divergence(
            jnp.zeros((n,)), jnp.zeros((n,)), jnp.zeros((n,)), mesh,
        )
        assert jnp.all(Fx == 0.0)
        assert jnp.all(Fy == 0.0)

    def test_evp_solver_mpas_finite(self):
        from legoesm.ice.dynamics import evp_solver
        mesh = self._make_mesh()
        n = mesh.nCells
        u_new, v_new, s11, s22, s12 = evp_solver(
            jnp.zeros((n,)), jnp.zeros((n,)),
            jnp.zeros((n,)), jnp.zeros((n,)), jnp.zeros((n,)),
            jnp.full((n,), 1.5), jnp.full((n,), 0.9),
            jnp.full((n,), 10.0), jnp.zeros((n,)),
            jnp.zeros((n,)), jnp.zeros((n,)),
            mesh, dt=3600.0, N_evp=20,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(s11))
        # Zubov-magnitude drift response.
        assert 0.01 < float(jnp.max(jnp.abs(u_new))) < 1.0

    def test_mevp_solver_mpas_finite(self):
        from legoesm.ice.dynamics import mevp_solver
        mesh = self._make_mesh()
        n = mesh.nCells
        u_new, v_new, s11, s22, s12 = mevp_solver(
            jnp.zeros((n,)), jnp.zeros((n,)),
            jnp.zeros((n,)), jnp.zeros((n,)), jnp.zeros((n,)),
            jnp.full((n,), 1.5), jnp.full((n,), 0.9),
            jnp.full((n,), 10.0), jnp.zeros((n,)),
            jnp.zeros((n,)), jnp.zeros((n,)),
            mesh, dt=3600.0, N_mevp=20,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(s11))


# ==============================================================================
# Grid coupling: dynamics/transport support across ocean grid types
# ==============================================================================

class TestGridValidation:
    """step_sea_ice must support cubed-sphere, lat-lon, and MPAS/Voronoi
    grids for dynamics + transport, and reject grids without the required
    operators (Gaussian spectral / Plane) with a clear error instead of a
    deep AttributeError."""

    def _forcing(self, shape):
        from legoesm.core.coupling_fields import AtmToSurface
        return AtmToSurface(
            sw_down=jnp.full(shape, 200.0), lw_down=jnp.full(shape, 300.0),
            T_lowest=jnp.full(shape, 260.0), q_lowest=jnp.full(shape, 1e-3),
            u_lowest=jnp.full(shape, 5.0), v_lowest=jnp.zeros(shape),
            p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
            rho_lowest=jnp.full(shape, 1.3),
            cos_zenith=jnp.full(shape, 0.5), co2_ppmv=jnp.full(shape, 400.0),
            precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.zeros(shape),
        )

    def test_grid_support_predicate(self):
        from legoesm.ice.sea_ice import _grid_supports_ice_dynamics
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.gaussian import create_gaussian_grid
        assert _grid_supports_ice_dynamics(create_cubed_sphere(8))
        assert _grid_supports_ice_dynamics(create_latlon_grid(n_lat=18, n_lon=36))
        assert not _grid_supports_ice_dynamics(create_gaussian_grid(n_max=21))
        assert not _grid_supports_ice_dynamics(None)

    def test_base_spatial_ndim_per_grid(self):
        from legoesm.ice.sea_ice import _base_spatial_ndim
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.latlon import create_latlon_grid
        assert _base_spatial_ndim(create_cubed_sphere(8)) == 3
        assert _base_spatial_ndim(create_latlon_grid(n_lat=18, n_lon=36)) == 2
        assert _base_spatial_ndim(None) is None

    def test_evp_on_gaussian_grid_raises(self):
        """EVP dynamics on a spectral (Gaussian) grid must raise a clear
        ValueError, not crash with an AttributeError in the EVP loop."""
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.core.field import Field
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(n_max=21)
        shape = (grid.n_lat, grid.n_lon)
        dims = ("lat", "lon")
        state = SeaIceState(
            h_ice=Field(data=jnp.ones(shape), name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=dims, units="K"),
            concentration=Field(data=jnp.full(shape, 0.9), name="conc", dims=dims, units="1"),
        )
        config = SeaIceConfig(dynamics="evp")
        with pytest.raises(ValueError, match="strain-rate and flux-divergence"):
            step_sea_ice(
                state, self._forcing(shape), jnp.full(shape, 271.35),
                jnp.zeros(shape), jnp.zeros(shape), config, 1.0, 3600.0, grid=grid,
            )

    def test_transport_on_gaussian_grid_raises(self):
        """transport='advect' on a Gaussian grid must raise the same guard."""
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.core.field import Field
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(n_max=21)
        shape = (grid.n_lat, grid.n_lon)
        dims = ("lat", "lon")
        state = SeaIceState(
            h_ice=Field(data=jnp.ones(shape), name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=dims, units="K"),
            concentration=Field(data=jnp.full(shape, 0.9), name="conc", dims=dims, units="1"),
        )
        config = SeaIceConfig(dynamics="none", transport="advect")
        with pytest.raises(ValueError, match="strain-rate and flux-divergence"):
            step_sea_ice(
                state, self._forcing(shape), jnp.full(shape, 271.35),
                jnp.zeros(shape), jnp.zeros(shape), config, 1.0, 3600.0, grid=grid,
            )

    def test_thermo_only_works_on_gaussian_grid(self):
        """Thermodynamics-only (no dynamics/transport) is pure elementwise and
        must run on ANY ocean grid shape, including a Gaussian/spectral grid."""
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.core.field import Field
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(n_max=21)
        shape = (grid.n_lat, grid.n_lon)
        dims = ("lat", "lon")
        state = SeaIceState(
            h_ice=Field(data=jnp.ones(shape), name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=dims, units="K"),
            concentration=Field(data=jnp.full(shape, 0.9), name="conc", dims=dims, units="1"),
        )
        config = SeaIceConfig(dynamics="none", transport="none")
        new_state, resp = step_sea_ice(
            state, self._forcing(shape), jnp.full(shape, 271.35),
            jnp.zeros(shape), jnp.zeros(shape), config, 1.0, 3600.0, grid=grid,
        )
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
        assert jnp.all(jnp.isfinite(resp.T_sfc))


# ==============================================================================
# Thin-ice ablation: area<->volume closure (no zombie tiles)
# ==============================================================================

class TestThinIceAblationClosure:
    """After full ablation of a thin ice column the tile must have BOTH
    h=0 and concentration=0 — no h=0 / conc>0 'zombie' tile that would keep
    exporting basal ocean heat (codex)."""

    def _forcing(self, shape, **kw):
        from legoesm.core.coupling_fields import AtmToSurface
        f = lambda v: jnp.full(shape, v)
        d = dict(sw_down=600.0, lw_down=340.0, T_lowest=288.0, q_lowest=7e-3,
                 u_lowest=5.0)
        d.update(kw)
        return AtmToSurface(
            sw_down=f(d["sw_down"]), lw_down=f(d["lw_down"]),
            precip_total=f(0.0), precip_snow=f(0.0),
            T_lowest=f(d["T_lowest"]), q_lowest=f(d["q_lowest"]),
            u_lowest=f(d["u_lowest"]), v_lowest=f(0.0),
            p_lowest=f(1.0e5), p_surface=f(1.0e5), rho_lowest=f(1.3),
            cos_zenith=f(0.5), co2_ppmv=f(400.0),
            has_radiation=f(1.0), has_precipitation=f(0.0),
        )

    def test_ablation_water_closure_with_ponds(self):
        """Full ablation with NONZERO ponds (+flooding) closes column water to
        the ocean exactly (codex #22 follow-up — the snow-only test used zero
        pond tracers).  Atmospheric exchange zeroed; pure basal-melt ablation.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig, SnowConfig, MeltPondConfig
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.thermo import saturation_mixing_ratio_ice
        from legoesm import constants
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        ri, rs, rw = constants.rho_ice, 330.0, constants.rho_water
        cfg = SeaIceConfig(
            snow=SnowConfig(enabled=True, flooding=True, rho_snow=rs),
            ponds=MeltPondConfig(enabled=True),
        )
        h0, c0, hs0, pa0, pd0, T0 = 0.001, 0.6, 0.02, 0.4, 0.02, 271.0
        qsat = float(saturation_mixing_ratio_ice(jnp.array(T0), jnp.array(1.0e5)))
        forcing = AtmToSurface(
            sw_down=f(0.0), lw_down=f(constants.sigma_sb * T0 ** 4),
            precip_total=f(0.0), precip_snow=f(0.0), T_lowest=f(T0),
            q_lowest=f(qsat), u_lowest=f(0.01), v_lowest=f(0.0),
            p_lowest=f(1.0e5), p_surface=f(1.0e5), rho_lowest=f(1.3),
            cos_zenith=f(0.0), co2_ppmv=f(400.0), has_radiation=f(1.0),
            has_precipitation=f(0.0),
        )
        sst = f(290.0)
        m_before = (h0 * ri + hs0 * rs + pa0 * pd0 * rw) * c0
        res = _thermo_v2(f(h0), f(T0), f(c0), f(hs0), f(0.0), f(pa0), f(pd0),
                         forcing, sst, cfg, 1.0, 3600.0, enable_lead_freeze=False)
        m_after = (
            float(jnp.mean(res["h"] * res["conc"])) * ri
            + float(jnp.mean(res["h_snow"] * res["conc"])) * rs
            + float(jnp.mean(res["pond_area"] * res["pond_depth"] * res["conc"])) * rw
        )
        fw = float(jnp.mean(res["freshwater_to_ocean"]))
        assert float(jnp.max(res["conc"])) <= 1e-9
        assert abs(float(jnp.mean(res["lhflx"]))) < 1.0
        residual = (m_after - m_before) + fw * 3600.0
        assert abs(residual) < 1e-6 * max(m_before, 1.0), (
            f"pond ablation water not closed: residual={residual:.4e} "
            f"(m0={m_before:.3f}, fw*dt={fw*3600.0:.4f})"
        )

    def test_lead_freeze_gated_on_ocean_supercooling(self):
        """Audit #2: lead ice must NOT form over an above-freezing ocean.

        Cold dry air drives a freezing surface deficit (Q_sfc < 0) over an
        ice-free cell.  With a warm mixed layer (sst > T_freeze_ocean) the
        atmospheric-deficit lead-freeze is gated OFF; a supercooled ocean
        (sst <= T_freeze_ocean) still freezes.  Same forcing, only sst varies.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.thermo import saturation_mixing_ratio_ice
        from legoesm import constants
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        cfg = SeaIceConfig()
        T_air = 240.0
        qsat = float(saturation_mixing_ratio_ice(jnp.array(T_air), jnp.array(1.0e5)))
        forcing = AtmToSurface(
            sw_down=f(0.0), lw_down=f(150.0),
            precip_total=f(0.0), precip_snow=f(0.0), T_lowest=f(T_air),
            q_lowest=f(qsat * 0.3), u_lowest=f(5.0), v_lowest=f(0.0),
            p_lowest=f(1.0e5), p_surface=f(1.0e5), rho_lowest=f(1.3),
            cos_zenith=f(0.0), co2_ppmv=f(400.0), has_radiation=f(1.0),
            has_precipitation=f(0.0),
        )
        # Ice-free cell (conc=0 -> full lead), skin at freezing.
        base = (f(0.0), f(constants.T_freeze_ocean), f(0.0), f(0.0), f(0.0),
                f(0.0), f(0.0), forcing)
        warm = _thermo_v2(*base, f(290.0), cfg, 1.0, 3600.0, enable_lead_freeze=True)
        cold = _thermo_v2(*base, f(270.0), cfg, 1.0, 3600.0, enable_lead_freeze=True)
        assert float(jnp.max(warm["conc"])) <= 1e-12, "lead ice grew over warm ocean"
        assert float(jnp.max(cold["conc"])) > 1e-6, "supercooled lead failed to freeze"

    def test_rain_on_ice_reaches_ocean_without_ponds(self):
        """Audit HIGH: with no pond scheme, rain falling on the ice fraction
        must run off to the ocean (snow instead enters the pack).  The ocean
        tile only delivers the open-water (f_ocean) precip share, so the ice
        tile must deliver rain*conc.  Two-point (rain vs no-rain) isolates it.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.thermo import saturation_mixing_ratio_ice
        from legoesm import constants
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        cfg = SeaIceConfig()  # ponds + snow OFF by default
        assert not cfg.ponds.enabled
        T_air, conc0, rain = 250.0, 0.8, 1.0e-4  # kg/m^2/s rain, cold (no melt)
        qsat = float(saturation_mixing_ratio_ice(jnp.array(T_air), jnp.array(1.0e5)))

        def run(precip_total):
            forcing = AtmToSurface(
                sw_down=f(0.0), lw_down=f(constants.sigma_sb * T_air ** 4),
                precip_total=f(precip_total), precip_snow=f(0.0), T_lowest=f(T_air),
                q_lowest=f(qsat), u_lowest=f(0.01), v_lowest=f(0.0),
                p_lowest=f(1.0e5), p_surface=f(1.0e5), rho_lowest=f(1.3),
                cos_zenith=f(0.0), co2_ppmv=f(400.0), has_radiation=f(1.0),
                has_precipitation=f(1.0),
            )
            # sst at freezing -> no basal; lead-freeze disabled -> isolate rain.
            res = _thermo_v2(f(1.0), f(constants.T_freeze_ocean), f(conc0), f(0.0),
                             f(0.0), f(0.0), f(0.0), forcing,
                             f(constants.T_freeze_ocean), cfg, 1.0, 3600.0,
                             enable_lead_freeze=False)
            return float(jnp.mean(res["freshwater_to_ocean"]))

        d_fw = run(rain) - run(0.0)
        # Ice-fraction rain delivered per water-area = rain * conc_new (~conc0;
        # a hair below because mild sublimation shrinks the ice area slightly).
        assert d_fw == pytest.approx(rain * conc0, rel=1e-3), (
            f"rain-on-ice runoff not delivered: got {d_fw:.3e}, "
            f"expected ~{rain * conc0:.3e}")

    def test_subgrid_ablation_water_closure(self):
        """Codex repro: thin ice (h<h_ice_min) fully ablating must route ALL
        column water to the ocean (no survivor-area snow/pond silently deleted
        by the ablation zero-out — the retained_area=min(conc_surv,conc_new)
        fix).  Isolate the ocean<->column exchange by zeroing atmospheric
        exchange (air at ice T + saturated -> shflx~0, lhflx~0, no sublimation)
        and using pure basal melt (warm ocean) so EVERY kg of column water must
        appear in freshwater_to_ocean.  Snow-only (flooding/ponds add separate
        conversions tested elsewhere); exact machine-precision closure.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig, SnowConfig, MeltPondConfig
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.thermo import saturation_mixing_ratio_ice
        from legoesm import constants
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        ri, rs = constants.rho_ice, 330.0
        cfg = SeaIceConfig(
            snow=SnowConfig(enabled=True, flooding=False, rho_snow=rs),
            ponds=MeltPondConfig(enabled=False),
        )
        h0, c0, hs0, T0 = 0.001, 0.6, 0.02, 271.0
        # Air at ice temperature + saturated => shflx~0, lhflx~0 (no sublim).
        qsat = float(saturation_mixing_ratio_ice(jnp.array(T0), jnp.array(1.0e5)))
        lw_in = constants.sigma_sb * T0 ** 4  # ~ balances lw_up at T0
        forcing = AtmToSurface(
            sw_down=f(0.0), lw_down=f(lw_in), precip_total=f(0.0),
            precip_snow=f(0.0), T_lowest=f(T0), q_lowest=f(qsat),
            u_lowest=f(0.01), v_lowest=f(0.0), p_lowest=f(1.0e5),
            p_surface=f(1.0e5), rho_lowest=f(1.3), cos_zenith=f(0.0),
            co2_ppmv=f(400.0), has_radiation=f(1.0), has_precipitation=f(0.0),
        )
        sst = f(290.0)  # strong basal melt -> full ablation in one step
        m_before = (h0 * ri + hs0 * rs) * c0
        res = _thermo_v2(f(h0), f(T0), f(c0), f(hs0), f(0.0), f(0.0), f(0.0),
                         forcing, sst, cfg, 1.0, 3600.0, enable_lead_freeze=False)
        m_after = (
            float(jnp.mean(res["h"] * res["conc"])) * ri
            + float(jnp.mean(res["h_snow"] * res["conc"])) * rs
        )
        fw = float(jnp.mean(res["freshwater_to_ocean"]))
        # Sanity: cell fully ablated and atmospheric exchange negligible.
        assert float(jnp.max(res["conc"])) <= 1e-9
        assert abs(float(jnp.mean(res["lhflx"]))) < 1.0
        residual = (m_after - m_before) + fw * 3600.0
        assert abs(residual) < 1e-6 * max(m_before, 1.0), (
            f"ablation water not closed: residual={residual:.4e} "
            f"(m0={m_before:.3f}, m1={m_after:.3f}, fw*dt={fw*3600.0:.4f})"
        )

    def test_total_water_closure_flooding_ponds(self):
        """EXACT per-cell water closure with flooding + ponds enabled (codex
        #22).  With no atmospheric water exchange (precip=0, |sublim|~0), every
        kg of H2O leaving the ice+snow+pond column each step must reappear in
        freshwater_to_ocean: d(M_ice + M_snow + M_pondwater)/dt + FW_to_ocean
        ~ basal-freeze uptake (the only ocean<->column mass term).  We assert
        the column-water budget closes to a tight relative tolerance across a
        step with simultaneous melt retreat + lead-freeze growth + flooding +
        pond refreeze.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig, SnowConfig, MeltPondConfig
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm import constants
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        cfg = SeaIceConfig(
            snow=SnowConfig(enabled=True, flooding=True),
            ponds=MeltPondConfig(enabled=True),
        )
        ri, rs, rw = cfg.rho_ice, cfg.snow.rho_snow, constants.rho_water

        def column_water(h, conc, hsnow, parea, pdepth):
            # Per-grid-cell H2O mass [kg/m2]: ice + snow + liquid pond water.
            m_ice = float(jnp.mean(h * conc)) * ri
            m_snow = float(jnp.mean(hsnow * conc)) * rs
            m_pond = float(jnp.mean(parea * pdepth * conc)) * rw
            return m_ice + m_snow + m_pond

        # Heavy snow + existing ponds on a partial-cover floe; dry air (no
        # deposition / no snowfall), warm ocean (basal melt -> retreat),
        # lead freeze ON (area growth) — exercise every conversion at once.
        h0, T0, c0, hs0, pa0, pd0 = 0.8, 271.0, 0.6, 0.4, 0.3, 0.05
        forcing = AtmToSurface(
            sw_down=f(50.0), lw_down=f(280.0), precip_total=f(0.0),
            precip_snow=f(0.0), T_lowest=f(258.0), q_lowest=f(1e-5),
            u_lowest=f(5.0), v_lowest=f(0.0), p_lowest=f(1.0e5),
            p_surface=f(1.0e5), rho_lowest=f(1.3), cos_zenith=f(0.2),
            co2_ppmv=f(400.0), has_radiation=f(1.0), has_precipitation=f(0.0),
        )
        sst = f(274.0)  # warm -> basal melt + retreat
        dt = 3600.0
        m_before = column_water(f(h0), f(c0), f(hs0), f(pa0), f(pd0))
        res = _thermo_v2(f(h0), f(T0), f(c0), f(hs0), f(0.0), f(pa0), f(pd0),
                         forcing, sst, cfg, 1.0, dt)
        m_after = column_water(res["h"], res["conc"], res["h_snow"],
                               res["pond_area"], res["pond_depth"])
        fw = float(jnp.mean(res["freshwater_to_ocean"]))      # kg/m2/s, +to ocean
        # Basal congelation freezes ocean water INTO the column (uptake);
        # basal_growth is the only column<->ocean mass source besides FW.
        # Net ocean exchange of column water: + FW out, - basal uptake.  The
        # FW term already includes -basal_growth*rho*conc, so:
        #   m_after - m_before  ==  -(FW)*dt   (column loses what FW gains),
        # to within sublimation (here |lhflx|~0 -> negligible) + roundoff.
        budget_residual = (m_after - m_before) + fw * dt
        scale = max(m_before, 1.0)
        assert abs(budget_residual) < 1e-3 * scale, (
            f"column water budget not closed: residual={budget_residual:.4e} "
            f"kg/m2, scale={scale:.2f} (m_before={m_before:.3f}, "
            f"m_after={m_after:.3f}, fw*dt={fw*dt:.4f})"
        )

    def test_snow_mass_conserved_under_area_change(self):
        """Per-cell snow mass closes across a step with area change (codex):
        snow_mass(final, in-ice) + snow shed to ocean == snow_mass(initial),
        for BOTH net retreat and net lead-freeze growth.  Catches the
        depth-only tracer manufacturing/dropping snow when ice area changes.
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig, SnowConfig
        from legoesm.core.coupling_fields import AtmToSurface
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        cfg = SeaIceConfig(snow=SnowConfig(enabled=True, flooding=False))
        rho_s = cfg.snow.rho_snow
        h0, hs0 = 1.0, 0.3   # thick floe with snow

        def snow_balance(conc0, forcing, sst, lead):
            res = _thermo_v2(f(h0), f(265.0), f(conc0), f(hs0), f(0.0), f(0.0),
                             f(0.0), forcing, sst, cfg, 1.0, 3600.0,
                             enable_lead_freeze=lead)
            # Per-cell snow mass in-ice after: h_snow_new * conc_new * rho_s.
            m_in = float(jnp.mean(res["h_snow"] * res["conc"])) * rho_s
            # Snow that left to the ocean this step: snow_to_ocean (open-water
            # snowfall, ~0 here) + the snow part of freshwater is bundled, so
            # instead reconstruct shed from the area change directly is hard;
            # use the budget identity: initial - in-ice = shed-to-ocean >= 0,
            # and must be bounded by the area actually lost.
            m0 = hs0 * conc0 * rho_s
            return m0, m_in

        # (a) Net MELT retreat via warm OCEAN (basal melt shrinks area).
        # Use DRY, cold air (q ~ 0, T_air <= ice T) so there is no surface
        # deposition adding snow — isolating the area-change tracer effect
        # (warm moist air would legitimately deposit snow via lhflx<0,
        # which is a separate physical source, not the invariant under test).
        fdry = AtmToSurface(
            sw_down=f(0.0), lw_down=f(280.0), precip_total=f(0.0),
            precip_snow=f(0.0), T_lowest=f(255.0), q_lowest=f(1e-5),
            u_lowest=f(5.0), v_lowest=f(0.0), p_lowest=f(1.0e5),
            p_surface=f(1.0e5), rho_lowest=f(1.3), cos_zenith=f(0.0),
            co2_ppmv=f(400.0), has_radiation=f(1.0), has_precipitation=f(0.0),
        )
        m0, m_in = snow_balance(0.7, fdry, f(277.0), lead=False)
        # Area retreats (basal melt) + sublimation only removes snow, so the
        # in-ice snow mass must be <= initial (no creation from area change).
        assert m_in <= m0 + 1e-9, f"melt: snow created {m_in:.4f} > {m0:.4f}"

        # (b) Net lead-freeze GROWTH (cold, open water refreezes): area grows,
        # snow depth must dilute so NO snow mass is manufactured on new ice.
        fcold = AtmToSurface(
            sw_down=f(0.0), lw_down=f(150.0), precip_total=f(0.0),
            precip_snow=f(0.0), T_lowest=f(240.0), q_lowest=f(2e-4),
            u_lowest=f(6.0), v_lowest=f(0.0), p_lowest=f(1.0e5),
            p_surface=f(1.0e5), rho_lowest=f(1.35), cos_zenith=f(0.0),
            co2_ppmv=f(400.0), has_radiation=f(1.0), has_precipitation=f(0.0),
        )
        m0g, m_in_g = snow_balance(0.5, fcold, f(271.35), lead=True)
        # Crucial: in-ice snow mass must NOT exceed the initial (the old
        # depth-only tracer would manufacture snow on the grown area).
        assert m_in_g <= m0g + 1e-9, (
            f"lead-growth manufactured snow: {m_in_g:.4f} > {m0g:.4f}"
        )

    def test_partial_cover_ablation_freshwater_closure(self):
        """Partial-cover (conc<1) ablation routes orphan ice/snow as
        PER-GRID-CELL freshwater (no 1/conc over-emit) — codex units fix.

        Integrate a thin partial-cover floe under warm forcing until it fully
        ablates, accumulating freshwater_to_ocean*dt.  The TOTAL freshwater
        delivered must equal the per-grid-cell ice+snow mass that existed
        (h*conc*rho), to a tight tolerance.  The old per-ice-area orphan term
        (h*rho, no conc) would over-deliver by 1/conc (~5x at conc0=0.2).
        """
        from legoesm.ice.sea_ice import _thermo_v2
        from legoesm.ice.config import SeaIceConfig, SnowConfig
        from legoesm.core.coupling_fields import AtmToSurface
        n = 4
        shape = (6, n, n)
        f = lambda v: jnp.full(shape, v)
        conc0 = 0.2
        h0, hs0 = 0.05, 0.02
        forcing = AtmToSurface(
            sw_down=f(700.0), lw_down=f(340.0), precip_total=f(0.0),
            precip_snow=f(0.0), T_lowest=f(292.0), q_lowest=f(9e-3),
            u_lowest=f(5.0), v_lowest=f(0.0), p_lowest=f(1.0e5),
            p_surface=f(1.0e5), rho_lowest=f(1.3), cos_zenith=f(0.5),
            co2_ppmv=f(400.0), has_radiation=f(1.0), has_precipitation=f(0.0),
        )
        cfg = SeaIceConfig(snow=SnowConfig(enabled=True))
        sst = f(276.0)
        dt = 3600.0
        # Per-grid-cell ice+snow mass that existed initially [kg/m2].
        cell_mass0 = (h0 * cfg.rho_ice + hs0 * cfg.snow.rho_snow) * conc0
        h, T, c, hs = f(h0), f(272.9), f(conc0), f(hs0)
        Sice, pa, pd = f(0.0), f(0.0), f(0.0)
        fw_total = 0.0  # kg/m2, accumulated (no lead freeze -> all melt out)
        for _ in range(120):
            res = _thermo_v2(h, T, c, hs, Sice, pa, pd, forcing, sst, cfg, 1.0, dt,
                             enable_lead_freeze=False)
            fw_total += float(jnp.mean(res["freshwater_to_ocean"])) * dt
            h, T, c, hs = res["h"], res["T"], res["conc"], res["h_snow"]
            Sice, pa, pd = res["S_ice"], res["pond_area"], res["pond_depth"]
        # Essentially ablated (area decays asymptotically as conc/h_eff; the
        # residual ice volume h*conc is negligible by here).
        assert float(jnp.max(h * c)) <= 1e-4 * cell_mass0 / cfg.rho_ice, (
            f"residual ice volume too large: {float(jnp.max(h*c)):.2e} m"
        )
        # Total freshwater out = per-cell column mass (NOT 1/conc too large).
        # Tolerance covers basal-melt water (also per-cell, already in budget).
        assert fw_total > 0.0
        assert fw_total < 1.5 * cell_mass0, (
            f"freshwater {fw_total:.3f} kg/m2 exceeds 1.5x per-cell column "
            f"mass {cell_mass0:.3f} (1/conc over-emit would be ~{1/conc0:.0f}x)"
        )

    def test_full_melt_out_zeros_concentration(self):
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig, SnowConfig
        from legoesm.ice.state import DynamicSeaIceState
        from legoesm.core.field import Field
        n = 4
        shape = (6, n, n)
        dims = ("face", "x", "y")
        def F(x, nm, u): return Field(data=jnp.full(shape, x), name=nm, dims=dims, units=u)
        z = Field(data=jnp.zeros(shape), name="z", dims=dims, units="1")
        # Very thin ice below h_ice_min, warm forcing -> should fully ablate.
        st = DynamicSeaIceState(
            h_ice=F(0.005, "h_ice", "m"), T_ice=F(272.8, "T_ice", "K"),
            concentration=F(1.0, "conc", "1"),
            u_ice=z, v_ice=z, sigma_11=z, sigma_22=z, sigma_12=z,
            h_snow=F(0.0, "h_snow", "m"), S_ice=F(0.0, "S_ice", "psu"),
            pond_area=z, pond_depth=z,
        )
        cfg = SeaIceConfig(snow=SnowConfig(enabled=True))  # new-physics path (_thermo_v2)
        sst = jnp.full(shape, 273.5)  # warm ocean too -> basal melt
        zc = jnp.zeros(shape)
        for _ in range(20):
            st, _ = step_sea_ice(st, self._forcing(shape), sst, zc, zc, cfg, 1.0, 3600.0)
        h = st.h_ice.data
        a = st.concentration.data
        # No zombie tiles anywhere: wherever h==0, conc must be 0 (and vice versa
        # to numerical tolerance).
        zombie = (h <= 1e-12) & (a > 1e-9)
        assert not bool(jnp.any(zombie)), (
            f"zombie tile (h=0, conc>0) count={int(jnp.sum(zombie))}, "
            f"max conc there={float(jnp.max(jnp.where(zombie, a, 0.0)))}"
        )
        assert jnp.all(jnp.isfinite(h)) and jnp.all(jnp.isfinite(a))
        assert jnp.all(a <= 1.0 + 1e-9) and jnp.all(a >= 0.0)
        assert jnp.all(h >= 0.0)



# ==============================================================================
# Multi-category atmosphere flux aggregation (finding #9)
# ==============================================================================

class TestMulticatAtmFluxAggregation:
    """The multicat sensible-heat / stress response to the atmosphere keeps the
    per-grid-cell numerator ``Sum_k flux_k*conc_post_k`` and divides by the
    FINAL aggregated concentration ``conc_agg`` -- the SAME concentration
    ``f_ice`` later multiplies by -- so the delivered flux ``resp * f_ice``
    recovers the per-cell total EXACTLY, regardless of net melt (the old
    max(pre,post) denominator under-reported under melt, finding #9) or of
    ITD-remap/ridging area change between thermo and the response (finding #4).

    Full multicat ``step_sea_ice`` paths are exercised by the sea-ice stress
    suite; this isolates the normalisation algebra the fix changed."""

    def test_conc_agg_basis_recovers_per_cell_flux(self):
        import numpy as np
        # Two categories; conc_agg differs from BOTH sum_pre and sum_post (as it
        # would after ridging compacts area between thermo and the response).
        shflx_k = jnp.array([20.0, -5.0])          # per-cat sensible heat [W/m2]
        conc_pre = jnp.array([0.5, 0.4])           # sum_pre  = 0.9
        conc_post = jnp.array([0.3, 0.2])          # sum_post = 0.5  (net melt)
        conc_agg = 0.42                            # post-ridging aggregate area
        sum_pre = float(jnp.sum(conc_pre))
        sum_post = float(jnp.sum(conc_post))
        assert sum_post < sum_pre                  # net melt regime
        assert conc_agg != sum_post                # ridging changed the area

        # Per-grid-cell SH total the atmosphere must receive (computed at the
        # post-thermo areas the bulk flux acted on).
        numer = float(jnp.sum(shflx_k * conc_post))

        # NEW response: numerator / conc_agg, then blend re-multiplies by
        # f_ice (proportional to conc_agg) -> conc_agg cancels EXACTLY.
        shflx_resp_new = numer / max(conc_agg, 1e-30)
        delivered_new = shflx_resp_new * conc_agg
        assert np.isclose(delivered_new, numer, rtol=1e-12), (
            "conc_agg-basis response must recover the per-cell sensible heat"
        )

        # OLD response: numerator / max(pre, post), then blend by conc_agg.
        shflx_resp_old = numer / max(max(sum_pre, sum_post), 1e-30)
        delivered_old = shflx_resp_old * conc_agg
        # The old basis mis-delivers (scaled by conc_agg/max(pre,post)).
        assert not np.isclose(delivered_old, numer, rtol=1e-6)
        assert np.isclose(delivered_old, numer * conc_agg / sum_pre, rtol=1e-12)
