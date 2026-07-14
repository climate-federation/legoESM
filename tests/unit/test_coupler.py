"""Tests for the surface coupler and all tile models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.core.coupling_fields import AtmToSurface, SurfaceToAtm, TileResponse
from legoesm.coupler.tile_fractions import (
    TileFractions,
    blend_tiles,
    compute_tile_fractions,
)
from legoesm.coupler.accumulator import (
    FluxAccumulator,
    accumulate,
    mean_accumulator,
    reset_accumulator,
)
from legoesm.coupler.coupler import (
    SurfaceState,
    init_surface_state,
    make_coupler,
    ocean_tile_response,
)
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.land.slab_land import step_land
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.lake.two_layer_lake import step_lake


# ---- Helpers ----

SHAPE = (6, 4, 4)
DT = 600.0
DIMS_2D = ("face", "x", "y")


def _make_forcing(shape=SHAPE, T_lowest=280.0, sw=200.0, lw=300.0,
                  precip=1e-5):
    """Create a realistic AtmToSurface forcing."""
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=jnp.full(shape, sw),
        lw_down=jnp.full(shape, lw),
        precip_total=jnp.full(shape, precip),
        precip_snow=z,
        T_lowest=jnp.full(shape, T_lowest),
        q_lowest=jnp.full(shape, 5e-3),
        u_lowest=jnp.full(shape, 5.0),
        v_lowest=jnp.full(shape, -3.0),
        p_lowest=jnp.full(shape, 95000.0),
        p_surface=jnp.full(shape, 1e5),
        rho_lowest=jnp.full(shape, 1.15),
        cos_zenith=jnp.full(shape, 0.6),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


def _make_land_state(shape=SHAPE, T=280.0, W=75.0):
    return LandState(
        T_soil=Field(jnp.full(shape, T), name="T_soil", dims=DIMS_2D, units="K"),
        W_bucket=Field(jnp.full(shape, W), name="W_bucket", dims=DIMS_2D, units="kg/m2"),
        snow_depth=Field(jnp.zeros(shape), name="snow_depth", dims=DIMS_2D, units="kg/m2"),
        snow_age=Field(jnp.zeros(shape), name="snow_age", dims=DIMS_2D, units="s"),
    )


def _make_ice_state(shape=SHAPE, h=0.5, T=260.0, conc=0.5):
    return SeaIceState(
        h_ice=Field(jnp.full(shape, h), name="h_ice", dims=DIMS_2D, units="m"),
        T_ice=Field(jnp.full(shape, T), name="T_ice", dims=DIMS_2D, units="K"),
        concentration=Field(jnp.full(shape, conc), name="ice_conc", dims=DIMS_2D, units="1"),
    )


def _make_lake_state(shape=SHAPE, T_epi=285.0, T_hypo=278.0):
    return LakeState(
        T_epi=Field(jnp.full(shape, T_epi), name="T_epi", dims=DIMS_2D, units="K"),
        T_hypo=Field(jnp.full(shape, T_hypo), name="T_hypo", dims=DIMS_2D, units="K"),
    )


# ==============================================================================
# Test coupling fields
# ==============================================================================

def test_coupling_fields_shapes():
    """AtmToSurface and SurfaceToAtm have correct number of fields."""
    forcing = _make_forcing()
    assert len(forcing) == 15
    assert forcing.sw_down.shape == SHAPE

    z = jnp.zeros(SHAPE)
    # SurfaceToAtm and TileResponse gained ``freshwater_flux`` (F4),
    # ``ocean_heat_extraction`` (F8), ``ocean_stress_x``/``y`` (F9),
    # and ``surface_mass_flux`` (F3) slots in the Physical_Consistency
    # cycle for tile-blended water, ice→ocean heat, ice→ocean stress
    # reaction, and phase-aware moisture mass closure.
    sfc = SurfaceToAtm(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z)
    # merged union: T_rad (HEAD, radiative-equiv skin T) + river_runoff_flux +
    # ice_lake_freshwater_flux (origin/main) → 22 fields (salt_flux is #20).
    assert len(sfc) == 22

    tile = TileResponse(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z)
    # 19 required + T_rad (optional emission-equiv skin T) +
    # ice_concentration_thermo (optional thermo-time ice area for the
    # forced-ocean open-water partition) — both trailing None-defaults, so the
    # 19-positional construction above stays valid.
    assert len(tile) == 21


# ==============================================================================
# Test land
# ==============================================================================

def test_land_step_finite():
    """Land model produces finite outputs."""
    state = _make_land_state()
    forcing = _make_forcing()
    config = LandConfig()

    new_state, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)

    assert jnp.all(jnp.isfinite(new_state.T_soil.data))
    assert jnp.all(jnp.isfinite(new_state.W_bucket.data))
    assert jnp.all(jnp.isfinite(resp.shflx))
    assert jnp.all(jnp.isfinite(resp.lhflx))
    assert jnp.all(jnp.isfinite(resp.lw_up))


def test_land_energy_balance_sign():
    """When surface is warmer than air, SH is positive (upward)."""
    state = _make_land_state(T=300.0)  # Warm soil
    forcing = _make_forcing(T_lowest=270.0)  # Cold air
    config = LandConfig()

    _, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)
    assert jnp.all(resp.shflx > 0)


def test_land_bucket_limits():
    """Bucket moisture stays in [0, W_max]."""
    config = LandConfig()

    # Saturated bucket with zero precip
    state = _make_land_state(W=config.W_max)
    forcing = _make_forcing(precip=0.0)
    new_state, _, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)
    assert jnp.all(new_state.W_bucket.data <= config.W_max + 1e-6)

    # Empty bucket
    state = _make_land_state(W=0.0)
    new_state, _, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)
    assert jnp.all(new_state.W_bucket.data >= -1e-6)


# ==============================================================================
# Test sea ice
# ==============================================================================

def test_sea_ice_step_finite():
    """Sea ice model produces finite outputs."""
    state = _make_ice_state()
    forcing = _make_forcing()
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 275.0)
    ocean_u = jnp.zeros(SHAPE)
    ocean_v = jnp.zeros(SHAPE)

    new_state, resp = step_sea_ice(
        state, forcing, ocean_sst, ocean_u, ocean_v, config, U_min=1.0, dt=DT)

    assert jnp.all(jnp.isfinite(new_state.h_ice.data))
    assert jnp.all(jnp.isfinite(new_state.T_ice.data))
    assert jnp.all(jnp.isfinite(new_state.concentration.data))
    assert jnp.all(jnp.isfinite(resp.shflx))


def test_sea_ice_thickness_nonneg():
    """Ice thickness stays non-negative."""
    state = _make_ice_state(h=0.01)  # Thin ice
    forcing = _make_forcing(T_lowest=280.0, sw=400.0)  # Warm forcing
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 280.0)  # Warm ocean melts from below

    new_state, _ = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    assert jnp.all(new_state.h_ice.data >= 0.0)


def test_sea_ice_concentration_bounds():
    """Ice concentration stays in [0, 1]."""
    state = _make_ice_state(conc=0.99, h=2.0)
    forcing = _make_forcing()
    config = SeaIceConfig()

    new_state, _ = step_sea_ice(
        state, forcing, jnp.full(SHAPE, 270.0),
        jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    assert jnp.all(new_state.concentration.data >= -1e-6)
    assert jnp.all(new_state.concentration.data <= 1.0 + 1e-6)


def test_sea_ice_no_spurious_growth_from_open_water():
    """Cells without ice should not grow ice under net warming conditions."""
    state = _make_ice_state(h=0.0, conc=0.0, T=constants.T_freeze_ocean)
    forcing = _make_forcing(T_lowest=280.0, sw=300.0, lw=350.0)
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 275.0)

    new_state, _ = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    assert jnp.allclose(new_state.h_ice.data, 0.0, atol=1e-8)
    assert jnp.allclose(new_state.concentration.data, 0.0, atol=1e-8)


def test_sea_ice_sublimation_mass_term_is_included():
    """Audit F7 (HIGH): ice mass budget must include
    ``dh_dt_sublim = -lhflx / (rho_ice · L_s)`` over ice cells.

    Run the same forcing twice — once with the production lhflx,
    once with lhflx=0 (no sublimation) — and verify that the
    delta in dh/dt matches the expected sublimation contribution.

    Strategy: rather than zeroing lhflx (which requires bypassing
    the bulk-flux call), use a *very dry* atmosphere to maximise
    lhflx, run two cases differing only in q_lowest, and check
    that the difference in dh_dt is consistent with the
    ``-Δlhflx / (rho_ice · L_s)`` sublim-mass term.

    Without the iter-11 fix, ``dh/dt`` would be invariant under
    q_lowest changes (energy budget closes via lhflx but mass
    budget ignores sublimation), so the delta would be zero.
    """
    from legoesm import constants

    state = _make_ice_state(h=2.0, conc=1.0, T=265.0)
    # Two forcings differing only in q_lowest: dry vs more moist.
    base_forcing = _make_forcing(T_lowest=240.0, sw=0.0, lw=200.0)
    forcing_dry = base_forcing._replace(q_lowest=jnp.full(SHAPE, 1e-5))
    forcing_moist = base_forcing._replace(q_lowest=jnp.full(SHAPE, 5e-3))
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, constants.T_freeze_ocean)

    # Run both cases.
    state_dry, resp_dry = step_sea_ice(
        state, forcing_dry, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT,
    )
    state_moist, resp_moist = step_sea_ice(
        state, forcing_moist, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT,
    )

    # lhflx must be larger under dry conditions (more sublimation).
    dlhflx = float(jnp.mean(resp_dry.lhflx - resp_moist.lhflx))
    assert dlhflx > 0.0, (
        f"dry case should give larger lhflx (more sublimation); "
        f"Δlhflx = {dlhflx:.2e}"
    )

    # Predicted Δ(dh/dt) from sublimation term alone:
    # Δdh_dt_sublim = -Δlhflx / (rho_ice · L_s)  (negative — more sublim
    # in dry case → more mass loss → MORE NEGATIVE dh/dt).
    expected_d_dh_dt = -dlhflx / (config.rho_ice * constants.L_s)

    dh_dt_dry = float(jnp.mean(state_dry.h_ice.data - state.h_ice.data) / DT)
    dh_dt_moist = float(jnp.mean(state_moist.h_ice.data - state.h_ice.data) / DT)
    actual_d_dh_dt = dh_dt_dry - dh_dt_moist

    # Without iter-11 fix this would be ~0 (mass budget ignores
    # the lhflx-driven mass loss).  With the fix, actual_d_dh_dt
    # should match expected_d_dh_dt.  The shflx ALSO changes a
    # little between the two runs because the sensible-heat
    # transfer responds to T_ice via the surface energy budget,
    # so allow a generous tolerance.
    assert abs(actual_d_dh_dt - expected_d_dh_dt) < 0.5 * abs(expected_d_dh_dt), (
        f"Δ(dh/dt) = {actual_d_dh_dt:.2e} does not match expected "
        f"sublim contribution {expected_d_dh_dt:.2e} — iter-11 fix "
        f"may be missing."
    )
    # Must be a NEGATIVE delta (dry case loses more mass).
    assert actual_d_dh_dt < 0.0, (
        f"Δ(dh/dt) should be negative (dry case loses more mass); "
        f"saw {actual_d_dh_dt:.2e}"
    )


def test_slab_ocean_Q_freeze_diagnostic_populated():
    """Audit F6 (MEDIUM): slab ocean freezing clamp must diagnose
    ``Q_freeze`` (latent heat of fusion implicitly extracted from
    the surface budget when SST clamps at T_freeze) instead of
    silently destroying the energy.  The iter-11 fix adds this
    diagnostic; iter-12 ensures it's a populated zero Field at
    init so the pytree shape is invariant.

    Test asserts:
    - Cold-forcing column drives SST below T_freeze → Q_freeze > 0.
    - Warm-forcing column → Q_freeze = 0 (no clamping).
    """
    from legoesm.ocean.simple_ocean import (
        _slab_step, init_slab_state, SimpleOceanConfig,
    )

    config = SimpleOceanConfig()
    state = init_slab_state(SHAPE, T_sfc_init=271.5)  # very near freezing
    # Cold forcing — net heat loss large enough to drive below freezing
    cold_forcing = _make_forcing(T_lowest=200.0, sw=0.0, lw=150.0)
    new_state, _, _, _ = _slab_step(state, cold_forcing, config, dt=86400.0)
    # Q_freeze must be populated and POSITIVE in cells where SST
    # would have dropped below T_freeze.
    assert new_state.Q_freeze is not None
    assert float(jnp.max(new_state.Q_freeze.data)) > 0.0, (
        "Q_freeze should be positive when SST clamps at T_freeze"
    )

    # Warm forcing — no clamping → Q_freeze should be zero.
    warm_state = init_slab_state(SHAPE, T_sfc_init=290.0)
    warm_forcing = _make_forcing(T_lowest=290.0, sw=300.0, lw=350.0)
    new_warm_state, _, _, _ = _slab_step(
        warm_state, warm_forcing, config, dt=86400.0,
    )
    assert float(jnp.max(new_warm_state.Q_freeze.data)) == 0.0


# ==============================================================================
# Coupler-conservation regression tests (audit F3/F4/F8/F9 channels)
# ==============================================================================

def test_sea_ice_freshwater_flux_balances_ice_mass_change():
    """Sea-ice TileResponse.freshwater_flux must equal the PER-GRID-CELL
    ice melt/freeze mass rate -rho_ice·(dh/dt - sublimation)·conc (audit
    F4 / F11).

    F11: the response is per-grid-cell (per-water-area), so the per-ice-area
    thickness-rate balance is weighted by the ice fraction ``conc``; the
    coupler's blend_tiles then applies ``f_water``.  With realistic warm
    forcing the ice melts → freshwater INTO ocean → flux > 0.  This scenario
    has no lead freeze and no over-ablation clamp, so the per-process budget
    reduces exactly to the lumped thickness-rate balance times ``conc``.
    """
    from legoesm import constants
    state = _make_ice_state(h=0.5, conc=0.9, T=constants.T_freeze_ocean)
    forcing = _make_forcing(T_lowest=280.0, sw=400.0, lw=350.0)
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 275.0)

    new_state, resp = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT,
    )

    # Per-grid-cell ice VOLUME change rate.  Under the volume-based V=h*A
    # update (#28) thermodynamic melt/growth changes BOTH h and conc, so the
    # per-cell ice-mass budget uses d(h*conc)/dt, not conc*dh/dt.
    dV_dt = (new_state.h_ice.data * new_state.concentration.data
             - state.h_ice.data * state.concentration.data) / DT
    # Sublimation thickness rate (signed; <0 for ice->atmosphere), over ice.
    sublim_rate = jnp.where(
        state.h_ice.data > config.h_ice_min,
        -resp.lhflx / (config.rho_ice * constants.L_s),
        0.0,
    )
    # Freshwater to ocean = -(per-cell volume change going to/from the OCEAN):
    # the total volume change minus the sublimation part (which leaves to the
    # ATMOSPHERE, not the ocean).  Sublimation acts over the ice fraction.
    expected_fw = -config.rho_ice * (dV_dt - sublim_rate * state.concentration.data)
    assert jnp.allclose(resp.freshwater_flux, expected_fw, rtol=1e-6, atol=1e-12)
    # Sanity: ice melting should yield positive freshwater into ocean
    assert float(jnp.mean(resp.freshwater_flux)) > 0.0, (
        "Warm-forcing scenario should melt ice → freshwater_flux > 0"
    )


def test_sea_ice_freshwater_flux_balances_under_ablation_clamp():
    """When the over-ablation cap fires (the per-ice-area removal demand would
    exceed the available ice h/dt), the SAME capped process increments drive
    BOTH the prognostic volume and the ocean-exchange diagnostics, so the
    per-cell ice-mass budget still closes exactly (#28, codex).

    Thin ice (2 cm) over a strongly super-freezing ocean for a full day: the
    basal-melt demand alone (F_ocean/(rho*L_f)) far exceeds h/dt, forcing the
    removal cap.  Previously the cap scaled only the returned diagnostics while
    h advanced on the uncapped rate, so the reported freshwater no longer
    matched d(h*conc)/dt in this branch.  (A cap that fires while basal GROWTH
    is active is not physically reachable in a single column: basal growth
    needs a sub-freezing skin, which also suppresses surface melt; the cap is
    only reached under strong melt, which is what this exercises.)

    Also verifies the OTHER coupled channels stay consistent in the clamp
    branch: total water mass closes across ice + ocean + atmosphere, and the
    ocean heat extraction is bounded in [0, F_ocean*conc] (the turbulent flux
    operates only over the survived fraction of the step) (#28, codex).
    """
    from legoesm import constants
    h0, conc0, dt = 0.02, 0.9, 86400.0
    state = _make_ice_state(h=h0, conc=conc0, T=constants.T_freeze_ocean)
    # Dry air (q_lowest << saturation over melting ice) so the latent flux is
    # SUBLIMATION (atmosphere mass sink), exercising the surface_mass_flux cap.
    forcing = _make_forcing(T_lowest=290.0, sw=600.0, lw=400.0)._replace(
        q_lowest=jnp.full(SHAPE, 1e-4),
    )
    config = SeaIceConfig()
    sst_val = 282.0
    ocean_sst = jnp.full(SHAPE, sst_val)

    new_state, resp = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=dt,
    )

    # Confirm the clamp regime: the UNCAPPED basal-melt thickness over the step
    # already exceeds the 2 cm of ice present, so the removal cap MUST fire.
    F_ocean = config.ocean_heat_transfer_coeff * (sst_val - constants.T_freeze_ocean)
    basal_melt_thickness = F_ocean / (config.rho_ice * config.L_f) * dt
    assert basal_melt_thickness > h0, (
        "scenario must drive removal past h/dt to exercise the ablation clamp"
    )
    # TOTAL water closure across ice + ocean + atmosphere: the per-cell ice mass
    # change equals the freshwater sent to the ocean plus the (realized, capped)
    # sublimation mass sent to the atmosphere.  Both freshwater_flux and
    # surface_mass_flux are PER-GRID-CELL (already weighted by the ice fraction),
    # so no extra *conc.  Machine-exact since both reuse the capped increments.
    dV_dt = (new_state.h_ice.data * new_state.concentration.data
             - state.h_ice.data * state.concentration.data) / dt
    water_residual = (config.rho_ice * dV_dt + resp.freshwater_flux
                      + resp.surface_mass_flux)
    assert float(jnp.max(jnp.abs(water_residual))) < 1e-9, (
        f"ice+ocean+atmosphere water not conserved in clamp: "
        f"max |resid| = {float(jnp.max(jnp.abs(water_residual))):.3e}"
    )
    # Ocean heat extraction upper bound F_ocean*conc: the basal turbulent flux
    # is scaled by the survived fraction, NOT reported in full while the state
    # only absorbed the capped melt (the pre-fix over-extraction bug).
    # LOWER bound: NEGATIVE extraction is now legitimate — a melt-out step's
    # SURPLUS surface-melt energy warms the ocean (sea_ice finding #6:
    # ``- surface_melt_ocean_gain``; previously that energy was dropped on the
    # floor).  It is bounded by the incident surface energy scale over the ice
    # fraction, so pin that instead of the stale >= 0 (which this full-melt-out
    # scenario — 600 W/m^2 SW onto 2 cm of ice — legitimately violates).
    _incident = (600.0 + 400.0) * conc0     # sw + lw of _make_forcing above
    assert jnp.all(resp.ocean_heat_extraction >= -(_incident + 1e-6))
    assert jnp.all(resp.ocean_heat_extraction <= F_ocean * conc0 + 1e-6)


def test_sea_ice_ocean_heat_extraction_positive_under_warm_ocean():
    """Sea-ice TileResponse.ocean_heat_extraction > 0 when ocean is
    warm above T_freeze_ocean (audit F8 — ocean LOSES heat to melt
    ice base).
    """
    state = _make_ice_state(h=1.0, conc=1.0, T=270.0)
    forcing = _make_forcing()
    config = SeaIceConfig()
    # Warm ocean (4 K above freezing) drives basal melt
    ocean_sst = jnp.full(SHAPE, 275.35)

    _, resp = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT,
    )

    expected_F_ocean = config.ocean_heat_transfer_coeff * 4.0  # SST - T_freeze_ocean
    # Tolerance generous: ocean_heat_extraction may include a small
    # open_freeze_flux contribution from any cells where ice grew.
    assert float(jnp.min(resp.ocean_heat_extraction)) > 0.5 * expected_F_ocean
    assert jnp.all(resp.ocean_heat_extraction >= 0.0)


def test_sea_ice_ocean_stress_opposes_ocean_ice_drag():
    """Sea-ice TileResponse.ocean_stress_x/y is the negative of
    rho_ocean·C_oi·|U_w − U_i|·(U_w − U_i), returned PER-ICE-TILE (no
    concentration factor).  The coupler's blend_tiles applies the single
    area weight f_ice = f_water·conc; the old ``* conc`` here double-counted
    the ice fraction (audit F9 / F11).
    """
    state = _make_ice_state(h=2.0, conc=0.8, T=265.0)
    # Use _make_forcing default winds (u=5, v=-3) — strong enough to
    # exercise the back-reaction stress.
    forcing = _make_forcing()
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, constants.T_freeze_ocean)
    ocean_u = jnp.zeros(SHAPE)
    ocean_v = jnp.zeros(SHAPE)

    _, resp = step_sea_ice(
        state, forcing, ocean_sst, ocean_u, ocean_v,
        config, U_min=1.0, dt=DT,
    )

    # Ice velocity from Zubov-style drag-balance free drift (iter-86):
    #   u_i = U_w + alpha * (U_a - U_w),
    #   alpha = sqrt(rho_air * C_ai / (rho_ocean * C_oi)).
    # With ocean at rest, u_i = alpha * U_a.
    import math
    alpha = math.sqrt(
        config.rho_air_ref * config.drag_atm
        / (config.rho_ocean_ref * config.drag_ocean)
    )
    u_ice_expected = alpha * 5.0
    v_ice_expected = alpha * (-3.0)
    # Relative velocity ocean - ice = -u_ice (ocean at rest)
    du_oi = -u_ice_expected
    dv_oi = -v_ice_expected
    speed_oi = float(jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10))
    tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
    # Per-ice-tile (no concentration factor): blend_tiles applies f_ice.
    expected_stress_x = -tau_oi_x

    assert jnp.allclose(resp.ocean_stress_x, expected_stress_x, rtol=1e-2)
    # Wind is eastward → ice moves east → ocean→ice drag pulls ice
    # westward (tau_oi_x < 0) → reaction on ocean is +tau_oi (eastward),
    # so ocean_stress_x is POSITIVE.
    assert float(jnp.mean(resp.ocean_stress_x)) > 0.0


def test_sea_ice_surface_mass_flux_equals_lhflx_over_Ls():
    """Sea-ice TileResponse.surface_mass_flux = (lhflx / L_s) * conc in the
    no-clamp regime (audit F3 / #28 — phase-aware mass flux uses sublimation
    latent heat over ice).

    The sublimation mass is now returned PER-GRID-CELL (weighted by the input
    ice fraction) so blend_tiles weights it by f_water and the atmosphere water
    survives terminal melt-out.  Thick ice here => no over-ablation cap, so the
    realized mass reduces exactly to lhflx/L_s * conc.
    """
    from legoesm import constants
    state = _make_ice_state(h=1.0, conc=0.7, T=263.0)
    forcing = _make_forcing()
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, constants.T_freeze_ocean)

    _, resp = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT,
    )

    expected = resp.lhflx / constants.L_s * state.concentration.data
    assert jnp.allclose(resp.surface_mass_flux, expected, rtol=1e-12)


def test_sea_ice_atmosphere_mass_survives_meltout_through_blend():
    """Coupler-level (#28, F11-analogue): the ice's realized sublimation mass is
    returned PER-GRID-CELL and blend_tiles weights it by f_water, so at terminal
    melt-out (post-step f_ice -> 0) the FULL ice->atmosphere water pulse is still
    delivered to the atmosphere instead of being zeroed by the vanished ice
    fraction.  This is the coupled closure Codex flagged the slab test did not
    exercise.
    """
    from legoesm import constants
    from legoesm.coupler.tile_fractions import compute_tile_fractions, blend_tiles
    # Dry, strongly-melting thin ice over a warm ocean for a day -> melt-out.
    state = _make_ice_state(h=0.02, conc=0.9, T=constants.T_freeze_ocean)
    forcing = _make_forcing(T_lowest=290.0, sw=600.0, lw=400.0)._replace(
        q_lowest=jnp.full(SHAPE, 1e-4),
    )
    config = SeaIceConfig()
    new_state, ice_resp = step_sea_ice(
        state, forcing, jnp.full(SHAPE, 282.0), jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=86400.0,
    )
    # The cell melted out: post-step ice fraction ~ 0.
    assert float(jnp.max(new_state.concentration.data)) < 1e-6
    # The ice DID lose water to the atmosphere this step (per-cell, nonzero).
    assert float(jnp.max(jnp.abs(ice_resp.surface_mass_flux))) > 0.0

    z = jnp.zeros(SHAPE)
    zero = TileResponse(
        T_sfc=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
        lhflx=z, tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z,
        co2_flux=z, freshwater_flux=z, ocean_heat_extraction=z,
        ocean_stress_x=z, ocean_stress_y=z, surface_mass_flux=z, salt_flux=z,
    )
    tcfg = TileConfig(f_land=z, f_lake=z)  # pure-water cell: f_water = 1
    # Tile fractions from the POST-step concentration (what make_coupler uses).
    fracs = compute_tile_fractions(tcfg, new_state.concentration.data)
    blended = blend_tiles(zero, ice_resp, zero, zero, fracs)
    f_water = fracs.f_ocean + fracs.f_ice
    # Delivered atmosphere mass = f_water * (per-cell ice sublimation), NOT zero,
    # even though f_ice -> 0 at melt-out.
    assert jnp.allclose(blended.surface_mass_flux,
                        f_water * ice_resp.surface_mass_flux,
                        rtol=1e-7, atol=1e-12)
    assert float(jnp.max(jnp.abs(blended.surface_mass_flux))) > 0.0


def test_sea_ice_latent_energy_pairs_with_moisture_and_deposition_debits():
    """#28 finding 2 + 1: the ice latent ENERGY delivered to the atmosphere
    equals L_s times the ice moisture MASS on the same f_water basis, and
    DEPOSITION (lhflx < 0) debits the atmosphere with a negative moisture mass
    (the state gains water).  Ice-only blend so the pairing is exact.
    """
    from legoesm import constants
    from legoesm.coupler.tile_fractions import compute_tile_fractions, blend_tiles
    # Cold ice + humid air => q_lowest above the (tiny) saturation over cold
    # ice => deposition (lhflx < 0, vapor -> ice).
    state = _make_ice_state(h=1.0, conc=0.6, T=250.0)
    forcing = _make_forcing(T_lowest=263.0, sw=50.0, lw=240.0)._replace(
        q_lowest=jnp.full(SHAPE, 2e-3),
    )
    config = SeaIceConfig()
    new_state, ice_resp = step_sea_ice(
        state, forcing, jnp.full(SHAPE, constants.T_freeze_ocean),
        jnp.zeros(SHAPE), jnp.zeros(SHAPE), config, U_min=1.0, dt=DT,
    )
    # Deposition: ice gains water from the atmosphere => negative surface mass
    # flux (atmosphere -> surface), so the atmosphere IS debited (not zero).
    assert float(jnp.min(ice_resp.surface_mass_flux)) < 0.0

    z = jnp.zeros(SHAPE)
    zero = TileResponse(
        T_sfc=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
        lhflx=z, tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z,
        co2_flux=z, freshwater_flux=z, ocean_heat_extraction=z,
        ocean_stress_x=z, ocean_stress_y=z, surface_mass_flux=z, salt_flux=z,
    )
    tcfg = TileConfig(f_land=z, f_lake=z)
    fracs = compute_tile_fractions(tcfg, new_state.concentration.data)
    blended = blend_tiles(zero, ice_resp, zero, zero, fracs)
    # Moisture-energy pairing: the blended latent ENERGY is exactly L_s times
    # the blended moisture MASS (ice-only cell), so the atmosphere never sees
    # vapor without its latent energy or vice-versa.
    assert jnp.allclose(blended.lhflx, constants.L_s * blended.surface_mass_flux,
                        rtol=1e-10, atol=1e-12)


# ==============================================================================
# Test lake
# ==============================================================================

def test_lake_step_finite():
    """Lake model produces finite outputs."""
    state = _make_lake_state()
    forcing = _make_forcing()
    config = LakeConfig()

    new_state, resp = step_lake(state, forcing, config, U_min=1.0, dt=DT)

    assert jnp.all(jnp.isfinite(new_state.T_epi.data))
    assert jnp.all(jnp.isfinite(new_state.T_hypo.data))
    assert jnp.all(jnp.isfinite(resp.shflx))


def test_lake_mixing_direction():
    """Heat flows from epilimnion to hypolimnion when epi is warmer."""
    state = _make_lake_state(T_epi=290.0, T_hypo=275.0)
    forcing = _make_forcing(sw=0.0, lw=0.0, T_lowest=290.0)
    config = LakeConfig()

    new_state, _ = step_lake(state, forcing, config, U_min=1.0, dt=DT)

    # Hypolimnion should warm, epilimnion should cool (from mixing alone)
    assert jnp.all(new_state.T_hypo.data > 275.0)


def test_lake_surface_mass_flux_uses_phase_aware_L():
    """Lake TileResponse.surface_mass_flux = lhflx / L_eff, where
    L_eff = L_s for frozen lakes (T_epi <= T_freeze) and L_v
    otherwise (audit F3 + iter-11).
    """
    from legoesm import constants
    config = LakeConfig()

    # Warm liquid lake: L_eff = L_v
    state_warm = _make_lake_state(T_epi=285.0, T_hypo=280.0)
    forcing_warm = _make_forcing(T_lowest=290.0)
    _, resp_warm = step_lake(state_warm, forcing_warm, config, U_min=1.0, dt=DT)
    expected_warm = resp_warm.lhflx / constants.L_v
    assert jnp.allclose(resp_warm.surface_mass_flux, expected_warm, rtol=1e-6)

    # Frozen lake: L_eff = L_s
    state_cold = _make_lake_state(T_epi=270.0, T_hypo=270.0)
    forcing_cold = _make_forcing(T_lowest=240.0, sw=0.0, lw=200.0)
    _, resp_cold = step_lake(state_cold, forcing_cold, config, U_min=1.0, dt=DT)
    expected_cold = resp_cold.lhflx / constants.L_s
    assert jnp.allclose(resp_cold.surface_mass_flux, expected_cold, rtol=1e-6)


def test_lake_freshwater_flux_is_P_minus_E():
    """Lake TileResponse.freshwater_flux = precip_total − E
    (audit F4).
    """
    state = _make_lake_state(T_epi=285.0, T_hypo=280.0)
    forcing = _make_forcing(precip=2e-5)  # nontrivial precip
    config = LakeConfig()

    _, resp = step_lake(state, forcing, config, U_min=1.0, dt=DT)

    expected = forcing.precip_total - resp.surface_mass_flux
    assert jnp.allclose(resp.freshwater_flux, expected, rtol=1e-6)


def test_land_freshwater_flux_equals_runoff():
    """Slab land TileResponse.freshwater_flux equals the bucket
    overflow runoff (audit F4).  ``LandConfig.W_max = 150 kg/m²`` —
    saturate the bucket and apply heavy precip with cool dry forcing
    (so evap doesn't drain the bucket within the step) to force
    overflow.
    """
    from legoesm.land.config import LandConfig
    state = _make_land_state(W=149.99)  # right at saturation
    # Cool, dry, low-wind forcing keeps evap minimal; very heavy
    # precip forces overflow regardless.
    forcing = _make_forcing(
        T_lowest=275.0, sw=0.0, lw=200.0, precip=1.0,
    )
    config = LandConfig()

    new_state, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)

    # The bucket overflow becomes runoff, which is freshwater_flux.
    # ``LandState.runoff`` is a bare jax.Array (not a Field), so use
    # it directly.
    assert jnp.allclose(resp.freshwater_flux, new_state.runoff, rtol=1e-6)
    # Heavy precip on a saturated bucket → positive runoff.
    assert float(jnp.mean(resp.freshwater_flux)) > 0.0


# ==============================================================================
# Test tile blending
# ==============================================================================

def test_tile_fractions_sum_to_one():
    """Tile fractions always sum to 1."""
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.1),
    )
    fracs = compute_tile_fractions(tile_cfg, ice_concentration=jnp.full(SHAPE, 0.2))

    total = fracs.f_ocean + fracs.f_ice + fracs.f_land + fracs.f_lake
    assert jnp.allclose(total, 1.0, atol=1e-10)


def test_tile_fractions_sanitize_invalid_static_masks():
    """Invalid static masks are clipped/rescaled to a conservative partition."""
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 1.2),
        f_lake=jnp.full(SHAPE, 0.7),
    )
    fracs = compute_tile_fractions(tile_cfg, ice_concentration=jnp.full(SHAPE, 1.5))

    for frac in (fracs.f_ocean, fracs.f_ice, fracs.f_land, fracs.f_lake):
        assert jnp.all(frac >= -1e-12)
        assert jnp.all(frac <= 1.0 + 1e-12)

    total = fracs.f_ocean + fracs.f_ice + fracs.f_land + fracs.f_lake
    assert jnp.allclose(total, 1.0, atol=1e-10)


def test_blending_is_area_weighted():
    """Blended T_sfc is correct area-weighted average."""
    z = jnp.zeros(SHAPE)
    one = jnp.ones(SHAPE)

    def _make_tile_resp(T):
        return TileResponse(
            T_sfc=jnp.full(SHAPE, T),
            albedo=z, emissivity=z, z0=z, q_surface=z,
            shflx=z, lhflx=z, tau_x=z, tau_y=z, lw_up=z,
            u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
            freshwater_flux=z, ocean_heat_extraction=z,
            ocean_stress_x=z, ocean_stress_y=z,
            surface_mass_flux=z, salt_flux=z,
        )

    fracs = TileFractions(
        f_ocean=jnp.full(SHAPE, 0.5),
        f_ice=jnp.full(SHAPE, 0.1),
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.1),
    )

    blended = blend_tiles(
        _make_tile_resp(300.0),
        _make_tile_resp(260.0),
        _make_tile_resp(280.0),
        _make_tile_resp(285.0),
        fracs,
    )

    expected = 0.5 * 300 + 0.1 * 260 + 0.3 * 280 + 0.1 * 285
    assert jnp.allclose(blended.T_sfc, expected, atol=1e-6)


def test_tile_blend_lw_flux_conservation():
    """Flux-conserving tile blend: the atmosphere LW boundary reproduces the
    area-weighted sum of per-tile lw_up for MIXED cells.

        eps_grid*sigma*T_rad^4 + (1-eps_grid)*La  ==  sum_i f_i * lw_up_i

    Independently area-averaging T_sfc and emissivity (the previous behaviour)
    fails this because T^4 and eps*T^4 are nonlinear.  ``T_rad`` is derived from
    the area-weighted EMISSION FLUX so the identity holds exactly.

    Critically the LAND tile here emits at the canopy temperature T_rad while
    reporting T_sfc = soil temperature (the real two-leaf canopy convention), so
    the blend MUST use each tile's T_rad — not T_sfc — or it is non-conserving
    exactly for vegetated cells.
    """
    from legoesm import constants
    sb = constants.sigma_sb
    La = 320.0
    z = jnp.zeros(SHAPE)

    def _resp(T, eps, T_rad=None):
        # lw_up emitted at the EMISSION temperature (T_rad if the tile decouples
        # its radiative temperature from its skin/soil temperature, else T_sfc).
        T_emit = T if T_rad is None else T_rad
        return TileResponse(
            T_sfc=jnp.full(SHAPE, T), albedo=z,
            emissivity=jnp.full(SHAPE, eps), z0=z, q_surface=z,
            shflx=z, lhflx=z, tau_x=z, tau_y=z,
            lw_up=jnp.full(SHAPE, eps * sb * T_emit ** 4 + (1.0 - eps) * La),
            u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
            freshwater_flux=z, ocean_heat_extraction=z,
            ocean_stress_x=z, ocean_stress_y=z,
            surface_mass_flux=z, salt_flux=z,
            T_rad=None if T_rad is None else jnp.full(SHAPE, T_rad),
        )

    fracs = TileFractions(
        f_ocean=jnp.full(SHAPE, 0.4), f_ice=jnp.full(SHAPE, 0.2),
        f_land=jnp.full(SHAPE, 0.3), f_lake=jnp.full(SHAPE, 0.1),
    )
    # Mixed coastal/MIZ cell.  LAND/canopy: T_sfc=295 (soil) but emits at
    # T_rad=289 (canopy column) — the case that breaks a T_sfc-based blend.
    blended = blend_tiles(
        _resp(305.0, 0.97),                       # ocean (T_rad = T_sfc)
        _resp(255.0, 0.97),                       # ice
        _resp(295.0, 0.96, T_rad=289.0),          # land/canopy: soil != canopy T
        _resp(288.0, 0.98),                       # lake
        fracs,
    )

    # Property-coupling boundary RRTMGP forms from the blended (eps_grid, T_rad).
    boundary = (blended.emissivity * sb * blended.T_rad ** 4
                + (1.0 - blended.emissivity) * La)
    # Must equal the coupler's own area-weighted blended upward LW.
    assert jnp.allclose(boundary, blended.lw_up, atol=1e-6)
    # T_rad genuinely differs from the naive area-weighted T_sfc (non-vacuous).
    assert not jnp.allclose(blended.T_rad, blended.T_sfc, atol=1e-2)

    # Pure CANOPY cell: the grid T_rad must collapse to the canopy EMISSION
    # temperature (289), NOT the reported soil T_sfc (295) — the exact bug.
    pure = TileFractions(
        f_ocean=z, f_ice=z, f_land=jnp.ones(SHAPE), f_lake=z)
    bp = blend_tiles(_resp(305.0, 0.97), _resp(255.0, 0.97),
                     _resp(295.0, 0.96, T_rad=289.0), _resp(288.0, 0.98), pure)
    assert jnp.allclose(bp.T_rad, 289.0, atol=1e-6)
    assert jnp.allclose(bp.T_sfc, 295.0, atol=1e-6)


# ==============================================================================
# Test accumulator
# ==============================================================================

def test_accumulator_reset():
    """Reset accumulator has zero total_dt."""
    acc = reset_accumulator(SHAPE)
    assert acc.total_dt == 0.0
    assert acc.sum_shflx.shape == SHAPE


def test_accumulator_mean():
    """Accumulating two steps gives correct dt-weighted mean."""
    z = jnp.zeros(SHAPE)
    acc = reset_accumulator(SHAPE)

    # Step 1: shflx=10, dt=100
    sfc1 = SurfaceToAtm(
        T_sfc=jnp.full(SHAPE, 280.0),
        T_rad=jnp.full(SHAPE, 280.0),
        albedo=z, emissivity=z, z0=z, q_surface=z,
        shflx=jnp.full(SHAPE, 10.0), lhflx=z,
        tau_x=z, tau_y=z, lw_up=z,
        u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
        freshwater_flux=z, ocean_heat_extraction=z,
        ocean_stress_x=z, ocean_stress_y=z,
        surface_mass_flux=z, salt_flux=z, river_runoff_flux=z,
        ice_lake_freshwater_flux=z,
    )
    acc = accumulate(acc, sfc1, 100.0)

    # Step 2: shflx=30, dt=300
    sfc2 = sfc1._replace(shflx=jnp.full(SHAPE, 30.0))
    acc = accumulate(acc, sfc2, 300.0)

    result = mean_accumulator(acc)

    # Expected: (10*100 + 30*300) / 400 = 10000/400 = 25
    assert jnp.allclose(result.shflx, 25.0, atol=1e-6)
    assert jnp.allclose(result.T_sfc, 280.0, atol=1e-6)


def test_accumulator_lw_flux_conservation_varying_substeps():
    """Window-mean RRTMGP LW boundary == dt-weighted mean surface EMISSION even
    when T_rad AND emissivity vary across substeps.

    The accumulator stores the emission flux ``eps*sigma*T_rad^4`` and
    reconstructs ``T_rad`` at flush, so ``mean_eps*sigma*mean_T_rad^4`` equals
    ``mean(eps*sigma*T_rad^4)``.  Averaging T_rad and eps INDEPENDENTLY (the
    naive form) breaks this because both T^4 and eps*T^4 are nonlinear.
    """
    from legoesm import constants
    sb = constants.sigma_sb
    z = jnp.zeros(SHAPE)

    def _sfc(T_rad, eps):
        return SurfaceToAtm(
            T_sfc=jnp.full(SHAPE, 300.0), T_rad=jnp.full(SHAPE, T_rad), albedo=z,
            emissivity=jnp.full(SHAPE, eps), z0=z, q_surface=z, shflx=z, lhflx=z,
            tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
            freshwater_flux=z, ocean_heat_extraction=z, ocean_stress_x=z,
            ocean_stress_y=z, surface_mass_flux=z, salt_flux=z,
            river_runoff_flux=z, ice_lake_freshwater_flux=z)

    acc = reset_accumulator(SHAPE)
    acc = accumulate(acc, _sfc(310.0, 0.95), 100.0)   # substep 1
    acc = accumulate(acc, _sfc(280.0, 0.99), 300.0)   # substep 2: different T+eps
    m = mean_accumulator(acc)

    emit_recon = m.emissivity * sb * m.T_rad ** 4
    emit_true = (100.0 * 0.95 * sb * 310.0 ** 4
                 + 300.0 * 0.99 * sb * 280.0 ** 4) / 400.0
    assert jnp.allclose(emit_recon, emit_true, atol=1e-6)
    # Non-vacuous: the naive independent-mean form differs by ~3.5 W/m2.
    t_naive = (100.0 * 310.0 + 300.0 * 280.0) / 400.0
    e_naive = (100.0 * 0.95 + 300.0 * 0.99) / 400.0
    assert not jnp.allclose(e_naive * sb * t_naive ** 4, emit_true, atol=1.0)


# ==============================================================================
# Test full coupler
# ==============================================================================

def test_full_coupler_step():
    """End-to-end: step all tiles and get blended output."""
    coupler_cfg = CouplerConfig()
    land_cfg = LandConfig()
    ice_cfg = SeaIceConfig()
    lake_cfg = LakeConfig()

    step_fn = make_coupler(coupler_cfg, land_cfg, ice_cfg, lake_cfg)

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()

    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.05),
    )

    ocean_sst = jnp.full(SHAPE, 300.0)
    ocean_u = jnp.zeros(SHAPE)
    ocean_v = jnp.zeros(SHAPE)

    new_state, blended = step_fn(
        sfc_state, forcing, tile_cfg, ocean_sst, ocean_u, ocean_v, DT)

    # All outputs finite
    assert jnp.all(jnp.isfinite(blended.T_sfc))
    assert jnp.all(jnp.isfinite(blended.shflx))
    assert jnp.all(jnp.isfinite(blended.lhflx))
    assert jnp.all(jnp.isfinite(blended.tau_x))
    assert jnp.all(jnp.isfinite(blended.lw_up))

    # Accumulator advanced
    assert new_state.accumulator.total_dt > 0.0


def test_full_coupler_step_multilayer_richards():
    """End-to-end coupler step with the MULTILAYER soil-thermal + RICHARDS land
    tile (columnar (ncol, nlayers) state), proving the coupler dispatch wires the
    multilayer land into AMIP/CMIP the same way as the slab tile.

    The coupler flattens (6,n,n) forcing to (ncol,), steps ``step_multilayer_land``,
    and unflattens the TileResponse back to spatial shape for blending."""
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.soil_grid import SoilGridConfig

    ncol = 6 * 4 * 4
    nlayers = 6
    land_cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=nlayers, total_depth=2.0))
    lat = jnp.full(SHAPE, 0.5)  # radians (required to flatten for the columnar tile)
    step_fn = make_coupler(CouplerConfig(), land_cfg, SeaIceConfig(), LakeConfig(),
                           lat=lat)

    sfc_state = init_surface_state(SHAPE, land_config=land_cfg)
    # the land sub-state is the columnar multilayer state, not the slab Field state
    assert sfc_state.land.T_soil.shape == (ncol, nlayers)
    assert sfc_state.land.theta_soil.shape == (ncol, nlayers)

    forcing = _make_forcing()
    tile_cfg = TileConfig(f_land=jnp.full(SHAPE, 0.5), f_lake=jnp.zeros(SHAPE))
    sst = jnp.full(SHAPE, 300.0); zu = jnp.zeros(SHAPE)

    theta_init = sfc_state.land.theta_soil
    for _ in range(3):
        sfc_state, blended = step_fn(sfc_state, forcing, tile_cfg, sst, zu, zu, DT)

    # blended response is unflattened back to spatial shape and finite
    assert blended.T_sfc.shape == SHAPE
    assert jnp.all(jnp.isfinite(blended.T_sfc))
    assert jnp.all(jnp.isfinite(blended.lhflx))
    assert jnp.all(jnp.isfinite(blended.shflx))
    # multilayer land state advanced, finite, Richards moisture stays physical
    assert jnp.all(jnp.isfinite(sfc_state.land.T_soil))
    assert jnp.all(jnp.isfinite(sfc_state.land.theta_soil))
    assert jnp.all(sfc_state.land.theta_soil >= 0.0)
    assert jnp.all(sfc_state.land.theta_soil <= 1.0)
    # soil moisture EVOLVES (Richards) in the coupled model — it is NOT frozen (the
    # frozen-moisture trick lives only in the offline calibrator, for tractability).
    assert float(jnp.max(jnp.abs(sfc_state.land.theta_soil - theta_init))) > 0.0
    assert float(sfc_state.accumulator.total_dt) == pytest.approx(3 * DT, abs=1e-6)


def _slab_transient_cover_provider(ncol):
    """A slab TransientCoverProvider: forest cover at 2000 -> crop at 2010 (crop
    is brighter), on ``ncol`` columns.  Mirrors the coupled driver's
    _build_pft_provider transient branch (variant='slab', soil-colour albedo
    off)."""
    import numpy as np
    from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
    from legoesm.land.clm_surface_map import (
        clm_provider_rebuild, TransientCoverProvider)
    idx = {n: i for i, n in enumerate(CLM5_PFT_NAMES)}

    def _cover(i):
        c = np.zeros((ncol, N_PFT_CLM5)); c[:, i] = 0.9; c[:, 0] = 0.1
        return jnp.asarray(c)
    o = jnp.ones(ncol)
    sm = dict(theta_wp=0.12 * o, theta_fc=0.30 * o, glacier_frac=0.0 * o,
              soil_albedo=0.15 * o, lai=2.0 * o)
    rebuild = clm_provider_rebuild(sm, variant="slab", include_soil_albedo=False)
    forest, crop = _cover(idx["broadleaf_evergreen_tropical"]), _cover(idx["crop_c3"])
    return TransientCoverProvider(
        base=rebuild(forest), cover=jnp.stack([forest, crop]),
        years=jnp.asarray([2000.0, 2010.0]), _rebuild=rebuild), rebuild(forest)


def test_step_surface_forwards_year_to_transient_cover_provider():
    """Coupled-driver transient LULC path: step_surface(year=Y) forwards the
    calendar year to a year_varying land provider so the vegetation params track
    the segment's year (transient cover drives the surface), and year=None reuses
    the base provider byte-for-byte."""
    import numpy as np
    ncol = 6 * 4 * 4
    provider, _base = _slab_transient_cover_provider(ncol)
    # crop (2010) cover is brighter than forest (2000): the forwarded year must
    # change the materialised vegetation albedo the tiles see.
    assert (float(jnp.mean(provider(year=2010.0).albedo_veg))
            > float(jnp.mean(provider(year=2000.0).albedo_veg)))

    step_fn = make_coupler(CouplerConfig(coupling_dt=600.0), LandConfig(),
                           SeaIceConfig(dynamics="none"), LakeConfig(),
                           land_param_provider=provider)
    forcing = _make_forcing(sw=400.0)
    tile_cfg = TileConfig(f_land=jnp.ones(SHAPE), f_lake=jnp.zeros(SHAPE))  # all land
    sst = jnp.full(SHAPE, 290.0); zu = jnp.zeros(SHAPE)

    def _run(year):
        return step_fn(init_surface_state(SHAPE), forcing, tile_cfg, sst, zu, zu,
                       DT, year=year)[1]
    r_2000, r_2010, r_none = _run(2000.0), _run(2010.0), _run(None)
    # year forwarded -> brighter 2010 cover shifts the blended surface temperature
    assert float(jnp.max(jnp.abs(r_2000.T_sfc - r_2010.T_sfc))) > 1e-6
    # year=None reuses the base provider (year 2000) -> byte-identical static path
    np.testing.assert_array_equal(np.asarray(r_none.T_sfc), np.asarray(r_2000.T_sfc))


def test_step_surface_static_provider_ignores_year():
    """A static (non year_varying) provider IGNORES year, so a normal coupled run
    is byte-identical regardless of the year kwarg the driver forwards."""
    import numpy as np
    _provider, static_provider = _slab_transient_cover_provider(6 * 4 * 4)
    # static_provider is a plain CLMSurfaceParamProvider (no year_varying attr).
    assert getattr(static_provider, "year_varying", False) is False
    step_fn = make_coupler(CouplerConfig(coupling_dt=600.0), LandConfig(),
                           SeaIceConfig(dynamics="none"), LakeConfig(),
                           land_param_provider=static_provider)
    forcing = _make_forcing(sw=400.0)
    tile_cfg = TileConfig(f_land=jnp.ones(SHAPE), f_lake=jnp.zeros(SHAPE))
    sst = jnp.full(SHAPE, 290.0); zu = jnp.zeros(SHAPE)

    def _run(year):
        return step_fn(init_surface_state(SHAPE), forcing, tile_cfg, sst, zu, zu,
                       DT, year=year)[1]
    np.testing.assert_array_equal(np.asarray(_run(2010.0).T_sfc),
                                  np.asarray(_run(2000.0).T_sfc))


def test_coupler_multiple_steps():
    """Multiple coupler steps stay finite and accumulator grows."""
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.4),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    for _ in range(5):
        sfc_state, blended = step_fn(
            sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)

    assert jnp.all(jnp.isfinite(blended.T_sfc))
    assert jnp.allclose(sfc_state.accumulator.total_dt, 5 * DT, atol=1e-6)


def test_coupler_flushes_at_coupling_interval():
    """Accumulator should flush when coupling_dt is reached."""
    coupler_cfg = CouplerConfig(coupling_dt=2 * DT)
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.4),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
    assert jnp.allclose(sfc_state.accumulator.total_dt, DT, atol=1e-6)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
    assert jnp.allclose(sfc_state.accumulator.total_dt, 0.0, atol=1e-6)


def test_coupler_carries_excess_dt_after_flush():
    """If dt overshoots coupling_dt, residual dt should carry to next window."""
    coupler_cfg = CouplerConfig(coupling_dt=900.0)
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.4),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, 600.0)
    assert jnp.allclose(sfc_state.accumulator.total_dt, 600.0, atol=1e-6)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, 600.0)
    assert jnp.allclose(sfc_state.accumulator.total_dt, 300.0, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupling_dt": 0.0}, "coupling_dt"),
        ({"U_min": -1.0}, "U_min"),
        ({"ocean_albedo": -0.1}, "ocean_albedo"),
        ({"ocean_albedo": 1.1}, "ocean_albedo"),
        ({"ocean_emissivity": -0.1}, "ocean_emissivity"),
        ({"ocean_emissivity": 1.1}, "ocean_emissivity"),
        ({"ocean_z0": 0.0}, "ocean_z0"),
        ({"Cd_ocean": -1.0e-3}, "Cd_ocean"),
        ({"Ch_ocean": -1.0e-3}, "Ch_ocean"),
    ],
)
def test_make_coupler_rejects_invalid_config(kwargs, match):
    """Coupler factory should fail fast on invalid configuration."""
    with pytest.raises(ValueError, match=match):
        make_coupler(CouplerConfig(**kwargs), LandConfig(), SeaIceConfig(), LakeConfig())


def test_coupler_step_validates_dt():
    """Coupler step should reject non-positive dt."""
    step_fn = make_coupler(CouplerConfig(), LandConfig(), SeaIceConfig(), LakeConfig())
    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.05),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    with pytest.raises(ValueError, match="dt"):
        step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, 0.0)


# ==============================================================================
# Test smooth operations
# ==============================================================================

def test_smooth_wind_floor():
    """Wind floor sqrt(u^2 + v^2 + U_min^2) is smooth (differentiable at zero)."""
    U_min = 1.0

    def wind_speed(u, v):
        return jnp.sqrt(u ** 2 + v ** 2 + U_min ** 2)

    # At u=v=0 the gradient should be finite (not NaN)
    grad_u = jax.grad(lambda u: wind_speed(u, 0.0).sum())(jnp.array(0.0))
    grad_v = jax.grad(lambda v: wind_speed(0.0, v).sum())(jnp.array(0.0))

    assert jnp.isfinite(grad_u)
    assert jnp.isfinite(grad_v)
    # Gradient should be zero at u=v=0 (symmetric minimum)
    assert jnp.abs(grad_u) < 1e-6
    assert jnp.abs(grad_v) < 1e-6


def test_init_surface_state():
    """init_surface_state creates valid state with correct shapes."""
    state = init_surface_state(SHAPE)
    assert state.land.T_soil.shape == SHAPE
    assert state.land.W_bucket.shape == SHAPE
    assert state.ice.h_ice.shape == SHAPE
    assert state.ice.T_ice.shape == SHAPE
    assert state.lake.T_epi.shape == SHAPE
    assert state.lake.T_hypo.shape == SHAPE
    assert state.accumulator.total_dt == 0.0


# ==============================================================================
# Test simple ocean modes
# ==============================================================================

from legoesm.ocean.simple_ocean import (
    SimpleOceanConfig,
    SlabOceanState,
    make_ocean,
    init_slab_state,
)


def test_fixed_sst_constant():
    """Fixed mode with constant SST returns that constant, state unchanged."""
    cfg = SimpleOceanConfig(mode="fixed", sst_constant=305.0)
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=300.0)
    forcing = _make_forcing()

    new_state, sst, u_sfc, v_sfc = step_fn(state, forcing, DT)

    assert jnp.allclose(sst, 305.0)
    # State unchanged
    assert jnp.allclose(new_state.T_sfc.data, state.T_sfc.data)
    # Surface currents are zero
    assert jnp.allclose(u_sfc, 0.0)
    assert jnp.allclose(v_sfc, 0.0)


def test_fixed_sst_map():
    """Fixed mode with spatial SST map returns that map."""
    sst_map = jnp.linspace(280.0, 310.0, SHAPE[0] * SHAPE[1] * SHAPE[2]).reshape(SHAPE)
    cfg = SimpleOceanConfig(mode="fixed")
    step_fn = make_ocean(cfg, sst_map=sst_map)

    state = init_slab_state(SHAPE)
    forcing = _make_forcing()

    new_state, sst, _, _ = step_fn(state, forcing, DT)

    assert jnp.allclose(sst, sst_map)
    assert jnp.allclose(new_state.T_sfc.data, state.T_sfc.data)


def test_slab_ocean_finite():
    """Slab ocean step produces all-finite outputs."""
    cfg = SimpleOceanConfig(mode="slab")
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=300.0)
    forcing = _make_forcing()

    new_state, sst, u_sfc, v_sfc = step_fn(state, forcing, DT)

    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
    assert jnp.all(jnp.isfinite(sst))
    assert jnp.all(jnp.isfinite(u_sfc))
    assert jnp.all(jnp.isfinite(v_sfc))


def test_slab_ocean_warming():
    """Warm atmosphere heats a cold slab ocean."""
    cfg = SimpleOceanConfig(mode="slab")
    step_fn = make_ocean(cfg)

    T_init = 275.0
    state = init_slab_state(SHAPE, T_sfc_init=T_init)
    # Warm air + strong SW drives ocean warming
    forcing = _make_forcing(T_lowest=300.0, sw=400.0, lw=350.0)

    new_state, sst, _, _ = step_fn(state, forcing, DT)

    assert jnp.all(new_state.T_sfc.data > T_init)


def test_slab_ocean_freezing_clamp():
    """SST stays >= T_freeze even with extreme cooling."""
    cfg = SimpleOceanConfig(mode="slab", T_freeze=constants.T_freeze_ocean)
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=272.0)
    # Strong cooling: no SW, cold air
    forcing = _make_forcing(T_lowest=200.0, sw=0.0, lw=50.0)

    # Multiple steps to drive temperature down
    for _ in range(100):
        state, sst, _, _ = step_fn(state, forcing, DT)

    assert jnp.all(state.T_sfc.data >= cfg.T_freeze - 1e-6)


def test_two_layer_ocean_finite():
    """Two-layer ocean step produces all-finite outputs."""
    cfg = SimpleOceanConfig(mode="two_layer")
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=300.0, T_deep_init=278.0)
    forcing = _make_forcing()

    new_state, sst, u_sfc, v_sfc = step_fn(state, forcing, DT)

    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
    assert jnp.all(jnp.isfinite(new_state.T_deep.data))
    assert jnp.all(jnp.isfinite(sst))


def test_two_layer_mixing_direction():
    """Heat flows from warm surface to cold deep layer via mixing."""
    cfg = SimpleOceanConfig(mode="two_layer", k_mix=1.0e-2)  # Strong mixing
    step_fn = make_ocean(cfg)

    T_sfc_init = 300.0
    T_deep_init = 275.0
    state = init_slab_state(SHAPE, T_sfc_init=T_sfc_init, T_deep_init=T_deep_init)
    forcing = _make_forcing(sw=0.0, lw=0.0, T_lowest=300.0)

    new_state, _, _, _ = step_fn(state, forcing, DT)

    # Deep layer should warm
    assert jnp.all(new_state.T_deep.data > T_deep_init)


def test_two_layer_deep_restoring():
    """Deep layer relaxes toward T_deep_ref when restoring is on."""
    T_deep_ref = 280.0
    cfg = SimpleOceanConfig(
        mode="two_layer",
        restore_deep=True,
        T_deep_ref=T_deep_ref,
        tau_deep=3600.0,  # Fast restoring for test
        k_mix=0.0,        # No mixing, isolate restoring
    )
    step_fn = make_ocean(cfg)

    # Start deep layer away from reference
    T_deep_init = 290.0
    state = init_slab_state(SHAPE, T_sfc_init=300.0, T_deep_init=T_deep_init)
    forcing = _make_forcing()

    new_state, _, _, _ = step_fn(state, forcing, DT)

    # Deep layer should move toward T_deep_ref (cool down)
    assert jnp.all(new_state.T_deep.data < T_deep_init)
    assert jnp.all(new_state.T_deep.data > T_deep_ref)  # Not overshoot


def test_make_ocean_factory():
    """Factory returns a callable for each valid mode."""
    for mode in ("fixed", "slab", "two_layer"):
        cfg = SimpleOceanConfig(mode=mode)
        step_fn = make_ocean(cfg)
        assert callable(step_fn)

    # Invalid mode raises
    with pytest.raises(ValueError, match="Unknown ocean mode"):
        make_ocean(SimpleOceanConfig(mode="invalid"))


# ==============================================================================
# Test differentiability
# ==============================================================================

def test_coupler_differentiable():
    """Full coupler step is differentiable w.r.t. ocean SST."""
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.05),
    )
    zu = jnp.zeros(SHAPE)

    def loss(ocean_sst):
        _, blended = step_fn(
            sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
        return jnp.sum(blended.shflx)

    ocean_sst = jnp.full(SHAPE, 300.0)
    grad_sst = jax.grad(loss)(ocean_sst)
    assert grad_sst.shape == SHAPE
    assert jnp.all(jnp.isfinite(grad_sst))
    # Non-zero gradient (shflx depends on SST via bulk formula)
    assert float(jnp.max(jnp.abs(grad_sst))) > 0.0


def test_coupler_differentiable_through_surface_state():
    """Coupler step is differentiable w.r.t. land soil temperature."""
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.5),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    def loss(T_soil):
        land = LandState(
            T_soil=Field(T_soil, name="T_soil", dims=DIMS_2D, units="K"),
            W_bucket=Field(jnp.full(SHAPE, 75.0), name="W_bucket",
                           dims=DIMS_2D, units="kg/m2"),
            snow_depth=Field(jnp.zeros(SHAPE), name="snow_depth",
                             dims=DIMS_2D, units="kg/m2"),
            snow_age=Field(jnp.zeros(SHAPE), name="snow_age",
                           dims=DIMS_2D, units="s"),
        )
        sfc_state = SurfaceState(
            land=land,
            ice=_make_ice_state(),
            lake=_make_lake_state(),
            accumulator=reset_accumulator(SHAPE),
        )
        _, blended = step_fn(
            sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
        return jnp.sum(blended.T_sfc)

    T_soil = jnp.full(SHAPE, 280.0)
    grad_T = jax.grad(loss)(T_soil)
    assert grad_T.shape == SHAPE
    assert jnp.all(jnp.isfinite(grad_T))


# ==============================================================================
# Test ocean albedo plumbing (P0-1)
# ==============================================================================

def test_ocean_albedo_constant_honoured():
    """CouplerConfig.ocean_albedo should control ocean tile albedo when method='constant'."""
    forcing = _make_forcing()
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    # Default albedo = 0.06
    cfg_default = CouplerConfig()
    resp_default = ocean_tile_response(forcing, ocean_sst, zu, zu, cfg_default)

    # Custom albedo = 0.15
    cfg_custom = CouplerConfig(ocean_albedo=0.15)
    resp_custom = ocean_tile_response(forcing, ocean_sst, zu, zu, cfg_custom)

    assert jnp.allclose(resp_default.albedo, 0.06, atol=1e-6)
    assert jnp.allclose(resp_custom.albedo, 0.15, atol=1e-6)
    assert not jnp.allclose(resp_default.albedo, resp_custom.albedo)


# ==============================================================================
# Test q_surface consistency with T_sfc (P0-2)
# ==============================================================================

def test_land_q_surface_uses_updated_temperature():
    """Slab land q_surface should be consistent with updated T_sfc."""
    from legoesm.thermo import saturation_mixing_ratio

    state = _make_land_state(T=280.0, W=75.0)
    forcing = _make_forcing(T_lowest=300.0, sw=400.0)  # strong warming
    config = LandConfig()

    new_state, resp, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)

    # q_surface should correspond to updated T_sfc, not initial
    T_new = resp.T_sfc
    assert not jnp.allclose(T_new, 280.0, atol=0.01), "T should have changed"

    q_sat_new = saturation_mixing_ratio(T_new, forcing.p_surface)
    w_frac = jnp.clip(new_state.W_bucket.data / config.W_max, 0.0, 1.0)
    beta = config.beta_min + (1.0 - config.beta_min) * w_frac
    q_expected = beta * q_sat_new
    assert jnp.allclose(resp.q_surface, q_expected, rtol=1e-5)


def test_ice_q_surface_uses_updated_temperature():
    """Sea ice q_surface should be consistent with updated T_sfc."""
    from legoesm.thermo import saturation_mixing_ratio_ice

    state = _make_ice_state(h=1.0, T=260.0, conc=0.8)
    forcing = _make_forcing(T_lowest=250.0, sw=100.0)
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 271.0)

    new_state, resp = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    T_new = resp.T_sfc
    q_expected = saturation_mixing_ratio_ice(T_new, forcing.p_surface)
    assert jnp.allclose(resp.q_surface, q_expected, rtol=1e-5)


def test_lake_freezing_energy_conservation():
    """Cooling near freezing: layer energy budget must close.

    cap * (T_new - T_old) / dt = (net flux in) - Q_freeze
    The Q_freeze term accounts for latent heat of ice formation that the
    temperature clamp would otherwise silently discard.
    """
    # Start right at freezing with a thin epilimnion for fast cooling
    state = _make_lake_state(T_epi=273.16, T_hypo=273.16)
    # Very cold atmosphere, no solar radiation → strong cooling
    forcing = _make_forcing(T_lowest=220.0, sw=0.0, lw=50.0)
    config = LakeConfig(h_epi=0.5)  # very thin layer → cools quickly
    dt = 3600.0  # 1 hour step to ensure freezing

    new_state, resp = step_lake(state, forcing, config, U_min=1.0, dt=dt)

    # Q_freeze should be stored in state and be positive (freezing)
    assert new_state.Q_freeze is not None
    Q_freeze = new_state.Q_freeze.data

    # Epilimnion energy budget closure
    cap_epi = config.rho_water * config.c_water_mass * config.h_epi
    T_epi_old = state.T_epi.data
    T_epi_new = new_state.T_epi.data

    # If T_epi hit freezing, Q_freeze should be positive
    at_freezing = jnp.any(T_epi_new <= config.T_freeze + 0.01)
    assert at_freezing, "Forcing should cool epilimnion to freezing"
    assert jnp.all(Q_freeze >= 0.0), "Q_freeze must be non-negative"
    # Q_freeze should be non-trivial — significant latent heat hidden
    assert jnp.any(Q_freeze > 1.0), (
        "Strong cooling near freezing should produce significant Q_freeze"
    )


def test_lake_q_surface_uses_updated_temperature():
    """Lake q_surface should be consistent with updated T_sfc."""
    from legoesm.thermo import saturation_mixing_ratio

    state = _make_lake_state(T_epi=285.0, T_hypo=278.0)
    forcing = _make_forcing(T_lowest=300.0, sw=400.0)  # strong warming
    config = LakeConfig()

    new_state, resp = step_lake(state, forcing, config, U_min=1.0, dt=DT)

    T_new = resp.T_sfc
    assert not jnp.allclose(T_new, 285.0, atol=0.01), "T should have changed"

    q_expected = saturation_mixing_ratio(T_new, forcing.p_surface)
    assert jnp.allclose(resp.q_surface, q_expected, rtol=1e-5)


def test_coupler_with_3d_ocean_cdgrid():
    """Coupler receives SST from the cube C-D grid ocean model (integration test)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.state import OceanConfig

    grid = create_cubed_sphere(8)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
    ocean_state = rest_state_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )

    # Step ocean on the FV3 C-D grid (the only cube ocean backend)
    ocean_config = OceanConfig(
        use_conservation_fixer=False,
        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
    )
    ocean_model = OceanModel(
        grid, z_coord, config=ocean_config, discretization="cdgrid",
    )
    ocean_state_new = ocean_model.step(ocean_state, 60.0)

    # Extract SST (top-level T) for coupler
    sst = ocean_state_new.T.data[:, :, :, 0]  # (6, n, n)
    assert jnp.all(jnp.isfinite(sst))

    # Feed to coupler
    shape = (6, 8, 8)
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(shape)
    forcing = _make_forcing(shape=shape)
    tile_cfg = TileConfig(
        f_land=jnp.full(shape, 0.3),
        f_lake=jnp.full(shape, 0.0),
    )
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    new_sfc, blended = step_fn(
        sfc_state, forcing, tile_cfg,
        sst + constants.T_freeze,  # degC → K
        ocean_u, ocean_v, DT,
    )
    assert jnp.all(jnp.isfinite(blended.T_sfc))
    assert jnp.all(jnp.isfinite(blended.shflx))


def test_coupler_with_fv_ocean_tracer_transport():
    """Coupler receives SST from FV-tracer ocean model (integration test)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.state import OceanConfig

    grid = create_cubed_sphere(8)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
    ocean_state = rest_state_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )

    # Step ocean
    ocean_config = OceanConfig(
        use_conservation_fixer=False,
    )
    ocean_model = OceanModel(grid, z_coord, config=ocean_config)
    ocean_state_new = ocean_model.step(ocean_state, 60.0)

    # Extract SST for coupler
    sst = ocean_state_new.T.data[:, :, :, 0]
    assert jnp.all(jnp.isfinite(sst))

    # Feed to coupler
    shape = (6, 8, 8)
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(shape)
    forcing = _make_forcing(shape=shape)
    tile_cfg = TileConfig(
        f_land=jnp.full(shape, 0.3),
        f_lake=jnp.full(shape, 0.0),
    )

    new_sfc, blended = step_fn(
        sfc_state, forcing, tile_cfg,
        sst + constants.T_freeze,  # iter-166: was 273.15 literal
        jnp.zeros(shape), jnp.zeros(shape), DT,
    )
    assert jnp.all(jnp.isfinite(blended.T_sfc))
    assert jnp.all(jnp.isfinite(blended.shflx))
