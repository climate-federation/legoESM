"""Category 1: Slab Land -- Smoke Tests & Energy Balance.

Tests the slab thermal land model with bucket hydrology for:
  - finite outputs and physical bounds
  - surface energy balance closure
  - flux sign conventions
  - moisture availability
  - bucket hydrology water conservation
  - Stefan-Boltzmann consistency
  - multi-step stability
  - wind speed floor
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.land.slab_land import step_land
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm import constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE = (4,)
DIMS = ("ncol",)


def _field(val: float, name: str = "", units: str = "") -> Field:
    return Field(data=jnp.full(SHAPE, val, dtype=jnp.float64), name=name, dims=DIMS, units=units)


def make_state(T_soil=280.0, W_bucket=50.0, snow_depth=0.0, snow_age=0.0) -> LandState:
    return LandState(
        T_soil=_field(T_soil, "T_soil", "K"),
        W_bucket=_field(W_bucket, "W_bucket", "kg/m2"),
        snow_depth=_field(snow_depth, "snow_depth", "kg/m2"),
        snow_age=_field(snow_age, "snow_age", "s"),
    )


def make_forcing(
    sw_down=300.0,
    lw_down=300.0,
    T_lowest=280.0,
    q_lowest=0.008,
    u_lowest=5.0,
    v_lowest=2.0,
    p_lowest=95000.0,
    p_surface=1e5,
    rho_lowest=1.2,
    cos_zenith=0.5,
    co2_ppmv=415.0,
    precip_total=0.0,
    precip_snow=0.0,
    has_radiation=1.0,
    has_precipitation=1.0,
) -> AtmToSurface:
    f = jnp.float64
    s = SHAPE
    return AtmToSurface(
        sw_down=jnp.full(s, sw_down, f),
        lw_down=jnp.full(s, lw_down, f),
        precip_total=jnp.full(s, precip_total, f),
        precip_snow=jnp.full(s, precip_snow, f),
        T_lowest=jnp.full(s, T_lowest, f),
        q_lowest=jnp.full(s, q_lowest, f),
        u_lowest=jnp.full(s, u_lowest, f),
        v_lowest=jnp.full(s, v_lowest, f),
        p_lowest=jnp.full(s, p_lowest, f),
        p_surface=jnp.full(s, p_surface, f),
        rho_lowest=jnp.full(s, rho_lowest, f),
        cos_zenith=jnp.full(s, cos_zenith, f),
        co2_ppmv=jnp.full(s, co2_ppmv, f),
        has_radiation=jnp.full(s, has_radiation, f),
        has_precipitation=jnp.full(s, has_precipitation, f),
    )


CONFIG = LandConfig()
DT = 3600.0


# ===================================================================
# 1a  Smoke test -- step_land runs and returns finite outputs
# ===================================================================


class Test1a_SmokeSlab:
    def test_returns_finite(self):
        state = make_state()
        forcing = make_forcing()
        new_state, response, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)

        # LandState fields
        for name in LandState._fields:
            arr = getattr(new_state, name).data
            assert jnp.all(jnp.isfinite(arr)), f"LandState.{name} has non-finite values"

        # TileResponse fields
        for name in TileResponse._fields:
            arr = getattr(response, name)
            assert jnp.all(jnp.isfinite(arr)), f"TileResponse.{name} has non-finite values"

    def test_T_soil_bounded(self):
        state = make_state()
        forcing = make_forcing()
        new_state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        T = new_state.T_soil.data
        assert jnp.all(T > 150.0), f"T_soil below 150 K: {T}"
        assert jnp.all(T < 400.0), f"T_soil above 400 K: {T}"

    def test_W_bucket_bounded(self):
        state = make_state()
        forcing = make_forcing()
        new_state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        W = new_state.W_bucket.data
        assert jnp.all(W >= 0.0)
        assert jnp.all(W <= CONFIG.W_max)

    def test_snow_nonneg(self):
        state = make_state(snow_depth=5.0)
        forcing = make_forcing()
        new_state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        assert jnp.all(new_state.snow_depth.data >= 0.0)


# ===================================================================
# 1b  Surface energy balance closure
# ===================================================================


class Test1b_EnergyBalance:
    def test_energy_balance_closure(self):
        """dT/dt should equal (Q_net - melt_energy) / (C * d)."""
        state = make_state(T_soil=280.0, W_bucket=50.0, snow_depth=0.0)
        forcing = make_forcing()
        config = CONFIG

        new_state, response, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)

        T_old = state.T_soil.data
        T_new = new_state.T_soil.data

        # Recompute radiation at OLD T (same as slab_land.py does)
        alpha = jnp.full(SHAPE, config.albedo_land, jnp.float64)
        sw_net = (1.0 - alpha) * forcing.sw_down
        lw_up_old = config.emissivity_land * constants.sigma_sb * T_old ** 4
        lw_net = config.emissivity_land * forcing.lw_down - lw_up_old

        shflx = response.shflx
        lhflx = response.lhflx

        Q_net = sw_net + lw_net - shflx - lhflx

        heat_cap = config.C_soil * config.d_soil
        # No snow => no melt energy
        dT_expected = DT * Q_net / heat_cap
        dT_actual = T_new - T_old

        rel_err = jnp.abs(dT_actual - dT_expected) / jnp.maximum(jnp.abs(dT_expected), 1e-12)
        assert jnp.all(rel_err < 0.01), (
            f"Energy balance not closed: rel_err = {float(rel_err[0]):.4e}"
        )


# ===================================================================
# 1c  Surface flux signs
# ===================================================================


class Test1c_FluxSigns:
    def test_warm_surface_shflx_positive(self):
        """T_soil > T_air => shflx > 0 (upward)."""
        state = make_state(T_soil=300.0)
        forcing = make_forcing(T_lowest=280.0)
        _, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        assert jnp.all(resp.shflx > 0), f"shflx = {resp.shflx}"

    def test_cold_surface_shflx_negative(self):
        """T_soil < T_air => shflx < 0 (downward)."""
        state = make_state(T_soil=260.0)
        forcing = make_forcing(T_lowest=280.0)
        _, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        assert jnp.all(resp.shflx < 0), f"shflx = {resp.shflx}"

    def test_lw_up_always_positive(self):
        """Surface always emits upward LW."""
        state = make_state(T_soil=280.0)
        forcing = make_forcing(cos_zenith=0.5)
        _, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        assert jnp.all(resp.lw_up > 0), f"lw_up = {resp.lw_up}"

    def test_evaporation_sign(self):
        """When surface q > atm q, lhflx > 0 (upward evaporation)."""
        # Wet surface, dry air
        state = make_state(T_soil=300.0, W_bucket=150.0)
        forcing = make_forcing(T_lowest=300.0, q_lowest=0.001)  # very dry air
        _, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        assert jnp.all(resp.lhflx > 0), f"lhflx = {resp.lhflx}"


# ===================================================================
# 1d  Moisture availability
# ===================================================================


class Test1d_MoistureAvailability:
    def test_wet_vs_dry(self):
        """More soil water => more evaporation."""
        forcing = make_forcing(T_lowest=290.0, q_lowest=0.005)

        state_wet = make_state(T_soil=295.0, W_bucket=CONFIG.W_max)
        _, resp_wet, _ = step_land(state_wet, forcing, CONFIG, U_min=1.0, dt=DT)

        state_dry = make_state(T_soil=295.0, W_bucket=0.0)
        _, resp_dry, _ = step_land(state_dry, forcing, CONFIG, U_min=1.0, dt=DT)

        # Wet soil should have larger (more positive) latent heat flux
        assert jnp.all(resp_wet.lhflx > resp_dry.lhflx), (
            f"lhflx wet={float(resp_wet.lhflx[0]):.2f}, "
            f"dry={float(resp_dry.lhflx[0]):.2f}"
        )

    def test_beta_at_wmax(self):
        """At W = W_max, beta should be near 1.0."""
        w_frac = CONFIG.W_max / CONFIG.W_max
        beta = CONFIG.beta_min + (1.0 - CONFIG.beta_min) * w_frac
        assert abs(beta - 1.0) < 1e-10

    def test_beta_at_zero(self):
        """At W = 0, beta should equal beta_min."""
        w_frac = 0.0 / CONFIG.W_max
        beta = CONFIG.beta_min + (1.0 - CONFIG.beta_min) * w_frac
        assert abs(beta - CONFIG.beta_min) < 1e-10


# ===================================================================
# 1e  Bucket hydrology water conservation
# ===================================================================


class Test1e_WaterConservation:
    def test_precip_accumulation(self):
        """Precipitation without evaporation increases W correctly."""
        precip = 1e-4  # kg/m2/s
        state = make_state(T_soil=260.0, W_bucket=50.0)
        forcing = make_forcing(
            precip_total=precip, precip_snow=0.0,
            T_lowest=260.0, q_lowest=0.02, sw_down=0.0, lw_down=280.0,
        )
        new_state, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        W_old = state.W_bucket.data[0]
        W_new = new_state.W_bucket.data[0]
        evap_rate = resp.lhflx / constants.L_v  # kg/m2/s
        dW_expected = (precip - float(evap_rate[0])) * DT
        dW_actual = float(W_new - W_old)
        rel_err = abs(dW_actual - dW_expected) / max(abs(dW_expected), 1e-12)
        assert rel_err < 0.02, f"dW actual={dW_actual:.4f}, expected={dW_expected:.4f}"

    def test_runoff_at_capacity(self):
        """When W = W_max and precip > evap, W stays capped."""
        state = make_state(T_soil=260.0, W_bucket=CONFIG.W_max)
        forcing = make_forcing(
            precip_total=1e-3, precip_snow=0.0,
            T_lowest=260.0, q_lowest=0.02, sw_down=0.0, lw_down=280.0,
        )
        new_state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        W_new = new_state.W_bucket.data
        assert jnp.all(W_new <= CONFIG.W_max + 1e-10)

    def test_no_negative_water(self):
        """When W = 0 and evap > 0, W stays at 0."""
        state = make_state(T_soil=310.0, W_bucket=0.0)
        forcing = make_forcing(T_lowest=280.0, q_lowest=0.001, sw_down=400.0)
        new_state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        W_new = new_state.W_bucket.data
        assert jnp.all(W_new >= 0.0)


# ===================================================================
# 1f  Stefan-Boltzmann consistency
# ===================================================================


class Test1f_StefanBoltzmann:
    def test_lw_up_matches_stefan_boltzmann(self):
        """lw_up should equal emissivity * sigma * T_surface^4."""
        state = make_state(T_soil=290.0)
        forcing = make_forcing()
        new_state, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)

        T_sfc = new_state.T_soil.data
        expected_lw_up = CONFIG.emissivity_land * constants.sigma_sb * T_sfc ** 4

        rel_err = jnp.abs(resp.lw_up - expected_lw_up) / expected_lw_up
        assert jnp.all(rel_err < 0.001), (
            f"LW_up mismatch: got {float(resp.lw_up[0]):.2f}, "
            f"expected {float(expected_lw_up[0]):.2f}"
        )


# ===================================================================
# 1g  Multi-step stability
# ===================================================================


class Test1g_MultiStepStability:
    def test_100_steps_stable(self):
        """Run 100 steps with constant forcing; T stays bounded, no NaN."""
        state = make_state(T_soil=260.0, W_bucket=75.0)
        forcing = make_forcing()

        T_history = []
        for _ in range(100):
            state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
            T_history.append(float(state.T_soil.data[0]))

        T_arr = jnp.array(T_history)
        assert jnp.all(jnp.isfinite(T_arr)), "NaN in T history"
        assert jnp.all(T_arr > 200.0), f"T_soil below 200 K: min={float(jnp.min(T_arr))}"
        assert jnp.all(T_arr < 350.0), f"T_soil above 350 K: max={float(jnp.max(T_arr))}"

        # Check convergence: variance over last 50 steps < variance over first 50
        var_first = jnp.var(T_arr[:50])
        var_last = jnp.var(T_arr[50:])
        assert var_last <= var_first + 1e-6, (
            f"T not converging: var_first={float(var_first):.4e}, var_last={float(var_last):.4e}"
        )

    def test_100_steps_W_bounded(self):
        state = make_state(T_soil=280.0, W_bucket=75.0)
        forcing = make_forcing(precip_total=1e-5)
        for _ in range(100):
            state, _, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)
        W = state.W_bucket.data
        assert jnp.all(W >= 0.0)
        assert jnp.all(W <= CONFIG.W_max)


# ===================================================================
# 1h  Wind speed floor
# ===================================================================


class Test1h_WindFloor:
    def test_fluxes_nonzero_calm_wind(self):
        """With u=v=0, U_min ensures nonzero fluxes."""
        state = make_state(T_soil=300.0, W_bucket=75.0)
        forcing = make_forcing(u_lowest=0.0, v_lowest=0.0, T_lowest=280.0, q_lowest=0.005)
        _, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)

        assert jnp.all(resp.shflx != 0.0), "shflx is zero despite U_min"
        assert jnp.all(resp.lhflx != 0.0), "lhflx is zero despite U_min"

    def test_stress_near_zero_calm_wind(self):
        """With u=v=0, surface stress should be near zero."""
        state = make_state(T_soil=280.0, W_bucket=75.0)
        forcing = make_forcing(u_lowest=0.0, v_lowest=0.0)
        _, resp, _ = step_land(state, forcing, CONFIG, U_min=1.0, dt=DT)

        assert jnp.allclose(resp.tau_x, 0.0, atol=1e-10), f"tau_x = {resp.tau_x}"
        assert jnp.allclose(resp.tau_y, 0.0, atol=1e-10), f"tau_y = {resp.tau_y}"
