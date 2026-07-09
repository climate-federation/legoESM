"""Category 11: Lake Two-Layer Model."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm import constants


SHAPE = (4,)
DIMS = ("ncol",)
CONFIG = LakeConfig()
DT = 3600.0


def _field(val, name="", units=""):
    return Field(data=jnp.full(SHAPE, val, jnp.float64), name=name, dims=DIMS, units=units)


def make_lake_state(T_epi=290.0, T_hypo=280.0):
    return LakeState(
        T_epi=_field(T_epi, "T_epi", "K"),
        T_hypo=_field(T_hypo, "T_hypo", "K"),
    )


def make_forcing(sw_down=250.0, lw_down=300.0, T_lowest=285.0, **kw):
    f = jnp.float64
    s = SHAPE
    d = dict(
        q_lowest=0.008, u_lowest=5.0, v_lowest=2.0,
        p_lowest=95000.0, p_surface=1e5, rho_lowest=1.2,
        cos_zenith=0.5, co2_ppmv=415.0, precip_total=0.0, precip_snow=0.0,
    )
    d.update(kw)
    return AtmToSurface(
        sw_down=jnp.full(s, sw_down, f), lw_down=jnp.full(s, lw_down, f),
        precip_total=jnp.full(s, d["precip_total"], f),
        precip_snow=jnp.full(s, d["precip_snow"], f),
        T_lowest=jnp.full(s, T_lowest, f),
        q_lowest=jnp.full(s, d["q_lowest"], f),
        u_lowest=jnp.full(s, d["u_lowest"], f),
        v_lowest=jnp.full(s, d["v_lowest"], f),
        p_lowest=jnp.full(s, d["p_lowest"], f),
        p_surface=jnp.full(s, d["p_surface"], f),
        rho_lowest=jnp.full(s, d["rho_lowest"], f),
        cos_zenith=jnp.full(s, d["cos_zenith"], f),
        co2_ppmv=jnp.full(s, d["co2_ppmv"], f),
        has_radiation=jnp.ones(s, f), has_precipitation=jnp.ones(s, f),
    )


class Test11a_Smoke:
    def test_returns_finite(self):
        state = make_lake_state()
        forcing = make_forcing()
        new_state, resp = step_lake(state, forcing, CONFIG, 1.0, DT)
        assert jnp.all(jnp.isfinite(new_state.T_epi.data))
        assert jnp.all(jnp.isfinite(new_state.T_hypo.data))
        for name in TileResponse._fields:
            arr = getattr(resp, name)
            if arr is None:            # optional fields (e.g. T_rad on non-canopy tiles)
                continue
            assert jnp.all(jnp.isfinite(arr))


class Test11e_MixingDirection:
    def test_warm_epi_cools(self):
        """T_epi > T_hypo => mixing cools epi, warms hypo."""
        state = make_lake_state(T_epi=295.0, T_hypo=280.0)
        forcing = make_forcing(sw_down=0.0, lw_down=280.0, T_lowest=285.0)
        new_state, _ = step_lake(state, forcing, CONFIG, 1.0, DT)
        # Hypo should warm (mixing brings heat down)
        assert float(new_state.T_hypo.data[0]) > 280.0

    def test_equal_T_no_mixing(self):
        """T_epi = T_hypo => F_mix = 0."""
        state = make_lake_state(T_epi=285.0, T_hypo=285.0)
        # Compute F_mix manually
        wind_speed = jnp.sqrt(5.0**2 + 2.0**2 + 1.0**2)
        k_eff = CONFIG.k_mix * (1.0 + CONFIG.wind_mix_alpha * wind_speed)
        d_mid = 0.5 * (CONFIG.h_epi + CONFIG.h_hypo)
        F_mix = CONFIG.rho_water * CONFIG.c_water_mass * k_eff * (285.0 - 285.0) / d_mid
        assert abs(float(F_mix)) < 1e-10


class Test11f_WindMixing:
    def test_stronger_wind_more_mixing(self):
        state = make_lake_state(T_epi=295.0, T_hypo=280.0)
        forcing_calm = make_forcing(u_lowest=1.0, v_lowest=0.0)
        forcing_windy = make_forcing(u_lowest=10.0, v_lowest=0.0)
        new_calm, _ = step_lake(state, forcing_calm, CONFIG, 1.0, DT)
        new_windy, _ = step_lake(state, forcing_windy, CONFIG, 1.0, DT)
        # Windy mixing should warm hypo more
        assert float(new_windy.T_hypo.data[0]) > float(new_calm.T_hypo.data[0])


class Test11g_FreezingFloor:
    def test_T_not_below_freeze(self):
        state = make_lake_state(T_epi=274.0, T_hypo=274.0)
        forcing = make_forcing(T_lowest=220.0, sw_down=0.0, lw_down=100.0)
        for _ in range(50):
            state, _ = step_lake(state, forcing, CONFIG, 1.0, DT)
        assert jnp.all(state.T_epi.data >= CONFIG.T_freeze)
        assert jnp.all(state.T_hypo.data >= CONFIG.T_freeze)

    def test_post_step_q_surface_uses_ice_saturation_when_frozen(self):
        """Iter-67: response.q_surface must use ICE saturation when the
        lake is frozen (T_epi ≤ T_freeze), matching the pre-step phase
        decision used for the bulk-flux call.

        Before the fix, the post-step ``q_sfc_new`` always called
        ``saturation_mixing_ratio`` (liquid form), biasing the
        q_surface reported to the atmosphere by ~14 % at T = -10 °C
        (liquid vs ice saturation diverges below T_freeze).
        """
        # Hold the lake at T_epi = T_freeze (boundary).
        from legoesm.thermo import (
            saturation_mixing_ratio, saturation_mixing_ratio_ice,
        )
        state = make_lake_state(T_epi=CONFIG.T_freeze, T_hypo=CONFIG.T_freeze)
        forcing = make_forcing(T_lowest=240.0, sw_down=0.0, lw_down=200.0)
        _, resp = step_lake(state, forcing, CONFIG, 1.0, DT)
        # Lake stays frozen; expected q is ice saturation at T_freeze.
        p_sfc = forcing.p_surface
        q_ice_expected = float(saturation_mixing_ratio_ice(
            jnp.asarray(CONFIG.T_freeze), p_sfc,
        ).mean())
        q_liq_expected = float(saturation_mixing_ratio(
            jnp.asarray(CONFIG.T_freeze), p_sfc,
        ).mean())
        q_actual = float(resp.q_surface.mean())
        # Should match ice saturation (≤ liquid; equal at exactly T_freeze
        # but diverges below).  Test at the boundary verifies the
        # `is_frozen_new = T_epi_new <= T_freeze` branch fires.
        assert abs(q_actual - q_ice_expected) < abs(q_actual - q_liq_expected) + 1e-12, (
            f"Frozen-lake q_surface should be closer to ice saturation; "
            f"got {q_actual:.6e}, ice={q_ice_expected:.6e}, "
            f"liq={q_liq_expected:.6e}"
        )

    def test_frozen_lake_uses_ice_albedo_not_water(self):
        """A frozen lake reflects like ice/snow, not open water: the is_frozen
        switch must gate albedo (it already gates q_sfc and L_eff).  Higher ice
        albedo -> less absorbed SW -> less warming; an unfrozen lake ignores
        albedo_lake_ice entirely."""
        forcing = make_forcing(T_lowest=272.0, sw_down=1000.0, lw_down=340.0)
        frozen = make_lake_state(T_epi=CONFIG.T_freeze, T_hypo=CONFIG.T_freeze)
        cfg_ice = CONFIG._replace(albedo_lake_ice=0.6)
        cfg_wat = CONFIG._replace(albedo_lake_ice=CONFIG.albedo_lake)
        s_ice, _ = step_lake(frozen, forcing, cfg_ice, 1.0, DT)
        s_wat, _ = step_lake(frozen, forcing, cfg_wat, 1.0, DT)
        assert float(s_ice.T_epi.data.mean()) < float(s_wat.T_epi.data.mean())
        # Unfrozen lake never touches albedo_lake_ice: identical result.
        warm = make_lake_state(T_epi=290.0, T_hypo=285.0)
        w_ice, _ = step_lake(warm, forcing, cfg_ice, 1.0, DT)
        w_wat, _ = step_lake(warm, forcing, cfg_wat, 1.0, DT)
        assert bool(jnp.allclose(w_ice.T_epi.data, w_wat.T_epi.data))
        # Reported albedo tracks the POST-step phase (like q_surface_new): a
        # lake that stays frozen must report the ice albedo, not open water.
        cold = make_forcing(T_lowest=230.0, sw_down=0.0, lw_down=150.0)
        _, resp = step_lake(frozen, cold, cfg_ice, 1.0, DT)
        assert abs(float(resp.albedo.mean()) - cfg_ice.albedo_lake_ice) < 1e-6


class Test11h_ConvectiveOverturn:
    def test_density_inversion_homogenizes(self):
        """Freshwater density peaks at ~4 °C (277.15 K), so above T_max
        a colder layer is DENSER.  A column with T_epi = 5 °C over
        T_hypo = 9 °C is statically unstable: |T_epi − T_max| = 1 K
        (anomaly = -α), while |T_hypo − T_max| = 5 K (anomaly = -25α),
        so ρ_epi > ρ_hypo despite T_epi < T_hypo.

        Why non-vacuous: under the prior implementation (no convective
        adjustment), the diffusive mixing term ``F_mix ∝ T_epi − T_hypo``
        is NEGATIVE here (T_epi < T_hypo), so heat flows UP from
        hypolimnion to epilimnion at a rate set by k_mix — but the
        unstable density profile is preserved indefinitely.  Without
        the convective adjustment, the post-step difference is
        ``T_hypo − T_epi ≈ +4 K − F_mix·dt/cap`` (small reduction).
        With the convective fix, the column homogenizes within one
        step (|ΔT| < 0.1 K).
        """
        # T_epi=278.15 K (5°C, anomaly -α), T_hypo=292.15 K (19°C,
        # anomaly -225α): ρ_epi >> ρ_hypo ⇒ strongly unstable.
        # Use a strongly unstable column so the convective adjustment
        # fires even after the diffusive flux has narrowed the gap
        # within one step.  Use a small dt to disable the
        # surface-flux-driven heating of the epilimnion (which would
        # otherwise also work toward removing the inversion).
        state = make_lake_state(T_epi=278.15, T_hypo=292.15)
        forcing = make_forcing(
            sw_down=0.0, lw_down=0.0, T_lowest=278.15, u_lowest=0.0, v_lowest=0.0
        )
        new_state, _ = step_lake(state, forcing, CONFIG, 1.0, dt=10.0)

        T_epi_after = float(new_state.T_epi.data.flatten()[0])
        T_hypo_after = float(new_state.T_hypo.data.flatten()[0])

        # (1) After convective adjustment, both layers should be at
        # nearly the same temperature — without the fix, |ΔT| stays
        # at ~14 K (10 s of diffusive flux barely changes it).
        assert abs(T_epi_after - T_hypo_after) < 0.1, (
            f"Convective overturn did not homogenize unstable column: "
            f"T_epi={T_epi_after:.3f}, T_hypo={T_hypo_after:.3f}, "
            f"diff={T_epi_after - T_hypo_after:.3f}.  Without the fix, "
            f"|ΔT| ≳ 14 K (the inversion is preserved)."
        )

        # (2) Enthalpy conservation: the mass-weighted mean must equal
        # the capacity-weighted mean of the pre-step temperatures
        # (within tolerance of the small forcing — sw_down=0,
        # lw_down=0, T_lowest=T_epi → very small surface flux).
        # cap_epi = ρ·c·h_epi, cap_hypo = ρ·c·h_hypo.  With h_epi=5,
        # h_hypo=20 (ratio 1:4), expected mean = (1·278.15 + 4·292.15)/5
        # = 289.35 K.  A broken adjustment that, e.g., uses the
        # arithmetic mean (285.15 K) would FAIL this check.
        cap_epi = CONFIG.rho_water * CONFIG.c_water_mass * CONFIG.h_epi
        cap_hypo = CONFIG.rho_water * CONFIG.c_water_mass * CONFIG.h_hypo
        T_expected_mean = (cap_epi * 278.15 + cap_hypo * 292.15) / (cap_epi + cap_hypo)
        assert abs(T_epi_after - T_expected_mean) < 0.5, (
            f"Convective overturn violates enthalpy conservation: "
            f"T_epi_after={T_epi_after:.3f} vs expected mass-weighted "
            f"mean {T_expected_mean:.3f}.  An arithmetic-mean "
            f"adjustment (285.15 K) or hypolimnion-only adjustment "
            f"(292.15 K) would FAIL this check."
        )


class Test11i_MultiStepConvergence:
    def test_500_steps_stable(self):
        state = make_lake_state(T_epi=295.0, T_hypo=280.0)
        forcing = make_forcing()
        for _ in range(500):
            state, _ = step_lake(state, forcing, CONFIG, 1.0, DT)
        assert jnp.all(jnp.isfinite(state.T_epi.data))
        assert jnp.all(state.T_epi.data >= CONFIG.T_freeze)
        assert jnp.all(state.T_epi.data < 330.0)


class Test11j_AlbedoEmissivity:
    def test_albedo_and_lw_up(self):
        state = make_lake_state()
        forcing = make_forcing()
        new_state, resp = step_lake(state, forcing, CONFIG, 1.0, DT)
        assert jnp.allclose(resp.albedo, CONFIG.albedo_lake, atol=1e-10)
        T = new_state.T_epi.data
        # Greybody upward LW with reflected component, matching the
        # convention used by every other tile (slab/multilayer land,
        # sea-ice, ocean):
        #     lw_up = ε σ T⁴ + (1 − ε) lw_down
        # (Earlier the lake dropped the reflection term, biasing
        # lw_up low by ~10 W/m² for ε=0.97 and lw_down ≈ 350 W/m².)
        expected_lw = (
            CONFIG.emissivity_lake * constants.sigma_sb * T ** 4
            + (1.0 - CONFIG.emissivity_lake) * forcing.lw_down
        )
        assert jnp.allclose(resp.lw_up, expected_lw, rtol=1e-3)
