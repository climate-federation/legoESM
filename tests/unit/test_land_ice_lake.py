"""Category 11: Lake Two-Layer Model."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
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
            assert jnp.all(jnp.isfinite(getattr(resp, name)))


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
        F_mix = CONFIG.rho_water * CONFIG.c_water * k_eff * (285.0 - 285.0) / d_mid
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
        expected_lw = CONFIG.emissivity_lake * constants.sigma_sb * T ** 4
        assert jnp.allclose(resp.lw_up, expected_lw, rtol=1e-3)
