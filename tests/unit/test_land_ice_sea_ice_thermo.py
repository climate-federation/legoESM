"""Category 8: Sea Ice Thermodynamics -- Slab Model."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm import constants


SHAPE = (4,)
DIMS = ("ncol",)
CONFIG = SeaIceConfig()


def _field(val, name="", units=""):
    return Field(data=jnp.full(SHAPE, val, jnp.float64), name=name, dims=DIMS, units=units)


def make_ice_state(h=1.0, T_ice=265.0, conc=0.9):
    return SeaIceState(
        h_ice=_field(h, "h_ice", "m"),
        T_ice=_field(T_ice, "T_ice", "K"),
        concentration=_field(conc, "concentration", "1"),
    )


def make_forcing(sw_down=100.0, lw_down=200.0, T_lowest=260.0, **kw):
    f = jnp.float64
    s = SHAPE
    defaults = dict(
        q_lowest=0.002, u_lowest=5.0, v_lowest=2.0,
        p_lowest=95000.0, p_surface=1e5, rho_lowest=1.3,
        cos_zenith=0.3, co2_ppmv=415.0,
        precip_total=0.0, precip_snow=0.0,
    )
    defaults.update(kw)
    d = defaults
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


DT = 3600.0
OCEAN_SST = jnp.full(SHAPE, CONFIG.T_freeze_ocean, jnp.float64)
OCEAN_U = jnp.zeros(SHAPE, jnp.float64)
OCEAN_V = jnp.zeros(SHAPE, jnp.float64)


class Test8a_Smoke:
    def test_returns_finite(self):
        state = make_ice_state()
        forcing = make_forcing()
        new_state, resp = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        for name in SeaIceState._fields:
            assert jnp.all(jnp.isfinite(getattr(new_state, name).data)), f"{name} non-finite"
        for name in TileResponse._fields:
            assert jnp.all(jnp.isfinite(getattr(resp, name))), f"TileResponse.{name} non-finite"


class Test8b_IceGrowthCold:
    def test_ice_grows(self):
        """Very cold air => ice should thicken."""
        state = make_ice_state(h=1.0, T_ice=260.0, conc=0.9)
        forcing = make_forcing(T_lowest=240.0, sw_down=0.0, lw_down=150.0)
        new_state, _ = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(new_state.h_ice.data >= state.h_ice.data)


class Test8c_IceMeltWarm:
    def test_ice_melts(self):
        """Warm air + warm ocean => ice thins."""
        state = make_ice_state(h=1.0, T_ice=270.0, conc=0.9)
        warm_sst = jnp.full(SHAPE, 275.0, jnp.float64)
        forcing = make_forcing(T_lowest=280.0, sw_down=300.0, lw_down=300.0)
        new_state, _ = step_sea_ice(state, forcing, warm_sst, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(new_state.h_ice.data <= state.h_ice.data)


class Test8d_CompleteMelt:
    def test_h_never_negative(self):
        state = make_ice_state(h=0.01, T_ice=270.0, conc=0.5)
        warm_sst = jnp.full(SHAPE, 280.0, jnp.float64)
        forcing = make_forcing(T_lowest=290.0, sw_down=500.0, lw_down=350.0)
        for _ in range(10):
            state, _ = step_sea_ice(state, forcing, warm_sst, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(state.h_ice.data >= 0.0)


class Test8f_TemperatureBounds:
    def test_T_ice_bounded(self):
        state = make_ice_state(h=1.0, T_ice=250.0)
        forcing = make_forcing(T_lowest=200.0, sw_down=0.0, lw_down=100.0)
        new_state, _ = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(new_state.T_ice.data >= CONFIG.T_ice_min)
        assert jnp.all(new_state.T_ice.data <= CONFIG.T_freeze_ocean)


class Test8g_ConcentrationBounds:
    def test_conc_bounded(self):
        state = make_ice_state(h=1.0, T_ice=265.0, conc=0.95)
        forcing = make_forcing()
        for _ in range(20):
            state, _ = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(state.concentration.data >= 0.0)
        assert jnp.all(state.concentration.data <= 1.0)


class Test8i_StefanBoltzmann:
    def test_lw_up_matches(self):
        state = make_ice_state(h=2.0, T_ice=260.0)
        forcing = make_forcing()
        new_state, resp = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        T_sfc = new_state.T_ice.data
        expected = CONFIG.emissivity_ice * constants.sigma_sb * T_sfc ** 4
        rel_err = jnp.abs(resp.lw_up - expected) / expected
        assert jnp.all(rel_err < 0.001)


class Test8j_FluxSigns:
    def test_cold_ice_warm_air(self):
        """Cold ice, warm air => shflx < 0 (heat into ice)."""
        state = make_ice_state(h=2.0, T_ice=250.0)
        forcing = make_forcing(T_lowest=270.0)
        _, resp = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(resp.shflx < 0)

    def test_lw_up_always_positive(self):
        state = make_ice_state()
        forcing = make_forcing()
        _, resp = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(resp.lw_up > 0)


class Test8k_MultiStepStability:
    def test_200_steps_stable(self):
        state = make_ice_state(h=1.0, T_ice=260.0, conc=0.8)
        forcing = make_forcing()
        for _ in range(200):
            state, _ = step_sea_ice(state, forcing, OCEAN_SST, OCEAN_U, OCEAN_V, CONFIG, 1.0, DT)
        assert jnp.all(jnp.isfinite(state.T_ice.data))
        assert jnp.all(jnp.isfinite(state.h_ice.data))
        assert jnp.all(state.h_ice.data >= 0.0)
