"""Category 13: Unit Consistency & Cross-Component Checks."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.slab_land import step_land
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm import constants


SHAPE = (4,)
DIMS = ("ncol",)
DT = 3600.0


def _field(val, name="", units=""):
    return Field(data=jnp.full(SHAPE, val, jnp.float64), name=name, dims=DIMS, units=units)


def make_forcing(T_lowest=280.0, sw_down=300.0, lw_down=300.0):
    f = jnp.float64
    s = SHAPE
    return AtmToSurface(
        sw_down=jnp.full(s, sw_down, f), lw_down=jnp.full(s, lw_down, f),
        precip_total=jnp.zeros(s, f), precip_snow=jnp.zeros(s, f),
        T_lowest=jnp.full(s, T_lowest, f), q_lowest=jnp.full(s, 0.005, f),
        u_lowest=jnp.full(s, 5.0, f), v_lowest=jnp.full(s, 2.0, f),
        p_lowest=jnp.full(s, 95000.0, f), p_surface=jnp.full(s, 1e5, f),
        rho_lowest=jnp.full(s, 1.2, f), cos_zenith=jnp.full(s, 0.5, f),
        co2_ppmv=jnp.full(s, 415.0, f),
        has_radiation=jnp.ones(s, f), has_precipitation=jnp.ones(s, f),
    )


class Test13a_LandTendencyMagnitudes:
    def test_dT_magnitude(self):
        state = LandState(
            T_soil=_field(280.0, "T_soil", "K"),
            W_bucket=_field(50.0, "W_bucket", "kg/m2"),
            snow_depth=_field(0.0, "snow_depth", "kg/m2"),
            snow_age=_field(0.0, "snow_age", "s"),
        )
        cfg = LandConfig()
        new_state, resp, _ = step_land(state, make_forcing(), cfg, 1.0, DT)
        dT = jnp.abs(new_state.T_soil.data - state.T_soil.data) / DT
        assert jnp.all(dT < 0.01), f"dT/dt = {float(dT[0])}"
        assert jnp.all(jnp.abs(resp.shflx) < 500.0)
        assert jnp.all(jnp.abs(resp.lhflx) < 500.0)


class Test13b_IceTendencyMagnitudes:
    def test_dh_magnitude(self):
        state = SeaIceState(
            h_ice=_field(1.0, "h_ice", "m"),
            T_ice=_field(265.0, "T_ice", "K"),
            concentration=_field(0.9, "concentration", "1"),
        )
        cfg = SeaIceConfig()
        sst = jnp.full(SHAPE, cfg.T_freeze_ocean, jnp.float64)
        new_state, _ = step_sea_ice(
            state, make_forcing(T_lowest=260.0, sw_down=50.0, lw_down=200.0),
            sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE), cfg, 1.0, DT,
        )
        dh = jnp.abs(new_state.h_ice.data - state.h_ice.data) / DT
        assert jnp.all(dh < 1e-5), f"dh/dt = {float(dh[0])}"


class Test13d_ConstantsCrossCheck:
    def test_rho_water(self):
        assert abs(constants.rho_water - 1000.0) / 1000.0 < 0.01

    def test_rho_ice(self):
        assert abs(constants.rho_ice - 917.0) / 917.0 < 0.01

    def test_L_f(self):
        assert abs(constants.L_f - 3.337e5) / 3.337e5 < 0.01

    def test_sigma_sb(self):
        assert abs(constants.sigma_sb - 5.67e-8) / 5.67e-8 < 0.01

    def test_c_water(self):
        assert abs(constants.c_pw - 4218.0) / 4218.0 < 0.01


class Test13e_CouplingInterface:
    def test_land_T_surface_matches(self):
        state = LandState(
            T_soil=_field(285.0, "T_soil", "K"),
            W_bucket=_field(50.0, "W_bucket", "kg/m2"),
            snow_depth=_field(0.0, "snow_depth", "kg/m2"),
            snow_age=_field(0.0, "snow_age", "s"),
        )
        new_state, resp, _ = step_land(state, make_forcing(), LandConfig(), 1.0, DT)
        # T_surface in response should equal updated T_soil
        assert jnp.allclose(resp.T_surface, new_state.T_soil.data, atol=1e-10)


class Test13f_ConsistentSignConventions:
    def test_shflx_positive_up_all_tiles(self):
        """With warm surface, cold air: shflx > 0 for all tiles."""
        forcing = make_forcing(T_lowest=270.0, sw_down=200.0, lw_down=250.0)

        # Land
        land_state = LandState(
            T_soil=_field(290.0, "T_soil", "K"),
            W_bucket=_field(50.0, "W_bucket", "kg/m2"),
            snow_depth=_field(0.0, "snow_depth", "kg/m2"),
            snow_age=_field(0.0, "snow_age", "s"),
        )
        _, resp_land, _ = step_land(land_state, forcing, LandConfig(), 1.0, DT)

        # Lake
        lake_state = LakeState(
            T_epi=_field(290.0, "T_epi", "K"),
            T_hypo=_field(280.0, "T_hypo", "K"),
        )
        _, resp_lake = step_lake(lake_state, forcing, LakeConfig(), 1.0, DT)

        # All should have positive (upward) shflx with warm surface, cold air
        assert jnp.all(resp_land.shflx > 0), f"Land shflx: {resp_land.shflx}"
        assert jnp.all(resp_lake.shflx > 0), f"Lake shflx: {resp_lake.shflx}"

    def test_lw_up_positive_all_tiles(self):
        forcing = make_forcing()

        land_state = LandState(
            T_soil=_field(280.0), W_bucket=_field(50.0),
            snow_depth=_field(0.0), snow_age=_field(0.0),
        )
        _, resp_land, _ = step_land(land_state, forcing, LandConfig(), 1.0, DT)

        ice_state = SeaIceState(
            h_ice=_field(1.0), T_ice=_field(265.0), concentration=_field(0.9),
        )
        sst = jnp.full(SHAPE, constants.T_freeze_ocean, jnp.float64)
        _, resp_ice = step_sea_ice(
            ice_state, forcing, sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
            SeaIceConfig(), 1.0, DT,
        )

        lake_state = LakeState(T_epi=_field(285.0), T_hypo=_field(280.0))
        _, resp_lake = step_lake(lake_state, forcing, LakeConfig(), 1.0, DT)

        assert jnp.all(resp_land.lw_up > 0)
        assert jnp.all(resp_ice.lw_up > 0)
        assert jnp.all(resp_lake.lw_up > 0)
