"""Category 14: Integrated Surface Model Tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.slab_land import step_land
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import init_carbon_state
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


def make_forcing(**kw):
    f = jnp.float64
    s = SHAPE
    d = dict(
        sw_down=300.0, lw_down=300.0, T_lowest=280.0, q_lowest=0.008,
        u_lowest=5.0, v_lowest=2.0, p_lowest=95000.0, p_surface=1e5,
        rho_lowest=1.2, cos_zenith=0.5, co2_ppmv=415.0,
        precip_total=0.0, precip_snow=0.0,
    )
    d.update(kw)
    return AtmToSurface(
        sw_down=jnp.full(s, d["sw_down"], f), lw_down=jnp.full(s, d["lw_down"], f),
        precip_total=jnp.full(s, d["precip_total"], f),
        precip_snow=jnp.full(s, d["precip_snow"], f),
        T_lowest=jnp.full(s, d["T_lowest"], f),
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


class Test14a_LandPlusCarbon:
    def test_carbon_evolves(self):
        carbon_cfg = CarbonConfig(scheme="differland")
        cfg = LandConfig(carbon=carbon_cfg)
        state = LandState(
            T_soil=_field(285.0, "T_soil", "K"),
            W_bucket=_field(50.0, "W_bucket", "kg/m2"),
            snow_depth=_field(0.0, "snow_depth", "kg/m2"),
            snow_age=_field(0.0, "snow_age", "s"),
        )
        carbon_state = init_carbon_state(SHAPE, carbon_cfg)
        forcing = make_forcing()
        lat = jnp.full(SHAPE, 0.8)

        new_state, resp, new_carbon = step_land(
            state, forcing, cfg, 1.0, DT, lat=lat,
            carbon_state=carbon_state, doy=200.0,
        )
        assert new_carbon is not None
        # CO2 flux should be non-zero during daytime
        assert jnp.any(resp.co2_flux != 0.0)
        # All carbon pools should be finite
        for name in CarbonState._fields:
            assert jnp.all(jnp.isfinite(getattr(new_carbon, name)))


class Test14g_AllTilesSameForcing:
    def test_all_produce_valid_response(self):
        forcing = make_forcing()

        # Land
        land_state = LandState(
            T_soil=_field(280.0, "T_soil", "K"),
            W_bucket=_field(50.0, "W_bucket", "kg/m2"),
            snow_depth=_field(0.0, "snow_depth", "kg/m2"),
            snow_age=_field(0.0, "snow_age", "s"),
        )
        _, resp_land, _ = step_land(land_state, forcing, LandConfig(), 1.0, DT)

        # Ice
        ice_state = SeaIceState(
            h_ice=_field(1.0, "h_ice", "m"),
            T_ice=_field(265.0, "T_ice", "K"),
            concentration=_field(0.9, "concentration", "1"),
        )
        sst = jnp.full(SHAPE, 271.35, jnp.float64)
        _, resp_ice = step_sea_ice(
            ice_state, forcing, sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
            SeaIceConfig(), 1.0, DT,
        )

        # Lake
        lake_state = LakeState(T_epi=_field(285.0, "T_epi", "K"), T_hypo=_field(280.0, "T_hypo", "K"))
        _, resp_lake = step_lake(lake_state, forcing, LakeConfig(), 1.0, DT)

        # All responses should be finite
        for name in TileResponse._fields:
            assert jnp.all(jnp.isfinite(getattr(resp_land, name))), f"Land.{name} not finite"
            assert jnp.all(jnp.isfinite(getattr(resp_ice, name))), f"Ice.{name} not finite"
            assert jnp.all(jnp.isfinite(getattr(resp_lake, name))), f"Lake.{name} not finite"

        # T_sfc should differ between tiles
        assert not jnp.allclose(resp_land.T_sfc, resp_ice.T_sfc, atol=0.1)
        assert not jnp.allclose(resp_land.T_sfc, resp_lake.T_sfc, atol=0.1)

        # Albedo should differ
        assert not jnp.allclose(resp_land.albedo, resp_ice.albedo, atol=0.01)
