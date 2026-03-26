"""Physics validation: surface models (land, ice, lake) physical consistency.

Category 8: Energy balance, flux signs, bucket hydrology, ice thermo bounds.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.coupler.coupling_fields import AtmToSurface


def make_forcing(shape, **overrides):
    ones = jnp.ones(shape)
    defaults = dict(
        sw_down=200.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=280.0, q_lowest=5e-3, u_lowest=5.0, v_lowest=2.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    defaults.update(overrides)
    return AtmToSurface(**{k: v * ones for k, v in defaults.items()})


# ============================================================================
# 8a  Slab land: temperature bounded
# ============================================================================

class TestSlabLandPhysics:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        self.step = step_land
        self.config = LandConfig()
        ncol = 16
        self.state = LandState(
            T_soil=Field(280.0 * jnp.ones(ncol), name="T_soil"),
            W_bucket=Field(50.0 * jnp.ones(ncol), name="W_bucket"),
            snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
            snow_age=Field(jnp.zeros(ncol), name="snow_age"),
        )
        self.forcing = make_forcing((ncol,))

    def test_temperature_bounded(self):
        state = self.state
        for _ in range(20):
            state, _, _ = self.step(state, self.forcing, self.config, U_min=1.0, dt=300.0)
        assert jnp.all(state.T_soil.data > 200), f"T_soil too cold: {state.T_soil.data.min()}"
        assert jnp.all(state.T_soil.data < 350), f"T_soil too warm: {state.T_soil.data.max()}"

    def test_shflx_sign(self):
        """Sensible heat flux sign should match T_sfc - T_air."""
        # Warm surface, cool air → SH upward (positive)
        warm_state = self.state._replace(T_soil=Field(310.0 * jnp.ones(16), name="T_soil"))
        cool_forcing = make_forcing((16,), T_lowest=270.0)
        _, resp, _ = self.step(warm_state, cool_forcing, self.config, U_min=1.0, dt=300.0)
        assert jnp.mean(resp.shflx) > 0, "SH should be positive when T_sfc > T_air"

    def test_bucket_bounded(self):
        """Bucket should stay in [0, W_max]."""
        state = self.state
        heavy_rain = make_forcing((16,), precip_total=1e-2)
        for _ in range(50):
            state, _, _ = self.step(state, heavy_rain, self.config, U_min=1.0, dt=300.0)
        assert jnp.all(state.W_bucket.data >= 0), "Bucket negative"
        assert jnp.all(state.W_bucket.data <= self.config.W_max + 1.0), "Bucket overflow"


# ============================================================================
# 8b  Sea ice: thermodynamic bounds
# ============================================================================

class TestSeaIcePhysics:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState

        self.step = step_sea_ice
        self.config = SeaIceConfig(dynamics="none")
        shape = (6, 4, 4)
        self.state = SeaIceState(
            h_ice=Field(1.0 * jnp.ones(shape), name="h_ice"),
            T_ice=Field(265.0 * jnp.ones(shape), name="T_ice"),
            concentration=Field(0.8 * jnp.ones(shape), name="concentration"),
        )
        self.forcing = make_forcing(shape, T_lowest=260.0, sw_down=100.0, lw_down=250.0)
        self.ocean_sst = 271.35 * jnp.ones(shape)
        self.zeros = jnp.zeros(shape)

    def test_ice_thickness_non_negative(self):
        state = self.state
        warm_forcing = make_forcing((6, 4, 4), T_lowest=280.0, sw_down=400.0, lw_down=350.0)
        for _ in range(20):
            state, _ = self.step(
                state, warm_forcing, self.ocean_sst, self.zeros, self.zeros,
                self.config, U_min=1.0, dt=3600.0,
            )
        assert jnp.all(state.h_ice.data >= 0), "Ice thickness negative"

    def test_concentration_bounded(self):
        state = self.state
        for _ in range(10):
            state, _ = self.step(
                state, self.forcing, self.ocean_sst, self.zeros, self.zeros,
                self.config, U_min=1.0, dt=3600.0,
            )
        assert jnp.all(state.concentration.data >= 0), "Concentration negative"
        assert jnp.all(state.concentration.data <= 1.0), "Concentration > 1"

    def test_T_ice_bounded(self):
        state = self.state
        for _ in range(10):
            state, _ = self.step(
                state, self.forcing, self.ocean_sst, self.zeros, self.zeros,
                self.config, U_min=1.0, dt=3600.0,
            )
        assert jnp.all(state.T_ice.data > 200), f"T_ice too cold: {state.T_ice.data.min()}"
        assert jnp.all(state.T_ice.data < 280), f"T_ice too warm: {state.T_ice.data.max()}"


# ============================================================================
# 8c  EOS physical sign convention
# ============================================================================

class TestEOSPhysics:

    def test_thermal_expansion(self):
        from legoesm.ocean.eos import wright_eos
        # Warmer water is lighter
        rho_cold = wright_eos(jnp.array(5.0), jnp.array(35.0), jnp.array(0.0))
        rho_warm = wright_eos(jnp.array(25.0), jnp.array(35.0), jnp.array(0.0))
        assert rho_warm < rho_cold, "Warm water should be lighter than cold water"

    def test_haline_contraction(self):
        from legoesm.ocean.eos import wright_eos
        # Saltier water is heavier
        rho_fresh = wright_eos(jnp.array(15.0), jnp.array(30.0), jnp.array(0.0))
        rho_salty = wright_eos(jnp.array(15.0), jnp.array(38.0), jnp.array(0.0))
        assert rho_salty > rho_fresh, "Salty water should be heavier than fresh water"

    def test_density_in_range(self):
        from legoesm.ocean.eos import wright_eos
        # Seawater density should be ~1020-1030 kg/m3 at surface
        rho = wright_eos(jnp.array(15.0), jnp.array(35.0), jnp.array(0.0))
        assert 1020 < rho < 1035, f"Surface density out of range: {rho}"
