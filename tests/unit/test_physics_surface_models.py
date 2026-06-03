"""Category 8: Land, Sea Ice, and Lake -- Physical Consistency.

Tests energy balance, flux signs, bucket hydrology, ice thickness bounds,
concentration bounds, lake temperature evolution, and EOS sign conventions.
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
# 8a  Slab land
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
        """T_soil stays in [200, 350] K after many steps."""
        state = self.state
        for _ in range(20):
            state, _, _ = self.step(state, self.forcing, self.config, U_min=1.0, dt=300.0)
        assert jnp.all(state.T_soil.data > 200), f"T_soil too cold"
        assert jnp.all(state.T_soil.data < 350), f"T_soil too warm"

    def test_shflx_sign(self):
        """SH > 0 when T_sfc > T_air (upward heat flux)."""
        warm_state = self.state._replace(T_soil=Field(310.0 * jnp.ones(16), name="T_soil"))
        cool_forcing = make_forcing((16,), T_lowest=270.0)
        _, resp, _ = self.step(warm_state, cool_forcing, self.config, U_min=1.0, dt=300.0)
        assert jnp.mean(resp.shflx) > 0, "SH should be positive when T_sfc > T_air"

    def test_bucket_bounded(self):
        """Bucket stays in [0, W_max]."""
        state = self.state
        heavy_rain = make_forcing((16,), precip_total=1e-2)
        for _ in range(50):
            state, _, _ = self.step(state, heavy_rain, self.config, U_min=1.0, dt=300.0)
        assert jnp.all(state.W_bucket.data >= 0), "Bucket negative"
        assert jnp.all(state.W_bucket.data <= self.config.W_max + 1.0), "Bucket overflow"

    def test_evaporation_limited_by_bucket(self):
        """With empty bucket, evaporation should be limited."""
        dry_state = self.state._replace(
            W_bucket=Field(jnp.zeros(16), name="W_bucket"),
        )
        dry_forcing = make_forcing((16,), precip_total=0.0, T_lowest=300.0)
        state2, resp, _ = self.step(dry_state, dry_forcing, self.config, U_min=1.0, dt=300.0)
        # Bucket should not go negative
        assert jnp.all(state2.W_bucket.data >= -1e-10), "Bucket went negative"

    def test_all_outputs_finite(self):
        """All output fields should be finite."""
        state, resp, _ = self.step(self.state, self.forcing, self.config, U_min=1.0, dt=300.0)
        assert jnp.all(jnp.isfinite(state.T_soil.data)), "T_soil NaN"
        assert jnp.all(jnp.isfinite(state.W_bucket.data)), "W_bucket NaN"
        assert jnp.all(jnp.isfinite(resp.shflx)), "SH NaN"
        assert jnp.all(jnp.isfinite(resp.lhflx)), "LH NaN"


# ============================================================================
# 8f  Sea ice
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
        """h_ice >= 0 even under warm forcing."""
        state = self.state
        warm_forcing = make_forcing((6, 4, 4), T_lowest=280.0, sw_down=400.0, lw_down=350.0)
        for _ in range(20):
            state, _ = self.step(
                state, warm_forcing, self.ocean_sst, self.zeros, self.zeros,
                self.config, U_min=1.0, dt=3600.0,
            )
        assert jnp.all(state.h_ice.data >= 0), "Ice thickness negative"

    def test_concentration_bounded(self):
        """Ice concentration in [0, 1]."""
        state = self.state
        for _ in range(10):
            state, _ = self.step(
                state, self.forcing, self.ocean_sst, self.zeros, self.zeros,
                self.config, U_min=1.0, dt=3600.0,
            )
        assert jnp.all(state.concentration.data >= 0), "Concentration negative"
        assert jnp.all(state.concentration.data <= 1.0), "Concentration > 1"

    def test_T_ice_bounded(self):
        """Ice temperature should stay in physically reasonable range."""
        state = self.state
        for _ in range(10):
            state, _ = self.step(
                state, self.forcing, self.ocean_sst, self.zeros, self.zeros,
                self.config, U_min=1.0, dt=3600.0,
            )
        assert jnp.all(state.T_ice.data > 200), "T_ice too cold"
        assert jnp.all(state.T_ice.data < 280), "T_ice too warm"

    def test_ice_grows_in_cold(self):
        """In very cold conditions, ice should grow (dh > 0)."""
        cold_forcing = make_forcing((6, 4, 4), T_lowest=240.0, sw_down=0.0, lw_down=150.0)
        cold_sst = 271.0 * jnp.ones((6, 4, 4))
        state = self.state._replace(
            h_ice=Field(0.5 * jnp.ones((6, 4, 4)), name="h_ice"),
        )
        state2, _ = self.step(
            state, cold_forcing, cold_sst, self.zeros, self.zeros,
            self.config, U_min=1.0, dt=86400.0,  # 1 day
        )
        # Ice VOLUME (h*conc) should grow or stay the same.  Under the
        # volume-based V=h*A update (#28) the MEAN thickness h can drop as the
        # refreezing lead averages in thin new ice, so assert on the conserved
        # volume rather than the (misleading) mean thickness.
        vol_before = float(jnp.mean(state.h_ice.data * state.concentration.data))
        vol_after = float(jnp.mean(state2.h_ice.data * state2.concentration.data))
        assert vol_after >= vol_before - 0.01, (
            f"Ice volume shrank in cold conditions: {vol_before:.3f} -> {vol_after:.3f}"
        )

    def test_all_outputs_finite(self):
        state, resp = self.step(
            self.state, self.forcing, self.ocean_sst, self.zeros, self.zeros,
            self.config, U_min=1.0, dt=3600.0,
        )
        assert jnp.all(jnp.isfinite(state.h_ice.data)), "h_ice NaN"
        assert jnp.all(jnp.isfinite(state.T_ice.data)), "T_ice NaN"
        assert jnp.all(jnp.isfinite(resp.shflx)), "SH NaN"


# ============================================================================
# 8i  Two-layer lake
# ============================================================================

class TestLakePhysics:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.coupler.lake.two_layer_lake import step_lake
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.coupler.lake.state import LakeState

        self.step = step_lake
        self.config = LakeConfig()
        ncol = 16
        self.state = LakeState(
            T_epi=Field(290.0 * jnp.ones(ncol), name="T_epi"),
            T_hypo=Field(278.0 * jnp.ones(ncol), name="T_hypo"),
        )
        self.forcing = make_forcing((ncol,), T_lowest=285.0)

    def test_temperature_bounded(self):
        """Lake temperatures stay in [250, 320] K."""
        state = self.state
        for _ in range(20):
            state, _ = self.step(state, self.forcing, self.config, U_min=1.0, dt=3600.0)
        assert jnp.all(state.T_epi.data > 250), "T_epi too cold"
        assert jnp.all(state.T_epi.data < 320), "T_epi too warm"
        assert jnp.all(state.T_hypo.data > 250), "T_hypo too cold"
        assert jnp.all(state.T_hypo.data < 320), "T_hypo too warm"

    def test_surface_cooling_cools_epi(self):
        """Strong cooling should reduce epilimnion temperature."""
        cold_forcing = make_forcing((16,), T_lowest=260.0, sw_down=0.0, lw_down=150.0)
        state, _ = self.step(self.state, cold_forcing, self.config, U_min=1.0, dt=3600.0)
        assert float(jnp.mean(state.T_epi.data)) < 290.0, (
            "Epilimnion did not cool under cold forcing"
        )

    def test_all_outputs_finite(self):
        state, resp = self.step(self.state, self.forcing, self.config, U_min=1.0, dt=3600.0)
        assert jnp.all(jnp.isfinite(state.T_epi.data)), "T_epi NaN"
        assert jnp.all(jnp.isfinite(state.T_hypo.data)), "T_hypo NaN"
        assert jnp.all(jnp.isfinite(resp.shflx)), "SH NaN"


# ============================================================================
# EOS physical sign convention
# ============================================================================

class TestEOSPhysics:

    def test_thermal_expansion(self):
        from legoesm.ocean.eos import wright_eos
        rho_cold = wright_eos(jnp.array(5.0), jnp.array(35.0), jnp.array(0.0))
        rho_warm = wright_eos(jnp.array(25.0), jnp.array(35.0), jnp.array(0.0))
        assert rho_warm < rho_cold, "Warm water should be lighter"

    def test_haline_contraction(self):
        from legoesm.ocean.eos import wright_eos
        rho_fresh = wright_eos(jnp.array(15.0), jnp.array(30.0), jnp.array(0.0))
        rho_salty = wright_eos(jnp.array(15.0), jnp.array(38.0), jnp.array(0.0))
        assert rho_salty > rho_fresh, "Salty water should be heavier"

    def test_density_in_range(self):
        from legoesm.ocean.eos import wright_eos
        rho = wright_eos(jnp.array(15.0), jnp.array(35.0), jnp.array(0.0))
        assert 1020 < rho < 1035, f"Surface density out of range: {rho}"
