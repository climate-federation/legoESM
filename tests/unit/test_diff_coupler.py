"""Differentiability tests for coupler components.

Categories:
  6a) Bulk flux (MOST iteration)
  6b) Tile blending
  6c) Full coupler step
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


# ============================================================================
# 6a  Bulk flux (MOST iteration)
# ============================================================================

class TestBulkFluxGrad:

    @pytest.mark.parametrize("scheme", ["coare3", "large_yeager"])
    @pytest.mark.parametrize("n_iter", [1, 5])
    def test_grad_wrt_T_sfc(self, scheme, n_iter):
        from legoesm.coupler.bulk_flux import compute_most_fluxes

        ncol = 32
        T_sfc = 300.0 * jnp.ones(ncol)
        T_atm = 295.0 * jnp.ones(ncol)
        u_rel = 5.0 * jnp.ones(ncol)
        v_rel = 2.0 * jnp.ones(ncol)
        q_atm = 5e-3 * jnp.ones(ncol)
        q_sfc = 8e-3 * jnp.ones(ncol)
        rho = 1.2 * jnp.ones(ncol)

        def loss(T_sfc):
            tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho,
                scheme=scheme, n_iter=n_iter,
            )
            return jnp.sum(shflx ** 2)

        grad = jax.grad(loss)(T_sfc)
        assert_gradient_ok(grad, f"MOST({scheme}, n_iter={n_iter}) w.r.t. T_sfc")


# ============================================================================
# 6b  Tile blending
# ============================================================================

class TestTileBlendingGrad:

    def test_grad_wrt_ice_concentration(self):
        from legoesm.coupler.tile_fractions import compute_tile_fractions, blend_tiles
        from legoesm.coupler.config import TileConfig
        from legoesm.coupler.coupling_fields import TileResponse

        n = 4
        shape = (6, n, n)
        ones = jnp.ones(shape)
        tile_config = TileConfig(
            f_land=0.3 * ones,
            f_lake=0.1 * ones,
        )

        def make_tile_response(T_sfc):
            return TileResponse(
                T_surface=T_sfc,
                albedo=0.1 * ones,
                emissivity=0.97 * ones,
                z0=1e-4 * ones,
                q_surface=5e-3 * ones,
                shflx=20.0 * ones,
                lhflx=10.0 * ones,
                tau_x=0.1 * ones,
                tau_y=0.05 * ones,
                lw_up=350.0 * ones,
                u_ocean_sfc=jnp.zeros(shape),
                v_ocean_sfc=jnp.zeros(shape),
                co2_flux=jnp.zeros(shape),
            )

        ocean_resp = make_tile_response(295.0 * ones)
        ice_resp = make_tile_response(265.0 * ones)
        land_resp = make_tile_response(290.0 * ones)
        lake_resp = make_tile_response(285.0 * ones)

        def loss(ice_conc):
            fracs = compute_tile_fractions(tile_config, ice_conc)
            blended = blend_tiles(ocean_resp, ice_resp, land_resp, lake_resp, fracs)
            return jnp.sum(blended.T_surface ** 2)

        ice_conc = 0.5 * ones
        grad = jax.grad(loss)(ice_conc)
        assert_gradient_ok(grad, "Tile blending w.r.t. ice_concentration")


# ============================================================================
# 6c  Full coupler step
# ============================================================================

class TestFullCouplerGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.coupler.coupler import make_coupler
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.coupler.coupling_fields import AtmToSurface
        from legoesm.coupler.accumulator import reset_accumulator
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.coupler.lake.state import LakeState
        from legoesm.coupler.coupler import SurfaceState

        n = 4
        shape = (6, n, n)
        ones = jnp.ones(shape)

        coupler_config = CouplerConfig()
        land_config = LandConfig()
        ice_config = SeaIceConfig(dynamics="none")
        lake_config = LakeConfig()

        self.step_surface = make_coupler(coupler_config, land_config, ice_config, lake_config)

        self.sfc_state = SurfaceState(
            land=LandState(
                T_soil=Field(280.0 * ones, name="T_soil"),
                W_bucket=Field(50.0 * ones, name="W_bucket"),
                snow_depth=Field(jnp.zeros(shape), name="snow_depth"),
                snow_age=Field(jnp.zeros(shape), name="snow_age"),
            ),
            ice=SeaIceState(
                h_ice=Field(1.0 * ones, name="h_ice"),
                T_ice=Field(265.0 * ones, name="T_ice"),
                concentration=Field(0.5 * ones, name="concentration"),
            ),
            lake=LakeState(
                T_epi=Field(285.0 * ones, name="T_epi"),
                T_hypo=Field(280.0 * ones, name="T_hypo"),
            ),
            accumulator=reset_accumulator(shape),
        )

        self.tile_config = TileConfig(
            f_land=0.3 * ones,
            f_lake=0.1 * ones,
        )
        self.ocean_sst = 295.0 * ones
        self.forcing = AtmToSurface(
            sw_down=200.0 * ones,
            lw_down=300.0 * ones,
            precip_total=1e-5 * ones,
            precip_snow=0.0 * ones,
            T_lowest=280.0 * ones,
            q_lowest=5e-3 * ones,
            u_lowest=5.0 * ones,
            v_lowest=2.0 * ones,
            p_lowest=1e5 * ones,
            p_surface=1.013e5 * ones,
            rho_lowest=1.2 * ones,
            cos_zenith=0.7 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones,
            has_precipitation=1.0 * ones,
        )
        self.dt = 300.0

    def test_grad_wrt_sst(self):
        step_fn = self.step_surface
        sfc_state = self.sfc_state
        forcing = self.forcing
        tile_config = self.tile_config

        def loss(sst):
            _, sfc_to_atm = step_fn(
                sfc_state, forcing, tile_config, sst,
                jnp.zeros_like(sst), jnp.zeros_like(sst),
                dt=self.dt,
            )
            return jnp.sum(sfc_to_atm.T_surface ** 2)

        grad = jax.grad(loss)(self.ocean_sst)
        assert_gradient_ok(grad, "Full coupler w.r.t. SST")
