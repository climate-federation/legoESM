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
        from legoesm.core.coupling_fields import TileResponse

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
                freshwater_flux=jnp.zeros(shape),
                ocean_heat_extraction=jnp.zeros(shape),
                ocean_stress_x=jnp.zeros(shape),
                ocean_stress_y=jnp.zeros(shape),
                surface_mass_flux=jnp.zeros(shape),
                salt_flux=jnp.zeros(shape),
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

    def test_grad_wrt_tile_fraction_finite_at_pure_ocean(self):
        """d/d(f_land) must be finite at pure-ocean cells (f_land=f_lake=0).

        Regression: the static renormalization ``where(total>1, 1/total, 1.0)``
        differentiated the *unselected* ``1/total`` at total=0 — a pure-ocean
        cell, the most common cell — giving ``0*inf = NaN`` gradients w.r.t. the
        tile masks (tile-mask sensitivity / coupled adjoint).  Flooring the
        reciprocal denominator fixes it; the forward is unchanged (the reciprocal
        is only selected when total>1).  A mix of pure-ocean, sub-unity, and
        over-unity cells is exercised.
        """
        from legoesm.coupler.tile_fractions import compute_tile_fractions
        from legoesm.coupler.config import TileConfig

        n = 4
        shape = (6, n, n)
        f_land0 = jnp.zeros(shape).at[0, 0, 0].set(0.3).at[0, 1, 1].set(0.7)
        # cell (0,1,1): f_land+f_lake = 1.3 > 1 (renorm path); rest pure ocean.
        f_lake0 = jnp.zeros(shape).at[0, 1, 1].set(0.6)
        ice_conc = 0.2 * jnp.ones(shape)

        def loss(f_land):
            cfg = TileConfig(f_land=f_land, f_lake=f_lake0)
            fr = compute_tile_fractions(cfg, ice_conc)
            return jnp.sum(fr.f_ocean ** 2 + fr.f_land ** 2 + fr.f_lake ** 2)

        grad = jax.grad(loss)(f_land0)
        assert bool(jnp.all(jnp.isfinite(grad))), (
            "tile-fraction gradient not finite at a pure-ocean cell"
        )


# ============================================================================
# 6c  Full coupler step
# ============================================================================

class TestFullCouplerGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.coupler.coupler import make_coupler
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.core.coupling_fields import AtmToSurface
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


# ============================================================================
# 6d  Flux accumulator differentiability
# ============================================================================

class TestFluxAccumulatorGrad:

    def test_accumulate_and_mean_grad(self):
        """Gradient flows through accumulate -> mean_accumulator."""
        from legoesm.coupler.accumulator import (
            reset_accumulator, accumulate, mean_accumulator,
        )
        from legoesm.core.coupling_fields import SurfaceToAtm

        shape = (6, 4, 4)
        ones = jnp.ones(shape)

        def make_sfc_to_atm(T_surface):
            return SurfaceToAtm(
                T_surface=T_surface,
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
                freshwater_flux=jnp.zeros(shape),
                ocean_heat_extraction=jnp.zeros(shape),
                ocean_stress_x=jnp.zeros(shape),
                ocean_stress_y=jnp.zeros(shape),
                surface_mass_flux=jnp.zeros(shape),
                salt_flux=jnp.zeros(shape),
            )

        def loss(T_surface):
            acc = reset_accumulator(shape)
            sfc1 = make_sfc_to_atm(T_surface)
            acc = accumulate(acc, sfc1, dt=100.0)
            sfc2 = make_sfc_to_atm(T_surface + 1.0)
            acc = accumulate(acc, sfc2, dt=200.0)
            mean = mean_accumulator(acc)
            return jnp.sum(mean.T_surface ** 2)

        T_sfc = 290.0 * ones
        grad = jax.grad(loss)(T_sfc)
        assert_gradient_ok(grad, "Flux accumulator w.r.t. T_surface")


# ============================================================================
# 6e  Cross-component: atmosphere -> coupler -> ocean
# ============================================================================

class TestCrossComponentAtmOcean:

    def test_grad_T_atm_to_shflx(self):
        """Gradient of sensible heat flux w.r.t. atmospheric temperature."""
        from legoesm.coupler.bulk_flux import compute_most_fluxes

        ncol = 32
        sst = 295.0 * jnp.ones(ncol)
        u_rel = 5.0 * jnp.ones(ncol)
        v_rel = 2.0 * jnp.ones(ncol)
        q_atm = 5e-3 * jnp.ones(ncol)
        q_sfc = 8e-3 * jnp.ones(ncol)
        rho = 1.2 * jnp.ones(ncol)

        def loss(T_atm):
            _, _, shflx, _, _ = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, sst, q_sfc, rho,
                scheme="coare3", n_iter=5,
            )
            return jnp.sum(shflx ** 2)

        T_atm = 290.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(T_atm)
        assert_gradient_ok(grad, "Cross-component T_atm -> shflx")


# ============================================================================
# 6f  Lake model differentiability
# ============================================================================

class TestLakeModelGrad:

    def test_grad_wrt_T_epi(self):
        """Gradient through two-layer lake step w.r.t. epilimnion temperature."""
        from legoesm.coupler.lake import step_lake, LakeConfig, LakeState

        ncol = 32
        config = LakeConfig()
        ones = jnp.ones(ncol)
        state = LakeState(
            T_epi=Field(285.0 * ones, name="T_epi"),
            T_hypo=Field(280.0 * ones, name="T_hypo"),
        )
        forcing = self._make_lake_forcing(ncol)

        def loss(T_epi_data):
            s = state._replace(T_epi=state.T_epi.replace(data=T_epi_data))
            out, _ = step_lake(s, forcing, config, U_min=1.0, dt=3600.0)
            return jnp.sum(out.T_epi.data ** 2)

        grad = jax.grad(loss)(state.T_epi.data)
        assert_gradient_ok(grad, "Lake model w.r.t. T_epi")

    def test_grad_wrt_sw_down(self):
        """Gradient through lake step w.r.t. shortwave forcing."""
        from legoesm.coupler.lake import step_lake, LakeConfig, LakeState

        ncol = 32
        config = LakeConfig()
        ones = jnp.ones(ncol)
        state = LakeState(
            T_epi=Field(285.0 * ones, name="T_epi"),
            T_hypo=Field(280.0 * ones, name="T_hypo"),
        )
        forcing = self._make_lake_forcing(ncol)

        def loss(sw_down):
            f = forcing._replace(sw_down=sw_down)
            out, _ = step_lake(state, f, config, U_min=1.0, dt=3600.0)
            return jnp.sum(out.T_epi.data ** 2)

        grad = jax.grad(loss)(forcing.sw_down)
        assert_gradient_ok(grad, "Lake model w.r.t. sw_down")

    @staticmethod
    def _make_lake_forcing(ncol):
        from legoesm.core.coupling_fields import AtmToSurface
        ones = jnp.ones(ncol)
        return AtmToSurface(
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
