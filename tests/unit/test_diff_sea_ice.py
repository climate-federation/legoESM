"""Differentiability tests for sea ice model.

Categories:
  5a) Slab sea ice thermodynamics
  5b) Dynamic sea ice (EVP)
  5c) Ice strength and rheology
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


def make_ice_forcing(shape):
    from legoesm.coupler.coupling_fields import AtmToSurface
    ones = jnp.ones(shape)
    return AtmToSurface(
        sw_down=100.0 * ones,
        lw_down=250.0 * ones,
        precip_total=0.0 * ones,
        precip_snow=0.0 * ones,
        T_lowest=260.0 * ones,
        q_lowest=1e-3 * ones,
        u_lowest=5.0 * ones,
        v_lowest=2.0 * ones,
        p_lowest=1e5 * ones,
        p_surface=1.013e5 * ones,
        rho_lowest=1.4 * ones,
        cos_zenith=0.5 * ones,
        co2_ppmv=400.0 * ones,
        has_radiation=1.0 * ones,
        has_precipitation=1.0 * ones,
    )


# ============================================================================
# 5a  Slab sea ice thermodynamics
# ============================================================================

class TestSlabIceGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState

        self.step_fn = step_sea_ice
        self.config = SeaIceConfig(dynamics="none")
        n = 4
        shape = (6, n, n)
        self.state = SeaIceState(
            h_ice=Field(1.0 * jnp.ones(shape), name="h_ice"),
            T_ice=Field(265.0 * jnp.ones(shape), name="T_ice"),
            concentration=Field(0.8 * jnp.ones(shape), name="concentration"),
        )
        self.forcing = make_ice_forcing(shape)
        self.ocean_sst = 271.35 * jnp.ones(shape)
        self.ocean_u = jnp.zeros(shape)
        self.ocean_v = jnp.zeros(shape)
        self.dt = 3600.0

    def test_grad_wrt_T_ice(self):
        state = self.state

        def loss(T_data):
            s = state._replace(T_ice=state.T_ice.replace(data=T_data))
            out, _ = self.step_fn(
                s, self.forcing, self.ocean_sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt,
            )
            return jnp.sum(out.T_ice.data ** 2)

        grad = jax.grad(loss)(state.T_ice.data)
        assert_gradient_ok(grad, "Slab ice w.r.t. T_ice")

    def test_grad_wrt_ocean_sst(self):
        """SST affects ice thickness via basal melt."""
        state = self.state

        def loss(sst):
            out, _ = self.step_fn(
                state, self.forcing, sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt,
            )
            return jnp.sum(out.h_ice.data ** 2)

        grad = jax.grad(loss)(self.ocean_sst)
        assert_gradient_ok(grad, "Slab ice h_ice w.r.t. ocean_sst")


# ============================================================================
# 5b  Dynamic sea ice (EVP)
# ============================================================================

class TestDynamicIceGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import DynamicSeaIceState

        n = 4
        self.grid = create_cubed_sphere(n)
        self.step_fn = step_sea_ice
        self.config = SeaIceConfig(
            dynamics="evp",
            N_evp=10,  # small for speed
            differentiable_dynamics=True,
        )
        shape = (6, n, n)
        self.state = DynamicSeaIceState(
            h_ice=Field(1.5 * jnp.ones(shape), name="h_ice"),
            T_ice=Field(263.0 * jnp.ones(shape), name="T_ice"),
            concentration=Field(0.9 * jnp.ones(shape), name="concentration"),
            u_ice=Field(jnp.zeros(shape), name="u_ice"),
            v_ice=Field(jnp.zeros(shape), name="v_ice"),
            sigma_11=Field(jnp.zeros(shape), name="sigma_11"),
            sigma_22=Field(jnp.zeros(shape), name="sigma_22"),
            sigma_12=Field(jnp.zeros(shape), name="sigma_12"),
        )
        self.forcing = make_ice_forcing(shape)
        self.ocean_sst = 271.35 * jnp.ones(shape)
        self.ocean_u = jnp.zeros(shape)
        self.ocean_v = jnp.zeros(shape)
        self.dt = 3600.0

    def test_grad_wrt_h_ice(self):
        state = self.state

        def loss(h_data):
            s = state._replace(h_ice=state.h_ice.replace(data=h_data))
            out, _ = self.step_fn(
                s, self.forcing, self.ocean_sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt, grid=self.grid,
            )
            return jnp.sum(out.h_ice.data ** 2)

        grad = jax.grad(loss)(state.h_ice.data)
        assert_gradient_ok(grad, "Dynamic ice w.r.t. h_ice")


# ============================================================================
# 5c  Ice strength and rheology
# ============================================================================

class TestRheologyGrad:

    def test_ice_strength_grad(self):
        from legoesm.ice.rheology import ice_strength

        h = 2.0 * jnp.ones((6, 4, 4))
        A = 0.9 * jnp.ones((6, 4, 4))

        dP_dh = jax.grad(lambda h: jnp.sum(ice_strength(h, A)))(h)
        assert jnp.all(jnp.isfinite(dP_dh)), "dP/dh not finite"
        assert jnp.all(dP_dh > 0), "dP/dh should be positive (thicker = stronger)"

    def test_evp_stress_update_grad(self):
        from legoesm.ice.rheology import evp_stress_update

        shape = (6, 4, 4)
        sigma_11 = jnp.zeros(shape)
        sigma_22 = jnp.zeros(shape)
        sigma_12 = jnp.zeros(shape)
        eps_11 = 1e-6 * jnp.ones(shape)
        eps_22 = -0.5e-6 * jnp.ones(shape)
        eps_12 = 0.3e-6 * jnp.ones(shape)
        P = 1e4 * jnp.ones(shape)

        def loss(eps_11):
            s11, s22, s12 = evp_stress_update(
                sigma_11, sigma_22, sigma_12,
                eps_11, eps_22, eps_12, P,
                e_yield=2.0, T_evp=0.36, dt_s=30.0,
            )
            return jnp.sum(s11 ** 2 + s22 ** 2 + s12 ** 2)

        grad = jax.grad(loss)(eps_11)
        assert_gradient_ok(grad, "EVP stress update w.r.t. eps_11")
