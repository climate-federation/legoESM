"""Sea ice validation tests.

Covers:
  - Transport: conservation, non-negativity, temperature bounds
  - ITD remap: category-bound preservation, volume conservation,
    concentration bounds, temperature bounds
  - EVP: zero-stress / internal-stress sanity checks
  - Configuration guardrails: error on missing grid
  - Slab vs dynamic flux consistency
  - Free drift: basic sanity
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.dynamics import evp_solver, free_drift_velocity
from legoesm.ice.itd import (
    category_bounds,
    linear_remap,
    upper_bounds,
)
from legoesm.ice.rheology import ice_strength
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.state import (
    DynamicSeaIceState,
    SeaIceState,
    init_dynamic_ice_state,
)
from legoesm.ice.transport import advect_ice_tracers


# ---- Helpers ----

def _make_grid(n=8):
    return create_cubed_sphere(n)


def _make_forcing(shape=(6, 8, 8)):
    from legoesm.coupler.coupling_fields import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.full(shape, 100.0),
        lw_down=jnp.full(shape, 200.0),
        precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, 250.0),
        q_lowest=jnp.full(shape, 1e-3),
        u_lowest=jnp.full(shape, 5.0),
        v_lowest=jnp.full(shape, -3.0),
        p_lowest=jnp.full(shape, 9.5e4),
        p_surface=jnp.full(shape, 1e5),
        rho_lowest=jnp.full(shape, 1.2),
        cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.full(shape, 400.0),
        has_radiation=jnp.ones(shape),
        has_precipitation=jnp.ones(shape),
    )


# ==============================================================================
# Transport validation
# ==============================================================================

class TestTransportConservation:
    """Test that advect_ice_tracers preserves physical bounds."""

    def test_h_nonnegative_after_advection(self):
        """Thickness must be >= 0 after transport."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        # Sharp interface: ice on face 0 only
        h = jnp.zeros(shape).at[0].set(2.0)
        a = jnp.zeros(shape).at[0].set(0.9)
        T = jnp.full(shape, 265.0)
        u = jnp.full(shape, 0.05)
        v = jnp.full(shape, 0.03)
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, grid, 3600.0)
        assert jnp.all(h_new >= 0.0), "Negative thickness after transport"

    def test_concentration_bounded_after_advection(self):
        """Concentration must stay in [0, 1]."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        h = jnp.ones(shape) * 1.0
        a = jnp.full(shape, 0.95)
        T = jnp.full(shape, 260.0)
        u = jnp.full(shape, 0.1)
        v = jnp.full(shape, -0.05)
        _, a_new, _ = advect_ice_tracers(h, a, T, u, v, grid, 3600.0)
        assert jnp.all(a_new >= 0.0), "Negative concentration"
        assert jnp.all(a_new <= 1.0), "Concentration > 1"

    def test_temperature_bounded_after_advection(self):
        """Temperature must stay in [T_ice_min, T_freeze_ocean]."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        # Create a sharp temperature gradient
        T = jnp.full(shape, 200.0).at[0].set(270.0)
        h = jnp.ones(shape) * 1.5
        a = jnp.full(shape, 0.8)
        u = jnp.full(shape, 0.08)
        v = jnp.full(shape, 0.04)
        _, _, T_new = advect_ice_tracers(h, a, T, u, v, grid, 3600.0)
        assert jnp.all(T_new >= 180.0), "T below T_ice_min after transport"
        assert jnp.all(T_new <= 271.35), "T above T_freeze_ocean after transport"

    def test_zero_velocity_preserves_state(self):
        """Zero velocity should leave state unchanged."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        h = jnp.ones(shape) * 1.5
        a = jnp.full(shape, 0.8)
        T = jnp.full(shape, 260.0)
        h_new, a_new, T_new = advect_ice_tracers(
            h, a, T, jnp.zeros(shape), jnp.zeros(shape), grid, 3600.0,
        )
        assert jnp.allclose(h_new, h, atol=1e-12)
        assert jnp.allclose(a_new, a, atol=1e-12)
        assert jnp.allclose(T_new, T, atol=1e-12)


# ==============================================================================
# ITD remap validation
# ==============================================================================

class TestITDCategoryBounds:
    """Category mean thickness should stay within bounds after remap."""

    def test_category_means_within_bounds(self):
        """After remap, each category's h should be in [lo, hi]."""
        n_cat = 5
        lo = category_bounds(n_cat)
        hi = upper_bounds(n_cat)

        # Start with valid state
        h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a_old = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
        # Growth pushes some categories beyond bounds
        h_new = jnp.array([0.3, 1.5, 2.5, 3.8, 6.0])
        a_new = a_old

        h_remap, a_remap = linear_remap(h_old, a_old, h_new, a_new, n_cat)

        for k in range(n_cat):
            if float(a_remap[k]) > 0.0:
                assert float(h_remap[k]) >= float(lo[k]) - 1e-10, (
                    f"Cat {k}: h={float(h_remap[k]):.4f} < lo={float(lo[k]):.4f}"
                )
                assert float(h_remap[k]) <= float(hi[k]) + 1e-10, (
                    f"Cat {k}: h={float(h_remap[k]):.4f} > hi={float(hi[k]):.4f}"
                )

    def test_volume_approximately_conserved(self):
        """Total volume should be approximately conserved."""
        n_cat = 5
        h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a_old = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
        h_new = h_old + 0.2
        a_new = a_old * 1.02

        vol_before = jnp.sum(h_new * a_new)
        h_remap, a_remap = linear_remap(h_old, a_old, h_new, a_new, n_cat)
        vol_after = jnp.sum(h_remap * a_remap)
        # Approximate conservation (clamping can break exact conservation)
        assert float(jnp.abs(vol_after - vol_before)) / float(vol_before) < 0.05

    def test_total_concentration_bounded(self):
        """Total concentration across categories should be <= 1."""
        n_cat = 5
        h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a_old = jnp.array([0.15, 0.15, 0.15, 0.15, 0.15])
        h_new = h_old + 0.3
        a_new = a_old * 1.1

        _, a_remap = linear_remap(h_old, a_old, h_new, a_new, n_cat)
        assert jnp.all(a_remap >= 0.0)
        assert jnp.all(a_remap <= 1.0)

    def test_temperature_bounded_after_remap(self):
        """Temperature should stay in [T_ice_min, T_freeze_ocean] after remap."""
        n_cat = 5
        shape = (n_cat,)
        h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a_old = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
        h_new = h_old + 0.3
        a_new = a_old
        T_new = jnp.array([200.0, 250.0, 260.0, 265.0, 270.0])

        _, _, T_remap = linear_remap(h_old, a_old, h_new, a_new, n_cat, T_new=T_new)
        assert jnp.all(T_remap >= 180.0), "T below T_ice_min after ITD remap"
        assert jnp.all(T_remap <= 271.35), "T above T_freeze_ocean after ITD remap"


# ==============================================================================
# EVP sanity checks
# ==============================================================================

class TestEVPSanity:
    """Basic sanity checks for EVP solver."""

    def test_zero_ice_zero_stress(self):
        """No ice => stress should be zero."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        _, _, s11, s22, s12 = evp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.zeros(shape),
            concentration=jnp.zeros(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_evp=5,
        )
        assert jnp.allclose(s11, 0.0, atol=1e-10)
        assert jnp.allclose(s22, 0.0, atol=1e-10)
        assert jnp.allclose(s12, 0.0, atol=1e-10)

    def test_zero_velocity_isotropic_stress(self):
        """If ice is still and forcing is zero, stress relaxes toward -P/2.

        The VP constitutive law gives sigma = -P/2 * I when strain is zero
        (h=1, A=0.9 → P = P*·exp(-20·0.1) ≈ 3726 N/m, so
        sigma_11 = sigma_22 → -P/2 ≈ -1863 N/m, sigma_12 → 0).

        EVP reaches this plastic rest state over SEVERAL dynamic steps, not
        in one: the elastic regularisation relaxes the carried-over stress
        toward the VP target by only ``1 - exp(-1/(2·T_evp))`` ≈ 75% per
        dynamic step (T_evp = 0.36), by design.  So we iterate a handful of
        dynamic steps (carrying sigma forward, as the model does) and assert
        convergence — a single call from zero stress only reaches ~0.75·(-P/2)
        and must NOT be expected to fully converge.
        """
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)
        s11 = jnp.zeros(shape)
        s22 = jnp.zeros(shape)
        s12 = jnp.zeros(shape)

        # Iterate dynamic steps, carrying the stress forward (as the coupled
        # model does).  The elastic stress relaxes toward the VP rest state
        # over several steps; ~6 is ample for T_evp=0.36 (per-step factor
        # 1 - exp(-1/(2·0.36)) ≈ 0.75, so the residual after 6 steps is
        # 0.25**6 ≈ 2e-4).
        for _ in range(6):
            u, v, s11, s22, s12 = evp_solver(
                u, v, s11, s22, s12,
                h_ice=jnp.ones(shape),
                concentration=jnp.full(shape, 0.9),
                wind_u=jnp.zeros(shape),
                wind_v=jnp.zeros(shape),
                ocean_u=jnp.zeros(shape),
                ocean_v=jnp.zeros(shape),
                grid=grid, dt=3600.0, N_evp=120,
            )
        P = 2.75e4 * 1.0 * jnp.exp(-20.0 * 0.1)  # ~ 3726
        # Converged normal stresses should be near -P/2 (isotropic
        # compression — the VP plastic rest state).
        assert jnp.allclose(s11, -P / 2, rtol=0.05), (
            f"s11 should converge to -P/2={float(-P/2):.1f}, "
            f"got {float(s11[0, 0, 0]):.1f}"
        )
        assert jnp.allclose(s22, -P / 2, rtol=0.05)
        # Shear stress should be near zero (isotropic state).
        assert jnp.max(jnp.abs(s12)) < 100.0

    def test_ice_strength_zero_when_no_ice(self):
        """P should be essentially zero when h=0."""
        P = ice_strength(jnp.array(0.0), jnp.array(0.9))
        assert float(P) < 1e-3


# ==============================================================================
# Configuration guardrails
# ==============================================================================

class TestConfigGuardrails:
    """step_sea_ice should raise on invalid config+grid combinations."""

    def test_evp_without_grid_raises(self):
        config = SeaIceConfig(dynamics="evp")
        state = init_dynamic_ice_state((6, 4, 4))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 4, 4))),
            concentration=state.concentration.replace(
                data=jnp.full((6, 4, 4), 0.8)
            ),
        )
        forcing = _make_forcing((6, 4, 4))
        shape = (6, 4, 4)
        with pytest.raises(ValueError, match="dynamics='evp' requires a grid"):
            step_sea_ice(
                state, forcing, jnp.full(shape, 271.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=None,
            )

    def test_advect_without_grid_raises(self):
        config = SeaIceConfig(dynamics="free_drift", transport="advect")
        state = init_dynamic_ice_state((6, 4, 4))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 4, 4))),
            concentration=state.concentration.replace(
                data=jnp.full((6, 4, 4), 0.8)
            ),
        )
        forcing = _make_forcing((6, 4, 4))
        shape = (6, 4, 4)
        with pytest.raises(ValueError, match="transport='advect' requires a grid"):
            step_sea_ice(
                state, forcing, jnp.full(shape, 271.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=None,
            )

    def test_slab_without_grid_ok(self):
        """Slab mode should work fine without grid."""
        config = SeaIceConfig()
        dims = ("face", "x", "y")
        shape = (6, 4, 4)
        state = SeaIceState(
            h_ice=Field(data=jnp.ones(shape), name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=dims, units="K"),
            concentration=Field(data=jnp.full(shape, 0.8), name="conc", dims=dims, units="1"),
        )
        forcing = _make_forcing(shape)
        new_state, _ = step_sea_ice(
            state, forcing, jnp.full(shape, 271.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0, grid=None,
        )
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))


# ==============================================================================
# Slab vs dynamic flux consistency
# ==============================================================================

class TestSlabDynamicFluxConsistency:
    """Slab and dynamic (free_drift, single-category) should produce
    consistent surface flux responses for the same state and forcing."""

    def test_response_fields_consistent(self):
        shape = (6, 8, 8)
        dims = ("face", "x", "y")

        # Same initial state
        h_init = jnp.ones(shape) * 1.5
        T_init = jnp.full(shape, 260.0)
        a_init = jnp.full(shape, 0.85)

        slab_state = SeaIceState(
            h_ice=Field(data=h_init, name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=T_init, name="T_ice", dims=dims, units="K"),
            concentration=Field(data=a_init, name="conc", dims=dims, units="1"),
        )
        dyn_state = DynamicSeaIceState(
            h_ice=Field(data=h_init, name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=T_init, name="T_ice", dims=dims, units="K"),
            concentration=Field(data=a_init, name="conc", dims=dims, units="1"),
            u_ice=Field(data=jnp.zeros(shape), name="u_ice", dims=dims, units="m/s"),
            v_ice=Field(data=jnp.zeros(shape), name="v_ice", dims=dims, units="m/s"),
            sigma_11=Field(data=jnp.zeros(shape), name="s11", dims=dims, units="N/m"),
            sigma_22=Field(data=jnp.zeros(shape), name="s22", dims=dims, units="N/m"),
            sigma_12=Field(data=jnp.zeros(shape), name="s12", dims=dims, units="N/m"),
            # Tier-1/2 new-physics tracer fields (snow/brine/ponds).  Pass-
            # through zeros here; the new-physics gates are off in this test.
            h_snow=Field(data=jnp.zeros(shape), name="h_snow", dims=dims, units="m"),
            S_ice=Field(data=jnp.zeros(shape), name="S_ice", dims=dims, units="psu"),
            pond_area=Field(data=jnp.zeros(shape), name="pond_area", dims=dims, units="1"),
            pond_depth=Field(data=jnp.zeros(shape), name="pond_depth", dims=dims, units="m"),
        )

        forcing = _make_forcing(shape)
        ocean_sst = jnp.full(shape, 271.35)
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        slab_config = SeaIceConfig(dynamics="none")
        dyn_config = SeaIceConfig(dynamics="free_drift")

        _, resp_slab = step_sea_ice(
            slab_state, forcing, ocean_sst, ocean_u, ocean_v,
            slab_config, U_min=1.0, dt=3600.0,
        )
        _, resp_dyn = step_sea_ice(
            dyn_state, forcing, ocean_sst, ocean_u, ocean_v,
            dyn_config, U_min=1.0, dt=3600.0,
        )

        # Both use constant bulk_scheme, both should use L_s now.
        # Response fields should have same sign and similar magnitude.
        # They won't be exactly equal because slab computes fluxes from
        # the pre-thermo T_ice while dynamic _build_response recomputes
        # from the post-thermo aggregated T.  But signs and order of
        # magnitude should match.
        for name in ("shflx", "lhflx", "tau_x", "tau_y"):
            s = getattr(resp_slab, name)
            d = getattr(resp_dyn, name)
            assert jnp.all(jnp.isfinite(s)), f"Slab {name} not finite"
            assert jnp.all(jnp.isfinite(d)), f"Dynamic {name} not finite"
            # Same sign
            assert jnp.all(jnp.sign(s) == jnp.sign(d)), (
                f"{name}: sign mismatch between slab and dynamic"
            )
            # Within factor of 2 (generous due to different T_ice evaluation points)
            ratio = jnp.where(
                jnp.abs(s) > 1e-10,
                jnp.abs(d / s),
                1.0,
            )
            assert jnp.all(ratio < 2.0) and jnp.all(ratio > 0.5), (
                f"{name}: magnitude mismatch > 2x between slab and dynamic"
            )


# ==============================================================================
# Free drift sanity
# ==============================================================================

class TestFreeDriftSanity:
    """Basic checks that free_drift_velocity is reasonable."""

    def test_zero_forcing_zero_velocity(self):
        u, v = free_drift_velocity(
            jnp.array(0.0), jnp.array(0.0),
            jnp.array(0.0), jnp.array(0.0),
        )
        assert float(u) == 0.0
        assert float(v) == 0.0

    def test_wind_dominates_direction(self):
        """Strong wind, weak current => ice moves in wind direction."""
        u, v = free_drift_velocity(
            jnp.array(0.0), jnp.array(0.0),    # no ocean current
            jnp.array(10.0), jnp.array(0.0),    # eastward wind
        )
        assert float(u) > 0.0, "Ice should move in wind direction"

    def test_ocean_contributes(self):
        """Non-zero ocean current should contribute to ice velocity."""
        u_no_ocean, _ = free_drift_velocity(
            jnp.array(0.0), jnp.array(0.0),
            jnp.array(5.0), jnp.array(0.0),
        )
        u_with_ocean, _ = free_drift_velocity(
            jnp.array(1.0), jnp.array(0.0),
            jnp.array(5.0), jnp.array(0.0),
        )
        assert float(u_with_ocean) > float(u_no_ocean)
