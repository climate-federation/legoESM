"""Physics integration tests for GM/Redi using the Eady uniform experiment.

The Eady uniform setup has:
- Linear EOS (rho = rho_0 * (1 - alpha_T * (T - T_ref)))  →  isopycnals = isotherms
- Uniform N², linear vertical shear, depth-uniform dT/dy
- Moderate slopes (~1e-3) → no tapering needed (taper ≈ 1 everywhere)

This gives two analytically motivated tests:

**Redi only (kappa_GM=0, kappa_Redi>0):**
  Since T is constant along isopycnals (T = T(rho) by linear EOS),
  the isopycnal gradient of T is zero everywhere.
  → Redi tendency must be zero.

**GM only (kappa_GM>0, kappa_Redi=0):**
  The skew-flux adiabatically flattens isopycnals.
  → Nonzero T tendency, APE must decrease.

Run with:
    JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_gm_redi_eady_physics.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star, compute_ocean_jacobian
from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig,
    create_initial_conditions,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_isopycnal_slopes_latlon_cgrid,
    gm_redi_tracer_tendency_latlon_cgrid,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _make_eady_setup(n_lat=12, n_lon=6, nlev=10):
    """Create a low-resolution Eady uniform setup for GM/Redi testing.

    Uses a small grid (12 lat x 6 lon, 10 levels) for fast execution.
    No perturbation — we test GM/Redi on the smooth background state.
    """
    config = EadyUniformConfig(
        H_max=5500.0,
        T_perturbation_K=0.0,   # No perturbation — clean background state
        U_surface=0.5,
        N=1.2e-3,
    )

    # Build a grid that covers the Eady domain.
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=config.H_max,
        dz_surface=200.0, dz_deep=1000.0,
    )

    # Create initial conditions (thermal-wind balanced).
    state = create_initial_conditions("latlon_channel", grid, z_coord, config)

    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    eta = state.eta.data
    H_bathy = state.H_bathy.data
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)

    return grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config


class TestRediOnlyIsZero:
    """With linear EOS, T is constant along isopycnals.
    Redi (isopycnal diffusion) should produce zero T tendency."""

    def test_redi_only_tendency_is_zero(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()

        # Compute density from T using linear EOS: rho = rho_0 * (1 - alpha_T * (T - T_ref))
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref))

        # Redi only: kappa_GM = 0, kappa_Redi = 1000
        cfg_redi = GMRediConfig(kappa_GM=0.0, kappa_Redi=1000.0, S_max=0.01)

        S_x, S_y, taper = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg_redi,
        )

        # Verify slopes are nonzero (the setup has tilted isopycnals).
        max_slope = float(jnp.max(jnp.abs(S_y) * mask[:, :, jnp.newaxis]))
        assert max_slope > 1e-5, f"Slopes should be nonzero, got max|S_y|={max_slope:.2e}"

        # Verify taper is ~1 everywhere (moderate slopes, no tapering needed).
        min_taper = float(jnp.min(taper + (1 - mask[:, :, jnp.newaxis])))
        assert min_taper > 0.9, f"Taper should be ~1, got min={min_taper:.4f}"

        # Compute Redi-only tendency.
        dT_redi = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=0.0, kappa_Redi=1000.0,
        )

        # The Redi tendency for T should be zero because T = f(rho).
        # With kappa_GM=0, horizontal flux = kappa_Redi * dq/dx + kappa_Redi * S_x * dq/dz
        # and vertical flux = kappa_Redi * (S_x*dq/dx + S_y*dq/dy) + kappa_Redi * S^2 * dq/dz
        # For T aligned with isopycnals, these should cancel to zero.
        max_dT = float(jnp.max(jnp.abs(dT_redi)))
        # Allow small residual from discrete averaging (face→center→interface).
        T_scale = float(jnp.max(jnp.abs(T)))
        relative = max_dT / max(T_scale, 1e-10)
        assert relative < 1e-4, (
            f"Redi-only tendency should be ~zero for T aligned with isopycnals, "
            f"but got max|dT|={max_dT:.4e} (relative {relative:.2e})"
        )


class TestGMOnlyFlattensIsopycnals:
    """GM (skew flux) should adiabatically flatten isopycnals,
    producing nonzero T tendency and reducing APE."""

    def test_gm_only_tendency_is_nonzero(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref))

        # GM only: kappa_GM = 1000, kappa_Redi = 0
        cfg_gm = GMRediConfig(kappa_GM=1000.0, kappa_Redi=0.0, S_max=0.01)

        S_x, S_y, taper = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg_gm,
        )

        dT_gm = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=1000.0, kappa_Redi=0.0,
        )

        # GM tendency should be nonzero (it flattens the tilted isopycnals).
        max_dT = float(jnp.max(jnp.abs(dT_gm)))
        assert max_dT > 1e-12, f"GM tendency should be nonzero, got max|dT|={max_dT:.2e}"
        assert jnp.all(jnp.isfinite(dT_gm)), "GM tendency has NaN/Inf"

    def test_gm_only_reduces_ape(self):
        """APE tendency = sum(dT * T * h * area) should be negative (flattening)."""
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref))

        cfg_gm = GMRediConfig(kappa_GM=1000.0, kappa_Redi=0.0, S_max=0.01)

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg_gm,
        )

        dT_gm = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=1000.0, kappa_Redi=0.0,
        )

        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        mask_3d = mask[:, :, jnp.newaxis]

        # d/dt(0.5 * T^2) = T * dT/dt.  For GM flattening, this should be < 0.
        ape_tendency = float(jnp.sum(dT_gm * T * dz * area * mask_3d))
        assert ape_tendency < 0, (
            f"GM should reduce APE (tracer variance), "
            f"but APE tendency = {ape_tendency:.4e}"
        )

    def test_gm_only_conserves_tracer(self):
        """sum(dT * h * area) should be zero (GM doesn't create/destroy tracer)."""
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref))

        cfg_gm = GMRediConfig(kappa_GM=1000.0, kappa_Redi=0.0, S_max=0.01)

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg_gm,
        )

        dT_gm = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=1000.0, kappa_Redi=0.0,
        )

        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        mask_3d = mask[:, :, jnp.newaxis]

        integral = float(jnp.sum(dT_gm * dz * area * mask_3d))
        max_dT = float(jnp.max(jnp.abs(dT_gm)))
        total_vol = float(jnp.sum(dz * area * mask_3d))
        relative = abs(integral) / (max_dT * total_vol) if max_dT > 0 else 0
        assert relative < 1e-8, (
            f"GM should conserve tracer, relative error = {relative:.2e}"
        )
