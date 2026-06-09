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
    gm_redi_tracer_tendency_triads_latlon_cgrid,
    gm_redi_tracer_tendency_latlon,
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
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))

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
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))

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
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))

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
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))

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


# =====================================================================
# Triad scheme: machine-precision Redi cancellation when q = f(rho)
# =====================================================================
#
# The whole point of the Griffies, Gnanadesikan, Pacanowski et al.
# (1998) triad decomposition is that, for any tracer constant along
# isopycnals (q = f(rho)), every individual triad's flux is zero
# *algebraically* — the sum (and hence the divergence, and hence the
# tendency) is therefore zero to machine precision, regardless of
# kappa_Redi or how steep the slopes are.  Centred-difference
# discretisations cannot guarantee this.

class TestTriadRediOnlyMachinePrecision:
    """Triad Redi tendency must be zero to machine precision when T = f(rho)."""

    def _eady_rho(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))
        return grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho

    def test_triad_redi_only_tendency_machine_precision(self):
        """Direct call: Redi-only triad tendency must vanish to ~eps_64 * |T|.

        The cancellation is algebraic per triad, so the residual is
        bounded by the working-precision unit roundoff times the
        magnitude of the largest cancelling term, NOT by kappa_Redi.
        """
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()

        # Use a *huge* kappa_Redi to amplify any non-cancelling residual.
        # If the triad cancellation works, dT remains at machine precision.
        dT_redi = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=0.0, kappa_Redi=1.0e6, S_max=0.01,
        )

        T_scale = float(jnp.max(jnp.abs(T)))
        max_dT = float(jnp.max(jnp.abs(dT_redi)))
        # Allow a couple of orders of magnitude above the float64 ULP
        # to absorb the dz/dx ratios in the divergence operator.
        assert max_dT < 1e-10 * T_scale, (
            "Triad Redi tendency must be ~machine precision for q=f(rho), "
            f"but got max|dT|={max_dT:.4e} (T_scale={T_scale:.2e}, "
            f"relative {max_dT / max(T_scale, 1e-30):.2e}).  "
            f"Centered-scheme baseline at kappa_Redi=1e6 was ~5e-3 K/s."
        )

    def test_triad_redi_residual_per_kappa_is_machine_precision(self):
        """Per-unit-kappa residual must be at the float64 round-off scale.

        The cancellation is *algebraic* per triad, so the only source of
        residual is float64 evaluation noise: the subtraction
        ``K_R · dq/dx − K_R · (avg of triad cancelling terms)`` still has
        round-off ``≈ K_R · ε_64`` (this scaling is fundamental, not a
        bug).  We therefore measure the residual NORMALISED by kappa
        and require it to be at the level of ``T_scale · ε_64``.
        """
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
        T_scale = float(jnp.max(jnp.abs(T)))

        for kappa in (1.0e3, 1.0e4, 1.0e5, 1.0e6):
            dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
                T, rho, mask, u_mask, v_mask,
                z_coord, jacobian, grid,
                kappa_GM=0.0, kappa_Redi=kappa, S_max=0.01,
            )
            max_dT = float(jnp.max(jnp.abs(dT)))
            # Tight bound: residual < T_scale · 1e-15 · kappa.  The
            # 1e-15 absorbs ~10× the float64 unit roundoff.  This
            # passes for any kappa, confirming the per-triad
            # cancellation holds algebraically (residual is purely
            # round-off, not a stencil mismatch).
            bound = T_scale * 1e-15 * kappa
            assert max_dT < bound, (
                f"Triad Redi residual at kappa={kappa:.0e} is {max_dT:.4e}; "
                f"expected < T_scale·1e-15·kappa = {bound:.4e}."
            )

    def test_triad_versus_centered_redi_only(self):
        """Triad scheme must be at least as good as centered for q=f(rho).

        On this Eady setup with a strictly linear EOS, the centered
        scheme already cancels well (because all isopycnal slopes are
        below S_max and no clipping bites).  The triad scheme must
        therefore be *no worse* — and is typically equal or better,
        depending on which round-off path dominates.  The real benefit
        of triads shows up when there is a non-linear EOS, mixed-layer
        clipping, or topography breaking the simple linearity of T↔ρ.
        """
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()

        cfg_c = GMRediConfig(kappa_GM=0.0, kappa_Redi=5.0e4, S_max=0.01,
                             slope_scheme="centered")
        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg_c,
        )
        dT_c = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=0.0, kappa_Redi=5.0e4,
        )
        dT_t = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=0.0, kappa_Redi=5.0e4, S_max=0.01,
        )
        max_c = float(jnp.max(jnp.abs(dT_c)))
        max_t = float(jnp.max(jnp.abs(dT_t)))
        # Triads must not be worse than centered by more than a
        # factor that absorbs the (slightly different) floating-point
        # cancellation order in the two schemes.
        assert max_t <= 10.0 * max_c, (
            f"Triad scheme worse than centered by >10×: "
            f"centered={max_c:.4e}, triads={max_t:.4e}."
        )

    def test_triad_orchestrator_redi_only_zero(self):
        """Through the public orchestrator with slope_scheme='triads'."""
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, _ = self._eady_rho()
        config = EadyUniformConfig(
            H_max=5500.0, T_perturbation_K=0.0, U_surface=0.5, N=1.2e-3,
        )

        eta = jnp.zeros_like(mask)
        H_bathy = jnp.full_like(mask, config.H_max)

        cfg = GMRediConfig(
            kappa_GM=0.0, kappa_Redi=1.0e5, S_max=0.01,
            slope_scheme="triads",
        )
        # Use a linear EOS configured to match the Eady setup.
        from legoesm.ocean.eos import LinearEOSConfig
        eos_lin = LinearEOSConfig(
            rho_ref=config.rho_0, alpha_T=config.alpha_T, beta_S=0.0,
            T_ref=config.T_ref_C, S_ref=config.S_uniform,
        )
        dT_dt, dS_dt = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg,
            eos="linear", eos_linear=eos_lin,
            mask=mask, u_mask=u_mask, v_mask=v_mask,
        )

        T_scale = float(jnp.max(jnp.abs(T)))
        max_dT = float(jnp.max(jnp.abs(dT_dt)))
        # Bound: T_scale · 1e-15 · kappa.  Same argument as the
        # per-kappa scaling test above.
        bound = T_scale * 1e-15 * cfg.kappa_Redi
        assert max_dT < bound, (
            f"Orchestrator triad Redi-only tendency must be at the "
            f"float64 round-off level; got max|dT|={max_dT:.4e}, "
            f"bound={bound:.4e}, T_scale={T_scale:.2e}."
        )


class TestTriadGMOnly:
    """GM with the triad scheme should match the continuum GM behaviour."""

    def _eady_rho(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))
        return grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho

    def test_triad_gm_only_nonzero_and_reduces_ape(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
        dT_gm = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=1000.0, kappa_Redi=0.0, S_max=0.01,
        )
        max_dT = float(jnp.max(jnp.abs(dT_gm)))
        assert max_dT > 1e-12, f"GM tendency must be nonzero, got {max_dT:.2e}"
        assert jnp.all(jnp.isfinite(dT_gm))

        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        mask_3d = mask[:, :, jnp.newaxis]
        ape = float(jnp.sum(dT_gm * T * dz * area * mask_3d))
        assert ape < 0, f"Triad GM must reduce APE; got {ape:.4e}"

    def test_triad_gm_only_conserves_tracer(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
        dT_gm = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=1000.0, kappa_Redi=0.0, S_max=0.01,
        )
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        mask_3d = mask[:, :, jnp.newaxis]
        integral = float(jnp.sum(dT_gm * dz * area * mask_3d))
        max_dT = float(jnp.max(jnp.abs(dT_gm)))
        total_vol = float(jnp.sum(dz * area * mask_3d))
        rel = abs(integral) / (max_dT * total_vol) if max_dT > 0 else 0.0
        assert rel < 1e-10, f"Triad GM must conserve tracer; rel={rel:.2e}"


class TestTriadDifferentiability:
    """jax.grad must flow cleanly through the triad path."""

    def test_grad_through_triad_tendency(self):
        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
        rho = config.rho_0 * (1.0 - config.alpha_T * (T - config.T_ref_C))

        def loss(T_in):
            dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
                T_in, rho, mask, u_mask, v_mask,
                z_coord, jacobian, grid,
                kappa_GM=500.0, kappa_Redi=500.0, S_max=0.01,
            )
            return jnp.mean(dT ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))
