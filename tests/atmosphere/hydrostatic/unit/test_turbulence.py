"""Unit tests for atmospheric turbulence / boundary layer module.

Tests cover:
- Vertical diffusion: shape, conservation, smoothing, differentiability
- Surface layer: shape/sign, wind sensitivity, warm surface
- Smagorinsky: output shapes, mixing, differentiability
- Louis: stable vs unstable Ri, shapes, differentiability
- TKE: shear response, minimum TKE, shapes, differentiability
- Integration: hydrostatic/NH shapes, nonzero tendencies, jax.grad, scheme selection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.turbulence.config import (
    SurfaceLayerConfig,
    SmagorinskyConfig,
    LouisConfig,
    TKEConfig,
    CLUBBLiteConfig,
    HoltslagBovilleConfig,
    YSUConfig,
    TurbulentEDMFConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    PBLHeightConfig,
    compute_bulk_richardson,
    diagnose_pbl_height,
    diagnose_pbl_height_interp,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.atmosphere.physics.turbulence.integration import (
    make_turbulence_physics,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_column_data(ncol=4, nlev=10):
    """Create test column data with a sheared wind profile.

    Returns u, v, T, q_v, p_full, p_half, z_full, z_half, rho.
    Levels ordered top (index 0) to bottom (index nlev-1).
    """
    # Pressure: linearly spaced interfaces from 100 Pa (top) to 1e5 Pa (surface)
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Temperature: warm at surface, cool aloft
    T = jnp.broadcast_to(
        jnp.linspace(220.0, 290.0, nlev)[None, :],
        (ncol, nlev),
    )

    # Heights: approximate from hydrostatic balance
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)

    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])

    # Density from ideal gas
    rho = p_full / (constants.R_d * T)

    # Wind: sheared profile (jet aloft)
    u = jnp.broadcast_to(
        jnp.linspace(20.0, 2.0, nlev)[None, :],
        (ncol, nlev),
    )
    v = jnp.broadcast_to(
        jnp.linspace(5.0, 1.0, nlev)[None, :],
        (ncol, nlev),
    )

    # Moisture
    q_sat = saturation_mixing_ratio(T, p_full)
    rh = jnp.linspace(0.1, 0.8, nlev)[None, :]
    q_v = rh * q_sat

    return u, v, T, q_v, p_full, p_half, z_full, z_half, rho


# ===========================================================================
# Vertical Diffusion tests
# ===========================================================================

class TestVerticalDiffusion:
    """Tests for the implicit Thomas solver."""

    def test_shape_preservation(self):
        """Output should have the same shape as input."""
        ncol, nlev = 4, 10
        phi = jnp.ones((ncol, nlev))
        K_half = jnp.full((ncol, nlev - 1), 10.0)
        rho = jnp.ones((ncol, nlev))
        dz = jnp.full((ncol, nlev), 1000.0)
        dz_half = jnp.full((ncol, nlev - 1), 1000.0)
        sflx = jnp.zeros(ncol)

        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
        assert result.shape == (ncol, nlev)

    def test_float64_inputs_coerced_to_field_dtype_no_warning(self):
        """A float32 field with WIDER (float64) coefficients — e.g. a per-column C_K
        diagnosed in float64 by the LES-informed correction loop feeding a float32 run —
        stays in the field's dtype: the coefficients are coerced so the tridiagonal
        scatters do NOT emit a float64→float32 JAX FutureWarning (a future error), and
        the result is BIT-IDENTICAL to passing the same values already in float32."""
        import warnings

        import numpy as np

        ncol, nlev = 4, 10
        phi = jnp.ones((ncol, nlev), dtype=jnp.float32)
        wide = dict(
            K_half=jnp.full((ncol, nlev - 1), 12.0, dtype=jnp.float64),
            rho=jnp.full((ncol, nlev), 1.2, dtype=jnp.float64),
            dz=jnp.full((ncol, nlev), 900.0, dtype=jnp.float64),
            dz_half=jnp.full((ncol, nlev - 1), 950.0, dtype=jnp.float64),
            sflx=jnp.zeros(ncol, dtype=jnp.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)    # the cast must silence it
            out = implicit_vertical_diffusion(
                phi, wide["K_half"], wide["rho"], wide["dz"], wide["dz_half"], 60.0,
                wide["sflx"])
        assert out.dtype == jnp.float32                      # stays in the field's dtype
        # Behavior-preserving: identical to the same values supplied already in float32.
        out_f32 = implicit_vertical_diffusion(
            phi, wide["K_half"].astype(jnp.float32), wide["rho"].astype(jnp.float32),
            wide["dz"].astype(jnp.float32), wide["dz_half"].astype(jnp.float32), 60.0,
            wide["sflx"].astype(jnp.float32))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(out_f32))

    def test_conserves_column_integral(self):
        """With zero surface flux, column integral should be conserved."""
        ncol, nlev = 3, 8
        key = jax.random.PRNGKey(42)
        phi = jax.random.normal(key, (ncol, nlev)) + 10.0
        K_half = jnp.full((ncol, nlev - 1), 5.0)
        rho = jnp.ones((ncol, nlev))
        dz = jnp.full((ncol, nlev), 500.0)
        dz_half = jnp.full((ncol, nlev - 1), 500.0)
        sflx = jnp.zeros(ncol)

        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 300.0, sflx)

        # Column integral: sum(phi * rho * dz) should be conserved
        integral_before = jnp.sum(phi * rho * dz, axis=1)
        integral_after = jnp.sum(result * rho * dz, axis=1)
        # Use rtol=1e-4 to accommodate float32 precision
        assert jnp.allclose(integral_before, integral_after, rtol=1e-4)

    def test_smooths_sharp_gradient(self):
        """Diffusion should smooth a step function."""
        ncol, nlev = 2, 20
        phi = jnp.zeros((ncol, nlev))
        phi = phi.at[:, nlev // 2:].set(10.0)  # step function

        K_half = jnp.full((ncol, nlev - 1), 50.0)
        rho = jnp.ones((ncol, nlev))
        dz = jnp.full((ncol, nlev), 200.0)
        dz_half = jnp.full((ncol, nlev - 1), 200.0)
        sflx = jnp.zeros(ncol)

        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)

        # The max gradient should be reduced
        max_grad_before = float(jnp.max(jnp.abs(jnp.diff(phi, axis=1))))
        max_grad_after = float(jnp.max(jnp.abs(jnp.diff(result, axis=1))))
        assert max_grad_after < max_grad_before

    def test_differentiable(self):
        """jax.grad should work through the solver."""
        ncol, nlev = 2, 6

        def loss(phi):
            K_half = jnp.full((ncol, nlev - 1), 10.0)
            rho = jnp.ones((ncol, nlev))
            dz = jnp.full((ncol, nlev), 500.0)
            dz_half = jnp.full((ncol, nlev - 1), 500.0)
            sflx = jnp.zeros(ncol)
            result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
            return jnp.sum(result ** 2)

        phi = jax.random.normal(jax.random.PRNGKey(0), (ncol, nlev)) + 5.0
        g = jax.grad(loss)(phi)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == phi.shape


# ===========================================================================
# Surface Layer tests
# ===========================================================================

class TestSurfaceLayer:
    """Tests for bulk aerodynamic surface fluxes."""

    def test_shapes(self):
        """Surface flux outputs should have correct shapes."""
        ncol = 4
        u = jnp.full(ncol, 5.0)
        v = jnp.full(ncol, 2.0)
        T = jnp.full(ncol, 290.0)
        q_v = jnp.full(ncol, 0.01)
        T_sfc = jnp.full(ncol, 295.0)
        q_sfc = jnp.full(ncol, 0.015)
        rho = jnp.full(ncol, 1.2)
        config = SurfaceLayerConfig()

        tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho, config,
        )

        assert tau_x.shape == (ncol,)
        assert tau_y.shape == (ncol,)
        assert shflx.shape == (ncol,)
        assert lhflx.shape == (ncol,)
        assert ustar.shape == (ncol,)

    def test_sign_conventions(self):
        """Verify sign: tau_x < 0 for u > 0, shflx > 0 for warm surface."""
        ncol = 2
        u = jnp.full(ncol, 10.0)
        v = jnp.zeros(ncol)
        T = jnp.full(ncol, 285.0)
        q_v = jnp.full(ncol, 0.005)
        T_sfc = jnp.full(ncol, 295.0)  # warm surface
        q_sfc = jnp.full(ncol, 0.015)
        rho = jnp.full(ncol, 1.2)
        config = SurfaceLayerConfig()

        tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho, config,
        )

        assert jnp.all(tau_x < 0)  # drag opposes wind
        assert jnp.all(shflx > 0)  # warm surface → upward heat
        assert jnp.all(lhflx > 0)  # q_sfc > q_v → upward moisture
        assert jnp.all(ustar > 0)

    def test_stronger_wind_larger_fluxes(self):
        """Stronger wind should produce larger magnitude fluxes."""
        ncol = 1
        T = jnp.full(ncol, 285.0)
        q_v = jnp.full(ncol, 0.005)
        T_sfc = jnp.full(ncol, 295.0)
        q_sfc = jnp.full(ncol, 0.015)
        rho = jnp.full(ncol, 1.2)
        config = SurfaceLayerConfig()

        # Weak wind
        tau_x_weak, _, shflx_weak, _, _ = compute_surface_fluxes(
            jnp.full(ncol, 2.0), jnp.zeros(ncol),
            T, q_v, T_sfc, q_sfc, rho, config,
        )

        # Strong wind
        tau_x_strong, _, shflx_strong, _, _ = compute_surface_fluxes(
            jnp.full(ncol, 20.0), jnp.zeros(ncol),
            T, q_v, T_sfc, q_sfc, rho, config,
        )

        assert float(jnp.abs(tau_x_strong[0])) > float(jnp.abs(tau_x_weak[0]))
        assert float(shflx_strong[0]) > float(shflx_weak[0])


# ===========================================================================
# Smagorinsky tests
# ===========================================================================

class TestSmagorinsky:
    """Tests for the Smagorinsky–Lilly turbulence closure."""

    def test_output_shapes(self):
        """Smagorinsky output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = SmagorinskyConfig()

        out = smagorinsky_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)
        assert out.lhflx.shape == (ncol,)
        assert out.ustar.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """Smagorinsky should produce nonzero tendencies for sheared profiles."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0  # warm surface
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = SmagorinskyConfig()

        out = smagorinsky_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through Smagorinsky turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = SmagorinskyConfig()

        def loss(T_in):
            out = smagorinsky_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_strain_dependent_and_lilly_cutoff(self):
        """Faithful Smagorinsky–Lilly: K_m grows with deformation |S|,
        is enhanced when unstable, and shuts off exactly at Ri ≥ Pr_t —
        with the cutoff gradient finite (the √(max(·,0)) AD trap)."""
        ncol, nlev = 2, 12
        z_half = jnp.linspace(2000.0, 0.0, nlev + 1)[None, :].repeat(ncol, 0)
        z_full = 0.5 * (z_half[:, 1:] + z_half[:, :-1])
        p_half = jnp.linspace(8e4, 1.0e5, nlev + 1)[None, :].repeat(ncol, 0)
        p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
        rho = p_full / (constants.R_d * 280.0)
        cfg = SmagorinskyConfig()

        def run(shear, dTdz):
            u = shear * z_full
            v = jnp.zeros_like(u)
            T = 288.0 + dTdz * z_full
            q_v = jnp.full_like(T, 1e-3)
            T_sfc = T[:, -1]
            q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
            return smagorinsky_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=cfg,
            )

        # (1) strain-dependence: stronger shear lowers Ri -> more mixing
        assert float(jnp.mean(run(0.02, -0.002).Km)) > \
            float(jnp.mean(run(0.002, -0.002).Km))
        # (2) Lilly cutoff: a strong inversion (Ri >= Pr_t) zeroes K_m
        assert float(jnp.max(run(0.01, +0.05).Km)) < 1e-9
        # (3) unstable enhances mixing over the stable case
        assert float(jnp.mean(run(0.01, -0.02).Km)) > \
            float(jnp.mean(run(0.01, +0.05).Km))
        # (4) gradient finite straddling the Ri = Pr_t cutoff
        g = jax.grad(lambda d: jnp.sum(run(0.01, d).Km))(0.0098)
        assert jnp.isfinite(g)

    def test_free_convection_limit(self):
        """At zero resolved shear the floored S² + Lilly factor give a
        well-defined buoyancy-driven free-convection limit
        K_m → (C_s·l)²·√(|N²|/Pr_t) when unstable, but K_m = 0 when stable
        (Lilly cutoff).  Guards the limit against the AD-safety floor
        value (it must be O(1), not O(√floor))."""
        ncol, nlev = 2, 12
        z_half = jnp.linspace(2000.0, 0.0, nlev + 1)[None, :].repeat(ncol, 0)
        z_full = 0.5 * (z_half[:, 1:] + z_half[:, :-1])
        p_half = jnp.linspace(8e4, 1.0e5, nlev + 1)[None, :].repeat(ncol, 0)
        p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
        rho = p_full / (constants.R_d * 280.0)
        cfg = SmagorinskyConfig()
        u = jnp.zeros((ncol, nlev))            # ZERO resolved shear
        v = jnp.zeros((ncol, nlev))
        q_v = jnp.full((ncol, nlev), 1e-3)

        def run(dTdz):
            T = 288.0 + dTdz * z_full
            T_sfc = T[:, -1]
            q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
            return smagorinsky_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=cfg,
            )

        Km_unstable = run(-0.02).Km            # super-adiabatic (unstable in θ)
        Km_stable = run(+0.005).Km             # inversion (stable)
        assert 1e-3 < float(jnp.max(Km_unstable)) < 1e2  # O(1), not floor-tied
        assert float(jnp.max(Km_stable)) < 1e-9          # Lilly cutoff


# ===========================================================================
# Louis tests
# ===========================================================================

class TestLouis:
    """Tests for Louis (1979) stability-dependent turbulence."""

    def test_output_shapes(self):
        """Louis output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = LouisConfig()

        out = louis_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)

    def test_stable_reduces_Km(self):
        """Stable stratification should reduce Km compared to unstable."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        # Stable: inversion (T increases with height at bottom)
        T_stable = T.at[:, -1].set(T[:, -2] - 5.0)
        T_sfc = T_stable[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = LouisConfig()

        out_stable = louis_turbulence(
            u, v, T_stable, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        # Unstable: very warm surface
        T_unstable = T.at[:, -1].set(T[:, -2] + 20.0)
        T_sfc_u = T_unstable[:, -1]
        q_sfc_u = saturation_mixing_ratio(T_sfc_u, p_full[:, -1])

        out_unstable = louis_turbulence(
            u, v, T_unstable, q_v, p_full, p_half, z_full, z_half,
            T_sfc_u, q_sfc_u, rho, dt=300.0, config=config,
        )

        # Unstable should have larger Km near surface
        mean_Km_stable = float(jnp.mean(out_stable.Km[:, -2:]))
        mean_Km_unstable = float(jnp.mean(out_unstable.Km[:, -2:]))
        assert mean_Km_unstable > mean_Km_stable

    def test_differentiable(self):
        """jax.grad should work through Louis turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = LouisConfig()

        def loss(T_in):
            out = louis_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))

    def test_separate_heat_function_prandtl(self):
        """Faithful Louis: heat function f_h ≠ f_m (b_h/b_m = 1.5), giving
        a stratification-dependent Pr_t = K_m/K_h > 1 stable, < 1 unstable;
        ``b_heat_ratio=1`` recovers the old f_h=f_m (Pr_t≡1).  Momentum K_m
        must be unaffected by the ratio."""
        ncol, nlev = 2, 10
        z_half = jnp.linspace(2000.0, 0.0, nlev + 1)[None, :].repeat(ncol, 0)
        z_full = 0.5 * (z_half[:, 1:] + z_half[:, :-1])
        p_half = jnp.linspace(8.0e4, 1.0e5, nlev + 1)[None, :].repeat(ncol, 0)
        p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
        u = 0.01 * z_full
        v = jnp.zeros_like(u)
        q_v = jnp.full((ncol, nlev), 5e-3)

        def mean_prandtl(dTdz, ratio=1.5):
            T = 290.0 + dTdz * z_full
            rho = p_full / (constants.R_d * T)
            T_sfc = T[:, -1]
            q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
            out = louis_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0,
                config=LouisConfig()._replace(b_heat_ratio=ratio),
            )
            mask = out.Km > 1e-8
            Pr = jnp.where(mask, out.Km / jnp.clip(out.Kh, 1e-12, None), jnp.nan)
            return float(jnp.nanmean(Pr)), out

        pr_unstable, _ = mean_prandtl(-0.012)   # super-adiabatic
        pr_stable, out_stable = mean_prandtl(+0.005)  # inversion
        assert pr_unstable < 1.0
        assert pr_stable > 1.0

        # b_heat_ratio = 1.0 collapses to Pr_t ≡ 1, and K_m is identical.
        _, out_ratio1 = mean_prandtl(+0.005, ratio=1.0)
        assert jnp.allclose(out_ratio1.Km, out_ratio1.Kh)
        assert jnp.allclose(out_stable.Km, out_ratio1.Km)  # momentum unaffected


# ===========================================================================
# TKE tests
# ===========================================================================

class TestTKE:
    """Tests for prognostic TKE turbulence."""

    def test_output_shapes(self):
        """TKE output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TKEConfig()

        out, tke_new = tke_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert tke_new.shape == (ncol, nlev)

    def test_tke_increases_with_shear(self):
        """More shear should produce higher TKE."""
        ncol, nlev = 2, 10
        _, _, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TKEConfig()

        # Weak shear
        u_weak = jnp.full((ncol, nlev), 1.0)
        v_weak = jnp.zeros((ncol, nlev))

        _, tke_weak = tke_turbulence(
            u_weak, v_weak, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        # Strong shear
        u_strong = jnp.broadcast_to(
            jnp.linspace(30.0, 0.0, nlev)[None, :], (ncol, nlev),
        )
        v_strong = jnp.zeros((ncol, nlev))

        _, tke_strong = tke_turbulence(
            u_strong, v_strong, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.mean(tke_strong)) > float(jnp.mean(tke_weak))

    def test_tke_stays_above_minimum(self):
        """TKE should never fall below tke_min."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = TKEConfig(tke_min=1e-4)

        # Start with very small TKE
        tke = jnp.full((ncol, nlev), 1e-8)

        _, tke_new = tke_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(tke_new >= config.tke_min)

    def test_differentiable(self):
        """jax.grad should work through TKE turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TKEConfig()

        def loss(T_in):
            out, _ = tke_turbulence(
                u, v, T_in, q_v, tke, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Tests for make_turbulence_physics integration bridge."""

    def test_hydrostatic_tendency_shapes(self):
        """Hydrostatic turbulence tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        n = grid.n
        nlev = sigma.n_levels
        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dv_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dp_s_dt.data.shape == (6, n, n)

    def test_hydrostatic_nonzero_momentum_tendency(self):
        """Hydrostatic turbulence should produce nonzero wind tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        # Give state some wind to mix
        n = grid.n
        nlev = sigma.n_levels
        u_data = jnp.ones((6, n, n, nlev)) * 10.0
        state = state._replace(
            u=Field(data=u_data, name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        # Should have nonzero du_dt (from surface drag at minimum)
        max_du = float(jnp.max(jnp.abs(tendencies.du_dt.data)))
        assert max_du > 0.0

    def test_hydrostatic_nonzero_heat_tendency(self):
        """Hydrostatic turbulence should produce nonzero T tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        # Smagorinsky-Lilly is a shear-driven closure with a Lilly
        # buoyancy factor, so the resting, stably-stratified Held-Suarez
        # init gives K_m≈0.  Impose vertical shear and a super-adiabatic
        # bottom layer (unstable interface → f_buoy>1) so the deformation
        # actually drives interior mixing of the θ-gradient.
        from legoesm.core.field import Field
        n = grid.n
        nlev = sigma.n_levels
        u_prof = jnp.linspace(2.0, 25.0, nlev)  # sheared (level 0 = top)
        T_unstable = state.T.data.at[:, :, :, -1].add(30.0)
        state = state._replace(
            T=Field(data=T_unstable, name="T",
                    dims=("face", "x", "y", "level"), units="K"),
            u=Field(data=jnp.broadcast_to(u_prof, (6, n, n, nlev)),
                    name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        max_dT = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_dT > 0.0

    def test_nonhydrostatic_tendency_shapes(self):
        """Non-hydrostatic turbulence tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        n = 8
        nlev = 10
        grid = create_cubed_sphere(n)
        height_coord = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros((6, n, n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        state = NonHydrostaticState(
            u=Field(data=jnp.ones((6, n, n, nlev)) * 5.0, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.ones((6, n, n, nlev)) * 2.0, name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 1)), name="tracers", dims=dims_tr, units="kg/kg"),
        )

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="nonhydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, height_coord, terrain_metric)

        assert tendencies.dtheta_prime_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dw_dt.data.shape == (6, n, n, nlev + 1)
        assert tendencies.dtracers_dt.data.shape == (6, n, n, nlev, 1)

    def test_grad_through_hydrostatic_turbulence(self):
        """jax.grad should work through hydrostatic turbulence physics."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tendencies, _ = physics_fn(new_state, grid, sigma)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))

    def test_scheme_selection(self):
        """Different schemes should produce different tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        # Give state vertically-sheared wind so both deformation-based
        # (Smagorinsky) and stability-function (Louis) closures produce
        # nonzero — and distinct — interior diffusivities.  A uniform
        # u would give |S|≈0 and K≈0 for both, masking the difference.
        n = grid.n
        nlev = sigma.n_levels
        u_data = jnp.broadcast_to(jnp.linspace(2.0, 25.0, nlev), (6, n, n, nlev))
        state = state._replace(
            u=Field(data=u_data, name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        fn_smag = make_turbulence_physics(
            TurbulenceConfig(scheme="smagorinsky"), model_type="hydrostatic", dt=300.0,
        )
        fn_louis = make_turbulence_physics(
            TurbulenceConfig(scheme="louis"), model_type="hydrostatic", dt=300.0,
        )

        tend_smag, _ = fn_smag(state, grid, sigma)
        tend_louis, _ = fn_louis(state, grid, sigma)

        # Both nonzero but different
        assert float(jnp.max(jnp.abs(tend_smag.du_dt.data))) > 0
        assert float(jnp.max(jnp.abs(tend_louis.du_dt.data))) > 0
        assert not jnp.allclose(tend_smag.du_dt.data, tend_louis.du_dt.data, atol=1e-10)

    def test_all_scheme_strings_accepted(self):
        """All supported scheme strings plus 'none' should be accepted."""
        schemes = [
            "smagorinsky", "louis", "tke", "clubb_lite",
            "holtslag_boville", "ysu", "edmf", "none",
        ]
        for scheme in schemes:
            config = TurbulenceConfig(scheme=scheme)
            fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
            assert callable(fn), f"Factory should return callable for scheme={scheme!r}"

    def test_none_scheme_gives_zeros(self):
        """scheme='none' should produce zero tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="none")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tendencies.dT_dt.data, 0.0)
        assert jnp.allclose(tendencies.du_dt.data, 0.0)
        assert jnp.allclose(tendencies.dv_dt.data, 0.0)

# ===========================================================================
# Holtslag-Boville tests
# ===========================================================================

class TestHoltslagBoville:
    """Tests for Holtslag-Boville nonlocal K-profile turbulence."""

    def test_output_shapes(self):
        """HB output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)
        assert out.lhflx.shape == (ncol,)
        assert out.ustar.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """HB should produce nonzero tendencies for sheared profiles."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through HB turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        def loss(T_in):
            out = holtslag_boville_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_finite_outputs(self):
        """All outputs should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.Km))
        assert jnp.all(jnp.isfinite(out.Kh))

    def test_counter_gradient_effect(self):
        """Warm surface should activate counter-gradient, changing T tendency."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 10.0  # very warm surface
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        # With counter-gradient (oracle default: fakn=7.2 drives cgs via
        # fak3 = fakn*wstar/wm; the nonlocal countergradient feature).
        config_cg = HoltslagBovilleConfig(fakn=7.2)
        out_cg = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_cg,
        )

        # Without counter-gradient: fakn=0 zeroes fak3 -> cgs=0 -> no
        # nonlocal countergradient transport (pure local K-profile).
        config_no_cg = HoltslagBovilleConfig(fakn=0.0)
        out_no_cg = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_no_cg,
        )

        # T tendencies should differ when counter-gradient is active
        assert not jnp.allclose(out_cg.dT_dt, out_no_cg.dT_dt, atol=1e-10)


# ===========================================================================
# YSU tests
# ===========================================================================

class TestYSU:
    """Tests for YSU PBL turbulence scheme."""

    def test_output_shapes(self):
        """YSU output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """YSU should produce nonzero tendencies for sheared profiles."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through YSU turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        def loss(T_in):
            out = ysu_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_finite_outputs(self):
        """All outputs should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.Km))

    def test_louis_constants_are_config_driven(self):
        """The Louis (1982) ``b``, ``c``, ``d`` stability constants and
        the Ri-blend sharpness must all come from ``YSUConfig`` and not
        be hardcoded inside ``ysu.py``.

        Why each sub-check is non-vacuous: each constant controls a
        distinct part of the f_stable / f_unstable / blend formula.  If
        ``ysu.py`` ignored any of them (i.e. they remained hardcoded),
        the corresponding sensitivity output would be bit-identical to
        the default — the tests below would fail by construction.

        - louis_b: enters both branches → affects all Ri regimes.
        - louis_d: stable-branch sqrt coefficient → affects only Ri > 0.
        - louis_c: unstable-branch denominator coefficient → only Ri < 0.
        - blend_ri_sharpness: stable/unstable blend smoothness → only
          materially affects |Ri| ≲ 1/sharpness regions.
        """
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        out_default = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=YSUConfig(),
        )

        # Sub-check A: doubling louis_b must change Km (both branches)
        out_b = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=YSUConfig(louis_b=10.0),
        )
        assert not jnp.allclose(out_default.Km, out_b.Km, atol=1e-12), (
            "Km did not change under config.louis_b doubling — louis_b "
            "is still hardcoded inside ysu.py."
        )

        # Sub-check B: doubling louis_d must change Km (stable branch)
        out_d = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=YSUConfig(louis_d=10.0),
        )
        assert not jnp.allclose(out_default.Km, out_d.Km, atol=1e-12), (
            "Km did not change under config.louis_d doubling — louis_d "
            "(stable-branch sqrt coefficient) is still hardcoded."
        )

        # Sub-check C (louis_c): louis_c controls f_unstable in
        # ``Km_local = l_mix² · S · f_m``, but Km_local is only weighted
        # ABOVE the PBL (where blend_pbl ≈ 1).  In a normally-stratified
        # atmosphere, Ri > 0 above the PBL — so f_unstable is gated to
        # zero (Ri_neg = min(Ri, 0) = 0) at every contributing level.
        # We therefore verify louis_c is consumed at the source level
        # rather than through the full Km output.  Reading the file
        # contents and asserting the literal token ``config.louis_c`` is
        # used in ysu.py is non-vacuous: the prior hardcoded version had
        # no such reference.
        from tests.legoesm_paths import legoesm_source_path
        ysu_src = legoesm_source_path(
            "src/legoesm/atmosphere/physics/turbulence/ysu.py"
        )
        ysu_text = ysu_src.read_text()
        assert "config.louis_c" in ysu_text, (
            "ysu.py must consume config.louis_c (the unstable-branch "
            "Louis denominator coefficient).  The previous hardcoded "
            "literal ``5.0`` would not match this assertion."
        )

        # Sub-check D (blend_ri_sharpness): like louis_c, this only
        # affects ``Km_local`` (the local Richardson-based diffusivity
        # ABOVE the PBL).  In a normally-stratified column with
        # Ri > 0 above the PBL, blend_ri (= sigmoid(s · Ri)) is already
        # saturated to 1.0 at any reasonable sharpness, so changing s
        # from 10 → 100 has no measurable effect on the full Km output.
        # Verify code-level consumption instead.
        assert "config.blend_ri_sharpness" in ysu_text, (
            "ysu.py must consume config.blend_ri_sharpness.  The "
            "previous hardcoded ``100.0`` literal would not match this."
        )

    def test_entrainment_near_pbl_top(self):
        """YSU with entrainment should differ from zero-entrainment.

        ``entrainment_ratio`` is the Hong06 prescribed entrainment-flux
        ratio (w'th')_h = -e_ratio*(w'th')_0 (renamed from the old
        ``entrainment_coeff`` Gaussian-K magnitude — a different quantity).
        """
        ncol, nlev = 2, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 15.0  # very warm surface for strong convective BL
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        # With strong entrainment
        config_ent = YSUConfig(entrainment_ratio=0.3)
        out_ent = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_ent,
        )

        # Without entrainment
        config_no_ent = YSUConfig(entrainment_ratio=0.0)
        out_no_ent = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_no_ent,
        )

        # Tendencies should differ when entrainment is active
        assert not jnp.allclose(out_ent.dT_dt, out_no_ent.dT_dt, atol=1e-12)

    def test_entrainment_flux_matches_prescribed_ratio(self):
        """The Hong06 entrainment closure pins the PBL-top heat flux to the
        surface flux: at the inversion the ADDED entrainment diffusivity
        satisfies -K_ent*(dtheta_v/dz) = -e_ratio*(w'th')_0*envelope,
        INDEPENDENT of the gradient magnitude — the defining Hong06 ratio
        closure the old Gaussian down-gradient K_ent = c*w**h blob did not
        satisfy.

        h_pbl and shflx are identical between the e_ratio=0.15 and e_ratio=0
        runs (entrainment K enters after the PBL height and surface fluxes),
        so the half-level Kh DIFFERENCE is exactly the entrainment K.  We
        reconstruct half-level Kh from the full-level diagnostic via the
        endpoint row (Kh_full[:, 0] == Kh_half[:, 0]) plus the interior
        interpolation recursion.
        """
        import numpy as np

        ncol, nlev = 1, 30
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        # Gentle uniform wind; strong surface heating -> convective BL.
        u = jnp.zeros_like(u) + 3.0
        v = jnp.zeros_like(v)
        T_sfc = T[:, -1] + 12.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        cfg0 = YSUConfig(entrainment_ratio=0.0)
        cfg1 = YSUConfig(entrainment_ratio=0.15)
        out0 = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=cfg0,
        )
        out1 = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=cfg1,
        )
        # PBL height and surface flux are entrainment-independent.
        np.testing.assert_allclose(
            np.asarray(out1.h_pbl), np.asarray(out0.h_pbl), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(out1.shflx), np.asarray(out0.shflx), rtol=1e-12)

        # Reconstruct the half-level Kh difference (== K_ent_h) from the
        # full-level diagnostic: dK_full[:, 0] = dK_half[:, 0];
        # interior dK_full[k] = 0.5*(dK_half[k-1] + dK_half[k]).
        dK_full = np.asarray(out1.Kh - out0.Kh)  # (ncol, nlev)
        nhalf = nlev - 1
        dK_half = np.zeros((ncol, nhalf))
        dK_half[:, 0] = dK_full[:, 0]
        for k in range(1, nhalf):
            dK_half[:, k] = 2.0 * dK_full[:, k] - dK_half[:, k - 1]

        # Rebuild the module's own geometry to locate the inversion interface.
        from legoesm.atmosphere.physics._shared import (
            virtual_temperature, exner_function,
        )
        exner_pref = 1.0 / exner_function(p_full)
        theta_v = np.asarray(virtual_temperature(T, q_v) * exner_pref)
        z_half_inner = np.asarray(0.5 * (z_full[:, :-1] + z_full[:, 1:]))
        dz_half = np.clip(
            np.abs(np.asarray(z_full[:, :-1] - z_full[:, 1:])), 1.0, None)
        dthdz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half

        h = float(np.asarray(out1.h_pbl)[0])
        wth0 = float(np.asarray(out1.shflx)[0]) / (
            float(np.asarray(rho)[0, -1]) * constants.c_pd
        )
        assert wth0 > 0.0, "test column must be unstable (upward surface flux)"

        # Interface nearest the PBL top.
        j = int(np.argmin(np.abs(z_half_inner[0] - h)))
        width = float(cfg1.entrainment_width_frac) * h
        env = float(np.exp(-((z_half_inner[0, j] - h) / max(width, 1.0)) ** 2))
        # The fixture's theta_v increases with height, so the inversion
        # gradient must be resolved (above the module's 1e-4 K/m floor) —
        # this keeps the flux-matching assertion below non-vacuous.
        assert dthdz[0, j] > 1.0e-4, "fixture must resolve the inversion"

        # Implied entrainment heat flux at that interface (positive-up):
        # F_ent = -K_ent*(dtheta_v/dz) == -e_ratio*(w'th')_0*envelope when
        # the flux-matching branch is active and the stability cap does not
        # bind (K_ent < cap by construction here: modest wth0, strong grad).
        F_ent = -dK_half[0, j] * dthdz[0, j]
        expected = -0.15 * wth0 * env
        np.testing.assert_allclose(F_ent, expected, rtol=5e-2)
        # The prescribed flux is DOWNWARD (negative) at the inversion and
        # the added diffusivity is non-negative everywhere.
        assert F_ent < 0.0
        assert np.all(dK_half >= -1e-8)

    def test_countergradient_consistent_with_excess_parcel(self):
        """gamma_c must use the MIXED-LAYER velocity scale w_s0 (= w_s_sfc,
        the same scale as the excess parcel theta_T), so gamma_c = theta_T/h
        by construction (Troen-Mahrt 1986 / Hong06 define both through
        w_s0).  Source-structural assertion (same pattern as
        test_louis_constants_are_config_driven): the old code divided by the
        pure convective w_star, which blows up gamma_c in windy
        weakly-convective columns (u* >> w*).
        """
        from tests.legoesm_paths import legoesm_source_path
        ysu_src = legoesm_source_path(
            "src/legoesm/atmosphere/physics/turbulence/ysu.py"
        )
        ysu_text = ysu_src.read_text()
        assert "counter_grad = excess_theta / h_pbl" in ysu_text, (
            "YSU countergradient must be gamma_c = theta_T/h (the excess "
            "parcel and gamma_c share the SAME mixed-layer velocity scale "
            "w_s0 per Troen-Mahrt/Hong06); dividing by the pure convective "
            "w_star overestimates gamma_c when u* >> w*."
        )

    def test_countergradient_bounded_in_windy_weakly_convective_column(self):
        """Windy, weakly-convective column (u* >> w*): the nonlocal
        countergradient must stay bounded (the old pure-w* denominator
        inflated gamma_c by w_s0/w* ~ u*/w* >> 1 there).  Physical bound:
        no level warms/cools faster than 100 K/day from boundary-layer
        mixing in a barely-unstable, strongly-sheared column.
        """
        ncol, nlev = 2, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        # Strong wind (large u*), tiny surface heating (tiny w*).
        u = u + 10.0
        T_sfc = T[:, -1] + 0.2   # barely unstable
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=YSUConfig(),
        )
        assert bool(jnp.all(jnp.isfinite(out.dT_dt)))
        max_dT_day = float(jnp.max(jnp.abs(out.dT_dt))) * 86400.0
        assert max_dT_day < 100.0, (
            f"windy weakly-convective column: |dT_dt| = {max_dT_day:.1f} "
            "K/day — countergradient blow-up (w* denominator?)"
        )


# ===========================================================================
# EDMF tests
# ===========================================================================

class TestEDMF:
    """Tests for EDMF eddy-diffusivity mass-flux turbulence."""

    def test_output_shapes(self):
        """EDMF output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TurbulentEDMFConfig()

        out, tke_new = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """EDMF should produce nonzero tendencies."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TurbulentEDMFConfig()

        out, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through EDMF turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TurbulentEDMFConfig()

        def loss(T_in):
            out, _ = edmf_turbulence(
                u, v, T_in, q_v, tke, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_finite_outputs(self):
        """All outputs should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TurbulentEDMFConfig()

        out, tke_new = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.Km))
        assert jnp.all(jnp.isfinite(tke_new))

    def test_returns_tke(self):
        """EDMF should return a TKE array with correct shape."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TurbulentEDMFConfig()

        out, tke_new = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert isinstance(tke_new, jax.Array)
        assert tke_new.shape == (ncol, nlev)
        assert jnp.all(tke_new >= config.tke_min)

    def test_mass_flux_active(self):
        """Nonzero MF contribution with warm (unstable) surface."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 15.0  # very warm surface for strong updrafts
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)

        # With mass flux
        config_mf = TurbulentEDMFConfig(a_updraft=0.1)
        out_mf, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_mf,
        )

        # Without mass flux (zero updraft area)
        config_no_mf = TurbulentEDMFConfig(a_updraft=0.0)
        out_no_mf, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_no_mf,
        )

        # T tendencies should differ when MF is active
        assert not jnp.allclose(out_mf.dT_dt, out_no_mf.dT_dt, atol=1e-10)




# ===========================================================================
# PBL Height Diagnosis tests (Task 9)
# ===========================================================================

class TestPBLHeight:
    """Tests for PBL height diagnosis via bulk Richardson method."""

    def test_bulk_richardson_shape(self):
        """Bulk Ri should have shape (ncol, nlev)."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        Ri_bulk, theta_v = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
        assert Ri_bulk.shape == (ncol, nlev)
        assert theta_v.shape == (ncol, nlev)

    def test_bulk_richardson_surface_zero(self):
        """Ri at the surface level should be near zero (no buoyancy difference)."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        Ri_bulk, _ = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
        # Bottom level (surface): theta_v(sfc) - theta_v(sfc) = 0 so Ri ~ 0
        assert float(jnp.max(jnp.abs(Ri_bulk[:, -1]))) < 0.1

    def test_bulk_richardson_finite(self):
        """Ri should be finite everywhere."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        Ri_bulk, theta_v = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
        assert jnp.all(jnp.isfinite(Ri_bulk))
        assert jnp.all(jnp.isfinite(theta_v))

    def test_diagnose_pbl_height_shape(self):
        """h_pbl should have shape (ncol,)."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
        assert h_pbl.shape == (ncol,)

    def test_diagnose_pbl_height_bounds(self):
        """h_pbl should be within [h_min, h_max]."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        config = PBLHeightConfig(h_min=100.0, h_max=5000.0)

        h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full, config)
        assert jnp.all(h_pbl >= config.h_min)
        assert jnp.all(h_pbl <= config.h_max)

    def test_diagnose_pbl_height_finite(self):
        """h_pbl should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
        assert jnp.all(jnp.isfinite(h_pbl))

    def test_interp_method_shape_and_bounds(self):
        """Interp method should return correct shape within bounds."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        config = PBLHeightConfig(h_min=100.0, h_max=5000.0)

        h_pbl = diagnose_pbl_height_interp(T, q_v, u, v, p_full, z_full, config)
        assert h_pbl.shape == (ncol,)
        assert jnp.all(h_pbl >= config.h_min)
        assert jnp.all(h_pbl <= config.h_max)
        assert jnp.all(jnp.isfinite(h_pbl))

    def test_strong_shear_deeper_pbl(self):
        """Stronger wind shear should produce a deeper PBL (more Ri < Ri_crit)."""
        ncol, nlev = 4, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        # Weak wind -> shallow PBL
        u_weak = u * 0.2
        v_weak = v * 0.2
        h_weak = diagnose_pbl_height(T, q_v, u_weak, v_weak, p_full, z_full)

        # Strong wind -> deeper PBL
        u_strong = u * 3.0
        v_strong = v * 3.0
        h_strong = diagnose_pbl_height(T, q_v, u_strong, v_strong, p_full, z_full)

        # Stronger shear means larger V^2, so Ri is smaller -> more levels
        # where Ri < Ri_crit -> deeper PBL
        assert float(jnp.mean(h_strong)) > float(jnp.mean(h_weak))

    def test_barotropic_wind_does_not_deepen_pbl(self):
        """Adding a uniform wind offset (no extra shear) should not change PBL height.

        The bulk Ri denominator must use wind shear from surface, not absolute
        wind speed.  A barotropic wind adds the same vector to every level,
        so the shear is unchanged and the PBL height should be identical.
        """
        ncol, nlev = 4, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        h_base = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
        # Add a large uniform barotropic wind (no shear added)
        h_baro = diagnose_pbl_height(T, q_v, u + 50.0, v + 30.0, p_full, z_full)

        # PBL height should be essentially unchanged
        assert jnp.allclose(h_base, h_baro, rtol=1e-4), (
            f"Barotropic wind changed PBL: {float(jnp.mean(h_base)):.1f} -> "
            f"{float(jnp.mean(h_baro)):.1f}"
        )

    def test_differentiable(self):
        """jax.grad should work through PBL height diagnosis."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        def loss(T_in):
            h = diagnose_pbl_height(T_in, q_v, u, v, p_full, z_full)
            return jnp.sum(h ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_backends_return_h_pbl(self):
        """All turbulence backends should return h_pbl in TurbulenceOutput."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)

        # Smagorinsky
        out = smagorinsky_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=SmagorinskyConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # Louis
        out = louis_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=LouisConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # YSU
        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=YSUConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # Holtslag-Boville
        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=HoltslagBovilleConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # TKE
        out, _ = tke_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=TKEConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # EDMF
        out, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=TurbulentEDMFConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))
