"""Tests for the Visbeck (1997) adaptive GM coefficient (issue #192).

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_visbeck_gm.py -v
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    VisbeckConfig,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi import (
    compute_visbeck_kappa_gm,
    gm_redi_lateral_mixing,
    _compute_tapered_slopes,
    _tracer_tendency_gm_redi,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _make_setup(n=4, nlev=10):
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (6, n, n, nlev)
    jacobian = jnp.ones((6, n, n))
    return grid, z_coord, jacobian, shape


def _stratified_fields(n=4, nlev=10, slope=1e-4):
    """Meridionally tilted isopycnals with uniform stratification.

    Builds a density field whose mean vertical gradient is set so that
    ``N² ≈ g·(ρ(top)-ρ(bottom))/(H·ρ_0)`` is a sensible 1e-5 s⁻², then
    adds a horizontal component matching the requested slope.
    """
    grid, z_coord, jacobian, shape = _make_setup(n=n, nlev=nlev)
    # Vertical profile: 1025 at surface -> 1027 at bottom, so rho[0] < rho[-1]
    # giving N² > 0.  compute_buoyancy_frequency uses +(g/ρ₀)·|drho/dz|.
    rho_z = jnp.linspace(1025.0, 1027.0, nlev)
    # Add a meridional gradient proportional to latitude index.  At slope
    # = 1e-4 and dρ/dz ≈ 2/H, horizontal gradient ≈ slope × dρ/dz.
    dz_full = z_coord.dz_ref
    H_total = float(jnp.sum(dz_full))
    drho_dz = 2.0 / H_total
    drho_dh = slope * drho_dz

    # Build a density field with a zonal gradient along the first axis of
    # each face (arbitrary but nonzero).  Broadcasting gives (6,n,n,nlev).
    n_ax = shape[1]
    grad_h = drho_dh * jnp.arange(n_ax, dtype=jnp.float64)
    rho = (rho_z[None, None, None, :]
           + grad_h[None, :, None, None]
           + jnp.zeros(shape, dtype=jnp.float64))

    T = jnp.broadcast_to(
        jnp.linspace(20.0, 5.0, nlev)[None, None, None, :], shape,
    )
    S = jnp.ones(shape) * 35.0
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)
    return grid, z_coord, jacobian, shape, u, v, T, S, rho


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestVisbeckConfig:
    def test_defaults(self):
        cfg = VisbeckConfig()
        assert cfg.enabled is False
        assert cfg.alpha == 0.015
        assert cfg.L_min < cfg.L_max
        assert cfg.kappa_min < cfg.kappa_max
        assert cfg.f_min > 0

    def test_gm_redi_config_carries_visbeck_default_off(self):
        cfg = GMRediConfig()
        assert hasattr(cfg, "visbeck")
        assert cfg.visbeck.enabled is False

    def test_visbeck_exported_from_public_packages(self):
        """Downstream callers should be able to import the new type from
        the established public entry points."""
        from legoesm.ocean.physics.lateral_mixing import VisbeckConfig as VC1
        from legoesm.ocean.physics import VisbeckConfig as VC2

        assert VC1 is VC2
        assert VC1().enabled is False

    def test_gm_redi_config_accepts_visbeck_override(self):
        vb = VisbeckConfig(enabled=True, alpha=0.05,
                           L_min=10e3, L_max=50e3,
                           kappa_min=50.0, kappa_max=2.0e3)
        cfg = GMRediConfig(visbeck=vb)
        assert cfg.visbeck.enabled is True
        assert cfg.visbeck.alpha == 0.05


# ---------------------------------------------------------------------------
# Coefficient mathematics
# ---------------------------------------------------------------------------


class TestVisbeckCoefficient:
    def test_shape_one_per_column(self):
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
        S_x, S_y, _ = _compute_tapered_slopes(
            rho, z_coord, jacobian, grid, GMRediConfig())
        kappa = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), VisbeckConfig(enabled=True))
        assert kappa.shape == jacobian.shape

    def test_non_negative_and_bounded(self):
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
        S_x, S_y, _ = _compute_tapered_slopes(
            rho, z_coord, jacobian, grid, GMRediConfig())
        cfg = VisbeckConfig(enabled=True, kappa_min=50.0, kappa_max=3000.0)
        kappa = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg)
        assert jnp.all(kappa >= cfg.kappa_min - 1e-10)
        assert jnp.all(kappa <= cfg.kappa_max + 1e-10)

    def test_zero_slope_hits_floor(self):
        """Flat isopycnals ⇒ N·|S| ≈ 0 ⇒ κ collapses to ``kappa_min``."""
        grid, z_coord, jacobian, shape = _make_setup()
        rho = jnp.broadcast_to(
            jnp.linspace(1025.0, 1027.0, shape[-1])[None, None, None, :],
            shape,
        )
        # Zero slopes by construction
        S_x = jnp.zeros(shape[:-1] + (shape[-1] - 1,))
        S_y = jnp.zeros_like(S_x)
        cfg = VisbeckConfig(enabled=True, kappa_min=123.0, kappa_max=4000.0)
        kappa = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg)
        assert jnp.allclose(kappa, cfg.kappa_min)

    def test_scales_with_alpha_below_ceiling(self):
        """Below the ceiling, κ is linear in ``alpha`` (and the sum
        stays below kappa_max here by construction)."""
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields(
            slope=1e-5)  # small slope so κ stays below the ceiling
        S_x, S_y, _ = _compute_tapered_slopes(
            rho, z_coord, jacobian, grid, GMRediConfig())
        cfg_a = VisbeckConfig(enabled=True, alpha=0.01,
                              kappa_min=1e-8, kappa_max=1e6,
                              use_rossby_radius=False, L_fixed=5e4)
        cfg_b = cfg_a._replace(alpha=0.02)
        ka = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg_a)
        kb = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg_b)
        # κ(α=0.02) / κ(α=0.01) → 2 wherever the slope is non-trivial.
        active = ka > 1.1 * cfg_a.kappa_min
        ratio = jnp.where(active, kb / jnp.where(active, ka, 1.0), 2.0)
        assert jnp.allclose(ratio, 2.0, rtol=1e-10)

    def test_quadratic_in_L_fixed(self):
        """At fixed-L, κ ∝ L² below the ceiling."""
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields(
            slope=1e-5)
        S_x, S_y, _ = _compute_tapered_slopes(
            rho, z_coord, jacobian, grid, GMRediConfig())
        cfg_a = VisbeckConfig(enabled=True, use_rossby_radius=False,
                              L_fixed=5e4, alpha=0.01,
                              kappa_min=1e-8, kappa_max=1e12)
        cfg_b = cfg_a._replace(L_fixed=1e5)
        ka = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg_a)
        kb = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg_b)
        active = ka > 1.1 * cfg_a.kappa_min
        ratio = jnp.where(active, kb / jnp.where(active, ka, 1.0), 4.0)
        assert jnp.allclose(ratio, 4.0, rtol=1e-10)

    def test_realistic_magnitude(self):
        """κ for realistic (N, S, L_ρ) is O(1e2–1e3) m²/s.

        Regression against a codex-caught bug where the buoyancy frequency
        was computed with ``rho_ref=1`` instead of ``ρ₀≈1025``, which
        inflated N² (and therefore κ) by ~10³ and saturated the ceiling.
        Pass slope fields directly (rather than building a density
        profile) so the numerics aren't confounded by grid geometry.
        """
        grid, z_coord, jacobian, shape = _make_setup(n=4, nlev=10)
        # Density profile: Δρ = 2 kg/m³ over 1000 m → N² ≈ 1.9e-5,
        # giving N ≈ 4.4e-3 s⁻¹ for ρ₀ = 1025.
        rho = jnp.broadcast_to(
            jnp.linspace(1025.0, 1027.0, shape[-1])[None, None, None, :],
            shape,
        )
        S_mag = 1.0e-3   # 1 m per 1000 m — Southern Ocean frontal scale
        S_x = jnp.full(shape[:-1] + (shape[-1] - 1,), S_mag)
        S_y = jnp.zeros_like(S_x)
        cfg = VisbeckConfig(enabled=True, alpha=0.015,
                            kappa_min=1.0, kappa_max=1.0e6)
        kappa = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg)
        k_max = float(jnp.max(kappa))
        k_mean = float(jnp.mean(kappa))
        # With ρ₀=1025 and S=1e-3, analytical κ ≈ α·L²·N·S is O(100).
        # The rho_ref=1 bug would inflate κ by ~10³ to >1e5.
        assert 1.0 < k_mean < 1.0e4, (
            f"κ_mean={k_mean:.3g} is outside the realistic [1, 1e4] m²/s "
            "window — check rho_ref and the depth-average")
        assert k_max < 1.0e5, (
            f"κ_max={k_max:.3g} — ceiling bug or inflated N²?")

    def test_nbar_uses_arithmetic_mean(self):
        """With a depth-varying N, κ uses ⟨N⟩ (not ``sqrt(⟨N²⟩)``).

        Construct a density field whose N² is strongly concentrated near
        the surface (factor-of-10 between top and bottom interfaces).
        The correct Rossby-radius length uses ``⟨N⟩``; the buggy form
        ``sqrt(⟨N²⟩)`` is strictly larger and would inflate κ.
        """
        grid, z_coord, jacobian, shape = _make_setup(n=4, nlev=10)
        # Density profile crafted so N² is large at the surface and small
        # below: ρ(z) = ρ₀ + a·(1 - exp(-z/z0))·Δρ with z0 shallow.
        nlev = shape[-1]
        z_centers = jnp.cumsum(z_coord.dz_ref) - 0.5 * z_coord.dz_ref
        H_max = float(jnp.sum(z_coord.dz_ref))
        z0 = 100.0
        rho_prof = 1025.0 + 2.0 * (1.0 - jnp.exp(-z_centers / z0))
        rho = jnp.broadcast_to(rho_prof[None, None, None, :], shape)

        S_x = jnp.full(shape[:-1] + (nlev - 1,), 1.0e-3)
        S_y = jnp.zeros_like(S_x)
        cfg = VisbeckConfig(enabled=True, alpha=0.015,
                            kappa_min=1.0, kappa_max=1.0e8)

        kappa = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian,
            jnp.asarray(grid.grid_coriolis), cfg)

        # Sanity: value is finite and far below what the RMS-bias form
        # would give.  A local check against a manual ⟨N⟩ average:
        from legoesm.ocean.eos import compute_buoyancy_frequency
        from legoesm import constants

        N2 = compute_buoyancy_frequency(
            rho, z_coord.dz_ref, jacobian, rho_ref=1025.0, g=constants.g)
        N = jnp.sqrt(jnp.maximum(N2, 0.0))
        dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
        dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
        N_bar_arith = (jnp.sum(N * dz_half, axis=-1)
                       / jnp.sum(dz_half, axis=-1))
        N2_bar = jnp.sum(N2 * dz_half, axis=-1) / jnp.sum(dz_half, axis=-1)
        N_bar_rms = jnp.sqrt(jnp.maximum(N2_bar, 0.0))

        ratio = float(jnp.max(N_bar_rms / jnp.maximum(N_bar_arith, 1e-30)))
        # For this profile the RMS form is at least 10 % larger — ensures
        # the test is actually discriminating between the two.
        assert ratio > 1.1, (
            f"Test profile too uniform to distinguish means; ratio={ratio}")

        f_safe = jnp.maximum(
            jnp.abs(jnp.asarray(grid.grid_coriolis)), cfg.f_min)
        H_col = jnp.sum(dz_actual, axis=-1)
        L_arith = jnp.clip(
            N_bar_arith * H_col / f_safe, cfg.L_min, cfg.L_max)
        sigma_bar = (jnp.sum(N * 1.0e-3 * dz_half, axis=-1)
                     / jnp.sum(dz_half, axis=-1))
        kappa_expected = jnp.clip(
            cfg.alpha * L_arith ** 2 * sigma_bar,
            cfg.kappa_min, cfg.kappa_max)
        # Require agreement with the arithmetic-mean formula.
        assert jnp.allclose(kappa, kappa_expected, rtol=1e-8, atol=1e-8)

    def test_finite_and_differentiable(self):
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
        cfg = VisbeckConfig(enabled=True)
        f_cor = jnp.asarray(grid.grid_coriolis)

        def loss(rho_in):
            S_x, S_y, _ = _compute_tapered_slopes(
                rho_in, z_coord, jacobian, grid, GMRediConfig())
            k = compute_visbeck_kappa_gm(
                rho_in, S_x, S_y, z_coord, jacobian, f_cor, cfg)
            return jnp.sum(k ** 2)

        grad = jax.grad(loss)(rho)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# End-to-end GM/Redi pipeline with Visbeck
# ---------------------------------------------------------------------------


class TestGMRediWithVisbeck:
    def test_tendency_shapes_and_finite(self):
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
        cfg = GMRediConfig(visbeck=VisbeckConfig(enabled=True))
        out = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg)
        assert out.dT_dt.shape == shape
        assert out.dS_dt.shape == shape
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.dS_dt))

    def test_disabled_matches_scalar_exactly(self):
        """With ``visbeck.enabled = False`` the output is exactly the
        same as the pre-#192 code path (scalar ``kappa_GM``)."""
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
        cfg_off = GMRediConfig(kappa_GM=750.0, kappa_Redi=1e3)
        cfg_on = GMRediConfig(
            kappa_GM=750.0, kappa_Redi=1e3,
            visbeck=VisbeckConfig(enabled=False, alpha=0.05),
        )
        o1 = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg_off)
        o2 = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg_on)
        assert jnp.allclose(o1.dT_dt, o2.dT_dt, rtol=0, atol=0)
        assert jnp.allclose(o1.dS_dt, o2.dS_dt, rtol=0, atol=0)

    def test_broadcast_array_kappa_matches_scalar(self):
        """Feeding a constant ``kappa_GM`` array to
        ``_tracer_tendency_gm_redi`` reproduces the scalar result."""
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
        cfg = GMRediConfig()
        S_x, S_y, _ = _compute_tapered_slopes(
            rho, z_coord, jacobian, grid, cfg)

        scalar_k = 650.0
        arr_k = jnp.full(jacobian.shape, scalar_k)

        dT_scalar = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid, scalar_k, cfg.kappa_Redi)
        dT_array = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid, arr_k, cfg.kappa_Redi)
        assert jnp.allclose(dT_scalar, dT_array, rtol=1e-12, atol=1e-14)

    def test_enabling_changes_output_when_baroclinic(self):
        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields(
            slope=5e-4)  # strong slope → visbeck κ >> floor
        cfg_scalar = GMRediConfig(kappa_GM=300.0, kappa_Redi=1e3)
        cfg_visbeck = GMRediConfig(
            kappa_GM=300.0, kappa_Redi=1e3,
            visbeck=VisbeckConfig(enabled=True, alpha=0.03,
                                   kappa_min=100.0, kappa_max=3e3),
        )
        o1 = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg_scalar)
        o2 = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg_visbeck)
        max_diff = float(jnp.max(jnp.abs(o1.dT_dt - o2.dT_dt)))
        assert max_diff > 0.0

    def test_visbeck_kappa_grad_finite_in_unstable_column(self):
        """``compute_visbeck_kappa_gm`` must produce finite gradients
        even when the column has statically unstable layers (N²<0).

        Regression guard for ``sqrt(max(N², 0))``: in JAX, ``sqrt(0)``
        has an infinite backward derivative, and chained with the
        ``maximum(., 0)`` mask whose VJP is zero on the negative side,
        the autodiff produces ``inf*0 = NaN``.  The fixed form
        ``sqrt(max(N², 1e-30))`` keeps N tiny but positive, giving a
        finite (very large) derivative which is still gradient-zero
        through the ``maximum`` VJP — well-defined zero, not NaN.
        """
        nlev = 6
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
        shape = (4, nlev)
        # Column with N² < 0 in mid-layer (unstable).
        T = jnp.array([20.0, 5.0, 18.0, 8.0, 3.0, 1.0]).astype(jnp.float64)
        T = jnp.broadcast_to(T, shape)
        from legoesm.ocean.eos import wright_eos
        S = jnp.full(shape, 35.0, dtype=jnp.float64)
        p = jnp.full(shape, 1e7, dtype=jnp.float64)
        rho = wright_eos(T, S, p)

        S_x = jnp.full((4, nlev - 1), 1e-3, dtype=jnp.float64)
        S_y = jnp.zeros((4, nlev - 1), dtype=jnp.float64)
        J = jnp.ones(4, dtype=jnp.float64)
        f_coriolis = jnp.full(4, 1e-4, dtype=jnp.float64)
        cfg = VisbeckConfig()

        def _loss(rho_in):
            kappa = compute_visbeck_kappa_gm(
                rho_in, S_x, S_y, z_coord, J, f_coriolis, cfg,
            )
            return jnp.sum(kappa)

        g = jax.grad(_loss)(rho)
        assert not bool(jnp.any(jnp.isnan(g))), (
            f"Visbeck kappa gradient contains NaN — sqrt(max(N²,0)) "
            f"backward pass blew up at N²<0.  Use sqrt(max(N², 1e-30)).\n"
            f"Gradient: {g}"
        )
        assert not bool(jnp.any(jnp.isinf(g))), (
            f"Visbeck kappa gradient contains Inf — likely sqrt(0) "
            f"backward.  Gradient: {g}"
        )
