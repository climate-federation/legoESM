"""Phase 3 of the density-Jacobian PGF (Shchepetkin & McWilliams 2003).

Tests the full ``density_jacobian_pgf_smc03_x/y`` operators in
``legoesm.ocean.dynamics.latlon_cgrid_operators``.

Four checks:

1. **Shape & boundary**: ``_x`` returns ``(n_lat, n_lon+1, nlev)`` with
   periodic wrap; ``_y`` returns ``(n_lat+1, n_lon, nlev)`` with
   zero pole rows.
2. **Linear-ρ stepped-bathymetry rest state**: SMC03 gives
   **machine-zero** PGF; Adcroft (the legacy face correction) leaves
   a residual that is orders of magnitude larger.  This is the
   load-bearing property — for the BH stratification (linear T(z) →
   linear ρ(z) under linear EOS), SMC03 closes the gap by
   construction.
3. **Smoothness in k**: at a face between adjacent partial-cell
   columns, the per-level PGF is a smooth function of level — no
   single-level spike like the Adcroft correction at the
   partial-bottom level.  This is what kills the 2Δz computational
   mode that the prior session diagnosed in
   ``partial_cells_results.md``.
4. **AD smoothness**: ``jax.grad`` of a sum-of-squared-PGF wrt T-driven
   ρ is finite end-to-end through Phase 1 + Phase 2 + Phase 3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    density_jacobian_pgf_smc03_x,
    density_jacobian_pgf_smc03_y,
    partial_cell_pgf_correction_x,
    partial_cell_pgf_correction_y,
    gradient_x_cgrid,
    gradient_y_cgrid,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


G = 9.80616


# ---------------------------------------------------------------------------
# 1. Shape & boundary
# ---------------------------------------------------------------------------


class TestShapeAndBoundary:

    def test_x_shape_and_periodic_wrap(self):
        grid = create_latlon_grid(n_lat=8, n_lon=16)
        nlev = 5
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=2000.0, dz_surface=20.0, dz_deep=600.0,
        )
        H = jnp.full((8, 16), 2000.0)
        partial = create_partial_cell_coordinate(z_coord, H)
        rho_prime = jnp.full((8, 16, nlev), -1.0)

        out = density_jacobian_pgf_smc03_x(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )
        assert out.shape == (8, 17, nlev)
        # Periodic wrap: face at j=n_lon should equal face at j=0.
        np.testing.assert_allclose(np.array(out[:, 0, :]), np.array(out[:, -1, :]))

    def test_y_shape_and_pole_walls(self):
        grid = create_latlon_grid(n_lat=8, n_lon=16)
        nlev = 5
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=2000.0, dz_surface=20.0, dz_deep=600.0,
        )
        H = jnp.full((8, 16), 2000.0)
        partial = create_partial_cell_coordinate(z_coord, H)
        # Random rho_prime so non-trivial values are non-zero
        rho_prime = jnp.asarray(
            np.random.default_rng(0).uniform(-2, 2, size=(8, 16, nlev))
        )

        out = density_jacobian_pgf_smc03_y(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )
        assert out.shape == (9, 16, nlev)
        # Pole rows i=0 and i=n_lat must be exactly zero (wall BC)
        assert float(jnp.max(jnp.abs(out[0]))) == 0.0
        assert float(jnp.max(jnp.abs(out[-1]))) == 0.0


# ---------------------------------------------------------------------------
# Helpers for the rest-state tests
# ---------------------------------------------------------------------------


def _make_step_bathymetry(grid, H_deep=2000.0, H_shallow=1500.0):
    n_lat, n_lon = grid.n_lat, grid.n_lon
    H = jnp.full((n_lat, n_lon), H_deep)
    H = H.at[: n_lat // 2, :].set(H_shallow)
    return H


def _uniform_dz_coord(n_levels=10, H_max=2000.0):
    """Uniform-dz vertical coordinate.  Picking dz_surface=dz_deep gives
    ``z_full_ref[k] = (k + 0.5) · dz`` so we can construct partial-cell
    columns where ``z_full_ref[bot]`` is unambiguously inside the
    partial cell."""
    dz = H_max / n_levels
    return create_ocean_z_star(
        n_levels=n_levels, H_max=H_max, dz_surface=dz, dz_deep=dz,
    )


def _linear_rho_per_cell(centroid_depth, slope, rho_offset):
    """ρ' = slope · z_centroid + rho_offset (positive z downward)."""
    return slope * centroid_depth + rho_offset


# ---------------------------------------------------------------------------
# 2. Linear-ρ stepped bathymetry: SMC03 → machine-zero PGF
# ---------------------------------------------------------------------------


class TestLinearRhoSteppedBathymetryMachineZero:
    """For a linear ρ(z) initial condition on stepped bathymetry (so
    columns have different ``bottom_level`` and different
    partial-cell thicknesses), the SMC03 horizontal Jacobian must be
    machine-zero.  This is the load-bearing rest-state property: the
    harmonic-mean σ recovers the analytic slope in every column,
    every column reconstructs the same ρ(z), and the integral at
    ``z_full_ref[k]`` is identical at any pair of columns at any
    level k.

    The legacy Adcroft correction, by contrast, treats ρ as
    piecewise-constant within each cell and only shifts pressures by
    ``ρ_k · g · (centroid_offset)`` — leaving an O(h · ρ' ·
    centroid_offset) residual.  We verify SMC03 beats it by orders of
    magnitude.
    """

    def test_smc03_pgf_machine_zero(self):
        grid = create_latlon_grid(n_lat=6, n_lon=12)
        nlev = 10
        z_coord = _uniform_dz_coord(n_levels=nlev, H_max=2000.0)
        # Bathymetry: deep column has a full bottom (no partial),
        # shallow column has bot_level=8 with partial thickness 150m.
        # |z_full_ref[8]| = 1700 < H_shallow = 1750 so the SMC03
        # face-reference depth is unambiguously inside the shallow
        # column's partial cell.
        H = _make_step_bathymetry(grid, H_deep=2000.0, H_shallow=1750.0)
        partial = create_partial_cell_coordinate(z_coord, H)

        centroid = compute_centroid_depth(jnp.zeros_like(H), H, partial)
        rho_prime = _linear_rho_per_cell(centroid, slope=2.5e-3, rho_offset=-1.0)

        # SMC03 PGF: should be machine zero in the active region.
        dpx_smc = density_jacobian_pgf_smc03_x(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )
        dpy_smc = density_jacobian_pgf_smc03_y(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )

        # Active levels: the shallow column has its partial bottom at
        # ``bot_shallow``; levels 0..bot_shallow are inside both
        # columns.  By choice of ``H_shallow`` (above), the SMC03
        # face-reference depth ``|z_full_ref[bot_shallow]|`` lies inside
        # the partial cell, so the partial level itself is also a
        # legitimate target of the machine-zero check.
        bot_shallow = int(jnp.min(partial.bottom_level))
        if bot_shallow < 1:
            pytest.fail("Test fixture has no levels above the shallow seafloor.")

        smc_active = dpx_smc[..., : bot_shallow + 1]
        smc_active_y = dpy_smc[..., : bot_shallow + 1]
        smc_max = float(jnp.max(jnp.abs(smc_active)))
        smc_y_max = float(jnp.max(jnp.abs(smc_active_y)))
        # Machine-zero target: linear ρ(z), harmonic σ exact, integral exact.
        assert smc_max < 1.0e-12, (
            f"SMC03 PGF |x| = {smc_max} on linear ρ — expected machine zero."
        )
        assert smc_y_max < 1.0e-12, (
            f"SMC03 PGF |y| = {smc_y_max} on linear ρ — expected machine zero."
        )

    def test_smc03_dramatically_better_than_adcroft(self):
        """Same setup as above; compare SMC03 vs (gradient + Adcroft).

        For a linear ρ(z), SMC03 is machine-zero while Adcroft is
        non-zero (residual scales with the centroid offset between
        adjacent partial cells).  The ratio should be at least 1e6.
        """
        grid = create_latlon_grid(n_lat=6, n_lon=12)
        nlev = 10
        z_coord = _uniform_dz_coord(n_levels=nlev, H_max=2000.0)
        H = _make_step_bathymetry(grid, H_deep=2000.0, H_shallow=1750.0)
        partial = create_partial_cell_coordinate(z_coord, H)

        centroid = compute_centroid_depth(jnp.zeros_like(H), H, partial)
        rho_prime = _linear_rho_per_cell(centroid, slope=2.5e-3, rho_offset=-1.0)

        # Build the Adcroft path: standard p' cumsum at centroids + face correction.
        # p'[k] = sum_{j<k} ρ'_j g h_j + 0.5 ρ'_k g h_k  (from
        # iterate_eos_and_pressure_anomaly).
        h = partial.h_partial
        dp_layer = rho_prime * G * h
        p_prime = jnp.cumsum(dp_layer, axis=-1) - 0.5 * dp_layer
        # The bathymetry varies in latitude only, so we look at the
        # y-direction PGF where the gradient is non-trivial.
        dpy_grad = gradient_y_cgrid(p_prime, grid)
        dpy_corr = partial_cell_pgf_correction_y(centroid, rho_prime, grid, G)
        dpy_adcroft = dpy_grad + dpy_corr

        dpy_smc = density_jacobian_pgf_smc03_y(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )

        bot_shallow = int(jnp.min(partial.bottom_level))
        smc_max = float(jnp.max(jnp.abs(dpy_smc[..., : bot_shallow + 1])))
        adcroft_max = float(
            jnp.max(jnp.abs(dpy_adcroft[..., : bot_shallow + 1]))
        )
        # SMC03 should beat Adcroft by many orders of magnitude on linear ρ.
        assert smc_max < adcroft_max * 1.0e-6, (
            f"SMC03 ({smc_max:.3e}) not orders better than Adcroft "
            f"({adcroft_max:.3e})."
        )


# ---------------------------------------------------------------------------
# 3. Smoothness in k at a partial-cell face
# ---------------------------------------------------------------------------


class TestSmoothnessInK:
    """At a partial-cell face the SMC03 PGF varies smoothly with k.
    The Adcroft correction has a single-level spike at the partial-
    bottom level — it is identically zero everywhere except that
    level.  SMC03 has no such spike: the per-level |PGF| profile is
    bounded by O(local ρ-curvature × dz²) and varies smoothly with
    level index.
    """

    def test_smc03_no_single_level_spike(self):
        grid = create_latlon_grid(n_lat=6, n_lon=12)
        nlev = 12
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=4000.0, dz_surface=20.0, dz_deep=500.0,
        )
        # Stepped bathymetry where columns differ by ~one half-cell
        # at level ~6.
        H = _make_step_bathymetry(grid, H_deep=2200.0, H_shallow=1700.0)
        partial = create_partial_cell_coordinate(z_coord, H)

        centroid = compute_centroid_depth(jnp.zeros_like(H), H, partial)
        # Use a NON-LINEAR ρ profile so the residual is non-zero
        # (otherwise both schemes get 0).  Exponential thermocline.
        rho_prime = -1.0 * jnp.exp(-centroid / 700.0) - 0.5

        dpx_smc = density_jacobian_pgf_smc03_x(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )
        dpx_adc = (
            gradient_x_cgrid(
                jnp.cumsum(rho_prime * G * partial.h_partial, axis=-1)
                - 0.5 * rho_prime * G * partial.h_partial,
                grid,
            )
            + partial_cell_pgf_correction_x(centroid, rho_prime, grid, G)
        )

        # Find a face between adjacent (deep, shallow) columns: the
        # transition row in latitude.  Pick lat=n_lat//2-1 (last shallow
        # row) where the equator-step makes north of it deep.  Use a
        # u-face — but here the step is in latitude so x-face won't
        # actually probe the transition.  Use y-face instead.
        dpy_smc = density_jacobian_pgf_smc03_y(
            rho_prime, partial.h_partial, partial.is_active,
            partial.z_full_ref, grid, G,
        )
        dpy_adc = (
            gradient_y_cgrid(
                jnp.cumsum(rho_prime * G * partial.h_partial, axis=-1)
                - 0.5 * rho_prime * G * partial.h_partial,
                grid,
            )
            + partial_cell_pgf_correction_y(centroid, rho_prime, grid, G)
        )

        # Transition v-face is at i = n_lat//2 (between shallow and deep).
        # Pick a column j=0 and look at the per-level |PGF|.
        i_face = grid.n_lat // 2
        bot_shallow = int(jnp.min(partial.bottom_level))
        smc_profile = jnp.abs(dpy_smc[i_face, 0, : bot_shallow + 1])
        adc_profile = jnp.abs(dpy_adc[i_face, 0, : bot_shallow + 1])

        # "Spike" metric: max-to-mean ratio.  A pure single-level
        # spike has ratio ≈ nlev_active.  A smooth profile has
        # ratio ≈ 1 to a few.
        smc_max = float(jnp.max(smc_profile))
        smc_mean = float(jnp.mean(smc_profile))
        adc_max = float(jnp.max(adc_profile))
        adc_mean = float(jnp.mean(adc_profile))

        smc_ratio = smc_max / max(smc_mean, 1.0e-30)
        adc_ratio = adc_max / max(adc_mean, 1.0e-30)
        # Adcroft's correction is concentrated at the partial-bottom level
        # (the in-cell face PGF is ≈ 0 elsewhere except the standard
        # gradient piece — but for a horizontally-uniform stratification
        # the gradient is zero, leaving only the spike).  So adc_ratio
        # ought to be much larger than smc_ratio.
        assert smc_ratio < adc_ratio, (
            f"SMC03 not smoother than Adcroft: SMC ratio={smc_ratio:.2f}, "
            f"Adcroft ratio={adc_ratio:.2f}."
        )


# ---------------------------------------------------------------------------
# 4. AD smoothness through the full chain
# ---------------------------------------------------------------------------


class TestADSmoothness:

    def test_grad_wrt_rho_finite(self):
        grid = create_latlon_grid(n_lat=6, n_lon=12)
        nlev = 8
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=2000.0, dz_surface=50.0, dz_deep=400.0,
        )
        H = _make_step_bathymetry(grid, H_deep=1800.0, H_shallow=1200.0)
        partial = create_partial_cell_coordinate(z_coord, H)
        centroid = compute_centroid_depth(jnp.zeros_like(H), H, partial)
        rho_prime = _linear_rho_per_cell(centroid, slope=2.5e-3, rho_offset=-1.0)

        def loss(rho):
            dpx = density_jacobian_pgf_smc03_x(
                rho, partial.h_partial, partial.is_active,
                partial.z_full_ref, grid, G,
            )
            dpy = density_jacobian_pgf_smc03_y(
                rho, partial.h_partial, partial.is_active,
                partial.z_full_ref, grid, G,
            )
            return jnp.sum(dpx ** 2) + jnp.sum(dpy ** 2)

        grad = jax.grad(loss)(rho_prime)
        assert jnp.all(jnp.isfinite(grad))
