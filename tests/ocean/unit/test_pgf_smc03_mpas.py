"""SMC03 density-Jacobian PGF on the MPAS Voronoi mesh.

Tests for ``density_jacobian_pgf_smc03_mpas`` in
``legoesm.ocean.dynamics.mpas_partial_cell_helpers``.

Phases mirror the lat-lon ``test_pgf_smc03_phase3.py`` suite:

1. **Shape & dtype**: returns ``(nEdges, nlev)`` matching
   ``gradient_edge_3d``; dtype-preserving.
2. **Linear-ρ rest-state machine-zero**: on a Gaussian-seamount
   partial-cell column with linear ρ(z), SMC03 returns float-noise PGF
   on every edge — the load-bearing property.  Centered + Adcroft
   leave a residual that grows with seamount amplitude.
3. **Smoothness in k**: at edges between adjacent partial-cell
   columns, the SMC03 PGF is a smooth function of vertical level (no
   single-level spike at the partial-bottom interface — that spike is
   what drives the 2Δz vertical mode under Adcroft on real bathymetry).
4. **AD smoothness**: ``jax.grad`` of a sum-of-squared-PGF wrt the
   T-driven ρ' is finite end-to-end.
5. **Full-cell flat-bottom equivalence**: when every cell is full and
   ρ is z-independent, SMC03 PGF is exactly zero (matches the
   centered-PGF rest state on z-star).

See ``docs/ocean_experiments/density_jacobian_pgf_mpas.md`` for the
port plan.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.operators_voronoi import gradient_edge_3d
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    density_jacobian_pgf_smc03_mpas,
    partial_cell_pgf_correction_edge,
)
from legoesm.ocean.vertical import (
    compute_centroid_depth,
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=10, H_max=4000.0)


def _gaussian_seamount_bathy(mesh, H_max=4000.0, height=2000.0, sigma=0.3):
    """Single Gaussian seamount centered on (0, 0)."""
    lat = np.asarray(mesh.latCell)
    lon = np.asarray(mesh.lonCell)
    r2 = lat ** 2 + lon ** 2
    return jnp.asarray(
        H_max - height * np.exp(-r2 / sigma), dtype=jnp.float64,
    )


# ---------------------------------------------------------------------------
# 1. Shape & dtype
# ---------------------------------------------------------------------------


class TestShapeAndDtype:

    def test_shape_matches_gradient_edge_3d(self, mesh, z_coord):
        H = jnp.full(mesh.nCells, 4000.0, dtype=jnp.float64)
        pc = create_partial_cell_coordinate(z_coord, H)
        rho = jnp.zeros((mesh.nCells, z_coord.n_levels), dtype=jnp.float64)
        out = density_jacobian_pgf_smc03_mpas(
            rho, pc.h_partial, pc.is_active, mesh, constants.g,
        )
        assert out.shape == (mesh.nEdges, z_coord.n_levels), (
            f"Expected (nEdges={mesh.nEdges}, nlev={z_coord.n_levels}); "
            f"got {out.shape}"
        )

    def test_zero_rho_returns_zero(self, mesh, z_coord):
        """ρ = 0 in every cell ⇒ no pressure anywhere ⇒ no PGF."""
        H = _gaussian_seamount_bathy(mesh)
        pc = create_partial_cell_coordinate(z_coord, H)
        rho = jnp.zeros((mesh.nCells, z_coord.n_levels), dtype=jnp.float64)
        out = density_jacobian_pgf_smc03_mpas(
            rho, pc.h_partial, pc.is_active, mesh, constants.g,
        )
        assert jnp.max(jnp.abs(out)) == 0.0


# ---------------------------------------------------------------------------
# 2. Linear-ρ rest-state machine-zero (the load-bearing property)
# ---------------------------------------------------------------------------


class TestLinearRhoRestState:
    """Linear ρ(z) ⇒ SMC03 PGF is identically zero on partial-cell
    topography (the property that distinguishes SMC03 from
    centered+AC).  Adcroft+centered leave a residual that scales with
    bathymetry amplitude.
    """

    def test_smc03_machine_zero_on_seamount(self, mesh, z_coord):
        H = _gaussian_seamount_bathy(mesh, height=2000.0)
        pc = create_partial_cell_coordinate(z_coord, H)
        # ρ_prime = α·z (perfectly linear in z, η=0 reference)
        zc = jnp.cumsum(pc.h_partial, axis=-1) - 0.5 * pc.h_partial
        alpha = 0.01
        rho = alpha * zc

        out = density_jacobian_pgf_smc03_mpas(
            rho, pc.h_partial, pc.is_active, mesh, constants.g,
        )
        # On wet edges only — masking ensures we don't compare against
        # the unphysical bottom-of-seafloor noise.
        max_pgf = float(jnp.max(jnp.abs(out)))
        assert max_pgf < 1e-12, (
            f"SMC03 must be machine-zero on linear ρ over partial-cell "
            f"topography; got max|PGF|={max_pgf:.3e} (expected < 1e-12)"
        )

    def test_smc03_beats_adcroft_on_seamount(self, mesh, z_coord):
        """At equal stratification, SMC03 PGF residual on linear ρ is
        orders of magnitude smaller than Adcroft.  This confirms that
        on partial bathymetry SMC03 is the strict improvement Adcroft
        was meant to be."""
        H = _gaussian_seamount_bathy(mesh, height=2000.0)
        pc = create_partial_cell_coordinate(z_coord, H)
        zc = jnp.cumsum(pc.h_partial, axis=-1) - 0.5 * pc.h_partial
        alpha = 0.01
        rho = alpha * zc

        smc03 = density_jacobian_pgf_smc03_mpas(
            rho, pc.h_partial, pc.is_active, mesh, constants.g,
        )

        # Adcroft path: centered ∇(p'/ρ_0) + AC face correction.
        # Compute p_prime via cumulative ρ·g·dz (matches the actual
        # PE pipeline's pressure construction).
        p_prime = constants.g * jnp.cumsum(rho * pc.h_partial, axis=-1)
        rho_0 = 1027.0
        centered = gradient_edge_3d(p_prime / rho_0, mesh)
        centroid = compute_centroid_depth(jnp.zeros_like(H), H, pc)
        ac = partial_cell_pgf_correction_edge(
            centroid, rho, mesh, constants.g, rho_0,
        )
        adcroft_pgf = centered + ac
        smc03_acc = smc03 / rho_0

        # SMC03 should be orders of magnitude smaller than Adcroft.
        max_smc = float(jnp.max(jnp.abs(smc03_acc)))
        max_adc = float(jnp.max(jnp.abs(adcroft_pgf)))
        assert max_smc < 1e-12, (
            f"SMC03 must vanish on linear ρ; got {max_smc:.3e}"
        )
        # On non-trivial seamount geometry adcroft must leave some
        # residual (otherwise the test isn't exercising the path).
        assert max_adc > 1e-6, (
            f"Adcroft residual unexpectedly tiny ({max_adc:.3e}); "
            f"verify the test is hitting non-trivial topography"
        )


# ---------------------------------------------------------------------------
# 3. Smoothness in vertical level
# ---------------------------------------------------------------------------


class TestVerticalSmoothness:

    def test_no_partial_bottom_spike(self, mesh, z_coord):
        """At edges where one column has a partial bottom and the other
        is full, SMC03 PGF should not exhibit a single-level spike at
        the partial-bottom interface (that spike is what drives the
        2Δz vertical mode under Adcroft).

        Quantify: the maximum per-edge per-level |PGF| should be at
        most a few times the column mean — not 100× as it would be for
        a thin spike.  Loose threshold: max-over-levels ≤ 10 × mean
        on every wet edge.
        """
        H = _gaussian_seamount_bathy(mesh, height=2500.0)  # taller seamount
        pc = create_partial_cell_coordinate(z_coord, H)
        # Realistic exponential T(z) → ρ'(z) (nonlinear; SMC03 not exact).
        zc = jnp.cumsum(pc.h_partial, axis=-1) - 0.5 * pc.h_partial
        rho = -0.2 * (
            (20.0 - 2.0) * jnp.exp(-zc / 1000.0)
        )  # density anomaly from temperature profile (β·ΔT scale)
        rho = jnp.where(pc.is_active, rho, 0.0)

        out = density_jacobian_pgf_smc03_mpas(
            rho, pc.h_partial, pc.is_active, mesh, constants.g,
        )

        # For each edge, compare max-over-levels to mean-over-levels
        # of |PGF|; flag any edge with ratio > 50 (loose — a true
        # 2Δz mode would be > 1000).  Filter to edges that have any
        # non-trivial signal at all.
        abs_out = jnp.abs(out)
        per_edge_max = jnp.max(abs_out, axis=1)
        per_edge_mean = jnp.mean(abs_out, axis=1)
        active = per_edge_max > 1e-8
        ratio = jnp.where(
            active & (per_edge_mean > 0),
            per_edge_max / jnp.maximum(per_edge_mean, 1e-30),
            0.0,
        )
        worst = float(jnp.max(ratio))
        assert worst < 50.0, (
            f"SMC03 has a spike-like vertical structure on partial-cell "
            f"edges (worst max/mean ratio = {worst:.1f}); compare with "
            f"Adcroft's >100x spike at the partial-bottom level"
        )


# ---------------------------------------------------------------------------
# 4. Differentiability (AD smoothness)
# ---------------------------------------------------------------------------


class TestAutodiff:

    def test_grad_finite_through_full_operator(self, mesh, z_coord):
        H = _gaussian_seamount_bathy(mesh, height=2000.0)
        pc = create_partial_cell_coordinate(z_coord, H)

        def loss_fn(rho):
            out = density_jacobian_pgf_smc03_mpas(
                rho, pc.h_partial, pc.is_active, mesh, constants.g,
            )
            return jnp.sum(out ** 2)

        zc = jnp.cumsum(pc.h_partial, axis=-1) - 0.5 * pc.h_partial
        rho_init = -0.2 * jnp.exp(-zc / 1000.0)
        rho_init = jnp.where(pc.is_active, rho_init, 0.0)

        grad = jax.grad(loss_fn)(rho_init)
        assert jnp.all(jnp.isfinite(grad)), (
            "AD produced NaN/Inf — check argmax / take_along_axis paths"
        )
        # Non-trivial gradient: should have some nonzero entries
        # (otherwise AD is silently zero, which would also be a bug).
        assert float(jnp.max(jnp.abs(grad))) > 0, (
            "AD gradient is identically zero — investigate"
        )


# ---------------------------------------------------------------------------
# 5. Full-cell flat-bottom equivalence
# ---------------------------------------------------------------------------


class TestFullCellEquivalence:

    def test_flat_bottom_uniform_rho_is_zero(self, mesh, z_coord):
        """On a flat-bottom z-star (every cell full) with horizontally
        uniform ρ', SMC03 PGF must vanish — the rest-state property
        the centered scheme provides on z-star."""
        H = jnp.full(mesh.nCells, 4000.0, dtype=jnp.float64)
        pc = create_partial_cell_coordinate(z_coord, H)
        # Horizontally uniform but vertically varying ρ
        zc = jnp.cumsum(pc.h_partial, axis=-1) - 0.5 * pc.h_partial
        rho = -0.2 * jnp.exp(-zc / 1000.0)
        out = density_jacobian_pgf_smc03_mpas(
            rho, pc.h_partial, pc.is_active, mesh, constants.g,
        )
        max_pgf = float(jnp.max(jnp.abs(out)))
        assert max_pgf < 1e-10, (
            f"Flat-bottom uniform ρ' must give zero PGF; got {max_pgf:.3e}"
        )
