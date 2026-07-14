"""Tests for energy backscatter (issue #193).

Covers the Jansen–Held (2014) deterministic backscatter closure on the
lat-lon C-grid and the MPAS TRiSK mesh, plus the shared energy-budget
helpers.

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_backscatter.py -v
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    viscous_tendency_cgrid,
)
from legoesm.core.operators_voronoi import vector_laplacian_del2_3d
from legoesm.ocean.physics.lateral_mixing import (
    BackscatterConfig,
    backscatter_tendency_cgrid,
    backscatter_tendency_mpas,
    backscatter_power_density_cgrid,
    update_eddy_energy,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def latlon_grid():
    grid, wall_mask = create_regional_latlon_grid(
        16, 34, 16.0, 34.0,
        periodic_x=True, lon_west=0.0, lon_east=10.0,
        dtype=jnp.float64,
    )
    u_mask, v_mask = compute_face_masks(wall_mask)
    return grid, wall_mask, u_mask, v_mask


@pytest.fixture(scope="module")
def mpas_mesh():
    return create_voronoi_mesh(subdivision_level=2)


def _random_velocity_latlon(grid, wall_mask, u_mask, v_mask,
                            seed=42, amplitude=0.1):
    rng = np.random.RandomState(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    u_inner = jnp.array(rng.randn(n_lat, n_lon) * amplitude, dtype=jnp.float64)
    u = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1) * u_mask
    v = jnp.array(rng.randn(n_lat + 1, n_lon) * amplitude,
                  dtype=jnp.float64) * v_mask
    return u, v


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestBackscatterConfig:
    def test_defaults_off(self):
        cfg = BackscatterConfig()
        assert cfg.enabled is False
        assert cfg.c_bs >= 0.0
        assert 0.0 <= cfg.efficiency <= 1.0
        assert cfg.E_min <= cfg.E_max

    def test_override(self):
        cfg = BackscatterConfig(
            enabled=True, c_bs=0.02,
            tau_relax_days=5.0, E_min=1e-6, E_max=0.05,
            efficiency=0.8)
        assert cfg.enabled is True
        assert cfg.c_bs == 0.02
        assert cfg.efficiency == 0.8

    def test_exported_from_public_packages(self):
        from legoesm.ocean.physics import BackscatterConfig as B1
        from legoesm.ocean.physics.lateral_mixing import BackscatterConfig as B2
        assert B1 is B2


# ---------------------------------------------------------------------------
# LatLon C-grid backscatter
# ---------------------------------------------------------------------------


class TestBackscatterLatLon:
    def test_disabled_is_noop(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        E = jnp.ones(grid.grid_shape_2d) * 1e-2
        cfg = BackscatterConfig(enabled=False, c_bs=0.1)
        tu, tv = backscatter_tendency_cgrid(
            u, v, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert jnp.all(tu == 0.0)
        assert jnp.all(tv == 0.0)

    def test_zero_energy_zero_tendency(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        E = jnp.zeros(grid.grid_shape_2d)
        cfg = BackscatterConfig(enabled=True, c_bs=0.1)
        tu, tv = backscatter_tendency_cgrid(
            u, v, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        # E = 0 ⇒ ν_bs EXACTLY 0 (double-where safe sqrt; the former
        # sqrt(E + 1e-30) leaked a ~1e-15·Δ nonzero negative viscosity).
        assert jnp.count_nonzero(tu) == 0
        assert jnp.count_nonzero(tv) == 0

    def test_sqrt_E_scaling(self, latlon_grid):
        """Quadrupling E must double the tendency (νbs ∝ √E)."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        cfg = BackscatterConfig(enabled=True, c_bs=0.05)
        E = jnp.full(grid.grid_shape_2d, 1e-3)
        tu1, tv1 = backscatter_tendency_cgrid(
            u, v, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        tu4, tv4 = backscatter_tendency_cgrid(
            u, v, 4.0 * E, grid, cfg, mask=mask,
            u_mask=u_mask, v_mask=v_mask)
        ratio_u = tu4 / jnp.where(jnp.abs(tu1) > 1e-20, tu1, 1.0)
        active = jnp.abs(tu1) > 1e-10
        assert jnp.allclose(
            jnp.where(active, ratio_u, 2.0), 2.0, rtol=1e-8)

    def test_is_negated_harmonic_viscosity(self, latlon_grid):
        """Backscatter is a NEGATIVE-Laplacian (∇², harmonic), NOT the old
        two-pass ``+∇²(A∇²u)`` anti-biharmonic (∇⁴).

        Its returned tendency must equal EXACTLY the negation of the
        single-pass harmonic viscous stress-divergence driven by the same
        ``ν_bs = c_bs·Δ·√E`` coefficient.  The caller then ADDs it
        (``du/dt += tend``) to realise ``-ν_bs∇²u``, injecting energy.
        This equality FAILS for the biharmonic two-pass form (different
        values), so it is a genuine regression lock on the fix.
        """
        from legoesm.ocean.physics.lateral_mixing.backscatter import (
            _A_bs_h_cgrid, _A_bs_q_cgrid,
        )
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        cfg = BackscatterConfig(enabled=True, c_bs=0.1)
        E = jnp.full(grid.grid_shape_2d, 1e-2)
        bs_u, bs_v = backscatter_tendency_cgrid(
            u, v, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        # Reference: SINGLE-pass harmonic viscous tendency (dissipative when
        # ADDED), negated ⇒ the backscatter injection.
        A_h = _A_bs_h_cgrid(E, grid, cfg.c_bs)
        A_q = _A_bs_q_cgrid(E, grid, cfg.c_bs)
        diss_u, diss_v = viscous_tendency_cgrid(
            u, v, grid, A_h, A_q, mask=mask, u_mask=u_mask, v_mask=v_mask,
            normalize=True)
        assert jnp.allclose(bs_u, -diss_u, rtol=1e-10, atol=1e-14)
        assert jnp.allclose(bs_v, -diss_v, rtol=1e-10, atol=1e-14)
        # And the injection must be non-trivial (not a degenerate all-zero).
        assert jnp.any(jnp.abs(bs_u) > 0)

    def test_injects_energy_on_average(self, latlon_grid):
        """⟨u · tend_bs⟩ area-weighted is ≥ 0 — the backscatter never
        removes energy from the resolved flow (on average).
        """
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(
            grid, mask, u_mask, v_mask, amplitude=1.0)
        cfg = BackscatterConfig(enabled=True, c_bs=0.1)
        E = jnp.full(grid.grid_shape_2d, 1e-2)
        tu, tv = backscatter_tendency_cgrid(
            u, v, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        # Area-weighted inner product; approximate face areas by cell
        # area (shared by symmetry to leading order for this test).
        area = grid.area
        area_u = jnp.concatenate([area, area[:, 0:1]], axis=1)
        power_u = jnp.sum(u * tu * area_u * u_mask)
        area_v_int = 0.5 * (area[:-1] + area[1:])
        area_v = jnp.concatenate(
            [0.5 * area[0:1], area_v_int, 0.5 * area[-1:]], axis=0)
        power_v = jnp.sum(v * tv * area_v * v_mask)
        total_power = float(power_u + power_v)
        # Globally the backscatter injects energy ⇒ total_power ≥ 0.
        # Allow a tiny numerical slop for the regional sponge region.
        assert total_power > -1e-10

    def test_grad_finite(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        cfg = BackscatterConfig(enabled=True, c_bs=0.05)
        E = jnp.full(grid.grid_shape_2d, 1e-3)

        def loss(uv):
            tu, tv = backscatter_tendency_cgrid(
                uv[0], uv[1], E, grid, cfg,
                mask=mask, u_mask=u_mask, v_mask=v_mask)
            return jnp.sum(tu ** 2) + jnp.sum(tv ** 2)

        du, dv = jax.grad(loss)((u, v))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_coefficient_dimensions_are_m2_per_s(self, latlon_grid):
        """Regression: the backscatter coefficient driving the
        negative-Laplacian stress-divergence must have harmonic-viscosity
        units ``m²/s`` = (length) · (velocity).  Doubling the cell
        size at fixed E should double ν_bs (not quadruple it, which
        would indicate cell-AREA scaling).
        """
        from legoesm.ocean.physics.lateral_mixing.backscatter import (
            _A_bs_h_cgrid,
        )

        grid, mask, u_mask, v_mask = latlon_grid
        E = jnp.full(grid.grid_shape_2d, 1e-3)

        class _GridLike:
            """Minimal grid stand-in with an ``area`` attribute."""
            def __init__(self, area):
                self.area = area

        # Reference coefficient at the original cell area.
        g1 = _GridLike(jnp.asarray(grid.area))
        A1 = _A_bs_h_cgrid(E, g1, 1.0)

        # Quadruple the cell area — doubles Δ (= √area).  Expect A to
        # DOUBLE (m²/s ∝ m) rather than quadruple (would be m²/s ∝ m²).
        g4 = _GridLike(jnp.asarray(grid.area) * 4.0)
        A4 = _A_bs_h_cgrid(E, g4, 1.0)

        ratio = float(jnp.max(A4 / jnp.maximum(A1, 1e-30)))
        # Tolerance must be looser than float32 precision (~6e-8) so
        # the test is reliable even when the shared fixture was built
        # before the autouse x64 switch.  The real failure mode this
        # guards against is ratio = 4.0 (area scaling).
        assert abs(ratio - 2.0) < 1e-5, (
            f"A scales as cell-AREA not length; ratio={ratio:.6f}")

    def test_3d_multilevel_tendency(self, latlon_grid):
        """Regression: 3-D velocity with ``nlev > 1`` must run without
        a ``vmap got inconsistent sizes`` error and produce a tendency
        shape matching the input at every level.  Uniform E and
        vertically repeated velocity must give identical per-level
        tendencies (no unintended vertical coupling).
        """
        grid, mask, u_mask, v_mask = latlon_grid
        u2, v2 = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        nlev = 4
        u3 = jnp.broadcast_to(u2[..., None], u2.shape + (nlev,))
        v3 = jnp.broadcast_to(v2[..., None], v2.shape + (nlev,))
        cfg = BackscatterConfig(enabled=True, c_bs=0.05)
        E = jnp.full(grid.grid_shape_2d, 1e-3)

        tu3, tv3 = backscatter_tendency_cgrid(
            u3, v3, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert tu3.shape == u3.shape
        assert tv3.shape == v3.shape
        tu2_ref, tv2_ref = backscatter_tendency_cgrid(
            u2, v2, E, grid, cfg, mask=mask, u_mask=u_mask, v_mask=v_mask)
        for k in range(nlev):
            assert jnp.allclose(tu3[..., k], tu2_ref, rtol=1e-10, atol=1e-14)
            assert jnp.allclose(tv3[..., k], tv2_ref, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# MPAS backscatter
# ---------------------------------------------------------------------------


class TestBackscatterMPAS:
    def test_disabled_is_noop(self, mpas_mesh):
        n_edges = mpas_mesh.dcEdge.shape[0]
        u = jnp.ones((n_edges, 2)) * 0.1
        E = jnp.full((mpas_mesh.areaCell.shape[0],), 1e-2)
        tend = backscatter_tendency_mpas(
            u, E, mpas_mesh, BackscatterConfig(enabled=False))
        assert jnp.all(tend == 0.0)

    def test_shape_and_finite(self, mpas_mesh):
        rng = np.random.RandomState(7)
        n_edges = mpas_mesh.dcEdge.shape[0]
        n_cells = mpas_mesh.areaCell.shape[0]
        u = jnp.array(rng.randn(n_edges, 3) * 0.1)
        E = jnp.full((n_cells,), 1e-2)
        tend = backscatter_tendency_mpas(
            u, E, mpas_mesh, BackscatterConfig(enabled=True, c_bs=0.05))
        assert tend.shape == u.shape
        assert jnp.all(jnp.isfinite(tend))

    def test_zero_energy_zero_tendency(self, mpas_mesh):
        rng = np.random.RandomState(7)
        n_edges = mpas_mesh.dcEdge.shape[0]
        n_cells = mpas_mesh.areaCell.shape[0]
        u = jnp.array(rng.randn(n_edges, 1) * 0.1)
        E = jnp.zeros((n_cells,))
        tend = backscatter_tendency_mpas(
            u, E, mpas_mesh, BackscatterConfig(enabled=True, c_bs=0.1))
        # E = 0 ⇒ ν_bs EXACTLY 0 — same double-where argument as latlon.
        assert jnp.count_nonzero(tend) == 0

    def test_sqrt_E_scaling(self, mpas_mesh):
        rng = np.random.RandomState(7)
        n_edges = mpas_mesh.dcEdge.shape[0]
        n_cells = mpas_mesh.areaCell.shape[0]
        u = jnp.array(rng.randn(n_edges, 1) * 0.1)
        E = jnp.full((n_cells,), 1e-3)
        t1 = backscatter_tendency_mpas(
            u, E, mpas_mesh, BackscatterConfig(enabled=True, c_bs=0.05))
        t4 = backscatter_tendency_mpas(
            u, 4.0 * E, mpas_mesh,
            BackscatterConfig(enabled=True, c_bs=0.05))
        active = jnp.abs(t1) > 1e-10
        ratio = jnp.where(active, t4 / jnp.where(active, t1, 1.0), 2.0)
        assert jnp.allclose(ratio, 2.0, rtol=1e-8)

    def test_grad_finite(self, mpas_mesh):
        rng = np.random.RandomState(7)
        n_edges = mpas_mesh.dcEdge.shape[0]
        n_cells = mpas_mesh.areaCell.shape[0]
        u = jnp.array(rng.randn(n_edges, 1) * 0.1)
        E = jnp.full((n_cells,), 1e-3)
        cfg = BackscatterConfig(enabled=True, c_bs=0.05)

        def loss(uu):
            return jnp.sum(backscatter_tendency_mpas(
                uu, E, mpas_mesh, cfg) ** 2)
        du = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(du))

    def test_is_negated_single_laplacian(self, mpas_mesh):
        """MPAS backscatter is ``-ν_bs ∇²u`` — a SINGLE (negated) vector
        Laplacian with the HARMONIC ``ν_bs = c_bs·Δ·√E`` [m²/s], NOT the
        old double-∇² anti-biharmonic ``+∇²(A∇²u)`` with A = c_bs·Δ³·√E.

        Uses a SPATIALLY-VARYING E (per-cell) so the lock also pins the
        coefficient PLACEMENT under spatial variation: ν_bs is the negated
        edge-coefficient form ``-ν_bs·∇²u`` — the exact anti-dissipation of
        the wired MPAS harmonic viscosity ``smagorinsky_laplacian_3d``
        (``A_e·∇²u``), for energetic consistency with the paired MPAS
        dissipation — NOT the strict exact-adjoint ``-∇·(ν∇u)``.  (Uniform E
        would not distinguish coefficient placement.)
        """
        rng = np.random.RandomState(7)
        n_edges = mpas_mesh.dcEdge.shape[0]
        n_cells = mpas_mesh.areaCell.shape[0]
        u = jnp.array(rng.randn(n_edges, 1) * 0.1)
        # Spatially-varying per-cell reservoir (positive, O(1e-3–1e-1)).
        E = jnp.array(np.abs(rng.randn(n_cells)) * 1e-2 + 1e-3)
        cfg = BackscatterConfig(enabled=True, c_bs=0.05)
        tend = backscatter_tendency_mpas(u, E, mpas_mesh, cfg)
        # Reference: ν_bs = c_bs·Δ·√E at edges (Δ¹, harmonic), single ∇².
        delta_edge = jnp.sqrt(mpas_mesh.dcEdge * mpas_mesh.dvEdge)
        c1, c2 = mpas_mesh.cellsOnEdge[0], mpas_mesh.cellsOnEdge[1]
        E_edge = (0.5 * (E[c1] + E[c2]))[:, jnp.newaxis]
        nu_bs = cfg.c_bs * delta_edge[:, jnp.newaxis] * jnp.sqrt(
            jnp.maximum(E_edge, 0.0) + 1e-30)
        ref = -nu_bs * vector_laplacian_del2_3d(u, mpas_mesh)
        assert jnp.allclose(tend, ref, rtol=1e-10, atol=1e-14)
        assert jnp.any(jnp.abs(tend) > 0)
        # ν_bs genuinely varies across edges (guards against a uniform-E
        # regression silently passing this as a constant-coefficient check).
        assert float(jnp.std(nu_bs)) > 0.0


# ---------------------------------------------------------------------------
# Energy budget
# ---------------------------------------------------------------------------


class TestEnergyBudget:
    def test_reservoir_update_decays_to_zero_without_source(self):
        E = jnp.full((4, 4), 1e-2)
        cfg = BackscatterConfig(
            enabled=True, tau_relax_days=1.0,
            E_min=0.0, E_max=1.0)
        dt = 3600.0
        zero = jnp.zeros_like(E)
        E_new = update_eddy_energy(E, dt, zero, zero, cfg)
        # Forward Euler: E_new = E · (1 − dt/τ) with τ = 86400.
        expected = E * (1.0 - dt / (cfg.tau_relax_days * 86400.0))
        assert jnp.allclose(E_new, expected, rtol=1e-12)
        assert jnp.all(E_new < E)

    def test_reservoir_responds_to_dissipation(self):
        E = jnp.zeros((4, 4))
        cfg = BackscatterConfig(
            enabled=True, tau_relax_days=10.0, E_max=1.0,
            efficiency=1.0)
        dt = 100.0
        eps_d = jnp.full_like(E, 1e-6)
        eps_b = jnp.zeros_like(E)
        E_new = update_eddy_energy(E, dt, eps_d, eps_b, cfg)
        # Forward Euler from zero: E_new = dt · ε_d − 0 − 0.
        assert jnp.allclose(E_new, dt * eps_d, rtol=1e-12)

    def test_reservoir_clips_to_bounds(self):
        E = jnp.full((2, 2), 0.5)
        cfg = BackscatterConfig(
            enabled=True, tau_relax_days=10.0,
            E_min=0.1, E_max=0.2, efficiency=1.0)
        # Large dissipation should push E above E_max; clamp.
        eps_d = jnp.full_like(E, 1e3)
        eps_b = jnp.zeros_like(E)
        E_new = update_eddy_energy(E, 1.0, eps_d, eps_b, cfg)
        assert float(jnp.max(E_new)) == cfg.E_max
        # Strongly negative net tendency should floor at E_min.
        E_hot = jnp.full((2, 2), 0.15)
        eps_b_neg = jnp.full_like(E_hot, 1e3)
        E_neg = update_eddy_energy(E_hot, 1.0, eps_d * 0.0, eps_b_neg, cfg)
        assert float(jnp.min(E_neg)) == cfg.E_min

    def test_zero_c_bs_update_is_noop(self):
        """With ``c_bs == 0`` the tendency is zero, so ``E`` must also
        stay put — otherwise callers that run the update every step
        get a reservoir that drifts while no energy is actually
        exchanged with the flow.
        """
        E = jnp.full((3, 3), 1.0e-2)
        cfg = BackscatterConfig(
            enabled=True, c_bs=0.0, tau_relax_days=1.0,
            E_min=0.0, E_max=1.0, efficiency=1.0)
        eps_d = jnp.full_like(E, 1.0e-4)
        eps_b = jnp.full_like(E, 1.0e-5)
        E_new = update_eddy_energy(E, 3600.0, eps_d, eps_b, cfg)
        assert jnp.all(E_new == E)

    def test_disabled_update_is_noop(self):
        """``update_eddy_energy`` must return E unchanged when
        ``cfg.enabled`` is ``False``; otherwise callers that invoke
        it every step regardless of the flag would see E drift.
        """
        E = jnp.full((3, 3), 1.0e-2)
        cfg = BackscatterConfig(
            enabled=False, tau_relax_days=1.0,
            E_min=0.0, E_max=1.0, efficiency=1.0)
        eps_d = jnp.full_like(E, 1.0e-4)
        eps_b = jnp.full_like(E, 1.0e-5)
        E_new = update_eddy_energy(E, 3600.0, eps_d, eps_b, cfg)
        assert jnp.all(E_new == E)

    def test_efficiency_is_applied(self):
        E = jnp.zeros((3, 3))
        eps_d = jnp.full_like(E, 1e-4)
        eps_b = jnp.zeros_like(E)
        dt = 10.0
        cfg_full = BackscatterConfig(
            enabled=True, tau_relax_days=1e6, efficiency=1.0,
            E_min=0.0, E_max=1.0)
        cfg_half = cfg_full._replace(efficiency=0.5)
        E_full = update_eddy_energy(E, dt, eps_d, eps_b, cfg_full)
        E_half = update_eddy_energy(E, dt, eps_d, eps_b, cfg_half)
        assert jnp.allclose(E_half, 0.5 * E_full)


# ---------------------------------------------------------------------------
# Power-density helper
# ---------------------------------------------------------------------------


class TestPowerDensity:
    def test_shape_2d(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        tu = jnp.zeros_like(u)
        tv = jnp.zeros_like(v)
        p = backscatter_power_density_cgrid(
            u, v, tu, tv, grid, u_mask=u_mask, v_mask=v_mask)
        assert p.shape == grid.grid_shape_2d

    def test_zero_tendency_zero_power(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        tu = jnp.zeros_like(u)
        tv = jnp.zeros_like(v)
        p = backscatter_power_density_cgrid(
            u, v, tu, tv, grid, u_mask=u_mask, v_mask=v_mask)
        assert jnp.allclose(p, 0.0)

    def test_dz_weighted_mean(self, latlon_grid):
        """The 3-D power is the depth-MEAN (per-mass) power, not the
        depth-integral (#backscatter dimensional-consistency fix).  A
        UNIFORM ``dz`` cancels in the thickness-weighted mean, so
        ``dz = 2`` gives the SAME result as the default unit spacing
        (an integral would double).
        """
        grid, mask, u_mask, v_mask = latlon_grid
        u2, v2 = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        nlev = 4
        u3 = jnp.broadcast_to(u2[..., None], u2.shape + (nlev,))
        v3 = jnp.broadcast_to(v2[..., None], v2.shape + (nlev,))
        tu3 = u3
        tv3 = v3
        p_default = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask)
        p_dz2 = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask,
            dz=jnp.full((nlev,), 2.0))
        # Depth-MEAN is invariant to a uniform thickness rescale.
        assert jnp.allclose(p_dz2, p_default, rtol=1e-10)

    def test_nonuniform_dz_is_thickness_weighted_mean(self, latlon_grid):
        """With non-uniform ``dz`` the result is the thickness-weighted
        MEAN ``Σ p_k dz_k / Σ dz_k`` — bounded by the per-level values,
        never their sum.
        """
        grid, mask, u_mask, v_mask = latlon_grid
        u2, v2 = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        nlev = 3
        u3 = jnp.broadcast_to(u2[..., None], u2.shape + (nlev,))
        v3 = jnp.broadcast_to(v2[..., None], v2.shape + (nlev,))
        # Identical levels ⇒ the weighted mean equals the single-level
        # power regardless of the (non-uniform) weights.
        p1 = backscatter_power_density_cgrid(
            u2, v2, u2, v2, grid, u_mask=u_mask, v_mask=v_mask)
        p3 = backscatter_power_density_cgrid(
            u3, v3, u3, v3, grid, u_mask=u_mask, v_mask=v_mask,
            dz=jnp.asarray([1.0, 5.0, 2.0]))
        assert jnp.allclose(p3, p1, rtol=1e-12)

    def test_3d_input_is_depth_mean_not_sum(self, latlon_grid):
        """For 3-D inputs the reservoir sees the depth-MEAN per-mass
        power (matching the per-mass EKE reservoir budget), NOT the
        depth-integrated SUM of layers.  With ``nlev`` identical levels
        the 3-D result equals the single 2-D level (a sum would give
        ``nlev ×``).  The 2-D face masks broadcast internally.
        """
        grid, mask, u_mask, v_mask = latlon_grid
        u2, v2 = _random_velocity_latlon(grid, mask, u_mask, v_mask)
        nlev = 5
        u3 = jnp.broadcast_to(u2[..., None], u2.shape + (nlev,))
        v3 = jnp.broadcast_to(v2[..., None], v2.shape + (nlev,))
        tu3 = jnp.broadcast_to(u2[..., None], u2.shape + (nlev,))
        tv3 = jnp.broadcast_to(v2[..., None], v2.shape + (nlev,))
        p3 = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask)
        p2 = backscatter_power_density_cgrid(
            u2, v2, u2, v2, grid, u_mask=u_mask, v_mask=v_mask)
        assert p3.shape == grid.grid_shape_2d
        # Depth-mean of nlev identical levels == the single level.
        assert jnp.allclose(p3, p2, rtol=1e-12)
