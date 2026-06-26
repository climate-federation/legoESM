"""CI tests for Leith viscosity on lat-lon C-grid and MPAS.

Mirrors ``tests/ocean/unit/test_smagorinsky.py``.  All tests use small
grids and run in seconds.  Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/ocean/unit/test_leith.py -v

References
----------
- Leith (1996) J. Atmos. Sci. 53, 2487-2492.
- Fox-Kemper & Menemenlis (2008) "Can Large Eddy Simulation Techniques
  Improve Mesoscale Rich Ocean Models?" AGU GM177.
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
    leith_viscosity_cgrid,
    leith_viscosity_q_cgrid,
    leith_biharmonic_tendency_cgrid,
    curl_vertex_cgrid,
)
from legoesm.core.operators_voronoi import (
    leith_viscosity_edge_3d,
    leith_biharmonic_3d,
    curl_vertex_3d,
    vector_laplacian_del2_3d,
)
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.mpas_config import MPASOceanConfig


# ============================================================================
# Fixtures
# ============================================================================

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


def _random_velocity_latlon(grid, wall_mask, u_mask, v_mask, seed=42,
                            amplitude=0.1):
    rng = np.random.RandomState(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    u_inner = jnp.array(rng.randn(n_lat, n_lon) * amplitude, dtype=jnp.float64)
    u = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1) * u_mask
    v = jnp.array(rng.randn(n_lat + 1, n_lon) * amplitude,
                  dtype=jnp.float64) * v_mask
    return u, v


def _random_velocity_mpas(mesh, seed=42, amplitude=0.1, nlev=1):
    rng = np.random.RandomState(seed)
    n_edges = mesh.dcEdge.shape[0]
    return jnp.array(rng.randn(n_edges, nlev) * amplitude, dtype=jnp.float64)


# ============================================================================
# 1. Config plumbing
# ============================================================================


class TestLeithConfig:
    """The two ocean configs must expose C_leith and C_leith_modified."""

    def test_latlon_config_has_leith_fields_default_off(self):
        cfg = LatLonCGridOceanConfig.from_flat()
        assert hasattr(cfg, "C_leith")
        assert hasattr(cfg, "C_leith_modified")
        assert cfg.C_leith == 0.0
        assert cfg.C_leith_modified is False

    def test_mpas_config_has_leith_fields_default_off(self):
        cfg = MPASOceanConfig()
        assert hasattr(cfg, "C_leith")
        assert hasattr(cfg, "C_leith_modified")
        assert cfg.C_leith == 0.0
        assert cfg.C_leith_modified is False

    def test_latlon_config_accepts_leith_override(self):
        cfg = LatLonCGridOceanConfig.from_flat(C_leith=1.5, C_leith_modified=True)
        assert cfg.C_leith == 1.5
        assert cfg.C_leith_modified is True


# ============================================================================
# 2. Lat-lon C-grid Leith
# ============================================================================


class TestLeithLatLon:
    """Tests for ``leith_viscosity_cgrid`` and biharmonic tendency."""

    def test_A_leith_non_negative(self, latlon_grid):
        """A_L must be ≥ 0 everywhere — it is a magnitude × grid scale cube."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        A = leith_viscosity_cgrid(
            u, v, grid, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert A.shape == (grid.n_lat, grid.n_lon)
        assert jnp.all(A >= 0.0)

    def test_A_leith_q_non_negative(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        A_q = leith_viscosity_q_cgrid(
            u, v, grid, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert A_q.shape == (grid.n_lat + 1, grid.n_lon + 1)
        assert jnp.all(A_q >= 0.0)

    def test_A_leith_zero_for_uniform_flow(self, latlon_grid):
        """A_L vanishes when ζ is spatially constant (uniform flow ⇒ ζ=0).

        The sqrt-regularisation ε=1e-30 means A_L does not reach exactly
        zero; bound by the analytic floor ``(C_L·Δ)³ · √ε`` which is tiny
        compared to any realistic A_h (~10⁴ m²/s).
        """
        grid, mask, u_mask, v_mask = latlon_grid
        C_L = 2.0
        u = jnp.zeros((grid.n_lat, grid.n_lon + 1)) * u_mask
        v = jnp.zeros((grid.n_lat + 1, grid.n_lon)) * v_mask

        A = leith_viscosity_cgrid(u, v, grid, C_L,
                                   mask=mask, u_mask=u_mask, v_mask=v_mask)
        Delta_max = float(jnp.sqrt(jnp.max(grid.area)))
        floor = (C_L * Delta_max) ** 3 * jnp.sqrt(1e-30).item() * 10.0
        assert float(jnp.max(A)) < floor

    def test_A_leith_cubic_in_C(self, latlon_grid):
        """Scaling: A_L(2·C_L) = 8 · A_L(C_L) at every point."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        A1 = leith_viscosity_cgrid(u, v, grid, 1.0,
                                    mask=mask, u_mask=u_mask, v_mask=v_mask)
        A2 = leith_viscosity_cgrid(u, v, grid, 2.0,
                                    mask=mask, u_mask=u_mask, v_mask=v_mask)
        # Where A1 > 0, A2/A1 should equal 8.  Use relative comparison.
        mask_active = A1 > 1e-12
        ratio = jnp.where(mask_active, A2 / jnp.where(mask_active, A1, 1.0),
                           8.0)
        assert jnp.allclose(ratio, 8.0, rtol=1e-10)

    def test_A_leith_modified_geq_classical(self, latlon_grid):
        """Modified Leith adds |∇δ|² ≥ 0 in quadrature, so A_L_mod ≥ A_L."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        A_classic = leith_viscosity_cgrid(
            u, v, grid, 1.0, modified=False,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        A_modified = leith_viscosity_cgrid(
            u, v, grid, 1.0, modified=True,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert jnp.all(A_modified + 1e-12 >= A_classic)

    def test_tendency_shapes(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        tu, tv = leith_biharmonic_tendency_cgrid(
            u, v, grid, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert tu.shape == u.shape
        assert tv.shape == v.shape
        assert jnp.all(jnp.isfinite(tu))
        assert jnp.all(jnp.isfinite(tv))

    def test_tendency_zero_for_zero_velocity(self, latlon_grid):
        grid, mask, u_mask, v_mask = latlon_grid
        u = jnp.zeros((grid.n_lat, grid.n_lon + 1))
        v = jnp.zeros((grid.n_lat + 1, grid.n_lon))

        tu, tv = leith_biharmonic_tendency_cgrid(
            u, v, grid, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask)
        # Strain of zero velocity is zero, ζ=0, A_L ≈ 0, tendency ≈ 0.
        assert float(jnp.max(jnp.abs(tu))) < 1e-10
        assert float(jnp.max(jnp.abs(tv))) < 1e-10

    def test_tendency_energy_dissipative(self, latlon_grid):
        """⟨u, −tend⟩ ≥ 0: biharmonic Leith never injects energy.

        Convention: ``du/dt -= tend_u``, so the work done on the flow is
        ``u · (−tend_u)``.  Sum over all cells, weighted by area.
        """
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        tu, tv = leith_biharmonic_tendency_cgrid(
            u, v, grid, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask)

        # Approximate dual face areas (cell area interpolated to each face).
        area_u = 0.5 * (grid.area[:, :] + jnp.roll(grid.area, 1, axis=1))
        area_u = jnp.concatenate([area_u, area_u[:, 0:1]], axis=1)  # periodic
        area_v_interior = 0.5 * (grid.area[:-1] + grid.area[1:])
        area_v = jnp.concatenate([
            grid.area[0:1] * 0.5, area_v_interior, grid.area[-1:] * 0.5,
        ], axis=0)

        power_u = jnp.sum(u * (-tu) * area_u * u_mask)
        power_v = jnp.sum(v * (-tv) * area_v * v_mask)
        total_power = float(power_u + power_v)
        # Dissipative ⇒ power done on flow ≤ 0.
        assert total_power <= 1e-10

    def test_A_leith_q_periodic_seam(self):
        """∂ζ/∂x at q-points must agree at column 0 and column n_lon.

        Regression test: the previous implementation rolled ``zeta_q`` over
        the wrap column (``[:, n_lon] == [:, 0]``), which gave column 0's
        west neighbour the duplicate value instead of column ``n_lon-1``
        and therefore produced a wrong (usually smaller) A_leith on the
        periodic seam.  Use a FULLY periodic global grid so the seam is a
        real physical boundary.
        """
        from legoesm.grids.latlon import create_latlon_grid

        grid = create_latlon_grid(16, 32, dtype=jnp.float64)
        n_lat, n_lon = grid.n_lat, grid.n_lon

        # A periodic-in-lon, non-trivial velocity: u = sin(2λ), v = cos(2λ).
        # u lives on east faces with shape (n_lat, n_lon+1); its wrap column
        # must equal column 0 for the q-point curl to be periodic.
        lon_u = (jnp.arange(n_lon + 1) * grid.dlon)[None, :]
        u_core = jnp.sin(2.0 * lon_u)
        u_core = jnp.broadcast_to(u_core, (n_lat, n_lon + 1))
        # enforce wrap column equality explicitly
        u = jnp.concatenate([u_core[:, :n_lon], u_core[:, 0:1]], axis=1)

        lon_v = (jnp.arange(n_lon) * grid.dlon)[None, :]
        v = jnp.cos(2.0 * lon_v)
        v = jnp.broadcast_to(v, (n_lat + 1, n_lon))

        A_q = leith_viscosity_q_cgrid(u, v, grid, 1.0)
        # Column 0 == column n_lon on a periodic global grid.
        left = A_q[:, 0]
        right = A_q[:, n_lon]
        assert jnp.allclose(left, right, rtol=0, atol=1e-10)

    def test_tendency_grad_finite(self, latlon_grid):
        """JAX grad through the biharmonic tendency must be finite."""
        grid, mask, u_mask, v_mask = latlon_grid
        u, v = _random_velocity_latlon(grid, mask, u_mask, v_mask)

        def loss(uv):
            tu, tv = leith_biharmonic_tendency_cgrid(
                uv[0], uv[1], grid, 1.0,
                mask=mask, u_mask=u_mask, v_mask=v_mask)
            return jnp.sum(tu ** 2) + jnp.sum(tv ** 2)

        du, dv = jax.grad(loss)((u, v))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))


# ============================================================================
# 3. MPAS Leith
# ============================================================================


class TestLeithMPAS:
    """Tests for ``leith_viscosity_edge_3d`` and biharmonic tendency."""

    def test_A_leith_non_negative(self, mpas_mesh):
        u = _random_velocity_mpas(mpas_mesh, seed=42, nlev=3)
        A = leith_viscosity_edge_3d(u, mpas_mesh, 1.0)
        assert A.shape == u.shape
        assert jnp.all(A >= 0.0)

    def test_A_leith_cubic_in_C(self, mpas_mesh):
        u = _random_velocity_mpas(mpas_mesh, seed=42, nlev=1)
        A1 = leith_viscosity_edge_3d(u, mpas_mesh, 1.0)
        A2 = leith_viscosity_edge_3d(u, mpas_mesh, 2.0)
        mask = A1 > 1e-12
        ratio = jnp.where(mask, A2 / jnp.where(mask, A1, 1.0), 8.0)
        assert jnp.allclose(ratio, 8.0, rtol=1e-10)

    def test_A_leith_modified_geq_classical(self, mpas_mesh):
        u = _random_velocity_mpas(mpas_mesh, seed=42, nlev=1)
        A_classic = leith_viscosity_edge_3d(u, mpas_mesh, 1.0, modified=False)
        A_modified = leith_viscosity_edge_3d(u, mpas_mesh, 1.0, modified=True)
        assert jnp.all(A_modified + 1e-12 >= A_classic)

    def test_A_leith_zero_for_zero_velocity(self, mpas_mesh):
        """Same bound as the latlon test: tiny analytic floor from sqrt(ε)."""
        C_L = 2.0
        u = jnp.zeros((mpas_mesh.dcEdge.shape[0], 1))
        A = leith_viscosity_edge_3d(u, mpas_mesh, C_L)
        delta_max = float(jnp.sqrt(jnp.max(mpas_mesh.dcEdge * mpas_mesh.dvEdge)))
        floor = (C_L * delta_max) ** 3 * jnp.sqrt(1e-30).item() * 10.0
        assert float(jnp.max(A)) < floor

    def test_tendency_shapes_and_finite(self, mpas_mesh):
        u = _random_velocity_mpas(mpas_mesh, seed=42, nlev=3)
        tend = leith_biharmonic_3d(u, mpas_mesh, 1.0)
        assert tend.shape == u.shape
        assert jnp.all(jnp.isfinite(tend))

    def test_tendency_zero_for_zero_velocity(self, mpas_mesh):
        u = jnp.zeros((mpas_mesh.dcEdge.shape[0], 1))
        tend = leith_biharmonic_3d(u, mpas_mesh, 1.0)
        assert float(jnp.max(jnp.abs(tend))) < 1e-14

    def test_tendency_multi_level_independence(self, mpas_mesh):
        """Running two independent levels together must equal running each
        level separately — no unintended vertical coupling."""
        u = _random_velocity_mpas(mpas_mesh, seed=42, nlev=2)
        tend_3d = leith_biharmonic_3d(u, mpas_mesh, 1.0)
        for k in range(2):
            tend_1d = leith_biharmonic_3d(u[:, k:k + 1], mpas_mesh, 1.0)
            max_diff = float(jnp.max(jnp.abs(tend_3d[:, k:k + 1] - tend_1d)))
            assert max_diff < 1e-14, (
                f"Multi-level/single-level mismatch at level {k}: {max_diff}"
            )

    def test_tendency_grad_finite(self, mpas_mesh):
        u = _random_velocity_mpas(mpas_mesh, seed=42, nlev=1)

        def loss(uu):
            return jnp.sum(leith_biharmonic_3d(uu, mpas_mesh, 1.0) ** 2)

        du = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(du))

    def test_checkerboard_damping(self, mpas_mesh):
        """Alternating-sign edge pattern must yield a non-zero biharmonic
        response — grid-scale noise should be damped."""
        n_edges = mpas_mesh.dcEdge.shape[0]
        u = ((-1.0) ** jnp.arange(n_edges))[:, None].astype(jnp.float64) * 0.1
        tend = leith_biharmonic_3d(u, mpas_mesh, 1.0)
        assert float(jnp.max(jnp.abs(tend))) > 0.0


# ============================================================================
# 4. Consistency between grids
# ============================================================================


class TestLeithCrossGridConsistency:
    """Sanity checks that both implementations agree on qualitative behavior."""

    def test_both_respond_to_large_velocity_gradients(
        self, latlon_grid, mpas_mesh
    ):
        """Increasing velocity amplitude must monotonically increase the
        Leith magnitude on both grids."""
        grid, mask, u_mask, v_mask = latlon_grid
        amps = [0.01, 0.1, 1.0]

        max_A_ll = []
        for a in amps:
            u, v = _random_velocity_latlon(
                grid, mask, u_mask, v_mask, seed=7, amplitude=a)
            A = leith_viscosity_cgrid(u, v, grid, 1.0,
                                       mask=mask, u_mask=u_mask, v_mask=v_mask)
            max_A_ll.append(float(jnp.max(A)))

        max_A_mpas = []
        for a in amps:
            u = _random_velocity_mpas(mpas_mesh, seed=7, amplitude=a, nlev=1)
            A = leith_viscosity_edge_3d(u, mpas_mesh, 1.0)
            max_A_mpas.append(float(jnp.max(A)))

        assert max_A_ll[0] < max_A_ll[1] < max_A_ll[2]
        assert max_A_mpas[0] < max_A_mpas[1] < max_A_mpas[2]
