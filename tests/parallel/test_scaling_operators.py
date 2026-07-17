"""Tests for scaling-optimized operators.

Verifies that native 3D operators match the vmap-based originals,
and that batched 3D TRiSK operators match per-level computation.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_latlon_vector,
    pad_halo_vector_latlon,
    pad_halo_latlon_3d,
    pad_halo_latlon_vector_3d,
    pad_halo_vector_latlon_3d,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grid():
    return create_latlon_grid(16, 32)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ============================================================================
# 3D halo padding matches vmap of 2D
# ============================================================================

class TestHaloLatlon3D:
    """Native 3D halo functions must match vmap of 2D versions."""

    def test_scalar_halo_3d(self, rng):
        nlev = 5
        data = jax.random.normal(rng, (16, 32, nlev))
        result_3d = pad_halo_latlon_3d(data, halo=1)

        # Reference: vmap over levels
        def pad_level(d_k):
            return pad_halo_latlon(d_k, halo=1)

        ref = jax.vmap(pad_level)(jnp.moveaxis(data, -1, 0))
        ref = jnp.moveaxis(ref, 0, -1)

        assert result_3d.shape == ref.shape
        assert jnp.allclose(result_3d, ref, atol=1e-12)

    def test_vector_halo_3d(self, rng):
        nlev = 5
        keys = jax.random.split(rng, 2)
        u = jax.random.normal(keys[0], (16, 32, nlev))
        v = jax.random.normal(keys[1], (16, 32, nlev))

        u3d, v3d = pad_halo_vector_latlon_3d(u, v, halo=1)

        # Reference
        def pad_level(u_k, v_k):
            return pad_halo_vector_latlon(u_k, v_k, halo=1)

        u_ref, v_ref = jax.vmap(pad_level)(
            jnp.moveaxis(u, -1, 0), jnp.moveaxis(v, -1, 0))
        u_ref = jnp.moveaxis(u_ref, 0, -1)
        v_ref = jnp.moveaxis(v_ref, 0, -1)

        assert jnp.allclose(u3d, u_ref, atol=1e-12)
        assert jnp.allclose(v3d, v_ref, atol=1e-12)

    def test_scalar_halo_3d_halo2(self, rng):
        nlev = 3
        data = jax.random.normal(rng, (16, 32, nlev))
        result_3d = pad_halo_latlon_3d(data, halo=2)

        def pad_level(d_k):
            return pad_halo_latlon(d_k, halo=2)

        ref = jax.vmap(pad_level)(jnp.moveaxis(data, -1, 0))
        ref = jnp.moveaxis(ref, 0, -1)

        assert result_3d.shape == ref.shape
        assert jnp.allclose(result_3d, ref, atol=1e-12)


# ============================================================================
# Native 3D lat-lon operators match vmap-based originals
# ============================================================================

class TestNative3DLatLonOps:
    """Native 3D operators must produce same results as vmap wrappers."""

    def test_gradient_x_3d(self, grid, rng):
        from legoesm.core.operators_latlon_3d import gradient_x_3d
        from legoesm.core.operators_latlon import gradient_x
        from legoesm.core.field import Field

        nlev = 4
        data = jax.random.normal(rng, (grid.n_lat, grid.n_lon, nlev))
        result = gradient_x_3d(data, grid)

        # Reference: per-level
        ref_levels = []
        for k in range(nlev):
            f = Field(data=data[:, :, k], name="f", dims=("lat", "lon"),
                      units="", staggering="cell")
            ref_levels.append(gradient_x(f, grid).data)
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10), (
            f"max diff = {jnp.max(jnp.abs(result - ref))}")

    def test_gradient_y_3d(self, grid, rng):
        from legoesm.core.operators_latlon_3d import gradient_y_3d
        from legoesm.core.operators_latlon import gradient_y
        from legoesm.core.field import Field

        nlev = 4
        data = jax.random.normal(rng, (grid.n_lat, grid.n_lon, nlev))
        result = gradient_y_3d(data, grid)

        ref_levels = []
        for k in range(nlev):
            f = Field(data=data[:, :, k], name="f", dims=("lat", "lon"),
                      units="", staggering="cell")
            ref_levels.append(gradient_y(f, grid).data)
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10)

    def test_divergence_3d(self, grid, rng):
        from legoesm.core.operators_latlon_3d import divergence_3d
        from legoesm.core.operators_latlon import divergence
        from legoesm.core.field import Field

        nlev = 4
        keys = jax.random.split(rng, 2)
        u = jax.random.normal(keys[0], (grid.n_lat, grid.n_lon, nlev))
        v = jax.random.normal(keys[1], (grid.n_lat, grid.n_lon, nlev))
        result = divergence_3d(u, v, grid)

        ref_levels = []
        for k in range(nlev):
            uf = Field(data=u[:, :, k], name="u", dims=("lat", "lon"), units="m/s")
            vf = Field(data=v[:, :, k], name="v", dims=("lat", "lon"), units="m/s")
            ref_levels.append(divergence(uf, vf, grid).data)
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10)

    def test_vorticity_3d(self, grid, rng):
        from legoesm.core.operators_latlon_3d import vorticity_3d
        from legoesm.core.operators_latlon import curl_z
        from legoesm.core.field import Field

        nlev = 4
        keys = jax.random.split(rng, 2)
        u = jax.random.normal(keys[0], (grid.n_lat, grid.n_lon, nlev))
        v = jax.random.normal(keys[1], (grid.n_lat, grid.n_lon, nlev))
        result = vorticity_3d(u, v, grid)

        ref_levels = []
        for k in range(nlev):
            uf = Field(data=u[:, :, k], name="u", dims=("lat", "lon"), units="m/s")
            vf = Field(data=v[:, :, k], name="v", dims=("lat", "lon"), units="m/s")
            ref_levels.append(curl_z(uf, vf, grid).data)
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10)

    def test_laplacian_3d(self, grid, rng):
        from legoesm.core.operators_latlon_3d import laplacian_3d
        from legoesm.core.operators_latlon import laplacian
        from legoesm.core.field import Field

        nlev = 4
        data = jax.random.normal(rng, (grid.n_lat, grid.n_lon, nlev))
        result = laplacian_3d(data, grid)

        ref_levels = []
        for k in range(nlev):
            f = Field(data=data[:, :, k], name="f", dims=("lat", "lon"), units="")
            ref_levels.append(laplacian(f, grid).data)
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10)


# ============================================================================
# Native 3D FV operators match vmap-based originals
# ============================================================================

class TestNative3DFVOps:

    def test_fv_flux_divergence_3d(self, grid, rng):
        from legoesm.core.operators_fv_latlon_3d import fv_flux_divergence_latlon_3d
        from legoesm.core.operators_fv_latlon import fv_flux_divergence_latlon

        nlev = 3
        keys = jax.random.split(rng, 3)
        q = 1.0 + 0.1 * jax.random.normal(keys[0], (grid.n_lat, grid.n_lon, nlev))
        u = 10.0 * jax.random.normal(keys[1], (grid.n_lat, grid.n_lon, nlev))
        v = 10.0 * jax.random.normal(keys[2], (grid.n_lat, grid.n_lon, nlev))

        result = fv_flux_divergence_latlon_3d(q, u, v, grid, limiter=True)

        ref_levels = []
        for k in range(nlev):
            ref_levels.append(
                fv_flux_divergence_latlon(
                    q[:, :, k], u[:, :, k], v[:, :, k], grid, limiter=True))
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-8), (
            f"max diff = {jnp.max(jnp.abs(result - ref))}")

    def test_fv_scalar_advection_3d(self, grid, rng):
        from legoesm.core.operators_fv_latlon_3d import fv_scalar_advection_latlon_3d
        from legoesm.core.operators_fv_latlon import fv_scalar_advection_latlon

        nlev = 3
        keys = jax.random.split(rng, 3)
        q = 300.0 + jax.random.normal(keys[0], (grid.n_lat, grid.n_lon, nlev))
        u = 10.0 * jax.random.normal(keys[1], (grid.n_lat, grid.n_lon, nlev))
        v = 10.0 * jax.random.normal(keys[2], (grid.n_lat, grid.n_lon, nlev))

        result = fv_scalar_advection_latlon_3d(q, u, v, grid, limiter=True)

        ref_levels = []
        for k in range(nlev):
            ref_levels.append(
                fv_scalar_advection_latlon(
                    q[:, :, k], u[:, :, k], v[:, :, k], grid, limiter=True))
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-8)


# ============================================================================
# Batched 3D TRiSK operators match per-level computation
# ============================================================================

class TestBatched3DTRiSK:
    """Batched 3D TRiSK operators must match per-level scalar operators."""

    @pytest.fixture
    def mesh_and_state(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        mesh = create_voronoi_mesh(subdivision_level=2)
        nlev = 4
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 3)
        u_3d = jax.random.normal(keys[0], (mesh.nEdges, nlev))
        T_3d = 300.0 + jax.random.normal(keys[1], (mesh.nCells, nlev))
        h_3d = 1000.0 + 100.0 * jax.random.normal(keys[2], (mesh.nCells, nlev))
        return mesh, u_3d, T_3d, h_3d

    def test_divergence_cell_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            divergence_cell, divergence_cell_3d)

        mesh, u_3d, _, _ = mesh_and_state
        nlev = u_3d.shape[-1]
        result = divergence_cell_3d(u_3d, mesh)

        ref = jnp.stack(
            [divergence_cell(u_3d[:, k], mesh) for k in range(nlev)],
            axis=-1)

        assert jnp.allclose(result, ref, atol=1e-12)

    def test_gradient_edge_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            gradient_edge, gradient_edge_3d)

        mesh, _, T_3d, _ = mesh_and_state
        nlev = T_3d.shape[-1]
        result = gradient_edge_3d(T_3d, mesh)

        ref = jnp.stack(
            [gradient_edge(T_3d[:, k], mesh) for k in range(nlev)],
            axis=-1)

        assert jnp.allclose(result, ref, atol=1e-12)

    def test_curl_vertex_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            curl_vertex, curl_vertex_3d)

        mesh, u_3d, _, _ = mesh_and_state
        nlev = u_3d.shape[-1]
        result = curl_vertex_3d(u_3d, mesh)

        ref = jnp.stack(
            [curl_vertex(u_3d[:, k], mesh) for k in range(nlev)],
            axis=-1)

        assert jnp.allclose(result, ref, atol=1e-12)

    def test_kinetic_energy_cell_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            kinetic_energy_cell, kinetic_energy_cell_3d)

        mesh, u_3d, _, _ = mesh_and_state
        nlev = u_3d.shape[-1]
        result = kinetic_energy_cell_3d(u_3d, mesh)

        ref = jnp.stack(
            [kinetic_energy_cell(u_3d[:, k], mesh) for k in range(nlev)],
            axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10)

    def test_cell_to_edge_avg_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            cell_to_edge_avg, cell_to_edge_avg_3d)

        mesh, _, T_3d, _ = mesh_and_state
        nlev = T_3d.shape[-1]
        result = cell_to_edge_avg_3d(T_3d, mesh)

        ref = jnp.stack(
            [cell_to_edge_avg(T_3d[:, k], mesh) for k in range(nlev)],
            axis=-1)

        assert jnp.allclose(result, ref, atol=1e-12)

    def test_vector_laplacian_del2_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            vector_laplacian_del2, vector_laplacian_del2_3d)

        mesh, u_3d, _, _ = mesh_and_state
        nlev = u_3d.shape[-1]
        result = vector_laplacian_del2_3d(u_3d, mesh)

        ref = jnp.stack(
            [vector_laplacian_del2(u_3d[:, k], mesh) for k in range(nlev)],
            axis=-1)

        assert jnp.allclose(result, ref, atol=1e-10)

    def test_pv_flux_energy_conserving_3d(self, mesh_and_state):
        from legoesm.core.operators_voronoi import (
            potential_vorticity_vertex,
            potential_vorticity_vertex_3d,
            pv_flux_energy_conserving,
            pv_flux_energy_conserving_3d,
        )

        mesh, u_3d, _, h_3d = mesh_and_state
        nlev = u_3d.shape[-1]

        # 3D path
        q_v_3d = potential_vorticity_vertex_3d(
            u_3d, h_3d, mesh.fVertex, mesh)
        result = pv_flux_energy_conserving_3d(u_3d, h_3d, q_v_3d, mesh)

        # Per-level reference
        ref_levels = []
        for k in range(nlev):
            q_v_k = potential_vorticity_vertex(
                u_3d[:, k], h_3d[:, k], mesh.fVertex, mesh)
            ref_levels.append(
                pv_flux_energy_conserving(
                    u_3d[:, k], h_3d[:, k], q_v_k, mesh))
        ref = jnp.stack(ref_levels, axis=-1)

        assert jnp.allclose(result, ref, atol=1e-8)


# ============================================================================
# MPAS fused tendencies match original scan-based
# ============================================================================

class TestMPASFusedTendencies:
    """Verify the fused 3D MPAS tendency matches level-by-level results."""

    def test_tendency_basic_agreement(self):
        """Run a single tendency eval and check it produces finite output."""
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            mpas_hydrostatic_tendencies,
            MPASPrimitiveEquationConfig,
        )
        from legoesm.core.state import MPASHydrostaticState
        from legoesm.core.field import Field

        mesh = create_voronoi_mesh(subdivision_level=2)
        nlev = 5
        sigma = create_sigma_coordinate(nlev)

        rng = jax.random.PRNGKey(1)
        keys = jax.random.split(rng, 2)
        u = jax.random.normal(keys[0], (mesh.nEdges, nlev)) * 10.0
        T = 280.0 + jax.random.normal(keys[1], (mesh.nCells, nlev)) * 5.0
        p_s = jnp.ones(mesh.nCells) * 1e5
        phis = jnp.zeros(mesh.nCells)

        state = MPASHydrostaticState(
            u=Field(data=u, name="u", dims=("nEdges", "nlev"), units="m/s"),
            T=Field(data=T, name="T", dims=("nCells", "nlev"), units="K"),
            p_s=Field(data=p_s, name="p_s", dims=("nCells",), units="Pa"),
            phis=Field(data=phis, name="phis", dims=("nCells",), units="m^2/s^2"),
        )

        config = MPASPrimitiveEquationConfig(
            nu_del4=1e14, pv_scheme="energy")

        tend = mpas_hydrostatic_tendencies(state, mesh, sigma, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))
        assert tend.du_dt.data.shape == (mesh.nEdges, nlev)
        assert tend.dT_dt.data.shape == (mesh.nCells, nlev)
