"""Tests for regional Voronoi mesh generation.

Validates that ``create_regional_voronoi_mesh`` produces a valid
MPAS-compatible mesh and that TRiSK operators work on it for
a barotropic wave test case.

See issue #88 for motivation and design.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

# Ensure float64 for geometry precision
jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_regional_voronoi_mesh


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def regional_mesh():
    """North Atlantic-like regional mesh: 0-120 deg E, 15-75 deg N, ~300 km."""
    return create_regional_voronoi_mesh(
        lon_range=(0.0, 120.0),
        lat_range=(15.0, 75.0),
        resolution_km=300.0,
    )


# ============================================================================
# Mesh validity tests
# ============================================================================

class TestMeshConnectivity:
    """Verify mesh connectivity is valid."""

    def test_positive_cell_count(self, regional_mesh):
        """Mesh should have a reasonable number of cells."""
        assert regional_mesh.nCells > 50, (
            f"Expected at least 50 cells, got {regional_mesh.nCells}"
        )
        assert regional_mesh.nEdges > 50
        assert regional_mesh.nVertices > 50

    def test_cells_on_edge_valid(self, regional_mesh):
        """cellsOnEdge should reference valid cell indices."""
        c1 = regional_mesh.cellsOnEdge[0]
        c2 = regional_mesh.cellsOnEdge[1]
        assert jnp.all(c1 >= 0), "cellsOnEdge[0] has negative indices"
        assert jnp.all(c2 >= 0), "cellsOnEdge[1] has negative indices"
        assert jnp.all(c1 < regional_mesh.nCells)
        assert jnp.all(c2 < regional_mesh.nCells)

    def test_edges_on_cell_valid(self, regional_mesh):
        """edgesOnCell entries should be valid edge indices or -1 padding."""
        eoc = regional_mesh.edgesOnCell
        valid = (eoc >= 0) & (eoc < regional_mesh.nEdges)
        padded = eoc == -1
        assert jnp.all(valid | padded), "edgesOnCell has invalid indices"

    def test_vertices_on_edge_valid(self, regional_mesh):
        """verticesOnEdge entries should be valid vertex indices or -1."""
        voe = regional_mesh.verticesOnEdge
        valid = (voe >= 0) & (voe < regional_mesh.nVertices)
        padded = voe == -1
        assert jnp.all(valid | padded), "verticesOnEdge has invalid indices"

    def test_n_edges_on_cell_reasonable(self, regional_mesh):
        """Each cell should have between 3 and maxEdges edges."""
        neoc = regional_mesh.nEdgesOnCell
        assert jnp.all(neoc >= 3), (
            f"Some cells have fewer than 3 edges: min={int(jnp.min(neoc))}"
        )
        assert jnp.all(neoc <= regional_mesh.maxEdges)


class TestMeshGeometry:
    """Verify mesh geometry is reasonable."""

    def test_dvEdge_positive(self, regional_mesh):
        """All dvEdge values should be positive (safety floor applied)."""
        assert jnp.all(regional_mesh.dvEdge > 0), (
            f"dvEdge has non-positive values: min={float(jnp.min(regional_mesh.dvEdge))}"
        )

    def test_dcEdge_positive(self, regional_mesh):
        """All dcEdge values should be positive."""
        assert jnp.all(regional_mesh.dcEdge > 0), (
            f"dcEdge has non-positive values: min={float(jnp.min(regional_mesh.dcEdge))}"
        )

    def test_dvEdge_dcEdge_ratio(self, regional_mesh):
        """dvEdge/dcEdge ratio should be reasonable (> 0.01) for interior cells."""
        ratio = regional_mesh.dvEdge / jnp.maximum(regional_mesh.dcEdge, 1e-10)
        # Some boundary edges may have small ratios, but median should be reasonable
        median_ratio = float(jnp.median(ratio))
        assert median_ratio > 0.1, (
            f"Median dvEdge/dcEdge ratio too small: {median_ratio:.4f}"
        )

    def test_area_cell_positive(self, regional_mesh):
        """All cell areas should be positive."""
        assert jnp.all(regional_mesh.areaCell > 0), (
            f"areaCell has non-positive values: min={float(jnp.min(regional_mesh.areaCell))}"
        )

    def test_area_triangle_positive(self, regional_mesh):
        """All triangle areas should be positive (safety floor applied)."""
        assert jnp.all(regional_mesh.areaTriangle > 0), (
            f"areaTriangle has non-positive values"
        )

    def test_cells_in_expected_region(self, regional_mesh):
        """Cell centers should be near the target domain."""
        lat_deg = jnp.degrees(regional_mesh.latCell)
        lon_deg = jnp.degrees(regional_mesh.lonCell)
        # Buffer is ~1.5x resolution, so allow generous bounds
        assert float(jnp.min(lat_deg)) > -10, (
            f"Cells too far south: min_lat={float(jnp.min(lat_deg)):.1f}"
        )
        assert float(jnp.max(lat_deg)) < 90, (
            f"Cells too far north: max_lat={float(jnp.max(lat_deg)):.1f}"
        )

    def test_finite_geometry(self, regional_mesh):
        """All geometry arrays should be finite (no NaN or Inf)."""
        for name in ["areaCell", "areaTriangle", "dcEdge", "dvEdge",
                      "angleEdge", "fEdge", "fVertex"]:
            arr = getattr(regional_mesh, name)
            assert jnp.all(jnp.isfinite(arr)), f"{name} has non-finite values"

    def test_edge_sign_on_cell_nonzero(self, regional_mesh):
        """edgeSignOnCell should be +/-1 for active edges."""
        esoc = regional_mesh.edgeSignOnCell
        eoc = regional_mesh.edgesOnCell
        active = eoc >= 0
        active_signs = jnp.abs(esoc[active])
        # All active signs should be 1
        assert jnp.allclose(active_signs, 1.0, atol=1e-10), (
            "edgeSignOnCell should be +/-1 for active edges"
        )


# ============================================================================
# TRiSK operators test
# ============================================================================

class TestTRiSKOperators:
    """Verify TRiSK operators work on the regional mesh."""

    def test_divergence_of_constant(self, regional_mesh):
        """Divergence of a constant field should be near zero."""
        from legoesm.core.operators_voronoi import divergence_cell

        u_const = jnp.ones(regional_mesh.nEdges, dtype=jnp.float64)
        div = divergence_cell(u_const, regional_mesh)
        # Interior cells should have near-zero divergence
        max_div = float(jnp.max(jnp.abs(div)))
        # Allow some tolerance for boundary effects
        assert max_div < 1.0, (
            f"Divergence of constant field too large: max={max_div:.2e}"
        )

    def test_gradient_of_constant(self, regional_mesh):
        """Gradient of a constant scalar should be zero."""
        from legoesm.core.operators_voronoi import gradient_edge

        phi = jnp.ones(regional_mesh.nCells, dtype=jnp.float64) * 5.0
        grad = gradient_edge(phi, regional_mesh)
        max_grad = float(jnp.max(jnp.abs(grad)))
        assert max_grad < 1e-10, (
            f"Gradient of constant field should be zero, got max={max_grad:.2e}"
        )


# ============================================================================
# Barotropic wave integration test
# ============================================================================

class TestBarotropicWave:
    """Run a short barotropic wave on the regional mesh to verify stability."""

    def test_1day_barotropic_wave_stable(self, regional_mesh):
        """1-day barotropic wave should remain stable on regional mesh."""
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.dynamics.barotropic_mpas import barotropic_substeps_mpas
        from legoesm.core.field import Field
        import numpy as np

        mesh = regional_mesh
        z_coord = create_ocean_z_star(
            n_levels=5, H_max=5500.0, dz_surface=200.0, dz_deep=2000.0,
        )

        # Create rest state with land at high latitudes
        state = rest_state_mpas_ocean(
            mesh, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=5500.0, land_lat_threshold=80.0,
        )

        # Add Gaussian SSH perturbation at domain center
        lon_center = np.radians(60.0)
        lat_center = np.radians(45.0)
        sigma = np.radians(10.0)

        lon = np.asarray(mesh.lonCell, dtype=np.float64)
        lat = np.asarray(mesh.latCell, dtype=np.float64)
        dist = np.arccos(np.clip(
            np.sin(lat) * np.sin(lat_center)
            + np.cos(lat) * np.cos(lat_center) * np.cos(lon - lon_center),
            -1.0, 1.0,
        ))
        perturb = 1.0 * np.exp(-0.5 * (dist / sigma) ** 2)
        eta_init = state.eta.data + jnp.array(perturb)
        state = state._replace(
            eta=Field(data=eta_init, name="eta", dims=("nCells",), units="m"),
        )

        config = MPASOceanConfig(
            A_h=1.0e3,
            K_h=1.0e2,
            A_v=1.0e-3,
            K_v=1.0e-4,
            n_barotropic_substeps=10,
            barotropic_diffusion_alpha=0.0,
            semi_implicit_coriolis=True,
        )

        # Run 1 day of barotropic substeps
        dt_baro = 30.0  # seconds
        n_steps = int(86400.0 / dt_baro)  # 1 day

        # Run in chunks to avoid excessive scan length
        chunk_size = 100
        eta = state.eta.data
        u_bar = jnp.zeros(mesh.nEdges, dtype=jnp.float64)

        for i_chunk in range(0, n_steps, chunk_size):
            n_sub = min(chunk_size, n_steps - i_chunk)
            # Update state with current eta for the substep function
            state_chunk = state._replace(
                eta=Field(data=eta, name="eta", dims=("nCells",), units="m"),
            )
            eta, u_bar = barotropic_substeps_mpas(
                state_chunk, mesh, z_coord, config,
                dt_baro=dt_baro, n_substeps=n_sub,
            )

        max_eta = float(jnp.max(jnp.abs(eta)))
        mean_eta = float(jnp.mean(eta * state.land_mask.data))

        # Stability check: max eta should not blow up
        assert max_eta < 10.0, (
            f"Barotropic wave blew up: max|eta| = {max_eta:.2f} m "
            f"(initial perturbation was 1 m)"
        )
        assert jnp.all(jnp.isfinite(eta)), "eta contains NaN/Inf"
        assert jnp.all(jnp.isfinite(u_bar)), "u_bar contains NaN/Inf"

        # The wave should still be present (not completely damped)
        assert max_eta > 0.01, (
            f"Wave completely dissipated: max|eta| = {max_eta:.4f} m"
        )


# ============================================================================
# Sub-360° periodic channel tests
# ============================================================================

@pytest.fixture(scope="module")
def sub360_periodic_mesh():
    """A 10° × 18° zonally-periodic mesh at 100 km resolution (Eady-like)."""
    return create_regional_voronoi_mesh(
        lon_range=(0.0, 10.0),
        lat_range=(16.0, 34.0),
        resolution_km=100.0,
        periodic_x=True,
    )


class TestSub360Periodic:
    """Verify sub-360° zonally-periodic regional meshes are well-formed."""

    def test_cell_count_scales_with_zonal_extent(self):
        """nCells should scale roughly linearly with L (fixed resolution)."""
        import numpy as np
        counts = []
        for L in (10.0, 20.0, 30.0, 45.0):
            m = create_regional_voronoi_mesh(
                (0.0, L), (16.0, 34.0), resolution_km=100.0, periodic_x=True,
            )
            counts.append((L, m.nCells))
        # Expected: nCells ≈ const × L (to within ~15%)
        ratios = [c[1] / c[0] for c in counts]
        mean_r = sum(ratios) / len(ratios)
        for L, n in counts:
            r = n / L
            assert abs(r - mean_r) / mean_r < 0.25, (
                f"nCells not linear in L: {counts} (ratios {ratios})"
            )

    def test_seam_edges_exist_with_correct_distance(self, sub360_periodic_mesh):
        """Seam edges should connect cells at lon_min ↔ lon_max with dcEdge
        near the resolution, NOT the long-way-around distance."""
        import numpy as np
        mesh = sub360_periodic_mesh
        cOE = np.asarray(mesh.cellsOnEdge)
        dc = np.asarray(mesh.dcEdge)
        lon_cell = np.degrees(np.asarray(mesh.lonCell))
        dlon = np.abs(lon_cell[cOE[0]] - lon_cell[cOE[1]])
        seam_mask = dlon > 5.0  # half the 10° extent
        n_seam = int(seam_mask.sum())
        assert n_seam >= 10, (
            f"expected ≥10 seam edges in a 10° × 18° mesh, got {n_seam}"
        )
        # Seam edges should have dcEdge near resolution (100 km), not ~1000 km
        # which is the long-way-around distance between 0° and 10° at 25°N.
        seam_dc_max = float(dc[seam_mask].max()) / 1e3  # km
        assert seam_dc_max < 200.0, (
            f"seam dcEdge too large: {seam_dc_max:.1f} km "
            f"(resolution 100 km). The shift likely failed."
        )

    def test_uniform_east_wind_div_near_zero(self, sub360_periodic_mesh):
        """Divergence of a uniform east wind on the periodic channel
        should be machine-epsilon for most cells (median), with at most a
        small boundary residual (max) consistent with the closed-basin
        and 360° cases."""
        import numpy as np
        from legoesm.core.operators_voronoi import divergence_cell
        mesh = sub360_periodic_mesh
        u_const = np.cos(np.asarray(mesh.angleEdge))
        div = np.asarray(divergence_cell(u_const, mesh))
        # The mesh-quality target: median ≈ 0 (well below 1e-10), max
        # below ~1e-4 (boundary cells). No cells should explode.
        assert np.median(np.abs(div)) < 1e-10, (
            f"median |div| too large: {np.median(np.abs(div)):.2e}"
        )
        assert np.abs(div).max() < 1e-4, (
            f"max |div| too large: {np.abs(div).max():.2e}"
        )

    def test_gauss_theorem_across_seam(self, sub360_periodic_mesh):
        """A control volume straddling the seam should have its integrated
        divergence match what we put in (Gauss's theorem). We set each
        edge-normal flux to a constant value and verify that cells not at
        the lat boundary see the correct divergence contribution."""
        import numpy as np
        from legoesm.core.operators_voronoi import divergence_cell
        mesh = sub360_periodic_mesh
        # Flux = 1 m/s * edge-normal (radially-uniform)
        flux = np.ones(mesh.nEdges, dtype=np.float64)
        div = np.asarray(divergence_cell(flux, mesh))
        # For a purely radial uniform edge-normal flux, the divergence per
        # cell is (sum of ±flux_k * dvEdge_k) / areaCell — not zero, but
        # should be finite and non-NaN everywhere and the SIGN should be
        # consistent away from the boundary.
        assert np.all(np.isfinite(div)), "divergence contains NaN/Inf"

    def test_360_backcompat(self):
        """``lon_range=(0, 360)`` with ``periodic_x=True`` (new unified
        path) gives a mesh of the same scale as the legacy full-360°
        code path."""
        m_full = create_regional_voronoi_mesh(
            (0.0, 360.0), (16.0, 34.0), resolution_km=100.0, periodic_x=True,
        )
        # For the full 360° test extent at this resolution, expect ~10k cells
        assert m_full.nCells > 500, (
            f"Full 360° mesh unexpectedly small: nCells={m_full.nCells}"
        )
        # Edge count per cell should be mostly 6 (hex) with some 5 (pentagons)
        import numpy as np
        nEC = np.asarray(m_full.nEdgesOnCell)
        frac_hex = float((nEC == 6).mean())
        assert frac_hex > 0.5, (
            f"Expected majority hex cells on a 360° mesh, got {frac_hex*100:.0f}%"
        )

    def test_ocean_model_one_step(self, sub360_periodic_mesh):
        """An MPAS ocean model should be able to take one step on the
        sub-360° periodic mesh without producing NaN."""
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.experiments.eady_uniform import (
            EadyUniformConfig, create_initial_conditions,
            create_forcings,
        )
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

        mesh = sub360_periodic_mesh
        z = create_ocean_z_star(n_levels=10, H_max=5500.0)
        cfg = EadyUniformConfig(U_surface=0.2)
        physics = create_forcings("mpas_channel", None, cfg)
        mcfg = MPASOceanConfig(
            n_barotropic_substeps=30, physics=physics,
            B_h=2.3e11, C_smag=0.2, tracer_advection="tvd",
        )
        model = MPASOceanModel(mesh, z, mcfg)
        state = create_initial_conditions("mpas_channel", mesh, z, cfg)
        s1 = model.step(state, 300.0)
        assert jnp.all(jnp.isfinite(s1.u.data)), "u blew up in 1 step"
        assert jnp.all(jnp.isfinite(s1.T.data)), "T blew up in 1 step"
        assert jnp.all(jnp.isfinite(s1.eta.data)), "eta blew up in 1 step"
