"""Tests for MPAS Voronoi ocean model.

Tests cover:
- State construction
- Initialization (bathymetry, rest state)
- Velocity reconstruction
- Baroclinic tendencies
- Barotropic substeps
- Full model stepping
- Simple ocean modes
- Conservation fixers
- Coupler integration
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Ensure float64
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.mpas_config import MPASOceanConfig, MPASSimpleOceanConfig
from legoesm.ocean.init_mpas import (
    idealized_bathymetry_mpas,
    rest_state_mpas_ocean,
    reconstruct_cell_velocity,
)
from legoesm.ocean.conservation_mpas import (
    mpas_ocean_conservation_fixer,
)
from legoesm.ocean.simple_ocean_mpas import (
    MPASSlabOceanState,
    init_mpas_slab_state,
    make_mpas_ocean,
)
from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic_mpas import (
    barotropic_substeps_mpas,
    reconcile_3d_velocity,
)
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def mesh():
    """Level 2 Voronoi mesh (162 cells)."""
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_coord():
    """Shallow ocean z-star coordinate (5 levels)."""
    return create_ocean_z_star(n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0)


@pytest.fixture(scope="module")
def state(mesh, z_coord):
    """Rest-state initial condition."""
    return rest_state_mpas_ocean(
        mesh, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=500.0, land_lat_threshold=85.0,
    )


@pytest.fixture(scope="module")
def config():
    """Default MPAS ocean config with reduced viscosity for test mesh."""
    return MPASOceanConfig(
        A_h=1.0e3,
        K_h=1.0e2,
        A_v=1.0e-3,
        K_v=1.0e-4,
        n_barotropic_substeps=5,
    )


# ============================================================================
# Test: State Construction
# ============================================================================

class TestStateConstruction:
    """Test MPAS ocean state types."""

    def test_mpas_ocean_state_fields(self):
        """MPASOceanState has expected fields."""
        nCells, nEdges, nlev = 10, 20, 5
        state = MPASOceanState(
            u=Field(jnp.zeros((nEdges, nlev)), "u", ("nEdges", "nlev"), "m/s"),
            T=Field(jnp.ones((nCells, nlev)) * 15.0, "T", ("nCells", "nlev"), "degC"),
            S=Field(jnp.ones((nCells, nlev)) * 35.0, "S", ("nCells", "nlev"), "PSU"),
            eta=Field(jnp.zeros(nCells), "eta", ("nCells",), "m"),
            w=Field(jnp.zeros((nCells, nlev + 1)), "w", ("nCells", "nlev+1"), "m/s"),
            H_bathy=Field(jnp.full(nCells, 5000.0), "H_bathy", ("nCells",), "m"),
            land_mask=Field(jnp.ones(nCells), "land_mask", ("nCells",), "1"),
        )
        assert state.u.data.shape == (nEdges, nlev)
        assert state.T.data.shape == (nCells, nlev)
        assert state.eta.data.shape == (nCells,)
        assert state.w.data.shape == (nCells, nlev + 1)

    def test_mpas_ocean_tendencies_fields(self):
        """MPASOceanTendencies has expected fields."""
        nCells, nEdges, nlev = 10, 20, 5
        tend = MPASOceanTendencies(
            du_dt=Field(jnp.zeros((nEdges, nlev)), "du_dt", ("nEdges", "nlev"), "m/s²"),
            dT_dt=Field(jnp.zeros((nCells, nlev)), "dT_dt", ("nCells", "nlev"), "degC/s"),
            dS_dt=Field(jnp.zeros((nCells, nlev)), "dS_dt", ("nCells", "nlev"), "PSU/s"),
            deta_dt=Field(jnp.zeros(nCells), "deta_dt", ("nCells",), "m/s"),
        )
        assert tend.du_dt.data.shape == (nEdges, nlev)
        assert tend.deta_dt.data.shape == (nCells,)


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInitialization:
    """Test bathymetry and initial condition generation."""

    def test_bathymetry_shapes(self, mesh):
        """Bathymetry and land mask have correct shapes."""
        H_bathy, land_mask = idealized_bathymetry_mpas(mesh)
        assert H_bathy.shape == (mesh.nCells,)
        assert land_mask.shape == (mesh.nCells,)

    def test_bathymetry_values(self, mesh):
        """Bathymetry is H_max everywhere for smooth z-star Jacobian."""
        H_bathy, land_mask = idealized_bathymetry_mpas(mesh, H_max=1000.0)
        # H_bathy = H_max everywhere (including land) so that the z-star
        # Jacobian (eta + H)/H_max is smooth across coastlines.
        # The land_mask prevents actual flow on land cells.
        assert jnp.allclose(H_bathy, 1000.0)
        # Ocean cells identified correctly
        ocean_h = H_bathy[land_mask > 0.5]
        assert jnp.all(ocean_h > 0)

    def test_land_mask_threshold(self, mesh):
        """Land mask respects latitude threshold."""
        _, land_mask = idealized_bathymetry_mpas(mesh, land_lat_threshold=60.0)
        lat_deg = jnp.abs(jnp.degrees(mesh.latCell))
        # High latitudes should be land
        high_lat = lat_deg > 65.0
        assert jnp.all(land_mask[high_lat] == 0.0)

    def test_rest_state_shapes(self, state, mesh, z_coord):
        """Rest state has correct shapes."""
        assert state.u.data.shape == (mesh.nEdges, z_coord.n_levels)
        assert state.T.data.shape == (mesh.nCells, z_coord.n_levels)
        assert state.S.data.shape == (mesh.nCells, z_coord.n_levels)
        assert state.eta.data.shape == (mesh.nCells,)

    def test_rest_state_temperature_profile(self, state):
        """Temperature decreases with depth."""
        # Pick an ocean cell
        mask = state.land_mask.data
        ocean_cells = jnp.where(mask > 0.5)[0]
        if ocean_cells.size > 0:
            T_profile = state.T.data[ocean_cells[0]]
            # Surface should be warmer than deep
            assert T_profile[0] > T_profile[-1]

    def test_rest_state_zero_velocity(self, state):
        """Rest state has zero velocity."""
        assert jnp.allclose(state.u.data, 0.0)

    def test_rest_state_zero_eta(self, state):
        """Rest state has zero SSH."""
        assert jnp.allclose(state.eta.data, 0.0)


# ============================================================================
# Test: Velocity Reconstruction
# ============================================================================

class TestVelocityReconstruction:
    """Test reconstruction of cell-center velocity from edge normals."""

    def test_zero_velocity(self, mesh):
        """Zero edge velocity gives zero cell velocity."""
        u_edge = jnp.zeros(mesh.nEdges)
        u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
        assert jnp.allclose(u_east, 0.0)
        assert jnp.allclose(v_north, 0.0)

    def test_reconstruction_shape_1d(self, mesh):
        """1D reconstruction has correct shape."""
        u_edge = jnp.ones(mesh.nEdges) * 0.1
        u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
        assert u_east.shape == (mesh.nCells,)
        assert v_north.shape == (mesh.nCells,)

    def test_reconstruction_shape_3d(self, mesh):
        """3D reconstruction has correct shape."""
        nlev = 5
        u_edge = jnp.ones((mesh.nEdges, nlev)) * 0.1
        u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
        assert u_east.shape == (mesh.nCells, nlev)
        assert v_north.shape == (mesh.nCells, nlev)

    def test_reconstruction_finite(self, mesh):
        """Reconstructed velocity is finite."""
        u_edge = jnp.sin(mesh.angleEdge) * 0.5
        u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
        assert jnp.all(jnp.isfinite(u_east))
        assert jnp.all(jnp.isfinite(v_north))


# ============================================================================
# Test: Baroclinic Tendencies
# ============================================================================

class TestBaroclinicTendencies:
    """Test MPAS ocean baroclinic tendency computation."""

    def test_tendency_shapes(self, state, mesh, z_coord, config):
        """Tendencies have correct shapes."""
        tend = mpas_ocean_baroclinic_tendencies(state, mesh, z_coord, config)
        assert tend.du_dt.data.shape == state.u.data.shape
        assert tend.dT_dt.data.shape == state.T.data.shape
        assert tend.dS_dt.data.shape == state.S.data.shape
        assert tend.deta_dt.data.shape == state.eta.data.shape

    def test_tendency_finite(self, state, mesh, z_coord, config):
        """All tendencies are finite."""
        tend = mpas_ocean_baroclinic_tendencies(state, mesh, z_coord, config)
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_rest_state_small_tendencies(self, state, mesh, z_coord, config):
        """Rest state should have small tendencies."""
        tend = mpas_ocean_baroclinic_tendencies(state, mesh, z_coord, config)
        # Velocity tendency should be small (only from pressure gradient)
        assert jnp.max(jnp.abs(tend.du_dt.data)) < 1.0
        # SSH tendency should be very small (zero velocity)
        assert jnp.max(jnp.abs(tend.deta_dt.data)) < 1e-10

    def test_land_masking(self, state, mesh, z_coord, config):
        """Tendencies are zero on land cells."""
        tend = mpas_ocean_baroclinic_tendencies(state, mesh, z_coord, config)
        mask = state.land_mask.data
        land_cells = mask < 0.5
        if jnp.any(land_cells):
            assert jnp.allclose(tend.dT_dt.data[land_cells], 0.0)
            assert jnp.allclose(tend.dS_dt.data[land_cells], 0.0)
            assert jnp.allclose(tend.deta_dt.data[land_cells], 0.0)


class TestScalarLaplacianOperators:
    """Unit tests for ``laplacian_cell_3d`` / ``bilaplacian_cell_3d``."""

    def test_laplacian_of_constant_is_zero(self, mesh):
        """div(grad(const)) == 0 for cell-centered constant field."""
        from legoesm.core.operators_voronoi import laplacian_cell_3d
        nlev = 4
        f = jnp.full((mesh.nCells, nlev), 3.7)
        lap = laplacian_cell_3d(f, mesh)
        assert float(jnp.max(jnp.abs(lap))) == 0.0

    def test_bilaplacian_of_constant_is_zero(self, mesh):
        """Bilaplacian of a constant field is identically zero."""
        from legoesm.core.operators_voronoi import bilaplacian_cell_3d
        nlev = 3
        f = jnp.full((mesh.nCells, nlev), -1.25)
        bilap = bilaplacian_cell_3d(f, mesh)
        assert float(jnp.max(jnp.abs(bilap))) == 0.0

    def test_bilaplacian_equals_lap_of_lap(self, mesh):
        """``bilaplacian_cell_3d(f) == laplacian_cell_3d(laplacian_cell_3d(f))``."""
        import numpy as np
        from legoesm.core.operators_voronoi import (
            laplacian_cell_3d, bilaplacian_cell_3d,
        )
        rng = np.random.default_rng(0)
        f = jnp.asarray(rng.normal(size=(mesh.nCells, 5)))
        direct = bilaplacian_cell_3d(f, mesh)
        chained = laplacian_cell_3d(laplacian_cell_3d(f, mesh), mesh)
        assert jnp.allclose(direct, chained)

    def test_mask_zeroes_land_output(self, mesh):
        """Cell mask zeroes the output on land and gradients at coastlines."""
        import numpy as np
        from legoesm.core.operators_voronoi import (
            laplacian_cell_3d, bilaplacian_cell_3d,
        )
        rng = np.random.default_rng(1)
        f = jnp.asarray(rng.normal(size=(mesh.nCells, 4)))
        land = jnp.arange(mesh.nCells) < mesh.nCells // 4
        mask = (~land).astype(jnp.float64)
        lap = laplacian_cell_3d(f, mesh, mask=mask)
        bilap = bilaplacian_cell_3d(f, mesh, mask=mask)
        assert jnp.allclose(lap[land], 0.0)
        assert jnp.allclose(bilap[land], 0.0)

    def test_mask_matches_hand_applied_mask(self, mesh):
        """mask= kw equals explicitly applying edge+cell masks step by step."""
        import numpy as np
        from legoesm.core.operators_voronoi import (
            laplacian_cell_3d, divergence_cell_3d, gradient_edge_3d,
        )
        rng = np.random.default_rng(2)
        f = jnp.asarray(rng.normal(size=(mesh.nCells, 3)))
        land = jnp.arange(mesh.nCells) < mesh.nCells // 3
        mask = (~land).astype(jnp.float64)
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        edge_mask = mask[c1] * mask[c2]
        grad = gradient_edge_3d(f, mesh) * edge_mask[:, None]
        lap_manual = divergence_cell_3d(grad, mesh) * mask[:, None]
        lap_op = laplacian_cell_3d(f, mesh, mask=mask)
        assert jnp.allclose(lap_manual, lap_op)


class TestBiharmonicTracerDiffusion:
    """Issue #206: scalar biharmonic tracer diffusion (K_bih) on MPAS."""

    def _perturbed_state(self, state, seed=0, amp=0.1):
        import numpy as np
        rng = np.random.default_rng(seed)
        noise = jnp.asarray(rng.normal(size=state.T.data.shape)) * amp
        return state._replace(T=state.T.replace(data=state.T.data + noise))

    def test_config_default_is_zero(self):
        """K_bih defaults to 0 — no behaviour change when unused."""
        assert MPASOceanConfig().K_bih == 0.0

    def test_kbih_zero_reproduces_explicit_harmonic_formula(self, state, mesh, z_coord):
        """K_bih=0 must reproduce the K_h-only harmonic tracer tendency *formula*.

        Instead of comparing two semantically identical configs, rebuild the
        pre-K_bih horizontal tracer tendency directly from its algebraic
        definition

            dT/dt_horiz = K_h * div(grad(T) * edge_mask) / h_safe * h_k

        on a minimal setup (no vertical mixing, no harmonic advection) and
        assert the MPAS tendency equals this formula when K_bih=0.  If the
        new branch ever leaks computation into K_bih=0, the hand-derived
        reference will diverge.
        """
        from legoesm.core.operators_voronoi import (
            divergence_cell_3d, gradient_edge_3d,
        )
        from legoesm.ocean.vertical import compute_layer_thickness

        K_h = 1.0e2
        cfg = MPASOceanConfig(K_h=K_h, K_bih=0.0, A_v=0.0, K_v=0.0,
                               n_barotropic_substeps=5)
        perturbed = self._perturbed_state(state)
        tend = mpas_ocean_baroclinic_tendencies(perturbed, mesh, z_coord, cfg)

        # Hand-computed harmonic horizontal tracer tendency.
        cell_mask = state.land_mask.data
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        edge_mask = cell_mask[c1] * cell_mask[c2]
        h_k = compute_layer_thickness(
            perturbed.eta.data, perturbed.H_bathy.data, z_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        h_safe = jnp.maximum(h_k, 1e-10)

        def _kh_lap(f):
            grad = gradient_edge_3d(f, mesh) * edge_mask[:, jnp.newaxis]
            return (K_h * divergence_cell_3d(grad, mesh) / h_safe * h_k
                    * cell_mask[:, jnp.newaxis])

        expected_dT = _kh_lap(perturbed.T.data)
        expected_dS = _kh_lap(perturbed.S.data)
        # The K_bih=0 model path shares the inner ``div(grad*edge_mask)`` between
        # the K_h Laplacian and the (skipped) biharmonic, and computes the tracer
        # gradient batched over a stacked [T,S] field.  That reorders the
        # floating-point ops relative to this per-tracer hand formula, so the two
        # agree to ~1 ULP (measured max relative diff 2.6e-16), NOT bit-for-bit.
        # A real leak of the K_bih term into the K_bih=0 path would be
        # order-unity relative, so a machine-precision tolerance still guards it.
        assert jnp.allclose(tend.dT_dt.data, expected_dT, atol=1e-15, rtol=1e-9)
        assert jnp.allclose(tend.dS_dt.data, expected_dS, atol=1e-15, rtol=1e-9)

    def test_kbih_activation_changes_tracer_tendency(self, state, mesh, z_coord):
        """K_bih>0 must measurably alter the tracer tendency."""
        cfg_off = MPASOceanConfig(K_h=0.0, K_bih=0.0, A_v=0.0, K_v=0.0,
                                   n_barotropic_substeps=5)
        cfg_on = MPASOceanConfig(K_h=0.0, K_bih=1.0e10, A_v=0.0, K_v=0.0,
                                  n_barotropic_substeps=5)
        perturbed = self._perturbed_state(state)
        t_off = mpas_ocean_baroclinic_tendencies(perturbed, mesh, z_coord, cfg_off)
        t_on = mpas_ocean_baroclinic_tendencies(perturbed, mesh, z_coord, cfg_on)
        diff = jnp.max(jnp.abs(t_on.dT_dt.data - t_off.dT_dt.data))
        assert float(diff) > 0.0
        assert jnp.all(jnp.isfinite(t_on.dT_dt.data))
        assert jnp.all(jnp.isfinite(t_on.dS_dt.data))

    def test_kbih_respects_land_mask(self, state, mesh, z_coord):
        """Bilaplacian diffusion must not inject tendency on land cells."""
        cfg_on = MPASOceanConfig(K_h=0.0, K_bih=1.0e10, A_v=0.0, K_v=0.0,
                                  n_barotropic_substeps=5)
        perturbed = self._perturbed_state(state)
        tend = mpas_ocean_baroclinic_tendencies(perturbed, mesh, z_coord, cfg_on)
        land = state.land_mask.data < 0.5
        if jnp.any(land):
            assert jnp.allclose(tend.dT_dt.data[land], 0.0)
            assert jnp.allclose(tend.dS_dt.data[land], 0.0)

    def test_kbih_opposes_harmonic_sign_on_perturbation(self, state, mesh, z_coord):
        """K_bih applies ``-K_bih * ∇⁴T`` (scale-selective dissipation).

        For a small-scale tracer perturbation, bilap(T) has the same sign
        as T at the perturbation peak, so the biharmonic tendency damps
        the peak (opposite sign to the perturbation), matching the
        latlon ``bilaplacian_cgrid`` convention in ocean_pe_latlon_cgrid.
        """
        mask = state.land_mask.data
        ocean_idx = int(jnp.argmax(mask))
        bump = jnp.zeros_like(state.T.data).at[ocean_idx, 0].set(1.0)
        perturbed = state._replace(T=state.T.replace(data=state.T.data + bump))
        cfg_on = MPASOceanConfig(K_h=0.0, K_bih=1.0e10, A_v=0.0, K_v=0.0,
                                  n_barotropic_substeps=5)
        cfg_off = MPASOceanConfig(K_h=0.0, K_bih=0.0, A_v=0.0, K_v=0.0,
                                   n_barotropic_substeps=5)
        t_on = mpas_ocean_baroclinic_tendencies(perturbed, mesh, z_coord, cfg_on)
        t_off = mpas_ocean_baroclinic_tendencies(perturbed, mesh, z_coord, cfg_off)
        delta = t_on.dT_dt.data - t_off.dT_dt.data
        # Damping at the bump location (top level where the bump lives):
        # with -K_bih*bilap and bump at centre → ΔdT/dt < 0 at centre.
        assert float(delta[ocean_idx, 0]) < 0.0

    def test_bilaplacian_area_sum_near_zero_on_closed_mesh(self, mesh):
        """``Σ_c A_c · bilap(T)_c ≈ 0`` on a closed (unmasked) mesh.

        Two passes of ``div(grad(·))`` telescope on a closed manifold:
        the area-weighted sum of the divergence of a cell-centred field
        is zero by discrete Stokes. This is the finite-volume
        conservation property that prevents the biharmonic from leaking
        area-integrated tracer mass/heat/salt in the open ocean interior.
        """
        import numpy as np
        from legoesm.core.operators_voronoi import bilaplacian_cell_3d
        rng = np.random.default_rng(123)
        nlev = 3
        f = jnp.asarray(rng.normal(size=(mesh.nCells, nlev)))
        bilap = bilaplacian_cell_3d(f, mesh)
        area_weighted_sum = jnp.sum(mesh.areaCell[:, None] * bilap, axis=0)
        scale = float(jnp.sum(mesh.areaCell)) * float(jnp.max(jnp.abs(f)))
        # Discrete Stokes identity — tolerance scaled to area·peak magnitude.
        assert float(jnp.max(jnp.abs(area_weighted_sum))) < 1e-10 * max(scale, 1.0)

    def test_bilaplacian_quadratic_form_dissipative(self, mesh):
        """``∫ f·∇⁴f dA = ∫ (∇²f)² dA ≥ 0`` — the key damping property.

        For a closed Voronoi mesh, integration by parts yields
        ``⟨f, bilap(f)⟩_A = ⟨lap(f), lap(f)⟩_A ≥ 0`` (for the scalar
        Laplacian built from ``div(grad())``).  Therefore
        ``-K_bih · bilap`` dissipates the quadratic tracer variance
        ``⟨f, f⟩_A`` at a scale-selective rate proportional to ``k⁴`` —
        the defining feature of scale-selective biharmonic damping.
        """
        import numpy as np
        from legoesm.core.operators_voronoi import (
            bilaplacian_cell_3d, laplacian_cell_3d,
        )
        rng = np.random.default_rng(7)
        f = jnp.asarray(rng.normal(size=(mesh.nCells, 1)))
        bilap = bilaplacian_cell_3d(f, mesh)
        lap = laplacian_cell_3d(f, mesh)
        lhs = float(jnp.sum(mesh.areaCell[:, None] * f * bilap))
        rhs = float(jnp.sum(mesh.areaCell[:, None] * lap * lap))
        assert lhs >= 0.0
        # Integration-by-parts identity (closed mesh, no coastlines).
        assert abs(lhs - rhs) < 1e-10 * max(abs(rhs), 1.0)


# ============================================================================
# Test: Barotropic Substeps
# ============================================================================

class TestBarotropicSubsteps:
    """Test barotropic forward-backward solver."""

    def test_barotropic_shapes(self, state, mesh, z_coord, config):
        """Barotropic substeps return correct shapes."""
        dt_baro = 10.0
        eta_new, u_bar_new, Hu_avg = barotropic_substeps_mpas(
            state, mesh, z_coord, config, dt_baro, 3,
        )
        assert eta_new.shape == state.eta.data.shape
        assert u_bar_new.shape == (mesh.nEdges,)
        assert Hu_avg.shape == (mesh.nEdges,)

    def test_barotropic_finite(self, state, mesh, z_coord, config):
        """Barotropic output is finite."""
        dt_baro = 10.0
        eta_new, u_bar_new, Hu_avg = barotropic_substeps_mpas(
            state, mesh, z_coord, config, dt_baro, 3,
        )
        assert jnp.all(jnp.isfinite(eta_new))
        assert jnp.all(jnp.isfinite(u_bar_new))
        assert jnp.all(jnp.isfinite(Hu_avg))

    def test_rest_state_barotropic_stable(self, state, mesh, z_coord, config):
        """Rest state remains at rest through barotropic substeps."""
        dt_baro = 10.0
        eta_new, u_bar_new, Hu_avg = barotropic_substeps_mpas(
            state, mesh, z_coord, config, dt_baro, 5,
        )
        assert jnp.max(jnp.abs(eta_new - state.eta.data)) < 1e-10
        assert jnp.max(jnp.abs(u_bar_new)) < 1e-10

    def test_reconcile_shapes(self, state, mesh):
        """Velocity reconciliation preserves shape."""
        nlev = state.u.data.shape[1]
        u_bar_old = jnp.zeros(mesh.nEdges)
        u_bar_new = jnp.ones(mesh.nEdges) * 0.01
        h_k = jnp.ones((mesh.nCells, nlev)) * 100.0
        mask = state.land_mask.data

        u_3d_new = reconcile_3d_velocity(
            state.u.data, u_bar_old, u_bar_new, mesh, mask,
        )
        assert u_3d_new.shape == state.u.data.shape

    def test_barotropic_u_viscosity_damps_grid_noise(
        self, state, mesh, z_coord, config,
    ):
        """``barotropic_u_viscosity`` damps grid-scale noise on u_bar.

        Targets the TRiSK rotational null branch on hexagonal C-grids
        (Thuburn 2008; Ringler et al. 2010, JCP §6) — a noise mode
        that has both ∇·u_bar ≈ 0 and is not damped by eta diffusion
        or divergence damping.

        Strategy: seed the 3D velocity with random edge noise (which
        projects onto all wavenumbers including the null branch),
        run a single barotropic substep with and without viscosity,
        and assert that the viscous run has significantly lower
        u_bar variance.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        u_noise = jnp.asarray(
            0.01 * rng.standard_normal(state.u.data.shape),
            dtype=state.u.data.dtype,
        )
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        edge_mask = state.land_mask.data[c1] * state.land_mask.data[c2]
        u_noise = u_noise * edge_mask[:, None]

        state_noisy = state._replace(u=state.u.replace(data=u_noise))

        dt_baro = 30.0
        n_sub = 30

        # Run without viscosity (baseline)
        cfg_off = config._replace(
            barotropic_u_viscosity=0.0,
            barotropic_diffusion_alpha=0.0,
            barotropic_div_damp=0.0,
        )
        _, u_bar_off, _ = barotropic_substeps_mpas(
            state_noisy, mesh, z_coord, cfg_off, dt_baro, n_sub,
        )

        # Run with viscosity ON. On the level-2 mesh (~2400 km),
        # forward-Euler stability requires A * dt / dx² < 0.5; with
        # dt=30 s and dx²≈7e12 this gives A < 1.2e11. Pick A=1e10:
        # diffusion timescale dx²/A ≈ 700 s, so ~30 substeps × 30 s
        # = 900 s gives an O(1) reduction at the highest wavenumbers.
        cfg_on = config._replace(
            barotropic_u_viscosity=1.0e10,
            barotropic_diffusion_alpha=0.0,
            barotropic_div_damp=0.0,
        )
        _, u_bar_on, _ = barotropic_substeps_mpas(
            state_noisy, mesh, z_coord, cfg_on, dt_baro, n_sub,
        )

        var_off = float(jnp.var(u_bar_off))
        var_on = float(jnp.var(u_bar_on))
        assert var_on < 0.5 * var_off, (
            f"u_bar viscosity should reduce variance by ≥2×; "
            f"got var_off={var_off:.3e}, var_on={var_on:.3e}"
        )

    def test_barotropic_u_viscosity_default_off(self, config):
        """Default config keeps u_bar viscosity disabled for bit-stability."""
        assert config.barotropic_u_viscosity == 0.0


# ============================================================================
# Test: Full Model
# ============================================================================

class TestMPASOceanModel:
    """Test the full MPAS ocean model."""

    def test_model_creation(self, mesh, z_coord, config):
        """Model can be created."""
        model = MPASOceanModel(mesh, z_coord, config)
        assert model.mesh is mesh
        assert model.z_coord is z_coord

    def test_single_step(self, mesh, z_coord, config, state):
        """Model can take a single step."""
        model = MPASOceanModel(mesh, z_coord, config)
        dt = 60.0
        state_new = model.step(state, dt)
        assert state_new.u.data.shape == state.u.data.shape
        assert state_new.T.data.shape == state.T.data.shape
        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))

    def test_static_fields_preserved(self, mesh, z_coord, config, state):
        """Static fields (H_bathy, land_mask) unchanged after step."""
        model = MPASOceanModel(mesh, z_coord, config)
        state_new = model.step(state, 60.0)
        assert jnp.allclose(state_new.H_bathy.data, state.H_bathy.data)
        assert jnp.allclose(state_new.land_mask.data, state.land_mask.data)

    def test_multi_step_stability(self, mesh, z_coord, config, state):
        """Model is stable for multiple steps."""
        model = MPASOceanModel(mesh, z_coord, config)
        s = state
        dt = 30.0
        for _ in range(5):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.eta.data))

    def test_step_checked_finite_state(self, mesh, z_coord, config, state):
        """step_checked must validate finite state without crashing.

        Regression test for an AttributeError on jax bool-array `.broadcast_to`
        in `_assert_runtime_invariants` (#174 item 6). Default T/S bounds are
        wide enough for the rest state, so the validator should pass through
        cleanly and return a finite stepped state.
        """
        config_rc = config._replace(enable_runtime_checks=True)
        model = MPASOceanModel(mesh, z_coord, config_rc)
        state_new = model.step_checked(state, dt=60.0)
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.S.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))

    def test_step_preserves_land_tracer_values(
        self, mesh, z_coord, config, state,
    ):
        """Regression for issue #164 bug 1.

        The old code zeroed T and S on land cells at the end of every
        flux-form tracer update (``jnp.where(mask > 0.5, tr_new, 0.0)``
        in ``ocean_model_mpas.py``). Combined with the single-iteration
        Neumann fill, that produced a cold/fresh front that propagated
        one cell per step along coastlines. The fix preserves the
        pre-step tracer value on land, matching the lat-lon pattern.
        """
        mask_1d = state.land_mask.data
        is_land = mask_1d < 0.5
        # Sanity check: the default rest state has at least one land cell.
        assert int(jnp.sum(is_land)) > 0, (
            "rest_state_mpas_ocean fixture has no land cells; test cannot "
            "discriminate the #164 fix."
        )
        T_land_before = state.T.data[is_land]
        S_land_before = state.S.data[is_land]
        # Rest state is non-trivially warm and salty.
        assert float(jnp.min(T_land_before)) > 1.0
        assert float(jnp.min(S_land_before)) > 10.0

        model = MPASOceanModel(mesh, z_coord, config)
        state_new = model.step(state, 60.0)

        T_land_after = state_new.T.data[is_land]
        S_land_after = state_new.S.data[is_land]
        # The old bug would make these exactly zero.
        assert float(jnp.min(T_land_after)) > 1.0, (
            f"Land T zeroed after step — bug #164 is back. "
            f"min={float(jnp.min(T_land_after))}"
        )
        assert float(jnp.min(S_land_after)) > 10.0, (
            f"Land S zeroed after step — bug #164 is back. "
            f"min={float(jnp.min(S_land_after))}"
        )
        # And the land values should track the pre-step values closely
        # (preservation, not arbitrary drift). Tolerance accommodates
        # fp32 round-off accumulated through the fill average and the
        # h_k_old / h_k_new ratio in the flux-form tracer update —
        # orders of magnitude below the ~20 K delta the old bug
        # would produce.
        assert float(jnp.max(jnp.abs(T_land_after - T_land_before))) < 1e-3
        assert float(jnp.max(jnp.abs(S_land_after - S_land_before))) < 1e-3


# ============================================================================
# Test: Land-cell Neumann fill
# ============================================================================

class TestMPASLandFill:
    """Regression tests for ``fill_land_cells_mpas`` (issue #164 bug 2)."""

    @staticmethod
    def _chain_connectivity(n):
        """Connectivity of a linear chain of n cells: edge i joins cells i
        and i+1.  Returns (c1, c2, edgesOnCell (2, n), nEdgesOnCell)."""
        c1 = jnp.arange(n - 1)
        c2 = jnp.arange(1, n)
        left = jnp.arange(n) - 1                     # edge to the left
        right = jnp.arange(n)                        # edge to the right
        # end cells have one edge; pad the unused slot with a valid index
        first = jnp.where(left < 0, right, left)
        second = jnp.where((left >= 0) & (right < n - 1), right, 0)
        eoc = jnp.stack([first, second])
        neoc = jnp.where((jnp.arange(n) == 0) | (jnp.arange(n) == n - 1), 1, 2)
        return c1, c2, eoc, neoc

    def test_single_iter_reaches_only_one_ring(self):
        """With n_iter=1, only land cells adjacent to ocean get filled.

        Documents the old (single-iteration) behaviour as a regression
        guard: n_iter=1 fills the first ring but leaves deeper interior
        land cells at their stale value. This is what the MPAS ocean
        had before issue #164 — any land cell two or more edges from
        ocean stayed zero.
        """
        from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

        # Chain: ocean ocean ocean land land land
        field = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        c1, c2, eoc, neoc = self._chain_connectivity(6)

        filled = fill_land_cells_mpas(field, mask, c1, c2, eoc, neoc, n_iter=1)
        # Cell 3 (1 edge from ocean) → filled with cell-2 value.
        assert float(filled[3]) == 3.0
        # Cells 4, 5 (2-3 edges from ocean) → unfilled.
        assert float(filled[4]) == 0.0
        assert float(filled[5]) == 0.0

    def test_three_iter_reaches_three_rings(self):
        """With n_iter=3 (new default), land cells up to 3 edges from
        ocean get filled. This matches the lat-lon
        ``neumann_fill_cgrid`` 3-pass behaviour.
        """
        from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

        field = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        c1, c2, eoc, neoc = self._chain_connectivity(6)

        filled = fill_land_cells_mpas(field, mask, c1, c2, eoc, neoc)  # default n_iter=3
        # Every land cell in the chain is reachable within 3 edges.
        assert float(filled[3]) == 3.0
        assert float(filled[4]) == 3.0
        assert float(filled[5]) == 3.0
        # Ocean cells untouched.
        assert float(filled[0]) == 1.0
        assert float(filled[1]) == 2.0
        assert float(filled[2]) == 3.0

    def test_fill_preserves_ocean_values_2d(self):
        """2D (per-level) field: fill must not mutate ocean cells or
        collapse across levels.
        """
        from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

        nlev = 3
        field = jnp.stack(
            [
                jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
                jnp.array([4.0, 5.0, 6.0, 0.0, 0.0, 0.0]),
                jnp.array([7.0, 8.0, 9.0, 0.0, 0.0, 0.0]),
            ],
            axis=-1,
        )  # (6, 3)
        mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        c1, c2, eoc, neoc = self._chain_connectivity(6)

        filled = fill_land_cells_mpas(field, mask, c1, c2, eoc, neoc)  # n_iter=3
        # Ocean cells preserved per-level.
        assert jnp.allclose(filled[0], jnp.array([1.0, 4.0, 7.0]))
        assert jnp.allclose(filled[2], jnp.array([3.0, 6.0, 9.0]))
        # Land cells get the last-ocean-cell value per level.
        assert jnp.allclose(filled[3], jnp.array([3.0, 6.0, 9.0]))
        assert jnp.allclose(filled[5], jnp.array([3.0, 6.0, 9.0]))

    def test_deep_land_beyond_n_iter_unchanged(self):
        """Land cells deeper than n_iter edges from ocean stay at
        their stale value. Documents the known limitation of the
        iterative fill — the 'proper' fix (precomputed nearest-ocean
        lookup) is still future work per issue #164's suggested fix.
        """
        from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

        # 6 ocean + 5 land; land cells 4..8 are 1..5 edges from ocean.
        n = 11
        field = jnp.array(
            [1.0] * 6 + [-99.0] * 5, dtype=jnp.float32,
        )
        mask = jnp.array([1.0] * 6 + [0.0] * 5, dtype=jnp.float32)
        c1, c2, eoc, neoc = self._chain_connectivity(n)

        filled = fill_land_cells_mpas(field, mask, c1, c2, eoc, neoc, n_iter=3)
        # Cells 6, 7, 8 are 1, 2, 3 edges from ocean → filled.
        assert float(filled[6]) == 1.0
        assert float(filled[7]) == 1.0
        assert float(filled[8]) == 1.0
        # Cells 9, 10 are 4, 5 edges from ocean → unchanged.
        assert float(filled[9]) == -99.0
        assert float(filled[10]) == -99.0


# ============================================================================
# Test: Conservation
# ============================================================================

    def test_gather_matches_frozen_edge_scatter_on_mesh(self):
        """The per-cell gather visits exactly the edges the former per-edge
        scatter-add did: on a real SCVT mesh with a random coastline the two
        agree to rounding (only the summation order differs)."""
        import numpy as np
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

        def scatter_ref(field, m, c1, c2, n_iter=3):
            filled = field
            for _ in range(n_iter):
                cnt = jnp.zeros_like(m).at[c1].add(m[c2]).at[c2].add(m[c1])
                has = cnt > 0.0
                if filled.ndim == 1:
                    vs = jnp.zeros_like(filled).at[c1].add(filled[c2] * m[c2])
                    vs = vs.at[c2].add(filled[c1] * m[c1])
                    avg, can = vs / jnp.maximum(cnt, 1.0), (m < 0.5) & has
                else:
                    vs = jnp.zeros_like(filled).at[c1].add(filled[c2] * m[c2, None])
                    vs = vs.at[c2].add(filled[c1] * m[c1, None])
                    avg = vs / jnp.maximum(cnt[:, None], 1.0)
                    can = ((m < 0.5) & has)[:, None]
                filled = jnp.where(can, avg, filled)
                m = jnp.where((m < 0.5) & has, 1.0, m)
            return filled

        mesh = create_voronoi_mesh(subdivision_level=4, lloyd_iterations=0)
        rng = np.random.default_rng(7)
        lon, lat = np.asarray(mesh.lonCell), np.asarray(mesh.latCell)
        mask = jnp.asarray((np.sin(3 * lon) * np.cos(2 * lat) < 0.3).astype(np.float64))
        c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
        for shape in ((mesh.nCells,), (mesh.nCells, 5)):
            f = jnp.asarray(rng.normal(size=shape))
            new = fill_land_cells_mpas(f, mask, c1, c2, mesh.edgesOnCell, mesh.nEdgesOnCell)
            ref = scatter_ref(f, mask, c1, c2)
            np.testing.assert_allclose(np.asarray(new), np.asarray(ref), rtol=1e-13, atol=1e-15)
            assert int(np.sum(np.asarray(new) != np.asarray(f))) > 0  # something was filled

class TestConservation:
    """Test conservation fixers."""

    def test_volume_conservation(self, state, mesh, z_coord, config):
        """Volume fixer restores total volume.

        Exercises the combined ``mpas_ocean_conservation_fixer`` with only the
        volume leg enabled (the standalone ``fix_volume_mpas`` had no
        production caller and was removed; the eta correction is identical).
        ``min_water_column_m=None`` reproduces the unfloored standalone call.
        """
        mask = state.land_mask.data
        area = mesh.areaCell

        # Perturb eta
        eta_perturbed = state.eta.data + 0.1 * jnp.sin(mesh.latCell) * mask
        state_new = state._replace(
            eta=state.eta.replace(data=eta_perturbed),
        )

        cfg = config._replace(
            fix_volume=True, fix_heat=False, fix_salt=False,
            min_water_column_m=None,
        )
        # Verify in float64 — the fixer accumulates in float64 but the
        # corrected state may be float32 under the default precision policy.
        _f64 = jnp.float64
        vol_before = jnp.sum(state.eta.data.astype(_f64) * mask.astype(_f64) * area.astype(_f64))
        state_fixed = mpas_ocean_conservation_fixer(
            state_new, state, mesh, z_coord, cfg,
        )
        vol_after = jnp.sum(state_fixed.eta.data.astype(_f64) * mask.astype(_f64) * area.astype(_f64))

        # Use absolute tolerance when reference volume is near zero.
        # With float32 state, the correction's float32 representation
        # introduces O(nCells * eps_f32 * |correction|) residual.
        total_area = jnp.sum(mask.astype(_f64) * area.astype(_f64))
        assert jnp.abs(vol_after - vol_before) < 1e-5 * total_area

    def test_heat_conservation(self, state, mesh, z_coord, config):
        """Heat fixer restores total heat content.

        Heat-only leg of the combined ``mpas_ocean_conservation_fixer`` (volume
        off, so the layer thickness is unchanged and the correction matches the
        removed standalone ``fix_heat_mpas``).  ``min_water_column_m=None``
        reproduces the unfloored standalone call.
        """
        mask = state.land_mask.data
        H_bathy = state.H_bathy.data

        # Perturb temperature
        T_perturbed = state.T.data + 0.5 * jnp.sin(
            mesh.latCell[:, jnp.newaxis]
        ) * mask[:, jnp.newaxis]
        state_new = state._replace(
            T=state.T.replace(data=T_perturbed),
        )

        cfg = config._replace(
            fix_volume=False, fix_heat=True, fix_salt=False,
            min_water_column_m=None,
        )
        _f64 = jnp.float64
        h_k = compute_layer_thickness(state.eta.data, H_bathy, z_coord)
        heat_before = jnp.sum(
            state.T.data.astype(_f64) * h_k.astype(_f64)
            * mask.astype(_f64)[:, jnp.newaxis] * mesh.areaCell.astype(_f64)[:, jnp.newaxis]
        )

        state_fixed = mpas_ocean_conservation_fixer(
            state_new, state, mesh, z_coord, cfg,
        )
        h_k_new = compute_layer_thickness(state_fixed.eta.data, H_bathy, z_coord)
        heat_after = jnp.sum(
            state_fixed.T.data.astype(_f64) * h_k_new.astype(_f64)
            * mask.astype(_f64)[:, jnp.newaxis] * mesh.areaCell.astype(_f64)[:, jnp.newaxis]
        )

        rel_err = jnp.abs(heat_after - heat_before) / jnp.maximum(
            jnp.abs(heat_before), 1e-30,
        )
        assert rel_err < 1e-5

    def test_full_fixer(self, state, mesh, z_coord, config):
        """Full conservation fixer chain works."""
        state_perturbed = state._replace(
            eta=state.eta.replace(
                data=state.eta.data + 0.01 * state.land_mask.data,
            ),
        )
        state_fixed = mpas_ocean_conservation_fixer(
            state_perturbed, state, mesh, z_coord, config,
        )
        assert jnp.all(jnp.isfinite(state_fixed.eta.data))
        assert jnp.all(jnp.isfinite(state_fixed.T.data))

    def test_fixer_runs_under_fp32_policy(
        self, state, mesh, z_coord, config,
    ):
        """Regression for issue #167.

        The previous implementation hard-coded float64 upcasts in every
        reduction, which crashed on backends without x64 support (notably
        Apple Metal). The fix routes all accumulations through the
        ``ocean_diagnostics`` precision policy. Under a pure fp32 policy
        with no module overrides — simulating the Metal backend on an
        x64-capable host — the fixer must run and return finite, fp32
        output instead of silently upcasting back to fp64.
        """
        from legoesm.core.precision import (
            PrecisionPolicy,
            get_policy,
            set_policy,
            clear_module_overrides,
            get_module_overrides,
            set_module_override,
        )

        prev_policy = get_policy()
        prev_overrides = get_module_overrides()

        try:
            set_policy(PrecisionPolicy.fp32())
            clear_module_overrides()

            # Cast the fixture state down to fp32 to match the policy.
            def _to_fp32(leaf):
                if (
                    isinstance(leaf, jax.Array)
                    and jnp.issubdtype(leaf.dtype, jnp.floating)
                ):
                    return leaf.astype(jnp.float32)
                return leaf

            state_fp32 = jax.tree.map(_to_fp32, state)

            state_new = state_fp32._replace(
                eta=state_fp32.eta.replace(
                    data=state_fp32.eta.data
                    + jnp.float32(0.01) * state_fp32.land_mask.data,
                ),
            )

            state_fixed = mpas_ocean_conservation_fixer(
                state_new, state_fp32, mesh, z_coord, config,
            )

            assert jnp.all(jnp.isfinite(state_fixed.eta.data))
            assert jnp.all(jnp.isfinite(state_fixed.T.data))
            assert jnp.all(jnp.isfinite(state_fixed.S.data))
            # No silent upcast — outputs must stay in fp32.
            assert state_fixed.eta.data.dtype == jnp.float32
            assert state_fixed.T.data.dtype == jnp.float32
            assert state_fixed.S.data.dtype == jnp.float32
        finally:
            set_policy(prev_policy)
            clear_module_overrides()
            for module, roles in prev_overrides.items():
                if roles:
                    set_module_override(module, **roles)


# ============================================================================
# Test: Simple Ocean
# ============================================================================

class TestSimpleOcean:
    """Test simplified ocean modes on Voronoi mesh."""

    def test_init_slab_state(self):
        """Slab state initialization."""
        state = init_mpas_slab_state(100, T_sfc_init=290.0)
        assert state.T_sfc.data.shape == (100,)
        assert jnp.allclose(state.T_sfc.data, 290.0)

    def test_fixed_mode(self):
        """Fixed SST mode returns constant."""
        config = MPASSimpleOceanConfig(mode="fixed", sst_constant=300.0)
        step = make_mpas_ocean(config)
        state = init_mpas_slab_state(10)
        _, sst, u, v = step(state, None, 3600.0)
        assert jnp.allclose(sst, 300.0)
        assert jnp.allclose(u, 0.0)

    def test_fixed_mode_with_map(self):
        """Fixed SST with prescribed spatial map."""
        sst_map = jnp.linspace(280.0, 300.0, 20)
        config = MPASSimpleOceanConfig(mode="fixed")
        step = make_mpas_ocean(config, sst_map=sst_map)
        state = init_mpas_slab_state(20)
        _, sst, _, _ = step(state, None, 3600.0)
        assert jnp.allclose(sst, sst_map)


# ============================================================================
# Test: Simple Ocean freezing-energy conservation (coupler audit F6)
# ============================================================================

def _cooling_forcing(nCells):
    """Strongly cooling MPAS surface forcing (drives trial SST below freezing)."""
    z = jnp.zeros(nCells)
    return AtmToSurface(
        sw_down=z,                              # polar night: no shortwave
        lw_down=jnp.full(nCells, 150.0),        # weak downwelling longwave
        precip_total=z,
        precip_snow=z,
        T_lowest=jnp.full(nCells, 240.0),       # cold air -> strong sensible loss
        q_lowest=z,                             # dry air -> strong latent loss
        u_lowest=jnp.full(nCells, 15.0),        # strong wind -> large exchange
        v_lowest=z,
        p_lowest=jnp.full(nCells, 95000.0),
        p_surface=jnp.full(nCells, 1.0e5),
        rho_lowest=jnp.full(nCells, 1.25),
        cos_zenith=z,
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


def _warm_forcing(nCells):
    """Heating forcing (clamp never fires; Q_freeze stays zero)."""
    z = jnp.zeros(nCells)
    return AtmToSurface(
        sw_down=jnp.full(nCells, 400.0),
        lw_down=jnp.full(nCells, 350.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(nCells, 305.0),
        q_lowest=jnp.full(nCells, 2.0e-2),
        u_lowest=jnp.full(nCells, 3.0),
        v_lowest=z,
        p_lowest=jnp.full(nCells, 95000.0),
        p_surface=jnp.full(nCells, 1.0e5),
        rho_lowest=jnp.full(nCells, 1.15),
        cos_zenith=jnp.full(nCells, 0.8),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


class TestSimpleOceanFreezeConservation:
    """The MPAS slab/two-layer freezing clamp must book the heat it removes
    as ``Q_freeze`` so the surface energy budget closes (coupler audit F6).

    Strategy: run the SAME cooling forcing twice — once with the default
    freezing point (clamp fires) and once with ``T_freeze`` set unreachably
    low (clamp disabled, identical fluxes since ``T_freeze`` never enters the
    energy balance).  The clamp injects exactly the latent heat of fusion, so
    ``C_mix * (T_clamp - T_noclamp) / dt`` must equal the diagnosed
    ``Q_freeze`` to machine precision, with NO re-derivation of the bulk-flux
    formulas in the test.
    """

    DT = 86400.0   # daily step: cooling forcing drives a ~0.5 K drop over h_mix=50 m
    NCELLS = 16

    def _C_mix(self, cfg):
        return cfg.rho_ocean * cfg.c_ocean * cfg.h_mix

    def test_slab_freeze_energy_conservation(self):
        forcing = _cooling_forcing(self.NCELLS)
        cfg = MPASSimpleOceanConfig(mode="slab")              # T_freeze = 271.35 K
        cfg_noclamp = cfg._replace(T_freeze=100.0)            # clamp never fires
        C_mix = self._C_mix(cfg)
        st = init_mpas_slab_state(self.NCELLS, T_sfc_init=271.5)  # just above freezing

        sf, _, _, _ = make_mpas_ocean(cfg)(st, forcing, self.DT)
        sn, _, _, _ = make_mpas_ocean(cfg_noclamp)(st, forcing, self.DT)

        # Clamp holds SST at/above the freezing point.
        assert bool(jnp.all(sf.T_sfc.data >= cfg.T_freeze - 1e-9))
        # The unclamped twin genuinely undershoots — the test exercises the clamp.
        assert bool(jnp.any(sn.T_sfc.data < cfg.T_freeze))
        # Energy closure: the heat the clamp injected == diagnosed Q_freeze.
        injected = C_mix * (sf.T_sfc.data - sn.T_sfc.data) / self.DT
        assert jnp.allclose(injected, sf.Q_freeze.data, atol=1e-6, rtol=1e-9)
        # Sign + units.
        assert bool(jnp.all(sf.Q_freeze.data >= 0.0))
        assert bool(jnp.any(sf.Q_freeze.data > 0.0))
        assert sf.Q_freeze.units == "W/m2"

    def test_two_layer_freeze_energy_conservation(self):
        forcing = _cooling_forcing(self.NCELLS)
        cfg = MPASSimpleOceanConfig(mode="two_layer")
        cfg_noclamp = cfg._replace(T_freeze=100.0)
        C_mix = self._C_mix(cfg)
        st = init_mpas_slab_state(self.NCELLS, T_sfc_init=271.5, T_deep_init=275.0)

        sf, _, _, _ = make_mpas_ocean(cfg)(st, forcing, self.DT)
        sn, _, _, _ = make_mpas_ocean(cfg_noclamp)(st, forcing, self.DT)

        assert bool(jnp.all(sf.T_sfc.data >= cfg.T_freeze - 1e-9))
        assert bool(jnp.any(sn.T_sfc.data < cfg.T_freeze))
        # Deep layer evolves identically (T_freeze does not touch it).
        assert jnp.allclose(sf.T_deep.data, sn.T_deep.data, atol=1e-9)
        injected = C_mix * (sf.T_sfc.data - sn.T_sfc.data) / self.DT
        assert jnp.allclose(injected, sf.Q_freeze.data, atol=1e-6, rtol=1e-9)
        assert bool(jnp.any(sf.Q_freeze.data > 0.0))

    def test_warm_case_no_freeze(self):
        for mode in ("slab", "two_layer"):
            cfg = MPASSimpleOceanConfig(mode=mode)
            st = init_mpas_slab_state(8, T_sfc_init=300.0)
            s, _, _, _ = make_mpas_ocean(cfg)(st, _warm_forcing(8), self.DT)
            assert jnp.allclose(s.Q_freeze.data, 0.0)

    def test_q_freeze_differentiable(self):
        """Total diagnosed freezing heat is differentiable wrt initial SST."""
        cfg = MPASSimpleOceanConfig(mode="slab")
        forcing = _cooling_forcing(self.NCELLS)

        def total_q_freeze(T0):
            st = MPASSlabOceanState(
                T_sfc=Field(jnp.full(self.NCELLS, T0), "T_sfc", ("nCells",), "K"),
                T_deep=Field(jnp.full(self.NCELLS, 275.0), "T_deep", ("nCells",), "K"),
                Q_freeze=Field(jnp.zeros(self.NCELLS), "Q_freeze", ("nCells",), "W/m2"),
            )
            s, _, _, _ = make_mpas_ocean(cfg)(st, forcing, self.DT)
            return jnp.sum(s.Q_freeze.data)

        g = jax.grad(total_q_freeze)(271.5)
        assert jnp.isfinite(g)


# ============================================================================
# Test: Coupler Integration
# ============================================================================

class TestCouplerIntegration:
    """Test that MPAS ocean can interface with the coupler."""

    def test_mpas_tile_config(self, mesh):
        """Create TileConfig from mesh data."""
        from legoesm.coupler.mpas_adapter import make_mpas_tile_config

        tc = make_mpas_tile_config(mesh)
        assert tc.f_land.shape == (mesh.nCells,)
        assert jnp.allclose(tc.f_land, 0.0)  # all ocean by default

    def test_mpas_tile_config_with_land(self, mesh):
        """TileConfig with land mask."""
        from legoesm.coupler.mpas_adapter import make_mpas_tile_config

        land = (jnp.abs(mesh.latCell) > jnp.radians(70)).astype(jnp.float64)
        tc = make_mpas_tile_config(mesh, land_mask=land)
        assert jnp.any(tc.f_land > 0)

    def test_init_mpas_surface_state(self, mesh):
        """Initialize surface state with MPAS shapes."""
        from legoesm.coupler.mpas_adapter import init_mpas_surface_state

        sfc = init_mpas_surface_state(mesh.nCells)
        # Check land state shape
        assert sfc.land.T_soil.data.shape == (mesh.nCells,)

    def test_reconstruct_for_coupler(self, mesh, state):
        """Velocity reconstruction provides coupler-compatible fields."""
        # Surface velocity for coupler
        u_sfc = state.u.data[:, 0]  # surface level
        u_east, v_north = reconstruct_cell_velocity(u_sfc, mesh)
        assert u_east.shape == (mesh.nCells,)
        assert v_north.shape == (mesh.nCells,)
        assert jnp.all(jnp.isfinite(u_east))


# ============================================================================
# Test: Surface Forcing Physics
# ============================================================================

class TestSurfaceForcing:
    """Test MPAS ocean physics pipeline (surface forcing, bottom drag)."""

    def test_physics_fn_creation(self):
        """make_mpas_ocean_physics returns a callable."""
        from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            PrescribedForcingConfig, SurfaceForcingConfig,
        )

        config = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(
                scheme="prescribed",
                prescribed=PrescribedForcingConfig(
                    wind_profile="single_gyre", tau_max=0.1),
            ),
        )
        fn = make_mpas_ocean_physics(config)
        assert callable(fn)

    def test_prescribed_wind_produces_tendency(self, mesh, z_coord, state):
        """Prescribed wind forcing produces nonzero edge-normal momentum tendency."""
        from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            PrescribedForcingConfig, SurfaceForcingConfig,
        )

        config = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(
                scheme="prescribed",
                prescribed=PrescribedForcingConfig(
                    wind_profile="single_gyre", tau_max=0.1),
            ),
        )
        fn = make_mpas_ocean_physics(config)
        tend = fn(state, mesh, z_coord)

        assert tend.du_dt.data.shape == state.u.data.shape
        assert tend.dT_dt.data.shape == state.T.data.shape
        assert tend.dS_dt.data.shape == state.T.data.shape
        # Wind stress should produce nonzero top-layer momentum tendency
        assert float(jnp.max(jnp.abs(tend.du_dt.data[:, 0]))) > 0
        # Below top layer should be zero (no bottom drag)
        assert float(jnp.max(jnp.abs(tend.du_dt.data[:, 1:]))) == 0.0

    def test_bottom_drag_produces_tendency(self, mesh, z_coord):
        """Dynamics-level linear bottom drag produces nonzero tendency."""
        from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean

        config = MPASOceanConfig(bottom_drag_r=1e-4)

        # Create state with nonzero bottom velocity
        s = rest_state_mpas_ocean(mesh, z_coord, H_max=500.0)
        u_data = s.u.data.at[:, -1].set(1.0)
        s = s._replace(u=s.u.replace(data=u_data))

        tend = mpas_ocean_baroclinic_tendencies(
            s, mesh, z_coord, config)
        # Bottom layer should have drag: du/dt = -r * u / dz_bottom
        assert float(jnp.max(jnp.abs(tend.du_dt.data[:, -1]))) > 0

    def test_model_step_with_physics(self, mesh, z_coord):
        """Full model step with wind + dynamics bottom drag produces circulation."""
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            PrescribedForcingConfig, SurfaceForcingConfig,
        )
        from legoesm.ocean.init_mpas import wind_driven_gyre_mpas

        physics = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(
                scheme="prescribed",
                prescribed=PrescribedForcingConfig(
                    wind_profile="single_gyre", tau_max=0.1),
            ),
        )
        config = MPASOceanConfig(
            n_barotropic_substeps=5, physics=physics, A_h=1e3,
            bottom_drag_r=1e-4)
        model = MPASOceanModel(mesh, z_coord, config)

        state = wind_driven_gyre_mpas(mesh, z_coord, H_max=500.0)
        state_new = model.step(state, 60.0)

        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))
        # Wind stress should produce motion
        assert float(jnp.max(jnp.abs(state_new.u.data))) > 0


# ============================================================================
# Test: TVD Advection on MPAS
# ============================================================================

class TestMPASTVDAdvection:
    """Test TVD tracer advection on Voronoi mesh."""

    def test_upup_cell_shapes(self, mesh):
        """compute_upup_cells returns correct shapes."""
        from legoesm.ocean.dynamics.advection_mpas import compute_upup_cells

        upup_pos, upup_neg = compute_upup_cells(mesh)
        assert upup_pos.shape == (mesh.nEdges,)
        assert upup_neg.shape == (mesh.nEdges,)

    def test_upup_cell_valid_indices(self, mesh):
        """Upup cell indices are within valid range."""
        from legoesm.ocean.dynamics.advection_mpas import compute_upup_cells

        upup_pos, upup_neg = compute_upup_cells(mesh)
        assert int(jnp.min(upup_pos)) >= 0
        assert int(jnp.max(upup_pos)) < mesh.nCells
        assert int(jnp.min(upup_neg)) >= 0
        assert int(jnp.max(upup_neg)) < mesh.nCells

    def test_upup_cell_differs_from_neighbor(self, mesh):
        """Upup cell is generally different from the direct neighbor.

        For most interior edges, the opposite-cell lookup should yield
        a cell that is distinct from both c1 and c2.
        """
        from legoesm.ocean.dynamics.advection_mpas import compute_upup_cells

        upup_pos, upup_neg = compute_upup_cells(mesh)
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]

        # At least 50% of edges should have upup != c2 for pos flow
        # (the opposite cell of c1 should not be c2 itself)
        frac_distinct_pos = float(jnp.mean((upup_pos != c2).astype(jnp.float32)))
        assert frac_distinct_pos > 0.5, (
            f"Only {frac_distinct_pos:.0%} of upup_pos differ from c2 — "
            f"opposite-cell lookup may be broken"
        )

    def test_tvd_to_edges_shapes(self, mesh):
        """tvd_tracer_to_edges returns correct shapes."""
        from legoesm.ocean.dynamics.advection_mpas import (
            compute_upup_cells, tvd_tracer_to_edges,
        )

        nlev = 5
        upup_pos, upup_neg = compute_upup_cells(mesh)
        tr = jnp.ones((mesh.nCells, nlev)) * 15.0
        mass_flux = jnp.ones((mesh.nEdges, nlev)) * 0.01

        tr_edge = tvd_tracer_to_edges(tr, mass_flux, mesh, upup_pos, upup_neg)
        assert tr_edge.shape == (mesh.nEdges, nlev)

    def test_tvd_recovers_constant_field(self, mesh):
        """TVD reconstruction of a uniform field is exact."""
        from legoesm.ocean.dynamics.advection_mpas import (
            compute_upup_cells, tvd_tracer_to_edges,
        )

        nlev = 3
        upup_pos, upup_neg = compute_upup_cells(mesh)
        tr = jnp.ones((mesh.nCells, nlev)) * 20.0
        mass_flux = jnp.sin(mesh.angleEdge)[:, jnp.newaxis] * jnp.ones(nlev)

        tr_edge = tvd_tracer_to_edges(tr, mass_flux, mesh, upup_pos, upup_neg)
        assert jnp.allclose(tr_edge, 20.0, atol=1e-12)

    def test_tvd_model_step(self, mesh, z_coord, state):
        """MPAS model with TVD advection takes a step successfully."""
        config_tvd = MPASOceanConfig(
            A_h=1.0e3, K_h=1.0e2, A_v=1.0e-3, K_v=1.0e-4,
            n_barotropic_substeps=5,
            tracer_advection="tvd",
        )
        model = MPASOceanModel(mesh, z_coord, config_tvd)
        state_new = model.step(state, 60.0)

        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.S.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))

    def test_tvd_multi_step_stability(self, mesh, z_coord, state):
        """TVD model is stable for multiple steps."""
        config_tvd = MPASOceanConfig(
            A_h=1.0e3, K_h=1.0e2, A_v=1.0e-3, K_v=1.0e-4,
            n_barotropic_substeps=5,
            tracer_advection="tvd",
        )
        model = MPASOceanModel(mesh, z_coord, config_tvd)
        s = state
        for _ in range(5):
            s = model.step(s, 30.0)
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.S.data))

    def test_tvd_less_diffusive_than_upwind(self, mesh, z_coord):
        """TVD produces less numerical diffusion than upwind.

        Creates a state with a sharp temperature gradient at the equator,
        applies a uniform flow, and verifies that TVD preserves the
        gradient better than upwind after several steps.
        """
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean

        state = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=500.0, land_lat_threshold=85.0,
        )
        # Sharp T front at equator — broadcast to (nCells, nlev)
        nlev = state.T.data.shape[1]
        north = (mesh.latCell > 0).astype(state.T.data.dtype)
        T_front = (north * 20.0 + (1 - north) * 10.0)[:, jnp.newaxis]
        T_front = jnp.broadcast_to(T_front, (mesh.nCells, nlev))
        mask = state.land_mask.data
        T_front = T_front * mask[:, jnp.newaxis]
        state_front = state._replace(T=state.T.replace(data=T_front))

        var_T_initial = float(jnp.var(T_front[mask > 0.5]))

        dt = 30.0
        n_steps = 5

        results = {}
        for scheme in ["upwind", "tvd"]:
            cfg = MPASOceanConfig(
                A_h=1.0e3, K_h=0.0, A_v=1.0e-3, K_v=0.0,
                n_barotropic_substeps=5,
                tracer_advection=scheme,
            )
            model = MPASOceanModel(mesh, z_coord, cfg)
            s = state_front
            for _ in range(n_steps):
                s = model.step(s, dt)
            var_T_final = float(jnp.var(s.T.data[mask > 0.5]))
            results[scheme] = var_T_final

        # TVD should preserve more variance (less diffusive)
        assert results["tvd"] >= results["upwind"] * 0.99, (
            f"TVD Var(T)={results['tvd']:.6f} should be >= "
            f"upwind Var(T)={results['upwind']:.6f}"
        )


# ============================================================================
# Test: freeze-floor (sea-ice surrogate)
# ============================================================================

class TestFreezeFloor:
    """``config.freeze_floor`` floors the SURFACE SST at the seawater freezing
    point (sea-ice thermodynamic surrogate, surface-only, applied after the
    step) — the MPAS port of LatLonCGridOceanModel._apply_freeze_floor that
    closes the no-ice Arctic over-cool vs NEMO."""

    def _cold_surface(self, state):
        # Set the surface (k=0) to -5 C everywhere (below the -1.8 C floor),
        # leaving the subsurface profile intact.
        T = state.T.data
        return state._replace(T=state.T.replace(data=T.at[..., 0].set(-5.0)))

    def test_freeze_floor_clamps_surface(self, mesh, z_coord, config, state):
        cfg = config._replace(freeze_floor=True)
        model = MPASOceanModel(mesh, z_coord, cfg)
        out = model.step(self._cold_surface(state), 60.0)
        sfc = np.asarray(out.T.data)[..., 0]
        floor = cfg.freeze_floor_temp_c
        assert np.all(sfc >= floor - 1e-9), (
            f"surface T below the freeze floor: min={sfc.min():.4f} < {floor:.4f}")
        assert np.all(np.isfinite(sfc))

    def test_freeze_floor_off_does_not_clamp(self, mesh, z_coord, config, state):
        """Default (freeze_floor=False): the cold -5 C surface is NOT clamped —
        one 60 s step barely warms it — so the min stays well below the floor."""
        model = MPASOceanModel(mesh, z_coord, config)  # default freeze_floor=False
        out = model.step(self._cold_surface(state), 60.0)
        sfc = np.asarray(out.T.data)[..., 0]
        assert sfc.min() < config.freeze_floor_temp_c, (
            "freeze_floor=False must NOT floor the surface (gate not bit-exact off)")

    def test_freeze_floor_surface_only(self, mesh, z_coord, config, state):
        """The floor touches ONLY the surface level; subsurface levels evolve
        identically with the floor on vs off (the clamp is k=0 only)."""
        st = self._cold_surface(state)
        m_on = MPASOceanModel(mesh, z_coord, config._replace(freeze_floor=True))
        m_off = MPASOceanModel(mesh, z_coord, config)
        T_on = np.asarray(m_on.step(st, 60.0).T.data)
        T_off = np.asarray(m_off.step(st, 60.0).T.data)
        # Subsurface (k>=1) identical; only k=0 differs.
        np.testing.assert_allclose(T_on[..., 1:], T_off[..., 1:], rtol=0, atol=0)
        assert not np.allclose(T_on[..., 0], T_off[..., 0])


class TestMPASMultiRankFreshwaterGuard:
    """Codex round-2/round-3: a multi-rank MPAS run with normalize_freshwater=True
    must FAIL-FAST (the eta + top-layer-salt freshwater means are rank-local with
    no owned-cell mask -> silent halo-double-count + inconsistent volume vs salt
    correction on MPI Voronoi).  The guard lives at the SINGLE reduction source
    (mpas_ocean_baroclinic_tendencies), so it also covers a DIRECT tendency call.
    Inert single-rank (asserted across the rest of this file)."""

    def _fw(self, mesh):
        from legoesm.ocean.freshwater import FreshwaterForcing
        import jax.numpy as jnp
        n = mesh.areaCell.shape[0]
        # Nonzero, non-zero-mean P-E so normalization would actually act.
        z = jnp.zeros(n)
        return FreshwaterForcing(
            precip=jnp.full(n, 1e-6), evap=z, runoff=z, ice_fw=z, restoring=z)

    def test_multirank_normalize_freshwater_raises(self, mesh, z_coord, state,
                                                   monkeypatch):
        cfg = MPASOceanConfig(
            normalize_freshwater=True, freshwater_closure="virtual_salt_flux")
        import legoesm.parallel.reductions as R
        # Simulate a multi-rank launch (the layout-less Voronoi MPI path keeps
        # is_multi_process False, so the world-size check is what trips).
        monkeypatch.setattr(R, "mpi_world_size", lambda: 2)
        with pytest.raises(NotImplementedError, match="normalize_freshwater"):
            mpas_ocean_baroclinic_tendencies(
                state, mesh, z_coord, cfg, freshwater=self._fw(mesh))

    def test_singlerank_normalize_freshwater_ok(self, mesh, z_coord, state):
        """The same config single-rank must NOT raise (guard inert)."""
        cfg = MPASOceanConfig(
            normalize_freshwater=True, freshwater_closure="virtual_salt_flux")
        tend = mpas_ocean_baroclinic_tendencies(
            state, mesh, z_coord, cfg, freshwater=self._fw(mesh))
        assert tend is not None
