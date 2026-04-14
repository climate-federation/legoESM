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

# Ensure float64
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
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
    fix_volume_mpas,
    fix_heat_mpas,
    fix_salt_mpas,
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
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
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


# ============================================================================
# Test: Conservation
# ============================================================================

class TestConservation:
    """Test conservation fixers."""

    def test_volume_conservation(self, state, mesh, z_coord):
        """Volume fixer restores total volume."""
        mask = state.land_mask.data
        area = mesh.areaCell

        # Perturb eta
        eta_perturbed = state.eta.data + 0.1 * jnp.sin(mesh.latCell) * mask
        state_new = state._replace(
            eta=state.eta.replace(data=eta_perturbed),
        )

        # Verify in float64 — the fixer accumulates in float64 but the
        # corrected state may be float32 under the default precision policy.
        _f64 = jnp.float64
        vol_before = jnp.sum(state.eta.data.astype(_f64) * mask.astype(_f64) * area.astype(_f64))
        state_fixed = fix_volume_mpas(state_new, state, mesh, z_coord)
        vol_after = jnp.sum(state_fixed.eta.data.astype(_f64) * mask.astype(_f64) * area.astype(_f64))

        # Use absolute tolerance when reference volume is near zero.
        # With float32 state, the correction's float32 representation
        # introduces O(nCells * eps_f32 * |correction|) residual.
        total_area = jnp.sum(mask.astype(_f64) * area.astype(_f64))
        assert jnp.abs(vol_after - vol_before) < 1e-5 * total_area

    def test_heat_conservation(self, state, mesh, z_coord):
        """Heat fixer restores total heat content."""
        mask = state.land_mask.data
        H_bathy = state.H_bathy.data

        # Perturb temperature
        T_perturbed = state.T.data + 0.5 * jnp.sin(
            mesh.latCell[:, jnp.newaxis]
        ) * mask[:, jnp.newaxis]
        state_new = state._replace(
            T=state.T.replace(data=T_perturbed),
        )

        _f64 = jnp.float64
        h_k = compute_layer_thickness(state.eta.data, H_bathy, z_coord)
        heat_before = jnp.sum(
            state.T.data.astype(_f64) * h_k.astype(_f64)
            * mask.astype(_f64)[:, jnp.newaxis] * mesh.areaCell.astype(_f64)[:, jnp.newaxis]
        )

        state_fixed = fix_heat_mpas(state_new, state, mesh, z_coord)
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
        """Linear bottom drag produces nonzero bottom-layer tendency."""
        from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.bottom_drag.config import (
            BottomDragConfig, LinearDragConfig,
        )
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean

        config = OceanPhysicsConfig(
            bottom_drag=BottomDragConfig(
                scheme="linear", linear=LinearDragConfig(r=1e-4)),
        )
        fn = make_mpas_ocean_physics(config)

        # Create state with nonzero bottom velocity
        s = rest_state_mpas_ocean(mesh, z_coord, H_max=500.0)
        u_data = s.u.data.at[:, -1].set(1.0)
        s = s._replace(u=s.u.replace(data=u_data))

        tend = fn(s, mesh, z_coord)
        # Bottom layer should have drag: du/dt = -r * u = -1e-4
        assert float(jnp.max(jnp.abs(tend.du_dt.data[:, -1]))) > 0

    def test_model_step_with_physics(self, mesh, z_coord):
        """Full model step with physics produces circulation."""
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            PrescribedForcingConfig, SurfaceForcingConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import (
            BottomDragConfig, LinearDragConfig,
        )
        from legoesm.ocean.init_mpas import wind_driven_gyre_mpas

        physics = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(
                scheme="prescribed",
                prescribed=PrescribedForcingConfig(
                    wind_profile="single_gyre", tau_max=0.1),
            ),
            bottom_drag=BottomDragConfig(
                scheme="linear", linear=LinearDragConfig(r=1e-4)),
        )
        config = MPASOceanConfig(
            n_barotropic_substeps=5, physics=physics, A_h=1e3)
        model = MPASOceanModel(mesh, z_coord, config)

        state = wind_driven_gyre_mpas(mesh, z_coord, H_max=500.0)
        state_new = model.step(state, 60.0)

        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))
        # Wind stress should produce motion
        assert float(jnp.max(jnp.abs(state_new.u.data))) > 0
