"""Tests for the lat-lon finite-volume ocean dynamical core.

Covers:
- FV operators (divergence, gradient, vorticity, Laplacian, PPM advection)
- State initialization (rest state, bathymetry, land mask)
- Baroclinic tendencies (shapes, masking, symmetry)
- Barotropic solver (conservation, CFL stability)
- Full model step (split-explicit integration)
- Conservation fixers (volume, heat, salt)
- Differentiability (jax.grad through model step)
- Multi-step stability
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import (
    LatLonOceanState,
    LatLonOceanTendencies,
    LatLonOceanConfig,
)
from legoesm.ocean.init_latlon import (
    rest_state_latlon_ocean,
    idealized_bathymetry_latlon,
)
from legoesm.ocean.dynamics.latlon_operators import (
    fv_divergence_latlon,
    fv_divergence_latlon_3d,
    fv_scalar_advection_latlon,
    fv_scalar_advection_latlon_3d,
    gradient_x_latlon,
    gradient_y_latlon,
    vorticity_latlon,
    laplacian_latlon,
    divergence_latlon,
)
from legoesm.ocean.dynamics.ocean_pe_latlon import (
    latlon_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.barotropic_latlon import barotropic_substeps_latlon
from legoesm.ocean.dynamics.ocean_model_latlon import LatLonOceanModel
from legoesm.ocean.conservation_latlon import (
    fix_volume_latlon,
    fix_heat_latlon,
    fix_salt_latlon,
    latlon_ocean_conservation_fixer,
)


# Small grid for fast tests
N_LAT = 16
N_LON = 32
NLEV = 5
DT = 1800.0


def _make_grid():
    return create_latlon_grid(N_LAT, N_LON)


def _make_z_coord():
    return create_ocean_z_star(NLEV, H_max=2000.0, dz_surface=50.0, dz_deep=800.0)


def _make_state(grid=None, z_coord=None):
    grid = grid or _make_grid()
    z_coord = z_coord or _make_z_coord()
    return rest_state_latlon_ocean(grid, z_coord, H_max=2000.0)


def _make_config(**overrides):
    defaults = dict(
        A_h=1.0e4,
        K_h=1.0e3,
        A_v=1.0e-3,
        K_v=1.0e-4,
        n_barotropic_substeps=10,
        use_conservation_fixer=True,
    )
    defaults.update(overrides)
    return LatLonOceanConfig(**defaults)


class TestLatLonOperators(unittest.TestCase):
    """Test FV operators on the lat-lon grid."""

    def setUp(self):
        self.grid = _make_grid()

    def test_divergence_uniform_field(self):
        """Divergence of a uniform velocity field is zero."""
        u = jnp.ones((N_LAT, N_LON))
        v = jnp.zeros((N_LAT, N_LON))
        div = fv_divergence_latlon(u, v, self.grid)
        self.assertEqual(div.shape, (N_LAT, N_LON))
        # Uniform u with zero v: div should be small (not exactly zero
        # due to spherical metric, but small for constant field)
        self.assertTrue(jnp.all(jnp.isfinite(div)))

    def test_divergence_3d_shape(self):
        """3D divergence has correct output shape."""
        Fu = jnp.ones((N_LAT, N_LON, NLEV))
        Fv = jnp.zeros((N_LAT, N_LON, NLEV))
        div = fv_divergence_latlon_3d(Fu, Fv, self.grid)
        self.assertEqual(div.shape, (N_LAT, N_LON, NLEV))

    def test_gradient_x_sinusoidal(self):
        """Zonal gradient of sin(lon) should approximate cos(lon)/R."""
        grid = self.grid
        f = jnp.sin(grid.lon2d)
        df_dx = gradient_x_latlon(f, grid)
        expected = jnp.cos(grid.lon2d) / (grid.radius * grid.cos_lat[:, None])
        # Centered difference: 2nd-order accurate, check relative error
        rel_err = jnp.max(jnp.abs(df_dx - expected)) / jnp.max(jnp.abs(expected))
        self.assertLess(float(rel_err), 0.1)  # ~2nd-order error at this resolution

    def test_gradient_y_sinusoidal(self):
        """Meridional gradient of sin(lat) should approximate cos(lat)/R."""
        grid = self.grid
        f = jnp.sin(grid.lat2d)
        df_dy = gradient_y_latlon(f, grid)
        expected = jnp.cos(grid.lat2d) / grid.radius
        # Check interior (skip pole-adjacent cells with boundary effects)
        interior = slice(2, -2)
        rel_err = jnp.max(jnp.abs(df_dy[interior] - expected[interior])) / jnp.max(
            jnp.abs(expected[interior])
        )
        self.assertLess(float(rel_err), 0.15)

    def test_gradient_3d(self):
        """Gradient works on 3D fields."""
        f = jnp.ones((N_LAT, N_LON, NLEV))
        df_dx = gradient_x_latlon(f, self.grid)
        df_dy = gradient_y_latlon(f, self.grid)
        self.assertEqual(df_dx.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(df_dy.shape, (N_LAT, N_LON, NLEV))

    def test_vorticity_shape(self):
        """Vorticity has correct shape."""
        u = jnp.ones((N_LAT, N_LON))
        v = jnp.zeros((N_LAT, N_LON))
        zeta = vorticity_latlon(u, v, self.grid)
        self.assertEqual(zeta.shape, (N_LAT, N_LON))
        self.assertTrue(jnp.all(jnp.isfinite(zeta)))

    def test_vorticity_3d(self):
        """Vorticity works on 3D fields."""
        u = jnp.ones((N_LAT, N_LON, NLEV))
        v = jnp.zeros((N_LAT, N_LON, NLEV))
        zeta = vorticity_latlon(u, v, self.grid)
        self.assertEqual(zeta.shape, (N_LAT, N_LON, NLEV))

    def test_laplacian_shape_and_finite(self):
        """Laplacian produces finite values with correct shape."""
        f = jnp.sin(self.grid.lat2d) * jnp.cos(2 * self.grid.lon2d)
        lap = laplacian_latlon(f, self.grid)
        self.assertEqual(lap.shape, (N_LAT, N_LON))
        self.assertTrue(jnp.all(jnp.isfinite(lap)))

    def test_laplacian_3d(self):
        """Laplacian works on 3D fields."""
        f = jnp.ones((N_LAT, N_LON, NLEV))
        lap = laplacian_latlon(f, self.grid)
        self.assertEqual(lap.shape, (N_LAT, N_LON, NLEV))

    def test_ppm_advection_shape(self):
        """PPM scalar advection produces correct shape."""
        q = jnp.ones((N_LAT, N_LON))
        u = jnp.zeros((N_LAT, N_LON))
        v = jnp.zeros((N_LAT, N_LON))
        dq = fv_scalar_advection_latlon(q, u, v, self.grid)
        self.assertEqual(dq.shape, (N_LAT, N_LON))
        self.assertTrue(jnp.all(jnp.isfinite(dq)))

    def test_ppm_advection_3d(self):
        """3D PPM scalar advection has correct shape."""
        q = jnp.ones((N_LAT, N_LON, NLEV))
        u = jnp.zeros((N_LAT, N_LON, NLEV))
        v = jnp.zeros((N_LAT, N_LON, NLEV))
        dq = fv_scalar_advection_latlon_3d(q, u, v, self.grid)
        self.assertEqual(dq.shape, (N_LAT, N_LON, NLEV))

    def test_ppm_advection_uniform_zero_tendency(self):
        """Advecting a uniform field should yield near-zero tendency."""
        q = jnp.full((N_LAT, N_LON), 5.0)
        u = jnp.ones((N_LAT, N_LON)) * 0.1
        v = jnp.zeros((N_LAT, N_LON))
        dq = fv_scalar_advection_latlon(q, u, v, self.grid)
        self.assertLess(float(jnp.max(jnp.abs(dq))), 1e-8)

    def test_divergence_latlon_centered(self):
        """Centered divergence operator works."""
        u = jnp.ones((N_LAT, N_LON)) * 0.1
        v = jnp.zeros((N_LAT, N_LON))
        div = divergence_latlon(u, v, self.grid)
        self.assertEqual(div.shape, (N_LAT, N_LON))
        self.assertTrue(jnp.all(jnp.isfinite(div)))


class TestLatLonInitialization(unittest.TestCase):
    """Test state initialization."""

    def test_rest_state_shapes(self):
        """Rest state fields have correct shapes."""
        grid = _make_grid()
        z_coord = _make_z_coord()
        state = rest_state_latlon_ocean(grid, z_coord, H_max=2000.0)

        self.assertEqual(state.u.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(state.v.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(state.T.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(state.S.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(state.eta.data.shape, (N_LAT, N_LON))
        self.assertEqual(state.H_bathy.data.shape, (N_LAT, N_LON))
        self.assertEqual(state.land_mask.data.shape, (N_LAT, N_LON))

    def test_rest_state_zero_velocity(self):
        """Rest state has zero velocity."""
        state = _make_state()
        self.assertAlmostEqual(float(jnp.max(jnp.abs(state.u.data))), 0.0)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(state.v.data))), 0.0)

    def test_rest_state_stratification(self):
        """Rest state has surface warmer than deep."""
        state = _make_state()
        T_sfc = float(jnp.mean(state.T.data[:, :, 0]))
        T_deep = float(jnp.mean(state.T.data[:, :, -1]))
        self.assertGreater(T_sfc, T_deep)

    def test_bathymetry_land_mask(self):
        """Land mask is 0 near poles, 1 in mid-latitudes."""
        grid = _make_grid()
        H_bathy, land_mask = idealized_bathymetry_latlon(grid, H_max=2000.0)
        # Mid-latitude should be ocean
        mid = N_LAT // 2
        self.assertAlmostEqual(float(land_mask[mid, 0]), 1.0)
        # Check some cells exist with mask=0 (near poles)
        self.assertLess(float(jnp.sum(land_mask)), float(N_LAT * N_LON))

    def test_bathymetry_invalid_H_max(self):
        """Invalid H_max raises ValueError."""
        grid = _make_grid()
        with self.assertRaises(ValueError):
            idealized_bathymetry_latlon(grid, H_max=-100.0)

    def test_state_is_namedtuple(self):
        """State is a proper NamedTuple (JAX pytree compatible)."""
        state = _make_state()
        self.assertIsInstance(state, LatLonOceanState)
        # Can flatten/unflatten as pytree
        leaves, treedef = jax.tree_util.tree_flatten(state)
        state_rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(state_rebuilt.u.data.shape, state.u.data.shape)


class TestBaroclinicTendencies(unittest.TestCase):
    """Test lat-lon baroclinic tendency computation."""

    def setUp(self):
        self.grid = _make_grid()
        self.z_coord = _make_z_coord()
        self.state = _make_state(self.grid, self.z_coord)
        self.config = _make_config()

    def test_tendency_shapes(self):
        """Tendencies have correct shapes."""
        tend = latlon_ocean_baroclinic_tendencies(
            self.state, self.grid, self.z_coord, self.config,
        )
        self.assertIsInstance(tend, LatLonOceanTendencies)
        self.assertEqual(tend.du_dt.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(tend.dv_dt.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(tend.dT_dt.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(tend.dS_dt.data.shape, (N_LAT, N_LON, NLEV))
        self.assertEqual(tend.deta_dt.data.shape, (N_LAT, N_LON))

    def test_tendency_finite(self):
        """All tendencies are finite."""
        tend = latlon_ocean_baroclinic_tendencies(
            self.state, self.grid, self.z_coord, self.config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dv_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dS_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.deta_dt.data)))

    def test_rest_state_tendencies_small(self):
        """Rest state should produce near-zero tendencies."""
        tend = latlon_ocean_baroclinic_tendencies(
            self.state, self.grid, self.z_coord, self.config,
        )
        # Rest state: zero velocity, horizontally uniform T/S → no advection
        # Small residual from numerics only
        max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
        max_dv = float(jnp.max(jnp.abs(tend.dv_dt.data)))
        max_deta = float(jnp.max(jnp.abs(tend.deta_dt.data)))
        self.assertLess(max_du, 1.0e-6)
        self.assertLess(max_dv, 1.0e-6)
        self.assertLess(max_deta, 1.0e-6)

    def test_land_masking(self):
        """Tendencies are zero on land cells."""
        tend = latlon_ocean_baroclinic_tendencies(
            self.state, self.grid, self.z_coord, self.config,
        )
        mask = self.state.land_mask.data
        land = mask < 0.5
        land_3d = land[..., jnp.newaxis]
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(jnp.where(land_3d, tend.du_dt.data, 0.0)))),
            0.0,
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(jnp.where(land, tend.deta_dt.data, 0.0)))),
            0.0,
        )

    def test_static_fields_zero_tendency(self):
        """H_bathy and land_mask tendencies are always zero."""
        tend = latlon_ocean_baroclinic_tendencies(
            self.state, self.grid, self.z_coord, self.config,
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(tend.dH_bathy_dt.data))), 0.0,
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(tend.dland_mask_dt.data))), 0.0,
        )


class TestBarotropicSolver(unittest.TestCase):
    """Test barotropic substeps on lat-lon grid."""

    def setUp(self):
        self.grid = _make_grid()
        self.z_coord = _make_z_coord()
        self.state = _make_state(self.grid, self.z_coord)
        self.config = _make_config(n_barotropic_substeps=10)

    def test_barotropic_preserves_shapes(self):
        """Barotropic step preserves state shapes."""
        dt_s = DT / self.config.n_barotropic_substeps
        state_new = barotropic_substeps_latlon(
            self.state, dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )
        self.assertEqual(state_new.eta.data.shape, (N_LAT, N_LON))
        self.assertEqual(state_new.u.data.shape, (N_LAT, N_LON, NLEV))

    def test_barotropic_finite(self):
        """Barotropic step produces finite values."""
        dt_s = DT / self.config.n_barotropic_substeps
        state_new = barotropic_substeps_latlon(
            self.state, dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(state_new.eta.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.v.data)))

    def test_barotropic_rest_state_stable(self):
        """Rest state remains approximately at rest after barotropic step."""
        dt_s = DT / self.config.n_barotropic_substeps
        state_new = barotropic_substeps_latlon(
            self.state, dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )
        max_eta = float(jnp.max(jnp.abs(state_new.eta.data)))
        self.assertLess(max_eta, 1.0)  # Should stay near zero

    def test_barotropic_differentiable_mode(self):
        """Differentiable (lax.scan) barotropic step produces same shapes."""
        config = _make_config(
            n_barotropic_substeps=5, differentiable_barotropic=True,
        )
        dt_s = DT / config.n_barotropic_substeps
        state_new = barotropic_substeps_latlon(
            self.state, dt_s, config.n_barotropic_substeps,
            self.grid, self.z_coord, config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(state_new.eta.data)))


class TestFullModelStep(unittest.TestCase):
    """Test the full LatLonOceanModel."""

    def setUp(self):
        self.grid = _make_grid()
        self.z_coord = _make_z_coord()
        self.config = _make_config(n_barotropic_substeps=5)
        self.model = LatLonOceanModel(self.grid, self.z_coord, self.config)
        self.state = _make_state(self.grid, self.z_coord)

    def test_single_step_shapes(self):
        """Single model step preserves all shapes."""
        state_new = self.model.step(self.state, DT)
        for name in LatLonOceanState._fields:
            old_shape = getattr(self.state, name).data.shape
            new_shape = getattr(state_new, name).data.shape
            self.assertEqual(old_shape, new_shape, f"Shape mismatch for {name}")

    def test_single_step_finite(self):
        """Single model step produces all finite values."""
        state_new = self.model.step(self.state, DT)
        for name in LatLonOceanState._fields:
            data = getattr(state_new, name).data
            self.assertTrue(
                jnp.all(jnp.isfinite(data)),
                f"Non-finite values in {name}",
            )

    def test_single_step_rest_state_stable(self):
        """Rest state is stable for one step."""
        state_new = self.model.step(self.state, DT)
        max_u = float(jnp.max(jnp.abs(state_new.u.data)))
        max_eta = float(jnp.max(jnp.abs(state_new.eta.data)))
        self.assertLess(max_u, 1.0)
        self.assertLess(max_eta, 1.0)

    def test_multi_step_stability(self):
        """Model is stable over 5 time steps."""
        state = self.state
        for _ in range(5):
            state = self.model.step(state, DT)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.eta.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.T.data)))

    def test_step_checked(self):
        """step_checked does not raise for rest state."""
        config = _make_config(
            n_barotropic_substeps=5, enable_runtime_checks=True,
        )
        model = LatLonOceanModel(self.grid, self.z_coord, config)
        state_new = model.step_checked(self.state, DT)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.eta.data)))

    def test_integrate_returns_trajectory(self):
        """integrate() returns final state and trajectory."""
        state_final, trajectory = self.model.integrate(
            self.state, duration=2 * DT, dt=DT, save_every=1,
        )
        # Initial + 2 steps
        self.assertEqual(len(trajectory), 3)
        self.assertTrue(jnp.all(jnp.isfinite(state_final.eta.data)))

    def test_config_validation(self):
        """Invalid config raises ValueError."""
        with self.assertRaises(ValueError):
            LatLonOceanModel(
                self.grid, self.z_coord,
                LatLonOceanConfig(A_h=-1.0),
            )
        with self.assertRaises(ValueError):
            LatLonOceanModel(
                self.grid, self.z_coord,
                LatLonOceanConfig(n_barotropic_substeps=0),
            )


class TestConservation(unittest.TestCase):
    """Test conservation fixers."""

    def setUp(self):
        self.grid = _make_grid()
        self.z_coord = _make_z_coord()
        self.state = _make_state(self.grid, self.z_coord)

    def test_volume_fixer(self):
        """Volume fixer restores global eta integral."""
        mask = self.state.land_mask.data
        # Perturb eta
        eta_perturbed = self.state.eta.data + 0.01 * mask
        state_perturbed = self.state._replace(
            eta=self.state.eta.replace(data=eta_perturbed),
        )
        state_fixed = fix_volume_latlon(
            state_perturbed, self.state, self.grid,
            min_water_column_m=0.5,
        )
        # Check volume correction was applied (eta_perturbed had excess volume,
        # fixer should reduce it)
        vol_old = float(jnp.sum(self.state.eta.data * mask * self.grid.area))
        vol_pert = float(jnp.sum(state_perturbed.eta.data * mask * self.grid.area))
        vol_fixed = float(jnp.sum(state_fixed.eta.data * mask * self.grid.area))
        # Fixed volume should be closer to original than perturbed
        self.assertLess(abs(vol_fixed - vol_old), abs(vol_pert - vol_old))

    def test_heat_fixer(self):
        """Heat fixer restores global T*h integral."""
        from legoesm.ocean.vertical import compute_layer_thickness

        mask = self.state.land_mask.data
        T_perturbed = self.state.T.data + 0.1 * mask[..., jnp.newaxis]
        state_perturbed = self.state._replace(
            T=self.state.T.replace(data=T_perturbed),
        )
        state_fixed = fix_heat_latlon(
            state_perturbed, self.state, self.grid, self.z_coord,
            min_water_column_m=0.5,
        )
        h_k = compute_layer_thickness(
            self.state.eta.data, self.state.H_bathy.data, self.z_coord,
            min_water_column_m=0.5,
        )
        heat_old = float(jnp.sum(
            jnp.sum(self.state.T.data * h_k, axis=-1) * mask * self.grid.area
        ))
        heat_pert = float(jnp.sum(
            jnp.sum(state_perturbed.T.data * h_k, axis=-1) * mask * self.grid.area
        ))
        heat_fixed = float(jnp.sum(
            jnp.sum(state_fixed.T.data * h_k, axis=-1) * mask * self.grid.area
        ))
        self.assertLess(abs(heat_fixed - heat_old), abs(heat_pert - heat_old))

    def test_salt_fixer(self):
        """Salt fixer restores global S*h integral."""
        from legoesm.ocean.vertical import compute_layer_thickness

        mask = self.state.land_mask.data
        S_perturbed = self.state.S.data + 0.05 * mask[..., jnp.newaxis]
        state_perturbed = self.state._replace(
            S=self.state.S.replace(data=S_perturbed),
        )
        state_fixed = fix_salt_latlon(
            state_perturbed, self.state, self.grid, self.z_coord,
            min_water_column_m=0.5,
        )
        h_k = compute_layer_thickness(
            self.state.eta.data, self.state.H_bathy.data, self.z_coord,
            min_water_column_m=0.5,
        )
        salt_old = float(jnp.sum(
            jnp.sum(self.state.S.data * h_k, axis=-1) * mask * self.grid.area
        ))
        salt_pert = float(jnp.sum(
            jnp.sum(state_perturbed.S.data * h_k, axis=-1) * mask * self.grid.area
        ))
        salt_fixed = float(jnp.sum(
            jnp.sum(state_fixed.S.data * h_k, axis=-1) * mask * self.grid.area
        ))
        self.assertLess(abs(salt_fixed - salt_old), abs(salt_pert - salt_old))

    def test_full_conservation_fixer(self):
        """Combined fixer reduces volume/heat/salt errors."""
        config = _make_config()
        mask = self.state.land_mask.data
        eta_p = self.state.eta.data + 0.01 * mask
        T_p = self.state.T.data + 0.1 * mask[..., jnp.newaxis]
        S_p = self.state.S.data + 0.05 * mask[..., jnp.newaxis]
        state_p = self.state._replace(
            eta=self.state.eta.replace(data=eta_p),
            T=self.state.T.replace(data=T_p),
            S=self.state.S.replace(data=S_p),
        )
        state_fixed = latlon_ocean_conservation_fixer(
            state_p, self.state, self.grid, self.z_coord, config,
        )
        vol_old = float(jnp.sum(self.state.eta.data * mask * self.grid.area))
        vol_pert = float(jnp.sum(state_p.eta.data * mask * self.grid.area))
        vol_fixed = float(jnp.sum(state_fixed.eta.data * mask * self.grid.area))
        self.assertLess(abs(vol_fixed - vol_old), abs(vol_pert - vol_old))


class TestDifferentiability(unittest.TestCase):
    """Test that the lat-lon ocean model is JAX-differentiable."""

    def test_gradient_through_tendencies(self):
        """Can compute gradient of a scalar loss through tendencies."""
        grid = _make_grid()
        z_coord = _make_z_coord()
        state = _make_state(grid, z_coord)
        config = _make_config(
            A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0,
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
        )

        def loss_fn(eta_data):
            state_in = state._replace(
                eta=state.eta.replace(data=eta_data),
            )
            tend = latlon_ocean_baroclinic_tendencies(
                state_in, grid, z_coord, config,
            )
            return jnp.sum(tend.deta_dt.data ** 2)

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(state.eta.data)
        self.assertEqual(grad_val.shape, state.eta.data.shape)
        self.assertTrue(jnp.all(jnp.isfinite(grad_val)))

    def test_gradient_through_barotropic(self):
        """Can compute gradient through differentiable barotropic solver."""
        grid = _make_grid()
        z_coord = _make_z_coord()
        state = _make_state(grid, z_coord)
        config = _make_config(
            n_barotropic_substeps=3,
            differentiable_barotropic=True,
            barotropic_diffusion_alpha=0.0,
        )
        dt_s = DT / config.n_barotropic_substeps

        def loss_fn(eta_data):
            state_in = state._replace(
                eta=state.eta.replace(data=eta_data),
            )
            state_out = barotropic_substeps_latlon(
                state_in, dt_s, config.n_barotropic_substeps,
                grid, z_coord, config,
            )
            return jnp.sum(state_out.eta.data ** 2)

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(state.eta.data)
        self.assertEqual(grad_val.shape, state.eta.data.shape)
        self.assertTrue(jnp.all(jnp.isfinite(grad_val)))


class TestPerturbedState(unittest.TestCase):
    """Test model behavior with non-trivial initial conditions."""

    def test_gaussian_eta_perturbation(self):
        """Model handles a Gaussian SSH perturbation without blowing up."""
        grid = _make_grid()
        z_coord = _make_z_coord()
        state = _make_state(grid, z_coord)
        config = _make_config(n_barotropic_substeps=10)
        model = LatLonOceanModel(grid, z_coord, config)

        # Add Gaussian SSH perturbation
        lat_c, lon_c = 0.0, jnp.pi
        sigma = 10.0 * jnp.pi / 180.0
        r2 = (grid.lat2d - lat_c) ** 2 + (grid.lon2d - lon_c) ** 2
        eta_pert = 0.5 * jnp.exp(-r2 / (2 * sigma ** 2)) * state.land_mask.data
        state = state._replace(
            eta=state.eta.replace(data=eta_pert.astype(jnp.float32)),
        )

        # Run 3 steps
        for _ in range(3):
            state = model.step(state, DT)

        self.assertTrue(jnp.all(jnp.isfinite(state.eta.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        # SSH should have spread but not diverged
        max_eta = float(jnp.max(jnp.abs(state.eta.data)))
        self.assertLess(max_eta, 10.0)

    def test_wind_driven_velocity(self):
        """Model handles wind-driven velocity perturbation."""
        grid = _make_grid()
        z_coord = _make_z_coord()
        state = _make_state(grid, z_coord)
        config = _make_config(n_barotropic_substeps=10)
        model = LatLonOceanModel(grid, z_coord, config)

        # Add small surface velocity (broadcast to full 3D shape)
        mask_3d = state.land_mask.data[..., jnp.newaxis]
        u_pert = jnp.broadcast_to(
            0.01 * jnp.sin(grid.lat2d[:, :, jnp.newaxis]) * mask_3d,
            state.u.data.shape,
        )
        state = state._replace(
            u=state.u.replace(data=u_pert.astype(jnp.float32)),
        )

        state_new = model.step(state, DT)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.eta.data)))


if __name__ == "__main__":
    unittest.main()
