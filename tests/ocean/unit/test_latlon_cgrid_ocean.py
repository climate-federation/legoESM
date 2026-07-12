"""Tests for the lat-lon C-grid FV ocean model.

Verifies that the C-grid discretization produces correct, stable, and
checkerboard-free results.  The C-grid eliminates the 2*dx null space
present in the A-grid (LatLonOceanModel) formulation.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
    laplacian_smag_cfl_cap,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    gradient_x_cgrid,
    gradient_y_cgrid,
    divergence_cgrid,
    compute_face_masks,
)


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=36, n_lon=72)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=5, H_max=4000.0)


@pytest.fixture
def state(grid, z_coord):
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )


@pytest.fixture
def config():
    return LatLonCGridOceanConfig.from_flat()


# =========================================================================
# C-grid operator unit tests
# =========================================================================

class TestCGridOperators:
    """Compact-stencil C-grid operators."""

    def test_gradient_x_shape(self, grid):
        f = jnp.ones((grid.n_lat, grid.n_lon))
        gx = gradient_x_cgrid(f, grid)
        assert gx.shape == (grid.n_lat, grid.n_lon + 1)

    def test_gradient_y_shape(self, grid):
        f = jnp.ones((grid.n_lat, grid.n_lon))
        gy = gradient_y_cgrid(f, grid)
        assert gy.shape == (grid.n_lat + 1, grid.n_lon)

    def test_gradient_constant_field_zero(self, grid):
        """Gradient of a constant field should be zero everywhere."""
        f = jnp.ones((grid.n_lat, grid.n_lon)) * 42.0
        gx = gradient_x_cgrid(f, grid)
        gy = gradient_y_cgrid(f, grid)
        assert float(jnp.max(jnp.abs(gx))) < 1e-10
        assert float(jnp.max(jnp.abs(gy))) < 1e-10

    def test_divergence_shape(self, grid):
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        u = jnp.zeros((n_lat, n_lon + 1))
        v = jnp.zeros((n_lat + 1, n_lon))
        div = divergence_cgrid(u, v, grid)
        assert div.shape == (n_lat, n_lon)

    def test_divergence_uniform_zero(self, grid):
        """Divergence of a uniform field should be near zero."""
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        u = jnp.ones((n_lat, n_lon + 1))
        v = jnp.zeros((n_lat + 1, n_lon))
        div = divergence_cgrid(u, v, grid)
        # On a sphere, div of uniform u is not exactly zero due to
        # geometry, but should be bounded
        assert jnp.all(jnp.isfinite(div))

    def test_face_masks(self, grid):
        """Face masks should have correct shapes."""
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        mask = jnp.ones((n_lat, n_lon))
        u_mask, v_mask = compute_face_masks(mask)
        assert u_mask.shape == (n_lat, n_lon + 1)
        assert v_mask.shape == (n_lat + 1, n_lon)
        # All-ocean: u_mask should be 1 everywhere
        assert float(jnp.min(u_mask)) == 1.0
        # v_mask at poles should be 0 (wall BC)
        assert float(v_mask[0].max()) == 0.0
        assert float(v_mask[-1].max()) == 0.0

    def test_face_masks_with_land(self):
        """Face between ocean and land should be masked."""
        mask = jnp.array([[1.0, 0.0, 1.0, 1.0]])
        u_mask, v_mask = compute_face_masks(mask)
        # u_mask between ocean (col 0) and land (col 1) should be 0
        assert float(u_mask[0, 1]) == 0.0
        # u_mask between ocean cells should be 1
        assert float(u_mask[0, 3]) == 1.0


# =========================================================================
# C-grid tendency tests
# =========================================================================

class TestCGridTendencies:
    """C-grid ocean baroclinic tendencies."""

    def test_tendencies_finite(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_rest_state_small_tendencies(self, state, grid, z_coord, config):
        # At u=v=0 with horizontally uniform T, S and flat bathymetry,
        # every term in the baroclinic momentum tendency vanishes by
        # construction: ζ=0, KE=0, ∇p'=0. Under JAX_ENABLE_X64=1 the
        # result is exactly 0.0. The previous 1e-2 bound was six orders
        # of magnitude too slack and masked regressions. See #160.
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-14
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-14

    def test_tendency_shapes(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        nlev = z_coord.n_levels
        assert tend.du_dt.data.shape == (n_lat, n_lon + 1, nlev)
        assert tend.dv_dt.data.shape == (n_lat + 1, n_lon, nlev)
        assert tend.dT_dt.data.shape == (n_lat, n_lon, nlev)
        assert tend.dS_dt.data.shape == (n_lat, n_lon, nlev)
        assert tend.deta_dt.data.shape == (n_lat, n_lon)

    def test_static_fields_zero(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        assert jnp.all(tend.dH_bathy_dt.data == 0)
        assert jnp.all(tend.dland_mask_dt.data == 0)

    def test_baroclinic_tendency_not_galilean_invariant(
        self, state, grid, z_coord, config,
    ):
        """Barotropic-shift test for issue #160 (total-velocity KE/PV).

        Two states differ only by a spatially constant δU added to u.
        The total-velocity KE gradient produces an extra −δU·∂u/∂x
        tendency, so du_dt_A ≠ du_dt_B when ∂u/∂x ≠ 0.
        """
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        nlev = z_coord.n_levels

        # Nontrivial depth-dependent zonal flow: 0.5·sin(lon)·z_profile
        lon_u_1d = jnp.linspace(0.0, 2.0 * jnp.pi, n_lon + 1)
        z_profile = jnp.linspace(1.0, 0.3, nlev)
        u_A = (
            0.5
            * jnp.sin(lon_u_1d)[None, :, None]
            * z_profile[None, None, :]
            * jnp.ones((n_lat, n_lon + 1, nlev))
        )
        u_mask_3d = state.u_mask.data[:, :, None]
        u_A = u_A * u_mask_3d

        delta_U = 0.1
        u_B = u_A + delta_U * u_mask_3d

        state_A = state._replace(u=state.u.replace(data=u_A))
        state_B = state._replace(u=state.u.replace(data=u_B))

        tend_A = latlon_cgrid_ocean_baroclinic_tendencies(
            state_A, grid, z_coord, config,
        )
        tend_B = latlon_cgrid_ocean_baroclinic_tendencies(
            state_B, grid, z_coord, config,
        )

        diff_du = float(
            jnp.max(jnp.abs(tend_A.du_dt.data - tend_B.du_dt.data))
        )
        # Total-velocity KE gradient produces O(δU · ∂u/∂x) difference.
        assert diff_du > 1e-12, (
            f"Baroclinic du/dt should depend on barotropic offset δU "
            f"via the KE gradient; got diff={diff_du:.3e} (invariant → bug)."
        )


# =========================================================================
# C-grid model integration tests
# =========================================================================

class TestCGridOceanModel:
    """C-grid ocean model integration."""

    def test_single_step(self, grid, z_coord, config, state):
        model = LatLonCGridOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert jnp.all(jnp.isfinite(s1.u.data))
        assert jnp.all(jnp.isfinite(s1.v.data))
        assert jnp.all(jnp.isfinite(s1.T.data))
        assert jnp.all(jnp.isfinite(s1.eta.data))

    def test_shapes_preserved(self, grid, z_coord, config, state):
        model = LatLonCGridOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert s1.u.data.shape == state.u.data.shape
        assert s1.v.data.shape == state.v.data.shape
        assert s1.T.data.shape == state.T.data.shape
        assert s1.eta.data.shape == state.eta.data.shape

    def test_static_fields_unchanged(self, grid, z_coord, config, state):
        model = LatLonCGridOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert jnp.allclose(s1.H_bathy.data, state.H_bathy.data)
        assert jnp.allclose(s1.land_mask.data, state.land_mask.data)
        assert jnp.allclose(s1.u_mask.data, state.u_mask.data)
        assert jnp.allclose(s1.v_mask.data, state.v_mask.data)

    def test_rest_state_stability(self, grid, z_coord, state):
        """Rest state should show minimal drift (key C-grid advantage).

        A-grid models develop checkerboard noise from the 2*dx null space.
        The C-grid should remain quiet.
        """
        cfg = LatLonCGridOceanConfig.from_flat(A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = state
        for _ in range(10):
            s = model.step(s, 3600.0)

        # Should remain stable
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.eta.data))

        # SSH drift should be very small for a rest state
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        assert eta_max < 1.0, f"Rest state eta drift too large: {eta_max}"

    def test_no_eta_volume_drift(self, grid, z_coord, state):
        """Issue #271: ``sum(eta * area)`` must not drift step-to-step.

        Default Mercator C-grid config enables ``fix_eta_drift`` so the
        global volume integral is preserved to numerical precision
        regardless of bathymetry complexity.
        """
        cfg = LatLonCGridOceanConfig.from_flat()
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        area = grid.area
        mask = state.land_mask.data
        vol0 = float(jnp.sum(state.eta.data * area * mask))
        s = state
        for _ in range(20):
            s = model.step(s, 3600.0)
        vol_final = float(jnp.sum(s.eta.data * area * mask))
        ocean_area = float(jnp.sum(area * mask))
        # Mean eta drift per step (m).  Must be machine-precision.
        mean_drift = abs(vol_final - vol0) / max(ocean_area, 1.0)
        assert mean_drift < 1.0e-10, (
            f"Volume leak: mean eta drift {mean_drift} m after 20 steps"
        )

    def test_eta_drift_correction_can_be_disabled(self, grid, z_coord, state):
        """Backwards-compat: ``fix_eta_drift=False`` restores the legacy
        path so existing regression baselines can opt out."""
        cfg = LatLonCGridOceanConfig.from_flat(fix_eta_drift=False)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = model.step(state, 3600.0)
        # Just check that the disabled path runs and produces finite eta.
        assert jnp.all(jnp.isfinite(s.eta.data))

    def test_no_eta_volume_drift_partial_cells(self, grid):
        """Issue #271 root cause: partial-cell bathymetry amplifies the
        leak 7×.  Run with a non-flat bathymetry on the partial-cell
        coord and assert the default projection still pins the volume
        integral exactly.
        """
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )

        H_max = 4000.0
        # Stepped bathymetry — emulates the partial-cell mismatch path.
        n_lat, n_lon = grid.n_lat, grid.n_lon
        H_bathy = H_max * (
            0.5 + 0.5 * jnp.cos(
                jnp.pi * jnp.arange(n_lat)[:, None] / n_lat
            ) * jnp.cos(
                jnp.pi * jnp.arange(n_lon)[None, :] / n_lon
            )
        )
        zstar = create_ocean_z_star(n_levels=5, H_max=H_max)
        z_coord = create_partial_cell_coordinate(zstar, H_bathy)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=H_max,
            H_bathy_override=H_bathy,
        )
        cfg = LatLonCGridOceanConfig.from_flat()
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        area = grid.area
        mask = state.land_mask.data
        vol0 = float(jnp.sum(state.eta.data * area * mask))
        s = state
        for _ in range(10):
            s = model.step(s, 1800.0)
        vol_final = float(jnp.sum(s.eta.data * area * mask))
        ocean_area = float(jnp.sum(area * mask))
        mean_drift = abs(vol_final - vol0) / max(ocean_area, 1.0)
        # f32 storage cast at end of each step introduces ~1e-10 m/step
        # round-off even when the projection is exact in accumulation
        # precision.  Bound is still 10⁴× tighter than the unfixed
        # ETOPO leak (0.4 mm/yr ≈ 1e-7 m/step).
        assert mean_drift < 1.0e-8, (
            f"Partial-cell volume leak: mean eta drift {mean_drift} m"
        )

    def test_no_eta_volume_drift_explicit_substepping(
        self, grid, z_coord, state,
    ):
        """The explicit substepping barotropic solver does not project
        the global mean internally — the end-of-step fix must catch
        any leak it introduces."""
        cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="explicit_substep")
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        area = grid.area
        mask = state.land_mask.data
        vol0 = float(jnp.sum(state.eta.data * area * mask))
        s = state
        for _ in range(10):
            s = model.step(s, 600.0)
        vol_final = float(jnp.sum(s.eta.data * area * mask))
        ocean_area = float(jnp.sum(area * mask))
        mean_drift = abs(vol_final - vol0) / max(ocean_area, 1.0)
        assert mean_drift < 1.0e-10, (
            f"Substepping volume leak: mean eta drift {mean_drift} m"
        )

    def test_freshwater_target_volume(self, grid, z_coord, state):
        """With non-zero freshwater the projection target is
        ``vol(eta_old) + dt * sum(F_eta * area)``, not just
        ``vol(eta_old)``.  Drive a precipitation excess and verify the
        post-step volume matches that target."""
        from legoesm.ocean.freshwater import (
            zero_freshwater, freshwater_eta_tendency,
        )

        # Lat-lon C-grid uses 2-D freshwater forcing shape (n_lat, n_lon).
        z2 = jnp.zeros((grid.n_lat, grid.n_lon))
        precip = jnp.full_like(z2, 1.0e-4)  # ~ 8 mm/day
        from legoesm.ocean.freshwater import FreshwaterForcing
        fw = FreshwaterForcing(
            precip=precip, evap=z2, runoff=z2, ice_fw=z2, restoring=z2,
        )
        cfg = LatLonCGridOceanConfig.from_flat(freshwater_closure="virtual_salt_flux")
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        area = grid.area
        mask = state.land_mask.data
        vol0 = float(jnp.sum(state.eta.data * area * mask))
        F_eta = freshwater_eta_tendency(fw, cfg.rho_0) * mask
        expected_change = float(jnp.sum(F_eta * area * mask)) * 3600.0
        s = model.step(state, 3600.0, freshwater=fw)
        vol_final = float(jnp.sum(s.eta.data * area * mask))
        observed_change = vol_final - vol0
        ocean_area = float(jnp.sum(area * mask))
        # The relative match must be tight; the storage-precision cast
        # at the end of step bounds the absolute error to ~1e-9 m * area.
        rel_err = abs(observed_change - expected_change) / max(
            abs(expected_change), 1.0
        )
        assert rel_err < 1.0e-4, (
            f"Freshwater volume target mismatch: rel_err={rel_err} "
            f"(expected {expected_change}, observed {observed_change})"
        )

    def test_implicit_cn_no_op_when_no_leak(self, grid, z_coord, state):
        """When the implicit-CN solver already projects volume
        internally, our end-of-step projection must be a near-no-op
        — verify that toggling ``fix_eta_drift`` on vs off gives
        bit-close eta fields on a single step where no real leak
        exists."""
        cfg_on = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="implicit_cn", fix_eta_drift=True,
        )
        cfg_off = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="implicit_cn", fix_eta_drift=False,
        )
        s_on = LatLonCGridOceanModel(grid, z_coord, cfg_on).step(state, 3600.0)
        s_off = LatLonCGridOceanModel(grid, z_coord, cfg_off).step(state, 3600.0)
        diff = float(jnp.max(jnp.abs(s_on.eta.data - s_off.eta.data)))
        # Implicit-CN already conserves volume to ~1e-10 m, so our
        # extra projection should not visibly perturb eta.
        assert diff < 1.0e-7, (
            f"fix_eta_drift perturbed implicit-CN eta by {diff} m"
        )

    def test_eta_projection_consistent_with_layer_thickness(
        self, grid, z_coord, state,
    ):
        """The eta correction is applied BEFORE the tracer step's
        ``h_k_new`` recompute, so layer thicknesses, w, T and S must
        all be consistent with the corrected eta.  Verify by checking
        that ``sum_k h_k_new`` matches the post-step ``eta + H_bathy``
        column thickness within numerical precision.
        """
        from legoesm.ocean.vertical import compute_layer_thickness

        cfg = LatLonCGridOceanConfig.from_flat()
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = model.step(state, 3600.0)
        h_k = compute_layer_thickness(
            s.eta.data, s.H_bathy.data, model.z_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        wc = jnp.sum(h_k, axis=-1)
        expected = jnp.maximum(
            s.eta.data + s.H_bathy.data, cfg.min_water_column_m,
        )
        rel = jnp.max(
            jnp.abs(wc - expected) / jnp.maximum(expected, 1.0)
        )
        # f32 storage cast at end of step bounds the relative
        # consistency to ~1e-7; we just need to confirm the eta
        # correction did NOT desynchronise h_k from eta by an
        # O(correction) amount (which would be a real bug).
        assert float(rel) < 1.0e-6, (
            f"h_k_new not consistent with corrected eta: rel err {float(rel)}"
        )

    def test_barotropic_wave_stability(self, grid, z_coord):
        """Barotropic wave should propagate stably for 10+ steps.

        Apply a Gaussian SSH perturbation and verify the model
        stays stable without checkerboard artifacts.

        Uses dt=600s with 60 barotropic substeps to satisfy
        CFL ~ 0.08 (c ~ 198 m/s, dx_min ~ 24 km at high lat).
        """
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0,
        )

        # Add Gaussian SSH perturbation
        lat_deg = grid.lat2d * (180.0 / jnp.pi)
        lon_deg = grid.lon2d * (180.0 / jnp.pi)
        eta_pert = 1.0 * jnp.exp(
            -((lon_deg - 180.0) ** 2 + lat_deg ** 2) / (10.0 ** 2)
        )
        eta_pert = eta_pert * state.land_mask.data
        state = state._replace(
            eta=state.eta.replace(data=eta_pert),
        )

        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=1e4, K_h=1e3, n_barotropic_substeps=60,
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)

        s = state
        for _ in range(10):
            s = model.step(s, 600.0)

        assert jnp.all(jnp.isfinite(s.eta.data))
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.v.data))

        # SSH should remain physically reasonable (wave disperses)
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        assert eta_max < 10.0, f"Wave amplitude blow-up: {eta_max}"

    def test_differentiable(self, grid, z_coord, config, state):
        """C-grid model should be differentiable through tendencies."""
        def loss_fn(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            tend = latlon_cgrid_ocean_baroclinic_tendencies(
                s, grid, z_coord, config,
            )
            return jnp.mean(tend.deta_dt.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.eta.data)
        assert jnp.all(jnp.isfinite(g))


# =========================================================================
# Bugfix-knob coverage (PR #261)
# =========================================================================


class TestPR261ConfigDispatch:
    """Each new public-config dispatch branch from PR #261 must select
    cleanly via :class:`LatLonCGridOceanConfig` and produce finite
    tendencies.  These tests guard the slopbuster-flagged untested
    dispatch knobs (`A_h_merid`, `B_h_lat_scaling`)."""

    def test_A_h_merid_branch_active(self, grid, z_coord, state):
        cfg = LatLonCGridOceanConfig.from_flat(A_h_merid=5.0e4)
        assert cfg.lateral_viscosity.A_h_merid > 0.0
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))

    def test_A_h_merid_default_inactive(self, grid, z_coord, state):
        cfg = LatLonCGridOceanConfig.from_flat()
        assert cfg.lateral_viscosity.A_h_merid == 0.0

    def test_B_h_lat_scaling_on(self, grid, z_coord, state):
        cfg = LatLonCGridOceanConfig.from_flat(B_h=1.0e10, B_h_lat_scaling=True)
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))

    def test_B_h_lat_scaling_off(self, grid, z_coord, state):
        cfg = LatLonCGridOceanConfig.from_flat(B_h=1.0e10, B_h_lat_scaling=False)
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))


def test_gm_redi_surface_complement_config_defaults_round_trip():
    """The two new ``GMRediConfig`` surface-complement knobs from
    PR #261 must round-trip through the NamedTuple cleanly."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig

    cfg = GMRediConfig()
    assert cfg.surface_complement is True
    assert cfg.surface_complement_depth == 100.0

    cfg2 = GMRediConfig(surface_complement=False, surface_complement_depth=50.0)
    assert cfg2.surface_complement is False
    assert cfg2.surface_complement_depth == 50.0


class TestSmagCFLCap:
    """``smag_cfl_safety`` caps the Laplacian-Smagorinsky coefficient at the
    per-cell anisotropic viscous-CFL estimate
    ``smag_cfl_safety / (dt * (1/dx^2 + 1/dy^2))`` (``laplacian_smag_cfl_cap``)
    so a large ``C_smag_lap`` can damp sharp western-boundary-current jets
    without self-CFL-violating -- the eORCA025 WOA cold-start fix."""

    def _sharp_jet(self, grid, z_coord):
        st = rest_state_latlon_cgrid_ocean(
            grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0,
        )
        # Grid-scale shear: alternate +/-3 m/s zonal velocity row by row so the
        # strain rate |D| is large and the uncapped Smagorinsky coefficient
        # A=(C*dx)^2*|D| shoots past the per-cell CFL limit.
        nlat = st.u.data.shape[0]
        sign = (jnp.arange(nlat) % 2 * 2 - 1).astype(st.u.data.dtype)
        u = (st.u.data + 3.0 * sign[:, None, None]) * st.u_mask.data[..., None]
        return st._replace(u=st.u.replace(data=u))

    def test_cap_on_by_default(self):
        # Issue #939: the Laplacian-Smagorinsky CFL cap is ON by default
        # (smag_cfl_safety=0.125) so an uncapped A_smag cannot self-CFL-blow a
        # WOA cold-start.  0.0 opts OUT (uncapped; unsafe with a large C_smag_lap).
        assert LatLonCGridOceanConfig.from_flat().lateral_viscosity.smag_cfl_safety == 0.125

    def test_default_cap_protects_sharp_jet_over_steps(self, grid, z_coord):
        """Issue #939 regression: with a large ``C_smag_lap`` and NO explicit
        ``smag_cfl_safety`` (i.e. the DEFAULT 0.125 cap), a sharp grid-scale jet
        stays finite over many steps -- the uncapped-Smagorinsky self-CFL blowup
        (lat-lon 180x360x40 dt=300 WOA cold-start, non-finite by ~step 36) cannot
        happen.  The explicit opt-out ``smag_cfl_safety=0.0`` is UNSAFE at large
        ``C_smag_lap`` and DOES blow on the same jet, which is why the cap is the
        default."""
        st = self._sharp_jet(grid, z_coord)
        C = 100.0  # cranked, as the sharp WOA cold-start jets effectively are
        # DEFAULT config: smag_cfl_safety unset -> the 0.125 cap auto-applies.
        cfg_default = LatLonCGridOceanConfig.from_flat(C_smag_lap=C)
        assert cfg_default.lateral_viscosity.smag_cfl_safety == 0.125
        s = st
        m_default = LatLonCGridOceanModel(grid, z_coord, cfg_default)
        for _ in range(10):
            s = m_default.step(s, 300.0)
        assert bool(jnp.all(jnp.isfinite(s.u.data)))
        assert bool(jnp.all(jnp.isfinite(s.v.data)))
        assert float(jnp.max(jnp.abs(s.u.data))) < 50.0  # bounded near the jet scale
        # Explicit opt-out (uncapped) self-CFL-blows on the SAME jet.
        cfg_uncapped = LatLonCGridOceanConfig.from_flat(
            C_smag_lap=C, smag_cfl_safety=0.0)
        m_uncapped = LatLonCGridOceanModel(grid, z_coord, cfg_uncapped)
        s2 = st
        blew = False
        for _ in range(10):
            s2 = m_uncapped.step(s2, 300.0)
            if not bool(jnp.all(jnp.isfinite(s2.u.data))):
                blew = True
                break
        assert blew, ("uncapped Smagorinsky (smag_cfl_safety=0.0 opt-out) should "
                      "self-CFL-blow on the sharp jet -- the #939 mechanism")

    def test_cap_bounds_and_reduces_smag(self, grid, z_coord):
        dt = 75.0
        C = 100.0  # huge -> the cap is guaranteed to bind at the sharp jet
        st = self._sharp_jet(grid, z_coord)
        t_un = latlon_cgrid_ocean_baroclinic_tendencies(
            st, grid, z_coord,
            LatLonCGridOceanConfig.from_flat(C_smag_lap=C, smag_cfl_safety=0.0), dt=dt,
        )
        t_cap = latlon_cgrid_ocean_baroclinic_tendencies(
            st, grid, z_coord,
            LatLonCGridOceanConfig.from_flat(C_smag_lap=C, smag_cfl_safety=0.125), dt=dt,
        )
        # Capped tendency stays finite and viscous-Courant-bounded.
        assert bool(jnp.all(jnp.isfinite(t_cap.du_dt.data)))
        assert bool(jnp.all(jnp.isfinite(t_cap.dv_dt.data)))
        peak_un = float(jnp.max(jnp.abs(t_un.du_dt.data)))
        peak_cap = float(jnp.max(jnp.abs(t_cap.du_dt.data)))
        assert peak_cap * dt < 50.0   # bounded near the 3 m/s jet scale
        # The cap really bit: the uncapped peak is non-finite or >> the capped.
        assert (not math.isfinite(peak_un)) or (peak_un > 5.0 * peak_cap)

    def test_cap_helper_area_cos2_ceiling(self, grid):
        """``laplacian_smag_cfl_cap`` returns the tuned ceiling
        ``safety * area * cos^2(lat) / dt`` (cos^2 floored at 0.04). ``grid`` is
        a ``LatLonGrid`` (area/lat2d branch)."""
        dt, safety = 75.0, 0.125
        cap_h, cap_q = laplacian_smag_cfl_cap(grid, dt, safety)
        lat2d = getattr(grid, "lat_T", None)
        if lat2d is None:
            lat2d = grid.lat2d
        area = getattr(grid, "area_T", None)
        if area is None:
            area = grid.area
        cos2 = jnp.maximum(jnp.cos(lat2d) ** 2, 0.04)
        assert jnp.allclose(cap_h, safety * area * cos2 / dt, rtol=1e-6)
        assert bool(jnp.all(jnp.isfinite(cap_h))) and bool(jnp.all(cap_h > 0))
        assert bool(jnp.all(jnp.isfinite(cap_q))) and bool(jnp.all(cap_q > 0))
        # cap_q is the vertex field (n_lat+1, n_lon+1).
        assert cap_q.shape == (cap_h.shape[0] + 1, cap_h.shape[1] + 1)

    def test_cap_q_periodic_seam(self, grid):
        """The q-point cap wraps periodically in longitude (no edge-replication
        discontinuity): column n_lon == column 0."""
        _, cap_q = laplacian_smag_cfl_cap(grid, 75.0, 0.125)
        assert jnp.array_equal(cap_q[:, -1], cap_q[:, 0])
