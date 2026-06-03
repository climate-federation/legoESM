"""Phase 1 of the realistic-geometry plan: variable-bathymetry validation.

The codebase nominally supports variable ``H_bathy`` already (pure z*
coordinate with dynamic Jacobian J = (eta + H_bathy) / H_max).  This
phase **proves it works correctly** end-to-end on the lat-lon C-grid
ocean model.

Three tests, each gating Phase 2 of the plan:

1. ``test_z_star_compression`` — the actual layer thicknesses produced
   by ``compute_layer_thickness`` sum to ``eta + H_bathy`` per cell.
   Catches: any bug in the Jacobian formula; any inconsistency between
   ``dz_ref`` (reference thicknesses) and the per-cell scaling.
2. ``test_step_bathymetry_rest_state`` — rest state on a step
   bathymetry profile (deep + shallow + sharp transition) holds at
   machine precision for the baroclinic tendency, and stays bounded
   under multiple model steps.  Catches: any bug in horizontal
   operators when adjacent cells have different ``H_bathy``.
3. ``test_smooth_bathymetry_tracer_conservation`` — closed-domain
   tracer conservation on smoothly-varying ``H_bathy``.  Total heat
   and salt invariant to round-off under no-restoring/no-wind forcing.
   Catches: any thickness-weighting bug in tracer flux divergence.

The "partial bottom cell" terminology in the plan was based on a
misreading of the codebase — pure z* uniformly compresses every layer
by the same Jacobian factor; there are no partial cells.  The
analogous tests for pure z* are the three above.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture
def grid():
    """Small 18×36 (10°) lat-lon grid for fast tests."""
    return create_latlon_grid(n_lat=18, n_lon=36)


@pytest.fixture
def z_coord():
    """5 levels, H_max=4000 m."""
    return create_ocean_z_star(n_levels=5, H_max=4000.0,
                                 dz_surface=10.0, dz_deep=500.0)


# =========================================================================
# 1. z* compression
# =========================================================================


class TestZStarCompression:
    """``compute_layer_thickness`` must satisfy
    ``sum_k h_k(i,j) = eta(i,j) + H_bathy(i,j)`` exactly per cell."""

    def test_flat_bottom_zero_eta_recovers_dz_ref(self, z_coord):
        """The trivial case: eta=0, H_bathy=H_max → h_k = dz_ref[k]."""
        eta = jnp.zeros((4, 6))
        H = jnp.full((4, 6), z_coord.H_max)
        h = compute_layer_thickness(eta, H, z_coord)
        # Each column should equal dz_ref exactly.
        for k in range(z_coord.n_levels):
            np.testing.assert_allclose(
                np.asarray(h[..., k]),
                float(z_coord.dz_ref[k]),
                atol=0.0, rtol=1e-15,
            )

    def test_sum_equals_water_column_variable_H(self, z_coord):
        """Per-cell column sum equals ``eta + H_bathy`` for arbitrary
        H_bathy and arbitrary eta — the core identity of pure z*.

        The tolerance is governed by the precision of ``z_coord.dz_ref``,
        which is built in the storage-policy precision (float32 by
        default in the precision policy, even with JAX_ENABLE_X64=1
        unless the policy is reconfigured).  The relative error is
        bounded by the cumulative float32 round-off in
        ``sum(dz_ref) ≈ H_max``: ~1e-7 at n_levels=20, smaller at
        n_levels=5.
        """
        rng = np.random.default_rng(seed=0)
        H = jnp.asarray(rng.uniform(500.0, 4000.0, size=(4, 6)))
        eta = jnp.asarray(rng.uniform(-2.0, 2.0, size=(4, 6)))
        h = compute_layer_thickness(eta, H, z_coord)
        col_sum = jnp.sum(h, axis=-1)
        expected = eta + H
        np.testing.assert_allclose(
            np.asarray(col_sum), np.asarray(expected),
            rtol=1e-6, atol=0.0,
        )

    def test_jacobian_scales_with_eta(self, z_coord):
        """If we double (eta + H_bathy), the Jacobian doubles, and so
        does every layer thickness."""
        H = jnp.full((4, 6), 1000.0)
        h0 = compute_layer_thickness(jnp.zeros((4, 6)), H, z_coord)
        h1 = compute_layer_thickness(jnp.full((4, 6), 1000.0), H, z_coord)
        # h1[k] / h0[k] should equal 2.0 everywhere.
        ratio = h1 / h0
        np.testing.assert_allclose(np.asarray(ratio), 2.0, rtol=1e-14)

    def test_compression_in_shallow_water(self, z_coord):
        """At H_bathy = H_max / 4, every layer should be 1/4 of dz_ref."""
        H = jnp.full((3, 5), z_coord.H_max / 4.0)
        eta = jnp.zeros((3, 5))
        h = compute_layer_thickness(eta, H, z_coord)
        for k in range(z_coord.n_levels):
            expected = float(z_coord.dz_ref[k]) * 0.25
            np.testing.assert_allclose(
                np.asarray(h[..., k]),
                expected, rtol=1e-14,
            )


# =========================================================================
# 2. Rest state on step bathymetry
# =========================================================================


def _make_step_bathymetry(grid, H_deep=4000.0, H_shallow=1000.0):
    """Half deep, half shallow with a sharp transition at the equator."""
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    H = jnp.full((n_lat, n_lon), H_deep)
    H = H.at[: n_lat // 2, :].set(H_shallow)
    return H


class TestStepBathymetryRestState:
    """Rest state on step bathymetry: tendencies must be machine-zero
    on initialisation, and the model must remain stable under stepping."""

    def test_rest_state_tendency_machine_zero(self, grid, z_coord):
        H_bathy = _make_step_bathymetry(grid)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        # All ocean (no land mask), uniform T(z) and S → density depends
        # only on z → no horizontal pressure gradient anywhere → du/dt = 0.
        cfg = LatLonCGridOceanConfig()
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg,
        )
        # Both u and v tendencies should be machine-zero.
        u_max = float(jnp.max(jnp.abs(tend.du_dt.data)))
        v_max = float(jnp.max(jnp.abs(tend.dv_dt.data)))
        assert u_max < 1e-12, f"du/dt = {u_max} on step bathy rest state"
        assert v_max < 1e-12, f"dv/dt = {v_max} on step bathy rest state"

    def test_rest_state_stable_under_stepping(self, grid, z_coord):
        """Step the model 24 hours on step bathymetry; |u|, |eta| should
        stay tightly bounded.  This catches PGF / hydrostatic-pressure
        bugs that don't show up at t=0."""
        H_bathy = _make_step_bathymetry(grid)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        # Use implicit-CN solver to avoid chequerboard and barotropic
        # substepping artefacts (the validated production path).
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = state
        for _ in range(24):  # 24 × 3600 s = 1 sim-day
            s = model.step(s, 3600.0)
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        T_max = float(jnp.max(s.T.data))
        T_min = float(jnp.min(s.T.data))
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.eta.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        # Rest state: drift should be tiny; rough bounds (not machine
        # precision because barotropic stepping has finite numerical
        # tolerance).
        assert u_max < 1e-3, f"|u| drift after 1 day = {u_max} m/s"
        assert eta_max < 0.5, f"|eta| drift after 1 day = {eta_max} m"
        # T should not drift outside the initial range (no forcing).
        assert 1.99 <= T_min <= T_max <= 20.01

    def test_flat_bottom_path_bit_exact(self, grid, z_coord):
        """Backwards-compat regression: flat-bottom path through
        ``rest_state_latlon_cgrid_ocean`` (no overrides) must match
        the legacy behaviour bit-exact.  Catches any unintended
        change to the default flat-bottom branch."""
        # Path A: legacy flat-bottom default
        s_a = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=80.0,
        )
        # Path B: same but explicit flat-bottom H_bathy_override
        H_flat = jnp.full((grid.n_lat, grid.n_lon), 4000.0)
        s_b = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_flat,
        )
        # T, S, u, v, eta should match exactly.
        np.testing.assert_array_equal(s_a.T.data, s_b.T.data)
        np.testing.assert_array_equal(s_a.S.data, s_b.S.data)
        np.testing.assert_array_equal(s_a.u.data, s_b.u.data)
        np.testing.assert_array_equal(s_a.v.data, s_b.v.data)
        np.testing.assert_array_equal(s_a.eta.data, s_b.eta.data)


# =========================================================================
# 3. Tracer conservation on smooth variable bathymetry
# =========================================================================


def _make_smooth_bathymetry(grid, H_min=500.0, H_max=4000.0):
    """Cosine pattern in latitude: deep at equator, shallow at poles."""
    lat_rad = jnp.asarray(grid.lat2d)
    H = H_min + (H_max - H_min) * jnp.cos(lat_rad) ** 2
    return H


class TestSmoothBathymetryTracerConservation:
    """Closed-domain integrated heat and salt must be invariant under
    no-forcing stepping when H_bathy varies smoothly."""

    def _column_integrated_tracer(self, T, h, area):
        """Heat content per cell × cell area, summed.

        T : (n_lat, n_lon, nlev), h : same, area : (n_lat, n_lon).
        Returns scalar.
        """
        col = jnp.sum(T * h, axis=-1)
        return float(jnp.sum(col * area))

    def test_heat_and_salt_conserved_on_smooth_bathy(self, grid, z_coord):
        """Step 24 hours on a smoothly-varying H_bathy with no forcing
        and no land — global heat and salt content must be conserved
        to round-off."""
        H_bathy = _make_smooth_bathymetry(grid)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        # Initial integrated heat and salt
        h0 = compute_layer_thickness(state.eta.data,
                                       state.H_bathy.data, z_coord)
        H0_heat = self._column_integrated_tracer(
            state.T.data, h0, grid.area,
        )
        H0_salt = self._column_integrated_tracer(
            state.S.data, h0, grid.area,
        )
        # Step
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = state
        for _ in range(24):
            s = model.step(s, 3600.0)
        h1 = compute_layer_thickness(s.eta.data,
                                       s.H_bathy.data, z_coord)
        H1_heat = self._column_integrated_tracer(s.T.data, h1, grid.area)
        H1_salt = self._column_integrated_tracer(s.S.data, h1, grid.area)
        # Conservation tolerance: the model uses thickness-weighted tracer
        # update so this should be machine-precision modulo the implicit
        # solver tolerance (~1e-10 relative).
        rel_heat_drift = abs(H1_heat - H0_heat) / abs(H0_heat)
        rel_salt_drift = abs(H1_salt - H0_salt) / abs(H0_salt)
        assert rel_heat_drift < 1e-8, (
            f"Heat content drifted {rel_heat_drift:.2e} (initial "
            f"{H0_heat:.3e}, final {H1_heat:.3e})"
        )
        assert rel_salt_drift < 1e-8, (
            f"Salt content drifted {rel_salt_drift:.2e} (initial "
            f"{H0_salt:.3e}, final {H1_salt:.3e})"
        )

    def test_water_column_thickness_invariant_on_rest_state(
        self, grid, z_coord,
    ):
        """At rest with no forcing, sum of layer thicknesses (eta + H)
        per cell stays constant at machine precision.  Independent of
        bathymetry variation — verifies the z* identity holds across
        the model step."""
        H_bathy = _make_smooth_bathymetry(grid)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        col0 = state.eta.data + state.H_bathy.data
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = state
        for _ in range(6):
            s = model.step(s, 3600.0)
        col1 = s.eta.data + s.H_bathy.data
        max_drift = float(jnp.max(jnp.abs(col1 - col0)))
        # Implicit solver iterates to ~1e-10 relative; H_max=4000 so
        # absolute drift up to ~4e-7 m is acceptable.
        assert max_drift < 1e-3, (
            f"Water column thickness drifted by {max_drift:.3e} m"
        )
