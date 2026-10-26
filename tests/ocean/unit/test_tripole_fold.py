"""Tests for tripolar fold halo exchange.

Validates that the fold permutation in pad_ns_scalar / pad_ns_vector
correctly mirrors data across the northern boundary of a tripolar grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    pad_ns_scalar,
    pad_ns_vector_u,
    pad_ns_vector_v,
    pad_ns_vector_pair,
    gradient_y_cgrid,
    divergence_cgrid,
    curl_vertex_cgrid,
    coriolis_cgrid,
    laplacian_cgrid,
)
from legoesm.ocean.vertical import nemo_t_fold_f_owned


@pytest.fixture
def tripole_grid():
    """Create a synthetic tripolar grid for testing."""
    return create_synthetic_tripole(36, 72)


@pytest.fixture
def regular_grid():
    """Create a regular lat-lon grid for comparison."""
    from legoesm.grids.latlon import create_latlon_geometry
    return create_latlon_geometry(36, 72)


# =========================================================================
# Fold round-trip tests
# =========================================================================


class TestFoldRoundTrip:
    """Verify the fold permutation is correct."""

    def test_scalar_fold_is_i_reversed(self, tripole_grid):
        """The fold at the north boundary should reverse the i-index."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon

        # Create a known pattern: T[j, i] = j * 1000 + i
        interior = jnp.arange(n_lon, dtype=jnp.float32)[jnp.newaxis, :]
        interior = interior + jnp.arange(n_lat - 1, dtype=jnp.float32)[:, jnp.newaxis] * 1000

        padded = pad_ns_scalar(interior, tripole_grid)
        assert padded.shape == (n_lat + 1, n_lon)

        # South boundary should be zero (wall)
        assert jnp.all(padded[0] == 0.0)

        # North boundary should be fold-reflected: last interior row, i-reversed
        last_row = interior[-1]  # the row closest to the fold
        expected_north = last_row[::-1]  # i-reversed
        assert jnp.allclose(padded[-1], expected_north), (
            f"North fold mismatch: got {padded[-1][:5]}, "
            f"expected {expected_north[:5]}"
        )

    def test_nemo_t_pivot_f_owned_row_uses_f_origin(self, tripole_grid):
        """NEMO T-fold F fields copy row j-1 with ``(-1-i) mod N``."""
        n_lat, n_lon = tripole_grid.n_lat, tripole_grid.n_lon
        field = jnp.arange(n_lat * n_lon * 2, dtype=jnp.float64).reshape(
            n_lat, n_lon, 2)
        folded = nemo_t_fold_f_owned(field, tripole_grid)
        expected = field[-2, jnp.arange(n_lon - 1, -1, -1)]
        assert jnp.array_equal(folded[-1], expected)
        assert jnp.array_equal(folded[:-1], field[:-1])

    def test_vector_u_fold_has_sign_flip(self, tripole_grid):
        """u-component across fold should be sign-reversed and i-reversed."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon

        interior = jnp.ones((n_lat - 1, n_lon), dtype=jnp.float32)
        padded = pad_ns_vector_u(interior, tripole_grid)

        # South = zero
        assert jnp.all(padded[0] == 0.0)

        # North = sign-flipped
        assert jnp.allclose(padded[-1], -1.0), (
            f"Expected -1.0 at fold, got {padded[-1, 0]}"
        )

    def test_vector_v_fold_has_sign_flip(self, tripole_grid):
        """v-component across fold should be sign-reversed and i-reversed."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon

        interior = jnp.ones((n_lat - 1, n_lon), dtype=jnp.float32)
        padded = pad_ns_vector_v(interior, tripole_grid)

        # North = sign-flipped
        assert jnp.allclose(padded[-1], -1.0)

    def test_fold_permutation_is_involution(self, tripole_grid):
        """Applying the fold twice should give the identity."""
        fold = tripole_grid.fold
        perm = fold.perm_T

        # perm[perm[i]] == i for all i
        roundtrip = perm[perm]
        assert jnp.all(roundtrip == jnp.arange(tripole_grid.n_lon))

    def test_scalar_fold_3d(self, tripole_grid):
        """Fold should work on 3D fields (lat, lon, lev)."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon
        nlev = 5

        interior = jax.random.normal(
            jax.random.PRNGKey(42), (n_lat - 1, n_lon, nlev)
        )
        padded = pad_ns_scalar(interior, tripole_grid)
        assert padded.shape == (n_lat + 1, n_lon, nlev)

        # South = zero
        assert jnp.all(padded[0] == 0.0)

        # North = fold-reflected last row
        fold = tripole_grid.fold
        expected = interior[-1][fold.perm_T]
        assert jnp.allclose(padded[-1], expected)


# =========================================================================
# Operator tests on tripolar grid
# =========================================================================


class TestOperatorsOnTripole:
    """Verify operators produce finite results on a synthetic tripolar grid."""

    def test_gradient_y_finite(self, tripole_grid):
        """gradient_y should produce finite results on tripolar grid."""
        f = jax.random.normal(jax.random.PRNGKey(0), (36, 72))
        result = gradient_y_cgrid(f, tripole_grid)
        assert result.shape == (37, 72)
        assert jnp.all(jnp.isfinite(result))
        # North boundary should be nonzero (fold gives data, not wall)
        assert not jnp.all(result[-1] == 0.0), (
            "North boundary of gradient_y should be nonzero with fold"
        )

    def test_divergence_finite(self, tripole_grid):
        """divergence should produce finite results on tripolar grid."""
        u = jax.random.normal(jax.random.PRNGKey(1), (36, 73))
        v = jax.random.normal(jax.random.PRNGKey(2), (37, 72))
        result = divergence_cgrid(u, v, tripole_grid)
        assert result.shape == (36, 72)
        assert jnp.all(jnp.isfinite(result))

    def test_curl_finite(self, tripole_grid):
        """curl should produce finite results on tripolar grid."""
        u = jax.random.normal(jax.random.PRNGKey(1), (36, 73))
        v = jax.random.normal(jax.random.PRNGKey(2), (37, 72))
        result = curl_vertex_cgrid(u, v, tripole_grid)
        assert result.shape == (37, 73)
        assert jnp.all(jnp.isfinite(result))
        # Vorticity at north boundary should be nonzero with fold
        assert not jnp.all(result[-1] == 0.0)

    def test_coriolis_finite(self, tripole_grid):
        """Coriolis should produce finite results on tripolar grid."""
        u = jax.random.normal(jax.random.PRNGKey(1), (36, 73))
        v = jax.random.normal(jax.random.PRNGKey(2), (37, 72))
        cor_u, cor_v = coriolis_cgrid(u, v, tripole_grid)
        assert cor_u.shape == (36, 73)
        assert cor_v.shape == (37, 72)
        assert jnp.all(jnp.isfinite(cor_u))
        assert jnp.all(jnp.isfinite(cor_v))

    def test_laplacian_finite(self, tripole_grid):
        """Laplacian should produce finite results on tripolar grid."""
        f = jax.random.normal(jax.random.PRNGKey(0), (36, 72))
        result = laplacian_cgrid(f, tripole_grid)
        assert result.shape == (36, 72)
        assert jnp.all(jnp.isfinite(result))


# =========================================================================
# Regular lat-lon comparison (backward compat)
# =========================================================================


class TestRegularLatLonBackwardCompat:
    """Ensure pad_ns_* on regular lat-lon gives same results as jnp.pad."""

    def test_pad_ns_scalar_matches_jnp_pad(self, regular_grid):
        """pad_ns_scalar with inactive fold == jnp.pad zero."""
        interior = jax.random.normal(jax.random.PRNGKey(0), (35, 72))
        result = pad_ns_scalar(interior, regular_grid)
        expected = jnp.pad(interior, ((1, 1), (0, 0)))
        assert jnp.all(result == expected)

    def test_pad_ns_vector_u_matches_jnp_pad(self, regular_grid):
        """pad_ns_vector_u with inactive fold == jnp.pad zero."""
        interior = jax.random.normal(jax.random.PRNGKey(1), (35, 72))
        result = pad_ns_vector_u(interior, regular_grid)
        expected = jnp.pad(interior, ((1, 1), (0, 0)))
        assert jnp.all(result == expected)

    def test_pad_ns_scalar_3d_matches_jnp_pad(self, regular_grid):
        """pad_ns_scalar 3D with inactive fold == jnp.pad zero."""
        interior = jax.random.normal(jax.random.PRNGKey(2), (35, 72, 5))
        result = pad_ns_scalar(interior, regular_grid)
        expected = jnp.pad(interior, ((1, 1), (0, 0), (0, 0)))
        assert jnp.all(result == expected)


# =========================================================================
# Phase 3: Vector rotation in bipolar cap
# =========================================================================


class TestTripoleRestState:
    """Phase 4 gate: rest state preserved on tripolar grid."""

    def test_rest_state_single_step(self, tripole_grid):
        """One timestep from rest should stay at machine precision."""
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.state import LatLonCGridOceanConfig

        z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
        config = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="implicit_cn",
            A_h=1000.0,
            K_h=500.0,
            A_v=1e-3,
            K_v=1e-5,
            n_barotropic_substeps=10,
        )
        model = LatLonCGridOceanModel(tripole_grid, z_coord, config)
        state = rest_state_latlon_cgrid_ocean(
            tripole_grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
        )
        state_new = model.step(state, dt=600.0)

        eta = state_new.eta.data if hasattr(state_new.eta, 'data') else state_new.eta
        u = state_new.u.data if hasattr(state_new.u, 'data') else state_new.u
        v = state_new.v.data if hasattr(state_new.v, 'data') else state_new.v
        assert float(jnp.max(jnp.abs(eta))) < 1e-10, (
            f"eta drift: {float(jnp.max(jnp.abs(eta))):.2e}"
        )
        assert float(jnp.max(jnp.abs(u))) < 1e-10, (
            f"u drift: {float(jnp.max(jnp.abs(u))):.2e}"
        )
        assert float(jnp.max(jnp.abs(v))) < 1e-10, (
            f"v drift: {float(jnp.max(jnp.abs(v))):.2e}"
        )

    def test_wind_forced_stable(self, tripole_grid):
        """Model should remain stable and finite under wind forcing."""
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.core.field import Field

        z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
        config = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="implicit_cn",
            A_h=1000.0, K_h=500.0, A_v=1e-3, K_v=1e-5,
            n_barotropic_substeps=10,
        )
        model = LatLonCGridOceanModel(tripole_grid, z_coord, config)
        state = rest_state_latlon_cgrid_ocean(
            tripole_grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
        )
        # Apply a zonal velocity kick
        u_data = state.u.data
        cos_lat = tripole_grid.cos_lat
        u_kicked = u_data.at[:, :, 0].set(0.01 * cos_lat[:, jnp.newaxis])
        state = state._replace(u=Field(u_kicked))

        for _ in range(5):
            state = model.step(state, dt=600.0)

        eta = state.eta.data
        u = state.u.data
        v = state.v.data
        assert jnp.all(jnp.isfinite(eta)), "eta has non-finite values"
        assert jnp.all(jnp.isfinite(u)), "u has non-finite values"
        assert jnp.all(jnp.isfinite(v)), "v has non-finite values"

    def test_differentiability(self, tripole_grid):
        """jax.grad through a tripolar timestep should produce finite grads."""
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.state import LatLonCGridOceanConfig

        z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
        config = LatLonCGridOceanConfig.from_flat(
            barotropic_solver="implicit_cn",
            A_h=1000.0, K_h=500.0, A_v=1e-3, K_v=1e-5,
            n_barotropic_substeps=10,
        )
        model = LatLonCGridOceanModel(tripole_grid, z_coord, config)
        state = rest_state_latlon_cgrid_ocean(
            tripole_grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
        )

        def loss_fn(T_init):
            s = state._replace(T=T_init)
            s_new = model.step(s, dt=600.0)
            T_out = s_new.T.data if hasattr(s_new.T, "data") else s_new.T
            return jnp.sum(T_out ** 2)

        grad_T = jax.grad(loss_fn)(state.T)
        grad_arr = grad_T.data if hasattr(grad_T, "data") else grad_T
        assert jnp.all(jnp.isfinite(grad_arr)), (
            "Gradient through tripolar timestep has non-finite values"
        )


# =========================================================================
# Phase 3: Vector rotation in bipolar cap
# =========================================================================


def _make_rotated_tripole(n_lat=36, n_lon=72, max_angle_deg=30.0):
    """Create a synthetic tripolar grid with non-trivial rotation angles.

    In the bipolar cap, the rotation angle varies linearly with longitude
    from -max_angle_deg to +max_angle_deg.  This means the fold partner
    at ``perm[i] = n_lon - 1 - i`` has a DIFFERENT rotation angle than
    cell ``i``, mimicking the asymmetry of a real bipolar cap.
    """
    from legoesm.grids.tripole import create_synthetic_tripole
    geom = create_synthetic_tripole(n_lat, n_lon)

    cap_j = geom.fold.cap_j

    # Longitude-varying angle: -max at i=0, +max at i=n_lon-1
    # So cell i has angle α(i) and fold partner perm[i]=n_lon-1-i
    # has angle α(n_lon-1-i) = -α(i).  The difference Δα = 2α(i).
    i_frac = jnp.linspace(-1.0, 1.0, n_lon)
    alpha_1d = jnp.deg2rad(max_angle_deg) * i_frac  # (n_lon,)

    # Build 2D rotation angles for v-points (n_lat+1, n_lon)
    cos_alpha_v = jnp.ones((n_lat + 1, n_lon))
    sin_alpha_v = jnp.zeros((n_lat + 1, n_lon))
    mask_v = (jnp.arange(n_lat + 1) >= cap_j)[:, jnp.newaxis]
    cos_alpha_v = jnp.where(mask_v, jnp.cos(alpha_1d), cos_alpha_v)
    sin_alpha_v = jnp.where(mask_v, jnp.sin(alpha_1d), sin_alpha_v)

    # u-points (n_lat, n_lon+1) — use alpha at cell center
    cos_alpha_u = jnp.ones((n_lat, n_lon + 1))
    sin_alpha_u = jnp.zeros((n_lat, n_lon + 1))
    alpha_u = jnp.concatenate([alpha_1d, alpha_1d[0:1]])  # wrap
    mask_u = (jnp.arange(n_lat) >= cap_j)[:, jnp.newaxis]
    cos_alpha_u = jnp.where(mask_u, jnp.cos(alpha_u), cos_alpha_u)
    sin_alpha_u = jnp.where(mask_u, jnp.sin(alpha_u), sin_alpha_u)

    return geom._replace(
        cos_alpha_u=cos_alpha_u,
        sin_alpha_u=sin_alpha_u,
        cos_alpha_v=cos_alpha_v,
        sin_alpha_v=sin_alpha_v,
    )


class TestVectorRotation:
    """Test vector rotation in the bipolar cap (Phase 3)."""

    def test_pair_fold_zero_rotation_matches_individual(self, tripole_grid):
        """With zero rotation, pad_ns_vector_pair should match individual pads."""
        u_int = jax.random.normal(jax.random.PRNGKey(1), (35, 72))
        v_int = jax.random.normal(jax.random.PRNGKey(2), (35, 72))

        u_pair, v_pair = pad_ns_vector_pair(u_int, v_int, tripole_grid)
        u_ind = pad_ns_vector_u(u_int, tripole_grid)
        v_ind = pad_ns_vector_v(v_int, tripole_grid)

        assert jnp.allclose(u_pair, u_ind), "u mismatch between pair and individual"
        assert jnp.allclose(v_pair, v_ind), "v mismatch between pair and individual"

    def test_pair_fold_regular_latlon_is_zero(self, regular_grid):
        """On regular lat-lon, pad_ns_vector_pair gives zero at boundaries."""
        u_int = jax.random.normal(jax.random.PRNGKey(1), (35, 72))
        v_int = jax.random.normal(jax.random.PRNGKey(2), (35, 72))

        u_pad, v_pad = pad_ns_vector_pair(u_int, v_int, regular_grid)
        assert jnp.all(u_pad[0] == 0.0)
        assert jnp.all(u_pad[-1] == 0.0)
        assert jnp.all(v_pad[0] == 0.0)
        assert jnp.all(v_pad[-1] == 0.0)

    def test_rotated_fold_differs_from_unrotated(self):
        """With non-zero rotation, the fold ghost row should differ from
        the simple sign-flip result."""
        grid_rot = _make_rotated_tripole(max_angle_deg=30.0)
        grid_norot = create_synthetic_tripole(36, 72)

        u_int = jax.random.normal(jax.random.PRNGKey(1), (35, 72))
        v_int = jax.random.normal(jax.random.PRNGKey(2), (35, 72))

        u_rot, v_rot = pad_ns_vector_pair(u_int, v_int, grid_rot)
        u_norot, v_norot = pad_ns_vector_pair(u_int, v_int, grid_norot)

        # Interior should be identical (rotation only affects the fold row)
        assert jnp.all(u_rot[1:-1] == u_norot[1:-1])
        assert jnp.all(v_rot[1:-1] == v_norot[1:-1])

        # North fold row should differ (rotation mixes u and v)
        assert not jnp.allclose(u_rot[-1], u_norot[-1]), (
            "Expected rotated fold to differ from unrotated"
        )

    def test_rotation_orthogonality(self):
        """The combined fold+rotation transformation should be orthogonal:
        |u_d|^2 + |v_d|^2 = |u_s|^2 + |v_s|^2."""
        grid_rot = _make_rotated_tripole(max_angle_deg=45.0)

        u_int = jax.random.normal(jax.random.PRNGKey(1), (35, 72))
        v_int = jax.random.normal(jax.random.PRNGKey(2), (35, 72))

        u_pad, v_pad = pad_ns_vector_pair(u_int, v_int, grid_rot)

        # Energy at the last interior row (source)
        energy_src = u_int[-1]**2 + v_int[-1]**2
        # Energy at the north fold row (destination)
        # Note: the fold reverses i, so compare with i-reversed source
        perm = grid_rot.fold.perm_T
        energy_src_folded = u_int[-1, perm]**2 + v_int[-1, perm]**2
        energy_dst = u_pad[-1]**2 + v_pad[-1]**2

        assert jnp.allclose(energy_dst, energy_src_folded, atol=1e-6), (
            f"Rotation should preserve energy. Max diff: "
            f"{float(jnp.max(jnp.abs(energy_dst - energy_src_folded)))}"
        )

    def test_pair_fold_3d(self):
        """pad_ns_vector_pair should work on 3D fields."""
        grid = create_synthetic_tripole(36, 72)
        u_int = jax.random.normal(jax.random.PRNGKey(1), (35, 72, 5))
        v_int = jax.random.normal(jax.random.PRNGKey(2), (35, 72, 5))

        u_pad, v_pad = pad_ns_vector_pair(u_int, v_int, grid)
        assert u_pad.shape == (37, 72, 5)
        assert v_pad.shape == (37, 72, 5)
        assert jnp.all(jnp.isfinite(u_pad))
        assert jnp.all(jnp.isfinite(v_pad))
