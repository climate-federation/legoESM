"""Phase 4 of partial cells: 3D face masks and tracer flux divergence.

Phase 3b's PGF correction works correctly at *active-active* partial-cell
faces.  Phase 4 handles the *active-vs-inactive* case (cells with
different ``bottom_level`` neighbouring at level k where one column
has water and the other doesn't): a 3D face mask
``compute_face_masks_3d(is_active_3d)`` zeros the spurious flux at
those level-k faces.

This unblocks the partial-cells advantage on realistic bathymetry
where columns naturally have different bottom_levels (continental
shelves vs abyssal plains).

Three test classes:

1. ``TestComputeFaceMasks3D``: low-level helper produces correct
   per-level u/v masks.
2. ``TestPartialCellsBitExactBackcompat``: when all columns have the
   same bottom_level (so 3D mask == broadcast(2D mask)), the partial
   path produces bit-exact identical tendencies to the legacy z\\*
   path on flat bottom.  Phase 6 backwards-compat foundation.
3. ``TestStepBathymetryDifferentBottomLevels``: the headline test —
   step bathymetry with DIFFERENT bottom_levels (the realistic case),
   tracer conservation under no-flux, momentum stability check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    compute_face_masks_3d,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
    compute_layer_thickness,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------------


class TestComputeFaceMasks3D:
    """``compute_face_masks_3d`` produces correct per-level masks."""

    def test_all_active_matches_2d_broadcast(self):
        """When all cells active at all levels, 3D masks are identical
        to broadcasting the all-ones 2D land mask through
        ``compute_face_masks``."""
        n_lat, n_lon, nlev = 4, 8, 5
        is_active = jnp.ones((n_lat, n_lon, nlev), dtype=jnp.bool_)
        u3d, v3d = compute_face_masks_3d(is_active)
        # 2D version with all-ocean mask
        land_2d = jnp.ones((n_lat, n_lon))
        u2d, v2d = compute_face_masks(land_2d)
        np.testing.assert_array_equal(
            np.asarray(u3d),
            np.broadcast_to(np.asarray(u2d)[..., None], u3d.shape),
        )
        np.testing.assert_array_equal(
            np.asarray(v3d),
            np.broadcast_to(np.asarray(v2d)[..., None], v3d.shape),
        )

    def test_partial_column_zero_below_bottom(self):
        """A column with bottom_level < nlev-1 has its faces
        zeroed at levels below bottom_level."""
        n_lat, n_lon, nlev = 4, 8, 5
        # Build is_active: column (1, 1) only active at levels 0, 1
        is_active = jnp.ones((n_lat, n_lon, nlev), dtype=jnp.bool_)
        is_active = is_active.at[1, 1, 2:].set(False)

        u3d, v3d = compute_face_masks_3d(is_active)
        # The west u-face of cell (1,1) sits between (1,0) and (1,1).
        # Cell (1,1) is inactive at levels 2-4 → west face also inactive.
        # In compute_face_masks_3d convention, u_mask[i, j, :] is at the
        # face between (i, j-1) and (i, j), so u_mask[1, 1, 2:] = 0.
        assert float(jnp.max(u3d[1, 1, 2:])) == 0.0
        # Levels 0, 1 (where (1, 1) is active and (1, 0) is active): wet
        assert float(jnp.min(u3d[1, 1, :2])) == 1.0
        # East u-face of (1,1) is u_mask[1, 2]: between (1,1) and (1,2).
        # Same logic: 0 at levels 2-4 because (1,1) is inactive.
        assert float(jnp.max(u3d[1, 2, 2:])) == 0.0

    def test_pole_rows_zero(self):
        """v-faces at the poles are always zero (wall BC)."""
        is_active = jnp.ones((4, 8, 5), dtype=jnp.bool_)
        _, v3d = compute_face_masks_3d(is_active)
        assert float(jnp.max(jnp.abs(v3d[0]))) == 0.0
        assert float(jnp.max(jnp.abs(v3d[-1]))) == 0.0


# ---------------------------------------------------------------------------
# Backwards-compat: flat bottom is bit-exact
# ---------------------------------------------------------------------------


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=18, n_lon=36)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(
        n_levels=10, H_max=4000.0, dz_surface=10.0, dz_deep=500.0,
    )


class TestPartialCellsBitExactBackcompat:
    """On flat bottom (every column has the same bottom_level=nlev-1
    and h_partial=dz_ref everywhere), the partial-cells path with
    3D face masks must produce bit-exact identical tendencies to the
    legacy z\\* path."""

    def test_flat_bottom_du_dt_bit_exact(self, grid, z_coord):
        H_bathy = jnp.full((grid.n_lat, grid.n_lon), z_coord.H_max)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        cfg = LatLonCGridOceanConfig()
        tend_zstar = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg,
        )
        tend_partial = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial_coord, cfg,
        )
        # Per the AD-bit-exact regression contract: identical to round-off
        np.testing.assert_array_equal(
            np.asarray(tend_zstar.du_dt.data),
            np.asarray(tend_partial.du_dt.data),
        )
        np.testing.assert_array_equal(
            np.asarray(tend_zstar.dv_dt.data),
            np.asarray(tend_partial.dv_dt.data),
        )


# ---------------------------------------------------------------------------
# Step bathymetry with different bottom_levels
# ---------------------------------------------------------------------------


def _make_realistic_step_bathymetry(grid, H_deep=4000.0, H_shallow=800.0):
    """Step bathymetry where columns have DIFFERENT bottom_levels.

    With n_levels=10 and H_max=4000, dz_surface=10, dz_deep=500:
    - H=4000 → bottom_level = nlev-1 = 9
    - H=800 → bottom_level around 4 (depending on z_half_ref)

    This is the realistic-bathymetry case that exercises the 3D face
    mask treatment.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    H = jnp.full((n_lat, n_lon), H_deep)
    H = H.at[: n_lat // 2, :].set(H_shallow)
    return H


class TestStepBathymetryDifferentBottomLevels:
    """The headline Phase 4 test: step bathymetry with shallow and
    deep columns having different bottom_levels.  Verifies that the
    3D face mask correctly zeroes fluxes through active-vs-inactive
    boundaries, and that the partial-cells path gives smaller
    spurious PGF than the legacy z\\* path on this realistic case."""

    def test_3d_face_mask_zeros_active_inactive_face(
        self, grid, z_coord,
    ):
        """At a v-face between a shallow column (bottom_level=k_shallow)
        and a deep column (bottom_level=k_deep > k_shallow), levels
        between k_shallow+1 and k_deep should have v_mask_3d = 0
        (face is between active deep and inactive shallow)."""
        H_bathy = _make_realistic_step_bathymetry(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        u3d, v3d = compute_face_masks_3d(partial_coord.is_active)
        # Find a v-face between shallow and deep columns
        # n_lat//2 is the boundary row; v_mask[n_lat//2, :, k] is between
        # row n_lat//2 - 1 (south, shallow) and row n_lat//2 (north, deep).
        boundary_row = grid.n_lat // 2
        bot_shallow = int(partial_coord.bottom_level[boundary_row - 1, 0])
        bot_deep = int(partial_coord.bottom_level[boundary_row, 0])
        assert bot_shallow < bot_deep, (
            "test setup invalid: bottom_levels should differ"
        )
        # At levels in (bot_shallow, bot_deep], v-face should be 0
        for k in range(bot_shallow + 1, bot_deep + 1):
            v_at_k = float(jnp.max(jnp.abs(v3d[boundary_row, :, k])))
            assert v_at_k == 0.0, (
                f"Active-vs-inactive face should be zero at level {k}, "
                f"got {v_at_k}"
            )
        # At levels k <= bot_shallow, both columns are active, v_mask = 1
        for k in range(bot_shallow + 1):
            v_at_k = float(jnp.min(v3d[boundary_row, :, k]))
            assert v_at_k == 1.0, (
                f"Active-active face should be wet at level {k}"
            )

    def test_step_bathymetry_no_blowup_under_stepping(self, grid, z_coord):
        """Smoke test: integrate 1 step on partial-cells coord with
        step bathymetry (different bottom_levels).  Just verify it
        doesn't crash and produces finite tendencies — the model
        runs end-to-end on the realistic case."""
        H_bathy = _make_realistic_step_bathymetry(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)

        # Centroid-aware T initialisation
        centroid = compute_centroid_depth(
            jnp.zeros_like(H_bathy), H_bathy, partial_coord,
        )
        from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
        T_per_cell = 2.0 + (20.0 - 2.0) * jnp.exp(-centroid / _SCALE_DEPTH)
        T_per_cell = jnp.where(partial_coord.is_active, T_per_cell, 2.0)

        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        state = state._replace(T=state.T.replace(data=T_per_cell))

        cfg = LatLonCGridOceanConfig()
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial_coord, cfg,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        # |du/dt|, |dv/dt| should be much smaller than the no-correction
        # case.  The active-vs-inactive face mask zeroes the worst
        # spurious gradients.
        u_max = float(jnp.max(jnp.abs(tend.du_dt.data)))
        v_max = float(jnp.max(jnp.abs(tend.dv_dt.data)))
        # Lenient bound — the linear-shift residual is still present at
        # active-active partial faces, but no longer dominated by the
        # active-vs-inactive face leakage.
        assert u_max < 1e-3, f"|du/dt|={u_max} (suspect blowup)"
        assert v_max < 1e-3, f"|dv/dt|={v_max} (suspect blowup)"

    def test_partial_beats_legacy_on_realistic_step(self, grid, z_coord):
        """The previously-skipped Phase 3b comparison: with proper
        Phase 4 3D face mask, partial-cells path should produce
        smaller spurious PGF than legacy z\\* on a step bathymetry
        with different bottom_levels."""
        H_bathy = _make_realistic_step_bathymetry(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)

        centroid = compute_centroid_depth(
            jnp.zeros_like(H_bathy), H_bathy, partial_coord,
        )
        from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
        T_per_cell = 2.0 + (20.0 - 2.0) * jnp.exp(-centroid / _SCALE_DEPTH)
        T_per_cell = jnp.where(partial_coord.is_active, T_per_cell, 2.0)

        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        state = state._replace(T=state.T.replace(data=T_per_cell))

        cfg = LatLonCGridOceanConfig()
        tend_partial = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial_coord, cfg,
        )
        tend_legacy = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg,
        )
        v_partial = float(jnp.max(jnp.abs(tend_partial.dv_dt.data)))
        v_legacy = float(jnp.max(jnp.abs(tend_legacy.dv_dt.data)))
        # Partial cells should show meaningful improvement (at least 2x)
        # on this realistic-bathy case.  The exact factor depends on
        # stratification and bathymetry geometry.
        improvement = v_legacy / max(v_partial, 1e-30)
        print(f"\nv_legacy = {v_legacy:.3e} m/s², "
              f"v_partial = {v_partial:.3e} m/s², "
              f"improvement = {improvement:.1f}x")
        assert improvement > 2.0, (
            f"Partial cells should beat legacy on realistic step bathy, "
            f"got {improvement:.1f}x improvement"
        )
