"""Phase 0 of partial cells: data structure + factory tests.

Validates ``OceanPartialCellCoordinate`` and
``create_partial_cell_coordinate``:

  - Construction edge cases (full cells, partial cells, dry columns,
    over-deep columns)
  - Per-column thickness identity ``sum_k h_partial = H_bathy``
  - Bit-exact agreement with the legacy z* path on a flat-bottom
    field (sanity check that a uniform H_bathy = H_max produces all
    full cells)
  - Vectorisation over arbitrary column-shape

Differentiability tests live in a later phase; here we validate the
static structure only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    OceanZStarCoordinate,
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture
def z5():
    """5-level reference coordinate with H_max=4000m.

    Constructs interfaces at depths approximately
    [0, 10, 132.5, 255, 377.5, 500] but normalised so the column
    sums to H_max.  Use the actual values from create_ocean_z_star
    rather than hard-coding."""
    return create_ocean_z_star(
        n_levels=5, H_max=4000.0, dz_surface=10.0, dz_deep=500.0,
    )


@pytest.fixture
def z20():
    """20-level production-style coordinate."""
    return create_ocean_z_star(
        n_levels=20, H_max=4000.0, dz_surface=10.0, dz_deep=500.0,
    )


# ---------------------------------------------------------------------------
# Single-column edge cases
# ---------------------------------------------------------------------------


class TestSingleColumnEdgeCases:
    """Construct partial-cell coords for a single column with various
    H_bathy values to verify the index logic + thickness arithmetic.
    """

    def test_full_depth_all_full_cells(self, z5):
        """H_bathy = H_max: bottom_level = nlev-1; all cells full."""
        H = jnp.asarray([z5.H_max])
        coord = create_partial_cell_coordinate(z5, H)
        # All levels active
        assert coord.bottom_level.shape == (1,)
        assert int(coord.bottom_level[0]) == z5.n_levels - 1
        assert bool(jnp.all(coord.is_active))
        # Every layer thickness equals dz_ref
        np.testing.assert_allclose(
            np.asarray(coord.h_partial[0]), np.asarray(z5.dz_ref),
            rtol=1e-12, atol=0.0,
        )

    def test_dry_column(self, z5):
        """H_bathy = 0: bottom_level = -1; nothing active; all
        thicknesses zero."""
        H = jnp.asarray([0.0])
        coord = create_partial_cell_coordinate(z5, H)
        assert int(coord.bottom_level[0]) == -1
        assert not bool(jnp.any(coord.is_active))
        np.testing.assert_array_equal(
            np.asarray(coord.h_partial[0]), np.zeros(z5.n_levels),
        )

    def test_negative_H_treated_as_dry(self, z5):
        """H_bathy < 0 (e.g. ETOPO land elevations) is also dry."""
        H = jnp.asarray([-100.0])
        coord = create_partial_cell_coordinate(z5, H)
        assert int(coord.bottom_level[0]) == -1
        assert not bool(jnp.any(coord.is_active))

    def test_shallow_column_partial_top(self, z5):
        """H_bathy < dz_ref[0]: only the topmost cell is partial,
        rest are below seafloor."""
        # dz_ref[0] is normalized; pick H_bathy comfortably less than it.
        H_target = float(z5.dz_ref[0]) * 0.5
        H = jnp.asarray([H_target])
        coord = create_partial_cell_coordinate(z5, H)
        # bottom_level should be 0
        assert int(coord.bottom_level[0]) == 0
        # Only level 0 is active
        assert bool(coord.is_active[0, 0])
        assert not bool(jnp.any(coord.is_active[0, 1:]))
        # Thickness of level 0 = H_target
        assert abs(float(coord.h_partial[0, 0]) - H_target) < 1e-10
        # Other levels = 0
        np.testing.assert_array_equal(
            np.asarray(coord.h_partial[0, 1:]),
            np.zeros(z5.n_levels - 1),
        )

    def test_arbitrary_column_thickness_sums_to_H(self, z5):
        """Column thickness identity: sum h_partial = H_bathy when
        H <= H_max; sum h_partial = H_max when H > H_max (cap)."""
        rng = np.random.default_rng(seed=7)
        # Float32 precision in abs_z_half means the identity holds to
        # ~1e-7 relative, not to 1e-9.
        for _ in range(20):
            H_test = float(rng.uniform(50.0, z5.H_max - 50.0))
            coord = create_partial_cell_coordinate(
                z5, jnp.asarray([H_test]),
            )
            col_sum = float(jnp.sum(coord.h_partial[0]))
            assert abs(col_sum - H_test) / max(H_test, 1.0) < 1e-6, (
                f"H_test={H_test}, col_sum={col_sum}"
            )

    def test_H_exceeds_Hmax_caps_at_Hmax(self, z5):
        """When H_bathy > H_max, the column is at full reference depth
        with all cells full.  Per-cell thickness equals dz_ref exactly;
        column sum equals H_max within float32 precision (the cumsum
        ``sum(dz_ref)`` and the value the user passed for H_max may
        differ by ~1e-7 relative)."""
        H = jnp.asarray([z5.H_max + 500.0])
        coord = create_partial_cell_coordinate(z5, H)
        # Per-cell match to dz_ref bit-exact
        np.testing.assert_allclose(
            np.asarray(coord.h_partial[0]), np.asarray(z5.dz_ref),
            rtol=1e-12,
        )
        # Column sum within float32 precision of H_max
        col_sum = float(jnp.sum(coord.h_partial[0]))
        assert abs(col_sum - z5.H_max) / z5.H_max < 1e-6

    def test_partial_cell_at_step(self, z5):
        """For H_bathy at a known interface depth, the partial cell
        thickness is exactly the thickness up to that interface."""
        # |z_half_ref| includes 0 (surface) + nlev other interfaces.
        # Pick the depth of interface at index 3 → bottom_level should be 2
        # (the layer above that interface) with a full thickness, and
        # the next layer should be inactive — wait, no, that's wrong.
        # If H = |z_half_ref[k]|, then the seafloor is exactly at the upper
        # interface of level k.  The deepest *active* level is k-1, and
        # it's a full cell.  Level k is fully below the seafloor.
        target_depth = float(jnp.abs(z5.z_half_ref[3]))   # |z_half_ref[3]|
        H = jnp.asarray([target_depth])
        coord = create_partial_cell_coordinate(z5, H)
        # Number of interfaces strictly < H: indices 0, 1, 2 → count=3 → bottom=2
        assert int(coord.bottom_level[0]) == 2
        # Level 2 is fully active (full cell)
        np.testing.assert_allclose(
            float(coord.h_partial[0, 2]),
            float(z5.dz_ref[2]),
            rtol=1e-12,
        )
        # Levels 3, 4 inactive
        assert not bool(coord.is_active[0, 3])
        assert not bool(coord.is_active[0, 4])


# ---------------------------------------------------------------------------
# Vectorised over multiple columns
# ---------------------------------------------------------------------------


class TestMultiColumn:
    """Verify the factory works on 2D and 3D bathymetry shapes."""

    def test_2d_shape_preserved(self, z5):
        """(n_lat, n_lon) H_bathy → h_partial shape (n_lat, n_lon, nlev)."""
        n_lat, n_lon = 4, 6
        rng = np.random.default_rng(seed=1)
        # Stay below H_max to avoid the cap (tested separately).
        H = jnp.asarray(rng.uniform(100.0, z5.H_max - 50.0,
                                       size=(n_lat, n_lon)))
        coord = create_partial_cell_coordinate(z5, H)
        assert coord.h_partial.shape == (n_lat, n_lon, z5.n_levels)
        assert coord.bottom_level.shape == (n_lat, n_lon)
        assert coord.is_active.shape == (n_lat, n_lon, z5.n_levels)
        # Column-thickness identity per cell (float32 precision in
        # abs_z_half permits ~1e-6 relative).
        col_sum = np.asarray(jnp.sum(coord.h_partial, axis=-1))
        np.testing.assert_allclose(col_sum, np.asarray(H), rtol=1e-6)

    def test_step_bathymetry_two_levels(self, z5):
        """Two-region bathymetry: half deep, half shallow.  Verify both
        regions get the right bottom_level."""
        n_lat, n_lon = 4, 6
        H_deep = float(z5.H_max)
        H_shallow = 100.0
        H = jnp.full((n_lat, n_lon), H_deep)
        H = H.at[:, :3].set(H_shallow)
        coord = create_partial_cell_coordinate(z5, H)
        # Deep half: bottom_level = nlev-1
        assert bool(jnp.all(coord.bottom_level[:, 3:] == z5.n_levels - 1))
        # Shallow half: bottom_level depends on where 100 m falls in the
        # reference grid.  For this z5, dz_ref[0] is small (~31 m for
        # H_max=4000, n=5, dz_surf=10), so bottom_level is 1 or 2.
        shallow_bot = int(coord.bottom_level[0, 0])
        assert 0 <= shallow_bot < z5.n_levels - 1
        # Shallow column thickness sums to H_shallow
        col_sum_shallow = float(jnp.sum(coord.h_partial[0, 0]))
        assert abs(col_sum_shallow - H_shallow) < 1e-9

    def test_thickness_zero_below_bottom(self, z5):
        """Cells below bottom_level have h_partial = 0 exactly."""
        n_lat, n_lon = 3, 5
        H = jnp.asarray(np.linspace(50.0, z5.H_max - 100.0, n_lat * n_lon)
                        .reshape(n_lat, n_lon))
        coord = create_partial_cell_coordinate(z5, H)
        # For each column, every k > bottom_level should have h_partial = 0
        for i in range(n_lat):
            for j in range(n_lon):
                k_bot = int(coord.bottom_level[i, j])
                if k_bot + 1 < z5.n_levels:
                    assert float(jnp.max(jnp.abs(
                        coord.h_partial[i, j, k_bot + 1:]
                    ))) == 0.0


# ---------------------------------------------------------------------------
# Backwards-compatibility: matches z* on flat bottom
# ---------------------------------------------------------------------------


class TestFlatBottomEquivalence:
    """When H_bathy = H_max everywhere, partial-cell coord should be
    indistinguishable from z* (all full cells)."""

    def test_flat_bottom_h_partial_equals_dz_ref(self, z20):
        n_lat, n_lon = 36, 72
        H = jnp.full((n_lat, n_lon), z20.H_max)
        coord = create_partial_cell_coordinate(z20, H)
        # Per cell, h_partial[..., k] = dz_ref[k]
        expected = np.broadcast_to(
            np.asarray(z20.dz_ref), (n_lat, n_lon, z20.n_levels),
        )
        np.testing.assert_allclose(
            np.asarray(coord.h_partial), expected,
            rtol=1e-12, atol=0.0,
        )

    def test_flat_bottom_all_active(self, z20):
        H = jnp.full((36, 72), z20.H_max)
        coord = create_partial_cell_coordinate(z20, H)
        assert bool(jnp.all(coord.is_active))
        assert bool(jnp.all(coord.bottom_level == z20.n_levels - 1))

    def test_flat_bottom_reference_fields_preserved(self, z20):
        """The reference z* fields should pass through unchanged."""
        H = jnp.full((4, 6), z20.H_max)
        coord = create_partial_cell_coordinate(z20, H)
        # Reference fields are inherited verbatim from z_coord
        np.testing.assert_array_equal(
            np.asarray(coord.z_full_ref), np.asarray(z20.z_full_ref),
        )
        np.testing.assert_array_equal(
            np.asarray(coord.z_half_ref), np.asarray(z20.z_half_ref),
        )
        np.testing.assert_array_equal(
            np.asarray(coord.dz_ref), np.asarray(z20.dz_ref),
        )
        np.testing.assert_array_equal(
            np.asarray(coord.dz_half_ref), np.asarray(z20.dz_half_ref),
        )
        assert coord.n_levels == z20.n_levels
        assert coord.H_max == z20.H_max


# ---------------------------------------------------------------------------
# Realistic bathymetry pattern (smoothly-varying)
# ---------------------------------------------------------------------------


class TestRealisticPattern:
    """Smoothly-varying bathymetry produces gradually-varying
    bottom_level — no spurious jumps."""

    def test_smooth_bathy_smooth_bottom_level(self, z20):
        """A linear ramp from 1000m to 4000m should produce a
        monotonically non-decreasing bottom_level along the ramp."""
        n_cells = 30
        H = jnp.asarray(np.linspace(1000.0, 4000.0, n_cells))
        coord = create_partial_cell_coordinate(z20, H)
        bot = np.asarray(coord.bottom_level)
        # bottom_level must be non-decreasing as H increases
        diffs = np.diff(bot)
        assert np.all(diffs >= 0), (
            f"bottom_level not monotone: {bot}"
        )

    def test_cosine_bathy_thickness_identity(self, z20):
        """Cosine pattern in latitude — verify thickness identity per cell.

        Tolerance ~1e-6 governed by abs_z_half float32 precision (see
        Phase 1 plan note).
        """
        n_lat, n_lon = 18, 36
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        H_min, H_max_use = 500.0, 3500.0
        H_lat = H_min + (H_max_use - H_min) * np.cos(lat) ** 2
        H = jnp.asarray(np.broadcast_to(H_lat[:, None],
                                          (n_lat, n_lon)))
        coord = create_partial_cell_coordinate(z20, H)
        col_sum = np.asarray(jnp.sum(coord.h_partial, axis=-1))
        np.testing.assert_allclose(col_sum, np.asarray(H), rtol=1e-6)
