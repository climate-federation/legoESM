"""Per-level nearest-valid fill of the observed T/S source grid.

An observed column stops at the local seafloor, so a model column standing
over a shallow source cell has no observation below that cell's floor.  The
column fallback in ``_interp_profile_to_z_coord`` then propagated the
shallowest valid value down the whole column, writing surface water into the
abyss.  ``_fill_source_levels_nearest_valid`` removes the cause: every source
cell missing an observation at a depth takes it from the nearest source cell
that HAS one at that same depth.

Each test that asserts the fix also asserts what happens WITHOUT it, so none
of them can pass against the unfixed code.
"""
import numpy as np
import pytest

from legoesm.grids.regridding import fill_missing_nearest_valid
from legoesm.ocean.init_woa import (
    _fill_source_levels_nearest_valid,
    _interp_profile_to_z_coord,
)
from legoesm.ocean.vertical import create_ocean_z_star

# A 1 x 3 source grid: three cells on the equator, 1 degree apart.
SRC_LAT = np.array([0.0])
SRC_LON = np.array([0.0, 1.0, 2.0])
N_LEV_SRC = 6


def _source(profiles):
    """(1, 3, 6) source field from three per-column profiles."""
    return np.asarray(profiles, dtype=np.float64).reshape(1, 3, N_LEV_SRC)


def test_deep_levels_come_from_the_nearest_column_that_has_them():
    # Column 0 is a shelf: it observes only the surface.  Columns 1 and 2 are
    # open ocean with the full profile; column 1 is the nearer of the two.
    shelf = [20.0] + [np.nan] * 5
    near = [18.0, 12.0, 8.0, 5.0, 3.0, 2.0]
    far = [17.0, 11.0, 7.0, 4.0, 2.5, 1.5]
    (filled,), n, _ = _fill_source_levels_nearest_valid(
        [_source([shelf, near, far])], SRC_LAT, SRC_LON)

    assert n == 5
    # The shelf keeps its own surface observation ...
    assert filled[0, 0, 0] == 20.0
    # ... and takes every deeper level from the NEAREST column that has one.
    np.testing.assert_allclose(filled[0, 0, 1:], near[1:])
    # Without the fill the shelf column carries 20 degC to the seafloor.
    assert np.all(np.isnan(_source([shelf, near, far])[0, 0, 1:]))
    z = create_ocean_z_star(n_levels=4, H_max=2500.0)
    unfixed = _interp_profile_to_z_coord(
        np.asarray(shelf), np.arange(N_LEV_SRC) * 500.0, z)
    np.testing.assert_allclose(unfixed, 20.0)


def test_the_donor_changes_with_depth_not_once_per_column():
    # The near column observes only the top half; the far column observes
    # everything.  A per-COLUMN fill would take all six levels from one donor.
    gap = [np.nan] * N_LEV_SRC
    near = [18.0, 12.0, 8.0] + [np.nan] * 3
    far = [17.0, 11.0, 7.0, 4.0, 2.5, 1.5]
    (filled,), _, _ = _fill_source_levels_nearest_valid(
        [_source([gap, near, far])], SRC_LAT, SRC_LON)

    np.testing.assert_allclose(filled[0, 0, :3], near[:3])   # near wins
    np.testing.assert_allclose(filled[0, 0, 3:], far[3:])    # near has none
    # The two donors really are different columns, i.e. this is not a
    # per-column copy that happens to agree.
    assert not np.allclose(near[:3], far[:3])


def test_a_level_observed_nowhere_is_left_for_the_callers_fill():
    # The deepest level has no observation on the whole source grid: there is
    # no donor, so it must stay missing rather than invent one.
    a = [18.0, 12.0, 8.0, 5.0, 3.0, np.nan]
    b = [17.0, 11.0, 7.0, 4.0, 2.5, np.nan]
    c = [16.0, 10.0, 6.0, 3.5, 2.0, np.nan]
    (filled,), n, _ = _fill_source_levels_nearest_valid(
        [_source([a, b, c])], SRC_LAT, SRC_LON)
    assert n == 0
    assert np.all(np.isnan(filled[..., -1]))
    np.testing.assert_allclose(filled[..., :-1], _source([a, b, c])[..., :-1])


def test_a_complete_source_field_is_returned_unchanged():
    full = _source([[18.0, 12.0, 8.0, 5.0, 3.0, 2.0]] * 3)
    (filled,), n, _ = _fill_source_levels_nearest_valid([full], SRC_LAT, SRC_LON)
    assert n == 0
    np.testing.assert_array_equal(filled, full)


def test_nearest_is_measured_across_the_dateline():
    # Two cells either side of longitude 0: the nearest neighbour of 359.5 is
    # 0.5 (1 degree away), not 180 (a whole hemisphere away).  Differencing
    # degrees would pick the far one.
    lon = np.array([0.5, 180.0, 359.5])
    src = _source([[10.0] * N_LEV_SRC,
                   [-5.0] * N_LEV_SRC,
                   [np.nan] * N_LEV_SRC])
    (filled,), n, _ = _fill_source_levels_nearest_valid([src], SRC_LAT, lon)
    assert n == N_LEV_SRC
    np.testing.assert_allclose(filled[0, 2, :], 10.0)


def test_the_shared_helper_fills_each_slice_from_its_own_donors():
    # The property the per-level fill rests on: row 1's missing entry may not
    # borrow row 0's value at the same point.
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    data = np.array([[1.0, np.nan, 3.0],
                     [10.0, np.nan, 30.0]])
    out = fill_missing_nearest_valid(data, coords)
    assert out[0, 1] in (1.0, 3.0)
    assert out[1, 1] in (10.0, 30.0)
    assert out[1, 1] == 10.0 * out[0, 1]


def test_the_shared_helper_refuses_a_slice_with_no_donor():
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    data = np.array([[1.0, 2.0], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="entirely NaN"):
        fill_missing_nearest_valid(data, coords)


def test_each_source_cell_keeps_its_own_coordinate():
    # The single most dangerous line in the fill is the pairing of flattened
    # source cells with flattened coordinates: transposing one of the two
    # would hand every cell a stranger's position and still produce a finite,
    # entirely plausible field.  A 2 x 3 grid distinguishes the two orderings
    # (a square one would not), and every cell carries a unique value, so the
    # donor is identifiable.
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 90.0, 180.0])
    vals = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])
    src = np.repeat(vals[:, :, None], N_LEV_SRC, axis=2)
    src[1, 2, :] = np.nan                       # (1 N, 180 E) is missing
    (filled,), n, _ = _fill_source_levels_nearest_valid([src], lat, lon)

    assert n == N_LEV_SRC
    # Its true nearest neighbour is 1 degree away in latitude, at (0 N, 180 E),
    # value 3.0.  The two cells on its own parallel are 90 and 180 degrees
    # away.  Any scrambling of the cell/coordinate pairing picks a different
    # donor, because every cell carries a different value.
    np.testing.assert_allclose(filled[1, 2, :], 3.0)
    # Everything else is untouched.
    np.testing.assert_array_equal(filled[0], src[0])
    np.testing.assert_allclose(filled[1, 0, :], 4.0)
    np.testing.assert_allclose(filled[1, 1, :], 5.0)


def test_the_flatten_transpose_round_trip_moves_no_value():
    # Same round trip with nothing missing but one entry, on a grid whose
    # three dimensions all differ, so a wrong reshape cannot pass by symmetry.
    rng = np.random.default_rng(0)
    src = rng.normal(size=(4, 7, 5))
    ref = src.copy()
    src[2, 5, 3] = np.nan
    lat = np.linspace(-60.0, 60.0, 4)
    lon = np.linspace(0.0, 300.0, 7)
    (filled,), n, _ = _fill_source_levels_nearest_valid([src], lat, lon)
    assert n == 1
    ref[2, 5, 3] = filled[2, 5, 3]
    np.testing.assert_array_equal(filled, ref)


def test_temperature_and_salinity_come_from_the_SAME_donor_column():
    # A cell must take every field from ONE column.  A per-field search could
    # pair one column's temperature with another's salinity -- a density
    # anomaly built out of two real observations.
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 90.0, 180.0])
    T = np.repeat(np.array([[10.0, 20.0, 30.0],
                            [11.0, 21.0, 31.0]])[:, :, None], N_LEV_SRC, axis=2)
    S = np.repeat(np.array([[34.0, 35.0, 36.0],
                            [34.5, 35.5, 36.5]])[:, :, None], N_LEV_SRC, axis=2)
    # Salinity alone is missing at (1 N, 180 E); temperature there is present.
    S[1, 2, :] = np.nan
    (Tf, Sf), n, _ = _fill_source_levels_nearest_valid([T, S], lat, lon)

    # The cell counts as missing because ONE field is, so BOTH are replaced
    # from the same donor: (0 N, 180 E), one degree away.
    assert n == N_LEV_SRC
    np.testing.assert_allclose(Sf[1, 2, :], 36.0)
    np.testing.assert_allclose(Tf[1, 2, :], 30.0)
    # Its own temperature, 31.0, is NOT kept next to a stranger's salinity.
    assert not np.any(np.isclose(Tf[1, 2, :], 31.0))
