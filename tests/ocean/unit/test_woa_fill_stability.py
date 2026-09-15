"""The per-level donor fill must never manufacture a static inversion.

A basin with no observation below some depth takes its deep donor from the
nearest cell observed at that depth, which can be another basin: on ORCA12
the Aegean below ~1000 m was filled with Black Sea water (22 psu under 39 psu,
a 6 kg/m3 inversion) and the run blew up from the first step.  The guard
replaces such a donor by the level above; observed levels are untouched.
"""

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import wright_eos
from legoesm.ocean.init_woa import (_fill_source_levels_nearest_valid,
                                    _reject_unstable_donors)

LAT = np.array([38.5, 39.5, 40.5, 41.5])
LON = np.array([24.5, 25.5, 26.5])
DEPTH = np.array([0.0, 500.0, 1000.0, 1200.0, 1400.0])


def _aegean_and_black_sea():
    """Aegean columns observed to 1000 m, one Black Sea column observed to
    1400 m; the 40.5 row never observed (the 1-deg land row)."""
    T = np.full((4, 3, 5), np.nan)
    S = np.full((4, 3, 5), np.nan)
    for j in (0, 1):
        T[j, :, :3] = [[15.0, 14.0, 13.2]] * 3
        S[j, :, :3] = [[38.3, 38.8, 38.8]] * 3
    T[3, 2, :] = [10.0, 8.9, 8.9, 8.9, 8.9]          # Black Sea (41.5, 26.5)
    S[3, 2, :] = [18.5, 22.0, 22.3, 22.3, 22.3]
    return T, S


def _rho_at(T, S, depth):
    return np.asarray(wright_eos(np.asarray(T), np.asarray(S),
                                 np.asarray(constants.rho_ocean * constants.g * depth)))


def test_lighter_cross_basin_donor_is_replaced_by_the_level_above():
    T0, S0 = _aegean_and_black_sea()
    filled = ~(np.isfinite(T0) & np.isfinite(S0))
    (T, S), _, _ = _fill_source_levels_nearest_valid((T0, S0), LAT, LON)
    # the defect: the Aegean row 39.5 took Black Sea water at 1200 m
    assert S[1, 2, 3] < 30.0
    assert _rho_at(T[1, 2, 3], S[1, 2, 3], 1200.0) < _rho_at(T[1, 2, 2], S[1, 2, 2], 1200.0) - 1.0
    Tg, Sg, n = _reject_unstable_donors(T, S, filled & np.isfinite(T), DEPTH)
    assert n > 0
    # observed levels untouched, everywhere
    np.testing.assert_array_equal(Tg[~filled], T0[~filled])
    np.testing.assert_array_equal(Sg[~filled], S0[~filled])
    # the Black Sea column keeps its own (observed) deep water
    assert Sg[3, 2, 4] == 22.3
    # every filled column is now statically stable at every filled level
    for k in range(1, DEPTH.size):
        cand = filled[..., k]
        r_here = _rho_at(Tg[..., k][cand], Sg[..., k][cand], DEPTH[k])
        r_above = _rho_at(Tg[..., k - 1][cand], Sg[..., k - 1][cand], DEPTH[k])
        assert np.all(r_here >= r_above - 0.01)
    # the Aegean deep water is now the extended 1000 m Aegean water, not Black Sea
    assert Sg[1, 2, 3] == Sg[1, 2, 2] == 38.8
    assert Sg[2, 1, 4] == 38.8


def test_denser_donor_is_kept():
    """The PR #1732 case: a shelf column over a deep model column takes cold
    deep water from the nearest deep cell -- denser, so it stays."""
    T = np.full((4, 3, 5), np.nan); S = np.full((4, 3, 5), np.nan)
    T[:, :, :2] = 20.0; S[:, :, :2] = 35.0                  # everyone observed to 500 m
    T[0, 0, :] = [20.0, 12.0, 5.0, 3.0, 2.0]; S[0, 0, :] = [35.0, 35.0, 34.8, 34.7, 34.7]
    filled = ~(np.isfinite(T) & np.isfinite(S))
    (Tf, Sf), _, _ = _fill_source_levels_nearest_valid((T, S), LAT, LON)
    Tg, Sg, n = _reject_unstable_donors(Tf, Sf, filled & np.isfinite(Tf), DEPTH)
    assert n == 0
    np.testing.assert_array_equal(Tg, Tf)
    np.testing.assert_array_equal(Sg, Sf)


def test_a_level_with_no_donor_anywhere_does_not_hide_the_inversion():
    """A source depth observed nowhere stays NaN (bridged later); the walk
    must compare the next filled level against the last FINITE level."""
    T0, S0 = _aegean_and_black_sea()
    T0[..., 3] = np.nan; S0[..., 3] = np.nan            # 1200 m observed nowhere
    filled = ~(np.isfinite(T0) & np.isfinite(S0))
    (T, S), _, _ = _fill_source_levels_nearest_valid((T0, S0), LAT, LON)
    assert np.all(np.isnan(T[..., 3]))
    Tg, Sg, n = _reject_unstable_donors(T, S, filled & np.isfinite(T), DEPTH)
    assert n > 0
    assert Sg[1, 2, 4] == 38.8                           # 1400 m compared against 1000 m


def test_guard_holds_for_c_contiguous_inputs():
    T0, S0 = _aegean_and_black_sea()
    filled = ~(np.isfinite(T0) & np.isfinite(S0))
    (T, S), _, _ = _fill_source_levels_nearest_valid((T0, S0), LAT, LON)
    Tg, Sg, n = _reject_unstable_donors(np.ascontiguousarray(T), np.ascontiguousarray(S),
                                        filled & np.isfinite(T), DEPTH)
    assert n > 0 and Sg[1, 2, 3] == 38.8


def test_production_ic_builder_applies_the_guard(tmp_path):
    """End-to-end through init_ocean_from_woa: a basin observed to 100 m next
    to a fresh basin observed to 1000 m must not get the fresh water at depth."""
    xr = pytest.importorskip("xarray")
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_woa import init_ocean_from_woa
    from legoesm.ocean.vertical import create_ocean_z_star

    src_lat = np.arange(-89.5, 90.0, 1.0)
    src_lon = np.arange(0.5, 360.0, 1.0)
    depths = np.array([0.0, 100.0, 1000.0])
    T = np.full((1, depths.size, src_lat.size, src_lon.size), 13.0)
    S = np.full_like(T, 38.8)
    # a land block (never observed) holding two basins so the nearest cell
    # observed at 1000 m for the "Aegean" is the fresh "Black Sea", not the
    # open ocean outside the block
    T[:, :, 126:134, 20:34] = np.nan; S[:, :, 126:134, 20:34] = np.nan
    # the "Aegean": lat 39.5-40.5, lon 24.5-26.5 observed only to 100 m
    T[:, :2, 129:131, 24:27] = 13.0; S[:, :2, 129:131, 24:27] = 38.8
    # the "Black Sea": same rows, lon 29.5-31.5, fresh, observed to 1000 m
    T[:, :, 129:131, 29:32] = 9.0; S[:, :, 129:131, 29:32] = 22.0
    ds = xr.Dataset({"t_an": (("time", "depth", "lat", "lon"), T),
                     "s_an": (("time", "depth", "lat", "lon"), S)},
                    coords={"time": [0.0], "depth": depths, "lat": src_lat, "lon": src_lon})
    path = tmp_path / "woa_basins.nc"
    ds.to_netcdf(path)
    grid = create_latlon_grid(180, 360)
    z = create_ocean_z_star(n_levels=8, H_max=1000.0)
    T_out, S_out = init_ocean_from_woa(grid, z, path, path)
    S_out = np.asarray(S_out)
    # deepest model level of the Aegean cells: extended 100 m water, not 22 psu
    assert np.all(S_out[129:131, 25:26, -1] > 38.0)
