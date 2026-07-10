"""Tests for the LUH2 land-use source (reader + 12-state -> 17-PFT crosswalk)."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.surface_data.sources.luh2 import (
    LUH2Config,
    LUH2_STATE_NAMES,
    build_transient_pft_frac,
    c4_grass_fraction_from_base,
    luh2_states_to_pft_frac,
    read_luh2_states,
)

_IDX = {name: i for i, name in enumerate(CLM5_PFT_NAMES)}


def _uniform_states(nyear=2, ny=3, nx=4, value=0.05):
    """Every LUH2 state set to a constant fraction (sums to 12*value <= 1)."""
    return {s: np.full((nyear, ny, nx), value) for s in LUH2_STATE_NAMES}


def _forest_pnv(ny=3, nx=4):
    """A PNV shape that is pure broadleaf-evergreen-tropical (index 4)."""
    pnv = np.zeros((N_PFT_CLM5, ny, nx))
    pnv[_IDX["broadleaf_evergreen_tropical"]] = 1.0
    return pnv


def test_crosswalk_conserves_area():
    """sum over 17 PFTs == sum over 12 states, per cell and year (redistribution)."""
    states = _uniform_states()
    pnv = _forest_pnv()
    c4f = np.full((3, 4), 0.5)
    out = luh2_states_to_pft_frac(states, pnv, c4f)

    assert out.shape == (2, N_PFT_CLM5, 3, 4)
    state_total = sum(states[s] for s in LUH2_STATE_NAMES)     # (nyear, ny, nx)
    np.testing.assert_allclose(out.sum(axis=1), state_total, rtol=0, atol=1e-12)
    assert np.all(out >= 0.0)


def test_c3_c4_crop_routing():
    """C3 crop states -> crop_c3; C4 crop states -> crop_c4; nothing leaks."""
    states = {s: np.zeros((1, 2, 2)) for s in LUH2_STATE_NAMES}
    states["c3ann"][:] = 0.1
    states["c3per"][:] = 0.2
    states["c3nfx"][:] = 0.05
    states["c4ann"][:] = 0.3
    states["c4per"][:] = 0.15
    out = luh2_states_to_pft_frac(states, _forest_pnv(2, 2), np.zeros((2, 2)))

    np.testing.assert_allclose(out[:, _IDX["crop_c3"]], 0.35)   # 0.1+0.2+0.05
    np.testing.assert_allclose(out[:, _IDX["crop_c4"]], 0.45)   # 0.3+0.15
    # No crop area bled into natural or grass PFTs.
    natural_and_grass = [i for n, i in _IDX.items() if n not in ("crop_c3", "crop_c4")]
    np.testing.assert_allclose(out[:, natural_and_grass], 0.0, atol=1e-12)


def test_managed_grass_c4_split():
    """pastr+range split between c3_grass / c4_grass by the C4 fraction."""
    states = {s: np.zeros((1, 1, 1)) for s in LUH2_STATE_NAMES}
    states["pastr"][:] = 0.3
    states["range"][:] = 0.1
    c4f = np.array([[0.25]])
    out = luh2_states_to_pft_frac(states, _forest_pnv(1, 1), c4f)

    np.testing.assert_allclose(out[:, _IDX["c4_grass"]], 0.4 * 0.25)
    np.testing.assert_allclose(out[:, _IDX["c3_grass"]], 0.4 * 0.75)


def test_urban_maps_to_bare_soil():
    states = {s: np.zeros((1, 1, 1)) for s in LUH2_STATE_NAMES}
    states["urban"][:] = 0.6
    out = luh2_states_to_pft_frac(states, _forest_pnv(1, 1), np.zeros((1, 1)))
    np.testing.assert_allclose(out[:, _IDX["bare_soil"]], 0.6)


def test_natural_follows_pnv_shape():
    """Natural fraction distributes over PFTs in proportion to the PNV shape."""
    states = {s: np.zeros((1, 1, 1)) for s in LUH2_STATE_NAMES}
    states["primf"][:] = 0.4
    states["secdf"][:] = 0.2   # natural total 0.6
    # PNV = 3 parts needleleaf-temperate : 1 part broadleaf-decid-boreal.
    pnv = np.zeros((N_PFT_CLM5, 1, 1))
    pnv[_IDX["needleleaf_evergreen_temperate"]] = 3.0
    pnv[_IDX["broadleaf_deciduous_boreal"]] = 1.0
    out = luh2_states_to_pft_frac(states, pnv, np.zeros((1, 1)))

    np.testing.assert_allclose(out[:, _IDX["needleleaf_evergreen_temperate"]], 0.6 * 0.75)
    np.testing.assert_allclose(out[:, _IDX["broadleaf_deciduous_boreal"]], 0.6 * 0.25)


def test_empty_pnv_natural_falls_back_to_bare():
    """Natural LUH2 area over a cell with no PNV vegetation -> bare soil."""
    states = {s: np.zeros((1, 1, 1)) for s in LUH2_STATE_NAMES}
    states["primn"][:] = 0.5
    pnv = np.zeros((N_PFT_CLM5, 1, 1))   # empty shape (desert)
    out = luh2_states_to_pft_frac(states, pnv, np.zeros((1, 1)))
    np.testing.assert_allclose(out[:, _IDX["bare_soil"]], 0.5)
    np.testing.assert_allclose(out.sum(axis=1), 0.5)   # still conserved


def test_c4_grass_fraction_from_base():
    base = np.zeros((N_PFT_CLM5, 2, 1))
    base[_IDX["c3_grass"], 0, 0] = 3.0
    base[_IDX["c4_grass"], 0, 0] = 1.0    # 1/(3+1) = 0.25
    # cell (1,0): no grass in base -> default all-C3 (0.0)
    c4f = c4_grass_fraction_from_base(base)
    np.testing.assert_allclose(c4f[0, 0], 0.25)
    np.testing.assert_allclose(c4f[1, 0], 0.0)


def test_build_transient_pft_frac_end_to_end():
    """The producer convenience derives PNV+C4 from a base map and conserves area."""
    base = _forest_pnv(2, 2)
    base[_IDX["c3_grass"]] = 1.0
    base[_IDX["c4_grass"]] = 1.0          # -> C4 grass fraction 0.5
    luh2 = {"states": _uniform_states(nyear=1, ny=2, nx=2)}
    out = build_transient_pft_frac(luh2, base)
    state_total = sum(luh2["states"][s] for s in LUH2_STATE_NAMES)
    np.testing.assert_allclose(out.sum(axis=1), state_total, atol=1e-12)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
def _synthetic_luh2_dataset(nyear=3, ny=2, nx=2, year_base=850):
    xr = pytest.importorskip("xarray")
    time = np.arange(nyear, dtype=np.float64)          # years since year_base
    data = {
        s: (("time", "lat", "lon"), np.full((nyear, ny, nx), 1.0 / len(LUH2_STATE_NAMES)))
        for s in LUH2_STATE_NAMES
    }
    return xr.Dataset(
        data,
        coords={"time": time, "lat": np.linspace(-45, 45, ny), "lon": np.linspace(0, 270, nx)},
    )


def test_read_luh2_states_roundtrip():
    ds = _synthetic_luh2_dataset(nyear=3)
    out = read_luh2_states(dataset=ds, config=LUH2Config(year_base=850))
    assert out["years"].tolist() == [850, 851, 852]
    assert set(out["states"]) == set(LUH2_STATE_NAMES)
    assert out["states"]["primf"].shape == (3, 2, 2)


def test_read_luh2_year_window():
    ds = _synthetic_luh2_dataset(nyear=5)          # 850..854
    out = read_luh2_states(dataset=ds, years=(851, 853))
    assert out["years"].tolist() == [851, 852, 853]
    assert out["states"]["c3ann"].shape[0] == 3


def test_read_luh2_window_out_of_range_raises():
    ds = _synthetic_luh2_dataset(nyear=3)          # 850..852
    with pytest.raises(ValueError, match="no LUH2 years"):
        read_luh2_states(dataset=ds, years=(2000, 2010))


def test_read_luh2_missing_state_raises():
    ds = _synthetic_luh2_dataset(nyear=2).drop_vars("pastr")
    with pytest.raises(ValueError, match="missing required state"):
        read_luh2_states(dataset=ds)


def test_partial_nan_pnv_band_conserves_area():
    """A NaN PFT band at a valid land cell contributes 0 weight, not NaN."""
    states = {s: np.zeros((1, 1, 1)) for s in LUH2_STATE_NAMES}
    states["primf"][:] = 0.2
    states["pastr"][:] = 0.3
    states["urban"][:] = 0.1                       # input total 0.6
    pnv = np.zeros((N_PFT_CLM5, 1, 1))
    pnv[_IDX["broadleaf_evergreen_tropical"]] = np.nan   # NaN band at a live cell
    pnv[_IDX["broadleaf_evergreen_temperate"]] = 1.0
    out = luh2_states_to_pft_frac(states, pnv, np.array([[0.25]]))

    np.testing.assert_allclose(out.sum(axis=1), 0.6, atol=1e-12)   # conserved, finite
    np.testing.assert_allclose(out[:, _IDX["broadleaf_evergreen_tropical"]], 0.0)
    np.testing.assert_allclose(out[:, _IDX["broadleaf_evergreen_temperate"]], 0.2)


def test_read_luh2_transposes_dims_by_name():
    """A file stored (time, lon, lat) is returned as (nyear, lat, lon), not mislabelled."""
    xr = pytest.importorskip("xarray")
    nlat, nlon = 2, 3
    # primf carries a distinctive per-cell pattern so a transpose bug is visible.
    primf = np.arange(nlat * nlon, dtype=np.float64).reshape(1, nlon, nlat)  # (time,lon,lat)
    data = {
        s: (("time", "lon", "lat"),
            primf if s == "primf" else np.zeros((1, nlon, nlat)))
        for s in LUH2_STATE_NAMES
    }
    ds = xr.Dataset(
        data,
        coords={"time": [0.0], "lat": np.linspace(-45, 45, nlat),
                "lon": np.linspace(0, 240, nlon)},
    )
    out = read_luh2_states(dataset=ds)
    assert out["states"]["primf"].shape == (1, nlat, nlon)          # lat, lon order
    # value at (lat=0, lon=1) must equal the source (lon=1, lat=0) entry.
    src = primf[0]                                                   # (lon, lat)
    np.testing.assert_allclose(out["states"]["primf"][0], src.T)


def test_read_luh2_nonmonotonic_time_window():
    """A non-monotonic time axis must not admit out-of-window years."""
    xr = pytest.importorskip("xarray")
    # calendar years 850, 853, 900, 852 (time offsets 0, 3, 50, 2).
    time = np.array([0.0, 3.0, 50.0, 2.0])
    data = {s: (("time", "lat", "lon"), np.full((4, 1, 1), 0.05)) for s in LUH2_STATE_NAMES}
    ds = xr.Dataset(data, coords={"time": time, "lat": [0.0], "lon": [0.0]})
    out = read_luh2_states(dataset=ds, years=(852, 853))
    assert sorted(out["years"].tolist()) == [852, 853]              # 900 excluded
    assert out["states"]["primf"].shape[0] == 2
