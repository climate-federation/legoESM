"""Unit tests for the ESA CCI PFT cover/PFT derivation (format-independent)."""

import numpy as np
import pytest

from legoesm.land.surface_data.sources.esa_cci_pft import (
    derive_cover_and_pft,
    esa_to_clm5_pft,
    aggregate_esa_pft,
    ESA_ALL_LAYER_NAMES,
    ESA_NC_VARMAP,
    ESA_VEG_PFT_NAMES,
    LAND_PFT_NAMES,
    N_LAND_PFT,
)
from legoesm.land.surface_data.bioclimate import ClimateZones
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5


def _layers(n=3):
    """Three cells: [0]=mixed land, [1]=open ocean, [2]=pure natural grassland."""
    L = {name: np.zeros(n) for name in ESA_ALL_LAYER_NAMES}
    L["bare_soil"] = np.array([0.10, 0.0, 0.0])
    L["built"] = np.array([0.05, 0.0, 0.0])
    L["water"] = np.array([0.10, 1.0, 0.0])
    L["snow_ice"] = np.array([0.05, 0.0, 0.0])
    L["tree_broadleaf_evergreen"] = np.array([0.30, 0.0, 0.0])
    L["tree_broadleaf_deciduous"] = np.array([0.20, 0.0, 0.0])
    L["natural_grass"] = np.array([0.20, 0.0, 1.0])
    return L


def test_layer_axis_definition():
    # bare at index 0, then the 10 ESA veg PFTs, no built (folded into bare).
    assert LAND_PFT_NAMES[0] == "bare_soil"
    assert LAND_PFT_NAMES[1:] == ESA_VEG_PFT_NAMES
    assert N_LAND_PFT == 11
    assert "built" not in LAND_PFT_NAMES


def test_cover_fractions_and_ocean_masking():
    out = derive_cover_and_pft(_layers())
    np.testing.assert_allclose(out["f_land"], [0.85, 0.0, 1.0])
    np.testing.assert_allclose(out["f_glacier"], [0.05, 0.0, 0.0])
    # ocean water (cell 1) is dropped; inland water (cell 0) becomes lake.
    np.testing.assert_allclose(out["f_lake"], [0.10, 0.0, 0.0])


def test_pft_frac_partition_and_composition():
    out = derive_cover_and_pft(_layers())
    pf = out["pft_frac"]
    assert pf.shape == (3, N_LAND_PFT)
    # land cells sum to 1; ocean cell stays all-zero
    np.testing.assert_allclose(pf[0].sum(), 1.0)
    np.testing.assert_allclose(pf[2].sum(), 1.0)
    np.testing.assert_allclose(pf[1].sum(), 0.0)
    # cell 0: bare+built folded into index 0 = 0.15/0.85
    np.testing.assert_allclose(pf[0, 0], 0.15 / 0.85)
    # dominant PFT in cell 0 is tree_broadleaf_evergreen (index 1) = 0.30/0.85
    assert LAND_PFT_NAMES[int(np.argmax(pf[0]))] == "tree_broadleaf_evergreen"
    np.testing.assert_allclose(pf[0, 1], 0.30 / 0.85)
    # cell 2: pure natural grass
    gi = LAND_PFT_NAMES.index("natural_grass")
    np.testing.assert_allclose(pf[2, gi], 1.0)


def test_explicit_land_mask_overrides_heuristic():
    # Force cell 1 (open ocean) to be 'land' -> its water becomes lake.
    out = derive_cover_and_pft(_layers(), land_mask=np.array([True, True, True]))
    np.testing.assert_allclose(out["f_lake"], [0.10, 1.0, 0.0])


def test_pct_scale_converts_percent():
    L = {k: v * 100.0 for k, v in _layers().items()}   # same data in percent
    out = derive_cover_and_pft(L, pct_scale=0.01)
    np.testing.assert_allclose(out["f_land"], [0.85, 0.0, 1.0])


def test_missing_layer_raises():
    L = _layers()
    del L["managed_grass"]
    with pytest.raises(ValueError, match="missing ESA layers"):
        derive_cover_and_pft(L)


# ---------------------------------------------------------------------------
# ESA -> CLM5 crosswalk
# ---------------------------------------------------------------------------
def _land_frac_from(d):
    """Build an (n, N_LAND_PFT) ESA composition from a list of name->value dicts."""
    arr = np.zeros((len(d), N_LAND_PFT))
    for i, cell in enumerate(d):
        for name, val in cell.items():
            arr[i, LAND_PFT_NAMES.index(name)] = val
    return arr


def _ci(name):
    return CLM5_PFT_NAMES.index(name)


def test_crosswalk_climate_routing_and_shape():
    esa = _land_frac_from([
        {"tree_broadleaf_evergreen": 1.0},   # cell0 tropical
        {"tree_broadleaf_evergreen": 1.0},   # cell1 temperate
    ])
    zones = ClimateZones(
        tropical=np.array([True, False]),
        temperate=np.array([False, True]),
        boreal=np.array([False, False]),
    )
    out = esa_to_clm5_pft(esa, zones, np.zeros(2), np.zeros(2))
    assert out.shape == (2, N_PFT_CLM5)
    np.testing.assert_allclose(out[0, _ci("broadleaf_evergreen_tropical")], 1.0)
    np.testing.assert_allclose(out[1, _ci("broadleaf_evergreen_temperate")], 1.0)


def test_crosswalk_needleleaf_shrub_folding():
    esa = _land_frac_from([
        {"shrub_needleleaf_evergreen": 0.5, "shrub_needleleaf_deciduous": 0.5},
    ])
    zones = ClimateZones(np.array([False]), np.array([True]), np.array([False]))
    out = esa_to_clm5_pft(esa, zones, np.zeros(1), np.zeros(1))
    # NE shrub -> broadleaf-evergreen shrub; ND shrub -> broadleaf-deciduous-boreal shrub
    np.testing.assert_allclose(out[0, _ci("broadleaf_evergreen_shrub")], 0.5)
    np.testing.assert_allclose(out[0, _ci("broadleaf_deciduous_boreal_shrub")], 0.5)


def test_crosswalk_grass_c4_split_and_arctic():
    esa = _land_frac_from([{"natural_grass": 1.0}])
    zones = ClimateZones(np.array([False]), np.array([False]), np.array([True]))  # boreal
    out = esa_to_clm5_pft(esa, zones, c4_grass_frac=np.array([0.25]), c4_crop_frac=np.zeros(1))
    np.testing.assert_allclose(out[0, _ci("c4_grass")], 0.25)
    np.testing.assert_allclose(out[0, _ci("c3_arctic_grass")], 0.75)   # C3 remainder is arctic (boreal)
    np.testing.assert_allclose(out[0, _ci("c3_grass")], 0.0)


def test_aggregate_esa_pft_from_synthetic_dataset():
    xr = pytest.importorskip("xarray")
    nlat, nlon = 4, 8
    lat = 90.0 - (np.arange(nlat) + 0.5) * (180.0 / nlat)   # descending
    lon = -180.0 + (np.arange(nlon) + 0.5) * (360.0 / nlon)
    arr = {ESA_NC_VARMAP[k]: np.zeros((1, nlat, nlon), np.int8) for k in ESA_ALL_LAYER_NAMES}
    arr["TREES-BE"][0, 0:2, 0:2] = 100      # coarse cell (0,0): tropical forest
    arr["GRASS-NAT"][0, 0:2, 2:4] = 100     # coarse cell (0,1): natural grass
    arr["BARE"][0, 2:4, 0:2] = 100          # coarse cell (1,0): bare
    # remaining fine pixels all zero -> ocean (f_land=0)
    ds = xr.Dataset(
        {v: (("time", "lat", "lon"), a) for v, a in arr.items()},
        coords={"lat": lat, "lon": lon, "time": [0.0]},
    )
    out = aggregate_esa_pft(dataset=ds, res_deg=90.0, coarse_rows_per_chunk=1)
    assert out["pft_frac"].shape == (2, 4, N_LAND_PFT)
    assert out["lat"].shape == (2,) and out["lon"].shape == (4,)
    be = LAND_PFT_NAMES.index("tree_broadleaf_evergreen")
    ng = LAND_PFT_NAMES.index("natural_grass")
    np.testing.assert_allclose(out["f_land"][0, 0], 1.0)
    np.testing.assert_allclose(out["pft_frac"][0, 0, be], 1.0)
    np.testing.assert_allclose(out["pft_frac"][0, 1, ng], 1.0)
    np.testing.assert_allclose(out["f_land"][1, 0], 1.0)
    np.testing.assert_allclose(out["pft_frac"][1, 0, 0], 1.0)   # bare at index 0
    np.testing.assert_allclose(out["f_land"][1, 2], 0.0)        # ocean cell


def test_crosswalk_conserves_total_area():
    rng = np.random.default_rng(3)
    n = 20
    esa = rng.uniform(0, 1, size=(n, N_LAND_PFT))
    # random partition into the three zones
    z = rng.integers(0, 3, size=n)
    zones = ClimateZones(z == 0, z == 1, z == 2)
    out = esa_to_clm5_pft(esa, zones, rng.uniform(0, 1, n), rng.uniform(0, 1, n))
    # CLM5 total per cell == ESA total per cell (area conservation)
    np.testing.assert_allclose(out.sum(axis=-1), esa.sum(axis=-1), rtol=1e-12)
