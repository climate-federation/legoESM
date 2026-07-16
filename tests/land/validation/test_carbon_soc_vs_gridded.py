"""Unit tests for the SOC model-vs-gridded-obs validator
(``scripts/validate/carbon_soc_vs_gridded.py``).

Pure-logic tests of the area-weighted statistics (bias / RMSE / weighted-Pearson corr),
the zonal profile, the per-biome grouping (via the shared ``PFT_BIOME`` table), the exact
spherical cell-area (sums to 4 pi R^2), the units-mismatch guard, and a synthetic
end-to-end ``assess`` over a tiny IC npz + obs NetCDF.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PY = _REPO / "scripts" / "validate" / "carbon_soc_vs_gridded.py"


def _mod():
    spec = importlib.util.spec_from_file_location("carbon_soc_vs_gridded", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules["carbon_soc_vs_gridded"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# weighted statistics
# ---------------------------------------------------------------------------
def _simple_case():
    lat = np.array([0.0, 60.0])                       # w = 1, 0.5
    model = np.array([[10.0, 10.0], [4.0, 4.0]])
    obs = np.array([[8.0, 8.0], [6.0, 6.0]])
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(model)
    mask = np.ones_like(model, bool)
    return model, obs, w, mask


def test_weighted_stats_bias_rmse_manual():
    m = _mod()
    model, obs, w, mask = _simple_case()
    st = m.weighted_stats(model, obs, w, mask)
    np.testing.assert_allclose(st["bias"], 2.0 / 3.0, rtol=1e-12)     # cos-lat weighted
    np.testing.assert_allclose(st["rmse"], 2.0, rtol=1e-12)
    np.testing.assert_allclose(st["model_mean"], 8.0, rtol=1e-12)
    np.testing.assert_allclose(st["obs_mean"], 22.0 / 3.0, rtol=1e-12)
    assert st["n"] == 4


def test_weighted_stats_corr_perfect_and_anti():
    m = _mod()
    model, _, w, mask = _simple_case()
    st_perfect = m.weighted_stats(model, model.copy(), w, mask)
    np.testing.assert_allclose(st_perfect["corr"], 1.0, rtol=1e-12)
    anti = 20.0 - model                                # perfectly anti-correlated
    st_anti = m.weighted_stats(model, anti, w, mask)
    np.testing.assert_allclose(st_anti["corr"], -1.0, rtol=1e-12)


def test_weighted_stats_excludes_masked_and_nan():
    m = _mod()
    model, obs, w, mask = _simple_case()
    model = model.copy(); model[0, 1] = np.nan         # NaN cell dropped
    mask = mask.copy(); mask[1, 0] = False             # masked (non-land) cell dropped
    st = m.weighted_stats(model, obs, w, mask)
    assert st["n"] == 2                                # (0,0) and (1,1) survive
    assert np.isfinite(st["bias"])


def test_weighted_stats_empty_mask_is_nan_not_crash():
    m = _mod()
    model, obs, w, _ = _simple_case()
    st = m.weighted_stats(model, obs, w, np.zeros_like(model, bool))
    assert st["n"] == 0 and np.isnan(st["bias"]) and np.isnan(st["corr"])


# ---------------------------------------------------------------------------
# zonal profile
# ---------------------------------------------------------------------------
def test_zonal_profile_reflects_meridional_gradient():
    m = _mod()
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, 240.0])
    obs = np.array([[2.0] * 3, [10.0] * 3, [30.0] * 3])     # increases toward +lat
    model = obs - 3.0                                        # uniform low bias
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(obs)
    mask = np.ones_like(obs, bool)
    zlat, mz, oz, bz = m.zonal_profile(model, obs, w, mask, lat)
    np.testing.assert_allclose(oz, [2.0, 10.0, 30.0])
    np.testing.assert_allclose(bz, [-3.0, -3.0, -3.0])
    assert oz[0] < oz[1] < oz[2]                              # gradient preserved


# ---------------------------------------------------------------------------
# per-biome grouping
# ---------------------------------------------------------------------------
def test_per_biome_bias_grouping_and_sort():
    m = _mod()
    pft_names = ["bare_soil", "broadleaf_evergreen_tropical", "c3_arctic_grass"]
    # 1 lat x 3 lon; each cell a distinct dominant PFT -> distinct biome.
    dompft = np.array([[0, 1, 2]])                            # bare, tropical, tundra
    obs = np.array([[1.0, 12.0, 40.0]])
    model = np.array([[1.0, 10.0, 20.0]])                    # bias 0, -2, -20
    w = np.ones_like(obs)
    mask = np.ones_like(obs, bool)
    rows = m.per_biome_bias(model, obs, w, mask, dompft, pft_names)
    by = {r["biome"]: r for r in rows}
    assert set(by) == {"bare/none", "tropical_forest", "tundra"}
    np.testing.assert_allclose(by["tundra"]["bias"], -20.0)
    np.testing.assert_allclose(by["tropical_forest"]["bias"], -2.0)
    np.testing.assert_allclose(by["bare/none"]["bias"], 0.0)
    # sorted by descending |bias| -> tundra first.
    assert rows[0]["biome"] == "tundra"
    # tundra carries the Jobbagy 0-100 cm SOC range for context.
    assert by["tundra"]["jobbagy_soc_0_100"] is not None


# ---------------------------------------------------------------------------
# grid reshape + exact cell area
# ---------------------------------------------------------------------------
def test_reshape_flat_to_grid_places_by_coordinate():
    m = _mod()
    # deliberately shuffled flat order; scatter must place each cell by (lat,lon).
    lat_flat = np.array([10.0, 10.0, -10.0, -10.0])
    lon_flat = np.array([0.0, 90.0, 0.0, 90.0])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    perm = np.array([2, 0, 3, 1])
    g, lat, lon = m.reshape_flat_to_grid(vals[perm], lat_flat[perm], lon_flat[perm])
    np.testing.assert_array_equal(lat, [-10.0, 10.0])
    np.testing.assert_array_equal(lon, [0.0, 90.0])
    np.testing.assert_array_equal(g, [[3.0, 4.0], [1.0, 2.0]])


def test_cell_area_sums_to_sphere():
    m = _mod()
    from legoesm import constants
    lat = np.linspace(-90.0, 90.0, 96)
    lon = np.linspace(0.0, 357.5, 144)
    area = m.cell_area_m2(lat, lon)
    sphere = 4.0 * np.pi * float(constants.R_earth) ** 2
    np.testing.assert_allclose(area.sum(), sphere, rtol=2e-3)


def test_total_pgc_constant_field():
    m = _mod()
    lat = np.linspace(-88.0, 88.0, 20)
    lon = np.linspace(0.0, 355.0, 24)
    area = m.cell_area_m2(lat, lon)
    mask = np.ones((lat.size, lon.size), bool)
    field = np.full((lat.size, lon.size), 5.0)               # 5 kgC/m2 everywhere
    np.testing.assert_allclose(m.total_pgc(field, area, mask),
                               5.0 * area.sum() / 1e12, rtol=1e-12)


# ---------------------------------------------------------------------------
# units-mismatch guard + end-to-end assess
# ---------------------------------------------------------------------------
def _write_synthetic_ic(path, nlat=6, nlon=8):
    lat = np.linspace(-75.0, 75.0, nlat)
    lon = np.linspace(0.0, 315.0, nlon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = LAT.reshape(-1)
    lon_flat = LON.reshape(-1)
    # SOC-like: higher toward the poles; some ocean (mask False).
    soc_k = 4.0 + 0.1 * np.abs(lat_flat)                      # kgC/m2
    land = (lon_flat < 250.0)                                 # crude land mask
    dompft = np.where(np.abs(lat_flat) > 55.0, 12, 4)         # tundra vs tropical
    dompft = np.where(land, dompft, -1)
    pool = soc_k * 1000.0 / 3.0                               # split across 3 pools (gC/m2)
    np.savez(
        path,
        C_som_active=pool, C_som_slow=pool, C_som_passive=pool,
        lat=lat_flat, lon=lon_flat, land_mask=land,
        dominant_pft=dompft.astype(np.int64),
        pft_names=np.array(
            ["bare_soil", "needleleaf_evergreen_temperate", "needleleaf_evergreen_boreal",
             "needleleaf_deciduous_boreal", "broadleaf_evergreen_tropical",
             "broadleaf_evergreen_temperate", "broadleaf_deciduous_tropical",
             "broadleaf_deciduous_temperate", "broadleaf_deciduous_boreal",
             "broadleaf_evergreen_shrub", "broadleaf_deciduous_temperate_shrub",
             "broadleaf_deciduous_boreal_shrub", "c3_arctic_grass", "c3_grass",
             "c4_grass", "crop_c3", "crop_c4"], dtype="<U40"),
        n_layers=np.asarray(10), soil_depth=np.asarray(3.0), dt=np.asarray(7200.0),
        resolution_deg=np.asarray(1.0))
    return lat, lon, soc_k.reshape(nlat, nlon)


def _write_obs(path, lat, lon, soc2d):
    import xarray as xr
    xr.Dataset({"soc": (("lat", "lon"), soc2d)},
               coords={"lat": lat, "lon": lon}).assign_attrs(product="synthetic")\
        .to_netcdf(path)


def test_align_obs_rejects_unit_mismatch(tmp_path):
    m = _mod()
    ic = tmp_path / "ic.npz"
    lat, lon, soc2d = _write_synthetic_ic(ic)
    obs = tmp_path / "obs_gcm2.nc"
    _write_obs(obs, lat, lon, soc2d * 1000.0)                # gC/m2 (1000x too large)
    model2d, mlat, mlon, *_ = m.load_model_soc(ic)
    try:
        m.align_obs(obs, mlat, mlon)
        raise AssertionError("expected SystemExit on kgC/m2 vs gC/m2 unit mismatch")
    except SystemExit:
        pass


def test_assess_end_to_end_synthetic(tmp_path):
    m = _mod()
    ic = tmp_path / "ic.npz"
    lat, lon, soc2d = _write_synthetic_ic(ic)
    obs = tmp_path / "obs.nc"
    _write_obs(obs, lat, lon, soc2d + 2.0)                   # obs uniformly 2 higher
    result = m.assess(str(ic), str(obs))
    # model uniformly 2 kgC/m2 below obs -> area-weighted bias ~ -2, corr ~ +1.
    np.testing.assert_allclose(result["global_bias"], -2.0, atol=1e-9)
    np.testing.assert_allclose(result["global_corr"], 1.0, atol=1e-9)
    assert result["n_land"] > 0
    assert result["model_total_PgC"] < result["obs_total_PgC"]
    # both biomes present in the per-biome table.
    biomes = {r["biome"] for r in result["per_biome"]}
    assert {"tundra", "tropical_forest"} <= biomes
    # zonal bias is the uniform -2 everywhere there is land.
    bz = np.asarray(result["zonal"]["bias"])
    assert np.allclose(bz[np.isfinite(bz)], -2.0, atol=1e-9)


def test_assess_writes_plot_and_json(tmp_path):
    m = _mod()
    ic = tmp_path / "ic.npz"
    lat, lon, soc2d = _write_synthetic_ic(ic)
    obs = tmp_path / "obs.nc"
    _write_obs(obs, lat, lon, soc2d + 1.0)
    result = m.assess(str(ic), str(obs))
    png = m.plot_soc_comparison(result, str(tmp_path / "out.png"))
    assert pathlib.Path(png).exists() and pathlib.Path(png).stat().st_size > 0
