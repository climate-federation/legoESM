"""Unit tests for the gridded satellite-SIF fetcher
(scripts/data/build_sif_observations.py).

Exercises the PURE reduction/conversion offline on synthetic arrays -- the networked
Caltech-FTP ``_fetch_caltech_sif740`` wrapper is NOT exercised -- and proves a NetCDF
write -> read round-trip that ``legoesm.land.carbon.sif_observations.load_gridded_sif``
(the ``--sif-obs`` contract) actually accepts, plus a synthetic gridded-SIF ->
per-archetype-target chain through ``per_archetype_observed_sif``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

from legoesm import constants

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PY = _REPO / "scripts" / "data" / "build_sif_observations.py"


def _mod():
    spec = importlib.util.spec_from_file_location("build_sif_observations", _PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# radiance -> photon-flux conversion
# ---------------------------------------------------------------------------
def test_photon_flux_per_watt_matches_planck():
    m = _mod()
    got = m.photon_flux_per_watt(740.0)
    # Independent e_photon = h c / lambda; umol/m2/s per W/m2 = 1e6 / (e_photon * N_A).
    e_photon = constants.h_planck * constants.c_light / (740.0e-9)   # [J/photon]
    expect = 1.0e6 / (e_photon * constants.N_A)
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    # ~6.19 umol/m2/s per W/m2 at 740 nm (sanity magnitude).
    assert 6.0 < got < 6.4


def test_radiance_to_photon_flux_linear_sign_and_factor():
    m = _mod()
    rad = np.array([-0.5, 0.0, 1.0, 2.0])            # mW/m2/sr/nm (negatives = noise)
    out = m.radiance_to_photon_flux(rad)
    # linear + sign preserving: out = k * rad, k > 0
    k = out[2] - out[1]                              # slope per unit radiance
    assert k > 0
    np.testing.assert_allclose(out, k * rad, rtol=1e-12)
    assert out[0] < 0 and out[1] == 0.0              # negative stays negative, 0 -> 0
    # k equals the documented factor product.
    k_expect = (constants.PI * m._SIF_FARRED_EFFECTIVE_WIDTH_NM * m._MW_TO_W
                * m.photon_flux_per_watt(m._SIF_WAVELENGTH_NM))
    np.testing.assert_allclose(k, k_expect, rtol=1e-12)


# ---------------------------------------------------------------------------
# temporal aggregation
# ---------------------------------------------------------------------------
def test_growing_season_mean_picks_active_months_nan_aware():
    m = _mod()
    nlat, nlon = 2, 3
    monthly = np.full((12, nlat, nlon), np.nan)
    # Cell (0,0): a summer bump months 5-7 = [8,10,8], rest low (=1).
    monthly[:, 0, 0] = 1.0
    monthly[5:8, 0, 0] = [8.0, 10.0, 8.0]
    # active_fraction 0.5 -> threshold 5.0 -> only months 5,6,7 -> mean (8+10+8)/3.
    gs = m.growing_season_mean(monthly, active_fraction=0.5)
    np.testing.assert_allclose(gs[0, 0], (8.0 + 10.0 + 8.0) / 3.0)
    # Cell (0,1): all-NaN -> stays NaN (never observed).
    assert np.isnan(gs[0, 1])
    # Cell (1,2): observed only months 6-8 (winter NaN); max there governs.
    monthly[:, 1, 2] = np.nan
    monthly[6:9, 1, 2] = [4.0, 6.0, 2.0]             # thr 3.0 -> months 6,7 -> (4+6)/2
    gs = m.growing_season_mean(monthly, active_fraction=0.5)
    np.testing.assert_allclose(gs[1, 2], (4.0 + 6.0) / 2.0)


def test_growing_season_nonpositive_max_falls_back_not_nan():
    """A cell whose finite months are ALL <= 0 (barren retrieval noise) has no positive
    seasonal amplitude, so it falls back to the finite-month mean -- NOT silently NaN
    (regression for the codex-flagged threshold>max defect)."""
    m = _mod()
    monthly = np.full((12, 1, 3), np.nan)
    # Cell 0: all-negative finite months -> fallback = mean of the finite months.
    monthly[:, 0, 0] = np.nan
    monthly[3:6, 0, 0] = [-1.0, -2.0, -3.0]
    # Cell 1: max exactly 0 (0 and negatives) -> fallback mean, finite (not NaN).
    monthly[2:5, 0, 1] = [0.0, -1.0, -2.0]
    # Cell 2: genuinely all-NaN -> stays NaN.
    gs = m.growing_season_mean(monthly, active_fraction=0.5)
    np.testing.assert_allclose(gs[0, 0], (-1.0 - 2.0 - 3.0) / 3.0)   # observed, preserved
    assert np.isfinite(gs[0, 0])
    np.testing.assert_allclose(gs[0, 1], (0.0 - 1.0 - 2.0) / 3.0)
    assert np.isnan(gs[0, 2])                                        # never-observed -> NaN


def test_growing_season_rejects_out_of_range_fraction():
    """active_fraction outside (0, 1] is a hard error (a >1 fraction thresholds above the
    annual max and would silently drop positive observed cells)."""
    m = _mod()
    monthly = np.ones((12, 1, 1))
    for bad in (1.5, 0.0, -0.2):
        try:
            m.growing_season_mean(monthly, active_fraction=bad)
            raise AssertionError(f"expected ValueError for active_fraction={bad}")
        except ValueError:
            pass


def test_average_years_and_annual_mean_nan_aware():
    m = _mod()
    y1 = np.ones((12, 2, 2))
    y2 = np.full((12, 2, 2), 3.0)
    y2[:, 0, 0] = np.nan                              # a cell missing in year 2
    clim = m.average_years([y1, y2])
    np.testing.assert_allclose(clim[:, 0, 0], 1.0)    # only y1 observed -> 1
    np.testing.assert_allclose(clim[:, 1, 1], 2.0)    # (1+3)/2
    np.testing.assert_allclose(m.annual_mean(clim)[1, 1], 2.0)


# ---------------------------------------------------------------------------
# area weighting + pattern-preserving rescale
# ---------------------------------------------------------------------------
def test_area_weighted_nanmean_cos_lat_and_nan():
    m = _mod()
    lat = np.array([-60.0, 0.0, 60.0])
    field = np.array([[10.0, 10.0], [2.0, 2.0], [np.nan, 4.0]])   # (3 lat, 2 lon)
    w = np.cos(np.deg2rad(lat))
    # manual: exclude the NaN cell from both sums
    num = w[0] * 20.0 + w[1] * 4.0 + w[2] * 4.0
    den = w[0] * 2 + w[1] * 2 + w[2] * 1
    np.testing.assert_allclose(m.area_weighted_nanmean(field, lat), num / den)


def test_rescale_preserves_pattern_and_hits_reference():
    m = _mod()
    lat = np.array([0.0, 30.0])
    field = np.array([[1.0, np.nan], [2.0, 3.0]])
    scaled, factor = m.rescale_to_reference_mean(field, lat, reference_umol=5.0)
    np.testing.assert_allclose(m.area_weighted_nanmean(scaled, lat), 5.0, rtol=1e-12)
    # pattern (ratios) untouched; NaN preserved.
    np.testing.assert_allclose(scaled[1, 1] / scaled[0, 0], 3.0)
    assert np.isnan(scaled[0, 1])
    np.testing.assert_allclose(scaled, field * factor, equal_nan=True)


def test_rescale_raises_on_nonpositive_mean():
    m = _mod()
    lat = np.array([0.0, 10.0])
    bad = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    try:
        m.rescale_to_reference_mean(bad, lat, reference_umol=5.0)
        raise AssertionError("expected ValueError on all-NaN field")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# full pure pipeline + regrid orientation
# ---------------------------------------------------------------------------
def _synthetic_source(nlat=18, nlon=36):
    """A monthly SIF radiance field with a smooth lat gradient + a summer cycle,
    on the Caltech-like grid (ascending lat, -180..180 lon)."""
    src_lat = np.linspace(-85.0, 85.0, nlat)          # ascending
    src_lon = np.linspace(-175.0, 175.0, nlon)        # -180..180
    base = np.cos(np.deg2rad(src_lat))[:, None] * np.ones((1, nlon))   # bright tropics
    monthly = np.empty((12, nlat, nlon))
    for mo in range(12):
        season = 1.0 + 0.8 * np.cos(2 * np.pi * (mo - 6) / 12.0)       # summer peak ~ Jul
        monthly[mo] = base * season
    return monthly, src_lat, src_lon


def test_reduce_structure_units_and_nan_and_rescale():
    m = _mod()
    monthly, src_lat, src_lon = _synthetic_source()
    tgt_lat = np.linspace(88.0, -88.0, 12)            # DESCENDING target (CLM-like)
    tgt_lon = np.linspace(1.25, 358.75, 24)           # 0..360 convention
    ds, prov = m.reduce_sif_to_climatology(
        [monthly], src_lat, src_lon, tgt_lat, tgt_lon,
        aggregate="growing_season", rescale=True, reference_umol=5.0)
    assert set(ds.data_vars) == {"sif"}
    assert ds["sif"].dims == ("lat", "lon")
    assert ds["sif"].shape == (tgt_lat.size, tgt_lon.size)
    assert ds["sif"].attrs["units"] == "umol m-2 s-1"
    # rescaled area-weighted mean hits the reference.
    np.testing.assert_allclose(
        m.area_weighted_nanmean(ds["sif"].values, tgt_lat), 5.0, rtol=1e-9)
    assert prov["rescaled"] and prov["aggregate"] == "growing_season"
    # tropical (equator) cells brighter than polar cells -> pattern survived the regrid.
    sif = np.asarray(ds["sif"].values)
    eq = np.nanmean(sif[np.argmin(np.abs(tgt_lat))])
    pole = np.nanmean(sif[np.argmax(np.abs(tgt_lat))])
    assert eq > pole


def test_reduce_regrid_orientation_matches_source_pattern():
    """A source lat gradient must land on the target with the SAME sign of gradient
    (guards against a flipped lat axis in the regrid wiring)."""
    m = _mod()
    nlat, nlon = 18, 36
    src_lat = np.linspace(-85.0, 85.0, nlat)          # ascending
    src_lon = np.linspace(-175.0, 175.0, nlon)
    # value increases monotonically with latitude (north brightest) -- unambiguous.
    ramp = ((src_lat + 90.0) / 180.0)[:, None] * np.ones((1, nlon)) + 0.1
    monthly = np.repeat(ramp[None], 12, axis=0)
    tgt_lat = np.linspace(80.0, -80.0, 10)            # descending
    tgt_lon = np.linspace(5.0, 355.0, 18)
    ds, _ = m.reduce_sif_to_climatology(
        [monthly], src_lat, src_lon, tgt_lat, tgt_lon,
        aggregate="annual", rescale=False)
    sif = np.asarray(ds["sif"].values)
    zonal = np.nanmean(sif, axis=1)                   # per target-lat mean
    # target lat is descending, source ramps up with lat -> zonal must DECREASE.
    assert zonal[0] > zonal[-1]
    assert np.all(np.diff(zonal) < 0)


def test_reduce_validates_bad_aggregate_and_shape():
    m = _mod()
    monthly, src_lat, src_lon = _synthetic_source()
    for bad in ("median", "", None):
        try:
            m.reduce_sif_to_climatology([monthly], src_lat, src_lon,
                                        np.array([0.0]), np.array([0.0]), aggregate=bad)
            raise AssertionError(f"expected ValueError for aggregate={bad!r}")
        except ValueError:
            pass
    try:
        m.growing_season_mean(np.zeros((11, 2, 2)))   # not 12 months
        raise AssertionError("expected ValueError on non-12-month input")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# NetCDF round-trip through load_gridded_sif + per-archetype target
# ---------------------------------------------------------------------------
def test_roundtrip_through_load_gridded_sif(tmp_path):
    from legoesm.land.carbon.sif_observations import (
        load_gridded_sif,
        per_archetype_observed_sif,
    )

    m = _mod()
    monthly, src_lat, src_lon = _synthetic_source()
    tgt_lat = np.linspace(88.0, -88.0, 8)
    tgt_lon = np.linspace(2.5, 357.5, 10)
    ncell = tgt_lat.size * tgt_lon.size
    ds, _ = m.reduce_sif_to_climatology(
        [monthly], src_lat, src_lon, tgt_lat, tgt_lon, rescale=True)
    out = tmp_path / "sif.nc"
    ds.to_netcdf(out)

    sif_cell = load_gridded_sif(str(out), ncell=ncell)
    assert sif_cell.shape == (ncell,)
    # matches the in-memory field reshaped row-major (i_lat, i_lon).
    np.testing.assert_allclose(
        sif_cell, np.asarray(ds["sif"].values).reshape(-1), equal_nan=True, rtol=1e-9)

    # wrong ncell is a hard error (never a silent misalignment).
    try:
        load_gridded_sif(str(out), ncell=ncell + 1)
        raise AssertionError("expected SystemExit on ncell mismatch")
    except SystemExit:
        pass

    # synthetic gridded SIF -> per-archetype cover-weighted target, NaN excluded.
    npft = 2
    cell_id = np.full((ncell, npft), -1, dtype=int)
    cell_w = np.zeros((ncell, npft), dtype=float)
    cell_id[0, 0] = 0
    cell_w[0, 0] = 1.0
    cell_id[1, 0] = 0
    cell_w[1, 0] = 3.0
    tgt = per_archetype_observed_sif(sif_cell, cell_id, cell_w, n_arch=1)
    finite = np.isfinite(sif_cell[:2])
    if finite.all():
        exp = (1.0 * sif_cell[0] + 3.0 * sif_cell[1]) / 4.0
        np.testing.assert_allclose(tgt[0], exp, rtol=1e-9)
    assert tgt.shape == (1,)
