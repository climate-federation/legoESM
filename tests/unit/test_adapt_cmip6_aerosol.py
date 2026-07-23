"""Unit tests for the MACv2-SP aerosol adapter
(``scripts/data/adapt_cmip6_aerosol.py``).

The adapter evaluates an IDEALIZED simplified-plume surrogate (NOT the reference
Stevens et al. 2017 MACv2-SP model — one geometry per plume + a cosine seasonal
surrogate) onto a grid to give the deck's ``aod(time, lat, lon)`` field.  The
magnitudes are NOT validated against the reference here (that needs a real
MACv2-SP file + the ``mo_simple_plumes`` code); these tests pin the STRUCTURAL
physics — a plume
peaks at its centre, an east/west sigma difference makes the falloff asymmetric,
the annual + seasonal scalings multiply, the Angstrom slope, the parameter
reader, and a round-trip of the gridded output through the REAL aerosol loader.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PY = _REPO / "scripts" / "data" / "adapt_cmip6_aerosol.py"


def _mod():
    spec = importlib.util.spec_from_file_location("adapt_cmip6_aerosol", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _one_plume(**over):
    p = {"lon": 100.0, "lat": 20.0, "theta": 0.0,
         "sig_lon_e": 10.0, "sig_lon_w": 10.0, "sig_lat_n": 8.0, "sig_lat_s": 8.0,
         "aod": 0.4, "features": [(1.0, 0.0, 0.0, 0.0)], "year_weight": 1.0}
    p.update(over)
    return p


def test_plume_peaks_at_its_centre():
    m = _mod()
    lon = np.linspace(0, 360, 181, endpoint=False)
    lat = np.linspace(-89, 89, 90)
    aod = m.evaluate_plume_aod(lon, lat, [_one_plume()], 2000, 196.5)
    j, i = np.unravel_index(np.argmax(aod), aod.shape)
    assert abs(lat[j] - 20.0) < 3.0 and abs(lon[i] - 100.0) < 3.0
    # continuous peak == plume aod (0.4); the discrete grid misses the exact
    # centre so the sampled max is just below it, never above.
    assert aod.max() <= 0.4 + 1e-9
    assert aod.max() == pytest.approx(0.4, abs=1e-2)
    assert np.all(aod >= 0.0)


def test_eastwest_sigma_makes_asymmetric_falloff():
    m = _mod()
    lon = np.linspace(0, 360, 361, endpoint=False)
    lat = np.array([20.0])
    # wide east (sig_lon_e=20), narrow west (sig_lon_w=5): AOD 15deg EAST of the
    # centre > AOD 15deg WEST.
    aod = m.evaluate_plume_aod(lon, lat, [_one_plume(sig_lon_e=20.0,
                                                     sig_lon_w=5.0)], 2000, 196.5)[0]
    east = aod[np.argmin(np.abs(lon - 115.0))]
    west = aod[np.argmin(np.abs(lon - 85.0))]
    assert east > west * 2.0


def test_year_factor_dict_table_scalar():
    m = _mod()
    assert m._year_factor({1990: 0.5, 2000: 1.0}, 2000) == 1.0
    assert m._year_factor({1990: 0.5}, 1888) == 0.0          # absent -> 0
    # table: linear interp + end-clamp
    yw = ([1850.0, 2000.0], [0.0, 1.0])
    assert np.isclose(m._year_factor(yw, 1925), 0.5)
    assert np.isclose(m._year_factor(yw, 2100), 1.0)         # clamped
    assert m._year_factor(0.7, 1999) == 0.7                  # scalar


def test_annual_scaling_multiplies_aod():
    m = _mod()
    lon, lat = np.array([100.0]), np.array([20.0])
    full = m.evaluate_plume_aod(lon, lat, [_one_plume(year_weight=1.0)], 2000, 196.5)
    half = m.evaluate_plume_aod(lon, lat, [_one_plume(year_weight=0.5)], 2000, 196.5)
    assert np.isclose(half[0, 0], 0.5 * full[0, 0])


def test_seasonal_weight_cycle():
    m = _mod()
    # amplitude 0.5, phase at doy 196 (peak). peak ~1.5, trough (half year) ~0.5.
    assert np.isclose(m._seasonal_weight(196.0, 196.0, 0.5), 1.5)
    assert np.isclose(m._seasonal_weight(196.0 + 182.5, 196.0, 0.5), 0.5, atol=1e-2)
    # never negative (amplitude > 1 clipped)
    assert m._seasonal_weight(0.0, 182.5, 2.0) >= 0.0


def test_spectral_aod_angstrom():
    m = _mod()
    # alpha=1: at 275 nm (=550/2) the AOD is 2x the 550 nm value.
    assert np.isclose(m.spectral_aod_scale(275.0, 1.0), 2.0)
    assert np.isclose(m.spectral_aod_scale(550.0, 1.5), 1.0)
    assert m.spectral_aod_scale(1100.0, 1.0) < 1.0          # longer lambda -> less


def test_read_macv2sp_plumes_from_params():
    m = _mod()
    ds = xr.Dataset({
        "plume_lat": (("plume",), np.array([20.0, -10.0])),
        "plume_lon": (("plume",), np.array([100.0, 20.0])),
        "plume_theta": (("plume",), np.array([0.1, 0.0])),
        "aod_spmx": (("plume",), np.array([0.4, 0.2])),
        "sig_lon_E": (("plume",), np.array([15.0, 12.0])),
        "sig_lon_W": (("plume",), np.array([8.0, 12.0])),
        "sig_lat_N": (("plume",), np.array([6.0, 5.0])),
        "sig_lat_S": (("plume",), np.array([6.0, 5.0])),
        "ftr_weight": (("plume", "feature"), np.array([[0.7, 0.3], [1.0, 0.0]])),
        "years": (("year",), np.array([1850.0, 2000.0])),
        "year_weight": (("year", "plume"), np.array([[0.0, 0.0], [1.0, 0.8]])),
    })
    plumes = m.read_macv2sp_plumes(ds)
    assert len(plumes) == 2
    assert plumes[0]["lon"] == 100.0 and plumes[0]["aod"] == 0.4
    assert len(plumes[0]["features"]) == 2
    # year_weight table interpolates: plume 0 at 1925 -> 0.5
    assert np.isclose(m._year_factor(plumes[0]["year_weight"], 1925), 0.5)


def _full_macv2sp_ds():
    return xr.Dataset({
        "plume_lat": (("plume",), np.array([20.0, -10.0])),
        "plume_lon": (("plume",), np.array([100.0, 20.0])),
        "plume_theta": (("plume",), np.array([0.1, 0.0])),
        "aod_spmx": (("plume",), np.array([0.4, 0.2])),
        "sig_lon_E": (("plume",), np.array([15.0, 12.0])),
        "sig_lon_W": (("plume",), np.array([8.0, 12.0])),
        "sig_lat_N": (("plume",), np.array([6.0, 5.0])),
        "sig_lat_S": (("plume",), np.array([6.0, 5.0])),
        "ftr_weight": (("plume", "feature"), np.array([[0.7, 0.3], [1.0, 0.0]])),
        "years": (("year",), np.array([1850.0, 2000.0])),
        "year_weight": (("year", "plume"), np.array([[0.0, 0.0], [1.0, 0.8]])),
    })


def test_read_macv2sp_missing_centres_raises():
    m = _mod()
    with pytest.raises(ValueError, match="required variable 'plume_lat'"):
        m.read_macv2sp_plumes(xr.Dataset({"aod_spmx": (("p",), [0.1])}))


def test_read_macv2sp_fails_closed_on_missing_geometry():
    """A missing REQUIRED plume parameter raises — never a silent default (that
    would be invented garbage physics)."""
    m = _mod()
    for drop in ("aod_spmx", "sig_lon_E", "sig_lat_N", "plume_theta",
                 "year_weight", "ftr_weight"):
        ds = _full_macv2sp_ds().drop_vars(drop)
        with pytest.raises(ValueError, match=f"required variable {drop!r}"):
            m.read_macv2sp_plumes(ds)


def test_year_weight_oriented_by_dim_name_not_shape():
    """A SQUARE year_weight declared transposed ((plume, year) instead of the
    canonical (year, plume)) is oriented by DIMENSION NAME, so both give the same
    per-plume annual table — the shape-based guard would have silently swapped
    them (codex)."""
    m = _mod()
    canon = m.read_macv2sp_plumes(_full_macv2sp_ds())     # (year, plume)
    ds_t = _full_macv2sp_ds()
    yw = ds_t["year_weight"].values                        # (year, plume)
    ds_t["year_weight"] = (("plume", "year"), yw.T)        # transposed decl
    trans = m.read_macv2sp_plumes(ds_t)
    for a, b in zip(canon, trans):
        assert np.allclose(a["year_weight"][1], b["year_weight"][1])
    # and it is the RIGHT table: plume 0 interpolates 0->1 over 1850..2000.
    assert np.isclose(m._year_factor(canon[0]["year_weight"], 1925), 0.5)


def test_missing_plume_dim_raises():
    """A per-plume variable that does not carry the plume dimension raises rather
    than being silently mis-axised."""
    m = _mod()
    ds = _full_macv2sp_ds()
    ds["aod_spmx"] = (("year",), np.array([0.4, 0.2]))     # wrong (non-plume) dim
    with pytest.raises(ValueError, match="do not include the plume dimension"):
        m.read_macv2sp_plumes(ds)


def test_nonfinite_or_zero_width_raises():
    """A zero/NaN plume width would divide into NaN AOD; evaluate_plume_aod must
    reject it (np.clip does not repair NaN)."""
    m = _mod()
    lon, lat = np.array([100.0]), np.array([20.0])
    for bad in (0.0, np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="must be positive and finite"):
            m.evaluate_plume_aod(lon, lat, [_one_plume(sig_lon_e=bad)], 2000, 196.5)
    with pytest.raises(ValueError, match="aod.*must be finite"):
        m.evaluate_plume_aod(lon, lat, [_one_plume(aod=np.nan)], 2000, 196.5)
    # A non-finite reaching via the seasonal amplitude (widths+aod fine) is caught
    # by the final-output backstop, not the per-width guard.
    with pytest.raises(ValueError, match="non-finite"):
        m.evaluate_plume_aod(lon, lat,
                             [_one_plume(features=[(1.0, 0.0, 0.0, np.nan)])],
                             2000, 196.5)


def test_reader_rejects_nonfinite_or_unsorted_years():
    m = _mod()
    ds = _full_macv2sp_ds()
    ds["years"] = (("year",), np.array([1850.0, np.nan]))
    with pytest.raises(ValueError, match="years contains non-finite"):
        m.read_macv2sp_plumes(ds)
    ds2 = _full_macv2sp_ds()
    ds2["years"] = (("year",), np.array([2000.0, 1850.0]))   # descending
    with pytest.raises(ValueError, match="years must be strictly increasing"):
        m.read_macv2sp_plumes(ds2)


def test_climatology_shape_and_nonneg():
    m = _mod()
    mid, lat, lon, aod = m.build_aod_climatology([_one_plume()], 2000, 48, 96)
    assert aod.shape == (12, 48, 96) and mid.shape == (12,)
    assert np.all(aod >= 0.0) and aod.max() > 0.0


def test_roundtrip_through_real_aerosol_loader(tmp_path):
    """gridded MACv2-SP AOD -> NetCDF -> the REAL get_aerosol_at_time returns a
    finite, non-negative zonal AOD at the model latitudes."""
    m = _mod()
    # a single mid-latitude plume with NO seasonal/annual variation, so the
    # field is time-constant and the loader's time interpolation is a no-op.
    mid, lat, lon, aod = m.build_aod_climatology(
        [_one_plume(lat=20.0, aod=0.4)], 2000, 60, 120)
    out = tmp_path / "aerosol_macv2sp.nc"
    m.to_dataset(mid, lat, lon, aod).to_netcdf(out)

    import jax.numpy as jnp
    from legoesm.forcing.external import AerosolConfig, get_aerosol_at_time
    cfg = AerosolConfig(enabled=True, path=str(out))
    # get_aerosol_at_time takes lat_grid in RADIANS (it degrees() it internally).
    res = get_aerosol_at_time(cfg, float(mid[6]), lat_grid=jnp.radians(lat))
    arr = np.asarray(res)
    assert arr.shape == lat.shape
    assert np.all(np.isfinite(arr)) and np.all(arr >= 0.0) and arr.max() > 0.0
    # Field is time-constant (no seasonal/annual variation) and queried at the
    # exact July record, so the loader's zonal aggregation at the file latitudes
    # is EXACTLY the longitude mean of the plume field — proving it consumes the
    # aod (not the reference) and that lon-averaging + the plume structure (peak
    # at the plume latitude, 20 N) survive end to end.
    zonal = aod[6].mean(axis=1)
    assert np.allclose(arr, zonal, atol=1e-6)
    assert abs(lat[int(np.argmax(arr))] - 20.0) < 3.0
