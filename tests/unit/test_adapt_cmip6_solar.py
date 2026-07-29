"""Unit tests for the CMIP6 solar forcing adapter
(``scripts/data/adapt_cmip6_solar.py``).

The adapter bins a raw SOLARIS-HEPPA spectral ``ssi`` (a wavelength-BIN MEAN with
``wlen_bnds``, gregorian calendar) into the deck's TSI + 14-band ``SSI_frac``
schema (noleap ``days since 1850-01-01``).  The binning core is pure (numpy in/
out) and tested against ANALYTIC references (constant bin-mean ssi -> fractions
proportional to the band wavelength widths; ssi confined to one band -> ~1
there, landing in that band's g-points), band edges are checked against the
model's own SW table, the gregorian->noleap Feb-29 drop is exercised, and the
headline test ROUND-TRIPS a REAL-SHAPE file (``wlen_bnds`` + variable-width bins)
through the REAL solar loader.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
cftime = pytest.importorskip("cftime")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PY = _REPO / "scripts" / "data" / "adapt_cmip6_solar.py"


def _mod():
    spec = importlib.util.spec_from_file_location("adapt_cmip6_solar", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _tiling_bins(lo, hi, n):
    """n contiguous bins tiling [lo, hi] -> (bin_lo, bin_hi, centres)."""
    edges = np.linspace(lo, hi, n + 1)
    return edges[:-1], edges[1:], 0.5 * (edges[:-1] + edges[1:])


def test_band_edges_from_model_table():
    m = _mod()
    edges = m.rrtmgsw_band_edges_nm()
    assert edges.shape == (14, 2)
    assert np.all(edges[:, 0] < edges[:, 1])                 # lo < hi per band
    # RRTMG-SW order: the 820-2680 cm-1 near-IR overlap band is LAST.
    assert np.isclose(edges[13, 0], 1e7 / 2680.0, atol=0.5)  # 3731 nm
    assert np.isclose(edges[13, 1], 1e7 / 820.0, atol=1.0)   # 12195 nm


def test_constant_ssi_gives_bandwidth_fractions():
    """Flat bin-mean ssi over bins tiling the spectrum -> SSI_frac equals the
    normalised band widths (histogram integral is exact for bin-mean data)."""
    m = _mod()
    edges = m.rrtmgsw_band_edges_nm()
    lo, hi, _ = _tiling_bins(150.0, 13000.0, 6000)
    ssi = np.ones((2, lo.size))
    frac = m.ssi_to_band_fractions(lo, hi, ssi, edges)
    widths = edges[:, 1] - edges[:, 0]
    assert np.allclose(frac.sum(-1), 1.0)
    assert np.allclose(frac[0], widths / widths.sum(), atol=1e-3)


def test_variable_width_bins_weight_by_overlap():
    """Two unequal bins straddling one band -> the band's flux is
    sum(ssi_bin * overlap_width), NOT a centre trapezoid."""
    m = _mod()
    edges = np.array([[400.0, 500.0]])            # a single synthetic band
    bin_lo = np.array([390.0, 450.0])
    bin_hi = np.array([450.0, 520.0])             # widths 60 and 70
    ssi = np.array([[2.0, 5.0]])                  # bin means
    # overlaps with [400,500]: bin0 400..450=50 ; bin1 450..500=50
    # flux = 2*50 + 5*50 = 350 ; only one band -> frac = 1
    frac = m.ssi_to_band_fractions(bin_lo, bin_hi, ssi, edges)
    assert frac.shape == (1, 1) and np.isclose(frac[0, 0], 1.0)
    # sanity on the raw overlap flux via the internal contract
    band_lo, band_hi = edges[:, 0][:, None], edges[:, 1][:, None]
    overlap = np.clip(np.minimum(band_hi, bin_hi) - np.maximum(band_lo, bin_lo),
                      0.0, None)
    assert np.isclose((ssi @ overlap.T)[0, 0], 350.0)


def test_single_band_ssi_concentrates_there():
    m = _mod()
    edges = m.rrtmgsw_band_edges_nm()
    lo, hi, c = _tiling_bins(150.0, 13000.0, 6000)
    blo, bhi = edges[10]
    ssi = np.where((c >= blo) & (c <= bhi), 1.0, 0.0)[None, :]
    frac = m.ssi_to_band_fractions(lo, hi, ssi, edges)
    assert frac[0, 10] > 0.98 and frac.sum() == pytest.approx(1.0)


def test_bin_bounds_prefers_wlen_bnds_then_binsize_then_centres():
    m = _mod()
    c = np.array([100.0, 200.0, 400.0])
    # explicit bounds
    ds1 = xr.Dataset(coords={"wlen": ("wlen", c)},
                     data_vars={"wlen_bnds": (("wlen", "bnds"),
                                np.array([[90, 110], [110, 300], [300, 500.0]]))})
    lo, hi = m._bin_bounds_nm(ds1, "wlen", "wlen")
    assert np.allclose(lo, [90, 110, 300]) and np.allclose(hi, [110, 300, 500])
    # binsize form
    ds2 = xr.Dataset(coords={"wlen": ("wlen", c)},
                     data_vars={"wlenbinsize": (("wlen",), np.array([20, 40, 60.0]))})
    lo, hi = m._bin_bounds_nm(ds2, "wlen", "wlen")
    assert np.allclose(lo, [90, 180, 370]) and np.allclose(hi, [110, 220, 430])
    # centres-only fallback -> midpoints
    ds3 = xr.Dataset(coords={"wlen": ("wlen", c)})
    lo, hi = m._bin_bounds_nm(ds3, "wlen", "wlen")
    assert np.isclose(hi[0], 150.0) and np.isclose(lo[1], 150.0)  # midpoint shared


def test_gregorian_leap_year_drops_feb29_and_maps_noleap():
    """A gregorian axis over a leap year -> Feb 29 dropped, Mar-Dec mapped by
    noleap calendar date (not shifted by the leap day)."""
    m = _mod()
    # 1852 is a leap year: Feb 28, Feb 29, Mar 1.
    da = xr.DataArray([cftime.DatetimeGregorian(1852, 2, 28),
                       cftime.DatetimeGregorian(1852, 2, 29),
                       cftime.DatetimeGregorian(1852, 3, 1)], dims=("time",))
    days, keep = m._to_days_since_1850(da)
    assert list(keep) == [True, False, True]
    # noleap: Feb 28 -> doy 59 (day 58 since Jan1); Mar 1 -> doy 60 (day 59).
    base = (1852 - 1850) * 365
    assert np.isclose(days[0], base + 58) and np.isclose(days[2], base + 59)


def test_datetime64_leap_year_maps_like_cftime():
    """A numpy datetime64[ns] Gregorian axis (how xarray decodes in-range files)
    must map identically — the .item()-is-an-int trap for datetime64."""
    m = _mod()
    da = xr.DataArray(
        np.array(["1852-02-28", "1852-02-29", "1852-03-01"], dtype="datetime64[ns]"),
        dims=("time",))
    days, keep = m._to_days_since_1850(da)
    assert list(keep) == [True, False, True]
    base = (1852 - 1850) * 365
    assert np.isclose(days[0], base + 58) and np.isclose(days[2], base + 59)


def _real_shape_solar_ds(years, n_bins=2500, tsi=1360.5, calendar="gregorian"):
    """A REAL-shaped SOLARIS file: ssi bin-mean(time, wlen) + wlen_bnds
    (variable-width) + gregorian time (a leap year in the span)."""
    lo, hi, c = _tiling_bins(150.0, 13000.0, n_bins)
    # make the bins non-uniform: stretch the near-IR half
    stretch = np.linspace(1.0, 1.6, n_bins)
    hi = lo + (hi - lo) * stretch
    lo = np.concatenate([[lo[0]], hi[:-1]])          # keep contiguous
    c = 0.5 * (lo + hi)
    cls = getattr(cftime, "DatetimeGregorian")
    times = []
    for y in years:                                  # a few days incl. Feb 29
        for (mo, dy) in ((1, 1), (2, 28), (2, 29), (3, 1)):
            try:
                times.append(cls(int(y), mo, dy))
            except Exception:
                pass
    ssi = 1.0 + 0.2 * np.exp(-((c - 500.0) / 300.0) ** 2)
    ssi = np.repeat(ssi[None, :], len(times), axis=0)
    return xr.Dataset(
        {"tsi": (("time",), np.full(len(times), tsi)),
         "ssi": (("time", "wlen"), ssi, {"units": "W m-2 nm-1",
                                         "cell_methods": "wlen: mean"}),
         "wlen_bnds": (("wlen", "bnds"), np.stack([lo, hi], axis=1))},
        coords={"time": times, "wlen": ("wlen", c, {"units": "nm"})})


def test_adapt_solar_schema_and_leapday_drop():
    m = _mod()
    ds = _real_shape_solar_ds([1852])               # leap year -> Feb 29 present
    out = m.adapt_solar(ds)
    assert out["TSI"].dims == ("time",) and out["SSI_frac"].dims == ("time", "numwl")
    assert out.sizes["numwl"] == 14
    assert out["time"].attrs["units"] == "days since 1850-01-01 00:00:00"
    assert out["time"].attrs["calendar"] == "noleap"
    assert np.allclose(out["SSI_frac"].values.sum(-1), 1.0)
    # 4 input dates but Feb 29 dropped -> 3 output records.
    assert out.sizes["time"] == 3


def test_roundtrip_through_real_solar_loader(tmp_path):
    """Real-shape adapt (incl. a LEAP year) -> NetCDF -> the REAL
    get_solar_forcing_at_time yields TSI + a >14-length g-point spectral weight
    summing to ~1, both at day 0 and at a day PAST the dropped Feb 29 (the axis
    stays consistent across the leap boundary)."""
    m = _mod()
    ds = _real_shape_solar_ds([1850, 1851, 1852])   # 1852 is a leap year
    out = tmp_path / "solar_cmip6.nc"
    m.adapt_solar(ds).to_netcdf(out)

    from legoesm.forcing.external import SolarConfig, get_solar_forcing_at_time
    cfg = SolarConfig(source="spectral_file", path=str(out), tsi_var="TSI",
                      spectral_var="SSI_frac", spectral_band_order="rrtmg_sw",
                      start_year=1850)
    # day 0 (1850-01-01) and 1852-03-01 (noleap: 2*365 + 59 = 789).
    for day in (0.0, 789.0):
        res = get_solar_forcing_at_time(cfg, day)
        assert np.isclose(res["tsi"], 1360.5, rtol=1e-4)
        w = np.asarray(res["solar_fraction_by_gpt"])
        assert w.ndim == 1 and w.shape[0] > 14
        assert np.isclose(float(w.sum()), 1.0, atol=1e-3)


def test_missing_spectral_var_raises():
    m = _mod()
    ds = xr.Dataset({"tsi": (("time",), [1361.0])},
                    coords={"time": [cftime.DatetimeNoLeap(1850, 1, 1)]})
    with pytest.raises(ValueError, match="spectral SSI"):
        m.adapt_solar(ds)
