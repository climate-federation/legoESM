"""Unit tests for the CMIP6 GHG forcing adapter
(``scripts/data/adapt_cmip6_ghg.py``).

The adapter merges the SEPARATE per-gas input4MIPs concentration files into the
ONE ``CO2/CH4/N2O/CFC_11/CFC_12`` file the AMIP GHG loader wants.  The core
(``merge_ghg`` / ``_extract_gas_series`` / ``_to_fractional_year``) is pure
(xarray in, xarray out), so it is tested with SYNTHETIC per-gas Datasets — no
network, no real files.  Coverage includes the REAL file shape that bit the
first draft: a labelled ``sector=(Global, NH, SH)`` region dim (must select
Global, not average the hemispheres) and the dotted ``"1.e-6"`` units spelling
the UoM files use (must canonicalise to the loader-recognised ``"1e-6"``).  The
headline test round-trips synthetic inputs -> merge -> NetCDF -> the REAL
``_load_ghg_annual_file``; it exercises variable naming, dims, time decoding and
unit normalisation for all five gases, but is NOT a substitute for one live
end-to-end run against actual downloaded UoM files.
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
_PY = _REPO / "scripts" / "data" / "adapt_cmip6_ghg.py"


def _mod():
    spec = importlib.util.spec_from_file_location("adapt_cmip6_ghg", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _gas_ds(cf_var: str, years, values, *, units="1"):
    """A synthetic per-gas input4MIPs-style Dataset: mole_fraction_of_*_in_air
    on a mid-year noleap cftime axis with a singleton sector dim."""
    times = [cftime.DatetimeNoLeap(int(y), 7, 2) for y in years]  # ~mid-year
    da = xr.DataArray(
        np.asarray(values, dtype=np.float64)[:, None],
        dims=("time", "sector"), coords={"time": times},
        attrs={"units": units},
    )
    return xr.Dataset({cf_var: da})


def _synthetic_per_gas(years):
    m = _mod()
    vals = {"CO2": 400.0, "CH4": 1800.0, "N2O": 320.0,
            "CFC_11": 230.0, "CFC_12": 500.0}
    units = {"CO2": "1e-6", "CH4": "1e-9", "N2O": "1e-9",
             "CFC_11": "1e-12", "CFC_12": "1e-12"}
    per = {}
    for gas, cf in m.GAS_TO_CFVAR.items():
        # a distinct linear ramp per gas so a mis-mapping would be caught
        series = vals[gas] + np.arange(len(years), dtype=float)
        per[gas] = _gas_ds(cf, years, series, units=units[gas])
    return per, vals, units


def test_to_fractional_year_cftime_and_numeric():
    m = _mod()
    ds = _gas_ds("x", [1990, 1991], [1.0, 2.0])
    fy = m._to_fractional_year(ds["time"])
    # mid-year (doy 183 of 365, noleap) -> ~1990.5
    assert 1990.4 < fy[0] < 1990.6 and 1991.4 < fy[1] < 1991.6
    # already-numeric axis passes through unchanged
    num = xr.DataArray(np.array([1979.5, 1980.5]), dims=("time",))
    assert np.allclose(m._to_fractional_year(num), [1979.5, 1980.5])


def test_merge_ghg_produces_loader_schema():
    m = _mod()
    per, _, _ = _synthetic_per_gas([1990, 1991, 1992])
    merged = m.merge_ghg(per)
    assert set(m.GAS_TO_CFVAR) <= set(merged.data_vars)
    assert merged["CO2"].dims == ("time", "lat", "lon")
    assert merged.sizes["lat"] == 1 and merged.sizes["lon"] == 1
    assert merged["time"].attrs["units"] == "year as %Y.%f"
    assert np.allclose(merged["time"].values,
                       [1990.5, 1991.5, 1992.5], atol=0.01)
    # each gas keeps its native units attr (loader normalises)
    assert merged["CO2"].attrs["units"] == "1e-6"


def test_merge_ghg_missing_gas_raises():
    m = _mod()
    per, _, _ = _synthetic_per_gas([2000, 2001])
    del per["CFC_12"]
    with pytest.raises(ValueError, match="missing gas"):
        m.merge_ghg(per)


def test_merge_ghg_mismatched_time_axis_raises():
    m = _mod()
    per, _, _ = _synthetic_per_gas([2000, 2001, 2002])
    # give CH4 a different axis -> hard error (no silent reindex)
    per["CH4"] = _gas_ds(m.GAS_TO_CFVAR["CH4"], [2000, 2001, 2003],
                         [1.0, 2.0, 3.0], units="1e-9")
    with pytest.raises(ValueError, match="time axis differs"):
        m.merge_ghg(per)


def test_roundtrip_through_real_loader(tmp_path):
    """merge -> NetCDF -> the REAL _load_ghg_annual_file recovers the series."""
    m = _mod()
    years = [1979, 1980, 1981, 1982]
    per, base, _ = _synthetic_per_gas(years)
    merged = m.merge_ghg(per)
    out = tmp_path / "ghg_cmip6.nc"
    merged.to_netcdf(out)

    from legoesm.forcing.external import _load_ghg_annual_file
    loaded_years, data = _load_ghg_annual_file(str(out))
    assert np.allclose(loaded_years, [1979.5, 1980.5, 1981.5, 1982.5], atol=0.01)
    # loader returns mole fractions (native units normalised); the FIRST-year
    # CO2 synthetic value 400 ppmv -> 400e-6 mol/mol.
    assert np.isclose(float(np.asarray(data["CO2"]).reshape(-1)[0]),
                      base["CO2"] * 1.0e-6, rtol=1e-6)
    assert np.isclose(float(np.asarray(data["CH4"]).reshape(-1)[0]),
                      base["CH4"] * 1.0e-9, rtol=1e-6)


def _gas_ds_gmnhsh(cf_var, years, global_series, *, units="1.e-6"):
    """A REAL-shaped UoM concentration Dataset: an INTEGER ``sector`` coord
    (0,1,2) whose region names live in the ``ids`` attribute (the actual UoM
    encoding, NOT string coordinate labels), with distinct per-region values so
    averaging != selecting Global, and a dotted-exponent units spelling."""
    times = [cftime.DatetimeNoLeap(int(y), 7, 2) for y in years]
    g = np.asarray(global_series, dtype=np.float64)
    data = np.stack([g, g + 50.0, g - 50.0], axis=1)  # Global, NH, SH
    da = xr.DataArray(
        data, dims=("time", "sector"),
        coords={"time": times,
                "sector": ("sector", np.array([0, 1, 2]),
                           {"ids": "0: Global; 1: Northern Hemisphere; "
                                   "2: Southern Hemisphere"})},
        attrs={"units": units})
    return xr.Dataset({cf_var: da})


def test_canonical_units_dotted_exponent_and_nonpower():
    m = _mod()
    assert m._canonical_units("1.e-6") == "1e-6"
    assert m._canonical_units("1.0e-9") == "1e-9"
    assert m._canonical_units("1e-12") == "1e-12"
    assert m._canonical_units("mol mol-1") == "mol mol-1"   # named -> untouched
    assert m._canonical_units("ppmv") == "ppmv"
    # non-powers-of-ten must NOT collapse to 1e-N (atol=0.0 guard).
    assert m._canonical_units("2e-9") == "2e-9"
    assert m._canonical_units("2.0e-12") == "2.0e-12"


def test_selects_global_via_integer_sector_ids_attr():
    """The real UoM shape — integer sector 0/1/2 + names in the ``ids`` attr —
    must select GLOBAL (index of 'Global' in ids), not average or index-0-guess."""
    m = _mod()
    # NH/SH offset so index-0-fallback and the mean both differ from Global only
    # if selection is wrong; here Global is index 0 but we resolve it by NAME.
    ds = _gas_ds_gmnhsh(m.GAS_TO_CFVAR["CO2"], [2000, 2001], [400.0, 401.0])
    assert m._global_index(ds, "sector") == 0
    years, values, units = m._extract_gas_series(ds, m.GAS_TO_CFVAR["CO2"])
    assert np.allclose(values, [400.0, 401.0]) and units == "1e-6"


def test_selects_global_when_not_first_sector():
    """Global resolved by NAME even when it is NOT the first sector index."""
    m = _mod()
    times = [cftime.DatetimeNoLeap(y, 7, 2) for y in (2000, 2001)]
    g = np.array([354.0, 355.0])
    # order: NH, Global, SH -> Global is index 1; a naive index-0 would take NH.
    data = np.stack([g + 50.0, g, g - 50.0], axis=1)
    ds = xr.Dataset({m.GAS_TO_CFVAR["CO2"]: xr.DataArray(
        data, dims=("time", "sector"),
        coords={"time": times, "sector": ("sector", np.array([0, 1, 2]),
                {"ids": "0: Northern Hemisphere; 1: Global; 2: Southern "
                        "Hemisphere"})}, attrs={"units": "1.e-6"})})
    assert m._global_index(ds, "sector") == 1
    _, values, _ = m._extract_gas_series(ds, m.GAS_TO_CFVAR["CO2"])
    assert np.allclose(values, g)   # Global (index 1), not NH (index 0)


def test_coded_sector_value_maps_to_position_not_index():
    """A ``N: Global`` code is mapped to the coord POSITION of value N, never
    treated as a positional index (codes != positions)."""
    m = _mod()
    times = [cftime.DatetimeNoLeap(y, 7, 2) for y in (2000, 2001)]
    g = np.array([354.0, 355.0])
    data = np.stack([g, g + 50.0, g - 50.0], axis=1)
    # coord VALUES 10/20/30 (not 0/1/2); ids says Global is CODE 10 -> position 0
    ds = xr.Dataset({m.GAS_TO_CFVAR["CO2"]: xr.DataArray(
        data, dims=("time", "sector"),
        coords={"time": times, "sector": ("sector", np.array([10, 20, 30]),
                {"ids": "10: Global; 20: Northern Hemisphere; 30: Southern "
                        "Hemisphere"})}, attrs={"units": "1.e-6"})})
    assert m._global_index(ds, "sector") == 0   # position of code 10, not 10
    _, values, _ = m._extract_gas_series(ds, m.GAS_TO_CFVAR["CO2"])
    assert np.allclose(values, g)


def test_roundtrip_real_shape_gmnhsh_units(tmp_path):
    """End-to-end with the REAL file shape: (Global,NH,SH) sector + '1.e-6'
    units -> merge -> real loader recovers the GLOBAL mole fractions for every
    gas (CO2/CH4/N2O/CFC_11/CFC_12)."""
    m = _mod()
    years = [1990, 1991]
    base = {"CO2": 354.0, "CH4": 1700.0, "N2O": 310.0,
            "CFC_11": 260.0, "CFC_12": 480.0}
    uexp = {"CO2": "1.e-6", "CH4": "1.e-9", "N2O": "1.e-9",
            "CFC_11": "1.e-12", "CFC_12": "1.e-12"}
    scale = {"CO2": 1e-6, "CH4": 1e-9, "N2O": 1e-9,
             "CFC_11": 1e-12, "CFC_12": 1e-12}
    per = {g: _gas_ds_gmnhsh(m.GAS_TO_CFVAR[g], years,
                             [base[g], base[g] + 1.0], units=uexp[g])
           for g in m.GAS_TO_CFVAR}
    out = tmp_path / "ghg_real.nc"
    m.merge_ghg(per).to_netcdf(out)

    from legoesm.forcing.external import _load_ghg_annual_file
    _, data = _load_ghg_annual_file(str(out))
    for g in m.GAS_TO_CFVAR:      # every gas normalised from the GLOBAL series
        got = float(np.asarray(data[g]).reshape(-1)[0])
        assert np.isclose(got, base[g] * scale[g], rtol=1e-6), g


def test_main_requires_input(tmp_path):
    m = _mod()
    assert m.main(["--out", str(tmp_path / "x.nc")]) == 2  # no --in-dir/--co2…
