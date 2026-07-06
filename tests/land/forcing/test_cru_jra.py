"""Unit tests for the CRU-JRA (CLM datm) land forcing reader (M1).

Covers: the synthetic fallback, the NetCDF reader against an in-test CLM-format
fixture (including the dual start/midpoint time-stamp convention and time
subsetting), the regrid onto model columns, and the variable map ->
``AtmToSurface`` with its derived fields.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.forcing import cru_jra

jax.config.update("jax_enable_x64", True)

xr = pytest.importorskip("xarray")


# --------------------------------------------------------------------------
# fixtures: a tiny CLM-format CRU-JRA year on disk
# --------------------------------------------------------------------------
_NT, _NLAT, _NLON = 4, 6, 8


def _write_clm_fixture(tmp_path, year=2023, prefix=cru_jra.CRUJRA_FILE_PREFIX):
    lat = np.linspace(-87.5, 87.5, _NLAT)
    lon = np.linspace(0.0, 360.0, _NLON, endpoint=False)
    units = f"days since {year}-01-01"
    # Solar stamped at interval START (0, 6, 12, 18 h); others at MIDPOINT.
    t_solar = np.array([0.0, 0.25, 0.5, 0.75])
    t_state = t_solar + 0.125
    rng = np.random.default_rng(0)

    def _da(values, t):
        return xr.DataArray(
            values, dims=("time", "lat", "lon"),
            coords={"time": ("time", t, {"units": units, "calendar": "noleap"}),
                    "lat": lat, "lon": lon},
        )

    shape = (_NT, _NLAT, _NLON)
    solr = xr.Dataset({"FSDS": _da(rng.uniform(0, 400, shape), t_solar)})
    prec = xr.Dataset({"PRECTmms": _da(rng.uniform(0, 1e-4, shape), t_state)})
    tpqwl = xr.Dataset({
        "TBOT": _da(rng.uniform(250, 300, shape), t_state),
        "PSRF": _da(rng.uniform(9.5e4, 1.0e5, shape), t_state),
        "QBOT": _da(rng.uniform(1e-3, 1e-2, shape), t_state),
        "WIND": _da(rng.uniform(0.5, 8.0, shape), t_state),
        "FLDS": _da(rng.uniform(200, 400, shape), t_state),
    })
    solr.to_netcdf(tmp_path / f"{prefix}.Solr.{year}.nc")
    prec.to_netcdf(tmp_path / f"{prefix}.Prec.{year}.nc")
    tpqwl.to_netcdf(tmp_path / f"{prefix}.TPQWL.{year}.nc")
    return year


# --------------------------------------------------------------------------
# synthetic fallback
# --------------------------------------------------------------------------
def test_synthetic_shapes_and_time_axes():
    f = cru_jra.synthetic_land_forcing(2000, n_time=4)
    assert f.tbot.shape == (4, cru_jra.CRUJRA_NLAT, cru_jra.CRUJRA_NLON)
    assert f.fsds.dtype == np.float32
    # solar stamped at interval start, state at midpoint -> half-step offset.
    half_step = 0.5 * cru_jra.CRUJRA_FREQ_HOURS * 3600.0
    assert np.allclose(f.time_s - f.time_s_solar, half_step)
    assert np.all(f.tbot > 200.0) and np.all(f.tbot < 320.0)
    assert np.all(f.qbot >= 0.0)


def test_load_falls_back_to_synthetic(tmp_path):
    f = cru_jra.load_cru_jra(1999, data_dir=tmp_path, allow_synthetic=True)
    assert isinstance(f, cru_jra.LandForcing)
    with pytest.raises(FileNotFoundError):
        cru_jra.load_cru_jra(1999, data_dir=tmp_path, allow_synthetic=False)


# --------------------------------------------------------------------------
# NetCDF reader
# --------------------------------------------------------------------------
def test_read_clm_fixture(tmp_path):
    year = _write_clm_fixture(tmp_path)
    f = cru_jra.read_crujra_year(tmp_path, year)
    assert f.tbot.shape == (_NT, _NLAT, _NLON)
    assert f.lat.shape == (_NLAT,) and f.lon.shape == (_NLON,)
    # dual time axes: solar at start, state at midpoint (half a 6-h step apart).
    half_step = 0.5 * cru_jra.CRUJRA_FREQ_HOURS * 3600.0
    assert np.allclose(f.time_s - f.time_s_solar, half_step)
    # units carried through correctly (seconds since year start, 6-hourly).
    assert np.isclose(f.time_s_solar[1] - f.time_s_solar[0],
                      cru_jra.CRUJRA_FREQ_HOURS * 3600.0)


def test_read_with_year_suffix(tmp_path):
    # TRENDY glade files are named <prefix>.<stream>.<year>_cdf5.nc
    prefix = "clmforc.TRENDY.c2023_0.5x0.5"
    lat = np.linspace(-87.5, 87.5, _NLAT); lon = np.linspace(0.0, 360.0, _NLON, endpoint=False)
    units = "days since 2022-01-01"
    t_solar = np.array([0.0, 0.25, 0.5, 0.75]); t_state = t_solar + 0.125
    shape = (_NT, _NLAT, _NLON)

    def _da(vals, t):
        return xr.DataArray(vals, dims=("time", "lat", "lon"),
                            coords={"time": ("time", t, {"units": units}), "lat": lat, "lon": lon})
    xr.Dataset({"FSDS": _da(np.ones(shape), t_solar)}).to_netcdf(
        tmp_path / f"{prefix}.Solr.2022_cdf5.nc")
    xr.Dataset({"PRECTmms": _da(np.zeros(shape), t_state)}).to_netcdf(
        tmp_path / f"{prefix}.Prec.2022_cdf5.nc")
    xr.Dataset({"TBOT": _da(np.full(shape, 280.0), t_state),
                "PSRF": _da(np.full(shape, 1e5, dtype=float), t_state),
                "QBOT": _da(np.full(shape, 5e-3), t_state),
                "WIND": _da(np.full(shape, 3.0), t_state),
                "FLDS": _da(np.full(shape, 300.0), t_state)}).to_netcdf(
        tmp_path / f"{prefix}.TPQWL.2022_cdf5.nc")

    f = cru_jra.read_crujra_year(tmp_path, 2022, prefix=prefix, suffix="_cdf5")
    assert f.tbot.shape == (_NT, _NLAT, _NLON)
    assert np.allclose(f.tbot, 280.0)
    # load_cru_jra + stage_forcing also honour the suffix (no synthetic fallback)
    g = cru_jra.load_cru_jra(2022, data_dir=tmp_path, prefix=prefix, suffix="_cdf5",
                             allow_synthetic=False)
    assert np.allclose(g.wind, 3.0)


def test_read_time_subset(tmp_path):
    year = _write_clm_fixture(tmp_path)
    f = cru_jra.read_crujra_year(tmp_path, year, time_indices=[0, 2])
    assert f.tbot.shape == (2, _NLAT, _NLON)
    assert f.time_s_solar.shape == (2,)


def test_read_rejects_bad_time_units(tmp_path):
    year = _write_clm_fixture(tmp_path)
    # Rewrite the Solr stream with non-days time units and confirm the guard fires.
    lat = np.linspace(-87.5, 87.5, _NLAT)
    lon = np.linspace(0.0, 360.0, _NLON, endpoint=False)
    bad = xr.Dataset({"FSDS": xr.DataArray(
        np.zeros((_NT, _NLAT, _NLON)), dims=("time", "lat", "lon"),
        coords={"time": ("time", [0.0, 0.25, 0.5, 0.75],
                          {"units": "hours since 2023-01-01"}),
                "lat": lat, "lon": lon})})
    bad.to_netcdf(tmp_path / f"{cru_jra.CRUJRA_FILE_PREFIX}.Solr.{year}.nc")
    with pytest.raises(ValueError, match="time units"):
        cru_jra.read_crujra_year(tmp_path, year)


# --------------------------------------------------------------------------
# regrid + variable map
# --------------------------------------------------------------------------
def _target_columns():
    # A handful of model columns (radians).
    lat = np.radians(np.array([-45.0, 0.0, 30.0, 60.0]))
    lon = np.radians(np.array([10.0, 100.0, 200.0, 300.0]))
    return lat, lon


def test_regrid_shapes_and_range(tmp_path):
    f = cru_jra.synthetic_land_forcing(2000, n_time=4)
    lat, lon = _target_columns()
    w = cru_jra.build_forcing_weights(f, lat, lon)
    cols = cru_jra.regrid_forcing(f, w)
    assert cols.tbot.shape == (4, lat.size)
    # IDW stays within the source range (no overshoot for a smooth field).
    assert cols.tbot.min() >= f.tbot.min() - 1.0
    assert cols.tbot.max() <= f.tbot.max() + 1.0


def test_forcing_to_atm_surface(tmp_path):
    f = cru_jra.synthetic_land_forcing(2000, n_time=4)
    lat, lon = _target_columns()
    w = cru_jra.build_forcing_weights(f, lat, lon)
    cols = cru_jra.regrid_forcing(f, w)

    atm = cru_jra.forcing_to_atm_surface(cols, lat, lon, model_time_s=float(cols.time_s[1]))
    assert isinstance(atm, AtmToSurface)
    for name in ("sw_down", "lw_down", "T_lowest", "q_lowest", "rho_lowest"):
        v = np.asarray(getattr(atm, name))
        assert v.shape == (lat.size,)
        assert np.all(np.isfinite(v))
    # physical sanity
    assert np.all(np.asarray(atm.rho_lowest) > 0.8)
    assert np.all(np.asarray(atm.rho_lowest) < 1.5)
    assert np.all(np.asarray(atm.cos_zenith) >= 0.0)
    assert np.all(np.asarray(atm.cos_zenith) <= 1.0 + 1e-6)
    # scalar wind -> zonal component, zero meridional
    assert np.allclose(np.asarray(atm.v_lowest), 0.0)
    assert np.allclose(np.asarray(atm.u_lowest), np.asarray(cols.wind[1]))
    # snow never exceeds total precip
    assert np.all(np.asarray(atm.precip_snow) <= np.asarray(atm.precip_total) + 1e-12)
    assert np.all(np.asarray(atm.has_radiation) == 1.0)
    assert np.asarray(atm.has_radiation).shape == (lat.size,)


def test_snow_partition_sign():
    # Below freezing -> all snow; well above -> all rain.
    cold = float(cru_jra._snow_fraction(jnp.asarray(constants.T_freeze - 5.0),
                                        ramp_k=cru_jra._SNOW_RAIN_RAMP_K))
    warm = float(cru_jra._snow_fraction(jnp.asarray(constants.T_freeze + 5.0),
                                        ramp_k=cru_jra._SNOW_RAIN_RAMP_K))
    assert cold == 1.0
    assert warm == 0.0


def test_nearest_time_selection_uses_both_clocks(tmp_path):
    f = cru_jra.synthetic_land_forcing(2000, n_time=4)
    lat, lon = _target_columns()
    cols = cru_jra.regrid_forcing(f, cru_jra.build_forcing_weights(f, lat, lon))
    # request a time near the 3rd state stamp; solar index should track its own clock
    target = float(cols.time_s[2])
    it = cru_jra._time_index(cols.time_s, target)
    isol = cru_jra._time_index(cols.time_s_solar, target)
    assert it == 2
    # solar stamped half a step earlier, so nearest solar to a midpoint can be the
    # same index or the next — assert it is a valid, in-range index
    assert 0 <= isol < f.time_s_solar.size


# --------------------------------------------------------------------------
# opt-in: real example files on the developer's machine
# --------------------------------------------------------------------------
_REAL_DIR = os.environ.get("CRUJRA_EXAMPLE_DIR", "/Users/linnia/Desktop")
_REAL_YEAR = 2023


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(_REAL_DIR, f"{cru_jra.CRUJRA_FILE_PREFIX}.Solr.{_REAL_YEAR}.nc")
    ),
    reason="real CRU-JRA example files not present",
)
def test_real_example_files_first_steps():
    f = cru_jra.read_crujra_year(_REAL_DIR, _REAL_YEAR, time_indices=list(range(4)))
    assert f.tbot.shape == (4, cru_jra.CRUJRA_NLAT, cru_jra.CRUJRA_NLON)
    # CLM streams are SI already; sanity-check ranges.
    assert np.nanmin(f.tbot) > 180.0 and np.nanmax(f.tbot) < 340.0
    assert np.nanmin(f.psrf) > 4.0e4 and np.nanmax(f.psrf) < 1.1e5
    assert np.nanmin(f.fsds) >= 0.0
    # dual time-stamp convention holds in the real data
    half_step = 0.5 * cru_jra.CRUJRA_FREQ_HOURS * 3600.0
    assert np.allclose(f.time_s - f.time_s_solar, half_step)
