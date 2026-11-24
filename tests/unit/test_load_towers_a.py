"""Direct checks of the Cabauw, ARM SGP and North Slope loaders on small synthetic files."""
import importlib.util
import pathlib
import sys

import netCDF4
import numpy as np
import pytest
import xarray as xr

_DIR = pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"
sys.path.insert(0, str(_DIR))
_SPEC = importlib.util.spec_from_file_location("load_towers_a", _DIR / "load_towers_a.py")
ld = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ld)


def cesar_file(path, names, n=6):
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", n)
        d.createDimension("nv", 2)
        t = d.createVariable("time", "f8", ("time",))
        t.units = "hours since 2015-07-01 00:00:00 0:00"
        t[:] = np.arange(n) / 6.0
        b = d.createVariable("time_bnds", "f8", ("time", "nv"))
        b[:, 0], b[:, 1] = np.arange(n) / 6.0, (np.arange(n) + 1) / 6.0
        for k, v in names.items():
            d.createVariable(k, "f8", ("time",))[:] = v


def test_cesar_stamp_is_the_end_of_the_interval(tmp_path):
    p = tmp_path / "a.nc"
    cesar_file(p, {"H": np.arange(6.0)})
    t, v = ld.read_cabauw([str(p)], ("H",))
    start = int(np.datetime64("2015-07-01T00:00:00", "s").astype(np.int64))
    assert t[0] == start + 600 and t[-1] == start + 3600
    t30, h = ld.half_hour(t, v["H"])
    assert t30.tolist() == [start + 1800, start + 3600] and h.tolist() == [1.0, 4.0]


def test_cabauw_late_flux_keeps_only_the_sonic_source_and_no_latent_heat(tmp_path):
    a, b = tmp_path / "lb1.nc", tmp_path / "lc1.nc"
    cesar_file(a, {"HSON": np.ones(6), "USTAB": np.ones(6), "LEED": np.ones(6)})
    idx = np.array([2, 2, 2, 5, 6, 2.0])
    cesar_file(b, {"H": np.full(6, 9.0), "UST": np.full(6, 9.0), "IH": idx, "IUST": idx})
    _, h, ustar, le = ld.cabauw_flux([str(a)], [str(b)])
    assert h[:6].tolist() == [1.0] * 6 and np.isnan(le[6:]).all()
    assert h[6:9].tolist() == [9.0] * 3 and np.isnan(h[9:11]).all() and np.isnan(ustar[9:11]).all()


def test_stress_not_ustar_is_averaged():
    t = np.array([600, 1200, 1800])
    _, u = ld.stress_mean(t, np.array([0.1, 0.1, 0.4]))
    assert u[0] == pytest.approx(np.sqrt((0.01 + 0.01 + 0.16) / 3))


def test_highest_version_wins_per_month():
    paths = ["x_v1.2_202105.nc", "x_v1.3_202105.nc", "x_v1.2_202104.nc"]
    assert ld.latest_versions(paths) == ["x_v1.2_202104.nc", "x_v1.3_202105.nc"]


def test_on_axis_leaves_gaps_missing():
    out = ld.on_axis(np.array([1, 2, 3]), np.array([1, 3, 9]), np.array([5.0, 6.0, 7.0]))
    assert out[0] == 5.0 and np.isnan(out[1]) and out[2] == 6.0


def test_clock_lag_peaks_where_the_series_are_aligned():
    rng = np.random.default_rng(0)
    x = np.convolve(rng.normal(size=4000), np.ones(3), "same")
    ds = xr.Dataset({"wind": (("time", "u"), x[:, None]), "ustar": (("time", "f"), x[:, None])})
    lags = ld.clock_lag(ds)
    assert max(lags, key=lags.get) == 0
    shifted = ds.assign(ustar=(("time", "f"), np.roll(x, 1)[:, None]))
    lags = ld.clock_lag(shifted)
    assert max(lags, key=lags.get) == -1


def arm_tower_file(path, base, bad_minute=None, n=30):
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", n)
        d.createDimension("height", 2)
        d.qc_bit_1_description = "Value is equal to missing_value."
        d.qc_bit_1_assessment = "Bad"
        d.createVariable("base_time", "i4", ())[...] = base
        d.createVariable("time_offset", "f8", ("time",))[:] = 60.0 * np.arange(1, n + 1)
        d.createVariable("height", "i4", ("height",))[:] = [2, 10]
        for name, value in (("temp_mean", 1.0), ("wspd_vec_mean", 5.0), ("wspd_arith_mean", 5.1), ("wdir_vec_mean", 90.0)):
            d.createVariable(name, "f8", ("time", "height"))[:] = value
            q = np.zeros((n, 2), dtype="i4")
            if bad_minute is not None:
                q[bad_minute, 1] = 1
            d.createVariable("qc_" + name, "i4", ("time", "height"))[:] = q


def test_north_slope_tower_block_needs_all_thirty_good_minutes(tmp_path):
    a, b = tmp_path / "a.cdf", tmp_path / "b.cdf"
    arm_tower_file(a, 1800 * 1000)
    arm_tower_file(b, 1800 * 1001, bad_minute=3)
    t30, z, wind, temp = ld.nsa_tower([str(a), str(b)])
    assert t30.tolist() == [1800 * 1001, 1800 * 1002] and z.tolist() == [2.0, 10.0]
    assert wind[0].tolist() == pytest.approx([5.0, 5.0]) and temp[0, 0] == pytest.approx(1.0 + ld.constants.T_freeze)
    assert wind[1, 0] == pytest.approx(5.0) and np.isnan(wind[1, 1]) and np.isnan(temp[1, 1])


def test_dead_anemometer_zero_is_not_calm(tmp_path):
    p = tmp_path / "a.cdf"
    arm_tower_file(p, 1800 * 1000)
    with netCDF4.Dataset(p, "a") as d:
        d["wspd_vec_mean"][4, 0] = 0.0
        d["wspd_arith_mean"][4, 0] = -9999.0
    _, v, _ = ld.read_arm_tower(str(p))
    assert np.isnan(v["wspd_vec_mean"][4, 0]) and v["wspd_vec_mean"][5, 0] == 5.0


def test_sgp_keeps_ecor_first_and_never_reads_ebbr(tmp_path):
    n = 4
    t = np.datetime64("2015-07-01T00:30:00", "s") + 1800 * np.arange(n)
    h = np.array([2.0, 4.0, 25.0, 60.0])
    two = lambda v: (("time", "height"), np.full((n, 4), v))  # noqa: E731
    one = lambda v: ("time", np.full(n, v))  # noqa: E731
    ds = xr.Dataset({"u": two(3.0), "v": two(4.0), "T": (("time", "height"), np.tile([10.0, 11.0, 12.0, 13.0], (n, 1))), "ustar_level": two(0.2), "H": two(50.0),
                     "LE": two(60.0), "ustar": one(0.3), "H_surface": one(70.0), "LE_surface": one(80.0),
                     "H_ebbr": one(999.0), "LE_ebbr": one(999.0), "surface_pressure": one(9.6e4)},
                    coords={"time": t, "height": h})
    p = tmp_path / "u.nc"
    ds.to_netcdf(p)
    out = ld.load_sgp(str(p))
    assert out["h"][0].tolist() == [70.0, 50.0, 50.0, 50.0] and out["ustar"][0, 0] == pytest.approx(0.3)
    assert out["wind"][0].tolist() == [5.0, 5.0, 5.0] and out["z_f"][0].tolist() == [3.0, 4.0, 25.0, 60.0]
    assert out["temp"][0] == pytest.approx(np.array([10.0, 12.0, 13.0]) + ld.constants.T_freeze)
