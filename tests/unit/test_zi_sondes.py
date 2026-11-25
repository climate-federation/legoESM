"""Known-answer checks for sonde boundary-layer depth."""
import pathlib
import sys

import numpy as np

from legoesm import constants

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import zi_sondes as zs  # noqa: E402


def mixed_layer(zi=800.0, jump=4.0, wind=5.0):
    z = np.arange(0.0, 3000.0, 50.0)
    thv = 300.0 + np.where(z > zi, jump + 0.004 * (z - zi), 0.0)
    return z, thv, np.full(z.size, wind), np.zeros(z.size)


def test_bulk_ri_crossing_matches_hand_value():
    z, thv, u, v = mixed_layer()
    got = zs.bulk_ri_height(z, thv, u, v)
    ri = constants.g / 300.0 * (thv - 300.0) * z / 25.0
    k = np.nonzero(ri >= 0.25)[0][0]
    want = z[k - 1] + (0.25 - ri[k - 1]) / (ri[k] - ri[k - 1]) * (z[k] - z[k - 1])
    assert abs(got - want) < 1e-9 and 800.0 < got <= 850.0


def test_bulk_ri_rejects_bad_profiles():
    z, thv, u, v = mixed_layer()
    assert np.isnan(zs.bulk_ri_height(z + 60.0, thv, u, v))
    assert np.isnan(zs.bulk_ri_height(z[::15], thv[::15], u[::15], v[::15]))
    assert np.isnan(zs.bulk_ri_height(z, np.full(z.size, 300.0), u, v))


def test_theta_v_falls_back_to_theta_without_humidity():
    th = zs.theta_v(np.array([280.0]), np.array([80000.0]), np.array([np.nan]))
    np.testing.assert_allclose(th, 280.0 * (1e5 / 8e4) ** constants.kappa)
    assert zs.theta_v(np.array([280.0]), np.array([80000.0]), np.array([0.005]))[0] > th[0]


def test_igra_profile_heights_and_surface_wind():
    rows = ["21 -9999 101900B-9999   172B-9999    27    20    30 ",
            "10 -9999 100000   165B  158B-9999    18    25    80 ",
            "20 -9999  97600 -9999   140B-9999     6 -9999 -9999 ",
            "10 -9999  92500   826B  148B-9999    34   300    40 "]
    z, thv, u, v, moist = zs.igra_profile(rows, 2.0)
    assert z[0] == 0.0 and u[0] == 0.0 and v[0] == 0.0 and moist
    assert 140.0 < z[1] < 180.0 and 750.0 < z[3] < 900.0
    np.testing.assert_allclose([u[1], v[1]], [-8.0 * np.sin(np.deg2rad(25.0)), -8.0 * np.cos(np.deg2rad(25.0))])
    np.testing.assert_allclose([u[3], v[3]], [-4.0 * np.sin(np.deg2rad(300.0)), -4.0 * np.cos(np.deg2rad(300.0))])
    sounding = list(zs._igra_soundings(["#NLM00006260 2021 06 15 00 2330   70 ncdc-gts 521000 51833"] + rows))
    assert sounding[0][0] == np.datetime64("2021-06-14T23:30", "s").astype(np.int64) and len(sounding[0][1]) == 4


def test_crossing_just_below_top_is_kept():
    z = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 4900.0, 5100.0])
    ri_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 0.30])
    u = np.full(z.size, 10.0)
    thv = 300.0 + np.where(z > 0, ri_target * 100.0 * 300.0 / (constants.g * np.maximum(z, 1.0)), 0.0)
    got = zs.bulk_ri_height(z, thv, u, np.zeros(z.size))
    assert abs(got - (4900.0 + 0.01 / 0.06 * 200.0)) < 1e-6


def _table(path, t_end, h):
    import netCDF4
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", t_end.size)
        d.createDimension("level", 1)
        tv = d.createVariable("time", "i8", ("time",))
        tv[:], tv.units = t_end, "seconds since 1970-01-01 00:00:00 UTC"
        d.createVariable("h", "f4", ("time", "level"))[:] = h[:, None]
        d.averaging_minutes = 30


def _era5(path, hours, zi):
    import netCDF4
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("hour", hours.size)
        d.createVariable("hour_time", "i8", ("hour",))[:] = hours
        d.createVariable("zi_hourly", "f4", ("hour",))[:] = zi


def test_compare_uses_hourly_era5_at_launch_and_refuses_gaps(tmp_path):
    t_end = np.arange(1800, 4 * 3600 + 1, 1800)
    _table(tmp_path / "t.nc", t_end, np.where(t_end < 7200, 50.0, -20.0))
    _era5(tmp_path / "e.nc", np.array([0, 3600, 7200, 10800, 14400]), np.array([0.0, 100.0, 0.0, np.nan, 300.0]))
    sondes = [(np.int64(3600), 400.0, True), (np.int64(6300), 500.0, False), (np.int64(12600), 600.0, True)]
    rows = zs.compare(tmp_path / "t.nc", tmp_path / "e.nc", sondes, [52.0] * 3, [5.0] * 3)
    np.testing.assert_allclose(rows[:, 0], [3600, 6300])
    np.testing.assert_allclose(rows[:, 2], [100.0, 25.0])
    np.testing.assert_allclose(rows[:, 3], [50.0, -20.0])


def test_arm_launch_time_adds_time_offset(tmp_path):
    import netCDF4
    with netCDF4.Dataset(tmp_path / "sgp.nc", "w") as d:
        d.createDimension("time", 2)
        d.createVariable("base_time", "i8")[:] = 86400
        d.createVariable("time_offset", "f8", ("time",))[:] = [84480.0, 84482.0]
        for k, val in (("pbl_height_bulk_richardson_pt25", 1315.0), ("qc_pbl_height_bulk_richardson_pt25", 0),
                       ("alt", 315.0), ("lat", 36.6), ("lon", -97.5)):
            d.createVariable(k, "f8")[:] = val
    out = zs.arm_zi(str(tmp_path), 0, 10 ** 6)
    assert out[0][0] == 86400 + 84480 and out[0][1] == 1000.0
