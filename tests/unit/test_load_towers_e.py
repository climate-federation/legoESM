"""Known-answer checks for the Summit, Netheravon, BLLAST, LIAISE and extra boundary-layer depth loaders."""
import datetime as dt
import pathlib
import sys

import netCDF4
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import load_towers_e as le  # noqa: E402

from legoesm import constants  # noqa: E402


def utc(*a):
    return int(dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp())


def test_summit_ustar_inverts_source_density():
    rho, wu, wts = 0.95, -0.04, 0.01
    h = rho * constants.c_pd * wts
    ust = le.summit_ustar(np.array([-rho * wu]), np.array([0.0]), np.array([h]), np.array([wts]))
    assert ust[0] == pytest.approx(0.2)
    assert np.isnan(le.summit_ustar(np.array([0.04]), np.array([0.0]), np.array([1.0]), np.array([0.0]))[0])


def test_summit_minute_starts_fill_one_half_hour(tmp_path):
    p = tmp_path / "m.nc"
    t0 = utc(2022, 5, 1)
    with netCDF4.Dataset(p, "w") as d:
        d.createDimension("time", 30)
        d.createDimension("index", 1)
        d.createVariable("time", "f8", ("time",))[:] = t0 + 60 * np.arange(30)
        for k in ("wind_speed", "wind_from_direction", "height_above_snow_surface", "qc_flag"):
            d.createVariable(k, "f8", ("time", "index"))[:] = {"wind_speed": 3.0, "wind_from_direction": 90.0,
                                                               "height_above_snow_surface": 2.0, "qc_flag": 1}[k]
    m = le._summit_month(p, {"s": "wind_speed", "d": "wind_from_direction", "z": "height_above_snow_surface", "q": "qc_flag"})
    keys, wind = le.vector_half_hour(m["t"], m["s"], m["d"], m["q"] == 1, 30)
    assert keys.tolist() == [t0 + 1800]
    assert wind[0, 0] == pytest.approx(3.0)


def test_vector_mean_cancels_opposite_winds():
    t = 60 * np.arange(1, 31)
    s = np.full((30, 1), 2.0)
    d = np.where(np.arange(30) % 2, 0.0, 180.0)[:, None]
    _, w = le.vector_half_hour(t, s, d, np.ones((30, 1), bool), 30)
    assert w[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_summit_prefers_v2(tmp_path):
    for day, ver in (("20220501", 1), ("20220501", 2), ("20220502", 1)):
        q = tmp_path / "flux-estimates/ace-flux-1_30min/2022/05"
        q.mkdir(parents=True, exist_ok=True)
        (q / f"ace-flux-1_summit_{day}_flux-estimates_30min_v{ver}.nc").write_text("")
    got = [pathlib.Path(p).name for p in le.summit_flux_files(tmp_path)]
    assert got == ["ace-flux-1_summit_20220501_flux-estimates_30min_v2.nc", "ace-flux-1_summit_20220502_flux-estimates_30min_v1.nc"]


def test_netheravon_selection_and_mean():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    got = le.a_else_b(a, np.array([True, False, False]), b, np.array([False, True, False]))
    np.testing.assert_array_equal(got, [1.0, 20.0, np.nan])
    m = le.mean_of_good([np.array([1.0, 1.0]), np.array([3.0, 5.0])], [np.array([True, False]), np.array([True, False])])
    assert m[0] == pytest.approx(2.0) and np.isnan(m[1])


def test_bllast_clock_and_repeated_stamps():
    t = le.bllast_clock(np.array([20110614] * 4), np.array(["00002.125", "000012.13", "00592.125", "005912.13"]))
    assert (t - utc(2011, 6, 14)).tolist() == [2, 12, 59 * 60 + 2, 59 * 60 + 12]
    t = le.bllast_clock(np.array([20110614]), np.array(["140019"]))
    assert t[0] - utc(2011, 6, 14) == 14 * 3600 + 19
    np.testing.assert_array_equal(le.drop_repeated_stamps(np.array([1, 2, 2, 3])), [True, False, False, True])


def test_ecpack_end_time(tmp_path):
    p = tmp_path / "f.nc"
    with netCDF4.Dataset(p, "w") as d:
        d.createDimension("time", 1)
        for k, v in (("DOY_end", 166), ("hour_end", 0), ("min_end", 30), ("Mean(u)", 3.0), ("u_star", 0.3),
                     ("H_sonic", 10.0), ("LvE_sum", -9999.0)):
            d.createVariable(k, "f8", ("time",))[:] = v
    t, x = le._ecpack(p)
    assert t[0] == utc(2011, 6, 15, 0, 30)
    assert np.isnan(x["LvE_sum"][0]) and x["u_star"][0] == 0.3


def test_nearest_finite_skips_missing_levels_and_heights():
    z_t = np.array([[2.0, 10.0], [2.0, 10.0], [2.0, 10.0]])
    temp = np.array([[270.0, 271.0], [np.nan, 271.0], [270.0, 271.0]])
    got = le.nearest_finite(z_t, temp, np.array([1.0, 1.0, np.nan]))
    np.testing.assert_array_equal(got, [270.0, 271.0, np.nan])


UHF = """System: UHF
 Day: 15-Jun-2011
   HH   MM    SS   Zi (m agl)
   7    40    33     309
  17    52    42     870
"""
LIDAR = """System:Lidar ALS 300
Altitude:595 m
Date Time (UTC) Zi ASL (m)
15/06/2011 07:14:03 816.621
"""
SONDE = """Balloon release time                     2021-07-14 15:52:33 UTC
Station latitude                         41.59 N
Station longitude                        01.03 E

      Time    Height      Pres      Temp        RH      Dewp      MixR       Dir     Speed
         s         m       hPa     deg C         %     deg C      g/kg       deg       m/s

         0         0    977.54     27.35     25.00      5.62      5.85    309.00      1.20
         1         1    977.39     25.66     25.58      4.53      5.42    308.94      1.31
"""


def test_zi_text_parsers():
    assert le.uhf_zi(UHF) == [(utc(2011, 6, 15, 7, 40, 33), 309.0), (utc(2011, 6, 15, 17, 52, 42), 870.0)]
    (t, z), = le.lidar_zi(LIDAR, 595.0)
    assert t == utc(2011, 6, 15, 7, 14, 3) and z == pytest.approx(221.621)
    launch, lat, lon, c = le.liaise_sonde(SONDE)
    assert launch == utc(2021, 7, 14, 15, 52, 33) and (lat, lon) == (41.59, 1.03)
    np.testing.assert_array_equal(c["Height"], [0.0, 1.0])
    assert c["MixR"][1] == 5.42


def test_sonde_zi_uses_shared_bulk_ri():
    z = np.arange(0.0, 3000.0, 50.0)
    t = np.where(z < 1000.0, 300.0, 300.0 + 0.02 * (z - 1000.0))  # at p_ref, theta = T: mixed layer under an inversion
    zi, moist = le.sonde_profile_zi(z, t, np.full(z.size, constants.p_ref), np.full(z.size, np.nan), np.full(z.size, 5.0), np.zeros(z.size))
    assert 1000.0 < zi < 1100.0 and moist is False


def test_cardington_sonde_height_above_launch(tmp_path):
    z = np.arange(0.0, 3000.0, 50.0)
    (tmp_path / "s" / "2011").mkdir(parents=True)
    with netCDF4.Dataset(tmp_path / "s/2011/x.nc", "w") as d:
        d.createDimension("time", z.size)
        cols = {"time": 362000.0 + z / 18000.0, "height": z + 500.0, "air_pressure": np.full(z.size, constants.p_ref / 100.0),
                "air_temperature": np.where(z < 1000.0, 300.0, 300.0 + 0.02 * (z - 1000.0)), "mixr": np.full(z.size, np.nan),
                "wind_east": np.full(z.size, 5.0), "wind_north": np.zeros(z.size), "latitude": np.full(z.size, le.CARD_LAT),
                "longitude": np.full(z.size, le.CARD_LON)}
        for k, v in cols.items():
            d.createVariable(k, "f8", ("time",))[:] = v
    assert le.cardington_sondes(tmp_path / "s", tmp_path) == (1, 1)
    with netCDF4.Dataset(tmp_path / "cardington_sondes.nc") as o:
        assert 1000.0 < float(o["zi"][0]) < 1100.0 and float(o["distance_km"][0]) == pytest.approx(0.0, abs=1e-3)


def test_bllast_documented_calibration():
    raw = np.array([17.62])
    assert le.bllast_calibrated("T45", raw)[0] == pytest.approx(17.12)
    assert le.bllast_calibrated("T60", raw)[0] == pytest.approx(17.56)
