"""Small synthetic inputs for the four tower loaders."""
import pathlib
import sys

import netCDF4
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import load_towers_b as lb

from legoesm import constants


def mosaic_file(path, base=1578009600.0, bad_turb=None, ustar=0.3):
    n = 6
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", n)
        d.createVariable("base_time", "f8")[...] = base
        d.createVariable("time", "f8", ("time",))[:] = 600.0 * np.arange(n)

        def var(name, x, kind="f8", **attrs):
            v = d.createVariable(name, kind, ("time",))
            v[:] = x
            v.setncatts(attrs)

        var("lat_tower", 85.0)
        var("lon_tower", 100.0)
        heights = dict(height_start="2.0", height_end="2.5", height_change_time="2020-01-03 00:20:00")
        for i, lev in enumerate(lb.MOSAIC_LEVELS):
            z = 1 + i
            var(f"wspd_u_mean_{lev}", [3.0, -3.0, 3.0, 3.0, 3.0, 3.0], **heights)
            var(f"wspd_v_mean_{lev}", 4.0)
            var(f"temp_{lev}", -20.0 - z, **heights)
            var(f"ustar_{lev}", ustar)
            var(f"Hs_{lev}", 10.0, **heights)
            for q in (f"wspd_u_mean_{lev}_qc", f"wspd_v_mean_{lev}_qc", f"temp_{lev}_qc", f"wind_sector_qc_info_{lev}"):
                var(q, 0, "i4")
            qc = np.zeros(n, int)
            if bad_turb is not None:
                qc[bad_turb] = 1
            var(f"turbulence_{lev}_qc", qc, "i4")
        var("Hl_Webb", 5.0)
        var("Hl_qc", 0, "i4")
        var("atmos_pressure_2m", 1000.0)
        var("atmos_pressure_2m_qc", 0, "i4")


def test_mosaic_start_stamps_vector_wind_and_caution(tmp_path):
    mosaic_file(tmp_path / "mosseb.metcity.level3.4.10min.20200103.000000.nc", bad_turb=4)
    out = lb.load_mosaic(tmp_path)
    assert (out["time_end"] - 1578009600).tolist() == [1800, 3600]
    assert out["wind"][0, 0] == pytest.approx(np.hypot(1.0, 4.0))
    assert out["temp"][0, 0] == pytest.approx(-21.0 + constants.T_freeze)
    assert out["ustar"][0, 0] == pytest.approx(0.3) and np.isnan(out["ustar"][1, 0])
    assert out["z_u"][0, 0] == pytest.approx((2.0 + 2.0 + 2.5) / 3)
    assert out["pressure"][0] == pytest.approx(1e5) and out["le"][0, 0] == 5.0 and np.isnan(out["le"][0, 1])


def test_mosaic_u_star_averages_as_stress(tmp_path):
    mosaic_file(tmp_path / "mosseb.metcity.level3.4.10min.20200103.000000.nc", ustar=[0.1, 0.1, 0.4, 0.3, 0.3, 0.3])
    assert lb.load_mosaic(tmp_path)["ustar"][0, 0] == pytest.approx(0.06 ** 0.5)


def test_sheba_start_of_hour_missing_values_and_flag(tmp_path):
    names = ["JD", "lat", "lon", "Press", "zoph"] + [f"{v}{k}" for v in ("z", "ws", "T", "u*", "hs", "fl") for k in range(1, 6)] + ["hl"]
    row = {n: 1.0 for n in names}
    row.update({"JD": 366.0, "lat": 76.0, "lon": -160.0, "Press": 1014.0, "zoph": 4.0, "hl": 3.0})
    row.update({f"z{k}": float(k + 1) for k in range(1, 6)})
    row.update({f"T{k}": -10.0 for k in range(1, 6)})
    row.update({f"fl{k}": 0.0 for k in range(1, 6)})
    second = dict(row, JD=366.0417, fl2=1.0, ws1=9999.0, Press=9999.0)
    path = tmp_path / "p.txt"
    path.write_text("\t".join(names) + "\t\n" + "units\n"
                    + "\n".join("\t".join(str(r[n]) for n in names) for r in (row, second)) + "\n")
    out = lb.load_sheba(str(path))
    assert out["time_end"].tolist() == [lb.calendar.timegm((1998, 1, 1, 1, 0, 0)), lb.calendar.timegm((1998, 1, 1, 2, 0, 0))]
    assert np.isnan(out["wind"][1, 0]) and np.isnan(out["wind"][1, 1]) and np.isnan(out["h"][1, 1])
    assert np.isnan(out["pressure"][1]) and out["pressure"][0] == pytest.approx(101400.0)
    assert out["le"][0, 2] == 3.0 and np.isnan(out["le"][0, 0])
    assert out["temp"][0, 0] == pytest.approx(-10.0 + constants.T_freeze)


def test_fino_sorts_heights_blocks_ten_minutes_and_keeps_flags(tmp_path):
    path = tmp_path / "f.nc"
    n = 6
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("height", 3)
        d.createDimension("time", n)
        d.createVariable("height", "f8", ("height",))[:] = [81, 61, 41]
        d.createVariable("time", "i8", ("time",))[:] = 10 * np.arange(n)
        d.geospatial_lat_max, d.geospatial_lon_max, d.license = "54.0", "6.5", "CC BY 4.0"
        for name, value in (("wind_speed", 8.0), ("wind_from_direction", 270.0), ("wind_speed_qc", 0), ("wind_from_direction_qc", 0), ("qc_Tau", 0),
                            ("qc_H", 0), ("u*", 0.4), ("H", 10.0), ("sonic_temperature", 285.0)):
            v = d.createVariable(name, "f8", ("height", "time"))
            v[:] = value
        d["wind_speed"][0, :] = 9.0
        d["wind_speed_qc"][0, 4] = 3
        d["wind_from_direction_qc"][1, 1] = 3
        d["u*"][2, :3] = [0.1, 0.1, 0.4]
    out = lb.load_fino1(str(path))
    assert np.isnan(out["wind"][0, 1]), "bad direction must blank the vector wind"
    assert out["ustar"][0, 0] == pytest.approx(0.06 ** 0.5), "u* averages as stress"
    assert out["z_u"][0].tolist() == [41.0, 61.0, 81.0]
    assert out["time_end"].size == 2 and (out["time_end"] % 1800 == 0).all()
    assert out["wind"][0, 2] == pytest.approx(9.0) and out["wind"][0, 0] == pytest.approx(8.0)
    assert np.isnan(out["wind"][1, 2]) and out["wind"][1, 1] == pytest.approx(8.0)
    assert out["attrs"]["pressure_source"].startswith("assumed") and out["attrs"]["temp_kind"] == "sonic"


def test_falkenberg_joins_masts_and_blanks_flagged_fluxes():
    t = np.array([1800, 3600])
    mast = {"low": {"t": t, "z": np.array([1.0, 2.0]), "ta": np.full((2, 2), 290.0), "wspeed": np.full((2, 2), 3.0)},
            "high": {"t": t[1:], "z": np.array([10.0]), "ta": np.full((1, 1), 289.0), "wspeed": np.full((1, 1), 5.0)}}
    flux = {f: {"t": t, "ustar": np.full(2, 0.3), "ustar_flag": np.array([3, 2]), "hfss": np.full(2, 50.0),
                "hfss_flag": np.array([3, 3]), "hfls": np.full(2, 80.0), "hfls_flag": np.array([1, 3]),
                "tsonic": np.full(2, 291.0), "pa": np.full(2, 1e5)} for f in ("2m", "50m", "90m")}
    out = lb.falkenberg_table(mast, flux)
    assert out["z_u"][0].tolist() == [1.0, 2.0, 10.0] and np.isnan(out["wind"][0, 2]) and out["wind"][1, 2] == 5.0
    assert np.isnan(out["ustar"][1, 0]) and out["h"][1, 0] == 50.0 and np.isnan(out["le"][0, 0])
    assert out["z_f"][0].tolist() == [2.3, 50.3, 90.3]


def test_unknown_site_raises():
    with pytest.raises(ValueError):
        lb.build("nowhere", "/nonexistent")
