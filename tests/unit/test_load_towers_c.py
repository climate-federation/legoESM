"""Each loader on a tiny synthetic file with known stamps, units, and flags."""
import pathlib
import sys

import netCDF4
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import load_towers_c as lc  # noqa: E402

from legoesm import constants  # noqa: E402


def test_chats_columns_stamps_and_missing(tmp_path):
    row = np.zeros(151)
    row[:6] = [2007, 4, 1, 0, 15, 0]
    row[15] = 101000.0                  # column 16, pressure
    row[16:28] = 290 + np.arange(12)    # columns 17-28, temperature
    row[52:65] = 1 + np.arange(13)      # columns 53-65, wind
    row[78:91] = 0.3                    # columns 79-91, u*
    row[91:104] = 50.0                  # columns 92-104, H
    row[104:110] = 20 + np.arange(6)    # columns 105-110, LE
    row[64] = 1e36
    np.savetxt(tmp_path / "chats_30min_data_2007_04_ver_x.csv", row[None, :], delimiter=",")
    d = lc.load_chats(tmp_path)
    assert d["time_end"][0] == pd.Timestamp("2007-04-01 00:30").timestamp()
    assert d["wind"][0, 0] == 1 and d["wind"][0, 11] == 12 and np.isnan(d["wind"][0, 12])
    assert d["temp"][0, 11] == 301 and d["pressure"][0] == 101000
    assert d["le"][0, 11] == 25 and np.isnan(d["le"][0, 1])
    assert d["t_flux"][0, 12] == 301


def smear_file(path):
    names = (lc.SMEAR_T + [k for k in lc.SMEAR_CUP + lc.SMEAR_SONIC if k]
             + "H Qc_H LE Qc_LE u_star Qc_tau av_t H_radtow Qc_H_radtow LE_radtow Qc_LE_radtow u_star_radtow "
               "Qc_tau_radtow av_t_radtow H_460 LE_460 u_star_460 Pamb0".split())
    f = pd.DataFrame({"HYY_X." + k: [np.nan, np.nan] for k in dict.fromkeys(names)})
    f.insert(0, "Year", 2015), f.insert(1, "Month", 1), f.insert(2, "Day", 1)
    f.insert(3, "Hour", 2), f.insert(4, "Minute", [0, 30]), f.insert(5, "Second", 0)
    f["HYY_X.WS336"] = [3.0, 3.0]
    f["HYY_X.WSU336"] = [np.nan, 4.0]
    f["HYY_X.WS504"] = [5.0, 5.0]
    f["HYY_X.T42"] = [10.0, 10.0]
    f["HYY_X.H"] = [100.0, 100.0]
    f["HYY_X.Qc_H"] = [2.0, np.nan]
    f["HYY_X.Pamb0"] = [990.0, 990.0]
    f.to_csv(path, index=False)


def test_smear_clock_family_and_flag(tmp_path):
    smear_file(tmp_path / "hyy_CHECKED_2015.csv")
    d = lc.load_smear(tmp_path)
    assert d["time_end"][0] == pd.Timestamp("2015-01-01 00:30").timestamp()
    assert d["wind"][0, 3] == 3 and d["wind"][0, 4] == 5
    assert d["wind"][1, 3] == 4 and np.isnan(d["wind"][1, 4])
    assert d["temp"][0, 0] == pytest.approx(10 + constants.T_freeze)
    assert np.isnan(d["h"][0, 0]) and d["h"][1, 0] == 100
    assert d["pressure"][0] == 99000


def ibox_files(root):
    (root / "low_frequency_data").mkdir()
    with netCDF4.Dataset(root / "low_frequency_data/a.nc", "w") as d:
        d.createDimension("time", 61), d.createDimension("z_TRH", 2), d.createDimension("z_wind", 2)
        t = d.createVariable("time", "f8", ("time",))
        t.units = "minutes since 2022-06-09 00:00:00"
        t[:] = np.arange(61)
        d.createVariable("z_TRH", "f8", ("z_TRH",))[:] = [2, 4]
        d.createVariable("z_wind", "f8", ("z_wind",))[:] = [2, 4]
        d.createVariable("T_tower", "f8", ("z_TRH", "time"))[:] = 20.0
        speed = np.full((2, 61), 2.0)
        d.createVariable("wind_speed_tower", "f8", ("z_wind", "time"))[:] = speed
        turn = np.zeros((2, 61))
        turn[:, 1:31:2] = 180.0         # first half hour reverses every minute
        d.createVariable("wind_dir_tower", "f8", ("z_wind", "time"))[:] = turn
        diag = np.zeros((2, 61))
        diag[1, 40] = 1                 # second half hour loses a minute at the upper level
        d.createVariable("wind_diag_tower", "f8", ("z_wind", "time"))[:] = diag
        d.createVariable("P", "f8", ("time",))[:] = 950.0
    for level in lc.IBOX_LEVELS:
        (root / f"EC_flux_lvl{level}").mkdir()
        with netCDF4.Dataset(root / f"EC_flux_lvl{level}/a.nc", "w") as d:
            d.createDimension("time", 2)
            t = d.createVariable("time", "f8", ("time",))
            t.units = "minutes since 2022-06-09 00:15:00"
            t[:] = [0, 30]
            for name, value in ((f"ustr{level}", 0.3), (f"h{level}cc", 40.0), (f"tair{level}", 20.0)):
                d.createVariable(name, "f8", ("time",))[:] = value
            d.createVariable(f"qcflag_uw{level}", "f8", ("time",))[:] = [2, 2]
            d.createVariable(f"qcflag_sh{level}", "f8", ("time",))[:] = [2, 1]
            if level == 1:
                d.createVariable("le1cw", "f8", ("time",))[:] = 10.0
                d.createVariable("qcflag_le1", "f8", ("time",))[:] = 2


def test_ibox_vector_mean_flags_and_stamps(tmp_path):
    ibox_files(tmp_path)
    d = lc.load_ibox(tmp_path, 2)
    k = int(np.flatnonzero(d["time_end"] == pd.Timestamp("2022-06-09 00:30").timestamp())[0])
    assert d["wind"][k, 0] == pytest.approx(0.0, abs=1e-9)
    assert d["wind"][k + 1, 0] == pytest.approx(2.0) and np.isnan(d["wind"][k + 1, 1])
    assert d["h"][k, 0] == 40 and np.isnan(d["h"][k + 1, 0])
    assert d["le"][k, 0] == 10 and np.isnan(d["le"][k, 1])
    assert d["pressure"][k] == 95000
    assert lc.load_ibox(tmp_path, 1)["h"][k + 1, 0] == 40


def test_metcrax_density_and_stamp(tmp_path):
    with netCDF4.Dataset(tmp_path / "MetCrax_unstable_30m.nc", "w") as d:
        d.createDimension("time", 1), d.createDimension("heights", 2)
        t = d.createVariable("time", "f8", ("time",))
        t.units = "minutes since 2013-10-01"
        t[:] = [960]
        d.createVariable("heights", "f8", ("heights",))[:] = [3, 10]
        for name, value in (("meanU", 2.0), ("meanT", 290.0), ("ustar", 0.3), ("wT", 0.1)):
            d.createVariable(name, "f8", ("time", "heights"))[:] = value
    x = lc.load_metcrax(tmp_path)
    assert x["time_end"][0] == pd.Timestamp("2013-10-01 16:30").timestamp()
    rho = lc.METCRAX_PRESSURE_PA / (constants.R_d * 290.0)
    assert x["h"][0, 0] == pytest.approx(rho * constants.c_pd * 0.1)
    assert np.isnan(x["le"]).all()
