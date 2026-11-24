"""Known-answer checks for the Cardington, ASCOS, AO2018, N-ICE and ICOS loaders."""
import pathlib
import sys
import zipfile

import netCDF4
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import load_towers_d as ld  # noqa: E402

from legoesm import constants  # noqa: E402


def write_nc(path, n, **v):
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", n)
        for k, x in v.items():
            d.createVariable(k, "f8", ("time",))[:] = x


def test_sonic_flux_loses_humidity_term():
    t, p = 290.0, 1e5
    rho = p / (constants.R_d * t)
    h, le = ld.sensible_from_sonic(0.1, 5e-5, t, p)
    assert le == pytest.approx(constants.L_v * 5e-5)
    assert h == pytest.approx(rho * constants.c_pd * (0.1 - 0.51 * t * 5e-5 / rho))
    assert h < rho * constants.c_pd * 0.1


def card_rows(n, first_2m):
    x = {"time_end": 1800 * (np.arange(n) + 1), "wq": np.full(n, 2e-5), "p": np.full(n, 1000.0)}
    for z in ("2m", "10m", "25m", "50m"):
        x["u" + z], x["uw" + z], x["vw" + z], x["wt" + z] = (np.full(n, v) for v in (5.0, -0.09, 0.0, 0.05))
    for k in ld.CARD_T_Z:
        x["t" + k] = np.full(n, 10.0)
    for k in ("u2m", "uw2m", "vw2m", "wt2m"):
        x[k][:first_2m] = np.nan
    return x


def test_cardington_splits_where_the_2m_sonic_starts():
    early, late = ld.cardington_tables(card_rows(10, 4))
    assert early["time_end"].size == 4 and late["time_end"].size == 6
    assert early["z_f"][0].tolist() == [10.0, 25.0, 50.0]
    assert late["z_f"][0, 0] == 2.0 and late["ustar"][0, 0] == pytest.approx(0.3)
    assert np.isfinite(late["le"][:, 1]).all() and np.isnan(late["le"][:, [0, 2, 3]]).all(), "LE only where measured"
    assert early["attrs"]["site"] != late["attrs"]["site"]


def test_cardington_flag_blanks_and_end_stamp(tmp_path):
    path = tmp_path / "d.nc"
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", 2)
        d.createDimension("bounds", 2)
        d.createVariable("time_bnds", "f8", ("time", "bounds"))[:] = [[100.0, 100.5], [100.5, 101.0]]
        for k, v, q in (("wind_speed_vector_10m", 5.0, [1, 5]), ("wq_covariance_10m", 1e-5, [1, 1])):
            d.createVariable(k, "f8", ("time",))[:] = v
            d.createVariable("qc_flag_" + k, "f8", ("time",))[:] = q
    x = ld._card_day(path)
    assert x["time_end"].tolist() == [361800, 363600]
    assert x["u10m"][0] == 5.0 and np.isnan(x["u10m"][1]) and np.isnan(x["u2m"]).all()


def test_window_mean_needs_every_sample_and_covers_the_window_only():
    t = np.arange(30, 2400, 60)
    x = np.arange(t.size, dtype=float)
    ends = np.array([1200, 2400])
    assert ld.window_mean(t, x, ends, 1200, 20).tolist() == [9.5, 29.5]
    y = x.copy()
    y[3] = np.nan
    out = ld.window_mean(t, y, ends, 1200, 20)
    assert np.isnan(out[0]) and out[1] == 29.5


def test_nice_flips_downward_flux_and_blanks_reconstructed_wind(tmp_path):
    n = 60
    t = 1800.0 * 1000 + 60 * (np.arange(n) + 1)
    met = dict(unix_time=t, air_pressure_at_sea_level=np.full(n, 1e5), wind_speed_10m_flag=np.r_[np.zeros(30), np.ones(30)],
               wind_from_direction_10m_flag=np.zeros(n))
    for z in (2, 4, 10):
        met.update({f"wind_speed_{z}m": np.full(n, 6.0), f"wind_from_direction_{z}m": np.full(n, 90.0),
                    f"air_temperature_{z}m": np.full(n, 260.0)})
    write_nc(tmp_path / "m.nc", n, **met)
    write_nc(tmp_path / "f.nc", 2, unix_time=1800.0 * np.array([1001, 1002]), friction_velocity=np.full(2, 0.3),
             surface_downward_sensible_heat_flux=np.full(2, 20.0), surface_downward_latent_heat_flux=np.full(2, -5.0),
             latitude=np.full(2, 82.0), longitude=np.full(2, 15.0))
    d = ld.load_nice(tmp_path / "f.nc", tmp_path / "m.nc")
    assert d["h"][0, 0] == -20.0 and d["le"][0, 0] == 5.0 and d["z_f"][0, 0] == 2.2
    assert d["wind"][0, 2] == pytest.approx(6.0) and np.isnan(d["wind"][1, 2]) and d["wind"][1, 0] == pytest.approx(6.0)


def test_steps_take_the_last_entry_at_or_before_each_time():
    assert np.isnan(ld.steps([10, 20], [1.0, 2.0], [5])[0])
    assert ld.steps([20, 10], [2.0, 1.0], [10, 15, 20, 99]).tolist() == [1.0, 1.0, 2.0, 2.0]


def test_stamp_seconds_handles_both_lengths():
    assert ld.stamp_seconds(["20200101", "202001010030"]).tolist() == [1577836800.0, 1577838600.0]


def test_short_canopy_keeps_low_crop_and_nothing_outside_the_visits():
    keep = ld.short_canopy([0, 100], [0.0, 1.0], [-1, 10, 50, 101])
    assert keep.tolist() == [False, True, False, False]


def ascos_file(path):
    n = 3
    v = dict(time=719529 + (1000 * 1800 + np.array([300.0, 900.0, 1500.0])) / 86400, Pressure=np.full(n, 1000.0),
             latitude=np.full(n, 87.0), longitude=np.full(n, 10.0), flag_ship=np.zeros(n), flag_mast_csat=np.zeros(n),
             flag_mast_gill=np.zeros(n), flag_licor_404_cm=np.zeros(n), wq_404_cm=np.full(n, 1e-5))
    for c in ld.ASCOS_T_CM:
        v[f"TC_Air_{c}_cm"] = np.full(n, -2.0)
    for c in ld.ASCOS_CSAT_CM + ld.ASCOS_GILL_CM + [ld.ASCOS_TOP_CM]:
        v.update({f"ugeo_{c}_cm": np.full(n, 3.0), f"vgeo_{c}_cm": np.full(n, 4.0), f"uw_{c}_cm": np.full(n, -0.09),
                  f"vw_{c}_cm": np.zeros(n), f"wts_{c}_cm": np.full(n, 0.01)})
    for c in ld.ASCOS_CSAT_CM:
        v.update({f"flag_csat_{c}_cm": np.zeros(n), f"wtc_{c}_cm": np.full(n, 0.002)})
    for c in ld.ASCOS_GILL_CM:
        v[f"flag_gill_{c}_cm"] = np.zeros(n)
    v["flag_gill_404_cm"] = np.array([0.0, 2.0, 0.0])
    v["flag_csat_94_cm"] = np.array([0.0, 1.0, 0.0])
    write_nc(path, n, **v)


def test_ascos_end_stamp_finewire_flag_and_humidity_term(tmp_path):
    ascos_file(tmp_path / "a.nc")
    d = ld.load_ascos(tmp_path / "a.nc")
    assert d["time_end"].tolist() == [1001 * 1800]
    assert d["wind"][0, 0] == pytest.approx(5.0) and d["ustar"][0, 0] == pytest.approx(0.3)
    assert np.isnan(d["h"][0, 0]), "finewire heat needs CSAT flag 0"
    t, rho = d["t_flux"][0, 2], 1e5 / (constants.R_d * d["t_flux"][0, 2])
    assert d["h"][0, 2] == pytest.approx(rho * constants.c_pd * 0.002)
    assert np.isnan(d["le"][0, 1]) and np.isnan(d["h"][0, 1]), "Gill flag 2 blanks water vapour flux and heat"
    assert np.isnan(d["ustar"][0, 1]) and np.isnan(d["wind"][0, 1]), "Gill flag 2 stress is not trusted"
    assert d["ustar"][0, 2] == pytest.approx(0.3), "other sonics stay"
    assert np.isnan(np.delete(d["le"][0], 1)).all(), "LE only at the Licor level"


def test_ao2018_window_ends_flags_and_pressure(tmp_path):
    n = 40
    with netCDF4.Dataset(tmp_path / "oden-ao-2018-meteorology-ice-station-1.nc", "w") as d:
        d.createDimension("time", n)
        d.createDimension("tz", 2)
        d.createDimension("wsz", 2)
        d.createVariable("time", "f8", ("time",))[:] = 1200 * 1000 + 30 + 60 * np.arange(n)
        d.createVariable("tz", "f8", ("tz",))[:] = [1.0, 3.0]
        d.createVariable("wsz", "f8", ("wsz",))[:] = [1.0, 2.0]
        for k, dim, x in (("tprof", "tz", -1.0), ("wsprof", "wsz", 4.0)):
            d.createVariable(k, "f8", ("time", dim))[:] = x
            d.createVariable(k + "_corflag", "f8", ("time", dim))[:] = 0
    for name in ld.AO_FLUX_Z:
        write_nc(tmp_path / f"oden-ao-2018-turbulent-flux-{name}-ice-station-1.nc", 2, time=1200 * 1000 + 600 + 1200.0 * np.arange(2),
                 ustar=np.full(2, 0.2), tsflux=np.full(2, -3.0), airdensity=np.full(2, 1.3), qc_flag_quality_wu=np.array([1.0, 3.0]),
                 qc_flag_quality_wv=np.ones(2), qc_flag_quality_wt=np.array([0.0, 2.0]))
    d = ld.load_ao2018(tmp_path)
    assert d["time_end"].tolist() == [1200 * 1001, 1200 * 1002]
    assert d["wind"][0, 0] == 4.0 and d["ustar"][0, 0] == 0.2 and np.isnan(d["ustar"][1, 0])
    assert np.isnan(d["h"][0, 0]) and d["h"][1, 0] == -3.0
    assert d["pressure"][0] == pytest.approx(1.3 * constants.R_d * (constants.T_freeze - 1.0))


def icos_zip(path):
    p = "ICOSETC_FR-X_"
    stamps = ["202001011300", "202001011330"]
    group = lambda rows: "SITE_ID,GROUP_ID,VARIABLE_GROUP,VARIABLE,DATAVALUE\n" + "".join(
        f"FR-X,{g},G,{k},{v}\n" for g, row in enumerate(rows, 1) for k, v in row.items())
    files = {
        "METEOSENS": "TIMESTAMP_END,PA_1_1_1,TA_1_1_1,TA_1_2_1,WS_1_1_1,WS_1_2_1\n" + "".join(f"{s},100.0,10,11,5,4\n" for s in stamps),
        "FLUXES": ("TIMESTAMP_END,H,LE,USTAR,H_SSITC_TEST,LE_SSITC_TEST,TAU_SSITC_TEST\n"
                   f"{stamps[0]},50,20,0.3,0,2,1\n{stamps[1]},-9999,20,0.3,0,0,2\n"),
        "VARINFO_METEOSENS": group([{"VAR_INFO_VARNAME": n, "VAR_INFO_HEIGHT": h, "VAR_INFO_DATE": "20190101"}
                                    for n, h in (("TA_1_1_1", 5), ("TA_1_2_1", 2), ("WS_1_1_1", 5), ("WS_1_2_1", 2))]),
        "INST": group([{"INSTOM_MODEL": "SA-Gill", "INSTOM_HEIGHT": 3, "INSTOM_DATE": "20190101"}]),
        "ANCILLARY": group([{"HEIGHTC": 0.1, "HEIGHTC_STATISTIC": "Mean", "HEIGHTC_DATE": "20190101"},
                            {"HEIGHTC": 0.1, "HEIGHTC_STATISTIC": "Mean", "HEIGHTC_DATE": "20210101"}]),
        "SITEINFO": group([{"LOCATION_LAT": 45, "LOCATION_LONG": 1, "SITE_DESC": "grass"}]),
    }
    with zipfile.ZipFile(path, "w") as z:
        for k, text in files.items():
            z.writestr(f"{p}{k}_L2.csv", text)


def test_icos_utc_units_and_each_flux_has_its_own_flag(tmp_path):
    icos_zip(tmp_path / "x.zip")
    d = ld._icos_site(tmp_path / "x.zip", "FR-X", "", ["TA_1_1_1", "TA_1_2_1"], ["WS_1_1_1", "WS_1_2_1"])
    assert d["time_end"][0] == 1577883600 - 3600
    assert d["pressure"][0] == 1e5 and d["z_f"][0, 0] == 3.0
    assert np.isnan(d["le"][0, 0]) and d["le"][1, 0] == 20.0
    assert d["ustar"][0, 0] == 0.3 and np.isnan(d["ustar"][1, 0])


def test_icos_levels_ascend_and_values_follow_their_heights(tmp_path):
    icos_zip(tmp_path / "x.zip")
    d = ld._icos_site(tmp_path / "x.zip", "FR-X", "", ["TA_1_1_1", "TA_1_2_1"], ["WS_1_1_1", "WS_1_2_1"])
    assert d["z_u"][0].tolist() == [2.0, 5.0] and d["wind"][0].tolist() == [4.0, 5.0]
    assert d["z_t"][0].tolist() == [2.0, 5.0]
    assert d["temp"][0] == pytest.approx([11 + constants.T_freeze, 10 + constants.T_freeze])
    assert d["t_flux"][0, 0] == pytest.approx(11 + constants.T_freeze), "sonic at 3 m takes the 2 m thermometer"


def test_icos_record_stops_where_a_site_ends(tmp_path, monkeypatch):
    icos_zip(tmp_path / "x.zip")
    monkeypatch.setitem(ld.ICOS_END_BEFORE, "FR-X", "202001011330")
    d = ld._icos_site(tmp_path / "x.zip", "FR-X", "", ["TA_1_1_1", "TA_1_2_1"], ["WS_1_1_1", "WS_1_2_1"])
    assert d["time_end"].size == 1
