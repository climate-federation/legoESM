"""Known-answer checks for the CASES-99 / AHATS ISFF 5-minute loader."""
import pathlib
import sys

import netCDF4
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import load_towers_f as lf  # noqa: E402

from legoesm import constants  # noqa: E402


def test_covariance_recovers_the_full_half_hour_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 200)) + np.arange(6)[:, None] * 0.7
    b = 0.5 * a + rng.normal(size=(6, 200)) + np.array([0.3, -0.2, 0.9, 0.0, -0.6, 0.4])[:, None]
    cov5 = ((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).mean(1)
    t = 300 * np.arange(1, 7)
    _, c = lf.covariance(t, np.ones(6, bool), cov5, a.mean(1), b.mean(1))
    full = ((a - a.mean()) * (b - b.mean())).mean()
    assert c[0] == pytest.approx(full, rel=1e-12)
    assert abs(cov5.mean() - full) > 0.05, "the between-block part must matter in this fixture"


def test_half_hour_needs_all_six_good_blocks():
    t = 300 * np.arange(1, 13)
    ok = np.ones(12, bool)
    ok[8] = False
    keys, m = lf.half(t, np.arange(12.0), ok)
    assert keys.tolist() == [1800, 3600]
    assert m[0] == pytest.approx(2.5) and np.isnan(m[1])


def test_sonic_qc_follows_the_readme_rules():
    x = {"diag_1m": np.array([0.0, 0.01, np.nan]), "ldiag_2m": np.array([0.0, 0.0002, 1.0]),
         **{f"{c}samples_3m": np.array([10.0, 10.0, 9.9]) for c in "uvw"}}
    x["vsamples_3m"] = np.array([10.0, 9.0, 10.0])
    assert lf.sonic_ok(x, "1m", "diag").tolist() == [True, False, False]
    assert lf.sonic_ok(x, "2m", "ldiag").tolist() == [True, False, False]
    assert lf.sonic_ok(x, "3m", "ati").tolist() == [True, False, False]
    with pytest.raises(ValueError):
        lf.sonic_ok(x, "1m", "csat")


def test_read_days_stamps_block_ends_blanks_corrupt_and_fills_missing(tmp_path):
    for day, base, has_p in (("b", 86400, True), ("a", 0, False)):
        with netCDF4.Dataset(tmp_path / f"f_{day}.nc", "w") as d:
            d.createDimension("time", 2)
            d.createDimension("station", 3)
            d.createVariable("base_time", "i4")[...] = base
            d.createVariable("time", "f8", ("time",))[:] = [150.0, 450.0]
            d.createVariable("T_5m", "f4", ("time",))[:] = [10.0, 1.6e24]
            if has_p:
                d.createVariable("P", "f4", ("time", "station"))[:] = [[960, 961, 962], [963, 964, 965]]
    t, x = lf.read_days(str(tmp_path / "f_*.nc"), ["T_5m", "P"])
    assert t.tolist() == [300, 600, 86700, 87000]
    assert x["T_5m"][0] == 10.0 and np.isnan(x["T_5m"][1]) and np.isnan(x["T_5m"][3])
    assert x["P"].shape == (4, 3) and np.isnan(x["P"][:2]).all() and x["P"][2, 1] == 961.0


def synthetic(n_half=2):
    n = 6 * n_half
    t = 300 * np.arange(1, n + 1)
    sonics = [(1.5, "1_5m", "diag"), (12.0, "12m", "ati")]
    temps = [(5.0, "T_5m"), (15.0, "T_15m")]
    x = {"T_5m": np.full(n, 20.0), "T_15m": np.full(n, 21.0),
         "P": np.tile([[960.0, 970.0, 1000.0]], (n, 1))}
    for _, s, kind in sonics:
        x.update({f"u_{s}": np.full(n, 3.0), f"v_{s}": np.full(n, 4.0), f"w_{s}": np.zeros(n), f"tc_{s}": np.full(n, 20.0),
                  f"u_w_{s}": np.full(n, -0.09), f"v_w_{s}": np.full(n, -0.12), f"w_tc_{s}": np.full(n, 0.05)})
    x["diag_1_5m"] = np.zeros(n)
    x["diag_1_5m"][7] = 1.0
    for c in "uvw":
        x[f"{c}samples_12m"] = np.full(n, 10.0)
    return t, x, sonics, temps


def test_isff_table_columns_end_to_end():
    t, x, sonics, temps = synthetic()
    out = lf.isff_table(t, x, sonics, temps, "P")
    assert out["time_end"].tolist() == [1800, 3600]
    assert out["pressure"][0] == pytest.approx(97000.0), "station median"
    assert out["wind"][0].tolist() == pytest.approx([5.0, 5.0])
    assert out["ustar"][0, 0] == pytest.approx(np.hypot(0.09, 0.12) ** 0.5)
    t_k = 20.0 + constants.T_freeze
    assert out["t_flux"][0].tolist() == pytest.approx([t_k, t_k + 1.0]), "nearest thermometer"
    assert out["h"][0, 0] == pytest.approx(97000.0 / (constants.R_d * t_k) * constants.c_pd * 0.05)
    assert np.isnan(out["le"]).all()
    assert np.isnan(out["ustar"][1, 0]) and np.isfinite(out["ustar"][1, 1]), "one bad CSAT block blanks only its level"


def test_station_pressure_outside_range_is_dropped_before_the_median():
    t, x, sonics, temps = synthetic()
    x["P"] = np.tile([[44.0, 44.0, 970.0]], (t.size, 1))
    out = lf.isff_table(t, x, sonics, temps, "P")
    assert out["pressure"][0] == pytest.approx(97000.0)


def test_heat_only_table_writes_and_scores_no_wind_pair(tmp_path):
    import flux_gradient_sites as fs
    import flux_profile_table as table
    t, x, sonics, temps = synthetic(4)
    x["T_15m"] = x["T_15m"] + np.arange(t.size) * 0.01
    out = lf.isff_table(t, x, sonics, temps, "P", profile_winds=False)
    assert out["wind"].shape == (4, 0) and out["z_u"].shape == (4, 0)
    attrs = dict(site="s", latitude=0.0, longitude=0.0, surface="x", canopy_height_m=0.0, displacement_m=0.0,
                 source="x", licence="x", averaging_minutes=30, flux_kind="native", flux_method="x")
    table.write_table(tmp_path / "s.nc", **out, attrs=attrs)
    ds = fs.open_site(tmp_path / "s.nc")
    samples = fs.site_samples(ds, 0, 0.1, 0.0, False, False)
    assert samples and all(k.startswith("h:") for k in samples)


def test_cases_levels_leave_out_the_40m_sonic():
    assert 40.0 not in [z for z, _, _ in lf.CASES_SONICS]


def test_read_days_refuses_a_name_in_no_file(tmp_path):
    with netCDF4.Dataset(tmp_path / "f.nc", "w") as d:
        d.createDimension("time", 1)
        d.createVariable("base_time", "i4")[...] = 0
        d.createVariable("time", "f8", ("time",))[:] = [150.0]
    with pytest.raises(KeyError):
        lf.read_days(str(tmp_path / "f*.nc"), ["T_5m"])


def test_t_flux_takes_the_mean_of_two_equidistant_thermometers():
    t, x, _, temps = synthetic()
    for k, v in list(x.items()):
        if k.endswith("_12m"):
            x[k.replace("12m", "10m")] = v
    out = lf.isff_table(t, x, [(1.5, "1_5m", "diag"), (10.0, "10m", "ati")], temps, "P")
    assert out["t_flux"][0, 1] == pytest.approx(20.5 + constants.T_freeze)


def test_read_days_refuses_a_midpoint_in_the_next_day(tmp_path):
    with netCDF4.Dataset(tmp_path / "f.nc", "w") as d:
        d.createDimension("time", 1)
        d.createVariable("base_time", "i4")[...] = 0
        d.createVariable("time", "f8", ("time",))[:] = [86550.0]
        d.createVariable("T_5m", "f4", ("time",))[:] = [10.0]
    with pytest.raises(ValueError):
        lf.read_days(str(tmp_path / "f*.nc"), ["T_5m"])
