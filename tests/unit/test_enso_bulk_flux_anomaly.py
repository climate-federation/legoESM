import importlib.util
import pathlib

import numpy as np

_P = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/enso_bulk_flux_anomaly.py"
_spec = importlib.util.spec_from_file_location("enso_bulk_flux_anomaly", _P)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_signs_and_wind_dependence():
    # eastward wind over a warm, dry-air ocean: upward LHF, eastward stress on the surface
    u = np.array([5.0, 10.0]); v = np.zeros(2)
    t2 = np.full(2, 299.0); td2 = np.full(2, 295.0)
    sst = np.full(2, 301.0); ps = np.full(2, 101000.0)
    lhf, tau = mod.model_fluxes(u, v, t2, td2, sst, ps)
    assert np.all(lhf > 0) and np.all(tau > 0)
    assert lhf[1] > lhf[0] and tau[1] > tau[0]


def test_planted_ocean_nan_is_fatal():
    import pytest
    lat, lon = np.array([0.0]), np.array([200.0])
    t = lambda y: ((np.arange(124) + 1) * 21600 + (np.datetime64(f"{y}-12-01", "s")
                                            - np.datetime64("1970-01-01", "s")).astype("int64"))
    d = {y: dict(time=t(y), lat=lat, lon=lon,
                 **{v: np.ones((124, 1, 1)) for v in mod._IN + mod._OUT}) for y in mod._YEARS}
    ocean = np.ones((1, 1), bool)
    mod.check_inputs(d, ocean, lat, lon)
    d[1993]["10m_u_component_of_wind"][5, 0, 0] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        mod.check_inputs(d, ocean, lat, lon)


def test_alignment_guard_refuses_offset_hourly_store():
    import pytest
    rng = np.random.default_rng(0)
    six = {"2m_temperature": 290 + rng.random((124, 2, 2)),
           "sea_surface_temperature": 300 + rng.random((124, 2, 2))}
    six["sea_surface_temperature"][:, 0, 0] = np.nan  # a cell NaN in the 6h store only
    sst1 = six["sea_surface_temperature"].copy(); sst1[:, 0, 0] = 301.0
    ends = [0, -1]
    t2h = six["2m_temperature"][ends] + 5e-5
    mod.check_alignment(six, {"sea_surface_temperature": sst1}, t2h, ends)
    with pytest.raises(AssertionError, match="2 m temperature"):
        mod.check_alignment(six, {"sea_surface_temperature": sst1}, t2h + 4.0, ends)
    with pytest.raises(AssertionError, match="SST"):
        mod.check_alignment(six, {"sea_surface_temperature": sst1 + 0.01}, t2h, ends)


def test_stale_hourly_cache_is_refused_on_load(tmp_path):
    import pytest
    six = {"time": np.arange(124), "2m_temperature": np.full((124, 2, 2), 290.0),
           "sea_surface_temperature": np.full((124, 2, 2), 300.0)}
    np.savez(tmp_path / "era5_dec1997.npz", **six)
    hour = dict(time=six["time"], src=mod._WB2_1H, t2_ends=six["2m_temperature"][[0, -1]],
                **{"2m_dewpoint_temperature": np.full((124, 2, 2), 295.0),
                   "sea_surface_temperature": np.full((124, 2, 2), 300.0)})
    np.savez(tmp_path / "era5_dec1997_1h.npz", **hour)
    assert mod.load_year(str(tmp_path), 1997)["2m_dewpoint_temperature"][0, 0, 0] == 295.0
    hour["sea_surface_temperature"] = hour["sea_surface_temperature"] + 10.0
    np.savez(tmp_path / "era5_dec1997_1h.npz", **hour)
    with pytest.raises(AssertionError, match="SST"):
        mod.load_year(str(tmp_path), 1997)
