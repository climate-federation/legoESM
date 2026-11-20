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
