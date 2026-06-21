"""Unit tests for the ERA5 land validator (no network — synthetic climatology)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.validate_land_era5 import (
    build_monthly_forcing, validation_metrics, run_land, _SCHEMES)


def _synthetic_clim(nlat=4, nlon=6):
    """A tiny plausible ERA5 monthly climatology (no network)."""
    rng = np.random.default_rng(0)
    lat = np.linspace(80, -80, nlat); lon = np.linspace(0, 300, nlon)
    sh3 = (12, nlat, nlon)
    base = 285.0 - 0.3 * np.abs(lat)[None, :, None]
    d = dict(
        lat=lat, lon=lon,
        lsm=(rng.random((nlat, nlon)) > 0.4).astype(float),
        elev_m=rng.random((nlat, nlon)) * 1000.0,
        skin_temperature=np.broadcast_to(base, sh3) + rng.normal(0, 1, sh3),
        **{"2m_temperature": np.broadcast_to(base, sh3).copy(),
           "2m_dewpoint_temperature": np.broadcast_to(base - 5, sh3).copy(),
           "ssrd_wm2": np.full(sh3, 200.0),
           "strd_wm2": np.full(sh3, 320.0),
           "precip_kgms": np.full(sh3, 2e-5),
           "surface_pressure": np.full(sh3, 1.0e5),
           "10m_u_component_of_wind": np.full(sh3, 3.0),
           "10m_v_component_of_wind": np.full(sh3, 2.0)})
    return d


def test_build_monthly_forcing_shapes():
    clim = _synthetic_clim()
    cols = np.array([0, 1, 5, 7])
    forc, t2m0 = build_monthly_forcing(clim, cols)
    assert len(forc) == 12
    assert forc[0].sw_down.shape == (4,)
    assert t2m0.shape == (4,)
    # precip becomes snow exactly where T < freezing, rain elsewhere
    from legoesm import constants
    T = np.asarray(forc[6].T_lowest); snow = np.asarray(forc[6].precip_snow)
    cold = T < constants.T_freeze
    assert np.all(snow[cold] > 0.0) and np.all(snow[~cold] == 0.0)


def test_validation_metrics_math():
    m = np.full((12, 5), 290.0); s = np.full((12, 5), 288.0)
    out = validation_metrics(m, s)
    assert out["bias"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(2.0)
    assert out["annual_bias"] == pytest.approx(2.0)


def test_run_land_dispatch_raises():
    with pytest.raises(ValueError):
        run_land("bogus", _synthetic_clim(), years=1)
    assert set(_SCHEMES) == {"slab", "multilayer"}


def test_run_slab_end_to_end_finite():
    """Free-run the slab one year on the tiny synthetic grid -> finite, sensible."""
    clim = _synthetic_clim()
    mT, skt, cols, extra = run_land("slab", clim, years=1)
    assert mT.shape[0] == 12 and mT.shape[1] == cols.size
    assert np.all(np.isfinite(mT))
    assert 200.0 < mT.mean() < 340.0           # physical surface temperature
    M = validation_metrics(mT, skt)
    assert np.isfinite(M["rmse"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
