"""Direct test for ``scripts/validate/spectral_jet_decay_probe.py``: the band
diagnostics on synthetic rows with a known answer, and the retained-cut guard."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_PROBE = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate"
          / "spectral_jet_decay_probe.py")


def _load():
    spec = importlib.util.spec_from_file_location("_jet_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_jet_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _grid(nlat=4, nlon=48, nlev=3):
    lat1 = np.radians(np.linspace(-45.0, 45.0, nlat))
    lon1 = np.arange(nlon) * 2 * np.pi / nlon
    lat = np.repeat(lat1, nlon)
    lon = np.tile(lon1, nlat)
    p = np.tile(np.array([100.0, 150.0, 300.0]), (nlat * nlon, 1))
    return lat, lon, p


def _wave(lat, lon, m, amp=3.0, mean=10.0):
    u = np.zeros((lat.size, 3))
    u[:, 1] = mean + amp * np.cos(m * lon) * np.cos(lat)
    return u


@pytest.mark.parametrize("m,in_tail", [(18, True), (10, False)])
def test_spectrum_puts_a_single_wave_where_it_is(m, in_tail):
    mod = _load()
    lat, lon, p = _grid()
    u = _wave(lat, lon, m)
    v = np.zeros_like(u)
    d = mod.band_diagnostics(lat, lon, u, v, p)
    assert int(np.argmax(d["spectrum"][1:])) + 1 == m
    assert d["tail_frac"] == pytest.approx(1.0 if in_tail else 0.0, abs=1e-12)
    assert d["zonal_max_u"] == pytest.approx(10.0, abs=1e-12)
    assert d["zonal_max_p_hpa"] == 150.0


def test_slope_of_a_power_law_spectrum():
    mod = _load()
    lat, lon, p = _grid(nlat=2, nlon=64)
    u = np.zeros((lat.size, 3))
    for m in range(1, 31):
        u[:, 1] += m ** -1.5 * np.cos(m * lon + 0.3 * m)
    d = mod.band_diagnostics(lat, lon, u, np.zeros_like(u), p)
    assert d["slope"] == pytest.approx(-3.0, abs=1e-9)  # E ~ amp^2 ~ m^-3


def test_ke_is_cos_weighted_band_mean():
    mod = _load()
    lat, lon, p = _grid()
    u = np.zeros((lat.size, 3))
    u[:, 1] = 2.0
    v = np.zeros_like(u)
    v[:, 1] = 1.0
    d = mod.band_diagnostics(lat, lon, u, v, p)
    assert d["ke_band"] == pytest.approx(2.5, abs=1e-12)


def test_uneven_rows_nonfinite_and_nyquist_refused():
    mod = _load()
    lat, lon, p = _grid()
    u = _wave(lat, lon, 5)
    with pytest.raises(ValueError, match="uneven"):
        mod.band_diagnostics(lat[1:], lon[1:], u[1:], u[1:], p[1:])
    bad = u.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mod.band_diagnostics(lat, lon, bad, u, p)
    lat, lon, p = _grid(nlon=32)
    with pytest.raises(ValueError, match="Nyquist"):
        mod.band_diagnostics(lat, lon, _wave(lat, lon, 5), _wave(lat, lon, 5), p)


def test_tail_band_sits_inside_the_t31_retained_spectrum():
    mod = _load()
    assert mod.retained_zonal_cut(31) == 20
    assert mod.TAIL_M[1] == mod.retained_zonal_cut(31)
    assert mod.retained_zonal_cut(23) < mod.TAIL_M[1]  # T23 would be refused
