"""rsdt fix: the TOA-incident-SW diagnostic must be the PRESCRIBED insolation
(``S_0 cos(SZA)`` / daily-mean), not the quadratically-extrapolated top-halo
downwelling flux.

``PhysicsPipeline.compute_radiation_core`` previously set ``sw_down_toa`` (which
the CMOR ``rsdt`` reads) from ``sw_flux_down`` at the top halo, which at the
time was a range-limited quadratic extrapolation ~15 % below the true TOA
insolation (≈330 vs ≈340 W/m^2, C48; that overwrite has since been removed
because it zeroed the top layer's radiation).  ``_toa_insolation`` returns the
incoming solar with the EXACT convention the radiation solver uses, so
``rsdt`` is exact and consistent with ``rsut`` (same ``S_0``/zenith).  These
tests pin that behaviour on the method.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
)
from legoesm.driver.physics_pipeline import PhysicsPipeline

S0 = float(constants.S_0)


def _toa(diurnal, lat, lon, doy, seconds_of_day, s_0=S0):
    """Call the method with a minimal stub (it only reads ``diurnal_cycle``)."""
    stub = SimpleNamespace(diurnal_cycle=diurnal)
    return PhysicsPipeline._toa_insolation(stub, lat, lon, doy, seconds_of_day, s_0)


# Equator / mid-lat / high-lat, longitudes spread so some columns are in night.
_LAT = jnp.deg2rad(jnp.array([[0.0, 45.0, -80.0]]))
_LON = jnp.deg2rad(jnp.array([[0.0, 90.0, 180.0]]))
_DOY = 80.0  # near equinox


def test_diurnal_matches_prescribed_solar_insolation():
    """Diurnal rsdt == S_0 * max(cos(SZA), 0) — the solver's instantaneous
    incoming (non-vacuous: this is NOT the clamped halo value)."""
    hour = 12.0
    out = _toa(True, _LAT, _LON, _DOY, hour * 3600.0)
    ref = S0 * jnp.maximum(cos_zenith_angle(_LAT, _LON, _DOY, hour), 0.0)
    assert np.allclose(np.asarray(out), np.asarray(ref))


def test_subsolar_recovers_full_insolation():
    """At the subsolar point rsdt == S_0 (full TOA insolation), the exact
    quantity the ~15%-low halo clamp used to suppress."""
    # Equinox, local noon on the equator at lon=0 -> sun overhead -> cos(SZA)=1.
    lat = jnp.deg2rad(jnp.array([[0.0]]))
    lon = jnp.deg2rad(jnp.array([[0.0]]))
    out = float(_toa(True, lat, lon, _DOY, 12.0 * 3600.0)[0, 0])
    assert abs(out - S0) < 1.0  # within 1 W/m^2 of the solar constant


def test_diurnal_night_columns_are_zero():
    """Where the sun is below the horizon (cos(SZA) < 0) rsdt floors at 0 —
    no negative incident flux."""
    # Local midnight on the equator at lon=0 (hour=0) -> night.
    lat = jnp.deg2rad(jnp.array([[0.0]]))
    lon = jnp.deg2rad(jnp.array([[0.0]]))
    out = float(_toa(True, lat, lon, _DOY, 0.0)[0, 0])
    assert out == 0.0


def test_nondiurnal_matches_daily_mean_insolation():
    """Without a diurnal cycle, rsdt == the daily-mean insolation (the
    solver's non-diurnal incoming)."""
    out = _toa(False, _LAT, _LON, _DOY, 0.0)
    ref = daily_mean_insolation(_LAT, _DOY, S0)
    assert np.allclose(np.asarray(out), np.asarray(ref))


def test_bounded_by_solar_constant():
    """rsdt is a physical incident flux: 0 <= rsdt <= S_0 everywhere, both
    modes, across a full day of longitudes."""
    for diurnal in (True, False):
        for hour in (0.0, 6.0, 12.0, 18.0):
            out = np.asarray(_toa(diurnal, _LAT, _LON, _DOY, hour * 3600.0))
            assert np.all(out >= 0.0)
            assert np.all(out <= S0 + 1e-6)
