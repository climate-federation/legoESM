"""TOA insolation (CMOR ``rsdt``) convention + threading tests (#620).

The CMOR ``rsdt`` (TOA incident shortwave) diagnostic must report the
PRESCRIBED TOA insolation the radiation solver is given — ``S_0·cos(SZA)``
(diurnal) / the daily-mean / RCE insolation — its own boundary condition,
not a flux read back out of the solver (at the time of #620 the top-halo SW
flux was a clipped extrapolation ~15% low; that overwrite is gone, but the
boundary condition remains the exact definition).  The fix threads the
prescribed insolation onto a ``RadiationOutput.toa_insolation`` field.

These fast CPU unit tests pin, WITHOUT a full model run or an RRTMGP g-point
compile (gray solver + ``_compute_insolation`` only):

  * the insolation CONVENTION returned by ``_compute_insolation`` — bounded
    ``[0, S_0]`` everywhere, ``== S_0`` at the subsolar point (diurnal),
    ``0`` at night, and daily-mean bounded ``[0, S_0]``;
  * the THREADING of that value onto ``RadiationOutput.toa_insolation`` (the
    gray solver echoes its ``insolation`` argument), and the ``None`` default
    so existing 7-field constructors (e.g. the zero-radiation stub) keep
    working — ``rsdt`` then falls back to the zero halo (correct for no-rad).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.integration import _compute_insolation
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.solar import solar_declination

_S0 = constants.S_0
_NOON_SECONDS = 43200.0  # local solar noon at lon=0 -> hour angle h = 0
_MIDNIGHT_SECONDS = 0.0   # local midnight at lon=0 -> hour angle h = -pi


def test_insolation_bounded_diurnal():
    """(a) Diurnal ``S_0·max(cosθ,0)`` is bounded ``[0, S_0]`` for any
    column / time-of-day."""
    cfg = RadiationConfig(scheme="gray", diurnal_cycle=True)
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 19)
    lon = jnp.linspace(0.0, 2.0 * jnp.pi, 19)
    for sec in (0.0, 21600.0, 43200.0, 64800.0):
        insol, _, _, _ = _compute_insolation(
            lat, cfg, lon=lon, day_of_year=172.0, seconds_of_day=sec,
        )
        a = np.asarray(insol)
        assert np.all(a >= 0.0)
        assert np.all(a <= _S0 + 1.0e-3)


def test_subsolar_column_equals_S0():
    """(b) A subsolar column — latitude == solar declination, hour angle 0 —
    gives ``cosθ = sin²δ + cos²δ = 1`` exactly, so ``insolation == S_0``."""
    cfg = RadiationConfig(scheme="gray", diurnal_cycle=True)
    day = 172.0  # near boreal summer solstice -> declination != 0
    delta = float(solar_declination(day, cfg.gray.obliquity))  # radians
    lat = jnp.array([delta])
    lon = jnp.array([0.0])  # with hour = 12, h = 2*pi*(12/24) + 0 - pi = 0
    insol, cos_sza, f_day, _ = _compute_insolation(
        lat, cfg, lon=lon, day_of_year=day, seconds_of_day=_NOON_SECONDS,
    )
    assert cos_sza is not None
    assert f_day is None  # diurnal path returns no daylight-fraction rescale
    np.testing.assert_allclose(np.asarray(cos_sza), 1.0, atol=1.0e-5)
    np.testing.assert_allclose(np.asarray(insol), _S0, rtol=1.0e-4)


def test_night_column_is_zero():
    """(c) An anti-solar (midnight) column has ``cosθ < 0`` clipped to 0, so
    ``insolation == 0`` exactly."""
    cfg = RadiationConfig(scheme="gray", diurnal_cycle=True)
    lat = jnp.array([0.0])
    lon = jnp.array([0.0])  # with hour = 0, h = -pi -> cos(zenith) = -cos(delta) < 0
    insol, cos_sza, _, _ = _compute_insolation(
        lat, cfg, lon=lon, day_of_year=172.0, seconds_of_day=_MIDNIGHT_SECONDS,
    )
    assert cos_sza is not None
    np.testing.assert_allclose(np.asarray(cos_sza), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(insol), 0.0, atol=0.0)


def test_daily_mean_bounded():
    """(d) Non-diurnal daily-mean insolation is bounded ``[0, S_0]`` at every
    latitude (including the poles) and season."""
    cfg = RadiationConfig(scheme="gray", diurnal_cycle=False)
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 37)
    for day in (1.0, 80.0, 172.0, 264.0, 355.0):  # solstices + equinoxes
        insol, cos_sza, f_day, _ = _compute_insolation(lat, cfg, day_of_year=day)
        a = np.asarray(insol)
        assert cos_sza is None       # daily-mean path: no per-step cos_sza
        assert f_day is not None     # daylight fraction returned
        assert np.all(a >= 0.0)
        assert np.all(a <= _S0 + 1.0e-3)


def test_gray_output_carries_prescribed_insolation():
    """Threading (#620): the gray solver echoes its ``insolation`` argument into
    ``RadiationOutput.toa_insolation`` (what ``rsdt`` reads), independent of the
    SW flux profile (which the top-boundary clamp caps below the true TOA)."""
    ncol, nlev = 2, 4
    p_half = jnp.broadcast_to(
        jnp.array([1.0e3, 2.5e4, 5.0e4, 7.5e4, 1.0e5]), (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.broadcast_to(jnp.linspace(230.0, 290.0, nlev), (ncol, nlev))
    T_sfc = jnp.array([295.0, 280.0])
    lat = jnp.array([0.0, 0.5])
    insolation = jnp.array([_S0 * 0.75, 0.0])  # one daylit column, one night

    out = gray_radiation(
        T=T, p_full=p_full, p_half=p_half, sfc_temperature=T_sfc,
        lat=lat, q_v=None, insolation=insolation, config=GrayRadiationConfig(),
    )
    assert out.toa_insolation is not None
    assert out.toa_insolation.shape == (ncol,)
    np.testing.assert_allclose(
        np.asarray(out.toa_insolation), np.asarray(insolation), atol=0.0,
    )


def test_radiation_output_default_toa_insolation_is_none():
    """The new field is last + defaults to ``None`` so existing 7-field
    constructors (e.g. the zero-radiation stub) keep working; the ``rsdt``
    read then falls back to the zero top-halo SW flux."""
    z_half = jnp.zeros((2, 5))
    z_full = jnp.zeros((2, 4))
    out = RadiationOutput(
        z_half, z_half, z_half, z_half, z_full, z_full, z_full,
    )
    assert out.toa_insolation is None
