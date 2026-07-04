"""Diurnal cycle of daily-mean shortwave (NEMO sbcdcy, ln_dm2dc).

Faithful port of ``nemo_5.0.1/src/OCE/SBC/sbcdcy.F90`` (Bernie et al.
2007, Appendix A): redistribute a DAILY-MEAN downwelling shortwave flux
over the day with the analytic clear-sky diurnal shape, preserving the
daily mean exactly.  The ORCA1 oracle runs ``ln_dm2dc = .true.`` on
daily CORE-II radiation, so a faithful comparison needs the same
modulation.

Formulation (per cell; all times in FRACTIONS of a day):

* solar-elevation shape  ``E(t) = raa + rbb·cos(2πt + rcc)`` with
  ``raa = sin(lat)·sin(δ)``, ``rbb = cos(lat)·cos(δ)``,
  ``rcc = lon_rad − π`` and the orbit declination
  ``δ = −23.5° · cos(2π·(11 + day_of_year)/year_len)`` (winter solstice
  pinned to 21 December).
* dawn/dusk are the roots of ``E``; the daily normalisation is
  ``rscal = 1/∫_dawn^dusk E dt`` (daylight shorter than 1/1000 day is
  zeroed — NEMO ticket #1040); the per-window factor is
  ``F = ∫_window E dt · rscal / (t_up − t_lo)`` so that
  ``qsr(t) = qsr_daily · F`` integrates back to the daily mean.

References
----------
Bernie, D. J., E. Guilyardi, G. Madec, J. M. Slingo, S. J. Woolnough
(2007). Impact of resolving the diurnal cycle in an ocean-atmosphere
GCM. Part 1. *Climate Dynamics* 29, 575-590.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

# NEMO sbcdcy hard-wired orbit constants (sbcdcy.F90:190-193)
_OBLIQUITY_DEG = 23.5          # orbit declination amplitude
_SOLSTICE_OFFSET_DAYS = 11.0   # days from 21 Dec to 1 Jan
_MIN_DAYLIGHT_FRAC = 0.001     # NEMO ticket #1040 short-daylight guard


def _fintegral(t1, t2, raa, rbb, rcc):
    """∫ E dt between day-fractions t1..t2 (sbcdcy.F90 fintegral)."""
    two_pi = 2.0 * math.pi
    return (raa * t2 + rbb / two_pi * jnp.sin(rcc + two_pi * t2)
            - raa * t1 - rbb / two_pi * jnp.sin(rcc + two_pi * t1))


def diurnal_sw_factor(
    lon_deg,
    lat_deg,
    *,
    day_of_year,
    year_len_days: float = 365.0,
    t_frac_lo,
    t_frac_up,
):
    """Multiplicative diurnal factor for a daily-mean shortwave flux.

    Transliterates ``sbc_dcy`` + ``sbc_dcy_param`` (sbcdcy.F90) with the
    branch logic converted to ``jnp.where`` (elementwise, jit/AD-safe).

    Parameters
    ----------
    lon_deg, lat_deg : arrays (broadcastable)
        Cell longitude/latitude [degrees].
    day_of_year : int or float
        1-based day of the (perpetual, no-leap) year.
    year_len_days : float
        Days in the year (CORE-II NYF: 365).
    t_frac_lo, t_frac_up : float
        The averaging window as day fractions — NEMO uses
        ``zlo = (nsec_day − dt/2)/86400``, ``zup = zlo + nn_fsbc·dt/86400``.
        ``t_frac_lo`` may be slightly negative on the first step of a day
        (exactly as in NEMO); the clipped integrals handle it.

    Returns
    -------
    factor : array, >= 0
        ``qsr_inst = qsr_daily · factor``; the factor's window-weighted
        daily mean is exactly 1 wherever there is daylight, and 0 in
        polar night.
    """
    lon = jnp.asarray(lon_deg, dtype=jnp.float64)
    lat = jnp.asarray(lat_deg, dtype=jnp.float64)
    two_pi = 2.0 * math.pi
    window = float(t_frac_up) - float(t_frac_lo)

    rcc = jnp.deg2rad(lon) - math.pi
    # declination of the earths orbit (winter solstice = 21 December)
    zdsws = _SOLSTICE_OFFSET_DAYS + float(day_of_year)
    zdecrad = (-_OBLIQUITY_DEG * math.pi / 180.0) * math.cos(
        zdsws * two_pi / float(year_len_days))
    raa = jnp.sin(jnp.deg2rad(lat)) * math.sin(zdecrad)
    rbb = jnp.cos(jnp.deg2rad(lat)) * math.cos(zdecrad)
    # time of midday (fraction of day)
    rtmd = jnp.mod(0.5 - lon / 360.0 + 1.0, 1.0)

    # --- dawn / dusk (sbcdcy.F90:203-224) ---
    rbb_safe = jnp.where(jnp.abs(rbb) > 1.0e-300, rbb, 1.0e-300)
    rab = -raa / rbb_safe
    day_lt_24h = jnp.abs(rab) < 1.0
    ztx = 1.0 / two_pi * (jnp.arccos(jnp.clip(rab, -1.0, 1.0)) - rcc)
    ztest = -rbb * jnp.sin(rcc + two_pi * ztx)
    dawn0 = jnp.where(ztest > 0.0, ztx, rtmd - (ztx - rtmd))
    dusk0 = jnp.where(ztest > 0.0, rtmd + (rtmd - ztx), ztx)
    rdawn = jnp.mod(dawn0 + 1.0, 1.0)
    rdusk = jnp.mod(dusk0 + 1.0, 1.0)

    # --- daily normalisation rscal (sbcdcy.F90:228-252) ---
    one_part = rdawn < rdusk
    len_one = rdusk - rdawn
    len_two = rdusk + (1.0 - rdawn)
    int_one = _fintegral(rdawn, rdusk, raa, rbb, rcc)
    int_two = (_fintegral(0.0, rdusk, raa, rbb, rcc)
               + _fintegral(rdawn, 1.0, raa, rbb, rcc))
    day_int = jnp.where(one_part, int_one, int_two)
    day_len = jnp.where(one_part, len_one, len_two)
    day_int_safe = jnp.where(jnp.abs(day_int) > 1.0e-300, day_int, 1.0)
    rscal_lt = jnp.where(day_len >= _MIN_DAYLIGHT_FRAC,
                         1.0 / day_int_safe, 0.0)
    # 24h day / polar night
    int_full = _fintegral(0.0, 1.0, raa, rbb, rcc)
    int_full_safe = jnp.where(jnp.abs(int_full) > 1.0e-300, int_full, 1.0)
    rscal_24 = jnp.where(raa > rbb, 1.0 / int_full_safe, 0.0)
    rscal = jnp.where(day_lt_24h, rscal_lt, rscal_24)

    # --- window integral (sbc_dcy:96-151) ---
    zlo = float(t_frac_lo)
    zup = float(t_frac_up)
    # day time in one part
    zlousd = jnp.minimum(jnp.maximum(zlo, rdawn), zup)
    zupusd = jnp.maximum(jnp.minimum(zup, rdusk), zlo)
    ztmp_one = _fintegral(zlousd, zupusd, raa, rbb, rcc)
    # day time in two parts
    zlo1 = jnp.minimum(zlo, rdusk)
    zup1 = jnp.minimum(zup, rdusk)
    zlo2 = jnp.maximum(zlo, rdawn)
    zup2 = jnp.maximum(zup, rdawn)
    ztmp_two = (_fintegral(zlo1, zup1, raa, rbb, rcc)
                + _fintegral(zlo2, zup2, raa, rbb, rcc))
    ztmp_lt = jnp.where(one_part, ztmp_one, ztmp_two)
    ztmp_24 = _fintegral(zlo, zup, raa, rbb, rcc)
    ztmp = jnp.where(day_lt_24h, ztmp_lt, ztmp_24)

    # qsr_out = qsr_in * ztmp * rscal, with rscal scaled by rday/(dt*nn_fsbc)
    # = 1/window in day-fraction units (sbcdcy.F90:255-256).
    factor = ztmp * rscal / window
    # Polar night (24h, raa <= rbb) has rscal = 0 already; clip tiny
    # negative round-off from the clipped integrals.
    return jnp.maximum(factor, 0.0)
