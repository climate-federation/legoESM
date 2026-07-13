"""Solar geometry helpers for offline single-column / flux-tower drivers.

Factored from :mod:`scripts.run.run_chats7_offline` so the same
Spencer (1971) declination polynomial is reused by the CHATS7 runner
and the FLUXNET offline loader (and any future ec-site driver).  Any
smooth approximation would work for ``cos(zenith)`` — Spencer is the
same one the CLM/JULES adapter uses in its legacy test helper, so we
keep it as the canonical offline cos_zen provider.

The formula:

    B    = 2 pi * doy_frac / 365
    decl = 0.006918
           - 0.399912 cos B    + 0.070257 sin B
           - 0.006758 cos 2B   + 0.000907 sin 2B
           - 0.002697 cos 3B   + 0.00148  sin 3B
    HA_deg = (utc_hour - 12) * 15 + lon
    cos_zen = max(sin(lat) sin(decl) + cos(lat) cos(decl) cos(HA), 0)

The polynomial coefficients are annotated ``# coeff-ok: Spencer
(1971) declination polynomial`` per the CLAUDE.md param-hygiene
guardrail: they are a published fit, not a tunable knob.
"""

from __future__ import annotations

import math

import numpy as np


def spencer_cos_zen(lat_deg: float, lon_deg: float, doy_frac: float) -> float:
    """Cosine of solar zenith angle (Spencer 1971 declination + hour angle).

    Parameters
    ----------
    lat_deg
        Site latitude [deg N] (positive north).
    lon_deg
        Site longitude [deg E] (accepts either the ``(-180, 180]`` or
        ``[0, 360)`` convention; the function normalises internally).
    doy_frac
        0-based fractional day of year **in UTC**
        (0.0 = Jan 1 00:00 UTC, 1.0 = Jan 2 00:00 UTC, etc.).

    Returns
    -------
    float
        ``max(cos(zenith), 0.0)`` — night values are clipped to zero.
    """
    lat_r = math.radians(lat_deg)
    B = 2.0 * math.pi * doy_frac / 365.0  # coeff-ok: Julian year length (Spencer 1971)
    decl = (
        0.006918                           # coeff-ok: Spencer (1971) declination polynomial
        - 0.399912 * math.cos(B)           # coeff-ok: Spencer (1971) declination polynomial
        + 0.070257 * math.sin(B)           # coeff-ok: Spencer (1971) declination polynomial
        - 0.006758 * math.cos(2 * B)       # coeff-ok: Spencer (1971) declination polynomial
        + 0.000907 * math.sin(2 * B)       # coeff-ok: Spencer (1971) declination polynomial
        - 0.002697 * math.cos(3 * B)       # coeff-ok: Spencer (1971) declination polynomial
        + 0.00148 * math.sin(3 * B)        # coeff-ok: Spencer (1971) declination polynomial
    )
    frac = doy_frac % 1.0
    utc_hour = frac * 24.0                 # coeff-ok: exact hours-per-day (24 h/day)
    lon_norm = lon_deg - 360.0 if lon_deg > 180.0 else lon_deg
    ha_deg = (utc_hour - 12.0) * 15.0 + lon_norm  # coeff-ok: exact 360°/24h = 15°/h
    ha_r = math.radians(ha_deg)
    return max(
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha_r),
        0.0,
    )


def spencer_cos_zen_array(
    lat_deg: float,
    lon_deg: float,
    doy_frac_utc: np.ndarray,
) -> np.ndarray:
    """Vectorised :func:`spencer_cos_zen` over a UTC fractional-day array.

    Same formula as the scalar version, but expressed with ``numpy`` so
    a whole forcing time series can be computed in one call.  Output is
    ``max(cos_zen, 0.0)`` so night values are 0.
    """
    lat_r = math.radians(lat_deg)
    B = 2.0 * math.pi * doy_frac_utc / 365.0  # coeff-ok: Julian year length (Spencer 1971)
    decl = (
        0.006918                              # coeff-ok: Spencer (1971) declination polynomial
        - 0.399912 * np.cos(B)                # coeff-ok: Spencer (1971) declination polynomial
        + 0.070257 * np.sin(B)                # coeff-ok: Spencer (1971) declination polynomial
        - 0.006758 * np.cos(2 * B)            # coeff-ok: Spencer (1971) declination polynomial
        + 0.000907 * np.sin(2 * B)            # coeff-ok: Spencer (1971) declination polynomial
        - 0.002697 * np.cos(3 * B)            # coeff-ok: Spencer (1971) declination polynomial
        + 0.00148 * np.sin(3 * B)             # coeff-ok: Spencer (1971) declination polynomial
    )
    frac = np.mod(doy_frac_utc, 1.0)
    utc_hour = frac * 24.0                    # coeff-ok: exact hours-per-day (24 h/day)
    lon_norm = lon_deg - 360.0 if lon_deg > 180.0 else lon_deg
    ha_deg = (utc_hour - 12.0) * 15.0 + lon_norm  # coeff-ok: exact 360°/24h = 15°/h
    ha_r = np.radians(ha_deg)
    cos_zen = (
        math.sin(lat_r) * np.sin(decl)
        + math.cos(lat_r) * np.cos(decl) * np.cos(ha_r)
    )
    return np.maximum(cos_zen, 0.0)


__all__ = ["spencer_cos_zen", "spencer_cos_zen_array"]
