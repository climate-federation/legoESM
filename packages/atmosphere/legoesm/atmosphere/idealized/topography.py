"""Shared analytical mountain / terrain constructors for idealized test cases.

Provides Gaussian, conical, and cosine-bell orography on the sphere for use
across DCMIP 2012 (hydrostatic rest-state-with-topography), the
Held-Suarez-with-topography variant, and any other test that needs a
prescribed analytical mountain.

All constructors operate purely on input lon/lat arrays and return surface
geopotential ``phi_s = g * z_s`` (or surface height ``z_s`` when explicitly
named ``..._height``).  No grid-specific logic — callers pass either flat
``(lat, lon)`` 2D arrays (lat-lon grid), 1-D ``(nCells,)`` arrays (MPAS),
``(6, n, n)`` cubed-sphere arrays, or ``(n_lat, n_lon)`` Gaussian-grid
arrays — broadcasting handles the rest.

References
----------
- Williamson et al. (1992) — Test 5 conical mountain.
- Ullrich et al. (2012) — DCMIP 2012 test case document, §2-0-0
  (atmospheric rest state with steep topography).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.williamson_sw_analytic import williamson_5_mountain_height


def great_circle_distance(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    lon_c: float,
    lat_c: float,
    radius: float,
) -> jnp.ndarray:
    """Great-circle distance from a fixed centre point [m].

    Parameters
    ----------
    lon, lat : jnp.ndarray
        Longitude and latitude [rad] of the field points.  Any shape.
    lon_c, lat_c : float
        Centre coordinates [rad].
    radius : float
        Sphere radius [m].
    """
    cos_arg = (
        jnp.sin(lat_c) * jnp.sin(lat)
        + jnp.cos(lat_c) * jnp.cos(lat) * jnp.cos(lon - lon_c)
    )
    cos_arg = jnp.clip(cos_arg, -1.0, 1.0)
    return radius * jnp.arccos(cos_arg)


# ---------------------------------------------------------------------------
# Williamson (1992) Test 5 conical mountain
# ---------------------------------------------------------------------------


def williamson5_cone(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    *,
    h_0: float = 2000.0,
    lon_c: float = 3.0 * jnp.pi / 2.0,  # 270 E
    lat_c: float = jnp.pi / 6.0,         # 30 N
    R_m: float = jnp.pi / 9.0,           # 20 deg in lon/lat plane
) -> jnp.ndarray:
    """Williamson Test 5 conical mountain (planar lon/lat-plane radius).

    Returns surface height ``z_s`` [m].  Thin adapter over the one shared
    definition in :mod:`legoesm.core.williamson_sw_analytic`, which
    applies the Williamson 1992 / ``test_cases.F90:1185`` clipped planar
    radius.

    This body previously wrote ``min(R, sqrt(...))``; the oracle writes
    ``sqrt(min(R^2, ...))``.  Algebraically the same, different rounding
    -- the shared helper carries the oracle's order.
    """
    return williamson_5_mountain_height(lon, lat, center=(lon_c, lat_c),
                                        r0=R_m, h_s0=h_0, xp=jnp)


# ---------------------------------------------------------------------------
# Gaussian mountain
# ---------------------------------------------------------------------------


def gaussian_mountain(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    radius: float,
    *,
    h_0: float = 2000.0,
    lon_c: float = 3.0 * jnp.pi / 2.0,
    lat_c: float = jnp.pi / 6.0,
    sigma: float = 1.5e6,
) -> jnp.ndarray:
    """Gaussian mountain with great-circle distance.

    z_s(d) = h_0 * exp(-(d/sigma)^2)

    Parameters
    ----------
    sigma : float
        Gaussian e-folding width [m].  Default 1500 km gives a realistic
        sub-synoptic mountain.
    """
    d = great_circle_distance(lon, lat, lon_c, lat_c, radius)
    return h_0 * jnp.exp(-((d / sigma) ** 2))


# ---------------------------------------------------------------------------
# DCMIP 2012 §2-0-0 cosine-bell mountain
# ---------------------------------------------------------------------------


def dcmip_2_0_0_mountain(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    radius: float,
    *,
    h_0: float = 2000.0,
    lon_c: float = 3.0 * jnp.pi / 2.0,
    lat_c: float = jnp.pi / 6.0,
    R_m_frac: float = 3.0 / 4.0,
    zeta_frac: float = 1.0 / 16.0,
) -> jnp.ndarray:
    """DCMIP 2012 §2-0-0 ridged cosine-bell mountain (Ullrich et al. 2012).

    z_s(λ, φ) = h_0/2 · (1 + cos(π r/R_m)) · cos²(π r/ζ)   for r ≤ R_m
              = 0                                            otherwise

    where r is the great-circle distance from (λ_c, φ_c).  R_m and ζ
    default to ``3π/4 R`` and ``π/16 R`` respectively.

    The resulting surface has a broad envelope (≈3π/4 radians wide,
    ~7000 km) modulated by a sharp ridged structure (~1300 km
    wavelength).  This combination is what makes the test stress
    pressure-gradient-force consistency on terrain-following
    coordinates.

    Returns
    -------
    z_s : jnp.ndarray
        Surface height [m] at each (lon, lat) point.
    """
    R_m = R_m_frac * radius
    zeta = zeta_frac * radius

    d = great_circle_distance(lon, lat, lon_c, lat_c, radius)

    envelope = 0.5 * h_0 * (1.0 + jnp.cos(jnp.pi * d / R_m))
    ridges = jnp.cos(jnp.pi * d / zeta) ** 2
    z_s_full = envelope * ridges
    return jnp.where(d < R_m, z_s_full, 0.0)


# ---------------------------------------------------------------------------
# Convenience: convert z_s to surface geopotential phi_s = g * z_s
# ---------------------------------------------------------------------------


def surface_geopotential(z_s: jnp.ndarray) -> jnp.ndarray:
    """Convert surface height [m] to surface geopotential [m^2/s^2]."""
    return constants.g * z_s
