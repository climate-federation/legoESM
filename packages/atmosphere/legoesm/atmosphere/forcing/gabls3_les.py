"""GABLS3 LES case (Cabauw, 2 July 2006, 00-09 UTC): the forcing, defined ONCE.

Source: S. Basu, "GABLS3 LES intercomparison: case description", revised
6 Oct 2008 (www4.ncsu.edu/~sbasu5/GABLS3/GABLS3_LES_Revised.pdf). The LES
driver and the SCM it is scored against read the same tables.

TIME: ``t`` is seconds since 2006-07-02 00 UTC (the run start). Series with a
repeated time are step changes; :func:`series_mean` integrates them exactly.

The surface is prescribed θ and q at ``Z_REF_M`` = 0.25 m (Table 1, linear in
time) through Monin–Obukhov similarity; geostrophic wind is Table 4 at the
surface, linear in time, then linear in height to (-2, 2) m/s at 2000 m;
advective tendencies (Tables 5-7) are uniform in 200-800 m and linear to zero at
the ground. The momentum advection enters the LES exactly as a shift of the
geostrophic wind (``du/dt = f(v - v_g) + u_adv = f(v - (v_g - u_adv/f))``).
"""
from __future__ import annotations

import math

import numpy as np

from legoesm import constants

__all__ = [
    "LAT_DEG", "Z0_M", "Z_REF_M", "P_SFC_PA", "THETA_TOP_GRADIENT_K_M",
    "SPONGE_BOTTOM_M", "DURATION_S", "coriolis_parameter", "series_mean",
    "surface_mean", "geostrophic_mean", "advection_mean", "initial_profiles",
    "perturbation_std",
]

LAT_DEG = 51.9711
Z0_M = 0.15
Z_REF_M = 0.25                      # height of the prescribed θ, q (Table 1)
P_SFC_PA = 102210.0                 # Table 1 at 00 UTC, held (approved 2026-10-18)
THETA_TOP_GRADIENT_K_M = 0.0029     # upper-boundary inversion strength
SPONGE_BOTTOM_M = 600.0             # spec: lower limit ~550-600 m (approved 600)
DURATION_S = 9.0 * 3600.0
_H = 3600.0

# --- Table 1: surface (0.25 m) θ [K] and q [kg/kg], linear in time -------------
SURFACE_THETA = tuple((h * _H, th) for h, th in enumerate(
    (291.28, 290.34, 289.45, 288.62, 288.43, 289.95, 292.38, 294.16, 296.55, 298.45)))
SURFACE_Q = tuple((h * _H, q) for h, q in enumerate(
    (0.0100, 0.0099, 0.0100, 0.0099, 0.0099, 0.0104, 0.0109, 0.0113, 0.0121, 0.0129)))

# --- Table 2: initial wind (z [m], U, V [m/s]) ----------------------------------
WIND_INIT = (
    (10, -3.35, -0.04), (20, -4.31, 0.08), (40, -6.13, 0.32), (80, -9.00, 0.90),
    (140, -11.48, 3.50), (200, -10.16, 5.58), (203, -10.08, 5.57),
    (257, -9.15, 5.47), (308, -8.75, 5.54), (310, -8.74, 5.55), (363, -8.57, 5.64),
    (408, -8.46, 5.64), (426, -8.41, 5.61), (465, -8.23, 5.47), (520, -7.82, 5.10),
    (541, -7.60, 4.90), (575, -7.17, 4.53), (635, -6.22, 3.76), (657, -5.82, 3.47),
    (694, -5.14, 3.00), (715, -4.77, 2.76), (749, -4.19, 2.46), (772, -3.86, 2.34),
    (801, -3.54, 2.31), (830, -3.37, 2.46))

# --- Table 3: initial θ [K] and q [kg/kg] (z [m]) -------------------------------
THERMO_INIT = (
    (10, 292.72, 0.0098), (20, 293.02, 0.0097), (40, 293.41, 0.0096),
    (80, 294.30, 0.0096), (140, 295.68, 0.0092), (200, 297.35, 0.0089),
    (203, 297.35, 0.0089), (257, 297.51, 0.0089), (308, 297.66, 0.0089),
    (363, 297.81, 0.0089), (408, 297.94, 0.0089), (465, 298.11, 0.0089),
    (520, 298.27, 0.0088), (575, 298.42, 0.0088), (635, 298.60, 0.0088),
    (694, 298.77, 0.0088), (749, 298.93, 0.0088), (801, 299.08, 0.0088),
    (854, 299.23, 0.0087))

# --- Table 4: surface geostrophic wind, linear in time; linear in z to the top ---
GEO_SFC_U = ((-1 * _H, -6.5), (3 * _H, -5.0), (6 * _H, -5.0), (12 * _H, -6.5))
GEO_SFC_V = ((-1 * _H, 4.5), (3 * _H, 4.5), (6 * _H, 4.5), (12 * _H, 2.5))
GEO_TOP_Z_M, GEO_TOP_U, GEO_TOP_V = 2000.0, -2.0, 2.0

# --- Tables 5-7: advective tendencies (repeated time = step change) ------------
U_ADV = ((-1 * _H, 5.0e-4), (3 * _H, 5.0e-4), (3 * _H, 0.0), (12 * _H, 0.0))   # m/s2
V_ADV = ((-1 * _H, 0.0), (12 * _H, 0.0))
THETA_ADV = ((-12 * _H, -2.5e-5), (1 * _H, -2.5e-5), (1 * _H, 7.5e-5),         # K/s
             (6 * _H, 7.5e-5), (6 * _H, 0.0), (12 * _H, 0.0))
Q_ADV = ((0.0, 0.0), (2 * _H, 0.0), (2 * _H, -8.0e-8), (5 * _H, -8.0e-8),       # 1/s
         (5 * _H, 0.0), (12 * _H, 0.0))
ADV_FULL_M = 200.0          # advection uniform above this height, linear below

# --- initial perturbations, z <= 200 m ------------------------------------------
PERTURB_TOP_M = 200.0
UV_PERTURB_VAR = 0.2        # variance 0.2 (1 - z/200)^2 [m2/s2]
THETA_PERTURB_VAR = 0.1     # variance [K2] (approved 2026-10-18; no q noise, seed 0)


def coriolis_parameter() -> float:
    return 2.0 * constants.Omega * math.sin(math.radians(LAT_DEG))


def series_mean(knots, t0: float, t1: float) -> float:
    """Exact mean over ``[t0, t1]`` of the piecewise-linear series ``knots`` ((t, v)
    pairs, non-decreasing t, a repeated t is a step), held constant outside them.
    ``t0 == t1`` gives the right-continuous value."""
    ts = np.asarray([k[0] for k in knots], float)
    vs = np.asarray([k[1] for k in knots], float)
    if np.any(np.diff(ts) < 0):
        raise ValueError("series times must be non-decreasing")

    def at(t):                      # right-continuous value
        i = int(np.searchsorted(ts, t, side="right"))
        if i == 0:
            return vs[0]
        if i == len(ts):
            return vs[-1]
        a, b = ts[i - 1], ts[i]
        return vs[i - 1] + (vs[i] - vs[i - 1]) * (t - a) / (b - a)

    if t1 < t0:
        raise ValueError("series_mean needs t1 >= t0")
    if t1 == t0:
        return float(at(t0))
    # breakpoints strictly inside the interval; each sub-interval is linear, so the
    # trapezoid of its right-limit start and left-limit end is exact
    cuts = [t0] + sorted({float(t) for t in ts if t0 < t < t1}) + [t1]
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = 0.5 * (a + b)
        total += (b - a) * at(m)    # linear on (a, b): the midpoint value is the mean
    return total / (t1 - t0)


def surface_mean(t0: float, t1: float) -> tuple[float, float]:
    """Interval-mean prescribed (θ, q) at ``Z_REF_M``."""
    return series_mean(SURFACE_THETA, t0, t1), series_mean(SURFACE_Q, t0, t1)


def geostrophic_mean(z, t0: float, t1: float):
    """Interval-mean geostrophic wind profiles (u_g, v_g) at heights ``z``."""
    z = np.asarray(z, float)
    frac = z / GEO_TOP_Z_M
    us, vs = series_mean(GEO_SFC_U, t0, t1), series_mean(GEO_SFC_V, t0, t1)
    return us + (GEO_TOP_U - us) * frac, vs + (GEO_TOP_V - vs) * frac


def advection_shape(z):
    return np.clip(np.asarray(z, float) / ADV_FULL_M, 0.0, 1.0)


def advection_mean(z, t0: float, t1: float) -> dict:
    """Interval-mean advective tendency profiles ``u, v`` [m/s2], ``theta`` [K/s],
    ``q`` [1/s] at heights ``z`` (``∂φ/∂t = … + adv``)."""
    s = advection_shape(z)
    return {k: series_mean(tab, t0, t1) * s
            for k, tab in (("u", U_ADV), ("v", V_ADV), ("theta", THETA_ADV),
                           ("q", Q_ADV))}


def initial_profiles(z):
    """Initial (θ, q, u, v) at heights ``z`` (linear between table levels, held
    beyond the end levels)."""
    z = np.asarray(z, float)
    zt = [k[0] for k in THERMO_INIT]
    zw = [k[0] for k in WIND_INIT]
    return (np.interp(z, zt, [k[1] for k in THERMO_INIT]),
            np.interp(z, zt, [k[2] for k in THERMO_INIT]),
            np.interp(z, zw, [k[1] for k in WIND_INIT]),
            np.interp(z, zw, [k[2] for k in WIND_INIT]))


def perturbation_std(z):
    """Standard deviations (σ_uv, σ_θ) of the initial noise at heights ``z``."""
    z = np.asarray(z, float)
    below = z <= PERTURB_TOP_M
    ramp = np.where(below, 1.0 - z / PERTURB_TOP_M, 0.0)
    return math.sqrt(UV_PERTURB_VAR) * ramp, np.where(below,
                                                      math.sqrt(THETA_PERTURB_VAR), 0.0)
