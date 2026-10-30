"""Shared 60-level tropical Kuo test sounding (moved from
``tests/unit/test_kuo_oracle_faithful.py`` so other Kuo tests can use it;
the calculations are unchanged, the helpers lost their leading underscore).

The sounding uses the water branch of the oracle's Huang (2018) saturation
curve and oracle constants only to BUILD a humidity profile; the scheme
itself uses legoESM's.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Resolved ascent [m/s] handed to the w_lcl > 0 gate wherever a convergence is
# given (the oracle fixture's L32 value); Kuo raises without one.
W_UP = 0.01


# ---------------------------------------------------------------------------
# 60-level tropical sounding (matches .physics-validator/kuo/oracle).
# Column ordering: index 0 = model top (low p), index 59 = surface.
# ---------------------------------------------------------------------------

NLEV = 60
PS = 1.0e5
P_TOP = 50.0e2
RD = 287.06
G = 9.81
EPS = 287.04 / 461.5


def esat_huang(T):
    aw, bw, cw, d1w, d2w = 34.494, 4924.99, 1.57, 237.1, 105.0
    Tc = T - 273.15
    return np.exp(aw - bw / (Tc + d1w)) / (Tc + d2w) ** cw


def qsat_huang(p, T):
    e = esat_huang(T)
    return EPS * e / (p - e * (1.0 - EPS))


def build_sounding():
    """Return (T, qv, p_full, p_half, ptenq); p_half is (1, NLEV + 1), the rest (1, NLEV)."""
    ph = np.linspace(P_TOP, PS, NLEV + 1)
    p = 0.5 * (ph[:-1] + ph[1:])
    z = -8000.0 * np.log(p / PS)
    T = 300.0 - 6.5e-3 * z
    T = np.maximum(T, 196.0)
    T = T + 4.0 * np.exp(-((p - 150e2) / 60e2) ** 2)
    qs = qsat_huang(p, T)
    sigma = p / PS
    RH = 0.85 * np.clip((sigma - 0.15) / (1.0 - 0.15), 0.0, 1.0) + 0.1
    RH = np.clip(RH, 0.05, 0.95)
    qv = np.maximum(RH * qs, 1e-7)
    # Positive low/mid-tropospheric moisture-convergence bump.
    peak = 3.0e-3 / 86400.0
    ptenq = peak * np.exp(-((p - 850e2) / 120e2) ** 2)
    to = lambda a: jnp.asarray(a)[None, :]
    return to(T), to(qv), to(p), to(ph), to(ptenq)
