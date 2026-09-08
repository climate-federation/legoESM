"""Wangara Day 33 convective boundary layer: the forcing, defined ONCE.

Wangara Day 33 (Clarke et al. 1971; the Andre et al. 1978 / Yamada-Mellor
benchmark) is a dry convective boundary layer driven by a diurnal surface
heat flux under a height-dependent easterly geostrophic wind. Both sides of an
LES-vs-SCM comparison have to be driven by the SAME forcing, so it lives here
rather than in either driver: a single-column model scored against an LES that
was given a different surface flux or a different geostrophic profile measures
the difference between the two forcings, not the difference between the
turbulence closures.

TIME CONVENTION, which is the easy thing to get wrong: ``t_seconds`` is
ABSOLUTE seconds since local midnight, not seconds since the run started. The
run conventionally begins at 09:00 (``T_START_S``). Passing run-relative time
would evaluate the diurnal cosine at hour 0 instead of hour 9 and start the
convective case with a NEGATIVE surface heat flux.

This is NOT the Nieuwstadt CBL_N91 case, which is a different dry convective
boundary layer: constant +0.06 K m/s surface flux, no Coriolis, no geostrophic
wind and theta = 300 K. ``results/les_ref/wangara`` was for a long time a
Nieuwstadt run under a Wangara label; see ``scripts/cluster/les_scm/
production_les.sbatch``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants

__all__ = [
    "LATITUDE_DEG",
    "THETA_INIT_K",
    "T_START_S",
    "coriolis_parameter",
    "surface_theta_flux",
    "surface_qv_flux",
    "geostrophic_u",
]

# --- Wangara Day 33 site and epoch (Clarke et al. 1971; Andre et al. 1978) ---
LATITUDE_DEG = -34.5          # Hay, New South Wales
THETA_INIT_K = 277.0          # radiosonde-average initial potential temperature
T_START_S = 9.0 * 3600.0      # the benchmark starts at 09:00 local

# --- Diurnal surface fluxes: A cos((hour - peak)/halfwidth * pi) ---
_W_THETA_AMP_K_M_S = 0.216        # kinematic sensible heat flux amplitude
_W_QV_AMP_KG_KG_M_S = 2.29e-5     # kinematic moisture flux amplitude
_DIURNAL_PEAK_HOUR = 13.0         # local time of maximum
_DIURNAL_HALFWIDTH_HOUR = 11.0    # cosine argument scale

# --- Geostrophic wind: easterly, weakening with height, kinked at 1 km ---
_UG_SFC_M_S = -5.5
_UG_SHEAR_LOWER_PER_S = 2.9e-3
_UG_KINK_Z_M = 1000.0
_UG_KINK_M_S = -2.6
_UG_SHEAR_UPPER_PER_S = 1.4e-3


def coriolis_parameter() -> float:
    """``f = 2 Omega sin(lat)`` at the Wangara site [1/s] (negative, southern)."""
    return float(2.0 * constants.Omega * np.sin(np.deg2rad(LATITUDE_DEG)))


def _diurnal(t_seconds, amplitude: float):
    """The shared diurnal cosine. ``t_seconds`` is ABSOLUTE (see module doc)."""
    hour = t_seconds / 3600.0
    return amplitude * jnp.cos(
        (hour - _DIURNAL_PEAK_HOUR) / _DIURNAL_HALFWIDTH_HOUR * jnp.pi)


def surface_theta_flux(t_seconds):
    """Kinematic surface sensible-heat flux [K m/s], positive UP.

    ``t_seconds`` is absolute seconds since local midnight.
    """
    return _diurnal(t_seconds, _W_THETA_AMP_K_M_S)


def surface_qv_flux(t_seconds):
    """Kinematic surface moisture flux [(kg/kg) m/s], positive UP."""
    return _diurnal(t_seconds, _W_QV_AMP_KG_KG_M_S)


def geostrophic_u(z_m):
    """Geostrophic ``u`` [m/s] at height ``z`` [m]; ``v_g`` is zero.

    Easterly at the surface, weakening upward with a kink at 1 km.
    """
    z = jnp.asarray(z_m)
    return jnp.where(
        z < _UG_KINK_Z_M,
        _UG_SFC_M_S + _UG_SHEAR_LOWER_PER_S * z,
        _UG_KINK_M_S + _UG_SHEAR_UPPER_PER_S * (z - _UG_KINK_Z_M),
    )
