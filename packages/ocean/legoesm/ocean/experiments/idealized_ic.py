"""Shared analytic initial-condition primitives for the ocean experiments.

Several idealized ocean cases hand-rolled the same small analytic building
blocks for their initial conditions.  This module owns the pieces whose maths
is genuinely identical across cases so they are derived once:

* :func:`coriolis_f` / :func:`coriolis_f_safe` — the ``f = 2Ω sin φ`` Coriolis
  parameter (and the equator-safe clamped variant the jet cases use).
* :func:`gaussian_lat_envelope` — the ``exp(−((φ−φ_c)/w)²)`` meridional
  envelope used to localise jets/fronts.
* :func:`thermal_wind_dudz_coeff` / :func:`integrate_thermal_wind_bottom_up`
  — the thermal-wind coefficient ``−gα/f`` and the bottom-up ``∂u/∂z``
  integration with ``u(−H)=0`` and the depth-mean removed.

Scope note (honest): a SINGLE unified ``thermal_wind_balance`` covering all
the idealized cases is NOT achievable byte-for-byte — the cases use genuinely
different discretizations (Eady's bottom-up integral vs Silvestri's analytic
depth-linear solution; zonal-mean ∂T/∂y vs an analytic tanh-front ∂T/∂y;
latlon C-grid u-faces vs MPAS edge-normal projection).  Folding them onto one
discretization would change those cases' ICs and inject geostrophic-adjustment
shocks (an IC *physics* property, not cosmetic).  Only the maths that is the
same in every caller is shared here; each case keeps its own staggering / grid
glue.  These are pure NumPy helpers (host-side IC construction, not the traced
hot loop).
"""
from __future__ import annotations

import numpy as np

from legoesm import constants

__all__ = [
    "coriolis_f",
    "coriolis_f_safe",
    "gaussian_lat_envelope",
    "integrate_thermal_wind_bottom_up",
    "thermal_wind_dudz_coeff",
]

# Equator floor on |f| for the thermal-wind 1/f division (Silvestri jet).
_F_FLOOR = 1.0e-12


def coriolis_f(lat_rad):
    """Coriolis parameter ``f = 2 Ω sin φ`` [s⁻¹].

    ``lat_rad`` may be a scalar or array of latitudes in RADIANS.
    """
    return 2.0 * constants.Omega * np.sin(lat_rad)


def coriolis_f_safe(lat_rad, floor: float = _F_FLOOR):
    """``f = 2 Ω sin φ`` with ``|f|`` floored to ``floor`` near the equator so
    ``1/f`` thermal-wind balance stays finite.  ``lat_rad`` in radians."""
    f = coriolis_f(lat_rad)
    return np.where(np.abs(f) < floor, np.sign(f + 1e-30) * floor, f)


def gaussian_lat_envelope(lat, center, width):
    """Meridional Gaussian envelope ``exp(−((lat − center)/width)²)``.

    ``lat``, ``center`` and ``width`` must share units (all degrees OR all
    radians); the caller supplies them already consistent.  Returns 1 at the
    centre and decays away from it.
    """
    return np.exp(-((lat - center) / width) ** 2)


def thermal_wind_dudz_coeff(f0: float, alpha_T: float,
                            g: float = constants.g) -> float:
    """Thermal-wind shear coefficient ``∂u/∂z = (−gα_T/f₀) ∂T/∂y``.

    Returns ``−gα_T/f₀`` so callers multiply by their ``∂T/∂y``.
    """
    return -g * alpha_T / f0


def integrate_thermal_wind_bottom_up(dTdy, dz, coeff):
    """Integrate ``∂u/∂z = coeff·∂T/∂y`` from the bottom up with ``u(−H)=0``
    and remove the depth mean (the barotropic solver carries it).

    Parameters
    ----------
    dTdy : np.ndarray
        ``∂T/∂y`` per column, shape ``(..., nlev)`` (leading axes free).
    dz : np.ndarray
        Layer thicknesses ``(nlev,)`` in legoESM surface-first order.
    coeff : float
        The thermal-wind coefficient ``−gα_T/f₀`` from
        :func:`thermal_wind_dudz_coeff`.

    Returns
    -------
    (U_baroclinic, U_bar) : tuple of np.ndarray
        ``U_baroclinic(..., nlev)`` with the depth mean removed, and the
        depth-mean ``U_bar(...)`` (used for the geostrophic SSH balance).
    """
    # Match the hand-rolled loop bit-for-bit: do NOT force-cast the inputs
    # (the old code read ``dz_ref``/``dTdy`` at their native dtype); only the
    # accumulator ``U`` is float64 (old ``U_zonal = np.zeros(..., float64)``).
    # Likewise the per-step product is evaluated in the SAME left-to-right
    # order ``coeff * dTdy[..., k] * dz_half`` (fp multiply is not associative,
    # so callers must pass the same ``dTdy`` factoring they used inline).
    dTdy = np.asarray(dTdy)
    dz = np.asarray(dz)
    nlev = dz.shape[0]
    U = np.zeros(dTdy.shape, dtype=np.float64)
    # Bottom-up: U[k] = U[k+1] + coeff·dTdy[k]·dz_half (centred Δz).
    for k in range(nlev - 2, -1, -1):
        dz_half = 0.5 * (dz[k] + dz[k + 1]) if k + 1 < nlev else dz[k]
        U[..., k] = U[..., k + 1] + coeff * dTdy[..., k] * dz_half
    # Remove the depth mean (the barotropic solver handles it).
    H_col = np.sum(dz)
    U_bar = np.sum(U * dz, axis=-1) / H_col
    return U - U_bar[..., np.newaxis], U_bar
