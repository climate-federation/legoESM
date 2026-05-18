"""Analytical reference formulas for ocean fidelity assessment.

Each function is closed-form and pure: it takes physical parameters and
returns the analytical value expected at infinite resolution / in the
linear regime. Comparisons in :mod:`legoesm.ocean.fidelity.metrics` invoke
these to build pass/fail records against the corresponding measured value.

Notation follows Pedlosky (1987) and Vallis (2017). All physical constants
flow from :mod:`legoesm.constants` (no literals for ``g``, ``Ω``,
``R_earth``, ``ρ_water``).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from legoesm import constants

ArrayLike = Union[float, np.ndarray]


def kelvin_wave_speed(H: float) -> float:
    """Equatorial Kelvin wave phase speed (m/s).

    ``c = sqrt(g · H)`` for a single-layer reduced-gravity ocean of depth H.
    """
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")
    return math.sqrt(constants.g * H)


def igw_omega(k: ArrayLike, l: ArrayLike, H: float, f: float) -> np.ndarray:
    """Inertia-gravity wave angular frequency (rad/s).

    ``ω² = f² + g · H · (k² + l²)``.

    Parameters
    ----------
    k, l : zonal / meridional wavenumber (rad/m)
    H : layer thickness (m)
    f : Coriolis parameter (rad/s)
    """
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")
    k_arr = np.asarray(k, dtype=float)
    l_arr = np.asarray(l, dtype=float)
    return np.sqrt(f * f + constants.g * H * (k_arr * k_arr + l_arr * l_arr))


def rossby_dispersion(k: ArrayLike, l: ArrayLike, beta: float, L_d: float) -> np.ndarray:
    """Barotropic Rossby wave angular frequency (rad/s).

    ``ω = − β · k / (k² + l² + 1 / L_d²)``.

    Sign convention: westward phase propagation (``ω < 0`` for ``k > 0``).
    """
    if L_d <= 0:
        raise ValueError(f"L_d must be positive, got {L_d}")
    k_arr = np.asarray(k, dtype=float)
    l_arr = np.asarray(l, dtype=float)
    denom = k_arr * k_arr + l_arr * l_arr + 1.0 / (L_d * L_d)
    return -beta * k_arr / denom


def eady_growth_rate(f: float, Lambda: float, N: float) -> float:
    """Eady linear-growth rate of the most unstable mode (1/s).

    ``σ_max ≈ 0.31 · |f| · |Λ| / N`` where ``Λ = ∂u/∂z`` is the uniform
    vertical shear and ``N`` is the (constant) Brunt-Väisälä frequency.
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    return 0.31 * abs(f) * abs(Lambda) / N


def eady_most_unstable_wavelength(N: float, H: float, f: float) -> float:
    """Eady most-unstable horizontal wavelength (m).

    ``λ_max ≈ 2π · N · H / (1.61 · |f|)``.
    """
    if H <= 0 or N <= 0:
        raise ValueError("Require H > 0 and N > 0")
    if f == 0:
        raise ValueError("Require f != 0")
    return 2.0 * math.pi * N * H / (1.61 * abs(f))


def stommel_width(r: float, beta: float) -> float:
    """Stommel western-boundary-layer width (m): ``δ_S = r / β``.

    ``r`` is the linear bottom-drag coefficient (1/s).
    """
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return r / beta


def munk_width(A_h: float, beta: float) -> float:
    """Munk western-boundary-layer width (m): ``δ_M = (A_h / β)^(1/3)``.

    ``A_h`` is the harmonic lateral viscosity (m²/s).
    """
    if A_h <= 0:
        raise ValueError(f"A_h must be positive, got {A_h}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return (A_h / beta) ** (1.0 / 3.0)


def sverdrup_transport(
    curl_tau: ArrayLike,
    beta: float,
    rho_0: float = constants.rho_water,
) -> np.ndarray:
    """Sverdrup-balance vertically integrated meridional transport (m²/s).

    ``V_sv = curl(τ_z) / (ρ_0 · β)``.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if rho_0 <= 0:
        raise ValueError(f"rho_0 must be positive, got {rho_0}")
    return np.asarray(curl_tau, dtype=float) / (rho_0 * beta)


def held_larichev_slope() -> float:
    """Held-Larichev geostrophic-turbulence interior EKE slope (dimensionless).

    Returns ``-3.0`` (``E(k) ∝ k^{-3}`` in the enstrophy cascade range).
    """
    return -3.0


def coriolis(lat_rad: ArrayLike) -> np.ndarray:
    """Coriolis parameter ``f = 2 Ω sin(φ)`` (rad/s)."""
    return 2.0 * constants.Omega * np.sin(np.asarray(lat_rad, dtype=float))


def beta_plane(lat_rad: ArrayLike) -> np.ndarray:
    """Meridional gradient of Coriolis: ``β = 2 Ω cos(φ) / R_earth`` (1/(m·s))."""
    return (
        2.0 * constants.Omega * np.cos(np.asarray(lat_rad, dtype=float))
        / constants.R_earth
    )


def thermal_wind_shear(
    rho_y: ArrayLike,
    f: float,
    rho_0: float = constants.rho_water,
) -> np.ndarray:
    """Geostrophic vertical shear ``∂u/∂z = − g · ∂ρ/∂y / (ρ_0 · f)`` (1/s)."""
    if f == 0:
        raise ValueError("f must be nonzero (thermal-wind balance is undefined at the equator)")
    if rho_0 <= 0:
        raise ValueError(f"rho_0 must be positive, got {rho_0}")
    return -constants.g * np.asarray(rho_y, dtype=float) / (rho_0 * f)
