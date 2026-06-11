"""Mass-doubling bin grid + spectral moments for the fast-SBM scheme.

Oracle: WRF ``phys/module_mp_fast_sbm.F`` (FSBM-2, Hebrew University Cloud
Model; Khain et al. 2004 JAS 61:2963, Shpund et al. 2019 JGR 124:9800). The
oracle evolves four size-distribution functions (aerosol, drops, snow,
graupel/hail) on ``NKR=33`` mass-doubling bins (``m_{k+1} = 2 m_k``) and
*reads* the bin masses from data files; here the grid is reconstructed
analytically from the smallest drop radius (2 um), which reproduces the
oracle's documented landmarks: bin 15 ~= 50.8 um (its cloud/rain boundary,
``KRDROP=Bin 15 --> 50um``) and bin 33 ~= 3.25 mm (the FSBM drop-spectrum
upper end).

Distribution convention (oracle ``FAST_SBM`` QC/QNC diagnostics):

    f_k    = size-distribution value at bin k, number per (volume x mass),
             SI ``[m^-3 kg^-1]`` (oracle CGS: ``cm^-3 g^-1``)
    dm_k   = 3 COL m_k = ln(2) m_k   exact mass width of a doubling grid,
             where COL = ln(2)/3 = 0.23105... is the oracle's log-radius
             increment (``DOUBLE PRECISION, PARAMETER :: COL = 0.23105``)
    number density ``[m^-3]``    = sum_k f_k dm_k
    mass density   ``[kg m^-3]`` = sum_k f_k m_k dm_k

This module is SI end-to-end; CGS conversion happens only inside
oracle-comparison tests. All functions are pure jnp and differentiable.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm.init import lognormal_cdf

# Oracle log-radius increment d(ln r) = ln(2)/3 (mass doubles every bin,
# radius doubles every 3 bins). Exact value, not the oracle's 0.23105 rounding.
COL = math.log(2.0) / 3.0

# ln(2) = 3*COL — the exact mass-bin quadrature width factor (dm_k = ln2 * m_k).
_LN2 = math.log(2.0)

# (4/3)π — sphere volume prefactor (pure geometry).
_FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi

# Oracle bin counts (module_mp_fast_sbm.F parameter block: NKR=33,
# NKR_aerosol=43).
NKR_LIQUID = 33
NKR_AEROSOL = 43

# Smallest liquid-drop radius [m]. The oracle reads masses from
# ``masses.asc``; 2.0 um reproduces its documented bin-15 ~ 50 um cloud/rain
# boundary (2 um * 2^(14/3) = 50.8 um) and the 3.25 mm spectrum top.
R_MIN_LIQUID = 2.0e-6

# Oracle cloud/rain split: bins 1..KRDROP are cloud drops, KRDROP+1..NKR rain
# (``KRDROP=Bin 15 --> 50um``).
KRDROP = 15


def mass_doubling_grid(
    n_bins: int = NKR_LIQUID,
    r_min: float = R_MIN_LIQUID,
    rho: float = constants.rho_water,
    dtype=jnp.float64,
) -> jax.Array:
    """Bin-centre masses ``m_k = m_1 2^(k-1)`` [kg], ``m_1 = (4/3)πρ r_min³``.

    ``2.0**k`` is exact in floating point, so the doubling property
    ``m[k+1] == 2 m[k]`` holds bitwise.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    m1 = _FOUR_THIRDS_PI * rho * float(r_min) ** 3
    return m1 * (2.0 ** jnp.arange(n_bins, dtype=dtype))


def radius_from_mass(mass: jax.Array, rho: float = constants.rho_water) -> jax.Array:
    """Sphere radius ``r = (3 m / (4πρ))^(1/3)`` [m] (oracle DROPRADII)."""
    return jnp.cbrt(mass / (_FOUR_THIRDS_PI * rho))


def bin_mass_widths(masses: jax.Array) -> jax.Array:
    """Quadrature mass width ``dm_k = 3 COL m_k = ln(2) m_k`` [kg].

    This is the oracle's ``dm = 3*col*xl(kr)`` — the midpoint rule in ln(m)
    (d ln m = ln 2 per bin), used for every spectral moment.
    """
    return _LN2 * masses


def number_density(f: jax.Array, masses: jax.Array) -> jax.Array:
    """0th moment: number per volume ``[m^-3]``, ``sum_k f_k dm_k``.

    ``f`` has bins on the last axis (``(..., n_bins)``).
    """
    return jnp.sum(f * bin_mass_widths(masses), axis=-1)


def mass_density(f: jax.Array, masses: jax.Array) -> jax.Array:
    """1st moment: condensed mass per volume ``[kg m^-3]``,
    ``sum_k f_k m_k dm_k`` (oracle: ``QC = (1/ρ) Σ COL·f·x²·3``)."""
    return jnp.sum(f * masses * bin_mass_widths(masses), axis=-1)


def f_from_bin_mixing_ratios(
    q_bins: jax.Array, masses: jax.Array, rho_air: jax.Array
) -> jax.Array:
    """Distribution ``f_k`` from per-bin mass mixing ratios ``q_k`` [kg/kg].

    The WRF oracle carries one tracer per bin (``chem_new``): the mass mixing
    ratio of bin k, ``q_k = f_k m_k dm_k / ρ_air`` (its QC diagnostic is just
    ``Σ_k q_k``). This inverts that: ``f_k = q_k ρ_air / (m_k dm_k)``
    — the oracle's ``ρ/(3 COL x²)`` conversion in SI.

    ``q_bins``: ``(..., n_bins)``; ``rho_air``: broadcastable to ``(...,)``
    (unsqueezed on the bin axis internally).
    """
    return q_bins * rho_air[..., None] / (masses * bin_mass_widths(masses))


def bin_mixing_ratios_from_f(
    f: jax.Array, masses: jax.Array, rho_air: jax.Array
) -> jax.Array:
    """Per-bin mass mixing ratios ``q_k = f_k m_k dm_k / ρ_air`` [kg/kg]."""
    return f * masses * bin_mass_widths(masses) / rho_air[..., None]


def discretize_lognormal(
    masses: jax.Array,
    n_total: float | jax.Array,
    r_median: float | jax.Array,
    geom_std: float | jax.Array,
    rho: float = constants.rho_water,
) -> jax.Array:
    """Project a log-normal number distribution onto the bin grid.

    Bin k receives the *exact* number fraction between its geometric radius
    edges ``r_k 2^(∓1/6)`` (mass edges ``m_k 2^(∓1/2)``) via CDF differences,
    then converts to the distribution value ``f_k = ΔN_k / dm_k``. By
    construction ``number_density(f) = n_total * (CDF(r_top) − CDF(r_bot))``
    — exact up to spectrum truncation by the grid — while the mass moment is
    midpoint-approximate (few % for typical geometric widths).

    Args:
        masses: bin-centre masses [kg], shape ``(n_bins,)``.
        n_total: total number concentration of the mode ``[m^-3]``.
        r_median: median (geometric-mean) radius of the mode [m].
        geom_std: geometric standard deviation (> 1).
        rho: bulk particle density [kg/m^3] mapping mass to radius.

    Returns:
        ``f`` with shape ``(n_bins,)`` in ``[m^-3 kg^-1]``.
    """
    radii = radius_from_mass(masses, rho)
    sigma = jnp.log(jnp.asarray(geom_std, masses.dtype))
    # Geometric bin edges in radius: mass edges m_k 2^(±1/2) → radius edges
    # r_k 2^(±1/6).
    r_lo = radii * 2.0 ** (-1.0 / 6.0)
    r_hi = radii * 2.0 ** (+1.0 / 6.0)
    delta_n = n_total * (
        lognormal_cdf(r_hi, r_median, sigma) - lognormal_cdf(r_lo, r_median, sigma)
    )
    return delta_n / bin_mass_widths(masses)
