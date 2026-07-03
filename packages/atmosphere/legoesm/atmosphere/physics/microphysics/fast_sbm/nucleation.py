"""CCN activation / drop nucleation (oracle ``JERNUCL01_KS``/``WATER_NUCLEATION``).

Köhler-theory activation: an aerosol particle of dry radius ``r_d`` activates
into a cloud droplet when the ambient supersaturation ``s = S − 1`` exceeds
its critical supersaturation. Equivalently, at supersaturation ``s`` every
aerosol larger than the **critical dry radius**

    r_crit = (A/3) · (4 / (B s²))^(1/3)          (oracle ``RCRITI``)

activates, where (oracle ``AKOE``/``BKOE``):

    A = 2 σ_w / (ρ_w R_v T)        Kelvin (curvature) term [m]
    B = i · M_w / M_s · ρ_s/ρ_w    Raoult (solute) hygroscopicity [-]

(the oracle writes ``BKOE = i·m_aero⁻¹·(4/3)π ρ_s`` per-volume and pairs it
with ``r_d³`` — algebraically the same dimensionless ``B`` once the ``(4/3)π
ρ_s r_d³`` solute mass is formed). The activated NUMBER is the integral of a
prescribed log-normal aerosol mode above ``r_crit``; each new droplet is
seeded at a small multiple of its dry radius (haze size) and added to the
liquid spectrum's smallest bins.

This fills the column adapter's "supersaturated clear cell stays clear" gap
with a real activation closure. It reuses ``grid.lognormal_cdf`` (the curve
promoted from ``sdm.init``) for the above-threshold number — no re-derivation.

A is built from ``constants.sigma_water``/``R_v``/``rho_water`` (the oracle's
``AKOE = 3.3e-5/T`` CGS rounding ≈ the same to <1%); aerosol properties live
in :class:`legoesm...fast_sbm.config.FastSBMConfig`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    bin_mass_widths,
    lognormal_cdf,
    radius_from_mass,
)

__physics_contract__ = {
    "summary": (
        "Köhler-theory CCN activation: aerosols above the critical dry "
        "radius r_crit=(A/3)(4/(B s²))^{1/3} activate into cloud droplets "
        "(oracle JERNUCL01_KS/WATER_NUCLEATION)."
    ),
    "inputs": {
        "S": "1 (saturation ratio e/e_s)",
        "T": "K",
        "n_aerosol_available": "1/m^3 (unactivated CCN reservoir)",
    },
    "outputs": {
        "n_activated": "1/m^3 (newly nucleated droplet number)",
        "df": "m^-3 kg^-1 (added liquid distribution at haze size)",
    },
    "sign_convention": (
        "Activation only for s = S−1 > 0; n_activated >= 0 and is "
        "monotonically increasing in s (lower r_crit) and in the aerosol "
        "reservoir; subsaturated air activates nothing. The added droplet "
        "mass is taken from vapor by the subsequent condensation step "
        "(this module only seeds number + haze mass)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Köhler (1936); Khain & Pokrovsky (2002); WRF "
        "module_mp_fast_sbm.F JERNUCL01_KS / WATER_NUCLEATION"
    ),
    "idealized_test": (
        "r_crit ∝ s^{-2/3} and matches the oracle RCRITI formula; zero "
        "activation at or below water saturation; activated number rises "
        "monotonically with s and saturates at the aerosol reservoir; "
        "seeded droplets land in the smallest liquid bins; differentiable "
        "in (S, T)."
    ),
}

# 1e-12: dimensionless supersaturation floor keeping r_crit ∝ s^(-2/3) finite at s→0
_S_FLOOR = 1.0e-12


def kelvin_coefficient(T: jax.Array) -> jax.Array:
    """Köhler curvature term ``A = 2 σ_w / (ρ_w R_v T)`` [m] (oracle AKOE)."""
    return 2.0 * constants.sigma_water / (
        constants.rho_water * constants.R_v * T)


def hygroscopicity(config: FastSBMConfig) -> jax.Array:
    """Dimensionless Raoult term ``B = i · M_w/M_s · ρ_s/ρ_w`` (oracle BKOE
    once the solute mass is formed)."""
    return (config.aerosol_ions * constants.M_h2o / config.aerosol_molar_mass
            * config.aerosol_solute_density / constants.rho_water)


def critical_dry_radius(
    S: jax.Array, T: jax.Array, config: FastSBMConfig = FastSBMConfig()
) -> jax.Array:
    """Critical dry aerosol radius ``r_crit`` [m] activating at ``S``
    (oracle ``RCRITI = (AKOE/3)(4/(BKOE s²))^{1/3}``). Returns ``+inf`` at
    or below saturation (nothing activates)."""
    s = S - 1.0
    A = kelvin_coefficient(T)
    B = hygroscopicity(config)
    # Two float32 AD hazards, both avoided here (surfaced by the end-to-end
    # grad test):
    #  1. A constant `jnp.inf` in a `where(s>0, r_crit, inf)` branch poisons
    #     reverse-mode AD (inf·0 = NaN). So r_crit is evaluated on a floored
    #     s everywhere (finite, smooth) and the caller's `S>1` gate zeros the
    #     subsaturated activation — same result, no inf.
    #  2. Writing `(4/(B s²))^(1/3)` makes the reciprocal's VJP carry `1/u²`
    #     with u = B s² ≈ 1e-24, i.e. ≈1e48 → overflows float32 to inf, then
    #     inf·0 = NaN. The algebraically identical `s^(-2/3)` form has VJP
    #     `s^(-5/3)` (≈1e20 at the floor) which stays in float32 range.
    s_safe = jnp.maximum(s, _S_FLOOR)
    return (A / 3.0) * (4.0 / B) ** (1.0 / 3.0) * s_safe ** (-2.0 / 3.0)


class NucleationResult(NamedTuple):
    n_activated: jax.Array   # newly activated droplet number [1/m^3]
    df: jax.Array            # added liquid distribution (n_bins,) [m^-3 kg^-1]


def activate_ccn(
    S: jax.Array,
    T: jax.Array,
    n_aerosol_available: jax.Array,
    masses: jax.Array,
    config: FastSBMConfig = FastSBMConfig(),
) -> NucleationResult:
    """Activate the aerosol fraction above ``r_crit`` and seed droplets.

    The unactivated reservoir is a prescribed log-normal CCN mode
    (``aerosol_dry_median``, ``aerosol_geom_std``) scaled to
    ``n_aerosol_available``; the activated number is its tail above
    ``r_crit`` via :func:`grid.lognormal_cdf`. New droplets are seeded at
    the smallest liquid bin (haze size ≪ first bin), so they enter as
    number that the condensation step then grows.
    """
    r_crit = critical_dry_radius(S, T, config)
    sigma = jnp.log(jnp.asarray(config.aerosol_geom_std, masses.dtype))
    # Fraction of the lognormal aerosol mode with dry radius > r_crit.
    frac_above = 1.0 - lognormal_cdf(
        jnp.minimum(r_crit, 1.0e30),
        jnp.asarray(config.aerosol_dry_median, masses.dtype), sigma)
    n_activated = jnp.where(S > 1.0, n_aerosol_available * frac_above, 0.0)

    # Seed the activated number into the smallest bin as distribution
    # value (number / bin-mass-width); condensation grows it thereafter.
    df = jnp.zeros_like(masses)
    df = df.at[0].set(n_activated / bin_mass_widths(masses)[0])
    return NucleationResult(n_activated=n_activated, df=df)
