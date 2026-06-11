"""Immersion freezing of supercooled drops (oracle ``FREEZ``, Bigg 1953).

Below 0 °C every liquid bin freezes stochastically at the volume-proportional
Bigg rate (oracle ``FREEZ``, Fortran l. 6560–6636):

    P_k = m_k · A · exp(−B(m_k) · ΔT)         ΔT = T − T_freeze  (< 0)
    B(m) = B0 + (B_max − B0)/m_top · m         (mass-dependent slope; with
                                                the oracle defaults B_max=B0
                                                so B is constant)
    frozen fraction = 1 − exp(−P_k · dt)

The frozen liquid leaves the drop spectrum and enters the ice spectrum
(oracle routes ``KR ≤ KRFREEZ`` to pristine ice crystals and the larger
bins to hail; this warm→ice first step produces a single mass-preserving
ice spectrum and defers the crystal/hail habit split to the ice-collision
iteration). Freezing releases the latent heat of fusion:

    ΔT_heat = (L_f / c_pd) · Δq_ice           (oracle ``333·SUM_ICE/RO`` —
                                                333 ≈ L_f/c_pd in CGS)

The oracle's Bigg coefficients are CGS (drop mass in grams); the port keeps
them as ``FastSBMConfig`` fields and converts ``masses`` kg→g internally so
the dimensionless ``P·dt`` matches the oracle bit-for-bit at equal inputs.
Pure ``jnp``; the ``ΔT < 0`` gate is a ``jnp.where`` (no traced branch).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import mass_density

__physics_contract__ = {
    "summary": (
        "Bigg (1953) immersion freezing of supercooled liquid bins into "
        "ice with latent-heat-of-fusion release (oracle FREEZ)."
    ),
    "inputs": {
        "f_liquid": "m^-3 kg^-1 (drop size distribution)",
        "masses": "kg (bin centres)",
        "T": "K",
        "rho": "kg/m^3",
        "dt": "s",
    },
    "outputs": {
        "f_liquid": "m^-3 kg^-1 (depleted)",
        "f_ice": "m^-3 kg^-1 (frozen drops, same bin masses)",
        "dT": "K (fusion warming this step)",
    },
    "sign_convention": (
        "Freezing acts only for T < T_freeze; it moves number/mass from "
        "liquid to ice at fixed per-bin mass (no mass change), so "
        "q_liquid+q_ice is conserved and ΔT = (L_f/c_pd)·Δq_ice >= 0. "
        "At or above freezing nothing happens."
    ),
    "conserves": ["mass"],
    "differentiable": True,
    "reference": (
        "Bigg (1953) QJRMS 79:510; Khain et al. (2004) JAS 61:2963; WRF "
        "module_mp_fast_sbm.F FREEZ"
    ),
    "idealized_test": (
        "No freezing at T >= 0 °C; frozen fraction 1−exp(−m·A·exp(−B·ΔT)·dt) "
        "matches the oracle per bin; deeper supercooling and larger drops "
        "freeze more; liquid+ice mass conserved exactly; ΔT = "
        "(L_f/c_pd)·Δq_ice; differentiable in T."
    ),
}

# kg → g (oracle Bigg coefficients are CGS, drop mass in grams).
_KG_TO_G = 1.0e3


class FreezeResult(NamedTuple):
    f_liquid: jax.Array   # depleted drop distribution (n_bins,)
    f_ice: jax.Array      # frozen-drop ice distribution (n_bins,)
    dT: jax.Array         # fusion warming [K]


class FreezeRoutedResult(NamedTuple):
    """Bigg freezing with the oracle's habit routing (FREEZ): frozen drops
    in bins ``≤ KRFREEZ`` become pristine ice crystals, the larger bins
    become hail/graupel."""

    f_liquid: jax.Array   # depleted drop distribution (n_bins,)
    f_crystals: jax.Array  # frozen small drops → ice crystals (n_bins,)
    f_hail: jax.Array     # frozen large drops → hail/graupel (n_bins,)
    dT: jax.Array         # fusion warming [K]


def bigg_freezing_rate(
    masses: jax.Array, T: jax.Array, config: FastSBMConfig = FastSBMConfig()
) -> jax.Array:
    """Per-bin Bigg freezing rate ``P_k`` [1/s] (oracle ``PF``); zero at or
    above ``T_freeze``."""
    dT = T - constants.T_freeze
    m_g = masses * _KG_TO_G
    cfreeze = (config.bigg_b_max - config.bigg_b0) / (masses[-1] * _KG_TO_G)
    B = config.bigg_b0 + cfreeze * m_g
    rate = m_g * config.bigg_a * jnp.exp(-B * dT)
    return jnp.where(dT < 0.0, rate, 0.0)


def freeze_step(
    f_liquid: jax.Array,
    masses: jax.Array,
    T: jax.Array,
    rho: jax.Array,
    dt: float | jax.Array,
    config: FastSBMConfig = FastSBMConfig(),
) -> FreezeResult:
    """One Bigg immersion-freezing step on a drop spectrum.

    Returns the depleted liquid spectrum, the frozen-drop ice spectrum
    (same bin masses — freezing preserves particle mass), and the fusion
    warming ``ΔT = (L_f/c_pd)·Δq_ice``.
    """
    P = bigg_freezing_rate(masses, T, config)
    frozen_frac = -jnp.expm1(-P * dt)          # 1 − exp(−P dt), stable
    df = f_liquid * frozen_frac
    f_liquid_new = f_liquid - df
    dq_ice = mass_density(df, masses) / rho
    dT = (constants.L_f / constants.c_pd) * dq_ice
    return FreezeResult(f_liquid=f_liquid_new, f_ice=df, dT=dT)


def freeze_step_routed(
    f_liquid: jax.Array,
    masses: jax.Array,
    T: jax.Array,
    rho: jax.Array,
    dt: float | jax.Array,
    config: FastSBMConfig = FastSBMConfig(),
) -> FreezeRoutedResult:
    """Bigg freezing with the oracle's two-category habit routing.

    Oracle ``FREEZ`` (``module_mp_fast_sbm.F:6614-6618``):
    ``IF(KR<=KRFREEZ) FF2(crystals)+=YK2 ELSE FF5(hail)+=YK2`` — small frozen
    drops nucleate pristine ice crystals, large frozen drops (frozen rain)
    become hail/graupel. The freezing *rate* and fusion heat are identical
    to :func:`freeze_step`; only the destination category differs by bin.
    Total frozen mass and the depleted liquid are unchanged, so any caller
    that sums the two categories recovers the single-category result
    exactly.
    """
    P = bigg_freezing_rate(masses, T, config)
    frozen_frac = -jnp.expm1(-P * dt)
    df = f_liquid * frozen_frac
    f_liquid_new = f_liquid - df
    # Bin split: bins 0..KRFREEZ-1 (0-based) → crystals, ≥ KRFREEZ → hail.
    is_crystal = jnp.arange(masses.shape[0]) < config.krfreeze
    f_crystals = jnp.where(is_crystal, df, 0.0)
    f_hail = jnp.where(is_crystal, 0.0, df)
    dq_ice = mass_density(df, masses) / rho
    dT = (constants.L_f / constants.c_pd) * dq_ice
    return FreezeRoutedResult(f_liquid=f_liquid_new, f_crystals=f_crystals,
                              f_hail=f_hail, dT=dT)
