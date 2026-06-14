"""Ice melting (oracle ``J_W_MELT``, Jiwen Fan constant-timescale melting).

Above 0 °C ice melts to liquid at a size-dependent rate (oracle ``J_W_MELT``,
Fortran l. 6638–6745): the smallest bins melt completely in one step; mid
bins melt a fraction ``meltrate·dt`` with ``meltrate = 0.5/50 s⁻¹``; the
largest bins melt at ``0.683/120 s⁻¹``. Melted mass moves to the drop
spectrum at the same bin mass and the column cools by the latent heat of
fusion:

    ΔT = −(L_f/c_pd)·Δq_melt           (oracle ``−333·SUM_ICE/RO``)

The oracle applies category-specific bin thresholds (ice crystals ``KR≤10``
full-melt, snow ``KR≤14``, graupel/hail ``KR≤13``); this single-ice-spectrum
port exposes the thresholds as ``FastSBMConfig`` fields (snow defaults) and
documents that per-habit thresholds arrive with the multi-ice-category
iteration. Mirror image of ``freezing.py`` (same fusion constant, opposite
sign). Pure ``jnp``; the ``ΔT ≥ 0`` gate is a ``jnp.where``.
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
        "Constant-timescale ice melting to liquid above 0 °C with latent-"
        "heat-of-fusion cooling (oracle J_W_MELT)."
    ),
    "inputs": {
        "f_ice": "m^-3 kg^-1 (ice size distribution)",
        "masses": "kg (bin centres)",
        "T": "K",
        "rho": "kg/m^3",
        "dt": "s",
    },
    "outputs": {
        "f_ice": "m^-3 kg^-1 (depleted)",
        "f_liquid": "m^-3 kg^-1 (melt water, same bin masses)",
        "dT": "K (fusion cooling this step, <= 0)",
    },
    "sign_convention": (
        "Melting acts only for T >= T_freeze; it moves number/mass from ice "
        "to liquid at fixed per-bin mass, so q_ice+q_liquid is conserved "
        "and ΔT = −(L_f/c_pd)·Δq_melt <= 0. Below freezing nothing happens."
    ),
    "conserves": ["mass"],
    # Differentiable in f_ice (linear) and rho; the T-dependence is only the
    # on/off gate at T_freeze (a step, as in the oracle's IF(DEL_T>=0)) — the
    # constant-rate ladder carries NO temperature sensitivity above freezing,
    # so d/dT of the melt is zero there and undefined exactly at T_freeze.
    "differentiable": True,
    "reference": (
        "Khain et al. (2004) JAS 61:2963; WRF module_mp_fast_sbm.F "
        "J_W_MELT (Jiwen Fan melting)"
    ),
    "idealized_test": (
        "No melting at T < 0 °C; the smallest bins melt fully in one step, "
        "mid/large bins melt the oracle fractions meltrate·dt; ice+liquid "
        "mass conserved; ΔT = −(L_f/c_pd)·Δq_melt; gradient in f_ice is the "
        "per-bin melt fraction (the T-gate is a step, not smooth)."
    ),
}


class MeltResult(NamedTuple):
    f_ice: jax.Array      # depleted ice distribution (n_bins,)
    f_liquid: jax.Array   # melt-water drop distribution (n_bins,)
    dT: jax.Array         # fusion cooling [K] (<= 0)


def melt_fraction(
    n_bins: int, dt: float | jax.Array, config: FastSBMConfig = FastSBMConfig()
) -> jax.Array:
    """Per-bin melted fraction this step (oracle ``J_W_MELT`` rate ladder):
    1 for bins ≤ ``melt_full_bin``, ``min(meltrate_mid·dt,1)`` up to
    ``melt_mid_bin``, ``min(meltrate_high·dt,1)`` above. Clamped to [0,1]
    (the oracle does not clamp but its ``meltrate·dt`` stays < 1 at its
    intended dt; clamping keeps the fraction physical at any dt)."""
    k = jnp.arange(n_bins)
    frac_mid = jnp.minimum(config.melt_rate_mid * dt, 1.0)
    frac_high = jnp.minimum(config.melt_rate_high * dt, 1.0)
    return jnp.where(
        k <= config.melt_full_bin, 1.0,
        jnp.where(k <= config.melt_mid_bin, frac_mid, frac_high))


def melt_step(
    f_ice: jax.Array,
    masses: jax.Array,
    T: jax.Array,
    rho: jax.Array,
    dt: float | jax.Array,
    config: FastSBMConfig = FastSBMConfig(),
) -> MeltResult:
    """One Jiwen-Fan melting step on an ice spectrum.

    Returns the depleted ice spectrum, the melt-water drop spectrum (same
    bin masses), and the fusion cooling ``ΔT = −(L_f/c_pd)·Δq_melt``.
    """
    frac = melt_fraction(masses.shape[0], dt, config)
    active = jnp.where(T >= constants.T_freeze, 1.0, 0.0)
    dm = f_ice * frac * active
    f_ice_new = f_ice - dm
    dq_melt = mass_density(dm, masses) / rho
    dT = -(constants.L_f / constants.c_pd) * dq_melt
    return MeltResult(f_ice=f_ice_new, f_liquid=dm, dT=dT)
