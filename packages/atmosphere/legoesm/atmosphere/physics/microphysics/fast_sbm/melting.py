"""Ice melting (oracle ``J_W_MELT``, Jiwen Fan constant-timescale melting).

At or above 0 °C ice melts to liquid at a size-dependent rate (oracle
``J_W_MELT``, Fortran l. 6638–6745): the smallest bins melt completely in
one step; mid bins melt a fraction ``meltrate·dt`` with
``meltrate = 0.5/50 s⁻¹``; the largest bins melt at ``0.683/120 s⁻¹``
(no clamp, as in the oracle). Melted mass moves to the drop spectrum at the
same bin mass and the column cools by the oracle's fusion literal:

    ΔT = −333 K · Δq_melt               (oracle ``−333·SUM_ICE/RO``)

The bin ladder is category-specific in the oracle (1-based KR): ice
crystals habit 1 full ``≤10``, mid ``<18``; habits 2/3 full ``≤12``, mid
``<20``; snow full ``≤14``, mid ``<22``; graupel and hail full ``≤13``, mid
``<23``. ``category`` selects the ladder for the one spectrum passed; the
snow ladder is the ``FastSBMConfig`` fields, the others are the oracle's
literals. Mirror image of ``freezing.py`` (same fusion literal, opposite
sign). Pure ``jnp``; the ``T ≥ T_freeze`` gate is a ``jnp.where``.
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
        "and ΔT = −333 K·Δq_melt <= 0. Below freezing nothing happens."
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
        "mass conserved; ΔT = −333 K·Δq_melt; gradient in f_ice is the "
        "per-bin melt fraction (the T-gate is a step, not smooth)."
    ),
}


# Oracle J_W_MELT ladders (0-based last full-melt bin, last mid-rate bin)
# for the categories without config fields: crystals ICE=1 (KR<=10, KR<18),
# ICE=2/3 (KR<=12, KR<20), graupel FF4 / hail FF5 (KR<=13, KR<23).
_MELT_LADDERS = {
    "crystals_habit1": (9, 16),
    "crystals_habit23": (11, 18),
    "graupel_hail": (12, 21),
}


class MeltResult(NamedTuple):
    f_ice: jax.Array      # depleted ice distribution (n_bins,)
    f_liquid: jax.Array   # melt-water drop distribution (n_bins,)
    dT: jax.Array         # fusion cooling [K] (<= 0)


def melt_fraction(
    n_bins: int,
    dt: float | jax.Array,
    config: FastSBMConfig = FastSBMConfig(),
    category: str = "snow",
) -> jax.Array:
    """Per-bin melted fraction this step (oracle ``J_W_MELT`` rate ladder):
    1 for bins ≤ full, ``meltrate_mid·dt`` up to mid, ``meltrate_high·dt``
    above — unclamped, as the oracle (``FF − FF·(meltrate·dt)``)."""
    if category == "snow":
        full_bin, mid_bin = config.melt_full_bin, config.melt_mid_bin
    elif category in _MELT_LADDERS:
        full_bin, mid_bin = _MELT_LADDERS[category]
    else:
        raise ValueError(
            f"unknown melt category {category!r}; expected 'snow' or one of "
            f"{sorted(_MELT_LADDERS)}")
    k = jnp.arange(n_bins)
    return jnp.where(
        k <= full_bin, 1.0,
        jnp.where(k <= mid_bin, config.melt_rate_mid * dt,
                  config.melt_rate_high * dt))


def melt_step(
    f_ice: jax.Array,
    masses: jax.Array,
    T: jax.Array,
    rho: jax.Array,
    dt: float | jax.Array,
    config: FastSBMConfig = FastSBMConfig(),
    category: str = "snow",
) -> MeltResult:
    """One Jiwen-Fan melting step on an ice spectrum of ``category``.

    Returns the depleted ice spectrum, the melt-water drop spectrum (same
    bin masses), and the fusion cooling ``ΔT = −333 K·Δq_melt``.
    """
    frac = melt_fraction(masses.shape[0], dt, config, category)
    active = jnp.where(T >= constants.T_freeze, 1.0, 0.0)
    dm = f_ice * frac * active
    f_ice_new = f_ice - dm
    dq_melt = mass_density(dm, masses) / rho
    dT = -constants.lf_over_cp_fast_sbm * dq_melt   # TIN=TIN-333.*SUM_ICE/RO
    return MeltResult(f_ice=f_ice_new, f_liquid=dm, dT=dT)
