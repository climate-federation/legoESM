"""Kuo column moisture-excess convection scheme.

A column moisture-excess convection scheme (Kuo 1965/1974). Convective
heating and moistening are proportional to the column-integrated moisture
excess above saturation, partitioned by alpha_heat.

Algorithm (Kuo 1965/1974 — column moisture-excess formulation):
1. Compute column moisture excess above saturation [kg/m^2]
2. Smooth sigmoid trigger based on moisture excess
3. Compute moist adiabat reference profile
4. Relax temperature and moisture toward reference profiles
5. Diagnose precipitation from implied condensation

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Kuo, H. L. (1965). On formation and intensification of tropical
  cyclones through latent heat release by cumulus convection.
  J. Atmos. Sci., 22, 40-63.
- Kuo, H. L. (1974). Further studies of the parameterization of the
  influence of cumulus convection on large-scale flow.
  J. Atmos. Sci., 31, 1232-1240.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    compute_moist_adiabat,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import KuoConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


def kuo_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: KuoConfig = KuoConfig(),
) -> ConvectionOutput:
    """Compute Kuo column moisture-excess convection tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : KuoConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)

    # 1. Saturation mixing ratio
    q_sat = saturation_mixing_ratio(T, p_full)  # (ncol, nlev)

    # 2. Column moisture excess: positive part only (zero when subsaturated)
    excess = jnp.maximum(q_v - q_sat, 0.0)  # (ncol, nlev)
    MC = jnp.sum(excess * dp, axis=1) / constants.g  # (ncol,) [kg/m^2]

    # 3. Smooth trigger based on column moisture excess
    trigger = jax.nn.sigmoid(
        config.smooth_trigger_sharpness * (MC - config.me_threshold)
    )  # (ncol,)

    # 4. Moist adiabatic reference profile from surface temperature
    T_base = T[:, -1]  # (ncol,)
    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)

    # 5. Heating tendency: relax toward moist adiabat
    dT_dt = (
        trigger[:, None]
        * config.alpha_heat
        * (T_moist - T)
        / config.tau_relax
    )  # (ncol, nlev)

    # 6. Moistening tendency (budget-consistent with heating)
    #
    # Water conservation requires:
    #   column_integral(dq_v_dt * dp/g) + precipitation = 0
    #
    # The column moisture excess MC is the only moisture source.
    # Fraction alpha_heat goes to condensational heating (→ precipitation).
    # Fraction (1 - alpha_heat) goes to moistening the column.
    #
    # We distribute the moistening budget proportional to the local
    # subsaturation deficit, then normalize so the column integral
    # exactly equals (1 - alpha_heat) * MC / tau_relax.

    # Implied condensation rate from heating (moisture sink)
    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)

    # Subsaturation deficit profile for distributing moistening
    deficit = jnp.maximum(q_sat - q_v, 0.0)  # (ncol, nlev)
    deficit_integral = jnp.sum(deficit * dp, axis=1, keepdims=True) / constants.g  # (ncol, 1)
    deficit_integral_safe = jnp.maximum(deficit_integral, 1e-20)

    # Moistening budget: (1 - alpha_heat) * MC / tau_relax [kg/m^2/s]
    moistening_budget = (
        trigger * (1.0 - config.alpha_heat) * MC / config.tau_relax
    )  # (ncol,)

    # Distribute moistening proportional to deficit, normalized to budget
    dq_v_dt = (
        moistening_budget[:, None]
        * (deficit / deficit_integral_safe)
        * constants.g / dp
    )  # (ncol, nlev) [kg/kg/s]

    # Subtract condensation implied by heating
    dq_v_dt = dq_v_dt - implied_condensation

    # 7. Precipitation = net column moisture removal (water-conservative)
    # P = -∫ dq_v_dt dp/g = condensation_integral - moistening_budget
    precipitation = jnp.clip(
        -jnp.sum(dq_v_dt * dp, axis=1) / constants.g,
        0.0,
        None,
    )  # (ncol,)

    # 8. CAPE diagnostic
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        precipitation=precipitation,
        cape=cape,
        convective_mask=trigger,
    )
