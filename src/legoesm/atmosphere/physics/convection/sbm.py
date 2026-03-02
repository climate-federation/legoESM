"""Simplified Betts-Miller (SBM) convection scheme.

A relaxation-based convection parameterization for idealized aquaplanet
experiments. Convective columns are relaxed toward a moist adiabatic
temperature profile with an enthalpy-conserving correction.

Algorithm:
1. Compute moist adiabat from surface temperature upward
2. Construct reference moisture profile: q_ref = RH_ref * q_sat(T_moist, p)
3. Apply enthalpy-conserving correction (energy budget closure)
4. Compute CAPE and smooth trigger
5. Relax T and q_v toward reference profiles over timescale tau_c
6. Diagnose precipitation from column moisture convergence

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Frierson, D. M. W. (2007). The Dynamics of Idealized Convection
  Schemes and Their Effect on the Zonally Averaged Tropical Circulation.
  J. Atmos. Sci., 64, 1959-1976.
- Betts, A. K., & Miller, M. J. (1986). A new convective adjustment
  scheme. Part II: Single column tests using GATE wave, BOMEX, ATEX
  and arctic air-mass data sets. Q. J. R. Meteorol. Soc., 112, 693-709.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import (
    saturation_mixing_ratio,
    compute_moist_adiabat,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


def sbm_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: SBMConfig = SBMConfig(),
) -> ConvectionOutput:
    """Compute Simplified Betts-Miller convection tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : SBMConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev) layer thickness

    # 1. Surface temperature as parcel starting point
    T_base = T[:, -1]  # (ncol,)

    # 2. Compute moist adiabatic temperature profile
    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)

    # 3. Reference moisture profile (before enthalpy correction)
    q_ref_raw = config.RH_ref * saturation_mixing_ratio(T_moist, p_full)

    # 4. Enthalpy-conserving correction (Newton iteration)
    # We need delta_T such that:
    #   sum(c_pd * (T_moist + dT - T) + L_v * (RH * q_sat(T_moist + dT, p) - q_v)) * dp = 0
    # Use two Newton iterations for accuracy.
    def _newton_step(T_trial):
        q_trial = config.RH_ref * saturation_mixing_ratio(T_trial, p_full)
        residual = jnp.sum(
            (constants.c_pd * (T_trial - T) + constants.L_v * (q_trial - q_v)) * dp,
            axis=1,
        )  # (ncol,)
        # Clausius-Clapeyron: dq_sat/dT ≈ L_v * q_sat / (R_v * T^2)
        q_sat_trial = saturation_mixing_ratio(T_trial, p_full)
        dqsat_dT = constants.L_v * q_sat_trial / (constants.R_v * T_trial ** 2)
        jacobian = jnp.sum(
            (constants.c_pd + constants.L_v * config.RH_ref * dqsat_dT) * dp,
            axis=1,
        )  # (ncol,)
        dT = -residual / jnp.clip(jacobian, 1.0, None)
        return T_trial + dT[:, None]

    T_ref = _newton_step(T_moist)  # first iteration
    T_ref = _newton_step(T_ref)    # second iteration

    # Reference moisture at converged temperature
    q_ref = config.RH_ref * saturation_mixing_ratio(T_ref, p_full)

    # 5. Compute CAPE for trigger
    cape = compute_cape(T, T_ref, p_full, p_half)  # (ncol,)

    # Smooth trigger: sigmoid(sharpness * (CAPE - threshold))
    trigger = jax.nn.sigmoid(
        config.smooth_trigger_sharpness * (cape - config.CAPE_threshold)
    )  # (ncol,)

    # 6. Relaxation tendencies
    tau_c = config.tau_c
    dT_dt = trigger[:, None] * (T_ref - T) / tau_c  # (ncol, nlev)
    dq_v_dt = trigger[:, None] * (q_ref - q_v) / tau_c  # (ncol, nlev)

    # 7. Precipitation: column-integrated moisture sink
    # precip = -sum(dq_v_dt * dp) / g, clipped >= 0
    precipitation = jnp.clip(
        -jnp.sum(dq_v_dt * dp, axis=1) / constants.g,
        0.0,
        None,
    )  # (ncol,)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        precipitation=precipitation,
        cape=cape,
        convective_mask=trigger,
    )
