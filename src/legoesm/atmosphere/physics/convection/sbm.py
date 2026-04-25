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
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
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
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
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
    # ``jnp.full`` lowers to a single ``Broadcast`` HLO op; the previous
    # ``broadcast_to(jnp.asarray(scalar, dtype), shape)`` form additionally
    # forced a ``ConvertElementType`` for the implicit promotion of the
    # Python float, which is unnecessary work per convection step.
    tau_c = jnp.full((ncol,), config.tau_c, dtype=T.dtype)
    RH_ref = jnp.full((ncol,), config.RH_ref, dtype=T.dtype)
    CAPE_threshold = jnp.full((ncol,), config.CAPE_threshold, dtype=T.dtype)

    # 1. Surface temperature as parcel starting point
    T_base = T[:, -1]  # (ncol,)

    # 2. Compute moist adiabatic temperature profile
    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)

    # 3. Identify the convective layer: only levels where the moist adiabat
    #    is warmer than the environment (conditional instability).
    #    This prevents adjusting the stable stratosphere (Frierson 2007).
    cloud_mask = (T_moist >= T).astype(T.dtype)  # (ncol, nlev)

    # 4. Compute CAPE from the RAW moist adiabat (before enthalpy correction)
    #    to avoid artificial CAPE from the Newton correction.
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    # 5. Enthalpy-conserving correction (Newton iteration)
    #    Only over the cloud layer (masked levels).
    def _newton_step(T_trial):
        q_trial = RH_ref[:, None] * saturation_mixing_ratio(T_trial, p_full)
        residual = jnp.sum(
            cloud_mask * (constants.c_pd * (T_trial - T)
                          + constants.L_v * (q_trial - q_v)) * dp,
            axis=1,
        )  # (ncol,)
        q_sat_trial = saturation_mixing_ratio(T_trial, p_full)
        dqsat_dT = constants.L_v * q_sat_trial / (constants.R_v * T_trial ** 2)
        jacobian = jnp.sum(
            cloud_mask * (constants.c_pd
                          + constants.L_v * RH_ref[:, None] * dqsat_dT) * dp,
            axis=1,
        )  # (ncol,)
        dT = -residual / jnp.clip(jacobian, 1.0, None)
        return T_trial + dT[:, None]

    T_ref = _newton_step(T_moist)  # first iteration
    T_ref = _newton_step(T_ref)    # second iteration

    # Reference moisture at converged temperature
    q_ref = RH_ref[:, None] * saturation_mixing_ratio(T_ref, p_full)

    # 6. Smooth trigger: sigmoid(sharpness * (CAPE - threshold))
    trigger = jax.nn.sigmoid(
        config.smooth_trigger_sharpness * (cape - CAPE_threshold)
    )  # (ncol,)

    # 7. Relaxation tendencies — only within the convective (cloud) layer
    dT_dt = trigger[:, None] * cloud_mask * (T_ref - T) / tau_c[:, None]
    dq_v_dt = trigger[:, None] * cloud_mask * (q_ref - q_v) / tau_c[:, None]

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
