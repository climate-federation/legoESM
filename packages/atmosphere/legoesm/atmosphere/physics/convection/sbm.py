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
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_dT
from legoesm.atmosphere.physics.thermodynamics import (
    compute_moist_adiabat,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics._shared import safe_divide


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
    RH_ref = jnp.full((ncol,), config.rh_ref, dtype=T.dtype)
    CAPE_threshold = jnp.full((ncol,), config.cape_threshold, dtype=T.dtype)

    # 1. Surface temperature as parcel starting point
    T_base = T[:, -1]  # (ncol,)

    # 2. Compute moist adiabatic temperature profile
    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)

    # 3. Identify the convective layer: only levels where the moist adiabat
    #    is warmer than the environment (conditional instability).
    #    This prevents adjusting the stable stratosphere (Frierson 2007).
    #
    #    A pure ``(T_moist >= T).astype(...)`` boolean breaks
    #    differentiability (∂mask/∂T = 0 a.e.), but a pure sigmoid changes
    #    the FORWARD semantics — at the surface the moist adiabat is
    #    initialized from T[:,-1] so ``T_moist - T = 0`` gives mask = 0.5
    #    instead of the prior mask = 1.  Use a straight-through estimator:
    #    forward = hard step (preserve prior numerics exactly), backward =
    #    sigmoid' (keep gradients alive across layer membership).
    soft = jax.nn.sigmoid(config.cloud_mask_sharpness * (T_moist - T))
    hard = (T_moist >= T).astype(T.dtype)
    cloud_mask = soft + jax.lax.stop_gradient(hard - soft)  # (ncol, nlev)

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
        # Tetens-exact mixing-ratio derivative (matches the mixing-ratio residual
        # above + the emanuel.py convention) — not the CC-approximate inline form.
        dqsat_dT = saturation_mixing_ratio_dT(T_trial, p_full)
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

    # 7. Convective source for cloud water: vapor that condenses at each
    # level becomes cloud water rather than precipitating instantly.
    # Microphysics processes this through autoconversion, sedimentation,
    # and evaporation, and produces the surface precipitation diagnostic.
    #
    # Naive ``max(-dq_v_dt, 0)`` per level would *create* water
    # column-wide whenever the relaxation has both drying and
    # moistening layers (column-integrated dq_v + column-integrated
    # max(-dq_v, 0) = moistening_part > 0). To preserve column water
    # conservation we rescale the per-level condensation candidate so
    # its column integral equals the column-net drying — this matches
    # the legacy ``precipitation`` formula exactly. Per-level the
    # field is still non-negative (no negative q_c production); when
    # the column is net moistening (col_dq_v > 0) the scale is 0 and
    # dq_c_conv_dt = 0 everywhere, mirroring the legacy
    # ``clip(-col_dq_v, 0)`` behavior.
    local_cond = jnp.maximum(-dq_v_dt, 0.0)
    # Both column reductions share the ``* dp / g`` weight on the level
    # axis — stack the two integrands and reduce once.
    _col_pair = jnp.sum(
        jnp.stack([local_cond, dq_v_dt], axis=-1) * (dp / constants.g)[..., None],
        axis=-2,
    )
    col_local_cond = _col_pair[..., 0:1]
    col_net_drying = jnp.clip(-_col_pair[..., 1:2], 0.0, None)
    # AD-safe column rescaling: ``col_local_cond`` and ``col_net_drying``
    # vanish together when the column is barely triggered.  ``clip + divide``
    # is forward-safe but the divide's reverse-mode VJP still emits
    # ``-a/eps**2`` terms that overflow under ``jax.value_and_grad``
    # (issue #249).  ``safe_divide`` masks the bad branch *before* the
    # divide so neither cotangent path differentiates ``1/x²`` at tiny ``x``.
    dq_c_conv_dt = local_cond * safe_divide(
        col_net_drying, col_local_cond, eps=1e-20,
    )  # (ncol, nlev) [kg/kg/s]

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=trigger,
    )
