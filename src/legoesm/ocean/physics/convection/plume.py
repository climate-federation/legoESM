"""Entraining mass-flux convective plume parameterization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def plume_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    p_hydro: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: PlumeConfig,
) -> OceanConvectionOutput:
    """Apply entraining mass-flux plume convection.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    p_hydro : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : PlumeConfig

    Returns
    -------
    OceanConvectionOutput
    """
    nlev = T.shape[-1]
    shape_3d = T.shape
    dtype = T.dtype
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Detect unstable surface: rho(k=0) > rho(k=1)
    surface_unstable = rho[..., 0] > rho[..., 1]  # (6, n, n)

    # Initialize plume properties at surface
    T_plume_init = T[..., 0] + cfg.T_excess
    S_plume_init = S[..., 0]

    # Descend plume using scan over levels (starting from level 1)
    def scan_fn(carry, k):
        T_plume, S_plume, active = carry
        dz_k = dz_actual[..., k]

        # Entrain environment.  ``1 - exp(-epsilon*dz)`` is the exact
        # solution of dT_plume/dz = -epsilon*(T_plume - T_env) over a
        # layer of thickness ``dz``.  The first-order linearization
        # ``epsilon*dz`` exceeds 1 and goes negative for thick layers
        # (e.g. epsilon=1e-3 m^-1, dz>1000 m), which would produce an
        # unphysical sign-flip on the plume properties.  ``-expm1(-x)``
        # is monotone in [0, 1) for x>=0 and gradient-friendly.
        entrain = -jnp.expm1(-cfg.epsilon * dz_k)
        T_plume = (1.0 - entrain) * T_plume + entrain * T[..., k]
        S_plume = (1.0 - entrain) * S_plume + entrain * S[..., k]

        # Buoyancy check
        rho_plume = wright_eos(T_plume, S_plume, p_hydro[..., k])
        delta_rho = rho_plume - rho[..., k]

        # Plume is active where it's denser than environment (sinking):
        # delta_rho > 0 means rho_plume > rho_env → plume sinks → stay active
        active = active * jax.nn.sigmoid(delta_rho * 1e4)

        # Detrainment tendency at this level [K/s], [PSU/s].
        #
        # The mass-flux plume formulation gives a tendency of the form
        #   dT/dt_env = w_p * alpha_plume * epsilon * (T_plume - T_env)
        # where ``w_p [m/s]`` is the plume vertical velocity, ``epsilon
        # [1/m]`` is the entrainment rate, and ``alpha_plume`` is a
        # dimensionless detrainment efficiency.  Without the ``w_p``
        # factor, the units would be [K/m] instead of [K/s] (codex
        # adversarial review iter-1, finding #2).  ``cfg.w_plume_min``
        # is used as the constant plume velocity (the minimum-floor
        # interpretation of an unresolved plume's effective speed).
        dT_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
                * (T_plume - T[..., k]) * active)
        dS_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
                * (S_plume - S[..., k]) * active)

        return (T_plume, S_plume, active), (dT_k, dS_k, active)

    init_active = surface_unstable.astype(dtype)
    (_, _, _), (dT_levels, dS_levels, active_levels) = jax.lax.scan(
        scan_fn,
        (T_plume_init, S_plume_init, init_active),
        jnp.arange(1, nlev),
    )

    # dT_levels shape: (nlev-1, 6, n, n) — move level axis to last,
    # then ``jnp.pad`` along the trailing axis instead of
    # ``zeros + .at[..., 1:].set(...)`` which materialises a fresh
    # zero buffer + scatter.  Single Pad HLO op each.
    dT_levels_t = jnp.moveaxis(dT_levels, 0, -1)  # (6, n, n, nlev-1)
    dS_levels_t = jnp.moveaxis(dS_levels, 0, -1)
    pad_axes = ((0, 0),) * (dT_levels_t.ndim - 1)
    dT_dt = jnp.pad(dT_levels_t, (*pad_axes, (1, 0)))
    dS_dt = jnp.pad(dS_levels_t, (*pad_axes, (1, 0)))

    # Column-integral conservation: the plume sources heat/salt from the
    # surface mixed layer (the layer that "feeds" the plume at k=0).
    # Detraining heat/salt to k>=1 without a compensating surface sink
    # leaves ``Σ_k dT_dt[k] · dz[k]`` non-zero, which is a closed-column
    # conservation violation (codex adversarial-review finding #3).
    # Subtract the column integral from level 0 so heat/salt are
    # exactly conserved per closed column to machine precision.  Mass
    # is unchanged (this is a redistribution; no flux through the
    # surface or floor).
    #
    # NaN guard: dry / land columns have ``jacobian = 0`` → ``dz_top =
    # 0``.  The plume's ``active`` mask is also zero there, so
    # ``column_dT`` and ``column_dS`` are zero and the correction
    # *should* be zero — but ``0 / 0`` is NaN.  Use ``jnp.where`` to
    # zero the correction explicitly when ``dz_top == 0`` and feed a
    # safe denominator into the division so neither branch produces
    # NaN gradients (codex stop-time review).
    dz_top = dz_actual[..., 0]
    column_dT = jnp.sum(dT_dt * dz_actual, axis=-1)
    column_dS = jnp.sum(dS_dt * dz_actual, axis=-1)
    wet = dz_top > 0
    dz_top_safe = jnp.where(wet, dz_top, jnp.ones_like(dz_top))
    correction_T = jnp.where(wet, -column_dT / dz_top_safe, jnp.zeros_like(column_dT))
    correction_S = jnp.where(wet, -column_dS / dz_top_safe, jnp.zeros_like(column_dS))
    dT_dt = dT_dt.at[..., 0].add(correction_T)
    dS_dt = dS_dt.at[..., 0].add(correction_S)

    # Convection flag at interfaces (average of adjacent levels' activity)
    active_t = jnp.moveaxis(active_levels, 0, -1)  # (6, n, n, nlev-1)
    flag = active_t

    # Final dry-column mask.  The scan operates on ``T``, ``S``, ``rho``
    # without reference to ``dz_actual``, so dry columns (``jacobian=0``
    # → ``dz_top=0``) still produce non-zero detrainment tendencies on
    # ``k ≥ 1``.  Multiply the entire output by the wet mask so the
    # plume contributes nothing on land.  The scalar conservation
    # correction at ``k=0`` is already zero for dry columns, so this
    # final masking is consistent with it (codex stop-time review:
    # "dry-column fix is incomplete").
    wet_mask = wet[..., jnp.newaxis].astype(dtype)
    dT_dt = dT_dt * wet_mask
    dS_dt = dS_dt * wet_mask
    flag = flag * wet_mask

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
    )
