"""Entraining mass-flux convective plume parameterization."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "Entraining mass-flux convective plume: a surface-triggered plume "
        "descends while entraining ambient water, and its detrainment plus "
        "compensating subsidence redistribute heat and salt vertically (no "
        "momentum mixing)."
    ),
    "inputs": {
        "T": "degC", "S": "psu", "rho": "kg/m^3", "p_hydro": "Pa",
        "jacobian": "1 (z-star dimensionless)",
        "cfg.epsilon": "1/m (entrainment rate)", "cfg.T_excess": "K",
    },
    "outputs": {
        "dT_dt": "degC/s", "dS_dt": "psu/s", "convection_flag": "1 (active)",
    },
    "sign_convention": (
        "The plume is active where the parcel is denser than ambient (buoyant "
        "descent under surface destabilisation); a vertical REDISTRIBUTION only "
        "— an explicit surface-cell correction subtracts the column integral so "
        "the dz-weighted column integral of T and S is conserved to machine "
        "precision (no surface/floor flux); no momentum tendency; dry columns "
        "masked to zero; z positive up."
    ),
    # Adiabatic vertical redistribution with an exact column-integral
    # correction: conserves column-integrated heat (energy) and salt.
    "conserves": ["energy", "salt"],
    "differentiable": True,
    "reference": (
        "Entraining mass-flux plume convection; Paluszkiewicz & Romea (1997), "
        "Dyn. Atmos. Oceans 26, 95-130"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_plume_convection.py — a surface-cooled unstable "
        "column convects and the dz-weighted column integral of T, S is "
        "conserved to machine precision; a stable column gives zero tendency."
    ),
}


def plume_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    p_hydro: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: PlumeConfig,
    eos_fn: Callable | None = None,
) -> OceanConvectionOutput:
    """Apply entraining mass-flux plume convection.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
        Ambient in-situ density — MUST be computed with the same EOS as
        ``eos_fn`` so the parcel/ambient buoyancy comparison is consistent.
    p_hydro : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : PlumeConfig
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> rho`` for the plume-parcel density. If None,
        uses ``wright_eos`` (#518: every other physics module threads
        ``eos_fn``; do not hardcode an EOS that can disagree with the
        ambient ``rho`` passed in).

    Returns
    -------
    OceanConvectionOutput
    """
    if eos_fn is None:
        eos_fn = wright_eos
    nlev = T.shape[-1]
    # Carry/output dtype: promote across the state arrays *and* the config
    # params that enter the scan carry + tendencies.  This (a) keeps the
    # lax.scan carry dtype stable so it never promotes mid-scan, and (b) under
    # ``jax.grad`` w.r.t. a float64 param on a float32 state lifts the whole
    # computation to float64 so no float64→float32 downcast scatter occurs in
    # the conservation correction or the reverse pass.  For the usual
    # uniform-precision state with static (weakly-typed) python-float params
    # this is just ``T.dtype``.  (Differentiability audit 2026-06 + codex.)
    dtype = jnp.result_type(
        T, S, cfg.epsilon, cfg.T_excess, cfg.active_sigmoid_sharpness,
        cfg.w_plume_min, cfg.alpha_plume,
    )
    # Cast to the carry/output dtype: ``z_coord.dz_ref`` follows the precision
    # *control* policy and can be float64 while the state is float32, which
    # would otherwise reintroduce a mixed-dtype scatter at the k=0
    # column-integral correction below (codex review).
    dz_actual = (z_coord.dz_ref * jacobian[..., jnp.newaxis]).astype(dtype)

    # Detect unstable surface at a CONSISTENT pressure.  Comparing the raw
    # in-situ densities ``rho[...,0]`` (at p_hydro[...,0]) vs ``rho[...,1]``
    # (at p_hydro[...,1]) mixes two different reference pressures: seawater
    # compressibility makes the deeper level spuriously denser, so the trigger
    # UNDER-fires (a truly unstable surface can read as stable).  Displace the
    # surface parcel adiabatically to level-1 pressure via ``eos_fn`` so the
    # comparison is at the SAME pressure as the environment there — mirroring
    # the in-plume check ``eos_fn(T_plume, S_plume, p_hydro[k])`` below.
    # Sign convention: parcel DENSER than the environment (delta_rho > 0)
    # => statically unstable => plume active (denser water sinks).  ``eos_fn``
    # is always defined here (defaults to ``wright_eos`` above), so no
    # raw-density fallback path is reachable.
    rho_surf_at_1 = eos_fn(T[..., 0], S[..., 0], p_hydro[..., 1])
    surface_unstable = rho_surf_at_1 > rho[..., 1]  # (6, n, n)

    # Initialize plume properties at surface.  This is a DOWNWARD (sinking)
    # ocean convective plume, so the source parcel must be *denser* than the
    # surface water that feeds it — i.e. COLDER by ``cfg.T_excess`` (a
    # destabilizing magnitude), not warmer.  A positive (warm) perturbation
    # would make the parcel lighter and oppose sinking.  Cast to the state
    # dtype so a float64-traced ``cfg.T_excess`` (under ``jax.grad``) cannot
    # promote the scan-carry init relative to the in-loop carry (see scan_fn
    # dtype note).
    T_plume_init = (T[..., 0] - cfg.T_excess).astype(dtype)
    S_plume_init = S[..., 0].astype(dtype)

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
        # Pin the scan-carry dtype to the state dtype.  ``cfg.epsilon`` /
        # ``cfg.active_sigmoid_sharpness`` feed the carry (T_plume, S_plume,
        # active); under ``jax.grad`` w.r.t. one of these the param is a
        # float64 tracer while the ocean state is float32, so without the
        # cast ``entrain`` promotes the carry to float64 mid-scan and
        # ``lax.scan`` rejects the input≠output carry dtype.  Casting keeps
        # the carry stable in ``dtype`` and the gradient still flows through
        # the cast.  (Differentiability audit 2026-06.)
        T_plume = ((1.0 - entrain) * T_plume + entrain * T[..., k]).astype(dtype)
        S_plume = ((1.0 - entrain) * S_plume + entrain * S[..., k]).astype(dtype)

        # Buoyancy check
        rho_plume = eos_fn(T_plume, S_plume, p_hydro[..., k])
        delta_rho = rho_plume - rho[..., k]

        # Plume is active where it's denser than environment (sinking):
        # delta_rho > 0 means rho_plume > rho_env → plume sinks → stay active
        active = (
            active * jax.nn.sigmoid(delta_rho * cfg.active_sigmoid_sharpness)
        ).astype(dtype)

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
        # Cast the emitted tendencies to the state dtype as well: a
        # float64-traced ``epsilon`` / ``w_plume_min`` / ``alpha_plume``
        # (under ``jax.grad`` on a float32 state) would otherwise leave
        # ``dT_k`` float64 and trigger an unsafe float64→float32 scatter at
        # the k=0 conservation correction below.  Keeps the output dtype
        # equal to the state dtype; the gradient still flows through the cast.
        dT_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
                * (T_plume - T[..., k]) * active).astype(dtype)
        dS_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
                * (S_plume - S[..., k]) * active).astype(dtype)

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
