"""Simplified Betts-Miller (SBM) convection scheme.

A relaxation-based convection parameterization for idealized aquaplanet
experiments. Convective columns are relaxed toward a moist adiabatic
temperature profile with an approximately enthalpy-conserving (2-iteration
Newton, ~1e-3 residual) correction.

Algorithm (in code order):
1. Compute moist adiabat from surface temperature upward
2. Compute CAPE from the RAW moist adiabat (before the Newton correction, to avoid
   artificial CAPE from the enthalpy adjustment)
3. Newton-correct the reference temperature T_ref from the moist adiabat
   (approximately enthalpy-conserving, 2-iteration, ~1e-3 closure)
4. Reference moisture q_ref = RH_ref * q_sat(T_ref, p) at the Newton T_ref, then
   the Frierson shallow redistribution (net-moistening columns only: subtract the
   cloud-mean moistening deficit from q_ref, add the latent-equivalent to T_ref)
5. Relax T and q_v toward the (T_ref, q_ref) references over tau_c, scaled by the
   smooth CAPE trigger
6. Condensation source + P>=0 drying gate (precip deferred to microphysics)

Differentiability: the scheme is JAX-AD-compatible (jax.grad-safe), NOT globally
smooth.  The forward path deliberately contains a HARD membership comparison
(T_moist >= T), ``maximum``/``clip`` kinks, and a HARD P>=0 drying gate.  Gradients
flow via a straight-through estimator for the cloud mask (forward = hard step,
backward = sigmoid') plus a.e. subgradients at the clamps/gate.  Only the CAPE
trigger is a genuinely smooth sigmoid.

Faithfulness
------------
The SBM ASSEMBLY (the 2-iteration Newton enthalpy correction, q_ref = RH_ref*q_sat,
the relaxation dT_dt = trigger*mask*(T_ref-T)/tau_c, the condensation rescale
[column-water-conserving to the 1e-20 floor], and the P>=0 drying gate) is pinned to
round-off (rel 1e-12)
against an independent numpy reimplementation in ``tests/unit/test_sbm_faithful.py``,
which REUSES the shared, separately-tested thermodynamics (``compute_moist_adiabat``,
``compute_cape``, ``cape_trigger``, ``saturation_mixing_ratio``) as givens.  Truth-
tiers pinned there: total-water conservation ``Sum(dq_v_dt + dq_c_conv_dt)*dp/g = 0``
EXACTLY in a net-drying column whose column condensation candidate exceeds 1e-20 (every
physically active column; a <=1e-20 residual survives only in the degenerate near-zero-
condensation corner where the ``safe_divide`` AD-guard floors), and the (Newton-limited,
~1e-3) enthalpy closure.
NOTE — net-moistening handling, A/B RESOLVED (2026-07-17; the owed full-module A/B ran:
shallow branch toggled off, forward output + gradients diffed across a drying /
moistening / mixed / rh-sweep column battery).  VERDICT: the Frierson SHALLOW branch is
LIVE — KEEP.  It is NOT superseded by the ``drying_gate`` (issue #771); the two compose:

  * MODERATELY net-moistening columns (uniform rh ~ 0.5-0.6 soundings in the battery):
    the shallow redistribution zeros the column integral, the gate then passes the
    REDISTRIBUTED local tendencies (|dT_dt| ~ 1e-3 K/s, column water residual ~ 1e-20).
    With the branch removed the raw integral > 0 keeps the gate SHUT and the column is
    silently zeroed — convection off where the scheme should redistribute.  This live
    regime is pinned in test_sbm_faithful.py (deleting the branch turns it red).
  * STRONGLY net-moistening columns (rh ~ 0.2): forward output is 0 with or without the
    branch (the gate dominates) BUT the GRADIENTS differ (the shallow shift shapes the
    backward path through the zeroed output) — removal is not even AD-neutral there.
  * Net-drying / mixed columns: byte-identical with the branch removed (the shift is
    exactly 0), as designed.

The gate's job stays what the pins say: zeroing the residual LOCAL tendencies of columns
whose integral the shallow branch could not fully cancel.  Neither mechanism subsumes
the other; do not remove either.

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
from legoesm.atmosphere.physics.convection._triggers import cape_trigger
from legoesm.atmosphere.physics._shared import safe_divide


__physics_contract__ = {
    "summary": (
        "Simplified Betts-Miller convective adjustment: relax T and q_v toward "
        "a Newton- and shallow-adjusted moist-adiabatic reference (T_ref, q_ref; "
        "q_ref = RH_ref*q_sat at the Newton T_ref, less the shallow shift) over "
        "tau_c, with an "
        "approximately enthalpy-conserving 2-iteration Newton correction and a "
        "condensation source (column-water-conserving to the 1e-20 floor). "
        "JAX-AD-compatible "
        "(straight-through cloud mask + a.e. subgradients; NOT globally smooth — "
        "hard mask, maximum/clip kinks, hard P>=0 gate)."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (condensation -> cloud-water source to microphysics, >=0)",
        "cape": "J/kg", "convective_mask": "1 (0-1 CAPE trigger)",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Where CAPE>threshold the column relaxes "
        "toward the (warmer, moist-adiabatic) reference: "
        "dT_dt = trigger*mask*(T_ref-T)/tau_c and "
        "dq_v_dt = trigger*mask*(q_ref-q_v)/tau_c. The 2-iteration Newton "
        "correction drives column-integrated (c_pd*dT + L_v*dq_v) toward 0 "
        "(approximate enthalpy/MSE closure, ~1e-3 relative residual — NOT exact; "
        "a 3rd Newton step would reach ~1e-7); the condensation source "
        "dq_c_conv_dt >= 0 is rescaled so its column integral equals the column "
        "net drying (total water conserved EXACTLY when the column condensation "
        "candidate > 1e-20; a <=1e-20 residual survives only in the degenerate "
        "near-zero-condensation corner where the safe_divide AD-guard floors), "
        "deferred to microphysics for precip."
    ),
    # The ``moisture`` token = total water, conserved EXACTLY for every physically
    # active drying column (condensation rescale, truth-tier); a bounded <=1e-20
    # safe_divide AD-guard residual survives ONLY in the degenerate near-zero-
    # condensation corner (the token carries this documented caveat, matching the
    # Kain-Fritsch contract, which likewise lists the token with a degenerate-corner
    # residual note).  Energy is only APPROXIMATELY conserved (2-iteration Newton,
    # ~1e-3 residual — NOT machine precision), so it is documented in prose above but
    # NOT listed as a machine-readable conserved quantity here.
    "conserves": ["moisture"],  # exact to the 1e-20 floor (see caveat above)
    "differentiable": True,
    "reference": (
        "Frierson (2007), J. Atmos. Sci. 64, 1959-1976; "
        "Betts & Miller (1986), Q. J. R. Meteorol. Soc. 112, 693-709"
    ),
    "idealized_test": (
        "tests/unit/test_physics_convection.py; CAPE sufficiently below threshold "
        "-> negligible tendency (smooth sigmoid trigger: 0.5 at equality, strictly "
        "positive but small below); a conditionally-unstable column relaxes T and "
        "q_v toward the Newton-adjusted, shallow-corrected references (T_ref = moist "
        "adiabat + 2-iteration enthalpy correction + shallow shift; q_ref = "
        "RH_ref*q_sat(T_ref_newton) - shift, evaluated at the pre-shift Newton T_ref), "
        "NOT the raw moist adiabat, with column-integrated "
        "(c_pd*dT + L_v*dq_v) ~ 0 (approximate enthalpy, Newton-limited ~1e-3) "
        "and the dq_c column integral = the column net drying (total water, exact "
        "up to the 1e-20 safe_divide AD-guard floor)."
    ),
}


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

    # 5. Approximately enthalpy-conserving correction (2-iteration Newton, ~1e-3)
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

    # 5b. Frierson (2007) SHALLOW branch — conserve column water in the
    #     net-moistening regime.  The deep references can net-MOISTEN the cloud
    #     layer (cloud-layer mass-mean ``q_ref − q_v`` > 0); applied directly
    #     this CREATES water out of nothing, because the cloud-water source
    #     below clips ``col_net_drying`` to 0 and so cannot absorb it (the
    #     column then has Σ(dq_v + dq_c) > 0).  Frierson's shallow branch sets
    #     precipitation to zero and conserves column-integrated moisture (and,
    #     via the approximately enthalpy-conserving refs, column enthalpy) by a pure
    #     redistribution.  We realise this by subtracting the cloud-layer
    #     mass-weighted-mean MOISTENING deficit from ``q_ref`` (so the
    #     column-integrated dq_v vanishes in the shallow regime) and adding the
    #     matching latent-heat-equivalent shift to ``T_ref`` (so the deep
    #     references' net cooling is removed and ∫c_p dT ≈ −L_v ∫dq ≈ 0).
    #     ``jnp.maximum(·, 0)`` leaves the deep (net-drying) branch BYTE-
    #     IDENTICAL (the shift is exactly 0 there) and uses the same a.e.-
    #     differentiable clamp already employed for the cloud-water source.
    w_cloud = cloud_mask * dp                                   # (ncol, nlev)
    W_cloud = jnp.clip(jnp.sum(w_cloud, axis=1, keepdims=True), 1e-30, None)
    dq_def = jnp.sum(w_cloud * (q_ref - q_v), axis=1, keepdims=True) / W_cloud
    shallow_dq = jnp.maximum(dq_def, 0.0)                       # net moistening to remove (>=0)
    q_ref = q_ref - shallow_dq
    T_ref = T_ref + (constants.L_v / constants.c_pd) * shallow_dq

    # 6. Smooth trigger: cape_trigger == sigmoid(sharpness * (CAPE - threshold))
    trigger = cape_trigger(
        cape, CAPE_threshold, config.smooth_trigger_sharpness
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
    # Total-water bookkeeping: ``∫dq_c·dp/g = col_net_drying`` EXACTLY whenever
    # ``col_local_cond > eps`` (safe_divide returns the true ratio and
    # ``∫local_cond·dp/g = col_local_cond``).  Because ``col_net_drying <=
    # col_local_cond`` always, this holds for every physically active drying
    # column; ONLY in the degenerate corner where the WHOLE column's condensation
    # candidate ``col_local_cond <= eps = 1e-20`` does the floor return 0, leaving a
    # ``<= 1e-20`` water residual (negligible; the alternative 1/x² VJP overflow is
    # worse).
    dq_c_conv_dt = local_cond * safe_divide(
        col_net_drying, col_local_cond, eps=1e-20,
    )  # (ncol, nlev) [kg/kg/s]

    # Column-water conservation to the 1e-20 safe_divide floor (issue #771).  The
    # relaxation ``(q_ref - q_v)``
    # can NET-MOISTEN a column that is net-subsaturated relative to
    # ``q_ref = RH_ref*q_sat`` (column integral ``∫dq_v > 0``), injecting column
    # water with no source: a single-column adjustment has no moisture supply for
    # net moistening (that water would have to be imported by transport).  The
    # condensate rescale above already zeros precip in those columns
    # (``col_net_drying = 0``), but the vapour + heat tendencies would still
    # moisten.  Gate the WHOLE adjustment off there so convection only ever dries
    # or stays neutral (never creates water); this is the ``P >= 0`` constraint of
    # a Betts-Miller adjustment.  Net-DRYING columns are byte-identical
    # (``gate = 1``) and preserve whatever enthalpy residual the Newton step left:
    # ``dT_dt`` and ``dq_v_dt`` carry the SAME per-column gate, so the gate does not
    # change the Newton-limited (approximate, ~1e-3) ``∫(c_pd·dT + L_v·dq_v) ≈ 0``
    # closure — it neither tightens nor loosens it.  ``col_net_drying > 0`` iff the
    # column is net-drying (``col_net_drying = clip(-∫dq_v, 0)``).
    drying_gate = (col_net_drying > 0.0).astype(dq_v_dt.dtype)  # (ncol, 1)
    dT_dt = dT_dt * drying_gate
    dq_v_dt = dq_v_dt * drying_gate
    dq_c_conv_dt = dq_c_conv_dt * drying_gate  # already ~0 in moistening columns

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=trigger,
    )
