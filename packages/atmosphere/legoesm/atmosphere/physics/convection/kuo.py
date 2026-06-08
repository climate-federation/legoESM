"""Kuo column moisture-excess convection scheme.

A column moisture-excess convection scheme (Kuo 1965/1974). Convective
heating and moistening are proportional to the column-integrated moisture
excess above saturation, partitioned by alpha_heat.

Algorithm (Kuo 1965/1974 — column moisture-excess formulation):
1. Compute column moisture excess above saturation [kg/m^2]
2. Smooth sigmoid trigger based on moisture excess
3. Compute moist adiabat reference profile
4. Relax temperature and moisture toward reference profiles
5. Emit per-level cloud-water source (``dq_c_conv_dt``) from the
   implied condensation rate; microphysics processes it through
   autoconversion / sedimentation / evaporation and produces the
   surface precipitation diagnostic.

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

Water budget
------------
Kuo is a *non-conservative* scheme by design: a fraction
``(1 - alpha_heat)`` of the column moisture excess appears as
moistening that has no in-scheme sink — physically interpreted as
surface evaporation or large-scale moisture convergence implicit in
the parameterization. The column water residual is therefore
``(1 - alpha_heat) * MC / tau_relax_s`` per timestep (≈0.25 ⋅ MC/τ for
the default ``alpha_heat = 0.75``). This was already true under the
legacy ``ConvectionOutput.precipitation`` formulation; Option C
preserves it and exposes the full ``alpha_heat * MC / tau_relax_s``
condensation rate to microphysics rather than the column-net drying.

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
from legoesm.atmosphere.physics.convection.mass_flux import (
    stratosphere_mass_flux_gate,
)


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

    # 5. Heating tendency: relax toward moist adiabat, gated by MC.
    #
    # The smooth sigmoid trigger alone is not enough: at MC = 0 the
    # default ``(smooth_trigger_sharpness, me_threshold)`` give
    # ``trigger ≈ 0.475`` rather than zero, so a relaxation
    # ``(T_moist - T) / tau_relax_s`` not gated by MC would still
    # heat (and condense, and remove vapor) in undersaturated
    # columns — destroying water with no source.
    #
    # Kuo (1965/1974) actually prescribes column heating proportional
    # to ``alpha_heat * MC / tau_relax_s``; the legacy implementation
    # drifted from that design by using a pure relaxation rate. The
    # ``tanh(MC / me_threshold)`` factor restores the MC-proportional
    # scaling smoothly: zero at MC = 0, ≈1 once ``MC >> me_threshold``,
    # and differentiable everywhere. Combined with the existing
    # sigmoid trigger this guarantees ``dT_dt = 0``,
    # ``implied_condensation = 0``, ``dq_v_dt = 0``, and
    # ``dq_c_conv_dt = 0`` whenever MC = 0 — no spurious heating,
    # no destroyed vapor, no created cloud water.
    mc_gate = jnp.tanh(MC / jnp.clip(config.me_threshold, 1e-30, None))
    # Desired relaxation heating toward the moist adiabat (Kuo closure),
    # gated by the trigger and the MC-proportional factor.
    desired_heating = (
        trigger[:, None] * mc_gate[:, None]
        * config.alpha_heat
        * (T_moist - T)
        / config.tau_relax_s
    )  # (ncol, nlev) [K/s]

    # 6. External moistening source (the ``(1 - alpha_heat)`` Kuo branch)
    #
    # Fraction ``(1 - alpha_heat)`` of the column moisture excess appears
    # as a column-distributed vapor SOURCE (surface evaporation / large-
    # scale moisture convergence implicit in Kuo's design), distributed
    # proportional to the local subsaturation deficit and normalized so
    # the column integral equals ``(1 - alpha_heat) * MC / tau_relax_s``.
    # ``f = deficit / ∫(deficit·dp/g)`` has units m²/kg, so ``budget · f``
    # is kg/(kg·s) and ``∫(dq_v · dp/g) = budget``.
    deficit = jnp.maximum(q_sat - q_v, 0.0)  # (ncol, nlev)
    deficit_integral = jnp.sum(deficit * dp, axis=1, keepdims=True) / constants.g  # (ncol, 1)
    deficit_integral_safe = jnp.maximum(deficit_integral, 1e-20)
    moistening_budget = (
        trigger * (1.0 - config.alpha_heat) * MC / config.tau_relax_s
    )  # (ncol,)
    dq_v_moisten = (
        moistening_budget[:, None] * (deficit / deficit_integral_safe)
    )  # (ncol, nlev) [kg/kg/s], >= 0
    # Cap the per-level moistening so it cannot push a level past
    # saturation in one step.  When the column is nearly saturated
    # everywhere, ``deficit_integral`` collapses toward its 1e-20 floor
    # and the ``deficit / deficit_integral`` normalisation can spike the
    # one level that still has a sliver of deficit, injecting kilograms of
    # vapour in a single step — the slow runaway that drove Kuo to a
    # finite-but-growing instability and eventually NaN in long RCE.
    # Bounding the source at ``deficit/dt`` is the physically correct
    # resolution (the implicit moisture-convergence source fills
    # sub-saturated air; there is nowhere to put it once saturated) and
    # keeps ``q_v >= 0`` (the source is still non-negative).
    dq_v_moisten = jnp.minimum(dq_v_moisten, deficit / dt)

    # 7. Latent heating, condensation, and POSITIVITY.
    #
    # The relaxation heating is supplied by condensation: ``c_pd · dT =
    # L_v · cond``.  The per-level vapor sink ``cond`` is therefore
    # ``desired_heating · c_pd / L_v``.  Left unbounded (the previous
    # implementation), this sink could remove far more vapor than the
    # column held — in a cold RCE column where ``T_moist − T`` is tens of
    # K, the implied condensation drove ``q_v`` to −150 g/kg, poisoning
    # the whole profile.  We cap the *condensation* (positive sink) at
    # the per-step available vapor ``q_v/dt + moistening`` so a single
    # forward-Euler step keeps ``q_v >= 0`` exactly, then derive the
    # REALIZED heating from the capped condensation so latent heating and
    # the moisture sink stay budget-consistent (no energy injected
    # without a matching moisture sink).  Where the closure is *cooling*
    # (``T > T_moist``) the term is an evaporative vapor source and is
    # passed through unchanged — it cannot drive ``q_v`` negative.
    cond_signed = desired_heating * constants.c_pd / constants.L_v  # >0 sink, <0 source
    pos_cond = jnp.maximum(cond_signed, 0.0)
    neg_cond = jnp.minimum(cond_signed, 0.0)  # <= 0 (evaporative moistening)
    max_cond = jnp.clip(q_v / dt + dq_v_moisten, 0.0, None)  # available vapor / step
    pos_cond_lim = jnp.minimum(pos_cond, max_cond)
    realized_cond = pos_cond_lim + neg_cond  # signed condensation actually applied

    # Realized heating consistent with the realized condensation.
    dT_dt = realized_cond * constants.L_v / constants.c_pd  # (ncol, nlev) [K/s]
    # Net vapor tendency: external moistening minus realized condensation.
    # Guaranteed ``q_v + dt·dq_v_dt >= 0`` by the ``max_cond`` cap above.
    dq_v_dt = dq_v_moisten - realized_cond
    # Convective cloud-water source handed to microphysics — only the
    # positive (condensation) part makes cloud water; non-negative by
    # construction.
    dq_c_conv_dt = pos_cond_lim  # (ncol, nlev) [kg/kg/s], >= 0

    # Gate the tendencies out of the stratosphere.  The thin upper-model
    # layers (small Δp/g) amplify any residual heating into unphysical
    # spikes — without this the top layer overflowed to NaN in long
    # fixed-SST RCE — and convective heating/moistening is meaningless
    # above the tropopause.  ``stratosphere_mass_flux_gate`` is ≈1 in the
    # troposphere and →0 above ~100 hPa, matching the mass-flux schemes.
    # Multiplying by a factor in [0, 1] preserves the q_v >= 0 guarantee.
    strat_gate = stratosphere_mass_flux_gate(p_full)  # (ncol, nlev)
    dT_dt = dT_dt * strat_gate
    dq_v_dt = dq_v_dt * strat_gate
    dq_c_conv_dt = dq_c_conv_dt * strat_gate

    # 8. CAPE diagnostic
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=trigger,
    )
