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
from legoesm.atmosphere.physics._shared import safe_divide


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
    dT_dt = (
        trigger[:, None] * mc_gate[:, None]
        * config.alpha_heat
        * (T_moist - T)
        / config.tau_relax_s
    )  # (ncol, nlev)

    # 6. Moistening tendency (budget-consistent with heating)
    #
    # The column moisture excess MC is the scheme's internal source.
    # Fraction ``alpha_heat`` becomes condensational heating — this is
    # the cloud-water source rate handed to microphysics through
    # ``dq_c_conv_dt`` (see step 7). Fraction ``(1 - alpha_heat)``
    # appears as a column-distributed vapor source representing
    # external moistening implicit in Kuo's design (surface evaporation
    # / large-scale moisture convergence).
    #
    # We distribute the moistening budget proportional to the local
    # subsaturation deficit, then normalize so the column integral
    # exactly equals (1 - alpha_heat) * MC / tau_relax_s.

    # Implied condensation rate from heating (moisture sink, kg/kg/s).
    # This is the per-level rate at which Kuo converts vapor to cloud
    # water by latent heat balance: c_pd * dT_dt = L_v * (-dq_v) for
    # the condensation contribution. Microphysics receives this
    # directly via ``dq_c_conv_dt`` so it can process the convective
    # condensate through its full chain (autoconversion, sedimentation,
    # evaporation) rather than the legacy assumption that all of it
    # falls instantly to the surface.
    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)

    # Subsaturation deficit profile for distributing moistening
    deficit = jnp.maximum(q_sat - q_v, 0.0)  # (ncol, nlev)
    deficit_integral = jnp.sum(deficit * dp, axis=1, keepdims=True) / constants.g  # (ncol, 1)
    deficit_integral_safe = jnp.maximum(deficit_integral, 1e-20)

    # Moistening budget: (1 - alpha_heat) * MC / tau_relax_s [kg/m^2/s]
    moistening_budget = (
        trigger * (1.0 - config.alpha_heat) * MC / config.tau_relax_s
    )  # (ncol,)

    # Distribute moistening proportional to deficit, normalized so the
    # column integral exactly equals the budget.  With ``f = deficit /
    # ∫(deficit·dp/g)`` (units m²/kg), ``dq_v_dt = budget · f`` has units
    # ``kg/(m²·s) × m²/kg = kg/(kg·s)`` ✓ and ``∫(dq_v_dt · dp/g) =
    # budget · ∫(f · dp/g) = budget · 1`` ✓.  An earlier form multiplied
    # by an extra ``g/dp`` factor, which gave units ``m²/(kg·s)`` and
    # under-reported the column moistening budget by a factor of
    # ``~Σ_k g²/(dp_k · g) = nlev`` — for nlev=20 the column integral
    # was ~500× smaller than the design intent (audit Codex finding:
    # 'Kuo moistening-budget distribution has an extra g/dp').
    dq_v_dt = (
        moistening_budget[:, None]
        * (deficit / deficit_integral_safe)
    )  # (ncol, nlev) [kg/kg/s]

    # Subtract condensation implied by heating
    dq_v_dt = dq_v_dt - implied_condensation

    # 7. Convective source for cloud water — column integral equals
    # Kuo's design-intent condensation rate ``trigger * alpha_heat *
    # MC / tau_relax_s`` (the gross condensation that microphysics
    # processes), distributed per-level by the implied-condensation
    # profile from latent heating.
    #
    # Critically the target rate is *MC-gated*: it is zero whenever
    # MC = 0, so the scheme does **not** create cloud water in
    # undersaturated columns. (A naive ``max(implied_condensation, 0)``
    # at every level would create cloud water in an undersaturated
    # column whenever the smooth sigmoid trigger had any nonzero
    # value — at the default ``smooth_trigger_sharpness`` and
    # ``me_threshold`` the trigger is ≈0.475 at MC=0, large enough
    # to yield a spurious ``dT_dt`` and therefore a spurious
    # condensation rate from a column with no moisture excess.)
    #
    # When MC > 0 the rescaling produces the same column total as the
    # design-intent formula and the same per-level shape as the
    # implied-condensation profile (no underreporting of the
    # moistening fraction).
    local_cond = jnp.maximum(implied_condensation, 0.0)
    col_local_cond = jnp.sum(local_cond * dp / constants.g, axis=-1, keepdims=True)
    target_col_cond = (
        trigger * config.alpha_heat * MC / config.tau_relax_s
    )[:, None]
    # AD-safe column rescaling — see sbm.py for derivation; issue #249.
    dq_c_conv_dt = local_cond * safe_divide(
        target_col_cond, col_local_cond, eps=1e-20,
    )  # (ncol, nlev) [kg/kg/s]

    # 8. CAPE diagnostic
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=trigger,
    )
