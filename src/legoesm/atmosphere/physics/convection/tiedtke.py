"""Tiedtke (1989) bulk mass-flux convection.

The IFS heritage scheme — three-class soft assignment of cloud type
(deep / mid-level / shallow) blended on cloud depth, downdraft with
RH-dependent trigger, and convective momentum transport via Gregory
et al. 1997.  This is the **first scheme that exercises the
``conv_prog_profile = M_u(k)`` profile carry**: the diagnosed
per-level updraft mass flux relaxes via implicit Euler toward the
new diagnosis on every step, smoothing fast oscillations.

Smooth-everywhere implementation:

* Three-class blend on cloud depth — sigmoid weights, not hard
  threshold.
* Downdraft RH trigger — sigmoid on (downdraft_RH_min - column_RH).
* CAPE gate — sigmoid via :func:`._triggers.cape_trigger`.
* Moisture-convergence proxy — saturation-deficit
  ``MC_proxy = max(q_sat - q_v, 0) / tau_MC_proxy`` (a placeholder
  until the PR-0 ``compute_moisture_convergence`` diagnostic ships).
* Plume integrator's mass-flux profile is gated by buoyancy sigmoid.

References
----------
* Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models.  *Mon. Wea. Rev.*, 117,
  1779–1800.
* Gregory, D., et al. (1997). Parametrization of momentum transport
  by convection. II.  *Quart. J. Roy. Meteor. Soc.*, 123, 1153–1183.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
)
from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.mass_flux import (
    _apply_mass_flux_kernel,
    _compute_column_geometry,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
    smooth_positive_part,
    smooth_step,
)
from legoesm.atmosphere.physics.convection._plume import (
    cmt_gregory_1997,
    compute_lcl,
    compute_lfc_lnb,
    entraining_detraining_plume,
)


__all__ = ("tiedtke_convection",)


def tiedtke_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    u: jax.Array,
    v: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: TiedtkeConfig = TiedtkeConfig(),
) -> tuple[ConvectionOutput, jax.Array]:
    """Tiedtke (1989) convection (smooth, differentiable).

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Environmental temperature [K].
    q_v : jax.Array, shape (ncol, nlev)
        Water-vapor specific humidity [kg/kg].
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa].
    u, v : jax.Array, shape (ncol, nlev)
        Environmental wind components [m/s] for the CMT closure.
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Updraft mass-flux profile from the previous time step.
        Tiedtke is the first scheme that uses this as a real per-level
        carry — relaxation is implicit Euler toward
        ``M_u_diagnosed`` over ``tau_M_u_relax`` seconds.
    dt : float
        Time step [s].
    config : TiedtkeConfig
        Scheme tunables.

    Returns
    -------
    out : ConvectionOutput
        Tendencies on T, q_v, q_c plus CAPE diagnostic.  CMT
        ``du_dt_conv`` / ``dv_dt_conv`` populated when
        ``config.enable_cmt`` is ``True``.
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
        Implicit-Euler-relaxed ``M_u(k)`` profile.
    """
    ncol, nlev = T.shape

    # -- Column geometry, moist adiabat, CAPE ------------------------------
    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)

    # -- LCL, LFC, LNB diagnostics -----------------------------------------
    T_parcel = T_base + config.parcel_dT
    q_parcel = q_base + config.parcel_dq
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)

    # Cloud depth: smooth interpolation of z at fractional indices.
    levels = jnp.arange(nlev, dtype=T.dtype)
    weight_lcl = jax.nn.softmax(
        -2.0 * (levels[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
    )
    weight_lnb = jax.nn.softmax(
        -2.0 * (levels[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
    )
    z_lcl = jnp.sum(weight_lcl * z, axis=-1)
    z_lnb = jnp.sum(weight_lnb * z, axis=-1)
    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)

    # -- Three-class soft assignment ---------------------------------------
    # deep_weight rises with cloud depth; shallow_weight falls with it.
    # mid-level fills the gap.
    deep_weight = smooth_step(
        cloud_depth - config.cloud_depth_deep, config.depth_split_sharpness,
    )
    shallow_weight = smooth_step(
        config.cloud_depth_shallow_max - cloud_depth,
        config.depth_split_sharpness,
    )
    midlevel_weight = jnp.clip(
        1.0 - deep_weight - shallow_weight, 0.0, 1.0
    )

    # -- Per-class entrainment / detrainment profiles ----------------------
    eps_per_class = (
        deep_weight[:, None] * config.epsilon_deep
        + shallow_weight[:, None] * config.epsilon_shallow
        + midlevel_weight[:, None] * config.epsilon_midlevel
    )
    dlt_per_class = (
        deep_weight[:, None] * config.delta_deep
        + shallow_weight[:, None] * config.delta_shallow
        + midlevel_weight[:, None] * config.delta_midlevel
    )
    # Broadcast to (ncol, nlev) — entrainment is constant over the
    # column for a given class blend.
    eps_profile = jnp.broadcast_to(eps_per_class, T.shape)
    dlt_profile = jnp.broadcast_to(dlt_per_class, T.shape)

    # -- Closure: deep uses a moisture-convergence proxy; shallow and
    # midlevel use a CAPE-relaxation closure.  Combined per-column
    # closure is a class-weighted blend.
    q_sat_env = saturation_mixing_ratio(T, p_full)
    sat_deficit = jnp.maximum(q_sat_env - q_v, 0.0)
    # Column-mean MC proxy.
    dp = p_half[:, 1:] - p_half[:, :-1]
    column_MC_proxy = (
        jnp.sum(sat_deficit * dp, axis=-1)
        / (constants.g * config.tau_MC_proxy)
    )
    # Smooth gate on MC threshold for deep.
    mc_gate = smooth_positive_part(
        column_MC_proxy - config.moisture_convergence_threshold,
        config.moisture_convergence_sharpness,
    )

    # Cloud-base mass flux (per class, then blended).
    M_b_deep = mc_gate * cape_weight
    M_b_shallow = (
        cape_weight
        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
        / 3600.0
    )
    M_b_midlevel = M_b_shallow * 0.5
    M_b = (
        deep_weight * M_b_deep
        + shallow_weight * M_b_shallow
        + midlevel_weight * M_b_midlevel
    )
    M_b = jnp.maximum(M_b, 0.0)

    # -- Plume integration -------------------------------------------------
    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_lcl_smooth,
        eps_profile, dlt_profile, M_b,
    )

    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, dt)
    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
    M_u_new = jnp.maximum(M_u_new, 0.0)

    # Use the relaxed M_u for the actual environmental tendencies — this
    # smooths the time evolution of the convective forcing.
    M_u_for_kernel = M_u_new

    # -- Environmental tendencies ----------------------------------------
    # Build an effective per-class delta_0 for the kernel and the
    # cloud-water source.
    delta_0_eff = (
        deep_weight * config.delta_deep
        + shallow_weight * config.delta_shallow
        + midlevel_weight * config.delta_midlevel
    )
    dT_dt_raw, dq_v_dt_raw, _ = _apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, M_u_for_kernel,
        z, rho, float(config.delta_deep),  # nominal delta — overridden below
    )
    rho_safe = jnp.clip(rho, 0.01, None)
    dq_c_conv_dt_raw = (
        delta_0_eff[:, None] * M_u_for_kernel * plume.q_c_u / rho_safe
    )
    # The kernel's dT/dq computations used ``config.delta_deep`` as the
    # detrainment scale; rescale by the per-column class blend.
    rescale = delta_0_eff[:, None] / config.delta_deep
    dT_dt = dT_dt_raw * rescale
    dq_v_dt = dq_v_dt_raw * rescale
    dq_c_conv_dt = dq_c_conv_dt_raw

    # -- Optional downdraft (RH-dependent trigger) -------------------------
    if config.enable_downdraft:
        # Column-mean RH below LCL.
        levels = jnp.arange(nlev, dtype=T.dtype)
        below_lcl = jax.nn.sigmoid(
            2.0 * (levels[None, :] - k_lcl_smooth[:, None])
        )
        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
        below_mass = jnp.sum(below_lcl * dp, axis=-1) + 1e-6
        rh_below = (
            jnp.sum(below_lcl * rh_layer * dp, axis=-1) / below_mass
        )
        downdraft_trigger = jax.nn.sigmoid(
            10.0 * (config.downdraft_RH_min - rh_below)
        )
        # Downdraft mass flux = -alpha * M_b at cloud base.
        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
        # Distribute uniformly below cloud base.
        below_lcl_norm = below_lcl / jnp.sum(below_lcl, axis=-1, keepdims=True).clip(1e-6, None)
        # Downdraft cools by entraining colder layers above and
        # bringing them down — net cooling tendency on environment.
        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
            jnp.abs(M_d_base[:, None]) * below_lcl_norm
            * jnp.maximum(0.05, 0.0)  # rough evap rate proxy [kg/kg]
        ) / rho_safe
        dT_dt = dT_dt + dT_dt_dd

    # -- CMT --------------------------------------------------------------
    if config.enable_cmt:
        if config.enable_downdraft:
            M_d = -config.downdraft_alpha * M_u_for_kernel * 0.3
        else:
            M_d = None
        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
            u, v, M_u_for_kernel, M_d,
            p_full, p_half, rho,
            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
        )
    else:
        du_dt_conv = None
        dv_dt_conv = None

    # -- Convective mask ---------------------------------------------------
    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
        cape=cape,
        convective_mask=convective_mask,
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
    )
    return out, M_u_new
