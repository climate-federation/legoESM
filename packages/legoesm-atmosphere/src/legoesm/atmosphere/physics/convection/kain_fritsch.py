"""Kain & Fritsch (1990; 2004 update) deep & shallow convection.

Bulk mass-flux scheme distinguished by its boundary-layer trigger
function: convection fires when a smoothly-perturbed parcel
temperature at the LCL exceeds the environmental temperature there
by a sigmoid amount.  Cloud-depth-dependent blending of deep and
shallow branches.  No convective momentum transport — KF emits
``du_dt_conv = dv_dt_conv = None`` and the orchestrator zero-fills.

The scheme is **smooth-everywhere**:

* The trigger function uses
  ``trigger_weight = sigmoid(s * (T_LCL_perturbed - T_env_at_LCL))``
  in place of the original hard ``> 0`` switch.  This is the central
  AD-safety property of the smooth-everywhere KF: gradients flow
  through the trigger threshold so training-time perturbations to
  ``parcel_perturb_T``, ``w_thresh_offset``, and ``trigger_sharpness``
  all have non-zero gradient signal.
* The deep/shallow blend is a sigmoid on cloud depth.
* The CAPE gate is the same ``cape_trigger`` used by ZM.

References
----------
* Kain, J. S. & Fritsch, J. M. (1990). A one-dimensional entraining /
  detraining plume model and its application in convective
  parameterization.  *J. Atmos. Sci.*, 47, 2784–2802.
* Kain, J. S. (2004). The Kain–Fritsch convective parameterization:
  An update.  *J. Appl. Meteor.*, 43, 170–181.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
)

from legoesm.atmosphere.physics.convection.config import KainFritschConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.mass_flux import (
    _apply_mass_flux_kernel,
    _compute_column_geometry,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
    smooth_level_indicator,
    smooth_step,
)
from legoesm.atmosphere.physics.convection._plume import (
    compute_lcl,
    compute_lfc_lnb,
    entraining_detraining_plume,
)


__all__ = ("kain_fritsch_convection",)


def _interpolate_at_smooth_level(
    profile: jax.Array,
    k_smooth: jax.Array,
    sharpness: float = 2.0,
) -> jax.Array:
    """Smooth interpolation of ``profile`` at a fractional level index.

    Uses a soft level-membership weighting so that the result is
    differentiable in ``k_smooth``.  ``profile`` shape ``(ncol, nlev)``,
    ``k_smooth`` shape ``(ncol,)``; returns shape ``(ncol,)``.
    """
    nlev = profile.shape[-1]
    levels = jnp.arange(nlev, dtype=profile.dtype)
    # Centered Gaussian-like weight peaked at k_smooth.
    weight = jax.nn.softmax(
        -sharpness * (levels[None, :] - k_smooth[:, None]) ** 2,
        axis=-1,
    )
    return jnp.sum(weight * profile, axis=-1)


def kain_fritsch_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    w_grid: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: KainFritschConfig = KainFritschConfig(),
) -> tuple[ConvectionOutput, jax.Array]:
    """Kain-Fritsch convection (smooth, differentiable).

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Environmental temperature [K].  Surface at ``[:, -1]``.
    q_v : jax.Array, shape (ncol, nlev)
        Water-vapor specific humidity [kg/kg].
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa].
    w_grid : jax.Array, shape (ncol, nlev)
        Grid-scale vertical-velocity proxy [m/s].  The non-hydrostatic
        bridge passes ``state.w`` (interpolated to full levels).  The
        hydrostatic and spectral-PE bridges derive ``w`` from
        ``∇·v_h`` via the standard sigma-coord continuity (``σ̇`` →
        ``ω``) and then ``w = -ω/(ρ g)`` (see
        :func:`legoesm.atmosphere.physics._shared.diagnose_grid_w_from_omega`).
        On grids that do not expose a divergence operator the bridge
        falls back to zeros and the trigger is driven by
        ``parcel_perturb_T`` alone.
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Convection prognostic carry.  KF is fully diagnostic at the
        physics-state level — we pack the diagnosed cloud-base mass
        flux ``M_b`` at ``[:, -1]`` for visibility but do not use it
        for relaxation (unlike ZM).
    dt : float
        Time step [s].
    config : KainFritschConfig
        Scheme tunables.

    Returns
    -------
    out : ConvectionOutput
        Tendencies on environment T, q_v, q_c plus CAPE diagnostic.
        ``du_dt_conv = dv_dt_conv = None`` (KF has no CMT).
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
        Updated carry with diagnosed ``M_b`` packed at ``[:, -1]``.
    """
    ncol, nlev = T.shape
    del conv_prog_profile  # KF is diagnostic; we only emit a fresh profile.

    # -- Column geometry, moist adiabat, CAPE ------------------------------
    # Use virtual-T moist hydrostatic geometry (clean_physics iter-2 #2).
    dz, rho, z = _compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]

    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)

    # -- LCL, LFC, LNB diagnostics -----------------------------------------
    T_parcel = T_base + config.parcel_perturb_T
    q_parcel = q_base + config.parcel_perturb_q
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)

    # -- The KF trigger function (the AD chokepoint) -----------------------
    # Smooth-interpolate environment T and grid-scale w at the LCL.
    T_env_at_lcl = _interpolate_at_smooth_level(T, k_lcl_smooth)
    w_grid_at_lcl = _interpolate_at_smooth_level(w_grid, k_lcl_smooth)

    T_lcl_perturbed = (
        lcl.T_lcl + config.w_thresh_scale * w_grid_at_lcl - config.w_thresh_offset
    )
    trigger_weight = smooth_step(
        T_lcl_perturbed - T_env_at_lcl, config.trigger_sharpness,
    )

    # -- CAPE gate (a secondary safety net) --------------------------------
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
    overall_weight = trigger_weight * cape_weight

    # -- Cloud-base mass flux closure: CAPE / cape_consumption_time --------
    # Following Kain (2004) §3 — the cloud-base mass flux is
    # ``M_b = rho_BL * CAPE / (g * tau_consume)`` (kg/m^2/s) modulated
    # by the trigger.  The earlier formula omitted both ``rho_BL`` and
    # ``g``, leaving units of m^2/s^3 — the bug was masked operationally
    # by the ``M_b_max`` cap but produced an order-of-magnitude error in
    # the gradient w.r.t. CAPE and made ``M_b`` independent of surface
    # density.
    rho_BL = p_full[:, -1] / (constants.R_d * jnp.maximum(T[:, -1], 1.0))
    M_b = (
        overall_weight
        * rho_BL
        * cape
        / (constants.g * jnp.maximum(config.cape_consumption_time, dt))
    )
    # Bound M_b to a literature peak tropical value
    # (config.M_b_max, default 0.1 kg/m²/s) — see ZhangMcFarlaneConfig.
    M_b = jnp.clip(M_b, 0.0, config.M_b_max)

    # -- Plume integration -------------------------------------------------
    # Same entraining-detraining plume as ZM, with KF default
    # ``epsilon_0 = delta_0 = 2e-3`` (a slightly stronger entrainment).
    eps_profile = jnp.full_like(T, config.epsilon_0)
    dlt_profile = jnp.full_like(T, config.delta_0)
    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_lcl_smooth,
        eps_profile, dlt_profile, M_b,
        buoyancy_death_memory=config.buoyancy_death_memory,
    )

    # -- Cloud depth — z(LCL) → z(LNB) -------------------------------------
    z_lcl = _interpolate_at_smooth_level(z, k_lcl_smooth)
    z_lnb = _interpolate_at_smooth_level(z, k_lnb_smooth)
    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)

    # Deep vs shallow blend — applied as a per-column scalar weight.
    deep_weight = smooth_step(
        cloud_depth - config.cloud_depth_min, config.cloud_depth_sharpness,
    )
    if config.enable_shallow:
        shallow_weight = 1.0 - deep_weight
    else:
        shallow_weight = jnp.zeros_like(deep_weight)
    branch_weight = deep_weight + shallow_weight  # = 1 with shallow on; = deep_weight only

    # Cap plume.M_u once at the source so every downstream use sees
    # the bounded value (see ZM).
    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
    plume = plume._replace(M_u=plume_M_u_capped)

    # -- Environmental tendencies via the shared mass-flux kernel ----------
    # Plume splits vapor (``plume.q_u``) and cloud water
    # (``plume.q_c_u``) explicitly so we use the kernel's correct
    # cloud-water source directly (see ZM).
    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
        z, rho, config.delta_0, M_u_max=config.M_b_max,
    )

    # Apply the deep+shallow weight as a per-column scalar.
    dT_dt = dT_dt_raw * branch_weight[:, None]
    dq_v_dt = dq_v_dt_raw * branch_weight[:, None]
    dq_c_conv_dt = dq_c_conv_dt_raw * branch_weight[:, None]

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=overall_weight,
        # KF has no convective momentum transport.
        du_dt_conv=None,
        dv_dt_conv=None,
    )

    # Diagnostic carry: pack final M_b at [:, -1] for visibility; the
    # other slots stay zero — KF is fully diagnostic.
    conv_prog_profile_new = (
        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(M_b)
    )
    return out, conv_prog_profile_new
