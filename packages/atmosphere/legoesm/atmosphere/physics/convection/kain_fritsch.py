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


def _interp_profile_at_height(
    profile: jax.Array,
    z: jax.Array,
    z_target: jax.Array,
    *,
    sharpness_m: float = 200.0,
) -> jax.Array:
    """Smooth interpolation of ``profile`` at a target height ``z_target``.

    Differentiable Gaussian-in-height weighting (width ``sharpness_m`` [m])
    so the result is C^1 in ``z_target``.  ``profile``, ``z`` shape
    ``(ncol, nlev)``; ``z_target`` shape ``(ncol,)``; returns ``(ncol,)``.
    Used to read the environmental temperature at the dry-adiabatic LCL
    height (oracle reads TENV at ZLCL via linear z-interpolation; the
    Gaussian kernel is the smooth surrogate).
    """
    w = jax.nn.softmax(
        -((z - z_target[:, None]) / sharpness_m) ** 2,
        axis=-1,
    )
    return jnp.sum(w * profile, axis=-1)


def _usl_mass_weighted(
    field: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    usl_depth_pa: float,
    *,
    sharpness_pa: float = 1.0e3,
) -> jax.Array:
    """Pressure-mass-weighted mean of ``field`` over the ~50 hPa updraft
    source layer (USL) at the surface.

    Faithful to KF-Eta `KF_eta_PARA` lines 901-911: the trigger parcel's
    T and q are the mass-weighted average over the lowest model layers
    whose combined depth reaches ``DPMIN`` (~50 hPa).  Here the source
    layer is anchored at the surface (the SCM/idealised bridge launches
    from the surface parcel; the oracle sweeps candidate USL bases but
    the surface layer is the canonical deep-convection source for a
    conditionally-unstable tropical column).

    The layer membership is a smooth sigmoid in cumulative pressure
    depth from the surface so the result is differentiable.  ``field``,
    ``p_full`` shape ``(ncol, nlev)``; ``p_half`` shape ``(ncol, nlev+1)``;
    returns ``(ncol,)``.
    """
    dp = p_half[:, 1:] - p_half[:, :-1]  # > 0, surface-last
    p_sfc = p_half[:, -1:]
    # Depth below each full level (pressure increases toward surface, so
    # ``p_sfc - p_full`` is the height above surface in pressure units).
    depth_above = p_sfc - p_full  # 0 at surface, grows aloft
    # Smooth membership: ~1 within ``usl_depth_pa`` of the surface, ~0 above.
    member = jax.nn.sigmoid((usl_depth_pa - depth_above) / sharpness_pa)
    w = member * dp
    return jnp.sum(w * field, axis=-1) / jnp.maximum(
        jnp.sum(w, axis=-1), 1e-30
    )


def _faithful_dtlcl(
    w_at_lcl: jax.Array,
    z_lcl: jax.Array,
    config: KainFritschConfig,
) -> jax.Array:
    """Fritsch-Chappell w-dependent LCL temperature perturbation DTLCL [K].

    Faithful to Kain (2004) Eqs. 1-2 (oracle `KF_eta_PARA` lines 978-988):

        WKLCL = wklcl_ref * min(ZLCL, z_ref) / z_ref          (Eq. 2)
        WKL   = w_grid * (DX / 25 km) - WKLCL
        DTLCL = dtlcl_coeff * WKL^dtlcl_exponent  if WKL > 0   (Eq. 1)
              = 0                                  otherwise

    The hard ``WKL > 1e-4`` branch and the ``WKL^0.33`` (infinite slope
    at 0) are replaced by a smooth, C^1 surrogate so the trigger's
    gradient w.r.t. ``w_grid`` stays finite everywhere:

        WKL_+ = softplus(s*WKL)/s                  (smooth positive part)
        DTLCL = dtlcl_coeff * ((WKL_+ + eps)^p - eps^p)

    which is ~0 for WKL <= 0, matches ``dtlcl_coeff*WKL^p`` for WKL >> eps,
    and has a finite derivative ``dtlcl_coeff*p*eps^(p-1)`` at WKL = 0.
    """
    z_ratio = jnp.minimum(z_lcl, config.wklcl_zref) / config.wklcl_zref
    wklcl = config.wklcl_ref * z_ratio
    wkl = w_at_lcl * config.dtlcl_dx_scale - wklcl
    s = config.wkl_softplus_sharpness
    wkl_plus = jax.nn.softplus(s * wkl) / s
    eps = config.wkl_floor
    p = config.dtlcl_exponent
    dtlcl = config.dtlcl_coeff * (
        (wkl_plus + eps) ** p - eps ** p
    )
    return dtlcl, wkl


def _faithful_rad(
    wkl: jax.Array,
    config: KainFritschConfig,
) -> jax.Array:
    """KF updraft radius RAD [m] (Kain 2004 Eq. 6; oracle lines 1054-1061).

    1000 m for WKL<=0, 2000 m for WKL>=0.1 m/s, linear between.  Smooth
    clamp keeps the map differentiable.
    """
    frac = jnp.clip(wkl / config.rad_wkl_ref, 0.0, 1.0)
    return config.rad_min_m + (config.rad_max_m - config.rad_min_m) * frac


def _faithful_entrainment_profile(
    wkl: jax.Array,
    rho: jax.Array,
    config: KainFritschConfig,
) -> jax.Array:
    """Per-level bulk-plume fractional entrainment rate [1/m] from the KF
    updraft radius (Kain 2004 Eqs. 5-6; oracle lines 1054-1061, 1194).

    The oracle's environmental inflow (mass) rate is
    ``REI = VMFLCL * DP * entrain_const / RAD`` with ``DP = rho*g*dz``;
    the *fractional* entrainment per unit depth is therefore

        epsilon(z) = (1/M) dM/dz ~ REI/(VMFLCL * dz)
                   = rho(z) * g * entrain_const / RAD          [1/m]

    i.e. it carries the ``rho*g`` factor that converts the oracle's
    per-pressure inflow into a per-height fractional rate.  Larger
    background ascent (WKL) -> larger radius -> weaker fractional
    entrainment, exactly as in KF.  ``wkl`` shape ``(ncol,)``; ``rho``
    shape ``(ncol, nlev)``; returns ``(ncol, nlev)``.
    """
    rad = _faithful_rad(wkl, config)  # (ncol,)
    return rho * constants.g * config.entrain_const / rad[:, None]


def _precip_efficiency(
    z_lcl: jax.Array,
    config: KainFritschConfig,
) -> jax.Array:
    """Cloud-base-height precipitation efficiency PEFCBH (Kain 2004;
    oracle lines 1616-1627).

    ``CBH`` is the cloud-base height in kft; ``RCBH`` a 5th-order
    polynomial; ``PEFCBH = 1/(1+RCBH)`` clamped to ``pef_max``.  In the
    oracle the final PEF is the mean of this and a wind-shear term; the
    SCM/idealised bridge has no resolved shear, so we use PEFCBH alone
    (the no-shear limit gives PEF=1.591 -> clamped to pef_max=0.9, so the
    shear term contributes only its clamp).  Returns a value in
    ``[pef_min, pef_max]``.
    """
    cbh = (z_lcl) * 3.281e-3  # m -> kft
    rcbh_poly = 0.96729352 + cbh * (
        -0.70034167 + cbh * (
            0.162179896 + cbh * (
                -1.2569798e-2 + cbh * (4.2772e-4 - cbh * 5.44e-6)
            )
        )
    )
    # Smooth low-CBH branch: oracle uses RCBH=0.02 for CBH<3 kft.
    low = jax.nn.sigmoid((3.0 - cbh) * 5.0)
    rcbh = low * 0.02 + (1.0 - low) * rcbh_poly
    pefcbh = 1.0 / (1.0 + jnp.maximum(rcbh, 0.0))
    return jnp.clip(pefcbh, config.pef_min, config.pef_max)


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

    # -- Updraft source layer (USL): ~50 hPa mass-weighted parcel ----------
    # Faithful to KF-Eta: the trigger parcel's T and q are the mass-weighted
    # mean over the lowest ~50 hPa, NOT a single surface point.  This both
    # follows the oracle and makes the launched parcel less extreme (a
    # surface-point parcel over-states near-surface superheating).
    T_usl = _usl_mass_weighted(T, p_full, p_half, config.usl_depth_pa)
    q_usl = _usl_mass_weighted(q_v, p_full, p_half, config.usl_depth_pa)
    z_usl = _usl_mass_weighted(z, p_full, p_half, config.usl_depth_pa)

    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)

    # -- LCL, LFC, LNB diagnostics -----------------------------------------
    # Parcel launched from the USL-mixed thermodynamic state plus the
    # sub-cloud perturbations.
    T_parcel = T_usl + config.parcel_perturb_T
    q_parcel = q_usl + config.parcel_perturb_q
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)

    # -- The KF trigger function (the AD chokepoint) -----------------------
    if config.faithful_trigger:
        # Faithful KF-Eta trigger (oracle lines 928-1022).  The LCL height
        # is the DRY-adiabatic ascent of the USL parcel:
        #     ZLCL = ZMIX + (TLCL - TMIX) / GDRY,   GDRY = -g/c_pd
        # and the environmental T at the LCL is read at THAT height (the
        # parcel cools along the dry adiabat, 9.8 K/km, faster than the
        # environment, so the env-T reference must be at the true LCL
        # height — reading it at a too-low height (e.g. a Bolton p_lcl that
        # saturates a moist parcel prematurely) over-warms the reference and
        # spuriously suppresses the trigger).  ``lcl.T_lcl`` (Bolton) is the
        # LCL temperature; we recompute the LCL *height* the oracle way.
        gdry = -constants.g / constants.c_pd  # K/m, negative
        z_lcl_dry = z_usl + (lcl.T_lcl - T_usl) / gdry
        z_lcl_for_trigger = z_lcl_dry
        # Read env T and grid-scale w at the dry-adiabatic LCL height.
        T_env_at_lcl = _interp_profile_at_height(T, z, z_lcl_dry)
        w_grid_at_lcl = _interp_profile_at_height(w_grid, z, z_lcl_dry)
        # Fritsch-Chappell w-dependent DTLCL (Kain 2004 Eq. 1-2).  Fire when
        # the perturbed parcel temperature at the LCL exceeds the
        # environmental temperature there: TLCL + DTLCL(w) >= TENV.
        dtlcl, wkl = _faithful_dtlcl(w_grid_at_lcl, z_lcl_for_trigger, config)
        T_lcl_perturbed = lcl.T_lcl + dtlcl
    else:
        # Legacy: smooth-interpolate environment T and w at the level index.
        T_env_at_lcl = _interpolate_at_smooth_level(T, k_lcl_smooth)
        w_grid_at_lcl = _interpolate_at_smooth_level(w_grid, k_lcl_smooth)
        # Legacy linear trigger (back-compat for existing tuning).
        wkl = w_grid_at_lcl  # informational; not the FC form
        T_lcl_perturbed = (
            lcl.T_lcl
            + config.w_thresh_scale * w_grid_at_lcl
            - config.w_thresh_offset
        )
    trigger_weight = smooth_step(
        T_lcl_perturbed - T_env_at_lcl, config.trigger_sharpness,
    )

    # -- CAPE gate (a secondary safety net) --------------------------------
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
    # CAPE-based OR fallback for the dynamical trigger.  The w-trigger
    # above is starved when there is no resolved grid-scale ascent
    # (``w_grid = 0`` in SCM / divergence-free dycores), which otherwise
    # leaves a strongly-unstable column in near-radiative equilibrium.
    # Firing when undilute CAPE exceeds ``cape_or_threshold`` rescues that
    # case.  The fallback is itself gated by the ABSENCE of resolved
    # ascent — ``exp(-(w_grid_at_lcl / cape_or_w_ref)^2)`` is ≈1 only where
    # ``w_grid ≈ 0`` and →0 wherever the bridge supplies a real grid-scale
    # ``w`` — so in any 3-D run with resolved ascent the OR branch
    # vanishes and KF uses the pure w-trigger unchanged (preserving its
    # documented response to resolved divergence).  Set
    # ``cape_or_threshold = inf`` to disable the fallback entirely.
    # See KainFritschConfig.
    w_absent = jnp.exp(
        -(w_grid_at_lcl / jnp.maximum(config.cape_or_w_ref, 1e-30)) ** 2
    )
    cape_or_weight = w_absent * cape_trigger(
        cape, config.cape_or_threshold, config.cape_or_sharpness,
    )
    fire_weight = jnp.maximum(trigger_weight, cape_or_weight)
    overall_weight = fire_weight * cape_weight

    # -- Cloud-base mass flux closure: CAPE / TIMEC ------------------------
    # KF removes ~90% of CAPE over the advective timescale TIMEC by
    # iterating the cloud-base mass flux (oracle lines 1879-2281).  TIMEC =
    # DX/VCONV is bounded to [timec_min_s, timec_max_s]; the SCM/idealised
    # bridge does not expose the LCL/mid-trop wind that sets VCONV, so we
    # use ``cape_consumption_time`` as the operative TIMEC, clamped into the
    # faithful bounds.  The bulk closure then sets the cloud-base mass flux
    # as the (smooth, single-pass) CAPE-consumption rate
    # ``M_b = rho_BL * CAPE / (g * TIMEC)`` — the AD-safe surrogate for the
    # oracle's AINC iteration that lands CAPE near 5-10% of its original
    # value over TIMEC.  ``rho_BL`` and ``g`` are explicit (units kg/m^2/s).
    timec = jnp.clip(
        jnp.asarray(config.cape_consumption_time, dtype=T.dtype),
        config.timec_min_s,
        config.timec_max_s,
    )
    timec = jnp.maximum(timec, dt)
    rho_BL = p_full[:, -1] / (constants.R_d * jnp.maximum(T[:, -1], 1.0))
    # ``M_b_closure`` is the raw (uncapped) closure cloud-base mass flux —
    # the diagnostic packed into the carry, monotone in both the trigger
    # weight (hence in the resolved ``w_grid``) and CAPE.  Keeping the carry
    # on the *uncapped* value means the diagnostic stays responsive to the
    # trigger even when the *applied* mass flux saturates at ``M_b_max``
    # (the stability cap), so downstream diagnostics and the spectral-PE
    # w-grid response test see the genuine closure signal.
    M_b_closure = (
        overall_weight
        * rho_BL
        * cape
        / (constants.g * timec)
    )
    # Bound the *applied* M_b to a literature peak tropical value
    # (config.M_b_max) — see ZhangMcFarlaneConfig.  The cap protects the
    # integration from the unbounded CAPE/tau closure spiking in a
    # high-CAPE column; it does NOT alter the diagnostic carry above.
    M_b = jnp.clip(M_b_closure, 0.0, config.M_b_max)

    # -- Plume integration -------------------------------------------------
    # Entraining-detraining plume.  When ``faithful_entrainment`` is set the
    # fractional entrainment rate is derived from the KF updraft radius
    # (Kain 2004 Eq. 5-6): epsilon = entrain_const / RAD with RAD ramping
    # 1000 m (no background ascent) -> 2000 m (WKL>=0.1 m/s).  Stronger
    # resolved ascent -> larger radius -> weaker fractional entrainment,
    # exactly as in the oracle.  The detrainment rate tracks entrainment
    # (bulk single-plume; the oracle's PROF5 buoyancy-sorted per-level
    # detrainment is the acknowledged structural simplification).
    if config.faithful_entrainment:
        eps_profile = _faithful_entrainment_profile(wkl, rho, config)
        dlt_profile = eps_profile
    else:
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

    # Diagnostic carry: pack the (uncapped) closure cloud-base mass flux at
    # [:, -1] for visibility; the other slots stay zero — KF is fully
    # diagnostic.  Using ``M_b_closure`` (not the capped ``M_b``) keeps the
    # carry monotone in the resolved-``w_grid`` trigger response even where
    # the applied flux saturates at ``M_b_max``.
    conv_prog_profile_new = (
        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(M_b_closure)
    )
    return out, conv_prog_profile_new
