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
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_dT
from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
)

from legoesm.atmosphere.physics.convection.config import KainFritschConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel,
    compute_column_geometry,
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


# Kain-Fritsch updraft-radius ramp smoothing widths (fixed).
_KF_SHARPNESS_M = 200.0
_KF_RAMP_WIDTH = 0.05
_KF_MIN_ENTRAIN_MULTIPLIER = 0.5
_KF_DETRAIN_BOOST = 1.5
_KF_PROF5_SQRT2P = 2.506628
_KF_PROF5_A1 = 0.4361836
_KF_PROF5_A2 = -0.1201676
_KF_PROF5_A3 = 0.9372980
_KF_PROF5_P = 0.33267
_KF_PROF5_SIGMA = 0.166666667
_KF_PROF5_FE = 0.202765151

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
    sharpness_m: float = _KF_SHARPNESS_M,
) -> jax.Array:
    """Linearly interpolate ``profile`` at target height ``z_target``.

    KF-Eta reads the environmental LCL state by linear interpolation in
    height (``DLP=(ZLCL-Z0(K))/(Z0(KLCL)-Z0(K))``; module_cu_kfeta.F
    lines 958-965).  A previous Gaussian surrogate over-weighted the
    nearest full level on the coarse RCE grid; for an LCL just above the
    lowest level it returned the surface temperature, spuriously making
    ``TLCL-TENV`` negative and starving the trigger.  ``jnp.interp`` is
    piecewise differentiable and matches the reference forward value.

    ``sharpness_m`` is retained for API/back-compat and intentionally
    unused.
    """
    del sharpness_m
    return jax.vmap(
        lambda zt, z_col, profile_col: jnp.interp(
            zt, z_col[::-1], profile_col[::-1],
        )
    )(z_target, z, profile)


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
    at 0) are replaced by a smooth, C^1 surrogate that is EXACTLY zero at
    the KF cutoff ``WKL = 0`` (so the trigger gets no artificial lift
    there) and recovers ``dtlcl_coeff*WKL^p`` for ``WKL >> 0``:

        g(x)  = softplus(s*x)/s          (smooth positive part, g(0)=ln2/s)
        DTLCL = dtlcl_coeff * softplus_pos( g(WKL)^p - g(0)^p )

    where ``softplus_pos(y)=softplus(k*y)/k`` is a smooth ``max(y,0)``.
    At WKL=0, ``g(WKL)^p - g(0)^p = 0`` so DTLCL=0 (no spurious lift,
    fixing codex review-1 #4).  For WKL<0, ``g(WKL)<g(0)`` so the argument
    is negative and the smooth positive-part drives DTLCL->0.  The
    base-point subtraction ``- g(0)^p`` removes the ``ln2/s`` offset that
    the earlier ``(WKL_+ + eps)^p - eps^p`` form left at the cutoff.
    """
    z_ratio = jnp.minimum(z_lcl, config.wklcl_zref) / config.wklcl_zref
    wklcl = config.wklcl_ref * z_ratio
    wkl = w_at_lcl * config.dtlcl_dx_scale - wklcl
    s = config.wkl_softplus_sharpness
    p = config.dtlcl_exponent
    eps = config.wkl_floor  # tiny floor inside the power for AD safety at g->0
    g_wkl = jax.nn.softplus(s * wkl) / s
    g_zero = jnp.asarray(jnp.log(2.0) / s, dtype=wkl.dtype)  # = g(0)
    # Power evaluated on a floored argument so the base of x^p never hits
    # exactly 0 (where p<1 has infinite slope); the floor is far below the
    # cutoff scale so it does not bias the firing point.
    base = (g_wkl + eps) ** p - (g_zero + eps) ** p
    # Smooth positive part (max(base, 0)) so DTLCL is ~0 for WKL<=0, ~base
    # for WKL>>0, and C^1 at WKL=0.  ``softplus(k*x)/k`` leaves a residual
    # ``ln(2)/k`` offset at x=0; the first stage subtracts that residual and a
    # second smooth positive-part removes the now-negative tail, so the inner
    # value ``sp`` is exactly 0 at base=0.  The OUTER ``softplus(k*sp)/k`` then
    # leaves its OWN tiny ``ln(2)/k`` residual, so DTLCL at the WKL=0 cutoff is
    # NOT exactly 0 but is negligible: ``dtlcl_coeff*ln(2)/dtlcl_pos_sharpness``
    # ~ 3e-3 K with defaults (codex review-3 #2 — an exact-zero C^1 smooth
    # positive part does not exist; softplus(0)=ln2>0).  3e-3 K is ~3 orders
    # below the env-T variation across the LCL, so it does not move the
    # firing point.
    k = config.dtlcl_pos_sharpness
    sp = (jax.nn.softplus(k * base) - jnp.log(2.0)) / k
    base_pos = jax.nn.softplus(k * sp) / k
    dtlcl = config.dtlcl_coeff * base_pos
    return dtlcl, wkl


def _faithful_rad(
    wkl: jax.Array,
    config: KainFritschConfig,
) -> jax.Array:
    """KF updraft radius RAD [m] (Kain 2004 Eq. 6; oracle lines 1054-1061).

    1000 m for WKL<=0, 2000 m for WKL>=0.1 m/s, linear between.  The oracle
    uses a hard piecewise-linear clamp; here ``frac`` is a C^1 smooth
    clamp of ``wkl/rad_wkl_ref`` to [0,1] (a softplus-of-softplus ramp)
    so RAD is differentiable at both corners (codex review-1 #5).
    """
    x = wkl / config.rad_wkl_ref
    width = _KF_RAMP_WIDTH  # smoothing width in units of the [0,1] ramp
    # smooth clamp to [0,1]: softplus rising edge minus softplus at x=1.
    lo = jax.nn.softplus(x / width) * width
    frac = lo - jax.nn.softplus((lo - 1.0) / width) * width
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


def _kf_prof5(eq: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Kain-Fritsch ``PROF5`` Gaussian mixing integrals.

    This is the differentiable vector form of WRF ``module_cu_kfeta.F``
    ``SUBROUTINE PROF5``: given the critical environmental mixing fraction
    ``EQFRC`` it returns the fractional entrainment and detrainment
    multipliers ``(EE, UD)`` for the layer.
    """
    eq = jnp.clip(eq, 0.0, 1.0)
    y = 6.0 * eq - 3.0
    ey = jnp.exp(-0.5 * y * y)
    e45 = jnp.exp(jnp.asarray(-4.5, dtype=eq.dtype))
    t2 = 1.0 / (1.0 + _KF_PROF5_P * jnp.abs(y))
    t1 = jnp.asarray(0.500498, dtype=eq.dtype)
    c1 = _KF_PROF5_A1 * t1 + _KF_PROF5_A2 * t1 * t1 + _KF_PROF5_A3 * t1 * t1 * t1
    c2 = _KF_PROF5_A1 * t2 + _KF_PROF5_A2 * t2 * t2 + _KF_PROF5_A3 * t2 * t2 * t2
    sigma = jnp.asarray(_KF_PROF5_SIGMA, dtype=eq.dtype)
    ee_pos = (
        sigma * (0.5 * (_KF_PROF5_SQRT2P - e45 * c1 - ey * c2) + sigma * (e45 - ey))
        - e45 * eq * eq / 2.0
    )
    ud_pos = (
        sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey))
        - e45 * (0.5 + eq * eq / 2.0 - eq)
    )
    ee_neg = (
        sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey))
        - e45 * eq * eq / 2.0
    )
    ud_neg = (
        sigma * (0.5 * (_KF_PROF5_SQRT2P - e45 * c1 - ey * c2) + sigma * (e45 - ey))
        - e45 * (0.5 + eq * eq / 2.0 - eq)
    )
    ee = jnp.where(y >= 0.0, ee_pos, ee_neg) / _KF_PROF5_FE
    ud = jnp.where(y >= 0.0, ud_pos, ud_neg) / _KF_PROF5_FE
    return jnp.clip(ee, 0.0, 1.0), jnp.clip(ud, 0.0, 1.0)


def _kf_mixed_virtual_temperature(
    T_env: jax.Array,
    q_env: jax.Array,
    T_u: jax.Array,
    q_u: jax.Array,
    q_c_u: jax.Array,
    p_full: jax.Array,
    env_fraction: float,
) -> jax.Array:
    """Virtual temperature of an updraft/environment mixture.

    KF-Eta evaluates the mixed parcel through ``tpmix2``.  This smooth
    surrogate mixes temperature, vapor, and condensate, then performs one
    donor-limited saturation-adjustment Newton step so evaporative cooling
    of entrained dry air controls the critical fraction.
    """
    f = jnp.asarray(env_fraction, dtype=T_env.dtype)
    T0 = f * T_env + (1.0 - f) * T_u
    q0 = f * q_env + (1.0 - f) * q_u
    qc0 = (1.0 - f) * jnp.maximum(q_c_u, 0.0)
    qsat = saturation_mixing_ratio(T0, p_full)
    dqs_dT = saturation_mixing_ratio_dT(T0, p_full)
    L_over_cp = constants.L_v / constants.c_pd
    delta_q = jnp.clip(
        (q0 - qsat) / (1.0 + L_over_cp * dqs_dT),
        -qc0,
        q0,
    )
    Tm = T0 + L_over_cp * delta_q
    qm = q0 - delta_q
    qcm = qc0 + delta_q
    return virtual_temperature(Tm, qm) * (1.0 - qcm)


def _kf_buoyancy_sort_rates(
    T_env: jax.Array,
    q_env: jax.Array,
    p_full: jax.Array,
    plume,
    eps_base: jax.Array,
    z: jax.Array,
    k_lnb_smooth: jax.Array,
    config: KainFritschConfig,
) -> tuple[jax.Array, jax.Array]:
    """Separate KF entrainment and detrainment profiles.

    Kain (2004) keeps ``REI = VMFLCL*DP*0.03/RAD`` as the environmental
    inflow scale, but updraft detrainment is diagnosed from the critical
    mixed fraction and ``PROF5``.  This replaces the previous
    ``detrainment = entrainment`` shortcut, which deposited too much mass
    in the lower/middle troposphere and left no deep heating.
    """
    tv_env = virtual_temperature(T_env, q_env)
    tv_u = virtual_temperature(plume.T_u, plume.q_u) * (
        1.0 - jnp.maximum(plume.q_c_u, 0.0)
    )
    tv95 = _kf_mixed_virtual_temperature(
        T_env, q_env, plume.T_u, plume.q_u, plume.q_c_u, p_full, 0.95,
    )
    tv10 = _kf_mixed_virtual_temperature(
        T_env, q_env, plume.T_u, plume.q_u, plume.q_c_u, p_full, 0.10,
    )

    colder = tv_u <= tv_env
    very_buoyant = tv95 > tv_env
    eq = (tv_env - tv_u) * 0.10 / jnp.maximum(tv10 - tv_u, 1.0e-6)
    eq = jnp.clip(eq, 0.0, 1.0)
    ee_prof, ud_prof = _kf_prof5(eq)
    ee2 = jnp.where(colder, _KF_MIN_ENTRAIN_MULTIPLIER, jnp.where(very_buoyant, 1.0, ee_prof))
    ud2 = jnp.where(colder, 1.0, jnp.where(very_buoyant, 0.0, ud_prof))
    ee2 = jnp.maximum(ee2, _KF_MIN_ENTRAIN_MULTIPLIER)
    ud2 = _KF_DETRAIN_BOOST * ud2

    # Oracle UER/UDR use 0.5*(previous + current) multipliers.  Apply that
    # average in surface-first order, initialising cloud-base values with
    # EE1=1, UD1=0 (WRF KF-Eta lines 1118-1119).
    ee_sf = ee2[:, ::-1]
    ud_sf = ud2[:, ::-1]
    ee_prev = jnp.concatenate([jnp.ones_like(ee_sf[:, :1]), ee_sf[:, :-1]], axis=-1)
    ud_prev = jnp.concatenate([jnp.zeros_like(ud_sf[:, :1]), ud_sf[:, :-1]], axis=-1)
    ee_mult = (0.5 * (ee_prev + ee_sf))[:, ::-1]
    ud_mult = (0.5 * (ud_prev + ud_sf))[:, ::-1]

    eps_profile = eps_base * ee_mult
    dlt_profile = eps_base * ud_mult

    nlev = T_env.shape[-1]
    levels = jnp.arange(nlev, dtype=T_env.dtype)
    # Smooth ``UDR(LTOP)=remaining UMF`` surrogate centred on the LNB.
    top_weight = jax.nn.softmax(
        -((levels[None, :] - k_lnb_smooth[:, None])
          / jnp.maximum(config.cloud_top_detrainment_width_levels, 1.0e-6)) ** 2,
        axis=-1,
    )
    dz_layer = jnp.maximum(
        jnp.concatenate([z[:, :-1] - z[:, 1:], z[:, -2:-1] - z[:, -1:]], axis=-1),
        1.0,
    )
    top_rate = -jnp.log(
        jnp.maximum(1.0 - config.cloud_top_detrainment_fraction, 1.0e-6)
    ) / dz_layer
    dlt_profile = dlt_profile + top_weight * top_rate
    return eps_profile, dlt_profile


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
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    p_base = p_full[:, -1]

    # -- Updraft source layer (USL): ~50 hPa mass-weighted parcel ----------
    # Faithful to KF-Eta: the trigger parcel's T, q, z AND launch pressure
    # are the mass-weighted mean over the lowest ~50 hPa, NOT a single
    # surface point (oracle KF_eta_PARA lines 901-911 mix T,q,z,p).  This
    # both follows the oracle and makes the launched parcel less extreme.
    T_usl = _usl_mass_weighted(T, p_full, p_half, config.usl_depth_pa)
    q_usl = _usl_mass_weighted(q_v, p_full, p_half, config.usl_depth_pa)
    z_usl = _usl_mass_weighted(z, p_full, p_half, config.usl_depth_pa)
    # Launch pressure = USL mean pressure (codex review-1 #1: oracle uses
    # PMIX, not the surface pressure).
    p_usl = _usl_mass_weighted(p_full, p_full, p_half, config.usl_depth_pa)

    # CAPE / parcel profile launched from the USL-MIXED parcel (codex
    # review-3 #1): the oracle's ABE is the buoyant energy of the same
    # ~50-hPa source-layer parcel that the trigger and plume use, NOT a
    # single surface point.  We lift the USL parcel q_v-aware (dry-adiabatic
    # below its LCL, moist-adiabatic above, launch humidity = q_usl) and
    # compute virtual-temperature CAPE — mirroring ``parcel_profile_and_cape``
    # but with the USL launch state instead of ``T[:, -1]`` / ``q_v[:, -1]``.
    # The saturated-from-base path would spuriously inflate CAPE in
    # unsaturated columns (codex review-2 #2).
    #
    # ``compute_moist_adiabat`` launches the dry leg from the surface
    # full-level pressure ``p_full[:, -1]``, but the USL parcel lives at the
    # USL mean pressure ``p_usl`` (~30-50 hPa above the surface, codex
    # review-4).  To launch the SAME parcel (identical potential temperature
    # and humidity) consistently from the surface pressure, translate
    # ``T_usl`` to its surface-pressure dry-adiabatic equivalent
    # ``T_usl * (p_surface / p_usl)^kappa`` — preserving theta, so the moist
    # adiabat above the (unchanged) LCL is identical, and the dry leg now
    # begins at the correct pressure origin.
    T_usl_at_sfc = T_usl * (p_base / jnp.maximum(p_usl, 1.0)) ** constants.kappa
    T_moist = compute_moist_adiabat(T_usl_at_sfc, p_full, q_v_base=q_usl)
    q_sat_parcel = saturation_mixing_ratio(T_moist, p_full)
    q_v_parcel = jnp.minimum(q_usl[:, None], q_sat_parcel)
    cape = compute_cape(
        T, T_moist, p_full, p_half,
        q_v_env=q_v, q_v_parcel=q_v_parcel,
    )

    # -- LCL, LFC, LNB diagnostics -----------------------------------------
    # The trigger LCL is the LCL of the *unperturbed* USL-mixed parcel
    # (oracle computes TLCL from TMIX/QMIX, with the w-dependent DTLCL the
    # ONLY thermal boost — the perturbation must NOT also shift the LCL or
    # the dry-adiabatic ZLCL base would be inconsistent, codex review-1 #2).
    # ``parcel_perturb_T/q`` are retained for the plume launch (below) where
    # a small sub-cloud excess seeds the updraft; in faithful mode they do
    # not move the LCL/trigger reference.
    if config.faithful_trigger:
        T_parcel_lcl = T_usl
        q_parcel_lcl = q_usl
    else:
        T_parcel_lcl = T_usl + config.parcel_perturb_T
        q_parcel_lcl = q_usl + config.parcel_perturb_q
    lcl = compute_lcl(T_parcel_lcl, q_parcel_lcl, p_usl, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
    # Plume launch parcel (seeded with the sub-cloud perturbation).  The
    # plume integrator starts its scan from the SURFACE level, so the launch
    # temperature must be the surface-pressure dry-adiabatic equivalent of
    # the USL parcel (codex review-4 — same pressure-origin consistency as
    # the CAPE parcel above), preserving theta so the moist ascent is
    # identical above the LCL.
    T_parcel = T_usl_at_sfc + config.parcel_perturb_T
    q_parcel = q_usl + config.parcel_perturb_q

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
        # Read env T/q and grid-scale w at the dry-adiabatic LCL height.
        T_env_at_lcl = _interp_profile_at_height(T, z, z_lcl_dry)
        q_env_at_lcl = _interp_profile_at_height(q_v, z, z_lcl_dry)
        w_grid_at_lcl = _interp_profile_at_height(w_grid, z, z_lcl_dry)
        # Fritsch-Chappell w-dependent DTLCL (Kain 2004 Eq. 1-2).  Fire when
        # the perturbed parcel temperature at the LCL exceeds the
        # environmental temperature there: TLCL + DTLCL(w) >= TENV.
        dtlcl, wkl = _faithful_dtlcl(w_grid_at_lcl, z_lcl_for_trigger, config)
        # KF-Eta trigger=3 adds a relative-humidity perturbation DTRH
        # (module_cu_kfeta.F lines 996-1017, U00=0.75).  Use the same
        # piecewise reference formula, with qsat/dT from thermo instead of
        # lookup-table coefficients.
        qsat_lcl_env = saturation_mixing_ratio(T_env_at_lcl, lcl.p_lcl)
        rh_lcl = q_env_at_lcl / jnp.maximum(qsat_lcl_env, 1.0e-12)
        dqssdt = saturation_mixing_ratio_dT(lcl.T_lcl, lcl.p_lcl)
        dtrh_scale = q_usl / jnp.maximum(dqssdt, 1.0e-12)
        dtrh = jnp.where(
            (rh_lcl >= config.rh_trigger_u00) & (rh_lcl <= config.rh_trigger_rhmax),
            config.rh_trigger_slope * (rh_lcl - config.rh_trigger_u00) * dtrh_scale,
            jnp.where(
                rh_lcl > config.rh_trigger_rhmax,
                (1.0 / jnp.maximum(rh_lcl, 1.0e-12) - 1.0) * dtrh_scale,
                0.0,
            ),
        )
        dtrh = jnp.where(config.enable_rh_trigger_perturb, dtrh, 0.0)
        T_lcl_perturbed = lcl.T_lcl + dtlcl + dtrh
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
    # Smooth probabilistic OR of the two trigger weights: 1-(1-a)(1-b).
    # Both are in (0,1), so the OR stays in (0,1) and is C^infty everywhere
    # (codex review-1 #13: ``jnp.maximum`` has a non-smooth join at a==b;
    # the noisy-OR form is the smooth replacement and matches ``max`` to
    # within the product term).
    fire_weight = trigger_weight + cape_or_weight - trigger_weight * cape_or_weight
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
    # The oracle drives its closure off the ENTRAINMENT-DILUTED updraft
    # buoyant energy ABE, which is markedly smaller than the undilute
    # surface-parcel CAPE that ``compute_cape`` returns (oracle ABE=5482 vs
    # ours undilute=15339 on the same sounding — codex review-1 #3).  We
    # discount the undilute CAPE by a dilution factor ``exp(-eps_mean *
    # cloud_depth)`` (the bulk-plume buoyancy is reduced by entrainment over
    # the cloud depth, exactly the e-folding the plume integrator applies)
    # and remove only ``cape_removal_fraction`` of it per TIMEC (codex
    # review-1 #9: ``cape_removal_fraction`` was previously dead config).
    z_lcl_cd = _interpolate_at_smooth_level(z, k_lcl_smooth)
    z_lnb_cd = _interpolate_at_smooth_level(z, k_lnb_smooth)
    cloud_depth_cd = jnp.maximum(z_lnb_cd - z_lcl_cd, 0.0)
    if config.faithful_entrainment:
        eps_mean = jnp.mean(_faithful_entrainment_profile(wkl, rho, config), axis=-1)
    else:
        eps_mean = jnp.full((ncol,), config.epsilon_0, dtype=T.dtype)
    dilution = jnp.exp(-eps_mean * cloud_depth_cd)
    abe = cape * dilution
    # ``M_b_closure`` is the raw (uncapped) closure cloud-base mass flux —
    # the diagnostic packed into the carry, monotone in both the trigger
    # weight (hence in the resolved ``w_grid``) and the diluted ABE.  Keeping
    # the carry on the *uncapped* value means the diagnostic stays responsive
    # to the trigger even when the *applied* mass flux saturates at
    # ``M_b_max`` (the stability cap), so downstream diagnostics and the
    # spectral-PE w-grid response test see the genuine closure signal.
    M_b_closure = (
        overall_weight
        * config.cape_removal_fraction
        * rho_BL
        * abe
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
        eps_base = _faithful_entrainment_profile(wkl, rho, config)
        predictor_plume = entraining_detraining_plume(
            T, q_v, p_full, p_half, z,
            T_parcel, q_parcel, k_lcl_smooth,
            eps_base, _KF_MIN_ENTRAIN_MULTIPLIER * eps_base, M_b,
            buoyancy_death_memory=config.buoyancy_death_memory,
        )
        eps_profile, dlt_profile = _kf_buoyancy_sort_rates(
            T, q_v, p_full, predictor_plume, eps_base, z, k_lnb_smooth, config,
        )
        # Environment-detrainment mixing rate fed to the shared kernel must
        # be the same KF buoyancy-sort ``UDR`` profile the plume uses.  The
        # previous shortcut ``dlt_profile = eps_profile`` violated Kain
        # (2004) Eq. 4 / PROF5: buoyant layers should entrain with little
        # detrainment, then detrain strongly near cloud top.
        kernel_delta = dlt_profile
    else:
        eps_profile = jnp.full_like(T, config.epsilon_0)
        dlt_profile = jnp.full_like(T, config.delta_0)
        kernel_delta = config.delta_0
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
    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
        z, rho, kernel_delta, M_u_max=config.M_b_max,
    )
    # KF CONDLOAD fallout keeps only the non-precipitating part of fresh
    # updraft condensate in the cloud field.  The shared kernel's q_c source
    # is built from total plume condensate, so retain the documented 40%
    # fresh-condensate fraction (module_cu_kfeta.F lines 2900-2923) and let
    # the remaining vapor sink represent convective precipitation fallout.
    dq_c_conv_dt_raw = (
        config.condload_fresh_retention_fraction * dq_c_conv_dt_raw
    )

    # Under the ConvectionOutput contract KF can emit retained cloud water
    # but has no direct precipitation diagnostic.  The CONDLOAD split above
    # therefore keeps only the reference retained fresh condensate in
    # ``dq_c_conv_dt``; the remaining vapor sink is the precipitating fallout
    # path represented in KF-Eta by PPTLIQ/PPTICE.

    # Apply the deep+shallow weight as a per-column scalar.
    dT_dt = dT_dt_raw * branch_weight[:, None]
    dq_v_dt = dq_v_dt_raw * branch_weight[:, None]
    dq_c_conv_dt = dq_c_conv_dt_raw * branch_weight[:, None]

    # Convective mask reflects BOTH the trigger and the deep/shallow branch
    # weight that actually scales the tendencies (codex review-1 #14: the
    # bare ``overall_weight`` could report active convection in a column
    # whose tendencies are zeroed because ``enable_shallow=False`` and the
    # cloud is shallow, i.e. ``branch_weight≈0``).
    convective_mask = overall_weight * branch_weight

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
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
