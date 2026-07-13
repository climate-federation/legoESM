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

from typing import NamedTuple

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


__physics_contract__ = {
    "summary": (
        "Kain-Fritsch bulk mass-flux deep/shallow convection: a "
        "boundary-layer-triggered entraining-detraining plume with CONDLOAD "
        "precipitation fallout and an RH-controlled downdraft; smooth "
        "(differentiable) trigger and deep/shallow blend."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "w_grid": "m/s (resolved grid-scale vertical velocity)",
        "conv_prog_profile": "kg/m^2/s (cloud-base mass-flux carry, packed at [:, -1])",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained non-precipitating cloud-water source to microphysics, >=0)",
        "cape": "J/kg", "convective_mask": "1 (0-1 activation)",
        "conv_prog_profile_new": "kg/m^2/s (updated cloud-base mass-flux carry)",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Convection warms aloft and dries the "
        "sub-cloud/lower layers where it stabilizes a conditionally-unstable "
        "column; dq_c_conv_dt >= 0. The CONDLOAD condensate is handed to "
        "microphysics as dq_c (vapor -> cloud, ZM/Emanuel convention; micro "
        "owns precip), so the SCHEME conserves column total water "
        "int(dq_v + dq_c) and moist static energy h = int(c_p dT + L_v q_v) to "
        "MACHINE PRECISION on realistic (moist) columns: condensation warms "
        "(dT += L_v/c_p * cond), the vapor sink is drawn from available "
        "moisture, the kernel's detrained-cloud latent is released, and a "
        "conservative vapor relocation keeps q_v >= 0 without hiding water."
    ),
    # #M1b fix: switched to the conservative implicit_flux mass-flux kernel and
    # a coupled condensation water+energy budget (moisture-limited vapor sink +
    # detrained-cloud latent release + vapor relocation), so the SCHEME now
    # conserves column total water and MSE to machine precision on realistic
    # columns (a bounded residual survives only for degenerate q_v==0 columns).
    "conserves": ["energy", "moisture"],
    "differentiable": True,
    "reference": (
        "Kain & Fritsch (1990), J. Atmos. Sci. 47, 2784-2802; "
        "Kain (2004), J. Appl. Meteor. 43, 170-181"
    ),
    "idealized_test": (
        "tests/unit/test_kain_fritsch.py; neutral column (CAPE=0, no "
        "grid-scale ascent) -> ~zero tendency; a conditionally-unstable column "
        "with resolved ascent -> heating aloft, sub-cloud drying, positive "
        "convective_mask; mask -> 0 when the cloud is too shallow and "
        "enable_shallow=False. Conservation gated by "
        "test_convection_group_validator.py::"
        "{test_tier3_kf_precip_efficiency_leak_removed (column water + MSE h "
        "machine-zero, warm+dry), test_tier3_kf_mse_conserved_on_capped_finite"
        "_lnb_column, test_tier3_kf_vapor_stays_nonnegative}."
    ),
}


# Kain-Fritsch updraft-radius ramp smoothing widths (fixed).
_KF_SHARPNESS_M = 200.0
_KF_RAMP_WIDTH = 0.05
_KF_MIN_ENTRAIN_MULTIPLIER = 0.5
_KF_DETRAIN_BOOST = 1.5
# KF-Eta PROF5 Gaussian-mixing quadrature constants transcribed from WRF
# ``module_cu_kfeta.F`` ``SUBROUTINE PROF5``.
_KF_PROF5_SQRT2P = 2.506628
_KF_PROF5_A1 = 0.4361836
_KF_PROF5_A2 = -0.1201676
_KF_PROF5_A3 = 0.9372980
_KF_PROF5_P = 0.33267
_KF_PROF5_SIGMA = 0.166666667
_KF_PROF5_FE = 0.202765151
_KF_PROF5_EXP_NEG_HALF_3SIGMA_SQ = -4.5
_KF_PROF5_T1 = 0.500498
_KF_PROF5_VERY_BUOYANT_ENV_FRACTION = 0.95
_KF_PROF5_CRITICAL_ENV_FRACTION = 0.10
# KF-Eta initial updraft vertical velocity and condensate loading constants
# transcribed from WRF ``module_cu_kfeta.F``:
#   * lines 1038-1047: DTTOT, GDT=2*g*DTTOT*500/TVEN,
#     WLCL=1+0.5*sqrt(GDT), capped at 3 m/s;
#   * lines 2863-2927: CONDLOAD / Ogura-Cho Eq. 9 fallout with RATE=0.03,
#     60% of fresh condensate participating in conversion and 40% retained.
_KF_DTTOT_MIN_K = 1.0e-4
_KF_WLCL_BUOYANCY_DEPTH_M = 500.0
_KF_WLCL_BASE_M_S = 1.0
_KF_WLCL_SQRT_COEFF = 0.5
_KF_WLCL_MAX_M_S = 3.0
_KF_WTW_DENOM = 1.5
_KF_WTW_MIN = 1.0e-4
_KF_WTW_STOP = 1.0e-3
_KF_WTW_STOP_SHARPNESS = 25.0
_KF_CONDLOAD_RATE = 0.03
_KF_CONDLOAD_FRESH_DRAG_EXCLUDED = 0.2
_KF_CONDLOAD_RATIO_FLOOR = 1.0e-8
# Numerics safety floors for the column condensation water+energy budget
# (converting the CONDLOAD fallout mass flux [kg/m^2/s] to a per-level
# condensation rate [kg/kg/s] via g*precip_flux/dp, and forming the
# non-re-evaporated rain fraction evap_col/precip_col).  Bound only degenerate
# thin layers / vanishing-precip columns; never touched in the active regime.
_KF_DP_FLOOR_PA = 1.0  # coeff-ok: layer-thickness floor, avoids /0 on a null layer
_KF_PRECIP_FLOOR_KG = 1.0e-30  # coeff-ok: precip-column floor for the rain fraction

# KF-Eta downdraft / precipitation-efficiency constants:
#   * lines 1647-1660: start downdraft about 150 hPa above cloud base and
#     require at least 50 hPa of depth;
#   * line 1703: DMFFRC = 2*(1 - RHBAR), Kain (2004) Eq. 11;
#   * lines 1737-1760: relative humidity decreases 20% per km downward.
# The RH clip only bounds pathological supersaturation in the diagnostic mean.
_KF_DD_START_DEPTH_PA = 150.0e2
_KF_DD_MIN_DEPTH_PA = 50.0e2
_KF_DD_PRESSURE_SHARPNESS_PA = 1.0e3
_KF_DD_DMFFRC_COEFF = 2.0
_KF_DD_RH_DECREASE_PER_M = 0.2 / 1000.0
_KF_DD_RH_CLIP_MAX = 1.5


class _CondloadProfile(NamedTuple):
    """KF CONDLOAD-adjusted updraft diagnostics."""

    q_c_u: jax.Array
    M_u: jax.Array
    precip_flux: jax.Array
    w_u: jax.Array
    alive: jax.Array

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


def _kf_initial_wlcl(
    dtlcl: jax.Array,
    dtrh: jax.Array,
    Tv_env_at_lcl: jax.Array,
) -> jax.Array:
    """Initial KF updraft velocity at the LCL [m/s].

    KF-Eta computes ``DTTOT = DTLCL + DTRH`` after the trigger is
    satisfied, then applies the Kain (2004) Eq.-3 buoyant-acceleration
    estimate (WRF ``module_cu_kfeta.F`` lines 1038-1047):

        GDT  = 2*g*DTTOT*500/TVEN
        WLCL = 1 + 0.5*sqrt(GDT), capped at 3 m/s

    with ``WLCL=1`` when ``DTTOT`` is negligible.  The hard branch is kept
    as a JAX piecewise expression; forward values match the reference and
    AD sees finite subgradients away from the documented cutoff.
    """
    dttot = dtlcl + dtrh
    dttot_sqrt = jnp.maximum(dttot, jnp.asarray(_KF_DTTOT_MIN_K, dtype=dttot.dtype))
    gdt = (
        2.0
        * constants.g
        * dttot_sqrt
        * _KF_WLCL_BUOYANCY_DEPTH_M
        / jnp.maximum(Tv_env_at_lcl, 1.0)
    )
    wlcl_buoyant = _KF_WLCL_BASE_M_S + _KF_WLCL_SQRT_COEFF * jnp.sqrt(gdt)
    wlcl = jnp.where(dttot > _KF_DTTOT_MIN_K, wlcl_buoyant, _KF_WLCL_BASE_M_S)
    return jnp.minimum(wlcl, _KF_WLCL_MAX_M_S)


def _kf_condload_profile(
    T_env: jax.Array,
    q_env: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z: jax.Array,
    plume,
    M_b: jax.Array,
    wkl: jax.Array,
    wlcl: jax.Array,
    z_lcl: jax.Array,
    config: KainFritschConfig,
) -> _CondloadProfile:
    """Apply the KF-Eta CONDLOAD vertical-velocity/fallout recursion.

    This is a differentiable column-vector transcription of the reference
    loop in ``module_cu_kfeta.F`` lines 1169-1194 and subroutine
    ``CONDLOAD`` lines 2863-2927.  It updates the updraft vertical kinetic
    energy ``WTW``, removes Ogura-Cho Eq.-9 fallout from plume condensate,
    retains only the documented non-precipitating condensate in the plume,
    and tapers the reported updraft mass flux once loading/buoyancy drives
    ``WTW`` below the KF stopping threshold.

    The shared plume helper does not expose separate liquid/ice or fresh
    condensation increments.  Warm-rain SCM convection uses a liquid-only
    approximation: the fresh increment is the positive layer increase in
    plume condensate after the retained amount from the previous level.
    The CONDLOAD coefficients themselves remain the reference ones.
    """
    ncol, nlev = T_env.shape
    dtype = T_env.dtype
    g = jnp.asarray(constants.g, dtype=dtype)
    dp = (p_half[:, 1:] - p_half[:, :-1]).astype(dtype)
    Tv_env = virtual_temperature(T_env, q_env)
    B_unloaded = virtual_temperature(plume.T_u, plume.q_u) - Tv_env
    rad = _faithful_rad(wkl, config)

    # Surface-first arrays for the upward scan.
    qc_sf = jnp.maximum(plume.q_c_u[:, ::-1], 0.0).astype(dtype)
    M_sf = jnp.maximum(plume.M_u[:, ::-1], 0.0).astype(dtype)
    B_sf = B_unloaded[:, ::-1].astype(dtype)
    Tv_sf = Tv_env[:, ::-1].astype(dtype)
    z_sf = z[:, ::-1].astype(dtype)
    dp_sf = dp[:, ::-1].astype(dtype)
    in_cloud_sf = (z_sf >= z_lcl[:, None]).astype(dtype)

    inputs = (
        jnp.moveaxis(qc_sf, 1, 0),
        jnp.moveaxis(M_sf, 1, 0),
        jnp.moveaxis(B_sf, 1, 0),
        jnp.moveaxis(Tv_sf, 1, 0),
        jnp.moveaxis(z_sf, 1, 0),
        jnp.moveaxis(dp_sf, 1, 0),
        jnp.moveaxis(in_cloud_sf, 1, 0),
    )
    init = (
        (jnp.maximum(wlcl, _KF_WLCL_BASE_M_S) ** 2).astype(dtype),  # WTW
        jnp.zeros((ncol,), dtype=dtype),                  # retained QLIQ
        jnp.maximum(M_b, 0.0).astype(dtype),              # UPOLD
        z_lcl.astype(dtype),                              # previous z
        jnp.ones((ncol,), dtype=dtype),                  # cumulative alive
    )

    def step(carry, layer_inputs):
        wtw, qliq_prev, upold_prev, z_prev, alive_prev = carry
        qc_total, M_layer, B_layer, Tv_layer, z_layer, dp_layer, in_cloud = layer_inputs
        dz = jnp.maximum(z_layer - z_prev, 1.0)

        # Fresh condensate entering CONDLOAD.  The reference separates
        # QNEWLQ/QNEWIC from existing QLIQ/QICE; the shared plume only
        # stores total liquid, so the positive excess over retained liquid
        # is the available fresh source.
        qnew = jnp.maximum(qc_total - qliq_prev, 0.0)
        qtot = jnp.maximum(qliq_prev, 0.0)
        qest = 0.5 * (qtot + qnew)

        be = B_layer / jnp.maximum(Tv_layer, 1.0)
        boterm = 2.0 * dz * g * be / _KF_WTW_DENOM
        rei = (
            jnp.maximum(M_b, 0.0)
            * dp_layer
            * config.entrain_const
            / jnp.maximum(rad, 1.0)
        )
        # KF-Eta's UPOLD is the active updraft mass before the current
        # layer's entrainment/detrainment update; it is not the final
        # buoyancy-tapered reporting mass flux.  Keep the denominator at
        # least VMFLCL (M_b) while computing ENTERM, otherwise a smooth
        # near-zero reporting flux in marginal layers spuriously magnifies
        # entrainment and kills WTW far below the reference cloud top.
        upold_for_enterm = jnp.maximum(upold_prev, jnp.maximum(M_b, 0.0))
        enterm = 2.0 * rei * wtw / jnp.maximum(upold_for_enterm, 1.0e-30)

        g1 = jnp.maximum(
            wtw
            + boterm
            - enterm
            - 2.0 * g * dz * qest / _KF_WTW_DENOM,
            0.0,
        )
        sqrt_floor = jnp.asarray(_KF_WTW_MIN, dtype=dtype)
        wavg = 0.5 * (
            jnp.sqrt(jnp.maximum(wtw, sqrt_floor))
            + jnp.sqrt(jnp.maximum(g1, sqrt_floor))
        )
        conv = _KF_CONDLOAD_RATE * dz / jnp.maximum(wavg, 1.0e-6)

        fresh_retained = jnp.asarray(
            config.condload_fresh_retention_fraction, dtype=dtype
        )
        fresh_participating = 1.0 - fresh_retained
        oldq = qtot + fresh_participating * qnew
        qtot_after = oldq * jnp.exp(-conv)
        dq_fallout = jnp.maximum(oldq - qtot_after, 0.0)
        ratio_liq = (
            fresh_participating * qnew + qliq_prev
        ) / jnp.maximum(oldq, _KF_CONDLOAD_RATIO_FLOOR)
        qout = ratio_liq * dq_fallout

        ppt_drag = 0.5 * (
            oldq
            + qtot_after
            - _KF_CONDLOAD_FRESH_DRAG_EXCLUDED * qnew
        )
        wtw_new = (
            wtw
            + boterm
            - enterm
            - 2.0 * g * dz * ppt_drag / _KF_WTW_DENOM
        )
        # The reference exits the updraft loop when WTW < 1e-3 and floors
        # tiny magnitudes to 1e-4 inside CONDLOAD.  A fixed-length JAX scan
        # must keep carrying a value through inactive upper levels; clamp it
        # non-negative so the stopped plume cannot generate inf-inf in the
        # next layer's entrainment term.
        wtw_new = jnp.maximum(wtw_new, jnp.asarray(_KF_WTW_MIN, dtype=dtype))

        qliq_new = (qtot_after + fresh_retained * qnew).astype(dtype)
        alive_local = jax.nn.sigmoid(
            _KF_WTW_STOP_SHARPNESS * (wtw_new - _KF_WTW_STOP)
            / _KF_WTW_STOP
        ).astype(dtype)
        alive = (alive_prev * alive_local).astype(dtype)
        precip_flux = (qout * jnp.maximum(upold_prev, 0.0) * alive_prev).astype(dtype)
        w_u = (jnp.sqrt(jnp.maximum(wtw_new, sqrt_floor)) * alive).astype(dtype)

        new_carry_active = (
            wtw_new.astype(dtype),
            qliq_new,
            jnp.maximum(M_layer, 0.0),
            z_layer,
            alive,
        )
        new_carry = (
            jnp.where(in_cloud > 0.0, new_carry_active[0], wtw),
            jnp.where(in_cloud > 0.0, new_carry_active[1], qliq_prev),
            jnp.where(in_cloud > 0.0, new_carry_active[2], upold_prev),
            jnp.where(in_cloud > 0.0, new_carry_active[3], z_prev),
            jnp.where(in_cloud > 0.0, new_carry_active[4], alive_prev),
        )
        output = (
            jnp.where(in_cloud > 0.0, qliq_new * alive, 0.0),
            jnp.where(in_cloud > 0.0, M_layer * alive, 0.0),
            jnp.where(in_cloud > 0.0, precip_flux, 0.0),
            jnp.where(in_cloud > 0.0, w_u, 0.0),
            jnp.where(in_cloud > 0.0, alive, 0.0),
        )
        return new_carry, output

    _carry, outputs = jax.lax.scan(step, init, inputs)
    q_loaded_sf, M_loaded_sf, precip_sf, w_sf, alive_sf = outputs
    return _CondloadProfile(
        q_c_u=jnp.moveaxis(q_loaded_sf, 0, 1)[:, ::-1],
        M_u=jnp.moveaxis(M_loaded_sf, 0, 1)[:, ::-1],
        precip_flux=jnp.moveaxis(precip_sf, 0, 1)[:, ::-1],
        w_u=jnp.moveaxis(w_sf, 0, 1)[:, ::-1],
        alive=jnp.moveaxis(alive_sf, 0, 1)[:, ::-1],
    )


def _kf_downdraft_evaporation(
    T_env: jax.Array,
    q_env: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z: jax.Array,
    p_lcl: jax.Array,
    precip_flux: jax.Array,
    branch_weight: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """KF-Eta RH-controlled downdraft evaporation tendency.

    The reference downdraft starts roughly 150 hPa above cloud base,
    computes a pressure-weighted mean RH in that source layer, and scales
    the downdraft by ``DMFFRC = 2*(1-RHBAR)`` (Kain 2004 Eq. 11; WRF
    ``module_cu_kfeta.F`` lines 1647-1703).  The full Fortran routine then
    follows a saturated downdraft with RH decreasing 20%/km and limits
    evaporation by the available precipitation (lines 1737-1822).

    This model has no convective rain tracer, so the local fallout flux
    from CONDLOAD is re-evaporated directly into the sub-cloud layer and
    capped by the local saturation deficit over the explicit physics step.
    The returned tendencies conserve latent energy exactly for the
    evaporated amount: ``dT = -L_v/c_p*dq_v``.
    """
    dtype = T_env.dtype
    dp = p_half[:, 1:] - p_half[:, :-1]
    qsat = saturation_mixing_ratio(T_env, p_full)
    rh = jnp.clip(q_env / jnp.maximum(qsat, 1.0e-12), 0.0, _KF_DD_RH_CLIP_MAX)

    p_source_top = p_lcl[:, None] - _KF_DD_START_DEPTH_PA
    in_source = jax.nn.sigmoid(
        (p_full - p_source_top) / _KF_DD_PRESSURE_SHARPNESS_PA
    ) * jax.nn.sigmoid(
        (p_lcl[:, None] - p_full) / _KF_DD_PRESSURE_SHARPNESS_PA
    )
    source_mass = jnp.sum(in_source * dp, axis=-1)
    rhbar = jnp.sum(in_source * rh * dp, axis=-1) / jnp.maximum(source_mass, 1.0)
    has_depth = jax.nn.sigmoid(
        (source_mass - _KF_DD_MIN_DEPTH_PA) / _KF_DD_PRESSURE_SHARPNESS_PA
    )
    dmffrc = jnp.clip(_KF_DD_DMFFRC_COEFF * (1.0 - rhbar), 0.0, 1.0) * has_depth

    precip_col = jnp.sum(jnp.maximum(precip_flux, 0.0), axis=-1) * branch_weight
    evap_col = precip_col * dmffrc

    below_lcl = jax.nn.sigmoid(
        (p_full - p_lcl[:, None]) / _KF_DD_PRESSURE_SHARPNESS_PA
    )
    z_lcl = jax.vmap(
        lambda pl, p_col, z_col: jnp.interp(pl, p_col, z_col),
    )(p_lcl, p_full, z)
    rh_target = jnp.clip(
        1.0 - _KF_DD_RH_DECREASE_PER_M * jnp.maximum(z_lcl[:, None] - z, 0.0),
        0.0,
        1.0,
    )
    dry_shape = below_lcl * jnp.maximum(rh_target - rh, 0.0)
    fallback_shape = below_lcl * jnp.maximum(1.0 - rh, 0.0)
    shape = jnp.where(
        (jnp.sum(dry_shape * dp, axis=-1) > 0.0)[:, None],
        dry_shape,
        fallback_shape,
    )
    mass_weight = shape * dp / constants.g
    denom = jnp.sum(mass_weight, axis=-1)
    evap_rate = evap_col[:, None] * shape / jnp.maximum(denom[:, None], 1.0e-30)

    dt_safe = jnp.maximum(jnp.asarray(dt, dtype=dtype), 1.0)
    evap_capacity = jnp.maximum(qsat - q_env, 0.0) / dt_safe
    evap_rate = jnp.minimum(evap_rate, evap_capacity)
    dT_evap = -(constants.L_v / constants.c_pd) * evap_rate
    dqv_evap = evap_rate
    return dT_evap, dqv_evap


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
    e45 = jnp.exp(jnp.asarray(_KF_PROF5_EXP_NEG_HALF_3SIGMA_SQ, dtype=eq.dtype))
    t2 = 1.0 / (1.0 + _KF_PROF5_P * jnp.abs(y))
    t1 = jnp.asarray(_KF_PROF5_T1, dtype=eq.dtype)
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
    M_b: jax.Array,
    z: jax.Array,
    k_lnb_smooth: jax.Array,
    config: KainFritschConfig,
) -> tuple[jax.Array, jax.Array]:
    """Separate KF entrainment and detrainment profiles.

    Kain (2004) keeps ``REI = VMFLCL*DP*0.03/RAD`` as the environmental
    inflow scale, but converts UER/UDR back to fractional plume rates by
    dividing by the active updraft mass ``UPOLD``.  Because ``eps_base`` is
    ``REI/(VMFLCL*dz)``, the fractional rate is
    ``eps_base * (VMFLCL/UPOLD) * PROF5``.  The previous implementation
    omitted ``VMFLCL/UPOLD``; once the plume mass grew above cloud base it
    over-entrained/detrained by the mass-growth factor, making the KF column
    too shallow and leaving the cold point too high.
    """
    tv_env = virtual_temperature(T_env, q_env)
    tv_u = virtual_temperature(plume.T_u, plume.q_u) * (
        1.0 - jnp.maximum(plume.q_c_u, 0.0)
    )
    tv95 = _kf_mixed_virtual_temperature(
        T_env, q_env, plume.T_u, plume.q_u, plume.q_c_u, p_full,
        _KF_PROF5_VERY_BUOYANT_ENV_FRACTION,
    )
    tv10 = _kf_mixed_virtual_temperature(
        T_env, q_env, plume.T_u, plume.q_u, plume.q_c_u, p_full,
        _KF_PROF5_CRITICAL_ENV_FRACTION,
    )

    colder = tv_u <= tv_env
    very_buoyant = tv95 > tv_env
    eq = (
        (tv_env - tv_u)
        * _KF_PROF5_CRITICAL_ENV_FRACTION
        / jnp.maximum(tv10 - tv_u, 1.0e-6)
    )
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

    upold_factor = M_b[:, None] / jnp.maximum(
        plume.M_u,
        jnp.maximum(M_b[:, None], 1.0e-30),
    )
    eps_profile = eps_base * ee_mult * upold_factor
    dlt_profile = eps_base * ud_mult * upold_factor

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
    # LFC/LNB on the SAME virtual-T buoyancy as the CAPE above (same
    # parcel-vapor profile ``q_v_parcel``).
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(
        T, T_moist, sharpness=1.0, q_v_env=q_v, q_v_parcel=q_v_parcel,
    )
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
        z_lcl_for_trigger = _interpolate_at_smooth_level(z, k_lcl_smooth)
        T_env_at_lcl = _interpolate_at_smooth_level(T, k_lcl_smooth)
        q_env_at_lcl = _interpolate_at_smooth_level(q_v, k_lcl_smooth)
        w_grid_at_lcl = _interpolate_at_smooth_level(w_grid, k_lcl_smooth)
        # Legacy linear trigger (back-compat for existing tuning).
        wkl = w_grid_at_lcl  # informational; not the FC form
        dtlcl = config.w_thresh_scale * w_grid_at_lcl
        dtrh = jnp.zeros_like(dtlcl)
        T_lcl_perturbed = (
            lcl.T_lcl
            + dtlcl
            - config.w_thresh_offset
        )
    Tv_env_at_lcl = virtual_temperature(T_env_at_lcl, q_env_at_lcl)
    wlcl = _kf_initial_wlcl(dtlcl, dtrh, Tv_env_at_lcl)
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
    # MOIST boundary-layer density: reuse the surface level of the
    # virtual-T column geometry computed above (``compute_column_geometry
    # (..., q_v=q_v)`` → ``rho = p/(R_d·T_v)``).  A dry ``p/(R_d·T)``
    # here overestimated ``rho_BL`` (hence ``M_b``) by ~(1+0.61·q_v) ≈
    # 1-2 % in a humid tropical sub-cloud layer, inconsistent with the
    # module's stated moist-geometry convention.
    rho_BL = rho[:, -1]
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
            filter_negative_buoyancy=False,
        )
        eps_profile, dlt_profile = _kf_buoyancy_sort_rates(
            T, q_v, p_full, predictor_plume, eps_base, M_b,
            z, k_lnb_smooth, config,
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
        filter_negative_buoyancy=not config.faithful_entrainment,
    )

    # -- Cloud depth — z(LCL) → z(LNB) -------------------------------------
    z_lcl = z_lcl_for_trigger
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
    condload = _kf_condload_profile(
        T, q_v, p_full, p_half, z, plume, M_b, wkl, wlcl, z_lcl, config,
    )
    plume = plume._replace(M_u=condload.M_u, q_c_u=condload.q_c_u)

    # -- Environmental tendencies via the shared mass-flux kernel ----------
    # Plume splits vapor (``plume.q_u``) and cloud water
    # (``plume.q_c_u``) explicitly so we use the kernel's correct
    # cloud-water source directly (see ZM).
    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
        z, rho, kernel_delta, M_u_max=config.M_b_max,
        # Conservative flux-form subsidence: the advective donor-cell form
        # leaves a non-telescoping (phi/rho)dM/dz transport residual (measured
        # -2.01e-4 kg/m^2/s on the deep-tropical column) that the column
        # water+MSE budget below would otherwise carry; implicit_flux
        # telescopes it to the vanishing top/base boundary flux.
        subsidence_solve="implicit_flux", p_half=p_half, dt=dt,
    )
    # ``plume.q_c_u`` has already passed through CONDLOAD, so this retained
    # source is the non-precipitating cloud condensate.  The fallout flux
    # ``condload.precip_flux`` feeds the downdraft evaporation loop below.

    # Apply the deep+shallow weight as a per-column scalar.
    dT_dt = dT_dt_raw * branch_weight[:, None]
    dq_v_dt = dq_v_dt_raw * branch_weight[:, None]
    dq_c_conv_dt = dq_c_conv_dt_raw * branch_weight[:, None]

    # -- Release the kernel's DETRAINED-CLOUD latent as warming --------------
    # The implicit_flux kernel books ``dq_v -= dq_c`` (vapor -> detrained cloud)
    # but adds NO sensible heat for that condensation (it defers the latent into
    # the cloud, "released downstream by microphysics").  That leaves column MSE
    # h = c_p T + L_v q_v short by L_v*dq_c on a finite-LNB column where the
    # plume detrains real cloud (measured -549 W/m^2 on the capped column).
    # Release it here so convection is h-CONSERVING in-scheme and matches the
    # ZM/Emanuel convention (dq_c handed to microphysics is already-condensed
    # cloud, latent already released).  Measured: closes vapor-MSE on the capped
    # column to machine precision with NO double-count — the kernel's
    # subsidence/detrainment dT does not already carry this latent.
    dT_dt = dT_dt + (constants.L_v / constants.c_pd) * dq_c_conv_dt

    # Downdraft re-evaporation (POTENTIAL — scaled by the moisture limiter
    # below along with the rest of the CONDLOAD precip cycle).
    dT_dd, dqv_dd = _kf_downdraft_evaporation(
        T, q_v, p_full, p_half, z, lcl.p_lcl,
        condload.precip_flux, branch_weight, dt,
    )

    # -- Convective condensation water+energy budget (close the column) -----
    # CONDLOAD removes ``condload.precip_flux`` [kg/m^2/s] of condensate from the
    # updraft at each level; that water condensed OUT of the column vapor.  Book
    # the condensation EXPLICITLY so the column total-water AND MSE budgets
    # close.  Previously the fallout was re-evaporated by the downdraft as a
    # +dq_v source with NO matching vapor sink or latent warming — water and
    # energy appeared from nowhere, netting a spurious +3.16e-4 kg/m^2/s water
    # source and ~-1500 W/m^2 of phantom cooling (tier-3 validator leak).
    #
    # ``cond_rate = g*precip_flux/dp`` [kg/kg/s] is the per-level condensation
    # rate; its column integral is the net condensate produced.
    #   (a) latent WARM  dT += (L_v/c_p) * cond_rate  at the CONDENSATION levels
    #       (heating aloft — the physical convective Q1).  The kernel only
    #       TRANSPORTS s = c_p T + g z and q_v (MSE-conserving, ~0 NET heating),
    #       so this condensation latent heat is added here exactly once.
    #   (b) vapor SINK  dq_v -= vapor_sink.  The condensed water is drawn from
    #       the vapor the UPDRAFT ingests (boundary-layer + entrained), NOT from
    #       the environment vapor at the (dry) condensation level — debiting
    #       ``cond_rate`` there drives q_v NEGATIVE (the updraft condenses far
    #       more than the local mid-tropospheric q_v holds).  So distribute the
    #       column sink over the POST-TRANSPORT vapor (weight ∝ available
    #       moisture); the heating-aloft / drying-below split is the classic
    #       convective Q1/Q2 structure.
    #   (c) net condensate SOURCE  dq_c += cond_rate * rain_scale (fraction NOT
    #       re-evaporated by the downdraft, >=0), routed to CLOUD water for
    #       microphysics (ZM/Emanuel convention).
    #
    # MOISTURE LIMITER (positivity for ANY dt / dryness): a single Euler step
    # cannot precipitate more than the column holds.  If ``precip_col*dt`` would
    # exceed the post-transport column vapor ``w_col``, scale the WHOLE CONDLOAD
    # cycle (sink, its warming, cloud, AND the downdraft re-evap+cooling) by
    # ``precip_scale = min(1, w_col/(precip_col*dt))`` so every column integral
    # scales together (closure preserved) and ``vapor_sink*dt <= q_v_prelim``
    # holds EXACTLY ⇒ q_v >= 0 unconditionally.  precip_scale=1 (no-op) in the
    # normal regime.  Signs (dp>0 downward, +tend = source): cond_rate>=0 ->
    # +(L_v/c_p)cond_rate WARMS, -vapor_sink DRIES, +cond_rate*scale SOURCES
    # cloud; dqv_dd>=0 re-evap MOISTENS, dT_dd<=0 COOLS.
    dp_col = p_half[:, 1:] - p_half[:, :-1]
    cond_rate = (
        constants.g * jnp.maximum(condload.precip_flux, 0.0)
        / jnp.clip(dp_col, _KF_DP_FLOOR_PA, None)
    ) * branch_weight[:, None]
    precip_col_raw = jnp.sum(cond_rate * dp_col, axis=1, keepdims=True) / constants.g
    # Post-kernel column vapor (re-evap only ADDS vapor, so this is a safe floor
    # for the available moisture the sink can draw on).
    q_v_postk = jnp.clip(q_v + dq_v_dt * dt, 0.0, None)
    w_col = jnp.sum(q_v_postk * dp_col, axis=1, keepdims=True) / constants.g
    precip_scale = jnp.clip(
        w_col / jnp.clip(precip_col_raw * dt, _KF_PRECIP_FLOOR_KG, None),
        0.0, 1.0,
    )
    cond_rate = cond_rate * precip_scale
    dqv_dd = dqv_dd * precip_scale
    dT_dd = dT_dd * precip_scale
    precip_col = precip_col_raw * precip_scale
    evap_col = jnp.sum(dqv_dd * dp_col, axis=1, keepdims=True) / constants.g
    rain_scale = jnp.clip(
        1.0 - evap_col / jnp.clip(precip_col, _KF_PRECIP_FLOOR_KG, None),
        0.0, 1.0,
    )
    # Apply the (scaled) downdraft re-evaporation and the condensation warming.
    dT_dt = dT_dt + dT_dd + (constants.L_v / constants.c_pd) * cond_rate
    dq_v_dt = dq_v_dt + dqv_dd
    # Positivity-safe vapor sink: column magnitude precip_col distributed over
    # the post-transport vapor.  precip_col*dt <= w_col (limiter) ⇒
    # vapor_sink*dt <= q_v_prelim at every level ⇒ q_v stays >= 0.
    q_v_prelim = jnp.clip(q_v + dq_v_dt * dt, 0.0, None)
    w_col2 = jnp.sum(q_v_prelim * dp_col, axis=1, keepdims=True) / constants.g
    vapor_sink = precip_col * q_v_prelim / jnp.clip(w_col2, _KF_PRECIP_FLOOR_KG, None)
    dq_v_dt = dq_v_dt - vapor_sink
    dq_c_conv_dt = dq_c_conv_dt + cond_rate * rain_scale

    # -- Unconditional positivity by CONSERVATIVE VAPOR RELOCATION -----------
    # The shared implicit_flux kernel's compensating-subsidence transport can
    # over-dry a level at extreme dt / very-dry columns (a large-dt property of
    # the kernel itself, NOT the CONDLOAD budget above — the moisture limiter
    # bounds that sink).  But that transport is CONSERVATIVE: the vapor drawn out
    # of an over-dried level was deposited as SURPLUS at other levels, so fill
    # the deficits by draining that surplus in proportion — a pure VAPOR
    # RELOCATION.  It is exactly water-neutral (∫ of the adjustment = 0 whenever
    # the column surplus covers the deficit, which the conservative transport
    # guarantees) and MSE-h-neutral (no phase change, no dT term — just moving
    # q_v between levels), and it keeps q_v >= 0 (deficits -> 0, surplus levels
    # stay >= 0 since scale <= 1).  No-op in the normal regime; water + MSE close
    # to MACHINE PRECISION on any realistic (moist) column.
    # ponytail: a bounded residual (<~4e-5 kg/m^2/s, still >8x below the original
    # leak) survives ONLY when the deficit exceeds BOTH the column vapor surplus
    # and the convective cloud — a physically-degenerate q_v≈0 column where a
    # trigger perturbation drives a plume with no real column moisture to
    # conserve against.  A kernel-level positive-definite transport (shared with
    # EDMF) is the upgrade path if that regime ever matters in practice.
    q_v_new = q_v + dq_v_dt * dt
    deficit = jnp.clip(-q_v_new, 0.0, None)                     # per-level, >= 0
    surplus = jnp.clip(q_v_new, 0.0, None)                      # per-level, >= 0
    deficit_col = jnp.sum(deficit * dp_col, axis=1, keepdims=True) / constants.g
    surplus_col = jnp.sum(surplus * dp_col, axis=1, keepdims=True) / constants.g
    relocate_scale = jnp.clip(
        deficit_col / jnp.clip(surplus_col, _KF_PRECIP_FLOOR_KG, None), 0.0, 1.0
    )
    dq_v_dt = dq_v_dt + (deficit - relocate_scale * surplus) / dt

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
