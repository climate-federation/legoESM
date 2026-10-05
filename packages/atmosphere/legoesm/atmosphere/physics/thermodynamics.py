"""Shared thermodynamic utilities for atmospheric physics.

Provides fundamental thermodynamic functions used across physics
parameterizations (Kessler microphysics, convection, etc.):

1. Saturation mixing ratio (Tetens formula)
2. Temperature-potential temperature conversions
3. Pressure from equation of state
4. Moist adiabatic lapse rate and profiles
5. CAPE computation

All operations are pure JAX and compatible with jit, grad, vmap, scan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants

# Numerical guardrails used across atmosphere dynamics/physics bridges.
_THETA_MIN = 50.0       # [K]
_RHO_MIN = 1.0e-9       # [kg m^-3]
_P_MIN = 1.0            # [Pa]
_P_MAX = 2.0e7          # [Pa]


# ==============================================================================
# Basic thermodynamic relations (extracted from kessler.py)
# ==============================================================================

# Canonical implementations now live in legoesm.thermo so that non-atmosphere
# packages (land, ice, ocean, coupler) can import them without pulling in the
# full atmosphere.physics package.  Re-exported here for backward compatibility.
from legoesm.thermo import saturation_mixing_ratio as saturation_mixing_ratio  # noqa: F401
from legoesm.thermo import saturation_mixing_ratio_ice as saturation_mixing_ratio_ice  # noqa: F401
# Bolton (1980) LCL-temperature formula constants (fixed).
_LCL_T_OFFSET_K = 55.0
_LCL_BOLTON_DENOM = 2840.0



def temperature_from_theta(
    theta: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Recover temperature from potential temperature and pressure.

    T = theta * (p / p_0)^kappa

    Parameters
    ----------
    theta : jax.Array
        Potential temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Temperature [K].
    """
    theta_pos = jnp.clip(theta, _THETA_MIN, None)
    p_pos = jnp.clip(p, _P_MIN, _P_MAX)
    return theta_pos * (p_pos / constants.p_ref) ** constants.kappa


def potential_temperature_from_temperature(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Potential temperature from temperature and pressure — the exact inverse of
    :func:`temperature_from_theta`.

    θ = T * (p_0 / p)^kappa

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Potential temperature [K].
    """
    T_pos = jnp.clip(T, _THETA_MIN, None)
    p_pos = jnp.clip(p, _P_MIN, _P_MAX)
    return T_pos * (constants.p_ref / p_pos) ** constants.kappa


def sanitize_theta_rho(
    theta: jax.Array,
    rho: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Clip thermodynamic state to physically positive ranges.

    Returns
    -------
    theta_pos, rho_pos : jax.Array
        Potential temperature [K] and density [kg/m^3] clipped to
        positive finite floors for robust EOS/exner evaluations.
    """
    theta_pos = jnp.clip(theta, _THETA_MIN, None)
    rho_pos = jnp.clip(rho, _RHO_MIN, None)
    return theta_pos, rho_pos


def pressure_from_eos(
    rho: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    """Compute pressure from density and potential temperature.

    p = p_0 * (R_d * rho * theta / p_0)^(c_p / c_v)

    Parameters
    ----------
    rho : jax.Array
        Density [kg/m^3].
    theta : jax.Array
        Potential temperature [K].

    Returns
    -------
    jax.Array
        Pressure [Pa].
    """
    R_d = constants.R_d
    c_p = constants.c_pd
    c_v = constants.c_vd
    p_0 = constants.p_ref

    theta_pos, rho_pos = sanitize_theta_rho(theta, rho)
    base = jnp.clip(R_d * rho_pos * theta_pos / p_0, 1.0e-20, 1.0e20)
    p = p_0 * base ** (c_p / c_v)
    return jnp.clip(p, _P_MIN, _P_MAX)


def reconstruct_half_level_pressure_hydrostatic(
    p_full: jax.Array,
    rho_full: jax.Array,
    z_half: jax.Array,
) -> jax.Array:
    """Reconstruct interface pressure from full-level state via hydrostatic balance.

    This is intended for non-hydrostatic column physics bridges where
    full-level pressure comes from the local EOS but interface pressure is
    needed by parameterizations. Using the evolving column state avoids
    relying on a fixed reference half-level pressure profile.

    Parameters
    ----------
    p_full : jax.Array
        Full-level pressure [Pa], shape (..., nlev).
    rho_full : jax.Array
        Full-level density [kg/m^3], shape (..., nlev).
    z_half : jax.Array
        Interface height [m], shape (..., nlev+1), top-to-bottom ordering.

    Returns
    -------
    jax.Array
        Reconstructed half-level pressure [Pa], shape (..., nlev+1).
    """
    # Layer thicknesses are positive with top-to-bottom level indexing.
    dz = jnp.abs(z_half[..., :-1] - z_half[..., 1:])
    rho_pos = jnp.clip(rho_full, 1e-9, None)

    # Hydrostatic increment across each full layer.
    dp = constants.g * rho_pos * dz

    # Top interface: centered estimate from top full level.
    p_top = p_full[..., 0] - 0.5 * dp[..., 0]
    p_top = jnp.clip(p_top, 1.0, None)

    # Downward integration to all interfaces.
    p_interfaces_inner = p_top[..., None] + jnp.cumsum(dp, axis=-1)
    p_half = jnp.concatenate([p_top[..., None], p_interfaces_inner], axis=-1)
    return jnp.clip(p_half, 1.0, None)


# ==============================================================================
# Moist thermodynamic functions (for convection)
# ==============================================================================

def moist_adiabat_lapse_rate(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute the moist adiabatic lapse rate dT/dp.

    Gamma_m = (R_d * T / (c_pd * p)) *
              (1 + L_v * q_sat / (R_d * T)) /
              (1 + L_v^2 * q_sat / (c_pd * R_v * T^2))

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Moist adiabatic lapse rate dT/dp [K/Pa].
    """
    R_d = constants.R_d
    c_pd = constants.c_pd
    L_v = constants.L_v
    R_v = constants.R_v

    q_sat = saturation_mixing_ratio(T, p)

    numerator = 1.0 + L_v * q_sat / (R_d * T)
    denominator = 1.0 + L_v ** 2 * q_sat / (c_pd * R_v * T ** 2)

    return (R_d * T / (c_pd * p)) * numerator / denominator


def bolton_lcl_temperature(
    T_base: jax.Array,
    p_base: jax.Array,
    q_v_base: jax.Array,
) -> jax.Array:
    """Bolton (1980) Eq. 22 LCL temperature [K].

    ``T_LCL = 1 / [ 1/(T - 55) - ln(RH)/2840 ] + 55``.

    This is the CANONICAL Bolton LCL implementation — the 55 K offset and
    2840 K denominator live only here.  Consumers:
    :func:`compute_moist_adiabat` (this module) and
    :func:`legoesm.atmosphere.physics.convection._plume.compute_lcl`
    (which adds the plume-specific Poisson ``p_lcl`` and smooth crossing
    index on top).  Do not re-implement the formula elsewhere.

    Parameters
    ----------
    T_base : jax.Array
        Parcel temperature at the launch level [K].
    p_base : jax.Array
        Parcel launch pressure [Pa].
    q_v_base : jax.Array
        Parcel water-vapor mixing ratio at the launch level [kg/kg].
    """
    from legoesm.thermo import saturation_mixing_ratio as _q_sat

    q_sat_base = _q_sat(T_base, p_base)
    RH = jnp.clip(q_v_base / jnp.maximum(q_sat_base, 1.0e-12), 1.0e-4, 1.0)  # coeff-ok: RH floor
    T_minus_55 = jnp.maximum(T_base - _LCL_T_OFFSET_K, 1.0)
    return 1.0 / (1.0 / T_minus_55 - jnp.log(RH) / _LCL_BOLTON_DENOM) + _LCL_T_OFFSET_K


def compute_moist_adiabat(
    T_base: jax.Array,
    p_levels: jax.Array,
    q_v_base: jax.Array | None = None,
    lcl_sigmoid_width_pa: float = 100.0,
) -> jax.Array:
    """Compute parcel temperature profile from surface upward.

    When ``q_v_base`` is not provided the parcel is assumed saturated at
    every level and the routine integrates the moist adiabatic lapse rate
    ``dT/dp = Gamma_m(T, p)`` upward from the base via trapezoidal
    predictor-corrector with :func:`jax.lax.scan`.  This matches the
    historical (pre-audit-2026-05-12) behaviour and is preserved for
    callers that have not yet been migrated to thread launch humidity.

    When ``q_v_base`` is provided the parcel is lifted along a **dry
    adiabat** from the base until reaching the lifting condensation
    level (LCL — Bolton 1980 Eq. 22), then along the **moist adiabat**
    above LCL.  This gives the physically correct parcel curve for
    unsaturated boundary layers — the previous saturated-from-base
    assumption underestimated parcel buoyancy aloft and systematically
    biased CAPE *low* for typical tropical / midlatitude soundings
    where the launch parcel has RH < 100% (audit 2026-05-12 finding
    HIGH #4).

    Parameters
    ----------
    T_base : jax.Array, shape (ncol,)
        Temperature at the lowest level (surface) [K].
    p_levels : jax.Array, shape (ncol, nlev)
        Pressure at full levels [Pa], top-to-bottom ordering.
    q_v_base : jax.Array, shape (ncol,) or None
        Optional water-vapor mixing ratio at the launch level [kg/kg].
    lcl_sigmoid_width_pa : float
        Sigmoid transition width [Pa] for the smooth dry/moist switch
        across the LCL.  Default 100 Pa (~1 hPa) gives a tight
        differentiable transition relative to the ~10⁵ Pa column range.

    Returns
    -------
    jax.Array, shape (ncol, nlev)
        Parcel temperature [K], top-to-bottom ordering.
    """
    ncol, nlev = p_levels.shape

    # Reverse to scan from surface (bottom) upward (top)
    p_rev = p_levels[:, ::-1]  # (ncol, nlev), surface first
    p_base = p_rev[:, 0]

    # Promote to common dtype so scan carry types are consistent.
    # Physical constants in moist_adiabat_lapse_rate are Python float64;
    # computation internally uses the promoted dtype, but the carry must
    # have matching input/output dtypes for jax.lax.scan.
    _dtype = jnp.result_type(T_base, p_rev)
    T_base = T_base.astype(_dtype)
    p_rev = p_rev.astype(_dtype)
    p_base = p_rev[:, 0]

    if q_v_base is None:
        # Saturated-from-base path (legacy callers).  Sentinel: T_lcl
        # equals T_base so the dry branch is never taken inside the
        # scan body and the result reproduces the historical curve.
        T_lcl = T_base
        p_lcl = p_base
    else:
        q_v_base = q_v_base.astype(_dtype)
        T_lcl = bolton_lcl_temperature(T_base, p_base, q_v_base).astype(_dtype)
        # Poisson relation: dry-adiabatic descent (or ascent) between
        # the base and the LCL.
        p_lcl = p_base * (T_lcl / jnp.clip(T_base, 1.0, None)) ** (
            constants.c_pd / constants.R_d
        )
        p_lcl = p_lcl.astype(_dtype)
        # Guard against pathological RH = 1 columns where T_lcl ≈ T_base
        # and p_lcl ≈ p_base (no dry leg).  Numerically harmless.

    # Pre-compute the dry-adiabat constant ``theta = T (p_ref/p)^kappa``
    # at the base — same theta is preserved on the dry leg.
    theta_dry = T_base * (constants.p_ref / jnp.clip(p_base, 1.0, None)) ** constants.kappa
    theta_dry = theta_dry.astype(_dtype)

    def scan_step(carry, p_k):
        T_prev_val, p_prev_val = carry

        # Smooth dry/moist switches.  Pressure decreases upward, so
        # ``p > p_lcl`` ⇔ below LCL.  Transition width is configurable
        # via ``lcl_sigmoid_width_pa`` (default 100 Pa ≈ 1 hPa).
        if q_v_base is None:
            below_lcl_k = jnp.zeros_like(p_k)
            prev_below_lcl = jnp.zeros_like(p_k)
        else:
            below_lcl_k = jax.nn.sigmoid((p_k - p_lcl) / lcl_sigmoid_width_pa)
            prev_below_lcl = jax.nn.sigmoid((p_prev_val - p_lcl) / lcl_sigmoid_width_pa)

        # Straddling steps (previous below LCL, current above) must
        # start the moist integration at the LCL itself rather than
        # the previous (still-dry) level.  Otherwise the moist lapse
        # rate is applied across the entire dry-leg slab and warms
        # the first saturated level — the bias Codex 2026-05-12 P3
        # flagged.  Codex review (2026-05-12) explicitly calls for
        # initialising the moist scan at the diagnosed LCL on
        # cross-LCL steps; that is exactly what ``straddle_weight``
        # interpolates.
        straddle_weight = prev_below_lcl * (1.0 - below_lcl_k)
        # On a straddle, the *moist* starting point is (T_lcl, p_lcl)
        # because the dry leg has carried the parcel from p_prev to
        # p_lcl.  On a pure above-LCL step (straddle_weight ≈ 0) the
        # moist integration starts at the previous level.
        T_moist_start = straddle_weight * T_lcl + (1.0 - straddle_weight) * T_prev_val
        p_moist_start = straddle_weight * p_lcl + (1.0 - straddle_weight) * p_prev_val
        dp_moist = p_k - p_moist_start

        # Moist adiabat trapezoidal predictor-corrector from the
        # (possibly shifted) start to p_k.
        gamma_1 = moist_adiabat_lapse_rate(T_moist_start, p_moist_start)
        T_pred_moist = T_moist_start + gamma_1 * dp_moist
        gamma_2 = moist_adiabat_lapse_rate(T_pred_moist, p_k)
        T_moist = T_moist_start + 0.5 * (gamma_1 + gamma_2) * dp_moist

        # Dry adiabat exact: T = theta_dry * Pi(p), Pi from the canonical
        # Exner helper (physics._shared) rather than an inline power.
        from legoesm.atmosphere.physics._shared import exner_function
        T_dry = theta_dry * exner_function(jnp.clip(p_k, 1.0, None))

        T_new = below_lcl_k * T_dry + (1.0 - below_lcl_k) * T_moist
        T_new = jnp.clip(T_new, 100.0, 350.0).astype(_dtype)  # coeff-ok: physical T clip [K]
        return (T_new, p_k.astype(_dtype)), T_new

    init = (T_base, p_base)
    p_scan = jnp.moveaxis(p_rev[:, 1:], 1, 0)  # (nlev-1, ncol)
    _, T_scan = jax.lax.scan(scan_step, init, p_scan)
    T_scan = jnp.moveaxis(T_scan, 0, 1)  # (ncol, nlev-1)

    T_moist_rev = jnp.concatenate([T_base[:, None], T_scan], axis=1)
    return T_moist_rev[:, ::-1]


def compute_cape(
    T_env: jax.Array,
    T_parcel: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    q_v_env: jax.Array | None = None,
    q_v_parcel: jax.Array | None = None,
    p_source: jax.Array | None = None,
) -> jax.Array:
    """Compute Convective Available Potential Energy (CAPE).

    ``CAPE = R_d * Σ max(0, T_v_parcel - T_v_env) * dp / p_mid``

    where ``dp`` is the layer pressure thickness, ``p_mid`` is the
    half-level midpoint, and the sum is over all levels where the
    parcel is buoyant.  When ``q_v_env`` and ``q_v_parcel`` are
    provided the buoyancy proxy is the **virtual temperature**
    ``T_v = T (1 + (1/ε - 1) q_v)``; otherwise dry temperature is used
    (legacy behaviour, ≈1% bias in tropical columns — audit
    2026-05-12 HIGH #4).

    The parcel virtual-T uses ``q_sat(T_parcel, p)`` above the LCL
    (saturated adiabat) and ``q_v_parcel`` below the LCL.  For
    simplicity we use ``q_v_parcel`` everywhere when provided; for a
    parcel rising along a saturated moist adiabat this is approximate
    above LCL but the dominant CAPE contribution comes from the
    saturated upper troposphere where ``q_v_parcel`` and ``q_sat`` are
    close in the warm-rain regime.  Callers wanting the strict
    saturated-parcel T_v should pass the saturated column from
    :func:`saturation_mixing_ratio`.

    Parameters
    ----------
    T_env, T_parcel : jax.Array, shape (ncol, nlev)
        Environmental and parcel temperatures [K].
    p_full : jax.Array, shape (ncol, nlev)
        Full-level pressure [Pa].
    p_half : jax.Array, shape (ncol, nlev+1)
        Half-level pressure [Pa].
    q_v_env, q_v_parcel : jax.Array, shape (ncol, nlev) or None
        Optional water-vapor mixing ratio profiles [kg/kg].  Pass both
        for the virtual-temperature CAPE.
    p_source : jax.Array, shape (ncol,) or None
        Parcel DEPARTURE-level pressure [Pa].  When given, levels BELOW
        the departure level (``p_full > p_source``) contribute nothing:
        the parcel does not exist there, so any "buoyancy" at those
        levels is an artifact of relaunching an elevated parcel from
        the surface (textbook mean-layer-parcel convention integrates
        from the source level upward).  A STABLE boundary layer makes
        this artifact large: theta increases with height, so the
        PBL-mean parcel translated to surface pressure theta-preserving
        arrives WARMER than the actual surface air (+6.3 K on the
        tier-5 validator column) and the below-departure positive area
        alone reached ~53–66 J/kg fake CAPE — defeating Bechtold's
        launch gate and heating a quiescent column by ~886 W/m² (the
        C24 AMIP bechtold blowup, 2026-07-06).  ``None`` (default) and
        ``p_source = surface pressure`` are byte-identical to the
        legacy all-levels integral, so surface-parcel callers are
        unaffected.

    Returns
    -------
    jax.Array, shape (ncol,)
        CAPE [J/kg].
    """
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
    if q_v_env is not None and q_v_parcel is not None:
        coeff = 1.0 / constants.epsilon - 1.0
        Tv_env = T_env * (1.0 + coeff * q_v_env)
        Tv_parcel = T_parcel * (1.0 + coeff * q_v_parcel)
        buoyancy = jnp.maximum(0.0, Tv_parcel - Tv_env)
    else:
        buoyancy = jnp.maximum(0.0, T_parcel - T_env)

    if p_source is not None:
        # zero out levels below the parcel departure level (p > p_source)
        buoyancy = jnp.where(p_full <= p_source[:, None], buoyancy, 0.0)

    # Use the half-level midpoint pressure for the discrete ``∫ dlnp``
    # approximation: ``(p_half[k+1] - p_half[k]) / p_mid`` with
    # ``p_mid = 0.5 (p_half[k] + p_half[k+1])`` is the canonical
    # convention used by every sister physics helper
    # (``_shared.compute_layer_dz``, ``mass_flux``, ``dca``).  The
    # earlier code used ``p_full``, which on hybrid-sigma grids is a
    # layer-mean pressure (not a half-level midpoint) and produced a
    # 0.5-2 % CAPE bias relative to the moist-adiabat path that
    # consumed it (audit cycle iter-39 finding HIGH #1).
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    return constants.R_d * jnp.sum(buoyancy * dp / p_mid, axis=1)


def parcel_profile_and_cape(
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    q_v: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Lift a surface parcel and return its temperature profile and CAPE.

    Shared parcel -> CAPE recipe for every mass-flux / CAPE-closure convection
    scheme (mass flux, Zhang-McFarlane, Tiedtke).  Centralising the launch
    humidity handling keeps the schemes from drifting apart:

    * ``q_v`` provided — the parcel is lifted **dry-adiabatically below the LCL
      and moist-adiabatically above**, using the lowest-level (surface)
      water-vapor mixing ratio as the launch humidity, and CAPE uses the
      **virtual temperature**.  This is the physically correct trigger for
      unsaturated boundary layers; the legacy saturated-from-base parcel lifts a
      *saturated* parcel from the surface, which spuriously inflates CAPE (and
      fires deep convection) in dry columns and biases buoyancy aloft.
    * ``q_v = None`` — legacy saturated-from-base parcel with dry-T CAPE, kept
      only for callers that genuinely have no humidity to thread.

    The parcel-vapor profile is ``min(q_v_base, q_sat(T_moist, p))`` — exact
    below the LCL (dry-adiabatic ascent conserves mixing ratio) and tracking
    saturation above.  Profiles use the canonical ``(ncol, nlev)`` surface-last
    convention.

    Returns ``(T_moist, cape)`` with shapes ``(ncol, nlev)`` and ``(ncol,)``.
    """
    T_base = T[:, -1]
    q_v_base = None if q_v is None else q_v[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full, q_v_base=q_v_base)
    if q_v is None:
        cape = compute_cape(T, T_moist, p_full, p_half)
    else:
        q_sat_parcel = saturation_mixing_ratio(T_moist, p_full)
        q_v_parcel = jnp.minimum(q_v_base[:, None], q_sat_parcel)
        cape = compute_cape(
            T, T_moist, p_full, p_half,
            q_v_env=q_v, q_v_parcel=q_v_parcel,
        )
    return T_moist, cape
