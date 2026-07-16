"""Bechtold / IFS convection (Bechtold et al. 2008, 2014).

Builds on the Tiedtke 1989 skeleton (see :mod:`.tiedtke`) and adds:

1. **PBL-CAPE / departure-CAPE closure** (Bechtold 2008).  ``M_b`` is
   diagnosed from a mass-weighted parcel within the boundary layer
   rather than the surface parcel.  This sharpens the diurnal cycle
   of deep convection over land.
2. **AR1 stochastic perturbation** (Bechtold 2014).  ``M_b *= (1 +
   amplitude * ε)`` where ``ε`` is an AR1 process with prescribed
   decorrelation timescale.  The AR1 noise state is carried in
   :attr:`legoesm.atmosphere.physics.physics_state.PhysicsState.conv_stoch_state`.
   Stochasticity is OFF by default; when enabled, the leaf takes a
   ``prng_key`` argument.

The smooth-everywhere / differentiability properties are inherited
from Tiedtke; the AR1 stochastic factor is treated as a fixed
multiplier per call so ``jax.grad`` flows through the deterministic
``M_b``.

References
----------
* Bechtold, P., Köhler, M., Jung, T., Doblas-Reyes, F., Leutbecher,
  M., Rodwell, M. J., Vitart, F., & Balsamo, G. (2008). Advances in
  simulating atmospheric variability with the ECMWF model.  *Quart.
  J. Roy. Meteor. Soc.*, 134, 1337–1351.
* Bechtold, P., Semane, N., Lopez, P., Chaboureau, J.-P., Beljaars,
  A., & Bormann, N. (2014). Representing equilibrium and
  nonequilibrium convection in large-scale models.  *J. Atmos. Sci.*,
  71, 734–753.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics._shared import (
    compute_rho,
    exner_function,
    virtual_temperature,
)
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
)
from legoesm.atmosphere.physics.convection.config import BechtoldConfig
from legoesm.atmosphere.physics.convection.output import (
    ConvectionOutput,
    convective_autoconversion_split,
    split_convective_rain,
)
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel,
    stratosphere_mass_flux_gate,
    compute_column_geometry,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
    smooth_level_indicator,
    smooth_positive_part,
    smooth_step,
)
from legoesm.atmosphere.physics.convection._plume import (
    cmt_gregory_1997,
    compute_lcl,
    compute_lfc_lnb,
    entraining_detraining_plume,
)


__all__ = ("bechtold_convection",)


__physics_contract__ = {
    "summary": (
        "Bechtold/IFS mass-flux convection (Tiedtke 1989 skeleton + Bechtold "
        "2008 PBL/departure-CAPE closure + optional 2014 AR1 stochastic "
        "perturbation, RH-dependent downdraft with rain re-evaporation and an "
        "optional penetrative thermodynamic transport of low-MSE air (column "
        "s and q_v conserving), and Gregory-1997 convective momentum transport)."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "u": "m/s", "v": "m/s",
        "conv_prog_profile": "kg/m^2/s (updraft mass-flux carry)",
        "conv_stoch_state": "1 (AR1 noise state)", "dt": "s",
        "moisture_convergence": "kg/kg/s (optional closure enhancement)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_conv_dt": "kg/kg/s",
        "cape": "J/kg", "convective_mask": "1 (0-1 convective indicator)",
        "du_dt_conv": "m/s^2 (None unless CMT enabled)",
        "dv_dt_conv": "m/s^2 (None unless CMT enabled)",
        "dq_r_conv_dt": "kg/kg/s (convective rain source when precip_efficiency>0; else None)",
        "conv_prog_profile_new": "kg/m^2/s (updated mass-flux carry)",
        "conv_stoch_state_new": "1 (updated AR1 noise state)",
    },
    "sign_convention": (
        "Warms and dries the convecting layer via compensating subsidence and "
        "updraft transport (dT_dt, dq_v_dt); the condensed vapor becomes a "
        "non-negative detrained condensate source split by precip_efficiency "
        "into anvil cloud-water (dq_c_conv_dt>=0, handed to microphysics — its "
        "precipitation is DEFERRED to the next step) and in-updraft rain "
        "(dq_r_conv_dt>=0, None when precip_efficiency=0), a falling species the "
        "unified pipeline column-integrates into SAME-STEP surface precipitation "
        "(standalone bridges lacking a precip path route it to the rain tracer or "
        "fold it back into cloud so no water is lost); the optional downdraft cools and moistens the sub-cloud "
        "layer by rain evaporation; the optional CMT drag opposes the "
        "cloud-relative wind shear; surface at the last vertical index. "
        "Column enthalpy/total-water closure is delegated to the orchestrator "
        "rebalance + microphysics (the shared mass-flux kernel is not "
        "self-closing), so no hard conservation is claimed for the raw "
        "tendencies."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Bechtold et al. (2008), QJRMS 134, 1337-1351; Bechtold et al. "
        "(2014), J. Atmos. Sci. 71, 734-753; Tiedtke (1989), Mon. Wea. Rev. "
        "117, 1779-1800"
    ),
    "idealized_test": (
        "A CAPE-positive tropical sounding produces deep convective heating "
        "that stabilizes the column; a stable / zero-CAPE column quiesces "
        "(<1 W/m^2 spurious heating)."
    ),
}


# --- pspec autoblock
# IFS caps RH at 1.0 (MIN(1,q/qsat)) inside the (1.3-RH) entrainment and
# (1.6-RH) detrainment factors (cuascn.F90:510,673), so the factors floor at
# 0.3 / 0.6 in saturated air.  The earlier 1.3 cap let them fall BELOW those
# floors in supersaturated environments (audit F5); 1.0 matches the oracle.
_BECHTOLD_RH_CAP = 1.0
_BECHTOLD_RH_ENTR = 1.3
_BECHTOLD_RH_DETR = 1.6
# Finite/physical guards on the output tendencies fed to the dynamics. Day-4 of
# a C48 AMIP is fully normal, then a rare degenerate column produces a
# non-finite tendency at day 5 -> non-finite winds (the #856 plume rewrite
# exposes this only at C48's finer grid). nan_to_num maps NaN->0 and Inf->finite;
# the clip bounds far above any real convective rate (86 K/day, ~9 g/kg/day) so
# only the numerical spike is removed, real convection untouched.
_BECHTOLD_DTDT_MAX = 1.0e-3     # K/s (~86 K/day)
_BECHTOLD_DQVDT_MAX = 1.0e-4    # kg/kg/s (~9 g/kg/day, well above real conv drying)

# --- Penetrative-downdraft transport numerics (Tiedtke 1989) ---------------
# Softmax "temperature" [J/kg] for the differentiable minimum-MSE (level of
# free sinking) origin: ~ L_v · 1 g/kg, so layers within ~1 g/kg-equivalent of
# the MSE minimum share the origin.  Smooths the argmin; not a tuned closure.
_DD_MSE_SOFT_J_PER_KG = 2500.0
# Sharpness [1/m] of the "below the origin" gate on the downdraft mass-flux
# profile (~300 m transition, so the mass flux is negligible ABOVE the level
# of free sinking — a downdraft must not exist above its own source).
# Numerics only.
_DD_ORIGIN_SHARP_PER_M = 3.0e-3
# Per-column CFL/positivity limiter: the max fraction of a level's vapor the
# transport may remove in one step (and the |dT/dt| cap, _BECHTOLD_DTDT_MAX,
# it is scaled to respect).  A UNIFORM per-column scaling preserves the zero
# column integral => still exactly conservative; it only throttles pathological
# thin-layer / near-dry columns (aggressive-stack blow-up guard).
_DD_CFL_FRAC = 0.5


def _penetrative_downdraft_transport(
    T: jax.Array,
    q_v: jax.Array,
    z: jax.Array,
    dp_full: jax.Array,
    k_lcl_smooth: jax.Array,
    levels_arr: jax.Array,
    M_d_mag: jax.Array,
    entrain_rate: float,
    detrain_scale_m: float,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    r"""Tiedtke-1989 penetrative-downdraft thermodynamic transport.

    The bechtold downdraft branch drives only rain re-evaporation (which
    locally MOISTENS the sub-cloud layer) and CMT momentum — it has no
    mass-flux transport of air, so it can only wet the marine boundary layer.
    This adds the missing transport: a downdraft initiated at the level of
    minimum moist static energy (MSE; the level of free sinking) carries
    low-MSE — i.e. dry, low-``q_v`` — mid-tropospheric air DOWN into the
    sub-cloud layer, DRYING it (a larger sea-air humidity gradient => stronger
    surface evaporation; less BL liquid cloud => lower planetary albedo).

    Conservative environment tendency (Tiedtke 1989, compensated downdraft),
    evaluated in flux form so the column integrals of the dry static energy
    ``s = c_p T + g z`` and of ``q_v`` are conserved to machine precision::

        d(psi_bar)/dt = g * d/dp [ M_d * (psi_d - psi_bar) ]

    Sign convention: ``z`` is geometric height [m], POSITIVE UP; the downdraft
    mass flux ``M_d <= 0`` (downward).  In a humid marine BL fed by a drier
    free troposphere the deposited air has ``q_d < q_bar``, so the sub-cloud
    ``dq_v/dt`` is NEGATIVE (drying).  Surface-last level indexing (larger
    index = lower altitude; index -1 = surface).  The ``M_d -> 0`` boundary
    conditions at BOTH the origin and the surface are what make the flux-form
    column integral vanish.

    Returns
    -------
    (dT_dd, dq_v_dd) : tuple[jax.Array, jax.Array]
        Temperature [K/s] and vapor [(kg/kg)/s] tendencies from the downdraft
        mass-flux transport (add to the environment tendencies).
    """
    g = constants.g
    cpd = constants.c_pd
    lv = constants.L_v

    s = cpd * T + g * z                          # dry static energy [J/kg]
    h = s + lv * q_v                             # moist static energy [J/kg]

    # -- Origin = FREE-TROPOSPHERIC level of minimum MSE (level of free sinking)
    # HARD mask to layers strictly above the LCL: a below-LCL cell gets a
    # -1e30 logit so softmax gives it EXACTLY zero weight regardless of its MSE
    # advantage (a soft/multiplicative mask is penetrable — codex).  The soft
    # argmin over the eligible layers keeps the origin differentiable in the
    # state.  ``source_valid`` is False only for a degenerate column with no
    # layer above the LCL (LCL at the model top); it disables the whole
    # downdraft so ``m_d`` cannot fire on the (then uniform, spurious) origin.
    above_lcl = levels_arr[None, :] < k_lcl_smooth[:, None]      # hard bool mask
    neg_h = -(h - jnp.max(h, axis=1, keepdims=True)) / _DD_MSE_SOFT_J_PER_KG
    masked_neg_h = jnp.where(above_lcl, neg_h, -1.0e30)
    w_org = jax.nn.softmax(masked_neg_h, axis=1)
    source_valid = jnp.any(above_lcl, axis=1, keepdims=True).astype(z.dtype)
    z_org = jnp.sum(w_org * z, axis=1, keepdims=True)     # [ncol, 1]
    s_org = jnp.sum(w_org * s, axis=1, keepdims=True)
    q_org = jnp.sum(w_org * q_v, axis=1, keepdims=True)

    # -- Descending plume: entrainment relaxes it toward the environment -----
    descent = jnp.clip(z_org - z, 0.0, None)             # [ncol, nlev], >=0 m
    f_env = 1.0 - jnp.exp(-entrain_rate * descent)       # mixing fraction
    s_d = s_org * (1.0 - f_env) + s * f_env
    q_d = q_org * (1.0 - f_env) + q_v * f_env

    # -- Downdraft mass flux M_d(k) <= 0, zero AT+ABOVE the origin AND at the
    # surface.  HARD zero for z >= z_org (a downdraft must not exist above its
    # own source — codex), rising with depth below it; then tapering to zero
    # across the sub-cloud layer.  Gated off entirely for a no-source column.
    # ``descent`` = clip(z_org - z, 0, None) >= 0, so the exp argument is <= 0
    # => the (where-)UNSELECTED branch above the origin cannot overflow in fp32
    # (exp(+large) -> inf would poison gradients through z_org — codex).
    below_origin = jnp.where(
        z < z_org,
        1.0 - jnp.exp(-_DD_ORIGIN_SHARP_PER_M * descent),
        0.0,
    )
    z_sfc = z[:, -1:]                                    # surface-last
    surface_taper = 1.0 - jnp.exp(
        -jnp.clip(z - z_sfc, 0.0, None) / jnp.maximum(detrain_scale_m, 1.0)
    )
    shape = below_origin * surface_taper * source_valid
    m_d = -M_d_mag[:, None] * shape                      # [ncol, nlev], <=0

    # Interface support: only interfaces with BOTH adjacent cells below the
    # origin carry flux.  This zeros the origin-STRADDLING interface so NO
    # tendency leaks to the at/above-origin neighbour (the "no downdraft above
    # its own source" invariant, made exact at interfaces — cell-centre m_d=0
    # above the origin is not enough because the 0.5-average g_if straddles it,
    # codex).  Zeroing an INTERIOR interface preserves the telescoping column
    # conservation (both g_below and g_above drop the same term).
    below_mask = z < z_org                               # [ncol, nlev] bool
    iface_below = (below_mask[:, :-1] & below_mask[:, 1:]).astype(z.dtype)

    # -- Conservative flux form: d(psi)/dt = g d/dp[ M_d (psi_d - psi_bar) ] --
    def _transport(psi_d: jax.Array, psi_bar: jax.Array) -> jax.Array:
        excess = m_d * (psi_d - psi_bar)                 # [ncol, nlev]
        # Interface values (between level k and k+1); the model-top and surface
        # boundary interfaces carry zero flux (M_d boundary conditions) and the
        # origin-straddling interior interface is gated to zero, so the column
        # integral telescopes to zero.
        g_if = 0.5 * (excess[:, :-1] + excess[:, 1:]) * iface_below
        zeros = jnp.zeros((excess.shape[0], 1), excess.dtype)
        g_below = jnp.concatenate([g_if, zeros], axis=1)  # flux at k+1/2
        g_above = jnp.concatenate([zeros, g_if], axis=1)  # flux at k-1/2
        return g * (g_below - g_above) / dp_full

    ds_dt = _transport(s_d, s)
    dq_v_dd = _transport(q_d, q_v)
    dt_dd = ds_dt / cpd                                  # z fixed => dT=ds/c_p

    # -- CFL / positivity limiter (conservation-preserving) ------------------
    # Scale the WHOLE-column transport by ONE factor per column so that in one
    # step no level loses more than ``_DD_CFL_FRAC`` of its vapor and |dT/dt|
    # stays under the scheme cap.  A uniform per-column scaling keeps the zero
    # column integral => the transport stays exactly conservative; it only
    # throttles pathological thin-layer / inversion / near-dry columns (the
    # aggressive-stack blow-up guard).  ``r`` is smooth (min/abs/clip) => AD-safe.
    tiny = 1e-30
    drying = jnp.maximum(-dq_v_dd, 0.0)                  # >0 only where drying
    # Only DRYING levels constrain the vapor limiter; a non-drying level (incl.
    # q_v==0 with drying==0) returns a huge value so it never disables a column.
    r_q = jnp.where(
        drying > 0.0, _DD_CFL_FRAC * q_v / (drying * dt + tiny), 1.0e30
    )
    r_t = _BECHTOLD_DTDT_MAX / (jnp.abs(dt_dd) + tiny)
    r = jnp.minimum(
        jnp.min(jnp.minimum(r_q, r_t), axis=1, keepdims=True), 1.0
    )
    return dt_dd * r, dq_v_dd * r


# --- IFS convective-turnover CAPE-closure timescale (cumastrn.F90 / cuascn.F90) ---
# ZTAU = (PGEOH(top)-PGEOH(base))/((2 + min(15, PWMEAN)) * g) * ZTAURES * RTAUA;
# the geopotential-over-g is a height, so g cancels: ZTAU = cloud_depth[m] /
# (2 + w_mean) * ZTAURES * RTAUA.  RTAUA = 1.0 (sucumf.F90:182), matched by
# omission.  ZTAURES is a RESOLUTION factor (cumastrn.F90:762-768):
# 1 + 1.6*dx/125km for dx>=8 km (capped at 3 for dx>125 km) and 1 + log(8km/dx)^2
# for dx<8 km (dx floored at 100 m, cumastrn.F90:713, so this branch tops out at
# 1 + ln(80)^2 ~= 20.2).  ZTAURES >= 1 at every resolution; 1.0 is only its
# infimum (the dx->8 km- limit of the fine branch; at exactly 8 km the coarse
# branch gives 1.1024).  This column-capable scheme is resolution-agnostic (no dx
# plumbed in), so ZTAURES is held at its floor 1.0.  Because the true factor is
# >=1 everywhere, holding it at 1.0 makes the RAW (pre-clamp) tau_conv SHORTER
# (mass flux stronger) than IFS by the ZTAURES factor (up to 3x coarse, ~20x at
# the fine floor); the subsequent [720,10800] s clamp bounds and can even erase
# the difference when both the held and true tau clip to the same bound.  The
# closure STRUCTURE is IFS-faithful; the
# resolution-dependent MAGNITUDE is a DOCUMENTED approximation (a faithful value
# needs dx threaded into convection), so the default is not byte-level IFS at any
# single resolution.  Clamped to [3600/5, 3*3600] = [720, 10800] s
# (cumastrn.F90:827).  PWMEAN is the mean updraught velocity sqrt(2*<PKINEU>)
# (cuascn.F90:845-846), from the updraught kinetic-energy budget PKINEU
# (cuascn.F90:638-654):
#   ZDKBUO = ZFACBUO * B * dz                       (buoyancy KE production)
#   ZDKEN  = min(1, (1+Z_CWDRAG)*mix*dz), mix = eps if eps>0 else delta (cuascn.F90:646)
#   PKINEU(k) = max(-1e3, (PKINEU(k+1)(1-ZDKEN) + ZDKBUO)/(1+ZDKEN))
_IFS_KE_BUOY_FACTOR = 0.5 / (1.0 + 0.5)         # ZFACBUO = 1/3 (cuascn.F90:275)
_IFS_KE_CWDRAG = (3.0 / 8.0) * 0.506 / 0.2      # Z_CWDRAG ~= 0.9488 (cuascn.F90:281)
_IFS_KE_BASE = 0.5                              # PKINEU cloud-base launch 0.5*PWUBASE^2 (cuascn.F90:394); APPROXIMATION: IFS DIAGNOSES PWUBASE (cubasen.F90:636), we fix PWUBASE=1 m/s -> 0.5 m^2/s^2
_IFS_KE_MIN = -1.0e3                            # PKINEU recurrence lower bound MAX(-1.E3,..) (cuascn.F90:654)
_IFS_KE_FLOOR = 1.0e-2                          # mean-KE floor before sqrt (cuascn.F90:845) -> w_mean >= sqrt(0.02) ~ 0.14 m/s
_IFS_WMEAN_MAX = 15.0                           # min(15, PWMEAN) (cumastrn.F90:773)
_IFS_TAU_MIN = 3600.0 / 5.0                     # 720 s  (cumastrn.F90:827)
_IFS_TAU_MAX = 3.0 * 3600.0                     # 10800 s (cumastrn.F90:827)

# --- IFS convective sub-cloud rain evaporation (cuflxn.F90:436-475, sucumf.F90) ---
# Kessler-type evaporation of the convective rain flux below cloud base,
# limited so a layer never evaporates past the RH break ZRHEBC.  Published IFS
# constants (fixed, not tunables):
_IFS_RCPECONS = 5.44e-4 / constants.g   # RCPECONS=5.44E-4/RG (sucumf.F90:176) [Kessler coeff]
_IFS_EVAP_EXPONENT = 0.5777             # (..flux/area)**0.5777 (cuflxn.F90:453)
_IFS_RCVRFACTOR = 5.09e-3               # RCVRFACTOR (sucumf.F90:177) [flux normaliser]
_IFS_RCUCOV = 0.05                      # RCUCOV assumed conv. cloud cover (sucumf.F90:175)
_IFS_RCUCOV_DEEP_FACTOR = 0.6           # deep (KTYPE=1) area = RCUCOV*0.6 (cuflxn.F90:442)
_IFS_RCUCOV_RH_BASE = 0.8               # non-deep area RH enhancement threshold (cuflxn.F90:440)
_IFS_RCUCOV_RH_SLOPE = 1.0 / 0.025      # ...(max(0.8,RHm)-0.8)/0.025 (cuflxn.F90:440)
_IFS_RHEBC_OCEAN = 0.92                 # RHEBC over water (sucumf.F90:179)
_IFS_RHEBC_OCEAN_DEEP = 0.85            # deep KTYPE=1 over water (cuflxn.F90:226)
# Land values (0.75 / 0.70 deep, cuflxn.F90:222-223) need a land mask the leaf
# does not receive — documented gap; the ocean values are used everywhere.
_IFS_EVAP_FLUX_TINY = 1.0e-12           # IF(ZRFL > 1.E-12) evap gate (cuflxn.F90:450)

# --- IFS deep CAPE closure ZMFUB1 = ZCAPE*ZMFUB/(ZHEAT*ZXTAU) (cumastrn.F90:704-833)
_IFS_RETV = 1.0 / constants.epsilon - 1.0       # RETV = R_v/R_d - 1 (yomcst.F90:342)
_IFS_ZCAPE_MAX_PA = 5000.0                      # ZCAPE = MIN(ZCAPE, 5000) (cumastrn.F90:825)
_IFS_ZHEAT_FLOOR = 1.0e-4                       # ZHEAT = MAX(1e-4, ZHEAT) (cumastrn.F90:826)
_IFS_MB_DEEP_FLOOR = 1.0e-3                     # ZMFUB1 = MAX(ZMFUB1, 0.001) (cumastrn.F90:829)


def _ifs_updraft_mean_velocity(
    B_u: jax.Array,
    T_env: jax.Array,
    q_v_env: jax.Array,
    dz: jax.Array,
    eps_profile: jax.Array,
    dlt_profile: jax.Array,
    dp: jax.Array,
    above_base: jax.Array,
    in_cloud: jax.Array,
) -> jax.Array:
    """Mean updraught velocity ``w_mean = sqrt(2*<PKINEU>)`` (IFS cuascn.F90).

    Integrates the IFS updraught kinetic-energy budget (cuascn.F90:638-654)
    from the cloud-base launch KE upward, then takes the in-cloud
    pressure-weighted mean KE and converts it to a velocity (cuascn.F90:845-846),
    floored at the IFS ``1e-2 m^2/s^2`` mean-KE floor (w_mean >= ~0.14 m/s) and
    capped at 15 m/s (cumastrn.F90:773).  The buoyancy source is the plume
    virtual-temperature excess ``B_u`` (M_b-independent, so this is valid for a
    first-guess ascent); the mixing drag uses the fractional entrainment
    ``eps*dz`` wherever entrainment is ACTIVE and substitutes the fractional
    detrainment ``delta*dz`` where entrainment is off — mirroring the IFS switch
    ``IF(ZDMFEN>0) ZDKEN=ZDMFEN ELSE ZDKEN=ZDMFDE`` (cuascn.F90:646-652), keyed on
    entrainment>0, NOT on ``eps < delta``.  This is a SURROGATE, not exact: IFS
    keys on the dynamically-diagnosed ``ZDMFEN`` (which it zeroes after strong
    negative buoyancy, cuascn.F90:666-677), whereas our ``where(eps > 0, ...)``
    keys on the PRESCRIBED ``eps = eps0*(1.3-RH)*f_scale`` that stays > 0 in a
    negatively-buoyant layer — a documented deviation.  ``eps`` is USUALLY > 0 (RH
    capped at 1.0 so ``1.3 - RH >= 0.3``), so the drag is the entrainment rate in
    the common case.  Where ``eps`` reaches exactly 0 — an all-``epsilon_*``-zero
    config, or float32 cancellation in the low-``q_sat`` saturation cap plus the
    ``f_scale`` lower clip (T floored at 150 K, so not a cubed underflow) — it
    takes the delta branch: a finite drag JUMP at ``eps == 0`` (like IFS's own
    discrete switch), ``jax.grad`` still finite (``jnp.where`` selects a branch).

    KE production and drag are gated to at/above cloud base (``above_base``, a
    SIGMOID membership — soft, not a hard mask): a sub-cloud layer sees
    ``zdkbuo, zdken ~= 0`` so ``PKINEU(k) ~= PKINEU(k+1)`` and the cloud-base launch
    KE ``0.5*PWUBASE^2`` (``_IFS_KE_BASE``, PWUBASE~1 m/s) propagates approximately
    unchanged to the base — a smooth analog of IFS launching ``PKINEU`` at
    ``KCBOT`` (cuascn.F90:394) rather than integrating sub-cloud buoyancy into the
    cloud KE.  The soft gate leaks a small residual (bounded by the
    ``_subcloud_insensitive`` test).  AD-safe with finite JAX gradients (floored
    sqrt; the ``eps==0`` drag switch above is a finite-jump ``jnp.where``, not
    mathematically differentiable there).

    Parameters
    ----------
    B_u : (ncol, nlev)  plume virtual-T buoyancy ``T_v_u - T_v_e`` [K], surface-last.
    T_env, q_v_env : (ncol, nlev)  environment T [K] / q_v [kg/kg].
    dz : (ncol, nlev)  layer thickness [m] (KE production uses g*dz).
    eps_profile : (ncol, nlev)  fractional entrainment [1/m] (drag term).
    dlt_profile : (ncol, nlev)  fractional detrainment [1/m] (drag substitute aloft).
    dp : (ncol, nlev)  pressure thickness [Pa] (IFS PWMEAN weight, cuascn.F90:497).
    above_base : (ncol, nlev)  at/above cloud-base membership in [0, 1] (KE gate).
    in_cloud : (ncol, nlev)  in-cloud (base -> top) membership in [0, 1] (mean weight).

    Returns
    -------
    w_mean : (ncol,)  mean updraught velocity [m/s], in [sqrt(0.02), 15].
    """
    # Pin the scan carry to one dtype (geometry/thermo can promote f32 -> f64;
    # lax.scan requires carry-in dtype == carry-out dtype).  See CLAUDE.md
    # jax-scan-carry-dtype-stability.
    _dtype = jnp.result_type(B_u, T_env, q_v_env, dz, eps_profile)
    T_v_e = virtual_temperature(T_env.astype(_dtype), q_v_env.astype(_dtype))
    # Buoyancy acceleration [m/s^2] = g * (T_v_u - T_v_e) / T_v_e.  ZDKBUO in IFS
    # is geoh_diff*ZFACBUO*(B/T_v) = g*dz*ZFACBUO*(B/T_v); the g is explicit here
    # via B_acc while dz carries the layer thickness.
    # PRE-GATE the buoyancy source to at/above cloud base, so the KE-production
    # midpoint 0.5*(b_cur + b_below) can never pull sub-cloud buoyancy across the
    # base boundary into the first in-cloud layer (codex R2 BUG1: b_below still
    # carried sub-cloud buoyancy when only production was gated).  A sub-cloud
    # layer then has b=0 and (via the abv drag gate) zdken=0, so ke=ke_below and
    # the launch KE propagates unchanged to the cloud base.
    B_acc = (
        above_base.astype(_dtype) * constants.g * B_u.astype(_dtype)
        / jnp.maximum(T_v_e, 1.0)
    ).astype(_dtype)
    # March upward: reverse to surface-first (index 0 = surface) for the scan.
    inputs = (
        jnp.moveaxis(B_acc[:, ::-1], 1, 0),
        jnp.moveaxis(eps_profile[:, ::-1].astype(_dtype), 1, 0),
        jnp.moveaxis(dlt_profile[:, ::-1].astype(_dtype), 1, 0),
        jnp.moveaxis(dz[:, ::-1].astype(_dtype), 1, 0),
        jnp.moveaxis(above_base[:, ::-1].astype(_dtype), 1, 0),
    )

    def _ke_step(carry, layer):
        ke_below, b_below = carry
        b_cur, eps_cur, dlt_cur, dz_cur, abv_cur = layer
        # B_acc is pre-gated by above_base (sub-cloud buoyancy excluded from the
        # midpoint); only the drag needs the abv gate so sub-cloud KE propagates
        # unchanged (zdkbuo=zdken=0 => ke=ke_below).
        zdkbuo = _IFS_KE_BUOY_FACTOR * 0.5 * (b_cur + b_below) * dz_cur
        # Mixing drag: use the fractional ENTRAINMENT where it is active and
        # SUBSTITUTE the fractional detrainment where it is off — mirroring the IFS
        # switch ``IF(ZDMFEN>0) eps ELSE delta`` (cuascn.F90:646-652), keyed on
        # entrainment>0, NOT on eps<delta.  SURROGATE, not exact: IFS keys on the
        # DYNAMICALLY-DIAGNOSED mass entrainment ``ZDMFEN``, which it explicitly
        # ZEROES after sufficiently negative buoyancy (cuascn.F90:666-677); our
        # ``eps = eps0*(1.3-RH)*f_scale`` is a PRESCRIBED rate that stays > 0 in a
        # negatively-buoyant layer, so it keeps using eps where IFS would have
        # switched to delta — a documented deviation (we do not model the
        # negative-buoyancy entrainment shutoff).  ``eps`` is USUALLY > 0 (RH
        # capped at 1.0 so ``1.3 - RH >= 0.3``), so the drag is the entrainment
        # rate in the common case.  ``eps`` CAN still reach exactly 0 — a config
        # that zeroes all ``epsilon_*``, or float32 numerical cancellation in the
        # low-``q_sat`` smooth saturation cap followed by the explicit ``f_scale``
        # lower clip (T is floored at 150 K, so this is cancellation, not a cubed
        # underflow) — and there it takes the delta branch: a finite JUMP in the
        # drag coefficient at ``eps == 0`` (like IFS's own discrete switch), with
        # ``jnp.where`` keeping ``jax.grad`` finite (it selects a branch, no 0/0),
        # so the diagnostic ``w_mean`` — feeding a CLAMPED tau, then M_b — stays
        # well defined, just not continuous at eps==0.
        mixing_cur = jnp.where(eps_cur > 0.0, eps_cur, dlt_cur)
        zdken = abv_cur * jnp.minimum(
            1.0, (1.0 + _IFS_KE_CWDRAG) * mixing_cur * dz_cur
        )
        ke = jnp.maximum(
            _IFS_KE_MIN, (ke_below * (1.0 - zdken) + zdkbuo) / (1.0 + zdken)
        ).astype(_dtype)
        return (ke, b_cur), ke

    ncol = B_u.shape[0]
    init = (
        jnp.full((ncol,), _IFS_KE_BASE, dtype=_dtype),
        B_acc[:, ::-1][:, 0].astype(_dtype),
    )
    _, ke_sf = jax.lax.scan(_ke_step, init, inputs)      # (nlev, ncol)
    pkineu = jnp.moveaxis(ke_sf, 0, 1)[:, ::-1]          # surface-last (ncol, nlev)
    # In-cloud pressure-weighted mean KE, floored at the IFS 1e-2 m^2/s^2
    # (=> w_mean >= sqrt(0.02) ~ 0.14 m/s), then sqrt(2*KE), capped at 15 m/s
    # (cuascn.F90:845-846, cumastrn.F90:773).  The KE floor also keeps grad(sqrt)
    # finite at zero KE.
    w = in_cloud * dp.astype(_dtype)
    mean_ke = jnp.sum(w * pkineu, axis=-1) / jnp.clip(jnp.sum(w, axis=-1), 1e-6, None)
    w_mean = jnp.sqrt(2.0 * jnp.maximum(mean_ke, _IFS_KE_FLOOR))
    return jnp.minimum(w_mean, _IFS_WMEAN_MAX)


def _ifs_subcloud_rain_evaporation(
    q_v: jax.Array,
    q_sat_env: jax.Array,
    p_half: jax.Array,
    dp_full: jax.Array,
    dq_r_conv_dt: jax.Array,
    below_lcl: jax.Array,
    rh_cloud_base: jax.Array,
    rh_cloud_top: jax.Array,
    deep_weight: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    r"""IFS Kessler sub-cloud evaporation of convective rain (cuflxn.F90:436-475).

    Oracle recurrence, marched DOWNWARD (both the IFS ``JK`` and our level
    index increase toward the surface) on the accumulated rain flux ``ZRFL``
    [kg/m^2/s], applied below cloud base (``JK >= KCBOT``)::

        ZDRFL1 = RCPECONS * max(0, qsat - q) * A *
                 ( sqrt(p_top(k)/p_sfc) / RCVRFACTOR * ZRFL / A )**0.5777 * dp
        ZRNEW  = ZRFL - ZDRFL1
        ZRMIN  = ZRFL - A * max(0, RHEBC*qsat - q) / (g*dt) * dp
        ZRFLN  = max(max(ZRNEW, ZRMIN), 0)
        evap_k = ZRFL - ZRFLN

    * ``A`` is the convective-rain AREA fraction (cuflxn.F90:436-443): deep
      columns get ``RCUCOV*0.6``; non-deep get the RH-enhanced
      ``RCUCOV*(1 + (max(0.8, RHm) - 0.8)/0.025)`` with ``RHm`` the mean of
      the cloud-base and cloud-top environment RH.  The discrete KTYPE switch
      is blended with the scheme's smooth ``deep_weight`` (same doctrine as
      the CAPE closure's LDCUM/KTYPE analogs).
    * ``ZRMIN`` is the RH-BREAK limiter: evaporation stops once one timestep
      of it would moisten the layer past ``RHEBC*qsat`` (deep-ocean 0.85,
      non-deep-ocean 0.92, deep-weight blended; land values need an absent
      land mask — documented gap).  ``ZCONS2 = 1/(g*dt)`` here is an
      INTENTIONAL departure from the oracle default ``RMFCFL/(g*dt)`` with
      ``RMFCFL = 3`` (sucumf.F90:229-232, active because IFS defaults
      ``RMFSOLTQ = 1``, the implicit mass-flux T/q solver): the factor-3
      headroom lets one evaporation step moisten a layer up to 3x past the
      RH break, which the IFS implicit solve then damps — this scheme
      applies the evap tendency EXPLICITLY (``subsidence_solve`` affects
      only the mass-flux kernel, not this coupling), so ``RMFCFL = 1`` is
      the value that cannot overshoot the break within a step (codex R1 #3).
    * The rain SOURCE profile feeding the flux is the scheme's post-split
      ``dq_r_conv_dt`` (rain forms in the cloud layer and accumulates
      downward; a layer's own source joins the flux BELOW it —
      ``ZRFL(k) = sum_{k'<k} [dq_r(k')*dp(k')/g - evap(k')]``, the
      PMFLXR/PDMFUP bookkeeping).  Evaporation is gated to below the
      (smooth) cloud base by ``below_lcl``; the flux carries through
      un-evaporated above it.
    * IFS applies this to the TOTAL precip flux including snow, with melting
      (RTAUMEL) and the rain/snow phase split — this scheme's convective
      precip has no ice phase, so melt/glaciation are out of scope
      (documented gap).

    Column-water exact by construction: ``evap_k <= ZRFL`` per level (fluxes
    stay >= 0), so the caller can debit ``sum(evap)`` from the rain source
    with a scale in [0, 1].

    AD-safe: the ``**0.5777`` power has an infinite derivative at zero flux,
    so the sub-``1e-12`` flux branch is a guarded double-``where`` (safe
    operand inside, branch select outside — the JAX double-where NaN-grad
    trap); all other ops are max/min/clip.

    Parameters
    ----------
    q_v, q_sat_env : (ncol, nlev)  environment vapor / saturation [kg/kg].
    p_half : (ncol, nlev+1)  half-level pressures [Pa] (surface last).
    dp_full : (ncol, nlev)  layer thickness [Pa].
    dq_r_conv_dt : (ncol, nlev)  post-split convective rain source [kg/kg/s].
    below_lcl : (ncol, nlev)  smooth below-cloud-base membership in [0, 1].
    rh_cloud_base, rh_cloud_top : (ncol,)  environment RH at cloud base/top.
    deep_weight : (ncol,)  deep-class membership in [0, 1].
    dt : float  time step [s].

    Returns
    -------
    (evap_rate, rain_scale) : per-level vapor source [kg/kg/s] and the
        column-uniform factor in [0, 1] that debits the evaporated water
        from ``dq_r_conv_dt``.
    """
    _dtype = jnp.result_type(q_v, q_sat_env, dq_r_conv_dt, below_lcl)
    g = constants.g

    zrhm = 0.5 * (rh_cloud_base + rh_cloud_top)                # (ncol,)
    area_nondeep = _IFS_RCUCOV * (
        1.0
        + (jnp.maximum(_IFS_RCUCOV_RH_BASE, zrhm) - _IFS_RCUCOV_RH_BASE)
        * _IFS_RCUCOV_RH_SLOPE
    )
    area = (
        deep_weight * (_IFS_RCUCOV * _IFS_RCUCOV_DEEP_FACTOR)
        + (1.0 - deep_weight) * area_nondeep
    ).astype(_dtype)                                           # (ncol,)
    rhebc = (
        deep_weight * _IFS_RHEBC_OCEAN_DEEP
        + (1.0 - deep_weight) * _IFS_RHEBC_OCEAN
    ).astype(_dtype)                                           # (ncol,)
    zcons2 = 1.0 / (g * dt)                                    # RMFCFL=1 branch

    src_flux = jnp.maximum(dq_r_conv_dt, 0.0) * dp_full / g    # (ncol, nlev)
    sqrt_p = jnp.sqrt(p_half[:, :-1] / p_half[:, -1:])         # layer-top / sfc

    inputs = tuple(
        jnp.moveaxis(a.astype(_dtype), 1, 0)
        for a in (src_flux, q_sat_env, q_v, sqrt_p, dp_full, below_lcl)
    )

    def _step(flux_top, layer):
        src_k, qsat_k, q_k, sqrtp_k, dp_k, gate_k = layer
        # Oracle ordering (codex R1 #1): evaporation acts on the flux entering
        # the layer TOP (``ZRFL = PMFLXR(JK)``, cuflxn.F90:449); the layer's
        # OWN source joins the flux only downstream (``PMFLXR(JK+1) = ... +
        # ZPDR``, cuflxn.F90:470-472) — rain never re-evaporates in its
        # production layer.
        zrfl = flux_top
        zrfl_safe = jnp.where(zrfl > _IFS_EVAP_FLUX_TINY, zrfl, 1.0)
        zdrfl1 = (
            _IFS_RCPECONS
            * jnp.maximum(qsat_k - q_k, 0.0)
            * area
            * (sqrtp_k / _IFS_RCVRFACTOR * zrfl_safe / area) ** _IFS_EVAP_EXPONENT
            * dp_k
        )
        zrmin = zrfl - area * jnp.maximum(rhebc * qsat_k - q_k, 0.0) * zcons2 * dp_k
        zrfln = jnp.maximum(jnp.maximum(zrfl - zdrfl1, zrmin), 0.0)
        # Smooth below-cloud-base membership: the realized layer outflow is
        # the CONVEX BLEND of the two oracle branches — ``zrfl`` (inactive,
        # above base) and ``zrfln`` (active, below base):
        #   flux_out = (1-g)*zrfl + g*zrfln = zrfl - g*(zrfl - zrfln)
        # with the deposited vapor ``g*(zrfl - zrfln)`` matching it exactly
        # (conservation per level).  At g in {0,1} this IS the oracle
        # recurrence; the fractional band (~2 levels around the smooth LCL)
        # is the scheme's standard AD-compatible membership analog (same
        # doctrine as the deep_weight/LDCUM blends) — codex R1 #2.
        evap_k = jnp.where(
            zrfl > _IFS_EVAP_FLUX_TINY, (zrfl - zrfln) * gate_k, 0.0,
        )
        return (zrfl - evap_k + src_k).astype(_dtype), evap_k

    ncol = q_v.shape[0]
    init = jnp.zeros((ncol,), dtype=_dtype)
    _, evap_sf = jax.lax.scan(_step, init, inputs)             # (nlev, ncol)
    evap = jnp.moveaxis(evap_sf, 0, 1)                         # (ncol, nlev)

    evap_rate = evap * g / dp_full                             # [kg/kg/s]
    evap_total = jnp.sum(evap, axis=-1)
    rain_total = jnp.sum(src_flux, axis=-1)
    rain_scale = jnp.clip(
        1.0 - evap_total / jnp.maximum(rain_total, 1e-30), 0.0, 1.0,
    )
    return evap_rate, rain_scale[:, None]


def _ifs_cape_closure_target(
    T: jax.Array,
    q_v: jax.Array,
    z: jax.Array,
    p_full: jax.Array,
    T_u: jax.Array,
    q_u: jax.Array,
    q_c_u: jax.Array,
    M_u_fg: jax.Array,
    M_d_fg: jax.Array,
    in_cloud: jax.Array,
    tau_conv: jax.Array,
    M_b_fg: jax.Array,
    cape_weight: jax.Array,
) -> jax.Array:
    r"""IFS deep-convection CAPE-closure cloud-base mass flux (cumastrn.F90:704-833).

    Oracle formula (cumastrn.F90:828)::

        ZMFUB1 = ZCAPE * ZMFUB / (ZHEAT * ZXTAU)

    with, summed over the cloud layer ``KCTOP < JK <= KCBOT`` (both the IFS
    ``JK`` and our level index increase DOWNWARD, surface last, so "``k-1``" is
    the level ABOVE ``k`` in both):

    * ``ZCAPE`` (cumastrn.F90:728-733): pressure-integrated FIRST-GUESS-plume
      buoyancy INCLUDING condensate loading,
      ``sum [ (T_u - T)/T + RETV*(q_u - q) - q_c_u ] * (p(k) - p(k-1))``  [Pa],
      with the oracle's FULL-LEVEL pressure spacing ``ZDZ = PAP(k)-PAP(k-1)``
      — NOT the half-level layer thickness, a different measure on a
      stretched vertical grid (codex R1 #1).  Both ``ZCAPE`` and ``ZHEAT``
      need the ``k-1`` neighbor, so BOTH sums run over levels ``1..nlev-1``
      and the model-top level is excluded symmetrically (codex R1 #2: gating
      only ``ZHEAT`` at the top mismatched numerator and denominator).  IFS
      evaluates the environment at HALF levels ``ZTENH/ZQENH``; this smooth
      scheme has no half-level state, so the full-level environment stands in
      — a documented approximation.  Capped at 5000 Pa (cumastrn.F90:825) and
      floored at 0 (the IFS branch runs only where the trigger already found
      deep convection, ``LDCUM & KTYPE==1``; a negative plume integral must
      not drive a negative mass flux here).
    * ``ZHEAT`` (cumastrn.F90:722-727): the stabilization response per layer,
      ``sum max(0, (T_{k-1} - T_k + (phi_{k-1} - phi_k)/c_p)/T_k
      + RETV*(q_{k-1} - q_k)) * g*(M_u(k) + M_d(k))``  [Pa/s],
      evaluated on the FIRST-GUESS mass-flux profiles (``PMFU + PMFD``), i.e.
      the rate at which one unit of convective overturning consumes the
      environment's convective instability.  Floored at 1e-4
      (cumastrn.F90:826); because ``ZHEAT`` scales linearly with the
      first-guess ``M_b_fg`` (``M_u ∝ M_b``), the first guess CANCELS from
      ``ZMFUB1`` whenever the floor is inactive — the closure then depends
      only on the plume SHAPE, not the surrogate first-guess magnitude.
    * ``ZXTAU``: the clamped convective-turnover time ``tau_conv`` already
      diagnosed by the F1 machinery (cumastrn.F90:773,827).

    The deep floor ``MAX(ZMFUB1, 0.001)`` (cumastrn.F90:829) applies in IFS
    only inside the ``LDCUM & KTYPE==1`` (trigger-fired) branch; the smooth
    analog multiplies it by ``cape_weight**2`` (the scheme's quiescence gate,
    same power as the downstream ``M_u`` cap) so a sub-threshold column is not
    handed a 1e-3 kg/m^2/s mass flux by the floor.  The ``MIN(ZMFUB1, ZMFMAX)``
    cap is applied by the caller via ``_ifs_deep_target_scale`` (cap AFTER
    rescale, cumastrn.F90:830-831) against the scheme's ``M_b_max``; IFS's
    ``ZMFMAX = dp_base * RMFCFL/(g*dt)`` CFL form is a documented departure
    (a constant literature cap instead of a timestep-dependent one).

    NOT reproduced (documented gaps — each needs an input this column scheme
    does not have plumbed): the ``RCAPQADV=0.8`` moisture/temperature
    advection correction ``ZCAPE2/ZDQCV/ZSATFR`` (cumastrn.F90:734-760,819-823;
    needs the DYNAMICS T and q advective tendencies and vertical velocity),
    the ``RCAPDCYCL=2`` diurnal-cycle PBL-CAPE subtraction ``ZCAPDCYCL``
    (cumastrn.F90:783-796; needs surface buoyancy flux and a land mask), and
    the ``RMINCAPE=0.05`` floor ``MAX(RMINCAPE*ZCAPE, ZCAPE - ZCAPDCYCL)``
    (cumastrn.F90:820-823) which is inert when ``ZCAPDCYCL = ZDQCV = 0``.

    AD-safe: floors/caps are ``jnp.maximum``/``jnp.clip`` (piecewise-smooth);
    the only division is by the FLOORED ``ZHEAT`` and the clamped ``tau_conv``,
    both bounded away from zero.

    Parameters
    ----------
    T, q_v, z : (ncol, nlev)  environment T [K], vapor [kg/kg], height [m].
    p_full : (ncol, nlev)  full-level pressure [Pa] (the ZCAPE spacing measure).
    T_u, q_u, q_c_u : (ncol, nlev)  first-guess plume T / vapor / condensate.
    M_u_fg, M_d_fg : (ncol, nlev)  first-guess updraft (>=0) and downdraft
        (<=0) mass-flux profiles [kg/m^2/s] (``PMFU``, ``PMFD``).
    in_cloud : (ncol, nlev)  smooth cloud-layer membership in [0, 1].
    tau_conv : (ncol,)  clamped turnover time [s] (``ZXTAU``).
    M_b_fg : (ncol,)  first-guess cloud-base mass flux [kg/m^2/s] (``ZMFUB``).
    cape_weight : (ncol,)  CAPE trigger in [0, 1] (the ``LDCUM`` analog).

    Returns
    -------
    M_b_target : (ncol,)  deep-closure cloud-base mass flux [kg/m^2/s],
        BEFORE the ``M_b_max`` cap (the caller caps after the rescale).
    """
    # ZCAPE: plume buoyancy + condensate loading, weighted by the oracle's
    # FULL-LEVEL pressure spacing ZDZ = PAP(k)-PAP(k-1) (cumastrn.F90:728) and
    # restricted to levels 1..nlev-1 exactly like ZHEAT (both need k-1; the
    # model-top level is excluded from BOTH sums, matching the oracle loop).
    dp_lev = p_full[:, 1:] - p_full[:, :-1]
    buoy = (
        (T_u[:, 1:] - T[:, 1:]) / jnp.maximum(T[:, 1:], 1.0)
        + _IFS_RETV * (q_u[:, 1:] - q_v[:, 1:])
        - q_c_u[:, 1:]
    )
    zcape = jnp.sum(in_cloud[:, 1:] * buoy * dp_lev, axis=-1)
    zcape = jnp.clip(zcape, 0.0, _IFS_ZCAPE_MAX_PA)

    # ZHEAT: environment stability consumption per unit overturning.
    # Neighbor differences k-1 (above) minus k, aligned to levels 1..nlev-1.
    dT_up = T[:, :-1] - T[:, 1:]
    dphi_up = constants.g * (z[:, :-1] - z[:, 1:])
    dq_up = q_v[:, :-1] - q_v[:, 1:]
    stab = jnp.maximum(
        0.0,
        (dT_up + dphi_up / constants.c_pd) / jnp.maximum(T[:, 1:], 1.0)
        + _IFS_RETV * dq_up,
    )
    m_net = constants.g * (M_u_fg[:, 1:] + M_d_fg[:, 1:])
    zheat = jnp.sum(in_cloud[:, 1:] * stab * m_net, axis=-1)
    zheat = jnp.maximum(zheat, _IFS_ZHEAT_FLOOR)

    # ``cape_weight`` multiplies the target as the smooth ``LDCUM`` membership
    # (IFS runs this closure ONLY where the trigger fired; our trigger is a
    # sigmoid, not a bool).  Above the ZHEAT floor the (M_b_fg, M_u_fg) pair
    # cancels, so without this factor a HALF-triggered column would receive
    # the full shape-only target while its realized updraft is capped at
    # cape_weight**2 * M_b_max downstream — the closure M_b (which also
    # drives the downdraft branches) would then disagree with the applied
    # transport across the whole trigger-transition band (codex R3 #1).  A
    # fully-triggered column (cape_weight = 1) is untouched.
    M_b_target = cape_weight * zcape * M_b_fg / (zheat * tau_conv)
    # Deep floor, quiescence-gated (see docstring).
    return jnp.maximum(M_b_target, _IFS_MB_DEEP_FLOOR * cape_weight**2)


def _ifs_deep_target_scale(
    M_b_target: jax.Array,
    M_b_capped: jax.Array,
    deep_weight: jax.Array,
    M_b_max: float,
    divisor_floor: float = 0.0,
) -> jax.Array:
    """Deep-only rescale factor sending ``M_b_capped`` to ``min(M_b_target, M_b_max)``.

    Shared cap-after-rescale + deep-weighted-blend kernel (cumastrn.F90:828-831
    applies ``MIN(ZMFUB1, ZMFMAX)`` AFTER diagnosing the closure target, and
    only for the deep ``KTYPE == 1`` class): for the deep class the returned
    factor turns ``M_b_capped`` into the capped target; for shallow/mid
    (``deep_weight -> 0``) it is exactly 1.  The quiescent ``M_b_capped -> 0``
    edge takes a constant-0 branch with divisor 1, so reverse-mode AD never
    forms ``1 / M_b_capped**2``.
    """
    M_b_target_capped = jnp.minimum(M_b_target, M_b_max)
    # ``divisor_floor`` (safety floor, not a tunable): the CAPE-closure target
    # is NOT proportional to M_b_capped (its deep floor is trigger-gated, not
    # flux-gated), so a subnormal-but-positive M_b_capped would otherwise
    # overflow the ratio to inf in fp32 — and ``deep_weight * inf`` is NaN for
    # the non-deep classes (codex R1 #4).  With the floor the factor is
    # bounded by M_b_max/divisor_floor (finite in fp32) and a sub-floor flux
    # column under-realizes the target toward 0 — physically quiescent anyway.
    # The TURNOVER wrapper passes the 0.0 default (no-op floor): its target IS
    # proportional to M_b (ratio bounded by tau_correction), and flooring
    # there would perturb the legacy default-on path vs main (codex R2 #2).
    divisor = jnp.where(
        M_b_capped > 0.0,
        jnp.maximum(M_b_capped, divisor_floor),
        jnp.ones_like(M_b_capped),
    )
    scale_deep = jnp.where(
        M_b_capped > 0.0, M_b_target_capped / divisor, jnp.zeros_like(M_b_capped),
    )
    return deep_weight * scale_deep + (1.0 - deep_weight)


def _ifs_cape_closure_scale(
    M_b_target: jax.Array,
    M_b_true: jax.Array,
    M_b_launch: jax.Array,
    deep_weight: jax.Array,
    M_b_max: float,
) -> jax.Array:
    """Blend factor sending ``M_b_launch`` to ``d*min(target, max) + (1-d)*M_b_true``.

    The deep share realizes the capped closure target; the non-deep share must
    ride on the TRUE first-guess ``M_b_true``, NOT the floored ``M_b_launch``
    (codex R7): with a hard-zero ``M_b_true`` and partial ``deep_weight = d``,
    blending SCALES around the launch flux lets the restart floor bleed into
    the ``(1-d)`` share — the realized flux came out ``floor*d*(2-d)`` instead
    of the convex ``floor*d`` (a 50% excess at ``d = 0.5``).  Blending the
    FLUXES keeps the restart convex; whenever the launch floor is inactive
    (``M_b_true == M_b_launch``, the generic case) this reduces EXACTLY to the
    ``_ifs_deep_target_scale`` blend.  Same guarded-division AD safety.
    """
    s_deep = _ifs_deep_target_scale(
        M_b_target, M_b_launch, jnp.ones_like(deep_weight), M_b_max,
        divisor_floor=1e-20,
    )
    launch_div = jnp.where(
        M_b_launch > 0.0, jnp.maximum(M_b_launch, 1e-20),
        jnp.ones_like(M_b_launch),
    )
    s_base = jnp.where(
        M_b_launch > 0.0, M_b_true / launch_div, jnp.zeros_like(M_b_launch),
    )
    return deep_weight * s_deep + (1.0 - deep_weight) * s_base


def _ifs_profile_scale_limit(
    mb_scale: jax.Array,
    M_u_profile: jax.Array,
    M_b_max: float,
) -> jax.Array:
    """Column-uniform ZMFS cap limiter (cumastrn.F90:913-932).

    IFS applies ONE closure scale ``ZMFS = ZMFUB1/ZMFUB`` to the whole column
    and then REDUCES it — ``ZMFS = MIN(ZMFS, ZMFMAX/PMFU)`` per level — until
    no level of the scaled profile exceeds its mass-flux cap, keeping a single
    profile-shape-preserving factor.  Without this, an ``mb_scale > 1`` on a
    profile already touching ``M_b_max`` is silently truncated by the
    downstream per-level clips, so ZHEAT anticipates ``s * M_b_max`` transport
    while the kernel realizes only ``M_b_max`` — under-removing CAPE exactly
    in the common saturated-profile case (codex R10).  Levels here share one
    constant cap, so the per-level MIN collapses to the profile peak.  The
    ``1e-20`` guard keeps a dead (all-zero) profile a no-op (limit -> huge,
    MIN inert); AD-safe (min/max are piecewise-smooth, guarded division).
    """
    peak = jnp.max(M_u_profile, axis=-1)
    s_limit = M_b_max / jnp.maximum(peak, 1e-20)
    return jnp.minimum(mb_scale, s_limit)


def _ifs_deep_turnover_scale(
    M_b_uncapped: jax.Array,
    M_b_capped: jax.Array,
    tau_correction: jax.Array,
    deep_weight: jax.Array,
    M_b_max: float,
) -> jax.Array:
    """Deep-only convective-turnover rescale factor for ``M_b`` / ``M_u``.

    Returns ``mb_scale`` such that ``M_b_capped * mb_scale`` equals, for the DEEP
    class, ``min(M_b_uncapped * tau_correction, M_b_max)`` and, for shallow/mid
    (``deep_weight -> 0``), leaves ``M_b_capped`` unchanged.

    Two IFS-faithfulness constraints are encoded here (codex adversarial review):

    * **Cap AFTER rescale** (cumastrn.F90:828-831): IFS diagnoses ``ZMFUB1`` from
      the turnover time and applies ``MIN(ZMFUB1, ZMFMAX)`` afterwards.  Building
      the target from the UNCAPPED flux (not the pre-capped ``M_b_capped``) avoids
      the ``min(M_b_diag, cap) * r`` under-scaling when the uncapped flux exceeds
      the cap and the turnover time lengthens (``r < 1``).
    * **Deep-weighted** (cumastrn.F90:762): IFS guards the turnover CAPE closure
      by the DISCRETE ``KTYPE == 1``; shallow (KTYPE=2) / mid (KTYPE=3) keep their
      own closure.  This scheme replaces IFS's discrete KTYPE switch with the same
      smooth cloud-depth ``deep_weight`` membership it already uses for the
      per-class entrainment/detrainment blend, so the rescale fades off smoothly
      for the non-deep classes (``deep_weight -> 0`` gives an exact no-op) rather
      than switching hard.  A transitional cloud (``0 < deep_weight < 1``) gets a
      correspondingly partial turnover correction — the AD-compatible analog of
      the hard gate, consistent with the rest of the scheme's class blending.

    AD-safe: the quiescent ``M_b_capped -> 0`` edge takes a constant-0 branch with
    divisor 1, so the reverse-mode derivative never forms ``1 / M_b_capped**2``.
    The RAW diagnosed plume ``M_u ∝ M_b`` exactly (plume buoyancy is
    M_b-independent), so the same factor rescales that raw plume ``M_u`` here —
    BEFORE it is blended with the additive prognostic carry in the downstream
    implicit-Euler relaxation (the returned ``M_u_new`` is therefore NOT ∝ M_b when
    a nonzero prior carry is present; the rescale is applied upstream of that).
    """
    return _ifs_deep_target_scale(
        M_b_uncapped * tau_correction, M_b_capped, deep_weight, M_b_max,
    )


def _ifs_cloud_base_qsat(
    q_sat_env: jax.Array,
    k_lcl_smooth: jax.Array,
    levels_arr: jax.Array,
) -> jax.Array:
    """Saturation mixing ratio at the (smooth) cloud base for the entrainment
    height factor ``f_scale = (q_sat / q_sat_base)**3`` (audit F4).

    IFS anchors ``f_scale`` at the cloud base ``PQSEN(JL,IKB)`` with ``IKB=KCBOT``
    (cuascn.F90:674), NOT the lowest (warmer, moister) model level.  Since the
    smooth scheme has no integer ``KCBOT``, we soft-gather ``q_sat`` with a
    Gaussian softmax kernel peaked at the fractional cloud-base index
    ``k_lcl_smooth`` (same kernel/sharpness as the ``z_lcl`` gather), returning a
    value ~= ``q_sat`` at cloud base that exponentially downweights distant
    (e.g. surface) levels — every level keeps a nonzero Gaussian weight, so this
    is negligible sensitivity, not literal independence.

    Returns ``q_sat_base`` of shape ``(ncol, 1)`` (broadcastable over levels).
    """
    base_weight = jax.nn.softmax(
        -2.0 * (levels_arr[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
    )
    return jnp.sum(base_weight * q_sat_env, axis=-1, keepdims=True)


def bechtold_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    u: jax.Array,
    v: jax.Array,
    conv_prog_profile: jax.Array,
    conv_stoch_state: jax.Array,
    prng_key: jax.Array | None,
    dt: float,
    config: BechtoldConfig = BechtoldConfig(),
    moisture_convergence: jax.Array | None = None,
    col_index: jax.Array | None = None,
) -> tuple[ConvectionOutput, jax.Array, jax.Array]:
    """Bechtold/IFS convection (smooth, differentiable).

    Parameters
    ----------
    T, q_v : jax.Array, shape (ncol, nlev)
        Environmental temperature and water-vapor specific humidity.
    p_full, p_half : jax.Array
        Full / half-level pressures.
    u, v : jax.Array, shape (ncol, nlev)
        Environmental winds (for CMT).
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Updraft mass-flux profile from the previous step.
    conv_stoch_state : jax.Array, shape (ncol,)
        AR1 noise state from the previous step.
    prng_key : jax.Array or None
        PRNG key for the stochastic perturbation.  When
        ``config.enable_stochastic`` is ``False`` this argument is
        ignored.  When stochastic is on but ``prng_key`` is ``None``
        the leaf falls back to a deterministic (zero-noise)
        realization.  The convection bridge derives a per-step
        sub-key from ``PhysicsState.prng_key`` (split + ``fold_in``
        with module id ``0xBEC4``) when stochasticity is enabled.
    dt : float
        Time step [s].
    config : BechtoldConfig
    col_index : jax.Array or None, shape (ncol,) int32
        GLOBAL column ids for the decomposition-invariant per-column
        stochastic draw (``PhysicsState.col_index``; a lat-band SPMD
        shard passes its own chunk).  ``None`` falls back to
        ``arange(ncol)`` — identical for any undecomposed caller.

    Returns
    -------
    out : ConvectionOutput
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
        Updated M_u profile (implicit-Euler relaxed).
    conv_stoch_state_new : jax.Array, shape (ncol,)
        Updated AR1 noise state.
    """
    ncol, nlev = T.shape

    # -- Column geometry, moist adiabat, CAPE ------------------------------
    # Pass q_v so dz / rho / z use virtual-temperature moist hydrostatic
    # geometry (~1 % thicker / less dense in tropics) — consistent with
    # the mass-flux closure path (mass_flux.diagnose_mass_flux_closure)
    # and the simplified EDMF entry point.  See clean_physics iter-2 #2.
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]

    # -- PBL parcel: mass-weighted average over the boundary-layer
    # depth.  Smooth weighting via ``smooth_level_indicator`` so the
    # PBL-depth threshold is differentiable.  The mass weight is
    # ``pbl_weight * dp`` (dp/g per layer is mass per unit area) — an
    # earlier form averaged with ``pbl_weight`` alone, which is only
    # correct for uniform-thickness layers and gave a height-weighted,
    # not mass-weighted, mean (audit Codex finding: "Bechtold PBL
    # parcel is not actually mass weighted").
    dp_full = p_half[:, 1:] - p_half[:, :-1]
    pbl_weight = smooth_level_indicator(
        z, threshold=config.cape_pbl_depth, sharpness=2.0e-3,  # coeff-ok: CAPE PBL-depth gate sharpness
        direction="below",
    )                                                       # (ncol, nlev)
    pbl_mass_weight = pbl_weight * dp_full
    # Iter-86: batch the 4 column sums into one stacked reduction so XLA
    # plans a single column-sum sweep instead of 4 separate ones.  Same
    # arithmetic; cleaner code and slightly fewer HLO ops.
    _pbl_sum_stack = jnp.stack(
        [
            pbl_mass_weight,
            pbl_mass_weight * T,
            pbl_mass_weight * q_v,
            pbl_mass_weight * p_full,
        ],
        axis=-1,
    )  # (ncol, nlev, 4)
    _pbl_sums = jnp.sum(_pbl_sum_stack, axis=-2)  # (ncol, 4)
    pbl_norm_val = _pbl_sums[..., 0].clip(1e-6, None)
    T_pbl = _pbl_sums[..., 1] / pbl_norm_val
    q_pbl = _pbl_sums[..., 2] / pbl_norm_val
    # Mass-weighted PBL pressure for the LCL launch level when the
    # parcel comes from the PBL mean (otherwise use surface pressure).
    p_pbl = _pbl_sums[..., 3] / pbl_norm_val
    if config.use_pbl_cape:
        T_parcel_source = T_pbl
        q_parcel_source = q_pbl
        p_parcel_source = p_pbl
    else:
        T_parcel_source = T_base
        q_parcel_source = q_base
        p_parcel_source = p_base

    T_parcel = T_parcel_source + config.parcel_dT
    q_parcel = q_parcel_source + config.parcel_dq

    # Launch the moist adiabat HUMIDITY-AWARE and from the correct pressure
    # origin (the over-firing fix; mirrors the Kain-Fritsch / Zhang-McFarlane
    # treatment).  Bechtold was the only convection scheme still using the
    # legacy saturated-from-base, dry-temperature CAPE that "spuriously
    # inflates CAPE and fires deep convection in dry columns".
    # ``compute_moist_adiabat`` starts its dry leg from the surface full-level
    # pressure ``p_base``, but the parcel lives at ``p_parcel_source``
    # (PBL-mean or base); translate the parcel temperature to its
    # surface-pressure dry-adiabatic equivalent (preserving theta, so the LCL
    # and the moist leg above it are unchanged) and pass ``q_v_base=q_parcel``
    # so the sub-LCL leg is DRY adiabatic, not saturated.  A SINGLE adiabat is
    # used for CAPE, LFC and LNB (a second separate scan tripped an XLA CPU
    # compile abort); consistency is the physically correct choice anyway.
    # Dry-adiabatic (theta-preserving) translation of the parcel temperature
    # from ``p_parcel_source`` to ``p_base``: T(p_base) = theta * Pi(p_base),
    # theta = T_parcel / Pi(p_parcel_source), so the ratio is
    # Pi(p_base)/Pi(p_parcel_source) = (p_base/p_parcel_source)^kappa. Route
    # through the shared Exner helper (no inline (p/p_ref)^kappa power).
    T_parcel_at_sfc = (
        T_parcel
        * exner_function(p_base)
        / exner_function(jnp.maximum(p_parcel_source, 1.0))
    )
    T_moist = compute_moist_adiabat(T_parcel_at_sfc, p_full, q_v_base=q_parcel)
    # Virtual-temperature CAPE: parcel vapour (capped at saturation along the
    # ascent) and environment vapour, so buoyancy uses virtual T, not dry T.
    q_sat_parcel = saturation_mixing_ratio(T_moist, p_full)
    q_v_parcel = jnp.minimum(q_parcel[:, None], q_sat_parcel)
    # Integrate CAPE from the parcel's DEPARTURE level upward only: the
    # theta-preserving surface relaunch makes an elevated (PBL-mean) parcel
    # WARMER than the actual surface air whenever the boundary layer is
    # STABLE (theta increases with height), and the below-departure
    # "buoyancy" is an artifact of a parcel that does not exist there —
    # it alone reached ~53–66 J/kg on the tier-5 stable dry column,
    # defeating the cape_weight² launch gate (886 W/m² spurious heating;
    # the C24 AMIP bechtold blowup).  For a well-mixed convective BL,
    # theta is uniform, the relaunched parcel matches the surface air,
    # and the masked levels contribute ~nothing — convecting columns are
    # essentially unchanged.
    cape_pbl = compute_cape(
        T, T_moist, p_full, p_half,
        q_v_env=q_v, q_v_parcel=q_v_parcel,
        p_source=p_parcel_source,
    )

    cape_weight = cape_trigger(
        cape_pbl, config.cape_threshold, config.cape_sharpness,
    )

    # -- LCL, LFC/LNB ------------------------------------------------------
    lcl = compute_lcl(T_parcel, q_parcel, p_parcel_source, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    # LFC/LNB on the SAME virtual-T buoyancy as the CAPE above (same
    # parcel-vapor profile ``q_v_parcel``).
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(
        T, T_moist, sharpness=1.0, q_v_env=q_v, q_v_parcel=q_v_parcel,
    )

    # Cloud depth.
    levels_arr = jnp.arange(nlev, dtype=T.dtype)
    weight_lcl = jax.nn.softmax(
        -2.0 * (levels_arr[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
    )
    weight_lnb = jax.nn.softmax(
        -2.0 * (levels_arr[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
    )
    # Both reductions share the level axis with weight ``z`` — fuse.
    _z_pair = jnp.sum(
        jnp.stack([weight_lcl, weight_lnb], axis=-1) * z[..., None], axis=-2,
    )
    z_lcl, z_lnb = _z_pair[..., 0], _z_pair[..., 1]
    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)

    # -- Three-class blend -------------------------------------------------
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

    # -- PBL-CAPE closure for cloud-base mass flux -------------------------
    # Bechtold 2008 / 2014 use a hybrid closure: PBL-CAPE drives the
    # baseline mass flux, optionally enhanced where the column is
    # moisture-convergent.  We add the (column-integrated) MC term as
    # a multiplicative enhancement (1 + MC_normalized) so the closure
    # gracefully reduces to pure PBL-CAPE when MC is unavailable
    # (zero-filled by the bridge for spectral PE and other dycores
    # without an MC diagnostic).
    # Generic CAPE-relaxation closure SURROGATE (dimensionally consistent):
    #     M_b = rho_BL * (CAPE_pbl - threshold)+ / (g * tau_bl)   [kg/m^2/s]
    # NOTE: this is NOT the Bechtold (2014) PCAPE/tau buoyancy-sorting closure,
    # nor a closed-form Kain (2004) expression (KF removes CAPE by *iterating*
    # M_b over TIMEC).  It is a first-order CAPE-consumption surrogate; the
    # ``g/rho_BL`` factor stands in for the ZM cloud-work-function sensitivity.
    # The earlier formula omitted ``rho_BL`` and ``g``; magnitude was
    # masked operationally only by ``M_b_max``.
    # Dry boundary-layer density via the shared ideal-gas helper (same
    # 1 K temperature clip as the previous inline form).
    rho_BL = compute_rho(T[:, -1], p_full[:, -1])
    M_b_pbl_cape = (
        cape_weight
        * rho_BL
        * smooth_positive_part(cape_pbl - config.cape_threshold, config.cape_sharpness)
        / (constants.g * config.tau_bl)
    )
    if moisture_convergence is not None:
        column_MC = jnp.sum(
            jnp.maximum(moisture_convergence, 0.0) * dp_full, axis=-1,
        ) / constants.g
        # Normalize the MC term so it acts as an O(1) multiplier.
        # ``mc_normalize_scale`` (default 0.05 kg/m²/s) is a typical
        # strong-convergence value over tropical convective regions
        # (Bechtold 2008 Fig. 2).  Lifted from a literal per CLAUDE.md
        # 'no hardcoded tunables in physics body' (audit B9).
        mc_enhancement = column_MC / config.mc_normalize_scale
        M_b_deterministic = M_b_pbl_cape * (1.0 + mc_enhancement)
    else:
        M_b_deterministic = M_b_pbl_cape

    # -- AR1 stochastic perturbation ---------------------------------------
    if config.enable_stochastic and prng_key is not None:
        alpha_AR1 = jnp.exp(-dt / config.stochastic_decorrelation)
        # Decomposition-INVARIANT draw: fold the per-step sub-key with each
        # column's GLOBAL id and draw one variate per column.  Under
        # lat-band SPMD each shard receives its own contiguous chunk of
        # ``PhysicsState.col_index``, so a given physical column sees the
        # SAME innovation as the serial run — a bulk
        # ``normal(key, (ncol_local,))`` would instead give every band the
        # first ncol_local variates of one stream (decomposition-variant).
        # NOTE: this changes the noise REALIZATION (not the statistics) of
        # serial stochastic runs vs the pre-2026-07 bulk draw.
        _ids = (col_index if col_index is not None
                else jnp.arange(ncol, dtype=jnp.int32))
        innovation = jax.vmap(
            lambda i: jax.random.normal(
                jax.random.fold_in(prng_key, i), dtype=T.dtype)
        )(_ids)
        # Floor the AR(1) innovation-variance sqrt argument at a tiny positive
        # rather than 0: as dt -> 0, alpha_AR1 -> 1 and ``1 - alpha^2 -> 0``,
        # where sqrt'(0) = inf would give a NaN gradient w.r.t. dt /
        # stochastic_decorrelation.  Forward is unchanged for any finite dt.
        conv_stoch_state_new = (
            alpha_AR1 * conv_stoch_state
            + jnp.sqrt(jnp.maximum(1.0 - alpha_AR1 ** 2, 1e-12)) * innovation
        )
        stoch_factor = 1.0 + config.stochastic_amplitude * conv_stoch_state_new
    else:
        # Either stochastic disabled or no PRNG provided — preserve
        # input AR1 state and use deterministic factor 1.
        conv_stoch_state_new = conv_stoch_state
        stoch_factor = jnp.ones_like(M_b_deterministic)

    # Preserve the UNCAPPED diagnostic flux: the IFS turnover closure (F1 below)
    # diagnoses the deep flux from the uncapped M_b and applies M_b_max only AFTER
    # the turnover rescale (cumastrn.F90:828-831), so capping here first would
    # under-scale a column whose uncapped flux exceeds the cap when the turnover
    # time lengthens.  The non-turnover path uses the capped value as before.
    M_b_uncapped = M_b_deterministic * jnp.maximum(stoch_factor, 0.0)
    # See ZhangMcFarlaneConfig.M_b_max.
    M_b = jnp.clip(M_b_uncapped, 0.0, config.M_b_max)
    if config.use_ifs_cape_closure:
        # IFS floors the triggered deep cloud-base flux at 0.001 kg/m^2/s
        # (ZMFUB1 = MAX(..., 0.001), cumastrn.F90:829).  The closure below is a
        # RESCALE of the first-guess plume, so a hard-zero M_b (e.g. the AR1
        # stochastic factor clipping to 0) could never be resurrected by the
        # target floor alone — a scale of zero is zero (codex R5).  Launch the
        # plume from a FLOORED flux (gated by the same smooth memberships the
        # closure floor uses) so a shape exists to rescale — but keep the TRUE
        # (possibly zero) M_b as the closure's ZMFUB: in the oracle a zero
        # first guess zeroes the formula NUMERATOR while the ZHEAT floor holds
        # the denominator, so ZMFUB1 = MAX(0, 0.001) restarts at EXACTLY the
        # floor, not at the full shape-only closure target (codex R6 — feeding
        # the floored flux to both sides resurrected the numerator and let the
        # cancellation promote the restart to the target/cap).  The launch
        # substitution applies ONLY at exact zero (codex R8): a small-but-
        # POSITIVE first guess must launch with M_b itself so (ZMFUB, PMFU)
        # stay the same first guess and the cancellation yields the full
        # shape-only target, as in the oracle — maximum-flooring all small
        # values halved the target at M_b = floor/2.  (An exact-zero flux is
        # what the AR1 clip produces; a sub-1e-20 positive flux instead
        # under-realizes toward 0 via the guarded divisor — physically
        # quiescent either way.)  Feature-gated so the legacy paths stay
        # byte-identical.
        M_b_launch = jnp.where(
            M_b > 0.0, M_b,
            _IFS_MB_DEEP_FLOOR * cape_weight**2 * deep_weight,
        )
    else:
        M_b_launch = M_b

    # -- Per-class entrainment / detrainment profiles (IFS Cy49r1) ---------
    # Uses the nominal IFS bulk-plume RH/height factors (Part IV, Ch. 6;
    # cross-checked against ecmwf-ifs/openifs cuascn/cuentr):
    #   E = ε₀ · f_ε · (1.3 − RH) · f_scale ,   f_scale = (q_sat(T̄)/q_sat(T̄_base))³
    #   D = δ₀ · (1.6 − RH)
    # ENTRAINMENT uses the nominal ALOFT IFS organized-entrainment factors
    # ``ENTRORG·(1.3−RH)·(q_sat/q_sat(KCBOT))³`` (cuascn.F90:672-673).  This is a
    # smooth REIMPLEMENTATION, NOT a port — it omits several IFS mechanisms (a
    # NON-EXHAUSTIVE list): the cloud-base ``(1−RH)`` organized-entrainment
    # initialisation (cuascn.F90:490), IFS's ``JK-1`` RH staggering vs our ``JK``
    # (cuascn.F90:673), IFS ``PQSEN`` = saturation SPECIFIC HUMIDITY vs our
    # ``saturation_mixing_ratio``, the ``ZBUO<=-0.2`` entrainment-zeroing and
    # ``MIN(0.4)`` incremental cap (cuascn.F90:666-677), and the smooth cloud-base
    # soft-gather approximating IFS's integer ``q_sat(KCBOT)`` lookup.
    # DETRAINMENT: the ``(1.6 − RH)`` factor IS faithful — IFS multiplies the
    # in-cloud detrainment ``ZDMFDE`` by ``(1.6 − MIN(1,RH))`` UNIVERSALLY for ALL
    # KTYPEs (deep included), ``JK<KCBOT``, cuascn.F90:510.  Our ``D = δ₀·(1.6−RH)``
    # uses that factor with the ``δ₀`` fractional rate [1/m] standing in for the IFS
    # baseline detrainment RATE ``DETRPEN`` [1/m] (cuentr.F90 forms the increment
    # ``DETRPEN·PMFU·dz``).  This is still a smooth REIMPLEMENTATION, NOT a port; the
    # departures (NON-EXHAUSTIVE) are the per-layer ``0.75·M`` detrainment limiter
    # (cuascn.F90:483), the ``ZMFMAX`` entrainment/detrainment redistribution
    # (cuascn.F90:495+), the negative-buoyancy ``ZOCUDET`` MAX enhancement
    # (``IF(ZBUOC<0)``, cuascn.F90:656-661), the KTYPE>=2 in-cloud ``ZDMFDE=ZDMFEN``
    # tie which we do NOT reproduce (we keep prescribed ``δ_shallow``/``δ_midlevel``,
    # cuascn.F90:500-507), and prescribed-vs-diagnosed rates.  F5's contribution is
    # the RH CAP: our ``MIN(1,RH)`` cap operation matches cuascn.F90:510's
    # ``MIN(1,·)``, though IFS applies it to specific humidity ``PQEN/PQSEN`` while
    # we use ``q_v/saturation_mixing_ratio``.
    # IFS ties shallow detrainment to the shallow entrainment
    # (``D_shallow = E_shallow·(1.6 − RH)``); we keep the simpler
    # prescribed-δ₀ shallow form because the entrainment-tied variant
    # regressed the column MSE budget on shallow-weighted columns (see the
    # ``dlt_profile`` comment).  The (1.6 − RH) RH factor is applied to
    # every class.
    # The fractional rates [1/m] passed to the plume are the bracketed
    # height factors times the per-class base rates ``ε₀``/``δ₀``.  The two
    # physical ingredients the earlier CONSTANT profiles were missing — and
    # the cause of the deep-plume over-dilution / cold RCE collapse — are:
    #   (1) the RH factor ``(1.3 − RH)`` (dry environments entrain more,
    #       moist ones less), and
    #   (2) the vertical scaling ``f_scale = (q_sat/q_sat_base)³`` which
    #       decays strongly with height (q_sat drops as the column cools),
    #       so entrainment is large near cloud base and →0 aloft.
    # Because δ₀ has no f_scale, aloft ``δ > ε`` and the mass flux turns
    # over — reproducing the IFS bell-shaped M(z) that peaks in the lower
    # troposphere and detrains near cloud top instead of diluting the
    # updraught to neutral buoyancy in the lower troposphere.
    q_sat_env = saturation_mixing_ratio(T, p_full)               # (ncol, nlev)
    # RH capped at 1.0 (_BECHTOLD_RH_CAP) to match IFS MIN(1,q/qsat) in the
    # (1.3-RH)/(1.6-RH) factors below (audit F5; cuascn.F90:510,673).
    RH = jnp.clip(q_v / jnp.maximum(q_sat_env, 1e-12), 0.0, _BECHTOLD_RH_CAP)
    # IFS scales the organized entrainment by (qs(z)/qs(cloud_base))**3 with the
    # base saturation taken AT CLOUD BASE (cuascn.F90:674 uses PQSEN(JL,IKB),
    # IKB=KCBOT), NOT the lowest model level (audit F4).  Soft-gather q_sat at
    # the smooth cloud-base index k_lcl_smooth (same softmax kernel/sharpness as
    # z_lcl above) so f_scale decays relative to the (higher, cooler) cloud base as
    # in the oracle — a smooth APPROXIMATION of IFS's integer PQSEN(KCBOT) lookup —
    # instead of from the warmer surface level.
    q_sat_base = _ifs_cloud_base_qsat(q_sat_env, k_lcl_smooth, levels_arr)
    f_scale = jnp.clip(q_sat_env / jnp.maximum(q_sat_base, 1e-12), 0.0, 1.0) ** 3
    rh_entr = jnp.clip(_BECHTOLD_RH_ENTR - RH, 0.0, None)                       # eq 6.7
    rh_detr = jnp.clip(_BECHTOLD_RH_DETR - RH, 0.0, None)                       # eq 6.8 / 6.9
    entr_factor = rh_entr * f_scale                              # (ncol, nlev)
    # Per-class entrainment ε₀ (shallow carries the f_ε=2 factor in its
    # config default); every class entrains with the same height factor
    # ``entr_factor = (1.3 − RH)·f_scale`` so ``E_class = ε₀_class ·
    # entr_factor`` (the deep branch — what RCE selects — uses the nominal aloft
    # IFS organized-entrainment factors cuascn.F90:672-673, with the documented
    # departures listed in the header block above).
    eps_per_class = (
        deep_weight[:, None] * config.epsilon_deep
        + shallow_weight[:, None] * config.epsilon_shallow
        + midlevel_weight[:, None] * config.epsilon_midlevel
    )
    eps_profile = eps_per_class * entr_factor

    # Detrainment uses the per-class δ₀ scaled by the ``(1.6 − RH)`` factor for ALL
    # classes — matching IFS's universal in-cloud ``ZDMFDE·(1.6−MIN(1,RH))``
    # (cuascn.F90:510, all KTYPEs); ``δ₀`` stands in for the IFS baseline rate.  The
    # departures (0.75·M limiter, ZMFMAX redistribution, negative-buoyancy ZOCUDET,
    # prescribed-vs-diagnosed) are listed in the header block above.
    # IFS additionally ties the *shallow*
    # detrainment to the shallow entrainment (``D_shallow =
    # E_shallow·(1.6 − RH)``); we keep the simpler prescribed-δ₀ shallow
    # form here on purpose: tying shallow detrainment to its (large)
    # entrainment regressed the column MSE budget from ~21 % to ~88 % on
    # shallow-weighted columns because the mass-flux kernel's
    # compensating-subsidence term does not balance that large a
    # detrainment on coarse grids (Codex adversarial review,
    # IFS-faithfulness iter-2/iter-3).  The shallow-branch deviation is a
    # documented faithfulness gap; conservation takes precedence (project
    # rule: mass/energy budgets are hard invariants).  ``dlt_profile`` is
    # reused for the environmental detrainment kernel below so the plume
    # mass budget and the detrained-property tendencies stay consistent.
    dlt_per_class = (
        deep_weight[:, None] * config.delta_deep
        + shallow_weight[:, None] * config.delta_shallow
        + midlevel_weight[:, None] * config.delta_midlevel
    )
    dlt_profile = dlt_per_class * rh_detr

    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_lcl_smooth,
        eps_profile, dlt_profile, M_b_launch,
        buoyancy_death_memory=config.buoyancy_death_memory,
    )

    # -- IFS convective-turnover CAPE-closure timescale (audit F1) ---------
    # The legacy closure divided PBL-CAPE by a FIXED ``tau_bl``; IFS
    # (cumastrn.F90:773,828) removes CAPE over a STATE-DEPENDENT convective
    # turnover time ``tau_conv = cloud_depth / (2 + w_mean)`` (clamped
    # [720, 10800] s), where ``w_mean`` is the mean updraught velocity from the
    # plume kinetic-energy budget (``_ifs_updraft_mean_velocity``, cuascn.F90).
    # The plume buoyancy ``B_u`` and the T/q profiles are INDEPENDENT of ``M_b``
    # (M_b only scales ``M_u`` linearly), so the first-guess ascent above already
    # yields the correct ``w_mean``.  We therefore rescale the cloud-base mass
    # flux — and, by exact linearity, ``M_u`` — from the UNCAPPED diagnostic flux,
    # mirroring the IFS "first-guess ascent -> PWMEAN -> rescale M_b -> cap"
    # sequence without a second plume evaluation.  Static-bool feature gate (NOT a
    # traced branch); ``False`` restores the byte-identical fixed-``tau_bl``
    # closure.  The downstream ``cape_weight**2 * M_b_max`` cap still gates
    # quiescence, so a zero-CAPE column (``cape_weight ~ 0``) stays quiescent.
    #
    # DEEP-WEIGHTED (codex R5 finding-2): the IFS turnover CAPE closure is guarded
    # by the DISCRETE ``KTYPE == 1`` (deep) — ``ZTAU`` and ``ZMFUB1`` live inside
    # that branch (cumastrn.F90:762,815); shallow (KTYPE=2) and mid (KTYPE=3) keep
    # their own closure.  We replace that hard switch with the scheme's existing
    # smooth ``deep_weight`` cloud-depth membership (same one blending the
    # per-class entrainment): ``deep_weight -> 0`` is an exact no-op, a
    # transitional cloud gets a partial correction — the AD-compatible analog of
    # the discrete gate, consistent with the rest of the scheme's class blends.
    #
    # CAP AFTER RESCALE (codex R5 finding-1): IFS diagnoses ``ZMFUB1`` from the
    # turnover time, THEN ``MIN(ZMFUB1, ZMFMAX)`` (cumastrn.F90:828-831).  Capping
    # M_b to M_b_max *before* the rescale (as the shared clip above does for the
    # non-turnover path) would compute ``min(M_b_diag, cap)*r`` instead of the
    # required ``min(M_b_diag*r, cap)`` — under-scaling a column whose uncapped
    # flux exceeds the cap when ``r < 1``.  We therefore build the deep target from
    # ``M_b_uncapped`` and cap once, then express the change as a scale on the
    # plume's pre-cap ``M_u`` (``M_u ∝ M_b`` exactly).
    if config.use_ifs_cape_closure or config.use_convective_turnover_tau:
        # SHARP (4/level) cloud-base and cloud-top gates so the KE mean is taken
        # over an essentially exclusive [LNB, LCL] cloud window; a broad
        # unit-sharpness sigmoid leaked sub-cloud / above-top KE into w_mean
        # (codex R2).  4.0 is an exempt math constant.
        above_base = jax.nn.sigmoid(4.0 * (k_lcl_smooth[:, None] - levels_arr[None, :]))
        in_cloud = above_base * jax.nn.sigmoid(
            4.0 * (levels_arr[None, :] - k_lnb_smooth[:, None])
        )
        w_mean = _ifs_updraft_mean_velocity(
            plume.B_u, T, q_v, dz, eps_profile, dlt_profile, dp_full,
            above_base, in_cloud,
        )
        tau_conv = jnp.clip(
            cloud_depth / (2.0 + w_mean), _IFS_TAU_MIN, _IFS_TAU_MAX,
        )
    if config.use_ifs_cape_closure:
        # -- Full IFS deep CAPE closure ZMFUB1 = ZCAPE*ZMFUB/(ZHEAT*ZXTAU)
        # (cumastrn.F90:704-833; see _ifs_cape_closure_target).  Supersedes the
        # tau-only turnover rescale below (the turnover time is one FACTOR of
        # this closure); deep-weighted + cap-after-rescale exactly like it.
        # ZHEAT downdraft first guess: IFS includes PMFD from the LFS-initiated
        # cuddrafn descent (cudlfsn.F90/cuddrafn.F90).  This scheme's downdraft
        # branch applies NO mass-flux transport of environment air (it drives
        # rain re-evaporation only), so the self-consistent PMFD analog here is
        # ZERO — an invented -alpha*M_u*trigger profile would put stabilization
        # into ZHEAT that the scheme never applies (codex R1 #3).  Omitting the
        # (negative) PMFD makes ZHEAT at most ~RMFDEPS = 30% larger, i.e. the
        # closure at most ~30% WEAKER than IFS on downdraft-active columns — a
        # documented gap until a faithful cuddrafn-shaped descent exists (the
        # opt-in penetrative transport's m_d profile is the natural donor).
        M_d_fg = jnp.zeros_like(plume.M_u)
        # ZMFUB and PMFU describe the SAME first guess (codex R2 #1): the
        # plume was integrated from the CAPPED M_b_launch, and IFS itself
        # builds the deep first-guess ZMFUB at/below ZMFMAX before the closure
        # (cumastrn.F90:544-563: 0.1*ZMFMAX or MIN(ZDHPBL/ZDH, ZMFMAX)).
        # Passing the uncapped flux would inflate the target by
        # M_b_uncapped/M_b whenever the cap is active (ZHEAT ∝ the capped
        # plume).  Above the ZHEAT floor the pair cancels, so the target is
        # shape-only either way — the pairing matters exactly when the cap or
        # floor is active.  Cap-after-rescale is preserved by the final
        # min(target, M_b_max) inside _ifs_deep_target_scale.
        # ONE deliberate asymmetry (codex R6): the numerator ZMFUB is the TRUE
        # M_b, not the floored M_b_launch the plume integrated — so a
        # hard-zero first guess zeroes the numerator and the target lands
        # exactly on the MAX(., 0.001) restart floor, as in the oracle (where
        # ZHEAT's own floor holds the denominator).  Whenever M_b >= the
        # launch floor the two are identical and the pair is exactly
        # consistent.
        # ZHEAT must measure the flux the kernel ACTUALLY applies (codex R3 #2,
        # R4 #1/#2): the realized transport is (a) gated TWICE at p_conv_top_pa
        # — once on the carry, once inside apply_mass_flux_kernel — so the
        # first-guess profile carries the SQUARED gate, and (b) clipped per
        # level at M_b_max by the kernel (M_u_max), mirrored here with the same
        # min.  IFS's own first-guess ascent is per-level ZMFMAX-limited
        # (cuascn.F90 ZMFMAX redistribution), so the mirror is also
        # oracle-shaped.  Residual: the carry's cape_weight**2 * M_b_max clip
        # is NOT mirrored (a trigger-transition-band effect already faded by
        # the cape_weight factor on the target).  ZCAPE keeps the ungated
        # in_cloud window: it measures the instability that EXISTS (the oracle
        # integrates the real cloud extent), not the transport.
        _top_gate = stratosphere_mass_flux_gate(p_full, config.p_conv_top_pa)
        M_u_capped_fg = jnp.minimum(plume.M_u, config.M_b_max)
        M_u_closure_fg = M_u_capped_fg * _top_gate**2
        M_b_target = _ifs_cape_closure_target(
            T, q_v, z, p_full,
            plume.T_u, plume.q_u, plume.q_c_u, M_u_closure_fg, M_d_fg,
            in_cloud, tau_conv, M_b, cape_weight,
        )
        # The rescale divides by the flux the plume was ACTUALLY launched with
        # (M_b_launch), so M_u * mb_scale realizes the target profile; the
        # downstream M_b (driving the downdraft branches) is the same product.
        # The non-deep blend share rides on the TRUE M_b (see
        # _ifs_cape_closure_scale — codex R7).
        mb_scale = _ifs_cape_closure_scale(
            M_b_target, M_b, M_b_launch, deep_weight, config.M_b_max,
        )
        # Column-uniform ZMFS limiter (cumastrn.F90:913-932; codex R10): shrink
        # the ONE closure scale until no level of the capped profile exceeds
        # M_b_max, so an s > 1 request is realized exactly (never silently
        # truncated by the downstream per-level clips ZHEAT knows nothing of).
        mb_scale = _ifs_profile_scale_limit(
            mb_scale, M_u_capped_fg, config.M_b_max,
        )
        M_b = M_b_launch * mb_scale
        # Rescale the SAME per-level-capped profile ZHEAT diagnosed (codex R9:
        # cap and rescale do not commute — rescaling the RAW plume lets a
        # level pinned at M_b_max ignore a requested reduction s < 1, because
        # min(s*A, M_b_max) stays at the cap for A >> M_b_max while ZHEAT
        # assumed s*M_b_max).  The downstream carry/kernel top gates then act
        # on this exactly as ZHEAT's gate**2 assumed.  The raw-profile rescale
        # remains the legacy turnover path's behavior below (pre-existing,
        # feature-gated apart).
        plume = plume._replace(M_u=M_u_capped_fg * mb_scale[:, None])
    elif config.use_convective_turnover_tau:
        tau_correction = config.tau_bl / tau_conv          # M_b is proportional to 1/tau
        # Deep-only, cap-after-rescale scale on the pre-cap M_b (and M_u ∝ M_b).
        mb_scale = _ifs_deep_turnover_scale(
            M_b_uncapped, M_b, tau_correction, deep_weight, config.M_b_max,
        )
        M_b = M_b * mb_scale
        plume = plume._replace(M_u=plume.M_u * mb_scale[:, None])

    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
    # ``dt / tau`` (floor tau against zero), NOT ``dt / max(tau, dt)``;
    # the latter under-stepped the relaxation when ``dt > tau`` (audit
    # Codex finding).
    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, 1e-30)
    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
    # Launch-aware mass-flux cap (root-cause fix for the stable-column
    # spurious-heating runaway — validator codex review round-2).  Bechtold's
    # IFS entrainment ``ε`` far exceeds its detrainment ``δ``, so the plume
    # ``M_u = M_b·exp(∫(ε−δ)dz)`` exponentiates by many orders of magnitude.  On
    # a stable / zero-CAPE column the CAPE trigger makes the launch mass flux
    # ``M_b`` ~ ``cape_weight·… ≈ 0``, BUT the exponential growth then drives
    # ``M_u`` straight into a *constant* ``M_b_max`` clip — ERASING the
    # launch-time ``cape_weight`` gate and spuriously heating a quiescent column
    # by ~3900 W/m².  Capping ``M_u_new`` by a launch-aware bound that scales
    # with the CAPE trigger preserves the launch gate through the downstream
    # transport: a genuinely-convecting column (``cape_weight → 1``) keeps the
    # full legacy ``M_b_max`` bound (BYTE-IDENTICAL when ``cape_weight == 1``, so
    # the closed column-MSE budget on a convecting column is untouched), while a
    # zero-CAPE column is capped far below ``M_b_max`` and stays quiescent.
    #
    # The trigger ``cape_weight = smooth_step(CAPE − threshold, sharpness)`` does
    # NOT decay all the way to 0 for a strongly sub-threshold column — it floors
    # at ~9e-4 on the validator stable-dry column — and a 9e-4·M_b_max ≈ 4.6e-5
    # kg/m²/s residual mass flux still drives ~4.6 W/m² of spurious heating
    # (above the <1 W/m² quiescence bar) because the dead plume's ``(T_u − T)``
    # is large.  Squaring the trigger (``cape_weight²·M_b_max``) collapses that
    # soft floor (9e-4 → 8e-7) so the sub-threshold column genuinely quiesces
    # (stable-dry heating 3929 → ~4e-3 W/m²) while leaving a fully-triggered
    # column (``cape_weight = 1 ⇒ 1² = 1``) on the exact legacy ``M_b_max`` cap.
    # ``cape_weight ∈ [0, 1]`` and the square are smooth, so the cap *bound* is
    # differentiable; the surrounding ``jnp.clip`` keeps the SAME piecewise AD
    # behaviour as the legacy constant cap (zero gradient through a clipped
    # value, full gradient through the active bound).  ``cape_weight²·M_b_max ≤
    # M_b_max`` keeps the literature peak as the hard upper bound.
    M_u_cap = (cape_weight ** 2)[:, None] * config.M_b_max
    M_u_new = jnp.clip(M_u_new, 0.0, M_u_cap)

    # Terminate the plume at the convective-top pressure ``p_conv_top_pa``.
    # Bechtold's entraining plume does NOT self-detrain to zero at its LNB —
    # the relaxed carry plateaus at ``M_b_max`` all the way to the model top
    # (confirmed in a C24 AMIP checkpoint: M_u == M_b_max at level 0, a
    # non-detraining profile).  The shared kernel gates only at the fixed
    # 100 hPa stratosphere cutoff, which still leaves the plateaued mass flux
    # at ~cap through the 50-100 hPa levels; the compensating subsidence from
    # that top-heavy profile bakes the upper troposphere / lower stratosphere
    # (+70..86 K over 15 days -> a slow blow-up that #856's cloud-base fix
    # only DELAYED, day 15 -> day 40).  Gating the carry at ``p_conv_top_pa``
    # (default 150 hPa — a physical deep-convection top; tighter than the
    # kernel's 100 hPa) makes the plume terminate there, and threading the
    # SAME cutoff into the kernel keeps the tendency subsidence consistent
    # with the carry.  Tiedtke does not need this — its plume decays.
    M_u_new = M_u_new * stratosphere_mass_flux_gate(
        p_full, config.p_conv_top_pa)

    # -- Environmental tendencies (using relaxed M_u) ---------------------
    # The detrainment rate that feeds the *environmental* tendencies
    # (heat/vapor/cloud-water detrained from the plume) MUST be the SAME
    # height-dependent ``dlt_profile = δ₀·(1.6 − RH)`` that shaped the
    # plume's mass budget ``dM/dz = (ε − δ)·M`` in
    # ``entraining_detraining_plume``.  Passing the unscaled per-class δ₀
    # here while the plume detrained at ``δ₀·(1.6 − RH)`` would account the
    # detrained MASS and the detrained PROPERTIES with different rates — in
    # saturated upper layers (RH capped at 1.0 by F5, so the ``1.6 − RH`` factor
    # floors at 0.6) the environment would receive plume air at ``1.0·δ₀`` while
    # the plume only shed ``0.6·δ₀`` of mass, a ~1.67× over-detrainment that breaks
    # the column MSE / water bookkeeping (Codex adversarial review,
    # IFS-faithfulness iter-1 HIGH #1).  Reusing ``dlt_profile`` keeps the plume
    # and the kernel on one
    # consistent detrainment.  The kernel's subsidence terms are
    # δ-independent, so a per-level δ here only rescales the genuinely
    # δ-proportional detrainment terms (see ``apply_mass_flux_kernel``).
    dT_dt, dq_v_dt, _ = apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, M_u_new,
        z, rho, dlt_profile, M_u_max=config.M_b_max,
        subsidence_solve=config.subsidence_solve,
        p_half=p_half,
        dt=dt,
        theta_implicit=config.theta_implicit,
        p_min_convection=config.p_conv_top_pa,
    )
    rho_safe = jnp.clip(rho, 0.01, None)  # coeff-ok: density floor
    p_gate_qc = stratosphere_mass_flux_gate(p_full, config.p_conv_top_pa)
    dq_c_conv_dt = (
        dlt_profile * M_u_new * p_gate_qc * plume.q_c_u / rho_safe
    )

    # -- Early precip split (IFS sub-cloud evap path only) -------------------
    # The Kessler evaporation needs the POST-SPLIT rain-source profile (only
    # the rain fraction is an evaporable falling flux; the anvil fraction
    # stays aloft as cloud), so the split dispatch runs EARLY here and the
    # legacy late-split block below is skipped.  The OFF path keeps its
    # original op order — byte-identical.  Same dispatch (incl. the
    # unknown-scheme raise) as the late block.
    _split_done = False
    if config.use_ifs_subcloud_evap:
        dq_c_conv_dt = jnp.nan_to_num(jnp.maximum(dq_c_conv_dt, 0.0))
        if config.precip_split_scheme == "constant":
            dq_c_conv_dt, dq_r_conv_dt = split_convective_rain(
                dq_c_conv_dt, config.precip_efficiency)
        elif config.precip_split_scheme == "autoconversion":
            dq_c_conv_dt, dq_r_conv_dt = convective_autoconversion_split(
                dq_c_conv_dt, plume.q_c_u,
                config.autoconv_q_c_crit, config.autoconv_pe_max)
        else:
            raise ValueError(
                f"unknown precip_split_scheme {config.precip_split_scheme!r}; "
                "expected 'constant' or 'autoconversion'")
        _split_done = True

    # -- Optional downdraft (RH-dependent) ---------------------------------
    if config.enable_downdraft:
        # ``lcl_membership_sharpness`` is a LEVEL-INDEX sharpness
        # [1/level] (surface-last: larger index = below LCL altitude).
        below_lcl = jax.nn.sigmoid(
            config.lcl_membership_sharpness
            * (levels_arr[None, :] - k_lcl_smooth[:, None])
        )
        q_sat_env = saturation_mixing_ratio(T, p_full)
        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
        # The 4 ``* dp_full`` column reductions in this branch
        # (below-LCL mass for ``rh_below`` denom, RH-weighted below-LCL
        # for ``rh_below`` num, below-LCL mass for ``evap_rate`` denom,
        # and rain-source positive part for ``evap_total``) all reduce
        # over the same level axis with the same ``dp_full`` weight.
        # Fuse the 3 distinct integrands into one stacked reduction;
        # ``below_mass`` and ``below_lcl_mass`` reuse the first column.
        _stack = jnp.stack(
            [below_lcl, below_lcl * rh_layer, jnp.maximum(dq_c_conv_dt, 0.0)],
            axis=-1,
        ) * dp_full[..., None]
        _col_triple = jnp.sum(_stack, axis=-2)
        _below_lcl_dp = _col_triple[..., 0]
        below_mass = _below_lcl_dp + 1e-6
        rh_below = _col_triple[..., 1] / below_mass
        # RH-FRACTION sharpness [1/RH]: the argument is O(0.1), so the
        # default 10 gives a crisp trigger around ``downdraft_RH_min``.
        downdraft_trigger = jax.nn.sigmoid(
            config.downdraft_rh_sharpness
            * (config.downdraft_RH_min - rh_below)
        )
        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
        # Subcloud rain-evaporation cooling — see tiedtke.py for the
        # full derivation.  ``E_layer`` [kg/(kg·s)] mass-weighted over
        # below-LCL layers; the rain mass that re-evaporates is drawn
        # from ``dq_c_conv_dt`` so the column water budget closes
        # (Codex stop-time review: "downdraft fix still creates column
        # water" — earlier form added vapor without removing the
        # corresponding cloud-water source).
        # SUPERSEDED by the oracle Kessler evaporation when
        # ``use_ifs_subcloud_evap`` is on (two evap paths would
        # double-count); the trigger/M_d above still feed CMT.
        if not config.use_ifs_subcloud_evap:
            below_lcl_mass = _below_lcl_dp[:, None].clip(1e-6, None)
            rain_source_total = _col_triple[..., 2] / constants.g
            evap_total = jnp.minimum(
                jnp.abs(M_d_base) * config.downdraft_evap_efficiency,
                rain_source_total,
            )
            evap_rate = (
                evap_total[:, None] * below_lcl * constants.g / below_lcl_mass
            )
            dT_dt_dd = -(constants.L_v / constants.c_pd) * evap_rate
            dT_dt = dT_dt + dT_dt_dd
            dq_v_dt = dq_v_dt + evap_rate
            rain_source_safe = jnp.clip(rain_source_total[:, None], 1e-30, None)
            rain_scale = 1.0 - evap_total[:, None] / rain_source_safe
            dq_c_conv_dt = jnp.where(
                dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt,
            )

    # -- IFS Kessler sub-cloud rain evaporation (cuflxn.F90:436-475, default ON) --
    # INDEPENDENT of ``enable_downdraft`` (IFS evaporates the precip flux
    # below cloud base wherever the sub-cloud air is drier than the RH break,
    # downdraft or not).  Vapor deposits AT the evaporating levels (the
    # oracle's per-layer recurrence — better spatial structure than the
    # legacy uniform mass-weighted deposit); the evaporated water is debited
    # from the rain source by a column-uniform scale so ``dq_r_conv_dt`` stays
    # a non-negative source (contract) and column water closes exactly.
    if config.use_ifs_subcloud_evap and dq_r_conv_dt is not None:
        _below_lcl_evap = jax.nn.sigmoid(
            config.lcl_membership_sharpness
            * (levels_arr[None, :] - k_lcl_smooth[:, None])
        )
        _q_sat_evap = saturation_mixing_ratio(T, p_full)
        _rh_evap = jnp.clip(q_v / jnp.maximum(_q_sat_evap, 1e-12), 0.0, None)
        # Soft-gather the environment RH at the smooth cloud base / top for
        # the RCUCOV area RH-enhancement (the shared Gaussian-softmax gather;
        # the helper is q_sat-named but is a generic level-index gather).
        _rh_base = _ifs_cloud_base_qsat(_rh_evap, k_lcl_smooth, levels_arr)[:, 0]
        _rh_top = _ifs_cloud_base_qsat(_rh_evap, k_lnb_smooth, levels_arr)[:, 0]
        evap_rate_ifs, rain_scale_ifs = _ifs_subcloud_rain_evaporation(
            q_v, _q_sat_evap, p_half, dp_full, dq_r_conv_dt,
            _below_lcl_evap, _rh_base, _rh_top, deep_weight, dt,
        )
        dT_dt = dT_dt - (constants.L_v / constants.c_pd) * evap_rate_ifs
        dq_v_dt = dq_v_dt + evap_rate_ifs
        dq_r_conv_dt = dq_r_conv_dt * rain_scale_ifs

    # -- Penetrative-downdraft thermodynamic transport (opt-in) --------------
    # INDEPENDENT of the re-evaporation downdraft above (hence OUTSIDE the
    # ``enable_downdraft`` block — it is NOT a no-op when that is off): it is
    # driven by CONVECTIVE activity — its mass flux is a fraction
    # ``downdraft_alpha`` of the cloud-base updraft mass flux ``M_b`` (Tiedtke
    # 1989 M_d ∝ M_u at the LFS) — NOT the re-evap ``downdraft_trigger`` (which
    # fires only where the sub-cloud RH is LOW, i.e. NOT the humid marine BL
    # this lever targets).  It advects low-MSE (dry) mid-tropospheric air DOWN
    # into the sub-cloud layer, DRYING it (the ventilation the
    # re-evaporation-only downdraft lacks).  Conservative (column s and q_v
    # integrals preserved) + CFL/positivity-limited.
    if config.downdraft_transport:
        dT_dt_transport, dq_v_dt_transport = _penetrative_downdraft_transport(
            T, q_v, z, dp_full, k_lcl_smooth, levels_arr,
            config.downdraft_alpha * M_b,
            config.downdraft_entrain_rate,
            config.downdraft_detrain_scale_m,
            dt,
        )
        dT_dt = dT_dt + dT_dt_transport
        dq_v_dt = dq_v_dt + dq_v_dt_transport

    # -- CMT --------------------------------------------------------------
    if config.enable_cmt:
        if config.enable_downdraft:
            # CMT downdraft profile = -downdraft_alpha · M_u · trigger,
            # matching ``M_d_base = -downdraft_alpha · M_b · trigger``
            # used by the subcloud rain-evap path above.  An earlier
            # formulation had an extra hardcoded ``× 0.3`` factor on
            # top of ``downdraft_alpha`` (effective 9 % instead of the
            # canonical Tiedtke 30 %) AND omitted the RH trigger gate
            # — fixed jointly with the Tiedtke scheme in iter-102/103.
            M_d = (
                -config.downdraft_alpha
                * M_u_new
                * downdraft_trigger[:, None]
            )
        else:
            M_d = None
        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
            u, v, M_u_new, M_d,
            p_full, p_half, rho,
            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
        )
    else:
        du_dt_conv = None
        dv_dt_conv = None

    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)

    # In-updraft precipitation: split the detrained condensate into a
    # precipitating rain fraction + suspended anvil remainder via the shared
    # helper (same knob + mass proof as Tiedtke). precip_efficiency=0
    # (default) ⇒ no split, byte-identical to the pre-split Bechtold.
    # Finite guard (before the rain split so it gets finite input): a rare
    # degenerate C48 column can emit a non-finite thermal/moisture/condensate
    # tendency that the dynamics amplify to non-finite winds; bound it. nan_to_num
    # maps NaN->0, Inf->finite; the clip is far above any real convective rate.
    # nan_to_num ONLY (NaN->0, Inf->finite) — the actual fix for the rare C48
    # non-finite-winds NaN.  The former MAGNITUDE clip (±_BECHTOLD_*_MAX) is
    # REMOVED here: clipping dT and dq_v to DIFFERENT bounds is not MSE-neutral,
    # so on a strongly-convecting column it broke the column enthalpy budget
    # (test_bechtold_mse_conservation, ~40% residual on the campaign branch;
    # main has no such clip and conserves).  nan_to_num alone keeps the
    # tendencies finite without the asymmetric conservation leak.
    dT_dt = jnp.nan_to_num(dT_dt)
    dq_v_dt = jnp.nan_to_num(dq_v_dt)
    dq_c_conv_dt = jnp.nan_to_num(jnp.maximum(dq_c_conv_dt, 0.0))

    # Precip split dispatch (validated on the STATIC config value at scheme
    # entry — dispatch-hardening: an unknown value runs different physics, so
    # raise rather than silently defaulting). "autoconversion" derives the
    # precip fraction from the plume updraft cloud water q_c_u (the same field
    # dq_c_conv_dt is built from at L571), so the two are per-level aligned.
    # Skipped when the IFS sub-cloud-evap path already split early (the
    # evaporation needed the rain profile); ``dq_r_conv_dt`` then already
    # carries the evap debit.
    if _split_done:
        pass
    elif config.precip_split_scheme == "constant":
        dq_c_conv_dt, dq_r_conv_dt = split_convective_rain(
            dq_c_conv_dt, config.precip_efficiency)
    elif config.precip_split_scheme == "autoconversion":
        dq_c_conv_dt, dq_r_conv_dt = convective_autoconversion_split(
            dq_c_conv_dt, plume.q_c_u,
            config.autoconv_q_c_crit, config.autoconv_pe_max)
    else:
        raise ValueError(
            f"unknown precip_split_scheme {config.precip_split_scheme!r}; "
            "expected 'constant' or 'autoconversion'")

    # Sanitize the returned carry + diagnostics too — M_u_new is a real
    # physics-state carry (feeds next step's relaxation), so a non-finite here
    # would re-enter the scheme; cape/mask are diagnostics but kept finite.
    M_u_new = jnp.clip(jnp.nan_to_num(M_u_new), 0.0, config.M_b_max)
    cape_pbl = jnp.nan_to_num(cape_pbl)
    convective_mask = jnp.nan_to_num(convective_mask)
    conv_stoch_state_new = jnp.nan_to_num(conv_stoch_state_new)

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape_pbl,
        convective_mask=convective_mask,
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
        dq_r_conv_dt=dq_r_conv_dt,
    )
    return out, M_u_new, conv_stoch_state_new
