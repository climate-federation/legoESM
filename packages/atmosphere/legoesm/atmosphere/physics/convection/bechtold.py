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
from legoesm.atmosphere.physics._shared import compute_rho, exner_function
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
        "conv_prog_profile_new": "kg/m^2/s (updated mass-flux carry)",
        "conv_stoch_state_new": "1 (updated AR1 noise state)",
    },
    "sign_convention": (
        "Warms and dries the convecting layer via compensating subsidence and "
        "updraft transport (dT_dt, dq_v_dt); the condensed vapor becomes a "
        "non-negative detrained cloud-water source (dq_c_conv_dt>=0) handed to "
        "microphysics; the optional downdraft cools and moistens the sub-cloud "
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
_BECHTOLD_RH_CAP = 1.3
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

    M_b = M_b_deterministic * jnp.maximum(stoch_factor, 0.0)
    # See ZhangMcFarlaneConfig.M_b_max.
    M_b = jnp.clip(M_b, 0.0, config.M_b_max)

    # -- Per-class entrainment / detrainment profiles (IFS Cy49r1) ---------
    # Faithful to the IFS bulk-plume formulation (Part IV, Ch. 6, eqs 6.7,
    # 6.8, 6.9; cross-checked against ecmwf-ifs/openifs cuascn/cuentr):
    #   E = ε₀ · f_ε · (1.3 − RH) · f_scale ,   f_scale = (q_sat(T̄)/q_sat(T̄_base))³
    #   D = δ₀ · (1.6 − RH)
    # The deep branch — what RCE selects — is exactly the IFS deep form.
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
    RH = jnp.clip(q_v / jnp.maximum(q_sat_env, 1e-12), 0.0, _BECHTOLD_RH_CAP)
    # Saturation at the lowest model level (surface-last index −1) is the
    # IFS departure-level base for the f_scale vertical scaling.  This is
    # the lowest-model-level convention, not the LCL/cloud-base level; the
    # scheme launches its parcel from the PBL mean / LCL, so f_scale here
    # decays relative to the surface, which is a deliberately slightly
    # stronger decay than measuring it from the (higher, cooler) cloud
    # base would give.
    q_sat_base = q_sat_env[:, -1:]
    f_scale = jnp.clip(q_sat_env / jnp.maximum(q_sat_base, 1e-12), 0.0, 1.0) ** 3
    rh_entr = jnp.clip(_BECHTOLD_RH_ENTR - RH, 0.0, None)                       # eq 6.7
    rh_detr = jnp.clip(_BECHTOLD_RH_DETR - RH, 0.0, None)                       # eq 6.8 / 6.9
    entr_factor = rh_entr * f_scale                              # (ncol, nlev)
    # Per-class entrainment ε₀ (shallow carries the f_ε=2 factor in its
    # config default); every class entrains with the same height factor
    # ``entr_factor = (1.3 − RH)·f_scale`` so ``E_class = ε₀_class ·
    # entr_factor`` (the deep branch — what RCE selects — is then exactly
    # eq 6.7).
    eps_per_class = (
        deep_weight[:, None] * config.epsilon_deep
        + shallow_weight[:, None] * config.epsilon_shallow
        + midlevel_weight[:, None] * config.epsilon_midlevel
    )
    eps_profile = eps_per_class * entr_factor

    # Detrainment uses the per-class δ₀ scaled by the IFS ``(1.6 − RH)``
    # factor for ALL classes.  The deep branch — what RCE selects — is
    # then exactly eqs 6.7/6.8.  IFS additionally ties the *shallow*
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
        eps_profile, dlt_profile, M_b,
        buoyancy_death_memory=config.buoyancy_death_memory,
    )

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
    # saturated upper layers (RH clipped to 1.3) the environment would
    # receive plume air at ``1.0·δ₀`` while the plume only shed ``0.3·δ₀``
    # of mass, a 3.3× over-detrainment that breaks the column MSE / water
    # bookkeeping (Codex adversarial review, IFS-faithfulness iter-1 HIGH
    # #1).  Reusing ``dlt_profile`` keeps the plume and the kernel on one
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
    dT_dt = jnp.clip(jnp.nan_to_num(dT_dt), -_BECHTOLD_DTDT_MAX, _BECHTOLD_DTDT_MAX)
    dq_v_dt = jnp.clip(jnp.nan_to_num(dq_v_dt), -_BECHTOLD_DQVDT_MAX, _BECHTOLD_DQVDT_MAX)
    dq_c_conv_dt = jnp.clip(jnp.nan_to_num(jnp.maximum(dq_c_conv_dt, 0.0)),
                            0.0, _BECHTOLD_DQVDT_MAX)

    # Precip split dispatch (validated on the STATIC config value at scheme
    # entry — dispatch-hardening: an unknown value runs different physics, so
    # raise rather than silently defaulting). "autoconversion" derives the
    # precip fraction from the plume updraft cloud water q_c_u (the same field
    # dq_c_conv_dt is built from at L571), so the two are per-level aligned.
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
