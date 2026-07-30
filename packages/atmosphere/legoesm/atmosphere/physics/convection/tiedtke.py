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
* Deep (penetrative) closure — driven by the large-scale moisture
  convergence: production passes the real PER-LEVEL
  ``_shared.compute_moisture_convergence`` field, which this closure
  column-integrates.  When ``moisture_convergence`` is ``None`` (configs
  where large-scale MC is unavailable) it falls back to the
  saturation-EXCESS proxy
  ``MC_proxy = ∫ max(q_v - RH_crit·q_sat, 0) dp / (g·tau_MC_proxy)``.
* Plume integrator's mass-flux profile is gated by buoyancy sigmoid.

Faithfulness to Tiedtke (1989)
------------------------------
Audited term-by-term against Tiedtke (1989) — the ORIGINAL
moisture-convergence-closure scheme.  Note the widely-used code
descendants (ECMWF IFS ``cumastrn``, WRF ``module_cu_tiedtke``) adopt
the later Nordeng (1994) CAPE closure for deep convection, so the 1989
*paper* — not those codes — is the oracle for the closure below.  The
downdraft ratio matches the cloned IFS source (``sucumf``:
``RMFDEPS = 0.30``); the turbulent ENTRAINMENT rates below are the
Tiedtke-1989 PAPER values, which the IFS later retuned (e.g. deep
detrainment ``DETRPEN = 0.75e-4``), so those follow the paper, not the
modern IFS.  One legoESM DETRAINMENT default (mid-level ``δ = 2e-4``)
departs from the paper — see DEPARTURES.

FAITHFUL to Tiedtke (1989):

* Turbulent ENTRAINMENT rates — deep ε = δ = 1e-4 m⁻¹, shallow
  ε = δ = 3e-4 m⁻¹, mid-level ε = 1e-4 m⁻¹ (``TiedtkeConfig`` defaults;
  the canonical Tiedtke-1989 turbulent-mixing coefficients).  The
  mid-level DETRAINMENT default (δ = 2e-4) departs — see below.
* Downdraft — mass flux at the level of free sinking is
  ``downdraft_alpha`` (0.3) × the diagnosed cloud-base updraft mass flux
  in the RH-gate's active limit (Tiedtke-1989 downdraft closure; the
  ratio matches IFS ``RMFDEPS = 0.30``, verified in the cloned source).
* Deep (penetrative, Type-1) closure driven by large-scale moisture
  convergence — production passes the real PER-LEVEL
  ``_shared.compute_moisture_convergence`` field, which this closure
  column-integrates.
* Three cloud types (deep / shallow / mid-level) — Tiedtke's discrete
  type SELECTION rendered as a smooth cloud-depth blend (differentiable
  surrogate; identical to the discrete choice in the crisp limit).

DEPARTURES / SURROGATES (NOT faithful to the 1989 paper — documented):

* **Shallow closure** uses a CAPE-relaxation SURROGATE, not Tiedtke's
  sub-cloud moisture-supply closure (``M_b`` balancing surface
  evaporation + sub-cloud turbulent moisture-flux convergence).  The
  surface latent-heat flux is a coupler quantity NOT passed to the
  convection scheme, so the exact 1989 shallow closure is not currently
  computable here — a structural input gap, not a numerical choice.
* **Mid-level closure** is tied to the shallow surrogate
  (``M_b = M_b_shallow × midlevel_M_b_fraction``) rather than Tiedtke's
  elevated large-scale moisture convergence.  Unlike the shallow case
  the required input (``mc_col``) is available in gridded configurations,
  so this is fixable there; it is deferred pending an RCE-gated
  controlled test — a closure can be oracle-faithful yet regress
  equilibrium (cf. the Bechtold F6 mismapping, +19 K RCE).
* **Mid-level detrainment** default ``delta_midlevel = 2e-4 m⁻¹``
  exceeds the 1e-4 turbulent entrainment (a legoESM tuning choice;
  Tiedtke's turbulent ε and δ are symmetric), so mid-level plumes
  detrain faster than the paper.
* **Deep closure** is a proportional response to the (gated) moisture
  convergence, not Tiedtke's exact balance requiring the convective
  moisture SINK to equal the large-scale supply; it also carries a CAPE
  weight (a Nordeng-flavoured trigger), not a pure 1989 parcel test.
* **Attenuation of the deep MC closure (known issue, deferred).** The
  shallow CAPE-surrogate cloud-base mass flux
  ``M_b_shallow = cape_weight·ρ_BL·(CAPE−thr)₊/(g·tau_shallow_M_b)``
  exceeds the ``M_b_max`` cap (0.05) for moderate-and-larger CAPE
  (≈0.16 kg/m²/s at CAPE≈5000, ≈0.7 at CAPE≈22000), so wherever the
  shallow class weight is non-negligible the class-blended ``M_b`` clips
  at the cap and the deep moisture-convergence contribution is
  attenuated.  This compounds with a small deep-class weight from the
  smooth ``cloud_depth = z_lnb − z_lcl`` metric under-detecting the cloud
  top (a CAPE≈5000 sounding reports ~1.8 km depth).  For one deep test
  sounding, feeding a large ``moisture_convergence`` changed the summed
  updraft mass flux by ≈0 at the default config but by ≈+0.004 kg/m²/s
  once the shallow surrogate was suppressed — i.e. the deep MC closure
  functions but has limited influence in typical columns.  Unmasking it
  (raising the shallow saturation and/or fixing the depth metric) is
  BEHAVIORAL and must pass the SCM-RCE realism gate (cf. Bechtold F6) —
  tracked as a follow-up, not fixed here.
* **Organized entrainment/detrainment** (Tiedtke's cloud-base organized
  inflow / cloud-top organized outflow) is not represented — entrainment
  is purely turbulent (constant per class).

References
----------
* Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models.  *Mon. Wea. Rev.*, 117,
  1779–1800.
* Nordeng, T. E. (1994). Extended versions of the convective
  parametrization scheme at ECMWF.  ECMWF Tech. Memo. 206 (CAPE closure
  adopted by the IFS/WRF descendants; not used for the deep closure here).
* Gregory, D., et al. (1997). Parametrization of momentum transport
  by convection. II.  *Quart. J. Roy. Meteor. Soc.*, 123, 1153–1183.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics._shared import compute_rho
from legoesm.atmosphere.physics.thermodynamics import (
    parcel_profile_and_cape,
)
from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
from legoesm.atmosphere.physics.convection.output import (
    ConvectionOutput,
    convective_autoconversion_split,
    split_convective_rain,
)
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel,
    release_detrained_condensate_latent,
    stratosphere_mass_flux_gate,
    compute_column_geometry,
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


__physics_contract__ = {
    "summary": (
        "Tiedtke (1989) bulk mass-flux convection: three-class (deep/mid/"
        "shallow) soft blend, RH-triggered downdraft, a per-level M_u(k) "
        "prognostic carry, and optional Gregory-97 convective momentum "
        "transport. Uses the shared subsidence+detrainment kernel; condensate "
        "handed to microphysics. Smooth (differentiable)."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "u": "m/s", "v": "m/s",
        "conv_prog_profile": "kg/m^2/s (previous-step updraft mass-flux profile M_u(k))",
        "dt": "s",
        "moisture_convergence": "kg/kg/s (large-scale dq/dt|dyn for the deep closure; optional)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained cloud-water source to microphysics, >=0)",
        "cape": "J/kg", "convective_mask": "1 (0-1 activation)",
        "du_dt_conv": "m/s^2 (CMT; None if disabled)",
        "dv_dt_conv": "m/s^2 (CMT; None if disabled)",
        "dq_r_conv_dt": "kg/kg/s (convective rain source when precip_efficiency>0; else None)",
        "conv_prog_profile_new": "kg/m^2/s (implicit-Euler-relaxed M_u(k))",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Convection warms aloft and dries where the "
        "updraft detrains; dq_c_conv_dt, dq_r_conv_dt >= 0 are SOURCES to "
        "microphysics (precip deferred). Compensating subsidence + detrainment "
        "APPROXIMATELY conserve column moist static energy and total water "
        "(exact only in the opt-in implicit_flux solve; truncation-order in the "
        "default advective solve); the downdraft "
        "rain-evaporation is energy-consistent (dT=-L_v/c_pd*dq_v) and "
        "column-water conserving (net column d(q_v+q_c)=0). Optional "
        "Gregory-97 CMT redistributes momentum vertically (transport-dominant; "
        "its pressure-gradient term is not exactly momentum-conserving, so "
        "momentum is not claimed)."
    ),
    # The DEFAULT public path uses the shared mass-flux kernel's advective
    # subsidence solve, conservative only to TRUNCATION ORDER (exact only in
    # the opt-in implicit_flux path), so no contract-level conservation is
    # guaranteed; the column budget is closed downstream.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Tiedtke (1989), Mon. Wea. Rev. 117, 1779-1800; "
        "Gregory et al. (1997), Q. J. R. Meteorol. Soc. 123, 1153-1183"
    ),
    "idealized_test": (
        "tests/unit/test_tiedtke.py; CAPE<=threshold -> zero tendency; a deep "
        "conditionally-unstable column -> deep-class heating aloft + drying "
        "with a positive dq_c source; the downdraft branch conserves column "
        "water; CMT populated only when enable_cmt=True."
    ),
}


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
    moisture_convergence: jax.Array | None = None,
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
    # Use virtual-T moist hydrostatic geometry (clean_physics iter-2 #2).
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]
    # Lift the surface parcel along its dry->LCL->moist path with the actual
    # boundary-layer humidity and compute virtual-T CAPE (shared recipe).
    # The legacy saturated-from-base parcel spuriously inflates CAPE in dry
    # columns; passing q_v is required for a faithful trigger.
    T_moist, cape = parcel_profile_and_cape(T, p_full, p_half, q_v=q_v)
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)

    # -- LCL, LFC, LNB diagnostics -----------------------------------------
    T_parcel = T_base + config.parcel_dT
    q_parcel = q_base + config.parcel_dq
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth
    # LFC/LNB on the SAME virtual-T buoyancy as the CAPE above.  Parcel
    # vapor follows the shared recipe inside ``parcel_profile_and_cape``:
    # launch humidity conserved on the dry leg, saturation-capped above
    # the LCL.
    q_v_parcel_ma = jnp.minimum(
        q_base[:, None], saturation_mixing_ratio(T_moist, p_full)
    )
    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(
        T, T_moist, sharpness=1.0, q_v_env=q_v, q_v_parcel=q_v_parcel_ma,
    )

    # Cloud depth: smooth interpolation of z at fractional indices.
    levels = jnp.arange(nlev, dtype=T.dtype)
    weight_lcl = jax.nn.softmax(
        -2.0 * (levels[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
    )
    weight_lnb = jax.nn.softmax(
        -2.0 * (levels[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
    )
    # Both reductions share the level axis with weight ``z`` — fuse.
    _z_pair = jnp.sum(
        jnp.stack([weight_lcl, weight_lnb], axis=-1) * z[..., None], axis=-2,
    )
    z_lcl, z_lnb = _z_pair[..., 0], _z_pair[..., 1]
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

    # -- Closure: deep uses moisture convergence; shallow and midlevel
    # use a CAPE-relaxation closure.  Combined per-column closure is a
    # class-weighted blend.  When the bridge supplies a real per-level
    # ``moisture_convergence`` field (from
    # ``_shared.compute_moisture_convergence``, available for cubed-
    # sphere and lat-lon dycores), column-integrate and use it directly;
    # otherwise (argument None) fall back to a saturation-EXCESS proxy
    # that is qualitatively similar (positive in moist columns, vanishing
    # in dry ones).
    q_sat_env = saturation_mixing_ratio(T, p_full)
    dp = p_half[:, 1:] - p_half[:, :-1]
    if moisture_convergence is not None:
        # Real MC — column-integrate per (kg/m^2/s).
        column_MC = (
            jnp.sum(moisture_convergence * dp, axis=-1) / constants.g
        )
        column_MC_proxy = jnp.maximum(column_MC, 0.0)
    else:
        # Saturation-EXCESS proxy: vapor in excess of RH_crit * q_sat,
        # column-integrated and divided by a relaxation timescale.
        # Positive in moist columns (q_v > RH_crit * q_sat), vanishing
        # in dry ones — this is the qualitative signature of moisture
        # convergence over a long timescale.
        #
        # The earlier formulation used the saturation DEFICIT
        # ``max(q_sat - q_v, 0)`` which has the opposite sign: large
        # in dry columns, vanishing in moist columns — convection
        # would be suppressed exactly where it should fire.
        sat_excess = jnp.maximum(
            q_v - config.mc_proxy_RH_crit * q_sat_env, 0.0,
        )
        column_MC_proxy = (
            jnp.sum(sat_excess * dp, axis=-1)
            / (constants.g * config.tau_MC_proxy)
        )
    # Smooth gate on MC threshold for deep.
    mc_gate = smooth_positive_part(
        column_MC_proxy - config.moisture_convergence_threshold,
        config.moisture_convergence_sharpness,
    )

    # Cloud-base mass flux (per class, then blended).
    # ``M_b_deep`` is driven by column moisture convergence (kg/m^2/s units
    # — already dimensionally correct) and stays as-is.
    # ``M_b_shallow`` and ``M_b_midlevel`` use a generic first-order
    # CAPE-relaxation SURROGATE (NOT a published closure).  This is *not* the
    # Zhang-McFarlane (1995) closure (CAPE consumed at a cloud-work-function /
    # quasi-equilibrium rate) and *not* a Kain (2004) formula (Kain 2004 has no
    # closed-form M_b — it iterates M_b to remove CAPE over TIMEC).  The
    # ``g / rho_BL`` factor is a dimensional stand-in for that CAPE-consumption
    # sensitivity, giving a kg/m^2/s mass flux:
    #     M_b = rho_BL * (CAPE - threshold)+ / (g * tau)   [kg/m^2/s]
    # The earlier formula omitted ``rho_BL`` and ``g``; magnitude was
    # masked operationally only by ``M_b_max``.
    # Dry boundary-layer density via the shared ideal-gas helper (same
    # 1 K temperature clip as the previous inline form).
    rho_BL = compute_rho(T[:, -1], p_full[:, -1])
    M_b_deep = mc_gate * cape_weight
    M_b_shallow = (
        cape_weight
        * rho_BL
        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
        / (constants.g * config.tau_shallow_M_b)
    )
    M_b_midlevel = M_b_shallow * config.midlevel_M_b_fraction
    M_b = (
        deep_weight * M_b_deep
        + shallow_weight * M_b_shallow
        + midlevel_weight * M_b_midlevel
    )
    # See ZhangMcFarlaneConfig.M_b_max.
    M_b = jnp.clip(M_b, 0.0, config.M_b_max)

    # -- Plume integration -------------------------------------------------
    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_lcl_smooth,
        eps_profile, dlt_profile, M_b,
        buoyancy_death_memory=config.buoyancy_death_memory,
    )

    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
    # ``dt / tau`` (with ``tau`` floored against zero), NOT
    # ``dt / max(tau, dt)`` — see ZM for the audit context.
    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, 1e-30)
    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
    # Cap M_u_new at config.M_b_max so every downstream use (kernel
    # tendencies, dq_c_conv_raw, downdraft trigger, CMT, carry update)
    # sees the same bounded value.
    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)

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
    # Pass the per-column blended delta_0 directly to the kernel.
    # ``apply_mass_flux_kernel`` uses ``delta_0`` ONLY in the
    # detrainment terms (``delta_0 * M * (T_u - T) / rho``,
    # ``delta_0 * M * (q_v_u - q_v) / rho``); the compensating-subsidence
    # contributions are independent of ``delta_0``.  The earlier
    # implementation called the kernel with ``config.delta_deep`` and
    # then multiplied the FULL kernel output by ``delta_0_eff /
    # delta_deep``, which incorrectly rescaled subsidence too — in a
    # shallow-only column with ``delta_shallow > delta_deep`` this
    # over-amplifies the subsidence drying / warming by the same factor
    # the detrainment is enhanced.
    dT_dt, dq_v_dt, dq_c_kernel = apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, M_u_for_kernel,
        z, rho, delta_0_eff[:, None], M_u_max=config.M_b_max,
        # Selectable vertical solve (config default "advective" = shipped
        # behaviour, byte-identical).  ``p_half``/``dt`` are only consumed by
        # the implicit_flux branch; passing them unconditionally keeps the
        # call site single-form, and the kernel raises on an unknown value.
        subsidence_solve=config.subsidence_solve,
        p_half=p_half, dt=dt, theta_implicit=config.theta_implicit,
    )
    # Condensation latent heat of the DETRAINED condensate.  Paired against
    # ``dq_c_kernel`` -- the kernel's OWN third return, i.e. exactly the
    # condensate whose vapor the implicit_flux solve debited -- so the heating
    # and the vapor sink cannot drift apart if the local ``dq_c_conv_dt``
    # formula below ever changes.  No-op on the advective default.
    dT_dt = release_detrained_condensate_latent(
        dT_dt, dq_c_kernel, config.subsidence_solve)
    rho_safe = jnp.clip(rho, 0.01, None)  # coeff-ok: density floor
    # Reuse the same stratospheric gate the kernel applies so this
    # custom q_c path does not detrain condensate above the tropopause.
    p_gate_qc = stratosphere_mass_flux_gate(p_full)
    dq_c_conv_dt = (
        delta_0_eff[:, None] * M_u_for_kernel * p_gate_qc * plume.q_c_u / rho_safe
    )

    # -- Optional downdraft (RH-dependent trigger) -------------------------
    if config.enable_downdraft:
        # Column-mean RH below LCL.  ``lcl_membership_sharpness`` is a
        # LEVEL-INDEX sharpness [1/level] (surface-last: index larger
        # than the LCL index = below LCL altitude).
        levels = jnp.arange(nlev, dtype=T.dtype)
        below_lcl = jax.nn.sigmoid(
            config.lcl_membership_sharpness
            * (levels[None, :] - k_lcl_smooth[:, None])
        )
        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
        # Both below-LCL reductions share ``below_lcl * dp`` — fuse them
        # into a single column reduction to halve the device work.
        _below_weight = below_lcl * dp
        _below_sums = jnp.sum(
            jnp.stack([_below_weight, _below_weight * rh_layer], axis=-1),
            axis=-2,
        )
        below_mass = _below_sums[..., 0] + 1e-6
        rh_below = _below_sums[..., 1] / below_mass
        # RH-FRACTION sharpness [1/RH]: the argument is O(0.1), so the
        # default 10 gives a crisp trigger around ``downdraft_RH_min``.
        downdraft_trigger = jax.nn.sigmoid(
            config.downdraft_rh_sharpness
            * (config.downdraft_RH_min - rh_below)
        )
        # Downdraft mass flux = -alpha * M_b at cloud base [kg/(m²·s)].
        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
        # Subcloud rain-evaporation cooling — dimensionally consistent,
        # locally AND column-water conserving.
        #
        # Physical model: a fraction ``downdraft_evap_efficiency`` of the
        # downdraft mass flux re-evaporates as rain falls through the
        # subcloud layer.  Mass conservation requires that re-evaporated
        # water be drawn from the same convective rain source that would
        # otherwise reach the surface — implemented by reducing
        # ``dq_c_conv_dt`` (the cloud-water source that microphysics
        # converts to surface precip) by the same column-integrated rate
        # that appears as a vapor source.  Capping ``evap_total`` at the
        # available rain rate guarantees we never extract more rain than
        # was generated this step.
        #
        # Earlier formulations were broken in two ways: (1) a literal
        # ``0.05`` divided by ``rho_safe`` only — units came out as
        # K·m/s not K/s; (2) cooling was added to dT_dt with no matching
        # dq_v source, then in the next iteration the dq_v source was
        # added but with no matching dq_c_conv sink — the column water
        # budget gained mass every step (audit GWD/convection: "downdraft
        # `0.05` cooling — dimensionally wrong AND non-water-conserving";
        # Codex stop-time review: "downdraft fix still creates column
        # water").
        # Below-LCL mass [kg/m^2] and column-integrated convective rain
        # source [kg/(m^2*s)] both reduce ``* dp`` over the level axis;
        # stack and reduce once.  ``rain_source_total`` is the positive
        # part of ``dq_c_conv_dt`` — cloud water is generated where
        # ``M_u`` detrains, never destroyed by this term.
        _stack = jnp.stack(
            [below_lcl, jnp.maximum(dq_c_conv_dt, 0.0)], axis=-1,
        ) * dp[..., None]
        _col_pair = jnp.sum(_stack, axis=-2)
        below_lcl_mass = _col_pair[..., 0:1].clip(1e-6, None)
        rain_source_total = _col_pair[..., 1] / constants.g
        # Total downdraft evap mass flux [kg/(m²·s)], capped at available
        # convective rain so dq_c_conv_dt stays non-negative after the
        # correction below.
        evap_total = jnp.minimum(
            jnp.abs(M_d_base) * config.downdraft_evap_efficiency,
            rain_source_total,
        )
        # Per-layer evap rate [kg/(kg·s)], mass-weighted over below-LCL.
        evap_rate = (
            evap_total[:, None] * below_lcl * constants.g / below_lcl_mass
        )
        dT_dt_dd = -(constants.L_v / constants.c_pd) * evap_rate
        dT_dt = dT_dt + dT_dt_dd
        # Local water source: rain → vapor in subcloud layer.
        dq_v_dt = dq_v_dt + evap_rate
        # Column conservation: subtract the same column-integrated rate
        # from the convective cloud-water source (proportional scaling
        # over levels where it is positive).  Net column ∫(dq_v + dq_c)
        # contribution from this term is then zero.
        rain_source_safe = jnp.clip(rain_source_total[:, None], 1e-30, None)
        rain_scale = 1.0 - evap_total[:, None] / rain_source_safe
        dq_c_conv_dt = jnp.where(
            dq_c_conv_dt > 0.0, dq_c_conv_dt * rain_scale, dq_c_conv_dt,
        )

    # -- CMT --------------------------------------------------------------
    if config.enable_cmt:
        if config.enable_downdraft:
            # Downdraft mass flux profile for CMT:
            #     M_d(z) = -downdraft_alpha · M_u(z) · downdraft_trigger
            # where ``downdraft_alpha`` is the canonical Tiedtke 1989
            # ~30 % LFS ratio and ``downdraft_trigger`` is the RH-based
            # column gate computed for the subcloud rain-evap branch
            # (sigmoid on below-LCL RH < downdraft_RH_min, ≈1 in dry
            # columns, ≈0 in moist columns).
            #
            # An earlier formulation multiplied by a hardcoded ``× 0.3``
            # on top of ``downdraft_alpha`` and *omitted* the
            # ``downdraft_trigger`` gate (iter-102): the literal × 0.3
            # gave a 9 % effective ratio, and removing it alone (without
            # the trigger) amplified CMT downdraft in moist columns
            # where no real downdraft forms — codex stop-time follow-up.
            # iter-103 applies BOTH the canonical ``downdraft_alpha``
            # ratio AND the RH trigger so CMT downdraft is consistent
            # with the rain-evap path's M_d_base.
            M_d = (
                -config.downdraft_alpha
                * M_u_for_kernel
                * downdraft_trigger[:, None]
            )
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

    # -- In-updraft precipitation (convective precipitation efficiency) ---
    # Split the detrained condensate into a precipitating rain fraction and
    # the suspended anvil remainder (same split used by Bechtold — see
    # split_convective_rain for the rationale + mass proof).  Dispatch is on the
    # STATIC config value at scheme entry (dispatch-hardening: raise on unknown).
    # "autoconversion" derives the precip fraction physically from the plume
    # updraft cloud water q_c_u (the field dq_c_conv_dt is built from at L366).
    if config.precip_split_scheme == "constant":
        dq_c_pos, dq_r_conv_dt = split_convective_rain(
            dq_c_conv_dt, config.precip_efficiency)
    elif config.precip_split_scheme == "autoconversion":
        dq_c_pos, dq_r_conv_dt = convective_autoconversion_split(
            dq_c_conv_dt, plume.q_c_u,
            config.autoconv_q_c_crit, config.autoconv_pe_max)
    else:
        raise ValueError(
            f"unknown precip_split_scheme {config.precip_split_scheme!r}; "
            "expected 'constant' or 'autoconversion'")

    # -- Convective mask ---------------------------------------------------
    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_pos,
        cape=cape,
        convective_mask=convective_mask,
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
        dq_r_conv_dt=dq_r_conv_dt,
    )
    return out, M_u_new
