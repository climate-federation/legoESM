"""Emanuel (1991) buoyancy-sorting convection.

The distinctive feature versus ZM (single bulk plume) and KF
(single plume with deep/shallow blend) is the **buoyancy-sorted
ensemble**: at each cloud level the parcel may mix with environmental
air in a discrete spectrum of mixing fractions ``f_i ∈ [0, 1]``.  The
mixed parcel's buoyancy at that level determines whether it
contributes to the upward mass flux (positive buoyancy) or detrains
into a downdraft (negative buoyancy).  This produces a per-level
spread in detrainment height that is impossible with a single bulk
plume.

The smooth-everywhere replacement: instead of a hard ``if B_mix > 0:
ascend`` switch we weight each mixing fraction's contribution by
``sigmoid(s * B_mix_i)``.  This preserves training-time gradients
through the buoyancy threshold while still producing the
qualitatively-correct detrainment-height spread.

Optional unsaturated-downdraft branch (rain evaporation cooling) is
toggled by ``enable_unsaturated_downdraft``.  Implementation: a
fraction ``downdraft_efficiency`` of the column-integrated detrained
condensate is moved as a per-level cooling + moistening tendency in
the cloud layer below LCL.

No convective momentum transport (Emanuel CMT is a separate
extension, deferred).

References
----------
* Emanuel, K. A. (1991). A scheme for representing cumulus convection
  in large-scale models.  *J. Atmos. Sci.*, 48, 2313–2335.
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
from legoesm.atmosphere.physics.convection.config import EmanuelConfig
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
    compute_lcl,
    entraining_detraining_plume,
)


__all__ = ("emanuel_convection",)


def _mixture_buoyancy(T_e, q_e, T_u, q_u, q_c_u, p, fractions):
    """Buoyancy of cloud–environment mixtures across a mixing spectrum
    (Emanuel 1991 buoyancy sorting).

    For an environmental mixing fraction ``χ`` the saturated,
    condensate-laden cloud air and the (sub-saturated) environment mix
    linearly in (T, q_v, q_c); the entrained dry air then evaporates
    condensate in a one-step saturation adjustment, cooling the mixture.
    The resulting virtual-temperature buoyancy ``B(χ)`` equals the
    undilute updraft buoyancy at ``χ=0`` and decreases as χ grows,
    **crossing zero** at a critical fraction for a sufficiently dry
    environment — so some mixtures become negatively buoyant and
    detrain.  That sign reversal is the feature distinguishing Emanuel
    from a single bulk plume (the old ``B_mix = χ·B_u`` form never
    reversed sign).

    Parameters
    ----------
    T_e, q_e : (ncol, nlev)
        Environment temperature [K] / water-vapor mixing ratio [kg/kg].
    T_u, q_u, q_c_u : (ncol, nlev)
        Updraft temperature / vapor / condensate.
    p : (ncol, nlev)
        Pressure [Pa].
    fractions : (n_frac,)
        Environmental mixing fractions χ ∈ (0, 1).

    Returns
    -------
    (ncol, nlev, n_frac) mixture buoyancy [m/s^2].
    """
    chi = fractions[None, None, :]
    pe = p[:, :, None]
    # Linear mixing of conserved-ish variables (cloud ← χ → environment).
    T_m0 = (1.0 - chi) * T_u[:, :, None] + chi * T_e[:, :, None]
    q_m0 = (1.0 - chi) * q_u[:, :, None] + chi * q_e[:, :, None]
    qc_m0 = (1.0 - chi) * q_c_u[:, :, None]          # condensate only from cloud
    # One-step (Newton) saturation adjustment toward q_sat(T_m0).  ``Δq``
    # is the vapor→condensate conversion: positive condenses the
    # super-saturation (mixing saturated cloud with cooler air) and warms;
    # negative evaporates condensate (entrained dry air) and cools.
    # Bounded to [−q_c, q_v] so we never make negative condensate or
    # negative vapor; ``clip`` keeps finite subgradients (AD-safe).
    q_sat_m = saturation_mixing_ratio(T_m0, pe)
    dqs_dT = saturation_mixing_ratio_dT(T_m0, pe)
    L_over_cp = constants.L_v / constants.c_pd
    delta_q = jnp.clip(
        (q_m0 - q_sat_m) / (1.0 + L_over_cp * dqs_dT), -qc_m0, q_m0,
    )
    T_m = T_m0 + L_over_cp * delta_q
    q_m = q_m0 - delta_q
    qc_m = qc_m0 + delta_q
    # Virtual-temperature buoyancy relative to environment, including the
    # condensate loading term (−q_c) on the mixture.
    Tv_m = virtual_temperature(T_m, q_m) - T_m * qc_m
    Tv_e = virtual_temperature(T_e, q_e)[:, :, None]
    return constants.g * (Tv_m - Tv_e) / jnp.maximum(Tv_e, 1.0)


def emanuel_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: EmanuelConfig = EmanuelConfig(),
) -> tuple[ConvectionOutput, jax.Array]:
    """Emanuel buoyancy-sorting convection (smooth, differentiable).

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Environmental temperature [K].
    q_v : jax.Array, shape (ncol, nlev)
        Water-vapor specific humidity [kg/kg].
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa].
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Convection prognostic carry — Emanuel is diagnostic; we pack
        the diagnosed cloud-base mass flux at ``[:, -1]`` for
        visibility.
    dt : float
        Time step [s].
    config : EmanuelConfig
        Scheme tunables.

    Returns
    -------
    out : ConvectionOutput
        Tendencies on environment T, q_v, q_c plus CAPE diagnostic.
        ``du_dt_conv = dv_dt_conv = None`` (Emanuel has no CMT in
        this implementation).
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
    """
    ncol, nlev = T.shape
    del conv_prog_profile  # diagnostic carry only — emit a fresh profile

    # -- Column geometry, moist adiabat, CAPE ------------------------------
    # Use virtual-T moist hydrostatic geometry (clean_physics iter-2 #2).
    dz, rho, z = _compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]

    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)

    # -- Smooth CAPE trigger -----------------------------------------------
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)

    # -- LCL and cloud base ------------------------------------------------
    T_parcel = T_base + config.parcel_perturb_T
    q_parcel = q_base + config.parcel_perturb_q
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_lcl_smooth = lcl.k_lcl_smooth

    # -- Cloud-base mass flux closure (CAPE-relaxation, Emanuel style) ----
    # Emanuel 1991 uses a sub-cloud-layer relaxation.  Dimensionally-
    # correct CAPE-relaxation closure (Kain 2004 §3 form):
    #     M_b = rho_BL * (CAPE - threshold)+ / (g * tau)   [kg/m^2/s]
    # The earlier formula ``(CAPE - threshold)+ / tau`` had units
    # ``m^2/s^3``; magnitude masked operationally only by ``M_b_max``.
    rho_BL = p_full[:, -1] / (constants.R_d * jnp.maximum(T[:, -1], 1.0))
    M_b_eq = (
        cape_weight
        * rho_BL
        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
        / (constants.g * config.sub_cloud_relaxation)
    )
    # See ZhangMcFarlaneConfig.M_b_max.
    M_b = jnp.clip(M_b_eq, 0.0, config.M_b_max)

    # -- Standard entraining plume from cloud base ------------------------
    eps_profile = jnp.full_like(T, config.epsilon_0)
    dlt_profile = jnp.full_like(T, config.delta_0)
    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_lcl_smooth,
        eps_profile, dlt_profile, M_b,
    )

    # -- Buoyancy-sorted ensemble (Emanuel 1991) ---------------------------
    # Build a discrete grid of environmental mixing fractions χ_i ∈ (0,1)
    # with N equal-weight bins.  At each level the mixed parcel's
    # virtual-temperature buoyancy B_mix_i is computed with a genuine
    # evaporative saturation adjustment (``_mixture_buoyancy``), so it
    # **crosses zero** at a critical χ for dry environments — mixtures
    # with B>0 ascend, B<0 detrain/sink.  Its smooth ascending weight is
    # sigmoid(s · B_mix_i).  The detrainment enhancement is the fraction
    # of the spectrum that is negatively buoyant: ≈0 deep in the cloud
    # (undilute parcel strongly buoyant, χ_c→1) and →1 near cloud top
    # (undilute loses buoyancy, χ_c→0).  This reproduces the
    # buoyancy-sorting detrainment-height spread that the old
    # ``B_mix = χ·B_u`` (sign-definite) form could not.
    n_frac = config.n_mixing_fractions
    fractions = jnp.linspace(
        1.0 / (2 * n_frac), 1.0 - 1.0 / (2 * n_frac), n_frac
    )  # environmental mixing-fraction bin midpoints χ
    B_mix = _mixture_buoyancy(
        T, q_v, plume.T_u, plume.q_u, plume.q_c_u, p_full, fractions,
    )                                                    # (ncol, nlev, n_frac)
    ascending_weight_per_frac = jax.nn.sigmoid(
        config.smooth_trigger_sharpness * B_mix
    )                                                    # (ncol, nlev, n_frac)
    ascending_mean = jnp.mean(ascending_weight_per_frac, axis=-1)
    detrained_fraction = 1.0 - ascending_mean            # negatively-buoyant share
    # Buoyancy-sort detrainment multiplier in [1, 1 + cu].
    sort_multiplier = 1.0 + config.cu_coefficient * detrained_fraction

    # Cap plume.M_u once at the source so every downstream use sees
    # the bounded value (see ZM).
    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
    plume = plume._replace(M_u=plume_M_u_capped)

    # -- Environment tendencies via the shared mass-flux kernel ------------
    # Plume splits vapor (``plume.q_u``) and cloud water (``plume.q_c_u``)
    # explicitly so we use the kernel's correct cloud-water source.
    # ``sort_multiplier`` is Emanuel's per-level detrainment enhancement
    # from the buoyancy-sorted ensemble.  Pass it through ``delta_0`` so
    # only the detrainment terms in the kernel are scaled — multiplying
    # the full kernel output by ``sort_multiplier`` also rescaled the
    # delta-independent subsidence terms (compensating-subsidence drying
    # / warming and the adiabatic ``g/c_p`` correction), which is wrong.
    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
        z, rho, config.delta_0 * sort_multiplier, M_u_max=config.M_b_max,
    )
    # Un-enhanced cloud-water source — used by the downdraft bookkeeping
    # below.  ``dq_c_conv_dt`` from the kernel is already enhanced by
    # ``sort_multiplier`` (since we passed ``delta_0 * sort_multiplier``);
    # dividing by ``sort_multiplier`` reconstructs the pre-enhancement
    # value so the downdraft column-budget bookkeeping matches the
    # original implementation's intent (downdraft uses the basic
    # condensate, not the buoyancy-sort-enhanced version).
    # AD-safe floor on the divisor (1e-15) so the VJP
    # ``-dq_c / sort_multiplier²`` cannot overflow fp32 when the
    # buoyancy sort gives a tiny weight.  Codex iter-35 audit pattern.
    dq_c_conv_dt_raw = dq_c_conv_dt / jnp.maximum(sort_multiplier, 1e-15)

    # -- Optional unsaturated-downdraft cooling ---------------------------
    # Implemented as a static Python branch (closure-time decision) so
    # it does NOT add a JAX trace overhead when disabled.  When enabled
    # the column-integrated condensate evaporates a fraction
    # ``downdraft_efficiency`` below LCL, cooling and moistening the
    # sub-cloud layer.
    if config.enable_unsaturated_downdraft:
        # Smooth indicator of "below LCL" (surface-last: index larger
        # than k_lcl_smooth ⇒ below).
        nlev_idx = jnp.arange(nlev, dtype=T.dtype)
        below_lcl = jax.nn.sigmoid(
            2.0 * (nlev_idx[None, :] - k_lcl_smooth[:, None])
        )                                                # (ncol, nlev)
        # Column-integrated condensate source [kg/m^2/s] and below-LCL
        # mass [kg/m^2] both reduce ``* dp / g`` over the level axis —
        # fuse them into one stacked reduction.
        dp = p_half[:, 1:] - p_half[:, :-1]
        _col_pair = jnp.sum(
            jnp.stack([dq_c_conv_dt_raw, below_lcl], axis=-1) * dp[..., None],
            axis=-2,
        ) / constants.g
        column_condensate = _col_pair[..., 0]
        below_mass = _col_pair[..., 1]
        evap_rate = (
            config.downdraft_efficiency
            * column_condensate[:, None]
            * below_lcl
            / jnp.maximum(below_mass[:, None], 1e-6)
        )
        # Evaporation cools T and moistens q (BL).  Conserve column
        # water by removing the same column-integrated mass from the
        # cloud-water source — distributed proportional to where
        # cloud water is *produced* (i.e. dq_c_conv_dt_raw), not where
        # it evaporates (BL).  The earlier formulation subtracted
        # ``evap_rate`` from ``dq_c_conv_dt`` *at the BL*, then clipped
        # to zero — which lost the bookkeeping (the BL has little
        # ``dq_c_conv_dt_raw``) and effectively created vapor from
        # nothing, flipping the sign of column ``Q_v`` on CAPE-positive
        # soundings.
        dT_evap = -(constants.L_v / constants.c_pd) * evap_rate
        dq_v_evap = evap_rate
        dT_dt = dT_dt + dT_evap
        dq_v_dt = dq_v_dt + dq_v_evap
        # ``col_dq_c`` is the same column reduction as
        # ``column_condensate`` above; reuse it instead of recomputing.
        col_dq_c = column_condensate
        weight = dq_c_conv_dt_raw / jnp.maximum(col_dq_c[:, None], 1e-12)
        weight = jnp.where(
            (col_dq_c > 1e-12)[:, None], weight, 0.0,
        )
        # ``∫ weight * dp/g = 1`` when ``col_dq_c > 0``, so
        # ``∫ subtract * dp/g = downdraft_efficiency * col_dq_c``.
        subtract = (
            config.downdraft_efficiency * col_dq_c[:, None] * weight
        )
        dq_c_conv_dt = dq_c_conv_dt - subtract

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
        cape=cape,
        convective_mask=cape_weight,
        du_dt_conv=None,
        dv_dt_conv=None,
    )
    conv_prog_profile_new = (
        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(M_b)
    )
    return out, conv_prog_profile_new
