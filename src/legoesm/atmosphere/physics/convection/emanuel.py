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
from legoesm.thermo import saturation_mixing_ratio
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
    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
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
    # Emanuel 1991 uses a sub-cloud-layer relaxation.  We approximate
    # it as a CAPE-driven mass flux with the configured timescale.
    M_b_eq = (
        cape_weight
        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
        / config.sub_cloud_relaxation
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

    # -- Buoyancy-sorted ensemble enhancement to detrainment ---------------
    # Build a discrete grid of mixing fractions f_i ∈ [0, 1] with N
    # equal-weight bins.  At each level k the mixed parcel buoyancy is
    #     B_mix_i(k) = f_i * (T_u(k) - T_env(k))
    # and its smooth contribution to "ascending" mass is sigmoid(s *
    # B_mix_i).  The buoyancy-sort multiplier on detrainment is the
    # variance of the ascending-weight distribution: where all
    # fractions agree (deep in the cloud or above LNB) it's small;
    # where the ensemble is split (near LNB) it's large.  This acts as
    # a per-level enhancement of the bulk detrainment, producing the
    # height-spread that distinguishes Emanuel from a single plume.
    n_frac = config.n_mixing_fractions
    fractions = jnp.linspace(
        1.0 / (2 * n_frac), 1.0 - 1.0 / (2 * n_frac), n_frac
    )  # midpoint fractions
    B_u = plume.B_u                                       # (ncol, nlev)
    # Outer-product: (ncol, nlev, n_frac).
    B_mix = B_u[:, :, None] * fractions[None, None, :]
    ascending_weight_per_frac = jax.nn.sigmoid(
        config.smooth_trigger_sharpness * B_mix
    )                                                    # (ncol, nlev, n_frac)
    # Mean ascending fraction at each level.
    ascending_mean = jnp.mean(ascending_weight_per_frac, axis=-1)
    # Variance — peaks where the ensemble is split (B_u ≈ 0).
    ascending_var = jnp.mean(
        (ascending_weight_per_frac - ascending_mean[..., None]) ** 2, axis=-1
    )
    # Buoyancy-sort detrainment multiplier in [1, 1 + cu].
    sort_multiplier = 1.0 + 4.0 * config.cu_coefficient * ascending_var

    # Cap plume.M_u once at the source so every downstream use sees
    # the bounded value (see ZM).
    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
    plume = plume._replace(M_u=plume_M_u_capped)

    # -- Environment tendencies via the shared mass-flux kernel ------------
    # Plume splits vapor (``plume.q_u``) and cloud water (``plume.q_c_u``)
    # explicitly so we use the kernel's correct cloud-water source.
    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
        z, rho, config.delta_0, M_u_max=config.M_b_max,
    )

    # Emanuel's per-level detrainment enhancement:
    dT_dt = dT_dt_raw * sort_multiplier
    dq_v_dt = dq_v_dt_raw * sort_multiplier
    dq_c_conv_dt = dq_c_conv_dt_raw * sort_multiplier

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
        # Column-integrated condensate source [kg/m^2/s].
        dp = p_half[:, 1:] - p_half[:, :-1]
        column_condensate = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
        # Distribute evaporation cooling proportional to below_lcl
        # mass.
        below_mass = jnp.sum(below_lcl * dp, axis=-1) / constants.g
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
        # Per-column total evap [kg/m²/s] = downdraft_efficiency *
        # column_condensate by construction.  Subtract from the source
        # at the levels where condensate is produced to keep
        # ``dq_c_conv_dt ≥ 0`` and column water conserved.
        col_dq_c = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
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
