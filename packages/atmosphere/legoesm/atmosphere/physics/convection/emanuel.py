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
    latent_heat_vaporization,
)
from legoesm.atmosphere.physics.convection.config import EmanuelConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel,
    compute_column_geometry,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
)
from legoesm.atmosphere.physics.convection._plume import (
    compute_lcl,
    entraining_detraining_plume,
)
from legoesm.atmosphere.physics.convection._emanuel_mixing import (
    emanuel_mixing_tendencies,
)
from legoesm.atmosphere.physics.convection._emanuel_downdraft import (
    emanuel_downdraft,
)


__all__ = ("emanuel_convection",)


__physics_contract__ = {
    "summary": (
        "Emanuel (1991) buoyancy-sorting convection with a prognostic "
        "cloud-base mass flux (DTMA sub-cloud quasi-equilibrium closure); "
        "returns convective T/q_v/q_c tendencies plus the updated CBMF carry."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "conv_prog_profile": "kg/m^2/s (prognostic CBMF carry in slot -1)",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_conv_dt": "kg/kg/s",
        "cape": "J/kg", "convective_mask": "1 (0-1 CAPE-trigger indicator)",
        "conv_prog_profile_new": "kg/m^2/s (updated prognostic CBMF carry)",
    },
    "sign_convention": (
        "Warms and dries the convecting layer (dT_dt>0, dq_v_dt<0); the "
        "condensed vapor becomes non-negative cloud water (dq_c_conv_dt>=0) "
        "handed to microphysics; a final ENTS pass enforces column vapor-side "
        "moist-enthalpy closure (c_pd*dT_dt + L_v*dq_v_dt integrates to 0 over "
        "the convective layer); total water is closed in the legacy surrogate "
        "path but the genuine mixer sheds a precipitating fraction; the "
        "prognostic cloud-base mass flux is >=0; surface at the last vertical "
        "index."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": "Emanuel (1991), J. Atmos. Sci. 48, 2313-2335",
    "idealized_test": (
        "Validated against the convect43c.f oracle (.physics-validator/"
        "emanuel) for RCE realism; a rest / no-CAPE column produces ~zero "
        "tendency and holds the prognostic CBMF at zero."
    ),
}


# Cloud-base mass-flux relaxation coefficient (Emanuel closure).
_CBMF_RELAX = 0.1

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
    (ncol, nlev, n_frac) mixture virtual-temperature buoyancy excess
        ``Tv_mixture − Tv_env`` [K].  Positive ⇒ buoyant mixture (ascends);
        negative ⇒ detrains.  Returned as a temperature excess (not the
        acceleration ``g·ΔTv/Tv``) so the downstream
        ``sigmoid(smooth_trigger_sharpness[1/K] · B_mix)`` is unit-consistent.
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
    # Virtual-temperature buoyancy EXCESS [K] relative to environment,
    # including the condensate loading term (−q_c) on the mixture.  We
    # return the temperature excess ``Tv_m − Tv_e`` [K] rather than the
    # acceleration ``g·ΔTv/Tv`` [m/s²] so the downstream sigmoid weight
    # ``sigmoid(smooth_trigger_sharpness · B_mix)`` has consistent units:
    # ``smooth_trigger_sharpness`` is documented as ``[1/K]``.  (The
    # earlier acceleration form scaled ΔTv by ``g/Tv ≈ 0.033`` /K, so a
    # 0.5 /K sharpness gave a near-flat sigmoid that barely sorted; the
    # sign-crossing — the defining buoyancy-sort feature — is identical
    # because ``g/Tv > 0``.)
    Tv_m = virtual_temperature(T_m, q_m) - T_m * qc_m
    Tv_e = virtual_temperature(T_e, q_e)[:, :, None]
    return Tv_m - Tv_e


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
        Convection **prognostic** carry.  Emanuel's cloud-base mass flux
        (CBMF) is genuinely prognostic: the previous value is read from
        ``[:, -1]`` and relaxed toward the sub-cloud quasi-equilibrium
        (DTMA closure), then written back to ``[:, -1]`` on output.  This
        scheme must therefore NOT be advanced with a stale stage carry
        under multi-stage integrators (the SCM rejects it under RK).
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

    # -- Column geometry, moist adiabat, CAPE ------------------------------
    # Use virtual-T moist hydrostatic geometry (clean_physics iter-2 #2).
    #
    # PARCEL ORIGIN — SCOPE (codex iter-2 #1).  The oracle diagnoses the
    # parcel-origin level ``NK`` as the level of maximum moist static energy
    # below the level of minimum MSE (convect43c.f lines 359-377), and uses
    # ``T(NK)/Q(NK)`` for the LCL and lifted-parcel thermodynamics.  This
    # implementation launches from the **lowest model level** (surface)
    # instead.  For the canonical tropical / RCE soundings the max-MSE level
    # IS the surface (MSE decreases monotonically upward in the boundary
    # layer), so the two agree; the divergence is confined to columns with a
    # sub-surface MSE maximum (e.g. an elevated mixed layer over a cool
    # surface), which the smooth-surrogate scheme does not yet diagnose.  A
    # smooth max-MSE origin selector is a future faithfulness upgrade; the
    # surface-origin choice is the documented scope of this iteration.
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
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

    # Unperturbed cloud base for the DTMA closure (codex iter-1 #2).  The
    # oracle's quasi-equilibrium responds to the *actual* parcel-origin
    # state ``Q(NK)`` with no temperature/humidity perturbation; our
    # ``parcel_perturb_T/q`` is a numerical nudge used only to launch the
    # plume.  Computing the DTMA cloud-base level and parcel buoyancy from
    # the unperturbed base keeps the closure internally consistent with the
    # oracle (the perturbed LCL above is reserved for the plume).
    lcl_unpert = compute_lcl(T_base, q_base, p_base, p_full)

    # -- Prognostic cloud-base mass-flux closure (Emanuel DTMA, FAITHFUL) --
    # FIDELITY (oracle convect43c.f lines 549-573).  Emanuel does NOT use a
    # bulk-CAPE relaxation; the cloud-base mass flux ``CBMF`` is a
    # *prognostic* quantity relaxed toward the sub-cloud-layer
    # quasi-equilibrium.  Each call:
    #     DTMA = (Tv_parcel - Tv_env) at cloud base  +  DTMAX  +  DTPBL
    #     CBMF = (1 - DAMP·dt/300)·CBMF_old  +  0.1·ALPHA·DTMA   (≥ 0)
    # where ``DTMA`` [K] is the lifted-parcel virtual-temperature excess at
    # the LCL (plus the sub-cloud mean buoyancy and the allowed negative
    # perturbation ``DTMAX``).  CBMF therefore ramps smoothly from ~0 when
    # the cloud-base parcel is only marginally buoyant to large values for
    # a strongly unstable sub-cloud layer — reproducing the oracle's
    # CBMF(lapse) ramp (≈0 below 7.3 K/km, growing through 9 K/km) that a
    # bulk-CAPE trigger (full strength at any positive CAPE) cannot.  The
    # earlier CAPE-relaxation closure fired at full strength on near-moist-
    # adiabatic columns the oracle treats as non-convective.
    #
    # ``CBMF_old`` is read back from the prognostic carry slot
    # ``conv_prog_profile[:, -1]`` (where this function stores it on
    # output), so CBMF is genuinely carried between calls as the oracle
    # requires.
    cbmf_old = jnp.maximum(conv_prog_profile[:, -1], 0.0)        # (ncol,)

    # FIDELITY (oracle DTMA, convect43c.f lines 549-558).  The closure's
    # forcing is the *sub-cloud* lifted-parcel buoyancy excess, NOT the
    # deep moist-adiabat buoyancy.  Below the LCL the parcel ascends along
    # a **dry adiabat** carrying its launch water ``q_base`` and is only
    # weakly buoyant; that weak excess is what discriminates a convecting
    # column from a stable one (the deep moist-adiabat buoyancy aloft is
    # always large and would fire on every column).  Oracle DTMA:
    #     DTMA = TVPPLCL − TVAPLCL + DTMAX + DTPBL
    # with TVPPLCL the parcel virtual-T extrapolated to the LCL, TVAPLCL
    # the environment virtual-T at the LCL, and DTPBL the mass-mean
    # sub-cloud buoyancy excess.  We build a smooth equivalent: the
    # dry-adiabatic sub-cloud parcel virtual-T minus the environment
    # virtual-T, mass-averaged over the layers at/below the LCL, plus the
    # allowed negative perturbation DTMAX.
    nlev_idx = jnp.arange(nlev, dtype=T.dtype)
    dp = p_half[:, 1:] - p_half[:, :-1]

    # Dry-adiabatic sub-cloud parcel temperature: T_base − (g/c_pd)·(z −
    # z_base).  ``z`` is height above the surface (z[:, -1] ≈ 0), so the
    # parcel cools upward.  Virtual-T uses the *unperturbed* launch state
    # (T_base, q_base) — the closure's quasi-equilibrium responds to the
    # actual sub-cloud parcel, not the +parcel_perturb_T trigger nudge
    # (which is only used to launch the plume).  Same specific-humidity
    # virtual-T convention as the environment, so DTMA matches the oracle
    # to ~0.1 K (verified against the instrumented Fortran DTMIN/DTPBL/
    # TVPPLCL/TVAPLCL dump in .physics-validator/emanuel).
    z_base = z[:, -1:]
    T_dry_parcel = T_base[:, None] - (constants.g / constants.c_pd) * (
        z - z_base
    )
    Tv_parcel_sub = virtual_temperature(T_dry_parcel, q_base[:, None])
    Tv_env = virtual_temperature(T, q_v)
    buoy_excess_sub = Tv_parcel_sub - Tv_env                    # (ncol, nlev) [K]

    # ``TVPPLCL - TVAPLCL``: dry-adiabatic parcel buoyancy excess AT the
    # LCL.  CONVECT v4.3c evaluates this at ``PLCL`` (lines 549-552), not
    # with a level-space average.  The old Gaussian selector averaged in
    # the first level above cloud base on coarse RCE grids; when PLCL sat
    # at the surface it produced a spurious ~2 K negative DTMA and shut
    # off convection despite large free-tropospheric CAPE.
    # CONVECT does not recompute parcel virtual temperature at PLCL from
    # Bolton's TLCL plus a separate saturation lookup.  It extrapolates the
    # lifted parcel virtual temperature dry-adiabatically from the parcel
    # origin to PLCL (convect43c.f lines 611-612):
    #
    #   TVPPLCL = TVP(ICB-1)
    #             - R_d TVP(ICB-1) (P(ICB-1)-PLCL)/(CPN(ICB-1) P(ICB-1))
    #
    # The distinction matters in marginal quasi-equilibrium RCE: the
    # independent Bolton+saturation reconstruction was biased low by
    # O(0.05-0.1 K), enough to make ``TVPPLCL-TVAPLCL+DTMAX`` slightly
    # negative and collapse a physically active prognostic CBMF to zero.
    cpn_base = constants.c_pd * (1.0 - q_base) + constants.c_pv * q_base
    Tv_base = virtual_temperature(T_base, q_base)
    Tv_parcel_lcl = Tv_base - (
        constants.R_d
        * Tv_base
        * (p_base - lcl_unpert.p_lcl)
        / (cpn_base * p_base)
    )
    Tv_env_lcl = jax.vmap(
        lambda p_lcl_i, p_i, tv_i: jnp.interp(p_lcl_i, p_i, tv_i),
    )(lcl_unpert.p_lcl, p_full, Tv_env)
    be_lcl = Tv_parcel_lcl - Tv_env_lcl                         # (ncol,) [K]

    # ``DTPBL``: mass-weighted mean sub-cloud buoyancy excess over the
    # pressure-bounded layer below PLCL.
    below_lcl = jax.nn.sigmoid(
        config.lcl_pressure_sharpness * (p_full - lcl_unpert.p_lcl[:, None])
    )                                                          # ~1 below PLCL
    sub_mass = jnp.sum(below_lcl * dp, axis=-1)
    dtpbl = jnp.sum(below_lcl * buoy_excess_sub * dp, axis=-1) / jnp.maximum(
        sub_mass, 1.0
    )

    # DTMA [K] (oracle DTMIN = TVPPLCL − TVAPLCL + DTMAX + DTPBL).  Gated
    # by the smooth CAPE trigger so deeply-stable columns (no positive
    # CAPE at all) stay off.
    dtma = cape_weight * (be_lcl + config.dtmax + dtpbl)

    # Prognostic relaxation (oracle line 565-567).  ``DAMPS = DAMP·dt/300``.
    damps = config.damp_coefficient * dt / 300.0  # coeff-ok: reference timestep [s]
    cbmf_new = (1.0 - damps) * cbmf_old + _CBMF_RELAX * config.alpha_closure * dtma
    # CONVECT v4.3c line 567 uses an exact ``MAX(CBMF,0.0)``.  A softplus
    # surrogate leaves a nonzero ln(2)/sharpness mass-flux floor; in RCE
    # that floor drove persistent heating even after CAPE was exhausted.
    cbmf_new = jnp.maximum(cbmf_new, 0.0)
    # See ZhangMcFarlaneConfig.M_b_max.
    M_b = jnp.clip(cbmf_new, 0.0, config.M_b_max)

    if config.use_genuine_mixing:
        # == GENUINE Emanuel (i,j) episodic-mixing buoyancy sort =========
        # Build the full ``(nlev, nlev)`` SIJ/ELIJ/MENT mixing matrix
        # (faithful CONVECT v4.3c port; see ``_emanuel_mixing.py``) and
        # assemble the environmental tendencies from the detrained mass
        # fluxes.  This REPLACES the single-sigmoid ``_mixture_buoyancy``
        # surrogate: every origin level i forms the neutral-buoyancy
        # mixing-fraction spectrum with environment air, each mixture's
        # buoyancy sets its detrainment level j, and MENT(i,j) carries
        # the detrained mass.
        mixing = emanuel_mixing_tendencies(
            T, q_v, p_full, p_half, M_b,
            c_l=config.c_l_emanuel,
            elcrit=config.elcrit,
            tlcrit=config.tlcrit,
            entp=config.entp,
            level_window_sharpness=config.level_window_sharpness,
            sij_gate_sharpness=config.sij_gate_sharpness,
            sij_upper_gate=config.sij_upper_gate,
            denom_floor=config.denom_floor,
            mse_min_search_offset=config.mse_min_search_offset,
            sat_branch_sharpness=config.sat_branch_sharpness,
            strict_index_sharpness=config.strict_index_sharpness,
            epmax=config.precip_efficiency_max,
        )
        dT_dt = mixing.dT_dt
        dq_v_dt = mixing.dq_v_dt
        dq_c_conv_dt = mixing.dq_c_conv_dt
        # The genuine path delivers the in-cloud condensate source
        # directly; the downdraft bookkeeping below reuses it as the raw
        # cloud-water source.
        dq_c_conv_dt_raw = dq_c_conv_dt
    else:
        # == Legacy single-sigmoid surrogate (ablation / back-compat) ====
        # -- Standard entraining plume from cloud base ----------------
        eps_profile = jnp.full_like(T, config.epsilon_0)
        dlt_profile = jnp.full_like(T, config.delta_0)
        plume = entraining_detraining_plume(
            T, q_v, p_full, p_half, z,
            T_parcel, q_parcel, k_lcl_smooth,
            eps_profile, dlt_profile, M_b,
        )

        # -- Buoyancy-sorted ensemble (smooth surrogate) --------------
        n_frac = config.n_mixing_fractions
        fractions = jnp.linspace(
            1.0 / (2 * n_frac), 1.0 - 1.0 / (2 * n_frac), n_frac
        )  # environmental mixing-fraction bin midpoints χ
        B_mix = _mixture_buoyancy(
            T, q_v, plume.T_u, plume.q_u, plume.q_c_u, p_full, fractions,
        )                                                # (ncol, nlev, n_frac)
        ascending_weight_per_frac = jax.nn.sigmoid(
            config.smooth_trigger_sharpness * B_mix
        )                                                # (ncol, nlev, n_frac)
        ascending_mean = jnp.mean(ascending_weight_per_frac, axis=-1)
        detrained_fraction = 1.0 - ascending_mean        # negatively-buoyant share
        sort_multiplier = 1.0 + config.cu_coefficient * detrained_fraction

        plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
        plume = plume._replace(M_u=plume_M_u_capped)

        dT_dt, dq_v_dt, dq_c_conv_dt = apply_mass_flux_kernel(
            T, q_v, p_full,
            plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
            z, rho, config.delta_0 * sort_multiplier, M_u_max=config.M_b_max,
            # NOTE (SCM-RCE matched-kernel campaign): this scheme is
            # deliberately NOT given a ``subsidence_solve`` selector.  Two
            # reasons, both structural:
            #   (a) This call is inside the ``use_genuine_mixing=False``
            #       LEGACY surrogate branch.  The shipped default is
            #       ``use_genuine_mixing=True`` (config.py), whose genuine
            #       buoyancy-sorting mixing matrix computes its own
            #       dT/dq tendencies and never touches this kernel -- so a
            #       selector here would be dead code in production.
            #   (b) The surrogate passes ``delta_0 * sort_multiplier`` IN but
            #       divides only the RETURNED condensate by
            #       ``sort_multiplier`` below.  Under a vapor-debiting solve
            #       that leaves an unpaired ``-(s-1)*dq_c`` vapor sink, i.e.
            #       it would NOT be "the same kernel" anyway.
            # Emanuel is therefore reported as OUTSIDE the matched-kernel
            # family rather than silently kernel-matched.
        )
        # AD-safe floor on the divisor so the VJP cannot overflow.
        dq_c_conv_dt_raw = dq_c_conv_dt / jnp.maximum(sort_multiplier, 1e-15)

    # -- Optional unsaturated-downdraft cooling ---------------------------
    # Implemented as a static Python branch (closure-time decision) so
    # it does NOT add a JAX trace overhead when disabled.  When enabled
    # the column-integrated condensate evaporates a fraction
    # ``downdraft_efficiency`` below LCL, cooling and moistening the
    # sub-cloud layer.
    if config.enable_unsaturated_downdraft:
        # The PORTED shaft (convect43c.f lines 713-830), not the previous
        # column-integrated stand-in.  It needs the mixer's own intermediates
        # (ELIJ, CLW, EP, QS, H, GZ, LV), so it is only available on the
        # genuine-mixing path; requesting it on the legacy surrogate would
        # silently run different physics than the flag advertises.
        if not config.use_genuine_mixing:
            raise ValueError(
                "enable_unsaturated_downdraft requires use_genuine_mixing: "
                "the ported CONVECT downdraft is driven by the buoyancy-sort "
                "mixing matrix (ELIJ/MENT/CLW/EP), which the legacy surrogate "
                "does not produce.")
        # The mixer's intermediates are SURFACE-FIRST (oracle ordering); the
        # environmental profiles here are surface-LAST.  Flip the latter, and
        # flip the shaft's tendencies back before adding them.  PH(i) is the
        # interface at the BOTTOM of level i, which surface-last is
        # ``p_half[k+1]``, i.e. ``p_half[:, 1:]`` reversed.
        _rev = lambda x: x[:, ::-1]  # noqa: E731 - local, one use per array
        cpn_profile = (constants.c_pd * (1.0 - q_v)
                       + constants.c_pv * q_v)
        downdraft = emanuel_downdraft(
            T=_rev(T), q=_rev(q_v), qs=mixing.qs,
            p_full=_rev(p_full), p_half=_rev(p_half[:, 1:]),
            h_dry_static=mixing.h_dry_static, gz=mixing.gz, lv=mixing.lv,
            cpn=_rev(cpn_profile),
            m_profile=mixing.m_profile, ment=mixing.ment, elij=mixing.elij,
            clw=mixing.clw, ep=mixing.ep,
            sigd=config.downdraft_sigd,
            sigs=config.downdraft_sigs,
            omtrain=config.downdraft_omtrain_pa_s,
            omtsnow=config.downdraft_omtsnow_pa_s,
            coeffr=config.downdraft_coeffr,
            coeffs=config.downdraft_coeffs,
            freeze_transition_K=config.downdraft_freeze_transition_K,
            inertia_scale_hPa=config.downdraft_inertia_scale_hPa,
            taper_p_fraction=config.downdraft_taper_p_fraction,
            dhdp_min=config.downdraft_dhdp_min,
            ep_gate_threshold=config.downdraft_ep_gate_threshold,
            ep_gate_width=config.downdraft_ep_gate_width,
        )
        dT_dt = dT_dt + _rev(downdraft.dT_dt)
        dq_v_dt = dq_v_dt + _rev(downdraft.dq_v_dt)

        # WATER BOOKKEEPING, and it differs from the oracle's on purpose.
        # CONVECT debits its own PRECIP diagnostic by what evaporated on the
        # way down.  This model has no such diagnostic: the precipitating
        # fraction EP*CLW leaves the mixer and the REST is handed to
        # microphysics as ``dq_c_conv_dt``.  So the vapour the shaft adds is
        # taken back out of that condensate source, column-integral for
        # column-integral, distributed where the condensate is PRODUCED
        # rather than where it evaporates -- the sub-cloud layer has almost
        # no ``dq_c_conv_dt_raw``, and subtracting there then clipping at zero
        # is exactly the bookkeeping loss that once flipped the sign of the
        # column vapour budget.
        col_evap = jnp.sum(
            _rev(downdraft.dq_v_dt) * dp, axis=-1) / constants.g
        col_dq_c = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
        weight = dq_c_conv_dt_raw / jnp.maximum(col_dq_c[:, None], 1e-12)
        weight = jnp.where((col_dq_c > 1e-12)[:, None], weight, 0.0)
        dq_c_conv_dt = dq_c_conv_dt - col_evap[:, None] * weight

    dq_c_conv_dt = jnp.maximum(dq_c_conv_dt, 0.0)

    # -- Exact water and layer enthalpy conservation ---------------------
    # FIDELITY (oracle convect43c.f).  Emanuel's tendencies force exact
    # vapor-side enthalpy conservation over levels 1..INB in a final pass
    # (oracle lines 969-984).  The genuine mixer also has an explicit
    # precipitating condensate path (EP·CLW); only the legacy surrogate
    # needs column-total-water bookkeeping.  Our scheme uses the shared
    # Tiedtke/Siebesma mass-flux kernel whose detrained condensate source
    # ``δ·M·q_c_u/ρ`` is NOT drawn from a matching column vapor sink, so
    # the raw tendencies break BOTH invariants:
    #   * total water ``∫(dq_v + dq_c)dp/g`` came out ≈ +680 W/m²-equiv on
    #     a CAPE-positive sounding instead of ~0;
    #   * column heating ``c_p·∫FT`` leaked ~2700 W/m² relative to the
    #     latent heat of the net condensation.
    # The physically-correct division of labour for THIS model: convection
    # condenses vapor into cloud water ``dq_c_conv_dt`` (handed to the
    # ``q_c`` tracer → microphysics owns precip/evaporation), and the
    # latent heat of that condensation is *released as convective heating*.
    # The legacy surrogate therefore needs two invariants:
    #
    #   (1) Total water: ``∫(dq_v + dq_c)dp = 0`` — every kg of vapor the
    #       legacy convection removes reappears as cloud water in the column.
    #       In the genuine path, negative net water is retained as the
    #       reference precipitating sink instead of being added back to cloud.
    #   (2) Moist enthalpy of the VAPOR reservoir over levels 1..INB:
    #       ``c_p·∫FT + L_v·∫FQ_v = 0`` — i.e. heating in the convective
    #       layer equals the latent heat released condensing the vapor that
    #       became cloud, ``c_p·∫FT = L_v·∫FQ_c``.
    #       Crucially the detrained-condensate term ``FQ_c`` is EXCLUDED
    #       from this MSE invariant: that condensate is liquid water whose
    #       latent heat has *already been released into FT*, so including
    #       it (as an earlier version did) zeroed the convective heating
    #       entirely and the free troposphere then cooled radiatively with
    #       no convective offset — a ~40 K-too-cold RCE.  The oracle's ENTS
    #       pass conserves exactly this layer-limited vapor-side enthalpy
    #       (its FQ is vapor only; the condensate left the column as PRECIP).
    #
    # Both corrections are uniform/activity-weighted shifts built from sums
    # and divides by positive column masses — AD-safe (finite subgradients)
    # everywhere.  The water step does ``clip(qc_scale, 0, 1)`` as a
    # defensive guard that binds only in the two pathologies noted below
    # (net drying, or more water created than condensate exists); in the
    # normal CAPE-positive case ``qc_scale ∈ (0, 1)`` and the clip is
    # inactive, so gradients flow.
    col_mass = jnp.sum(dp, axis=-1)                      # ∝ ∫dp
    L_v = constants.L_v
    c_pd = constants.c_pd

    # (1) Legacy total-water conservation / genuine positive-water cleanup.
    # If the kernel creates net water (``net_water > 0``), remove it from
    # the condensate channel by *rescaling* ``dq_c`` by a uniform per-column
    # factor, so the subtraction is distributed exactly where condensate
    # exists and can never drive ``dq_c`` negative.  Negative net water is
    # only added back in the legacy surrogate; in the genuine mixer it is
    # the reference precipitating condensate sink.
    #
    #   ∫dq_c_new = ∫dq_c − net_water·g   ⇒   scale = 1 − net_water·g/∫dq_c
    #
    # ``scale`` is clipped to [0, 1] (net_water ≤ ∫dq_c in practice since
    # the spurious source IS the excess condensate; the clip is a defensive
    # floor only).  For inactive columns ``∫dq_c ≈ 0`` and ``net_water ≈ 0``
    # so ``scale → 1`` and the condensate is untouched.
    net_water = jnp.sum((dq_v_dt + dq_c_conv_dt) * dp, axis=-1) / constants.g
    col_dq_c_mass = jnp.sum(dq_c_conv_dt * dp, axis=-1) / constants.g

    # The correction is BIDIRECTIONAL (required by the genuine (i,j)
    # mixing, codex-found):
    #
    #  * ``net_water > 0`` — the kernel created spurious condensate; remove
    #    it by scaling ``dq_c`` DOWN (the legacy surrogate case).
    #  * ``net_water < 0`` — convection removed MORE vapor than it produced
    #    as retained in-cloud condensate.  In the genuine Emanuel mixer this
    #    is the reference precipitation path: EP·CLW feeds the oracle's
    #    precipitation/downdraft budget and must NOT be reinserted as cloud
    #    water, or the RCE column accumulates radiatively active anvils.  The
    #    legacy mass-flux-kernel surrogate lacks that precipitation split, so
    #    it still gets the old bookkeeping add-back for compatibility.
    #
    # Both branches are smooth/AD-safe: the down-scale is a clipped ratio,
    # the up-add distributes ``|net_water|`` by the (non-negative) drying
    # weight ``max(-dq_v, 0)`` normalised over the column.
    deficit = jnp.where(
        config.use_genuine_mixing,
        0.0,
        jnp.maximum(-net_water, 0.0),
    )                                                     # legacy add-back only
    drying = jnp.maximum(-dq_v_dt, 0.0)                   # where vapor removed
    drying_col = jnp.sum(drying * dp, axis=-1) / constants.g
    add_weight = drying / jnp.maximum(drying_col[:, None], 1e-30)  # ∫w dp/g = 1
    dq_c_add = deficit[:, None] * add_weight              # ∫ dq_c_add dp/g = deficit

    # THE PRECIPITATING BRANCH (the water this scheme used to destroy).
    #
    # On the genuine mixing path the negative net water IS the oracle's
    # ``EP·CLW``: the precipitating fraction of the adiabatic condensate, which
    # ``QTI = Q(NK) − EP·CLW`` removes from the updraught's total water and
    # which CONVECT hands to ``WDTRAIN`` → ``PRECIP``.  Before this it was
    # returned to NOTHING — ``deficit`` is pinned to 0 above, the receiving
    # unsaturated downdraft ships OFF, and its ``precip_mm_day`` was consumed by
    # no one even when ON.  MEASURED in a 100-day RCEMIP1 column: evaporation
    # 2.41 mm/day (correct, the CRM's own value) against precipitation 0.28
    # (CRM 2.4) with the column in steady state, i.e. 2.13 mm/day of water
    # destroyed, ≈61 W/m² of spurious latent sink.  It is invisible in the
    # temperature field because the latent heat was already released during
    # ascent, which is why it survived so long.
    #
    # It is emitted as ``dq_r_conv_dt``, the in-updraught rain source the
    # bechtold / tiedtke / mass_flux schemes already use for exactly this
    # quantity (their ``precip_efficiency``-diverted condensate, #929); the
    # physics pipeline column-integrates it into same-step surface precip.
    # Deliberately NOT routed into ``dq_c_conv_dt``: ``EP`` already
    # parameterises autoconversion, so the cloud route would convert it a
    # second time, delay it by the host's autoconversion timescale, and load
    # radiatively active anvil condensate the comment below rightly warns of.
    # NO heating accompanies it — the latent heat is already in ``dT_dt``, and
    # adding more would double-count L_v.
    dq_r_conv_dt = jnp.where(
        config.use_genuine_mixing,
        jnp.maximum(-net_water, 0.0)[:, None] * add_weight,
        0.0,
    )                                                     # ∫dq_r dp/g = |deficit|
    # Down-scale only the spurious-excess case; when net_water<0 qc_scale=1.
    qc_scale = jnp.clip(
        1.0 - jnp.maximum(net_water, 0.0)
        / jnp.maximum(col_dq_c_mass, 1e-30), 0.0, 1.0,
    )
    dq_c_conv_dt = dq_c_conv_dt * qc_scale[:, None] + dq_c_add

    # (2) Vapor-side enthalpy conservation (oracle ENTS pass).  CONVECT
    # applies this correction ONLY over levels 1..INB (surface through the
    # diagnosed convection top), with the denominator PH(1)-PH(INB+1).
    # Applying a uniform full-column shift cools the quiescent stratosphere
    # every convective step and drove the SCM cold point to ~133 K.  The
    # genuine mixing path returns the smooth 1..INB mask; the legacy
    # surrogate falls back to the active mass-flux profile.
    cpn = constants.c_pd * (1.0 - q_v) + constants.c_pv * q_v
    # Kirchhoff L(T) via the shared helper, with Emanuel's CONVECT
    # tunable liquid heat capacity CL in place of ``constants.c_pw``.
    lv_eff = latent_heat_vaporization(T, c_liquid=config.c_l_emanuel)
    if config.use_genuine_mixing:
        ents_mask = mixing.convective_layer_mask
    else:
        ents_mask = jnp.where(jnp.abs(dq_v_dt) + dq_c_conv_dt > 0.0, 1.0, 0.0)
    layer_mass = jnp.sum(ents_mask * dp, axis=-1)
    mse_tend = cpn * dT_dt + lv_eff * dq_v_dt             # (ncol, nlev) [W/kg]
    ents = jnp.sum(ents_mask * mse_tend * dp, axis=-1) / jnp.maximum(
        layer_mass, 1.0,
    )
    dT_dt = dT_dt - ents_mask * (ents[:, None] / cpn)

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        dq_r_conv_dt=dq_r_conv_dt,
        cape=cape,
        convective_mask=cape_weight,
        du_dt_conv=None,
        dv_dt_conv=None,
    )
    # Carry the *prognostic* CBMF (the relaxed value before the per-step
    # ``M_b_max`` transport safety clip) so the closure's memory ramps
    # faithfully between calls.  FIDELITY: the oracle's CBMF can exceed the
    # per-step transport limit (its CBMFMAX ≈ 10·Δp·g⁻¹·Δt⁻¹ is a loose CFL
    # bound, ~several kg/m²/s) and reaches ~0.12 kg/m²/s in deep tropical
    # columns.  Clipping the carry to the much tighter transport cap
    # ``M_b_max`` (0.05) would freeze the prognostic state and destroy that
    # memory above 0.05 (codex iter-1 #3).  We bound the carry to a
    # separate, looser ``cbmf_carry_max`` (≈ the oracle CBMFMAX scale) only
    # to prevent an unphysical runaway; the actual plume transport is still
    # bounded at ``M_b_max`` via ``M_b`` above.
    cbmf_carry = jnp.clip(cbmf_new, 0.0, config.cbmf_carry_max)
    conv_prog_profile_new = (
        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(cbmf_carry)
    )
    return out, conv_prog_profile_new
