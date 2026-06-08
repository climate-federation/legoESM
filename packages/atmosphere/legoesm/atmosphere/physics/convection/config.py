"""Configuration for atmospheric convection schemes.

Provides configuration NamedTuples for:
1. Simplified Betts-Miller (SBM) — relaxation-based convection (Frierson 2007)
2. Deep Convective Adjustment (DCA) — simplest baseline adjustment
3. Kuo — column moisture-excess (Kuo 1965/1974)
4. Prognostic Mass-Flux — Arakawa-Wu type (1 prognostic var: M_c)
5. Simplified EDMF — eddy-diffusivity mass-flux (1 prognostic var: a_u)
6. Top-level ConvectionConfig that selects the active scheme.

References
----------
- Frierson, D. M. W. (2007). The Dynamics of Idealized Convection
  Schemes and Their Effect on the Zonally Averaged Tropical Circulation.
  J. Atmos. Sci., 64, 1959-1976.
- Kuo, H. L. (1974). Further studies of the parameterization of the
  influence of cumulus convection on large-scale flow. J. Atmos. Sci.,
  31, 1232-1240.
- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
  moist convection in numerical modeling of the atmosphere. Part I.
  J. Atmos. Sci., 70, 1977-1992.
"""

from __future__ import annotations

from typing import NamedTuple


class SBMConfig(NamedTuple):
    """Configuration for Simplified Betts-Miller convection.

    Fields
    ------
    tau_c : float
        Relaxation timescale [s] (default 7200 = 2 hours).
    RH_ref : float
        Reference relative humidity for moisture profile (default 0.7).
    CAPE_threshold : float
        Minimum CAPE [J/kg] to trigger convection (default 70.0).
    T_min_convect : float
        Minimum temperature [K] for convection (default 200.0).
    smooth_trigger_sharpness : float
        Sigmoid sharpness for smooth trigger [1/(J/kg)] (default 0.01).
    """
    tau_c: float = 7200.0
    RH_ref: float = 0.7
    CAPE_threshold: float = 70.0
    T_min_convect: float = 200.0
    # Default 0.1 (was 0.01).  At CAPE=0 the looser 0.01 gives
    # ``sigmoid(0.01·-70) ≈ 0.33`` (33 % activation when CAPE is zero
    # — gating leaks).  0.1 gives ``sigmoid(-7) ≈ 9e-4`` (effectively 0)
    # while preserving smoothness near the threshold.
    smooth_trigger_sharpness: float = 0.1
    # Sigmoid sharpness for the cloud-layer mask (T_moist - T) [1/K].
    # Hard ``T_moist >= T`` boolean masks kill ``jax.grad`` through layer
    # top/bottom transitions; the smoothed sigmoid keeps gradients alive.
    # 5/K gives ~0.5 at T_moist == T and ~0.01 at T_moist - T = -1 K, which
    # is sharp enough to behave like a hard mask in forward integration but
    # differentiable for training.
    cloud_mask_sharpness: float = 5.0


class DCAConfig(NamedTuple):
    """Configuration for Deep Convective Adjustment.

    Fields
    ------
    n_iterations : int
        Number of bottom-to-top adjustment sweeps per call (default 3).
        A single sweep only partially relaxes a deep column toward the
        moist adiabat, so one call per physics step leaves the free
        troposphere several K too cold under steady radiative cooling;
        free-tropospheric ``mean|T - T_moist|`` falls monotonically with
        sweeps (≈9.0/8.4/7.3/5.5 K at 1/3/5/10) with the column maximum
        temperature unchanged (the simultaneous pair solve keeps every
        sweep enthalpy-conserving and bounded).
    mixing_fraction : float
        Fraction of adjustment applied per iteration (default 1.0).
    cape_threshold : float
        Minimum CAPE [J/kg] to trigger convection (default 100.0).
        Columns with CAPE below this are not adjusted.
    cape_sharpness : float
        Sigmoid sharpness [1/(J/kg)] for smooth CAPE gating (default 0.02).
    instability_blend_sharpness : float
        Dimensionless sigmoid sharpness on the superadiabatic-instability
        metric controlling per-pair adjustment blending inside the
        ``lax.scan`` sweep (default 10.0).  Lifted from a hardcoded literal
        so the trigger transition width is tunable.
    """
    n_iterations: int = 3
    mixing_fraction: float = 1.0
    cape_threshold: float = 100.0
    cape_sharpness: float = 0.1   # sigmoid(-10)≈5e-5 at CAPE=0; 0.5 at threshold
    instability_blend_sharpness: float = 10.0


class KuoConfig(NamedTuple):
    """Configuration for Kuo column moisture-excess convection.

    Fields
    ------
    alpha_heat : float
        Fraction of column moisture excess going to heating vs moistening.
    me_threshold : float
        Minimum column moisture excess to trigger convection [kg/m^2].
    smooth_trigger_sharpness : float
        Sigmoid sharpness on column moisture excess trigger [1/(kg/m^2)].
    tau_relax_s : float
        Relaxation timescale [s].  Default 7200 (2 h).  The earlier
        default 3600 (1 h) ate the entire column moisture excess every
        hour, which combined with the surface-evap supply rate gave
        ~10× too much precipitation in tropical RCE.  CCM2/CCM3 used
        21600 (6 h); 7200 is a compromise that keeps the scheme
        responsive to real precipitating columns without
        over-precipitating.
    """
    alpha_heat: float = 0.75
    me_threshold: float = 1e-5
    smooth_trigger_sharpness: float = 1e4
    tau_relax_s: float = 7200.0


class MassFluxConfig(NamedTuple):
    """Configuration for Prognostic Mass-Flux convection (Arakawa-Wu type).

    Fields
    ------
    tau_adj : float
        Mass flux relaxation timescale [s].
    epsilon_0 : float
        Entrainment rate [1/m].
    delta_0 : float
        Detrainment rate [1/m].
    M_scale : float
        Equilibrium mass flux scale [kg/m^2/s].
    cape_activation_scale : float
        Sigmoid scale for CAPE trigger [J/kg].
    cape_threshold : float
        CAPE threshold [J/kg].
    M_c_init : float
        Initial base mass flux [kg/m^2/s].
    """
    tau_adj: float = 3600.0
    epsilon_0: float = 1e-3
    delta_0: float = 1e-3
    M_scale: float = 0.01
    # Trigger: ``convective_mask = sigmoid((cape − cape_threshold)
    # / cape_activation_scale)``.  Defaults below give ``sigmoid(-7)
    # ≈ 9e-4`` at CAPE=0 (effectively no firing) and full activation
    # 100 J/kg above the 70 J/kg threshold.  The earlier defaults
    # (``cape_threshold=0``, ``cape_activation_scale=100``) gave 50 %
    # activation at CAPE=0 — the gating-leak that the validation
    # script flags.
    cape_activation_scale: float = 10.0
    cape_threshold: float = 70.0
    M_c_init: float = 0.0
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max


class ZhangMcFarlaneConfig(NamedTuple):
    """Configuration for the Zhang & McFarlane (1995) deep-convection scheme.

    Single-plume mass-flux scheme with CAPE-relaxation closure and
    optional Gregory et al. 1997 convective momentum transport.

    Fields
    ------
    tau_cape : float
        CAPE relaxation timescale [s] (default 3600 = 1 hour).
    cape_threshold : float
        CAPE threshold [J/kg] below which convection is suppressed
        (default 70.0, the classic ZM 1995 value).
    cape_sharpness : float
        Sigmoid sharpness on the CAPE trigger [1/(J/kg)].  Default
        ``0.02`` gives ~95% activation 100 J/kg above threshold.
    parcel_dT : float
        Sub-cloud parcel temperature perturbation [K] (default 0.5).
    parcel_dq : float
        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
    epsilon_0 : float
        Bulk-plume entrainment rate [1/m] (default 1e-3).
    delta_0 : float
        Bulk-plume detrainment rate [1/m] (default 1e-3).
    enable_cmt : bool
        Whether to compute convective momentum transport tendencies
        (default ``True``).  When ``False``, the bridge zero-fills
        ``du_dt_conv`` / ``dv_dt_conv``.
    cmt_c_u, cmt_c_d : float
        Pressure-gradient correction coefficients in the Gregory et
        al. 1997 closure (default 0.55 each — the canonical value).
    M_b_max : float
        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).  The
        CAPE/τ_cape closure is unbounded above; without this cap a
        column with CAPE >> 5 kJ/kg yields M_b that drives
        column-integrated heating > 10⁴ W/m² and blows up the
        integration on the next dynamics step.
    """
    tau_cape: float = 3600.0
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    parcel_dT: float = 0.5
    parcel_dq: float = 1.0e-3
    epsilon_0: float = 1.0e-3
    delta_0: float = 1.0e-3
    # --- FAITHFUL ZM dilute-parcel CAPE (Raymond-Blyth 1992) -----------
    # The ZM trigger and closure use the CAPE of a DILUTE entraining
    # plume (``buoyan_dilute``/``parcel_dilute`` in zm_conv.F90), not an
    # undilute moist adiabat.  ``dmpdz`` is the fractional entrainment
    # rate [1/m] (E3SM/CAM default ``−1.0e-3``; NEGATIVE by the oracle's
    # ``mp`` sign convention).  ``tiedke_add`` is the buoyancy offset [K]
    # added at every cloud level (oracle default 0.5 K).  ``tp_fac`` ×
    # ``parcel_tpert`` is the optional PBL temperature perturbation (both
    # default 0).  ``pbl_top_pa`` bounds the launch-level (max-MSE)
    # search to the PBL.  Set ``use_dilute_cape = False`` to recover the
    # legacy undilute moist-adiabat CAPE.
    use_dilute_cape: bool = True
    dmpdz: float = -1.0e-3
    tiedke_add: float = 0.5
    tp_fac: float = 0.0
    parcel_tpert: float = 0.0
    pbl_top_pa: float = 7.0e4
    enable_cmt: bool = True
    cmt_c_u: float = 0.55
    cmt_c_d: float = 0.55
    M_b_max: float = 0.05
    # Buoyancy-death memory is OFF by default.  The audit's cycle-2 P2
    # concern (a plume terminated by negative buoyancy can revive
    # above an inversion) is real, but several attempted detector
    # designs each introduced their own edge cases (sub-LCL warm-
    # bubble leak, miss of weak positive CAPE, miss of cloud-base
    # launch, growth of the launched gate during revival).  The
    # ``_plume.entraining_detraining_plume`` ``buoyancy_death_memory``
    # kwarg is wired through and the option is fully tested in
    # isolation, but enabling it by default would require a more
    # robust state-machine design plus a dedicated validation
    # campaign.  Schemes that *want* the single-plume monotone
    # termination can opt in by setting this to True; the default
    # preserves the legacy local-only filter behaviour that all
    # existing scheme test fixtures were calibrated against.
    buoyancy_death_memory: bool = False


class KainFritschConfig(NamedTuple):
    """Configuration for the Kain & Fritsch (1990, 2004 update) scheme.

    Single-plume bulk mass-flux scheme distinguished by its
    boundary-layer trigger function: convection fires when the
    perturbed parcel temperature at the LCL exceeds the environmental
    temperature at the LCL.  The trigger is smoothed via a sigmoid
    (``trigger_sharpness``) to preserve gradients.  Deep-vs-shallow
    cloud branches are blended on cloud depth.  No convective
    momentum transport — KF emits ``du_dt_conv = dv_dt_conv = None``.

    Fields
    ------
    w_thresh_offset : float
        Trigger offset [K] (default 2.0; the canonical KF 1990 value).
    w_thresh_scale : float
        Conversion factor from ``w_grid`` [m/s] to a temperature
        perturbation [K] in the trigger function.  Default 1.0 K per
        m/s — the dimensionful scaling depends on resolution; users
        with grid-scale ``w`` available should tune this.
    trigger_sharpness : float
        Sigmoid sharpness on the trigger threshold [1/K].  Larger
        values approach a hard ``> 0`` step; smaller values broaden
        the transition.  Default 5.0 → ~95% activation 0.6 K above
        threshold.
    cape_consumption_time : float
        CAPE-removal timescale [s] (default 1800.0).
    parcel_perturb_T : float
        Sub-cloud parcel temperature perturbation [K] (default 0.5).
    parcel_perturb_q : float
        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
    epsilon_0 : float
        Bulk-plume entrainment rate [1/m] (default 2e-3).
    delta_0 : float
        Bulk-plume detrainment rate [1/m] (default 2e-3).
    cloud_depth_min : float
        Cloud-depth threshold [m] separating shallow and deep
        branches (default 4000.0).
    cloud_depth_sharpness : float
        Sigmoid sharpness on the deep/shallow blend [1/m] (default
        1e-3 — ~95% activation 1500 m above threshold).
    enable_shallow : bool
        Whether to include the shallow-cloud branch.  ``False``
        disables shallow tendencies regardless of cloud depth
        (default ``True``).
    cape_threshold : float
        CAPE threshold [J/kg] below which convection is gated off.
        Default 0.0 — KF gates primarily on the trigger function,
        not on CAPE.
    cape_sharpness : float
        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
    M_b_max : float
        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).
    """
    w_thresh_offset: float = 2.0
    w_thresh_scale: float = 1.0
    trigger_sharpness: float = 5.0
    cape_consumption_time: float = 1800.0
    parcel_perturb_T: float = 0.5
    parcel_perturb_q: float = 1.0e-3
    # Deep-convection entrainment/detrainment (~2e-4 /m).  The earlier
    # ``2e-3`` (shallow-cumulus range) over-diluted the single plume so it
    # lost buoyancy in the lower troposphere and could not warm the free
    # troposphere — see ZhangMcFarlaneConfig / EmanuelConfig.
    epsilon_0: float = 2.0e-4
    delta_0: float = 2.0e-4
    cloud_depth_min: float = 4000.0
    cloud_depth_sharpness: float = 1.0e-3
    enable_shallow: bool = True
    # Single-plume scheme — see ZhangMcFarlaneConfig.buoyancy_death_memory.
    buoyancy_death_memory: bool = False
    cape_threshold: float = 0.0
    cape_sharpness: float = 0.1
    # CAPE-based OR fallback for the dynamical trigger.  Kain-Fritsch fires
    # where resolved grid-scale ascent lifts a parcel past its LCL
    # (``T_lcl + w_thresh_scale·w_grid − w_thresh_offset > T_env``).  In a
    # single-column model (and any dycore that does not expose a divergence
    # operator) ``w_grid`` is zero, so the 2 K ``w_thresh_offset`` becomes a
    # permanent suppression and the scheme never fires — leaving the column
    # in near-radiative equilibrium.  When the undilute CAPE exceeds
    # ``cape_or_threshold`` the trigger fires regardless of ``w_grid``.  The
    # threshold is set to a deliberately EXTREME value (2000 J/kg, deep
    # maritime-tropical CAPE) with a tight sigmoid so the fallback is a
    # near-no-op for the moderate-CAPE columns of a 3-D run (≈0.007 at
    # 1000 J/kg, 0.5 at 2000) — those columns are handled by the resolved
    # w-trigger, and any 3-D column carrying ≳2000 J/kg essentially always
    # has the resolved ascent to satisfy it anyway.  The branch therefore
    # only rescues the ``w = 0`` SCM/idealised case (RCE CAPE ~ 10⁴ J/kg).
    # Set ``cape_or_threshold = inf`` to disable and recover the pure
    # w-trigger behaviour exactly.
    cape_or_threshold: float = 2000.0
    cape_or_sharpness: float = 0.005
    # The CAPE-OR fallback is itself gated by the ABSENCE of resolved
    # grid-scale ascent, ``exp(-(w_grid_at_lcl / cape_or_w_ref)^2)``, so it
    # only engages where ``w_grid ≈ 0`` (SCM, or a dycore with no
    # divergence operator).  Wherever the bridge supplies a non-negligible
    # ``w_grid`` (any 3-D run with resolved ascent) the gate →0 and the
    # pure w-trigger is used unchanged — so KF's documented response to
    # resolved divergence is preserved exactly.  ``cape_or_w_ref`` is the
    # vertical-velocity scale [m/s] at which the fallback is suppressed.
    cape_or_w_ref: float = 0.02
    M_b_max: float = 0.05


class EmanuelConfig(NamedTuple):
    """Configuration for the Emanuel (1991) buoyancy-sorting scheme.

    Single-plume mass-flux scheme with a buoyancy-sorted ensemble of
    mixed parcels: at every cloud level the parcel may mix with
    environmental air in a discrete spectrum of mixing fractions; the
    fractions with positive buoyancy continue to ascend while the
    fractions with negative buoyancy descend.  The smooth-everywhere
    formulation replaces the hard ascend/descend switch with a
    sigmoid weighting on buoyancy.

    Distinct from Zhang-McFarlane (single bulk plume) and Kain-Fritsch
    (single plume with deep/shallow blend) by the per-level
    distribution of detrainment that the buoyancy-sorted ensemble
    produces.

    Fields
    ------
    n_mixing_fractions : int
        Number of discrete mixing fractions in the buoyancy-sort
        ensemble.  Default 8 — Emanuel 1991 uses 50; the smaller
        value here is a cost / accuracy compromise.
    cu_coefficient : float
        Entrainment scale factor (Emanuel's α).  Default 0.7.
    precip_efficiency_water : float
        Precipitation efficiency above LCL (default 1.0).
    precip_efficiency_lcl : float
        Precipitation efficiency below LCL (default 0.0).
    precip_threshold_qc : float
        Cloud-water threshold above which precipitation falls
        [kg/kg] (default 1e-3).
    cape_threshold : float
        CAPE gate [J/kg] (default 70.0).
    cape_sharpness : float
        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
    parcel_perturb_T : float
        Sub-cloud parcel temperature perturbation [K] (default 0.5).
    parcel_perturb_q : float
        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
    sub_cloud_relaxation : float
        Sub-cloud layer mixing timescale [s] (default 100.0).
    enable_unsaturated_downdraft : bool
        Whether to include the unsaturated downdraft branch (rain
        evaporation cooling) (default ``True``).
    downdraft_efficiency : float
        Fraction of precipitation that re-evaporates below cloud base
        in the downdraft (default 0.2).
    smooth_trigger_sharpness : float
        Sigmoid sharpness on the buoyancy-sort weighting [1/K]
        (default 0.5).
    epsilon_0 : float
        Bulk-plume entrainment rate [1/m] (default 1.5e-3).
    delta_0 : float
        Bulk-plume detrainment rate [1/m] (default 1.5e-3).
    M_b_max : float
        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).
    """
    n_mixing_fractions: int = 8
    cu_coefficient: float = 0.7
    precip_efficiency_water: float = 1.0
    precip_efficiency_lcl: float = 0.0
    precip_threshold_qc: float = 1.0e-3
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    parcel_perturb_T: float = 0.5
    parcel_perturb_q: float = 1.0e-3
    # Emanuel 1991 §3 uses a sub-cloud-layer mixing timescale of
    # several thousand seconds.  The earlier default of 100 s gave
    # M_b ~72× larger than published values and produced 28 MW/m²
    # of column heating from a CAPE-positive sounding.  NOTE: with the
    # faithful prognostic DTMA closure (``alpha_closure`` / ``damp_*``)
    # this field is no longer read by ``emanuel_convection``; it is kept
    # for back-compat with configs/tests that set it.
    sub_cloud_relaxation: float = 7200.0
    # --- Prognostic cloud-base mass-flux (CBMF) closure ----------------
    # FAITHFUL to oracle convect43c.f (CONVECT v4.3c).  CBMF is a
    # prognostic quantity relaxed each call toward the sub-cloud
    # quasi-equilibrium:
    #     CBMF = (1 - DAMP·dt/300)·CBMF_old + 0.1·ALPHA·DTMA   (≥0)
    # ``alpha_closure`` (ALPHA) and ``damp_coefficient`` (DAMP) are the
    # oracle's standard values 0.2 and 0.1 (DAMP < 1).  ``dtmax`` (DTMAX)
    # is the maximum negative temperature perturbation [K] a lifted
    # parcel is allowed below its LFC (oracle 0.9 K).
    alpha_closure: float = 0.2
    damp_coefficient: float = 0.1
    dtmax: float = 0.9
    # Width [levels] of the smooth cloud-base-level selector used to read
    # the parcel buoyancy excess at the LCL for the DTMA closure.  A
    # narrow Gaussian (≈1 level) localises the buoyancy to cloud base
    # while staying differentiable.
    cloud_base_index_width: float = 1.0
    # Sharpness [1/(kg/m²/s)] of the softplus positive-part applied to the
    # relaxed CBMF so it is ~0 when the relaxation target goes negative
    # (stable column) without a hard ``max`` that would kill the gradient.
    # Large because CBMF magnitudes are O(0.01-0.1) kg/m²/s.
    cbmf_positive_sharpness: float = 1.0e3
    # Upper bound [kg/m²/s] on the *carried* prognostic CBMF — looser than
    # the per-step transport cap ``M_b_max`` so the closure's memory can
    # ramp to the oracle's deep-tropical CBMF (~0.12 kg/m²/s) instead of
    # being frozen at ``M_b_max``.  Set well above the oracle peak; the
    # plume transport is still bounded at ``M_b_max``.  Acts only as an
    # anti-runaway guard for a persistently violently-unstable column.
    cbmf_carry_max: float = 0.3
    # Default-OFF.  Emanuel 1991's downdraft re-evaporates a fraction
    # of *precipitation* (rain) back to vapor in the BL.  In a model
    # without an explicit q_r tracer the implementation can only draw
    # from ``dq_c_conv_dt`` (the cloud-water source), so enabling it
    # produces a column-net moistening on CAPE-positive soundings —
    # the wrong sign of ``Q_v`` that the validation script flags.
    # Production runs with a full microphysics chain that owns q_r
    # should override this to ``True``.
    enable_unsaturated_downdraft: bool = False
    downdraft_efficiency: float = 0.2
    smooth_trigger_sharpness: float = 0.5
    # Bulk-plume entrainment/detrainment for the cloud-base updraft.
    # Deep-convection value (~2e-4 /m): in Emanuel's scheme the
    # cloud-environment mixing is represented explicitly by the
    # buoyancy-sorted ensemble (``n_mixing_fractions``), so the bulk
    # ascent should be near-undilute.  The earlier ``1.5e-3`` (shallow-
    # cumulus range) double-counted dilution — it over-entrained the
    # bulk plume on top of the ensemble mixing, collapsing the updraft
    # buoyancy in the lower troposphere so deep convection could not
    # warm the free troposphere (anti-convective, super-adiabatic,
    # ~30 K-too-cold RCE).
    epsilon_0: float = 2.0e-4
    delta_0: float = 2.0e-4
    M_b_max: float = 0.05
    # --- GENUINE (i,j) episodic-mixing buoyancy-sort spectrum ----------
    # Faithful port of the Fortran CONVECT v4.3c SIJ/ELIJ/MENT mixing
    # matrix (convect43c.f lines 588-712).  When ``use_genuine_mixing``
    # is True (default) the scheme builds the full ``(nlev, nlev)``
    # mixing matrix — every origin level i mixes with environment air in
    # the neutral-buoyancy fraction spectrum, each mixture's buoyancy
    # sets its detrainment level j, and the environmental tendencies are
    # assembled from MENT(i,j).  When False it falls back to the legacy
    # single-sigmoid ``_mixture_buoyancy`` surrogate (kept for back-compat
    # / ablation).  See ``_emanuel_mixing.py``.
    use_genuine_mixing: bool = True
    # Emanuel's effective liquid-water heat capacity CL [J/kg/K] (oracle
    # value 2500).  This is a scheme-internal thermodynamic coefficient
    # in CONVECT's liquid-water-static-energy formulation, distinct from
    # the canonical ``constants.c_pw`` (4218 J/kg/K at standard
    # conditions); kept here so the SIJ/ELIJ algebra matches the oracle
    # term-for-term rather than monkey-patching a global constant.
    c_l_emanuel: float = 2500.0
    # Autoconversion threshold ELCRIT [kg/kg] and critical temperature
    # TLCRIT [degC] for the precipitation efficiency EP (oracle .0011 /
    # -55.0).
    elcrit: float = 1.1e-3
    tlcrit: float = -55.0
    # Mixing-rate coefficient ENTP in M(i) (oracle 1.5).
    entp: float = 1.5
    # SIGD / SIGS — fractional area of unsaturated downdraught / fraction
    # of precip falling outside cloud (oracle 0.05 / 0.12).  Kept as
    # config for the downdraught bookkeeping in the orchestrator.
    sigd: float = 0.05
    sigs: float = 0.12
    # Rain / snow evaporation coefficients COEFFR / COEFFS and the CU
    # momentum-transport coefficient + BETA downdraught velocity scale
    # (oracle 1.0 / 0.8 / 0.7 / 10.0).  Threaded for completeness of the
    # precip-downdraught handoff; the model owns precip via q_c.
    coeffr: float = 1.0
    coeffs: float = 0.8
    cu_momentum: float = 0.7
    beta_downdraft: float = 10.0
    # --- Smoothing sharpnesses for the discrete sort (AD-safety) -------
    # Each replaces a hard Fortran switch with a smooth surrogate; the
    # forward result tracks the discrete sort to a stated tolerance (see
    # the oracle-vs-ours mixing-matrix comparison in
    # ``.physics-validator/emanuel``).
    # Sigmoid sharpness [1/level] for the ICB/INB cloud-layer windows.
    level_window_sharpness: float = 6.0
    # Sigmoid sharpness [dimensionless] for the ``0 < SIJ < 0.9``
    # entrainment gate (oracle counts a mixture only inside this band).
    sij_gate_sharpness: float = 40.0
    # Upper SIJ gate (oracle 0.9).
    sij_upper_gate: float = 0.9
    # Magnitude floor for the SIJ denominator (oracle ``ABS(DENOM)<0.01``).
    denom_floor: float = 0.01
    # Offset [Pa] for the smooth max-MSE (NK) source-level selection.
    mse_min_search_offset: float = 5.0e4
    # Sigmoid sharpness for the saturated-mixture re-solve switch
    # (oracle ``SIJ<0 .or. SIJ>1 .or. ALTEM>CWAT``).
    sat_branch_sharpness: float = 100.0
    # Sharpness [1/level] for the STRICT integer-index inequalities
    # (``j>i``, ``k<i``, AMP1/AD ``j>t``/``k<t``).  These compare integer
    # level indices, so the 0.5-shifted sigmoid is evaluated at half-integer
    # arguments; a high sharpness makes it ≈binary (σ(±10)≈4.5e-5 at the
    # diagonal) so the strict ``J.GT.I`` / ``K=1,I-1`` Fortran bounds do not
    # leak onto the diagonal (codex iter-5 #2).  Distinct from the FRACTIONAL
    # ``level_window_sharpness`` (ICB/INB cloud edges), which must stay
    # moderate to keep the cloud-top/base transition differentiable.
    strict_index_sharpness: float = 20.0


class TiedtkeConfig(NamedTuple):
    """Configuration for the Tiedtke (1989) bulk mass-flux scheme.

    Three-class scheme with deep, mid-level, and shallow branches
    blended on cloud depth.  Downdraft is included with an RH-based
    trigger; convective momentum transport via Gregory et al. 1997.
    First scheme that actually exercises the new ``(ncol, nlev)``
    profile carry for ``M_u(k)``.

    The full Tiedtke 1989 closure uses column moisture convergence
    for the deep branch.  Until the PR-0 ``compute_moisture_convergence``
    diagnostic ships, we use a saturation-deficit proxy
    ``MC_proxy = (q_sat - q_v) / tau_relax_s`` that has the same
    qualitative behavior (positive in moist columns, zero in dry
    columns).

    Fields
    ------
    epsilon_deep, delta_deep : float
        Entrainment / detrainment rates for deep branch [1/m].
    epsilon_shallow, delta_shallow : float
        Same for shallow branch.
    epsilon_midlevel, delta_midlevel : float
        Same for mid-level branch.
    enable_downdraft : bool
        Whether to include the downdraft branch (default ``True``).
    downdraft_alpha : float
        Downdraft / updraft mass flux ratio at LFS (default 0.3).
    downdraft_RH_min : float
        Below this column-mean RH the downdraft fires (default 0.2).
    moisture_convergence_threshold : float
        Saturation-deficit proxy threshold [kg/kg/s].  Below this the
        deep branch is suppressed (default 1e-8).
    moisture_convergence_sharpness : float
        Sigmoid sharpness on the MC threshold [s/(kg/kg)] (default 1e8).
    cape_threshold : float
        Secondary CAPE gate [J/kg] (default 70.0).
    cape_sharpness : float
        Sigmoid sharpness on CAPE gate (default 0.02).
    cloud_depth_deep : float
        Depth threshold separating mid-level from deep branches [m]
        (default 3000.0).
    cloud_depth_shallow_max : float
        Depth threshold separating shallow from mid-level branches
        [m] (default 1500.0).
    depth_split_sharpness : float
        Sigmoid sharpness on the depth thresholds [1/m] (default 1e-3).
    enable_cmt : bool
        Whether to compute CMT (default ``True``).
    cmt_c_u, cmt_c_d : float
        Gregory et al. 1997 closure coefficients (default 0.7).
    smooth_trigger_sharpness : float
        Sigmoid sharpness on the buoyancy / RH soft triggers (default
        0.02).
    tau_M_u_relax : float
        Implicit-Euler relaxation timescale [s] for the profile carry
        ``M_u`` toward its diagnosed equilibrium (default 1800.0).
    parcel_dT, parcel_dq : float
        Sub-cloud parcel perturbations (defaults 0.5 K, 1e-3 kg/kg).
    """
    epsilon_deep: float = 1.0e-4
    delta_deep: float = 1.0e-4
    epsilon_shallow: float = 3.0e-4
    delta_shallow: float = 3.0e-4
    epsilon_midlevel: float = 1.0e-4
    delta_midlevel: float = 2.0e-4
    enable_downdraft: bool = True
    downdraft_alpha: float = 0.3
    downdraft_RH_min: float = 0.2
    # Fraction of the downdraft mass flux that re-evaporates as rain
    # falling through the subcloud layer (default 0.05 — matches a
    # historical hardcoded literal that was previously dimensionally
    # wrong; the current implementation distributes the resulting
    # evaporation rate over below-LCL layers by mass weight, with a
    # matching dq_v source so the column water budget closes).
    downdraft_evap_efficiency: float = 0.05
    moisture_convergence_threshold: float = 1.0e-8
    moisture_convergence_sharpness: float = 1.0e8
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    cloud_depth_deep: float = 3000.0
    cloud_depth_shallow_max: float = 1500.0
    depth_split_sharpness: float = 1.0e-3
    enable_cmt: bool = True
    cmt_c_u: float = 0.7
    cmt_c_d: float = 0.7
    smooth_trigger_sharpness: float = 0.02
    tau_M_u_relax: float = 1800.0
    parcel_dT: float = 0.5
    parcel_dq: float = 1.0e-3
    tau_MC_proxy: float = 3600.0   # for saturation-excess MC proxy
    # Critical column-mean RH above which the MC proxy starts firing.
    # The proxy approximates moisture convergence as the column-integrated
    # vapor in excess of ``RH_crit * q_sat``: positive in moist columns,
    # vanishing in dry ones.
    mc_proxy_RH_crit: float = 0.6
    tau_shallow_M_b: float = 3600.0  # Shallow-cloud-base mass-flux timescale [s]
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
    midlevel_M_b_fraction: float = 0.5  # M_b_midlevel = M_b_shallow * this
    # Single-plume scheme — see ZhangMcFarlaneConfig.buoyancy_death_memory.
    buoyancy_death_memory: bool = False


class BechtoldConfig(NamedTuple):
    """Configuration for the Bechtold/IFS convection scheme.

    Builds on :class:`TiedtkeConfig` (three-class blend, downdraft,
    CMT) with two distinguishing features:

    * **PBL-CAPE / departure-CAPE closure** (Bechtold 2008):
      ``M_b ∝ (CAPE_pbl - CAPE_eq)+ / tau_bl`` where ``CAPE_pbl`` is
      diagnosed from a mass-weighted parcel within the boundary layer
      rather than the surface parcel.
    * **AR1 stochastic perturbation** (Bechtold 2014): ``M_b *= (1 +
      amplitude * ε)`` where ``ε`` is an AR1-process realization with
      decorrelation timescale ``stochastic_decorrelation``.  When
      ``enable_stochastic`` is ``False`` the multiplier is 1.

    Stochasticity defaults to OFF for reproducibility.  When enabled,
    the leaf consumes a ``prng_key`` argument; the convection bridge
    splits ``PhysicsState.prng_key`` into a Bechtold sub-key (folded
    with module id ``0xBEC4``) and an advanced master key, returning
    the latter as part of the multi-field carry update so subsequent
    steps see independent random streams.

    Inherits sensible defaults from Tiedtke 1989 with the entrainment
    revision from Bechtold et al. 2008.

    Fields
    ------
    epsilon_deep, delta_deep : float
        Deep-branch base entrainment / detrainment [1/m].  Default
        1.75e-3 / 0.75e-4 (IFS Cy49r1 Part IV Ch.6).  ``bechtold.py``
        multiplies these by the IFS height factors
        ``(1.3 − RH)·(q_sat/q_sat_base)³`` (entrainment) and
        ``(1.6 − RH)`` (detrainment); the f_scale decay — not a small
        constant ε — is what makes the deep plume penetrate.
    epsilon_shallow, delta_shallow : float
        Shallow-branch base rates [1/m] (default 3.5e-3 = 2× deep ε via
        the IFS f_ε factor, 0.75e-4).  Note: IFS ties shallow detrainment
        to the shallow entrainment (``D_shallow = E_shallow·(1.6 − RH)``);
        ``bechtold.py`` keeps the simpler prescribed-δ₀ shallow form for
        conservation, so ``delta_shallow`` is the shallow detrainment
        base directly.
    epsilon_midlevel, delta_midlevel : float
        Mid-level branch [1/m] (default 1e-4, 2e-4).
    cape_pbl_depth : float
        PBL depth [m] for the parcel-source mass weighting (default
        500.0).
    tau_bl : float
        PBL closure timescale [s] (default 3600.0).
    enable_stochastic : bool
        Whether to apply the AR1 stochastic perturbation (default
        ``False`` — reproducibility).
    stochastic_amplitude : float
        Multiplicative perturbation amplitude (default 0.5).
    stochastic_decorrelation : float
        AR1 decorrelation timescale [s] (default 7200.0).
    enable_downdraft, downdraft_alpha, downdraft_RH_min : as Tiedtke.
    enable_cmt, cmt_c_u, cmt_c_d : as Tiedtke.
    cape_threshold, cape_sharpness, smooth_trigger_sharpness,
    parcel_dT, parcel_dq, tau_M_u_relax,
    cloud_depth_deep, cloud_depth_shallow_max, depth_split_sharpness :
        as Tiedtke.
    """
    # Tiedtke-inherited / revised.
    # IFS Cy49r1 base entrainment/detrainment rates [1/m] (Part IV Ch.6
    # eqs 6.7/6.8; ecmwf-ifs/openifs).  These are the *base* fractional
    # rates ε₀/δ₀; ``bechtold.py`` multiplies them by the IFS height
    # factors ``(1.3 − RH)·(q_sat/q_sat_base)³`` (entrainment) and
    # ``(1.6 − RH)`` (detrainment).  ε₀_deep = 1.75e-3, δ₀_deep = 0.75e-4
    # are the published IFS deep values; the f_scale decay (not a smaller
    # constant ε) is what makes the deep plume penetrate — a constant ε of
    # 1.75e-3 over-dilutes and collapses the updraught (the cold-RCE bug),
    # which is why this is now applied with the IFS vertical scaling rather
    # than as a constant.  Shallow ε₀ carries the IFS f_ε = 2 factor
    # (3.5e-3 = 2 × deep).
    epsilon_deep: float = 1.75e-3
    delta_deep: float = 0.75e-4
    epsilon_shallow: float = 3.5e-3
    delta_shallow: float = 0.75e-4
    epsilon_midlevel: float = 1.0e-4
    delta_midlevel: float = 2.0e-4
    enable_downdraft: bool = True
    downdraft_alpha: float = 0.3
    downdraft_RH_min: float = 0.2
    # See TiedtkeConfig.downdraft_evap_efficiency for definition.
    downdraft_evap_efficiency: float = 0.05
    enable_cmt: bool = True
    cmt_c_u: float = 0.7
    cmt_c_d: float = 0.7
    # The earlier ``cape_threshold = 0.0`` with ``cape_sharpness = 0.005``
    # left the closure essentially always-on (``softplus(0)/0.005 ≈ 138
    # J/kg`` of phantom CAPE even when CAPE = 0).  Match ZM/Tiedtke with a
    # meaningful 70 J/kg trigger threshold; ``cape_sharpness = 0.1``
    # [1/(J/kg)] gives a tight CAPE sigmoid around it.
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    smooth_trigger_sharpness: float = 0.02
    parcel_dT: float = 0.5
    parcel_dq: float = 1.0e-3
    tau_M_u_relax: float = 1800.0
    cloud_depth_deep: float = 3000.0
    cloud_depth_shallow_max: float = 1500.0
    depth_split_sharpness: float = 1.0e-3
    # Bechtold-specific
    use_pbl_cape: bool = True
    cape_pbl_depth: float = 500.0
    tau_bl: float = 3600.0
    enable_stochastic: bool = False
    stochastic_amplitude: float = 0.5
    stochastic_decorrelation: float = 7200.0
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
    # Strong-convergence normaliser used to make the moisture-convergence
    # enhancement an O(1) multiplier of M_b (Bechtold 2008 Fig. 2 — typical
    # tropical strong-convergence is ≈ 0.05 kg/m²/s).
    mc_normalize_scale: float = 0.05
    # Single-plume scheme — see ZhangMcFarlaneConfig.buoyancy_death_memory.
    buoyancy_death_memory: bool = False


class ConvectiveEDMFConfig(NamedTuple):
    """Configuration for simplified EDMF convection (mass-flux part only).

    Fields
    ------
    epsilon_0 : float
        Entrainment rate [1/m].
    delta_0 : float
        Detrainment rate [1/m].
    a_u_init : float
        Initial updraft area fraction.
    tau_a : float
        Relaxation timescale for a_u [s].
    w_u_min : float
        Minimum updraft velocity [m/s].
    cape_activation_scale : float
        Sigmoid scale for CAPE trigger [J/kg].
    cape_threshold : float
        CAPE threshold [J/kg].
    """
    epsilon_0: float = 2e-3
    delta_0: float = 2e-3
    a_u_init: float = 0.1
    tau_a: float = 1800.0
    w_u_min: float = 0.1
    # Trigger gating — see MassFluxConfig for rationale (sharper
    # ``cape_activation_scale`` and a 70 J/kg threshold close the
    # CAPE=0 leak from the earlier 50 % activation).
    cape_activation_scale: float = 10.0
    cape_threshold: float = 70.0
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max


class ConvectionConfig(NamedTuple):
    """Top-level convection configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active convection scheme: ``"sbm"``, ``"dca"``, ``"kuo"``,
        ``"mass_flux"``, ``"edmf"``, ``"zhang_mcfarlane"``, or
        ``"none"``.  Future PRs (KF, Emanuel, Tiedtke, Bechtold) add
        their literal here.
    sbm, dca, kuo, mass_flux, edmf, zhang_mcfarlane :
        Per-scheme configuration NamedTuples.
    update_interval_steps : int
        Recompute convection every N time steps (1 = every step).
    """
    scheme: str = "sbm"
    sbm: SBMConfig = SBMConfig()
    dca: DCAConfig = DCAConfig()
    kuo: KuoConfig = KuoConfig()
    mass_flux: MassFluxConfig = MassFluxConfig()
    edmf: ConvectiveEDMFConfig = ConvectiveEDMFConfig()
    zhang_mcfarlane: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig()
    kain_fritsch: KainFritschConfig = KainFritschConfig()
    emanuel: EmanuelConfig = EmanuelConfig()
    tiedtke: TiedtkeConfig = TiedtkeConfig()
    bechtold: BechtoldConfig = BechtoldConfig()
    update_interval_steps: int = 1
