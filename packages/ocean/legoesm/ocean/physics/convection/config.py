"""Configuration for ocean convection schemes."""

from __future__ import annotations

from typing import NamedTuple


__param_spec__ = {
    "EnhancedDiffusionConfig": {
        "scheme_key": "ocean.conv.enhanced_diffusion",
        "excluded": {
            "cfl_dt_estimate": "numerics: solver/CFL/smoothing parameter",
            "cfl_safety": "numerics: solver/CFL/smoothing parameter",
            "n2_threshold": "convention: NEMO zdfevd static-instability trigger threshold on N^2 [1/s^2] (detection noise floor, hard-threshold path only; not a trainable closure)",
            "nu_bg": "default 0 = disabled/off (enable via config, not training)",
            "nu_conv": "default 0 = disabled/off (enable via config, not training)",
            "sigmoid_sharpness": "numerics: solver/CFL/smoothing parameter",
        },
        "params": {
            "K_bg": {"units": "m2/s", "bounds": (0.0, 3e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "mixing", "reference": "convective enhanced diffusion", "shape": None},
            # Upper bound covers NEMO's own zdfevd coefficient (ORCA1
            # rn_evd = 100 m2/s): the previous (0.33, 3.0) declared a range
            # the oracle's setting sits 33x outside, i.e. two ranges for one
            # parameter, with the namelist the authoritative one.
            "K_conv": {"units": "m2/s", "bounds": (0.33, 300.0), "tunable_tier": 2, "transform": "sigmoid", "category": "mixing", "reference": "convective enhanced diffusion; NEMO zdfevd rn_evd", "shape": None},
        },
    },
    "PlumeConfig": {
        "scheme_key": "ocean.conv.plume",
        "excluded": {
            "active_sigmoid_sharpness": "numerics: solver/CFL/smoothing parameter",
            "w_plume_min": "numerics: floor/cap",
        },
        "params": {
            "T_excess": {"units": "K", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "convection", "reference": "convective plume", "shape": None},
            "alpha_plume": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "convection", "reference": "convective plume", "shape": None},
            "epsilon": {"units": "1", "bounds": (0.00033, 0.003), "tunable_tier": 2, "transform": "sigmoid", "category": "convection", "reference": "convective plume", "shape": None},
        },
    },
}


class EnhancedDiffusionConfig(NamedTuple):
    """Enhanced diffusion where N^2 < 0.

    Mirrors Oceananigans' ``ConvectiveAdjustmentVerticalDiffusivity``:
    large vertical diffusivity/viscosity wherever the column is
    statically unstable (``N² < 0``), background values elsewhere.
    Oceananigans keeps the convective **tracer diffusivity**
    (``convective_κz`` → ``K_conv``) and **momentum viscosity**
    (``convective_νz`` → ``nu_conv``) as *independent* parameters; this
    config does likewise.

    The convective diffusivity ``K_conv`` is normally large (~1 m²/s) to
    rapidly homogenize an unstable column.  When the diffusion operator
    is applied explicitly (``apply_diffusion=True`` in
    ``enhanced_diffusion_convection``), explicit-Euler stability requires
    ``K · dt / dz² ≤ 0.5``.  With ``dz ≈ 10 m`` and ``dt ≈ 3600 s`` this
    forces ``K ≤ 0.014`` — three orders of magnitude below the desired
    value.  ``cfl_dt_estimate`` and ``cfl_safety`` parameterize the
    safety cap applied internally to ``K``/``A`` along the explicit path;
    the implicit path (``apply_diffusion=False``) bypasses the cap
    because backward-Euler is unconditionally stable.

    ``nu_conv`` / ``nu_bg`` are the momentum counterparts of
    ``K_conv`` / ``K_bg``.  Defaults are ``0`` to match Oceananigans'
    ``convective_νz = 0`` (convective adjustment acts on tracers only by
    default).  Set ``nu_conv > 0`` to additionally mix momentum where
    ``N² < 0``.  Note: this differs from the pre-feature implicit path,
    which incidentally reused the tracer ``K_conv`` as a momentum
    viscosity; that side-effect is now opt-in and explicit via ``nu_conv``.

    Scope: ``nu_conv`` / ``nu_bg`` mix momentum only on the cell-centred
    cubed-sphere / A-grid (where u/v share the tracer stagger).  On the
    lat-lon C-grid the convective momentum viscosity is applied to the
    face velocities through the model's implicit (backward-Euler) vertical
    solve, so it requires ``implicit_vertical_mixing=True``; the explicit
    C-grid path raises rather than silently dropping it.  MPAS (edge-normal
    velocity) applies the convective adjustment to tracers only — its
    momentum mixing needs a TRiSK cell->edge reconstruction (follow-up).
    When KPP is the vertical-mixing scheme, the convective momentum
    viscosity is suppressed here (KPP already enhances interior momentum
    for N²<0).
    """
    K_conv: float = 1.0        # Convective tracer diffusivity κ [m^2/s]
    K_bg: float = 1e-5         # Background tracer diffusivity [m^2/s]
    nu_conv: float = 0.0       # Convective momentum viscosity ν [m^2/s]
    nu_bg: float = 0.0         # Background momentum viscosity [m^2/s]
    smooth_transition: bool = True
    sigmoid_sharpness: float = 1e6
    cfl_dt_estimate: float = 3600.0  # Reference dt for explicit-CFL cap [s]
    cfl_safety: float = 0.45         # Stability margin (≤ 0.5)
    # ----- Static-stability N^2 mode for the convective trigger -----
    # ``"insitu"`` (default, BIT-IDENTICAL legacy) / ``"insitu_signed"``: the
    #   convective trigger uses the in-situ density N^2
    #   (``eos.compute_buoyancy_frequency``). Compressibility can leave a
    #   statically-unstable column with N^2 > 0, so convection is MISSED.
    # ``"adiabatic"``: the TRUE static stability via adiabatic parcel
    #   displacement (``eos.compute_buoyancy_frequency_adiabatic``), SIGNED —
    #   a compressibility-masked unstable column then gives N^2 < 0 and the
    #   trigger fires. Requires the caller to thread T, S, cell-centre
    #   pressure ``p_cell`` (+ the model EOS) to ``convective_K_A_flag`` /
    #   ``enhanced_diffusion_convection``; both integration factory and the
    #   implicit k_profiles path supply them when this is selected.
    n2_mode: str = "insitu"
    # Which alpha/beta the ``n2_mode="nemo_bn2"`` assembly uses. ``"seos"``
    # (default, BIT-IDENTICAL legacy) is NEMO's 3-term simplified EOS;
    # ``"teos10"`` is NEMO's Roquet polynomial with the TEOS-10 coefficient
    # set -- what ORCA1 runs (``ln_teos10 = .true.``). Same axis, same name
    # and same default as ``TKEConfig.n2_eos_form``: the EVD trigger's N²
    # was hard-wired to the simplified EOS while its TKE sibling was already
    # configurable.
    # SCOPE, measured (dual review 2026-08-19): NO shipped card is changed
    # or fixed by this field today. ORCA1 leaves the EVD ``n2_mode`` at its
    # ``"insitu"`` default, so the bn2 kernel is never reached from the
    # trigger and this selector is inert there; the only cards selecting
    # ``"nemo_bn2"`` are the DINO Kamm ones, whose NEMO namelist selects the
    # simplified EOS, so ``"seos"`` is already the right answer for them.
    # The field exists so a future TEOS-10 card that also selects
    # ``"nemo_bn2"`` cannot silently take the simplified fit. Note ORCA1 has
    # a LARGER and still-open split this does not address: its EVD trigger
    # runs on the in-situ density difference while its TKE closure runs
    # NEMO bn2 with TEOS-10 -- two genuinely different static stabilities in
    # one column. Ignored by every other ``n2_mode``.
    n2_eos_form: str = "seos"
    # Static-instability trigger threshold on N² [1/s²]: EVD fires where
    # N² < n2_threshold. Default 0.0 (fire on any negative N²). NEMO zdfevd
    # (ln_zdfevd) fires where MIN(rn2, rn2b) <= -1e-12 — a small NEGATIVE
    # threshold that IGNORES marginally-neutral interfaces (N² in [-1e-12, 0)).
    # Matters during spring restratification: firing on near-zero-negative N²
    # noise re-mixes the shoaling ML every step and blocks the seasonal
    # thermocline rebuild (NEMO GYRE fidelity, plan §G). Only consulted on the
    # hard-threshold path (smooth_transition=False).
    n2_threshold: float = 0.0
    # NEMO zdfevd two-time-level trigger: fire where MIN(rn2, rn2b) < thr —
    # i.e. where EITHER the now or the before N² is unstable (zdfevd.F90:
    # MIN(rn2,rn2b) <= -1e-12). The hysteresis keeps EVD on one extra step in
    # marginal columns, preventing per-step ON/OFF FLICKER of the 100 m²/s
    # coefficient (a grid-scale noise generator in winter convecting regions).
    # Implemented as max(K_now, K_before) — equivalent for a hard threshold.
    # Requires the before-advection tracers (n2_tracers) to be threaded (the
    # TKE n2_before_advection machinery); silently single-level when absent.
    two_level_trigger: bool = False
    # ----- Time levels the two EVD trigger arms are evaluated at -----
    # NEMO zdfevd fires on MIN(rn2, rn2b) (src/OCE/ZDF/zdfevd.F90:93-94 for
    # avt, :119-120 for avm), and the DINO driver builds that pair at
    # cfgs/DINO/MY_SRC/stpmlf.F90:186-187:
    #   CALL bn2( ts(:,:,:,:,Nbb), rab_b, rn2b, Nnn )  ! BEFORE T/S, NOW geom
    #   CALL bn2( ts(:,:,:,:,Nnn), rab_n, rn2 , Nnn )  ! NOW    T/S, NOW geom
    # -- note the geometry index is Nnn for BOTH arms.
    #
    # "solver_state" (default, BIT-IDENTICAL legacy): both arms are built from
    #   whatever state the caller hands to the mixing coefficients. Under the
    #   leap-frog implicit solve that state is the POST-EXPLICIT (Kaa) one, so
    #   arm A is N²(Kaa) and the geometry (eta -> gdept / z* jacobian) is Kaa
    #   for BOTH arms.
    # "nemo_now_before": NEMO's own pair — arm A on the NOW (Nnn) tracers, arm
    #   B on the BEFORE (Nbb) tracers, BOTH on NOW (Nnn) geometry. Requires
    #   ``two_level_trigger=True`` (the name promises both arms) and a caller
    #   that threads the Nnn tracers, the Nbb tracers AND the Nnn eta;
    #   k_profiles.compute_vertical_K_profiles raises if any is missing rather
    #   than silently falling back to the solver state.
    evd_n2_time_level: str = "solver_state"
    # ----- How the EVD coefficient COMPOSES with everything else -----
    # NEMO's zdf_evd REPLACES the already-assembled coefficient where the
    # trigger fires -- ``p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)`` inside
    # the IF, zdfevd.f90:107-110 of the SMT-1 build's ppsrc -- and it runs
    # AFTER the background/closure assembly (zdfphy.f90:348-351 copies
    # avt_k into avt, :359 then calls zdf_evd).  The momentum arm is guarded
    # by ``IF( nn_evdm == 1 )`` (:121) and likewise REPLACES avm (:133-135).
    # legoESM's own (Oceananigans) composition ADDS the scheme's K on top of
    # every other contribution, which on a NEMO card leaves the background
    # and the closure riding on top of rn_evd.
    #   ""              unset.  A card whose trigger is NEMO's (n2_mode=
    #                   "nemo_bn2") must STATE this -- unset raises; every
    #                   other card keeps the additive composition.
    #   "additive"      legoESM/Oceananigans: K_total = K_other + K_evd.
    #   "nemo_replace"  NEMO zdfevd: K_total = rn_evd where the trigger
    #                   fires, K_other elsewhere.  Requires the hard
    #                   threshold (smooth_transition=False), K_bg < K_conv,
    #                   and -- the nn_evdm statement -- nu_conv either 0.0
    #                   (nn_evdm=0, avm untouched) or exactly K_conv
    #                   (nn_evdm=1, avm replaced by rn_evd too).
    evd_composition: str = ""


class PlumeConfig(NamedTuple):
    """Entraining mass-flux convective plume."""
    epsilon: float = 1e-3       # Entrainment rate [1/m]
    alpha_plume: float = 0.1    # Detrainment tendency scaling
    # Plume vertical velocity [m/s].  Used directly as ``w_p`` in
    # dT/dt = w_p · α · ε · (T_p − T_env).  Despite the legacy
    # ``_min`` suffix, this is the actual plume speed for the
    # unresolved-plume detrainment closure (see plume.py:75-88).
    w_plume_min: float = 0.01
    T_excess: float = 0.05      # Plume cold (destabilizing) excess magnitude [K];
                                # the source parcel is set to T_surface - T_excess
                                # so it is denser and sinks (down-plume)
    # Sharpness of the smooth active-mask transition on ``delta_rho``.
    # Larger values approach a hard switch; ``1e4`` corresponds to a
    # transition width of ~1e-4 kg/m^3 in density anomaly.  Configurable
    # so coarser/finer EOS regimes can retune without source edits.
    active_sigmoid_sharpness: float = 1e4


class OceanConvectionConfig(NamedTuple):
    """Top-level ocean convection configuration."""
    scheme: str = "none"  # "enhanced_diffusion", "plume", "none"
    enhanced_diffusion: EnhancedDiffusionConfig = EnhancedDiffusionConfig()
    plume: PlumeConfig = PlumeConfig()
