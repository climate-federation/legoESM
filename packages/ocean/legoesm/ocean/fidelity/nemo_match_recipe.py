"""Reusable OMIP NEMO-*match* ocean recipe cards (MPAS ico6 + tripole eORCA025).

These are the two PROVEN OMIP CORE-II configurations that match the NEMO ORCA1
*climate* (not the bit-faithful NEMO-numerics card in :mod:`nemo_recipe`).  They
were validated against NEMO Mar day-90 SST (``docs/dev-notes/ocean_faithfulness_nemo.md``):

* **MPAS ico6 (~115 km ≈ ORCA1)** — SST RMSE **0.84** vs NEMO (best grid).
* **tripole eORCA025 (¼°)** — SST RMSE **1.15**, corr 0.99.

Unlike the NEMO-style ``legoesm_nemo_like_v1`` card (TKE / EEN / GSW /
split-explicit; renamed from ``nemo_v1`` 2026-09-02 — its barotropic solver
is legoESM's own generic arm, not NEMO's ``dyn_spg_ts``), these are a
*climate*-match stack: KPP vertical mixing, Laplacian Smagorinsky
lateral viscosity, the ``implicit_cn`` barotropic solver, GM/Redi at κ=600, and a
``tvd`` / ``adcroft`` dycore.  This module is the SINGLE SOURCE OF TRUTH for that
proven dycore + coefficient block; ``scripts/run/run_omip.py::_create_setup``
builds its dycore via these factories and overlays only the run-dependent SETUP
(mesh, forcing-mode-dependent surface forcing, the per-run vertical-mixing
config).  The catalog recipes ``omip_nemo_match_mpas_v1`` /
``omip_nemo_match_tripole_v1`` in :mod:`legoesm.ocean.recipes` are the scheme
identity of these factories, locked against drift by
``tests/ocean/unit/test_recipes.py``.

Like :mod:`nemo_recipe`, this is a pure config layer: it selects canonical
legoESM blocks (no bespoke solver / bridge / grid / forcing reader).  Physics is
SETUP, not recipe — the factory accepts a fully-built ``physics`` config (the
run-dependent piece) or builds a documented standalone default for recipe-only
use.  Unknown dispatch values raise; the shared model validators raise again for
unknown lower-level literals.
"""

from __future__ import annotations

from dataclasses import dataclass

from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    validate_treguier_cfg,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    LateralMixingConfig,
    TreguierConfig,
    VisbeckConfig,
)
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    RestoringConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.state import LatLonCGridOceanConfig


# --- documented block mapping (audit / provenance) --------------------------

NEMO_MATCH_BLOCK_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("EOS", "eos", "wright (the config default; NEMO-match uses it, not linear)"),
    ("tracer advection", "tracer_advection", "tvd (Van Leer monotone limiter)"),
    ("pressure gradient", "pgf_scheme", "adcroft density-Jacobian PGF"),
    (
        "barotropic solver",
        "barotropic_solver",
        "implicit_cn (Crank-Nicolson implicit free surface + PCG)",
    ),
    ("potential-vorticity flux (MPAS)", "pv_scheme", "enstrophy (the config default)"),
    (
        "KE gradient (tripole)",
        "ke_gradient_scheme",
        "centered (the config default; hollingsworth is NOT fold-aware — see "
        "_create_setup tripole note)",
    ),
    (
        "baroclinic momentum / tracer integrator (tripole)",
        "outer_integrator / tracer_time_integrator",
        "forward_euler / euler split-explicit (the config defaults)",
    ),
    ("vertical mixing", "implicit_vertical_mixing", "True (backward-Euler A_v/K_v)"),
    (
        "lateral viscosity",
        "A_h / C_smag_lap",
        "A_h derived from the mesh's narrowest wet cell (anchored so "
        "eORCA1.2 keeps 1e5) + Laplacian Smagorinsky C_smag_lap=0.33",
    ),
    ("GM/Redi", "gm_redi", "GMRediConfig kappa_GM=kappa_Redi=600, S_max=0.005, centered slopes"),
    ("bottom drag", "bottom_drag_r / bbl", "linear r=1e-3 over a 100 m BBL, bg vel 0.1"),
    ("free-surface filter (tripole)", "n_barotropic_substeps", "30"),
    ("vertical mixing (physics, SETUP)", "physics.vertical_mixing", "kpp (run-dependent)"),
    ("convection (physics, SETUP)", "physics.convection", "enhanced_diffusion K_conv=1.0"),
)


# --- frozen recipe cards (defaults = the PROVEN winning coefficients) --------

@dataclass(frozen=True)
class NEMOMatchMPASRecipeConfig:
    """Tunable coefficients around the fixed MPAS ico6 NEMO-match block choices.

    Defaults reproduce ``scripts/run/run_omip.py::_create_setup("mpas", ...)``'s
    ``MPASOceanConfig`` EXACTLY (the SST-RMSE-0.84 configuration).  The scheme
    names are intentionally narrow; unknown dispatch values raise in the model
    validators.
    """

    # dycore scheme identity (the 6-key MPAS recipe bundle)
    eos: str = "wright"
    tracer_advection: str = "tvd"
    pgf_scheme: str = "adcroft"
    barotropic_solver: str = "implicit_cn"
    pv_scheme: str = "enstrophy"
    implicit_vertical_mixing: bool = True

    # lateral viscosity / dissipation
    A_h: float = 1.0e5
    C_smag_lap: float = 0.33
    B_h: float = 0.0  # biharmonic viscosity [m^4/s]; 0 = off, byte-identical default
    C_smag: float = 0.0  # biharmonic Smagorinsky coefficient; 0 = off, byte-identical default
    C_leith: float = 0.0  # Leith coefficient; 0 = off, byte-identical default
    # Explicit biharmonic filter on relative vorticity, OFF by default.
    #
    # Fixed 1e14 (not mesh-scaled) gave stability number K*dt*lambda_max^2 =
    # 1.4 on the level-7 mesh and 11 on level 8, which blew up in 10 steps
    # (2026-09-04); the filter was switched off by user decision 2026-09-06.
    #
    # main introduced a mesh-scaled rule instead (``None`` ->
    # ``resolution_scaled_k_zeta_bih``, K_ref*(dx/dx_ref)^3 anchored on ico6).
    # That machinery is KEPT and remains selectable by passing ``None``; only
    # the DEFAULT stays off. Both reviewers reached that conclusion
    # independently: the stability number goes as K*dt/dx^4, so a dx^3 scaling
    # leaves it proportional to dt/dx -- mesh-invariant only if dt shrinks with
    # the mesh, and codex confirmed the runner passes ``float(args.dt)``
    # unchanged, so a finer mesh can still receive a fixed timestep. The rule
    # rescues the level-8 case (K = 1e14/64) but is mitigation, not a fix.
    # A nonzero or derived value is gated at the first step by
    # MPASOceanModel.check_vorticity_filter_stability (codex: the gate covers
    # BOTH derived and pinned coefficients, raising at stability number >= 2).
    K_zeta_bih: float | None = 0.0

    # vertical mixing (explicit-block coefficients; implicit solve uses them)
    A_v: float = 1.0e-4
    K_v: float = 1.0e-5

    # implicit barotropic PCG solver controls
    barotropic_implicit_pcg_tol: float = 1.0e-10
    barotropic_implicit_pcg_maxiter: int = 300

    # bottom drag (linear, distributed over a BBL)
    bottom_drag_r: float = 1.0e-3
    bottom_drag_bbl_thickness: float = 100.0
    bottom_drag_bg_velocity: float = 0.1

    # freshwater
    normalize_freshwater: bool = True

    # GM/Redi (mesoscale eddy parameterization).
    # NOTE (MPAS recipe): `gm_treguier=True` is refused here, but the reason
    # CHANGED in 2026-09.  The MPAS GM/Redi now implements the Treguier block
    # (`gm_redi_mpas.py`), using a wet-edge Perot reconstruction of the slope
    # vector.  What it implements is the SHARED variant, NOT the
    # `nemo_native` one the ORCA1-faithful tripole card selects, and no CLI
    # flag reaches this lane yet — so a recipe that set it would silently run
    # a different discretisation from the tripole it is being compared with.
    # The refusal stands until the CLI is wired and the two variants have been
    # compared; it is a harmonization guard now, not a missing-code guard.
    gm_redi: bool = True
    kappa_GM: float = 600.0
    kappa_Redi: float = 600.0
    redi_S_max: float = 0.005
    # NEMO ldf_eiv flow-dependent kappa_GM — see `_nemo_match_gm_redi` below for
    # the namelist provenance.  False keeps the constant `kappa_GM` above
    # (byte-identical default); True selects the NEMO-faithful Treguier scaling.
    gm_treguier: bool = False
    # 900 = 1/2*rn_Ue*rn_Le: ldftra.F90:290-293 sets zUfac = r1_2*rn_Ud for the
    # LAPLACIAN operator, which ORCA1 runs (ln_traldf_lap=.true.), and NEMO's
    # emitted aeiu_2d maxes at exactly 900 on eORCA1.  (Was 1800 = rn_Ue*rn_Le,
    # i.e. the same product with NEMO's 1/2 simply omitted -- NOT a bilaplacian
    # value; NEMO's bilaplacian coefficient has different units and its own
    # 1/12 factor.)
    gm_aei0: float = 900.0
    # Floor on the Treguier kappa_GM [m^2/s]; only meaningful with
    # gm_treguier=True.
    #
    # DEFAULT 0.0 = RAW NEMO (capped-only; kappa -> 0 at the equator).  This is
    # a NEMO-MATCH recipe, so its default must be the oracle form: a nonzero
    # floor is a deliberate NON-NEMO closure change (see TreguierConfig).
    #
    # This DELIBERATELY differs from `run_omip_core2.py --gm-kappa-min`, whose
    # default is 200.0: that is a production stability knob (an unfloored
    # equatorial band destabilised a 1-degree global run), not a fidelity
    # setting.  The divergence is intentional and tested
    # (test_nemo_match_recipe / test_run_omip_core2_gm_treguier) -- what was a
    # BUG before was the recipe silently taking 0.0 with nobody aware of it,
    # because the field did not exist and could not be set.
    gm_kappa_min: float = 0.0


@dataclass(frozen=True)
class NEMOMatchTripoleRecipeConfig:
    """Tunable coefficients around the fixed tripole eORCA025 NEMO-match choices.

    Defaults reproduce ``scripts/run/run_omip.py::_create_setup("tripole", ...)``'s
    ``LatLonCGridOceanConfig`` EXACTLY (the SST-RMSE-1.15 configuration).
    """

    # dycore scheme identity (matches the LATLON recipe bundle key set)
    eos: str = "wright"
    momentum_advection: str = "vector_invariant"
    tracer_advection: str = "tvd"
    pgf_scheme: str = "adcroft"
    ke_gradient_scheme: str = "centered"
    barotropic_solver: str = "implicit_cn"
    coriolis_scheme: str = "matsuno_split"
    outer_integrator: str = "forward_euler"
    tracer_time_integrator: str = "euler"
    implicit_vertical_mixing: bool = True
    n_barotropic_substeps: int = 30

    # lateral viscosity / dissipation
    # None = DERIVE from the mesh's narrowest wet cell, anchored so eORCA1.2
    # keeps 1e5 (legoesm.ocean.state.resolution_scaled_lateral_viscosity).  One
    # number cannot serve 1 degree and 1/12 degree: measured, 1e5 puts ORCA12
    # 26x over the explicit Laplacian limit and its cold start diverges at
    # step 10 with a 193 m/s current.
    A_h: float | None = None
    B_h: float = 0.0
    C_smag_lap: float = 0.33
    C_smag: float = 0.0  # biharmonic Smagorinsky coefficient; 0 = off, byte-identical default
    C_leith: float = 0.0  # Leith coefficient; 0 = off, byte-identical default
    B_h_gamma0: float = 0.0  # FESOM2 gamma0*h^3 biharmonic on the 2-D metrics; 0 = off

    # vertical mixing (explicit-block coefficients)
    A_v: float = 1.0e-4
    K_v: float = 1.0e-5

    # implicit barotropic PCG solver controls
    barotropic_implicit_pcg_tol: float = 1.0e-10
    barotropic_implicit_pcg_maxiter: int = 300

    # bottom drag (linear, distributed over a BBL)
    bottom_drag_r: float = 1.0e-3
    bottom_drag_bbl_thickness: float = 100.0
    bottom_drag_bg_velocity: float = 0.1

    # freshwater
    freshwater_closure: str = "virtual_salt_flux"
    # OMIP global freshwater correction (NEMO-faithful): remove the area-mean
    # of the net freshwater flux from the virtual-salt closure so surface
    # freshwater conserves GLOBAL SALT.  _create_setup("tripole") sets this
    # True (run_omip.py); the recipe omitting it silently fell back to the
    # LatLonCGridOceanConfig default False (codex) — a real drift from the
    # proven configuration.
    normalize_freshwater: bool = True

    # GM/Redi (mesoscale eddy parameterization)
    gm_redi: bool = True
    kappa_GM: float = 600.0
    kappa_Redi: float = 600.0
    redi_S_max: float = 0.005
    # NEMO ldf_eiv flow-dependent kappa_GM — see `_nemo_match_gm_redi` below for
    # the namelist provenance.  False keeps the constant `kappa_GM` above
    # (byte-identical default); True selects the NEMO-faithful Treguier scaling.
    gm_treguier: bool = False
    # 900 = 1/2*rn_Ue*rn_Le: ldftra.F90:290-293 sets zUfac = r1_2*rn_Ud for the
    # LAPLACIAN operator, which ORCA1 runs (ln_traldf_lap=.true.), and NEMO's
    # emitted aeiu_2d maxes at exactly 900 on eORCA1.  (Was 1800 = rn_Ue*rn_Le,
    # i.e. the same product with NEMO's 1/2 simply omitted -- NOT a bilaplacian
    # value; NEMO's bilaplacian coefficient has different units and its own
    # 1/12 factor.)
    gm_aei0: float = 900.0
    # Floor on the Treguier kappa_GM [m^2/s]; only meaningful with
    # gm_treguier=True.
    #
    # DEFAULT 0.0 = RAW NEMO (capped-only; kappa -> 0 at the equator).  This is
    # a NEMO-MATCH recipe, so its default must be the oracle form: a nonzero
    # floor is a deliberate NON-NEMO closure change (see TreguierConfig).
    #
    # This DELIBERATELY differs from `run_omip_core2.py --gm-kappa-min`, whose
    # default is 200.0: that is a production stability knob (an unfloored
    # equatorial band destabilised a 1-degree global run), not a fidelity
    # setting.  The divergence is intentional and tested
    # (test_nemo_match_recipe / test_run_omip_core2_gm_treguier) -- what was a
    # BUG before was the recipe silently taking 0.0 with nobody aware of it,
    # because the field did not exist and could not be set.
    gm_kappa_min: float = 0.0


def _nemo_match_gm_redi(cfg) -> GMRediConfig | None:
    """Build the shared GM/Redi block both proven configs use, or None."""
    if not cfg.gm_redi:
        if cfg.gm_treguier:
            # Otherwise the selected kappa_GM scheme (and its floor) vanish at
            # this early return -- the same silent-discard the CLI rejects via
            # the --gm-treguier/--no-gm-redi conflict guard.
            raise ValueError(
                "gm_treguier=True with gm_redi=False: the Treguier kappa_GM "
                "scheme cannot apply when GM/Redi is disabled entirely. Set "
                "gm_redi=True, or leave gm_treguier=False.")
        return None
    # NEMO ORCA1 runs GM with a FLOW-DEPENDENT coefficient (&namtra_eiv:
    # ln_ldfeiv=T, nn_aei_ijk_t=21 => aeiu/aeiv = F(growth rate of baroclinic
    # instability), capped at aei0 = 1/2*rn_Ue*rn_Le = 0.5*0.018*100e3 = 900
    # m^2/s for the laplacian operator ORCA1 runs; ldftra.F90:290-293).  `gm_treguier=True` selects that scaling (TreguierConfig,
    # the same block the DINO oracle card uses); the default keeps the constant
    # `kappa_GM` so existing runs stay byte-identical.  Treguier and Visbeck are
    # mutually exclusive (both are adaptive-kappa schemes) — enforced by
    # GMRediConfig, so Visbeck stays disabled on both branches.
    # `gm_kappa_min` is threaded so the recipe and `run_omip_core2.py` build the
    # SAME block: without it the recipe silently took kappa_min=0.0 (no floor)
    # while the CLI defaulted to 200 — one scheme, two opposite defaults.
    _treguier = (
        TreguierConfig(enabled=True, aei0=cfg.gm_aei0,
                       kappa_min=cfg.gm_kappa_min)
        if cfg.gm_treguier else TreguierConfig()
    )
    validate_treguier_cfg(_treguier)
    return GMRediConfig(
        kappa_GM=cfg.kappa_GM,
        kappa_Redi=cfg.kappa_Redi,
        S_max=cfg.redi_S_max,
        visbeck=VisbeckConfig(enabled=False),
        treguier=_treguier,
        slope_scheme="centered",
    )


def _default_match_physics(
    *,
    forcing_mode: str = "restoring",
    vertical_mixing: VerticalMixingConfig | None = None,
    wind_profile: str = "global_wind",
    tropical_wind_scale: float = 0.5,
) -> OceanPhysicsConfig:
    """Documented standalone physics for recipe-only use (matches _create_setup).

    Mirrors the SETUP physics that ``_create_setup`` builds for the proven runs:
    KPP vertical mixing (overridable via ``vertical_mixing``), enhanced-diffusion
    convection (``K_conv=1.0``), and ``lateral_mixing``/``bottom_drag`` set to
    ``"none"`` (the dissipation lives in the model config's ``A_h``/``gm_redi``).

    ``forcing_mode="jra55_do_tropical"`` selects ``surface_forcing="none"`` (the
    dynamics-core external-tau block is the sole consumer); any other value
    selects the idealized prescribed-wind + restoring forcing used for the
    restoring-mode runs.  This default is for recipe-only use; production
    ``_create_setup`` passes a fully-built ``physics`` so the two never drift.
    """
    if forcing_mode == "jra55_do_tropical":
        sf_config = SurfaceForcingConfig(scheme="none")
    elif forcing_mode == "restoring":
        sf_config = SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile=wind_profile,
                tropical_wind_scale=tropical_wind_scale,
            ),
            restoring=RestoringConfig(),
        )
    else:
        raise ValueError(
            "unknown NEMO-match forcing_mode "
            f"{forcing_mode!r}; expected 'restoring' or 'jra55_do_tropical'."
        )
    return OceanPhysicsConfig(
        surface_forcing=sf_config,
        vertical_mixing=vertical_mixing or VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
        ),
        shortwave_penetration=None,
    )


def nemo_match_mpas_model_config(
    cfg: NEMOMatchMPASRecipeConfig | None = None,
    *,
    physics: OceanPhysicsConfig | None = None,
) -> MPASOceanConfig:
    """Return the proven OMIP NEMO-match MPAS ico6 ``MPASOceanConfig``.

    Reproduces ``_create_setup("mpas", ...)``'s dycore + coefficients EXACTLY.
    ``physics`` (run-dependent SETUP) is passed through; when ``None`` a
    documented restoring-mode default is built (recipe-only use).
    """
    if cfg is None:
        cfg = NEMOMatchMPASRecipeConfig()
    if cfg.gm_treguier:
        # gm_redi_mpas raises NotImplementedError for the Treguier block, but
        # only inside the first GM tendency -- i.e. after a full model build.
        # Reject here so the failure is at config build, next to the flag.
        raise NotImplementedError(
            "NEMOMatchMPASRecipeConfig.gm_treguier=True is not supported yet. "
            "The MPAS GM/Redi DOES now implement a Treguier kappa_GM, but it "
            "is the SHARED variant, not the nemo_native one the ORCA1-faithful "
            "tripole card runs, and no CLI flag selects it on this lane. "
            "Enabling it from a recipe would quietly compare two different "
            "discretisations. Use the tripole recipe, or leave "
            "gm_treguier=False for constant kappa_GM.")
    if physics is None:
        physics = _default_match_physics()
    return MPASOceanConfig(
        A_h=cfg.A_h,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        C_smag_lap=cfg.C_smag_lap,
        B_h=cfg.B_h,
        C_smag=cfg.C_smag,
        C_leith=cfg.C_leith,
        K_zeta_bih=cfg.K_zeta_bih,
        barotropic_solver=cfg.barotropic_solver,
        barotropic_implicit_pcg_tol=cfg.barotropic_implicit_pcg_tol,
        barotropic_implicit_pcg_maxiter=cfg.barotropic_implicit_pcg_maxiter,
        pgf_scheme=cfg.pgf_scheme,
        eos=cfg.eos,
        pv_scheme=cfg.pv_scheme,
        implicit_vertical_mixing=cfg.implicit_vertical_mixing,
        normalize_freshwater=cfg.normalize_freshwater,
        tracer_advection=cfg.tracer_advection,
        bottom_drag_r=cfg.bottom_drag_r,
        bottom_drag_bbl_thickness=cfg.bottom_drag_bbl_thickness,
        bottom_drag_bg_velocity=cfg.bottom_drag_bg_velocity,
        gm_redi=_nemo_match_gm_redi(cfg),
        physics=physics,
    )


def nemo_match_tripole_model_config(
    cfg: NEMOMatchTripoleRecipeConfig | None = None,
    *,
    physics: OceanPhysicsConfig | None = None,
) -> LatLonCGridOceanConfig:
    """Return the proven OMIP NEMO-match tripole eORCA025 ``LatLonCGridOceanConfig``.

    Reproduces ``_create_setup("tripole", ...)``'s dycore + coefficients EXACTLY.
    ``physics`` (run-dependent SETUP) is passed through; when ``None`` a
    documented restoring-mode default is built (recipe-only use).

    NOTE: ``ke_gradient_scheme`` is intentionally left at the ``"centered"``
    default — the Hollingsworth KE stencil is not yet north-fold-aware on the
    tripole (see the ``_create_setup`` tripole comment).
    """
    if cfg is None:
        cfg = NEMOMatchTripoleRecipeConfig()
    for name in ("B_h", "C_smag", "C_leith"):
        if (getattr(cfg, name) or 0.0) > 0.0:
            raise ValueError(
                f"NEMOMatchTripoleRecipeConfig.{name}={getattr(cfg, name)!r}: "
                "the biharmonic, biharmonic-Smagorinsky and Leith operators use "
                "the regular lat-lon 1-D metrics (biharmonic_scaling_factor / "
                "_cos_lat_uv divide by dlon = 0 on a tripolar mesh -> "
                "non-finite; vertex_area_1d broadcasts one column of corner "
                "areas), so they are refused on the tripole recipe until a "
                "2-D-metric operator exists; use A_h / C_smag_lap.")
    if cfg.B_h_gamma0 > 0.0 and (cfg.A_h is None or cfg.A_h > 0.0 or cfg.C_smag_lap > 0.0):
        raise ValueError(
            f"NEMOMatchTripoleRecipeConfig.B_h_gamma0={cfg.B_h_gamma0!r} (FESOM2 "
            "biharmonic) requires A_h = 0 and C_smag_lap = 0: FESOM2 has no "
            f"Laplacian, and the A_h branches would ignore gamma0 (got A_h="
            f"{cfg.A_h!r}, C_smag_lap={cfg.C_smag_lap!r}).")
    if physics is None:
        physics = _default_match_physics()
    return LatLonCGridOceanConfig.from_flat(
        A_h=cfg.A_h,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        B_h=cfg.B_h,
        C_smag_lap=cfg.C_smag_lap,
        C_smag=cfg.C_smag,
        C_leith=cfg.C_leith,
        C_leith_modified=(cfg.C_leith > 0),
        # FESOM2 gamma0*h^3 biharmonic (2-D metrics): lat scaling off when
        # selected (it already scales with the local face size); default
        # (gamma0 = 0) keeps B_h_lat_scaling True byte-identically.
        B_h_gamma0=cfg.B_h_gamma0,
        B_h_lat_scaling=(cfg.B_h_gamma0 <= 0.0),
        n_barotropic_substeps=cfg.n_barotropic_substeps,
        barotropic_solver=cfg.barotropic_solver,
        barotropic_implicit_pcg_tol=cfg.barotropic_implicit_pcg_tol,
        barotropic_implicit_pcg_maxiter=cfg.barotropic_implicit_pcg_maxiter,
        pgf_scheme=cfg.pgf_scheme,
        eos=cfg.eos,
        momentum_advection=cfg.momentum_advection,
        ke_gradient_scheme=cfg.ke_gradient_scheme,
        coriolis_scheme=cfg.coriolis_scheme,
        outer_integrator=cfg.outer_integrator,
        tracer_time_integrator=cfg.tracer_time_integrator,
        implicit_vertical_mixing=cfg.implicit_vertical_mixing,
        tracer_advection=cfg.tracer_advection,
        bottom_drag_r=cfg.bottom_drag_r,
        bottom_drag_bbl_thickness=cfg.bottom_drag_bbl_thickness,
        bottom_drag_bg_velocity=cfg.bottom_drag_bg_velocity,
        freshwater_closure=cfg.freshwater_closure,
        normalize_freshwater=cfg.normalize_freshwater,
        gm_redi=_nemo_match_gm_redi(cfg),
        physics=physics,
    )


__all__ = (
    "NEMO_MATCH_BLOCK_MAPPING",
    "NEMOMatchMPASRecipeConfig",
    "NEMOMatchTripoleRecipeConfig",
    "nemo_match_mpas_model_config",
    "nemo_match_tripole_model_config",
)
