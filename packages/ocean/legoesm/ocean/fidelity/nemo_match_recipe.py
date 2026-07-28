"""Reusable OMIP NEMO-*match* ocean recipe cards (MPAS ico6 + tripole eORCA025).

These are the two PROVEN OMIP CORE-II configurations that match the NEMO ORCA1
*climate* (not the bit-faithful NEMO-numerics card in :mod:`nemo_recipe`).  They
were validated against NEMO Mar day-90 SST (``docs/dev-notes/ocean_faithfulness_nemo.md``):

* **MPAS ico6 (~115 km ≈ ORCA1)** — SST RMSE **0.84** vs NEMO (best grid).
* **tripole eORCA025 (¼°)** — SST RMSE **1.15**, corr 0.99.

Unlike the numerics-faithful ``nemo_v1`` card (TKE / EEN / GSW / split-explicit),
these are a *climate*-match stack: KPP vertical mixing, Laplacian Smagorinsky
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
        "constant A_h=1e5 + Laplacian Smagorinsky C_smag_lap=0.33",
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
    K_zeta_bih: float = 1.0e14

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

    # GM/Redi (mesoscale eddy parameterization)
    gm_redi: bool = True
    kappa_GM: float = 600.0
    kappa_Redi: float = 600.0
    redi_S_max: float = 0.005
    # NEMO ldf_eiv flow-dependent kappa_GM (`nn_aei_ijk_t=21`: aeiu/aeiv =
    # F(growth rate of baroclinic instability), capped at aei0 = rn_Ue*rn_Le).
    # ORCA1 runs `ln_ldfeiv=.true., nn_aei_ijk_t=21, rn_Ue=0.018, rn_Le=100e3`
    # => aei0 = 1800 m^2/s, i.e. NEMO uses a SPACE/TIME-VARYING coefficient where
    # this recipe otherwise pins the constant `kappa_GM` above.  False keeps the
    # constant (byte-identical default); True selects the NEMO-faithful scaling
    # (TreguierConfig — already used by the DINO oracle card).
    gm_treguier: bool = False
    gm_aei0: float = 1800.0


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
    A_h: float = 1.0e5
    B_h: float = 0.0
    C_smag_lap: float = 0.33

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
    # NEMO ldf_eiv flow-dependent kappa_GM (`nn_aei_ijk_t=21`: aeiu/aeiv =
    # F(growth rate of baroclinic instability), capped at aei0 = rn_Ue*rn_Le).
    # ORCA1 runs `ln_ldfeiv=.true., nn_aei_ijk_t=21, rn_Ue=0.018, rn_Le=100e3`
    # => aei0 = 1800 m^2/s, i.e. NEMO uses a SPACE/TIME-VARYING coefficient where
    # this recipe otherwise pins the constant `kappa_GM` above.  False keeps the
    # constant (byte-identical default); True selects the NEMO-faithful scaling
    # (TreguierConfig — already used by the DINO oracle card).
    gm_treguier: bool = False
    gm_aei0: float = 1800.0


def _nemo_match_gm_redi(cfg) -> GMRediConfig | None:
    """Build the shared GM/Redi block both proven configs use, or None."""
    if not cfg.gm_redi:
        return None
    # NEMO ORCA1 runs GM with a FLOW-DEPENDENT coefficient (&namtra_eiv:
    # ln_ldfeiv=T, nn_aei_ijk_t=21 => aeiu/aeiv = F(growth rate of baroclinic
    # instability), capped at aei0 = rn_Ue*rn_Le = 0.018*100e3 = 1800 m^2/s;
    # ldftra.F90:386).  `gm_treguier=True` selects that scaling (TreguierConfig,
    # the same block the DINO oracle card uses); the default keeps the constant
    # `kappa_GM` so existing runs stay byte-identical.  Treguier and Visbeck are
    # mutually exclusive (both are adaptive-kappa schemes) — enforced by
    # GMRediConfig, so Visbeck stays disabled on both branches.
    _treguier = TreguierConfig(enabled=True, aei0=cfg.gm_aei0) \
        if getattr(cfg, "gm_treguier", False) else TreguierConfig()
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
    if physics is None:
        physics = _default_match_physics()
    return MPASOceanConfig(
        A_h=cfg.A_h,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        C_smag_lap=cfg.C_smag_lap,
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
    if physics is None:
        physics = _default_match_physics()
    return LatLonCGridOceanConfig.from_flat(
        A_h=cfg.A_h,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        B_h=cfg.B_h,
        C_smag_lap=cfg.C_smag_lap,
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
