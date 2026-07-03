"""Reusable NEMO ocean numerics recipe card.

This module is intentionally a pure config layer: it selects the canonical
legoESM blocks that already implement NEMO-like numerics.  It does not contain a
NEMO-specific solver, bridge, grid, forcing reader, halo convention, or state
translation.  Setups below are thin demonstrations that pair the same card with
different domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from legoesm import constants
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    LateralMixingConfig,
    VisbeckConfig,
)
from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig
from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


NEMO_CONSTANTS_CONFIG = ConstantsConfig(
    g=constants.g_nemo,
    rho_0=constants.rho_ocean,
    c_sw=constants.c_p_seawater,
    Omega=constants.Omega,
    R_earth=constants.R_earth,
)


class NEMORecipe(NamedTuple):
    """ACCRecipe-shaped setup container using the reusable NEMO model card."""

    model_config: LatLonCGridOceanConfig
    physics_config: OceanPhysicsConfig
    grid: object
    z_coord: object
    land_mask: object
    initial_state: object


@dataclass(frozen=True)
class NEMOModelRecipeConfig:
    """Tunable coefficients around the fixed NEMO block choices.

    The scheme names are intentionally narrow.  Unknown dispatch values raise in
    :func:`nemo_lat_lon_model_config`, while the shared model validators raise
    again for unknown lower-level literals.
    """

    momentum_core: str = "vector_invariant_een"
    eos: str = "veros_gsw"
    tracer_advection: str = "ppm_fct"
    pgf_scheme: str = "smc03"
    barotropic_solver: str = "explicit_substep"
    n_barotropic_substeps: int = 30
    barotropic_time_filter: str = "cosine"
    momentum_time_integrator: str = "rk3"
    adaptive_implicit_vertadv: bool = True
    implicit_vertical_mixing: bool = True

    A_h: float = 5.0e4
    A_h_lat_scaling: bool = True
    A_h_cos_power: int = 1
    A_h_floor: float = 2.0e3
    A_h_merid: float = 0.0
    A_h_eq_boost: float = 1.0
    A_h_eq_sigma_deg: float = 5.0
    C_smag_lap: float = 0.33
    B_h: float = 0.0
    K_h: float = 0.0
    K_bih: float = 0.0

    bottom_drag_cd: float = 2.5e-3
    bottom_drag_bg_velocity: float = 0.1
    bottom_drag_bbl_thickness: float = 100.0
    # NEMO zdfdrg drag law ("legacy" = the historical MOM6-style
    # quadratic-with-floor above, baseline-preserving default;
    # "nemo_quadratic" = zdfdrg np_non_lin, the ORCA1 namelist selection
    # (namdrg ln_non_lin, rn_Cd0=1e-3, rn_ke0=2.5e-3);
    # "nemo_loglayer" = np_loglayer with rn_z0/rn_Cdmax).  The NEMO
    # schemes ignore bottom_drag_cd/bottom_drag_bg_velocity.
    bottom_drag_scheme: str = "legacy"
    bottom_drag_cd0: float = 1.0e-3     # NEMO rn_Cd0 [-]
    bottom_drag_cdmax: float = 0.1      # NEMO rn_Cdmax [-]
    bottom_drag_z0: float = 3.0e-3      # NEMO rn_z0 [m]
    bottom_drag_ke0: float = 2.5e-3     # NEMO rn_ke0 [m²/s²]

    gm_redi: bool = True
    kappa_GM: float = 600.0
    kappa_Redi: float = 600.0
    redi_S_max: float = 0.005

    mle: bool = True
    rgb_shortwave: bool = True
    runoff_depth_spread_m: float = 150.0
    normalize_freshwater: bool = True
    freeze_floor: bool = False


NEMO_BLOCK_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("constants", "ConstantsConfig", "NEMO_CONSTANTS_CONFIG from legoesm.constants"),
    (
        "EOS",
        "LatLonCGridOceanConfig.eos",
        "veros_gsw (TEOS-10-like GSW polynomial)",
    ),
    ("momentum advection", "momentum_advection", "vector_invariant"),
    (
        "vorticity/PV flux",
        "vector_invariant path",
        "AL81/EEN pv_flux_al81_partial_cell",
    ),
    ("KE gradient", "ke_gradient_scheme", "hollingsworth"),
    (
        "UP3 momentum option",
        "momentum_core",
        "flux_form_upwind3 selects momentum_flux_scheme=upwind3",
    ),
    (
        "tracer advection",
        "tracer_advection",
        "ppm_fct (nearest existing FCT-style block)",
    ),
    ("pressure gradient", "pgf_scheme", "smc03 density-Jacobian partial-cell PGF"),
    ("barotropic solver", "barotropic_solver", "explicit_substep + cosine filter"),
    ("baroclinic momentum time step", "momentum_time_integrator", "rk3"),
    ("vertical momentum advection", "adaptive_implicit_vertadv", "True"),
    (
        "vertical mixing",
        "physics.vertical_mixing",
        "tke prognostic Gaspar/Burchard closure",
    ),
    (
        "lateral viscosity",
        "A_h/C_smag_lap",
        "cos-lat Laplacian + Laplacian Smagorinsky",
    ),
    (
        "meridional viscosity scaling",
        "A_h_lat_scaling/A_h_cos_power",
        "rn_ahm0-style cosine latitude scaling",
    ),
    (
        "equatorial viscosity boost",
        "A_h_eq_boost/A_h_eq_sigma_deg",
        "available NEMO-matching knob; inactive by default in the card",
    ),
    ("GM/Redi", "gm_redi", "GMRediConfig with neutral triads"),
    ("mixed-layer eddies", "physics.mle", "MLEConfig nn_mle=1"),
    ("shortwave penetration", "physics.shortwave_penetration", "rgb_chl"),
    ("runoff spreading", "runoff_depth_spread_m", "150 m"),
)


NEMO_DEFERRED_BLOCKS: tuple[str, ...] = (
    "Dedicated NEMO EOS polynomial dispatch remains deferred; the card uses the "
    "existing veros_gsw TEOS-10-like block.",
    "Exact NEMO leapfrog/Robert-Asselin tracer stepping is not available as a "
    "lat-lon C-grid selector; the card uses the existing RK3 momentum path and "
    "documents the time-stepping fidelity cost.",
    "Exact NEMO split-explicit Higdon/doubled-boxcar free-surface filtering is "
    "not a separate selector; explicit_substep + cosine is the closest existing "
    "canonical barotropic block.",
    "Exact NEMO TKE namelist/coefficient parity remains deferred; the card uses "
    "the existing prognostic Gaspar/Burchard-style TKE block.",
    "Implicit quadratic NEMO bottom drag is approximated by the existing "
    "quadratic-with-floor model-level drag knobs.",
)


def _nemo_tke_config() -> TKEConfig:
    return TKEConfig(
        prognostic=True,
        n2_mode="adiabatic",
        veros_dz_slots=True,
        positivity="veros_surface_correction",
        kappa_convention="veros_sqrte",
        buoyancy_timing="post_mixing_veros",
        shear_production="realized_veros",
        prandtl_mode="richardson",
    )


def _nemo_physics_config(cfg: NEMOModelRecipeConfig) -> OceanPhysicsConfig:
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke",
            tke=_nemo_tke_config(),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=(
            ShortwavePenetrationConfig(scheme="rgb_chl")
            if cfg.rgb_shortwave else None
        ),
        mle=MLEConfig() if cfg.mle else None,
        constants=NEMO_CONSTANTS_CONFIG,
    )


def _momentum_options(momentum_core: str) -> dict[str, object]:
    if momentum_core == "vector_invariant_een":
        return {
            "momentum_advection": "vector_invariant",
            "ke_gradient_scheme": "hollingsworth",
        }
    if momentum_core == "flux_form_upwind3":
        return {
            "momentum_advection": "flux_form",
            "momentum_flux_scheme": "upwind3",
            "ke_gradient_scheme": "centered",
        }
    raise ValueError(
        "unknown NEMO momentum_core "
        f"{momentum_core!r}; expected 'vector_invariant_een' or "
        "'flux_form_upwind3'."
    )


def nemo_lat_lon_model_config(
    cfg: NEMOModelRecipeConfig | None = None,
) -> LatLonCGridOceanConfig:
    """Return the domain-agnostic NEMO lat-lon C-grid model config."""

    if cfg is None:
        cfg = NEMOModelRecipeConfig()

    bottom_drag_r = cfg.bottom_drag_cd * cfg.bottom_drag_bg_velocity
    gm_redi_cfg = None
    if cfg.gm_redi:
        gm_redi_cfg = GMRediConfig(
            kappa_GM=cfg.kappa_GM,
            kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.redi_S_max,
            visbeck=VisbeckConfig(enabled=False),
            slope_scheme="triads",
            slope_density="neutral",
            implicit_K33=True,
        )

    return LatLonCGridOceanConfig.from_flat(
        g=NEMO_CONSTANTS_CONFIG.g,
        rho_0=NEMO_CONSTANTS_CONFIG.rho_0,
        constants=NEMO_CONSTANTS_CONFIG,
        eos=cfg.eos,
        tracer_advection=cfg.tracer_advection,
        pgf_scheme=cfg.pgf_scheme,
        barotropic_solver=cfg.barotropic_solver,
        n_barotropic_substeps=cfg.n_barotropic_substeps,
        barotropic_time_filter=cfg.barotropic_time_filter,
        momentum_time_integrator=cfg.momentum_time_integrator,
        adaptive_implicit_vertadv=cfg.adaptive_implicit_vertadv,
        implicit_vertical_mixing=cfg.implicit_vertical_mixing,
        A_h=cfg.A_h,
        A_h_lat_scaling=cfg.A_h_lat_scaling,
        A_h_cos_power=cfg.A_h_cos_power,
        A_h_floor=cfg.A_h_floor,
        A_h_merid=cfg.A_h_merid,
        A_h_eq_boost=cfg.A_h_eq_boost,
        A_h_eq_sigma_deg=cfg.A_h_eq_sigma_deg,
        C_smag_lap=cfg.C_smag_lap,
        B_h=cfg.B_h,
        K_h=cfg.K_h,
        K_bih=cfg.K_bih,
        bottom_drag_r=bottom_drag_r,
        bottom_drag_bg_velocity=cfg.bottom_drag_bg_velocity,
        bottom_drag_bbl_thickness=cfg.bottom_drag_bbl_thickness,
        bottom_drag_scheme=cfg.bottom_drag_scheme,
        bottom_drag_cd0=cfg.bottom_drag_cd0,
        bottom_drag_cdmax=cfg.bottom_drag_cdmax,
        bottom_drag_z0=cfg.bottom_drag_z0,
        bottom_drag_ke0=cfg.bottom_drag_ke0,
        gm_redi=gm_redi_cfg,
        physics=_nemo_physics_config(cfg),
        normalize_freshwater=cfg.normalize_freshwater,
        runoff_depth_spread_m=cfg.runoff_depth_spread_m,
        freeze_floor=cfg.freeze_floor,
        **_momentum_options(cfg.momentum_core),
    )


def build_nemo_rest_recipe(
    *,
    n_lat: int = 8,
    n_lon: int = 16,
    nlev: int = 4,
    H_max: float = 1000.0,
    cfg: NEMOModelRecipeConfig | None = None,
) -> NEMORecipe:
    """Thin global-rest setup using the reusable NEMO model card."""

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    model_config = nemo_lat_lon_model_config(cfg)
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    initial_state = rest_state_latlon_cgrid_ocean(
        grid,
        z_coord,
        H_max=H_max,
        land_lat_threshold=90.0,
    )
    return NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=initial_state.land_mask.data,
        initial_state=initial_state,
    )


def build_nemo_eady_recipe(
    *,
    n_lat: int = 24,
    n_lon: int = 24,
    nlev: int = 12,
    cfg: NEMOModelRecipeConfig | None = None,
) -> NEMORecipe:
    """Thin Eady-channel setup using the reusable NEMO model card."""

    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup

    base = build_eady_uniform_setup(n_lat=n_lat, n_lon=n_lon, nlev=nlev)
    model_config = nemo_lat_lon_model_config(cfg)
    return NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=base.grid,
        z_coord=base.z_coord,
        land_mask=base.wall_mask,
        initial_state=base.initial_state,
    )


def build_nemo_recipe(
    *,
    setup: str = "rest",
    cfg: NEMOModelRecipeConfig | None = None,
    **kwargs: object,
) -> NEMORecipe:
    """One-stop NEMO recipe constructor using the shared model card."""

    if setup == "rest":
        return build_nemo_rest_recipe(cfg=cfg, **kwargs)
    if setup == "eady":
        return build_nemo_eady_recipe(cfg=cfg, **kwargs)
    raise ValueError("unknown NEMO setup " f"{setup!r}; expected 'rest' or 'eady'.")


__all__ = (
    "NEMO_BLOCK_MAPPING",
    "NEMO_CONSTANTS_CONFIG",
    "NEMO_DEFERRED_BLOCKS",
    "NEMOModelRecipeConfig",
    "NEMORecipe",
    "build_nemo_eady_recipe",
    "build_nemo_recipe",
    "build_nemo_rest_recipe",
    "nemo_lat_lon_model_config",
)
