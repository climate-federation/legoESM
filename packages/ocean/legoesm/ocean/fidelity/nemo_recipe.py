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

import numpy as np
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
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

from legoesm import constants

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
    # Iso-neutral tracer operator. "triads" (default) = the GM+Redi triad scheme
    # for a GM-on NEMO (ln_ldfeiv=T, ln_traldf_triad=T; ORCA-style, kappa_GM>0).
    # "nemo_iso_lap" = NEMO's standard rotated-Laplacian Redi (traldf_iso,
    # ln_traldf_triad=F) — verified vs GYRE's dumped ttrd_ldf (T corr 0.96); it is
    # PURE Redi (NEMO GYRE has ln_ldfeiv=F, no GM), so it forces kappa_GM=0.
    lateral_operator: str = "triads"

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
    "NEMO TKE amplitude (√e), Ri-Prandtl (nn_pdl=1, slope 1/ri_cri=4.5), c_k/c_eps "
    "(rn_ediff/rn_ediss), background Kz (rn_avm0/rn_avt0) AND the interior "
    "mixing-length response are now dump-verified NEMO-exact: fed NEMO's own N² "
    "(rn2), legoESM's diffusivity matches avm_k to corr 0.9998 for levels >=2 (the "
    "Veros nn_mxl=2 length is scalar-identical to NEMO nn_mxl=3 in the interior, so "
    "nn_mxl=3 is NOT the lever — TKE_DECONFOUNDED_FINDINGS.md). The one remaining "
    "TKE item is the ln_mxl0 surface mixing-length anchor + NEMO Dirichlet "
    "surface-TKE BC (top ~2 interfaces only), which matters solely for a full "
    "PROGNOSTIC spinup match, not the diagnostic/tendency oracle.",
    "Implicit quadratic NEMO bottom drag is approximated by the existing "
    "quadratic-with-floor model-level drag knobs.",
)


def _nemo_tke_config() -> TKEConfig:
    # NEMO zdftke coefficient parity (verified vs the GYRE step-12 dump —
    # avm_k/avt_k/en; TKE_FIDELITY_FINDINGS.md).  The amplitude
    # (kappa_convention="veros_sqrte" ⇒ K=c_k·l·√e, not √(2e) — removes an exact
    # √2 overshoot), the Ri-Prandtl (prandtl_ri_coeff=1/ri_cri), and the
    # background floors below are now NEMO-exact; c_k=0.1/c_eps=0.7 already match
    # rn_ediff/rn_ediss by default.  The remaining deferred item is the nn_mxl=3
    # length form (see NEMO_DEFERRED_BLOCKS).
    return TKEConfig(
        prognostic=True,
        n2_mode="adiabatic",
        # NEMO stp ordering: eosbn2 runs at step start (bn2(Nnow)), BEFORE
        # tra_adv. Sampling the diffusivity-stage N² on the before-advection
        # T/S stops the single-step fct2 bottom-cell drift from flipping the
        # marginal deepest interface to N²<0 (spurious deep convection;
        # BOTTOM_N2_DIAGNOSIS_FINDINGS.md).
        n2_before_advection=True,
        veros_dz_slots=True,
        positivity="veros_surface_correction",
        kappa_convention="veros_sqrte",
        buoyancy_timing="post_mixing_veros",
        shear_production="realized_veros",
        prandtl_mode="richardson",
        # NEMO nn_pdl=1: Pr = min(10, max(1, Ri/ri_cri)), ri_cri = 2/(2+rn_ediss/
        # rn_ediff) = 2/(2+0.7/0.1) = 2/9 ⇒ slope 1/ri_cri = 4.5 (NOT the Veros
        # 6.6 default). legoESM Pr=max(1,min(10,coeff·Ri)) is bit-identical (clamp
        # to [1,10] is order-independent), so this reproduces NEMO's pdl exactly.
        prandtl_ri_coeff=4.5,
        # NEMO background Kz: rn_avm0=1.2e-4 (avmb), rn_avt0=1.2e-5 (avtb).
        kappaM_min=1.2e-4,
        kappaH_min=1.2e-5,
        # NEMO GYRE (ln_zdfcst=F) uses a CONSTANT background avtb=1.2e-5, NOT the
        # Veros Bryan-Lewis abyssal depth profile. With BL on (the legoESM
        # default) the deep tracer floor would be ~3e-5..1.3e-4 (10x NEMO's
        # background), masking the independent-kappaH_min flooring fix. Turn it
        # off so the deep floor is NEMO's constant avtb.
        enable_kappaH_profile=False,
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
    if momentum_core == "vector_invariant_ene":
        # NEMO GYRE_BARE dynamical core: ln_dynvor_ene (Sadourny-1975 energy-
        # conserving vorticity) + nn_dynkeg=0 (c2 mean-of-squares KE gradient).
        # Distinct from vector_invariant_een above (which uses the al81 enstrophy
        # vorticity default + Hollingsworth KE for ORCA-style configs). Verified
        # term-by-term vs GYRE's utrd_rvo/pvo/keg dumps.
        return {
            "momentum_advection": "vector_invariant",
            "vorticity_scheme": "ene",
            "ke_gradient_scheme": "c2",
        }
    if momentum_core == "flux_form_upwind3":
        return {
            "momentum_advection": "flux_form",
            "momentum_flux_scheme": "upwind3",
            "ke_gradient_scheme": "centered",
        }
    raise ValueError(
        "unknown NEMO momentum_core "
        f"{momentum_core!r}; expected 'vector_invariant_een', "
        "'vector_invariant_ene', or 'flux_form_upwind3'."
    )


def nemo_lat_lon_model_config(
    cfg: NEMOModelRecipeConfig | None = None,
) -> LatLonCGridOceanConfig:
    """Return the domain-agnostic NEMO lat-lon C-grid model config."""

    if cfg is None:
        cfg = NEMOModelRecipeConfig()

    bottom_drag_r = cfg.bottom_drag_cd * cfg.bottom_drag_bg_velocity
    if cfg.lateral_operator not in ("triads", "nemo_iso_lap"):
        raise ValueError(
            f"Unknown NEMOModelRecipeConfig.lateral_operator="
            f"{cfg.lateral_operator!r}; expected 'triads' or 'nemo_iso_lap'.")
    gm_redi_cfg = None
    if cfg.gm_redi:
        # nemo_iso_lap (NEMO traldf_iso, ln_traldf_triad=F) is PURE Redi — NEMO
        # GYRE runs ln_ldfeiv=F (no GM bolus), and the operator raises on
        # kappa_GM≠0, so force kappa_GM=0 when it is selected.
        _kappa_gm = 0.0 if cfg.lateral_operator == "nemo_iso_lap" else cfg.kappa_GM
        gm_redi_cfg = GMRediConfig(
            kappa_GM=_kappa_gm,
            kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.redi_S_max,
            visbeck=VisbeckConfig(enabled=False),
            slope_scheme=cfg.lateral_operator,
            slope_density="neutral",
            implicit_K33=True,
            # nemo_iso_lap wants NEMO's ldfslp ML ramp + Shapiro slope fidelity.
            nemo_mld_slope_ramp=(cfg.lateral_operator == "nemo_iso_lap"),
            nemo_slope_shapiro=(cfg.lateral_operator == "nemo_iso_lap"),
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


# =====================================================================
# GYRE_BARE — native (non-bridge) assembly of NEMO's double-gyre demo.
# =====================================================================
#
# Reproduces ``cfgs/GYRE_BARE`` (NEMO 5.0.2) block-for-block from canonical
# legoESM pieces so the assembled model can be RUN and its trajectory compared
# to NEMO (not only per-term).  The bridge (``nemo_state_bridge``) builds an
# INCONSISTENT coordinate (``z_coord.H_max`` from the full 31-level e3t ladder =
# 4601.81 m, but the wet-column ``H_bathy`` = 4300.71 m from 30 wet levels),
# which leaves a spurious ``w != 0`` at the sea floor (FCT2_BOTTOM_DRIFT_
# FINDINGS.md).  This native setup builds the z* coordinate from ONLY the 30 wet
# levels, so ``H_max == H_bathy`` by construction → pure z* snaps
# ``z_half_ref[-1] = -H_max`` → ``sigma[nlev] = 0`` → ``diagnose_w`` gives w = 0
# at the sea floor with no masked-level inconsistency.

# --- Horizontal grid (usrdef_hgr.F90 / usrdef_nam.F90, nn_GYRE=1). -----------
# kpi = 30*nn_GYRE+2, kpj = 20*nn_GYRE+2 → 32 x 22 T-cells with a 1-cell closed
# land frame (30 x 20 = 600 wet).  Uniform 106 km metric (e1t=e2t=ze1, NO
# cos(lat) convergence — the un-rotated MY_SRC beta-plane), matching NEMO GYRE.
_NEMO_GYRE_NX: int = 32
_NEMO_GYRE_NY: int = 22
_NEMO_GYRE_DX_M: float = 106000.0            # ze1 = 106000/nn_GYRE [m]
# Coriolis f = f0 + beta*(phi - 15deg) on the beta-plane (usrdef_hgr): f0 at
# latitude 15deg (Hazeleger & Drijfhout 1998), beta at the reference F-point
# latitude 29deg.  The un-rotated T-cell latitudes are phi_t[j] = 29 +
# (j-0.5)*ze1deg (ze1deg = ze1/(R*rad)); the southernmost T-centre is 28.5234deg
# (verified vs mesh_mask gphit).
_NEMO_GYRE_LAT_ZPHI1_DEG: float = 29.0       # usrdef_hgr zphi1 (F-point ref lat)
_NEMO_GYRE_LAT_BETA0_DEG: float = 15.0       # usrdef_hgr zphi0 (Coriolis y=0 lat)
_NEMO_GYRE_RAD: float = np.pi / 180.0
_NEMO_GYRE_ZE1DEG: float = _NEMO_GYRE_DX_M / (constants.R_earth * _NEMO_GYRE_RAD)
_NEMO_GYRE_F0: float = (
    2.0 * constants.Omega * np.sin(_NEMO_GYRE_LAT_BETA0_DEG * _NEMO_GYRE_RAD))
_NEMO_GYRE_BETA: float = (
    2.0 * constants.Omega
    * np.cos(_NEMO_GYRE_LAT_ZPHI1_DEG * _NEMO_GYRE_RAD) / constants.R_earth)
# create_beta_plane centres y_c[j] = y_origin + (j+0.5)*dy and returns
# f = f0 + beta*y_c.  Choosing y to be arc distance from latitude 15deg
# ((phi_t[j]-15)*rad*R) makes f0/beta the NEMO values; the required origin is
# y_origin = (29-15)*rad*R - dy (the half-cell shift between NEMO's zphi0-anchored
# T-centre index and create_beta_plane's (j+0.5) convention).
_NEMO_GYRE_Y_ORIGIN_M: float = (
    (_NEMO_GYRE_LAT_ZPHI1_DEG - _NEMO_GYRE_LAT_BETA0_DEG)
    * constants.R_earth * _NEMO_GYRE_RAD - _NEMO_GYRE_DX_M)

# --- Vertical grid (usrdef_zgr.F90 zgr_z; MI96 = Madec & Imbard 1996). --------
# jpk=31 total levels, deepest is an always-dry rock level → 30 WET levels,
# H_wet = sum(e3t[:30]) = 4300.71 m.  Coefficients are the GYRE zgr_z literals.
_NEMO_GYRE_JPK: int = 31                      # jpkglo (namusr_def)
_MI96_ZSUR: float = -2033.194295283385
_MI96_ZA0: float = 155.8325369664153
_MI96_ZA1: float = 146.3615918601890
_MI96_ZKTH: float = 17.28520372419791
_MI96_ZACR: float = 5.0

# --- Time step (namelist_cfg rn_Dt). -----------------------------------------
_NEMO_GYRE_DT_S: float = 14400.0

# --- Surface thermal forcing (usrdef_sbc.F90; GYRE_BARE = thermal only, NO
# wind).  Haney restoring qns = -A*(SST - T*) - qsr with A = 40 W/m^2/K, plus a
# 2-band penetrative solar qsr.  SEASONAL analytic generators (mimicry glue →
# harness, per oracle_recipe_strategy) reproduced here so the setup is runnable.
_NEMO_GYRE_HANEY_A: float = 40.0             # usrdef_sbc ztrp magnitude [W/m^2/K]
_NEMO_GYRE_TAU_S_INERT: float = 1.0e30       # GYRE_BARE has ~no SSS restoring
_NEMO_GYRE_YEAR_DAYS: float = 360.0          # nn_leapy=30 (12 x 30-day months)
_NEMO_QSR_PI: float = 3.1415                 # usrdef_sbc literal (NOT math pi)


def _nemo_gyre_vertical_ladder() -> tuple[np.ndarray, np.ndarray]:
    """NEMO GYRE 30-wet-level reference ladder ``(e3t_1d, gdept_1d)`` [m].

    Reproduces ``usrdef_zgr.zgr_z`` analytically (MI96 depths → ``depth_to_e3`` →
    ``e3_to_depth``), matching mesh_mask ``e3t_1d``/``gdept_1d`` to machine
    precision.  Returns the WET column only (levels 1..30), so
    ``sum(e3t) == 4300.71 m == H_bathy`` — the consistent-coordinate property.
    """
    jpk = _NEMO_GYRE_JPK
    k = np.arange(1, jpk + 1, dtype=np.float64)
    depw = _MI96_ZSUR + _MI96_ZA0 * k + _MI96_ZA1 * _MI96_ZACR * np.log(
        np.cosh((k - _MI96_ZKTH) / _MI96_ZACR))
    dept = _MI96_ZSUR + _MI96_ZA0 * (k + 0.5) + _MI96_ZA1 * _MI96_ZACR * np.log(
        np.cosh((k + 0.5 - _MI96_ZKTH) / _MI96_ZACR))
    # NEMO domutil.depth_to_e3: e3t(k)=depw(k+1)-depw(k), e3w(k)=dept(k)-dept(k-1).
    e3t = np.empty(jpk)
    e3t[:-1] = depw[1:] - depw[:-1]
    e3t[-1] = 2.0 * (dept[-1] - depw[-1])
    e3w = np.empty(jpk)
    e3w[0] = 2.0 * (dept[0] - depw[0])
    e3w[1:] = dept[1:] - dept[:-1]
    # NEMO domutil.e3_to_depth: recover gdepw = cumsum(e3t), gdept via e3w.
    gdepw = np.empty(jpk)
    gdept = np.empty(jpk)
    gdepw[0] = 0.0
    gdept[0] = 0.5 * e3w[0]
    for jk in range(1, jpk):
        gdepw[jk] = gdepw[jk - 1] + e3t[jk - 1]
        gdept[jk] = gdept[jk - 1] + e3w[jk]
    n_wet = jpk - 1
    return e3t[:n_wet], gdept[:n_wet]


def nemo_gyre_latitudes(n_lat: int = _NEMO_GYRE_NY) -> np.ndarray:
    """NEMO GYRE un-rotated T-cell latitudes [deg], shape ``(n_lat,)``.

    ``phi_t[j] = zphi1 + (j - 0.5)*ze1deg`` (usrdef_hgr with sin_alpha=0).  These
    are the REAL NEMO latitudes (28.52..48.54 deg) the surface forcing needs — the
    beta-plane geometry carries meridional position in ``f``, not ``grid.lat``.
    """
    j = np.arange(n_lat, dtype=np.float64)
    return _NEMO_GYRE_LAT_ZPHI1_DEG + (j - 0.5) * _NEMO_GYRE_ZE1DEG


def _nemo_gyre_land_mask():
    """Closed-box land mask ``(NY, NX)`` (1=ocean, 0=land): 1-cell land frame.

    NEMO GYRE is a closed basin (jperio=0); the lat-lon C-grid's periodic-x roll
    is walled by the land columns, so the interior 30 x 20 stays a basin.
    """
    import jax.numpy as jnp

    mask = np.ones((_NEMO_GYRE_NY, _NEMO_GYRE_NX), dtype=np.float64)
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    return jnp.asarray(mask)


def nemo_gyre_initial_T_S(depth_pos_m):
    """NEMO GYRE analytic initial ``(T, S)`` profiles vs positive depth [m].

    Verbatim reproduction of ``usrdef_istate.usr_def_istate`` (horizontally
    uniform tanh profiles; T [degC], S [g/kg]).  ``depth_pos_m`` may be any shape;
    the profiles broadcast over it.
    """
    import jax.numpy as jnp

    d = jnp.asarray(depth_pos_m)
    # NEMO's two-piece blend weights (deep w1, shallow w2; each 0.5 at d=500 m).
    w1 = (-jnp.tanh((500.0 - d) / 150.0) + 1.0) / 2.0
    w2 = (-jnp.tanh((d - 500.0) / 150.0) + 1.0) / 2.0
    T = ((16.0 - 12.0 * jnp.tanh((d - 400.0) / 700.0)) * w1
         + (15.0 * (1.0 - jnp.tanh((d - 50.0) / 1500.0))
            - 1.4 * jnp.tanh((d - 100.0) / 100.0)
            + 7.0 * (1500.0 - d) / 1500.0) * w2)
    S = ((36.25 - 1.13 * jnp.tanh((d - 305.0) / 460.0)) * w1
         + (35.55 + 1.25 * (5000.0 - d) / 5000.0
            - 1.62 * jnp.tanh((d - 60.0) / 650.0)
            + 0.2 * jnp.tanh((d - 35.0) / 100.0)
            + 0.2 * jnp.tanh((d - 1000.0) / 5000.0)) * w2)
    return T, S


def nemo_gyre_seasonal_cosines(t_seconds: float = 0.0):
    """NEMO GYRE seasonal phase factors ``(zcos_sais1, zcos_sais2)``.

    Reproduces ``usrdef_sbc`` (nn_leapy=30, 360-day year).  ``t_seconds`` is model
    time since start; the solar (qsr) uses ``zcos_sais1`` and the restoring
    target (T*) uses ``zcos_sais2``.
    """
    import jax.numpy as jnp

    ztime = t_seconds / 3600.0                     # hours since start
    ztimemax1 = (5.0 * 30.0 + 21.0) * 24.0
    ztimemin1 = ztimemax1 + 24.0 * _NEMO_GYRE_YEAR_DAYS / 2.0
    ztimemax2 = (6.0 * 30.0 + 21.0) * 24.0
    ztimemin2 = ztimemax2 - 24.0 * _NEMO_GYRE_YEAR_DAYS / 2.0
    zcos_sais1 = jnp.cos((ztime - ztimemax1) / (ztimemin1 - ztimemax1) * jnp.pi)
    zcos_sais2 = jnp.cos((ztime - ztimemax2) / (ztimemax2 - ztimemin2) * jnp.pi)
    return zcos_sais1, zcos_sais2


def nemo_gyre_qsr(lat_deg, t_seconds: float = 0.0):
    """NEMO GYRE analytic penetrative solar ``qsr(lat)`` [W/m^2] (usrdef_sbc)."""
    import jax.numpy as jnp

    zcos_sais1, _ = nemo_gyre_seasonal_cosines(t_seconds)
    return 230.0 * jnp.cos(
        _NEMO_QSR_PI * (jnp.asarray(lat_deg) - 23.5 * zcos_sais1) / (0.9 * 180.0))


def nemo_gyre_t_star(lat_deg, t_seconds: float = 0.0):
    """NEMO GYRE analytic restoring target SST ``T*(lat)`` [degC] (usrdef_sbc)."""
    import jax.numpy as jnp

    _, zcos_sais2 = nemo_gyre_seasonal_cosines(t_seconds)
    return (28.3 * (1.0 + zcos_sais2 / 50.0) * jnp.cos(
        jnp.pi * (jnp.asarray(lat_deg) - 5.0)
        / (53.5 * (1.0 + 11.0 / 53.5 * zcos_sais2) * 2.0)))


# The GYRE-configured NEMO card: ENE vorticity + c2 KE (ln_dynvor_ene,
# nn_dynkeg=0) + EOS-80 + adcroft PGF + traldf_iso pure Redi (ln_ldfeiv=F).
# n_barotropic_substeps=120 (NOT the card default 30): the deep (H=4300 m)
# 106 km grid at dt=14400 s needs the finer external-mode substep for barotropic
# CFL — sqrt(g*H)*dt/(120*dx) ~ 0.23 (30 substeps blows up in a few steps;
# matches the reference gyre_assembled_step n_barotropic_substeps).
_NEMO_GYRE_CARD_CONFIG = NEMOModelRecipeConfig(
    momentum_core="vector_invariant_ene",
    eos="nemo_eos80",
    pgf_scheme="adcroft",
    lateral_operator="nemo_iso_lap",
    n_barotropic_substeps=120,
)


def build_nemo_gyre_recipe(
    *,
    cfg: NEMOModelRecipeConfig | None = None,
) -> NEMORecipe:
    """Native (non-bridge) NEMO GYRE_BARE setup on the reusable NEMO card.

    Assembles the already-verified NEMO-faithful blocks into a runnable GYRE:
    the un-rotated beta-plane C-grid, the 30-wet-level MI96 z* coordinate (with
    ``H_max == H_bathy == 4300.71 m`` so w = 0 at the sea floor by construction),
    the analytic tanh T/S initial state, and the GYRE card (ENE/c2/EOS-80/adcroft/
    nemo_iso_lap).  The horizontal/vertical dims are fixed by NEMO (32x22x30), so
    unlike ``build_nemo_rest_recipe``/``build_nemo_eady_recipe`` this takes no
    grid-size kwargs.  The thermal surface forcing (Haney restoring + 2-band
    solar, NO wind — GYRE_BARE) is applied POST-step by
    :func:`apply_nemo_gyre_surface_forcing` — the sanctioned harness route
    (mirrors DINO), since the in-step physics_fn protocol cannot thread the eq-8
    ``subtract_qsr`` split, and it carries the seasonal ``t_seconds`` phase.  A
    runnable step is therefore ``state = model.step(
    apply_nemo_gyre_surface_forcing(state, z_coord, dt), dt)`` (forcing then
    dynamics, iterated).
    """
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    if cfg is None:
        cfg = _NEMO_GYRE_CARD_CONFIG

    n_lat, n_lon = _NEMO_GYRE_NY, _NEMO_GYRE_NX

    # Consistent 30-wet-level z* (H_max == sum(e3t) == H_bathy == 4300.71 m).
    e3t_wet, gdept_wet = _nemo_gyre_vertical_ladder()
    z_coord = create_z_star_from_thicknesses(e3t_wet, gdept_wet)
    n_lev = int(e3t_wet.size)

    # Beta-plane geometry (uniform 106 km metric, NEMO f = f0 + beta*y).
    grid = create_beta_plane_cgrid_geometry(
        n_lat, n_lon,
        dx_m=_NEMO_GYRE_DX_M, dy_m=_NEMO_GYRE_DX_M,
        f0=_NEMO_GYRE_F0, beta=_NEMO_GYRE_BETA,
        y_origin_m=_NEMO_GYRE_Y_ORIGIN_M,
    )

    land_mask = _nemo_gyre_land_mask()

    # Analytic tanh IC at NEMO's own T-point depths (gdept), horizontally uniform.
    T_1d, S_1d = nemo_gyre_initial_T_S(jnp.asarray(gdept_wet))
    mask3 = land_mask[:, :, None]
    T = jnp.where(
        mask3 > 0.5, jnp.broadcast_to(T_1d, (n_lat, n_lon, n_lev)), 0.0)
    S = jnp.where(
        mask3 > 0.5, jnp.broadcast_to(S_1d, (n_lat, n_lon, n_lev)), 0.0)

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        H_max=z_coord.H_max,
        land_mask_override=land_mask,
    )
    state = state._replace(
        T=Field(data=T, name="T", dims=state.T.dims, units=state.T.units),
        S=Field(data=S, name="S", dims=state.S.dims, units=state.S.units),
    )

    # Model card.  GYRE_BARE thermal forcing (Haney restoring + 2-band solar) is
    # applied POST-step by :func:`apply_nemo_gyre_surface_forcing` — the
    # sanctioned harness route (mirrors DINO): the eq-8 ``subtract_qsr`` split
    # cannot be threaded through the in-step physics_fn protocol, and the lat-lon
    # C-grid consumes ``q_solar`` only via ``flux_feedback``.  physics thus keeps
    # surface_forcing "none" (no wind — GYRE_BARE) and no in-step shortwave, so
    # the model builds + steps unconditionally; all thermal forcing is the
    # applicator's.  mle off (nn_mle=0).
    model_config = nemo_lat_lon_model_config(cfg)
    physics_config = model_config.physics._replace(
        shortwave_penetration=None, mle=None)
    model_config = model_config._replace(physics=physics_config)

    return NEMORecipe(
        model_config=model_config,
        physics_config=physics_config,
        grid=grid,
        z_coord=z_coord,
        land_mask=land_mask,
        initial_state=state,
    )


class _GyreLatLonGridShim:
    """Minimal grid shim exposing the ``grid_lat`` (NEMO T-latitudes, radians)
    that ``restoring_surface_forcing`` reads (its sole grid access).

    The beta-plane geometry carries meridional position in ``f``, not
    ``grid.lat``, so the Haney target ``T*(lat)`` must be evaluated at the real
    NEMO T-latitudes — supplied here, not the pseudo-lat.  Land masking is applied
    by the caller (``mask3``), so the shim carries no land mask.
    """

    def __init__(self, grid_lat_rad):
        self.grid_lat = grid_lat_rad


def apply_nemo_gyre_surface_forcing(state, z_coord, dt, *, t_seconds=0.0):
    """Apply the GYRE Haney NON-solar restoring POST-step (harness route).

    NEMO thermal surface forcing, reproduced with the shared canonical kernels:

    * Haney NON-solar restoring — ``restoring_surface_forcing`` with the eq-8
      ``subtract_qsr`` split (``qns = -A*(SST - T*) - qsr``), analytical implicit
      Euler (stable for any dt).  SSS restoring inert (GYRE_BARE).
    * 2-band penetrative solar ``qsr`` — the shared Jerlov type-I column
      (``shortwave_penetration_tendency``, NEMO traqsr).

    eq-8 + eq-10 together = the pure Haney relaxation ``A*(T* - SST)`` distributed
    NEMO-faithfully (the ``-qsr`` the restoring subtracts is re-added by the
    penetration column, conserving heat).  ``t_seconds`` sets the seasonal phase.
    Returns a new state with T/S updated; land cells untouched.
    """
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig,
        shortwave_penetration_tendency,
    )
    from legoesm.ocean.physics.surface_forcing.config import (
        RestoringConfig,
        tau_from_flux_coefficient,
    )
    from legoesm.ocean.physics.surface_forcing.restoring import (
        restoring_surface_forcing,
    )

    n_lat, n_lon = state.T.data.shape[0], state.T.data.shape[1]
    lat_t = jnp.asarray(nemo_gyre_latitudes(n_lat))
    shape_2d = (n_lat, n_lon)
    T_star_2d = jnp.broadcast_to(nemo_gyre_t_star(lat_t, t_seconds)[:, None], shape_2d)
    qsr_2d = jnp.broadcast_to(nemo_gyre_qsr(lat_t, t_seconds)[:, None], shape_2d)

    dz_0 = float(z_coord.dz_ref[0])
    cell_mask = state.land_mask.data
    grid_lat_rad = jnp.broadcast_to(jnp.radians(lat_t)[:, None], shape_2d)
    restoring_cfg = RestoringConfig(
        T_star_array=T_star_2d,
        tau_T=tau_from_flux_coefficient(
            _NEMO_GYRE_HANEY_A,
            NEMO_CONSTANTS_CONFIG.rho_0, NEMO_CONSTANTS_CONFIG.c_sw, dz_0),
        tau_S=_NEMO_GYRE_TAU_S_INERT,
        subtract_qsr=True,            # eq-8 split (qsr re-added by penetration)
        implicit=True,
    )
    out = restoring_surface_forcing(
        state.T.data, state.S.data,
        _GyreLatLonGridShim(grid_lat_rad), restoring_cfg,
        sw_down=qsr_2d, dt=dt,
        rho_0=NEMO_CONSTANTS_CONFIG.rho_0, c_p=NEMO_CONSTANTS_CONFIG.c_sw,
        dz_0=dz_0,
    )
    # eq-10 Jerlov 2-band penetration through the column (NEMO traqsr).
    dT_dt_sw = shortwave_penetration_tendency(
        sw_down=qsr_2d,
        z_coord_dz_ref=z_coord.dz_ref,
        z_coord_z_half_ref=z_coord.z_half_ref,
        jacobian=jnp.ones_like(state.eta.data),
        config=ShortwavePenetrationConfig(scheme="jerlov_2band", water_type="I"),
        rho_0=NEMO_CONSTANTS_CONFIG.rho_0, c_sw=NEMO_CONSTANTS_CONFIG.c_sw,
    )
    mask3 = cell_mask[..., None]
    new_T = state.T.data + dt * (out.dT_dt + dT_dt_sw) * mask3
    new_S = state.S.data + dt * out.dS_dt * mask3
    return state._replace(
        T=Field(data=new_T, name=state.T.name, dims=state.T.dims, units=state.T.units),
        S=Field(data=new_S, name=state.S.name, dims=state.S.dims, units=state.S.units),
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
    if setup == "gyre":
        return build_nemo_gyre_recipe(cfg=cfg, **kwargs)
    raise ValueError(
        "unknown NEMO setup "
        f"{setup!r}; expected 'rest', 'eady', or 'gyre'.")


__all__ = (
    "NEMO_BLOCK_MAPPING",
    "NEMO_CONSTANTS_CONFIG",
    "NEMO_DEFERRED_BLOCKS",
    "NEMOModelRecipeConfig",
    "NEMORecipe",
    "apply_nemo_gyre_surface_forcing",
    "build_nemo_eady_recipe",
    "build_nemo_gyre_recipe",
    "build_nemo_recipe",
    "build_nemo_rest_recipe",
    "nemo_gyre_initial_T_S",
    "nemo_gyre_latitudes",
    "nemo_gyre_qsr",
    "nemo_gyre_seasonal_cosines",
    "nemo_gyre_t_star",
    "nemo_lat_lon_model_config",
)
