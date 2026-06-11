"""legoESM-Veros recipe: ACC *basic* channel (the transfer test).

THE TRANSFER TEST -- the oracle-recipe method's generalization experiment.

``veros.setups.acc_basic.acc_basic.ACCBasicSetup`` is the analytic TKE-only
twin of ``acc``: SAME grid (30x42x15, 2deg, ~1900 m), SAME dt
(dt_mom=4800, dt_tracer=43200), SAME topography (single western wall north of
-20deg, re-entrant channel south), SAME wind stress + T* restoring + linear T(z)
initial condition. The ONLY physics deltas are:

  1. ``enable_eke = False`` (acc has it True with 8 eke_* params + isopycnal
     diffusion). With EKE off, Veros's ``set_eke_diffusivities_kernel``
     (veros/core/eke.py:68-77) sets the GM skew diffusivity ``K_gm = K_gm_0 =
     1000`` CONSTANT and the Redi isopycnal diffusivity ``K_iso = K_iso_0 =
     1000`` CONSTANT (the ``not enable_eke`` branch + the ``always constant``
     K_iso else-branch). No prognostic eddy energy; no K_iso=K_gm coupling.
     => legoESM mapping: ``GMRediConfig.eke = None`` (constant kappa_GM /
        kappa_Redi), which is exactly the EKE-OFF FALLBACK the ACC recipe
        already documents (kappa_GM=kappa_Redi=1000).
  2. ``enable_Prandtl_tke = False`` (acc leaves it the Veros default True). With
     it False, Veros's tke.py:87-92 uses a CONSTANT Prandtl number
     ``Prandtlnumber = Prandtl_tke0 = 10`` (instead of the Richardson-dependent
     ``max(1, min(10, 6.6*Ri))``); ``kappaH = max(kappaH_min, kappaM/Prandtl)``.
     => legoESM mapping: ``TKEConfig.prandtl_mode = "constant"`` with
        ``Prandtl_tke0 = 10`` -- exactly ``K_H = max(kappaH_min, K_M/10)``.

Everything else is IDENTICAL to acc, so this recipe REUSES the acc builders
(``build_acc_grid`` / ``build_acc_z_coord`` / ``build_acc_land_mask`` /
``build_acc_state`` / ``build_acc_wind_stress`` / ``build_acc_restoring_config``)
and only swaps the two configs above. The shared ``build_acc_state`` was
parametrized (gm_redi/tke kwargs, defaulting to the ACC configs) so the
EKE-field seed branch is skipped here (eke=None) while the prognostic-TKE field
is still seeded.

THE RULE (transfer-test integrity): this recipe is built ENTIRELY from existing
canonical config options (eke=None, prandtl_mode="constant") + harness reuse. NO
numerics module is modified. acc_basic needs NO option legoESM lacks -> the
transfer succeeds at the config-construction level (the climate-closeness verdict
is measured separately under .physics-validator/transfer_acc_basic/).

Source of truth: ``veros/setups/acc_basic/acc_basic.py``.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig, LateralMixingConfig,
)
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.state import LatLonCGridOceanConfig, LatLonCGridOceanState
from legoesm.ocean.vertical import OceanZStarCoordinate

# Reuse the ACC recipe builders + shared constants verbatim. acc_basic shares
# the grid / z-coord / topography / dt / forcing with acc, so these are the
# SAME canonical harness glue -- factored, not copied (THE RULE).
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACCRecipe,
    DT_MOM_S, DT_TRACER_S, NX, NY, NZ, R_BOT, T_RESTORING_DAYS,
    X_ORIGIN_DEG, Y_ORIGIN_DEG,
    acc_A_h,
    build_acc_grid, build_acc_land_mask, build_acc_restoring_config,
    build_acc_state, build_acc_t_star, build_acc_wind_stress, build_acc_z_coord,
)


# ---------------------------------------------------------------------------
# acc_basic physics configs (the ONLY delta vs acc).
# ---------------------------------------------------------------------------

# (1) GM/Redi with EKE OFF: constant K_gm = K_iso = 1000 (Veros eke.py:68-77
#     not-enable_eke branch + always-constant K_iso). Same neutral-density
#     isoneutral slopes, same K_iso_steep=500 floor, same implicit K_33, same
#     S_max/taper as acc -- the slope/taper numerics are EKE-independent in
#     Veros (isoneutral.py). Only the diffusivity VALUE changes (prognostic ->
#     constant), so ``eke=None``.
ACC_BASIC_GM_REDI_CONFIG = GMRediConfig(
    kappa_GM=1000.0,             # Veros K_gm_0 (constant; enable_eke=False)
    kappa_Redi=1000.0,           # Veros K_iso_0 (constant; enable_eke=False)
    S_max=0.01,                  # <-> iso_slopec
    taper_width_frac=0.5,        # = iso_dslope / iso_slopec = 0.005 / 0.01
    implicit_K33=True,           # Veros applies the vertical K_33 implicitly
    K_iso_steep=500.0,           # <-> Veros K_iso_steep (acc_basic.py:42)
    slope_density="neutral",     # Veros neutral (locally-referenced) slopes
    veros_triad_weights=True,    # Veros dzw(pair)/(4 dzt) triad weights (see doc)
    double_redi_diagonal=True,   # Veros adds K_11/K_22 in BOTH the iso and skew
    #                              passes (acc_basic runs neutral+skew; see the
    #                              GMRediConfig field doc) — 2× horiz. diagonal.
    eke=None,                    # enable_eke=False -> constant kappa, no EKE
)

# (2) TKE with enable_Prandtl_tke=False: constant Prandtl = Prandtl_tke0 = 10.
#     Everything else identical to acc's TKE (prognostic carried TKE, same
#     c_k/c_eps/alpha_tke/mxl_min/tke_mxl_choice/kappaM_min/kappaH_min/
#     enable_kappaH_profile, adiabatic N^2 for the convective ventilation).
#     EKE-derived TKE sources are OFF (enable_eke=False -> Veros adds no
#     eke_diss_tke; enable_idemix=False -> no K_diss_bot into TKE either,
#     tke.py:154-166).
ACC_BASIC_TKE_CONFIG = TKEConfig(
    c_k=0.1,
    c_eps=0.7,
    alpha_tke=30.0,
    mxl_min=1.0e-8,
    tke_mxl_choice=2,
    kappaM_min=2.0e-4,
    kappaM_max=100.0,            # Veros kappaM_max default (convective ceiling)
    kappaH_min=2.0e-5,
    enable_kappaH_profile=True,
    n2_mode="adiabatic",         # Veros adiabatic static stability (convection)
    # enable_Prandtl_tke=False  ==>  CONSTANT Prandtl = Prandtl_tke0 = 10.
    prandtl_mode="constant",
    Prandtl_tke0=10.0,
    prognostic=True,             # Veros enable_tke prognostic carried-TKE form
    # EKE off + IDEMIX off => Veros adds NO recycled energy to the TKE forc.
    source_eke_diss=False,
    source_bottom_drag_diss=False,
)


def build_acc_basic_physics_config(grid: LatLonGrid | None = None, *,
                                   with_surface_forcing: bool = False,
                                   ) -> OceanPhysicsConfig:
    """acc_basic physics: TKE (constant-Prandtl) + GM/Redi (constant K) + linear
    bottom drag (via model config) + implicit vertical viscosity. IDEMIX and EKE
    are both OFF. Surface forcing identical to acc (T* restoring)."""
    if with_surface_forcing:
        if grid is None:
            grid = build_acc_grid()
        surface_forcing = SurfaceForcingConfig(
            scheme="restoring", restoring=build_acc_restoring_config(grid))
    else:
        surface_forcing = SurfaceForcingConfig(scheme="prescribed")
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke", tke=ACC_BASIC_TKE_CONFIG),
        # GM/Redi is wired at the top-level config.gm_redi (dynamics path);
        # physics.lateral_mixing stays "none" (same as the acc recipe).
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=surface_forcing,
        bottom_drag=BottomDragConfig(scheme="none"),  # see model config bottom_drag_r
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def build_acc_basic_model_config(grid: LatLonGrid | None = None, *,
                                 with_surface_forcing: bool = False,
                                 ) -> LatLonCGridOceanConfig:
    """acc_basic dynamics config -- identical to acc EXCEPT the GM/Redi (constant
    K, eke=None) and TKE (constant Prandtl) configs. All dycore numerics options
    (flux-form advection, centered flux, centered tracer, flux-divergence
    viscosity with cos(lat), nonlin2 EOS, implicit vmix, K_v=0, faithful bottom
    drag, implicit surface forcing) are SHARED with acc verbatim -- which is the
    whole point of the transfer test."""
    return LatLonCGridOceanConfig(
        g=VEROS_CONSTANTS_CONFIG.g,
        rho_0=VEROS_CONSTANTS_CONFIG.rho_0,
        constants=VEROS_CONSTANTS_CONFIG,
        A_h=acc_A_h(VEROS_CONSTANTS_CONFIG.R_earth),
        A_h_lat_scaling=True,
        A_h_cos_power=1,
        lateral_viscosity_operator="flux_divergence",
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        vertical_momentum_scheme="centered_full",
        tracer_advection="centered",
        bottom_drag_r=R_BOT,
        eos="veros_nonlin2",
        implicit_vertical_mixing=True,
        K_v=0.0,
        gm_redi=ACC_BASIC_GM_REDI_CONFIG,
        surface_forcing_implicit=with_surface_forcing,
        physics=build_acc_basic_physics_config(
            grid, with_surface_forcing=with_surface_forcing),
    )


def build_acc_basic_recipe(*, with_surface_forcing: bool = False) -> ACCRecipe:
    """One-stop constructor for the legoESM-Veros acc_basic recipe.

        from legoesm.ocean.fidelity.veros_acc_basic_recipe import (
            build_acc_basic_recipe)
        recipe = build_acc_basic_recipe(with_surface_forcing=True)

    Returns the SAME ``ACCRecipe`` NamedTuple the acc recipe uses (grid, z_coord,
    land_mask, model/physics configs, initial_state, wind_forcing). All Veros
    constants are pinned through config; no monkey-patch needed.
    """
    grid = build_acc_grid()
    z_coord = build_acc_z_coord()
    land_mask = build_acc_land_mask(grid)
    # Pass the acc_basic configs so build_acc_state seeds the prognostic-TKE
    # field but NOT an EKE field (eke=None) -- the only seed-branch difference.
    initial_state = build_acc_state(
        grid, z_coord,
        gm_redi=ACC_BASIC_GM_REDI_CONFIG, tke=ACC_BASIC_TKE_CONFIG)
    wind_forcing = (
        build_acc_wind_stress(grid) if with_surface_forcing else None)
    return ACCRecipe(
        model_config=build_acc_basic_model_config(
            grid, with_surface_forcing=with_surface_forcing),
        physics_config=build_acc_basic_physics_config(
            grid, with_surface_forcing=with_surface_forcing),
        grid=grid,
        z_coord=z_coord,
        land_mask=land_mask,
        initial_state=initial_state,
        wind_forcing=wind_forcing,
    )


__all__ = (
    "ACC_BASIC_GM_REDI_CONFIG",
    "ACC_BASIC_TKE_CONFIG",
    "build_acc_basic_model_config",
    "build_acc_basic_physics_config",
    "build_acc_basic_recipe",
)
