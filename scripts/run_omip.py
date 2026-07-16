#!/usr/bin/env python
"""Reference OMIP simulation — ocean-only forced integration on all grids.

Runs a realistic ocean simulation using WOA18-based initialization (or
analytical fallback), full physics stack (KPP, GM/Redi, SW penetration,
convection, bottom drag), and restoring surface forcing.  Supports all
four ocean grids: cubed-sphere, lat-lon, MPAS, and spectral.

Usage:
    # Quick 30-day smoke test on cubed-sphere:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py --quick --grid cubed_sphere

    # All grids, 1 year:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py --days 365 --grid all

    # Full physics with WOA18 data:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \\
        --days 365 --grid cubed_sphere --woa-t woa18_t.nc --woa-s woa18_s.nc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

# ===========================================================================
# Grid types and default resolutions / timesteps
# ===========================================================================

GRID_TYPES = ["cubed_sphere", "latlon", "mpas", "spectral"]

GRID_DEFAULTS: dict[str, dict] = {
    "cubed_sphere": {"resolution": "C24", "dt": 300.0},
    "latlon":       {"resolution": "36x72", "dt": 300.0},
    "mpas":         {"resolution": "ico3", "dt": 300.0},
    "spectral":     {"resolution": "T21", "dt": 300.0},
}

ALL_RESULTS: list[dict] = []
_WOA_BGC_PATHS: dict = {}  # WOA nutrient paths for BGC init


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Reference OMIP simulation on all ocean grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", type=str, default="all",
                   choices=GRID_TYPES + ["all"])
    p.add_argument("--resolution", type=str, default=None,
                   help="Grid resolution (e.g. C24, 36x72, ico3, T21)")
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--dt", type=float, default=None,
                   help="Timestep [s] (default: grid-specific)")
    p.add_argument("--days", type=float, default=365.0)
    p.add_argument("--quick", action="store_true",
                   help="Short 30-day run for CI")
    p.add_argument("--output", type=str, default="results/omip")
    p.add_argument("--checkpoint-days", type=float, default=30.0)
    p.add_argument("--biogeo", type=str, default="none",
                   choices=["none", "abiotic", "npzd", "npzd_v2"],
                   help="BGC scheme (default: none)")
    p.add_argument("--pco2-atm", type=float, default=400.0,
                   help="Atmospheric pCO2 [uatm] (default: 400)")
    p.add_argument("--woa-t", type=str, default=None,
                   help="WOA18 temperature NetCDF path")
    p.add_argument("--woa-s", type=str, default=None,
                   help="WOA18 salinity NetCDF path")
    p.add_argument("--woa-no3", type=str, default=None,
                   help="WOA18 nitrate NetCDF path for BGC init")
    p.add_argument("--woa-po4", type=str, default=None,
                   help="WOA18 phosphate NetCDF path for BGC init")
    p.add_argument("--woa-si", type=str, default=None,
                   help="WOA18 silicate NetCDF path for BGC init")
    p.add_argument("--woa-init", action="store_true",
                   help="Initialize T/S from WOA18 instead of rest state. "
                        "Requires --woa-t and --woa-s.")
    p.add_argument("--nudge-woa-tau", type=float, default=0.0,
                   help="Nudge T toward WOA18 with this restoring timescale [days]. "
                        "Applied after each block step. 0=disabled. "
                        "Typical 90-180 days for gentle spinup.")
    p.add_argument("--bathymetry", type=str, default=None,
                   help=(
                       "Path to ETOPO/GEBCO NetCDF bathymetry file. "
                       "When provided, loads realistic topography instead "
                       "of a flat-bottom domain."
                   ))
    p.add_argument("--H-min", type=float, default=10.0,
                   help="Minimum ocean depth [m]; shallower cells become land (default 10).")
    p.add_argument("--smoothing-passes", type=int, default=2,
                   help="Laplacian smoothing passes for bathymetry (default 2).")
    p.add_argument("--r-factor-max", type=float, default=0.2,
                   help="Maximum bathymetric slope r-factor for partial cells (default 0.2).")
    p.add_argument("--north-cap-lat", type=float, default=80.0,
                   help="Latitude [°N] above which all cells become land (default 80).")
    p.add_argument("--south-cap-lat", type=float, default=-80.0,
                   help=(
                       "Latitude [°S] below which all cells become land. "
                       "Set to -90 to disable the southern cap and let "
                       "ETOPO define Antarctica naturally (default -80)."
                   ))
    p.add_argument("--A-h", type=float, default=None,
                   help="Override Laplacian viscosity A_h [m²/s] (default: grid-dependent).")
    p.add_argument("--B-h", type=float, default=None,
                   help="Override biharmonic viscosity B_h [m⁴/s] (default: 5e9 for bathymetry).")
    p.add_argument("--K-h", type=float, default=None,
                   help="Override horizontal tracer diffusivity K_h [m²/s] (default: 1e3 with bathy).")
    p.add_argument("--no-lat-scaling", action="store_true",
                   help=(
                       "Disable cos²(lat) scaling of A_h. By default A_h is "
                       "scaled by cos²(lat) to keep the viscous CFL latitude-"
                       "independent.  This flag uses a constant A_h everywhere, "
                       "useful for diagnosing whether cos² scaling drives "
                       "high-latitude instability."
                   ))
    p.add_argument("--A-h-eq-boost", type=float, default=1.0,
                   help=(
                       "Equatorial A_h boost (>=1). Multiplies A_h by "
                       "1 + (boost-1)*exp(-(lat/sigma)^2) so eq momentum gets "
                       "extra dissipation. K_h (tracers) untouched. "
                       "Typical 3-10. 1=disabled."
                   ))
    p.add_argument("--A-h-eq-sigma", type=float, default=5.0,
                   help="Eq A_h boost Gaussian half-width [degrees]. Typical 3-7.")
    p.add_argument("--C-smag", type=float, default=None,
                   help="Smagorinsky biharmonic coefficient (dimensionless, OM4 uses 0.06).")
    p.add_argument("--C-smag-lap", type=float, default=0.15,
                   help="Laplacian Smagorinsky coefficient (dimensionless, default 0.15).")
    p.add_argument("--A-h-floor", type=float, default=2000.0,
                   help="Minimum effective A_h after latitude scaling [m²/s] (default 2000).")
    p.add_argument("--C-leith", type=float, default=None,
                   help="Leith biharmonic coefficient (dimensionless, typical 1.0-2.0).")
    p.add_argument("--pgf-scheme", type=str, default=None,
                   choices=["adcroft", "smc03"],
                   help="PGF scheme override (default smc03 with bathymetry).")
    p.add_argument("--slope-foot-alpha", type=float, default=0.0,
                   help=(
                       "Slope-foot viscosity enhancement (MOM6 OM4 KH_BG_2D analog). "
                       "Multiplies horizontal viscosity in bottom-N levels by "
                       "1 + alpha*tanh(|grad H|/H/0.1). 0=disabled, 3.0=production. "
                       "Targets African shelf, ITF, equatorial trench instabilities."
                   ))
    p.add_argument("--min-passage-width", type=int, default=0,
                   help=(
                       "Minimum passage width in grid cells. Passages narrower "
                       "than this are filled (become land). Set to 2 to eliminate "
                       "all 1-cell-wide straits that bottleneck WBCs (default 0=disabled)."
                   ))
    p.add_argument("--close-arctic-lat", type=float, default=None,
                   help=(
                       "Close off the Arctic completely above this latitude. "
                       "Unlike --north-cap-lat which just caps land, this makes "
                       "ALL cells above the latitude into land, creating a solid "
                       "wall. E.g. 65.0 closes off the entire Arctic basin."
                   ))
    p.add_argument("--sw-down", type=float, default=200.0,
                   help="Constant downwelling SW [W/m²]")
    p.add_argument("--physics", type=str, default="full",
                   choices=["full", "minimal", "none"])
    p.add_argument("--water-type", type=str, default="II",
                   choices=["I", "IA", "IB", "II", "III"])
    p.add_argument("--no-conservation-fixer", action="store_true")
    p.add_argument("--restoring-timescale", type=float, default=1095.0,
                   help="SST/SSS restoring timescale [days] (default: 1095)")
    p.add_argument("--no-restoring", action="store_true",
                   help="Disable SST/SSS restoring")
    p.add_argument("--diag-every", type=int, default=None,
                   help="Diagnostic interval in steps (default: ~1 day)")
    # Tropical-OMIP forcing (Item 4 of tropical_omip_plan.md).
    p.add_argument("--forcing-mode", type=str, default="restoring",
                   choices=["restoring", "jra55_do_tropical"],
                   help=(
                       "Surface forcing source. 'restoring' (default) uses "
                       "Haney SST/SSS restoring toward WOA. 'jra55_do_tropical' "
                       "uses LY09 bulk fluxes from a pre-built JRA55-do cache "
                       "(see scripts/prepare_omip_forcing.py). Currently "
                       "supports only --grid latlon."
                   ))
    p.add_argument("--jra55-cache", type=str, default=None,
                   help=(
                       "Path to JRA55-do Zarr cache. Required when "
                       "--forcing-mode=jra55_do_tropical."
                   ))
    p.add_argument("--jra55-co2-ppmv", type=float, default=400.0,
                   help=(
                       "Static atmospheric CO2 [ppmv] for JRA55-do mode "
                       "(default 400 — OMIP-2 protocol holds CO2 constant)."
                   ))
    p.add_argument("--jra55-cycle", action="store_true",
                   help=(
                       "Cycle the JRA55-do cache modulo its length. "
                       "Use this with a 1-year RYF cache (Stewart 2020) "
                       "for multi-year repeat-year-forcing runs. Default "
                       "off (cache must cover the requested run length, "
                       "e.g. for IAF mode)."
                   ))
    # Tropical-OMIP sponge / SSS restoring / freeze-cap (Day 3 of Item 4).
    p.add_argument("--sponge-lat-min", type=float, default=-60.0,
                   help="Southern boundary of tropical-OMIP active domain [°].")
    p.add_argument("--sponge-lat-max", type=float, default=60.0,
                   help="Northern boundary of tropical-OMIP active domain [°].")
    p.add_argument("--sponge-width-deg", type=float, default=5.0,
                   help="Sponge-zone width inside the active domain [°].")
    p.add_argument("--sponge-tau-days", type=float, default=5.0,
                   help="Sponge relaxation timescale at the boundary [days].")
    p.add_argument("--sss-piston-velocity", type=float, default=5.0e-7,
                   help=(
                       "SSS restoring piston velocity [m/s] "
                       "(default 5e-7 ≈ 200-day timescale at 10 m, NEMO/ORCA standard)."
                   ))
    p.add_argument("--T-ramp-days", type=float, default=1.0,
                   help=(
                       "Wind-stress spinup ramp timescale [days]. "
                       "tau is multiplied by min(1, t/T_ramp) to avoid "
                       "violent geostrophic adjustment from rest (default 1)."
                   ))
    p.add_argument("--jra55-no-sponge", action="store_true",
                   help="Disable the polar sponge layer.")
    p.add_argument("--jra55-no-sss-restoring", action="store_true",
                   help="Disable global SSS restoring.")
    p.add_argument("--jra55-no-freeze-cap", action="store_true",
                   help="Disable the T_freeze cap inside the sponge zone.")
    p.add_argument("--restart", type=str, default=None,
                   help=(
                       "Path to a restart_dayXXXXXX.npz file from a previous "
                       "run. When provided, the state is loaded from the "
                       "restart instead of initializing from rest. The time "
                       "loop starts from the restart day."
                   ))
    p.add_argument("--gpu-interp", action="store_true", default=True,
                   help=(
                       "Move JRA55 forcing interpolation from CPU to GPU "
                       "(DEFAULT, ~38%% faster). Loads only native 3-hourly "
                       "records and interpolates inside the lax.scan body, "
                       "reducing host-side I/O from ~288 to ~9 calls per "
                       "day-block. Use --no-gpu-interp to disable."
                   ))
    p.add_argument("--no-gpu-interp", action="store_false", dest="gpu_interp",
                   help="Disable GPU-side forcing interpolation (use CPU path).")
    p.add_argument("--no-gm-redi", action="store_true",
                   help="Disable GM/Redi isopycnal mixing (for diagnostic experiments).")
    p.add_argument("--implicit-vertical-mixing", action="store_true",
                   dest="implicit_vertical_mixing",
                   help=("Use backward-Euler implicit vertical viscosity and "
                         "diffusivity (issue #204).  Removes the explicit-CFL "
                         "limit dt < dz²/(2K) that becomes binding when "
                         "K_conv=1 m²/s convection fires with surface dz<30 m "
                         "or when vertical resolution is increased.  KPP non-"
                         "local fluxes remain explicit."))
    return p.parse_args()


# ===========================================================================
# Resolution parsing
# ===========================================================================

def _parse_resolution(grid_type: str, resolution: str) -> dict:
    if grid_type == "cubed_sphere":
        return {"n": int(resolution.lstrip("Cc"))}
    elif grid_type == "latlon":
        parts = resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif grid_type == "mpas":
        return {"level": int(resolution.replace("ico", ""))}
    elif grid_type == "spectral":
        return {"truncation": int(resolution.lstrip("Tt"))}
    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Physics config presets
# ===========================================================================

def _build_physics_config(preset: str, water_type: str):
    """Build OceanPhysicsConfig from a preset name."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    if preset == "none":
        return None

    if preset == "minimal":
        return OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant"),
            lateral_mixing=LateralMixingConfig(scheme="harmonic"),
            surface_forcing=SurfaceForcingConfig(scheme="restoring"),
            bottom_drag=BottomDragConfig(scheme="linear"),
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=ShortwavePenetrationConfig(water_type=water_type),
        )

    # "full" preset
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=LateralMixingConfig(scheme="gm_redi"),
        surface_forcing=SurfaceForcingConfig(scheme="restoring"),
        bottom_drag=BottomDragConfig(scheme="quadratic"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=ShortwavePenetrationConfig(water_type=water_type),
    )


# ===========================================================================
# Grid + model creation
# ===========================================================================

def _create_setup(grid_type: str, resolution: str, nlev: int, H_max: float,
                  physics_preset: str, water_type: str,
                  use_bathymetry: bool = False,
                  A_h_override: float = None,
                  B_h_override: float = None,
                  K_h_override: float = None,
                  A_h_eq_boost: float = 1.0,
                  A_h_eq_sigma_deg: float = 5.0,
                  C_smag: float = None,
                  C_smag_lap: float = 0.15,
                  A_h_floor: float = 2000.0,
                  C_leith: float = None,
                  pgf_scheme: str = None,
                  slope_foot_alpha: float = 0.0,
                  no_lat_scaling: bool = False,
                  no_gm_redi: bool = False,
                  implicit_vertical_mixing: bool = False,
                  biogeo: str = "none",
                  pco2_atm: float = 400.0,
                  woa_no3: str | None = None,
                  woa_po4: str | None = None,
                  woa_si:  str | None = None):
    """Create grid, z_coord, config, model for any grid type.

    All grids use the SAME config-based diffusion (A_h, K_h, A_v, K_v)
    via ``physics=None`` (built-in tendencies) so that the 4 grids are
    physically equivalent.  The ``physics_preset`` only affects which
    OceanPhysicsConfig modules are enabled on cubed-sphere grids where
    the modular pipeline is supported.

    Returns (grid, z_coord, config, model, coord_kind).
    """
    from legoesm.ocean.vertical import create_ocean_z_star
    if use_bathymetry:
        # Partial cells with ETOPO: use the same vertical stretching
        # as the global-overturning production scripts (dz_surface=20,
        # dz_deep=500) to avoid degenerate thin layers.
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=H_max, dz_surface=20.0, dz_deep=500.0,
        )
    else:
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    params = _parse_resolution(grid_type, resolution)

    # Mixing coefficients tuned per grid for equivalent effective diffusion
    # at ~5° resolution.  FV grids (cubed-sphere, latlon, MPAS) need higher
    # explicit K_h because the discrete Laplacian has truncation error;
    # spectral grids are spectrally accurate and rely on hyperdiffusion.
    A_v = 1.0e-3  # vertical viscosity [m²/s] (all grids)
    K_v = 1.0e-4  # vertical tracer diffusivity [m²/s] (all grids)
    if grid_type == "spectral":
        A_h = 1.0e4   # spectral Laplacian is exact: less needed
        K_h = 1.0e3
    elif grid_type == "mpas":
        A_h = 1.0e4   # MPAS ico3 is very coarse (~900 km); lower K_h stable
        K_h = 1.0e3
    else:
        A_h = 1.0e5   # cubed-sphere/latlon need more dissipation at ~5°
        K_h = 1.0e5

    if grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig

        grid = create_cubed_sphere(params["n"])
        # physics=None → use built-in A_h/K_h/A_v/K_v diffusion
        # (same as latlon/MPAS/spectral for consistency).
        config = OceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            n_barotropic_substeps=30,
            use_conservation_fixer=True,
            physics=None,
        )
        model = OceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "cube"

    elif grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        if use_bathymetry:
            # Production config for real ETOPO bathymetry, matching the
            # global-overturning realistic-geometry scripts that run
            # stable 50+ year integrations.  Key ingredients:
            #   - implicit CN barotropic solver (no checkerboard mode)
            #   - SMC03 density-Jacobian PGF (accurate on partial cells)
            #   - biharmonic viscosity (damps topographic comp. modes)
            #   - linear bottom drag with BBL
            #   - convective adjustment (enhanced diffusion)
            #   - GM/Redi isopycnal mixing (Visbeck adaptive)
            #   - K_h=0 (GM/Redi replaces horizontal tracer diffusion)
            from legoesm.ocean.physics.combined import OceanPhysicsConfig
            from legoesm.ocean.physics.convection.config import (
                OceanConvectionConfig, EnhancedDiffusionConfig,
            )
            from legoesm.ocean.physics.lateral_mixing.config import (
                GMRediConfig, VisbeckConfig, LateralMixingConfig,
            )
            from legoesm.ocean.physics.vertical_mixing.config import (
                VerticalMixingConfig, KPPConfig,
            )
            bathy_physics = OceanPhysicsConfig(
                vertical_mixing=VerticalMixingConfig(
                    scheme="kpp",
                    kpp=KPPConfig(),
                ),
                convection=OceanConvectionConfig(
                    scheme="enhanced_diffusion",
                    enhanced_diffusion=EnhancedDiffusionConfig(
                        K_conv=1.0, K_bg=1e-5,
                    ),
                ),
                # Disable the physics pipeline's lateral mixing — the
                # C-grid model applies its own A_h/B_h/K_h viscosity
                # and GM/Redi is wired via the gm_redi config field.
                lateral_mixing=LateralMixingConfig(scheme="none"),
                # Disable shortwave penetration — q_net already includes
                # SW, so the physics SW module would double-count.
                shortwave_penetration=None,
            )
            bathy_gm_redi = GMRediConfig(
                kappa_GM=800.0,
                kappa_Redi=800.0,
                S_max=0.005,
                visbeck=VisbeckConfig(
                    enabled=True,
                    alpha=0.015,
                    kappa_min=200.0,
                    kappa_max=2000.0,
                ),
            )
            _A_h = A_h_override if A_h_override is not None else 2.0e5
            _B_h = B_h_override if B_h_override is not None else 5.0e9
            _K_h = K_h_override if K_h_override is not None else 1e3
            _C_smag = C_smag if C_smag is not None else 0.0
            _C_leith = C_leith if C_leith is not None else 0.0
            config = LatLonCGridOceanConfig(
                A_h=_A_h, A_h_lat_scaling=(not no_lat_scaling),
                A_h_floor=A_h_floor,
                A_h_eq_boost=A_h_eq_boost,
                A_h_eq_sigma_deg=A_h_eq_sigma_deg,
                K_h=_K_h, A_v=A_v, K_v=K_v,
                B_h=_B_h,
                C_smag=_C_smag,
                C_smag_lap=C_smag_lap,
                C_leith=_C_leith,
                C_leith_modified=(_C_leith > 0),
                slope_foot_alpha=slope_foot_alpha,
                # A2: biharmonic hyperviscosity on the DEPTH-MEAN
                # (U_bar, V_bar) only.  Surgically damps the barotropic
                # standing mode at deep cells next to steep slopes
                # (Rhines 1969 bottom-trapped wave with f≈0) without
                # touching baroclinic geostrophy.  HIM/MOM6
                # BIHARMONIC_BAROTROPIC analog at 1° global resolution.
                # Scaled per Griffies-Hallberg 2000: ν₄ ≈ Δx³·U/8.
                # At 1° (Δx≈111 km, U≈1 m/s) that's 1.7e14 m⁴/s.
                B_h_barotropic=1.0e14,
                bottom_drag_r=2.5e-3,
                bottom_drag_bbl_thickness=100.0,
                # MOM6 OM4 DRAG_BG_VEL — quadratic-with-floor drag.
                # At standing-mode amplitudes (~0.05 m/s) this gives ~3×
                # more drag than pure linear, which is the cheapest
                # production fix for the deep-cell barotropic mode at
                # steep slopes.  Recovers linear drag (bit-exact) at
                # |u|→0; scales as Cd·|u| for |u|≫u_bg.
                bottom_drag_bg_velocity=0.1,
                n_barotropic_substeps=30,
                use_conservation_fixer=True,
                physics=bathy_physics,
                gm_redi=None if no_gm_redi else bathy_gm_redi,
                barotropic_solver="implicit_cn",
                pgf_scheme=pgf_scheme if pgf_scheme is not None else "smc03",
                # MOM6 MAXVEL: clip barotropic velocities to prevent
                # blowup from WBC intensification at coarse resolution.
                # MOM6 default is 6.0 m/s; we use 3.0 since realistic
                # currents at 1° shouldn't exceed ~2 m/s.
                maxvel_barotropic=0.0,  # disabled — let physics handle it
                implicit_vertical_mixing=implicit_vertical_mixing,
            )
        else:
            config = LatLonCGridOceanConfig(
                A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
                n_barotropic_substeps=30,
                use_conservation_fixer=True,
                physics=None,
                implicit_vertical_mixing=implicit_vertical_mixing,
            )
        # ── BGC setup (latlon) ───────────────────────────────────────────
        _bgc_cfg = None
        if biogeo != "none":
            from legoesm.ocean.biogeochemistry import BiogeoConfig
            _no3_tgt = _WOA_BGC_PATHS.get('NO3_target')
            _bgc_cfg = BiogeoConfig(
                scheme=biogeo, pCO2_atm=pco2_atm, wind_speed=7.0,
                nudge_nutrients=(_no3_tgt is not None),
                tau_nudge_days=365.0,
                NO3_target=_WOA_BGC_PATHS.get('NO3_target'),
                PO4_target=_WOA_BGC_PATHS.get('PO4_target'),
                Si_target =_WOA_BGC_PATHS.get('Si_target'),
            )
            # Store WOA nutrient paths in module-level dict for BGC init
            _WOA_BGC_PATHS["no3"] = woa_no3
            _WOA_BGC_PATHS["po4"] = woa_po4
            _WOA_BGC_PATHS["si"]  = woa_si
        model = LatLonCGridOceanModel(grid, z_coord, config, bgc_cfg=_bgc_cfg)
        return grid, z_coord, config, model, "latlon"

    elif grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        mesh = create_voronoi_mesh(params["level"])
        config = MPASOceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            n_barotropic_substeps=30,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        return mesh, z_coord, config, model, "mpas"

    elif grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig

        grid = create_gaussian_grid(params["truncation"])
        # Keep eta_hyperdiff for barotropic stability (required by unsplit
        # SSP-RK3), but reduce 3D hyperdiffusion to be more consistent
        # with the explicit A_h/K_h on the other grids.
        config = SpectralOceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            # Keep default hyperdiffusion coefficients — they are tuned
            # for stability of the unsplit SSP-RK3 spectral solver.
        )
        model = SpectralOceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "gaussian"

    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# State initialization with WOA T/S
# ===========================================================================

def _init_rest_state(grid_type, grid, z_coord, H_max,
                     H_bathy=None, land_mask=None):
    """Create rest-state initial condition (zero velocity, exponential T, uniform S).

    Uses each grid's standard rest_state function, which provides a
    horizontally uniform stratified profile.  This avoids creating strong
    pressure gradients from horizontal T/S contrasts.

    Parameters
    ----------
    H_bathy, land_mask : array or None
        When provided, override the default flat-bottom / idealized
        bathymetry with realistic topography (e.g. from ETOPO).
    """
    # land_lat_threshold=80 → land at high latitudes (standard for ocean
    # test cases).  The spectral model requires land boundaries to constrain
    # the barotropic mode (eta_hyperdiff is tuned for this case).
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=H_max,
            H_bathy_override=H_bathy,
            land_mask_override=land_mask,
        )
    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        return rest_state_spectral_ocean(grid, z_coord, H_max=H_max)
    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Surface forcing construction
# ===========================================================================

def _build_surface_forcing(grid_type, grid, sw_down_value):
    """Create OceanSurfaceForcing with constant SW for the given grid."""
    from legoesm.ocean.state import OceanSurfaceForcing

    if grid_type == "cubed_sphere":
        shape = (6, grid.n, grid.n)
    elif grid_type == "latlon":
        shape = (grid.lat.shape[0], grid.lon.shape[0])
    elif grid_type == "mpas":
        # MPAS model.step doesn't accept surface_forcing yet
        return None
    elif grid_type == "spectral":
        # Spectral model doesn't use surface_forcing pipeline
        return None
    else:
        return None

    sw = jnp.full(shape, sw_down_value)
    return OceanSurfaceForcing(sw_down=sw)


# ===========================================================================
# Grid-agnostic SST/SSS restoring
# ===========================================================================

def _apply_restoring(state, grid_type, grid, T_target, S_target, dt, tau_s):
    """Apply SST/SSS restoring toward WOA climatology.

    This is the OMIP-standard Haney (1971) surface flux restoring:
    dT/dt|surface = -(T_surface - T*) / tau, applied to the top layer only.

    Works identically across all grid types by operating on the state
    arrays directly.

    Parameters
    ----------
    state : ocean state (any grid type)
    grid_type : str
    T_target, S_target : jnp.ndarray
        Target surface T/S from WOA, shape matching the surface layer.
    dt : float
        Timestep [s].
    tau_s : float
        Restoring timescale [s].
    """
    # Restoring coefficient: fraction toward target per step
    alpha = dt / tau_s

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis
        # Apply restoring directly in spectral space to avoid aliasing
        # from repeated synthesis→modify→analysis cycles.
        # T_target and S_target are grid-space 2D fields that were
        # pre-transformed to spectral coefficients and stored alongside
        # the restoring targets (see run_omip_single).
        T_hat = state.T_hat.data
        S_hat = state.S_hat.data
        # T_target / S_target are spectral coefficients of the WOA
        # surface field (pre-computed once).
        T_hat_new = T_hat.at[:, 0].set(
            T_hat[:, 0] - alpha * (T_hat[:, 0] - T_target),
        )
        S_hat_new = S_hat.at[:, 0].set(
            S_hat[:, 0] - alpha * (S_hat[:, 0] - S_target),
        )
        return state._replace(
            T_hat=state.T_hat.replace(data=T_hat_new),
            S_hat=state.S_hat.replace(data=S_hat_new),
        )

    # FV grids: cubed-sphere (6,n,n,nlev), latlon (nlat,nlon,nlev),
    #           MPAS (nCells, nlev)
    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data

    # Restore surface layer only, ocean cells only
    if T.ndim == 4:
        # cubed-sphere: mask shape (6,n,n), target shape (6,n,n)
        mask_sfc = mask
    elif T.ndim == 3:
        # latlon: mask shape (nlat,nlon), target shape (nlat,nlon)
        mask_sfc = mask
    else:
        # MPAS: mask shape (nCells,), target shape (nCells,)
        mask_sfc = mask

    T_new = T.at[..., 0].set(
        T[..., 0] - alpha * (T[..., 0] - T_target) * mask_sfc,
    )
    S_new = S.at[..., 0].set(
        S[..., 0] - alpha * (S[..., 0] - S_target) * mask_sfc,
    )

    return state._replace(
        T=state.T.replace(data=T_new),
        S=state.S.replace(data=S_new),
    )


# ===========================================================================
# Tropical OMIP — JRA55-do forcing path (Item 4 of tropical_omip_plan.md)
# ===========================================================================

def _setup_jra55_forcing_state(args, grid, grid_type,
                               z_coord=None, T_woa=None, S_woa=None):
    """Open the JRA55-do cache and pre-compute per-step constants.

    Returns a dict with everything ``_jra55_step`` needs:
    cache path, ref_year, 2-D lat/lon arrays in radians, ``CouplerConfig``
    with the Item-1 LY09 / 0.98 q_sat / z_t/z_q fixes, plus optional
    sponge / SSS-restoring / T_freeze-cap state when ``z_coord`` and
    the WOA targets are provided.

    Validates that the cache target grid matches the model grid;
    mismatched cache must be rebuilt with the right resolution.

    Parameters
    ----------
    z_coord : OceanZStarCoordinate or None
        Vertical coordinate of the model. Required for SSS restoring
        (uses the top-layer thickness for the piston-velocity step).
    T_woa, S_woa : ndarray or None, shape (n_lat, n_lon, nlev)
        WOA monthly climatology targets. Required for the sponge T,S
        references and the SSS-restoring target. If either is None,
        the corresponding feature is disabled regardless of the
        ``--jra55-no-...`` flags.
    """
    if grid_type != "latlon":
        raise ValueError(
            "--forcing-mode jra55_do_tropical currently supports only "
            f"--grid latlon (got {grid_type!r}). Other grids will need "
            "an additional regridding step."
        )
    if args.jra55_cache is None:
        raise ValueError(
            "--forcing-mode jra55_do_tropical requires --jra55-cache PATH."
        )
    cache_path = Path(args.jra55_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"JRA55-do cache not found at {cache_path}. Build one with "
            "scripts/prepare_omip_forcing.py."
        )

    import xarray as xr
    ds = xr.open_zarr(str(cache_path), decode_times=False)
    cache_n_lat = int(ds.sizes["lat"])
    cache_n_lon = int(ds.sizes["lon"])
    model_n_lat = int(grid.lat.shape[0])
    model_n_lon = int(grid.lon.shape[0])
    if (cache_n_lat, cache_n_lon) != (model_n_lat, model_n_lon):
        raise ValueError(
            f"JRA55-do cache grid ({cache_n_lat}×{cache_n_lon}) does not "
            f"match model grid ({model_n_lat}×{model_n_lon}). Rebuild the "
            "cache with prepare_omip_forcing.py at the matching resolution."
        )

    # Build 2-D lat/lon (in radians) for cos_zenith / atm_to_surface.
    lat_2d = jnp.asarray(np.deg2rad(np.asarray(grid.lat))[:, None])
    lon_2d = jnp.asarray(np.deg2rad(np.asarray(grid.lon))[None, :])

    # Coupler config: LY09 bulk flux at 10 m winds, 2 m T/q (the JRA55-do
    # convention). The Item 1 fixes (LY09 U^6 term, 0.98 q_sat, separate
    # reference heights) are wired through CouplerConfig.
    from legoesm.coupler.config import CouplerConfig
    coupler_cfg = CouplerConfig(
        bulk_scheme="large_yeager",
        z_ref=10.0,
        z_t_atm=2.0,
        z_q_atm=2.0,
    )

    state: dict = {
        "cache_path": str(cache_path),
        "ref_year": int(ds.attrs.get("ref_year", 1958)),
        "lat_2d": lat_2d,
        "lon_2d": lon_2d,
        "coupler_cfg": coupler_cfg,
        "co2_ppmv": float(args.jra55_co2_ppmv),
        # Cycle the cache modulo its length when --jra55-cycle is set.
        # This is the Stewart 2020 RYF path: a single-year cache drives
        # a multi-year run by replaying the same 12 months.
        "cycle": bool(getattr(args, "jra55_cycle", False)),
        # Wind-stress spinup ramp: tau *= min(1, t / T_ramp).
        # Stored in seconds for direct use in the step functions.
        "T_ramp_seconds": float(getattr(args, "T_ramp_days", 1.0)) * 86400.0,
    }

    # Sponge layer at 60°S/60°N — uses the existing
    # legoesm.ocean.sponge.compute_sponge_gamma_latlon utility. Active
    # only when WOA targets are available (the sponge needs T_ref, S_ref).
    sponge_enabled = (
        not args.jra55_no_sponge
        and T_woa is not None
        and S_woa is not None
    )
    if sponge_enabled:
        from legoesm.ocean.sponge import compute_sponge_gamma_latlon
        sponge_gamma = compute_sponge_gamma_latlon(
            grid,
            lat_south=args.sponge_lat_min,
            lat_north=args.sponge_lat_max,
            width_deg=args.sponge_width_deg,
            timescale_days=args.sponge_tau_days,
        )
        state["sponge_gamma_2d"] = jnp.asarray(sponge_gamma)
        state["sponge_T_ref_3d"] = jnp.asarray(T_woa)
        state["sponge_S_ref_3d"] = jnp.asarray(S_woa)
    state["enable_sponge"] = sponge_enabled

    # SSS restoring — Haney piston-velocity formulation, applied
    # globally (i.e. on every ocean cell) after the dynamics step.
    sss_restoring_enabled = (
        not args.jra55_no_sss_restoring
        and S_woa is not None
        and z_coord is not None
    )
    if sss_restoring_enabled:
        state["sss_target_2d"] = jnp.asarray(S_woa[..., 0])
        state["sss_piston_velocity"] = float(args.sss_piston_velocity)
        state["dz_top"] = float(np.asarray(z_coord.dz_ref)[0])
    state["enable_sss_restoring"] = sss_restoring_enabled

    # T_freeze cap — stand-in for the missing sea-ice model.
    # When a sponge is active, cap only inside the sponge zone.
    # When no sponge, cap globally over all ocean cells.
    freeze_cap_enabled = not args.jra55_no_freeze_cap
    state["enable_freeze_cap"] = freeze_cap_enabled
    from legoesm import constants as _consts
    # State temperature is stored in °C per the lat-lon C-grid ocean
    # convention (init_latlon_cgrid + init_woa both use °C). The cap
    # threshold therefore lives in °C: T_freeze_ocean (271.35 K) minus
    # T_freeze (273.15 K) = -1.8 °C, the seawater freezing point.
    state["T_freeze_ocean_C"] = float(_consts.T_freeze_ocean - _consts.T_freeze)

    return state


def _build_sponge_forcing(jra55_state):
    """Build a ``SpongeForcing`` from the precomputed tropical-OMIP state."""
    from legoesm.ocean.sponge import SpongeForcing
    return SpongeForcing(
        gamma=jra55_state["sponge_gamma_2d"],
        T_ref=jra55_state["sponge_T_ref_3d"],
        S_ref=jra55_state["sponge_S_ref_3d"],
    )


def _apply_sss_restoring(state, jra55_state, dt):
    """Haney SSS restoring with a piston-velocity formulation.

    Applied as a post-step operation on the surface salinity layer
    of the lat-lon C-grid state. Restricted to ocean cells via the
    state's land mask.

    The discretised step is::

        alpha = piston_velocity * dt / dz_top
        S_top_new = S_top - alpha * (S_top - S_target)

    For ``piston_velocity = 5e-7 m/s`` and ``dz_top = 10 m``, the
    e-folding timescale is ~230 days — light enough to let the
    bulk-flux freshwater dominate but stiff enough to damp drift.
    """
    alpha = (
        jra55_state["sss_piston_velocity"]
        * dt
        / max(jra55_state["dz_top"], 1e-6)
    )
    S = state.S.data
    mask = state.land_mask.data  # 2-D
    # Cast target + alpha to S's dtype so scatter doesn't trip the
    # JAX dtype-promotion FutureWarning when the state runs in float32.
    S_target = jnp.asarray(jra55_state["sss_target_2d"], dtype=S.dtype)
    alpha = jnp.asarray(alpha, dtype=S.dtype)
    mask = jnp.asarray(mask, dtype=S.dtype)
    S_top_new = S[..., 0] - alpha * (S[..., 0] - S_target) * mask
    S_new = S.at[..., 0].set(S_top_new)
    return state._replace(S=state.S.replace(data=S_new))


def _apply_freeze_cap(state, jra55_state):
    """Cap surface T from below at ``T_freeze_ocean`` globally.

    Stand-in for the missing sea-ice model: prevent the surface layer
    from cooling below seawater's freezing point (-1.8°C). Without
    this, JRA55-do bulk-flux heat loss over polar regions (where the
    atmosphere is very cold) drives SST below freezing and produces
    unphysical densities.

    When a sponge is active, the cap is scoped to the sponge zone only
    (backward-compatible). When no sponge, the cap applies to all
    ocean cells.
    """
    sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
    if sponge_gamma is not None and jnp.any(sponge_gamma > 0):
        # Sponge active: cap only inside sponge zone
        freeze_mask = sponge_gamma > 0.0
    else:
        # No sponge: cap globally over all ocean cells
        freeze_mask = state.land_mask.data > 0.5
    T = state.T.data
    T_top = T[..., 0]
    T_freeze_C = jnp.asarray(jra55_state["T_freeze_ocean_C"], dtype=T.dtype)
    T_top_capped = jnp.where(
        freeze_mask,
        jnp.maximum(T_top, T_freeze_C),
        T_top,
    )
    T_new = T.at[..., 0].set(T_top_capped)
    return state._replace(T=state.T.replace(data=T_new))


def _jra55_step(state, step_idx, dt, model, jra55_state):
    """One forced-ocean step under JRA55-do bulk-flux forcing.

    Per-step pipeline:
      1. Map step index to fractional simulation day (noleap, since
         ``ref_year-01-01``).
      2. Load JRA55Slice from the cache (linear-in-time interp between
         the bracketing 3-hourly records).
      3. Build AtmToSurface from the slice + model lat/lon + day.
      4. Call ``ocean_tile_response`` with surface SST and zero ocean
         current (forced runs at 1° treat |u_o| << |u_a|).
      5. Compute net heat flux into ocean from SW/LW/SH/LH + LW up.
      6. Build FreshwaterForcing with E from the latent heat flux.
      7. Step the ocean model with ``freshwater=``, ``surface_forcing=``,
         and (if enabled) ``sponge=``.
      8. Apply post-step SSS restoring and the T_freeze cap.

    Steps 1–7 are the Day-2 path; steps 7-sponge and 8 are Day 3.
    The ``enable_sponge``, ``enable_sss_restoring``, ``enable_freeze_cap``
    flags in ``jra55_state`` gate each piece independently.
    """
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.forcing.jra55_do import (
        jra55_to_atm_surface,
        jra55_to_freshwater,
        load_jra55_slice,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    day = float(step_idx) * dt / 86400.0
    slc = load_jra55_slice(
        jra55_state["cache_path"], day,
        ref_year=jra55_state["ref_year"],
        cycle=jra55_state.get("cycle", False),
    )
    atm = jra55_to_atm_surface(
        slc,
        jra55_state["lat_2d"],
        jra55_state["lon_2d"],
        day,
        ref_year=jra55_state["ref_year"],
        co2_ppmv=jra55_state["co2_ppmv"],
    )

    # State T is stored in °C; the bulk-flux solver and q_sat lookup
    # need K. This matches the conversion in
    # ocean/physics/surface_forcing/bulk_formulas.py:48.
    from legoesm import constants as _consts
    sst_K = state.T.data[..., 0] + _consts.T_freeze
    # Forced-ocean approximation: at 1° non-eddying resolution
    # |u_ocean| ~ 0.1 m/s << |u_atm| ~ 10 m/s, so we drop the
    # surface-current correction in the bulk-flux solver.
    u_o = jnp.zeros_like(sst_K)
    v_o = jnp.zeros_like(sst_K)

    tile_resp = ocean_tile_response(
        atm, sst_K, u_o, v_o, jra55_state["coupler_cfg"],
    )

    # Wind-stress spinup ramp: scale tau by min(1, t/T_ramp) so that
    # a rest-state ocean accelerates gently under the applied forcing,
    # avoiding violent geostrophic adjustment on the first few steps.
    T_ramp = jra55_state.get("T_ramp_seconds", 86400.0)
    t_sim = float(step_idx) * dt
    ramp = min(1.0, t_sim / T_ramp) if T_ramp > 0 else 1.0

    # Net heat into ocean (positive into ocean):
    #   Q_net = SW_absorbed + LW_down − LW_up − SH − LH
    # where SH and LH are positive upward (out of ocean) per the
    # TileResponse contract.
    sw_net = atm.sw_down * (1.0 - tile_resp.albedo)
    q_net = (
        sw_net
        + atm.lw_down
        - tile_resp.lw_up
        - tile_resp.shflx
        - tile_resp.lhflx
    )

    fw = jra55_to_freshwater(slc, tile_resp.lhflx)
    sf = OceanSurfaceForcing(
        sw_down=atm.sw_down,
        q_net=q_net,
        tau_x=tile_resp.tau_x * ramp,
        tau_y=tile_resp.tau_y * ramp,
        freshwater=None,  # using the structured FreshwaterForcing path
    )

    if jra55_state.get("enable_sponge", False):
        sponge = _build_sponge_forcing(jra55_state)
        # Ramp sponge strength alongside wind stress so the relaxation
        # doesn't create violent pressure gradients from a standing start.
        sponge = sponge._replace(gamma=sponge.gamma * ramp)
    else:
        sponge = None

    state = model.step(
        state, dt,
        freshwater=fw,
        surface_forcing=sf,
        sponge=sponge,
    )

    # Post-step closure-domain operations.
    if jra55_state.get("enable_sss_restoring", False):
        state = _apply_sss_restoring(state, jra55_state, dt)
    if jra55_state.get("enable_freeze_cap", False):
        state = _apply_freeze_cap(state, jra55_state)

    return state


# ===========================================================================
# JRA55-do scan-block path (Item 4 follow-up — multi-core utilisation)
# ===========================================================================
#
# The plain Python time loop calling ``model.step`` once per step is
# correct but single-core-bound: XLA can't see across the loop, so the
# Eigen / BLAS threadpools don't get a useful work item per call at 1°
# resolution. The realistic-geometry GO continuation scripts wrap N
# steps in ``jax.lax.scan`` inside a single ``@jax.jit``; that gives
# XLA one big computation graph and 5–10× speedup at 1° on multi-core
# CPU.
#
# We can't put ``load_jra55_slice`` inside ``lax.scan`` (Zarr I/O is
# not a JAX op). Instead the outer Python layer pre-loads N steps of
# forcing into stacked JAX arrays once, then calls a JIT-compiled
# block function that scans through them.

def _preload_jra55_forcing_block(start_step_idx, n_steps, dt, jra55_state):
    """Pre-load N steps of JRA55-do forcing into stacked JAX arrays.

    Returns a tuple ``(atm_stack, runoff_stack)`` where ``atm_stack``
    is a dict of stacked AtmToSurface fields with shape
    ``(n_steps, n_lat, n_lon)`` and ``runoff_stack`` is the per-step
    friver field with the same shape.

    Pure host-side I/O — runs once per block, then the JIT-compiled
    block_fn consumes the result.
    """
    from legoesm.forcing.jra55_do import (
        jra55_to_atm_surface,
        load_jra55_block,
    )

    cache_path = jra55_state["cache_path"]
    ref_year = jra55_state["ref_year"]
    cycle = jra55_state.get("cycle", False)
    lat_2d = jra55_state["lat_2d"]
    lon_2d = jra55_state["lon_2d"]
    co2_ppmv = jra55_state["co2_ppmv"]

    # Bulk-read all N steps in one Zarr open + contiguous slab read.
    start_day = start_step_idx * dt / 86400.0
    slices = load_jra55_block(
        cache_path, start_day, n_steps, dt,
        ref_year=ref_year, cycle=cycle,
    )

    # Names match AtmToSurface field set; collected per-step then stacked.
    fields = (
        "sw_down", "lw_down", "precip_total", "precip_snow",
        "T_lowest", "q_lowest", "u_lowest", "v_lowest",
        "p_lowest", "p_surface", "rho_lowest", "cos_zenith",
    )
    accum: dict[str, list] = {f: [] for f in fields}
    runoffs: list = []

    for k, slc in enumerate(slices):
        day = (start_step_idx + k) * dt / 86400.0
        atm = jra55_to_atm_surface(
            slc, lat_2d, lon_2d, day,
            ref_year=ref_year, co2_ppmv=co2_ppmv,
        )
        for f in fields:
            accum[f].append(getattr(atm, f))
        runoffs.append(slc.friver)

    atm_stack = {f: jnp.stack(accum[f]) for f in fields}
    runoff_stack = jnp.stack(runoffs)
    return atm_stack, runoff_stack


def _preload_jra55_raw_records(start_step_idx, n_steps, dt, jra55_state):
    """Pre-load only the native 3-hourly JRA55 records that bracket a block.

    **GPU-interp path (default, --gpu-interp).**

    The JRA55 cache has 8 records/day (3-hourly).  At dt=300s, each
    simulated day has 288 timesteps.  The old CPU path called
    ``jra55_to_atm_surface`` 288 times per day in Python, spending
    ~1.9s/day on host-side I/O.  This path loads only the ~9 native
    records that bracket the block (~0.2s/day) and defers the linear
    interpolation + solar zenith to the JIT-compiled ``lax.scan`` body
    on GPU.  Result: 38% overall speedup (9.5x I/O reduction).

    Returns ``(raw_stack, runoff_stack, record_meta)`` where:
    - ``raw_stack``: dict of ``(n_records, n_lat, n_lon)`` arrays for
      each JRA55 raw variable (uas, vas, tas, huss, psl, rsds, rlds,
      prra, prsn)
    - ``runoff_stack``: ``(n_records, n_lat, n_lon)`` friver
    - ``record_meta``: dict with ``record_days`` (fractional day of each
      record), ``block_start_day``, ``dt``, ``n_steps`` — enough for the
      scan body to compute interpolation weights
    """
    import xarray as xr
    from legoesm.forcing.jra55_do import (
        JRA55_VARIABLES, RECORDS_PER_DAY,
    )

    cache_path = jra55_state["cache_path"]
    ref_year = jra55_state["ref_year"]
    cycle = jra55_state.get("cycle", False)

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    n_cache_records = int(ds.attrs["n_records"])
    cache_length_days = n_cache_records / RECORDS_PER_DAY

    # Find the range of 3-hourly record indices needed.
    start_day = start_step_idx * dt / 86400.0
    end_day = (start_step_idx + n_steps - 1) * dt / 86400.0

    if cycle:
        start_day_c = start_day % cache_length_days
        end_day_c = end_day % cache_length_days
    else:
        start_day_c = start_day
        end_day_c = end_day

    i_first = int(np.floor(start_day_c * RECORDS_PER_DAY))
    i_last = int(np.floor(end_day_c * RECORDS_PER_DAY)) + 1  # +1 for upper bracket
    i_last = min(i_last, n_cache_records - 1)

    # Handle wrap-around for cycling
    if cycle and i_last < i_first:
        # Block spans the cache boundary — read both pieces
        indices = list(range(i_first, n_cache_records)) + list(range(0, i_last + 1))
    else:
        indices = list(range(i_first, i_last + 1))

    n_records = len(indices)

    # Bulk-read each variable
    var_data = {}
    for var in JRA55_VARIABLES:
        slab = ds[var].isel(time=indices).values
        var_data[var] = jnp.asarray(slab, dtype=jnp.float64)

    ds.close()

    # Record fractional days (for interpolation inside scan)
    record_days = jnp.asarray(
        [idx / RECORDS_PER_DAY for idx in indices], dtype=jnp.float64,
    )

    raw_stack = {var: var_data[var] for var in JRA55_VARIABLES}
    runoff_stack = var_data["friver"]

    record_meta = {
        "record_days": record_days,          # (n_records,) fractional days
        "block_start_day": float(start_day),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "cache_length_days": float(cache_length_days),
        "cycle": cycle,
    }

    return raw_stack, runoff_stack, record_meta


def _build_jra55_block_fn(model, jra55_state, dt):
    """Return a JIT-compiled block function that runs N steps via lax.scan.

    Captures everything that's static across the block (sponge, SSS
    target, freeze-cap mask, coupler config, dt) in the closure so
    the scan body has a clean ``(state, idx) → (state', None)`` signature.
    Re-using the returned function across blocks reuses the JIT cache.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    coupler_cfg = jra55_state["coupler_cfg"]
    co2_ppmv = float(jra55_state["co2_ppmv"])
    T_ramp_seconds = float(jra55_state.get("T_ramp_seconds", 86400.0))
    enable_ramp = T_ramp_seconds > 0
    enable_sponge = bool(jra55_state.get("enable_sponge", False))
    enable_sss = bool(jra55_state.get("enable_sss_restoring", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
    else:
        sss_alpha_static = 0.0
        sss_target_static = None

    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
        else:
            # No sponge: cap globally over all ocean cells.
            # Use the land_mask from the initial state (captured below).
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    # 3D velocity clip — caps ALL velocity components (barotropic +
    # baroclinic) after each step.  The barotropic-only MAXVEL inside
    # the split-explicit solver doesn't prevent baroclinic blowup.
    _maxvel_3d = model.config.maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    @jax.jit
    def block_fn(state, atm_stack, runoff_stack, block_start_step):
        def step_body(state_in, idx):
            atm = AtmToSurface(
                sw_down=atm_stack["sw_down"][idx],
                lw_down=atm_stack["lw_down"][idx],
                precip_total=atm_stack["precip_total"][idx],
                precip_snow=atm_stack["precip_snow"][idx],
                T_lowest=atm_stack["T_lowest"][idx],
                q_lowest=atm_stack["q_lowest"][idx],
                u_lowest=atm_stack["u_lowest"][idx],
                v_lowest=atm_stack["v_lowest"][idx],
                p_lowest=atm_stack["p_lowest"][idx],
                p_surface=atm_stack["p_surface"][idx],
                rho_lowest=atm_stack["rho_lowest"][idx],
                cos_zenith=atm_stack["cos_zenith"][idx],
                co2_ppmv=jnp.asarray(co2_ppmv, dtype=atm_stack["T_lowest"].dtype),
                has_radiation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
                has_precipitation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
            )

            sst_K = state_in.T.data[..., 0] + _const.T_freeze
            u_o = jnp.zeros_like(sst_K)
            v_o = jnp.zeros_like(sst_K)
            tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)

            # Wind-stress spinup ramp (gated at compile time).
            if enable_ramp:
                abs_step = block_start_step + idx
                t_sim = abs_step.astype(jnp.float64) * dt
                ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
                tau_x = tile.tau_x * ramp
                tau_y = tile.tau_y * ramp
            else:
                tau_x = tile.tau_x
                tau_y = tile.tau_y

            sw_net = atm.sw_down * (1.0 - tile.albedo)
            q_net = (sw_net + atm.lw_down
                     - tile.lw_up - tile.shflx - tile.lhflx)

            evap = tile.lhflx / _const.L_v
            fw = FreshwaterForcing(
                precip=atm.precip_total,
                evap=evap,
                runoff=runoff_stack[idx],
                ice_fw=jnp.zeros_like(runoff_stack[idx]),
            )
            sf = OceanSurfaceForcing(
                sw_down=atm.sw_down,
                q_net=q_net,
                tau_x=tau_x,
                tau_y=tau_y,
                freshwater=None,
            )
            # Ramp sponge strength alongside wind stress.
            if enable_ramp and enable_sponge:
                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
            else:
                sponge_step = sponge

            new_state = model._step_impl(
                state_in, dt,
                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
            )

            # SSS restoring (gated at compile time via Python `if`).
            if enable_sss:
                S = new_state.S.data
                target = jnp.asarray(sss_target_static, dtype=S.dtype)
                alpha = jnp.asarray(sss_alpha_static, dtype=S.dtype)
                mask = jnp.asarray(new_state.land_mask.data, dtype=S.dtype)
                S_top_new = (
                    S[..., 0] - alpha * (S[..., 0] - target) * mask
                )
                new_state = new_state._replace(
                    S=new_state.S.replace(data=S.at[..., 0].set(S_top_new)),
                )

            # T_freeze cap inside sponge.
            if enable_freeze:
                T = new_state.T.data
                T_freeze_C = jnp.asarray(T_freeze_C_static, dtype=T.dtype)
                T_top = T[..., 0]
                T_top_capped = jnp.where(
                    freeze_mask_static,
                    jnp.maximum(T_top, T_freeze_C),
                    T_top,
                )
                new_state = new_state._replace(
                    T=new_state.T.replace(data=T.at[..., 0].set(T_top_capped)),
                )

            # 3D velocity clip (MOM6 MAXVEL analog for full field).
            if enable_maxvel:
                u_clipped = jnp.clip(new_state.u.data, -_maxvel_3d, _maxvel_3d)
                v_clipped = jnp.clip(new_state.v.data, -_maxvel_3d, _maxvel_3d)
                new_state = new_state._replace(
                    u=new_state.u.replace(data=u_clipped),
                    v=new_state.v.replace(data=v_clipped),
                )

            return new_state, None

        n = atm_stack["sw_down"].shape[0]
        final_state, _ = jax.lax.scan(
            step_body, state, jnp.arange(n, dtype=jnp.int32),
        )
        return final_state

    return block_fn


def _build_jra55_block_fn_interp(model, jra55_state, dt):
    """JIT-compiled block function with GPU-side forcing interpolation.

    Like ``_build_jra55_block_fn``, but instead of receiving pre-
    interpolated per-step forcing, receives the native 3-hourly records
    and computes the linear interpolation + solar zenith inside the
    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
    per block.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle, solar_declination,
    )
    from legoesm.forcing.jra55_do import RECORDS_PER_DAY

    coupler_cfg = jra55_state["coupler_cfg"]
    co2_ppmv = float(jra55_state["co2_ppmv"])
    T_ramp_seconds = float(jra55_state.get("T_ramp_seconds", 86400.0))
    enable_ramp = T_ramp_seconds > 0
    enable_sponge = bool(jra55_state.get("enable_sponge", False))
    enable_sss = bool(jra55_state.get("enable_sss_restoring", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
    else:
        sss_alpha_static = 0.0
        sss_target_static = None

    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
        else:
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    _maxvel_3d = model.config.maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    lat_2d = jra55_state["lat_2d"]
    lon_2d = jra55_state["lon_2d"]
    _rpd = float(RECORDS_PER_DAY)

    def _make_block_fn(n_steps_block):
        """Create a JIT-compiled block function for a fixed block size."""
        @jax.jit
        def block_fn(state, raw_stack, runoff_records, record_days,
                     block_start_day):
            dt_days = dt / 86400.0

            def step_body(state_in, idx):
                # Current fractional day
                day = block_start_day + idx * dt_days

                # Find bracketing records: record_days is sorted,
                # find floor position relative to the first record.
                local_pos = day * _rpd - record_days[0] * _rpd
                i_lo = jnp.clip(
                    jnp.floor(local_pos).astype(jnp.int32),
                    0, record_days.shape[0] - 2,
                )
                i_hi = i_lo + 1
                day_lo = record_days[i_lo]
                day_hi = record_days[i_hi]
                alpha = jnp.clip(
                    jnp.where(day_hi > day_lo,
                              (day - day_lo) / (day_hi - day_lo), 0.0),
                    0.0, 1.0,
                )

                def _interp(arr):
                    return (1.0 - alpha) * arr[i_lo] + alpha * arr[i_hi]

                rsds = _interp(raw_stack["rsds"])
                rlds = _interp(raw_stack["rlds"])
                tas = _interp(raw_stack["tas"])
                huss = _interp(raw_stack["huss"])
                uas = _interp(raw_stack["uas"])
                vas = _interp(raw_stack["vas"])
                psl = _interp(raw_stack["psl"])
                prra = _interp(raw_stack["prra"])
                prsn = _interp(raw_stack["prsn"])
                friver = _interp(runoff_records)

                # Derived: virtual-T density + solar zenith
                T_v = tas * (1.0 + 0.61 * huss)
                rho_a = psl / (_const.R_d * T_v)
                doy = jnp.mod(day, 365.0) + 1.0
                hour = jnp.mod(day, 1.0) * 24.0
                cos_z = cos_zenith_angle(lat_2d, lon_2d, doy, hour)

                atm = AtmToSurface(
                    sw_down=rsds, lw_down=rlds,
                    precip_total=prra + prsn, precip_snow=prsn,
                    T_lowest=tas, q_lowest=huss,
                    u_lowest=uas, v_lowest=vas,
                    p_lowest=psl, p_surface=psl,
                    rho_lowest=rho_a, cos_zenith=cos_z,
                    co2_ppmv=jnp.asarray(co2_ppmv, dtype=tas.dtype),
                    has_radiation=jnp.asarray(1.0, dtype=tas.dtype),
                    has_precipitation=jnp.asarray(1.0, dtype=tas.dtype),
                )

                sst_K = state_in.T.data[..., 0] + _const.T_freeze
                u_o = jnp.zeros_like(sst_K)
                v_o = jnp.zeros_like(sst_K)
                tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)

                if enable_ramp:
                    t_sim = day * 86400.0
                    ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
                else:
                    ramp = 1.0

                sw_net = atm.sw_down * (1.0 - tile.albedo)
                q_net = (sw_net + atm.lw_down - tile.lw_up
                         - tile.shflx - tile.lhflx)

                _dtype = state_in.T.data.dtype
                E_rate = tile.lhflx / jnp.asarray(_const.L_v, dtype=_dtype)
                fw = FreshwaterForcing(
                    precip=jnp.asarray(prra + prsn, dtype=_dtype),
                    evap=jnp.asarray(E_rate, dtype=_dtype),
                    runoff=jnp.asarray(friver, dtype=_dtype),
                    ice_fw=jnp.zeros_like(sst_K, dtype=_dtype),
                )
                sf = OceanSurfaceForcing(
                    sw_down=atm.sw_down, q_net=q_net,
                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
                    freshwater=None,
                )

                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
                            if enable_sponge else None)
                new_state = model._step_impl(
                    state_in, dt, freshwater=fw,
                    surface_forcing=sf, sponge=sponge_k,
                )

                if enable_sss:
                    S = new_state.S.data
                    S_new = S.at[..., 0].set(
                        S[..., 0] - sss_alpha_static * (
                            S[..., 0] - sss_target_static))
                    new_state = new_state._replace(
                        S=new_state.S.replace(data=S_new))
                if enable_freeze:
                    T = new_state.T.data
                    T_top = jnp.where(
                        freeze_mask_static,
                        jnp.maximum(T[..., 0], T_freeze_C_static),
                        T[..., 0],
                    )
                    new_state = new_state._replace(
                        T=new_state.T.replace(
                            data=T.at[..., 0].set(T_top)))
                if enable_maxvel:
                    new_state = new_state._replace(
                        u=new_state.u.replace(
                            data=jnp.clip(new_state.u.data,
                                          -_maxvel_3d, _maxvel_3d)),
                        v=new_state.v.replace(
                            data=jnp.clip(new_state.v.data,
                                          -_maxvel_3d, _maxvel_3d)))

                return new_state, None

            final_state, _ = jax.lax.scan(
                step_body, state,
                jnp.arange(n_steps_block, dtype=jnp.int32),
            )
            return final_state

        return block_fn

    # Cache block functions by size to avoid recompilation.
    _block_fn_cache = {}

    def _get_block_fn(n):
        if n not in _block_fn_cache:
            _block_fn_cache[n] = _make_block_fn(n)
        return _block_fn_cache[n]

    return _get_block_fn


# ===========================================================================
# Diagnostics
# ===========================================================================

def _extract_bgc_scalars(state, mask):
    """Extract surface BGC diagnostics from biogeo state (if present)."""
    result = {}
    b = getattr(state, "biogeo", None)
    if b is None:
        return result
    import numpy as np
    wet = np.asarray(mask) > 0.5
    def _surf_mean(arr):
        a = np.asarray(arr)[..., 0]
        return float(np.nanmean(np.where(wet, a, np.nan))) if wet.any() else 0.0
    result["DIC_surf"]   = _surf_mean(b.DIC)
    result["ALK_surf"]   = _surf_mean(b.ALK)
    if b.NO3   is not None: result["NO3_surf"]   = _surf_mean(b.NO3)
    if b.Phyto is not None: result["Phyto_surf"] = _surf_mean(b.Phyto)
    if b.Zoo   is not None: result["Zoo_surf"]   = _surf_mean(b.Zoo)
    if b.Det   is not None: result["Det_surf"]   = _surf_mean(b.Det)
    return result


def _extract_scalars(state, grid_type, grid, z_coord):
    """Compute scalar diagnostics from ocean state."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d, sh_synthesis
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real
        eta_grid = sh_synthesis(grid, state.eta_hat.data).real
        mask = np.asarray(state.land_mask_grid.data)
        sst = float(np.nanmean(np.where(mask > 0.5, np.asarray(T_grid[..., 0]), np.nan)))
        sss = float(np.nanmean(np.where(mask > 0.5, np.asarray(S_grid[..., 0]), np.nan)))
        ssh = float(np.nanmean(np.where(mask > 0.5, np.asarray(eta_grid), np.nan)))
        return {"SST": sst, "SSS": sss, "SSH": ssh}

    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    eta = np.asarray(state.eta.data)
    mask = np.asarray(state.land_mask.data)

    if grid_type == "mpas":
        # MPAS: (nCells, nlev), mask: (nCells,)
        wet = mask > 0.5
        sst = float(np.mean(T[wet, 0])) if wet.any() else 0.0
        sss = float(np.mean(S[wet, 0])) if wet.any() else 0.0
        ssh = float(np.mean(eta[wet])) if wet.any() else 0.0
        u = np.asarray(state.u.data)
        max_u = float(np.max(np.abs(u)))
    else:
        # Cubed-sphere (6,n,n,nlev) or latlon (nlat,nlon,nlev)
        if T.ndim == 4:
            mask_3d = mask[..., np.newaxis]
        else:
            mask_3d = mask[..., np.newaxis]
        wet = mask > 0.5
        wet_3d = mask_3d > 0.5
        sst = float(np.mean(T[..., 0][wet]))
        sss = float(np.mean(S[..., 0][wet]))
        ssh = float(np.mean(eta[wet]))
        u_raw = np.asarray(state.u.data)
        v_raw = np.asarray(state.v.data) if hasattr(state, 'v') else np.zeros_like(u_raw)
        # C-grid lat-lon: u is (nlat, nlon+1, nlev), v is (nlat+1, nlon, nlev).
        # Interpolate staggered velocities to cell centers before computing speed.
        if grid_type == "latlon" and u_raw.shape[1] != T.shape[1]:
            u_c = 0.5 * (u_raw[:, :-1] + u_raw[:, 1:])
            v_c = 0.5 * (v_raw[:-1, :] + v_raw[1:, :])
        else:
            u_c = u_raw
            v_c = v_raw
        speed_3d = np.sqrt(u_c**2 + v_c**2)
        max_u = float(np.max(speed_3d))

    # ---- B2 standing-mode purity diagnostic P_bt ----
    # P_bt = ⟨|U_bar|²⟩ / ⟨|u_3d|²⟩ — fraction of KE in the depth-mean
    # (barotropic) component.  At a healthy spinup P_bt ≈ 0.05–0.15
    # depending on the regime; a barotropic standing mode locked onto
    # a single column drives P_bt → 1 there.  Uses simple unweighted
    # depth-mean (partial-cell thickness ignored — proxy good enough
    # for monitoring; it overweights deep columns slightly which is
    # exactly where the failure lives).  Also reports max\|u\| location.
    pbt = 0.0
    j_max = i_max = -1
    if grid_type != "spectral" and grid_type != "mpas":
        nlev_state = u_c.shape[-1]
        U_bar = np.mean(u_c, axis=-1)
        V_bar = np.mean(v_c, axis=-1)
        ke_baro = 0.5 * (U_bar**2 + V_bar**2)
        ke_3d = 0.5 * speed_3d**2
        wet_3d_b = np.broadcast_to(wet[..., np.newaxis], ke_3d.shape)
        ke_baro_total = float(np.sum(np.where(wet, ke_baro, 0.0))) * nlev_state
        ke_3d_total = float(np.sum(np.where(wet_3d_b, ke_3d, 0.0)))
        pbt = ke_baro_total / max(ke_3d_total, 1e-30)
        speed_masked = np.where(wet[..., np.newaxis], speed_3d, -1.0)
        idx = np.unravel_index(np.argmax(speed_masked), speed_3d.shape)
        j_max, i_max = int(idx[0]), int(idx[1])

    scalars = {
        "SST": sst, "SSS": sss, "SSH": ssh,
        "max_speed": max_u if grid_type != "spectral" else 0.0,
        "P_bt": float(pbt),
        "j_maxu": j_max,
        "i_maxu": i_max,
    }
    # Add BGC surface diagnostics if biogeo state is present
    if grid_type not in ("spectral",):
        mask_2d = np.asarray(state.land_mask.data) if grid_type != "mpas" else np.asarray(state.land_mask.data)
        scalars.update(_extract_bgc_scalars(state, mask_2d))
    return scalars


def _check_finite(state, grid_type):
    """Check if state contains finite and physically sensible values."""
    if grid_type == "spectral":
        ok_finite = bool(jnp.all(jnp.isfinite(state.T_hat.data)))
        if not ok_finite:
            return False
        T0_mag = float(jnp.abs(state.T_hat.data[0, 0]))
        return T0_mag < 1000.0

    T = state.T.data
    eta = state.eta.data
    mask = state.land_mask.data

    # Mask to ocean cells only (land cells may have uncontrolled values)
    if grid_type == "mpas":
        mask_3d = mask[:, jnp.newaxis]
        mask_2d = mask
    else:
        mask_3d = mask[..., jnp.newaxis]
        mask_2d = mask

    T_ocean = jnp.where(mask_3d > 0.5, T, 0.0)
    eta_ocean = jnp.where(mask_2d > 0.5, eta, 0.0)

    ok_finite = bool(
        jnp.all(jnp.isfinite(T_ocean))
        & jnp.all(jnp.isfinite(eta_ocean))
    )
    if not ok_finite:
        return False

    # Bound checks: ocean SSH variations are < 10 m even with
    # tsunamis (Mariana Trench depth ~11 km but η is the surface
    # elevation, not depth).  Use 1000 m as the sanity threshold —
    # well above any realistic dynamic range, but catches the
    # iter-71 cube C24 OMIP BLOWUP (eta_max=2677 m at step 500).
    # iter-79 added the η bound; previously only T was bounded
    # (< 100 °C), so an η-only blowup could in principle escape
    # detection (T might still be reasonable while η diverged).
    # The iter-71 BLOWUP was caught via the T bound at step 500
    # but the η bound is defensive.
    #
    # iter-81 codex LOW: state must remain below the threshold
    # (strict ``<``).  Exactly 1000 m would trigger a BLOWUP —
    # acceptable since 1000 m is already absurd for SSH.
    if not bool(jnp.max(jnp.abs(T_ocean)) < 100.0):
        return False
    return bool(jnp.max(jnp.abs(eta_ocean)) < 1000.0)


# ===========================================================================
# Restart I/O — minimum-viable npz format compatible with the
# global-overturning progress plotter.  We dump every field of the
# state that has a ``.data`` attribute, plus the simulation day and
# step index.  The plotter
# (``scripts/global_overturning/plot_realistic_geometry_progress.py``
# and the JRA55 sibling) reads these to compute MOC, BSF, snapshots.
# ===========================================================================


def _save_restart(state, day, step, output_dir):
    """Save a state restart in the global-overturning npz format.

    Mirrors ``scripts/global_overturning/run_global_overturning_*``
    so the same plotting helpers consume both runs without
    discrimination.

    Parameters
    ----------
    state : ocean state
    day : float
        Simulation day (filename uses ``int(round(day))``).
    step : int
        Step index (stored in npz for provenance only).
    output_dir : Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "time_days": float(day),
        "grid_type": "latlon",
    }
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        payload[f] = np.asarray(obj.data)
    # Save BGC tracer state if present
    biogeo = getattr(state, "biogeo", None)
    if biogeo is not None:
        for bf in biogeo._fields:
            arr = getattr(biogeo, bf)
            if arr is not None:
                payload[f"biogeo_{bf}"] = np.asarray(arr)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **payload)
    return fname


def _load_restart(restart_path, template_state):
    """Load a restart npz and populate the state from a template.

    The template state (from ``_init_rest_state``) provides the pytree
    structure, Field metadata (name, dims, units), and masks.  Only the
    prognostic data arrays (u, v, T, S, eta, and optional SOM/AB2 carry
    fields) are overwritten from the restart file.

    Parameters
    ----------
    restart_path : str or Path
        Path to a ``restart_dayXXXXXX.npz`` file.
    template_state : ocean state
        A freshly initialized state with correct grid, masks, and
        z-coordinate.

    Returns
    -------
    state : same type as template_state
        State with prognostic fields loaded from the restart.
    restart_day : float
        Simulation day at which the restart was saved.
    restart_step : int
        Step index at which the restart was saved.
    """
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    restart_step = int(data["step"])

    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = obj.replace(data=arr)

    state = template_state._replace(**replacements)
    return state, restart_day, restart_step


# ===========================================================================
# Time loop
# ===========================================================================

def _run_omip_loop(model, state, grid_type, grid, z_coord, dt, n_steps,
                   diag_every, label="",
                   restoring_targets=None, restoring_tau_s=None,
                   jra55_state=None,
                   checkpoint_days=None, checkpoint_dir=None,
                   start_step=0,
                   nudge_woa_tau=0.0, T_woa_3d=None):
    """Run time loop with diagnostics.

    Two forcing paths, mutually exclusive:

    * **restoring** (default): plain ``model.step(state, dt)`` followed
      by Haney SST/SSS restoring when ``restoring_targets`` is set.
    * **jra55_do_tropical**: ``_jra55_step(...)`` per step using a
      pre-built JRA55-do cache (set ``jra55_state``); the model
      receives bulk-flux fields and structured freshwater forcing.

    Parameters
    ----------
    restoring_targets : tuple(T_target, S_target) or None
        Surface T/S targets for SST/SSS restoring (restoring path).
    restoring_tau_s : float or None
        Restoring timescale [seconds] (restoring path).
    jra55_state : dict or None
        Output of :func:`_setup_jra55_forcing_state`.  When provided,
        replaces the restoring path with JRA55-do bulk-flux forcing.
    checkpoint_days : float or None
        If set, save a ``restart_dayXXXXXX.npz`` snapshot every
        ``checkpoint_days`` simulated days (and at the final step).
        Format mirrors the global-overturning runs so the existing
        progress plotters consume it directly.
    checkpoint_dir : Path or None
        Output directory for restarts. Required when
        ``checkpoint_days`` is set.

    Returns (final_state, diagnostics, wall_time, ok).
    """
    if jra55_state is not None and restoring_targets is not None:
        raise ValueError(
            "_run_omip_loop: jra55_state and restoring_targets are mutually "
            "exclusive — choose one forcing path."
        )
    if checkpoint_days is not None and checkpoint_dir is None:
        raise ValueError(
            "_run_omip_loop: checkpoint_days requires checkpoint_dir."
        )
    diag: dict[str, list] = {"day": [], "step": []}
    snapshots: dict[int, dict] = {}
    snap_steps = {0, n_steps}
    for i in range(1, min(10, n_steps)):
        snap_steps.add(max(1, int(i * n_steps / 10)))

    # iter-97: capture BLOWUP details so ``results.txt`` can
    # surface them rather than just reporting the last *clean*
    # diagnostic (which masks BLOWUPs as "PASS-shaped FAIL").
    # The iter-96 cube OMIP smoke saw SST=19.76 in
    # results.txt, and only by re-running with verbose output
    # was it visible that max|T|=8.3M K and η=2678 m had
    # actually blown up.  The BLOWUP info now lives in
    # ``blowup_info`` and is emitted in results.txt.
    blowup_info: dict | None = None

    # Initial diagnostics
    # Attach BGC state to ocean state if BGC is enabled
    _bgc_cfg_active = getattr(model, "_bgc_cfg", None)
    if _bgc_cfg_active is not None and hasattr(_bgc_cfg_active, "scheme") and _bgc_cfg_active.scheme != "none":
        from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state
        shape_3d = state.T.data.shape
        biogeo = init_biogeo_state(shape_3d, z_coord.z_full_ref, _bgc_cfg_active)

        # Override nutrient ICs from WOA18 if paths provided
        _woa_no3 = _WOA_BGC_PATHS.get("no3")
        _woa_po4 = _WOA_BGC_PATHS.get("po4")
        _woa_si  = _WOA_BGC_PATHS.get("si")
        if _bgc_cfg_active.scheme == "npzd_v2" and (
            _woa_no3 or _woa_po4 or _woa_si
        ):
            from legoesm.ocean.init_woa import init_bgc_from_woa
            print("  Loading WOA18 BGC nutrients...")
            woa_bgc = init_bgc_from_woa(
                grid, z_coord,
                no3_path=_woa_no3,
                po4_path=_woa_po4,
                si_path =_woa_si,
            )
            biogeo = biogeo._replace(
                NO3=woa_bgc["NO3"].astype(biogeo.DIC.dtype),
                PO4=woa_bgc["PO4"].astype(biogeo.DIC.dtype),
                Si =woa_bgc["Si" ].astype(biogeo.DIC.dtype),
                Fe =woa_bgc["Fe" ].astype(biogeo.DIC.dtype),
            )
            # Store WOA nutrient targets for nudging
            _WOA_BGC_PATHS["NO3_target"] = woa_bgc["NO3"]
            _WOA_BGC_PATHS["PO4_target"] = woa_bgc["PO4"]
            _WOA_BGC_PATHS["Si_target"]  = woa_bgc["Si"]
            # Update bgc_cfg with nutrient targets for nudging
            _bgc_cfg_active = _bgc_cfg_active._replace(
                nudge_nutrients=True,
                tau_nudge_days=30.0,
                NO3_target=woa_bgc["NO3"],
                PO4_target=woa_bgc["PO4"],
                Si_target =woa_bgc["Si"],
            )
            model._bgc_cfg = _bgc_cfg_active
            print(f"    NO3 surface mean: {float(woa_bgc['NO3'][...,0].mean()):.4f} mol/m3")
            print(f"    PO4 surface mean: {float(woa_bgc['PO4'][...,0].mean()):.4f} mol/m3")
            print(f"    Si  surface mean: {float(woa_bgc['Si' ][...,0].mean()):.4f} mol/m3")
            print(f"    Fe  surface mean: {float(woa_bgc['Fe' ][...,0].mean()):.2e} mol/m3")

        state = state._replace(biogeo=biogeo)
        print(f"  BGC state initialised: scheme={_bgc_cfg_active.scheme} "
              f"tracers={len([f for f in biogeo._fields if getattr(biogeo,f) is not None])}")

    scalars = _extract_scalars(state, grid_type, grid, z_coord)
    for k, v in scalars.items():
        diag.setdefault(k, []).append(v)
    diag["day"].append(0.0)
    diag["step"].append(0)

    t0 = time.time()
    last_print = t0
    blown_up = False

    # ---- B2 standing-mode time diagnostic χ ----
    # χ(t) = ||η^n - ½(η^{n−1} + η^{n+1})||² / ||η^n||²  (Williams 2009).
    # Tracks the 2-Δt-block computational-mode amplitude in η: a clean
    # integration sits at χ~1e-6, a growing computational mode shows χ
    # rising exponentially 5–10 days BEFORE max|u| spikes.  Buffer
    # holds the last 3 end-of-block η snapshots; we compute χ on the
    # middle once we have 3.
    eta_history: list[np.ndarray] = []

    # ----- JRA55-do block-scan path (multi-core friendly) ----------------
    # Wraps N ocean steps in lax.scan inside @jax.jit for ~30x GPU
    # speedup.  The scan body calls model._step_impl() (no inner JIT)
    # to avoid nested JIT boundaries that caused divergence with
    # partial-cell coordinates.
    use_scan_blocks = (jra55_state is not None
                       and not jra55_state.get("_use_single_step", False))
    use_gpu_interp = (jra55_state is not None
                      and jra55_state.get("_gpu_interp", False))
    if use_scan_blocks:
        if use_gpu_interp:
            _get_block_fn_interp = _build_jra55_block_fn_interp(
                model, jra55_state, dt)
            print("  GPU-interp mode: forcing interpolation on GPU")
        else:
            block_fn = _build_jra55_block_fn(model, jra55_state, dt)
        block_size = max(1, diag_every)
        if checkpoint_days is not None:
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
        else:
            steps_per_ckpt = None

        block_start = start_step
        while block_start < n_steps and not blown_up:
            actual = min(block_size, n_steps - block_start)
            t_io_start = time.time()

            if use_gpu_interp:
                raw_stack, runoff_records, record_meta = (
                    _preload_jra55_raw_records(
                        block_start, actual, dt, jra55_state))
            else:
                atm_stack, runoff_stack = _preload_jra55_forcing_block(
                    block_start, actual, dt, jra55_state,
                )
            io_dt = time.time() - t_io_start

            t_compute_start = time.time()
            if use_gpu_interp:
                bfn = _get_block_fn_interp(actual)
                state = bfn(
                    state, raw_stack, runoff_records,
                    record_meta["record_days"],
                    jnp.float64(record_meta["block_start_day"]),
                )
            else:
                state = block_fn(
                    state, atm_stack, runoff_stack,
                    jnp.int32(block_start),
                )
            jax.block_until_ready(state.T.data)
            compute_dt = time.time() - t_compute_start

            block_start += actual
            step = block_start
            day = step * dt / 86400.0

            if not _check_finite(state, grid_type):
                print(f"  BLOWUP at step {step}")
                blown_up = True
                break

            # WOA T nudging: dT/dt += (T_woa - T) / tau
            if nudge_woa_tau > 0 and T_woa_3d is not None:
                nudge_per_step = dt / (nudge_woa_tau * 86400.0)
                daily_frac = 1.0 - (1.0 - nudge_per_step) ** actual
                mask_3d = state.land_mask.data[..., jnp.newaxis]
                T_nudged = state.T.data + daily_frac * (
                    T_woa_3d - state.T.data) * mask_3d
                state = state._replace(
                    T=state.T.replace(data=T_nudged.astype(state.T.data.dtype)))

            scalars = _extract_scalars(state, grid_type, grid, z_coord)

            # B2: chi diagnostic from last 3 eta snapshots
            chi = 0.0
            if grid_type == "latlon":
                eta_now = np.asarray(state.eta.data)
                eta_history.append(eta_now)
                if len(eta_history) > 3:
                    eta_history.pop(0)
                if len(eta_history) == 3:
                    eta_m2, eta_m1, eta_0 = eta_history
                    mask_eta = np.asarray(state.land_mask.data) > 0.5
                    diff = (eta_m1 - 0.5 * (eta_0 + eta_m2)) * mask_eta
                    den = eta_m1 * mask_eta
                    num_sq = float(np.sum(diff * diff))
                    den_sq = float(np.sum(den * den))
                    chi = num_sq / max(den_sq, 1e-30)
            scalars["chi"] = chi

            # Convert max|u| location indices → lat/lon for readability
            if grid_type == "latlon" and "j_maxu" in scalars and scalars["j_maxu"] >= 0:
                lat_v = np.asarray(grid.lat) if hasattr(grid, "lat") else None
                lon_v = np.asarray(grid.lon) if hasattr(grid, "lon") else None
                if lat_v is not None and lon_v is not None:
                    j = scalars["j_maxu"]; i = scalars["i_maxu"]
                    if 0 <= j < len(lat_v) and 0 <= i < len(lon_v):
                        scalars["lat_maxu"] = float(lat_v[j])
                        scalars["lon_maxu"] = float(lon_v[i])

            diag["day"].append(day)
            diag["step"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            elapsed_total = time.time() - t0
            total_days = n_steps * dt / 86400.0
            # Custom summary string includes the new diagnostics
            scalar_summary = (
                f"SST={scalars['SST']:.3g} "
                f"max|u|={scalars['max_speed']:.3g} "
                f"P_bt={scalars['P_bt']:.3g} "
                f"χ={chi:.2e}"
            )
            if "lat_maxu" in scalars:
                scalar_summary += (
                    f" @({scalars['lat_maxu']:.0f},"
                    f"{scalars['lon_maxu']:.0f})"
                )
            summary = scalar_summary
            print(
                f"    [{label}] Day {day:7.2f}/{total_days:.0f} | {summary} "
                f"| io={io_dt:.1f}s compute={compute_dt:.1f}s "
                f"({compute_dt/actual:.2f} s/step) | {elapsed_total:.0f}s total",
                flush=True,
            )

            if (steps_per_ckpt is not None and
                    (step % steps_per_ckpt == 0 or step == n_steps)):
                fname = _save_restart(state, day, step, checkpoint_dir)
                print(f"    Restart saved: {fname.name}", flush=True)

        # After the block loop, jump to the post-loop tally below.
        if grid_type == "spectral":
            jax.block_until_ready(state.T_hat.data)
        else:
            jax.block_until_ready(state.T.data)
        wall = time.time() - t0
        ok = not blown_up and _check_finite(state, grid_type)
        return state, diag, wall, ok
    # --------------------------------------------------------------------

    # ----- JRA55-do single-step path (partial-cell fallback) -----------
    if jra55_state is not None and not use_scan_blocks:
        if checkpoint_days is not None:
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
        else:
            steps_per_ckpt = None

        for i in range(start_step, n_steps):
            state = _jra55_step(state, i, dt, model, jra55_state)

            step = i + 1
            if step % diag_every == 0 or step == n_steps:
                day = step * dt / 86400.0
                if not _check_finite(state, grid_type):
                    print(f"  BLOWUP at step {step}")
                    blown_up = True
                    break
                scalars = _extract_scalars(state, grid_type, grid, z_coord)
                diag["day"].append(day)
                diag["step"].append(step)
                for k, v in scalars.items():
                    diag.setdefault(k, []).append(v)
                elapsed_total = time.time() - t0
                total_days = n_steps * dt / 86400.0
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:4]
                )
                print(
                    f"    [{label}] Day {day:7.2f}/{total_days:.0f} | {summary} "
                    f"| {elapsed_total:.0f}s total",
                    flush=True,
                )
                if (steps_per_ckpt is not None and
                        (step % steps_per_ckpt == 0 or step == n_steps)):
                    fname = _save_restart(state, day, step, checkpoint_dir)
                    print(f"    Restart saved: {fname.name}", flush=True)

        jax.block_until_ready(state.T.data)
        wall = time.time() - t0
        ok = not blown_up and _check_finite(state, grid_type)
        return state, diag, wall, ok
    # --------------------------------------------------------------------

    for i in range(start_step, n_steps):
        state = model.step(state, dt)

        # Apply SST/SSS restoring (grid-agnostic, after dynamics step)
        if restoring_targets is not None:
            T_tgt, S_tgt = restoring_targets
            state = _apply_restoring(
                state, grid_type, grid, T_tgt, S_tgt,
                dt, restoring_tau_s,
            )

        step = i + 1

        if step % 100 == 0:
            if not _check_finite(state, grid_type):
                # Debug: identify what failed
                if grid_type != "spectral":
                    mask = state.land_mask.data
                    m3 = mask[:, jnp.newaxis] if grid_type == "mpas" else mask[..., jnp.newaxis]
                    T_oc = jnp.where(m3 > 0.5, state.T.data, 0.0)
                    eta_max = float(jnp.max(jnp.abs(state.eta.data)))
                    eta_finite = bool(jnp.all(jnp.isfinite(state.eta.data)))
                    T_max = float(jnp.max(jnp.abs(T_oc)))
                    T_finite = bool(jnp.all(jnp.isfinite(T_oc)))
                    print(
                        f"  BLOWUP step {step}: "
                        f"max|T|={T_max:.1f} "
                        f"T_finite={T_finite} "
                        f"eta_max={eta_max:.2f} "
                        f"eta_finite={eta_finite}"
                    )
                    # iter-81 codex LOW: report Reason for BOTH T
                    # and η triggers (iter-79 reported only η).
                    # Multiple conditions can fire simultaneously
                    # (e.g., a NaN cascade hits both T and η).
                    reasons = []
                    if not T_finite:
                        msg = "T contains NaN/Inf"
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    elif T_max >= 100.0:
                        msg = (f"|T| reached {T_max:.1f} °C "
                               f"(sanity threshold 100 °C)")
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    if not eta_finite:
                        msg = "η contains NaN/Inf"
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    elif eta_max >= 1000.0:
                        msg = (f"|η| reached {eta_max:.0f} m "
                               f"(iter-79 sanity threshold 1000 m)")
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    # iter-97: persist BLOWUP info so the report
                    # can surface it in results.txt.
                    blowup_info = {
                        "step": step,
                        "day": step * dt / 86400.0,
                        "T_max": T_max,
                        "T_finite": T_finite,
                        "eta_max": eta_max,
                        "eta_finite": eta_finite,
                        "reasons": reasons,
                    }
                else:
                    print(f"  BLOWUP at step {step}")
                    blowup_info = {
                        "step": step,
                        "day": step * dt / 86400.0,
                        "reasons": ["spectral state non-finite"],
                    }
                blown_up = True
                break

        if step % diag_every == 0 or step == n_steps:
            day = step * dt / 86400.0
            scalars = _extract_scalars(state, grid_type, grid, z_coord)
            diag["day"].append(day)
            diag["step"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            now = time.time()
            if now - last_print > 15:
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:4])
                elapsed = now - t0
                total_days = n_steps * dt / 86400.0
                print(f"    [{label}] Day {day:7.1f}/{total_days:.0f} | {summary} "
                      f"| {elapsed:.0f}s elapsed")
                last_print = now

        # Restart-checkpoint cadence (independent of the diag cadence
        # so checkpoints land on round-number simulation days).
        if checkpoint_days is not None:
            day_now = step * dt / 86400.0
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
            if step % steps_per_ckpt == 0 or step == n_steps:
                fname = _save_restart(state, day_now, step, checkpoint_dir)
                # Friendly progress; gated on the same 15-s cadence as
                # the diag print so we don't spam.
                if time.time() - last_print < 1.0:
                    print(f"    Restart saved: {fname.name}", flush=True)

    if grid_type == "spectral":
        jax.block_until_ready(state.T_hat.data)
    else:
        jax.block_until_ready(state.T.data)

    wall = time.time() - t0
    ok = not blown_up and _check_finite(state, grid_type)
    return state, diag, wall, ok, blowup_info


# ===========================================================================
# Output
# ===========================================================================

def _save_output(output_dir: Path, diag, args, grid_type, wall_time, ok,
                 blowup_info: dict | None = None):
    """Save diagnostics and metadata.

    iter-97: ``blowup_info`` (added kwarg) carries the BLOWUP
    step / max|T| / max|η| / reasons captured at the time the
    state went non-finite, so ``results.txt`` can clearly mark
    BLOWUP runs as such instead of silently reporting the last
    *clean* SST/SSS/SSH (which led to a false-improvement claim
    in iter-96).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timeseries CSV
    import csv
    csv_path = output_dir / "timeseries.csv"
    keys = list(diag.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n_rows = len(diag[keys[0]])
        for i in range(n_rows):
            w.writerow([diag[k][i] if i < len(diag[k]) else "" for k in keys])

    # iter-25: also write a matrix-compatible ``mean_timeseries.csv``
    # so ``run_ocean_test_matrix.py --replot`` (and the cross-grid
    # plotter generally) can pick this up.  The ocean-matrix
    # plotter expects ``time_days`` column + a set of mean_*
    # diagnostics; rename columns appropriately and write a
    # parallel CSV.  Don't replace ``timeseries.csv`` since that
    # filename + ``results.json`` is the existing OMIP output
    # contract.
    mean_csv_path = output_dir / "mean_timeseries.csv"
    column_renames = {
        "day": "time_days",
        "SST": "mean_SST",
        "SSS": "mean_SSS",
        "SSH": "mean_eta",
        "max_speed": "max_speed",
        "step": "step",
    }
    out_keys = [column_renames.get(k, k) for k in keys]
    with open(mean_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_keys)
        n_rows = len(diag[keys[0]])
        for i in range(n_rows):
            w.writerow([diag[k][i] if i < len(diag[k]) else "" for k in keys])

    # Results JSON — include full CLI args for reproducibility
    results = {
        "grid_type": grid_type,
        "resolution": args.resolution or GRID_DEFAULTS[grid_type]["resolution"],
        "nlev": args.nlev,
        "days": args.days,
        "dt": args.dt or GRID_DEFAULTS[grid_type]["dt"],
        "physics": args.physics,
        "water_type": args.water_type,
        "sw_down": args.sw_down,
        "wall_time_s": wall_time,
        "status": "PASS" if ok else "FAIL",
        "final_SST": diag["SST"][-1] if diag["SST"] else None,
        "final_SSS": diag["SSS"][-1] if diag["SSS"] else None,
        "final_SSH": diag["SSH"][-1] if diag["SSH"] else None,
        "blowup_info": blowup_info,
        "cli_args": vars(args),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # iter-25: also write a matrix-compatible ``results.txt`` next to
    # the existing ``results.json`` so the ocean cross-grid plotter
    # (which parses ``key: value`` lines from results.txt) can pick
    # this up.
    final_sst = diag["SST"][-1] if diag.get("SST") else None
    final_sss = diag["SSS"][-1] if diag.get("SSS") else None
    final_ssh = diag["SSH"][-1] if diag.get("SSH") else None
    notes_parts = []
    # iter-97: lead with BLOWUP marker when the run failed
    # because a BLOWUP was detected.  This is unambiguous —
    # readers no longer mistake "SST=19.76 (last clean diag)"
    # for a healthy run.
    if blowup_info is not None:
        notes_parts.append(
            f"BLOWUP at step {blowup_info['step']} "
            f"(day {blowup_info.get('day', 0):.2f})"
        )
        if "T_max" in blowup_info:
            notes_parts.append(f"max|T|={blowup_info['T_max']:.3e} °C")
        if "eta_max" in blowup_info:
            notes_parts.append(f"max|η|={blowup_info['eta_max']:.0f} m")
        for reason in blowup_info.get("reasons", []):
            notes_parts.append(f"reason: {reason}")
        # Also keep the last clean diagnostic so a reader can
        # see what the system looked like at the last sane
        # state — but mark it as such.
        if final_sst is not None:
            notes_parts.append(f"last clean SST={final_sst:.3f}")
    else:
        if final_sst is not None:
            notes_parts.append(f"SST={final_sst:.3f}")
        if final_sss is not None:
            notes_parts.append(f"SSS={final_sss:.3f}")
        if final_ssh is not None:
            notes_parts.append(f"SSH={final_ssh:.3e}")
    notes_str = ", ".join(notes_parts) if notes_parts else "OMIP complete"
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"test: omip\n")
        f.write(f"grid: {grid_type}\n")
        f.write(f"resolution: {results['resolution']}\n")
        f.write(f"days: {results['days']}\n")
        f.write(f"dt: {results['dt']}\n")
        f.write(f"levels: {results['nlev']}\n")
        f.write(f"physics: {results['physics']}\n")
        f.write(f"status: {results['status']}\n")
        f.write(f"notes: {notes_str}\n")
        f.write(f"wall_time: {wall_time:.1f}s\n")

    # Plot timeseries
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        days = diag["day"]

        for ax, key in zip(axes.flat, ["SST", "SSS", "SSH", "max_speed"]):
            if key in diag:
                ax.plot(days, diag[key])
                ax.set_xlabel("Day")
                ax.set_ylabel(key)
                ax.set_title(key)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"OMIP {grid_type} — {args.physics} physics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "timeseries.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass  # plotting is optional

    return results


# ===========================================================================
# Single-grid runner
# ===========================================================================

def run_omip_single(grid_type: str, args) -> dict:
    """Run OMIP simulation on a single grid type."""
    # iter-115 codex iter-114-followup HIGH-1: pre-iter-115,
    # ``--resolution 16`` was applied verbatim to every grid
    # type.  Cube/spectral parsed it (silently wrong: cube
    # ``int("16"[1:]) = 6``); latlon errored; mpas
    # interpreted as level 16 (4.29e+10 cells).  Now use the
    # iter-115 shared dispatch helper.
    if args.resolution is not None:
        from legoesm.driver.cli_resolution import (
            expand_cli_resolution, validate_cli_resolution,
        )
        N = validate_cli_resolution(
            args.resolution,
            additional_examples="'C24', 'ico3', '36x72', 'T21', '50km'",
        )
        if N is not None:
            resolution = expand_cli_resolution(N, grid_type)
        else:
            # Pre-formatted per-grid string — pass through.
            resolution = args.resolution
    else:
        resolution = GRID_DEFAULTS[grid_type]["resolution"]
    dt = args.dt or GRID_DEFAULTS[grid_type]["dt"]
    days = 30.0 if args.quick else args.days
    n_steps = int(days * 86400.0 / dt)
    diag_every = args.diag_every or max(1, int(86400.0 / dt))  # ~daily

    print(f"\n{'='*70}")
    print(f"  OMIP: {grid_type} | {resolution} | {args.nlev} levels | "
          f"dt={dt:.0f}s | {days:.0f} days ({n_steps} steps)")
    print(f"  Physics: {args.physics} | SW: {args.sw_down} W/m² | "
          f"Water type: {args.water_type}")
    print(f"{'='*70}")

    t_setup = time.time()

    # Create grid + model (all grids use identical config-based diffusion
    # for cross-grid consistency; physics pipeline disabled).
    grid, z_coord, config, model, coord_kind = _create_setup(
        grid_type, resolution, args.nlev, args.H_max,
        args.physics, args.water_type,
        use_bathymetry=(args.bathymetry is not None),
        A_h_override=args.A_h,
        B_h_override=args.B_h,
        K_h_override=args.K_h,
        A_h_eq_boost=args.A_h_eq_boost,
        A_h_eq_sigma_deg=args.A_h_eq_sigma,
        C_smag=args.C_smag,
        C_smag_lap=args.C_smag_lap,
        A_h_floor=args.A_h_floor,
        C_leith=args.C_leith,
        pgf_scheme=args.pgf_scheme,
        slope_foot_alpha=args.slope_foot_alpha,
        no_lat_scaling=args.no_lat_scaling,
        no_gm_redi=getattr(args, "no_gm_redi", False),
        implicit_vertical_mixing=getattr(
            args, "implicit_vertical_mixing", False),
        biogeo=getattr(args, "biogeo", "none"),
        pco2_atm=getattr(args, "pco2_atm", 400.0),
        woa_no3=getattr(args, "woa_no3", None),
        woa_po4=getattr(args, "woa_po4", None),
        woa_si=getattr(args, "woa_si", None),
    )

    # --- Initialization strategy ---
    # Start from rest state with uniform T/S, then restore toward WOA
    # climatology.  Starting from full WOA T/S creates extreme pressure
    # gradients that trigger violent geostrophic adjustment (200+ m/s
    # currents, SSS >70 PSU).  Uniform start + restoring is the standard
    # OMIP spin-up approach: the model gradually builds up the
    # climatological circulation from rest.
    from legoesm.ocean.init_woa import init_ocean_from_woa
    T_woa, S_woa = init_ocean_from_woa(grid, z_coord, args.woa_t, args.woa_s)

    # Bathymetry: realistic (ETOPO) or flat-bottom.
    H_bathy_init = None
    land_mask_init = None
    if args.bathymetry is not None:
        from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
        bathy_cfg = BathymetryConfig(
            source="file",
            path=args.bathymetry,
            H_max=args.H_max,
            H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            enforce_straits=True,
            fill_isolated_basins=True,
            depth_is_negative=True,
            r_factor_max=args.r_factor_max,
            north_cap_lat=args.north_cap_lat,
            south_cap_lat=args.south_cap_lat,
        )
        H_bathy_init, land_mask_init = init_ocean_bathymetry(grid, bathy_cfg)
        H_bathy_init = jnp.asarray(H_bathy_init, dtype=jnp.float64)
        land_mask_init = jnp.asarray(land_mask_init, dtype=jnp.float64)
        n_ocean = int(np.sum(np.asarray(land_mask_init) > 0.5))
        n_total = int(np.prod(np.asarray(land_mask_init).shape))
        print(f"  Bathymetry: {Path(args.bathymetry).name} "
              f"({n_ocean}/{n_total} ocean cells, "
              f"H_min={args.H_min}m, {args.smoothing_passes} smoothing passes, "
              f"r_max={args.r_factor_max})")

        # Equatorial-only extra smoothing.
        # The 30-day spinup diagnosed a barotropic standing-mode
        # instability at deep ocean cells in the equatorial belt
        # (Java Trench off Sumatra; deep Atlantic; deep Pacific east of
        # S. America), with growth factors of 500-1200× in 20 days.
        # Common features: |lat|<13°, H>4000 m, large ∂H/∂x near coast.
        # f≈0 there cannot damp PGF errors driven by steep H gradients.
        # The standard fix is more aggressive bathymetry smoothing in
        # the equatorial belt: the loss of "true" bathymetry detail at
        # ±10° is acceptable for a 1° model that can't resolve the EUC
        # anyway.  Apply a Laplacian smoother with cosine taper from
        # full strength at the equator to zero at the band edge, and
        # average only over wet neighbours (don't pull from land).
        eq_band = 15.0          # smooth within ±15° of equator
        eq_passes = 30           # extra Laplacian passes at the equator
        H_np = np.asarray(H_bathy_init)
        ocean_mask = np.asarray(land_mask_init) > 0.5
        # Cell-center latitudes (axis 0)
        from legoesm.grids.latlon import create_latlon_grid as _clg
        # we already have grid; pull lat
        lat_c = np.asarray(grid.lat) if hasattr(grid, 'lat') else None
        if lat_c is not None:
            # taper alpha(lat): quadratic, 1 at eq, 0 at ±eq_band
            # grid.lat is in radians; convert eq_band to radians
            eq_band_rad = np.deg2rad(eq_band)
            x = np.clip(np.abs(lat_c) / eq_band_rad, 0.0, 1.0)
            taper = (1.0 - x**2)                # (n_lat,)
            taper2d = np.broadcast_to(taper[:, None], H_np.shape)
            n_lat_g, n_lon_g = H_np.shape
            for _p in range(eq_passes):
                # neighbour sum over wet cells only, with periodic lon
                up    = np.roll(H_np, -1, axis=0); up_m    = np.roll(ocean_mask, -1, axis=0)
                down  = np.roll(H_np,  1, axis=0); down_m  = np.roll(ocean_mask,  1, axis=0)
                left  = np.roll(H_np,  1, axis=1); left_m  = np.roll(ocean_mask,  1, axis=1)
                right = np.roll(H_np, -1, axis=1); right_m = np.roll(ocean_mask, -1, axis=1)
                # No wrap in lat: zero the off-grid neighbour mask
                up_m[-1, :] = False; down_m[0, :] = False
                neigh_sum = (up * up_m + down * down_m
                             + left * left_m + right * right_m)
                neigh_cnt = up_m.astype(np.float64) + down_m + left_m + right_m
                avg = np.where(neigh_cnt > 0,
                               neigh_sum / np.maximum(neigh_cnt, 1.0),
                               H_np)
                # alpha = 0.5 * taper(lat) → mixes self with neighbour avg
                alpha = 0.5 * taper2d
                new_H = (1.0 - alpha) * H_np + alpha * avg
                # Keep land cells fixed
                H_np = np.where(ocean_mask, new_H, H_np)
            # Floor at H_min so the smoother doesn't accidentally
            # create cells shallower than the physical floor.
            H_np = np.where(ocean_mask, np.maximum(H_np, args.H_min), H_np)
            H_bathy_init = jnp.asarray(H_np, dtype=jnp.float64)
            print(f"  Equatorial smoothing: {eq_passes} extra Laplacian "
                  f"passes within ±{eq_band:.0f}° (cosine taper, wet-only)")

        # --- Close Arctic completely (solid wall) ---
        if args.close_arctic_lat is not None:
            ocean_mask = np.array(land_mask_init, copy=True) > 0.5
            lat_c = np.asarray(grid.lat) if hasattr(grid, 'lat') else None
            if lat_c is not None:
                # grid.lat is in radians; convert threshold to radians
                arctic_rows = lat_c > np.deg2rad(args.close_arctic_lat)
                n_closed = int(np.sum(ocean_mask[arctic_rows, :]))
                ocean_mask[arctic_rows, :] = False
                # Keep H_bathy unchanged — land cells retain depth values
                # but are masked out (setting H=0 confuses partial-cell coord).
                land_mask_init = jnp.asarray(ocean_mask.astype(np.float64))
                print(f"  Arctic closure: {n_closed} cells → land above "
                      f"{args.close_arctic_lat:.1f}°N")

        # --- Widen narrow passages ---
        if args.min_passage_width >= 2:
            H_np = np.array(H_bathy_init, copy=True)
            ocean_mask = np.array(land_mask_init, copy=True) > 0.5
            n_lat_g, n_lon_g = ocean_mask.shape
            fill_cells = np.zeros_like(ocean_mask)
            min_w = args.min_passage_width

            # Find cells that are part of passages narrower than min_w
            # in the zonal direction (land on both sides within min_w-1)
            for j in range(n_lat_g):
                for i in range(n_lon_g):
                    if not ocean_mask[j, i]:
                        continue
                    # Check zonal width: how many consecutive ocean cells
                    # in the east-west direction including this cell?
                    width = 1
                    # count east
                    for di in range(1, min_w):
                        ii = (i + di) % n_lon_g
                        if ocean_mask[j, ii]:
                            width += 1
                        else:
                            break
                    # count west
                    for di in range(1, min_w):
                        ii = (i - di) % n_lon_g
                        if ocean_mask[j, ii]:
                            width += 1
                        else:
                            break
                    if width < min_w:
                        fill_cells[j, i] = True

            # Same for meridional direction
            for j in range(n_lat_g):
                for i in range(n_lon_g):
                    if not ocean_mask[j, i]:
                        continue
                    width = 1
                    for dj in range(1, min_w):
                        jj = j + dj
                        if jj < n_lat_g and ocean_mask[jj, i]:
                            width += 1
                        else:
                            break
                    for dj in range(1, min_w):
                        jj = j - dj
                        if jj >= 0 and ocean_mask[jj, i]:
                            width += 1
                        else:
                            break
                    if width < min_w:
                        # Only fill if ALSO narrow zonally (avoid filling
                        # long coastlines). A true narrow passage is narrow
                        # in at least one direction.
                        fill_cells[j, i] = True

            # Actually we want cells that are narrow in BOTH directions
            # to be filled... No — a 1-cell-wide strait running N-S is
            # narrow zonally but wide meridionally. We want to fill cells
            # narrow in ANY direction. But let's be more careful:
            # Fill cells that are zonally narrow (land within min_w on both sides)
            fill_zonal = np.zeros_like(ocean_mask)
            fill_merid = np.zeros_like(ocean_mask)
            for j in range(n_lat_g):
                for i in range(n_lon_g):
                    if not ocean_mask[j, i]:
                        continue
                    # Zonal: find distance to land on each side
                    dist_e = 0
                    for di in range(1, min_w + 1):
                        ii = (i + di) % n_lon_g
                        if ocean_mask[j, ii]:
                            dist_e += 1
                        else:
                            break
                    dist_w = 0
                    for di in range(1, min_w + 1):
                        ii = (i - di) % n_lon_g
                        if ocean_mask[j, ii]:
                            dist_w += 1
                        else:
                            break
                    # Total passage width = dist_w + 1 + dist_e
                    if (dist_w + 1 + dist_e) < min_w:
                        fill_zonal[j, i] = True

                    # Meridional
                    dist_n = 0
                    for dj in range(1, min_w + 1):
                        jj = j + dj
                        if jj < n_lat_g and ocean_mask[jj, i]:
                            dist_n += 1
                        else:
                            break
                    dist_s = 0
                    for dj in range(1, min_w + 1):
                        jj = j - dj
                        if jj >= 0 and ocean_mask[jj, i]:
                            dist_s += 1
                        else:
                            break
                    if (dist_s + 1 + dist_n) < min_w:
                        fill_merid[j, i] = True

            # A cell in a narrow passage is one that's narrow in at least
            # one direction. But we only want to close actual straits, not
            # peninsulas. A narrow strait is narrow zonally OR meridionally.
            fill_cells = fill_zonal | fill_merid
            n_filled = int(np.sum(fill_cells))
            if n_filled > 0:
                ocean_mask[fill_cells] = False
                # Keep H_np unchanged — land cells retain their depth value
                # but are masked out. Setting H=0 confuses partial-cell coord.
                land_mask_init = jnp.asarray(ocean_mask.astype(np.float64))
                H_bathy_init = jnp.asarray(H_np, dtype=jnp.float64)
            print(f"  Narrow passage fill (min_width={min_w}): "
                  f"{n_filled} cells → land")

        # --- Remove small enclosed basins ---
        # After closing narrow passages and polar caps, some small bays
        # may remain connected to the open ocean only through 1-2 cells.
        # These drain over multi-year runs (no sea ice to buffer).
        # Fix: flood-fill from the largest connected ocean basin, then
        # remove any disconnected basins smaller than min_basin_size.
        from collections import deque
        H_np = np.array(H_bathy_init, copy=True)
        ocean_mask = np.array(land_mask_init, copy=True) > 0.5
        n_lat_g, n_lon_g = ocean_mask.shape
        labeled = np.zeros(ocean_mask.shape, dtype=np.int32)
        basin_id = 0
        basin_sizes = {}
        for j in range(n_lat_g):
            for i in range(n_lon_g):
                if ocean_mask[j, i] and labeled[j, i] == 0:
                    basin_id += 1
                    q = deque()
                    q.append((j, i))
                    labeled[j, i] = basin_id
                    count = 0
                    while q:
                        cj, ci = q.popleft()
                        count += 1
                        for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nj = cj + dj
                            ni = (ci + di) % n_lon_g
                            if 0 <= nj < n_lat_g and ocean_mask[nj, ni] and labeled[nj, ni] == 0:
                                labeled[nj, ni] = basin_id
                                q.append((nj, ni))
                    basin_sizes[basin_id] = count
        if basin_sizes:
            main_basin = max(basin_sizes, key=basin_sizes.get)
            n_removed = 0
            for bid, bsize in basin_sizes.items():
                if bid != main_basin:
                    small_cells = labeled == bid
                    ocean_mask[small_cells] = False
                    n_removed += int(np.sum(small_cells))
            if n_removed > 0:
                land_mask_init = jnp.asarray(ocean_mask.astype(np.float64))
                H_bathy_init = jnp.asarray(H_np, dtype=jnp.float64)
            n_basins_removed = len(basin_sizes) - 1
            print(f"  Small basin removal: {n_removed} cells in "
                  f"{n_basins_removed} disconnected basins → land "
                  f"(main basin: {basin_sizes[main_basin]} cells)")

        # Snap H_bathy to layer interfaces when the resulting partial
        # cell would be too thin.  Thin partial cells (<30% of full
        # dz_ref) at the bottom of deep equatorial columns drove the
        # day-13 PGF instability we diagnosed in the 30-day spinup —
        # f≈0 there, so geostrophy can't damp pressure-gradient errors
        # quickly, and a 67 m partial cell adjacent to a 470 m full
        # cell amplified SMC03 PGF errors enough to blow up.  The snap
        # is the standard MOM6/MITgcm fix: round H_bathy DOWN to the
        # nearest interface above (i.e., bottom moves up by one level)
        # whenever the partial cell would be thinner than the cutoff,
        # so that every column ends with a full bottom cell or a
        # "thick enough" partial cell.
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        H_np = np.asarray(H_bathy_init)
        z_half_np = np.asarray(z_coord.z_half_ref)        # negative
        dz_ref_np = np.asarray(z_coord.dz_ref)            # positive
        abs_z_half = np.abs(z_half_np)                    # positive
        H_snapped = H_np.copy()
        n_snapped = 0
        thin_threshold = 0.3
        for k in range(z_coord.n_levels):
            top = abs_z_half[k]
            bot = abs_z_half[k + 1]
            in_layer = (H_np > top) & (H_np <= bot)
            partial_h = H_np - top
            too_thin = in_layer & (partial_h < thin_threshold * dz_ref_np[k])
            H_snapped = np.where(too_thin, top, H_snapped)
            n_snapped += int(np.sum(too_thin))
        # Cells where the new H_bathy is at the surface (k=0 case
        # snapped down to 0) become land.  Update land_mask consistently.
        new_land = (H_snapped <= 0.0) & (np.asarray(land_mask_init) > 0.5)
        n_new_land = int(np.sum(new_land))
        if n_new_land > 0:
            land_mask_init = jnp.where(
                jnp.asarray(new_land), 0.0, land_mask_init,
            )
        H_bathy_init = jnp.asarray(H_snapped, dtype=jnp.float64)
        print(f"  Partial-cell snap (cutoff {thin_threshold*100:.0f}%): "
              f"{n_snapped} cells snapped, {n_new_land} → land")
        z_coord_partial = create_partial_cell_coordinate(z_coord, H_bathy_init)
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        model = LatLonCGridOceanModel(grid, z_coord_partial, config)
        # The scan body calls model._step_impl() (no inner JIT) so
        # partial-cell + lax.scan now works correctly.

    # Initial state: from restart, WOA, or rest.
    start_step = 0
    state = _init_rest_state(
        grid_type, grid, z_coord, args.H_max,
        H_bathy=H_bathy_init, land_mask=land_mask_init,
    )
    if args.woa_init and T_woa is not None and S_woa is not None:
        # Replace rest-state T/S with WOA18 climatology.
        # Keep zero velocity, zero eta — let the model adjust.
        T_woa_masked = T_woa * state.land_mask.data[..., jnp.newaxis]
        S_woa_masked = S_woa * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_woa_masked.astype(state.T.data.dtype)),
            S=state.S.replace(data=S_woa_masked.astype(state.S.data.dtype)),
        )
        print(f"  WOA18 initialization: T=[{float(T_woa_masked[state.land_mask.data > 0.5].min()):.1f}, "
              f"{float(T_woa_masked[state.land_mask.data > 0.5].max()):.1f}]°C, "
              f"S=[{float(S_woa_masked[state.land_mask.data > 0.5].min()):.1f}, "
              f"{float(S_woa_masked[state.land_mask.data > 0.5].max()):.1f}] PSU")
    if args.restart is not None:
        state, restart_day, restart_step = _load_restart(
            args.restart, state,
        )
        start_step = restart_step
        print(f"  Restart: loaded day {restart_day:.1f} (step {restart_step}) "
              f"from {Path(args.restart).name}")

    # Forcing dispatch: 'restoring' (default) vs JRA55-do bulk fluxes.
    restoring_targets = None
    restoring_tau_s = None
    jra55_state = None

    if args.forcing_mode == "jra55_do_tropical":
        # Tropical OMIP: bulk fluxes from JRA55-do cache. The setup
        # also builds the polar sponge layer (60°S/60°N), the SSS-
        # restoring target, and the T_freeze cap region from the WOA
        # climatology and z-coordinate.
        jra55_state = _setup_jra55_forcing_state(
            args, grid, grid_type,
            z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
        )
        # Provide the ocean mask for global freeze-cap when no sponge.
        jra55_state["_ocean_mask_2d"] = state.land_mask.data > 0.5
        jra55_state["_gpu_interp"] = getattr(args, "gpu_interp", False)
        flags = []
        if jra55_state.get("enable_sponge"):
            flags.append("sponge")
        if jra55_state.get("enable_sss_restoring"):
            flags.append("SSS-restoring")
        if jra55_state.get("enable_freeze_cap"):
            flags.append("freeze-cap")
        flag_str = ", ".join(flags) or "no closure-domain features"
        print(
            f"  Forcing: jra55_do_tropical "
            f"(cache={Path(args.jra55_cache).name}, "
            f"ref_year={jra55_state['ref_year']}, {flag_str})"
        )
    elif not args.no_restoring and grid_type != "spectral":
        # SST/SSS restoring toward WOA climatology for FV grids.
        # Spectral model: restoring is unstable at coarse resolution
        # (large-scale gradients trigger exponential growth that
        # hyperdiffusion cannot suppress); rely on dynamics + diffusion.
        restoring_targets = (T_woa[..., 0], S_woa[..., 0])
        restoring_tau_s = args.restoring_timescale * 86400.0  # days → seconds
        print(f"  Restoring: tau={args.restoring_timescale:.0f} days")
    elif grid_type == "spectral":
        print(f"  Restoring: disabled (spectral stability)")

    setup_time = time.time() - t_setup
    print(f"  Setup: {setup_time:.1f}s")

    # --- Save full run configuration ---
    # Dump every CLI arg plus the constructed physics config so that
    # restarts can be relaunched with identical parameters.  The file
    # is written at the START of the run (not the end) so it exists
    # even if the run crashes.
    config_dir = Path(args.output) / grid_type / resolution
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"cli_args": vars(args)}
    # Include the actual ocean config fields (these reflect defaults
    # that were applied inside _create_setup, not just the CLI overrides).
    if hasattr(config, "_fields"):
        ocean_cfg = {}
        for field_name in config._fields:
            val = getattr(config, field_name)
            # Serialize NamedTuples and configs as dicts recursively
            if hasattr(val, "_fields"):
                sub = {}
                for sf in val._fields:
                    sv = getattr(val, sf)
                    if hasattr(sv, "_fields"):
                        sub[sf] = {ssf: getattr(sv, ssf) for ssf in sv._fields
                                   if not callable(getattr(sv, ssf))}
                    elif callable(sv):
                        sub[sf] = str(sv)
                    else:
                        sub[sf] = sv
                ocean_cfg[field_name] = sub
            elif callable(val):
                ocean_cfg[field_name] = str(val)
            else:
                ocean_cfg[field_name] = val
        run_config["ocean_config"] = ocean_cfg
    config_path = config_dir / "run_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2, default=str)
        print(f"  Config saved: {config_path}")
    except Exception as e:
        print(f"  Warning: could not save config: {e}")

    # Restart cadence — only wired for the JRA55 path for now (the
    # restoring path is fast enough that re-running from scratch is
    # cheaper than maintaining restarts; revisit if needed).
    checkpoint_dir = None
    checkpoint_days = None
    if args.checkpoint_days > 0.0:
        checkpoint_dir = Path(args.output) / grid_type / resolution
        checkpoint_days = float(args.checkpoint_days)
        print(
            f"  Checkpoint cadence: every {checkpoint_days:g} simulated days "
            f"→ {checkpoint_dir}"
        )

    # Run time loop
    state, diag, wall_time, ok, blowup_info = _run_omip_loop(
        model, state, grid_type, grid, z_coord,
        dt, n_steps, diag_every,
        label=f"{grid_type}/{resolution}",
        restoring_targets=restoring_targets,
        restoring_tau_s=restoring_tau_s,
        jra55_state=jra55_state,
        checkpoint_days=checkpoint_days,
        checkpoint_dir=checkpoint_dir,
        start_step=start_step,
        nudge_woa_tau=args.nudge_woa_tau,
        T_woa_3d=(T_woa * state.land_mask.data[..., jnp.newaxis]).astype(
            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
    )

    status = "PASS" if ok else "FAIL"
    icon = "  " if ok else "**"
    sst_str = f"SST={diag['SST'][-1]:.2f}" if diag["SST"] else ""
    # iter-97: when the run blew up, ``diag['SST'][-1]`` is the
    # last *clean* diagnostic from BEFORE the BLOWUP, which can
    # mislead the reader into thinking the run is healthy.
    # Show the BLOWUP marker explicitly.
    if blowup_info is not None:
        sst_str = f"BLOWUP at step {blowup_info['step']}"
    print(f"\n  {icon} {status} | {grid_type}/{resolution} | "
          f"{wall_time:.1f}s | {sst_str}")

    # Save output
    output_dir = Path(args.output) / grid_type / resolution
    results = _save_output(
        output_dir, diag, args, grid_type, wall_time, ok,
        blowup_info=blowup_info,
    )

    ALL_RESULTS.append(results)
    return results


# ===========================================================================
# Summary
# ===========================================================================

def print_summary():
    """Print summary table of all runs."""
    if not ALL_RESULTS:
        return

    print(f"\n{'='*70}")
    print("  OMIP SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Grid':<15s} {'Resolution':<10s} {'Status':<8s} "
          f"{'Time (s)':<10s} {'Final SST':<10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for r in ALL_RESULTS:
        # iter-97: when the run blew up, show "BLOWUP@N" instead
        # of the last-clean SST (which misled the iter-96 audit
        # into a false-improvement claim).
        blowup = r.get("blowup_info")
        if blowup is not None:
            sst = f"BLOWUP@{blowup['step']}"
        elif r["final_SST"] is not None:
            sst = f"{r['final_SST']:.2f}"
        else:
            sst = "N/A"
        print(f"  {r['grid_type']:<15s} {r['resolution']:<10s} "
              f"{r['status']:<8s} {r['wall_time_s']:<10.1f} {sst:<10s}")

    n_pass = sum(1 for r in ALL_RESULTS if r["status"] == "PASS")
    n_total = len(ALL_RESULTS)
    print(f"\n  {n_pass}/{n_total} passed")
    print(f"{'='*70}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    grids = GRID_TYPES if args.grid == "all" else [args.grid]

    print(f"legoESM OMIP Reference Simulation")
    print(f"  Grids: {', '.join(grids)}")
    print(f"  Days: {'30 (quick)' if args.quick else args.days}")
    print(f"  Physics: {args.physics}")

    for grid_type in grids:
        try:
            run_omip_single(grid_type, args)
        except Exception:
            print(f"\n  !! ERROR running {grid_type}:")
            traceback.print_exc()
            ALL_RESULTS.append({
                "grid_type": grid_type,
                "resolution": args.resolution or GRID_DEFAULTS[grid_type]["resolution"],
                "status": "ERROR",
                "wall_time_s": 0.0,
                "final_SST": None,
                "final_SSS": None,
                "final_SSH": None,
            })

    print_summary()

    # Exit with error if any failed
    if any(r["status"] != "PASS" for r in ALL_RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
