"""legoESM-MITgcm recipe: wind- and buoyancy-driven baroclinic double gyre.

A pure-config legoESM reproduction of MITgcm ``verification/tutorial_baroclinic_gyre``
— the canonical baroclinic MITgcm oracle, and the first on the production's NATIVE
SPHERICAL grid (so the iteration-7/8 Cartesian-beta-plane metric work is not in play:
on a spherical grid the C-grid operators' ``cos(lat)`` metric is correct).

It validates the production model on a FORCED, STRATIFIED, mid-latitude spin-up:
wind-driven double gyre + surface buoyancy forcing + convective adjustment + the
implicit free surface.

MITgcm setup (``input/data`` + ``gendata.m``), reproduced field-for-field:

================  ===============================================================
grid              Spherical 62x62x15, dx=dy=1deg, xgOrigin=-1, ygOrigin=14 (so the
                  ocean interior is 60x60, lat 15.5-74.5N, lon 0.5-59.5E), with a
                  solid 1-cell wall ring (``bathy.bin``).
vertical          delR=[50,60,...,190] m (15 levels, Ho=1800 m, flat bottom).
forcing           wind tau_x(Y) = -tauMax cos(2pi (Y-15)/60), tauMax=0.1 (a DOUBLE
                  gyre: easterly N/S, westerly mid); surface T restoring to
                  Trest(Y) = (Tmax-Tmin)/60 (75 - Y), Tmax=30, Tmin=0, over
                  tauThetaClimRelax = 30 days.
physics           linear EOS rho=-rho0 alpha T' (tAlpha=2e-4, sBeta=0, salt frozen);
                  Laplacian viscAh=5000 + diffKhT=1000; viscAr=1e-2, diffKrT=1e-5;
                  ivdc_kappa=1 convective adjustment (enhanced vertical diffusivity
                  where statically unstable); no_slip_sides; implicit free surface;
                  implicit vertical diffusion.
initial           T(z) = tRef = [30,27,24,...,2] (horizontally uniform; no
                  hydrogThetaFile), S frozen.
time              deltaT=1200 s, abEps=0.1, 10 steps (endTime=12000) — short, fast.
================  ===============================================================

The 10-step MITgcm reference (shipped ``results/output.txt`` %MON, monitorSelect=2)
is ``eta_max=0.00844, uvel_max=0.01879, vvel_max=0.01603, theta in [2, 30]`` — parsed
by ``mitgcm_monitor.py``; no MITgcm rebuild needed.

Wind sign: MITgcm ``zonalWindFile`` IS the ocean-side stress; legoESM's external-
forcing path treats ``OceanSurfaceForcing.tau_x`` as the ATMOSPHERIC stress and flips
it, so this recipe passes ``+tauMax cos(...)`` (negated) to reproduce MITgcm's
``-tauMax cos(...)`` ocean stress (same convention as the barotropic gyre + ACC).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.grids.latlon import LatLonGrid, create_regional_latlon_grid
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.state import (
    LatLonCGridOceanConfig,
    LatLonCGridOceanState,
    OceanSurfaceForcing,
)
from legoesm.ocean.vertical import create_z_star_from_thicknesses

# --- MITgcm tutorial_baroclinic_gyre parameters (input/data + gendata.m). -----
NX_INT = 60                      # ocean interior cells (62 incl. the wall ring)
NY_INT = 60
LAT_SOUTH, LAT_NORTH = 15.0, 75.0    # interior bounds (deg); walls at 14.5 / 75.5
LON_WEST, LON_EAST = 0.0, 60.0
DELR_M = (50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
          150.0, 160.0, 170.0, 180.0, 190.0)        # 15 levels
HO_M = float(sum(DELR_M))                            # 1800 m
T_REF_DEGC = (30.0, 27.0, 24.0, 21.0, 18.0, 15.0, 13.0, 11.0, 9.0, 7.0,
              6.0, 5.0, 4.0, 3.0, 2.0)               # initial T(z)
RHO_NIL = 999.8
GRAVITY = 9.81
T_ALPHA = 2.0e-4
TAU_MAX = 0.1                    # [N/m^2]
WIND_Y0 = 15.0                   # gendata yo
N_OCEAN = 60                     # gendata ny-2 (wall-excluded count)
T_MAX_RESTORE, T_MIN_RESTORE = 30.0, 0.0
TAU_RESTORE_S = 2592000.0        # tauThetaClimRelax = 30 days
NO_SALT_RESTORE_TAU_S = 1.0e30   # salt not restored (salt frozen)
VISC_AH = 5000.0                 # Laplacian horizontal viscosity -> A_h
DIFF_KH_T = 1000.0               # horizontal tracer diffusivity -> K_h
VISC_AR = 1.0e-2                 # vertical viscosity -> A_v
DIFF_KR_T = 1.0e-5               # vertical tracer diffusivity -> K_v / K_bg
IVDC_KAPPA = 1.0                 # convective enhanced vertical diffusivity -> K_conv
DT_S = 1200.0                    # deltaT
AB_EPS = 0.1                     # abEps


class MitgcmBaroclinicGyreRecipe(NamedTuple):
    """Assembled legoESM reproduction of the MITgcm baroclinic gyre."""

    geometry: LatLonGrid
    z_coord: object
    config: LatLonCGridOceanConfig
    state: LatLonCGridOceanState
    wind_forcing: OceanSurfaceForcing
    land_mask: jnp.ndarray
    dt_s: float


def build_baroclinic_gyre_grid() -> tuple[LatLonGrid, jnp.ndarray]:
    """Spherical 62x62 closed box (60x60 ocean + 1-cell wall ring) matching the
    MITgcm grid + spherical Coriolis ``f = 2 Omega sin(lat)``."""
    grid, wall_mask = create_regional_latlon_grid(
        NY_INT, NX_INT, lat_south=LAT_SOUTH, lat_north=LAT_NORTH,
        lon_west=LON_WEST, lon_east=LON_EAST, periodic_x=False,
    )
    return grid, wall_mask


def baroclinic_gyre_wind(grid: LatLonGrid) -> OceanSurfaceForcing:
    """Double-gyre zonal wind stress. MITgcm ocean stress is
    ``-tauMax cos(2pi (Y-yo)/(ny-2))``; pass the NEGATED value so legoESM's
    atmosphere->ocean flip reproduces it (Y = cell-centre latitude [deg])."""
    lat_deg = np.degrees(np.asarray(grid.lat))                 # (n_lat,)
    tau_ocean = -TAU_MAX * np.cos(2.0 * np.pi * (lat_deg - WIND_Y0) / N_OCEAN)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    tau_x = jnp.asarray(np.broadcast_to(-tau_ocean[:, None], (n_lat, n_lon)))
    tau_y = jnp.zeros((n_lat, n_lon), dtype=tau_x.dtype)
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


def baroclinic_gyre_t_star(grid: LatLonGrid) -> jnp.ndarray:
    """Surface restoring target ``Trest(Y) = (Tmax-Tmin)/(ny-2) (yo+dy(ny-2) - Y)``
    = ``0.5 (75 - Y)`` [degC], 2-D (broadcast over longitude)."""
    lat_deg = np.degrees(np.asarray(grid.lat))
    trest = (T_MAX_RESTORE - T_MIN_RESTORE) / N_OCEAN * (
        (WIND_Y0 + N_OCEAN) - lat_deg)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    return jnp.asarray(np.broadcast_to(trest[:, None], (n_lat, n_lon)))


def build_baroclinic_gyre_restoring(grid: LatLonGrid) -> RestoringConfig:
    """Surface T restoring to ``Trest`` over 30 days; salt not restored."""
    return RestoringConfig(
        tau_T=TAU_RESTORE_S,
        tau_S=NO_SALT_RESTORE_TAU_S,
        T_star_array=baroclinic_gyre_t_star(grid),
        implicit=False,            # MITgcm relaxes explicitly (dt << 2 tau_T)
    )


def build_baroclinic_gyre_config(grid: LatLonGrid) -> LatLonCGridOceanConfig:
    """MITgcm baroclinic-gyre core numerics (spherical grid).

    Laplacian ``viscAh=5000`` (flux-divergence component operator, MITgcm
    ``useStrainTensionVisc=.FALSE.``) + ``diffKhT=1000`` horizontal tracer
    diffusion; ``viscAr=1e-2`` / ``diffKrT=1e-5`` vertical, applied implicitly;
    ``ivdc_kappa=1`` convective adjustment via the ``enhanced_diffusion`` scheme;
    no-slip walls; implicit free surface; linear EOS (T only); MITgcm unsplit
    explicit-Coriolis -> AB2(abEps=0.1) -> implicit free surface (face-f Coriolis
    is correct here — spherical grid, no Cartesian-metric special case)."""
    return LatLonCGridOceanConfig(
        g=GRAVITY,
        rho_0=RHO_NIL,
        eos="linear",
        eos_linear=LinearEOSConfig(rho_ref=RHO_NIL, alpha_T=T_ALPHA, beta_S=0.0),
        # Laplacian horizontal viscosity (constant; MITgcm viscAh, no cos-lat
        # scaling) + horizontal tracer diffusion.
        A_h=VISC_AH,
        A_h_lat_scaling=False,
        lateral_viscosity_operator="flux_divergence",
        lateral_side_bc="no_slip",
        B_h=0.0,
        C_smag=0.0,
        K_h=DIFF_KH_T,
        # Vertical viscosity/diffusion (implicit) + convective adjustment.
        A_v=VISC_AR,
        K_v=DIFF_KR_T,
        implicit_vertical_mixing=True,
        # No bottom drag, no eddy params.
        bottom_drag_r=0.0,
        gm_redi=None,
        # MITgcm flux-form centered momentum advection + centered tracer advection.
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        tracer_advection="centered",
        # MITgcm-faithful UNSPLIT implicit free surface (theta=1, backward-Euler
        # surface mode = MITgcm implicSurfPress=1). NO barotropic/baroclinic mode
        # split — one 3D predictor + one elliptic eta solve + uniform correction,
        # matching MITgcm's implicitFreeSurface (audited: no barotropic sub-cycle,
        # no barotropic-velocity prognostic). The split implicit_cn breaks the
        # discrete PGF/continuity adjointness at the grid scale and grows a spurious
        # 2dx baroclinic instability (the long-run velocity checkerboard, zig~1.4);
        # the unsplit solver keeps it smooth (zig~0.2) at the FAITHFUL viscAh=5000.
        # docs/ocean_fidelity/mitgcm_unsplit_freesurface_fix.md.
        barotropic_solver="implicit_unsplit",
        barotropic_implicit_theta_eta=1.0,
        barotropic_implicit_theta_pgf=1.0,
        coriolis_scheme="explicit_ab2",
        outer_integrator="ab2",
        ab2_epsilon=AB_EPS,
        differentiable_barotropic=True,
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        physics=OceanPhysicsConfig(
            # Constant vertical mixing (config A_v/K_v) — no KPP/TKE.
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            # ivdc_kappa=1 convective adjustment.  K_bg=0: the background
            # diapycnal diffusivity (DIFF_KR_T) is supplied ONCE by config.K_v
            # (the implicit-vmix solve adds the config K_v floor); the convection
            # scheme contributes only the convective ENHANCEMENT above it, so
            # stable-region K_v stays DIFF_KR_T and is not double-counted.
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=IVDC_KAPPA, K_bg=0.0),
            ),
            surface_forcing=SurfaceForcingConfig(
                scheme="restoring", restoring=build_baroclinic_gyre_restoring(grid)),
            bottom_drag=BottomDragConfig(scheme="none"),
            shortwave_penetration=None,
        ),
    )


def build_baroclinic_gyre_state(grid, z_coord, land_mask) -> LatLonCGridOceanState:
    """Rest state with the horizontally-uniform ``tRef`` stratification in T."""
    base = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=HO_M, land_mask_override=land_mask)
    nz = np.asarray(z_coord.z_full_ref).shape[0]
    t_ref = np.asarray(T_REF_DEGC)[:nz]                       # (nz,)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    mask3 = np.asarray(land_mask)[:, :, None]
    temp3d = np.broadcast_to(t_ref[None, None, :], (n_lat, n_lon, nz)) * mask3
    return base._replace(T=base.T.replace(data=jnp.asarray(temp3d)))


def build_baroclinic_gyre_recipe() -> MitgcmBaroclinicGyreRecipe:
    """Assemble the full legoESM-MITgcm baroclinic-gyre recipe."""
    grid, land_mask = build_baroclinic_gyre_grid()
    z_coord = create_z_star_from_thicknesses(np.asarray(DELR_M))
    config = build_baroclinic_gyre_config(grid)
    state = build_baroclinic_gyre_state(grid, z_coord, land_mask)
    wind = baroclinic_gyre_wind(grid)
    return MitgcmBaroclinicGyreRecipe(
        geometry=grid, z_coord=z_coord, config=config, state=state,
        wind_forcing=wind, land_mask=land_mask, dt_s=DT_S,
    )
