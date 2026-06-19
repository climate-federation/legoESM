"""legoESM-MITgcm recipe: zonally-reentrant ACC-like channel (Southern Ocean).

A pure-config legoESM reproduction of MITgcm ``verification/tutorial_reentrant_channel``
— the canonical idealized Southern-Ocean channel: a wind- and buoyancy-driven,
zonally-PERIODIC Cartesian beta-plane channel with Gent-McWilliams/Redi eddy
parameterization, surface temperature restoring, an RBCS northern sponge, and a
mid-depth N-S ridge. It validates the production lat-lon C-grid model on the
ACC regime that legoESM already exercises against Veros (``veros_acc_recipe``),
but now against a SECOND oracle (MITgcm) with a SIMPLER closure stack (constant
GM coefficient, constant vertical mixing + convective adjustment — no TKE/EKE).

MITgcm setup (``input/data`` + ``data.gmredi`` + ``data.rbcs`` + ``gendata_50km.m``),
reproduced field-for-field at the COARSE (50 km, GM-parameterized) resolution:

================  ===============================================================
grid              Cartesian beta-plane, nx=20 x ny=40 x nz=49, dx=dy=50 km,
                  ZONALLY REENTRANT (delX=20*50e3, no E/W walls); solid wall on
                  the SOUTH row (``bathy(:,1)=0``). f0=-1.363e-4, beta=1.313e-11
                  (Southern-Hemisphere: f<0).
vertical          delR = 49 stretched levels (Stewart et al. 2017 tanh grid),
                  5.49 m (surface) -> 149.4 m (deepest); Ho ~ 3984 m.
bathymetry        Flat bottom at -Ho with a mid-domain N-S ridge (rows 6-14,
                  +2000 m sin-bump) and a sloping notch (the f/H "leak" band).
forcing           wind tau_x(Y) = 0.2 sin(pi Y/Ly), Y the meridional index
                  (eastward westerlies peaking mid-channel); surface T restoring
                  to SST*(Y) = -2 + 12 Y/40 [degC] over tauThetaClimRelax=10 d.
rbcs sponge       3-D T restoring to the initial T(z) field over tauRelaxT=10 d,
                  with the gendata mask: full (1.0) at the NORTH row (y=40) for
                  k>=2, 0.25 at y=39 for k>=2, zero elsewhere (surface SST
                  restoring handles k=1).
physics           GM/Redi: GM_background_K=1000 (kappa_GM=kappa_Redi=1000),
                  dm95 taper, advective (skew-flux) form. Linear EOS
                  rho=-rho0 alpha T' (tAlpha=2e-4, sBeta=0, salt frozen).
                  viscAh=2000 (Laplacian), diffKhT=0; viscAr=3e-3, diffKrT=1e-5
                  (implicit vertical); ivdc_kappa=1 convective adjustment;
                  implicit free surface.
time              deltaT=1000 s, 10 steps (a short spin-up for the first oracle
                  tier; the full run is 30 yr).
================  ===============================================================

Wind sign: MITgcm ``zonalWindFile`` IS the ocean-side stress; legoESM's external-
forcing path treats ``OceanSurfaceForcing.tau_x`` as the ATMOSPHERIC stress and
flips it, so this recipe passes the NEGATED ocean stress (same convention as the
ACC / baroclinic-gyre recipes).

Coverage scope (v1, honest): the coarse 50 km GM-parameterized configuration is
reproduced field-for-field. The eddy-permitting 5 km branch (``useGMRedi=.FALSE.``,
Leith viscosity) is NOT built — it is a resolution/closure swap, not new
infrastructure, and is a documented follow-up. The MITgcm ``layers`` package
diagnostic (residual-mean overturning in density coordinates) is a post-processing
DIAGNOSTIC, not a prognostic process; it is out of scope for the recipe (legoESM
computes overturning from the model state in its own diagnostics layer).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_beta_plane_cgrid_geometry,
)
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    LateralMixingConfig,
)
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.sponge import SpongeForcing
from legoesm.ocean.state import (
    LatLonCGridOceanConfig,
    LatLonCGridOceanState,
    OceanSurfaceForcing,
)
from legoesm.ocean.vertical import create_z_star_from_thicknesses

# --- MITgcm tutorial_reentrant_channel parameters (input/data + gendata_50km.m).
NX = 20
NY = 40
DX_M = 50.0e3
DY_M = 50.0e3
F0 = -1.363e-4          # [s^-1] (Southern Hemisphere)
BETA = 1.313e-11        # [m^-1 s^-1]
# delR: 49 stretched levels (Stewart et al. 2017 tanh grid), surface -> deep.
DELR_M = (
    5.48716549, 6.19462098, 6.99291201, 7.89353689, 8.90937723, 10.05483267,
    11.34595414, 12.80056778, 14.43837763, 16.28102917, 18.35210877,
    20.67704362, 23.28285446, 26.1976981, 29.45012046, 33.06792588,
    37.07656002, 41.496912, 46.34247864, 51.61592052, 57.30518684,
    63.37960847, 69.78661289, 76.44996107, 83.27047568, 90.13003112,
    96.89898027, 103.44631852, 109.65099217, 115.4122275, 120.65692923,
    125.34295968, 129.45821977, 133.01641219, 136.05088105, 138.60793752,
    140.74074276, 142.50436556, 143.95220912, 145.133724, 146.09317287,
    146.86917206, 147.49475454, 147.99774783, 148.40131516, 148.72455653,
    148.98310489, 149.18968055, 149.35458582,
)
HO_M = float(sum(DELR_M))            # ~3984 m
RHO_CONST = 1035.0                   # rhoConst / rhoNil
GRAVITY = 9.81
T_ALPHA = 2.0e-4                     # tAlpha
TAU_MAX = 0.2                        # [N/m^2] wind amplitude (gendata 0.2 sin)
T_MAX_RESTORE = 10.0                 # SST_relax Tmax [degC]
T_MIN_RESTORE = -2.0                 # SST_relax Tmin [degC]
RIDGE_HEIGHT_M = 2000.0              # gendata ridge bump amplitude
T_EFOLD_M = 500.0                    # gendata h: T(z) e-folding scale
VISC_AH = 2000.0                     # Laplacian horizontal viscosity -> A_h
DIFF_KH_T = 0.0                      # horizontal tracer diffusivity (GM handles it)
VISC_AR = 3.0e-3                     # vertical viscosity -> A_v
DIFF_KR_T = 1.0e-5                   # vertical tracer diffusivity -> K_v
IVDC_KAPPA = 1.0                     # ivdc_kappa convective adjustment -> K_conv
GM_BACKGROUND_K = 1000.0            # GM_background_K (kappa_GM = kappa_Redi)
GM_S_MAX = 0.01                     # dm95 default slope-clip (MITgcm GM_maxSlope)
DT_S = 1000.0                       # deltaT
TAU_RESTORE_S = 864000.0           # tauThetaClimRelax = tauRelaxT = 10 days
NO_SALT_RESTORE_TAU_S = 1.0e30     # salt frozen (saltStepping=.FALSE.)


class MitgcmReentrantChannelRecipe(NamedTuple):
    """Assembled legoESM reproduction of the MITgcm reentrant channel."""

    geometry: LatLonCGridGeometry
    z_coord: object
    config: LatLonCGridOceanConfig
    state: LatLonCGridOceanState
    wind_forcing: OceanSurfaceForcing
    sponge: SpongeForcing
    land_mask: jnp.ndarray
    dt_s: float


def build_reentrant_channel_geometry() -> LatLonCGridGeometry:
    """Cartesian beta-plane channel matching MITgcm's grid + Coriolis.

    Zonally REENTRANT: the lat-lon C-grid operators are intrinsically zonal-
    periodic (axis-1 ``jnp.roll`` wrap), so a reentrant channel is built by
    simply NOT adding E/W wall columns (the south wall is imposed via the land
    mask). ``y_origin_m=0`` so the absolute ``y`` (hence ``f = f0 + beta·y``)
    matches MITgcm's ``ini_cori.F``; ``cartesian_pseudo_lat=True`` pins the
    pseudo-lat at 0 for the metric-consistent implicit free surface (same as the
    barotropic-gyre / front_relax recipes)."""
    return create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DY_M, f0=F0, beta=BETA,
        y_origin_m=0.0, x_origin_m=0.0,
        cartesian_pseudo_lat=True,
    )


def build_reentrant_channel_bathymetry() -> np.ndarray:
    """MITgcm ``gendata_50km.m`` bathymetry, depth [m] (positive down), shape
    ``(ny, nx)`` in legoESM order.

    gendata builds ``bathy`` as ``(nx=20, ny=40)`` (MATLAB ``ones(20,40)``):

    * the ridge ``bathy(6:14,:)`` runs N-S — it occupies ZONAL (nx) rows 6:14
      (9 cells) and is constant across all 40 meridional (ny) columns; the bump
      ``2000·sin(0:pi/8:pi)`` (9 values) varies ACROSS the zonal band;
    * the notch ``bathy(6:14,15:25)`` cuts through the ridge in the MERIDIONAL
      (ny) band 15:25 — the unblocked f/H corridor that lets the ACC equilibrate;
    * ``bathy(:,1)=0`` is the SOUTHERN wall (gendata's first ny column).

    We transpose to legoESM ``(ny, nx)``: the zonal ridge band -> nx columns
    5:14, the meridional notch -> ny rows 14:24, the south wall -> ny row 0.

    Returned as a DEPTH (>=0). The land mask is ``depth > 0``; the recipe carries
    a flat ``H_max`` z-star coordinate (the ridge stays wet at full depth —
    partial-cell ridge resolution is a documented follow-up).
    """
    # Build in gendata's native (nx, ny) order, then transpose.
    bathy = np.full((NX, NY), HO_M, dtype=np.float64)   # (nx, ny); flat -Ho
    # Ridge: zonal rows 6:14 (1-based) -> python nx-rows 5:14 (9 cells).
    ridge_x = np.arange(5, 14)                           # gendata 6:14
    bump = np.sin(np.linspace(0.0, np.pi, ridge_x.size))   # (9,), varies in nx
    # Base ridge bump applied across ALL meridional columns.
    bathy[ridge_x, :] -= RIDGE_HEIGHT_M * bump[:, None]
    # Notch in the MERIDIONAL band (gendata ny-columns, 1-based):
    #   15:18 sloping in, 19:21 fully unblocked, 22:25 sloping out.
    # 19:21 (1-based) -> python ny 18:21 fully unblocked (depth = Ho).
    bathy[np.ix_(ridge_x, np.arange(18, 21))] = HO_M
    # 15:18 (1-based) -> python ny 14:18: ramp toward the notch.
    for col in range(14, 18):
        frac = (19 - (col + 1)) / 5.0                   # gendata (19-(15:18))/5
        bathy[ridge_x, col] = HO_M - RIDGE_HEIGHT_M * bump * frac
    # 22:25 (1-based) -> python ny 21:25: ramp out the other side.
    for col in range(21, 25):
        frac = ((col + 1) - 21) / 5.0                   # gendata ((22:25)-21)/5
        bathy[ridge_x, col] = HO_M - RIDGE_HEIGHT_M * bump * frac
    # Southern wall: gendata bathy(:,1)=0 -> the first meridional column is land.
    bathy[:, 0] = 0.0
    return bathy.T.copy()                                # -> (ny, nx)


def build_reentrant_channel_land_mask() -> jnp.ndarray:
    """Cell-centre wet mask (1.0 ocean, 0.0 land): wet where bathy depth > 0.

    Only the southern row is land (zonally reentrant: no E/W walls). The ridge
    is a partial-depth feature, NOT land — it stays wet (the recipe carries a
    flat ``H_max`` z-star coordinate; resolving the ridge's partial cells is a
    documented follow-up, see module docstring)."""
    depth = build_reentrant_channel_bathymetry()
    return jnp.asarray((depth > 0.0).astype(np.float64))


def reentrant_channel_t_surf_restore(y_idx: np.ndarray) -> np.ndarray:
    """gendata SST relaxation target ``T_surf(Y) = Tmin + (Tmax-Tmin)·Y/40``,
    with ``Y = 0.5 .. 39.5`` the cell-centre meridional index (NOT metres).
    Returns a 1-D ``(ny,)`` profile [degC]."""
    return T_MIN_RESTORE + (T_MAX_RESTORE - T_MIN_RESTORE) * y_idx / NY


def reentrant_channel_initial_temperature(z_c_m: np.ndarray) -> np.ndarray:
    """gendata 3-D initial / RBCS-reference T field, shape ``(ny, nz)``.

    ``T_3D(y,z) = (T_surf(y) - Tmin)·(exp(z/h) - exp(H/h))/(1 - exp(H/h)) + Tmin``
    with ``h=500 m`` the e-folding scale, ``H = -Ho`` (z negative down), and
    ``T_surf(y)`` the SST-restore profile. Surface (z->0) -> T_surf; deep
    (z->H) -> Tmin."""
    y_idx = 0.5 + np.arange(NY)                          # gendata y=0.5:39.5
    t_surf = reentrant_channel_t_surf_restore(y_idx)     # (ny,)
    z = np.asarray(z_c_m)                                # negative depths (nz,)
    h = T_EFOLD_M
    big_h = -HO_M                                        # gendata H = -sum(dz)
    shape = (np.exp(z / h) - np.exp(big_h / h)) / (1.0 - np.exp(big_h / h))  # (nz,)
    return (t_surf[:, None] - T_MIN_RESTORE) * shape[None, :] + T_MIN_RESTORE


def build_reentrant_channel_wind(geom: LatLonCGridGeometry) -> OceanSurfaceForcing:
    """gendata zonal wind ``taux(Y) = 0.2 sin(pi Y/(ny-1))`` [N/m^2].

    gendata: ``taux = 0.2*sin(0 : pi/39 : pi)`` over the 40 y-rows. This is the
    ocean-side stress; pass the NEGATED value so legoESM's atmosphere->ocean flip
    reproduces MITgcm's ``+taux`` (westerlies accelerate the ACC eastward)."""
    yy = np.linspace(0.0, np.pi, NY)                     # 0 : pi/39 : pi
    tau_ocean = TAU_MAX * np.sin(yy)                     # (ny,) ocean-side stress
    tau_x = jnp.asarray(
        np.broadcast_to(-tau_ocean[:, None], (NY, NX)))  # negated: see docstring
    tau_y = jnp.zeros((NY, NX), dtype=tau_x.dtype)
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


def build_reentrant_channel_restoring(geom: LatLonCGridGeometry) -> RestoringConfig:
    """Surface T restoring to ``SST*(Y)`` over 10 days (tauThetaClimRelax);
    salt frozen (saltStepping=.FALSE., huge tau_S)."""
    y_idx = 0.5 + np.arange(NY)
    t_star = reentrant_channel_t_surf_restore(y_idx)     # (ny,)
    t_star_2d = jnp.asarray(np.broadcast_to(t_star[:, None], (NY, NX)))
    return RestoringConfig(
        tau_T=TAU_RESTORE_S,
        tau_S=NO_SALT_RESTORE_TAU_S,
        T_star_array=t_star_2d,
        implicit=False,            # MITgcm relaxes explicitly (dt << 2 tau_T)
    )


def build_reentrant_channel_sponge(
    z_coord, land_mask: jnp.ndarray) -> SpongeForcing:
    """RBCS northern sponge (``data.rbcs`` + gendata ``T_relax_mask``).

    gendata builds a 3-D mask: full restoring (1.0) at the NORTH row (y=40,
    python -1) for k>=2 (i.e. below the surface layer where SST restoring acts),
    0.25 at the row just south (y=39, python -2) for k>=2, zero elsewhere. The
    relaxation timescale is ``tauRelaxT=10 d``, so ``gamma = mask / tauRelaxT``
    [1/s] — a FULL-RANK per-cell rate (legoESM ``SpongeForcing.gamma`` of rank
    ``T.ndim``). The reference field is the gendata 3-D initial T(z) (the same
    field used for the IC), broadcast over longitude. Salt is frozen so
    ``S_ref`` = the rest-state S (the sponge dS = gamma·(S_ref - S) -> 0)."""
    z_c = np.asarray(z_coord.z_full_ref)                 # negative depths (nz,)
    nz = z_c.shape[0]
    # 3-D restoring mask (ny, nx, nz): k>=2 means python k>=1 (k=0 is surface).
    mask3 = np.zeros((NY, NX, nz), dtype=np.float64)
    mask3[-1, :, 1:] = 1.0                               # north row, below surface
    mask3[-2, :, 1:] = 0.25                              # row south of it
    lm = np.asarray(land_mask)[:, :, None]               # (ny, nx, 1)
    gamma = (mask3 / TAU_RESTORE_S) * lm                  # 1/s, land-masked
    # Reference T = gendata 3-D T(z) (same as IC), broadcast over lon, masked.
    temp = reentrant_channel_initial_temperature(z_c)    # (ny, nz)
    t_ref = np.broadcast_to(temp[:, None, :], (NY, NX, nz)) * lm
    # Salt reference: uniform 35 (the rest-state value); frozen so inert.
    s_ref = np.broadcast_to(35.0, (NY, NX, nz)) * lm
    return SpongeForcing(
        gamma=jnp.asarray(gamma),
        T_ref=jnp.asarray(t_ref),
        S_ref=jnp.asarray(s_ref),
        u_ref=None, v_ref=None,
    )


def build_reentrant_channel_config(
    geom: LatLonCGridGeometry) -> LatLonCGridOceanConfig:
    """MITgcm reentrant-channel core numerics.

    GM/Redi (GM_background_K=1000, dm95 taper, advective/skew form) via the
    top-level ``gm_redi`` field (the lat-lon C-grid GM path); linear EOS (T only);
    Laplacian ``viscAh=2000``, no horizontal tracer diffusion (GM handles lateral
    tracer mixing); ``viscAr=3e-3`` / ``diffKrT=1e-5`` vertical, implicit;
    ``ivdc_kappa=1`` convective adjustment via the enhanced-diffusion scheme;
    implicit free surface; MITgcm unsplit explicit-Coriolis -> AB2 -> implicit FS
    (the faithful path on the metric-consistent Cartesian beta-plane)."""
    gm_redi = GMRediConfig(
        kappa_GM=GM_BACKGROUND_K,
        kappa_Redi=GM_BACKGROUND_K,    # MITgcm Redi K defaults to GM_background_K
        S_max=GM_S_MAX,
        # MITgcm GM_taper_scheme='dm95' -> the DM95 (Danabasoglu-McWilliams 1995)
        # taper, which legoESM's GMRediConfig implements (taper_width_frac sets
        # the transition band; the 0.1 default reproduces the standard DM95 curve).
    )
    # Selects the SHARED MITgcm-faithful numerics block via the recipe card
    # (mitgcm_recipe.py): flux-form centered momentum, centered tracer, explicit_ab2
    # Coriolis, AB2(total) + the MITgcm-faithful UNSPLIT implicit free surface,
    # implicit vertical mixing.  GM/Redi (GM_background_K=1000, dm95 taper, advective
    # skew form) is threaded through the unsplit step; the ACC case does not develop
    # the 2dx baroclinic checkerboard (GM/Redi removes the grid-scale APE).  This
    # supplies only the per-setup knobs: viscAh=2000, NO horizontal tracer diffusion
    # (diffKhT=0; GM handles lateral tracer mixing), viscAr=3e-3 / diffKrT=1e-5
    # vertical, ivdc_kappa=1 convective adjustment, linear EOS (T only).
    return mitgcm_canonical_ocean_config(
        g=GRAVITY,
        rho_0=RHO_CONST,
        eos_linear=LinearEOSConfig(rho_ref=RHO_CONST, alpha_T=T_ALPHA, beta_S=0.0),
        A_h=VISC_AH,
        K_h=DIFF_KH_T,
        A_v=VISC_AR,
        K_v=DIFF_KR_T,
        # GM/Redi is a TOP-LEVEL dynamics field on the lat-lon C-grid (the model
        # reads config.gm_redi; setting it only in physics.lateral_mixing would
        # leave it inactive — see the ACC recipe's note).
        gm_redi=gm_redi,
        physics=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                # K_bg=0: the stable-region background vertical diffusivity comes
                # SOLELY from the top-level config.K_v=diffKrT (the model step adds
                # config.K_v on top of the convection K). Passing K_bg=DIFF_KR_T
                # here too would DOUBLE-COUNT it (effective 2e-5 vs MITgcm's 1e-5).
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=IVDC_KAPPA, K_bg=0.0),
            ),
            surface_forcing=SurfaceForcingConfig(
                scheme="restoring",
                restoring=build_reentrant_channel_restoring(geom)),
            bottom_drag=BottomDragConfig(scheme="none"),
            shortwave_penetration=None,
        ),
    )


def build_reentrant_channel_state(
    geom, z_coord, land_mask) -> LatLonCGridOceanState:
    """Rest state with the gendata 3-D analytic T(z) stratification (S uniform 35,
    frozen)."""
    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord, H_max=HO_M, S_uniform=35.0,
        land_mask_override=land_mask)
    z_c = np.asarray(z_coord.z_full_ref)                 # negative depths (nz,)
    nz = z_c.shape[0]
    temp = reentrant_channel_initial_temperature(z_c)    # (ny, nz)
    lm = np.asarray(land_mask)[:, :, None]
    temp3d = np.broadcast_to(temp[:, None, :], (NY, NX, nz)) * lm
    return base._replace(T=base.T.replace(data=jnp.asarray(temp3d)))


def build_reentrant_channel_recipe() -> MitgcmReentrantChannelRecipe:
    """Assemble the full legoESM-MITgcm reentrant-channel recipe."""
    geom = build_reentrant_channel_geometry()
    z_coord = create_z_star_from_thicknesses(np.asarray(DELR_M))
    land_mask = build_reentrant_channel_land_mask()
    config = build_reentrant_channel_config(geom)
    state = build_reentrant_channel_state(geom, z_coord, land_mask)
    wind = build_reentrant_channel_wind(geom)
    sponge = build_reentrant_channel_sponge(z_coord, land_mask)
    return MitgcmReentrantChannelRecipe(
        geometry=geom, z_coord=z_coord, config=config, state=state,
        wind_forcing=wind, sponge=sponge, land_mask=land_mask, dt_s=DT_S,
    )


__all__ = (
    "BETA", "DELR_M", "DT_S", "DX_M", "DY_M", "F0", "GM_BACKGROUND_K", "HO_M",
    "NX", "NY", "TAU_MAX", "TAU_RESTORE_S", "T_MAX_RESTORE", "T_MIN_RESTORE",
    "VISC_AH", "VISC_AR",
    "MitgcmReentrantChannelRecipe",
    "build_reentrant_channel_bathymetry",
    "build_reentrant_channel_config",
    "build_reentrant_channel_geometry",
    "build_reentrant_channel_land_mask",
    "build_reentrant_channel_recipe",
    "build_reentrant_channel_restoring",
    "build_reentrant_channel_sponge",
    "build_reentrant_channel_state",
    "build_reentrant_channel_wind",
    "reentrant_channel_initial_temperature",
    "reentrant_channel_t_surf_restore",
)
