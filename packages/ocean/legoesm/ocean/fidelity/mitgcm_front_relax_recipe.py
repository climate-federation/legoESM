"""legoESM-MITgcm recipe: relaxation of a baroclinic front in a channel.

A pure-config legoESM reproduction of MITgcm ``verification/front_relax`` — the
FIRST BAROCLINIC MITgcm oracle (after the barotropic gyre). It validates the
production implicit free-surface solver (``barotropic_solver="implicit_cn"``,
which is already baroclinic-capable: one 3-D momentum update + one 2-D eta
elliptic solve + reconstruct) on STRATIFIED geostrophic adjustment — the regime
the single-layer gyre cannot exercise.

MITgcm setup (``input/data`` + ``input/gendata.m``), reproduced field-for-field:

================  ===============================================================
grid              Cartesian channel, nx=1 x ny=32 x nz, dx=dy=10 km, f-plane
                  (f0=1e-4, beta=0); solid wall on the NORTH row (``H(:,end)=0``).
                  A meridional-vertical (y-z) slice — zonally re-entrant (nx=1).
vertical          delR=[50,50,55,60,65,70,80,95,120,155,200,260,320,400,480] m
                  (15 active levels, Ho=2460 m).
stratification    N/f=20 -> N^2=(20 f0)^2; dT/dz=N^2/(alpha g).  A front:
                  T = 20 + dT/dz z + dT/dy Ly sin(pi Y/Ly) exp(-(3 z/Ho)^2),
                  dT/dy = -slope dT/dz, slope=1e-3.  Salt is a PASSIVE tracer
                  (sBeta=0): S = exp(-(2 Y/Ly)^2).
physics           linear EOS rho=-rho0 alpha T' (tAlpha=2e-4, sBeta=0); viscAr=1e-3
                  (vertical), viscA4=1e11 (BIHARMONIC horizontal), diffKrT=3e-5
                  (vertical), diffKhT=0; free-slip walls; implicit free surface;
                  implicit vertical diffusion.
time              deltaT=1800 s, abEps=0.1, 20 steps (a short geostrophic
                  adjustment — fast to validate).
================  ===============================================================

v1 fidelity scope: UNIFORM dy (MITgcm refines dy ~30% in the channel centre) and
15 active levels (MITgcm carries 10 extra 1-m levels below the 2460-m bathymetry
as partial/inactive cells). Both are documented fidelity follow-ups; neither
changes the geostrophic-adjustment physics this oracle validates. The MITgcm
reference for the monitor-statistics tier is the shipped ``results/output.txt``
(``%MON`` per-step dynstat), parsed by ``mitgcm_monitor.py`` — no MITgcm rebuild
needed; a field-dump reference (tendency tier) is a follow-up.
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
from legoesm.ocean.state import (
    LatLonCGridOceanConfig,
    LatLonCGridOceanState,
)
from legoesm.ocean.vertical import create_z_star_from_thicknesses

# --- MITgcm front_relax parameters (input/data + gendata.m). -----------------
NX = 1
NY = 32
DX_M = 10.0e3
DY_M = 10.0e3
F0 = 1.0e-4            # [s^-1]
BETA = 0.0            # f-plane
DELR_M = (50.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 95.0, 120.0, 155.0,
          200.0, 260.0, 320.0, 400.0, 480.0)   # 15 active levels
HO_M = float(sum(DELR_M))                       # 2460 m
RHO_CONST = 1000.0
GRAVITY = 9.81
T_ALPHA = 2.0e-4
N_OVER_F = 20.0
SLOPE = 1.0e-3
VISC_AR = 1.0e-3       # vertical viscosity  -> A_v
VISC_A4 = 1.0e11       # biharmonic horizontal viscosity -> B_h
DIFF_KR_T = 3.0e-5     # vertical T diffusivity -> K_v
DT_S = 1800.0          # deltaT
AB_EPS = 0.1           # abEps


class MitgcmFrontRelaxRecipe(NamedTuple):
    """Assembled legoESM reproduction of the MITgcm front_relax case."""

    geometry: LatLonCGridGeometry
    z_coord: object
    config: LatLonCGridOceanConfig
    state: LatLonCGridOceanState
    dt_s: float


def build_front_relax_geometry() -> LatLonCGridGeometry:
    """Cartesian f-plane channel matching MITgcm's grid + Coriolis.

    The front is centred at Y=0 (``sin(pi Y/Ly)``), so ``y_origin`` puts the
    domain centre at 0.  ``Ly = dy*(ny-1)`` (the wet span; the north row is a
    wall), matching ``gendata.m`` (``Ly=dy*(ny-1)``).
    """
    ly = DY_M * (NY - 1)
    return create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DY_M, f0=F0, beta=BETA,
        y_origin_m=-0.5 * ly, x_origin_m=0.0,
        # #514: operators read the stored dx_v; the workaround is obsolete.
        cartesian_pseudo_lat=False,
    )


def build_front_relax_land_mask() -> jnp.ndarray:
    """Solid wall on the NORTH row only (``gendata.m`` ``H(:,end)=0``); zonally
    re-entrant (nx=1, no E/W walls). 1.0 = ocean, 0.0 = land."""
    mask = np.ones((NY, NX), dtype=np.float64)
    mask[-1, :] = 0.0    # north wall
    return jnp.asarray(mask)


def front_relax_initial_temperature(y_c_m: np.ndarray, z_c_m: np.ndarray) -> np.ndarray:
    """MITgcm ``gendata.m`` analytic front, T(y, z), shape ``(ny, nz)``.

    ``T = 20 + dTdz z + dTdy Ly sin(pi Y/Ly) exp(-(3 z/Ho)^2)`` with
    ``N^2 = (N/f f0)^2``, ``dTdz = N^2/(alpha g)``, ``dTdy = -slope dTdz``.
    ``z_c_m`` are NEGATIVE cell-centre depths; ``y_c_m`` are cell-centre y.
    """
    ly = DY_M * (NY - 1)
    n2 = (N_OVER_F * F0) ** 2
    dtdz = n2 / (T_ALPHA * GRAVITY)
    dtdy = -SLOPE * dtdz
    yy = y_c_m[:, None]                      # (ny, 1)
    zz = z_c_m[None, :]                       # (1, nz)
    return (20.0 + dtdz * zz
            + dtdy * ly * np.sin(np.pi * yy / ly) * np.exp(-(3.0 * zz / HO_M) ** 2))


def front_relax_initial_salt(y_c_m: np.ndarray, nz: int) -> np.ndarray:
    """Passive-tracer salt ``S = exp(-(2 Y/Ly)^2)`` (sBeta=0), shape ``(ny, nz)``."""
    ly = DY_M * (NY - 1)
    s_col = np.exp(-(2.0 * y_c_m / ly) ** 2)   # (ny,)
    return np.broadcast_to(s_col[:, None], (NY, nz)).copy()


def build_front_relax_config(
    *, lateral_viscosity: str = "biharmonic",
) -> LatLonCGridOceanConfig:
    """Single-front baroclinic config matching MITgcm front_relax core numerics.

    * linear EOS ``rho = -rho0 alpha T'`` (tAlpha=2e-4, sBeta=0 -> salt passive);
    * BIHARMONIC horizontal viscosity ``B_h=viscA4=1e11`` (no Laplacian A_h) —
      the FAITHFUL MITgcm choice (``lateral_viscosity="biharmonic"``, default),
      realized by the COMPONENT biharmonic (``flux_divergence`` family applied
      twice; MITgcm ``useStrainTensionVisc=.FALSE.`` per-component del4) — stable,
      unlike the vector ``grad(div)−curl(curl)`` biharmonic which is ill-scaled
      here;
    * vertical viscosity ``A_v=viscAr=1e-3`` + IMPLICIT vertical diffusion
      ``K_v=diffKrT=3e-5`` (``implicitDiffusion=.TRUE.``);
    * free-slip side walls (``no_slip_sides=.FALSE.``);
    * implicit free surface (the solver under validation), fully-backward Euler;
    * MITgcm unsplit explicit-Coriolis -> AB2(abEps=0.1) -> implicit free surface
      (same faithful path as the barotropic gyre; face-f Coriolis on the
      metric-consistent Cartesian grid).

    ``lateral_viscosity="laplacian_stable"`` swaps the biharmonic for a harmonic
    ``flux_divergence`` Laplacian (A_h=400) — an alternative stable closure. Both
    reproduce MITgcm's eta_max (1%) and uvel_max (3%) over the 20-step adjustment;
    they differ only in the (small, transient) ``vvel_max``, which is set by the
    adjustment/time scheme here rather than the viscosity (both give ~0.04 vs
    MITgcm's 0.009 — a finer fidelity gap, not an instability).
    """
    if lateral_viscosity == "biharmonic":
        # flux_divergence family -> the COMPONENT biharmonic (MITgcm-faithful
        # per-component del4, useStrainTensionVisc=.FALSE.; stable, unlike the
        # vector grad(div)-curl(curl) biharmonic).
        a_h, b_h, visc_op = 0.0, VISC_A4, "flux_divergence"
    elif lateral_viscosity == "laplacian_stable":
        a_h, b_h, visc_op = 400.0, 0.0, "flux_divergence"
    else:
        raise ValueError(
            "lateral_viscosity must be 'biharmonic' or 'laplacian_stable', "
            f"got {lateral_viscosity!r}")
    # Selects the SHARED MITgcm-faithful numerics block via the recipe card
    # (mitgcm_recipe.py: flux-form centered momentum, explicit_ab2 Coriolis, AB2,
    # component flux_divergence friction, theta=1 free surface) and supplies the
    # per-deck knobs: BIHARMONIC horizontal viscosity (viscA4=1e11 via B_h; the
    # card pins the flux_divergence operator, so this is the component del4),
    # free-slip walls (no_slip_sides=.FALSE.), the split implicit_cn free surface
    # (the baroclinic-capable solver under validation), viscAr/diffKrT implicit
    # vertical mixing, and linear EOS (salt passive).  No convection/physics block.
    assert visc_op == "flux_divergence"   # the card pins this operator
    return mitgcm_canonical_ocean_config(
        g=GRAVITY,
        rho_0=RHO_CONST,
        eos_linear=LinearEOSConfig(rho_ref=RHO_CONST, alpha_T=T_ALPHA, beta_S=0.0),
        # Horizontal viscosity: faithful biharmonic (MITgcm viscA4) or the
        # runnable flux_divergence-Laplacian proxy (see docstring).
        A_h=a_h,
        K_h=0.0,
        A_v=VISC_AR,
        K_v=DIFF_KR_T,
        # Front advects T with the config-default tracer scheme (kept as-is).
        tracer_advection="tvd",
        # MITgcm no_slip_sides=.FALSE.
        lateral_side_bc="free_slip",
        # Implicit free surface, fully backward-Euler (the solver under test).
        barotropic_solver="implicit_cn",
        ab2_epsilon=AB_EPS,
        # Biharmonic viscosity (component del4) via the card's overrides.
        B_h=b_h,
        B_h_lat_scaling=False,
    )


def build_front_relax_state(geom, z_coord) -> LatLonCGridOceanState:
    """Rest state with the analytic baroclinic front in T (+ passive S)."""
    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord, land_mask_override=build_front_relax_land_mask(),
    )
    # Cell-centre y from the beta-plane origin (lat is pinned to 0 on the
    # Cartesian grid, so recover y directly: y_c = y_origin + (j+0.5) dy).
    ly = DY_M * (NY - 1)
    y_c = -0.5 * ly + (np.arange(NY) + 0.5) * DY_M
    z_c = np.asarray(z_coord.z_full_ref)       # negative depths, (nz,)
    nz = z_c.shape[0]
    temp = front_relax_initial_temperature(y_c, z_c)        # (ny, nz)
    salt = front_relax_initial_salt(y_c, nz)                # (ny, nz)
    mask2d = np.asarray(build_front_relax_land_mask())       # (ny, nx)
    temp3d = np.broadcast_to(temp[:, None, :], (NY, NX, nz)) * mask2d[:, :, None]
    salt3d = np.broadcast_to(salt[:, None, :], (NY, NX, nz)) * mask2d[:, :, None]
    return base._replace(
        T=base.T.replace(data=jnp.asarray(temp3d)),
        S=base.S.replace(data=jnp.asarray(salt3d)),
    )


def build_front_relax_recipe(
    *, lateral_viscosity: str = "biharmonic",
) -> MitgcmFrontRelaxRecipe:
    """Assemble the full legoESM-MITgcm front_relax recipe.

    ``lateral_viscosity`` selects the faithful biharmonic (default) or the
    runnable ``"laplacian_stable"`` proxy (see :func:`build_front_relax_config`).
    """
    geom = build_front_relax_geometry()
    z_coord = create_z_star_from_thicknesses(np.asarray(DELR_M))
    config = build_front_relax_config(lateral_viscosity=lateral_viscosity)
    state = build_front_relax_state(geom, z_coord)
    return MitgcmFrontRelaxRecipe(
        geometry=geom, z_coord=z_coord, config=config, state=state, dt_s=DT_S,
    )
