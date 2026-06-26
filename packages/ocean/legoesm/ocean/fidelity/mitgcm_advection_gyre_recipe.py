"""legoESM-MITgcm recipe: passive-tracer advection in a barotropic gyre.

A pure-config legoESM reproduction of MITgcm
``verification/tutorial_advection_in_gyre`` — the MITgcm passive-TRACER oracle.
It shares the barotropic gyre's grid / wind / single-layer dynamics
(``mitgcm_barotropic_gyre_recipe``) and ADDS a passive dye tracer advected by
the equilibrated gyre flow with a high-order flux-limited scheme.

MITgcm setup (``input/data`` + ``input/data.ptracers`` + ``gendata.m``):

================  ===============================================================
grid              Cartesian beta-plane, 60 x 60 cells x 1 level, dx = dy = 20 km,
                  solid walls on the last row/col (``topog.box5000``:
                  ``h(end,:)=0; h(:,end)=0``).  Ho = 5000 m.
Coriolis          f0 = 1e-4 (MITgcm default), beta = 1e-11 -> f = f0 + beta*y.
wind              tau_x(y) = -tauMax cos(2 pi Y), tauMax = 0.1 N/m^2,
                  Y = (j - 0.5)/(ny - 1)  (``windx.m01cos2y``).
dynamics          viscAh = 400 (Laplacian), no_slip_sides, no_slip_bottom,
                  implicit free surface, linear EOS (homogeneous single layer).
tracer            ONE passive dye (``PTRACERS_numInUse=1``), advection scheme 80
                  (3rd-order DST flux limiter, multi-dim), diffKh = diffK4 =
                  diffKr = 0 (PURE advection).  Initialized AFTER a 10-year
                  spin-up (``PTRACERS_Iter0=259200``) as a single cell of
                  concentration 1 at (i=2, j=30) [1-based Fortran], i.e.
                  (y=29, x=1) [0-based], everywhere-else zero (``dye.bin``).
time              deltaT = 1200 s, 4 steps (``nTimeSteps=4``).
================  ===============================================================

PASSIVE-TRACER CARRIER.  legoESM's ``LatLonCGridOceanState`` does not carry a
dedicated passive-tracer slot — it carries (T, S).  The dye is therefore carried
in the **salinity** field with a LINEAR EOS whose haline contraction is ZERO
(``beta_S = 0``), exactly matching MITgcm's ``sBeta = 0`` here: S is
dynamically INERT, so it is a true passive tracer.  This is the same convention
``ocean/experiments/stommel_gyre_tracer.py`` uses (a passive salinity blob) and
the same ``sBeta = 0`` MITgcm uses in this deck.

ADVECTION SCHEME.  MITgcm ``PTRACERS_advScheme=80`` = 3rd-order Direct-Space-Time
flux-limited, multi-dimensional.  legoESM's closest canonical block is
``tracer_advection = "dst3_multidim"`` (DST-3 with the transverse predictor
correction), which is what this recipe selects.

EQUILIBRATED FLOW.  The MITgcm tutorial restarts the DYNAMICS from a 10-year
pickup and advects the dye for 4 steps in the (effectively steady) equilibrated
gyre.  The honest, isolated tracer-advection comparison
(``scripts/validate/ocean_fidelity/compare_mitgcm_advection_gyre.py``) PRESCRIBES
MITgcm's equilibrated u, v field into the legoESM state and advects the dye blob
4 steps with ``dst3_multidim`` through the SAME canonical legoESM advection
operator the production model dispatches (``_compute_advection_flux_div``), then
compares to MITgcm's ``PTRACER01`` dumps.  This isolates the advection scheme
from any dynamical-spin-up difference (legoESM-from-rest vs MITgcm-from-pickup).

Wind sign: identical convention to the barotropic gyre recipe (legoESM's
external-forcing path treats ``OceanSurfaceForcing.tau_x`` as the ATMOSPHERIC
stress and flips it, so we pass ``+tauMax cos(...)``).
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
    OceanSurfaceForcing,
)
from legoesm.ocean.vertical import create_ocean_z_star

# --- MITgcm tutorial_advection_in_gyre parameters (input/data, .ptracers). ---
NX = 60                  # delX = 60*20E3
NY = 60                  # delY = 60*20E3
DX_M = 20.0e3
DY_M = 20.0e3
F0 = 1.0e-4              # [s^-1]  MITgcm default (f0 absent from data)
BETA = 1.0e-11           # [m^-1 s^-1]
Y_ORIGIN_M = 0.0
X_ORIGIN_M = 0.0
H_DEPTH_M = 5000.0       # Ho
TAU_MAX = 0.1            # [N/m^2]
VISC_AH = 400.0          # [m^2/s]  viscAh
DT_S = 1200.0            # deltaT
N_STEPS = 4              # nTimeSteps
RHO_CONST = 1000.0
G_BARO = 9.81

# Dye blob (gendata.m: dye(2,30)=1 -> 0-based (y=29, x=1)).
DYE_X = 1
DYE_Y = 29
DYE_AMPLITUDE = 1.0      # PTRACERS dye concentration


class MitgcmAdvectionGyreRecipe(NamedTuple):
    """Assembled legoESM reproduction of the MITgcm advection-in-gyre case."""

    geometry: LatLonCGridGeometry
    z_coord: object
    config: LatLonCGridOceanConfig
    state: LatLonCGridOceanState
    wind_forcing: OceanSurfaceForcing
    dt_s: float
    n_steps: int


def build_advgyre_geometry() -> LatLonCGridGeometry:
    """Cartesian beta-plane C-grid matching MITgcm's grid + Coriolis."""
    return create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DY_M, f0=F0, beta=BETA,
        y_origin_m=Y_ORIGIN_M, x_origin_m=X_ORIGIN_M,
        # #514: operators read the stored dx_v; the workaround is obsolete.
        cartesian_pseudo_lat=False,
    )


def build_advgyre_land_mask() -> jnp.ndarray:
    """Closed box: solid walls on the last row and last col only.

    Reproduces ``gendata.m`` ``h(end,:)=0; h(:,end)=0`` exactly (the south/west
    boundaries are OPEN cells walled by the C-grid face masks, matching MITgcm's
    topography).  1.0 = ocean, 0.0 = land.
    """
    mask = np.ones((NY, NX), dtype=np.float64)
    mask[-1, :] = 0.0
    mask[:, -1] = 0.0
    return jnp.asarray(mask)


def advgyre_taux_ocean_profile() -> np.ndarray:
    """MITgcm OCEAN-side zonal wind stress per row: ``-tauMax cos(2 pi Y)``.

    ``Y[j] = (j - 0.5)/(ny - 1)`` (``gendata.m``: ``y=((1:ny)-0.5)/(ny-1)``).
    Returns shape ``(NY,)`` [N/m^2].
    """
    j = np.arange(1, NY + 1, dtype=np.float64)   # gendata 1:ny
    y_norm = (j - 0.5) / (NY - 1)
    return -TAU_MAX * np.cos(2.0 * np.pi * y_norm)


def build_advgyre_wind() -> OceanSurfaceForcing:
    """Wind as ``OceanSurfaceForcing`` (negated for the atmospheric->ocean flip)."""
    taux_ocean = advgyre_taux_ocean_profile()                     # (NY,)
    tau_x = jnp.asarray(np.broadcast_to(-taux_ocean[:, None], (NY, NX)))
    tau_y = jnp.zeros((NY, NX), dtype=tau_x.dtype)
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


def build_dye_field() -> jnp.ndarray:
    """Dye blob: single cell of concentration 1 at (y=29, x=1), else 0.

    Shape ``(NY, NX, 1)`` to drop straight into the salinity slot of the
    single-layer ``LatLonCGridOceanState``.
    """
    dye = np.zeros((NY, NX, 1), dtype=np.float64)
    dye[DYE_Y, DYE_X, 0] = DYE_AMPLITUDE
    return jnp.asarray(dye)


def build_advgyre_config() -> LatLonCGridOceanConfig:
    """Single-layer barotropic config + passive-dye advection.

    Inherits the barotropic-gyre MITgcm-faithful core numerics and adds:

    * ``eos_linear`` with ``alpha_T = beta_S = 0`` -> constant density: BOTH T
      and S are dynamically inert, so the dye carried in S is a true passive
      tracer (MITgcm ``tAlpha=2e-4`` matters only for the spun-up dynamics, which
      we PRESCRIBE; ``sBeta=0`` is matched exactly).
    * ``tracer_advection = "dst3_multidim"`` -> MITgcm ``PTRACERS_advScheme=80``
      (3rd-order DST flux limiter, multi-dimensional).
    * ``K_h = 0`` -> MITgcm ``PTRACERS_diffKh=0`` (pure advection, no explicit
      tracer diffusion).

    Selects the SHARED MITgcm-faithful numerics block via
    :func:`mitgcm_canonical_ocean_config` (the MITgcm recipe card) — same
    single-layer homogeneous barotropic-gyre core (viscAh=400, abEps=0.01, split
    ``implicit_cn``) — and overrides only the tracer advection to MITgcm's
    ``PTRACERS_advScheme=80`` (``dst3_multidim``) with ``K_h=0`` (pure advection).
    """
    return mitgcm_canonical_ocean_config(
        g=G_BARO,
        rho_0=RHO_CONST,
        eos_linear=LinearEOSConfig(rho_ref=RHO_CONST, alpha_T=0.0, beta_S=0.0),
        A_h=VISC_AH,
        # PURE passive-tracer advection: no explicit tracer diffusion.
        K_h=0.0,
        # MITgcm PTRACERS_advScheme=80 (3rd-order DST flux limiter, multi-dim).
        tracer_advection="dst3_multidim",
        # Split implicit_cn free surface from the equilibrated-gyre pickup (the
        # tracer comparison PRESCRIBES MITgcm's u,v — see module docstring).
        barotropic_solver="implicit_cn",
        ab2_epsilon=0.01,
    )


def build_advgyre_state(z_coord) -> LatLonCGridOceanState:
    """Rest state with the dye blob in the (passive) salinity field."""
    geom = build_advgyre_geometry()
    state = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        land_mask_override=build_advgyre_land_mask(),
    )
    return state._replace(S=state.S.replace(data=build_dye_field()))


def build_advgyre_recipe() -> MitgcmAdvectionGyreRecipe:
    """Assemble the full legoESM-MITgcm advection-in-gyre recipe."""
    geom = build_advgyre_geometry()
    z_coord = create_ocean_z_star(1, H_max=H_DEPTH_M)
    config = build_advgyre_config()
    state = build_advgyre_state(z_coord)
    wind = build_advgyre_wind()
    return MitgcmAdvectionGyreRecipe(
        geometry=geom, z_coord=z_coord, config=config,
        state=state, wind_forcing=wind, dt_s=DT_S, n_steps=N_STEPS,
    )
