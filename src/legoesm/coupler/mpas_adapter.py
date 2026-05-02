"""Coupler adapter for MPAS Voronoi-mesh ocean.

Provides convenience functions to interface the MPAS ocean
(Voronoi mesh, (nCells,) shaped fields) with the coupler system.

The existing coupler (``make_coupler``, ``blend_tiles``, etc.) is
shape-agnostic — it performs element-wise JAX operations that work
with any array shape. This module provides:

1. Initialization helpers for MPAS-shaped surface state
2. A ``make_mpas_coupler`` factory that wraps the standard coupler
3. Velocity reconstruction from edge normals to cell centers
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupling_fields import AtmToSurface, SurfaceToAtm
from legoesm.ocean.freshwater import FreshwaterForcing, freshwater_from_coupler


def make_mpas_tile_config(
    mesh,
    land_mask=None,
    lake_fraction=None,
) -> TileConfig:
    """Create TileConfig from Voronoi mesh data.

    Parameters
    ----------
    mesh : VoronoiMesh
    land_mask : jax.Array or None, shape (nCells,)
        Land fraction [0-1]. If None, all ocean.
    lake_fraction : jax.Array or None, shape (nCells,)
        Lake fraction [0-1]. If None, no lakes.

    Returns
    -------
    TileConfig
    """
    nCells = mesh.nCells
    f_land = land_mask if land_mask is not None else jnp.zeros(nCells)
    f_lake = lake_fraction if lake_fraction is not None else jnp.zeros(nCells)
    return TileConfig(f_land=f_land, f_lake=f_lake)


def init_mpas_surface_state(
    nCells: int,
    T_soil_init: float = 280.0,
    W_bucket_init: float = 75.0,
    T_epi_init: float = 285.0,
    T_hypo_init: float = 278.0,
    T_ice_init: float = 260.0,
    land_config=None,
):
    """Initialize surface state for MPAS-shaped fields.

    Wraps the standard ``init_surface_state`` with ``shape=(nCells,)``.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.
    T_soil_init, W_bucket_init, T_epi_init, T_hypo_init, T_ice_init : float
        Initial temperatures and moisture.
    land_config : LandConfig or None

    Returns
    -------
    SurfaceState
    """
    from legoesm.coupler.coupler import init_surface_state

    return init_surface_state(
        shape=(nCells,),
        T_soil_init=T_soil_init,
        W_bucket_init=W_bucket_init,
        T_epi_init=T_epi_init,
        T_hypo_init=T_hypo_init,
        T_ice_init=T_ice_init,
        land_config=land_config,
    )


def make_mpas_coupler(
    coupler_config: CouplerConfig,
    land_config=None,
    ice_config=None,
    lake_config=None,
    lat=None,
):
    """Create a coupler step function for MPAS-shaped fields.

    Wraps the standard ``make_coupler`` — the underlying coupler is
    shape-agnostic, so this simply provides default configs and
    passes latitude as (nCells,) array.

    Parameters
    ----------
    coupler_config : CouplerConfig
    land_config : LandConfig or None
    ice_config : SeaIceConfig or None
    lake_config : LakeConfig or None
    lat : jax.Array or None, shape (nCells,)
        Cell-center latitudes [rad]. Needed for carbon cycle.

    Returns
    -------
    step_surface : callable
        ``(sfc_state, atm_forcing, tile_config, ocean_sst, ocean_u, ocean_v, dt, doy)``
        → ``(SurfaceState, SurfaceToAtm)``
    """
    from legoesm.coupler.coupler import make_coupler
    from legoesm.land.config import LandConfig
    from legoesm.ice.config import SeaIceConfig
    from legoesm.coupler.coupler import LakeConfig

    if land_config is None:
        land_config = LandConfig()
    if ice_config is None:
        ice_config = SeaIceConfig()
    if lake_config is None:
        lake_config = LakeConfig()

    return make_coupler(
        coupler_config=coupler_config,
        land_config=land_config,
        ice_config=ice_config,
        lake_config=lake_config,
        lat=lat,
    )


def compute_mpas_freshwater(
    atm_forcing: AtmToSurface,
    sfc_response: SurfaceToAtm,
    ocean_mask: jnp.ndarray,
    L_v: float = constants.L_v,
    land_state=None,
    ice_state_old=None,
    ice_state_new=None,
    ice_config=None,
    dt: float = 1.0,
) -> FreshwaterForcing:
    """Compute freshwater forcing for MPAS ocean from coupler fields.

    Extracts precipitation, evaporation (from latent heat), land runoff,
    and ice melt/freeze freshwater, then applies ocean mask.

    Parameters
    ----------
    atm_forcing : AtmToSurface
        Atmospheric forcing with precip_total.
    sfc_response : SurfaceToAtm
        Surface response with lhflx.
    ocean_mask : jax.Array, shape (nCells,)
        Ocean mask (1=ocean, 0=land).
    L_v : float
        Latent heat of vaporization [J/kg].
    land_state : MultiLayerLandState or None
        Land state with runoff fields.
    ice_state_old, ice_state_new : SeaIceState or None
        Ice states before/after step for ice freshwater.
    ice_config : SeaIceConfig or None
    dt : float
        Timestep [s].

    Returns
    -------
    FreshwaterForcing
    """
    # Land runoff
    runoff_sfc = None
    runoff_sub = None
    if land_state is not None:
        if hasattr(land_state, 'runoff_surface'):
            runoff_sfc = land_state.runoff_surface
        if hasattr(land_state, 'runoff_subsurface'):
            runoff_sub = land_state.runoff_subsurface

    return freshwater_from_coupler(
        precip_total=atm_forcing.precip_total,
        lhflx=sfc_response.lhflx,
        L_v=L_v,
        runoff_surface=runoff_sfc,
        runoff_subsurface=runoff_sub,
        ice_state_old=ice_state_old,
        ice_state_new=ice_state_new,
        ice_config=ice_config,
        ocean_mask=ocean_mask,
        dt=dt,
    )
