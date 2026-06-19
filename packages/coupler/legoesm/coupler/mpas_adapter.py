"""Coupler adapter for MPAS Voronoi-mesh ocean.

Provides convenience functions to interface the MPAS ocean
(Voronoi mesh, (nCells,) shaped fields) with the coupler system.

The existing coupler (``make_coupler``, ``blend_tiles``, etc.) is
shape-agnostic — it performs element-wise JAX operations that work
with any array shape. This module provides:

1. Initialization helpers for MPAS-shaped surface state
2. Velocity reconstruction from edge normals to cell centers
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.config import TileConfig
from legoesm.core.coupling_fields import AtmToSurface, SurfaceToAtm
from legoesm.coupler.coupler import init_surface_state
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
    T_soil_init: float = 280.0,  # coeff-ok: initial condition [K]
    W_bucket_init: float = 75.0,  # coeff-ok: initial condition [kg/m^2]
    T_epi_init: float = 285.0,  # coeff-ok: initial condition [K]
    T_hypo_init: float = 278.0,  # coeff-ok: initial condition [K]
    T_ice_init: float = 260.0,  # coeff-ok: initial condition [K]
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
    return init_surface_state(
        shape=(nCells,),
        T_soil_init=T_soil_init,
        W_bucket_init=W_bucket_init,
        T_epi_init=T_epi_init,
        T_hypo_init=T_hypo_init,
        T_ice_init=T_ice_init,
        land_config=land_config,
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
        Surface response with lhflx and (post-iter-16) the
        phase-aware ``surface_mass_flux`` channel.  When available,
        ``surface_mass_flux`` is used in preference to back-deriving
        evap from ``lhflx / L_v`` because the latter under-counts
        mass by ~13 % on tiles that sublimate (sea-ice / cold lakes /
        snow-covered land).  ``L_v`` is retained as a fallback for
        callers that wire a SurfaceToAtm without surface_mass_flux.
    ocean_mask : jax.Array, shape (nCells,)
        Ocean mask (1=ocean, 0=land).
    L_v : float
        Latent heat of vaporization [J/kg].  Used only as the
        fallback for back-deriving evap from lhflx when
        surface_mass_flux is not populated.
    land_state : MultiLayerLandState or LandState or None
        Land state with runoff fields.  Both multilayer
        (``runoff_surface``, ``runoff_subsurface``) and slab
        (``runoff``) state shapes are supported.  Audit F5.
    ice_state_old, ice_state_new : SeaIceState or None
        Ice states before/after step for ice freshwater.
    ice_config : SeaIceConfig or None
    dt : float
        Timestep [s].

    Returns
    -------
    FreshwaterForcing
    """
    # Land runoff: support multilayer (runoff_surface + runoff_subsurface)
    # AND slab (single ``runoff`` field).  Audit F5 caught that the
    # earlier code dropped the slab path silently.
    runoff_sfc = None
    runoff_sub = None
    if land_state is not None:
        if hasattr(land_state, 'runoff_surface'):
            runoff_sfc = land_state.runoff_surface
        if hasattr(land_state, 'runoff_subsurface'):
            runoff_sub = land_state.runoff_subsurface
        # Slab LandState exposes a single ``runoff`` field — fold it
        # into ``runoff_surface`` if multilayer fields aren't present.
        if (
            runoff_sfc is None
            and runoff_sub is None
            and hasattr(land_state, 'runoff')
            and land_state.runoff is not None
        ):
            runoff_sfc = land_state.runoff

    # Phase-aware evap mass flux (audit F22): ``surface_mass_flux`` is a
    # mandatory SurfaceToAtm field that correctly accounts for sublimation
    # over cold tiles, used in place of the L_v back-derivation.
    surface_mass_flux = sfc_response.surface_mass_flux

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
        surface_mass_flux=surface_mass_flux,
    )
