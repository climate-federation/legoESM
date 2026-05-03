"""Freshwater forcing for the MPAS ocean model.

Handles precipitation, evaporation, land runoff, and ice melt/freeze
freshwater fluxes. Applies virtual salt flux to salinity and real
freshwater mass flux to the free surface.

Conventions
-----------
- All fluxes in kg/m2/s (mass flux per unit area).
- Positive = freshwater entering ocean (precip, runoff, ice melt).
- Evaporation is positive upward in the coupler, so E enters here
  as a positive value that *removes* freshwater from the ocean.

References
----------
- Griffies, S. M. (2004). Fundamentals of Ocean Climate Models, Ch. 12.
- Large, W. G. et al. (1997). J. Phys. Oceanogr., 27(11), 2418-2447.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class FreshwaterForcing(NamedTuple):
    """Freshwater fluxes applied to the ocean surface.

    All fields have shape (nCells,) and units kg/m2/s.
    Positive = freshwater into ocean, except evaporation which is
    positive upward (i.e., freshwater leaving ocean).

    Fields
    ------
    precip : jax.Array
        Precipitation rate [kg/m2/s].
    evap : jax.Array
        Evaporation rate [kg/m2/s], positive upward.
    runoff : jax.Array
        Land runoff rate [kg/m2/s].
    ice_fw : jax.Array
        Ice melt/freeze freshwater [kg/m2/s], positive = melt.
    """
    precip: jnp.ndarray
    evap: jnp.ndarray
    runoff: jnp.ndarray
    ice_fw: jnp.ndarray


def zero_freshwater(nCells: int) -> FreshwaterForcing:
    """Create zero freshwater forcing.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.

    Returns
    -------
    FreshwaterForcing
    """
    # Init helper: keep at the JAX default float dtype.  Callers running
    # under a non-default precision policy can ``cast_pytree`` the
    # result to match their state.
    z = jnp.zeros(nCells)
    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z)


def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
    """Compute net freshwater flux into ocean [kg/m2/s].

    F_fw = P - E + R + M

    where P=precip, E=evaporation (positive up), R=runoff, M=ice melt.

    Parameters
    ----------
    fw : FreshwaterForcing

    Returns
    -------
    jax.Array, shape (nCells,)
        Net freshwater flux [kg/m2/s], positive into ocean.
    """
    return fw.precip - fw.evap + fw.runoff + fw.ice_fw


def freshwater_eta_tendency(fw: FreshwaterForcing, rho_0: float) -> jnp.ndarray:
    """Compute free-surface tendency from freshwater flux.

    deta/dt = F_fw / rho_0

    Parameters
    ----------
    fw : FreshwaterForcing
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Free-surface tendency [m/s].
    """
    return net_freshwater_flux(fw) / rho_0


def virtual_salt_flux(
    fw: FreshwaterForcing,
    S_ref: float,
    dz_0: jnp.ndarray,
    rho_0: float,
) -> jnp.ndarray:
    """Compute virtual salt flux for the top ocean layer.

    dS/dt = -S_ref * F_fw / (rho_0 * dz_0)

    This approximation maintains volume while adjusting salinity
    to account for freshwater dilution/concentration.

    Parameters
    ----------
    fw : FreshwaterForcing
    S_ref : float
        Reference salinity [PSU].
    dz_0 : jax.Array, shape (nCells,)
        Top layer thickness [m].
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Salinity tendency [PSU/s] for top layer.
    """
    F_fw = net_freshwater_flux(fw)
    dz_safe = jnp.maximum(dz_0, 1e-10)
    return -S_ref * F_fw / (rho_0 * dz_safe)


def freshwater_from_coupler(
    precip_total: jnp.ndarray,
    lhflx: jnp.ndarray,
    L_v: float,
    runoff_surface: jnp.ndarray | None = None,
    runoff_subsurface: jnp.ndarray | None = None,
    ice_state_old=None,
    ice_state_new=None,
    ice_config=None,
    ocean_mask: jnp.ndarray | None = None,
    dt: float = 1.0,
) -> FreshwaterForcing:
    """Compute freshwater forcing from coupler fields.

    Parameters
    ----------
    precip_total : jax.Array, shape (nCells,)
        Total precipitation [kg/m2/s].
    lhflx : jax.Array, shape (nCells,)
        Latent heat flux [W/m2], positive upward.
    L_v : float
        Latent heat of vaporization [J/kg].
    runoff_surface : jax.Array or None, shape (nCells,)
        Surface runoff from land [kg/m2/s].
    runoff_subsurface : jax.Array or None, shape (nCells,)
        Subsurface runoff from land [kg/m2/s].
    ice_state_old, ice_state_new : SeaIceState or None
        Ice state before/after ice step. Used to compute ice freshwater.
    ice_config : SeaIceConfig or None
        Ice config with rho_ice.
    ocean_mask : jax.Array or None, shape (nCells,)
        Ocean mask (1=ocean). Used to restrict fluxes to ocean cells.
    dt : float
        Timestep [s]. Used for ice thickness change rate.

    Returns
    -------
    FreshwaterForcing
    """
    nCells = precip_total.shape[0]

    # Precipitation over ocean
    precip = precip_total

    # Evaporation from latent heat flux: E = lhflx / L_v
    evap = lhflx / L_v

    # Land runoff (sum surface + subsurface).  Pin the zero-fallback
    # dtype to the precip path so a missing runoff input does not
    # silently widen the freshwater forcing struct to f64 under x64.
    # Both fields are summed independently (codex adversarial review,
    # iter-1, bug #3) — previously, ``runoff_subsurface`` was silently
    # dropped whenever ``runoff_surface`` was ``None``.
    runoff = jnp.zeros(nCells, dtype=precip.dtype)
    if runoff_surface is not None:
        runoff = runoff + runoff_surface
    if runoff_subsurface is not None:
        runoff = runoff + runoff_subsurface

    # Ice freshwater: based on areal ice mass change.
    # ice_mass = rho_ice * h * A  (per unit area of grid cell)
    # ice_fw = -(ice_mass_new - ice_mass_old) / dt
    # Melting (mass decrease) puts freshwater into ocean (positive fw).
    # Supports both single-category and multi-category ice.
    if ice_state_old is not None and ice_state_new is not None and ice_config is not None:
        h_old = ice_state_old.h_ice.data
        h_new = ice_state_new.h_ice.data
        A_old = ice_state_old.concentration.data
        A_new = ice_state_new.concentration.data
        rho_ice = ice_config.rho_ice
        ice_mass_old = rho_ice * h_old * A_old
        ice_mass_new = rho_ice * h_new * A_new
        # Multi-category: h has more dims than precip_total; sum categories.
        # Stack the two ice-mass arrays once and reduce the trailing axes
        # together — each loop pass becomes one ``jnp.sum`` instead of two.
        n_extra = ice_mass_old.ndim - precip_total.ndim
        if n_extra > 0:
            _ice_pair = jnp.stack([ice_mass_old, ice_mass_new], axis=-1)
            for _ in range(n_extra):
                _ice_pair = jnp.sum(_ice_pair, axis=-2)
            ice_mass_old, ice_mass_new = _ice_pair[..., 0], _ice_pair[..., 1]
        ice_fw = -(ice_mass_new - ice_mass_old) / jnp.maximum(dt, 1e-10)
    else:
        ice_fw = jnp.zeros(nCells, dtype=precip.dtype)

    # Mask to ocean cells
    if ocean_mask is not None:
        precip = precip * ocean_mask
        evap = evap * ocean_mask
        runoff = runoff * ocean_mask
        ice_fw = ice_fw * ocean_mask

    return FreshwaterForcing(
        precip=precip,
        evap=evap,
        runoff=runoff,
        ice_fw=ice_fw,
    )
