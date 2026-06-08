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

    All fields have shape (nCells,) and units kg/m²/s.
    Positive = freshwater into ocean, except evaporation which is
    positive upward (i.e., freshwater leaving ocean).

    Fields
    ------
    precip : jax.Array
        Precipitation rate [kg/m²/s].
    evap : jax.Array
        Evaporation rate [kg/m²/s], positive upward.
    runoff : jax.Array
        Land runoff rate [kg/m²/s].
    ice_fw : jax.Array
        Ice melt/freeze freshwater [kg/m²/s], positive = melt.
    restoring : jax.Array
        OMIP-2 SSS-restoring virtual freshwater flux [kg/m²/s,
        positive INTO ocean].  Computed from
        :func:`legoesm.ocean.forcing.sss_restoring.compute_sss_restoring_flux`.
        Zero by default for backward compatibility with the
        legacy 4-component constructor.
    """
    precip: jnp.ndarray
    evap: jnp.ndarray
    runoff: jnp.ndarray
    ice_fw: jnp.ndarray
    restoring: jnp.ndarray = None  # type: ignore[assignment]


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
    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z, restoring=z)


def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
    """Compute net freshwater flux into ocean [kg/m²/s].

    F_fw = P - E + R + M + R_restore

    where P=precip, E=evaporation (positive up), R=runoff,
    M=ice melt, R_restore=SSS-restoring virtual FW flux (zero
    when ``restoring`` field is None or absent — legacy
    callers built without the SSS-restoring extension).

    Parameters
    ----------
    fw : FreshwaterForcing

    Returns
    -------
    jax.Array, shape (nCells,)
        Net freshwater flux [kg/m²/s], positive into ocean.
    """
    base = fw.precip - fw.evap + fw.runoff + fw.ice_fw
    # ``restoring is None`` is a Python (trace-time) check — safe
    # under JIT because the field is structural metadata.
    if fw.restoring is None:
        return base
    return base + fw.restoring


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
    return virtual_salt_flux_from_net(
        net_freshwater_flux(fw), S_ref, dz_0, rho_0)


def virtual_salt_flux_from_net(
    F_fw: jnp.ndarray,
    S_ref: float,
    dz_0: jnp.ndarray,
    rho_0: float,
) -> jnp.ndarray:
    """Top-layer virtual-salt tendency [PSU/s] from a PRECOMPUTED net freshwater
    flux ``F_fw`` [kg/m²/s, +INTO ocean] (vs :func:`virtual_salt_flux`, which
    builds the net from a ``FreshwaterForcing``).  Lets a caller normalize the
    net (e.g. :func:`normalize_freshwater_net`) before the closure.

        dS/dt = -S_ref * F_fw / (rho_0 * dz_0)
    """
    # Guard thin cells: on partial-cell grids, dz_0 can be O(cm) at
    # shallow coastal cells.  Dividing by tiny dz produces huge dS/dt.
    # Zero the tendency where dz_0 < 1mm (same guard as prescribed
    # surface forcing's is_ocean threshold).
    is_wet = dz_0 > 1.0e-3
    dz_safe = jnp.maximum(dz_0, 1.0e-3)
    return jnp.where(is_wet, -S_ref * F_fw / (rho_0 * dz_safe), 0.0)


def normalize_freshwater_net(
    F_fw: jnp.ndarray,
    area: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Remove the ocean-area-weighted global mean of a freshwater flux.

    Returns ``F_fw - mean(F_fw)`` so the area integral over the ``mask`` cells is
    exactly zero.  Applied to the virtual-salt closure this CONSERVES GLOBAL SALT:
    the salt-mass tendency per area is ``-S_ref * F_fw * 1e-3`` (the top-layer
    thickness cancels), so ``∮(-S_ref*(F_fw - F_mean)*area) = 0`` (Griffies: a
    redistributive surface freshwater flux changes no salt mass).  Using the SAME
    area-mean removal that the free-surface (eta) path applies keeps volume and
    salt normalization consistent.

    The caller must pass the EFFECTIVE WET mask (ocean cells the salt flux
    actually touches), so the mean removal matches the applied flux exactly --
    ``apply_freshwater_virtual_salt_top`` uses ``mask * (h_top > 1e-3)``.

    NOTE: single-device local ``jnp.sum`` (matching the eta normalization in
    ``ocean_model_mpas.step``).  An MPI-sharded run would need ``global_sum_mpi``
    over both numerator and denominator -- a shared pre-existing caveat.
    """
    w = area * mask
    F_mean = jnp.sum(F_fw * w) / jnp.maximum(jnp.sum(w), 1.0e-10)
    return F_fw - F_mean * mask


def salt_flux_salinity_tendency(salt_flux, dz_0, rho_0: float):
    """Top-layer salinity tendency from a REAL salt-mass flux [PSU/s].

    A salt mass flux ``F_salt`` [kg(salt)/m²/s, positive INTO the ocean] adds
    salt to the top layer of thickness ``dz_0``:

        d(S * 1e-3 * rho_0 * dz_0)/dt = F_salt   (PSU = g/kg -> kg/kg via 1e-3)
        => dS/dt = F_salt * 1e3 / (rho_0 * dz_0)

    Inverse of the ``sss_restoring`` convention
    (``salt_flux = rho_0 * dz * dS/dt * 1e-3``).  This is the REAL-salt channel
    (e.g. sea-ice brine rejection); distinct from ``virtual_salt_flux`` (the
    freshwater dilution proxy).  Same thin-cell guard (dz_0 < 1 mm -> 0).

    Parameters
    ----------
    salt_flux : array
        Salt-mass flux into the ocean [kg(salt)/m²/s].
    dz_0 : array
        Top-layer thickness [m].
    rho_0 : float
        Reference seawater density [kg/m³].

    Returns
    -------
    array
        Top-layer salinity tendency [PSU/s].
    """
    is_wet = dz_0 > 1.0e-3
    dz_safe = jnp.maximum(dz_0, 1.0e-3)
    return jnp.where(is_wet, salt_flux * 1.0e3 / (rho_0 * dz_safe), 0.0)


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
    surface_mass_flux: jnp.ndarray | None = None,
) -> FreshwaterForcing:
    """Compute freshwater forcing from coupler fields.

    Parameters
    ----------
    precip_total : jax.Array, shape (nCells,)
        Total precipitation [kg/m2/s].
    lhflx : jax.Array, shape (nCells,)
        Latent heat flux [W/m2], positive upward.  Used only as the
        fallback when ``surface_mass_flux`` is None — see audit F22.
    L_v : float
        Latent heat of vaporization [J/kg].  Fallback factor for
        ``lhflx``-based evap; ignored when ``surface_mass_flux`` is
        provided.
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
    surface_mass_flux : jax.Array or None, shape (nCells,)
        Phase-aware moisture mass flux from SurfaceToAtm
        [kg/m²/s, positive up].  Added in iter-16 of the
        Physical_Consistency cycle (audit F3).  Preferred over the
        ``lhflx / L_v`` back-derivation because the latter
        under-counts mass by ~13 % on sublimating tiles.  When
        provided, this overrides the lhflx-based evap calculation.

    Returns
    -------
    FreshwaterForcing
    """
    nCells = precip_total.shape[0]

    # Precipitation over ocean
    precip = precip_total

    # Evaporation: prefer the phase-aware surface_mass_flux when
    # available, fall back to lhflx / L_v for legacy callers.
    if surface_mass_flux is not None:
        evap = surface_mass_flux
    else:
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
        restoring=jnp.zeros_like(precip),
    )


def with_sss_restoring(
    fw: FreshwaterForcing,
    *,
    S_model_top: jnp.ndarray,
    S_target: jnp.ndarray,
    lat_deg: jnp.ndarray,
    lon_deg: jnp.ndarray,
    ice_concentration: jnp.ndarray,
    restoring_config,
) -> FreshwaterForcing:
    """Augment a ``FreshwaterForcing`` with the OMIP-2 SSS restoring FW flux.

    The restoring is added to the dedicated ``restoring`` field so
    ``net_freshwater_flux`` automatically picks it up.  Per-cell
    salt-mass exchange is delegated to the standard virtual-salt
    convention via ``virtual_salt_flux``: the restoring contribution
    is salt-conserving by construction because it acts on the FW
    side only.

    When ``restoring_config.enabled`` is False this is a no-op and
    the original ``fw`` is returned unchanged.

    Parameters
    ----------
    fw : FreshwaterForcing
        Existing freshwater forcing (precip - evap + runoff + ice_fw).
    S_model_top : array
        Top-layer model salinity [PSU] on the ocean's per-cell grid.
    S_target : array
        Target SSS climatology [PSU] interpolated onto the same grid.
    lat_deg, lon_deg : array
        Cell-centred latitude and longitude [°].
    ice_concentration : array
        Cell ice fraction in [0, 1] used to gate restoring under ice.
    restoring_config : SSSRestoringConfig
        Configuration for region masks, piston velocity, ice gating.

    Returns
    -------
    FreshwaterForcing
        Same forcing with ``restoring`` populated.
    """
    # Local import to avoid a cycle with ocean.forcing → ocean.freshwater.
    from legoesm.ocean.forcing.sss_restoring import compute_sss_restoring_flux

    if not restoring_config.enabled:
        return fw

    out = compute_sss_restoring_flux(
        S_model_top=S_model_top,
        S_target=S_target,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        ice_concentration=ice_concentration,
        config=restoring_config,
    )
    # Combine with any pre-existing ``restoring`` component (e.g.
    # multiple restoring channels stacked).
    prior = fw.restoring if fw.restoring is not None else jnp.zeros_like(out["freshwater_flux"])
    return fw._replace(restoring=prior + out["freshwater_flux"])
