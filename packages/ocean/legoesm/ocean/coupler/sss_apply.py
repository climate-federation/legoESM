"""Apply OMIP-2 SSS restoring to the ocean state at every timestep.

Bridges the standalone restoring kernel
(``legoesm.ocean.forcing.sss_restoring.compute_sss_restoring_flux``)
with the per-step ocean update.  Convention follows the virtual-
salt formulation:

    dS_top/dt = − (S_top − S_target) / τ_eff

with ``τ_eff`` carrying the region masks + ice gating.  The
helper applies the discrete step

    S_top_new = S_top_old + dt · dS_top/dt

clipped to the land mask so dry cells are untouched.

Currently supports the lat-lon C-grid ocean state
(``LatLonCGridOceanState``).  Other grids may use the standalone
``compute_sss_restoring_flux`` directly and assemble the dS
themselves.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.ocean.forcing.sss_restoring import (
    SSSRestoringConfig,
    compute_sss_restoring_flux,
)


def apply_sss_restoring_step(
    state,
    *,
    S_target: np.ndarray | jnp.ndarray,
    ice_concentration: np.ndarray | jnp.ndarray | None,
    config: SSSRestoringConfig,
    grid,
    z_coord,
    dt: float,
    lat2d_deg: np.ndarray | None = None,
    lon2d_deg: np.ndarray | None = None,
    river_runoff: np.ndarray | jnp.ndarray | None = None,
) -> object:
    """Apply one timestep of OMIP-2 SSS restoring to ``state``.

    Updates the surface-layer salinity via

        S_top_new = S_top_old + dt · dS/dt|_restore

    where ``dS/dt|_restore`` comes from
    :func:`compute_sss_restoring_flux`.  Land cells (``land_mask=0``)
    are left untouched.

    Parameters
    ----------
    state : LatLonCGridOceanState
        Current ocean state (lat-lon C-grid).  ``state.S`` is a
        ``Field`` with shape ``(n_lat, n_lon, nlev)``.
    S_target : array ``(n_lat, n_lon)``
        Climatological target SSS interpolated to the grid [PSU].
    ice_concentration : array ``(n_lat, n_lon)`` or None
        Cell ice fraction in [0, 1].  When ``None``, restoring is
        applied everywhere without ice gating — appropriate for
        ocean-only experiments without a sea-ice tile.
    config : SSSRestoringConfig
        Restoring configuration (region masks, τ, ice gate,
        flux cap).
    grid : LatLonGrid-like
        Provides ``grid.lat`` and ``grid.lon`` (radians) for the
        region-mask builder.
    z_coord : OceanZStarCoordinate / OceanPartialCellCoordinate
        Used to read ``dz_ref[0]`` for the surface-layer thickness.
    dt : float
        Time step [s].

    Returns
    -------
    new_state : LatLonCGridOceanState
        Same state with the top-layer salinity updated.
    """
    if not config.enabled:
        return state

    # Surface salinity from the existing state.  Slice FIRST (device-side),
    # THEN convert: np.asarray on the full leaf would assemble the entire
    # 3-D (possibly lat-band-sharded) S field on the host every step (codex
    # batch4 HIGH); only the 2-D surface layer crosses.  Value-identical —
    # slicing commutes with the elementwise f64 upcast.
    S_top = np.asarray(state.S.data[..., 0], dtype=np.float64)

    # Region masks need the TRUE per-cell lat/lon.  On a CURVILINEAR grid
    # (tripole) ``grid.lat``/``grid.lon`` are 1-D row-mean / first-row
    # approximations, so the OMIP regional tau masks (Nordic/Labrador/Arctic)
    # would be misplaced in the folded north.  Prefer the caller-supplied 2-D
    # degree coordinates (the driver computes the authoritative ones per grid);
    # fall back to the 1-D broadcast only for a regular lat-lon grid.
    if lat2d_deg is not None and lon2d_deg is not None:
        _la = np.asarray(lat2d_deg, dtype=np.float64)
        _lo = np.asarray(lon2d_deg, dtype=np.float64)
        # Normalise semantic 1-D inputs so latitude varies by ROW and longitude
        # by COLUMN (a bare (n_lat,) would otherwise broadcast as the trailing
        # axis and mis-place the region masks on a square n_lat==n_lon grid).
        if _la.ndim == 1:
            _la = _la[:, None]
        if _lo.ndim == 1:
            _lo = _lo[None, :]
        lat2d = np.broadcast_to(_la, S_top.shape)
        lon2d = np.broadcast_to(_lo, S_top.shape)
    else:
        lat_deg = np.degrees(np.asarray(grid.lat))
        lon_deg = np.degrees(np.asarray(grid.lon))
        lat2d = np.broadcast_to(lat_deg[:, None], S_top.shape)
        lon2d = np.broadcast_to(lon_deg[None, :], S_top.shape)

    if ice_concentration is None:
        ice = np.zeros_like(S_top)
    else:
        ice = np.asarray(ice_concentration, dtype=np.float64)

    out = compute_sss_restoring_flux(
        S_model_top=jnp.asarray(S_top),
        S_target=jnp.asarray(S_target),
        lat_deg=jnp.asarray(lat2d),
        lon_deg=jnp.asarray(lon2d),
        ice_concentration=jnp.asarray(ice),
        config=config,
        river_runoff=(None if river_runoff is None
                      else jnp.asarray(np.asarray(river_runoff,
                                                  dtype=np.float64))),
    )

    dS_dt = np.asarray(out["dS_dt_top"], dtype=np.float64)
    land_mask = np.asarray(state.land_mask.data, dtype=np.float64)

    S_top_new = S_top + dt * dS_dt * land_mask

    # Surface-only write-back: scatter the updated 2-D layer into the leaf
    # DEVICE-SIDE instead of round-tripping the full 3-D field through the
    # host (codex batch4 HIGH).  Deep layers keep the original device buffer
    # (bit-identical — the old full round trip re-uploaded them unchanged;
    # for f32 leaves the old f32->f64->f32 detour was exact).  The set()
    # casts the f64 surface update to the leaf dtype exactly like the old
    # in-place numpy assign, and PRESERVES the leaf dtype.  A plain-numpy
    # host state (no ``.at``) uploads once — same cost as before.
    S_dev = state.S.data
    if not hasattr(S_dev, "at"):
        S_dev = jnp.asarray(S_dev)
    return state._replace(
        S=Field(
            S_dev.at[..., 0].set(jnp.asarray(S_top_new)),
            name=state.S.name,
            dims=state.S.dims,
            units=state.S.units,
        ),
    )


def apply_sss_restoring_step_mpas(
    state,
    *,
    S_target: np.ndarray | jnp.ndarray,
    ice_concentration: np.ndarray | jnp.ndarray | None,
    config: SSSRestoringConfig,
    mesh,
    dt: float,
    river_runoff: np.ndarray | jnp.ndarray | None = None,
) -> object:
    """Apply one timestep of OMIP-2 SSS restoring on a Voronoi mesh.

    Counterpart of :func:`apply_sss_restoring_step` for MPAS-style
    ocean states where:
        * ``state.S.data`` has shape ``(nCells, nlev)``;
        * ``mesh.latCell`` / ``mesh.lonCell`` are 1-D ``(nCells,)``
          arrays (radians).

    Parameters
    ----------
    state : MPAS-style ocean state
        ``state.S`` (``(nCells, nlev)``) and ``state.land_mask``
        (``(nCells,)``) are read; ``S`` is updated in-place via the
        NamedTuple ``_replace`` API.
    S_target : array ``(nCells,)``
        Climatological target SSS interpolated to the mesh [PSU].
    ice_concentration : array ``(nCells,)`` or None
        Cell ice fraction in [0, 1].  None ⇒ no ice gating.
    config : SSSRestoringConfig
    mesh : VoronoiMesh
    dt : float

    Returns
    -------
    new_state : same type as input
        With the surface salinity layer updated.
    """
    if not config.enabled:
        return state

    # Slice FIRST (device-side), THEN convert — same host-transfer contract
    # as the lat-lon variant above: only the (nCells,) surface layer crosses.
    S_top = np.asarray(state.S.data[..., 0], dtype=np.float64)   # (nCells,)

    lat_deg = np.degrees(np.asarray(mesh.latCell))           # (nCells,)
    lon_deg = np.degrees(np.asarray(mesh.lonCell))           # (nCells,)

    if ice_concentration is None:
        ice = np.zeros_like(S_top)
    else:
        ice = np.asarray(ice_concentration, dtype=np.float64)

    out = compute_sss_restoring_flux(
        S_model_top=jnp.asarray(S_top),
        S_target=jnp.asarray(S_target),
        lat_deg=jnp.asarray(lat_deg),
        lon_deg=jnp.asarray(lon_deg),
        ice_concentration=jnp.asarray(ice),
        config=config,
        river_runoff=(None if river_runoff is None
                      else jnp.asarray(np.asarray(river_runoff,
                                                  dtype=np.float64))),
    )
    dS_dt = np.asarray(out["dS_dt_top"], dtype=np.float64)
    land_mask = np.asarray(state.land_mask.data, dtype=np.float64)

    S_top_new = S_top + dt * dS_dt * land_mask

    # Surface-only device-side write-back (see the lat-lon variant above):
    # no full-3-D host round trip; deep layers keep the original buffer.
    S_dev = state.S.data
    if not hasattr(S_dev, "at"):
        S_dev = jnp.asarray(S_dev)
    return state._replace(
        S=Field(
            S_dev.at[..., 0].set(jnp.asarray(S_top_new)),
            name=state.S.name,
            dims=state.S.dims,
            units=state.S.units,
        ),
    )
