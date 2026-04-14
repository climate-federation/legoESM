"""MPI domain decomposition for lat-lon C-grid dynamical core.

Implements latitude-band decomposition: each MPI rank owns a contiguous
band of latitude rows.  All ranks own all longitudes (no decomposition
in the periodic direction).

Usage
-----
::

    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout,
        scatter_state_latlon,
        make_latlon_mpi_step,
    )

    layout = make_latlon_band_layout(rank, n_ranks, n_lat, n_lon)
    local_state = scatter_state_latlon(global_state, layout)
    step_fn = make_latlon_mpi_step(model, grid, layout, sigma, config)

    for _ in range(n_steps):
        local_state = step_fn(local_state, dt)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class LatLonBandLayout(NamedTuple):
    """Latitude-band decomposition layout for MPI.

    Attributes
    ----------
    rank : int
        This process's MPI rank.
    n_ranks : int
        Total MPI processes.
    n_lat_global : int
        Total latitude rows (global).
    n_lon_global : int
        Total longitude columns (global).
    n_lat_local : int
        Latitude rows owned by this rank (interior, excluding halos).
    lat_start : int
        Global index of the first owned latitude row.
    lat_end : int
        Global index one past the last owned latitude row.
    south_rank : int or None
        MPI rank of the southern neighbor, or None if at south pole.
    north_rank : int or None
        MPI rank of the northern neighbor, or None if at north pole.
    """
    rank: int
    n_ranks: int
    n_lat_global: int
    n_lon_global: int
    n_lat_local: int
    lat_start: int
    lat_end: int
    south_rank: int | None
    north_rank: int | None


def make_latlon_band_layout(
    rank: int,
    n_ranks: int,
    n_lat: int,
    n_lon: int,
) -> LatLonBandLayout:
    """Create a latitude-band decomposition layout.

    Divides ``n_lat`` rows as evenly as possible across ``n_ranks``.
    Remainder rows are distributed to the first few ranks.

    Parameters
    ----------
    rank : int
        This process's MPI rank.
    n_ranks : int
        Total MPI processes.
    n_lat : int
        Number of latitude rows (global).
    n_lon : int
        Number of longitude columns (global).

    Returns
    -------
    LatLonBandLayout
    """
    base = n_lat // n_ranks
    remainder = n_lat % n_ranks

    # Ranks [0, remainder) get one extra row.
    if rank < remainder:
        n_local = base + 1
        lat_start = rank * (base + 1)
    else:
        n_local = base
        lat_start = remainder * (base + 1) + (rank - remainder) * base

    lat_end = lat_start + n_local
    south_rank = rank - 1 if rank > 0 else None
    north_rank = rank + 1 if rank < n_ranks - 1 else None

    return LatLonBandLayout(
        rank=rank,
        n_ranks=n_ranks,
        n_lat_global=n_lat,
        n_lon_global=n_lon,
        n_lat_local=n_local,
        lat_start=lat_start,
        lat_end=lat_end,
        south_rank=south_rank,
        north_rank=north_rank,
    )


def _exchange_halo_latlon(
    field: jax.Array,
    layout: LatLonBandLayout,
    is_vector_v: bool = False,
) -> jax.Array:
    """Exchange 1-cell latitude halo between neighboring ranks.

    Parameters
    ----------
    field : jax.Array, shape (n_lat_local, n_lon, ...)
        Rank-local field (interior rows only).
    layout : LatLonBandLayout
    is_vector_v : bool
        If True, apply sign reversal at polar fold (for v-velocity).

    Returns
    -------
    jax.Array, shape (n_lat_local + 2, n_lon, ...)
        Padded field with 1 halo row on each side.
    """
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "Lat-lon MPI halo exchange requires mpi4jax and mpi4py."
        ) from exc

    from legoesm.parallel.halo_exchange import _get_sendrecv_vjp

    comm = MPI.COMM_WORLD
    rank = layout.rank
    n_lon = layout.n_lon_global
    trailing = field.shape[2:]  # (...) after (n_lat_local, n_lon)

    # Pad with zeros on south and north.
    padded = jnp.pad(field, [(1, 1)] + [(0, 0)] * (field.ndim - 1))

    sendrecv = _get_sendrecv_vjp(mpi4jax)

    # --- South halo ---
    if layout.south_rank is not None:
        # Send my southernmost row to my southern neighbor, receive their
        # northernmost row as my south halo.
        send_buf = field[0].reshape(-1)
        recv_template = jnp.zeros_like(send_buf)
        recv_buf = sendrecv(
            send_buf, recv_template,
            layout.south_rank, layout.south_rank,
            rank, layout.south_rank, comm,
        )
        padded = padded.at[0].set(recv_buf.reshape(field[0].shape))
    else:
        # South pole: fold (reflect with 180° lon shift).
        south_row = field[0]
        folded = jnp.roll(south_row, n_lon // 2, axis=0)
        if is_vector_v:
            folded = -folded
        padded = padded.at[0].set(folded)

    # --- North halo ---
    if layout.north_rank is not None:
        send_buf = field[-1].reshape(-1)
        recv_template = jnp.zeros_like(send_buf)
        recv_buf = sendrecv(
            send_buf, recv_template,
            layout.north_rank, layout.north_rank,
            rank, layout.north_rank, comm,
        )
        padded = padded.at[-1].set(recv_buf.reshape(field[-1].shape))
    else:
        # North pole: fold (reflect with 180° lon shift).
        north_row = field[-1]
        folded = jnp.roll(north_row, n_lon // 2, axis=0)
        if is_vector_v:
            folded = -folded
        padded = padded.at[-1].set(folded)

    return padded


def scatter_state_latlon(state, layout: LatLonBandLayout):
    """Extract rank-local band from a global lat-lon state.

    Parameters
    ----------
    state : CGridLatLonHydrostaticState
        Global state with fields shaped (n_lat, n_lon, ...) etc.
    layout : LatLonBandLayout

    Returns
    -------
    CGridLatLonHydrostaticState
        Rank-local state with latitude dimension = n_lat_local.
    """
    s, e = layout.lat_start, layout.lat_end

    # Scalar fields: (n_lat, n_lon, ...) → (n_lat_local, n_lon, ...)
    T_local = state.T[s:e]
    p_s_local = state.p_s[s:e]
    phis_local = state.phis[s:e]

    # C-grid u: (n_lat, n_lon+1, ...) → (n_lat_local, n_lon+1, ...)
    u_local = state.u[s:e]

    # C-grid v: (n_lat+1, n_lon, ...) → (n_lat_local+1, n_lon, ...)
    # v is at lat INTERFACES, so we need one extra row.
    v_local = state.v[s:e + 1]

    # Tracers
    tracers_local = {}
    if hasattr(state, 'tracers') and state.tracers:
        for name, tr in state.tracers.items():
            tracers_local[name] = tr[s:e]

    return state._replace(
        u=u_local,
        v=v_local,
        T=T_local,
        p_s=p_s_local,
        phis=phis_local,
        tracers=tracers_local if tracers_local else state.tracers,
    )


def gather_state_latlon(local_state, layout: LatLonBandLayout):
    """Gather rank-local bands into a global state on rank 0.

    Parameters
    ----------
    local_state : CGridLatLonHydrostaticState
        Rank-local state.
    layout : LatLonBandLayout

    Returns
    -------
    CGridLatLonHydrostaticState or None
        Global state on rank 0, None on other ranks.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return local_state

    comm = MPI.COMM_WORLD

    def _gather_field(local_field):
        """Gather a field along the latitude axis."""
        gathered = comm.gather(local_field, root=0)
        if layout.rank == 0:
            return jnp.concatenate(gathered, axis=0)
        return None

    T_global = _gather_field(local_state.T)
    p_s_global = _gather_field(local_state.p_s)
    phis_global = _gather_field(local_state.phis)
    u_global = _gather_field(local_state.u)
    # v needs special handling: each rank has n_lat_local+1 rows,
    # overlapping by 1. Take rank's [0:n_lat_local] rows, last rank
    # adds the final row.
    if layout.rank < layout.n_ranks - 1:
        v_to_send = local_state.v[:-1]
    else:
        v_to_send = local_state.v
    v_global = _gather_field(v_to_send)

    if layout.rank == 0:
        return local_state._replace(
            u=u_global, v=v_global, T=T_global,
            p_s=p_s_global, phis=phis_global,
        )
    return None


def _pad_state_halos(state, layout: LatLonBandLayout):
    """Add 1-cell latitude halos to a rank-local C-grid state.

    Each scalar field (T, p_s, phis) and u get 1 halo row on south
    and north via MPI sendrecv with pole-folding at boundaries.
    v (at lat interfaces) gets 1 halo row on each side as well.

    Parameters
    ----------
    state : CGridLatLonHydrostaticState
        Rank-local state (interior rows only).
    layout : LatLonBandLayout

    Returns
    -------
    CGridLatLonHydrostaticState
        State with shape (n_lat_local + 2, ...) for all fields.
    """
    T_pad = _exchange_halo_latlon(state.T, layout, is_vector_v=False)
    p_s_pad = _exchange_halo_latlon(state.p_s, layout, is_vector_v=False)
    phis_pad = _exchange_halo_latlon(state.phis, layout, is_vector_v=False)
    u_pad = _exchange_halo_latlon(state.u, layout, is_vector_v=False)
    v_pad = _exchange_halo_latlon(state.v, layout, is_vector_v=True)

    tracers_pad = {}
    if hasattr(state, 'tracers') and state.tracers:
        for name, tr in state.tracers.items():
            tracers_pad[name] = _exchange_halo_latlon(tr, layout, is_vector_v=False)

    return state._replace(
        u=u_pad, v=v_pad, T=T_pad, p_s=p_s_pad, phis=phis_pad,
        tracers=tracers_pad if tracers_pad else state.tracers,
    )


def _strip_halos(state, layout: LatLonBandLayout):
    """Remove 1-cell latitude halos from a padded C-grid state.

    Parameters
    ----------
    state : CGridLatLonHydrostaticState
        Padded state (n_lat_local + 2, ...).
    layout : LatLonBandLayout

    Returns
    -------
    CGridLatLonHydrostaticState
        Interior-only state (n_lat_local, ...).
    """
    def _strip(field):
        return field[1:-1]

    tracers_stripped = {}
    if hasattr(state, 'tracers') and state.tracers:
        for name, tr in state.tracers.items():
            tracers_stripped[name] = _strip(tr)

    return state._replace(
        u=_strip(state.u),
        v=_strip(state.v),
        T=_strip(state.T),
        p_s=_strip(state.p_s),
        phis=_strip(state.phis),
        tracers=tracers_stripped if tracers_stripped else state.tracers,
    )


def _build_padded_grid(grid, layout: LatLonBandLayout):
    """Build a sub-grid for the padded domain (interior + 1-cell halos).

    Slices the global grid's latitude-dependent arrays to cover
    ``[lat_start-1, lat_end+1)`` (clamped and edge-padded at poles),
    giving ``n_lat_local + 2`` rows.  All LatLonGrid fields are updated
    consistently so operators see correct metrics.
    """
    s, e = layout.lat_start, layout.lat_end
    s_pad = max(s - 1, 0)
    e_pad = min(e + 1, layout.n_lat_global)

    def _slice_1d(arr):
        sliced = arr[s_pad:e_pad]
        ps = 1 if s == 0 else 0
        pn = 1 if e == layout.n_lat_global else 0
        if ps + pn > 0:
            sliced = jnp.pad(sliced, (ps, pn), mode='edge')
        return sliced

    def _slice_2d(arr):
        sliced = arr[s_pad:e_pad]
        ps = 1 if s == 0 else 0
        pn = 1 if e == layout.n_lat_global else 0
        if ps + pn > 0:
            sliced = jnp.pad(sliced, [(ps, pn), (0, 0)], mode='edge')
        return sliced

    area_pad = _slice_2d(grid.area)
    return grid._replace(
        n_lat=layout.n_lat_local + 2,
        lat=_slice_1d(grid.lat),
        lat2d=_slice_2d(grid.lat2d),
        lon2d=_slice_2d(grid.lon2d),
        cos_lat=_slice_1d(grid.cos_lat),
        sin_lat=_slice_1d(grid.sin_lat),
        f=_slice_2d(grid.f),
        dx=_slice_2d(grid.dx),
        area=area_pad,
        total_area=jnp.sum(area_pad),
    )


def make_latlon_mpi_step(
    model,
    grid,
    layout: LatLonBandLayout,
    sigma,
    config,
) -> Callable:
    """Create an MPI-aware step function for the lat-lon C-grid dycore.

    Not yet implemented.  True lat-lon MPI scaling requires adapting
    the C-grid operators (gradient, divergence, Coriolis, polar filter)
    to work on latitude sub-domains with halo exchange, which is a
    significant refactor.  The infrastructure for domain decomposition
    (``LatLonBandLayout``, ``scatter_state_latlon``,
    ``gather_state_latlon``, ``_exchange_halo_latlon``) is in place;
    what remains is wiring halo exchange into each operator's boundary
    treatment and the model's conservation/polar-filter logic.

    Raises
    ------
    NotImplementedError
        Always.  Use single-rank execution for lat-lon benchmarks,
        or cubed-sphere / icosahedral grids for MPI scaling.
    """
    raise NotImplementedError(
        "Lat-lon MPI local-compute stepping is not yet implemented. "
        "The domain decomposition infrastructure (LatLonBandLayout, "
        "scatter/gather, halo exchange) exists, but the C-grid "
        "operators and model internals (polar filter, mass fixer) "
        "require adaptation for latitude sub-domains.  Run lat-lon "
        "benchmarks with a single rank, or use cubed-sphere or "
        "icosahedral grids for MPI scaling tests."
    )
