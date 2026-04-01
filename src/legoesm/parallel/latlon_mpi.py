"""MPI-based lat-band decomposition for FV lat-lon dynamics.

Each MPI rank owns a contiguous band of latitude rows.  Halo exchange
uses ``mpi4jax.sendrecv`` to exchange north/south boundary rows between
ranks, replacing the ``jax.lax.ppermute`` used in the single-process
multi-GPU path (``latlon_sharded.py``).

Design
------
1. Latitude axis is partitioned into ``n_ranks`` equal bands.
2. Each rank holds ``lats_per = n_lat // n_ranks`` owned rows plus
   ``halo`` ghost rows on each side.
3. Before computing tendencies (at each RK stage), each rank exchanges
   ``halo`` rows with its north and south neighbors via MPI sendrecv.
4. Pole-adjacent ranks apply pole-folding BCs for their outer boundary.
5. Per-rank local grids are pre-built with correct metric terms.

This module does NOT touch the global ``_active_topology`` or
``_active_layout`` singletons in ``distributed.py``.  It provides its
own initialization, scatter, gather, and step-function APIs.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import (
    FVLatLonPrimitiveEquationConfig,
    fv_latlon_hydrostatic_tendencies,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import compute_polar_filter_mask
from legoesm.parallel.reductions import _require_mpi_stack, _mpi4jax_array_result

logger = logging.getLogger(__name__)


# ============================================================================
# Layout
# ============================================================================

class LatLonBandLayout(NamedTuple):
    """MPI layout for 1-D latitude-band decomposition.

    Each rank owns a contiguous band of ``local_n_lat`` rows starting
    at global index ``lat_start``.  Halo rows extend ``halo`` rows
    beyond the owned band on each side.
    """
    rank: int
    n_ranks: int
    global_n_lat: int
    global_n_lon: int
    local_n_lat: int       # n_lat // n_ranks  (owned rows only)
    halo: int              # ghost rows per side (typically 2)
    lat_start: int         # global index of first owned row
    lat_end: int           # global index past last owned row
    north_rank: int        # rank to the north (-1 for northernmost)
    south_rank: int        # rank to the south (-1 for southernmost)


def make_latlon_band_layout(
    rank: int,
    n_ranks: int,
    n_lat: int,
    n_lon: int,
    halo: int = 2,
) -> LatLonBandLayout:
    """Build a latitude-band layout for rank ``rank``.

    Parameters
    ----------
    rank : int
        MPI rank (0-based).
    n_ranks : int
        Total number of MPI ranks.
    n_lat, n_lon : int
        Global grid dimensions.
    halo : int
        Number of ghost rows per side (default 2 for PPM).

    Returns
    -------
    LatLonBandLayout
    """
    if n_lat % n_ranks != 0:
        raise ValueError(
            f"n_lat={n_lat} must be divisible by n_ranks={n_ranks}")

    lats_per = n_lat // n_ranks
    lat_start = rank * lats_per
    lat_end = lat_start + lats_per

    # Neighbor ranks (-1 means pole boundary)
    south_rank = rank - 1 if rank > 0 else -1
    north_rank = rank + 1 if rank < n_ranks - 1 else -1

    return LatLonBandLayout(
        rank=rank,
        n_ranks=n_ranks,
        global_n_lat=n_lat,
        global_n_lon=n_lon,
        local_n_lat=lats_per,
        halo=halo,
        lat_start=lat_start,
        lat_end=lat_end,
        north_rank=north_rank,
        south_rank=south_rank,
    )


# ============================================================================
# Local grid builder (delegates to latlon_sharded._build_local_grid)
# ============================================================================

def build_local_grid(
    global_grid: LatLonGrid,
    layout: LatLonBandLayout,
) -> LatLonGrid:
    """Build a local LatLonGrid for this rank with halo rows.

    Delegates to the shared ``_build_local_grid`` in ``latlon_sharded``.
    """
    from legoesm.parallel.latlon_sharded import _build_local_grid
    return _build_local_grid(
        global_grid, layout.rank, layout.n_ranks, layout.halo,
    )


# ============================================================================
# Halo exchange via MPI
# ============================================================================

def exchange_latlon_halos(
    owned_fields: jnp.ndarray,
    layout: LatLonBandLayout,
    sign_flip: bool = False,
) -> jnp.ndarray:
    """Exchange north/south halo rows via MPI sendrecv.

    Parameters
    ----------
    owned_fields : jax.Array, shape ``(local_n_lat, n_lon, ...)``
        Owned data for this rank (no halos).
    layout : LatLonBandLayout
    sign_flip : bool
        If True, negate pole-folded halo values (for u, v winds).

    Returns
    -------
    jax.Array, shape ``(local_n_lat + 2*halo, n_lon, ...)``
        Data with halo rows prepended and appended.
    """
    mpi4jax, MPI = _require_mpi_stack()
    comm = MPI.COMM_WORLD

    halo = layout.halo
    n_lon = layout.global_n_lon
    trailing = owned_fields.shape[2:]
    half = n_lon // 2
    TAG_BASE = 200_000

    # Boundary rows to send
    send_south = owned_fields[:halo]     # first `halo` owned rows
    send_north = owned_fields[-halo:]    # last `halo` owned rows

    recv_shape = (halo, n_lon) + trailing

    # --- South halo (receive from south neighbor) ---
    if layout.south_rank >= 0:
        # Receive north boundary from south neighbor
        recv_buf = jnp.zeros(recv_shape, dtype=owned_fields.dtype)
        south_halo = _mpi4jax_array_result(
            mpi4jax.sendrecv(
                send_south,   # send my south boundary to south neighbor
                recv_buf,
                source=layout.south_rank,
                dest=layout.south_rank,
                sendtag=TAG_BASE + layout.rank * 1000 + layout.south_rank,
                recvtag=TAG_BASE + layout.south_rank * 1000 + layout.rank,
                comm=comm,
            ),
        )
    else:
        # South pole: fold owned rows
        folded = jnp.roll(owned_fields[:halo][::-1], half, axis=1)
        south_halo = -folded if sign_flip else folded

    # --- North halo (receive from north neighbor) ---
    if layout.north_rank >= 0:
        recv_buf = jnp.zeros(recv_shape, dtype=owned_fields.dtype)
        north_halo = _mpi4jax_array_result(
            mpi4jax.sendrecv(
                send_north,   # send my north boundary to north neighbor
                recv_buf,
                source=layout.north_rank,
                dest=layout.north_rank,
                sendtag=TAG_BASE + layout.rank * 1000 + layout.north_rank,
                recvtag=TAG_BASE + layout.north_rank * 1000 + layout.rank,
                comm=comm,
            ),
        )
    else:
        # North pole: fold owned rows
        folded = jnp.roll(owned_fields[-halo:][::-1], half, axis=1)
        north_halo = -folded if sign_flip else folded

    return jnp.concatenate([south_halo, owned_fields, north_halo], axis=0)


def _exchange_state_halos(
    state: HydrostaticState,
    layout: LatLonBandLayout,
) -> HydrostaticState:
    """Exchange halos for all fields in a HydrostaticState.

    Winds (u, v) get sign-flipped at poles; scalars (T, p_s, phis) do not.
    """
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    u_h = exchange_latlon_halos(state.u.data, layout, sign_flip=True)
    v_h = exchange_latlon_halos(state.v.data, layout, sign_flip=True)
    T_h = exchange_latlon_halos(state.T.data, layout, sign_flip=False)
    ps_h = exchange_latlon_halos(state.p_s.data[:, :, None], layout, sign_flip=False)[:, :, 0]
    phis_h = exchange_latlon_halos(state.phis.data[:, :, None], layout, sign_flip=False)[:, :, 0]

    return HydrostaticState(
        u=Field(data=u_h, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_h, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_h, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=ps_h, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis_h, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ============================================================================
# Scatter / gather
# ============================================================================

def scatter_latlon(
    global_field: jnp.ndarray,
    layout: LatLonBandLayout,
) -> jnp.ndarray:
    """Extract rank-local rows from a global ``(n_lat, n_lon, ...)`` field.

    Returns owned rows only (no halos).
    """
    return global_field[layout.lat_start:layout.lat_end]


def scatter_state_latlon(
    global_state: HydrostaticState,
    layout: LatLonBandLayout,
) -> HydrostaticState:
    """Extract rank-local state from a global HydrostaticState."""
    return jax.tree.map(
        lambda f: f._replace(data=scatter_latlon(f.data, layout))
        if hasattr(f, 'data') else f,
        global_state,
    )


def gather_latlon(
    local_field: jnp.ndarray,
    layout: LatLonBandLayout,
) -> jnp.ndarray:
    """Reconstruct global ``(n_lat, n_lon, ...)`` from rank-local data.

    Uses MPI Allgather to collect owned rows from all ranks.
    """
    mpi4jax, MPI = _require_mpi_stack()
    comm = MPI.COMM_WORLD

    # Allgather: each rank contributes its owned rows
    all_data = _mpi4jax_array_result(
        mpi4jax.allgather(local_field, comm=comm),
    )
    # all_data shape: (n_ranks, local_n_lat, n_lon, ...)
    # Reshape to (n_lat, n_lon, ...)
    return all_data.reshape(
        (layout.global_n_lat,) + local_field.shape[1:])


def gather_state_latlon(
    local_state: HydrostaticState,
    layout: LatLonBandLayout,
) -> HydrostaticState:
    """Reconstruct global HydrostaticState from rank-local data."""
    return jax.tree.map(
        lambda f: f._replace(data=gather_latlon(f.data, layout))
        if hasattr(f, 'data') else f,
        local_state,
    )


# ============================================================================
# Initialization
# ============================================================================

def initialize_latlon_mpi(
    n_lat: int,
    n_lon: int,
    halo: int = 2,
) -> tuple[int, int, LatLonBandLayout]:
    """Initialize MPI for lat-lon band decomposition.

    Returns (rank, n_ranks, layout).

    Does NOT call ``jax.distributed.initialize()`` or modify any global
    state in ``distributed.py``.  Uses mpi4jax for all communication.
    """
    _require_mpi_stack()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    layout = make_latlon_band_layout(rank, n_ranks, n_lat, n_lon, halo)

    logger.info(
        "Lat-lon MPI init: rank=%d/%d, lat_band=[%d:%d), halo=%d",
        rank, n_ranks, layout.lat_start, layout.lat_end, halo,
    )
    return rank, n_ranks, layout


# ============================================================================
# MPI-aware step function
# ============================================================================

def make_latlon_mpi_step(
    model,
    global_grid: LatLonGrid,
    layout: LatLonBandLayout,
    sigma_coord,
    config: FVLatLonPrimitiveEquationConfig | None = None,
):
    """Build an MPI-parallel step function for FV lat-lon dynamics.

    The returned function ``step(state, dt) -> state`` operates on
    rank-local state (owned rows only, no halos).  At each RK stage,
    it exchanges halos via MPI before computing tendencies.

    Parameters
    ----------
    model : FVLatLonPrimitiveEquationModel
    global_grid : LatLonGrid
        Full global grid (needed for metric terms and polar filter).
    layout : LatLonBandLayout
    sigma_coord : SigmaCoordinate
    config : FVLatLonPrimitiveEquationConfig, optional
        Uses model.config if not provided.

    Returns
    -------
    callable : ``(state, dt) -> state``
        JIT-compiled step operating on rank-local HydrostaticState.
    """
    if config is None:
        config = model.config

    local_grid = build_local_grid(global_grid, layout)
    halo = layout.halo
    lats_per = layout.local_n_lat

    # Polar filter mask for the local grid
    polar_mask = None
    if config.use_polar_filter:
        polar_mask = compute_polar_filter_mask(
            local_grid, 600.0, config.polar_filter_max_wave_speed,
            config.polar_filter_cutoff_deg,
        )

    # Global area for mass conservation (computed via MPI reduction)
    from legoesm.parallel.reductions import global_sum_mpi

    def _local_tendency(u, v, T, ps, phis, dt_val):
        """Halo exchange + tendency for the local band.

        Takes owned-only arrays, exchanges halos, computes tendencies
        on the padded local grid, returns owned-only tendencies.
        """
        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")

        # Build owned-only state for halo exchange
        owned_state = HydrostaticState(
            u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=ps, name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
        )

        # Exchange halos to get padded state
        padded_state = _exchange_state_halos(owned_state, layout)

        # Compute tendencies on padded local grid
        tend = fv_latlon_hydrostatic_tendencies(
            padded_state, local_grid, sigma_coord, config,
            physics_tendency=None, polar_mask=polar_mask,
        )

        # Extract owned portion (strip halos)
        h = halo
        return (
            tend.du_dt.data[h:h + lats_per],
            tend.dv_dt.data[h:h + lats_per],
            tend.dT_dt.data[h:h + lats_per],
            tend.dp_s_dt.data[h:h + lats_per],
        )

    # SSP-RK3 with per-stage halo exchange
    @jax.jit
    def _step(state, dt):
        u = state.u.data
        v = state.v.data
        T = state.T.data
        ps = state.p_s.data
        phis = state.phis.data

        # Stage 1
        du1, dv1, dT1, dps1 = _local_tendency(u, v, T, ps, phis, dt)
        u1 = u + dt * du1
        v1 = v + dt * dv1
        T1 = T + dt * dT1
        ps1 = ps + dt * dps1

        # Stage 2
        du2, dv2, dT2, dps2 = _local_tendency(u1, v1, T1, ps1, phis, dt)
        u2 = 0.75 * u + 0.25 * (u1 + dt * du2)
        v2 = 0.75 * v + 0.25 * (v1 + dt * dv2)
        T2 = 0.75 * T + 0.25 * (T1 + dt * dT2)
        ps2 = 0.75 * ps + 0.25 * (ps1 + dt * dps2)

        # Stage 3
        du3, dv3, dT3, dps3 = _local_tendency(u2, v2, T2, ps2, phis, dt)
        u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * du3)
        v_new = (1.0 / 3.0) * v + (2.0 / 3.0) * (v2 + dt * dv3)
        T_new = (1.0 / 3.0) * T + (2.0 / 3.0) * (T2 + dt * dT3)
        ps_new = (1.0 / 3.0) * ps + (2.0 / 3.0) * (ps2 + dt * dps3)

        # Floors
        if config.T_min > 0:
            T_new = jnp.maximum(T_new, config.T_min)
        if config.p_floor > 0:
            ps_new = jnp.maximum(ps_new, config.p_floor)

        # Mass conservation via global MPI reduction
        if config.use_conservation_fixer and config.fix_mass:
            # Use owned area only for this rank's contribution
            owned_area = global_grid.area[
                layout.lat_start:layout.lat_end]
            mass_old = global_sum_mpi(jnp.sum(ps * owned_area))
            mass_new = global_sum_mpi(jnp.sum(ps_new * owned_area))
            total_area = global_sum_mpi(jnp.sum(owned_area))
            ps_new = ps_new + (mass_old - mass_new) / total_area

        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")
        return HydrostaticState(
            u=Field(data=u_new, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=v_new, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T_new, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=ps_new, name="p_s", dims=dims_2d, units="Pa"),
            phis=state.phis,
        )

    logger.info(
        "Lat-lon MPI step ready: rank=%d/%d, %d lats/rank, halo=%d",
        layout.rank, layout.n_ranks, lats_per, halo,
    )
    return _step
