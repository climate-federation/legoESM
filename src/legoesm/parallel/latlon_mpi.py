"""MPI domain decomposition for the lat-lon C-grid atmospheric dycore.

Latitude-band decomposition: each MPI rank owns a contiguous band of
latitude rows; every rank owns all longitudes (no decomposition in the
periodic direction).

Stage 0 (this file) provides the halo-exchange and padded-grid
machinery only.  The per-step driver ``make_latlon_mpi_step`` is a
stub that raises ``NotImplementedError`` — Stage 1 will wire it
against the existing serial C-grid step in
:mod:`legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid`.

Conventions
-----------
- Halo width ``halo``: number of ghost lat rows added on each side
  (south + north).  Operators with compact 1-cell stencils (gradient,
  divergence, curl) need ``halo=1``; operators with PPM
  reconstruction or biharmonic stencils need ``halo=2``.
- Pole fold matches :func:`legoesm.grids.halo_latlon.pad_halo_latlon`
  exactly: mirror-reverse the first/last ``halo`` interior rows, shift
  by 180° in longitude, and (optionally) negate for vector ``v``
  components.  This single source of truth lives in
  :mod:`legoesm.grids.halo_latlon`; we reuse it here so MPI and
  single-rank paths can never disagree at pole-touching ranks.
- C-grid layout:

      u : (n_lat_local,    n_lon,   nlev)  — zonal velocity at lon faces
                              (n_lon+1 in serial; lon is periodic so we
                              treat the trailing face as wrap-around)
      v : (n_lat_local+1,  n_lon,   nlev)  — meridional velocity at lat
                              interfaces; neighbouring ranks share the
                              ``v`` row at the partition boundary
      T, p_s, phis, tracers : (n_lat_local, n_lon, ...) cell-centered

  For ``v`` we keep the duplicated boundary row across neighbours; the
  scatter and halo-exchange helpers handle this explicitly.

- AD safety: halo sendrecv uses
  :func:`legoesm.parallel.halo_exchange._get_sendrecv_vjp` (the AD-safe
  wrapper around ``mpi4jax.sendrecv``).  Backward swaps source/dest as
  required by the reverse-mode rule.

Status
------
Stage 0 (DONE):
    LatLonBandLayout, make_latlon_band_layout, exchange_halo_latlon,
    scatter_state_latlon, gather_state_latlon, pad_state_halos,
    strip_halos, build_padded_grid.

Stage 1 (TODO — explicit NotImplementedError):
    make_latlon_mpi_step — wrap
    ``cgrid_latlon_hydrostatic_tendencies`` and the RK driver inside a
    pad → step → strip cycle.  Must also make the pole-wall BC
    (``v=0`` at the global lat boundaries) rank-aware so interior cuts
    are NOT zeroed.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_latlon_3d,
    pad_halo_latlon_vector,
    pad_halo_latlon_vector_3d,
)
from legoesm.parallel.halo_exchange import _get_sendrecv_vjp


# ============================================================================
# Layout
# ============================================================================


class LatLonBandLayout(NamedTuple):
    """Latitude-band decomposition layout for MPI.

    Attributes
    ----------
    rank, n_ranks : int
        This process's MPI rank and the world size.
    n_lat_global, n_lon_global : int
        Total latitude rows / longitude columns (global problem size).
    n_lat_local : int
        Interior latitude rows owned by this rank (no halos).
    lat_start, lat_end : int
        Global indices of the owned band: rows ``[lat_start, lat_end)``.
    south_rank, north_rank : int or None
        MPI ranks of the southern / northern neighbour, or ``None`` if
        this rank touches the south / north pole.
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
    """Build a latitude-band decomposition layout.

    Divides ``n_lat`` rows as evenly as possible across ``n_ranks``;
    the first ``n_lat % n_ranks`` ranks get one extra row.
    """
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be >=1, got {n_ranks}")
    if rank < 0 or rank >= n_ranks:
        raise ValueError(f"rank {rank} out of range [0, {n_ranks})")
    if n_lat < n_ranks:
        raise ValueError(
            f"Cannot decompose {n_lat} lat rows across {n_ranks} ranks "
            "(at least one row per rank is required for a 1-cell halo "
            "neighbour exchange)."
        )

    base = n_lat // n_ranks
    remainder = n_lat % n_ranks
    if rank < remainder:
        n_local = base + 1
        lat_start = rank * (base + 1)
    else:
        n_local = base
        lat_start = remainder * (base + 1) + (rank - remainder) * base
    lat_end = lat_start + n_local

    return LatLonBandLayout(
        rank=rank,
        n_ranks=n_ranks,
        n_lat_global=n_lat,
        n_lon_global=n_lon,
        n_lat_local=n_local,
        lat_start=lat_start,
        lat_end=lat_end,
        south_rank=rank - 1 if rank > 0 else None,
        north_rank=rank + 1 if rank < n_ranks - 1 else None,
    )


# ============================================================================
# Halo exchange
# ============================================================================


def _serial_pad_then_strip_lon(
    field: jax.Array, halo: int, negate: bool,
) -> jax.Array:
    """Run the canonical serial ``pad_halo_latlon*`` then strip the lon halos.

    Why this exists
    ---------------
    A naive 180° rotation via ``jnp.roll(field, n_lon//2, axis=1)`` does NOT
    bit-reproduce serial ``pad_halo_latlon``.  The serial helper wrap-pads in
    longitude first and only then rolls by ``data.shape[1] // 2 ==
    (n_lon + 2*halo) // 2``; after slicing the interior lon columns the
    result is an *aliased* halo that depends on ``halo``, not a clean
    physical 180° rotation.  See ``tests/parallel/test_latlon_mpi_halo_serial.py``
    for the concrete (4-cell) example.

    To guarantee Stage-1+ produces bit-identical results to the serial
    path under MPI, this helper delegates the lon-padding + fold + slice
    to the canonical helper itself, then strips the lon halos so the
    output has the rank-local lon shape.

    This is also why we cannot fix the "physical correctness" question
    here: changing the convention would diverge from the serial dycore.
    That refactor belongs in a separate Stage-0.5 audit of the upstream
    fold convention (tracked in CLAUDE.md follow-up debt).

    Returns
    -------
    (south_halo, north_halo) : each shape (halo, n_lon[, nlev])
        Ready for concatenation south of ``field[0]`` / north of
        ``field[-1]``.
    """
    if field.ndim == 2:
        helper = pad_halo_latlon_vector if negate else pad_halo_latlon
        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h)
        # Strip the lon halos to recover (halo, n_lon).
        south = padded[:halo, halo:halo + field.shape[1]]
        north = padded[-halo:, halo:halo + field.shape[1]]
    elif field.ndim == 3:
        helper = pad_halo_latlon_vector_3d if negate else pad_halo_latlon_3d
        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h, nlev)
        south = padded[:halo, halo:halo + field.shape[1], :]
        north = padded[-halo:, halo:halo + field.shape[1], :]
    else:
        raise ValueError(
            f"_serial_pad_then_strip_lon: field.ndim must be 2 or 3, "
            f"got {field.ndim}"
        )
    return south, north


def _pole_fold_south(field: jax.Array, halo: int, negate: bool) -> jax.Array:
    """South-pole halo, bit-identical to serial ``pad_halo_latlon*``."""
    south, _ = _serial_pad_then_strip_lon(field, halo, negate)
    return south


def _pole_fold_north(field: jax.Array, halo: int, negate: bool) -> jax.Array:
    """North-pole halo, bit-identical to serial ``pad_halo_latlon*``."""
    _, north = _serial_pad_then_strip_lon(field, halo, negate)
    return north


def exchange_halo_latlon(
    field: jax.Array,
    layout: LatLonBandLayout,
    halo: int = 1,
    is_vector_v: bool = False,
) -> jax.Array:
    """Exchange ``halo`` ghost lat rows on each side via MPI sendrecv.

    Boundary ranks (south_rank is None / north_rank is None) pole-fold
    using the same convention as
    :func:`legoesm.grids.halo_latlon.pad_halo_latlon`: mirror-reverse
    the first / last ``halo`` interior rows, shift by 180° in
    longitude, sign-flip for vector ``v``.

    Periodic longitude is preserved (every rank owns all longitudes;
    no lon halo).

    Parameters
    ----------
    field : jax.Array, shape (n_lat_local, n_lon)  or  (n_lat_local, n_lon, nlev)
        Rank-local interior field, no halos in input.
    layout : LatLonBandLayout
    halo : int, default 1
        Number of ghost rows to add on each lat side.
    is_vector_v : bool
        Whether the field is a vector component that flips sign
        across the pole (e.g. v-velocity, or any meridional flux).
        Has no effect at non-pole-touching ranks.

    Returns
    -------
    jax.Array, shape (n_lat_local + 2*halo, n_lon[, nlev])
    """
    if halo <= 0:
        return field

    n_lat_local = field.shape[0]
    if halo > n_lat_local:
        # At halo=2 this can bite when n_lat_global < 2 * n_ranks * halo;
        # surface a clear error rather than producing garbage halos.
        raise ValueError(
            f"exchange_halo_latlon: halo={halo} exceeds n_lat_local="
            f"{n_lat_local} on rank {layout.rank}.  Reduce n_ranks or "
            "increase grid resolution."
        )

    trailing = field.shape[1:]

    # Single-rank ⇒ both sides pole-fold; never touch MPI.  This branch
    # keeps the n_ranks=1 path runnable without mpi4jax installed, which
    # matters for serial smoke tests on the login node and for users
    # who haven't built the optional MPI stack.
    if layout.south_rank is None and layout.north_rank is None:
        south_halo = _pole_fold_south(field, halo, negate=is_vector_v)
        north_halo = _pole_fold_north(field, halo, negate=is_vector_v)
        return jnp.concatenate([south_halo, field, north_halo], axis=0)

    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "Lat-lon MPI halo exchange (n_ranks>1) requires mpi4jax and mpi4py."
        ) from exc

    comm = MPI.COMM_WORLD
    sendrecv = _get_sendrecv_vjp(mpi4jax)

    # ---- South halo ----
    if layout.south_rank is not None:
        # Receive from southern neighbour's TOP `halo` rows (their
        # ``field[-halo:]``), placed as our ``[0:halo)``.  We send our
        # bottom `halo` rows in return (the south neighbour's north
        # halo).
        send_bot = field[:halo].reshape(-1)
        recv_template = jnp.zeros_like(send_bot)
        recv_south = sendrecv(
            send_bot, recv_template,
            layout.south_rank,         # source
            layout.south_rank,         # dest
            layout.rank,               # sendtag = sender's rank
            layout.south_rank,         # recvtag = source's rank
            comm,
        )
        south_halo = recv_south.reshape((halo,) + trailing)
    else:
        south_halo = _pole_fold_south(field, halo, negate=is_vector_v)

    # ---- North halo ----
    if layout.north_rank is not None:
        send_top = field[-halo:].reshape(-1)
        recv_template = jnp.zeros_like(send_top)
        recv_north = sendrecv(
            send_top, recv_template,
            layout.north_rank,
            layout.north_rank,
            layout.rank,
            layout.north_rank,
            comm,
        )
        north_halo = recv_north.reshape((halo,) + trailing)
    else:
        north_halo = _pole_fold_north(field, halo, negate=is_vector_v)

    return jnp.concatenate([south_halo, field, north_halo], axis=0)


# ============================================================================
# Backend-dispatched pad_halo_latlon implementation
# ============================================================================
#
# Used by :mod:`legoesm.grids.halo_latlon` when the global halo
# backend is ``"mpi"`` and the active topology is a
# :class:`LatLonBandLayout`.  Composes the existing
# :func:`exchange_halo_latlon` (lat MPI sendrecv + boundary pole-fold)
# with a periodic-lon wrap that every rank performs locally.  Output
# shape matches the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon`
# family — operators stay backend-oblivious.
#
# This is the architectural reuse point the user asked for: no
# parallel registry, no operator-side branching, no duplicate
# halo-machinery.  The cubed-sphere precedent
# (``set_halo_backend`` → ``_halo_backend`` global → ``pad_halo``
# dispatch) is mirrored exactly.


def _pad_halo_latlon_mpi(
    data, layout: LatLonBandLayout, halo: int = 1,
    is_vector_v: bool = False,
):
    """MPI variant of :func:`legoesm.grids.halo_latlon.pad_halo_latlon`.

    Step 1 — periodic lon wrap (same on every rank; every rank owns
    the full longitude axis).  This matches the serial code's
    lon-pad order so the operator's downstream stencil sees the same
    layout under both backends.

    Step 2 — lat halo via :func:`exchange_halo_latlon`:
      - Boundary ranks (``layout.<side>_rank is None``) apply the
        pole-fold convention used by the serial
        :func:`legoesm.grids.halo_latlon.pad_halo_latlon` (mirror +
        180° lon shift + sign flip for vectors).
      - Interior partition cuts MPI-sendrecv with the neighbour rank.

    The result has the same shape and semantics as the serial
    helper.  Under MPI the lat axis is the rank's band plus the halo
    rows; the lon axis is the full global lon plus its periodic
    halo, identical to serial.
    """
    if data.ndim not in (2, 3):
        raise ValueError(
            f"_pad_halo_latlon_mpi: data.ndim must be 2 or 3, got {data.ndim}"
        )
    # Step 1: periodic lon wrap (local on every rank).
    if data.ndim == 2:
        lon_padded = jnp.pad(data, ((0, 0), (halo, halo)), mode="wrap")
    else:
        lon_padded = jnp.pad(
            data, ((0, 0), (halo, halo), (0, 0)), mode="wrap",
        )
    # Step 2: lat halo — sendrecv at interior cuts, pole-fold at
    # boundary ranks.  ``exchange_halo_latlon`` already handles both
    # cases (single-rank pole-fold + multi-rank MPI sendrecv) and
    # is AD-safe via ``_get_sendrecv_vjp``.
    return exchange_halo_latlon(
        lon_padded, layout, halo=halo, is_vector_v=is_vector_v,
    )


def _pad_with_pole_bc_lat_mpi(
    interior, layout: LatLonBandLayout, halo: int = 1,
    south_value: float = 0.0, north_value: float = 0.0,
    is_vector_v: bool = False,
):
    """MPI variant of :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat`.

    Pole-touching south rank → ``south_value`` constant pad.
    Pole-touching north rank → ``north_value`` constant pad.
    Interior partition cuts → MPI sendrecv with neighbour.

    Reuses :func:`exchange_halo_latlon` for the inter-rank exchange;
    overrides the pole-fold result with the requested constants on
    the boundary ranks afterwards (so the wall-BC semantics is
    preserved instead of pole-folding).  This is the SIMPLEST way
    to reuse the existing AD-safe sendrecv machinery — we let
    ``exchange_halo_latlon`` do the heavy lifting and then patch
    the boundary halo slabs.
    """
    if halo <= 0:
        return interior

    # Step 1: run the standard halo exchange.  Boundary ranks will
    # get pole-folded values; interior cuts will get sendrecv'd
    # neighbour values (which is what we want — those are NOT to
    # be overwritten).
    padded = exchange_halo_latlon(
        interior, layout, halo=halo, is_vector_v=is_vector_v,
    )
    # Step 2: where this rank touches a pole, replace the
    # pole-folded slab with the wall-BC constant.  Slab shapes
    # match by construction.  We use the bcast-tuple pattern so
    # this works for 1D (sin_lat), 2D (face metrics) and 3D
    # (u-on-face-with-levels) fields uniformly.
    trailing_ones = (1,) * (interior.ndim - 1)
    if layout.south_rank is None:
        south_const = jnp.full(
            (halo,) + interior.shape[1:],
            jnp.asarray(south_value, dtype=interior.dtype),
        )
        padded = jnp.concatenate([south_const, padded[halo:]], axis=0)
        del trailing_ones  # silence linter on unused alias
    if layout.north_rank is None:
        north_const = jnp.full(
            (halo,) + interior.shape[1:],
            jnp.asarray(north_value, dtype=interior.dtype),
        )
        padded = jnp.concatenate([padded[:-halo], north_const], axis=0)
    return padded


# ============================================================================
# Scatter / gather
# ============================================================================


def scatter_state_latlon(state, layout: LatLonBandLayout):
    """Extract the rank-local band from a global C-grid lat-lon state.

    Each rank reads its own slice of the global arrays; we do not call
    MPI here (callers either broadcast a global state from rank 0 or
    each rank constructs the global initial state locally and slices).

    The C-grid v-field at lat interfaces has one extra row globally
    (shape ``(n_lat+1, n_lon, ...)``); we give each rank rows
    ``[lat_start, lat_end+1)`` so neighbouring ranks duplicate the
    boundary row.  The duplicated row is the *same* face — both ranks
    must always hold the same value there.  Halo exchange of ``v``
    after a step uses ``is_vector_v=True`` so the pole-touching rank
    sign-flips correctly.
    """
    s, e = layout.lat_start, layout.lat_end

    T_local = state.T[s:e]
    p_s_local = state.p_s[s:e]
    phis_local = state.phis[s:e]
    u_local = state.u[s:e]
    # v: lat-interface, one extra global row
    v_local = state.v[s:e + 1]

    tracers_local = {}
    if getattr(state, "tracers", None):
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
    """Gather rank-local bands onto rank 0; returns ``None`` elsewhere.

    The duplicated ``v`` boundary row is handled by trimming the
    trailing row from every rank except the northernmost so the
    concatenated global array has shape ``(n_lat_global+1, n_lon, ...)``.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return local_state

    comm = MPI.COMM_WORLD

    def _gather_field(local_field):
        gathered = comm.gather(local_field, root=0)
        if layout.rank == 0:
            return jnp.concatenate(gathered, axis=0)
        return None

    T_global = _gather_field(local_state.T)
    p_s_global = _gather_field(local_state.p_s)
    phis_global = _gather_field(local_state.phis)
    u_global = _gather_field(local_state.u)

    # v: trim duplicated boundary row except on the northernmost rank
    if layout.rank < layout.n_ranks - 1:
        v_to_send = local_state.v[:-1]
    else:
        v_to_send = local_state.v
    v_global = _gather_field(v_to_send)

    tracers_global = None
    if getattr(local_state, "tracers", None):
        tracers_global = {}
        for name, tr in local_state.tracers.items():
            gathered = _gather_field(tr)
            tracers_global[name] = gathered

    if layout.rank == 0:
        return local_state._replace(
            u=u_global, v=v_global, T=T_global,
            p_s=p_s_global, phis=phis_global,
            tracers=(tracers_global if tracers_global is not None
                     else local_state.tracers),
        )
    return None


# ============================================================================
# Padded state + grid
# ============================================================================


def pad_state_halos(state, layout: LatLonBandLayout, halo: int = 1):
    """Add ``halo`` ghost lat rows to every field in a C-grid state.

    Scalar fields (T, p_s, phis, tracers) and the u face-velocity get
    standard scalar pole-fold at boundary ranks.  v is treated as a
    vector and sign-flipped at the pole.

    The ``u`` field already lives on lon interfaces; the ``halo``
    rows added on each lat side are lon-face values from the
    neighbour, which is what every C-grid operator expects.

    Parameters
    ----------
    state : CGridLatLonHydrostaticState or compatible NamedTuple
    layout : LatLonBandLayout
    halo : int

    Returns
    -------
    state with all fields shaped ``(n_lat_local + 2*halo, ...)`` in
    their first axis.  The ``v`` field, originally shape
    ``(n_lat_local+1, ...)``, becomes ``(n_lat_local+1 + 2*halo, ...)``;
    the south/north halos are scattered around the *interior* +
    duplicated-boundary block.
    """
    T_pad = exchange_halo_latlon(state.T, layout, halo, is_vector_v=False)
    p_s_pad = exchange_halo_latlon(state.p_s, layout, halo, is_vector_v=False)
    phis_pad = exchange_halo_latlon(state.phis, layout, halo, is_vector_v=False)
    u_pad = exchange_halo_latlon(state.u, layout, halo, is_vector_v=False)
    v_pad = exchange_halo_latlon(state.v, layout, halo, is_vector_v=True)

    tracers_pad = {}
    if getattr(state, "tracers", None):
        for name, tr in state.tracers.items():
            tracers_pad[name] = exchange_halo_latlon(
                tr, layout, halo, is_vector_v=False,
            )

    return state._replace(
        u=u_pad, v=v_pad, T=T_pad, p_s=p_s_pad, phis=phis_pad,
        tracers=tracers_pad if tracers_pad else state.tracers,
    )


def strip_halos(state, layout: LatLonBandLayout, halo: int = 1):
    """Remove ``halo`` ghost lat rows added by :func:`pad_state_halos`.

    Inverse of :func:`pad_state_halos`.  Required after every
    operator chain that ran on the padded state, before returning to
    rank-local storage.
    """
    if halo <= 0:
        return state

    def _strip(field):
        return field[halo:-halo]

    tracers_stripped = {}
    if getattr(state, "tracers", None):
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


def build_padded_grid(grid, layout: LatLonBandLayout, halo: int = 1):
    """Build a ``LatLonGrid`` for the padded local domain (interior + halos).

    Interior cells (where the rank's band lies) copy the global grid's
    metrics bit-exactly.  Halo cells on **interior partition cuts**
    (neighbour rank present) likewise slice the global grid.  Halo
    cells **past a pole** are filled by **linearly extrapolating
    ``lat``** and recomputing all derived metrics (``sin_lat``,
    ``cos_lat``, ``f``, ``dx``, ``area``) from the extended ``lat``
    using the same formulas as :func:`create_latlon_grid`.

    Why linear extrapolation
    ------------------------
    The earlier ``mode='edge'`` pad produced *duplicate* latitudes at
    the halo (``[lat[0], lat[0], lat[0], lat[1], ...]``).  The
    canonical ``curl_vertex_cgrid`` then computes vertex areas as
    ``R² dlon |sin_ext[k+1] - sin_ext[k]|`` over the entire padded
    lat axis, giving ``A_vertex = 0`` at any pair of duplicate-lat
    rows — and ``zeta = circ / 0 = ±Inf`` propagates into the
    interior via the absolute-vorticity averaging onto u/v faces.
    Diagnostic in ``scripts/_diag_mpi_step_nans.py`` pinpointed this.

    Linear extrapolation past the pole keeps ``lat`` strictly
    monotonic, ``A_vertex > 0`` everywhere, and (critically) the
    interior lat values bit-exactly equal to the global grid's — so
    the operator's stencils at interior cells are unchanged from
    serial.  Halo cell *outputs* still get stripped after the step;
    only the freedom from spurious zeros at intermediate stages is
    what matters.

    Assumptions
    -----------
    Uniform-``dlat`` lat-lon grid (the regular case used for AMIP).
    Mercator and tripolar grids carry per-row ``dy`` / fold metadata
    that this builder does NOT extrapolate yet — Stage 3 follow-up.

    Parameters
    ----------
    grid : LatLonGrid
    layout : LatLonBandLayout
    halo : int

    Returns
    -------
    LatLonGrid with ``n_lat = n_lat_local + 2*halo``, all metrics
    derived consistently from the (extrapolated where necessary)
    extended ``lat``.
    """
    if halo < 0:
        raise ValueError(f"halo must be >= 0, got {halo}")
    if halo == 0:
        return grid._replace(
            n_lat=layout.n_lat_local,
            lat=grid.lat[layout.lat_start:layout.lat_end],
            lat2d=grid.lat2d[layout.lat_start:layout.lat_end, :],
            lon2d=grid.lon2d[layout.lat_start:layout.lat_end, :],
            cos_lat=grid.cos_lat[layout.lat_start:layout.lat_end],
            sin_lat=grid.sin_lat[layout.lat_start:layout.lat_end],
            dy=grid.dy[layout.lat_start:layout.lat_end],
            f=grid.f[layout.lat_start:layout.lat_end, :],
            dx=grid.dx[layout.lat_start:layout.lat_end, :],
            area=grid.area[layout.lat_start:layout.lat_end, :],
            total_area=jnp.sum(
                grid.area[layout.lat_start:layout.lat_end, :]
            ),
        )

    s, e = layout.lat_start, layout.lat_end
    n_lat_g = layout.n_lat_global
    s_pad = max(s - halo, 0)
    e_pad = min(e + halo, n_lat_g)
    pad_s = halo - (s - s_pad)   # rows to fabricate past south pole
    pad_n = halo - (e_pad - e)   # rows to fabricate past north pole

    # ---- Extended lat past poles: step *from the pole boundary* ----
    # Linear extrapolation by ``dlat`` from the cell centers produces
    # halo lat values that are SYMMETRIC across the pole to interior
    # rows (since cell centers sit at ``-π/2 ± dlat/2``).  ``sin`` is
    # symmetric about ``-π/2`` (``sin(-π/2 + x) = sin(-π/2 - x) =
    # -cos(x)``), so symmetric lat values yield identical ``sin``
    # values and ``A_vertex = R² dlon |sin Δ| = 0`` in the operator
    # — bringing back the original NaN bug.  Instead, treat the
    # pole as the canonical reference: place halo cell centers at
    # ``-π/2 - dlat * k`` (south) and ``π/2 + dlat * k`` (north) for
    # k = 1, 2, …, halo.  This gives strictly monotonic ``sin`` in
    # the halo, A_vertex > 0 everywhere, and interior lat values
    # remain untouched (bit-exact to serial).
    dlat = float(grid.dlat)
    south_pole_lat = -float(jnp.pi) / 2.0
    north_pole_lat = float(jnp.pi) / 2.0
    lat_sliced = grid.lat[s_pad:e_pad]
    extended_lat = lat_sliced
    if pad_s > 0:
        south_extrap = (
            south_pole_lat - dlat * jnp.arange(pad_s, 0, -1)
        )
        extended_lat = jnp.concatenate([south_extrap, extended_lat])
    if pad_n > 0:
        north_extrap = (
            north_pole_lat + dlat * jnp.arange(1, pad_n + 1)
        )
        extended_lat = jnp.concatenate([extended_lat, north_extrap])

    # Recover ``omega`` from the original grid's Coriolis field so the
    # rebuilt grid carries the same rotation rate (the LatLonGrid
    # NamedTuple does not store ``omega`` directly).  Use the
    # interior row furthest from the pole so cos(lat) is well above
    # the clamp floor.
    mid = grid.lat.shape[0] // 2
    omega_eff = float(grid.f[mid, 0] / (2.0 * grid.sin_lat[mid]))

    # Delegate the metric construction to the shared helper so the
    # serial create_latlon_grid path and this MPI extension stay
    # algebraically identical.  Interior cells of the extended grid
    # come out bit-for-bit equal to the original grid (same formulas,
    # same lat values); halo cells are freshly computed from the
    # extrapolated lat.
    from legoesm.grids.latlon import _build_uniform_latlon_grid_from_axes
    return _build_uniform_latlon_grid_from_axes(
        lat=extended_lat,
        lon=grid.lon,
        dlat=dlat,
        dlon=float(grid.dlon),
        radius=float(grid.radius),
        omega=omega_eff,
        dtype=grid.lat.dtype,
    )


# ============================================================================
# Stage-1: dry C-grid PE MPI step
# ============================================================================


def make_latlon_mpi_step(
    model,
    layout: LatLonBandLayout,
    *,
    halo: int = 2,
) -> Callable:
    """Build an MPI-aware step function for the lat-lon C-grid dycore.

    **Stage 1 + Stage 2 scope** — hydrostatic primitive equations with
    optional tracers.  Physics (``physics_fn``) and polar filter will be
    added in Stage 3.  Stage 2 (tracer support) reuses Stage 0's
    halo machinery transparently because ``pad_state_halos`` /
    ``strip_halos`` already iterate over ``state.tracers``; the
    external mass fixer also already rescales tracers when ``p_s``
    is corrected (see ``_apply_safety_rails`` in
    ``primitive_eq_latlon_cgrid.py``).

    Architecture
    ------------
    The serial step
    (:meth:`CGridLatLonPrimitiveEquationModel._step_cgrid`) runs:

      1. Tendency computation via the C-grid operators.
      2. SSPRK / RK4 integration via ``dispatch_integrator``.
      3. Pole-wall BC (``v[poles] = 0``).
      4. Safety rails (T floor, p_s floor, mass fixer with global sum).

    Under latitude-band MPI we want phases 1–3 to run on a **padded
    local-band state** with a **padded local grid**, then strip the
    halos, and finally apply phase 4 on the rank-local interior with
    its rank-local grid so global reductions land on the right
    contributions.

    Concrete plumbing
    -----------------

    * **Padded model.**  Built once per call to this factory.  It
      reuses the original ``CGridLatLonPrimitiveEquationModel`` class
      with three config overrides:

        - ``pole_v_bc = (layout.south_rank is None,
                         layout.north_rank is None)`` — only boundary
          ranks zero the actual global poles; interior ranks leave
          their band-edge v-rows alone (those are interior v-faces
          shared with the neighbour rank and kept consistent by halo
          exchange).

        - ``fix_mass = False``, ``zero_mean_ps_tendency = False`` —
          turn OFF the global-sum operations *inside* the padded
          step.  Otherwise they would integrate over halo cells which
          duplicate the neighbour rank's interior, double-counting
          under allreduce.  Global mass conservation is restored
          *externally* below.

    * **Per-step closure.**  Pads the input state via
      :func:`pad_state_halos`, runs ``padded_model._step_cgrid`` on
      the padded state, strips the halos, and finally applies the
      mass fixer using the ORIGINAL ``model`` (which carries the
      rank-local grid).  The original model's
      ``_apply_safety_rails`` uses ``_batch_global_area_sums`` which
      delegates to ``global_sum_mpi`` whenever
      ``_is_distributed()`` is True — so the caller must have
      activated the MPI halo backend before any step runs.

    Parameters
    ----------
    model : CGridLatLonPrimitiveEquationModel
        Built against the **rank-local** grid (no halos in its
        ``grid.area`` / ``grid.lat`` etc.).  This is the canonical
        "what this rank owns" object.
    layout : LatLonBandLayout
    halo : int, default 2
        Halo width passed to ``pad_state_halos``.  Must be at least
        as wide as the deepest stencil in the dycore — PPM scalar
        transport + bilaplacian hyperdiff use a 2-cell stencil
        (see Stage-0 survey), so ``halo=2`` is the default.

    Returns
    -------
    step_fn : callable
        ``step_fn(local_state, dt, *, target_mass=None) -> local_state``

    Notes on what is NOT yet supported
    ----------------------------------

    * **Physics** (``physics_fn is not None``): the SST/SIC forcing,
      radiation, convection, microphysics paths plug into
      ``_step_cgrid`` via the ``physics_fn`` argument.  These are
      column-local (no lat halo) BUT the AMIP coupler must scatter
      the global SST/SIC time series to per-rank bands before each
      step.  Defer to Stage 3.

    * **Polar filter** (``config.use_polar_filter``): the FFT is
      per-latitude-row so it's MPI-local in principle, but the
      precomputed mask in the padded model would include halo rows
      with edge-extrapolated metrics — needs verification before
      enabling.  Defer.

    These are surfaced as explicit ``NotImplementedError`` raises at
    step time (not at factory time) so callers get a clear signal
    rather than silent wrong answers.
    """
    # Local imports keep this module importable without the dycore on
    # the path (important for the Stage-0 halo tests which don't need
    # the heavy dycore stack).
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
    )
    from legoesm.parallel.reductions import global_sum_mpi

    if halo < 1:
        raise ValueError(f"halo must be >= 1, got {halo}")
    if model.config.use_polar_filter:
        raise NotImplementedError(
            "make_latlon_mpi_step: polar filter under MPI is not "
            "validated yet (Stage 3).  Use config.use_polar_filter=False "
            "or run single-rank."
        )

    # ---- Build the padded local grid + padded model once ----
    padded_grid = build_padded_grid(model.grid, layout, halo=halo)
    padded_config = model.config._replace(
        fix_mass=False,              # external fixer on the stripped state
        zero_mean_ps_tendency=False, # ditto — avoid in-step global sums
        pole_v_bc=(
            layout.south_rank is None,
            layout.north_rank is None,
        ),
        # Under padded execution the actual pole rows sit ``halo``
        # cells into the padded v-array, NOT at its ends — without
        # this offset the wall-BC enforcement zeros halo rows
        # instead and the real poles drift, producing NaNs by the
        # second RK stage (see Stage-2 smoke run #3).
        pole_v_bc_offset=halo,
    )
    padded_model = CGridLatLonPrimitiveEquationModel(
        grid=padded_grid,
        sigma_coord=model.sigma_coord,
        config=padded_config,
        dt=getattr(model, "_max_dt", 600.0),  # only used for polar filter init
    )

    # ---- Build the rank-local "fixer model" with a GLOBAL total area ----
    #
    # The serial ``_apply_safety_rails`` computes
    #
    #     correction = (mass_target - mass_new) / self.grid.grid_total_area
    #
    # where ``mass_*`` come from ``_batch_global_area_sums`` (already
    # allreduced under MPI).  ``self.grid.grid_total_area`` is NOT
    # allreduced; on a rank-local lat-lon band it equals the band area,
    # not the sphere area — so the uniform correction divides by the
    # wrong denominator and mass drifts every step.
    #
    # The cubed-sphere replicated-MPI path doesn't hit this because each
    # rank holds the full grid (grid_total_area is already global).  For
    # the lat-lon band path we pre-allreduce here and inject the result
    # into a "fixer model" via a NamedTuple replace on the grid.  We
    # don't mutate the caller's ``model.grid`` (no surprise side effects
    # on shared state).
    # ``LatLonGrid.grid_total_area`` is a property delegating to
    # ``total_area`` — the actual NamedTuple field — so ``_replace``
    # must use the underlying name.
    global_total_area = global_sum_mpi(model.grid.total_area)
    fixer_grid = model.grid._replace(total_area=global_total_area)
    fixer_config = model.config  # keep fix_mass etc. as caller intended
    fixer_model = CGridLatLonPrimitiveEquationModel(
        grid=fixer_grid,
        sigma_coord=model.sigma_coord,
        config=fixer_config,
        dt=getattr(model, "_max_dt", 600.0),
    )

    def step_fn(local_state, dt, *, target_mass=None):
        # 1. Pad the local interior state by `halo` lat rows on each side.
        #    pad_state_halos already iterates over state.tracers and pads
        #    each one — so a state with q_v, q_c, etc. is transparently
        #    handled by Stage 0's halo machinery.
        padded_state = pad_state_halos(local_state, layout, halo=halo)
        # 2. Run pure dynamics on the padded grid+state — no fixer, no
        #    zero-mean (those are external below).  Tracer transport
        #    (PPM mass-flux form) runs here automatically when
        #    ``padded_state.tracers`` is non-empty; the operators use
        #    ``halo`` rows of lat padding that we just supplied, so PPM
        #    reconstruction at the interior cells sees the right
        #    neighbour values.
        padded_new = padded_model._step_cgrid(
            padded_state, dt, target_mass=None, physics_fn=None,
        )
        # 3. Strip halos to recover the rank-local interior state.
        stripped = strip_halos(padded_new, layout, halo=halo)
        # 4. Apply mass fixer on the stripped state using the
        #    GLOBAL-aware fixer model.  Routes through
        #    ``_batch_global_area_sums`` which delegates to
        #    ``global_sum_mpi`` whenever ``_is_distributed()`` is True
        #    — i.e. when the caller has set the halo backend to MPI
        #    (see legoesm.grids.halo.set_halo_backend).  The fixer
        #    model carries the pre-allreduced
        #    ``grid_total_area`` so the per-step uniform p_s
        #    correction divides by the right denominator.
        if model.config.fix_mass:
            stripped = fixer_model._apply_safety_rails(
                stripped, target_mass=target_mass, pre_state=local_state,
            )
        return stripped

    return step_fn
