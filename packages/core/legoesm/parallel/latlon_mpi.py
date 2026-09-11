"""MPI domain decomposition for the lat-lon C-grid atmospheric dycore.

Latitude-band decomposition: each MPI rank owns a contiguous band of
latitude rows; every rank owns all longitudes (no decomposition in the
periodic direction).

This file provides the halo-exchange and padded-grid machinery
together with the per-step driver ``make_latlon_mpi_step``, which is
fully implemented: it delegates to the existing serial C-grid step in
:mod:`legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid` via the
backend-aware MPI halo path.

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

      u : zonal velocity at LON faces.  Global ``n_lon+1`` faces (the
                              trailing one the periodic closure
                              ``face n_lon == face 0``).  The 1-D band keeps
                              the full ``n_lon+1`` (lon is not split); the 2-D
                              pencil keeps ``n_lon_local+1`` faces, so W/E
                              neighbours SHARE the boundary east face — the
                              lon twin of v's shared row (see
                              ``scatter_state_latlon_2d`` / the ``is_u_face``
                              gather mode).
      v : (n_lat_local+1,  n_lon,   nlev)  — meridional velocity at lat
                              interfaces; N/S neighbouring ranks share the
                              ``v`` row at the partition boundary
      T, p_s, phis, tracers : (n_lat_local, n_lon, ...) cell-centered

  For ``v`` we keep the duplicated boundary row across neighbours; the
  scatter and halo-exchange helpers handle this explicitly.

- AD safety: halo sendrecv uses
  :func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp` (the AD-safe
  wrapper around ``mpi4jax.sendrecv``).  Backward swaps source/dest as
  required by the reverse-mode rule.

Status
------
Halo-exchange and padded-grid machinery (DONE):
    LatLonBandLayout, make_latlon_band_layout, exchange_halo_latlon,
    scatter_state_latlon, gather_state_latlon, pad_state_halos,
    strip_halos, build_padded_grid.

Per-step driver (DONE):
    make_latlon_mpi_step — wraps
    ``cgrid_latlon_hydrostatic_tendencies`` and the RK driver inside a
    pad → step → strip cycle.  The pole-wall BC (``v=0`` at the global
    lat boundaries) is rank-aware so interior cuts are NOT zeroed.
"""

from __future__ import annotations

import functools

from typing import TYPE_CHECKING, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from legoesm.grids.latlon import FoldDescriptor
    # NOTE: ocean.state.LatLonCGridOceanState is referenced only in docstrings
    # (:class: cross-refs), so it is intentionally NOT imported here — importing
    # it would make the low-level parallel layer depend UP on the ocean
    # component, which blocks component independence (see import-linter contracts).

from legoesm.grids.halo_latlon import (
    fold_pole_rows,
    fold_pole_rows_3d,
)
from legoesm.parallel.halo_exchange import get_sendrecv_vjp


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
    fold : FoldDescriptor or None
        Tripolar north-fold descriptor (issue #353).  When non-``None``
        and ``fold.is_active`` is True, the **northernmost** rank
        (``north_rank is None``) applies the permutation-based tripolar
        fold (``perm_T``/``perm_v`` + ``vector_sign_u``/``vector_sign_v``)
        at the north boundary instead of the geographic 180°-roll
        pole-fold.  Interior partition cuts and the south boundary are
        unaffected.  ``None`` (the default) ⇒ regular lat-lon /
        atmospheric pole-fold everywhere — fully backward compatible.
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
    # Trailing field with a default so all existing constructor call
    # sites (which never passed ``fold``) keep working unchanged.
    fold: "FoldDescriptor | None" = None


def wet_band_boundaries(
    wet_rows,
    n_ranks: int,
    *,
    min_rows: int = 1,
) -> tuple[int, ...]:
    """Contiguous latitude-band boundaries equalizing WET (ocean) cells.

    The default row-count split (``make_latlon_band_layout`` with no
    ``boundaries``) gives land-heavy bands the same row budget as open-ocean
    bands, so ranks whose band is mostly land idle at every collective (the
    lat-lon analogue of the Voronoi path's METIS weighting). This computes
    band boundaries from the per-ROW wet-cell counts instead: each band gets
    as close to ``total_wet / n_ranks`` wet cells as a contiguous row split
    allows.

    Pure host-side numpy — deterministic, no MPI; every rank computes the
    identical result from the identical global mask. Feed the result to
    ``make_latlon_band_layout(..., boundaries=...)`` (or
    ``initialize_distributed_latlon(band_boundaries=...)``).

    Parameters
    ----------
    wet_rows : (n_lat,) array-like
        Wet-cell count per latitude row, e.g. ``land_mask.sum(axis=1)``
        (any non-negative row weight works).
    n_ranks : int
        Number of contiguous bands.
    min_rows : int
        Minimum rows per band (halo floor — the MPI pads raise when
        ``halo > n_lat_local``; pass 2 for the halo=2 PPM/biharmonic paths).

    Returns
    -------
    boundaries : tuple of ``n_ranks + 1`` ints, ``boundaries[0] == 0``,
        ``boundaries[-1] == n_lat``, strictly increasing; band ``r`` owns rows
        ``[boundaries[r], boundaries[r+1])``.

    Algorithm: walk the cumulative wet count and cut band ``r`` at the first
    row where the running total reaches ``r / n_ranks`` of the global wet
    total, clamped so every band (including all remaining ones) keeps
    ``min_rows`` rows. All-land stretches (zero weight) attach to whichever
    band the running quantile puts them in — they cost compute-idle rows but
    carry no wet work, which is exactly the imbalance being minimized.
    """
    w = np.asarray(wet_rows, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"wet_rows must be 1-D (n_lat,), got shape {w.shape}")
    n_lat = int(w.shape[0])
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be >=1, got {n_ranks}")
    if min_rows < 1:
        raise ValueError(f"min_rows must be >=1, got {min_rows}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("wet_rows must be finite and non-negative")
    if n_lat < n_ranks * min_rows:
        raise ValueError(
            f"Cannot decompose {n_lat} lat rows across {n_ranks} ranks with "
            f"min_rows={min_rows} (need at least {n_ranks * min_rows} rows)."
        )
    total = float(w.sum())
    if total <= 0.0:
        # Degenerate all-land domain: fall back to the even row split (there
        # is no wet work to balance).
        base, rem = divmod(n_lat, n_ranks)
        sizes = [base + 1 if r < rem else base for r in range(n_ranks)]
        return tuple(int(x) for x in np.concatenate([[0], np.cumsum(sizes)]))

    cum = np.cumsum(w)
    boundaries = [0]
    for r in range(1, n_ranks):
        target = total * r / n_ranks
        # First row whose cumulative weight reaches the target; the cut is
        # AFTER that row (searchsorted on the running total).
        cut = int(np.searchsorted(cum, target, side="left")) + 1
        lo = boundaries[-1] + min_rows                 # this band keeps min_rows
        hi = n_lat - (n_ranks - r) * min_rows          # remaining bands too
        boundaries.append(int(np.clip(cut, lo, hi)))
    boundaries.append(n_lat)
    return tuple(boundaries)


def validate_band_boundaries(boundaries, n_ranks: int, n_lat: int,
                             ) -> tuple[int, ...]:
    """Validate explicit band boundaries; return them as a tuple of ints.

    Shared by :func:`make_latlon_band_layout` and the
    ``initialize_distributed_latlon`` re-init span comparison (which must
    reject invalid boundaries BEFORE deciding a stale layout may be reused —
    an invalid request must never be silently served; codex round 2).
    Checks: integral values (no silent float truncation), ``n_ranks + 1``
    entries, ``[0, n_lat]`` span, strictly increasing.
    """
    if any(int(x) != x for x in boundaries):
        raise ValueError(
            f"boundaries must be integers (a float like 1.9 would be "
            f"silently truncated), got {tuple(boundaries)}")
    b = tuple(int(x) for x in boundaries)
    if len(b) != n_ranks + 1:
        raise ValueError(
            f"boundaries must have n_ranks+1 = {n_ranks + 1} entries, "
            f"got {len(b)}")
    if b[0] != 0 or b[-1] != n_lat:
        raise ValueError(
            f"boundaries must span [0, n_lat={n_lat}], got "
            f"[{b[0]}, {b[-1]}]")
    if any(b[i + 1] <= b[i] for i in range(n_ranks)):
        raise ValueError(
            f"boundaries must be strictly increasing (every band >= 1 "
            f"row), got {b}")
    return b


def make_latlon_band_layout(
    rank: int,
    n_ranks: int,
    n_lat: int,
    n_lon: int,
    fold: "FoldDescriptor | None" = None,
    *,
    boundaries: tuple[int, ...] | None = None,
) -> LatLonBandLayout:
    """Build a latitude-band decomposition layout.

    Divides ``n_lat`` rows as evenly as possible across ``n_ranks``;
    the first ``n_lat % n_ranks`` ranks get one extra row.

    Parameters
    ----------
    fold : FoldDescriptor or None
        Tripolar north-fold descriptor (issue #353).  Carried on the
        layout so the MPI halo path can apply the permutation-based
        fold at the northernmost rank.  ``None`` ⇒ regular lat-lon.
    boundaries : tuple of ints or None
        Optional explicit band boundaries (``n_ranks + 1`` strictly
        increasing ints spanning ``[0, n_lat]``): band ``r`` owns rows
        ``[boundaries[r], boundaries[r+1])``.  Use
        :func:`wet_band_boundaries` to balance WET cells instead of row
        counts (land-heavy bands otherwise idle).  ``None`` (default)
        keeps the even row split byte-identically.  Every rank must pass
        the IDENTICAL boundaries (a deterministic host computation from
        global data).  Non-uniform bands are an MPI-band-path feature:
        the SPMD steps require uniform ``n_lat % n_devices == 0`` bands,
        and the banded-multigrid preconditioner needs even-aligned
        boundaries to coarsen (it degrades gracefully to fewer levels
        otherwise).
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

    if boundaries is not None:
        b = validate_band_boundaries(boundaries, n_ranks, n_lat)
        lat_start = b[rank]
        lat_end = b[rank + 1]
        n_local = lat_end - lat_start
    else:
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
        fold=fold,
    )


class LatLon2DLayout(NamedTuple):
    """2-D pencil (lat × lon) decomposition layout for MPI.

    Increment 2 of the lat-lon 2-D decomposition
    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Generalises
    :class:`LatLonBandLayout` from a 1-D latitude band to a 2-D
    ``(proc_lat, proc_lon)`` process grid (row-major rank =
    ``proc_row * proc_lon + proc_col``).  Latitude is a LINE (poles
    terminate it ⇒ ``south_rank``/``north_rank`` are ``None`` at the
    grid's top/bottom rows); longitude is a periodic RING (``west_rank``/
    ``east_rank`` are ALWAYS defined, wrapping ``proc_col`` modulo
    ``proc_lon``).  ``proc_lon == 1`` reproduces the 1-D band exactly
    (west==east==rank ⇒ the :func:`exchange_halo_lon` local-wrap path).

    The N/S neighbours feed :func:`exchange_halo_latlon`; the W/E
    neighbours feed :func:`exchange_halo_lon` (increment 1).
    """
    rank: int
    n_ranks: int
    proc_lat: int
    proc_lon: int
    proc_row: int
    proc_col: int
    n_lat_global: int
    n_lon_global: int
    n_lat_local: int
    n_lon_local: int
    lat_start: int
    lat_end: int
    lon_start: int
    lon_end: int
    south_rank: int | None
    north_rank: int | None
    west_rank: int
    east_rank: int
    fold: "FoldDescriptor | None" = None


def _even_split(n: int, parts: int, idx: int) -> tuple[int, int]:
    """Block ``idx`` of an even-as-possible split of ``n`` into ``parts``
    (first ``n % parts`` blocks get one extra).  Returns ``(start, len)``."""
    base, rem = n // parts, n % parts
    if idx < rem:
        return idx * (base + 1), base + 1
    return rem * (base + 1) + (idx - rem) * base, base


def make_latlon_2d_layout(
    rank: int,
    proc_lat: int,
    proc_lon: int,
    n_lat: int,
    n_lon: int,
    fold: "FoldDescriptor | None" = None,
) -> LatLon2DLayout:
    """Build a 2-D pencil decomposition layout for ``rank``.

    ``rank = proc_row * proc_lon + proc_col`` (row-major).  Latitude is
    split over ``proc_lat`` (line, pole-terminated), longitude over
    ``proc_lon`` (periodic ring).  ``proc_lat * proc_lon`` must equal the
    world size; ``proc_lon == 1`` gives the 1-D-band-equivalent layout.
    """
    if proc_lat < 1 or proc_lon < 1:
        raise ValueError(
            f"proc_lat and proc_lon must be >=1, got "
            f"proc_lat={proc_lat}, proc_lon={proc_lon}")
    n_ranks = proc_lat * proc_lon
    if rank < 0 or rank >= n_ranks:
        raise ValueError(f"rank {rank} out of range [0, {n_ranks})")
    if n_lat < proc_lat:
        raise ValueError(
            f"cannot split {n_lat} lat rows over proc_lat={proc_lat} "
            "(>=1 row/block required for a 1-cell halo)."
        )
    if n_lon < proc_lon:
        raise ValueError(
            f"cannot split {n_lon} lon cols over proc_lon={proc_lon} "
            "(>=1 col/block required for a 1-cell halo)."
        )

    proc_row, proc_col = divmod(rank, proc_lon)
    lat_start, n_lat_local = _even_split(n_lat, proc_lat, proc_row)
    lon_start, n_lon_local = _even_split(n_lon, proc_lon, proc_col)

    def _rank_at(r, c):
        return r * proc_lon + c

    # Latitude = pole-terminated line: None at the grid's top/bottom row.
    south_rank = _rank_at(proc_row - 1, proc_col) if proc_row > 0 else None
    north_rank = (_rank_at(proc_row + 1, proc_col)
                  if proc_row < proc_lat - 1 else None)
    # Longitude = periodic ring: always defined (wrap modulo proc_lon).
    west_rank = _rank_at(proc_row, (proc_col - 1) % proc_lon)
    east_rank = _rank_at(proc_row, (proc_col + 1) % proc_lon)

    return LatLon2DLayout(
        rank=rank, n_ranks=n_ranks, proc_lat=proc_lat, proc_lon=proc_lon,
        proc_row=proc_row, proc_col=proc_col,
        n_lat_global=n_lat, n_lon_global=n_lon,
        n_lat_local=n_lat_local, n_lon_local=n_lon_local,
        lat_start=lat_start, lat_end=lat_start + n_lat_local,
        lon_start=lon_start, lon_end=lon_start + n_lon_local,
        south_rank=south_rank, north_rank=north_rank,
        west_rank=west_rank, east_rank=east_rank, fold=fold,
    )


def scatter_field_latlon_2d(
    global_field: jax.Array, layout: LatLon2DLayout,
) -> jax.Array:
    """Slice the rank's 2-D block ``[lat_start:lat_end, lon_start:lon_end]``
    from a global (n_lat, n_lon[, ...]) field.  Deterministic slice (every
    rank derives the identical global field) — no MPI."""
    return global_field[
        layout.lat_start:layout.lat_end,
        layout.lon_start:layout.lon_end,
    ]


def gather_field_latlon_2d(
    local_field, layout: LatLon2DLayout,
    *, is_u_face: bool = False, is_v_face: bool = False,
):
    """Reassemble the global field from every rank's 2-D block (I/O only).

    mpi4py ``allgather`` of host blocks, placed by (proc_row, proc_col).
    Returns a numpy array on every rank; single-rank returns the input
    as numpy.

    Staggering (mirror of :func:`scatter_state_latlon_2d`): a cell-centred
    field tiles ``(n_lat_global, n_lon_global)`` exactly.  A FACE field has a
    shared boundary face duplicated on the neighbour, so its block is one
    wider/taller and the assembled global array gains the matching face:

    * ``is_u_face`` — lon-face ``u`` (block ``n_lon_local+1``): assemble to
      ``n_lon_global+1`` (faces ``0..n_lon``, the trailing the periodic
      closure).  Interior shared east faces overwrite with the identical
      neighbour value; the last block fills the closure column.
    * ``is_v_face`` — lat-face ``v`` (block ``n_lat_local+1``): assemble to
      ``n_lat_global+1``.

    Passing both flags is an error (a field is one stagger or the other).
    """
    import numpy as _np

    if is_u_face and is_v_face:
        raise ValueError(
            "gather_field_latlon_2d: a field is u-face OR v-face, not both."
        )

    local_np = _np.asarray(local_field)
    if layout.n_ranks == 1:
        return local_np

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() != layout.n_ranks:
        # The constructor takes proc_lat/proc_lon explicitly; this guard
        # catches a layout built for a different world size than the one
        # actually running (would silently mis-assemble the global field;
        # codex review 2026-06-13).
        raise ValueError(
            f"gather_field_latlon_2d: layout n_ranks={layout.n_ranks} "
            f"(proc_lat={layout.proc_lat} x proc_lon={layout.proc_lon}) "
            f"!= MPI world size {comm.Get_size()}."
        )
    blocks = comm.allgather(
        (layout.lat_start, layout.lon_start, local_np))
    trailing = local_np.shape[2:]
    # Face fields carry the shared boundary face, so the global array gains
    # the matching extra row/column (else an ``n_lon_local+1`` u block would
    # overrun an ``n_lon_global``-wide buffer — codex round-2).
    lat_g = layout.n_lat_global + (1 if is_v_face else 0)
    lon_g = layout.n_lon_global + (1 if is_u_face else 0)
    out = _np.zeros((lat_g, lon_g) + trailing, dtype=local_np.dtype)
    for ls, los, blk in blocks:
        out[ls:ls + blk.shape[0], los:los + blk.shape[1]] = blk
    return out


def make_lon_row_comm(layout: LatLon2DLayout):
    """Create the longitude-ring sub-communicator for this rank's
    ``proc_row`` (the ``proc_lon`` ranks that share a latitude band).

    COLLECTIVE over ``COMM_WORLD`` (every rank must call it).  ``key=
    proc_col`` orders the sub-comm ranks by longitude block, so an
    ``allgather`` over it returns blocks in west->east order.  Build
    this ONCE at layout/setup time and reuse — ``Split`` is not free.
    Returns the row sub-communicator.
    """
    from mpi4py import MPI

    return MPI.COMM_WORLD.Split(color=layout.proc_row, key=layout.proc_col)


# custom_vjp on SCALAR lon metadata only (NOT the whole layout): the
# layout's ``fold`` field can carry jax.Array permutations, which must
# not become custom-VJP static args (codex MAJOR 2026-06-13).  The public
# wrapper below extracts the scalar fields.  nondiff args = lon_start,
# lon_end, n_lon_global, proc_lon, row_comm (indices 1..5).
@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def _lon_gather_full_p(local_block, lon_start, lon_end, n_lon_global,
                       proc_lon, row_comm):
    from legoesm.parallel.reductions import mpi4jax_array_result, require_mpi_stack

    # Checked accessor (not a bare ``import mpi4jax``): runs the GPU-transport
    # preflight so this device-array allgather can't slip a GPU-direct
    # misconfiguration past the fail-closed check.
    mpi4jax, _ = require_mpi_stack()

    if n_lon_global % proc_lon != 0:
        raise ValueError(
            f"lon_gather_full needs an equal lon split (n_lon="
            f"{n_lon_global} % proc_lon={proc_lon} != 0) for the "
            "allgather; non-uniform splits need alltoallv (not yet "
            "implemented)."
        )
    # (proc_lon, n_lat_local, n_lon_local[, nlev]); key=proc_col ordered
    # the sub-comm ranks west->east, so axis 0 IS lon-block order.
    stacked = mpi4jax_array_result(mpi4jax.allgather(local_block, comm=row_comm))
    return jnp.concatenate(
        [stacked[i] for i in range(stacked.shape[0])], axis=1)


def _lon_gather_full_p_fwd(local_block, lon_start, lon_end, n_lon_global,
                           proc_lon, row_comm):
    # No residual: the adjoint depends only on the (static) scalar args.
    return _lon_gather_full_p(
        local_block, lon_start, lon_end, n_lon_global, proc_lon, row_comm), None


def _lon_gather_full_p_bwd(lon_start, lon_end, n_lon_global, proc_lon,
                           row_comm, _res, g_full):
    """Adjoint of the lat-pencil gather.

    Forward ``G`` maps each rank's block ``x_s`` into the full-lon array
    that is REPLICATED across the whole row ring (``y_r = concat_s x_s``
    for every ring rank ``r``).  Hence ``x̄_s = Σ_r ȳ_r[:, block_s]`` —
    sum the cotangent over the ring (``allreduce(SUM)``, the only AD-safe
    collective), then slice this rank's lon block.  (``allgather``'s true
    adjoint is a reduce-scatter; ``allreduce(SUM)`` + slice computes the
    same value without a reduce-scatter primitive.)
    """
    from legoesm.parallel.reductions import mpi4jax_array_result, require_mpi_stack

    # Checked accessor (see the forward primal): GPU-transport preflight before
    # this device-array allreduce.
    mpi4jax, MPI = require_mpi_stack()

    summed = mpi4jax_array_result(
        mpi4jax.allreduce(g_full, op=MPI.SUM, comm=row_comm))
    return (summed[:, lon_start:lon_end],)


_lon_gather_full_p.defvjp(_lon_gather_full_p_fwd, _lon_gather_full_p_bwd)


def lon_gather_full(local_block: jax.Array, layout: LatLon2DLayout,
                    row_comm) -> jax.Array:
    """In-trace lat-pencil transpose: assemble the FULL longitude axis
    for this rank's latitude band from the ``proc_lon`` lon-ring blocks.

    The crux primitive for any operation that needs all longitudes on a
    lon-split grid — the N/S pole-fold (180° lon shift / tripole perm)
    and the polar filter (per-row rfft) — design doc §3b/§4.  Uses
    ``mpi4jax.allgather`` over the ``proc_row`` sub-comm
    (:func:`make_lon_row_comm`), so each rank ends with
    ``(n_lat_local, n_lon_global[, nlev])``; apply the fold/filter on
    that, then :func:`lon_scatter_full` back to the rank's block.

    AD-SAFE (custom VJP on :func:`_lon_gather_full_p`): the forward is
    ``allgather`` (no native VJP), but the gather is a linear map whose
    adjoint is exact — a local block contributes to the full-lon array
    on EVERY rank of the row ring, so the cotangent's adjoint is
    ``allreduce(SUM)`` over the row ring (AD-safe) then a slice of this
    rank's lon block.  Requires an EQUAL lon split (``n_lon % proc_lon
    == 0``) so the allgather blocks share a shape; non-uniform splits
    need an ``alltoallv`` (future).

    COLLECTIVE PRECONDITION (deadlock safety): both the forward gather
    AND its reverse-mode ``allreduce`` are collectives over ``row_comm``,
    so EVERY rank of the row ring MUST execute both the primal and the
    backward pass.  Do NOT place a ``lon_gather_full`` result behind
    rank-dependent Python control flow (some ranks skipping the VJP while
    others enter the backward ``allreduce`` deadlocks).  For
    rank-selective use, keep the call in the traced graph on all ranks
    and gate with arithmetic masks (zero cotangents where inactive).
    """
    return _lon_gather_full_p(
        local_block, layout.lon_start, layout.lon_end,
        layout.n_lon_global, layout.proc_lon, row_comm)


def lon_scatter_full(full_field: jax.Array,
                     layout: LatLon2DLayout) -> jax.Array:
    """Slice this rank's longitude block ``[lon_start:lon_end]`` from a
    full-longitude field — inverse of :func:`lon_gather_full`."""
    return full_field[:, layout.lon_start:layout.lon_end]


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
    # IMPORTANT: call the *_local* helpers directly, NOT the
    # backend-dispatched public ``pad_halo_latlon*``.  This helper
    # runs INSIDE the MPI backend's pole-fold step, so calling the
    # public dispatcher would re-enter the MPI path and recurse
    # forever:
    #   pad_halo_latlon (public)
    #     → pad_halo_latlon_mpi   (MPI branch)
    #       → exchange_halo_latlon
    #         → _pole_fold_south / _north  (at boundary ranks)
    #           → _serial_pad_then_strip_lon
    #             → pad_halo_latlon (public, again!) ← recursion
    # The fix: pole-fold uses the SERIAL implementation directly.
    # That's what the docstring above promised; the early Stage 0
    # version pre-dated the backend-dispatch refactor and called
    # the dispatcher by accident.
    from legoesm.grids.halo_latlon import (
        pad_halo_latlon_local,
        pad_halo_latlon_vector_local,
        pad_halo_latlon_3d_local,
        pad_halo_latlon_vector_3d_local,
    )
    if field.ndim == 2:
        helper = (
            pad_halo_latlon_vector_local if negate
            else pad_halo_latlon_local
        )
        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h)
        # Strip the lon halos to recover (halo, n_lon).
        south = padded[:halo, halo:halo + field.shape[1]]
        north = padded[-halo:, halo:halo + field.shape[1]]
    elif field.ndim == 3:
        helper = (
            pad_halo_latlon_vector_3d_local if negate
            else pad_halo_latlon_3d_local
        )
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
    """South-pole halo, bit-identical to serial ``pad_halo_latlon*``
    (stripped of lon halos).

    Operates on ``field`` *without* lon padding — the helper internally
    lon-pads, pole-folds the lon-padded data (matching serial's
    ``(n_lon + 2*halo) // 2`` lon-shift), then strips the lon halos.
    Returns shape ``(halo, n_lon[, nlev])``.

    Stage-0 tests in ``tests/parallel/test_latlon_mpi_halo_serial.py``
    pin this contract: ``exchange_halo_latlon(unpadded, ...)`` at a
    pole-touching rank must equal serial
    ``pad_halo_latlon(unpadded, halo)[:, halo:-halo]``.
    """
    south, _ = _serial_pad_then_strip_lon(field, halo, negate)
    return south


def _pole_fold_north(field: jax.Array, halo: int, negate: bool) -> jax.Array:
    """North-pole halo — see :func:`_pole_fold_south`."""
    _, north = _serial_pad_then_strip_lon(field, halo, negate)
    return north


# ----------------------------------------------------------------------------
# Tripolar north fold (issue #353)
# ----------------------------------------------------------------------------
#
# A tripolar grid (ORCA / eORCA) has a *fold seam* at its northern row
# instead of a geographic pole: cell ``(i, fold_j)`` is identified with
# its fold partner ``(perm[i], fold_j)`` (an i-index reversal), and
# vector components flip sign across the seam.  This is fundamentally
# different from the atmospheric 180°-longitude-roll pole-fold above.
#
# These helpers replicate the serial ocean convention
# (``legoesm.ocean.dynamics.latlon_cgrid_operators.fold_row`` /
# ``pad_ns_scalar`` / ``pad_ns_vector_u`` / ``pad_ns_vector_v``) so the
# MPI northernmost rank produces a north halo that is *bit-identical* to
# the single-rank serial fold.  The descriptor (perm + signs) travels on
# ``LatLonBandLayout.fold``.


def _is_tripolar_layout(layout: LatLonBandLayout) -> bool:
    """True iff ``layout`` carries an active tripolar north-fold."""
    fold = layout.fold
    return fold is not None and bool(fold.is_active)


def _tripolar_fold_perm_sign(fold, *, is_vector_u: bool, is_vector_v: bool):
    """Resolve ``(perm, sign)`` for the tripolar north fold by field kind.

    - scalar  (T, S, eta, depth, mask): ``perm_T``, sign ``+1``
    - u-comp. (zonal face velocity):    ``perm_T``, sign ``vector_sign_u``
    - v-comp. (meridional face vel.):   ``perm_v``, sign ``vector_sign_v``

    Mirrors the serial operators ``pad_ns_scalar`` / ``pad_ns_vector_u``
    / ``pad_ns_vector_v`` in
    :mod:`legoesm.ocean.dynamics.latlon_cgrid_operators`.
    """
    if is_vector_v:
        return fold.perm_v, fold.vector_sign_v
    if is_vector_u:
        return fold.perm_T, fold.vector_sign_u
    return fold.perm_T, 1.0


def _fold_tripolar_north(field: jax.Array, halo: int, perm, sign) -> jax.Array:
    """North tripolar-fold ghost rows: i-reversal permutation + sign flip.

    The fold seam identifies cell ``(i, fold_j)`` with ``(perm[i],
    fold_j)``.  The ghost row immediately north of the boundary row is
    the fold partner of the boundary row; the ``k``-th ghost row north
    is the fold partner of the ``k``-th interior row counted from the
    boundary.  Hence reverse the last ``halo`` interior rows in latitude,
    then apply the column permutation + sign flip.

    For ``halo == 1`` this is bit-identical to the serial
    ``latlon_cgrid_operators.fold_row(field[-1:], perm, sign, n_lon)``;
    ``halo > 1`` is the natural multi-row generalisation.

    Handles the periodic wrap column: u-face / vertex fields carry
    ``n_lon + 1`` columns (column ``n_lon`` == column 0); scalar and
    v-face fields carry ``n_lon`` columns.

    Parameters
    ----------
    field : (n_lat_local, n_cols[, nlev]) — rank-local interior field.
    halo : int
    perm : (n_lon,) int array — ``fold.perm_T`` or ``fold.perm_v``.
    sign : float — ``+1`` (scalar) or ``vector_sign_u``/``vector_sign_v``.

    Returns
    -------
    (halo, n_cols[, nlev]) north ghost rows, ready to concatenate north
    of ``field``.
    """
    n_lon = perm.shape[0]
    # Reverse lat order of the last ``halo`` rows: the row closest to the
    # boundary folds to the ghost row closest to the boundary.
    last = field[-halo:][::-1]
    n_cols = last.shape[1]
    if n_cols == n_lon:
        return sign * last[:, perm]
    if n_cols == n_lon + 1:
        core = sign * last[:, :n_lon][:, perm]
        # Periodic wrap column: column n_lon == column 0 after folding.
        return jnp.concatenate([core, core[:, 0:1]], axis=1)
    raise ValueError(
        f"_fold_tripolar_north: field has {n_cols} columns; expected "
        f"n_lon={n_lon} (scalar/v) or n_lon+1={n_lon + 1} (u/vertex)."
    )


def _north_halo_boundary(
    field: jax.Array, halo: int, layout: LatLonBandLayout,
    is_vector_u: bool, is_vector_v: bool,
) -> jax.Array:
    """North ghost rows at a north-pole-touching rank.

    Tripolar fold (``perm`` + sign) when ``layout.fold`` is active, else
    the geographic 180°-roll pole-fold.
    """
    if _is_tripolar_layout(layout):
        perm, sign = _tripolar_fold_perm_sign(
            layout.fold, is_vector_u=is_vector_u, is_vector_v=is_vector_v,
        )
        return _fold_tripolar_north(field, halo, perm, sign)
    return _pole_fold_north(field, halo, negate=is_vector_v)


def exchange_halo_latlon(
    field: jax.Array,
    layout: LatLonBandLayout,
    halo: int = 1,
    is_vector_v: bool = False,
    is_vector_u: bool = False,
) -> jax.Array:
    """Exchange ``halo`` ghost lat rows on each side via MPI sendrecv.

    Boundary ranks (south_rank is None / north_rank is None) pole-fold
    using the same convention as
    :func:`legoesm.grids.halo_latlon.pad_halo_latlon`: mirror-reverse
    the first / last ``halo`` interior rows, shift by 180° in
    longitude, sign-flip for vector ``v``.

    Tripolar grids (issue #353): when ``layout.fold`` is active, the
    **northernmost** rank (``north_rank is None``) applies the
    permutation-based tripolar fold (``perm_T``/``perm_v`` +
    ``vector_sign_u``/``vector_sign_v``) at the north boundary instead of
    the 180°-roll pole-fold — bit-identical to the serial ocean
    operators ``pad_ns_scalar`` / ``pad_ns_vector_u`` /
    ``pad_ns_vector_v``.  The south boundary and interior partition cuts
    are unchanged (the south of an ORCA grid is a normal closed edge;
    the ocean's south wall BC is applied separately via
    ``pad_with_pole_bc_lat``).

    Periodic longitude is preserved (every rank owns all longitudes;
    no lon halo).

    Parameters
    ----------
    field : jax.Array, shape (n_lat_local, n_lon)  or  (n_lat_local, n_lon, nlev)
        Rank-local interior field, no halos in input.  ``n_lon`` may be
        ``n_lon+1`` for u-face / vertex fields (the wrap column is
        preserved through the tripolar fold).
    layout : LatLonBandLayout
    halo : int, default 1
        Number of ghost rows to add on each lat side.
    is_vector_v : bool
        Whether the field is a meridional vector component that flips
        sign across the pole / fold (v-velocity, meridional flux).  At a
        tripolar fold this selects ``perm_v`` + ``vector_sign_v``.
    is_vector_u : bool
        Whether the field is a zonal vector component (u-velocity).
        Only meaningful at a tripolar fold, where u flips sign
        (``perm_T`` + ``vector_sign_u``).  No effect on the atmospheric
        pole-fold (zonal velocity does not flip there).  Mutually
        exclusive with ``is_vector_v``.

    Returns
    -------
    jax.Array, shape (n_lat_local + 2*halo, n_lon[, nlev])
    """
    if is_vector_u and is_vector_v:
        raise ValueError(
            "exchange_halo_latlon: is_vector_u and is_vector_v are mutually "
            "exclusive (a field is u-type OR v-type, not both)."
        )
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
        north_halo = _north_halo_boundary(
            field, halo, layout, is_vector_u, is_vector_v,
        )
        return jnp.concatenate([south_halo, field, north_halo], axis=0)

    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "Lat-lon MPI halo exchange (n_ranks>1) requires mpi4jax and mpi4py."
        ) from exc

    comm = MPI.COMM_WORLD
    sendrecv = get_sendrecv_vjp(mpi4jax)

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
        north_halo = _north_halo_boundary(
            field, halo, layout, is_vector_u, is_vector_v,
        )

    return jnp.concatenate([south_halo, field, north_halo], axis=0)


def exchange_halo_lon(
    field: jax.Array,
    west_rank: int,
    east_rank: int,
    rank: int,
    halo: int = 1,
) -> jax.Array:
    """Exchange ``halo`` ghost LONGITUDE columns on each side (W/E).

    Increment 1 of the lat-lon 2-D pencil decomposition
    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Longitude is
    GLOBALLY PERIODIC, so — unlike the N/S :func:`exchange_halo_latlon`
    — there are no pole/wall ends: every rank in the longitude ring
    sends its west edge west and its east edge east, and the wrap is
    just the ring topology (the rank owning the last lon block has the
    rank owning the first as its east neighbour).  This is a STANDALONE
    primitive (takes plain neighbour ranks, not a layout) so the future
    ``LatLon2DLayout`` can call it with ``layout.west_rank`` etc.; it is
    NOT yet wired into any step.

    When the longitude ring has a single member (``west_rank == east_rank
    == rank`` — i.e. ``proc_lon == 1``, the current 1-D latitude-band
    case) the wrap is performed LOCALLY from the rank's own columns,
    byte-identical to the existing ``jnp.roll`` periodic wrap and
    runnable without mpi4jax.

    Parameters
    ----------
    field : jax.Array, shape (n_lat_local, n_lon_local[, nlev])
        Rank-local interior field, no lon halos in input.
    west_rank, east_rank : int
        MPI ranks of the western / eastern longitude neighbours (ring).
    rank : int
        This process's MPI rank.
    halo : int, default 1
        Number of ghost columns to add on each lon side.

    Returns
    -------
    jax.Array, shape (n_lat_local, n_lon_local + 2*halo[, nlev])
    """
    if halo <= 0:
        return field

    n_lon_local = field.shape[1]
    if halo > n_lon_local:
        raise ValueError(
            f"exchange_halo_lon: halo={halo} exceeds n_lon_local="
            f"{n_lon_local} on rank {rank}.  Reduce proc_lon or "
            "increase longitude resolution."
        )

    # Single-member lon ring (proc_lon == 1): local periodic wrap — the
    # west ghost is the rank's own EAST edge, the east ghost its WEST
    # edge.  Byte-identical to the legacy ``jnp.roll`` wrap; no MPI.
    if west_rank == rank and east_rank == rank:
        west_halo = field[:, -halo:]
        east_halo = field[:, :halo]
        return jnp.concatenate([west_halo, field, east_halo], axis=1)

    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "Lat-lon E/W MPI halo exchange (proc_lon>1) requires "
            "mpi4jax and mpi4py."
        ) from exc

    comm = MPI.COMM_WORLD
    sendrecv = get_sendrecv_vjp(mpi4jax)
    trailing_lat = field.shape[0]
    other = field.shape[2:]  # (nlev,) or ()

    # PHASE-CONSTANT tags with sendtag == recvtag (one per shift
    # direction, +1 for the opposite) — the AD-safe convention from
    # plane_mpi._TAG_EW.  The shared sendrecv VJP's backward swaps
    # source<->dest but KEEPS the tags, so a rank-as-tag scheme
    # (sendtag=rank, recvtag=source) mismatches on the reverse ring at
    # proc_lon>=3 (codex review MAJOR 2026-06-13: gradients would hang).
    # With a single tag per phase, forward AND backward messages match
    # for any ring size.  (Numerically distinct from plane_mpi's
    # 1000/2000; not strictly distinct from the N/S rank-tags, but a
    # cross-match would also need the same comm + same source + same
    # dest + concurrent outstanding recvs — E/W and N/S neighbour pairs
    # are disjoint at proc_lon>1, and proc_lon==1 uses the local-wrap
    # fast path, so no cross-match path exists.)
    _TAG_LON = 3_000

    # UNIFORM-DIRECTION SHIFT (deadlock-free on a periodic RING).  The
    # same-neighbour pattern (source==dest) that the N/S
    # ``exchange_halo_latlon`` uses only unwinds on a LINE — the poles
    # terminate the chain.  Longitude is a closed ring with no ends, and
    # the AD-safe sendrecv token serialises this rank's two exchanges,
    # so a same-neighbour west-then-east pattern makes phase-1 wait on
    # the neighbour's phase-2 → circular deadlock (observed: job
    # 8476475 timed out in mpi_sendrecv).  Instead each phase is a
    # UNIFORM shift where every send is matched by a recv IN THE SAME
    # phase (a permutation), so no cross-phase ring dependency exists.

    # Phase 1 — EASTWARD shift: send our EAST edge to the east neighbour,
    # receive the west neighbour's east edge into our WEST halo.
    send_e = field[:, -halo:].reshape(-1)
    recv_w = sendrecv(
        send_e, jnp.zeros_like(send_e),
        west_rank,   # source (recv from west)
        east_rank,   # dest   (send to east)
        _TAG_LON,    # sendtag == recvtag (phase-constant, AD-safe)
        _TAG_LON,
        comm,
    )
    west_halo = recv_w.reshape((trailing_lat, halo) + other)

    # Phase 2 — WESTWARD shift: send our WEST edge to the west neighbour,
    # receive the east neighbour's west edge into our EAST halo.
    send_w = field[:, :halo].reshape(-1)
    recv_e = sendrecv(
        send_w, jnp.zeros_like(send_w),
        east_rank,     # source (recv from east)
        west_rank,     # dest   (send to west)
        _TAG_LON + 1,  # phase-2 tag (sendtag == recvtag)
        _TAG_LON + 1,
        comm,
    )
    east_halo = recv_e.reshape((trailing_lat, halo) + other)

    return jnp.concatenate([west_halo, field, east_halo], axis=1)


def _pad_lat_wall_2d(
    field: jax.Array,
    layout: LatLon2DLayout,
    halo: int,
    south_value: float,
    north_value: float,
) -> jax.Array:
    """Lat-axis (N/S) wall pad for a 2-D pencil row — the shared core of
    :func:`pad_halo_latlon_2d` and :func:`pad_with_pole_bc_lat_2d`.

    Interior lat cuts MPI-sendrecv their edge row with the south / north
    neighbour (the AD-safe rank-as-tag LINE pattern — the lat axis is a
    pole-terminated line, not the periodic ring); pole-touching rows
    (``south_rank``/``north_rank is None``) fill the constant wall.
    Longitude is NOT touched.  Works for 1-D lat metrics and N-D face
    fields (trailing axes ride the reshape).  ``proc_lat == 1`` (both poles
    local) runs serially without mpi4jax.

    Returns ``(n_lat_local + 2*halo, …)`` — same trailing shape as input.
    """
    # halo must fit the SMALLEST local lat block: a neighbour owning fewer
    # than `halo` rows would send/recv a mismatched slab and the exchange
    # would truncate/hang (same guard rationale as pad_halo_latlon_2d's
    # lon side).  _even_split gives the first blocks one extra row, so the
    # smallest block is floor(n_lat_global / proc_lat).
    min_lat_block = layout.n_lat_global // layout.proc_lat
    if halo > min_lat_block:
        raise ValueError(
            f"_pad_lat_wall_2d: halo={halo} exceeds the smallest local lat "
            f"block ({min_lat_block}=n_lat_global {layout.n_lat_global}//"
            f"proc_lat {layout.proc_lat}); a neighbour would send/recv a "
            f"mismatched halo and the MPI exchange would abort/hang.")
    trailing = field.shape[1:]
    south_is_pole = layout.south_rank is None
    north_is_pole = layout.north_rank is None

    # Both lat ends are physical poles (proc_lat==1): pure local wall pad,
    # no MPI — keep the path runnable serially (the proc_lon==1, proc_lat==1
    # single-process equivalence test, and the proc_lon-only ring case).
    if south_is_pole and north_is_pole:
        south = jnp.full((halo,) + trailing, south_value, field.dtype)
        north = jnp.full((halo,) + trailing, north_value, field.dtype)
        return jnp.concatenate([south, field, north], axis=0)

    import mpi4jax
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    sendrecv = get_sendrecv_vjp(mpi4jax)

    if south_is_pole:
        south = jnp.full((halo,) + trailing, south_value, field.dtype)
    else:
        send_s = field[:halo].reshape(-1)
        recv_s = sendrecv(
            send_s, jnp.zeros_like(send_s),
            layout.south_rank, layout.south_rank,
            layout.rank, layout.south_rank, comm)
        south = recv_s.reshape((halo,) + trailing)

    if north_is_pole:
        north = jnp.full((halo,) + trailing, north_value, field.dtype)
    else:
        send_n = field[-halo:].reshape(-1)
        recv_n = sendrecv(
            send_n, jnp.zeros_like(send_n),
            layout.north_rank, layout.north_rank,
            layout.rank, layout.north_rank, comm)
        north = recv_n.reshape((halo,) + trailing)

    return jnp.concatenate([south, field, north], axis=0)


def pad_with_pole_bc_lat_2d(
    interior: jax.Array,
    layout: LatLon2DLayout,
    halo: int = 1,
    south_value: float = 0.0,
    north_value: float = 0.0,
) -> jax.Array:
    """Lat-axis-ONLY wall pad for a 2-D pencil layout (NO longitude halo).

    The 2-D analogue of the band
    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` MPI path: pad
    only the lat axis (interior cut sendrecv + pole wall constant), leaving
    longitude untouched.  The band path never split longitude, so its
    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
    operator adds lon ghosts through its own dispatched lon halo.  Routed
    here from ``halo_latlon.pad_with_pole_bc_lat`` when the active topology
    is a :class:`LatLon2DLayout` (wall poles only; the tripolar north fold /
    vector-u seam is excluded upstream).  AD-safe via the shared sendrecv
    VJP.
    """
    if halo <= 0:
        return interior
    return _pad_lat_wall_2d(interior, layout, halo, south_value, north_value)


def pad_halo_latlon_2d(
    field: jax.Array,
    layout: LatLon2DLayout,
    halo: int = 1,
    pole_bc: str = "wall",
    south_value: float = 0.0,
    north_value: float = 0.0,
) -> jax.Array:
    """Full 2-D halo pad (lat + lon) for ANY pencil row — the increment-3
    integration of the 2-D building blocks.

    N/S then E/W (corners ride the lat-padded edge columns into the E/W
    exchange).  N/S: interior cuts MPI-sendrecv with the lat neighbour;
    pole-touching rows fill the pole side per ``pole_bc``.  Every rank
    participates in its interior-facing sendrecv (the line terminates at
    the pole rows' local fill) so there is NO collective-line deadlock —
    the failure of the earlier "guard-and-skip" interior-only attempt.
    E/W: the periodic-ring :func:`exchange_halo_lon`.

    ``pole_bc``:
      * ``"wall"`` (default) — pole ghost rows = constant wall BC
        (``south_value``/``north_value``).  This is the REGULAR lat-lon
        case (and ocean, whose poles are closed walls).  Fully LOCAL at
        the poles ⇒ no longitude transpose, and the whole pad is
        DEADLOCK-FREE and AD-SAFE (sendrecv-VJP on both axes).
      * ``"fold"`` / ``"tripole"`` — the atmospheric 180° pole-fold and
        the ocean tripole north-fold need the FULL longitude axis at the
        pole row (180° shift / permutation), so they require the
        lat-pencil transpose (:func:`lon_gather_full`).  NOT YET wired
        (next increment); raises so a fold deck can't silently get a
        wall.

    Returns ``(n_lat_local + 2*halo, n_lon_local + 2*halo[, nlev])``.
    """
    if pole_bc not in ("wall", "fold", "tripole"):
        raise ValueError(
            f"pad_halo_latlon_2d: pole_bc must be 'wall', 'fold', or "
            f"'tripole', got {pole_bc!r}")
    if pole_bc in ("fold", "tripole"):
        raise NotImplementedError(
            "pad_halo_latlon_2d: pole_bc='fold'/'tripole' needs the "
            "lat-pencil transpose (lon_gather_full) to do the pole-fold's "
            "global-longitude shift/permutation on a lon-split row — next "
            "increment.  Use pole_bc='wall' for regular lat-lon / "
            "closed-pole ocean."
        )
    if halo <= 0:
        return field

    # halo must fit the SMALLEST local block on BOTH axes — with an
    # uneven split a neighbour can own fewer than `halo` rows/cols, so
    # its send/recv would be shorter than this rank expects and the MPI
    # exchange truncates/aborts/hangs before any reshape (codex MAJOR
    # 2026-06-13). Smallest block on an even-ish split = floor(n/proc).
    # _even_split gives the first (n % parts) blocks one extra row/col, so
    # the SMALLEST block is exactly floor(n_global / parts).
    min_lat_block = layout.n_lat_global // layout.proc_lat
    min_lon_block = layout.n_lon_global // layout.proc_lon
    if halo > min_lat_block or halo > min_lon_block:
        raise ValueError(
            f"pad_halo_latlon_2d: halo={halo} exceeds the smallest local "
            f"block (lat {min_lat_block}=n_lat_global "
            f"{layout.n_lat_global}//proc_lat {layout.proc_lat}, lon "
            f"{min_lon_block}=n_lon_global {layout.n_lon_global}//proc_lon "
            f"{layout.proc_lon}); a neighbour would send/recv a mismatched "
            f"halo and the MPI exchange would abort/hang.")

    # N/S — lat-axis wall pad (interior cut sendrecv at the AD-safe
    # rank-as-tag LINE pattern, pole side wall); shared verbatim with
    # pad_with_pole_bc_lat_2d so the two lat exchanges stay bit-identical.
    ns = _pad_lat_wall_2d(field, layout, halo, south_value, north_value)
    # E/W ring on the lat-padded block → fills lon ghosts + corners.
    return exchange_halo_lon(
        ns, layout.west_rank, layout.east_rank, layout.rank, halo=halo)


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


def pad_halo_latlon_mpi(
    data, layout: LatLonBandLayout, halo: int = 1,
    is_vector_v: bool = False, is_vector_u: bool = False,
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
            f"pad_halo_latlon_mpi: data.ndim must be 2 or 3, got {data.ndim}"
        )

    # Step 1: periodic lon wrap (local on every rank — every rank owns
    # the full lon axis).  This produces ``lon_padded`` shape
    # ``(n_lat_local, n_lon + 2*halo[, nlev])``.
    if data.ndim == 2:
        lon_padded = jnp.pad(data, ((0, 0), (halo, halo)), mode="wrap")
    else:
        lon_padded = jnp.pad(
            data, ((0, 0), (halo, halo), (0, 0)), mode="wrap",
        )

    # Step 2: lat halo — pole-fold the LON-PADDED data at boundary
    # ranks (matches serial ``pad_halo_latlon``'s lon-pad-then-fold
    # order, so the pole-fold lon-shift is ``(n_lon + 2*halo) // 2``
    # like serial does).  Interior partition cuts MPI sendrecv with
    # the neighbour rank.
    #
    # We do NOT call ``exchange_halo_latlon`` here because that
    # function operates on an *unpadded* field and is designed to
    # return serial-stripped (lon-halo-stripped) halos.  Routing
    # through it would either re-lon-pad or produce stripped halos
    # — neither matches the full-pad result serial expects.
    if halo > data.shape[0]:
        raise ValueError(
            f"pad_halo_latlon_mpi: halo={halo} exceeds n_lat_local="
            f"{data.shape[0]} on rank {layout.rank}."
        )

    # --- South halo ---
    if layout.south_rank is None:
        # Pole-fold the lon-padded first ``halo`` rows.  Uses
        # ``fold_pole_rows*`` whose lon-shift is ``data.shape[1] // 2``
        # — for lon-padded input that's exactly ``(n_lon + 2*halo) // 2``
        # which is the serial convention.
        if data.ndim == 2:
            south_halo, _ = fold_pole_rows(lon_padded, halo, negate=is_vector_v)
        else:
            south_halo, _ = fold_pole_rows_3d(
                lon_padded, halo, negate=is_vector_v,
            )
    else:
        # MPI sendrecv with south neighbour — exchanges the
        # lon-padded boundary rows.  AD-safe via get_sendrecv_vjp.
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "pad_halo_latlon_mpi: multi-rank lat halo requires "
                "mpi4jax and mpi4py."
            ) from exc
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)
        send_bot = lon_padded[:halo].reshape(-1)
        recv_template = jnp.zeros_like(send_bot)
        recv_south = sendrecv(
            send_bot, recv_template,
            layout.south_rank, layout.south_rank,
            layout.rank, layout.south_rank, comm,
        )
        # Inter-rank exchange: no sign flip (v is continuous across a
        # partition cut, only flips across the actual pole).
        south_halo = recv_south.reshape((halo,) + lon_padded.shape[1:])

    # --- North halo ---
    if layout.north_rank is None:
        if _is_tripolar_layout(layout):
            # Tripolar fold: permute the UNPADDED data (perm is defined on
            # the n_lon columns), then lon-wrap-pad to match ``lon_padded``.
            perm, sign = _tripolar_fold_perm_sign(
                layout.fold, is_vector_u=is_vector_u, is_vector_v=is_vector_v,
            )
            north_unpadded = _fold_tripolar_north(data, halo, perm, sign)
            if data.ndim == 2:
                north_halo = jnp.pad(
                    north_unpadded, ((0, 0), (halo, halo)), mode="wrap")
            else:
                north_halo = jnp.pad(
                    north_unpadded, ((0, 0), (halo, halo), (0, 0)), mode="wrap")
        elif data.ndim == 2:
            _, north_halo = fold_pole_rows(lon_padded, halo, negate=is_vector_v)
        else:
            _, north_halo = fold_pole_rows_3d(
                lon_padded, halo, negate=is_vector_v,
            )
    else:
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "pad_halo_latlon_mpi: multi-rank lat halo requires "
                "mpi4jax and mpi4py."
            ) from exc
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)
        send_top = lon_padded[-halo:].reshape(-1)
        recv_template = jnp.zeros_like(send_top)
        recv_north = sendrecv(
            send_top, recv_template,
            layout.north_rank, layout.north_rank,
            layout.rank, layout.north_rank, comm,
        )
        north_halo = recv_north.reshape((halo,) + lon_padded.shape[1:])

    return jnp.concatenate([south_halo, lon_padded, north_halo], axis=0)


def _pad_with_pole_bc_lat_mpi_1d(
    interior, layout: LatLonBandLayout, halo: int,
    south_value: float, north_value: float,
):
    """MPI pad for 1-D lat-axis metrics (sin_lat, cos_lat_v, dx_cell).

    1-D fields don't have a longitude axis to pole-fold over, so the
    standard ``exchange_halo_latlon`` (which assumes axis 1 = lon)
    can't be reused directly.  This helper builds the (n_lat_local +
    2*halo,) result by:

      * inter-rank sendrecv on the lat axis (just like
        ``exchange_halo_latlon``), and
      * constant ``south_value`` / ``north_value`` pad at pole-touching
        boundaries (instead of pole-fold).

    Pole-fold would be inappropriate here even with the right shape:
    a wall-BC pad on ``sin_lat`` wants the constant ``-1`` / ``+1``
    (sin at the pole), not the mirror-then-shift of ``sin(lat[0])``.
    Which is exactly what ``pad_with_pole_bc_lat`` is for.
    """
    interior_dtype = interior.dtype

    if halo > interior.shape[0]:
        raise ValueError(
            f"_pad_with_pole_bc_lat_mpi_1d: halo={halo} exceeds "
            f"n_lat_local={interior.shape[0]} on rank {layout.rank}"
        )

    # South side
    if layout.south_rank is None:
        south_slab = jnp.full(
            (halo,), jnp.asarray(south_value, dtype=interior_dtype),
        )
    else:
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "Lat-lon MPI 1-D pad (interior partition cut) requires "
                "mpi4jax and mpi4py."
            ) from exc
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)
        send_bot = interior[:halo]
        recv_template = jnp.zeros_like(send_bot)
        south_slab = sendrecv(
            send_bot, recv_template,
            layout.south_rank, layout.south_rank,
            layout.rank, layout.south_rank, comm,
        )

    # North side
    if layout.north_rank is None:
        north_slab = jnp.full(
            (halo,), jnp.asarray(north_value, dtype=interior_dtype),
        )
    else:
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "Lat-lon MPI 1-D pad (interior partition cut) requires "
                "mpi4jax and mpi4py."
            ) from exc
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)
        send_top = interior[-halo:]
        recv_template = jnp.zeros_like(send_top)
        north_slab = sendrecv(
            send_top, recv_template,
            layout.north_rank, layout.north_rank,
            layout.rank, layout.north_rank, comm,
        )

    return jnp.concatenate([south_slab, interior, north_slab], axis=0)


def _pad_static_wall_bc_mpi(
    interior, layout: LatLonBandLayout, halo: int,
    south_value: float, north_value: float,
):
    """Trace-time host-MPI pad of a CONCRETE (non-Tracer) field.

    Grid metrics (``grid.lat``, ``sin_lat``, tripolar ``dx_T`` rows, …)
    are closed-over constants inside the jitted step, so their wall-BC
    pads are static — yet the traced ``sendrecv`` op cannot be
    constant-folded by XLA, so every operator call re-exchanged the
    same bytes every step (census job 8459289: 14 metric pads/step, two
    of them inside the barotropic PCG ``fori_loop`` body = 120 executed
    sendrecv pairs/step at M=60).  This helper performs the exchange
    ONCE at trace time with host mpi4py; the result is a compile-time
    constant and the per-step collective disappears.

    Deadlock safety: tracing is SPMD-synchronous — every rank traces
    the same Python (same shapes/dtypes ⇒ same jit cache hits/misses;
    the persistent XLA cache caches compilation, not tracing), so all
    ranks execute the same eager ``Sendrecv`` schedule in the same
    order.  Blocking host ``Sendrecv`` pairs match neighbour-to-
    neighbour exactly like the traced path.

    Wall-BC semantics only (constants at pole-touching boundaries —
    callers with ``north_fold=True`` keep the traced path).

    ``mpi4py`` is imported only inside the neighbor branches (codex
    round-2 MAJOR): a single-rank armed-"mpi" backend (both neighbors
    ``None`` — ``make_latlon_mpi_step`` arms even at n_ranks=1) must
    keep working without the optional MPI stack, exactly like the
    traced 1-D path.
    """
    arr = np.asarray(interior)
    trailing = arr.shape[1:]

    if layout.south_rank is None:
        south = np.full((halo,) + trailing, south_value, dtype=arr.dtype)
    else:
        from mpi4py import MPI

        south = np.empty((halo,) + trailing, dtype=arr.dtype)
        MPI.COMM_WORLD.Sendrecv(
            np.ascontiguousarray(arr[:halo]), dest=layout.south_rank,
            sendtag=layout.rank,
            recvbuf=south, source=layout.south_rank,
            recvtag=layout.south_rank,
        )
    if layout.north_rank is None:
        north = np.full((halo,) + trailing, north_value, dtype=arr.dtype)
    else:
        from mpi4py import MPI

        north = np.empty((halo,) + trailing, dtype=arr.dtype)
        MPI.COMM_WORLD.Sendrecv(
            np.ascontiguousarray(arr[-halo:]), dest=layout.north_rank,
            sendtag=layout.rank,
            recvbuf=north, source=layout.north_rank,
            recvtag=layout.north_rank,
        )
    return jnp.concatenate(
        [jnp.asarray(south), jnp.asarray(interior), jnp.asarray(north)],
        axis=0,
    )


def pad_with_pole_bc_lat_mpi(
    interior, layout: LatLonBandLayout, halo: int = 1,
    south_value: float = 0.0, north_value: float = 0.0,
    is_vector_v: bool = False, is_vector_u: bool = False,
    north_fold: bool = False,
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

    Tripolar grids (issue #353): the north fold is OPT-IN via
    ``north_fold``.  The default (``north_fold=False``) keeps the
    historical WALL semantics at the north — even on a tripolar layout —
    so existing wall-BC callers (``pad_ns_zero`` and the like) are
    unchanged.  Only when ``north_fold=True`` AND ``layout.fold`` is
    active AND this rank owns the north boundary is the north slab left
    as the permutation-folded data from ``exchange_halo_latlon`` instead
    of being overwritten with ``north_value``.  ``is_vector_u`` /
    ``is_vector_v`` then select ``vector_sign_u`` / ``vector_sign_v`` at
    the fold.  The south boundary is always a wall (an ORCA grid's south
    is a closed edge).
    """
    if halo <= 0:
        return interior

    # Static-metric constant folding (census job 8459289): a concrete
    # (non-Tracer) 1-D input is a closed-over compile-time constant —
    # its wall-BC pad is exchanged ONCE at trace time via host MPI
    # instead of a per-step traced sendrecv that XLA cannot fold.
    # Codex-hardened gate (review 2026-06-10):
    #   * ndim == 1 only — covers every measured static pad (the
    #     1-D lat metrics; census 8459289) while keeping the rarely-
    #     trodden 2-D tripolar-metric pads on the traced path;
    #   * boundary constants must ALSO be non-Tracers (a traced
    #     south/north value must not be constant-folded);
    #   * same halo <= n_lat_local guard as the traced path (a silent
    #     short send would truncate/hang instead of raising).
    # INVARIANT (same class as every traced mpi4jax collective in this
    # module): tracing is SPMD-symmetric — every rank traces the same
    # jitted functions in the same order.  Rank-subset tracing would
    # block in the trace-time Sendrecv exactly like rank-subset
    # EXECUTION blocks the traced sendrecv.  Set
    # ``LEGOESM_LATLON_STATIC_METRIC_PAD=0`` to restore the traced
    # exchange (A/B + kill-switch lever).
    import os
    if (
        not north_fold
        and interior.ndim == 1
        and not isinstance(interior, jax.core.Tracer)
        and not isinstance(south_value, jax.core.Tracer)
        and not isinstance(north_value, jax.core.Tracer)
        and os.environ.get("LEGOESM_LATLON_STATIC_METRIC_PAD", "1") != "0"
    ):
        if halo > interior.shape[0]:
            raise ValueError(
                f"pad_with_pole_bc_lat_mpi: halo={halo} exceeds "
                f"n_lat_local={interior.shape[0]} on rank {layout.rank}"
            )
        return _pad_static_wall_bc_mpi(
            interior, layout, halo, south_value, north_value,
        )

    # 1D lat-axis metrics (sin_lat, cos_lat_v_interior, dx_cell, …)
    # don't have a lon axis to pole-fold over.  Build the result
    # directly: constant pad at pole-touching boundaries, sendrecv
    # at interior cuts — no call to ``exchange_halo_latlon`` (which
    # assumes axis 1 = lon and would fail on ndim=1).
    if interior.ndim == 1:
        return _pad_with_pole_bc_lat_mpi_1d(
            interior, layout, halo=halo,
            south_value=south_value, north_value=north_value,
        )

    # Step 1: run the standard halo exchange.  Boundary ranks will
    # get pole-folded values; interior cuts will get sendrecv'd
    # neighbour values (which is what we want — those are NOT to
    # be overwritten).
    padded = exchange_halo_latlon(
        interior, layout, halo=halo,
        is_vector_v=is_vector_v, is_vector_u=is_vector_u,
    )
    # Step 2: where this rank touches a pole, replace the
    # pole-folded slab with the wall-BC constant.  Slab shapes
    # match by construction.  We use the bcast-tuple pattern so
    # this works for 1D (sin_lat), 2D (face metrics) and 3D
    # (u-on-face-with-levels) fields uniformly.
    keep_north_fold = north_fold and _is_tripolar_layout(layout)
    if layout.south_rank is None:
        south_const = jnp.full(
            (halo,) + interior.shape[1:],
            jnp.asarray(south_value, dtype=interior.dtype),
        )
        padded = jnp.concatenate([south_const, padded[halo:]], axis=0)
    if layout.north_rank is None and not keep_north_fold:
        # Regular north pole / wall-BC caller → ``north_value`` constant.
        # Only an explicit ``north_fold=True`` request on an active
        # tripolar layout keeps the permutation-folded north slab that
        # ``exchange_halo_latlon`` produced above.
        north_const = jnp.full(
            (halo,) + interior.shape[1:],
            jnp.asarray(north_value, dtype=interior.dtype),
        )
        padded = jnp.concatenate([padded[:-halo], north_const], axis=0)
    return padded


# ============================================================================
# Fused multi-field halo exchange (scaling campaign, audit lever O4)
# ============================================================================
#
# The lat-lon band MPI step issues O(30) independent wall-BC cell pads per
# step, each paying its own token-serialized sendrecv pair (~200-300 us
# latency on Ginsburg CPU nodes — the measured rank-growing term of the
# baroclinic phase, jobs 8458934/8458989).  mpi4jax sendrecvs do NOT
# overlap (token chain), so N independent pads cost N x latency.  Fusing
# independent same-dataflow-level pads into ONE concatenated sendrecv per
# cut per dtype group cuts that latency term by the cluster size while
# exchanging bit-identical bytes (concat -> sendrecv -> split is value-
# identical to per-field sendrecvs; no arithmetic).
#
# Scope: scalar wall-BC fields ONLY (``pad_ns_zero`` /
# ``pad_with_pole_bc_lat`` with constant boundary values, no
# ``north_fold``, no ``is_vector_*``) — boundary slabs are constant fills,
# so per-field flags reduce to per-field constants and the interior cut
# exchange is flag-independent.  Pole-fold / tripolar-fold callers keep
# the single-field path.


def pad_with_pole_bc_lat_multi_mpi(
    fields,
    layout: LatLonBandLayout,
    halo: int = 1,
    south_values=None,
    north_values=None,
):
    """Fused MPI variant of N independent ``pad_with_pole_bc_lat`` calls.

    Pads every field in ``fields`` along the lat axis (axis 0) with
    ``halo`` rows per side: constant ``south_values[i]`` /
    ``north_values[i]`` at pole-touching boundaries, MPI-sendrecv'd
    neighbour rows at interior partition cuts.  All fields must share
    ``n_lat_local`` (axis 0); trailing shapes and dtypes may differ
    (fields are flattened and concatenated per dtype group — ONE
    sendrecv pair per cut per dtype group instead of one per field).

    Value-identical to ``tuple(pad_with_pole_bc_lat_mpi(f, layout,
    halo, sv, nv) for ...)`` for wall-BC scalars: the single-field path
    pole-folds at boundary ranks and then overwrites the boundary slabs
    with the constants, so skipping the fold and filling constants
    directly produces the same result with less local compute.

    AD-safe: the fused buffer goes through the same
    :func:`get_sendrecv_vjp` custom-vjp as the single-field path;
    ``concatenate``/``slice`` carry native JAX VJPs.

    Returns a tuple of padded arrays, in input order.
    """
    fields = tuple(fields)
    n = len(fields)
    if n == 0:
        return ()
    if south_values is None:
        south_values = (0.0,) * n
    if north_values is None:
        north_values = (0.0,) * n
    south_values = tuple(south_values)
    north_values = tuple(north_values)
    if len(south_values) != n or len(north_values) != n:
        raise ValueError(
            "pad_with_pole_bc_lat_multi_mpi: south_values/north_values "
            f"must match len(fields)={n}; got {len(south_values)}/"
            f"{len(north_values)}."
        )
    if halo <= 0:
        return fields

    n_lat_local = fields[0].shape[0]
    for i, f in enumerate(fields):
        if f.shape[0] != n_lat_local:
            raise ValueError(
                "pad_with_pole_bc_lat_multi_mpi: all fields must share "
                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
                f"{i} has {f.shape[0]}."
            )
    if halo > n_lat_local:
        raise ValueError(
            f"pad_with_pole_bc_lat_multi_mpi: halo={halo} exceeds "
            f"n_lat_local={n_lat_local} on rank {layout.rank}."
        )

    south_slabs: list = [None] * n
    north_slabs: list = [None] * n

    # Pole-touching boundaries: constant wall-BC fill, no comm.
    if layout.south_rank is None:
        for i, f in enumerate(fields):
            south_slabs[i] = jnp.full(
                (halo,) + f.shape[1:],
                jnp.asarray(south_values[i], dtype=f.dtype),
            )
    if layout.north_rank is None:
        for i, f in enumerate(fields):
            north_slabs[i] = jnp.full(
                (halo,) + f.shape[1:],
                jnp.asarray(north_values[i], dtype=f.dtype),
            )

    # Interior partition cuts: ONE fused sendrecv per cut per dtype group.
    if layout.south_rank is not None or layout.north_rank is not None:
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "Lat-lon fused MPI halo exchange (n_ranks>1) requires "
                "mpi4jax and mpi4py."
            ) from exc
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)

        # Group field indices by dtype in first-appearance order — the
        # order is trace-deterministic, so every rank issues the same
        # fused-message schedule (sendrecv pairing relies on it).
        groups: dict = {}
        for i, f in enumerate(fields):
            groups.setdefault(jnp.dtype(f.dtype), []).append(i)

        for idxs in groups.values():
            sizes = [
                halo * int(np.prod(fields[i].shape[1:], dtype=np.int64))
                for i in idxs
            ]
            offsets = np.concatenate([[0], np.cumsum(sizes)])

            if layout.south_rank is not None:
                send_bot = jnp.concatenate(
                    [fields[i][:halo].reshape(-1) for i in idxs]
                )
                recv_south = sendrecv(
                    send_bot, jnp.zeros_like(send_bot),
                    layout.south_rank,      # source
                    layout.south_rank,      # dest
                    layout.rank,            # sendtag = sender's rank
                    layout.south_rank,      # recvtag = source's rank
                    comm,
                )
                for k, i in enumerate(idxs):
                    south_slabs[i] = recv_south[
                        offsets[k]:offsets[k + 1]
                    ].reshape((halo,) + fields[i].shape[1:])

            if layout.north_rank is not None:
                send_top = jnp.concatenate(
                    [fields[i][-halo:].reshape(-1) for i in idxs]
                )
                recv_north = sendrecv(
                    send_top, jnp.zeros_like(send_top),
                    layout.north_rank,
                    layout.north_rank,
                    layout.rank,
                    layout.north_rank,
                    comm,
                )
                for k, i in enumerate(idxs):
                    north_slabs[i] = recv_north[
                        offsets[k]:offsets[k + 1]
                    ].reshape((halo,) + fields[i].shape[1:])

    return tuple(
        jnp.concatenate([south_slabs[i], fields[i], north_slabs[i]], axis=0)
        for i in range(n)
    )


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


def scatter_state_latlon_2d(state, layout: LatLon2DLayout):
    """Extract the rank-local 2-D block from a global C-grid lat-lon state.

    The 2-D analog of :func:`scatter_state_latlon`: slice BOTH the latitude
    band ``[lat_start, lat_end)`` and the longitude pencil
    ``[lon_start, lon_end)``.  Pure indexing — no MPI (each rank slices its own
    block; callers broadcast a global state or build it locally).

    Staggering (codex design pitfall — slice lat AND lon together, lat first):

    * ``T``/``p_s``/``phis``/tracers : cell-centred → ``[s:e, w:x]``.
    * ``u`` : LON-face.  The dycore stores u with ``n_lon+1`` faces (faces
      ``0..n_lon``, the trailing one the periodic closure ``face n_lon ==
      face 0``).  A block owns faces ``[w, x]`` → ``[w:x+1]`` (``n_lon_local+1``
      faces), so W/E-neighbouring ranks DUPLICATE the shared east face — the
      EXACT longitude analogue of v's shared lat-face row.  This matches the
      dycore u convention directly (no operator rewrite) and makes
      ``proc_lon==1`` reproduce the band's full ``n_lon+1`` u.  (The earlier
      ``[w:x]`` slice dropped to ``n_lon_local`` cell-aligned columns — a
      different, non-dycore convention that silently broke u at proc_lon==1
      and was masked by a cell-shaped fake-u test fixture.)
    * ``v`` : LAT-face, one extra global row at the poles (shape ``n_lat+1``);
      give rows ``[s, e+1)`` so N/S-neighbouring ranks DUPLICATE the shared
      boundary face (the same convention as the 1-D band; both ranks must hold
      the same value there).  Longitude is sliced ``[w:x]`` like the cells.

    ``proc_lon == 1`` reproduces :func:`scatter_state_latlon` exactly (u keeps
    its full ``n_lon+1`` faces, v its shared lat row).
    """
    s, e = layout.lat_start, layout.lat_end
    w, x = layout.lon_start, layout.lon_end

    T_local = state.T[s:e, w:x]
    p_s_local = state.p_s[s:e, w:x]
    phis_local = state.phis[s:e, w:x]
    # lon-face: faces [w, x] => one extra (shared east) column, the lon twin
    # of v's shared lat row.  The global u carries the n_lon+1 periodic-closure
    # column, so [w:x+1] is well-defined at the seam too (last block's east
    # face == global face n_lon == periodic dup of face 0).
    u_local = state.u[s:e, w:x + 1]
    v_local = state.v[s:e + 1, w:x]   # lat-face: shared boundary row duplicated

    tracers_local = {}
    if getattr(state, "tracers", None):
        for name, tr in state.tracers.items():
            tracers_local[name] = tr[s:e, w:x]

    return state._replace(
        u=u_local,
        v=v_local,
        T=T_local,
        p_s=p_s_local,
        phis=phis_local,
        tracers=tracers_local if tracers_local else state.tracers,
    )


def gather_field_latlon(
    local_arr,
    layout: LatLonBandLayout,
    *,
    is_v_face: bool = False,
):
    """Gather a rank-local lat-banded array onto rank 0.

    Used by both :func:`gather_state_latlon` (dycore-internal raw state)
    and ``ModelDriver._gather_state_for_global_checkpoint`` (driver-side
    Field-wrapped state).  Keep the v-row-trim convention in one place
    so save / restart and dynamics stay in sync.

    Parameters
    ----------
    local_arr : jax.Array or numpy.ndarray
        Rank-local field with first axis = lat (cell-centre for scalars
        or face for ``v``).
    layout : LatLonBandLayout
    is_v_face : bool
        When True, ``local_arr`` is a v-face array (shape
        ``(n_lat_local+1, n_lon, ...)``).  Every rank except the
        northernmost trims its trailing row before send so the
        concatenated global array has shape ``(n_lat_global+1, ...)``
        rather than ``(n_lat_global + n_ranks, ...)``.

    Returns
    -------
    jax.Array on rank 0 (concatenated global array), ``None`` on
    other ranks.  When mpi4py is unavailable returns the input as-is.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return local_arr

    comm = MPI.COMM_WORLD

    arr_to_send = local_arr
    if is_v_face and layout.rank < layout.n_ranks - 1:
        arr_to_send = local_arr[:-1]
    gathered = comm.gather(np.asarray(arr_to_send), root=0)
    if layout.rank == 0:
        return jnp.concatenate(gathered, axis=0)
    return None


def scatter_field_latlon(
    global_arr,
    layout: LatLonBandLayout,
    *,
    is_v_face: bool = False,
):
    """Scatter a rank-0-resident global array to all ranks' bands.

    Mirror of :func:`gather_field_latlon` for the restart path.  Rank 0
    broadcasts the global array; every rank then slices its own band.

    A ``bcast`` of the full global array is wasteful relative to a true
    MPI scatterv, but at AMIP resolutions (1° = 180x360x32, ~5 MB per
    3-D field) the simplicity wins: a restart is rare, and the
    bcast happens once per chained job.

    Parameters
    ----------
    global_arr : jax.Array or numpy.ndarray or None
        Global array on rank 0 (first axis = lat).  Other ranks may
        pass ``None``.
    layout : LatLonBandLayout
    is_v_face : bool
        When True, slice ``[lat_start : lat_end+1]`` so neighbouring
        ranks duplicate the boundary v-face row (same convention as
        :func:`scatter_state_latlon`).

    Returns
    -------
    jax.Array — this rank's band-local view of the global field.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        # Single-process: the caller's global array IS the rank-local
        # band (one rank == one band == the whole world).
        return global_arr

    comm = MPI.COMM_WORLD

    if layout.rank == 0:
        if global_arr is None:
            raise ValueError(
                "scatter_field_latlon: rank 0 must supply the global "
                "array; got None.  Other ranks may pass None."
            )
        payload = np.asarray(global_arr)
    else:
        payload = None
    payload = comm.bcast(payload, root=0)

    s, e = layout.lat_start, layout.lat_end
    band = payload[s : e + 1] if is_v_face else payload[s:e]
    return jnp.asarray(band)


def gather_state_latlon(local_state, layout: LatLonBandLayout):
    """Gather rank-local bands onto rank 0; returns ``None`` elsewhere.

    Thin wrapper over :func:`gather_field_latlon` that handles the
    canonical C-grid lat-lon state NamedTuple (raw jnp arrays, not
    Field-wrapped).  Driver-side callers that hold a Field-wrapped
    ``HydrostaticState`` should call :func:`gather_field_latlon` per
    ``.data`` attribute directly (see
    ``ModelDriver._gather_state_for_global_checkpoint``).
    """
    T_global = gather_field_latlon(local_state.T, layout)
    p_s_global = gather_field_latlon(local_state.p_s, layout)
    phis_global = gather_field_latlon(local_state.phis, layout)
    u_global = gather_field_latlon(local_state.u, layout)
    v_global = gather_field_latlon(local_state.v, layout, is_v_face=True)

    tracers_global = None
    if getattr(local_state, "tracers", None):
        tracers_global = {}
        for name, tr in local_state.tracers.items():
            tracers_global[name] = gather_field_latlon(tr, layout)

    if layout.rank == 0:
        return local_state._replace(
            u=u_global, v=v_global, T=T_global,
            p_s=p_s_global, phis=phis_global,
            tracers=(tracers_global if tracers_global is not None
                     else local_state.tracers),
        )
    return None


# ============================================================================
# Ocean C-grid state scatter / gather  (issue #353)
# ============================================================================
#
# :class:`~legoesm.ocean.state.LatLonCGridOceanState` differs from the
# atmospheric :class:`CGridLatLonHydrostaticState` in two ways that
# matter here:
#   * its fields are immutable ``Field`` objects (no ``__getitem__``) —
#     slice ``.data`` and re-wrap via ``.replace(data=...)``;
#   * it carries C-grid face masks (u_mask, v_mask) and optional moment /
#     AB2 fields that must travel with the band.
# The tripolar fold descriptor lives on the *grid* and is carried on
# ``LatLonBandLayout.fold``; the halo helpers pick it up automatically.


def _field_slice_lat(field, s: int, e: int):
    """Slice a ``Field`` along its leading latitude axis → new ``Field``."""
    return field.replace(data=field.data[s:e])


def _maybe_field_slice_lat(field, s: int, e: int):
    """Like :func:`_field_slice_lat`, but pass ``None`` through unchanged."""
    if field is None:
        return None
    return field.replace(data=field.data[s:e])


def scatter_state_latlon_cgrid_ocean(state, layout: LatLonBandLayout):
    """Extract this rank's latitude band from a global ocean C-grid state.

    Mirror of :func:`scatter_state_latlon` for
    :class:`~legoesm.ocean.state.LatLonCGridOceanState`.  Each rank
    slices the global ``Field`` arrays along latitude; no MPI call (the
    global state is replicated on / broadcast to every rank, then
    sliced — same convention as the atmospheric scatter).

    Stagger-aware slicing
    ---------------------
    * cell-centre scalars (T, S, eta, H_bathy, land_mask, w) and the
      u-face fields (u, u_mask) → rows ``[lat_start, lat_end)``;
    * v-face fields (v, v_mask) → rows ``[lat_start, lat_end+1)`` so
      neighbouring ranks duplicate the boundary v-face row (same
      convention as :func:`scatter_state_latlon`);
    * optional moment / AB2 fields (T_som, S_som, T_flux_div_prev,
      S_flux_div_prev) sliced like cell-centre scalars when present;
    * the optional ``store_mass_flux`` capture (#1442): ``mass_flux_u``
      like ``u`` (u-face, rows ``[s, e)``) and ``mass_flux_v`` like ``v``
      (v-face, rows ``[s, e+1)``).  Omitting them left a POPULATED global
      state un-sliced -- every rank would carry the full-domain flux while
      every other field was band-local (codex RED 4).

    Masks are sliced from the GLOBAL pre-computed masks (NOT recomputed
    locally): ``v_mask`` depends on lat-adjacent cells, so a local
    recompute would be wrong at a band's south edge — slicing the global
    mask is exact and makes scatter∘gather a round-trip identity.
    """
    s, e = layout.lat_start, layout.lat_end
    return state._replace(
        u=_field_slice_lat(state.u, s, e),
        v=_field_slice_lat(state.v, s, e + 1),
        T=_field_slice_lat(state.T, s, e),
        S=_field_slice_lat(state.S, s, e),
        eta=_field_slice_lat(state.eta, s, e),
        H_bathy=_field_slice_lat(state.H_bathy, s, e),
        land_mask=_field_slice_lat(state.land_mask, s, e),
        u_mask=_field_slice_lat(state.u_mask, s, e),
        v_mask=_field_slice_lat(state.v_mask, s, e + 1),
        w=_field_slice_lat(state.w, s, e),
        T_som=_maybe_field_slice_lat(state.T_som, s, e),
        S_som=_maybe_field_slice_lat(state.S_som, s, e),
        T_flux_div_prev=_maybe_field_slice_lat(state.T_flux_div_prev, s, e),
        S_flux_div_prev=_maybe_field_slice_lat(state.S_flux_div_prev, s, e),
        # #1442 store_mass_flux: u-face like ``u``, v-face like ``v``, and the
        # vertical partner cell-centred like ``w`` (it is (n_lat, n_lon,
        # nlev+1) -- staggered in the VERTICAL only, which the lat-band
        # decomposition never splits).
        mass_flux_u=_maybe_field_slice_lat(
            getattr(state, "mass_flux_u", None), s, e),
        mass_flux_v=_maybe_field_slice_lat(
            getattr(state, "mass_flux_v", None), s, e + 1),
        mass_flux_w=_maybe_field_slice_lat(
            getattr(state, "mass_flux_w", None), s, e),
        # store_salt_flux: 2-D column-integrated pair; u-face lat-sliced like
        # ``u``, v-face staggered like ``v`` (n_lat+1 rows -> s, e+1).
        salt_flux_u_int=_maybe_field_slice_lat(
            getattr(state, "salt_flux_u_int", None), s, e),
        salt_flux_v_int=_maybe_field_slice_lat(
            getattr(state, "salt_flux_v_int", None), s, e + 1),
    )


def _gather_field_ocean(field, layout: LatLonBandLayout, *, is_v_face=False):
    """Gather a ``Field``'s data onto rank 0; re-wrap on rank 0, ``None``
    on other ranks.  ``None`` field → ``None``.  When mpi4py is
    unavailable (single process) returns the field unchanged."""
    if field is None:
        return None
    gathered = gather_field_latlon(field.data, layout, is_v_face=is_v_face)
    if gathered is None:
        return None
    return field.replace(data=gathered)


def gather_state_latlon_cgrid_ocean(local_state, layout: LatLonBandLayout):
    """Gather rank-local ocean bands onto rank 0; ``None`` elsewhere.

    Inverse of :func:`scatter_state_latlon_cgrid_ocean`.  v-face fields
    (v, v_mask, and #1442's ``mass_flux_v``) pass ``is_v_face=True`` so
    every rank except the northernmost trims its duplicated boundary row
    before the gather (global v shape ``(n_lat_global+1, n_lon, nlev)``).

    Every slot present on the local state MUST be gathered.  A slot left out
    of the ``_replace`` below keeps rank 0's BAND-LOCAL array in a state
    otherwise assembled to global shape -- shape-inconsistent, and silently
    wrong for any diagnostic that reads it (codex RED 4 on the ``mass_flux_*``
    pair).  ``_gather_field_ocean`` passes ``None`` through, so an unpopulated
    optional slot stays ``None``.
    """
    u_g = _gather_field_ocean(local_state.u, layout)
    v_g = _gather_field_ocean(local_state.v, layout, is_v_face=True)
    T_g = _gather_field_ocean(local_state.T, layout)
    S_g = _gather_field_ocean(local_state.S, layout)
    eta_g = _gather_field_ocean(local_state.eta, layout)
    H_g = _gather_field_ocean(local_state.H_bathy, layout)
    lm_g = _gather_field_ocean(local_state.land_mask, layout)
    um_g = _gather_field_ocean(local_state.u_mask, layout)
    vm_g = _gather_field_ocean(local_state.v_mask, layout, is_v_face=True)
    w_g = _gather_field_ocean(local_state.w, layout)
    tsom_g = _gather_field_ocean(local_state.T_som, layout)
    ssom_g = _gather_field_ocean(local_state.S_som, layout)
    tfd_g = _gather_field_ocean(local_state.T_flux_div_prev, layout)
    sfd_g = _gather_field_ocean(local_state.S_flux_div_prev, layout)
    # #1442 store_mass_flux: u-face like ``u``, v-face like ``v``.  Gathered
    # UNCONDITIONALLY (before the rank-0 branch) like every slot above, so all
    # ranks run the same number of collectives.
    mfu_g = _gather_field_ocean(getattr(local_state, "mass_flux_u", None),
                                layout)
    mfv_g = _gather_field_ocean(getattr(local_state, "mass_flux_v", None),
                                layout, is_v_face=True)
    mfw_g = _gather_field_ocean(getattr(local_state, "mass_flux_w", None),
                                layout)
    sfu_g = _gather_field_ocean(getattr(local_state, "salt_flux_u_int", None),
                                layout)
    sfv_g = _gather_field_ocean(getattr(local_state, "salt_flux_v_int", None),
                                layout, is_v_face=True)
    if layout.rank == 0:
        return local_state._replace(
            u=u_g, v=v_g, T=T_g, S=S_g, eta=eta_g, H_bathy=H_g,
            land_mask=lm_g, u_mask=um_g, v_mask=vm_g, w=w_g,
            T_som=tsom_g, S_som=ssom_g,
            T_flux_div_prev=tfd_g, S_flux_div_prev=sfd_g,
            mass_flux_u=mfu_g, mass_flux_v=mfv_g, mass_flux_w=mfw_g,
            salt_flux_u_int=sfu_g, salt_flux_v_int=sfv_g,
        )
    return None


# ============================================================================
# Ocean geometry / vertical-coordinate band-slicing  (issue #353)
# ============================================================================


def slice_cgrid_geometry_to_band(geom, layout: LatLonBandLayout):
    """Slice a global :class:`LatLonCGridGeometry` to this rank's band.

    Stagger-aware (issue #353 part 5):
      * T-point metrics (n_lat, n_lon) → ``[s:e]``;
      * u-point metrics (n_lat, n_lon+1) → ``[s:e]``;
      * v-point metrics (n_lat+1, n_lon) → ``[s:e+1]`` (shared boundary
        v-row, matching the state scatter);
      * q-point area (n_lat+1, n_lon+1) → ``[s:e+1]``;
      * 1-D lat arrays (cos_lat, sin_lat, lat) → ``[s:e]``; lon unchanged.

    ``n_lat`` becomes the rank-local row count; ``total_area`` keeps the
    GLOBAL value (so area-weighted-mean denominators stay correct on every
    rank).  The fold descriptor is ACTIVE on ALL ranks (so ``is_tripolar()``
    is consistent → same MPI call counts); on non-northernmost ranks,
    ``fold_j`` and ``cap_j`` are set to -1 as a sentinel meaning "fold
    exists but is not locally present."  The serial ``pad_ns_*`` operators
    check ``fold_j >= 0`` via ``fold_is_local()`` to decide whether to
    apply the fold permutation.  ``perm_T`` / ``perm_v`` are
    longitude-only and unaffected by latitude banding.
    """
    s, e = layout.lat_start, layout.lat_end

    def t(a):   # T-point / u-point (leading dim n_lat) → [s:e]
        return a[s:e]

    def vface(a):  # v-point / q-point (leading dim n_lat+1) → [s:e+1]
        return a[s:e + 1]

    area_T_band = t(geom.area_T)

    # Fold: all ranks carry an ACTIVE fold (is_tripolar() must be consistent
    # across ranks to avoid MPI call-count mismatches — issue #356).  On
    # non-northernmost ranks, fold_j and cap_j are set to -1 as a sentinel
    # meaning "fold exists but is not locally present."  The serial
    # pad_ns_* operators use fold_is_local() (checks fold_j >= 0) to
    # decide whether to apply the fold permutation; on non-northernmost
    # ranks they fall through to pad_ns_zero (MPI halo exchange) instead.
    if layout.north_rank is None:
        band_fold = geom.fold
    else:
        band_fold = geom.fold._replace(fold_j=-1, cap_j=-1)

    return geom._replace(
        n_lat=layout.n_lat_local,
        lat_T=t(geom.lat_T), lon_T=t(geom.lon_T),
        dx_T=t(geom.dx_T), dy_T=t(geom.dy_T),
        # total_area is the GLOBAL denominator (area-weighted means); keep the
        # input's global value on every rank rather than a band-local sum, to
        # match slice_latlon_grid_to_band's global-total semantics.
        area_T=area_T_band, total_area=geom.total_area,
        dx_u=t(geom.dx_u), dy_u=t(geom.dy_u),
        dx_v=vface(geom.dx_v), dy_v=vface(geom.dy_v),
        area_q=vface(geom.area_q),
        f_T=t(geom.f_T), f_u=t(geom.f_u), f_v=vface(geom.f_v),
        cos_alpha_u=t(geom.cos_alpha_u), sin_alpha_u=t(geom.sin_alpha_u),
        cos_alpha_v=vface(geom.cos_alpha_v), sin_alpha_v=vface(geom.sin_alpha_v),
        cos_lat=t(geom.cos_lat), sin_lat=t(geom.sin_lat),
        # cos_lat_v is a v-FACE (n_lat+1,) field -> vface(), not t().
        cos_lat_v=vface(geom.cos_lat_v),
        # lat_v is the same v-FACE axis.  Optional (None on builders with no
        # face array, e.g. tripole), and a band that kept the GLOBAL array
        # would then be sharded on the n_lat axis by sharded_ocean_step --
        # caught by test_build_band_grids' per-field stagger audit.
        lat_v=(vface(geom.lat_v) if geom.lat_v is not None else None),
        lat=t(geom.lat),
        fold=band_fold,
        # Partial-periodic seam wall is a per-lat-row (n_lat,) profile —
        # slice to the band like the other T-point fields so it stays
        # aligned with the band-local n_lat (None passes through).
        seam_wall_rows=(t(geom.seam_wall_rows)
                        if geom.seam_wall_rows is not None else None),
        native_lat_T_deg=(t(geom.native_lat_T_deg)
                          if geom.native_lat_T_deg is not None else None),
        # lon, dlon, dlat, radius, n_lon pass through unchanged.
    )


def slice_zcoord_to_band(zcoord, layout: LatLonBandLayout):
    """Slice a vertical coordinate's per-cell T-point arrays to this band.

    Only the known per-cell fields of an ``OceanPartialCellCoordinate``
    (``h_partial``, ``bottom_level``, ``is_active``) are sliced ``[s:e]``;
    reference profiles (``z_full_ref``, ``dz_ref``, …) and scalars
    (``n_levels``, ``H_max``) pass through unchanged.  Slicing explicit
    field names rather than a shape heuristic avoids mis-slicing any
    future auxiliary table whose leading dimension coincidentally equals
    ``n_lat_global``.  A z* / z-level coordinate carrying no per-cell
    fields is returned unchanged.  Issue #353 part 5.
    """
    n_lat = layout.n_lat_global
    s, e = layout.lat_start, layout.lat_end
    updates = {}
    for name in ("h_partial", "bottom_level", "is_active"):
        arr = getattr(zcoord, name, None)
        if (isinstance(arr, (jax.Array, np.ndarray))
                and arr.ndim >= 2 and arr.shape[0] == n_lat):
            updates[name] = arr[s:e]
    if not updates:
        return zcoord
    return zcoord._replace(**updates)


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


def slice_latlon_grid_to_band(grid, layout: LatLonBandLayout,
                              *, skip_total_area_reduce: bool = False):
    """Slice a global ``LatLonGrid`` to this rank's lat band.

    All latitude-dependent metric arrays (1-D: ``lat``, ``cos_lat``,
    ``sin_lat``, ``dy``; 2-D: ``lat2d``, ``lon2d``, ``f``, ``dx``,
    ``area``) are sliced along axis 0 to ``[lat_start:lat_end]``.
    The scalar ``total_area`` is REPLACED with the rank's *global*
    sphere area via ``global_sum_mpi(rank_local_band_area)`` so that
    the mass fixer's uniform p_s correction divides by the right
    denominator (the rank-local sum would be the band area only).

    The ``n_lat`` field on the returned grid is the rank-local row
    count (``layout.n_lat_local``), so downstream code that reads
    ``grid.n_lat`` sees what this rank actually owns.

    Used by
    -------
    * ``ModelDriver.setup()`` when ``grid_type=latlon`` and the
      runtime bootstrap activated MPI — see Stage 3-B in commit
      log.
    * ``make_latlon_mpi_step``'s ``fixer_model`` construction —
      same operation, kept inlined there for the wrapper-function
      entry-point that bypasses ModelDriver.
    * Multi-rank tests in
      ``tests/distributed/test_latlon_mpi_step.py`` (extract from
      the ad-hoc ``_make_local_model`` helper).

    Parameters
    ----------
    grid : LatLonGrid
        Global grid covering the full sphere.
    layout : LatLonBandLayout
        This rank's band.

    Returns
    -------
    LatLonGrid
        Rank-local grid with global ``total_area`` (allreduced).
    """
    s, e = layout.lat_start, layout.lat_end
    band_area = grid.area[s:e, :]
    if skip_total_area_reduce:
        # The caller passed a FULL grid replicated on every rank (e.g. the MG
        # coarse-grid construction): the global total is just the local sum of
        # ALL rows, with NO MPI collective — so the banded-MG preconditioner
        # build stays reduction-free (codex MED, the whole point of banded MG).
        global_total = jnp.sum(grid.area)
    else:
        from legoesm.parallel.reductions import global_sum_mpi
        band_total = jnp.sum(band_area)
        # Allreduce → global sphere area.  Under the local backend
        # this is a no-op (returns the local value unchanged), so the
        # function also works correctly at single-rank.
        global_total = global_sum_mpi(band_total)
    return grid._replace(
        n_lat=layout.n_lat_local,
        lat=grid.lat[s:e],
        lat2d=grid.lat2d[s:e, :],
        lon2d=grid.lon2d[s:e, :],
        cos_lat=grid.cos_lat[s:e],
        sin_lat=grid.sin_lat[s:e],
        # v-face arrays span ``[lat_start, lat_end+1)`` so each rank
        # holds the v-face row shared with its northern neighbour.
        # Matches the v-state-array slicing in ``scatter_state_latlon``.
        # Without this, the polar-filter v-face mask would be sized
        # against the GLOBAL v-face axis on every rank.
        lat_v=grid.lat_v[s:e + 1],
        cos_lat_v=grid.cos_lat_v[s:e + 1],
        dy=grid.dy[s:e],
        f=grid.f[s:e, :],
        dx=grid.dx[s:e, :],
        area=band_area,
        total_area=global_total,
    )


def slice_latlon_grid_to_block_2d(
    grid, layout: LatLon2DLayout, *, skip_total_area_reduce: bool = False,
):
    """Slice a global ``LatLonGrid`` to this rank's 2-D lat x lon block.

    The 2-D analog of :func:`slice_latlon_grid_to_band`: every lat-dependent
    array is sliced ``[lat_start:lat_end]`` AND every lon-dependent array
    ``[lon_start:lon_end]``.  v-face lat arrays span ``[lat_start:lat_end+1]``
    (the shared boundary row, matching :func:`scatter_state_latlon_2d`).
    ``n_lat``/``n_lon`` become the rank-local counts.  ``total_area`` is the
    GLOBAL sphere area (allreduce of the block area; or the full passed grid's
    area when ``skip_total_area_reduce`` — e.g. a replicated full grid) so the
    mass fixer's uniform correction divides by the right denominator.
    ``proc_lon == 1`` reproduces :func:`slice_latlon_grid_to_band`.
    """
    s, e = layout.lat_start, layout.lat_end
    w, x = layout.lon_start, layout.lon_end
    if skip_total_area_reduce:
        global_total = jnp.sum(grid.area)
    else:
        from legoesm.parallel.reductions import global_sum_mpi
        global_total = global_sum_mpi(jnp.sum(grid.area[s:e, w:x]))
    return grid._replace(
        n_lat=layout.n_lat_local,
        n_lon=layout.n_lon_local,
        lat=grid.lat[s:e],
        lon=grid.lon[w:x],
        lat2d=grid.lat2d[s:e, w:x],
        lon2d=grid.lon2d[s:e, w:x],
        cos_lat=grid.cos_lat[s:e],
        sin_lat=grid.sin_lat[s:e],
        lat_v=grid.lat_v[s:e + 1],
        cos_lat_v=grid.cos_lat_v[s:e + 1],
        f=grid.f[s:e, w:x],
        dx=grid.dx[s:e, w:x],
        dy=grid.dy[s:e],
        area=grid.area[s:e, w:x],
        total_area=global_total,
    )


def pole_v_bc_for_layout(layout: LatLonBandLayout) -> tuple[bool, bool]:
    """Return the ``(south_pole, north_pole)`` flags for a band.

    True at an end means that end of this rank's band is the *actual*
    global pole — the wall-BC v=0 enforcement must fire there.
    False means that end is an interior partition cut (shared with a
    neighbouring rank's v-row) and must NOT be zeroed.

    Used to set ``CGridLatLonPrimitiveEquationConfig.pole_v_bc``.
    """
    return (layout.south_rank is None, layout.north_rank is None)


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
            # v-face arrays span [lat_start, lat_end+1) — same as
            # slice_latlon_grid_to_band.  Every n_lat _replace MUST keep
            # lat_v consistent (n_lat+1): lat_v_interfaces() consumes it
            # for the PPM meridional flux metric (codex M3a findings 2/6).
            lat_v=grid.lat_v[layout.lat_start:layout.lat_end + 1],
            cos_lat_v=grid.cos_lat_v[layout.lat_start:layout.lat_end + 1],
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

    # The rebuilt grid carries the ORIGINAL grid's rotation rate:
    # LatLonGrid now stores the construction scalar (``grid.omega``,
    # #521), exact on any band.  The prior recovery from the local
    # Coriolis field (f[mid,0] / (2*sin_lat[mid])) was 0/0 -> NaN
    # whenever the selected mid row sat on the equator — odd-n_lat
    # global grids and small equatorial bands (codex 2026-07-03
    # round-3 HIGH) — and a non-rotating omega=0 grid hit the same
    # 0/0 at ANY row.
    omega_eff = float(grid.omega)

    # Delegate the metric construction to the shared helper so the
    # serial create_latlon_grid path and this MPI extension stay
    # algebraically identical.  Interior cells of the extended grid
    # come out bit-for-bit equal to the original grid (same formulas,
    # same lat values); halo cells are freshly computed from the
    # extrapolated lat.
    from legoesm.grids.latlon import build_uniform_latlon_grid_from_axes
    return build_uniform_latlon_grid_from_axes(
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
    halo: int = 2,  # kept for backward-compat signature; unused now
    physics_fn: Callable | None = None,
) -> Callable:
    """Build an MPI-aware step function for the lat-lon C-grid dycore.

    Thin backend-aware wrapper (Step 4 of the Option-(ii) refactor).
    Previous iterations of this function pre-padded state on a padded
    grid and ran a "padded model"; that approach hit an architectural
    mismatch at the pole because the dycore's operator-internal
    pole-BC pads were applied at the *padded* array's outer faces, not
    at the actual global pole rows.  The new design activates the
    backend-dispatched
    :func:`legoesm.grids.halo_latlon.pad_halo_latlon` family — so
    every operator that needs halo data fetches it via inter-rank
    sendrecv at partition cuts and pole-fold / wall-BC constants at
    boundary ranks, with no pre-padding of state on the caller side.

    Architecture
    ------------
    1. ``set_halo_backend("mpi", layout)`` is activated once at
       factory time.  Every subsequent ``pad_halo_latlon*``,
       ``pad_ns_zero``, and ``pad_with_pole_bc_lat`` call inside the
       dycore operators dispatches through the MPI sendrecv path.
       ``is_distributed()`` returns True, which gates the
       allreduce inside ``batch_global_area_sums`` and
       ``zero_mean_tendency`` (the mass-fixer reductions).

    2. A rank-local ``mpi_model`` is built once:

       * **Grid** — same rank-local grid as ``model.grid`` *except*
         ``total_area`` is replaced with the global sphere area
         (``global_sum_mpi`` over the rank-local band area).  This is
         what the mass fixer's uniform-correction divisor needs.
         The per-cell ``area`` field stays rank-local.

       * **Config** — ``pole_v_bc = (south_rank is None, north_rank
         is None)`` so only the boundary ranks zero the actual pole
         rows; interior ranks leave their band-edge v-rows alone
         (those are interior v-faces shared with the neighbour rank
         and kept consistent by halo exchange).  ``pole_v_bc_offset``
         stays 0 since we're operating on rank-local state, not on
         a pre-padded array.

    3. ``step_fn`` is a one-line delegate to ``mpi_model._step_cgrid``.
       No padding, no stripping, no external fixer call — the
       backend-aware operators inside ``_step_cgrid`` handle all the
       halo plumbing.

    Column physics is supported via ``physics_fn`` (forwarded per RK
    stage to ``_step_cgrid``, exactly like the serial
    ``model.step(state, dt, physics_fn=...)`` path).  Prescribed-SST
    scatter for AMIP remains the ``ModelDriver`` path's job.

    Parameters
    ----------
    model : CGridLatLonPrimitiveEquationModel
        Built against the **rank-local** grid (sliced from the
        global grid for this band).  ``model.grid.total_area`` may be
        either the band area or the sphere area — the wrapper ignores
        it and recomputes the global total by allreducing
        ``sum(model.grid.area)`` (the per-cell band area is
        unambiguous; re-allreducing an already-global ``total_area``
        scalar would double-count by ``n_ranks``).
    layout : LatLonBandLayout
    halo : int, default 2
        Kept for backward-compatible signature.  No longer used —
        the operator-internal halo widths (1 for compact stencils,
        2 for PPM and biharmonic) are determined by each operator
        as it calls ``pad_halo_latlon(field, halo=k)``.
    physics_fn : callable or None, default None
        Column-physics callable forwarded unchanged to
        ``_step_cgrid`` (where it is a *static* jit argument —
        pass a stable module-level function or a long-lived closure,
        never a fresh lambda per step, or every step recompiles).
        Same contract as the serial ``model.step``: either the
        3-arg form ``physics_fn(hs_state, grid, sigma_coord)`` or a
        1-arg closure ``physics_fn(hs_state)`` (see
        ``CGridLatLonPrimitiveEquationModel._call_physics``).  Under
        band MPI it receives the **rank-local** cell-centred
        ``HydrostaticState`` view plus the rank-local band grid /
        sigma — it must be column-local (no global-lat assumptions).
        Dynamics tendencies in the same RK stage are computed from
        backend-aware halo-exchanged operators on the same rank-local
        fields, so physics and dynamics see a consistent state.
        ``None`` (default) keeps the previous dynamics-only
        behaviour.

    Returns
    -------
    step_fn : callable
        ``step_fn(local_state, dt, *, target_mass=None) -> local_state``.

    Side effect
    -----------
    Activates ``set_halo_backend("mpi", layout)`` at the global
    module level.  Caller is responsible for resetting via
    ``set_halo_backend("local")`` once the run is finished — see
    ``ModelDriver._finalize_run`` for the cubed-sphere precedent.
    Tests that share a Python process should reset between cases.

    Note on ``halo`` kwarg
    ----------------------
    The previous (pre-pad) implementation took ``halo`` to size its
    padded state buffer.  In the new design no buffer exists — each
    operator picks its own halo width from the
    backend-dispatched ``pad_halo_latlon(..., halo=k)`` call.  The
    kwarg is retained for signature stability with existing test
    fixtures and the (now legacy) tests/parallel/test_latlon_mpi_step_serial.py.
    """
    # The MPI model is re-instantiated from the SAME class as the passed-in
    # ``model`` via ``type(model)`` — so this shared-substrate ``parallel`` module
    # does NOT import the atmosphere dycore (preserving component independence;
    # parallel must not reach up into a component).  The Stage-0 halo tests also
    # stay importable without the heavy dycore stack.
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.reductions import global_sum_mpi

    # Stage 3-E (commit 8b736e04): polar filter is now MPI-safe.  The
    # filter algorithm is lon-only FFT (``jnp.fft.rfft`` along axis -1)
    # so each rank applies it independently on its own band — no MPI
    # exchange is required for the filter itself.  Slice-equivariance
    # of the mask + filter output is pinned by
    # ``tests/distributed/test_latlon_mpi_polar_filter.py``.

    # Activate the MPI halo backend.  Every pad_halo_latlon /
    # pad_with_pole_bc_lat call inside the dycore now dispatches
    # through inter-rank sendrecv at partition cuts and constant /
    # pole-fold pads at boundary ranks.  This also flips
    # ``is_distributed()`` so the mass-fixer reductions allreduce.
    set_halo_backend("mpi", layout)
    from legoesm.grids.polar_filter import (
        get_polar_filter_lon_gather,
        set_polar_filter_lon_gather,
    )

    # Build the MPI-aware model: rank-local grid with allreduced
    # ``total_area`` (mass-fixer divisor); pole_v_bc tracks which
    # ends of the band touch actual global poles.
    #
    # Derive the global total from the rank-local PER-CELL ``area``
    # (unambiguous on every path) rather than allreducing the
    # ``total_area`` scalar: callers arrive with ``total_area`` either
    # band-local (tests' ``_make_local_model``) or already global
    # (``slice_latlon_grid_to_band``), and allreducing an
    # already-global scalar would give ``n_ranks * sphere_area`` — a
    # silently wrong mass-fixer denominator (Codex P1-fix review,
    # MAJOR 1).
    global_total_area = global_sum_mpi(jnp.sum(model.grid.area))
    mpi_grid = model.grid._replace(total_area=global_total_area)
    mpi_config = model.config._replace(
        pole_v_bc=(
            layout.south_rank is None,
            layout.north_rank is None,
        ),
        # No padding → no offset.  The dycore zeros v at v[0] / v[-1]
        # of its rank-local input — those ARE the band's lat ends,
        # which are the actual poles ONLY at pole-touching ranks
        # (handled by the pole_v_bc tuple above).
        pole_v_bc_offset=0,
    )
    # Use the SAME ``dt`` the input model was constructed with so the
    # polar-filter mask is sized identically to the serial reference
    # (Codex Stage 3-E round 2 BLOCK #2: ``_max_dt`` is the pole-cell
    # CFL — using it here would size the mask for the tiny pole-CFL
    # dt instead of the actual production dt, masking modes the
    # production dt does not require).  ``model.effective_dt`` is the
    # canonical post-clamp dt set by ``component_factory``;
    # ``model.dt`` is the constructor arg the model stashed.  Prefer
    # effective_dt and fall back to the constructor dt.
    _mpi_dt = getattr(
        model, "effective_dt",
        getattr(model, "dt", None),
    )
    if _mpi_dt is None:
        _mpi_dt = getattr(model, "_max_dt", 600.0)
    # Same class as the passed-in model — no atmosphere-dycore import needed.
    mpi_model = type(model)(
        grid=mpi_grid,
        sigma_coord=model.sigma_coord,
        config=mpi_config,
        dt=_mpi_dt,
    )

    # Issue #405/#413: this wrapper does not thread the PhysicsState
    # carry between steps — a stateful physics_fn would silently reseed
    # its prognostic fields every step.  Refuse loudly at build time.
    # The shared predicate also sees a stateful physics through a
    # functools.partial / __wrapped__ wrapper that hides the tag.
    from legoesm.timestepping.integration import (
        physics_requires_phys_state,
    )
    if physics_requires_phys_state(physics_fn):
        raise NotImplementedError(
            "make_latlon_mpi_step does not thread the PhysicsState "
            "carry yet, so a stateful physics_fn (prognostic TKE-family "
            "/ MYNN-2.5 turbulence, mass_flux/edmf/bechtold convection, "
            "prognostic-spectral GWD) would silently reseed every step "
            "(issue #405/#413).  Use a diagnostic scheme or the "
            "ModelDriver loops, which thread the carry."
        )

    def step_fn(local_state, dt, *, target_mass=None):
        # No pre-pad, no strip.  The operators inside ``_step_cgrid``
        # call backend-aware halo helpers that, under
        # ``_halo_backend == "mpi"``, MPI-sendrecv with neighbouring
        # ranks for halo cells at partition cuts and apply the
        # serial pole-fold / wall-BC constants at boundary ranks.
        # The mass fixer's allreduce fires through the same
        # ``is_distributed()`` gate.  ``physics_fn`` (static jit arg,
        # closure-captured so its identity is stable across steps) is
        # evaluated inside each RK stage on the rank-local state —
        # identical calling convention to the serial
        # ``model.step(state, dt, physics_fn=...)`` delegate.
        # ``_step_cgrid`` returns ``(state_new, phys_state_out)`` since
        # the #413 carry threading; no carry is threaded here (guarded
        # above), so unpack and return the state to preserve this
        # wrapper's documented contract.
        # Full lon per rank: the polar filter runs LOCALLY.  Save/set-None/
        # restore the lon-gather context around the jitted ``_step_cgrid``
        # trace so a band step traced after a 2-D step reads None, and this
        # step does not leak None to a later user (codex C7 P1/(a)).
        _prev = get_polar_filter_lon_gather()
        set_polar_filter_lon_gather(None, None)
        try:
            state_new, _ = mpi_model._step_cgrid(
                local_state, dt,
                target_mass=target_mass,
                physics_fn=physics_fn,
            )
        finally:
            set_polar_filter_lon_gather(*(_prev if _prev else (None, None)))
        return state_new

    return step_fn


def make_latlon_2d_mpi_step(
    model,
    layout: LatLon2DLayout,
    *,
    physics_fn: Callable | None = None,
) -> Callable:
    """Build an MPI step for the lat-lon C-grid dycore on a 2-D pencil.

    The 2-D ``(proc_lat × proc_lon)`` analogue of :func:`make_latlon_mpi_step`.
    Identical architecture — activate the MPI halo backend once with the
    :class:`LatLon2DLayout`, build a rank-local model with the global
    sphere area (mass-fixer divisor) and ``pole_v_bc`` tracking which lat
    ends touch a physical pole, then delegate each step to
    ``model._step_cgrid`` whose backend-aware operators fetch halo data via
    the 2-D dispatch (``pad_halo_latlon`` → :func:`pad_halo_latlon_2d`,
    ``pad_with_pole_bc_lat`` → :func:`pad_with_pole_bc_lat_2d`,
    ``zero_polar_lat_ends`` → the 2-D pole-touch test).

    Wall poles only — a labeled midlatitude throughput benchmark, NOT the
    atmosphere's 180° pole fold (that needs a lat-pencil transpose; see
    ``docs/performance/scaling/latlon_2d_build_plan.md``).

    ``proc_lon == 1`` (latitude split only) is the band-EQUIVALENT layout
    and is fully wired: the lat axis is exchanged via the 2-D dispatch and
    longitude is full on every rank, so the operators' local longitude
    wraps are correct.  This case reproduces :func:`make_latlon_mpi_step`
    and is the single-process / 2×1 conservation gate.

    ``proc_lon > 1`` (a genuine LONGITUDE split) is now WIRED for REGULAR /
    wall-pole grids: the C-grid operators route their longitude halo through
    the dispatched
    :func:`legoesm.grids.operators_latlon_cgrid.pad_lon_cgrid`
    (``exchange_halo_lon`` under a 2-D layout, bit-identical to the local wrap
    at ``proc_lon == 1``), so ``gradient_x_cgrid`` / ``interp_cell_to_uface`` /
    ``curl_vertex_cgrid`` / the mask ops span longitude partition cuts.
    TRIPOLAR grids are still REFUSED at ``proc_lon > 1``: the curl
    tripolar-cap fold keeps a local lon roll, and the 180° fold under a
    longitude split needs the lat-pencil transpose (fail loud, not a silent
    wrong-fold — the same doctrine as the ``pole_bc="fold"`` guard in
    :func:`pad_halo_latlon_2d`).

    Parameters / contract are otherwise identical to
    :func:`make_latlon_mpi_step` (rank-local ``model``; ``physics_fn`` a
    stable column-local closure; ``step_fn(local_state, dt, *,
    target_mass=None)``).
    """
    # Regular / wall-pole grids: proc_lon>1 is wired (operators route their
    # lon halo through pad_lon_cgrid -> exchange_halo_lon).  TRIPOLAR grids
    # still keep a local lon roll in the curl tripolar-cap fold, so refuse
    # them under a longitude split — fail loud rather than ship a silent
    # wrong-fold.  Key on ``fold.is_active`` ALONE (a grid-global property,
    # identical on every rank) — NOT ``fold_j`` (the rank-LOCAL fold-row
    # index, -1 on ranks that don't own the seam): a fold_j-keyed guard would
    # raise only on the seam-owning rank while the others enter the MPI
    # collectives and DEADLOCK (codex — SPMD-asymmetric raise).
    _fold = getattr(model.grid, "fold", None)
    _is_tripolar = _fold is not None and bool(getattr(_fold, "is_active", False))
    if layout.proc_lon > 1 and _is_tripolar:
        raise NotImplementedError(
            "make_latlon_2d_mpi_step: proc_lon>1 on a TRIPOLAR grid is not "
            "wired — curl_vertex_cgrid's tripolar-cap fold keeps a local "
            "jnp.roll(..., axis=1); the 180-deg fold under a longitude split "
            "needs the lat-pencil transpose.  Use proc_lon=1 for tripolar, or "
            "a regular / wall-pole grid for the 2-D throughput benchmark.  "
            f"Got proc_lon={layout.proc_lon}, proc_lat={layout.proc_lat}."
        )
    # The polar filter is a per-latitude-row LONGITUDE rFFT, so under a
    # longitude split (proc_lon>1) it needs the whole lon circle on one rank.
    # WIRED via the AD-safe lat-pencil transpose (``lon_gather_full`` /
    # ``lon_scatter_full``): the filter gathers full lon, rFFT/mask/irFFTs on
    # it, then scatters this rank's block.  The mask is rebuilt at the GLOBAL
    # n_lon below (rank-local lat rows, global n_freq), and the gather/scatter
    # pair is injected into ``polar_filter`` before the step is traced
    # (proc_lon==1 keeps full lon per rank -> no gather, unchanged).  Tripolar
    # is still refused above (the cap-fold needs the same transpose on the
    # curl, a separate follow-up).
    _polar_lon_split = (layout.proc_lon > 1 and bool(
        getattr(model.config, "use_polar_filter", False)))
    # Validate the EQUAL-split precondition of the polar-filter lon-gather
    # allgather HERE — before any mutation (set_halo_backend / the global
    # area allreduce / the model build) — so a rejected uneven split leaves
    # the process UNCONFIGURED (codex C7 P2), not half-set-up.
    if _polar_lon_split and layout.n_lon_global % layout.proc_lon != 0:
        raise NotImplementedError(
            "make_latlon_2d_mpi_step: use_polar_filter=True under a longitude "
            f"split needs an EQUAL split (n_lon={layout.n_lon_global} % "
            f"proc_lon={layout.proc_lon} != 0) — the polar-filter lon-gather "
            "allgather requires equal blocks.  Use a proc_lon that divides "
            "n_lon, or use_polar_filter=False."
        )

    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.reductions import global_sum_mpi
    from legoesm.timestepping.integration import (
        physics_requires_phys_state,
    )

    set_halo_backend("mpi", layout)

    global_total_area = global_sum_mpi(jnp.sum(model.grid.area))
    mpi_grid = model.grid._replace(total_area=global_total_area)
    mpi_config = model.config._replace(
        pole_v_bc=(
            layout.south_rank is None,
            layout.north_rank is None,
        ),
        pole_v_bc_offset=0,
    )
    _mpi_dt = getattr(
        model, "effective_dt",
        getattr(model, "dt", None),
    )
    if _mpi_dt is None:
        _mpi_dt = getattr(model, "_max_dt", 600.0)
    mpi_model = type(model)(
        grid=mpi_grid,
        sigma_coord=model.sigma_coord,
        config=mpi_config,
        dt=_mpi_dt,
    )

    if physics_requires_phys_state(physics_fn):
        raise NotImplementedError(
            "make_latlon_2d_mpi_step does not thread the PhysicsState "
            "carry yet, so a stateful physics_fn would silently reseed "
            "every step (issue #405/#413).  Use a diagnostic scheme or the "
            "ModelDriver loops, which thread the carry."
        )

    from legoesm.grids.polar_filter import (
        get_polar_filter_lon_gather,
        set_polar_filter_lon_gather,
    )
    _polar_ctx = (None, None)
    if _polar_lon_split:
        # Rebuild the polar-filter masks at the GLOBAL n_lon (the rank-local
        # ``mpi_grid`` carries the LOCAL n_lon, but the filter rFFTs the
        # gathered full-lon circle).  Latitude terms stay rank-local (the
        # owned rows).  Build the AD-safe lon gather/scatter closures so the
        # dycore's ``fourier_filter`` call sites transparently transpose.
        from legoesm.grids.polar_filter import compute_polar_filter_mask
        _pf_kw = dict(
            dt=_mpi_dt,
            max_wave_speed=mpi_config.polar_filter_max_wave_speed,
            cutoff_lat_deg=mpi_config.polar_filter_cutoff_deg,
            n_lon_override=layout.n_lon_global,
        )
        mpi_model._polar_mask = compute_polar_filter_mask(mpi_grid, **_pf_kw)
        mpi_model._polar_mask_v = compute_polar_filter_mask(
            mpi_grid, is_v_face=True, **_pf_kw)
        _row_comm = make_lon_row_comm(layout)

        def _gather_full(field, _layout=layout, _rc=_row_comm):
            return lon_gather_full(field, _layout, _rc)

        def _scatter_block(full, _layout=layout):
            return lon_scatter_full(full, _layout)

        _polar_ctx = (_gather_full, _scatter_block)

    def step_fn(local_state, dt, *, target_mass=None):
        # SAVE the caller's lon-gather context, install THIS step's for the
        # jitted ``_step_cgrid`` trace, and RESTORE it in finally — so the
        # filter's process-global (read at first-call trace time) is correct
        # for this step regardless of build/call interleaving AND does not
        # leak to a later unwrapped ``model.step`` / shallow-water / ocean
        # filter user (codex C7 P1/(a)).  ``(None, None)`` = local rFFT.
        _prev = get_polar_filter_lon_gather()
        set_polar_filter_lon_gather(*_polar_ctx)
        try:
            state_new, _ = mpi_model._step_cgrid(
                local_state, dt,
                target_mass=target_mass,
                physics_fn=physics_fn,
            )
        finally:
            set_polar_filter_lon_gather(*(_prev if _prev else (None, None)))
        return state_new

    return step_fn
