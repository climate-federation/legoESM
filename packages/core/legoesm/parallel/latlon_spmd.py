"""Single-controller SPMD halo for the lat-lon grid (multi-GPU, no mpi4jax).

The ocean lat-lon C-grid (and atm lat-lon) shard cleanly by LATITUDE BAND: each
device owns a contiguous lat band and the FULL longitude circle (lon is periodic
and kept local — the audit's "every rank owns all longitudes").  The halo is
therefore 1-D over the ``"lat"`` mesh axis:

  * longitude: periodic wrap — LOCAL ``jnp.pad(mode="wrap")`` (no comm), exactly
    like the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon_local`.
  * latitude interior band cuts: ``jax.lax.ppermute`` of the ``halo`` edge rows
    between adjacent bands (north neighbour's bottom rows / south neighbour's
    top rows).
  * poles: the end bands (axis_index 0 = south, N-1 = north) have no neighbour
    there, so they fold their OWN pole rows (mirror in lat + 180 deg in lon),
    selected by a ``jnp.where`` on the band index — bit-identical to the serial
    pole fold.

This is the lat-lon analogue of the cubed-sphere
``cubesphere_exchange.make_tiled_pad_body`` (which the cube SPMD uses), and the
foundation for the ocean lat-lon multi-GPU SPMD step (pure-jax ppermute over the
RTX8000 PCIe pair — no mpi4jax dependency).  Validated by BIT-IDENTITY vs the
serial local pad (``tests/parallel/test_latlon_spmd_halo.py``), the proven
methodology.  ``check_vma=False`` follows the cube SPMD halo bodies.
"""
from __future__ import annotations

import re
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from legoesm.parallel.shard_map_compat import shard_map

# Reference halo depth for the topology chooser's pole-fold cost model: the
# widest production halo (the PPM halo-2 exchange).  The chooser's score is
# otherwise normalized per unit halo depth; the partner fold's h-quadratic
# E/W-extension strips are charged at this reference so the constant stays
# honest without threading the runtime halo through the chooser (see
# choose_latlon_2d_topology).
_FOLD_REF_HALO = 2


def _resolve_halo_nocomm(env_value: str) -> bool:
    """Resolve ``LEGOESM_LATLON_HALO_NOCOMM``: ``'1'`` replaces every lat-lon
    halo ``ppermute`` with an optimization barrier; ``''``/``'0'`` off
    (default).

    A MEASUREMENT knob that DELIBERATELY BREAKS THE ANSWER -- each device
    keeps its OWN edge rows instead of the neighbour's, so every ghost row is
    wrong and the run is meaningless as physics.  It exists because the step
    time alone cannot be split: the profiler on this stack does not record
    the halo collectives, so ``full - nocomm`` is the only way to price the
    wire time plus the wait for the slowest peer against the SAME program.
    Every other line -- the pack, the concatenate, the split, the pole fold,
    the wall constants, the kernel count and the shapes -- is unchanged.

    The barrier (rather than a bare identity) is load-bearing: without it the
    receive-side slices fold back to the send-side buffer and XLA deletes the
    pack, so the arm would time a different program.  Same lesson as
    ``sharded_dynamics._resolve_halo_nocomm``, whose MPAS knob this mirrors.

    Never valid in production.  Unknown values raise (dispatch hardening).
    """
    if env_value in ("", "0"):
        return False
    if env_value == "1":
        return True
    raise ValueError(
        f"LEGOESM_LATLON_HALO_NOCOMM={env_value!r}: must be '0' or '1' "
        f"(empty = off). It is a timing knob that BREAKS the answer; "
        f"a typo must not silently enable it.")


#: Canonical decimal integer, no sign / whitespace / underscores / leading
#: zeros -- the spellings ``int()`` would silently accept.
_CANONICAL_INT_RE = re.compile(r"[1-9][0-9]*")


def _resolve_halo_ballast(env_value: str) -> int:
    """Resolve ``LEGOESM_LATLON_HALO_BALLAST``: wire-payload multiplier for
    the lat-band exchange; ``''``/``'1'`` = off (default).

    A MEASUREMENT knob, never a production one, and unlike the no-comm knob
    it is bit-identical: each message is sent ``N`` times and the copies are
    dropped on receipt, so the bytes scale while the collective count, the
    schedule, the arithmetic and the answer do not.  The step's response to
    ``N`` IS the bandwidth term.

    Why it is needed here: the lat-band decomposition gives every device a
    FIXED halo -- two rows of the whole longitude circle -- no matter how
    many devices there are, yet the measured communication time doubled
    between 32 and 64 devices (0.676 to 1.483 ms).  Constant bytes and
    rising time means the cost is per-operation, not payload, and that
    distinction decides whether a two-dimensional decomposition (which cuts
    bytes but adds collectives) can help at all.

    Unknown or non-canonical values raise (dispatch hardening): a typo must
    not silently run a different payload multiple.
    """
    if env_value in ("", "1"):
        return 1
    if _CANONICAL_INT_RE.fullmatch(env_value) is None:
        raise ValueError(
            f"LEGOESM_LATLON_HALO_BALLAST={env_value!r}: must be a canonical "
            f"decimal integer >= 1 (empty or '1' = off)")
    n = int(env_value)
    if not 1 <= n <= 8:
        raise ValueError(
            f"LEGOESM_LATLON_HALO_BALLAST={n}: out of range 1..8. It "
            f"multiplies every halo message, so a large value runs the node "
            f"out of memory rather than measuring anything.")
    return n


def _halo_ppermute(x, axis_name, perm):
    """Halo ``ppermute``, or one of the two timing substitutes.

    Single choke point for every lat-lon SPMD halo exchange so a budget arm
    covers all of them at once.  Both knobs resolve at TRACE time (a Python
    branch on a static env value), so the compiled program contains one path
    or the other, never a select.

    ``LEGOESM_LATLON_HALO_NOCOMM=1`` drops the collective entirely (wrong
    answers, times the program without communication).
    ``LEGOESM_LATLON_HALO_BALLAST=N`` sends N copies and keeps the first
    (bit-identical answers, times the payload slope).
    """
    import os

    if _resolve_halo_nocomm(os.environ.get("LEGOESM_LATLON_HALO_NOCOMM", "")):
        return jax.lax.optimization_barrier(x)
    ballast = _resolve_halo_ballast(
        os.environ.get("LEGOESM_LATLON_HALO_BALLAST", ""))
    if ballast > 1:
        rows = x.shape[0]
        recv = jax.lax.ppermute(
            jnp.concatenate([x] * ballast, axis=0), axis_name, perm)
        # The kept slice is the same buffer that would have been sent, so
        # the answer is unchanged; only the bytes on the wire scale.
        return recv[:rows]
    return jax.lax.ppermute(x, axis_name, perm)


def latlon_band_perms(n_dev: int):
    """Static (src, dst) permutation pairs over the 1-D ``lat`` band axis.

    ``perm_north``: each band ``b`` receives band ``b+1``'s bottom rows as its
    NORTH ghost -> source ``b+1`` sends to ``b`` (pairs ``(s, s-1)``); the top
    band (``N-1``) is not a destination -> its north_recv is zeros, replaced by
    the north pole fold.  ``perm_south``: band ``b`` receives band ``b-1``'s top
    rows as its SOUTH ghost -> ``(s, s+1)``; the bottom band (0) gets zeros ->
    south pole fold.

    PUBLIC (issue #353 SPMD step): the ocean lat-band SPMD wrapper
    (``ocean.dynamics.sharded_ocean_step``) reuses ``perm_north`` to lift the
    staggered-v north boundary row from band ``r+1`` (the no-private-cross-import
    rule — promoted from ``_latlon_band_perms``).
    """
    perm_north = tuple((s, s - 1) for s in range(1, n_dev))   # send up->down
    perm_south = tuple((s, s + 1) for s in range(0, n_dev - 1))  # down->up
    return perm_north, perm_south


def latlon_lon_ring_perms(p_lon: int):
    """Static (src, dst) permutation pairs over the periodic ``lon`` ring axis.

    Longitude is a periodic RING (unlike the pole-terminated lat LINE), so
    every tile is both a source and a destination — the perms are full cyclic
    permutations with NO non-target zeros.

    ``perm_to_west``: source ``s`` sends to its WEST neighbour
    ``(s-1) mod p`` — the receiver ``j`` gets tile ``j+1``'s payload, i.e.
    its EAST ghost.  ``perm_to_east``: source ``s`` sends to its EAST
    neighbour ``(s+1) mod p`` — the receiver gets its WEST ghost.
    """
    perm_to_west = tuple((s, (s - 1) % p_lon) for s in range(p_lon))
    perm_to_east = tuple((s, (s + 1) % p_lon) for s in range(p_lon))
    return perm_to_west, perm_to_east


def lon_ring_ghosts_spmd(f, mesh, halo: int = 1):
    """Periodic LONGITUDE ghosts (axis 1) under the 2-D lat-lon SPMD mesh.

    The SPMD twin of the local ``jnp.pad(mode="wrap")`` lon wrap (band path)
    and of the MPI 2-D pencil's ``exchange_halo_lon``: each tile owns a lon
    SECTOR, so its east/west ghost columns are the neighbouring tiles' edge
    columns, moved by ``jax.lax.ppermute`` over the ``"lon"`` mesh axis (the
    periodic wrap IS the cyclic ring permutation —
    :func:`latlon_lon_ring_perms`).  ``p_lon == 1`` (a degenerate lon axis /
    the 1-D-band-equivalent (N, 1) mesh) is the LOCAL wrap, chosen by a
    STATIC Python branch — bit-identical to the band path's ``jnp.pad`` (no
    collective is emitted at all).

    Lon axis is axis 1; the field must be CELL-ALIGNED in lon (width
    ``n_lon_local`` — never an ``n_lon_local+1`` u-face field, whose seam
    column would double-count under the ring shift; reconstruct u-faces via
    :func:`reconstruct_uface_left` instead).  2-D and 3-D fields supported
    (trailing axes ride through).  AD-safe: ``ppermute`` is
    self-transposing.  MUST be called INSIDE a shard_map over a mesh
    carrying the ``"lon"`` axis when ``p_lon > 1``.
    """
    if halo <= 0:
        return f
    p_lon = int(mesh.shape["lon"])
    if p_lon == 1:
        pad = ((0, 0), (halo, halo)) + ((0, 0),) * (f.ndim - 2)
        return jnp.pad(f, pad, mode="wrap")
    perm_to_west, perm_to_east = latlon_lon_ring_perms(p_lon)
    # My EAST ghost = east neighbour's west edge (sources send WEST edges to
    # their west neighbour); my WEST ghost = west neighbour's east edge.
    east_ghost = _halo_ppermute(f[:, :halo], "lon", perm_to_west)
    west_ghost = _halo_ppermute(f[:, -halo:], "lon", perm_to_east)
    return jnp.concatenate([west_ghost, f, east_ghost], axis=1)


def reconstruct_vface_lower(v_lower, axis: str, perm_north):
    """Rebuild the ``n_lat+1`` staggered v-faces from the ``n_lat``-row
    ``v_lower`` representation, INSIDE a ``shard_map`` over ``axis``.

    The staggered meridional velocity ``v`` has a leading dim ``n_lat+1`` (faces
    at latitude interfaces), coprime with ``n_lat`` for ``N>1`` so it cannot be
    sharded directly; it is carried as ``v_lower = v[:n_lat]`` (``n_lat`` rows,
    divisible by ``N``). Each band's NORTH boundary face is the next band's
    ``v_lower[0]`` (= the shared global interface row), lifted down via
    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
    receives the pole-wall zero (the ppermute non-target). Pure array core (no
    Field/state coupling) shared by the ocean and atmosphere lat-band SPMD steps
    so the v-stagger numerics are written ONCE (factored from the ocean step's
    ``_reconstruct_v`` closure). AD-safe: ``ppermute`` is self-transposing.

    Parameters
    ----------
    v_lower : array ``(n_lat_band, n_lon[, nlev])``
    axis : the ``shard_map`` mesh axis name (``"lat"``).
    perm_north : the ``(src, dst)`` pairs from :func:`latlon_band_perms`.

    Returns
    -------
    array ``(n_lat_band + 1, n_lon[, nlev])`` — the band's full v-faces.
    """
    boundary = _halo_ppermute(v_lower[0:1], axis, perm_north)
    return jnp.concatenate([v_lower, boundary], axis=0)


def to_vface_lower(v_full):
    """Inverse of :func:`reconstruct_vface_lower`: drop the north boundary face
    (owned by the next band) to return to the ``n_lat``-row ``v_lower``.

    Round-trip identity ``to_vface_lower(reconstruct_vface_lower(v_lower)) ==
    v_lower`` holds whenever the top boundary face is the pole-wall zero (true
    after any step that zeroes v at the pole)."""
    return v_full[:-1]


def reconstruct_vface_lower_multi(v_lowers, axis: str, perm_north):
    """FUSED multi-field twin of :func:`reconstruct_vface_lower` (message
    aggregation, scaling-M4): ONE ``ppermute`` per DTYPE GROUP for the whole
    staggered field group instead of one per field.

    Each field's single boundary row (``v_lower[0:1]``) is flattened on its
    trailing axes, concatenated into one ``(1, sum_flat)`` buffer per dtype
    group, exchanged once, then split and reshaped back — value-identical to
    the per-field reconstruction (the exchange is a bit-copy; flatten/concat/
    split are layout ops), including the north band's ppermute non-target
    zeros.  Mirrors :func:`make_latlon_band_wall_multi_pad_body`'s "per dtype
    group" packing contract; gated by
    ``tests/parallel/test_latlon_spmd_fused_halo.py``.

    Parameters
    ----------
    v_lowers : sequence of arrays ``(n_lat_band, n_lon[, ...])`` — the
        ``v_lower`` carriers to reconstruct (e.g. the ocean state's ``v`` and
        ``v_mask``).  Trailing shapes may differ; dtypes group internally.
    axis : the ``shard_map`` mesh axis name (``"lat"``).
    perm_north : the ``(src, dst)`` pairs from :func:`latlon_band_perms`.

    Returns
    -------
    tuple of arrays ``(n_lat_band + 1, ...)`` — full band v-faces, input order.
    """
    fields = tuple(v_lowers)
    if not fields:
        return ()

    # Group by dtype (static: dtypes are trace-time facts of the args).
    groups: dict = {}
    for i, f in enumerate(fields):
        groups.setdefault(str(f.dtype), []).append(i)

    boundaries: list = [None] * len(fields)
    for _, idxs in sorted(groups.items()):
        flats = []
        widths = []
        for i in idxs:
            row = fields[i][0:1]
            w = 1
            for s in row.shape[1:]:
                w *= int(s)
            widths.append(w)
            flats.append(row.reshape(1, w))
        buf = jnp.concatenate(flats, axis=1)
        # ONE ppermute for the whole dtype group (north band receives 0).
        recv = _halo_ppermute(buf, axis, perm_north)
        off = 0
        for k, i in enumerate(idxs):
            w = widths[k]
            tail = fields[i].shape[1:]
            boundaries[i] = recv[:, off:off + w].reshape((1,) + tail)
            off += w

    return tuple(
        jnp.concatenate([fields[i], boundaries[i]], axis=0)
        for i in range(len(fields))
    )


def reconstruct_uface_left(u_left, axis: str, p_lon: int):
    """Rebuild the ``n_lon_local+1`` staggered u-faces from the
    ``n_lon_local``-column ``u_left`` representation, INSIDE a ``shard_map``
    over the periodic ``"lon"`` ring axis — the u-stagger twin of
    :func:`reconstruct_vface_lower`.

    The zonal velocity ``u`` carries ``n_lon+1`` lon-interface columns whose
    LAST column is the periodic seam (``u[:, n_lon] == u[:, 0]`` by
    construction — ``interp_cell_to_uface`` builds it from the wrap, and every
    C-grid tendency preserves the identity because face ``n_lon`` and face
    ``0`` difference the same wrapped operands).  Under a 2-D lon split, ``u``
    is therefore carried as ``u_left = u[:, :n_lon]`` (``n_lon`` columns,
    divisible by ``p_lon``); each tile's EAST boundary face is its east
    neighbour's ``u_left[:, 0]`` (the shared global interface column), lifted
    via the cyclic ring permutation.  Unlike the v/pole case, longitude is
    periodic so EVERY tile is a ppermute target — the wrap pair
    ``(0, p_lon-1)`` delivers tile 0's first column as the LAST tile's seam,
    which equals the global ``u[:, n_lon]`` by the seam identity above.

    ``p_lon == 1``: STATIC local branch — the boundary is the tile's own
    first column (the serial periodic closure), no collective emitted.

    Parameters
    ----------
    u_left : array ``(n_lat_local, n_lon_local[, nlev])``
    axis : the ``shard_map`` mesh axis name (``"lon"``).
    p_lon : static size of the ``"lon"`` mesh axis.

    Returns
    -------
    array ``(n_lat_local, n_lon_local + 1[, nlev])`` — the tile's u-faces.
    """
    if p_lon == 1:
        boundary = u_left[:, 0:1]
    else:
        perm_to_west, _ = latlon_lon_ring_perms(p_lon)
        boundary = _halo_ppermute(u_left[:, 0:1], axis, perm_to_west)
    return jnp.concatenate([u_left, boundary], axis=1)


def to_uface_left(u_full):
    """Inverse of :func:`reconstruct_uface_left`: drop the east seam column
    (owned as the east neighbour's column 0) to return to the
    ``n_lon_local``-column ``u_left``.

    Round-trip identity holds because the dropped seam face is bit-equal to
    the east neighbour's first face (both tiles compute the shared interface
    from identical wrapped operands — the u-stagger analogue of the v
    round-trip's pole-wall-zero invariant)."""
    return u_full[:, :-1]


def replicate_leaf(arr, rep, *, multiprocess: bool):
    """Replicate one (possibly lat-sharded) leaf onto every device of ``rep``'s
    mesh — the gather primitive shared by the atm and ocean lat-band SPMD steps
    (``gather_state_atm_latlon`` / ``gather_state_latlon``).

    Single-process: plain ``jax.device_put`` (the historical path, unchanged).
    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``):
    a top-level ``device_put`` cannot reshard an array whose shards live on
    other processes' devices, so the replication runs as a jit-compiled
    identity with replicated ``out_shardings`` — the supported cross-process
    collective path (every process executes the same program; XLA inserts the
    all-gather). The fresh ``jax.jit`` per call recompiles per gather —
    acceptable at the segment/run output boundary where gathers happen (never
    in the step hot loop).

    Parameters
    ----------
    arr : jax.Array (any sharding on ``rep``'s mesh)
    rep : NamedSharding — the replicated ``P()`` sharding of the target mesh.
    multiprocess : pass ``jax.process_count() > 1`` (keyword-only so the
        branch is explicit at every call site).
    """
    if multiprocess:
        return jax.jit(lambda a: a, out_shardings=rep)(arr)
    return jax.device_put(arr, rep)


def shard_leaf(arr, sharding, *, multiprocess: bool):
    """Scatter one full-global leaf onto ``sharding``'s mesh — the SCATTER
    primitive symmetric to :func:`replicate_leaf`, shared by the atm and ocean
    lat-band SPMD steps (``shard_state_atm_latlon`` / ``shard_state_latlon``).

    Single-process: plain ``jax.device_put`` (the historical path, unchanged and
    byte-identical).

    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``,
    a mesh spanning processes): a top-level ``jax.device_put`` of the FULL global
    array to a cross-process ``NamedSharding`` cannot place shards on peer
    processes' devices, so XLA falls back to an all-gather that (a) transiently
    materialises a second global copy per process and (b) is the collective seen
    to crash under full-node CPU packing (issue #1100: 128 procs × global-state
    each, rc=137 OOM). Instead ``jax.make_array_from_callback`` invokes the
    callback ONCE PER ADDRESSABLE SHARD with that shard's global index, and each
    process reads only its own shards out of the global array it already holds —
    no all-gather, no transient global replica. Using the per-shard index
    callback (not an enclosing [min,max) span) makes it correct for ANY
    device→process placement, including a non-contiguous/interleaved mesh order.

    NOT differentiable: ``make_array_from_callback`` is a host construction API,
    so (unlike the historical ``device_put``) a ``jax.grad``/``vjp`` cannot be
    taken THROUGH the multiprocess scatter. This is fine — the scatter is an
    init-time boundary (``scatter_to_local`` before the step loop, per the MPI
    pattern), never inside a differentiated loss; gradients w.r.t. params flow
    through the already-sharded state, not the scatter itself.

    Parameters
    ----------
    arr : the FULL global array, present on every process (host or device).
    sharding : NamedSharding — the lat-band ``P("lat", None, ...)`` target.
    multiprocess : pass ``jax.process_count() > 1`` (keyword-only so the branch
        is explicit at every call site, mirroring :func:`replicate_leaf`).
    """
    if not multiprocess:
        return jax.device_put(arr, sharding)
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])


def _pole_fold(rows, negate: bool):
    """Serial pole fold of ``rows`` (lat-mirror + 180 deg lon roll [+ sign]).

    ``rows`` is ``(halo, n_lon_padded[, nlev])`` already lon-wrapped; matches
    :func:`legoesm.grids.halo_latlon.fold_pole_rows` (``half = n_lon_pad // 2``,
    ``roll(rows[::-1], half, axis=lon)``)."""
    half = rows.shape[1] // 2
    sign = -1.0 if negate else 1.0
    return sign * jnp.roll(rows[::-1], half, axis=1)


def partner_pole_fold_window(edge, lon_index, mesh, halo: int, negate: bool):
    """Tile's padded 180-deg pole-fold window via ONE antipodal ``ppermute``
    (even ``p_lon``) — value-identical to the all_gather construction in
    :func:`make_latlon_2d_pad_body`.

    The serial fold of the full wrap-padded circle (:func:`_pole_fold`:
    lat-mirror + roll by ``(W+2h)//2``) maps the window of tile ``c``
    (padded cols ``[c*w, c*w+w+2h)``) onto the ANTIPODE tile
    ``c' = c + p_lon/2``'s edge rows extended ``2h`` columns E/W:

    ``window[t] = sign * ext_antipode[::-1][:, r(t)]`` with
    ``r(t) = t + 2h`` where the global padded column ``j = c*w + t`` lies
    west of the antimeridian seam (``j < W/2 + h`` — there the serial roll
    reads through the wrap-pad copies) and ``r(t) = t`` east of it.  Both
    branches are exact mod-``W`` algebra of the serial padded roll
    (including its piecewise seam), so the result is BIT-equal to the
    gather path — gated by ``test_2d_pad_body_matches_serial_window`` and
    the direct twin test.  Per-tile pole traffic drops from the full
    ``(halo, n_lon)`` circle (all_gather on EVERY tile) to
    ``O(w + 4h)`` point-to-point.

    ``w >= 2*halo`` required (the E/W extension strips must fit the
    neighbour tile); ``min_tile`` in :func:`choose_latlon_2d_topology`
    already enforces ``w >= 2`` and production halos are ``<= 2``, so the
    guard only fires on hand-built degenerate meshes.

    AD-safe: ``ppermute`` is self-transposing; the window ``take`` is a
    gather with a defined transpose (scatter-add).
    """
    p_lon = int(mesh.shape["lon"])
    if p_lon % 2 != 0:
        raise ValueError(
            f"partner_pole_fold_window: p_lon must be even, got {p_lon}")
    w = edge.shape[1]
    h = halo
    if w < 2 * h:
        raise ValueError(
            f"partner_pole_fold_window: tile width {w} < 2*halo = {2 * h} "
            f"(the E/W extension strips would exceed the neighbour tile); "
            f"use the all_gather fold for this degenerate tiling")
    W = w * p_lon
    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
    ext = lon_ring_ghosts_spmd(edge, mesh, halo=2 * h)  # (halo, w+4h[, lev])
    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
    # bijection, and c' != c for every even p_lon >= 2).
    perm_anti = [(s, (s + p_lon // 2) % p_lon) for s in range(p_lon)]
    recv = _halo_ppermute(ext, "lon", perm_anti)
    # 3. lat-mirror + sign (the _pole_fold row flip), then the window
    # column map (derivation above; seam-straddling windows mix branches).
    sign = -1.0 if negate else 1.0
    recv = sign * recv[::-1]
    t = jnp.arange(w + 2 * h)
    j = lon_index * w + t                       # global padded column
    r = jnp.where(j < W // 2 + h, t + 2 * h, t)
    return jnp.take(recv, r, axis=1)


def activate_latlon_spmd_halo(mesh) -> None:
    """Arm the lat-lon SPMD halo backend on a 1-D ``("lat",)`` band mesh or a
    2-D ``("lat", "lon")`` tile mesh.

    Sets the halo backend to ``"spmd"`` and stores ``mesh`` so the per-grid
    ``pad_halo_latlon*`` dispatch routes through
    :func:`make_latlon_band_pad_body` (1-D) or :func:`make_latlon_2d_pad_body`
    (2-D) when called INSIDE a shard_map over the same axes (the ocean/atm
    lat-lon SPMD step).  Mirrors the cube
    ``cubesphere_exchange.activate_spmd_halo_backend``.
    """
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    # The pad_halo_latlon dispatch routes on ``"lat" in mesh.axis_names``; a
    # mesh named otherwise would activate but SILENTLY fall back to local
    # padding inside the shard_map (wrong interior band halos) — fail loud
    # (codex LOW).  Accepted shapes: 1-D ("lat",) band mesh (the historical
    # contract, byte-unchanged) or the 2-D ("lat", "lon") tile mesh (M3a).
    names = tuple(mesh.axis_names)
    if names not in (("lat",), ("lat", "lon")):
        raise ValueError(
            f"activate_latlon_spmd_halo: mesh axes must be ('lat',) or "
            f"('lat', 'lon') (the pad_halo_latlon SPMD dispatch keys on "
            f"them); got {names} with device shape "
            f"{tuple(mesh.devices.shape)}")
    set_spmd_mesh(mesh)
    set_halo_backend("spmd")


def deactivate_latlon_spmd_halo() -> None:
    """Clear the SPMD halo backend (-> ``"local"``)."""
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_spmd_mesh(None)
    set_halo_backend("local")


def make_latlon_band_pad_body(mesh, halo: int = 1, negate: bool = False):
    """Unwrapped lat-lon band halo body for use INSIDE a shard_map over the
    ``"lat"`` axis (the ocean/atm lat-lon SPMD step).

    Returns ``body(tile)`` where ``tile`` is one band's local
    ``(nl_lat, n_lon[, nlev])`` block and the result is the padded
    ``(nl_lat+2h, n_lon+2h[, nlev])`` block — lon-wrap (local) + lat-band
    ppermute (interior) + pole fold (ends).  ``negate=True`` folds with a sign
    flip for meridional-vector components (v).  Bit-identical to the serial
    ``pad_halo_latlon_local`` (scalar) / ``..._vector_local`` (vector) per band.
    """
    n_dev = mesh.devices.size
    if tuple(mesh.devices.shape) != (n_dev,):
        raise ValueError(
            f"make_latlon_band_pad_body: needs a 1-D (n_lat-band) mesh; got "
            f"shape {tuple(mesh.devices.shape)}")
    axis = mesh.axis_names[0]
    perm_north, perm_south = latlon_band_perms(n_dev)

    def body(tile):
        # 1. longitude periodic wrap (LOCAL — full lon circle per band).
        pad_lon = ((0, 0),) + ((halo, halo),) + ((0, 0),) * (tile.ndim - 2)
        data_lon = jnp.pad(tile, pad_lon, mode="wrap")  # (nl, n_lon+2h[, lev])

        # 2. latitude band ppermute of the edge rows (axis 0 = south->north,
        # so row 0 is the SOUTH edge, row -1 the NORTH edge).
        south_edge = data_lon[:halo]   # my south rows -> band below (b-1)'s N ghost
        north_edge = data_lon[-halo:]  # my north rows -> band above (b+1)'s S ghost
        north_recv = _halo_ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
        south_recv = _halo_ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge

        # 3. pole fold at the end bands (ppermute non-targets receive zeros).
        b = jax.lax.axis_index(axis)
        south_ghost = jnp.where(b == 0,
                                _pole_fold(data_lon[:halo], negate), south_recv)
        north_ghost = jnp.where(b == n_dev - 1,
                                _pole_fold(data_lon[-halo:], negate), north_recv)
        return jnp.concatenate([south_ghost, data_lon, north_ghost], axis=0)

    return body


def make_latlon_2d_pad_body(mesh, halo: int = 1, negate: bool = False):
    """Unwrapped 2-D lat-lon TILE halo body for use INSIDE a shard_map over a
    ``("lat", "lon")`` mesh — the 2-D twin of :func:`make_latlon_band_pad_body`
    (M3a: native 2-D tiling for the lat-lon SPMD step).

    Returns ``body(tile)`` where ``tile`` is one tile's local
    ``(nl_lat, nl_lon[, nlev])`` block and the result is the padded
    ``(nl_lat+2h, nl_lon+2h[, nlev])`` block, equal to tile ``(r, c)``'s
    window of the SERIAL :func:`legoesm.grids.halo_latlon.pad_halo_latlon_local`
    output (rows ``[r*nl, r*nl+nl+2h)``, cols ``[c*w, c*w+w+2h)``):

    * latitude interior cuts: ``ppermute`` of the ``halo`` edge rows over
      ``"lat"`` (identical to the band body);
    * longitude: :func:`lon_ring_ghosts_spmd` — the periodic wrap as a cyclic
      ring ``ppermute`` over ``"lon"`` (``p_lon == 1``: the LOCAL wrap, a
      static branch — no collective, bit-identical to the band body);
    * corners: filled by SEQUENCING lat-then-lon — the E/W neighbour's edge
      columns of its lat-EXTENDED block carry its own lat-ghost rows, which
      originate on the receiver's DIAGONAL tile.  No explicit corner message:
      every consumer stencil (vertex circulation, 4-pt Coriolis averages,
      PPM halo-2 reconstruction) reads corners only through this two-pass
      composition, matching the serial pad's corner values exactly (both are
      pure data movement — no arithmetic, so bit-equal).
    * poles: the PHYSICAL pole tiles (``axis_index("lat") == 0`` / ``p_lat-1``)
      replace their beyond-pole ghost rows with the SERIAL 180-deg pole fold.
      Under a lon split the fold's half-circle shift needs remote columns.
      EVEN ``p_lon`` (with ``w >= 2h``): ONE antipodal ``ppermute`` of the
      2h-E/W-extended pole edge rows + a local window column map —
      :func:`partner_pole_fold_window`, bit-equal to the gather construction
      including the serial fold's piecewise ``W//2``-of-the-PADDED-row roll,
      at ``O(w+4h)`` point-to-point traffic per tile.  ODD ``p_lon > 1``
      (and degenerate ``w < 2h``): the tile's pole edge rows are
      ``all_gather``'d over ``"lon"`` into the full ``(halo, n_lon)``
      circle, wrap-padded, folded with the SAME :func:`_pole_fold` the
      serial/band paths use, and the tile's own padded window
      ``[c*w, c*w+w+2h)`` dynamic-sliced back out — bit-identical to the
      serial fold BY CONSTRUCTION.  Uniform-program cost either way: the
      fold collective runs on EVERY tile (selection is a ``jnp.where`` on
      the lat index).  ``p_lon == 1`` skips all fold collectives statically
      (the fold is local, exactly the band body's).

    ``negate=True`` folds with a sign flip for meridional-vector components.
    AD-safe: ``ppermute`` is self-transposing, ``all_gather`` has a defined
    transpose (``psum_scatter``), and the partner fold's window ``take`` is
    a gather (transpose scatter-add).
    """
    p_lat = int(mesh.shape["lat"])
    p_lon = int(mesh.shape["lon"])
    perm_north, perm_south = latlon_band_perms(p_lat)

    def _fold_rows(edge, lon_index):
        """Serial pole fold of the tile's ``(halo, w[, nlev])`` edge rows ->
        the tile's ``(halo, w+2h[, nlev])`` padded fold window (local wrap /
        antipodal ppermute / all_gather — static branch on ``p_lon``)."""
        w = edge.shape[1]
        pad_lon = ((0, 0), (halo, halo)) + ((0, 0),) * (edge.ndim - 2)
        if p_lon == 1:
            # Full circle already local: wrap + fold, exactly the band body
            # (wrap(tile)[:halo] == wrap(tile[:halo]) — per-row lon pad).
            return _pole_fold(jnp.pad(edge, pad_lon, mode="wrap"), negate)
        if p_lon % 2 == 0 and w >= 2 * halo:
            # 180-deg partner-tile ppermute (the documented follow-up,
            # now the even-p_lon default): O(w+4h) point-to-point instead
            # of the full-circle all_gather on EVERY tile.  Bit-equal to
            # the gather construction (see partner_pole_fold_window).
            return partner_pole_fold_window(edge, lon_index, mesh, halo,
                                            negate)
        rows = jax.lax.all_gather(edge, "lon", axis=1, tiled=True)
        rows = jnp.pad(rows, pad_lon, mode="wrap")     # (halo, n_lon+2h[, lev])
        folded = _pole_fold(rows, negate)
        return jax.lax.dynamic_slice_in_dim(
            folded, lon_index * w, w + 2 * halo, axis=1)

    def body(tile):
        # 1. latitude ppermute of the (pre-lon-pad) edge rows over "lat".
        # p_lat == 1 (a pure lon split): both lat ends are physical poles —
        # no lat neighbour exists, so skip the (empty-perm) ppermute
        # statically; the zero rows are overwritten by the folds below.
        if p_lat == 1:
            north_recv = jnp.zeros_like(tile[:halo])
            south_recv = jnp.zeros_like(tile[-halo:])
        else:
            north_recv = _halo_ppermute(tile[:halo], "lat", perm_north)
            south_recv = _halo_ppermute(tile[-halo:], "lat", perm_south)
        ext = jnp.concatenate([south_recv, tile, north_recv], axis=0)

        # 2. longitude ring ghosts on the lat-EXTENDED block (fills corners
        # from the E/W neighbour's lat-ghost rows == the diagonal tile).
        ext = lon_ring_ghosts_spmd(ext, mesh, halo=halo)

        # 3. pole fold at the physical pole tiles (ppermute non-targets
        # received zeros in step 1; overwritten here on the pole rows).
        b = jax.lax.axis_index("lat")
        c = jax.lax.axis_index("lon") if p_lon > 1 else 0
        south_ghost = jnp.where(b == 0, _fold_rows(tile[:halo], c),
                                ext[:halo])
        north_ghost = jnp.where(b == p_lat - 1, _fold_rows(tile[-halo:], c),
                                ext[-halo:])
        return jnp.concatenate(
            [south_ghost, ext[halo:ext.shape[0] - halo], north_ghost], axis=0)

    return body


def make_latlon_band_wall_pad_body(mesh, halo: int = 1,
                                   south_value: float = 0.0,
                                   north_value: float = 0.0):
    """Unwrapped lat-ONLY band WALL pad body for use INSIDE a shard_map over the
    ``"lat"`` axis — the SPMD analogue of the LOCAL branch of
    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat`.

    Returns ``body(tile)`` where ``tile`` is one band's local
    ``(nl_lat, ...)`` block and the result is ``(nl_lat+2h, ...)`` — lat-band
    ppermute of the edge rows at INTERIOR cuts (so the cut ghost row is the
    neighbour band's true edge row, NOT a wall), and a CONSTANT pad
    (``south_value`` / ``north_value``) at the PHYSICAL pole end bands
    (``axis_index == 0`` / ``N-1``).  Unlike :func:`make_latlon_band_pad_body`
    this pads ONLY axis 0 (lon is left untouched, matching
    ``pad_with_pole_bc_lat``) and the pole ghost is a constant WALL, not the
    atmospheric pole fold.  Bit-identical, per band, to the serial
    ``pad_with_pole_bc_lat`` local pad.

    Regular-grid only: ``is_vector_*`` / ``north_fold`` (tripolar fold seam)
    are a follow-up — the SPMD ocean wrapper raises on an active fold.

    2-D ``("lat", "lon")`` meshes are supported (M3a): the pad is lat-ONLY,
    so each lon column of tiles exchanges independently over the ``"lat"``
    axis — the SAME ppermute, keyed on ``mesh.shape["lat"]`` (identical to
    ``devices.size`` on a 1-D band mesh, byte-unchanged there).
    """
    if "lat" not in tuple(mesh.axis_names):
        raise ValueError(
            f"make_latlon_band_wall_pad_body: needs a mesh carrying the "
            f"'lat' axis; got axes {tuple(mesh.axis_names)}")
    n_dev = int(mesh.shape["lat"])
    axis = "lat"
    perm_north, perm_south = latlon_band_perms(n_dev)

    def body(tile):
        # lat-band ppermute of the edge rows (axis 0 = south->north; row 0 is the
        # SOUTH edge, row -1 the NORTH edge).  No lon pad (pad_with_pole_bc_lat
        # leaves lon untouched).
        south_edge = tile[:halo]       # my south rows -> band below (b-1)'s N ghost
        north_edge = tile[-halo:]      # my north rows -> band above (b+1)'s S ghost
        north_recv = _halo_ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
        south_recv = _halo_ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge

        # CONSTANT wall pad at the physical pole end bands (ppermute non-targets
        # receive zeros from these rows, but the where below overrides them with
        # the wall constant of the right shape).
        b = jax.lax.axis_index(axis)
        south_wall = jnp.full_like(south_recv, south_value)
        north_wall = jnp.full_like(north_recv, north_value)
        south_ghost = jnp.where(b == 0, south_wall, south_recv)
        north_ghost = jnp.where(b == n_dev - 1, north_wall, north_recv)
        return jnp.concatenate([south_ghost, tile, north_ghost], axis=0)

    return body


def make_latlon_band_wall_multi_pad_body(mesh, halo: int = 1,
                                         south_values=None,
                                         north_values=None,
                                         n_fields: int = 0):
    """FUSED multi-field twin of :func:`make_latlon_band_wall_pad_body`.

    One ``ppermute`` pair per DIRECTION for the whole field GROUP instead of
    one pair per field: each band's ``halo`` edge rows of every field are
    flattened on the trailing axes, concatenated into a single
    ``(halo, sum_flat)`` buffer per direction, exchanged once, then split
    and reshaped back — value-identical to the per-field pads (the exchange
    is a bit-copy; flatten/concat/split are layout ops).  This is the SPMD
    leg of the message-aggregation lever (audit item 7): the mpi4jax leg
    already fuses via ``pad_with_pole_bc_lat_multi_mpi``, the SPMD leg
    expanded per field.

    STATIC group signature: ``n_fields`` (+ each field's dtype/shape at
    trace time) describes the group (shard_map bodies must be one uniform
    program).  Mixed dtypes are grouped by dtype internally — one buffer
    (and one ppermute pair) per dtype group, matching the MPI fused path's
    "per dtype group" contract.

    Returns ``body(*fields) -> tuple(padded_fields)``.

    2-D ``("lat", "lon")`` meshes: lat-only exchange per lon tile column,
    keyed on ``mesh.shape["lat"]`` (see :func:`make_latlon_band_wall_pad_body`).
    """
    if "lat" not in tuple(mesh.axis_names):
        raise ValueError(
            f"make_latlon_band_wall_multi_pad_body: needs a mesh carrying "
            f"the 'lat' axis; got axes {tuple(mesh.axis_names)}")
    n_dev = int(mesh.shape["lat"])
    if n_fields < 1:
        raise ValueError("n_fields must be >= 1")
    south_values = tuple(south_values or (0.0,) * n_fields)
    north_values = tuple(north_values or (0.0,) * n_fields)
    if len(south_values) != n_fields or len(north_values) != n_fields:
        raise ValueError("boundary value tuples must match n_fields")
    axis = "lat"
    perm_north, perm_south = latlon_band_perms(n_dev)

    def body(*fields):
        if len(fields) != n_fields:
            raise ValueError(
                f"fused pad body built for {n_fields} fields, got "
                f"{len(fields)}")
        b = jax.lax.axis_index(axis)

        # Group by dtype (static: dtypes are trace-time facts of the args).
        groups: dict = {}
        for i, f in enumerate(fields):
            groups.setdefault(str(f.dtype), []).append(i)

        south_ghosts: list = [None] * n_fields
        north_ghosts: list = [None] * n_fields
        for _, idxs in sorted(groups.items()):
            flats = []
            widths = []
            for i in idxs:
                edge_shape = fields[i][:halo].shape
                w = 1
                for s in edge_shape[1:]:
                    w *= int(s)
                widths.append(w)
                flats.append((fields[i][:halo].reshape(halo, w),
                              fields[i][-halo:].reshape(halo, w)))
            south_buf = jnp.concatenate([s for s, _ in flats], axis=1)
            north_buf = jnp.concatenate([n for _, n in flats], axis=1)
            # ONE ppermute pair for the whole dtype group.
            north_recv = _halo_ppermute(south_buf, axis, perm_north)
            south_recv = _halo_ppermute(north_buf, axis, perm_south)
            off = 0
            for k, i in enumerate(idxs):
                w = widths[k]
                tail = fields[i].shape[1:]
                sr = south_recv[:, off:off + w].reshape((halo,) + tail)
                nr = north_recv[:, off:off + w].reshape((halo,) + tail)
                s_wall = jnp.full_like(sr, south_values[i])
                n_wall = jnp.full_like(nr, north_values[i])
                south_ghosts[i] = jnp.where(b == 0, s_wall, sr)
                north_ghosts[i] = jnp.where(b == n_dev - 1, n_wall, nr)
                off += w

        return tuple(
            jnp.concatenate([south_ghosts[i], fields[i], north_ghosts[i]],
                            axis=0)
            for i in range(n_fields)
        )

    return body


def packed_exchange_mesh():
    """The armed 1-D lat-band SPMD mesh when the packed per-stage exchange
    (packing-plan bucket A) is enabled, else ``None``.

    Env gate ``LEGOESM_LATLON_PACKED_EXCHANGE``: ``''`` / ``'0'`` (default)
    is OFF; ``'1'`` is ON — any other value RAISES (dispatch-hardening house
    rule, so a typo cannot silently run the default path).  ON additionally
    requires the armed 1-D ``("lat",)`` band mesh: serial / MPI / the 2-D
    ``("lat", "lon")`` tile mesh return ``None`` so callers keep their
    default (byte-identical) per-exchange pads.
    """
    import os
    val = os.environ.get("LEGOESM_LATLON_PACKED_EXCHANGE", "")
    if val in ("", "0"):
        return None
    if val != "1":
        raise ValueError(
            f"LEGOESM_LATLON_PACKED_EXCHANGE must be '', '0' or '1'; "
            f"got {val!r}")
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None or tuple(getattr(mesh, "axis_names", ())) != ("lat",):
        return None
    return mesh


def make_latlon_band_packed_pad_body(mesh, specs):
    """PACKED multi-field, mixed-halo, mixed-SEMANTICS band exchange
    (packing-plan bucket A): ONE north+south ``ppermute`` pair per dtype
    group carries the RAW (un-lon-padded) edge rows of every field of a
    stage's exchange epoch; the wall constants, the 180-deg pole fold and
    the periodic lon wrap are applied LOCALLY after receipt.  Bit-exact vs
    the per-exchange bodies: the ``ppermute`` is a bit-copy and both the
    lon wrap (``jnp.pad(mode="wrap")``) and the fold act per-row, so they
    commute with the lat exchange exactly.

    ``specs`` — STATIC tuple, one entry per field (trace-time facts):

    * ``("fold", halo, negate)`` — full fold-family pad; output
      ``(nl + 2h, n_lon + 2h[, lev])``, bit-equal to
      :func:`make_latlon_band_pad_body` (``pad_halo_latlon*`` semantics).
    * ``("wall", halo, south_value, north_value)`` — lat-ONLY wall pad;
      output ``(nl + 2h, ...)``, bit-equal to
      :func:`make_latlon_band_wall_pad_body` (``pad_with_pole_bc_lat``
      semantics).

    An unknown kind raises ``ValueError`` (dispatch hardening).

    POLAR BANDS: the poleward ghost is NOT remote — ``perm_north`` /
    ``perm_south`` (from :func:`latlon_band_perms`) already omit the polar
    direction on the end bands (band ``N-1`` is no ``perm_north`` target,
    band ``0`` no ``perm_south`` target), so those bands receive zeros
    there, overwritten by the LOCAL fold / wall constant via ``jnp.where``
    — identical to the single-field bodies.  Per-field semantics (incl.
    the fold's staggering-dependent lon-reversal) are applied AFTER the
    buffer split, never on the packed buffer.

    AD: the comm envelope is linear (reshape / concat / slice + the
    self-transposing ``ppermute``; masks are data-independent selects) —
    no ``custom_vjp`` needed.

    Returns ``body(*fields) -> tuple(padded_fields)`` for use INSIDE a
    shard_map over the ``"lat"`` axis.
    """
    # 1-D BAND mesh only: the fold branch wraps longitude LOCALLY, which is
    # wrong for a lon-split ("lat", "lon") tile mesh (its lon ghosts are
    # remote — the 2-D pad body's ring ppermute).  Reject rather than
    # silently mis-wrap (codex r1 MINOR 1).
    if tuple(mesh.axis_names) != ("lat",):
        raise ValueError(
            f"make_latlon_band_packed_pad_body: needs a 1-D ('lat',) band "
            f"mesh (the fold branch wraps lon locally); got axes "
            f"{tuple(mesh.axis_names)}")
    n_dev = int(mesh.shape["lat"])
    axis = "lat"
    perm_north, perm_south = latlon_band_perms(n_dev)
    specs = tuple(tuple(s) for s in specs)
    for s in specs:
        if s[0] == "fold":
            if len(s) != 3:
                raise ValueError(f"fold spec must be (kind, halo, negate): {s}")
        elif s[0] == "wall":
            if len(s) != 4:
                raise ValueError(
                    f"wall spec must be (kind, halo, south, north): {s}")
        else:
            raise ValueError(
                f"make_latlon_band_packed_pad_body: unknown spec kind "
                f"{s[0]!r} (expected 'fold' or 'wall')")
    n_fields = len(specs)
    if n_fields < 1:
        raise ValueError("specs must name >= 1 field")

    def body(*fields):
        if len(fields) != n_fields:
            raise ValueError(
                f"packed pad body built for {n_fields} fields, got "
                f"{len(fields)}")
        b = jax.lax.axis_index(axis)

        # Group by dtype (static trace-time fact) — one buffer / one
        # ppermute pair per dtype group, matching the fused-pad contract.
        groups: dict = {}
        for i, f in enumerate(fields):
            groups.setdefault(str(f.dtype), []).append(i)

        south_recv: list = [None] * n_fields
        north_recv: list = [None] * n_fields
        for _, idxs in sorted(groups.items()):
            s_flat, n_flat, widths = [], [], []
            for i in idxs:
                h = int(specs[i][1])
                s_edge = fields[i][:h]     # my south rows
                n_edge = fields[i][-h:]    # my north rows
                w = 1
                for d in s_edge.shape:
                    w *= int(d)
                widths.append(w)
                # PRE-FLATTEN to (1, h*w*lev) BEFORE concat: mixed
                # trailing dims after concat would force XLA layout
                # copies; the payload stays contiguous on the minor axis.
                s_flat.append(s_edge.reshape(1, w))
                n_flat.append(n_edge.reshape(1, w))
            south_buf = jnp.concatenate(s_flat, axis=1)
            north_buf = jnp.concatenate(n_flat, axis=1)
            # ONE ppermute pair for the whole dtype group.  My NORTH ghost
            # = north neighbour's south rows (perm_north sends s -> s-1);
            # my SOUTH ghost = south neighbour's north rows.
            n_recv_buf = _halo_ppermute(south_buf, axis, perm_north)
            s_recv_buf = _halo_ppermute(north_buf, axis, perm_south)
            off = 0
            for k, i in enumerate(idxs):
                h = int(specs[i][1])
                w = widths[k]
                shp = (h,) + fields[i].shape[1:]
                south_recv[i] = s_recv_buf[:, off:off + w].reshape(shp)
                north_recv[i] = n_recv_buf[:, off:off + w].reshape(shp)
                off += w

        outs = []
        for i, f in enumerate(fields):
            spec = specs[i]
            h = int(spec[1])
            if spec[0] == "wall":
                _, _, sv, nv = spec
                s_wall = jnp.full_like(south_recv[i], sv)
                n_wall = jnp.full_like(north_recv[i], nv)
                s_ghost = jnp.where(b == 0, s_wall, south_recv[i])
                n_ghost = jnp.where(b == n_dev - 1, n_wall, north_recv[i])
                outs.append(
                    jnp.concatenate([s_ghost, f, n_ghost], axis=0))
            else:  # fold
                negate = bool(spec[2])
                pad_lon = ((0, 0), (h, h)) + ((0, 0),) * (f.ndim - 2)
                data_lon = jnp.pad(f, pad_lon, mode="wrap")
                # Lon-wrap the RAW received rows locally (per-row op —
                # commutes bit-exactly with the exchange of raw rows).
                s_recv_lon = jnp.pad(south_recv[i], pad_lon, mode="wrap")
                n_recv_lon = jnp.pad(north_recv[i], pad_lon, mode="wrap")
                s_ghost = jnp.where(
                    b == 0, _pole_fold(data_lon[:h], negate), s_recv_lon)
                n_ghost = jnp.where(
                    b == n_dev - 1, _pole_fold(data_lon[-h:], negate),
                    n_recv_lon)
                outs.append(
                    jnp.concatenate([s_ghost, data_lon, n_ghost], axis=0))
        return tuple(outs)

    return body


def spmd_pole_end_masks():
    """``(south_mask, north_mask)`` TRACED scalar booleans for the active band
    under the armed lat-band SPMD backend, or ``None`` if SPMD is not active.

    The SPMD twin of the operators' static ``lat_ends_are_poles()``: under the
    single-program SPMD ``shard_map`` there is no STATIC per-band pole answer (the
    same compiled body runs on every band), so an operator that restores the
    serial pole edge-clamp at PHYSICAL poles only must select it DATA-dependently
    —  ``south_mask = (axis_index("lat") == 0)``,
    ``north_mask = (axis_index("lat") == N-1)`` — via ``jnp.where`` rather than a
    Python ``if`` (which under SPMD would clamp every band's INTERIOR cut, the
    ``lat_ends_are_poles() == (True, True)`` SPMD-blind bug).  Returns ``None``
    for the local / MPI / cube paths so the caller keeps its existing static
    ``lat_ends_are_poles()`` branch unchanged (additive; serial/MPI byte-exact).

    MUST be called INSIDE a ``shard_map`` over the ``"lat"`` axis (``axis_index``
    is only defined there).  The masks broadcast against any array (scalar bool
    vs array in ``jnp.where``).  2-D ``("lat", "lon")`` meshes are supported:
    pole ownership is a property of the LAT axis alone (every lon tile of the
    end lat rows touches its pole), so the band count is ``mesh.shape["lat"]``
    — NOT ``devices.size``, which would mislabel every 2-D tile as
    non-polar."""
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None or "lat" not in tuple(getattr(mesh, "axis_names", ())):
        return None
    n_bands = int(mesh.shape["lat"])
    b = jax.lax.axis_index("lat")
    return (b == 0), (b == n_bands - 1)


def apply_pole_end_masks(field, masks, offset: int = 0):
    """Zero ``field`` at the south / north PHYSICAL pole rows of the active band,
    selecting the clamp DATA-dependently from ``masks = (south_mask, north_mask)``
    (the :func:`spmd_pole_end_masks` traced scalar booleans).

    Interior band cuts — whose ``masks`` are both ``False`` — pass through
    untouched, so their cut row keeps the cross-band meridional gradient.  This is
    the SPMD twin of the static ``_zero_v_at_pole`` (atmosphere PE) /
    ``zero_polar_lat_ends`` (halo) pole-wall clamp; ``offset`` mirrors their
    halo-row offset (``0`` under SPMD — bands are not pre-padded).  Both ends are
    applied via ``jnp.where`` so the same compiled body is bit-correct on every
    band (south pole, north pole, or interior cut).
    """
    south_mask, north_mask = masks
    n = field.shape[0]
    z_south = field.at[offset].set(jnp.zeros_like(field[offset]))
    out = jnp.where(south_mask, z_south, field)
    z_north = out.at[n - 1 - offset].set(jnp.zeros_like(out[n - 1 - offset]))
    out = jnp.where(north_mask, z_north, out)
    return out


def zero_polar_lat_ends_band_spmd(field, mesh):
    """SPMD analogue of the MPI branch of
    :func:`legoesm.grids.halo_latlon.zero_polar_lat_ends` — zero axis-0 index 0
    ONLY on the south band (``axis_index == 0``) and index ``-1`` ONLY on the
    north band (``axis_index == N-1``); INTERIOR band cuts are left intact (their
    cut v-row carries the cross-band gradient).

    Must be called INSIDE a shard_map over the same ``"lat"`` axis.  Mirrors the
    MPI ``south_rank is None`` / ``north_rank is None`` pole-touch test with
    ``jax.lax.axis_index`` (data-dependent ⇒ ``jnp.where``, both ends traced) and
    delegates the field-zeroing to the shared :func:`apply_pole_end_masks` core.
    2-D ``("lat", "lon")`` meshes: pole ownership keys on the LAT axis size
    (``mesh.shape["lat"]``), identical to ``devices.size`` on a 1-D band mesh.
    """
    n_bands = int(mesh.shape["lat"])
    b = jax.lax.axis_index("lat")
    return apply_pole_end_masks(field, ((b == 0), (b == n_bands - 1)), offset=0)


def cell_to_cgrid_winds_spmd(u_cell, v_cell):
    """Band-local cell -> C-grid wind conversion — the SPMD twin of
    :func:`legoesm.grids.operators_latlon_cgrid.cell_to_cgrid_winds` (non-fold),
    for the per-step cell<->C-grid round trip the operator-split lat-band lane
    reproduces from serial ``model.step``.

    * **u-face**: ``interp_cell_to_uface`` averages along LONGITUDE only; every
      band owns the full periodic lon circle, so it is row-by-row identical to
      serial with NO halo.
    * **v-face**: ``interp_cell_to_vface_halo`` lifts each INTERIOR band-cut
      face from the neighbour band's edge row (``ppermute`` via the armed spmd
      backend).  A naive band-local ``pad_ns_zero(0.5*(v[:-1]+v[1:]))`` would
      instead ZERO every band boundary (treating each interior cut as a pole)
      and silently mis-set the interior v-faces -> non-bitwise, wrong dynamics.
    * **physical poles**: re-zeroed to the ``v = 0`` wall via
      :func:`apply_pole_end_masks` — ONLY axis-0 index 0 on the south band and
      index -1 on the north band; interior cuts keep their halo'd value.

    Serial / local backend (``spmd_pole_end_masks()`` is ``None``):
    ``interp_cell_to_vface_halo`` delegates to the naive interior average with a
    pole EDGE-COPY, and the static both-ends zero below reproduces
    ``cell_to_cgrid_winds``'s ``pad_ns_zero`` EXACTLY (byte-identical serial).

    MUST run INSIDE a ``shard_map`` over ``"lat"`` WITH the lat-band backend
    ARMED (``activate_latlon_spmd_halo(mesh)``), or serially with it un-armed.
    A shard_map WITHOUT arming is the one silent-wrong state: ``spmd_pole_end_masks()``
    then returns ``None`` and BOTH local band ends get zeroed as poles — the
    caller (the operator-split step) owns the arm/restore, exactly as
    :func:`make_sharded_atm_latlon_step`'s ``sharded_step`` does. Non-fold only:
    the fn takes no grid so it cannot self-check — the operator-split SPMD lane
    refuses the tripole fold upstream (``make_sharded_atm_latlon_step``).

    Parameters
    ----------
    u_cell, v_cell : ``(n_lat_band, n_lon[, nlev])`` cell-centered winds.

    Returns
    -------
    (u_face, v_face) : ``(n_lat_band, n_lon+1, ...)`` and
        ``(n_lat_band + 1, n_lon, ...)`` C-grid face winds.
    """
    from legoesm.grids.operators_latlon_cgrid import (
        interp_cell_to_uface, interp_cell_to_vface_halo)
    u_face = interp_cell_to_uface(u_cell)
    v_face = interp_cell_to_vface_halo(v_cell)
    masks = spmd_pole_end_masks()
    if masks is None:                 # serial / local backend: both ends poles
        v_face = v_face.at[0].set(jnp.zeros_like(v_face[0]))
        v_face = v_face.at[-1].set(jnp.zeros_like(v_face[-1]))
    else:                             # SPMD: zero the PHYSICAL poles only
        v_face = apply_pole_end_masks(v_face, masks, offset=0)
    return u_face, v_face


def choose_latlon_2d_topology(
    n_devices: int,
    n_lat: int,
    n_lon: int,
    *,
    min_tile: int = 2,
    band_preference: float = 1.25,
) -> tuple[int, int]:
    """Choose the ``(p_lat, p_lon)`` factorization of ``n_devices`` for the
    2-D lat-lon SPMD tiling — minimizing the modeled per-tile halo
    COMMUNICATION VOLUME of the actual pad program (ppermute perimeter PLUS
    the pole-fold lon-all_gathers every lon split pays), with a documented
    preference for the 1-D band path.

    Feasibility (uniform tiles — one shard_map program):
      * ``p_lat * p_lon == n_devices``;
      * ``n_lat % p_lat == 0`` and ``n_lon % p_lon == 0``;
      * every SPLIT dimension keeps at least ``min_tile`` cells per tile
        (default 2 — the widest production halo, the PPM ``halo=2``
        exchange, moves edge blocks of that depth in ONE ppermute hop).

    Score = the per-tile RECEIVED communication volume of one full halo pad,
    normalized per unit halo depth (volume / 2h), with ``nl = n_lat/p_lat``,
    ``w = n_lon/p_lon``:

      * lat cut (``p_lat > 1``): the N/S ppermute pair moves ``2h*w`` cells
        -> ``w``;
      * lon cut (``p_lon > 1``): the E/W ring ppermute pair moves ``~2h*nl``
        -> ``nl``, PLUS the pole-fold term, which depends on ``p_lon``
        parity (``make_latlon_2d_pad_body._fold_rows``, uniform program —
        the fold collective runs on EVERY tile because both operands of the
        fold's ``jnp.where`` are evaluated):

        - ODD ``p_lon`` (or ``w < 2h``): two ``all_gather``s each deliver
          the full ``(h, n_lon)`` pole edge circle -> ``n_lon`` (codex M3a
          finding 4: a perimeter-only score ignored this and called the
          ``(1, N)`` split "E/W-perimeter only", which is false);
        - EVEN ``p_lon`` with ``w >= 2*_FOLD_REF_HALO`` (exactly the
          runtime partner-path condition): the partner fold
          (:func:`partner_pole_fold_window`) receives, per fold, the
          2h-wide E/W extension strips (``2 * h*2h``) plus the antipodal
          ``(h, w+4h)`` block; both folds / 2h -> ``w + 8*_FOLD_REF_HALO``
          with the ``h``-quadratic strip terms charged at the widest
          production halo (``_FOLD_REF_HALO = 2``, the PPM halo-2
          exchange — the score is otherwise halo-normalized).  NO cap at
          the gather cost: the runtime never falls back to the gather on
          this branch, so capping would undercharge forced-partner
          candidates (codex r1 #3);

      * an unsplit dimension — or one whose only boundary is the LOCALLY
        folded pole (``p_lon == 1``) — moves nothing.

    Selection policy: the band ``(N, 1)`` is returned WHENEVER FEASIBLE —
    the M3a contract every production lane and scaling receipt assumes;
    the model below ranks only the 2-D candidates that remain when the
    band is infeasible (``n_lat % N != 0`` or ``n_lat/N < min_tile``).
    With the partner fold an even-``p_lon`` candidate can now model-beat
    the band once ``w + 8*h_ref + nl < n_lon``, but flipping the produced
    topology is MEASUREMENT-GATED (M4): revisit with >=16-rank hardware
    receipts (``band_preference``, default 1.25x, is retained for that
    unlock and is currently dormant).  Among 2-D candidates ties break
    toward smaller ``p_lon`` (fewer collectives).

    Raises ``ValueError`` when NO factorization is feasible (so a caller can
    never silently run an invalid tiling).
    """
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    feasible: dict[tuple[int, int], float] = {}
    for p_lat in range(1, n_devices + 1):
        if n_devices % p_lat:
            continue
        p_lon = n_devices // p_lat
        if n_lat % p_lat or n_lon % p_lon:
            continue
        nl, w = n_lat // p_lat, n_lon // p_lon
        if (p_lat > 1 and nl < min_tile) or (p_lon > 1 and w < min_tile):
            continue
        # Received volume per tile per pad, / 2h (see docstring): N/S
        # ppermute pair (w) + E/W ring pair (nl) + the pole-fold term —
        # even p_lon (w >= 2*_FOLD_REF_HALO): partner-ppermute fold
        # (w + 8*h_ref); odd p_lon (or degenerate w): the TWO full-circle
        # lon-all_gathers (n_lon) — codex M3a finding 4.
        if p_lon > 1:
            if p_lon % 2 == 0 and w >= 2 * _FOLD_REF_HALO:
                # Charge exactly what the runtime does: _fold_rows takes
                # the partner path whenever (even p_lon, w >= 2h), even
                # where the gather would move fewer bytes (codex r1 #3 —
                # a min() cap undercharged forced-partner candidates and
                # could select a worse topology).  h_ref=2 is the widest
                # production halo; smaller runtime halos cost less.
                fold = float(w + 8 * _FOLD_REF_HALO)
            else:
                fold = float(n_lon)
            lon_term = nl + fold
        else:
            lon_term = 0.0
        feasible[(p_lat, p_lon)] = (
            (w if p_lat > 1 else 0.0) + lon_term)
    if not feasible:
        raise ValueError(
            f"choose_latlon_2d_topology: no feasible (p_lat, p_lon) for "
            f"n_devices={n_devices} on a {n_lat}x{n_lon} grid (need "
            f"p_lat*p_lon == n_devices with n_lat % p_lat == 0, "
            f"n_lon % p_lon == 0, and >= {min_tile} cells per split "
            f"dimension per tile).")
    band = (n_devices, 1)
    if band in feasible:
        # The band stays the default production lane WHENEVER feasible
        # (the M3a contract — sharded_atm_latlon_step's lane routing and
        # every existing scaling receipt assume it).  With the partner
        # fold, 2-D candidates are now genuinely competitive at large
        # n_lon by the model below, but flipping the produced topology is
        # MEASUREMENT-GATED (M4 doctrine): unlock only with >=16-rank
        # hardware receipts.  ``band_preference`` is retained for that
        # future unlock.
        return band
    # Band infeasible: best modeled volume among the 2-D candidates
    # (honest partner/gather fold costs); ties break toward smaller
    # p_lon (fewer collectives).
    best = min(feasible.items(), key=lambda kv: (kv[1], kv[0][1]))
    return best[0]


def pad_halo_latlon_band_spmd(mesh, halo: int = 1, negate: bool = False):
    """Wrapped shard_map lat-lon band halo exchange (parity-test entry).

    ``fn(field)`` where ``field`` is the GLOBAL ``(n_lat, n_lon[, nlev])`` field
    lat-sharded ``P("lat", None[, None])``; returns the padded global field with
    band-local halos, gathered ``(n_dev*(nl+2h)... )`` — band ``b``'s block ==
    the serial pad's ``[b*nl : b*nl+nl+2h]`` window (the tiled methodology)."""
    body = make_latlon_band_pad_body(mesh, halo=halo, negate=negate)
    axis = mesh.axis_names[0]

    def fn(field):
        nd = field.ndim
        isp = P(axis, *((None,) * (nd - 1)))
        osp = P(axis, *((None,) * (nd - 1)))

        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=osp,
                 check_vma=False)
        def _ex(x):
            return body(x)

        return _ex(field)

    return fn
