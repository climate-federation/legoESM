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

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from legoesm.parallel.shard_map_compat import shard_map


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
    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
    return jnp.concatenate([v_lower, boundary], axis=0)


def to_vface_lower(v_full):
    """Inverse of :func:`reconstruct_vface_lower`: drop the north boundary face
    (owned by the next band) to return to the ``n_lat``-row ``v_lower``.

    Round-trip identity ``to_vface_lower(reconstruct_vface_lower(v_lower)) ==
    v_lower`` holds whenever the top boundary face is the pole-wall zero (true
    after any step that zeroes v at the pole)."""
    return v_full[:-1]


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


def _pole_fold(rows, negate: bool):
    """Serial pole fold of ``rows`` (lat-mirror + 180 deg lon roll [+ sign]).

    ``rows`` is ``(halo, n_lon_padded[, nlev])`` already lon-wrapped; matches
    :func:`legoesm.grids.halo_latlon.fold_pole_rows` (``half = n_lon_pad // 2``,
    ``roll(rows[::-1], half, axis=lon)``)."""
    half = rows.shape[1] // 2
    sign = -1.0 if negate else 1.0
    return sign * jnp.roll(rows[::-1], half, axis=1)


def activate_latlon_spmd_halo(mesh) -> None:
    """Arm the lat-lon band SPMD halo backend on a 1-D ``"lat"`` mesh.

    Sets the halo backend to ``"spmd"`` and stores ``mesh`` so the per-grid
    ``pad_halo_latlon*`` dispatch routes through :func:`make_latlon_band_pad_body`
    when called INSIDE a shard_map over the same ``"lat"`` axis (the ocean/atm
    lat-lon SPMD step).  Mirrors the cube
    ``cubesphere_exchange.activate_spmd_halo_backend``.
    """
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    if tuple(mesh.devices.shape) != (mesh.devices.size,):
        raise ValueError(
            f"activate_latlon_spmd_halo: needs a 1-D lat-band mesh; got "
            f"shape {tuple(mesh.devices.shape)}")
    # The pad_halo_latlon dispatch routes on ``"lat" in mesh.axis_names``; a
    # mesh named otherwise would activate but SILENTLY fall back to local
    # padding inside the shard_map (wrong interior band halos) — fail loud
    # (codex LOW).
    if "lat" not in tuple(mesh.axis_names):
        raise ValueError(
            f"activate_latlon_spmd_halo: mesh axis must be named 'lat' (the "
            f"pad_halo_latlon SPMD dispatch keys on it); got "
            f"{tuple(mesh.axis_names)}")
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
        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge

        # 3. pole fold at the end bands (ppermute non-targets receive zeros).
        b = jax.lax.axis_index(axis)
        south_ghost = jnp.where(b == 0,
                                _pole_fold(data_lon[:halo], negate), south_recv)
        north_ghost = jnp.where(b == n_dev - 1,
                                _pole_fold(data_lon[-halo:], negate), north_recv)
        return jnp.concatenate([south_ghost, data_lon, north_ghost], axis=0)

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
    """
    n_dev = mesh.devices.size
    if tuple(mesh.devices.shape) != (n_dev,):
        raise ValueError(
            f"make_latlon_band_wall_pad_body: needs a 1-D (n_lat-band) mesh; "
            f"got shape {tuple(mesh.devices.shape)}")
    axis = mesh.axis_names[0]
    perm_north, perm_south = latlon_band_perms(n_dev)

    def body(tile):
        # lat-band ppermute of the edge rows (axis 0 = south->north; row 0 is the
        # SOUTH edge, row -1 the NORTH edge).  No lon pad (pad_with_pole_bc_lat
        # leaves lon untouched).
        south_edge = tile[:halo]       # my south rows -> band below (b-1)'s N ghost
        north_edge = tile[-halo:]      # my north rows -> band above (b+1)'s S ghost
        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge

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
    """
    n_dev = mesh.devices.size
    if tuple(mesh.devices.shape) != (n_dev,):
        raise ValueError(
            f"make_latlon_band_wall_multi_pad_body: needs a 1-D mesh; got "
            f"shape {tuple(mesh.devices.shape)}")
    if n_fields < 1:
        raise ValueError("n_fields must be >= 1")
    south_values = tuple(south_values or (0.0,) * n_fields)
    north_values = tuple(north_values or (0.0,) * n_fields)
    if len(south_values) != n_fields or len(north_values) != n_fields:
        raise ValueError("boundary value tuples must match n_fields")
    axis = mesh.axis_names[0]
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
            north_recv = jax.lax.ppermute(south_buf, axis, perm_north)
            south_recv = jax.lax.ppermute(north_buf, axis, perm_south)
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
    vs array in ``jnp.where``)."""
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None or "lat" not in tuple(getattr(mesh, "axis_names", ())):
        return None
    n_dev = mesh.devices.size
    b = jax.lax.axis_index("lat")
    return (b == 0), (b == n_dev - 1)


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
    """
    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]
    b = jax.lax.axis_index(axis)
    return apply_pole_end_masks(field, ((b == 0), (b == n_dev - 1)), offset=0)


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
