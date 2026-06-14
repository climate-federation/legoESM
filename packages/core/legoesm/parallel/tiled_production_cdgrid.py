"""Sub-face tiled PRODUCTION cube-dycore ops (task (a), design:
``docs/scaling/cube_production_tiling_design.md``).

Ports the U3/U4 approach-C sub-face tiling from the EXPERIMENTAL FB chain to the
PRODUCTION ``fv3_sw_tendencies`` / ``fv3_hydrostatic_tendencies`` op set, so the
production cube can run on np = 6*kt*kt devices on future fast-interconnect HW
(TPU pods / NVLink).  Each op = a local stencil → per-tile ``dynamic_slice`` of
the (face-replicated) global inputs + the device-uniform op core, reassembled
bit-exactly.  NOT Ginsburg-benchable (np>6 = CPU shard_map / cross-node ppermute
both anti-scale here) — gated by BIT-IDENTITY, the proven U3 methodology.

P-i: ``dgrid_vorticity`` (corner winds → cc relative vorticity).  PURELY LOCAL —
cc cell ``(i,j)`` reads only the 2x2 corner block ``[i:i+2, j:j+2]``; the corner
inputs are ``(n+1, n+1)`` so a cell-tile of ``nl`` cells slices ``nl+1`` corners
``[a:a+nl+1]`` (no halo, no pre-pad), and the cc cells partition EXACTLY (no
shared face → no lower-owns trim).
"""
from __future__ import annotations

from functools import partial

import jax
from jax.sharding import PartitionSpec as P

try:  # JAX >= 0.8 top-level export
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map


def dgrid_vorticity_tile_2d(u_d, v_d, cosa_corner, dx_edge_y, dy_edge_x, area,
                            a_i, a_j, nl: int):
    """Per-tile ``dgrid_vorticity`` (P-i) on a 2-D ``(tile_i, tile_j)`` tiling.

    ``u_d``/``v_d``/``cosa_corner`` ``(F, n+1, n+1)`` corner inputs;
    ``dx_edge_y`` ``(F, n, n+1)``, ``dy_edge_x`` ``(F, n+1, n)``, ``area``
    ``(F, n, n)`` raw metrics — all FACE-REPLICATED.  Slices the cell tile at
    ``(a_i, a_j)`` (``axis_index*nl`` in a shard, ``t*nl`` in a host test):
    corners ``[a:a+nl+1]`` (the ``nl+1`` corners bounding ``nl`` cells), metrics
    to their staggered tile extents.  Returns ``zeta_tile`` ``(F, nl, nl)``
    (cc cells partition exactly — caller concatenates with NO trim).
    """
    from legoesm.core.operators_cdgrid import dgrid_vorticity_core

    nfc = u_d.shape[1]            # n+1 (corners on i)
    nf = nfc - 1                  # n   (cells)
    # Static guards (codex U3d/U4a pattern): a mis-sized input would make
    # dynamic_slice_in_dim silently CLAMP instead of failing.  F (axis 0) is 6
    # in a host test, 1 in a face-shard; compare axes 1/2 only.
    if not (v_d.shape[1:] == cosa_corner.shape[1:] == (nfc, nfc)):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: v_d/cosa_corner must be corners "
            f"{(nfc, nfc)}; got v_d={v_d.shape[1:]}, "
            f"cosa_corner={cosa_corner.shape[1:]}")
    if dx_edge_y.shape[1:] != (nf, nfc):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: dx_edge_y must be {(nf, nfc)}; got "
            f"{dx_edge_y.shape[1:]}")
    if dy_edge_x.shape[1:] != (nfc, nf):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: dy_edge_x must be {(nfc, nf)}; got "
            f"{dy_edge_x.shape[1:]}")
    if area.shape[1:] != (nf, nf):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: area must be {(nf, nf)}; got "
            f"{area.shape[1:]}")

    def _s(arr, si, sj, ax_i=1, ax_j=2):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=ax_i)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=ax_j)

    u_t = _s(u_d, nl + 1, nl + 1)
    v_t = _s(v_d, nl + 1, nl + 1)
    cosa_t = _s(cosa_corner, nl + 1, nl + 1)
    dx_t = _s(dx_edge_y, nl, nl + 1)        # (F, nl, nl+1)
    dy_t = _s(dy_edge_x, nl + 1, nl)        # (F, nl+1, nl)
    area_t = _s(area, nl, nl)               # (F, nl, nl)
    return dgrid_vorticity_core(u_t, v_t, cosa_t, dx_t, dy_t, area_t)


def make_tiled_dgrid_vorticity_stage_2d(mesh, cdgrid, n: int, kt: int):
    """Build the sharded ``dgrid_vorticity`` stage on a ``(6, kt, kt)`` mesh
    (axis names ``("face", "tile_i", "tile_j")``).

    Returns ``stage(u_d, v_d) -> zeta`` where ``u_d``/``v_d`` are the
    FACE-REPLICATED corner winds; the static metrics are passed as face-sharded
    inputs (NOT closed over — a closed-over full ``(6,...)`` metric would
    broadcast the output back to face extent 6, the codex U4a HIGH).  Output is
    tile-sharded ``P("face","tile_i","tile_j")``, gathered extent
    ``(6, kt*nl, kt*nl)`` = ``(6, n, n)`` (cc cells partition exactly — NO
    duplicated shared face).
    """
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_dgrid_vorticity_stage_2d: n={n} != cdgrid.n={cdgrid.n}")
    nl = n // kt
    cosa_corner = cdgrid.cosa_corner
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    area = cdgrid.base.area
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo,) * 6, out_specs=co,
             check_vma=False)
    def _body(u_d, v_d, cosa_c, dxe, dye, ar):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return dgrid_vorticity_tile_2d(
            u_d, v_d, cosa_c, dxe, dye, ar, a_i, a_j, nl)

    def stage(u_d, v_d):
        return _body(u_d, v_d, cosa_corner, dx_edge_y, dy_edge_x, area)

    return stage


# ---------------------------------------------------------------------------
# P-ii: the two production 4-point box interps.
#   interp_corner_to_center (corner -> cc): PURELY LOCAL (cc cell reads its 2x2
#     corner block) — like dgrid_vorticity, no halo, exact cc partition.
#   interp_center_to_corner (cc -> corner): needs the cc 1-cell halo; tile the
#     GLOBAL face-replicated pre-pad ``pad_halo_auto(field)`` (corner staggered
#     -> lower-owns-shared reassembly).
# Both tile kernels CALL the production ops on the sliced blocks (no dup
# numerics — the box + its ndim handling live once in operators_cdgrid).
# ---------------------------------------------------------------------------

def interp_corner_to_center_tile_2d(field_d, a_i, a_j, nl: int):
    """Per-tile ``interp_corner_to_center`` (P-ii): cc tile ``[a:a+nl]`` from the
    ``nl+1`` corner block ``[a:a+nl+1]`` (no halo; cc cells partition exactly).
    ``field_d`` ``(F, n+1, n+1[, nlev])`` corner, FACE-REPLICATED.  Returns the
    cc tile ``(F, nl, nl[, nlev])``."""
    from legoesm.core.operators_cdgrid import interp_corner_to_center

    if field_d.shape[1] != field_d.shape[2]:
        raise ValueError(
            f"interp_corner_to_center_tile_2d: field_d must be square corners; "
            f"got {field_d.shape[1:3]}")
    blk = jax.lax.dynamic_slice_in_dim(field_d, a_i, nl + 1, axis=1)
    blk = jax.lax.dynamic_slice_in_dim(blk, a_j, nl + 1, axis=2)
    return interp_corner_to_center(blk)


def interp_center_to_corner_tile_2d(f_pad, cdgrid, a_i, a_j, nl: int):
    """Per-tile ``interp_center_to_corner`` (P-ii): corner tile (up to
    ``nl+1`` staggered) from the GLOBAL pre-pad ``f_pad =
    pad_halo_auto(field)`` ``(F, n+2, n+2[, nlev])`` (FACE-REPLICATED).  The
    corner block ``[a:a+nl+1]`` reads ``f_pad[a:a+nl+2]``.  Calls the production
    op with ``padded=blk`` (``field=blk`` only supplies ndim; cdgrid unused once
    padded is given).  Returns ``(F, nl+1, nl+1[, nlev])``; reassembly is
    lower-tile-owns-the-shared-corner-face."""
    from legoesm.core.operators_cdgrid import interp_center_to_corner

    if f_pad.shape[1] != f_pad.shape[2]:
        raise ValueError(
            f"interp_center_to_corner_tile_2d: f_pad must be square; got "
            f"{f_pad.shape[1:3]}")
    blk = jax.lax.dynamic_slice_in_dim(f_pad, a_i, nl + 2, axis=1)
    blk = jax.lax.dynamic_slice_in_dim(blk, a_j, nl + 2, axis=2)
    return interp_center_to_corner(blk, cdgrid, padded=blk)


def make_tiled_interp_corner_to_center_stage_2d(mesh, n: int, kt: int):
    """Sharded ``interp_corner_to_center`` on a ``(6, kt, kt)`` mesh.
    ``stage(field_d) -> cc`` (face-replicated corner in; tile-sharded cc out,
    gathered ``(6, n, n)`` — exact cc partition, no shared face)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo,), out_specs=co,
             check_vma=False)
    def _stage(field_d):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return interp_corner_to_center_tile_2d(field_d, a_i, a_j, nl)

    return _stage


def make_tiled_interp_center_to_corner_stage_2d(mesh, cdgrid, n: int, kt: int):
    """Sharded ``interp_center_to_corner`` on a ``(6, kt, kt)`` mesh.
    ``stage(f_pad) -> corner`` where ``f_pad`` is the GLOBAL face-replicated
    ``pad_halo_auto(field)`` ``(6, n+2, n+2)``; tile-sharded corner out, gathered
    ``(6, kt*(nl+1), kt*(nl+1))`` reassembling lower-owns-shared to
    ``(6, n+1, n+1)``.  cdgrid is closed over (unused — padded is always
    supplied in the tile body, so no compute reads it)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo,), out_specs=co,
             check_vma=False)
    def _stage(f_pad):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return interp_center_to_corner_tile_2d(f_pad, cdgrid, a_i, a_j, nl)

    return _stage
