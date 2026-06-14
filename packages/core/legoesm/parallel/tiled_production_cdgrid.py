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


# ---------------------------------------------------------------------------
# P-iii: arakawa_lamb_gradient (DEFAULT path — cc B -> D-grid corner gradient
# (dB_dx, dB_dy_perp)).  Like interp_center_to_corner it tiles the GLOBAL
# pad_halo_auto pre-pad ``B_pad`` (n+2), but the corner output ALSO needs the
# precomputed face-local matrix grad_c00/c01/c10/c11 (n+1) sliced per tile, so
# the kernel calls the extracted ``arakawa_lamb_gradient_core``.  Two staggered
# corner outputs -> lower-owns-shared reassembly each.  (The cube-vertex
# a2b/dir-aware diagnostics are OFF in production and not tiled.)
# ---------------------------------------------------------------------------

def arakawa_lamb_gradient_tile_2d(B_pad, grad_c00, grad_c01, grad_c10, grad_c11,
                                  a_i, a_j, nl: int):
    """Per-tile default-path ``arakawa_lamb_gradient`` (P-iii).  ``B_pad``
    ``(F, n+2, n+2[, nlev])`` = global ``pad_halo_auto(B)`` (FACE-REPLICATED);
    ``grad_c00..c11`` ``(F, n+1, n+1)`` the corner matrices.  Corner tile
    ``[a:a+nl+1]`` reads ``B_pad[a:a+nl+2]`` and ``grad_*[a:a+nl+1]``.  Returns
    ``(dB_dx, dB_dy_perp)`` ``(F, nl+1, nl+1[, nlev])`` — reassembly
    lower-owns-shared each."""
    from legoesm.core.operators_cdgrid import arakawa_lamb_gradient_core

    if B_pad.shape[1] != grad_c00.shape[1] + 1:
        raise ValueError(
            f"arakawa_lamb_gradient_tile_2d: B_pad i-dim must be grad i-dim+1 "
            f"(n+2 vs n+1); got B_pad={B_pad.shape[1]}, "
            f"grad_c00={grad_c00.shape[1]}")

    def _sb(arr, si):                       # B_pad block (nl+2)
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, si, axis=2)

    def _sg(arr):                           # grad block (nl+1)
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl + 1, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl + 1, axis=2)

    return arakawa_lamb_gradient_core(
        _sb(B_pad, nl + 2), _sg(grad_c00), _sg(grad_c01), _sg(grad_c10),
        _sg(grad_c11))


def make_tiled_arakawa_lamb_gradient_stage_2d(mesh, cdgrid, n: int, kt: int):
    """Sharded default-path ``arakawa_lamb_gradient`` on a ``(6, kt, kt)`` mesh.
    ``stage(B_pad) -> (dB_dx, dB_dy_perp)`` where ``B_pad`` is the GLOBAL
    face-replicated ``pad_halo_auto(B)`` ``(6, n+2, n+2)``; the corner matrices
    are passed as face-sharded inputs (NOT closed over -> codex U4a HIGH).  Both
    outputs tile-sharded, gathered ``(6, kt*(nl+1), kt*(nl+1))`` reassembling
    lower-owns-shared to ``(6, n+1, n+1)``."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo,) * 5, out_specs=(co, co),
             check_vma=False)
    def _body(B_pad, c00, c01, c10, c11):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return arakawa_lamb_gradient_tile_2d(
            B_pad, c00, c01, c10, c11, a_i, a_j, nl)

    def stage(B_pad):
        return _body(B_pad, gc00, gc01, gc10, gc11)

    return stage


# ---------------------------------------------------------------------------
# P-v: the velocity transforms fv3_d2cc (D-grid edge-midpoint -> cc) +
# fv3_cc2c (cc -> C-grid).  fv3_d2cc is PURELY LOCAL (2-pt avg, no halo, exact
# cc partition).  fv3_cc2c is VECTOR — it tiles the GLOBAL pad_halo_vector
# pre-pad (u_pad/v_pad carry the cube-edge rotation, face-replicated); a single
# (nl+2)-window fed to fv3_cc2c_core's [1:-1] trim yields both staggered
# C-grid outputs (lower-owns-shared each).
# ---------------------------------------------------------------------------

def fv3_d2cc_tile_2d(u_d, v_d, a_i, a_j, nl: int):
    """Per-tile ``fv3_d2cc`` (P-v): D-grid edge-midpoint -> cc (local 2-pt avg).
    ``u_d`` ``(F, n, n+1[, nlev])``, ``v_d`` ``(F, n+1, n[, nlev])``
    FACE-REPLICATED.  cc tile ``[a:a+nl]`` reads ``u_d[a_i:a_i+nl, a_j:a_j+nl+1]``
    + ``v_d[a_i:a_i+nl+1, a_j:a_j+nl]``.  Returns ``(u_cc, v_cc)`` ``(F, nl,
    nl[, nlev])`` — cc exact partition (no trim)."""
    from legoesm.core.operators_cdgrid import fv3_d2cc

    nf = u_d.shape[1]                          # n (u_d is (n, n+1))
    # compare only the spatial axes 1/2 (a trailing nlev is allowed: fv3_d2cc is
    # a pure 2-pt avg, ndim-agnostic) — codex P-v HIGH.
    if u_d.shape[2] != nf + 1 or v_d.shape[1:3] != (nf + 1, nf):
        raise ValueError(
            f"fv3_d2cc_tile_2d: shapes inconsistent; u_d {u_d.shape[1:3]} "
            f"(expected ({nf}, {nf + 1})), v_d {v_d.shape[1:3]} "
            f"(expected ({nf + 1}, {nf}))")
    u_blk = jax.lax.dynamic_slice_in_dim(u_d, a_i, nl, axis=1)
    u_blk = jax.lax.dynamic_slice_in_dim(u_blk, a_j, nl + 1, axis=2)
    v_blk = jax.lax.dynamic_slice_in_dim(v_d, a_i, nl + 1, axis=1)
    v_blk = jax.lax.dynamic_slice_in_dim(v_blk, a_j, nl, axis=2)
    return fv3_d2cc(u_blk, v_blk, None)        # fv3_d2cc never reads cdgrid


def fv3_cc2c_tile_2d(u_pad, v_pad, cosa_u, a_i, a_j, nl: int):
    """Per-tile ``fv3_cc2c`` (P-v): cc -> C-grid (vector).  ``u_pad``/``v_pad``
    ``(F, n+2, n+2[, nlev])`` = GLOBAL ``pad_halo_vector(u_cc, v_cc)``
    (FACE-REPLICATED, carries the cube-edge rotation); ``cosa_u`` ``(F, n+1, n)``.
    The ``[a:a+nl+2]`` window + ``fv3_cc2c_core``'s ``[1:-1]`` trim -> tile
    outputs ``u_c`` ``(F, nl+1, nl)`` (x-face) + ``v_c`` ``(F, nl, nl+1)``
    (y-face); reassembly lower-owns-shared each."""
    from legoesm.core.operators_cdgrid import fv3_cc2c_core

    if u_pad.shape[1] != u_pad.shape[2]:
        raise ValueError(
            f"fv3_cc2c_tile_2d: u_pad must be square; got {u_pad.shape[1:3]}")
    if v_pad.shape[1:] != u_pad.shape[1:]:
        raise ValueError(
            f"fv3_cc2c_tile_2d: v_pad shape {v_pad.shape[1:]} != u_pad "
            f"{u_pad.shape[1:]}")

    def _w(arr):                               # (nl+2)x(nl+2) window
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl + 2, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl + 2, axis=2)

    cu = jax.lax.dynamic_slice_in_dim(cosa_u, a_i, nl + 1, axis=1)
    cu = jax.lax.dynamic_slice_in_dim(cu, a_j, nl, axis=2)
    return fv3_cc2c_core(_w(u_pad), _w(v_pad), cu)


def make_tiled_fv3_d2cc_stage_2d(mesh, n: int, kt: int):
    """Sharded ``fv3_d2cc`` on a ``(6, kt, kt)`` mesh.  ``stage(u_d, v_d) ->
    (u_cc, v_cc)`` (face-replicated D-grid in; tile-sharded cc out, gathered
    ``(6, n, n)`` exact)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo), out_specs=(co, co),
             check_vma=False)
    def _stage(u_d, v_d):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return fv3_d2cc_tile_2d(u_d, v_d, a_i, a_j, nl)

    return _stage


def make_tiled_fv3_cc2c_stage_2d(mesh, cdgrid, n: int, kt: int):
    """Sharded ``fv3_cc2c`` on a ``(6, kt, kt)`` mesh.  ``stage(u_pad, v_pad) ->
    (u_c, v_c)`` where ``u_pad``/``v_pad`` are the GLOBAL face-replicated
    ``pad_halo_vector`` output; ``cosa_u`` is a face-sharded input (NOT closed
    over -> codex U4a HIGH).  Outputs tile-sharded, gathered
    ``(6, kt*(nl+1), kt*nl)`` / ``(6, kt*nl, kt*(nl+1))`` -> lower-owns-shared
    to ``(6, n+1, n)`` / ``(6, n, n+1)``."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    cosa_u = cdgrid.cosa_u
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo), out_specs=(co, co),
             check_vma=False)
    def _body(u_pad, v_pad, cu):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return fv3_cc2c_tile_2d(u_pad, v_pad, cu, a_i, a_j, nl)

    def stage(u_pad, v_pad):
        return _body(u_pad, v_pad, cosa_u)

    return stage
