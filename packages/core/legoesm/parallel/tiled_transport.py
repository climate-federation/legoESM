"""Sub-face tiled FV3 d_sw-chain SPMD kernels (P4 phase-1b, task #3).

Two operator families, both in the SAME approach-C form (cheap global
face-replicated pre-pad, then per-tile ``dynamic_slice`` + a device-uniform
body with NO in-stage halo ppermute):
  * the PPM transport SWEEPS (d_sw3 ``xtp_u``/``ytp_v``), and
  * the d_sw1 ut/vt transport-velocity RECOMPUTE
    (:func:`dsw1_ut_vt_tile_2d`, U4a — a 2x2 box + pointwise, fully local
    once uc/vc carry their 1-cell cross halo).

Approach C (mirrors :func:`legoesm.parallel.tiled_d2a2c.make_tiled_d2a2c_stage`):
the cross-face D-grid halo + the h3=4 PPM pre-pad run in the global
(face-replicated) view; the per-tile PPM sweep is sharded over the device
mesh — each device slices its tile's sweep window from the replicated
padded field and runs the device-uniform
:func:`legoesm.core.fv3_sw_core.ppm_transport_1d` with
``external_halo=h3, rd_prepadded=True`` (so the padded face already carries
the halo — NO in-stage halo ppermute, the same insight as the d2a2c stage).

The per-tile compute is bit-exact vs the global PPM sweep
(``tests/parallel/test_ppm_transport_tile.py`` U2/U2b); this module proves
it holds INSIDE a real ``shard_map`` (``tests/parallel/
test_tiled_transport_stage.py``).

This first stage tiles the SWEEP axis only (i-sweep ``xtp_u`` on a
``(6, kt)`` ``(face, tile_i)`` mesh, full cross axis) — np = 6*kt.  The 2-D
``(6, kt, kt)`` staggered-cross-axis tiling + the ``ytp_v`` j-sweep + the
real Courant / B-grid corner sync are follow-ups (design:
``docs/performance/scaling/cube_transport_tiling_design.md`` §U3).
"""
from __future__ import annotations

from functools import partial

import jax
from jax.sharding import PartitionSpec as P
from legoesm.parallel.shard_map_compat import shard_map


def transport_sweep_tile(vp_g, courant, rd_g, a, nl: int, h3: int = 4):
    """Per-tile i-sweep PPM flux: slice the tile window (SWEEP axis=1; the
    leading face axis and the trailing cross axis are kept) from the
    face-replicated padded inputs at start ``a``, then run the PPM flux.

    Shared by the shard_map stage (``a = axis_index("tile_i") * nl``) and a
    non-shard body test (``a = t * nl``) so the exact slice shapes + the
    ``ppm_transport_1d`` call are CI-covered WITHOUT a multi-device mesh
    (codex U3 MEDIUM — the shard_map test silently skips below 6*kt host
    devices).  ``vp_g`` is ``(F, n+2*h3, m)`` (F=1 inside the stage shard,
    F=6 in the host test); the tile's i-cells ``[a-h3 : a+nl+h3]`` live at
    padded indices ``[a : a+nl+2*h3]`` (the slice U2/U2b validated).
    ``dynamic_slice_in_dim`` takes a single start index (no mixed
    int32/int64 index-tuple error from ``axis_index``).  Returns
    ``(F, nl+1, m)`` i-interface flux.
    """
    from legoesm.core.fv3_sw_core import ppm_transport_1d

    vp_t = jax.lax.dynamic_slice_in_dim(vp_g, a, nl + 2 * h3, axis=1)
    c_t = jax.lax.dynamic_slice_in_dim(courant, a, nl + 1, axis=1)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_g, a, nl + 2, axis=1)
    return ppm_transport_1d(
        vp_t, c_t, rd_t, 1, external_halo=h3, rd_prepadded=True)


def make_tiled_transport_sweep_stage(mesh, n: int, kt: int, h3: int = 4):
    """Build the sharded i-sweep PPM transport stage for a ``(6, kt)`` mesh.

    mesh : ``jax.sharding.Mesh`` with axis names ``("face", "tile_i")`` and
        shape ``(6, kt)``.

    Returns ``stage(vp_g, courant, rd_g) -> flux`` where
      * ``vp_g`` : ``(6, n+2*h3, M)`` the GLOBAL field already padded to the
        PPM storage halo h3 (cross-face halo + edge-pad), FACE-REPLICATED;
      * ``courant`` : ``(6, n+1, M)`` i-interface Courant numbers;
      * ``rd_g`` : ``(6, n+2, M)`` the depth-1 edge-padded ``rdelta``;
    and the output ``flux`` is the tile-sharded i-interface flux of gathered
    extent ``(6, kt*(nl+1), M)`` — each tile contributes ``nl+1`` interfaces
    DUPLICATING the shared boundary with its neighbour (reassembly: lower
    tile owns the shared interface → the global ``(6, n+1, M)`` PPM sweep).
    """
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    fo = P("face", None, None)          # face-replicated padded inputs
    co = P("face", "tile_i", None)      # tile-i-sharded staggered flux

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo),
             out_specs=co, check_vma=False)
    def _stage(vp_g, courant, rd_g):
        a = jax.lax.axis_index("tile_i") * nl
        return transport_sweep_tile(vp_g, courant, rd_g, a, nl, h3)

    return _stage


# ---------------------------------------------------------------------------
# U3b: full 2-D (6, kt, kt) tiling — BOTH the cross axis tiled AND the
# symmetric j-sweep (ytp_v).  The two PPM sweeps are INDEPENDENT 1-D passes
# (fv3_sw_core._bgrid_ke_transport: axis=2 ytp_v, axis=1 xtp_u — no coupling
# at the sweep level), and each sweep is independent PER cross-row, so the
# CROSS axis is sliced to the tile WITHOUT a halo (only the sweep axis carries
# the h3 PPM halo).  np = 6*kt*kt (kt=2 → the np24 target).
# ---------------------------------------------------------------------------

def transport_sweep_tile_2d(vp_g, courant, rd_g, a_i, a_j, nl: int, h3: int = 4,
                            cross_nl: int | None = None,
                            apply_d_sw3_boundary_fix: bool = False,
                            boundary_fix_dx_field=None,
                            boundary_fix_edges=None):
    """Per-tile i-sweep (xtp_u, axis=1) PPM flux on a 2-D ``(tile_i, tile_j)``
    tiling.  Slice the SWEEP axis=1 to the window ``[a_i : a_i+nl+2*h3]`` AND
    the CROSS axis=2 to the tile's ``[a_j : a_j+cross_nl]`` (NO j-halo — the
    i-sweep is an independent 1-D PPM per j-row).  ``cross_nl`` defaults to
    ``nl`` (CELL cross, the synthetic/U3b case); pass ``nl+1`` for the REAL
    xtp_u whose cross axis is the ``n+1`` CORNER axis (U3d, shapes from job
    8480355).  ``vp_g`` ``(F, n+2*h3, n_cross)``; returns ``(F, nl+1, cn)``.

    d_sw3 boundary fix (2026-07-10 follow-up — the serial sweeps now run
    ``apply_d_sw3_boundary_fix=True`` always-on): pass the GLOBAL interior
    ``boundary_fix_dx_field`` (sliced to the tile here) plus the static
    ``boundary_fix_edges=(sweep_lo, sweep_hi, cross_lo, cross_hi)`` flags for
    the global face edges the tile touches (e.g. ``(a_i == 0, ti == kt-1,
    a_j == 0, tj == kt-1)``) — the one-sided overrides + vertex zeroing are
    only valid at GLOBAL face boundaries, never at tile cuts."""
    from legoesm.core.fv3_sw_core import ppm_transport_1d

    cn = nl if cross_nl is None else cross_nl
    # Static guards (codex U3d MED): the three inputs must share the CROSS
    # axis (=2 here) size and ``cn`` must fit, else dynamic_slice_in_dim
    # silently CLAMPS the start and mis-pairs cross rows instead of failing.
    if not (vp_g.shape[2] == courant.shape[2] == rd_g.shape[2]):
        raise ValueError(
            f"transport_sweep_tile_2d: cross-axis(2) sizes must match; got "
            f"vp_g={vp_g.shape[2]}, courant={courant.shape[2]}, "
            f"rd_g={rd_g.shape[2]}")
    if cn > vp_g.shape[2]:
        raise ValueError(
            f"transport_sweep_tile_2d: cross_nl={cn} exceeds cross dim "
            f"{vp_g.shape[2]}")
    vp_t = jax.lax.dynamic_slice_in_dim(vp_g, a_i, nl + 2 * h3, axis=1)
    vp_t = jax.lax.dynamic_slice_in_dim(vp_t, a_j, cn, axis=2)
    c_t = jax.lax.dynamic_slice_in_dim(courant, a_i, nl + 1, axis=1)
    c_t = jax.lax.dynamic_slice_in_dim(c_t, a_j, cn, axis=2)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_g, a_i, nl + 2, axis=1)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_t, a_j, cn, axis=2)
    dx_t = None
    if boundary_fix_dx_field is not None:
        # GLOBAL interior dx (sweep axis=1 length n) → tile window.  Only
        # the ≤2 cells nearest an ACTIVE global edge are read, where the
        # tile's internal edge-pad matches the global one bit-for-bit.
        dx_t = jax.lax.dynamic_slice_in_dim(
            boundary_fix_dx_field, a_i, nl, axis=1)
        dx_t = jax.lax.dynamic_slice_in_dim(dx_t, a_j, cn, axis=2)
    return ppm_transport_1d(
        vp_t, c_t, rd_t, 1, external_halo=h3, rd_prepadded=True,
        apply_d_sw3_boundary_fix=apply_d_sw3_boundary_fix,
        boundary_fix_dx_field=dx_t,
        boundary_fix_edges=boundary_fix_edges)


def transport_jsweep_tile_2d(vp_g, courant, rd_g, a_i, a_j, nl: int, h3: int = 4,
                             cross_nl: int | None = None,
                             apply_d_sw3_boundary_fix: bool = False,
                             boundary_fix_dx_field=None,
                             boundary_fix_edges=None):
    """Per-tile j-sweep (ytp_v, axis=2) PPM flux — the symmetric counterpart
    of :func:`transport_sweep_tile_2d` with the sweep/cross axes swapped.
    Slice the SWEEP axis=2 to ``[a_j : a_j+nl+2*h3]`` AND the CROSS axis=1 to
    ``[a_i : a_i+cross_nl]`` (NO i-halo).  ``cross_nl`` defaults to ``nl``;
    pass ``nl+1`` for the REAL ytp_v (cross axis = ``n+1`` CORNER, U3d).
    ``vp_g`` ``(F, n_cross, n+2*h3)``; returns ``(F, cn, nl+1)``.

    d_sw3 boundary-fix args: as :func:`transport_sweep_tile_2d`, with the
    sweep on the j axis — ``boundary_fix_edges=(a_j-lo, a_j-hi, a_i-lo,
    a_i-hi)`` global-edge flags."""
    from legoesm.core.fv3_sw_core import ppm_transport_1d

    cn = nl if cross_nl is None else cross_nl
    # Static guards (codex U3d MED): cross axis here is =1.
    if not (vp_g.shape[1] == courant.shape[1] == rd_g.shape[1]):
        raise ValueError(
            f"transport_jsweep_tile_2d: cross-axis(1) sizes must match; got "
            f"vp_g={vp_g.shape[1]}, courant={courant.shape[1]}, "
            f"rd_g={rd_g.shape[1]}")
    if cn > vp_g.shape[1]:
        raise ValueError(
            f"transport_jsweep_tile_2d: cross_nl={cn} exceeds cross dim "
            f"{vp_g.shape[1]}")
    vp_t = jax.lax.dynamic_slice_in_dim(vp_g, a_j, nl + 2 * h3, axis=2)
    vp_t = jax.lax.dynamic_slice_in_dim(vp_t, a_i, cn, axis=1)
    c_t = jax.lax.dynamic_slice_in_dim(courant, a_j, nl + 1, axis=2)
    c_t = jax.lax.dynamic_slice_in_dim(c_t, a_i, cn, axis=1)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_g, a_j, nl + 2, axis=2)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_t, a_i, cn, axis=1)
    dx_t = None
    if boundary_fix_dx_field is not None:
        # GLOBAL interior dx (sweep axis=2 length n) → tile window.
        dx_t = jax.lax.dynamic_slice_in_dim(
            boundary_fix_dx_field, a_j, nl, axis=2)
        dx_t = jax.lax.dynamic_slice_in_dim(dx_t, a_i, cn, axis=1)
    return ppm_transport_1d(
        vp_t, c_t, rd_t, 2, external_halo=h3, rd_prepadded=True,
        apply_d_sw3_boundary_fix=apply_d_sw3_boundary_fix,
        boundary_fix_dx_field=dx_t,
        boundary_fix_edges=boundary_fix_edges)


def make_tiled_transport_sweep_stage_2d(
    mesh, n: int, kt: int, h3: int = 4, sweep: str = "i",
    cross_nl: int | None = None,
):
    """Build a sharded PPM transport stage on a ``(6, kt, kt)`` mesh with axis
    names ``("face", "tile_i", "tile_j")``.

    ``sweep="i"`` tiles the i-sweep (xtp_u); ``sweep="j"`` the j-sweep
    (ytp_v).  Inputs are FACE-REPLICATED, pre-padded to the PPM storage halo
    h3 on the SWEEP axis (i: axis=1, j: axis=2).  Output is sharded
    ``P("face","tile_i","tile_j")``.  ``cross_nl=None`` (the synthetic/U3b
    case) → cross slice ``nl`` → gathered ``(6, kt*(nl+1), kt*nl)`` for ``i``
    / ``(6, kt*nl, kt*(nl+1))`` for ``j``.  ``cross_nl=nl+1`` (the REAL
    _bgrid_ke_transport case, cross axis = the ``n+1`` CORNER axis, U3d) →
    gathered ``(6, kt*(nl+1), kt*(nl+1))``; both axes reassemble
    lower-tile-owns-shared to ``(6, n+1, n+1)``."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    if sweep not in ("i", "j"):
        raise ValueError(f"sweep must be 'i' or 'j', got {sweep!r}")
    nl = n // kt
    # cross_nl is only meaningful as nl (CELL cross, tiles n) or nl+1 (CORNER
    # cross, tiles n+1) — both tile EXACTLY across kt tiles.  Reject anything
    # else so a stray value can't silently clamp the last tile's cross slice
    # (codex U3d-wire LOW).
    if cross_nl is not None and cross_nl not in (nl, nl + 1):
        raise ValueError(
            f"cross_nl must be nl={nl} (cell) or nl+1={nl + 1} (corner); "
            f"got {cross_nl}")
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")
    body = transport_sweep_tile_2d if sweep == "i" else transport_jsweep_tile_2d

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo),
             out_specs=co, check_vma=False)
    def _stage(vp_g, courant, rd_g):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return body(vp_g, courant, rd_g, a_i, a_j, nl, h3, cross_nl=cross_nl)

    return _stage


# ---------------------------------------------------------------------------
# U4a: d_sw1 ut/vt transport-velocity RECOMPUTE (the start of the D-grid
# stage).  ut/vt are re-derived from the UPDATED C-grid winds via a 4-cell
# box average of the OTHER component:
#     ut = (uc - 0.25*cosa_u*box(vc)) * rsin_u    # (6, n+1, n)
#     vt = (vc - 0.25*cosa_v*box(uc)) * rsin_v    # (6, n, n+1)
# (fv3_sw_core._d_sw1_recompute_ut_vt, duogrid Part 1, lines 220-242).  This
# is DISTINCT from the d2a2c ut/vt (d2a2c_tile_unified:1225/1235 uses the
# D-grid wind directly, `ut = (uc - v_d*cosa_u)*rsin_u`).  The box stencil is
# fully LOCAL once uc/vc carry their 1-cell cross halo — supplied by the
# GLOBAL face-replicated pre-pad `_pad_halo_uc_vc_new_via_neighbor_delta` (the same
# pre-pad proven in U3f) — so each tile only dynamic_slices, NO in-stage
# exchange.  Reassembly: lower tile owns the shared staggered face (ut: i-axis
# n+1; vt: j-axis n+1), the cell axis tiles exactly across kt.
# ---------------------------------------------------------------------------

def dsw1_ut_vt_tile_2d(uc, vc, uc_pad, vc_pad,
                       cosa_u, rsin_u, cosa_v, rsin_v,
                       a_i, a_j, nl: int):
    """Per-tile d_sw1 ut/vt recompute on a 2-D ``(tile_i, tile_j)`` tiling.

    ``uc`` ``(F, n+1, n)``, ``vc`` ``(F, n, n+1)`` are the (face-replicated)
    UPDATED C-grid winds; ``uc_pad`` ``(F, n+1, n+2)`` / ``vc_pad``
    ``(F, n+2, n+1)`` are their GLOBAL cross-halo pre-pad
    (``_pad_halo_uc_vc_new_via_neighbor_delta``).  ``cosa_u``/``rsin_u``
    ``(F, n+1, n)`` and ``cosa_v``/``rsin_v`` ``(F, n, n+1)`` are the static
    metrics.  Slices the tile at start ``(a_i, a_j)`` (``axis_index*nl`` in a
    shard, ``t*nl`` in a host test) and returns the staggered tile blocks
    ``ut_t`` ``(F, nl+1, nl)`` and ``vt_t`` ``(F, nl, nl+1)``.

    The ``ut`` tile owns i-faces ``[a_i : a_i+nl+1]`` (the box reads vc_pad
    rows ``[a_i : a_i+nl+2]``) and j-cells ``[a_j : a_j+nl]`` (vc_pad cols
    ``[a_j : a_j+nl+1]``); ``vt`` is symmetric.  All slices fit the padded
    extents EXACTLY at the last tile (start+size == dim) so none clamp.
    """
    # Static guards (codex U3d/U4a MED): a mis-sized input would make
    # dynamic_slice_in_dim silently CLAMP a window instead of failing — check
    # EVERY axis relationship.  ``F`` (axis 0) is 6 in a host test and 1 inside
    # a face-shard; ALL inputs must share it, else a face-1 uc * face-6 metric
    # would broadcast the output back to face 6 (codex U4a LOW).
    F = uc.shape[0]
    if not (vc.shape[0] == uc_pad.shape[0] == vc_pad.shape[0]
            == cosa_u.shape[0] == rsin_u.shape[0] == cosa_v.shape[0]
            == rsin_v.shape[0] == F):
        raise ValueError(
            "dsw1_ut_vt_tile_2d: all inputs must share the leading face-axis "
            f"extent {F}; got vc={vc.shape[0]}, uc_pad={uc_pad.shape[0]}, "
            f"vc_pad={vc_pad.shape[0]}, cosa_u={cosa_u.shape[0]}, "
            f"rsin_u={rsin_u.shape[0]}, cosa_v={cosa_v.shape[0]}, "
            f"rsin_v={rsin_v.shape[0]}")
    # nfi = n+1 (u-faces on i), nf = n (cells).  uc (n+1, n); vc (n, n+1) =
    # (nf, nfi); uc_pad (n+1, n+2) = (nfi, nf+2); vc_pad (n+2, n+1) =
    # (nfi+1, nf+1); cosa_u/rsin_u (n+1, n) = (nfi, nf); cosa_v/rsin_v (n, n+1)
    # = (nf, nfi).
    nfi, nf = uc.shape[1], uc.shape[2]
    if vc.shape[1:] != (nf, nfi):
        raise ValueError(
            f"dsw1_ut_vt_tile_2d: vc cross shape {vc.shape[1:]} inconsistent "
            f"with uc {(nfi, nf)} (expected ({nf}, {nfi}))")
    if uc_pad.shape[1:] != (nfi, nf + 2):
        raise ValueError(
            f"dsw1_ut_vt_tile_2d: uc_pad shape {uc_pad.shape[1:]} != "
            f"({nfi}, {nf + 2}) (j-halo of uc)")
    if vc_pad.shape[1:] != (nfi + 1, nf + 1):
        raise ValueError(
            f"dsw1_ut_vt_tile_2d: vc_pad shape {vc_pad.shape[1:]} != "
            f"({nfi + 1}, {nf + 1}) (i-halo of vc)")
    if not (cosa_u.shape[1:] == rsin_u.shape[1:] == (nfi, nf)):
        raise ValueError(
            f"dsw1_ut_vt_tile_2d: cosa_u/rsin_u must match uc {(nfi, nf)}; "
            f"got {cosa_u.shape[1:]}, {rsin_u.shape[1:]}")
    if not (cosa_v.shape[1:] == rsin_v.shape[1:] == (nf, nfi)):
        raise ValueError(
            f"dsw1_ut_vt_tile_2d: cosa_v/rsin_v must match vc {(nf, nfi)}; "
            f"got {cosa_v.shape[1:]}, {rsin_v.shape[1:]}")

    # --- ut tile: (F, nl+1, nl) from uc + 2x2 box of vc_pad ---
    uc_t = jax.lax.dynamic_slice_in_dim(uc, a_i, nl + 1, axis=1)
    uc_t = jax.lax.dynamic_slice_in_dim(uc_t, a_j, nl, axis=2)
    cau_t = jax.lax.dynamic_slice_in_dim(cosa_u, a_i, nl + 1, axis=1)
    cau_t = jax.lax.dynamic_slice_in_dim(cau_t, a_j, nl, axis=2)
    rsu_t = jax.lax.dynamic_slice_in_dim(rsin_u, a_i, nl + 1, axis=1)
    rsu_t = jax.lax.dynamic_slice_in_dim(rsu_t, a_j, nl, axis=2)
    vcp_t = jax.lax.dynamic_slice_in_dim(vc_pad, a_i, nl + 2, axis=1)
    vcp_t = jax.lax.dynamic_slice_in_dim(vcp_t, a_j, nl + 1, axis=2)
    vc_avg = (vcp_t[:, :-1, :-1] + vcp_t[:, 1:, :-1]
              + vcp_t[:, :-1, 1:] + vcp_t[:, 1:, 1:])          # (F, nl+1, nl)
    ut_t = (uc_t - 0.25 * cau_t * vc_avg) * rsu_t

    # --- vt tile: (F, nl, nl+1) from vc + 2x2 box of uc_pad ---
    vc_t = jax.lax.dynamic_slice_in_dim(vc, a_i, nl, axis=1)
    vc_t = jax.lax.dynamic_slice_in_dim(vc_t, a_j, nl + 1, axis=2)
    cav_t = jax.lax.dynamic_slice_in_dim(cosa_v, a_i, nl, axis=1)
    cav_t = jax.lax.dynamic_slice_in_dim(cav_t, a_j, nl + 1, axis=2)
    rsv_t = jax.lax.dynamic_slice_in_dim(rsin_v, a_i, nl, axis=1)
    rsv_t = jax.lax.dynamic_slice_in_dim(rsv_t, a_j, nl + 1, axis=2)
    ucp_t = jax.lax.dynamic_slice_in_dim(uc_pad, a_i, nl + 1, axis=1)
    ucp_t = jax.lax.dynamic_slice_in_dim(ucp_t, a_j, nl + 2, axis=2)
    uc_avg = (ucp_t[:, :-1, :-1] + ucp_t[:, 1:, :-1]
              + ucp_t[:, :-1, 1:] + ucp_t[:, 1:, 1:])          # (F, nl, nl+1)
    vt_t = (vc_t - 0.25 * cav_t * uc_avg) * rsv_t

    return ut_t, vt_t


def make_tiled_dsw1_ut_vt_stage_2d(mesh, cdgrid, n: int, kt: int):
    """Build the sharded d_sw1 ut/vt recompute stage on a ``(6, kt, kt)`` mesh
    (axis names ``("face", "tile_i", "tile_j")``).

    Closes over the static staggered metrics (``cosa_u``/``rsin_u``/
    ``cosa_v``/``rsin_v``) like :func:`tiled_d2a2c.make_tiled_d2a2c_stage`.
    Returns ``stage(uc, vc, uc_pad, vc_pad) -> (ut, vt)`` where the four
    inputs are FACE-REPLICATED (``uc``/``vc`` the updated C-grid winds,
    ``uc_pad``/``vc_pad`` their global ``_pad_halo_uc_vc_new_via_neighbor_delta``
    pre-pad) and the outputs are tile-sharded
    ``P("face","tile_i","tile_j")`` with gathered extents
    ``ut (6, kt*(nl+1), kt*nl)`` / ``vt (6, kt*nl, kt*(nl+1))`` (the staggered
    axis DUPLICATES the shared face; reassembly = lower tile owns it →
    ``(6, n+1, n)`` / ``(6, n, n+1)``).
    """
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_dsw1_ut_vt_stage_2d: n={n} != cdgrid.n={cdgrid.n}")
    nl = n // kt
    cosa_u, rsin_u = cdgrid.cosa_u, cdgrid.rsin_u
    cosa_v, rsin_v = cdgrid.cosa_v, cdgrid.rsin_v
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    # The static metrics are passed as FACE-SHARDED inputs (in_specs=fo), NOT
    # closed over: inside a face-sharded shard_map the local uc/vc have face
    # extent 1, so a closed-over full (6,...) metric would broadcast the output
    # back to face extent 6 and violate out_specs (codex U4a HIGH).  Mirror
    # tiled_d2a2c.make_tiled_d2a2c_stage, which passes every array as an input.
    @partial(shard_map, mesh=mesh, in_specs=(fo,) * 8,
             out_specs=(co, co), check_vma=False)
    def _body(uc, vc, uc_pad, vc_pad, cau, rsu, cav, rsv):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return dsw1_ut_vt_tile_2d(
            uc, vc, uc_pad, vc_pad, cau, rsu, cav, rsv, a_i, a_j, nl)

    def stage(uc, vc, uc_pad, vc_pad):
        """uc/vc the updated C-grid winds; uc_pad/vc_pad their global
        ``_pad_halo_uc_vc_new_via_neighbor_delta`` pre-pad — all FACE-REPLICATED."""
        return _body(uc, vc, uc_pad, vc_pad,
                     cosa_u, rsin_u, cosa_v, rsin_v)

    return stage
