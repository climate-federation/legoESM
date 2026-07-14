"""Sub-face tiled PRODUCTION cube-dycore ops (task (a), design:
``docs/performance/scaling/cube_production_tiling_design.md``).

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
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from legoesm.parallel.shard_map_compat import shard_map


def _check_shapes(n, **named):
    """Fail-loud stage-entry shape guard for the assembly/full stages.

    The leaf tile kernels guard their slice inputs (a mis-sized array would make
    ``dynamic_slice_in_dim`` silently CLAMP -> wrong answer), but the composed
    momentum/mass/full stages slice many inputs inline; validate them ONCE at
    the stage boundary instead.  ``named`` maps name -> (array, expected
    trailing-2D shape) with ``n`` the global cube extent.  Leading axis 0 (6
    faces) is not checked (it is 6 in a host call, 1 per face-shard)."""
    for name, (arr, want) in named.items():
        got = tuple(arr.shape[1:])
        if got != want:
            raise ValueError(
                f"{name}: expected trailing shape {want} (n={n}); got {got}")


def _check_tiled_mesh(mesh, n: int, kt: int):
    """Fail-loud factory-time guard for the (6, kt, kt) tiled cube stages
    (codex review).  The bit-identity gates only exercise a mesh whose shape
    MATCHES kt at nl=n//kt>=2, so two misuse modes slip past them:

    1. ``mesh.devices.shape != (6, kt, kt)``: the tiled halo tables derive their
       tiling from ``mesh`` while the per-tile ``dynamic_slice`` offsets use the
       caller's ``kt``; a mismatch silently CLAMPS the slices / exchanges halos
       on a different tiling -> wrong answer, no error.
    2. ``nl = n // kt < 2``: the in-stage halo's 2-cell offset slivers + the
       corner-staggered ``(nl+1)`` blocks assume nl>=2; nl==1 builds and then
       produces wrong clamped halo interpolation.

    ``n % kt`` is checked separately at each factory entry."""
    want = (6, kt, kt)
    got = tuple(mesh.devices.shape)
    if got != want:
        raise ValueError(
            f"tiled stage: mesh.devices.shape {got} != (6, kt, kt)={want} "
            f"(kt={kt}); the mesh tiling must match kt")
    if n // kt < 2:
        raise ValueError(
            f"tiled stage: nl = n//kt = {n // kt} < 2 (n={n}, kt={kt}); the "
            f"in-stage halo offset slivers + (nl+1) corner blocks need nl>=2")


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
    # in a host test, 1 in a face-shard; compare the HORIZONTAL axes 1/2 only
    # (``[1:3]``) so a trailing vertical ``nlev`` axis is allowed — u_d/v_d are
    # 4D ``(F, n+1, n+1, nlev)`` in the 3D PE dycore while the metrics stay
    # 2D-face (broadcast over nlev inside dgrid_vorticity_core).
    if not (v_d.shape[1:3] == cosa_corner.shape[1:3] == (nfc, nfc)):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: v_d/cosa_corner must be corners "
            f"{(nfc, nfc)}; got v_d={v_d.shape[1:3]}, "
            f"cosa_corner={cosa_corner.shape[1:3]}")
    if dx_edge_y.shape[1:3] != (nf, nfc):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: dx_edge_y must be {(nf, nfc)}; got "
            f"{dx_edge_y.shape[1:3]}")
    if dy_edge_x.shape[1:3] != (nfc, nf):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: dy_edge_x must be {(nfc, nf)}; got "
            f"{dy_edge_x.shape[1:3]}")
    if area.shape[1:3] != (nf, nf):
        raise ValueError(
            f"dgrid_vorticity_tile_2d: area must be {(nf, nf)}; got "
            f"{area.shape[1:3]}")

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


def make_tiled_dgrid_vorticity_stage_2d(mesh, cdgrid, n: int, kt: int,
                                        nlev: int | None = None):
    """Build the sharded ``dgrid_vorticity`` stage on a ``(6, kt, kt)`` mesh
    (axis names ``("face", "tile_i", "tile_j")``).

    Returns ``stage(u_d, v_d) -> zeta`` where ``u_d``/``v_d`` are the
    FACE-REPLICATED corner winds; the static metrics are passed as face-sharded
    inputs (NOT closed over — a closed-over full ``(6,...)`` metric would
    broadcast the output back to face extent 6, the codex U4a HIGH).  Output is
    tile-sharded ``P("face","tile_i","tile_j")``, gathered extent
    ``(6, kt*nl, kt*nl)`` = ``(6, n, n)`` (cc cells partition exactly — NO
    duplicated shared face).

    ``nlev`` (3D PE dycore): when given, ``u_d``/``v_d`` are 4D
    ``(6, n+1, n+1, nlev)`` and ``zeta`` is 4D ``(6, n, n, nlev)`` — the wind
    specs gain a trailing replicated axis (the vertical is NOT tiled), while the
    metrics stay 2D-face (broadcast over nlev in the core).  ``dgrid_vorticity``
    is the production cube vorticity op for BOTH fv3_sw_tendencies (2D) and
    fv3_hydrostatic_tendencies (4D)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_dgrid_vorticity_stage_2d: n={n} != cdgrid.n={cdgrid.n}")
    nl = n // kt
    cosa_corner = cdgrid.cosa_corner
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    area = cdgrid.base.area
    fo = P("face", None, None)                 # 2D-face metric / 2D wind
    co = P("face", "tile_i", "tile_j")
    # 4D wind/output gain a trailing replicated (vertical) axis.
    fw = P("face", None, None, None) if nlev else fo
    cz = P("face", "tile_i", "tile_j", None) if nlev else co

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fo, fo, fo, fo), out_specs=cz,
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
# P-3D-vel: the 3D PE-dycore velocity transforms — BOTH PURELY LOCAL (the 3D
# fv3_hydrostatic_tendencies uses dgrid_to_cgrid (D->C, within-face projection)
# + dgrid_to_center_vector (D->cc, box-avg), NOT the SW fv3_d2cc/fv3_cc2c
# vector-halo path).  Both read only adjacent corners (i,i+1)/(j,j+1) within the
# corner block, so the staggered (nl+1) tile slice already carries every needed
# value — NO halo, NO cross-face rotation.  4D-native cores; metrics 2D-face.
# ---------------------------------------------------------------------------

def dgrid_to_cgrid_tile_2d(u_d, v_d, cosa_u, a_i, a_j, nl: int):
    """Per-tile ``dgrid_to_cgrid`` (3D PE D->C).  ``u_d``/``v_d``
    ``(F, n+1, n+1[, nlev])`` corner winds, ``cosa_u`` ``(F, n+1, n)`` metric —
    FACE-REPLICATED.  Slices the ``(nl+1, nl+1)`` corner block at ``(a_i, a_j)``
    + ``cosa_u`` ``(nl+1, nl)``; returns ``u_c`` ``(F, nl+1, nl)`` (x-face) +
    ``v_c`` ``(F, nl, nl+1)`` (y-face) — staggered, reassembly lower-owns-shared
    each."""
    from legoesm.core.operators_cdgrid import dgrid_to_cgrid_core

    nfc = u_d.shape[1]
    if u_d.shape[1:3] != (nfc, nfc) or v_d.shape[1:3] != (nfc, nfc):
        raise ValueError(
            f"dgrid_to_cgrid_tile_2d: u_d/v_d must be square corners; got "
            f"u_d={u_d.shape[1:3]}, v_d={v_d.shape[1:3]}")
    if cosa_u.shape[1:3] != (nfc, nfc - 1):
        raise ValueError(
            f"dgrid_to_cgrid_tile_2d: cosa_u must be {(nfc, nfc - 1)}; got "
            f"{cosa_u.shape[1:3]}")

    def _sc(arr, si, sj):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

    u_blk = _sc(u_d, nl + 1, nl + 1)
    v_blk = _sc(v_d, nl + 1, nl + 1)
    cosa_blk = _sc(cosa_u, nl + 1, nl)
    return dgrid_to_cgrid_core(u_blk, v_blk, cosa_blk)


def dgrid_to_center_vector_tile_2d(u_d, v_d, a_i, a_j, nl: int):
    """Per-tile ``dgrid_to_center_vector`` (3D PE D->cc, box-avg).  ``u_d``/
    ``v_d`` ``(F, n+1, n+1[, nlev])`` corner winds (FACE-REPLICATED).  cc tile
    ``[a:a+nl]`` from the ``(nl+1, nl+1)`` corner block; returns ``(u_cc, v_cc)``
    ``(F, nl, nl[, nlev])`` — cc cells partition exactly (no halo, no trim)."""
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector

    if (u_d.shape[1] != u_d.shape[2] or u_d.shape[1:3] != v_d.shape[1:3]
            or nl + 1 > u_d.shape[1]):
        raise ValueError(
            f"dgrid_to_center_vector_tile_2d: u_d/v_d must be equal square "
            f"corners with extent >= nl+1={nl + 1}; got u_d={u_d.shape[1:3]}, "
            f"v_d={v_d.shape[1:3]}")

    def _sc(arr):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl + 1, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl + 1, axis=2)

    return dgrid_to_center_vector(_sc(u_d), _sc(v_d))


def make_tiled_dgrid_to_cgrid_stage_2d(mesh, cdgrid, n: int, kt: int,
                                       nlev: int | None = None):
    """Sharded ``dgrid_to_cgrid`` (3D PE D->C) on a ``(6, kt, kt)`` mesh.
    ``stage(u_d, v_d) -> (u_c, v_c)``; corner winds FACE-REPLICATED, ``cosa_u``
    a face-sharded input (NOT closed over -> codex U4a HIGH).  Outputs
    tile-sharded, gathered ``(6, kt*(nl+1), kt*nl)`` / ``(6, kt*nl, kt*(nl+1))``
    reassembling lower-owns-shared to ``(6, n+1, n)`` / ``(6, n, n+1)``.
    ``nlev`` -> 4D winds (vertical replicated); ``cosa_u`` stays 2D-face."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_dgrid_to_cgrid_stage_2d: n={n} != cdgrid.n={cdgrid.n}")
    nl = n // kt
    cosa_u = cdgrid.cosa_u
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")
    fw = P("face", None, None, None) if nlev else fo
    cz = P("face", "tile_i", "tile_j", None) if nlev else co

    @partial(shard_map, mesh=mesh, in_specs=(fw, fw, fo), out_specs=(cz, cz),
             check_vma=False)
    def _body(u_d, v_d, cu):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return dgrid_to_cgrid_tile_2d(u_d, v_d, cu, a_i, a_j, nl)

    def stage(u_d, v_d):
        # Fail-loud stage-entry guard (codex MED): a mis-sized input would
        # otherwise dynamic_slice-CLAMP silently inside the shard body.
        if u_d.shape[1:3] != (n + 1, n + 1) or v_d.shape[1:3] != (n + 1, n + 1):
            raise ValueError(
                f"dgrid_to_cgrid stage: corner winds must be (n+1,n+1)="
                f"{(n + 1, n + 1)}; got u_d={u_d.shape[1:3]}, "
                f"v_d={v_d.shape[1:3]}")
        return _body(u_d, v_d, cosa_u)

    return stage


def make_tiled_dgrid_to_center_vector_stage_2d(mesh, n: int, kt: int,
                                               nlev: int | None = None):
    """Sharded ``dgrid_to_center_vector`` (3D PE D->cc box-avg) on a
    ``(6, kt, kt)`` mesh.  ``stage(u_d, v_d) -> (u_cc, v_cc)`` (corner winds
    FACE-REPLICATED; tile-sharded cc out, gathered ``(6, n, n)`` exact).
    ``nlev`` -> 4D (vertical replicated)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")
    fw = P("face", None, None, None) if nlev else fo
    cz = P("face", "tile_i", "tile_j", None) if nlev else co

    @partial(shard_map, mesh=mesh, in_specs=(fw, fw), out_specs=(cz, cz),
             check_vma=False)
    def _body(u_d, v_d):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return dgrid_to_center_vector_tile_2d(u_d, v_d, a_i, a_j, nl)

    def stage(u_d, v_d):
        # Fail-loud stage-entry guard (codex MED).
        if u_d.shape[1:3] != (n + 1, n + 1) or v_d.shape[1:3] != (n + 1, n + 1):
            raise ValueError(
                f"dgrid_to_center_vector stage: corner winds must be "
                f"(n+1,n+1)={(n + 1, n + 1)}; got u_d={u_d.shape[1:3]}, "
                f"v_d={v_d.shape[1:3]}")
        return _body(u_d, v_d)

    return stage


# ---------------------------------------------------------------------------
# P-3D-geo: compute_geopotential (Simmons-Burridge hydrostatic integration).
# PURELY VERTICAL + horizontally-pointwise — Phi_k is a per-COLUMN cumsum over
# levels of R_d*T*ln_ratio + alpha*R_d*T; no horizontal stencil.  The vertical
# axis is REPLICATED (not tiled), so the cumsum runs LOCAL per tile -> exact cc
# partition, NO halo.  sigma_coord carries only 1-D (nlev,) vertical arrays
# (ln_ratio/alpha) — safe to close over (NOT a (6,...) face array, so no U4a
# broadcast).
# ---------------------------------------------------------------------------

def compute_geopotential_tile_2d(T, p_s, phis, sigma_coord, a_i, a_j, nl: int):
    """Per-tile ``compute_geopotential`` (3D PE).  ``T`` ``(F, n, n, nlev)``,
    ``p_s``/``phis`` ``(F, n, n)`` — FACE-REPLICATED cc.  Slices the cc tile
    ``[a:a+nl]`` (no halo; cc partitions exactly) and runs the per-column
    integration; returns ``Phi`` ``(F, nl, nl, nlev)``."""
    from legoesm.grids.vertical import compute_geopotential

    if T.shape[1] != T.shape[2] or p_s.shape[1:3] != T.shape[1:3] \
            or phis.shape[1:3] != T.shape[1:3]:
        raise ValueError(
            f"compute_geopotential_tile_2d: T cc-square + p_s/phis matching "
            f"horizontal; got T={T.shape[1:3]}, p_s={p_s.shape[1:3]}, "
            f"phis={phis.shape[1:3]}")

    def _s2(arr):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

    return compute_geopotential(_s2(T), _s2(p_s), sigma_coord, _s2(phis))


def make_tiled_compute_geopotential_stage_2d(mesh, sigma_coord, n: int, kt: int):
    """Sharded ``compute_geopotential`` on a ``(6, kt, kt)`` mesh.
    ``stage(T, p_s, phis) -> Phi``; ``T`` 4D ``(6,n,n,nlev)`` + ``p_s``/``phis``
    2D cc ``(6,n,n)`` FACE-REPLICATED; tile-sharded ``Phi`` out, gathered
    ``(6, n, n, nlev)`` (exact cc partition, vertical replicated).  ``sigma_coord``
    (1-D vertical arrays) is closed over."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)                 # 2D cc (p_s, phis)
    fw = P("face", None, None, None)           # 4D T
    cz = P("face", "tile_i", "tile_j", None)   # 4D Phi out

    @partial(shard_map, mesh=mesh, in_specs=(fw, fo, fo), out_specs=cz,
             check_vma=False)
    def _body(T, p_s, phis):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return compute_geopotential_tile_2d(T, p_s, phis, sigma_coord,
                                            a_i, a_j, nl)

    def stage(T, p_s, phis):
        if T.shape[1:3] != (n, n) or p_s.shape[1:3] != (n, n) \
                or phis.shape[1:3] != (n, n):
            raise ValueError(
                f"compute_geopotential stage: T/p_s/phis must be cc (n,n)="
                f"{(n, n)}; got T={T.shape[1:3]}, p_s={p_s.shape[1:3]}, "
                f"phis={phis.shape[1:3]}")
        return _body(T, p_s, phis)

    return stage


# ---------------------------------------------------------------------------
# P-3D-geo-hybrid: compute_geopotential_hybrid (Simmons-Burridge on the HYBRID
# sigma-pressure coordinate — the PRODUCTION OMIP vertical coord).  Same
# per-COLUMN structure as compute_geopotential (vertical cumsum, NO horizontal
# stencil), but the ln_ratio/alpha are pressure-dependent (p_half from
# pressure_from_hybrid), still per-column.  Vertical REPLICATED, cc partitions
# exactly, NO halo.  ``coord`` (HybridSigmaPressureCoordinate, 1-D a/b vertical
# arrays) is closed over (not a (6,...) face array, so no U4a broadcast).
# ---------------------------------------------------------------------------

def compute_geopotential_hybrid_tile_2d(T, p_s, phis, coord, a_i, a_j, nl: int):
    """Per-tile ``compute_geopotential_hybrid`` (3D PE, hybrid coord).  Slices
    the cc tile ``[a:a+nl]`` (no halo) and runs the per-column hybrid
    integration; returns ``Phi`` ``(F, nl, nl, nlev)``.  Mirrors
    :func:`compute_geopotential_tile_2d` (sigma); only the core + coord differ."""
    from legoesm.grids.vertical import compute_geopotential_hybrid

    if T.shape[1] != T.shape[2] or p_s.shape[1:3] != T.shape[1:3] \
            or phis.shape[1:3] != T.shape[1:3]:
        raise ValueError(
            f"compute_geopotential_hybrid_tile_2d: T cc-square + p_s/phis "
            f"matching horizontal; got T={T.shape[1:3]}, p_s={p_s.shape[1:3]}, "
            f"phis={phis.shape[1:3]}")

    def _s2(arr):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

    return compute_geopotential_hybrid(_s2(T), _s2(p_s), coord, _s2(phis))


def make_tiled_compute_geopotential_hybrid_stage_2d(mesh, coord, n: int, kt: int):
    """Sharded ``compute_geopotential_hybrid`` on a ``(6, kt, kt)`` mesh — the
    hybrid-coord twin of :func:`make_tiled_compute_geopotential_stage_2d` (the
    production OMIP vertical coord).  ``stage(T, p_s, phis) -> Phi``; ``T`` 4D
    ``(6,n,n,nlev)`` + ``p_s``/``phis`` 2D cc face-replicated; tile-sharded
    ``Phi`` out (exact cc partition, vertical replicated).  ``coord`` closed over."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)                 # 2D cc (p_s, phis)
    fw = P("face", None, None, None)           # 4D T
    cz = P("face", "tile_i", "tile_j", None)   # 4D Phi out

    @partial(shard_map, mesh=mesh, in_specs=(fw, fo, fo), out_specs=cz,
             check_vma=False)
    def _body(T, p_s, phis):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return compute_geopotential_hybrid_tile_2d(T, p_s, phis, coord,
                                                   a_i, a_j, nl)

    def stage(T, p_s, phis):
        if T.shape[1:3] != (n, n) or p_s.shape[1:3] != (n, n) \
                or phis.shape[1:3] != (n, n):
            raise ValueError(
                f"compute_geopotential_hybrid stage: T/p_s/phis must be cc "
                f"(n,n)={(n, n)}; got T={T.shape[1:3]}, p_s={p_s.shape[1:3]}, "
                f"phis={phis.shape[1:3]}")
        return _body(T, p_s, phis)

    return stage


# ---------------------------------------------------------------------------
# P-3D-massflux-hybrid: compute_mass_flux_hybrid (vertical mass flux at
# half-levels on the HYBRID coord).  Takes the already-3D horizontal divergence
# ``div_3d`` (F,n,n,nlev) + ``p_s`` (F,n,n); every op is along the LEVEL axis
# (``dp_from_hybrid`` is per-column, cumsum over levels, frac_B from
# ``coord.B_half``) — NO horizontal stencil, so it tiles EXACTLY like
# geopotential.  TWO cc-local outputs: ``mass_flux`` (F,nl,nl,nlev+1) +
# ``D_total_p`` (F,nl,nl,1), vertical REPLICATED.  ``coord`` closed over.
# ---------------------------------------------------------------------------

def compute_mass_flux_hybrid_tile_2d(div_3d, p_s, coord, a_i, a_j, nl: int):
    """Per-tile ``compute_mass_flux_hybrid`` (3D PE, hybrid coord).  Slices the
    cc tile ``[a:a+nl]`` (no halo; cc partitions exactly) and runs the
    per-column vertical mass-flux integration; returns ``(mass_flux
    (F,nl,nl,nlev+1), D_total_p (F,nl,nl,1))``.  Mirrors
    :func:`compute_geopotential_hybrid_tile_2d`; only the core differs."""
    from legoesm.grids.vertical import compute_mass_flux_hybrid

    if div_3d.shape[1] != div_3d.shape[2] or p_s.shape[1:3] != div_3d.shape[1:3]:
        raise ValueError(
            f"compute_mass_flux_hybrid_tile_2d: div_3d cc-square + p_s matching "
            f"horizontal; got div_3d={div_3d.shape[1:3]}, p_s={p_s.shape[1:3]}")

    def _s2(arr):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

    return compute_mass_flux_hybrid(_s2(div_3d), _s2(p_s), coord)


def make_tiled_compute_mass_flux_hybrid_stage_2d(mesh, coord, n: int, kt: int):
    """Sharded ``compute_mass_flux_hybrid`` on a ``(6, kt, kt)`` mesh.
    ``stage(div_3d, p_s) -> (mass_flux, D_total_p)``; ``div_3d`` 4D
    ``(6,n,n,nlev)`` + ``p_s`` 2D cc FACE-REPLICATED; both outputs tile-sharded
    (exact cc partition, vertical replicated).  ``coord`` closed over."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)                 # 2D cc (p_s)
    fw = P("face", None, None, None)           # 4D div_3d
    cz = P("face", "tile_i", "tile_j", None)   # 4D out (mass_flux + D_total_p)

    @partial(shard_map, mesh=mesh, in_specs=(fw, fo), out_specs=(cz, cz),
             check_vma=False)
    def _body(div_3d, p_s):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return compute_mass_flux_hybrid_tile_2d(div_3d, p_s, coord,
                                                a_i, a_j, nl)

    def stage(div_3d, p_s):
        if div_3d.shape[1:3] != (n, n) or p_s.shape[1:3] != (n, n):
            raise ValueError(
                f"compute_mass_flux_hybrid stage: div_3d/p_s must be cc "
                f"(n,n)={(n, n)}; got div_3d={div_3d.shape[1:3]}, "
                f"p_s={p_s.shape[1:3]}")
        return _body(div_3d, p_s)

    return stage


# ---------------------------------------------------------------------------
# P-3D-vadv/omega-hybrid: vertical_advection_hybrid + compute_omega_hybrid.
# Both are per-COLUMN vertical ops (level-axis avg of the half-level mass_flux,
# diff/upwind over levels, pressure_from_hybrid per column) — NO horizontal
# stencil — so they tile EXACTLY like geopotential/mass_flux.  Single cc-local
# output each (..., nlev), vertical REPLICATED.  ``coord`` closed over.  These
# are on the REAL 3D-PE hydrostatic / tracer-transport path (vertical advection
# of T/tracers/momentum + pressure-velocity diagnostic).
# ---------------------------------------------------------------------------

def vertical_advection_hybrid_tile_2d(field, mass_flux, p_s, coord, a_i, a_j, nl: int):
    """Per-tile ``vertical_advection_hybrid`` (3D PE, hybrid coord).  ``field``
    ``(F,n,n,nlev)``, ``mass_flux`` ``(F,n,n,nlev+1)``, ``p_s`` ``(F,n,n)``
    FACE-REPLICATED.  Slices the cc tile ``[a:a+nl]`` (no halo) and runs the
    per-column upwind vertical advection; returns ``(F,nl,nl,nlev)``."""
    from legoesm.grids.vertical import vertical_advection_hybrid

    if field.shape[1] != field.shape[2] or mass_flux.shape[1:3] != field.shape[1:3] \
            or p_s.shape[1:3] != field.shape[1:3]:
        raise ValueError(
            f"vertical_advection_hybrid_tile_2d: field cc-square + mass_flux/p_s "
            f"matching horizontal; got field={field.shape[1:3]}, "
            f"mass_flux={mass_flux.shape[1:3]}, p_s={p_s.shape[1:3]}")

    def _s2(arr):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

    return vertical_advection_hybrid(_s2(field), _s2(mass_flux), _s2(p_s), coord)


def make_tiled_vertical_advection_hybrid_stage_2d(mesh, coord, n: int, kt: int):
    """Sharded ``vertical_advection_hybrid`` on a ``(6, kt, kt)`` mesh.
    ``stage(field, mass_flux, p_s) -> tend``; tile-sharded out (exact cc
    partition, vertical replicated).  ``coord`` closed over."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)                 # 2D cc (p_s)
    fw = P("face", None, None, None)           # 4D field / mass_flux
    cz = P("face", "tile_i", "tile_j", None)   # 4D tend out

    @partial(shard_map, mesh=mesh, in_specs=(fw, fw, fo), out_specs=cz,
             check_vma=False)
    def _body(field, mass_flux, p_s):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return vertical_advection_hybrid_tile_2d(field, mass_flux, p_s, coord,
                                                 a_i, a_j, nl)

    def stage(field, mass_flux, p_s):
        if field.shape[1:3] != (n, n) or mass_flux.shape[1:3] != (n, n) \
                or p_s.shape[1:3] != (n, n):
            raise ValueError(
                f"vertical_advection_hybrid stage: field/mass_flux/p_s must be "
                f"cc (n,n)={(n, n)}; got field={field.shape[1:3]}, "
                f"mass_flux={mass_flux.shape[1:3]}, p_s={p_s.shape[1:3]}")
        return _body(field, mass_flux, p_s)

    return stage


def compute_omega_hybrid_tile_2d(mass_flux, p_s, dp_s_dt, coord, a_i, a_j, nl: int):
    """Per-tile ``compute_omega_hybrid`` (3D PE, hybrid coord).  ``mass_flux``
    ``(F,n,n,nlev+1)``, ``p_s``/``dp_s_dt`` ``(F,n,n)`` FACE-REPLICATED.  Slices
    the cc tile ``[a:a+nl]`` (no halo) and runs the per-column omega diagnostic
    (``B_full*dp_s/dt + F_full``); returns ``(F,nl,nl,nlev)``."""
    from legoesm.grids.vertical import compute_omega_hybrid

    if mass_flux.shape[1] != mass_flux.shape[2] \
            or p_s.shape[1:3] != mass_flux.shape[1:3] \
            or dp_s_dt.shape[1:3] != mass_flux.shape[1:3]:
        raise ValueError(
            f"compute_omega_hybrid_tile_2d: mass_flux cc-square + p_s/dp_s_dt "
            f"matching horizontal; got mass_flux={mass_flux.shape[1:3]}, "
            f"p_s={p_s.shape[1:3]}, dp_s_dt={dp_s_dt.shape[1:3]}")

    def _s2(arr):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

    return compute_omega_hybrid(_s2(mass_flux), _s2(p_s), _s2(dp_s_dt), coord)


def make_tiled_compute_omega_hybrid_stage_2d(mesh, coord, n: int, kt: int):
    """Sharded ``compute_omega_hybrid`` on a ``(6, kt, kt)`` mesh.
    ``stage(mass_flux, p_s, dp_s_dt) -> omega``; tile-sharded out (exact cc
    partition, vertical replicated).  ``coord`` closed over."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)                 # 2D cc (p_s, dp_s_dt)
    fw = P("face", None, None, None)           # 4D mass_flux
    cz = P("face", "tile_i", "tile_j", None)   # 4D omega out

    @partial(shard_map, mesh=mesh, in_specs=(fw, fo, fo), out_specs=cz,
             check_vma=False)
    def _body(mass_flux, p_s, dp_s_dt):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return compute_omega_hybrid_tile_2d(mass_flux, p_s, dp_s_dt, coord,
                                            a_i, a_j, nl)

    def stage(mass_flux, p_s, dp_s_dt):
        if mass_flux.shape[1:3] != (n, n) or p_s.shape[1:3] != (n, n) \
                or dp_s_dt.shape[1:3] != (n, n):
            raise ValueError(
                f"compute_omega_hybrid stage: mass_flux/p_s/dp_s_dt must be cc "
                f"(n,n)={(n, n)}; got mass_flux={mass_flux.shape[1:3]}, "
                f"p_s={p_s.shape[1:3]}, dp_s_dt={dp_s_dt.shape[1:3]}")
        return _body(mass_flux, p_s, dp_s_dt)

    return stage


# ---------------------------------------------------------------------------
# P-3D-vertical-transport COMPOSED stage: the three cc-local vertical ops
# (compute_mass_flux_hybrid -> vertical_advection_hybrid (of a field) +
# compute_omega_hybrid) fused into ONE shard_map body — the cc tile is sliced
# ONCE and the ops chain locally, with NO inter-op gather/reshard between them
# (all per-column, no halo).  This is the ASSEMBLY direction (vs three separate
# tiled stages with a gather between each).  No new numerics — calls the raw
# vertical.py ops on the sliced tile.  ``coord`` closed over.
# ---------------------------------------------------------------------------

def make_tiled_vertical_pe_stage_2d(mesh, coord, n: int, kt: int):
    """Composed vertical 3D-PE transport stage on a ``(6, kt, kt)`` mesh.
    ``stage(div_3d, field, p_s, dp_s_dt) -> (mass_flux, tend, omega, D_total_p)``:
    div_3d/field 4D ``(6,n,n,nlev)`` + p_s/dp_s_dt 2D cc FACE-REPLICATED; all
    outputs tile-sharded (exact cc partition, vertical replicated).  The cc tile
    is sliced ONCE then mass_flux -> vertical_advection -> omega chain locally —
    bit-identical to running each global op then slicing (all per-column)."""
    from legoesm.grids.vertical import (
        compute_mass_flux_hybrid,
        compute_omega_hybrid,
        vertical_advection_hybrid,
    )
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fo = P("face", None, None)                 # 2D cc (p_s, dp_s_dt)
    fw = P("face", None, None, None)           # 4D div_3d, field
    cz = P("face", "tile_i", "tile_j", None)   # 4D outs

    @partial(shard_map, mesh=mesh, in_specs=(fw, fw, fo, fo),
             out_specs=(cz, cz, cz, cz), check_vma=False)
    def _body(div_3d, field, p_s, dp_s_dt):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s2(arr):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

        div_t, field_t, ps_t, dps_t = _s2(div_3d), _s2(field), _s2(p_s), _s2(dp_s_dt)
        mass_flux, d_total_p = compute_mass_flux_hybrid(div_t, ps_t, coord)
        tend = vertical_advection_hybrid(field_t, mass_flux, ps_t, coord)
        omega = compute_omega_hybrid(mass_flux, ps_t, dps_t, coord)
        return mass_flux, tend, omega, d_total_p

    def stage(div_3d, field, p_s, dp_s_dt):
        if div_3d.shape[1:3] != (n, n) or field.shape[1:3] != (n, n) \
                or p_s.shape[1:3] != (n, n) or dp_s_dt.shape[1:3] != (n, n):
            raise ValueError(
                f"vertical_pe stage: div_3d/field/p_s/dp_s_dt must be cc "
                f"(n,n)={(n, n)}; got div_3d={div_3d.shape[1:3]}, "
                f"field={field.shape[1:3]}, p_s={p_s.shape[1:3]}, "
                f"dp_s_dt={dp_s_dt.shape[1:3]}")
        return _body(div_3d, field, p_s, dp_s_dt)

    return stage


# ---------------------------------------------------------------------------
# P-3D-momentum COMPOSED stage: the full ``fv3_hydrostatic_tendencies`` D-grid
# MOMENTUM path (du_d_dt, dv_d_dt) — the 3D analogue of the SW momentum
# capstone, and the genuine 3D >6-device unlock.  Replicates
# ``primitive_eq_cdgrid.py:307-479`` base case (div_damp=0, hyperdiff=0,
# corner_div_damp=0, KE-heat off, non-duogrid, use_fv3_a2b_zeta_corner=False):
#   dgrid_to_center_vector -> Phi -> B=KE+Phi ; dgrid_vorticity(u_d,v_d) -> zeta
#   -> [PACKED SCALAR in-stage halo {zeta, B, inv_T, ln_ps(, hf)}]
#   -> interp_center_to_corner(zeta)+f_corner ; arakawa_lamb_gradient(B)
#   -> PGF: arakawa_lamb_gradient(ln_ps_hi) + T_corner=1/interp(inv_T) [+ hf]
#   -> du_d_dt = zeta_corner*v_d - dB_dx - pg_corr_x  (dv symmetric).
# UNLIKE the SW momentum, the 3D-PE momentum lives at the D-grid CORNERS
# ``(n+1, n+1)`` (u_d/v_d ARE corner winds) and needs NO vector halo — only the
# cc-field SCALAR halos (the in-stage analogue of the production's
# ``packed_pad_halo_4d`` stage pack).  All intermediates (B, zeta, inv_T, ln_ps,
# hf) have NO global pre-pad, so they are halo-exchanged WITHIN the shard_map via
# the unwrapped ``make_tiled_pad_body`` (ndim=4 — the 4D in-stage scalar halo,
# whose bit-identity to ``pad_halo_4d`` is pinned by test_tiled_pad_body's ndim=4
# lane).  The PGF higher-precision cast (``_pg_dt``) is replicated verbatim so the
# stage is faithful at any dtype; in the x64 gate it is a no-op.
# ---------------------------------------------------------------------------

def make_tiled_fv3_hydrostatic_momentum_stage_2d(mesh, cdgrid, coord, n: int,
                                                 kt: int, nlev: int):
    """Build the sharded tiled ``fv3_hydrostatic_tendencies`` D-grid MOMENTUM
    stage on a ``(6, kt, kt)`` mesh (axes ``("face","tile_i","tile_j")``).

    Returns ``stage(u_d, v_d, T, p_s, phis) -> (du_d_dt, dv_d_dt)`` where the
    state inputs are FACE-SHARDED, TILE-REPLICATED (``P("face",None,None,...)``):
    ``u_d``/``v_d`` D-grid CORNER winds ``(6, n+1, n+1, nlev)``, ``T``
    ``(6, n, n, nlev)`` cc, ``p_s``/``phis`` ``(6, n, n)`` cc.  Both outputs are
    corner-staggered tiles ``P("face","tile_i","tile_j",None)``; gathered
    ``(6, kt*(nl+1), kt*(nl+1), nlev)`` reassemble lower-owns-shared (drop the
    duplicated shared corner face) to ``(6, n+1, n+1, nlev)``.

    Base cut (matches ``primitive_eq_cdgrid.py`` defaults): orthogonal-rotation
    (non-duogrid) only; div_damp=0, hyperdiff=0, corner_div_damp=0, KE-heat off,
    ``use_fv3_a2b_zeta_corner=False``.  All static metrics pass as face-sharded
    shard_map inputs (NEVER closed over a full ``(6,...)`` array — that
    broadcasts the output back to face extent 6, the codex U4a HIGH).  ``coord``
    (sigma OR hybrid; 1-D vertical arrays) is closed over and the geopotential /
    hybrid-factor cores are dispatched on its type.  No new numerics.
    """
    from legoesm.core.operators_cdgrid import (
        arakawa_lamb_gradient_core,
        dgrid_to_center_vector,
        dgrid_vorticity_core,
        interp_center_to_corner,
    )
    from legoesm.core.precision import resolve_dtype
    from legoesm.grids.vertical import (
        HybridSigmaPressureCoordinate,
        compute_geopotential,
        compute_geopotential_hybrid,
        pressure_from_hybrid,
    )
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body

    from legoesm import constants

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_momentum_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_hydrostatic_momentum_stage_2d: base cut supports "
            "the orthogonal-rotation (non-duogrid) cube only; the in-stage "
            "scalar halo does not yet carry the duogrid kinked->extended remap.")
    # codex MED: guard the CLOSED-OVER vertical coord against nlev.  A wrong
    # singleton-level coord (n_levels=1) would broadcast SILENTLY through
    # _geo(...)/pressure_from_hybrid(...)/coord.B_full*... and produce plausible
    # but wrong tendencies (the state shape guard below only checks the 4D
    # trailing nlev of u_d/v_d/T, not the coord's vertical extent).
    if getattr(coord, "n_levels", nlev) != nlev:
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_momentum_stage_2d: coord.n_levels="
            f"{coord.n_levels} != nlev={nlev}")
    nl = n // kt
    R_d = constants.R_d
    _hybrid = isinstance(coord, HybridSigmaPressureCoordinate)
    _geo = compute_geopotential_hybrid if _hybrid else compute_geopotential

    # Static metrics (face-sharded, tile-replicated -> sliced per tile).
    cosa_corner = cdgrid.cosa_corner               # (6, n+1, n+1)
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x  # (6,n,n+1)/(6,n+1,n)
    area = grid.area                               # (6, n, n)
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01  # (6, n+1, n+1)
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    f_corner = cdgrid.f_corner                     # (6, n+1, n+1)
    offsets = grid.halo_interp_offsets             # (6, 4, n) — non-duogrid

    # Unwrapped in-stage SCALAR halo body (ndim=4: global field rank 4, the
    # per-device tile is 3-D (nl, nl, C)).  Built once; called per cc-field.
    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)

    fo = P("face", None, None)                     # 2D-face metric / cc 2D
    fw = P("face", None, None, None)               # 4D state (u_d, v_d, T)
    cz = P("face", "tile_i", "tile_j", None)       # 4D corner outputs

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fo, fo)          # u_d, v_d, T, p_s, phis
                       + (fo,) * 4                  # cosa_corner, dxe, dye, area
                       + (fo,) * 4                  # gc00..gc11
                       + (fo,)                      # f_corner
                       + (P(),),                    # offsets (replicated)
             out_specs=(cz, cz), check_vma=False)
    def _body(u_d, v_d, T, p_s, phis,
              cosa_c, dxe, dye, ar,
              c00, c01, c10, c11, fco, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            """Tile slice of a face-shard at (a_i, a_j) -> (1, si, sj[, C])."""
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        # ---- (1/3/5) Bernoulli B = KE + Phi (cc-local, NO halo) ----
        u_d_t = _s(u_d, nl + 1, nl + 1)        # (1, nl+1, nl+1, nlev) corner
        v_d_t = _s(v_d, nl + 1, nl + 1)
        T_t = _s(T, nl, nl)                    # (1, nl, nl, nlev) cc
        p_s_t = _s(p_s, nl, nl)                # (1, nl, nl) cc
        phis_t = _s(phis, nl, nl)
        u_cell, v_cell = dgrid_to_center_vector(u_d_t, v_d_t)   # (1, nl, nl, nlev)
        Phi = _geo(T_t, p_s_t, coord, phis_t)
        B = 0.5 * (u_cell ** 2 + v_cell ** 2) + Phi            # (1, nl, nl, nlev)

        # ---- (6) relative vorticity (cc-local from corner winds, NO halo) ----
        zeta = dgrid_vorticity_core(
            u_d_t, v_d_t, _s(cosa_c, nl + 1, nl + 1),
            _s(dxe, nl, nl + 1), _s(dye, nl + 1, nl),
            _s(ar, nl, nl))                    # (1, nl, nl, nlev)

        # ---- cc-field SCALAR fields packed by the production stage halo ----
        inv_T = 1.0 / T_t                      # (1, nl, nl, nlev)
        ln_ps = jnp.log(p_s_t)                 # (1, nl, nl)
        # PGF higher precision (primitive_eq_cdgrid.py:452-453): cast BEFORE the
        # halo so the padded ln_ps matches the single-device reference's
        # pad_halo_auto(ln_ps_hi).  No-op when ln_ps is already _pg_dt (x64).
        _pg_dt = jnp.result_type(
            ln_ps.dtype, resolve_dtype("atm_pressure_gradient", "compute"))
        ln_ps_3d = ln_ps.astype(_pg_dt)[..., None]             # (1, nl, nl, 1)

        # ---- in-stage SCALAR halos (ndim=4 pad body; per cc-field) ----
        # Each is bit-identical to the global op's internal pad_halo_auto (the
        # single-device reference path); coalescing them into one collective is
        # a perf-only optimisation that does NOT change the padded values.
        B_pad = scalar_body(B[0], offs)[None]                  # (1, nl+2, nl+2, nlev)
        zeta_pad = scalar_body(zeta[0], offs)[None]
        invT_pad = scalar_body(inv_T[0], offs)[None]
        lnps_pad = scalar_body(ln_ps_3d[0], offs)[None]        # (1, nl+2, nl+2, 1)

        # ---- (6) zeta_corner = interp(zeta) + f_corner ----
        zeta_corner = (interp_center_to_corner(zeta_pad, cdgrid, padded=zeta_pad)
                       + _s(fco, nl + 1, nl + 1)[..., None])   # (1, nl+1, nl+1, nlev)

        # ---- (7) Bernoulli gradient at D-grid corners (Arakawa-Lamb) ----
        gc = (_s(c00, nl + 1, nl + 1), _s(c01, nl + 1, nl + 1),
              _s(c10, nl + 1, nl + 1), _s(c11, nl + 1, nl + 1))
        dB_dx, dB_dy_perp = arakawa_lamb_gradient_core(B_pad, *gc)

        # ---- (8) pressure-gradient correction at D-grid corners ----
        dln_dx_hi, dln_dy_perp_hi = arakawa_lamb_gradient_core(lnps_pad, *gc)
        T_corner = 1.0 / interp_center_to_corner(invT_pad, cdgrid, padded=invT_pad)
        T_corner_hi = T_corner.astype(_pg_dt)
        # dln_*_hi already carries the trailing axis (lnps_pad is (...,1)), so the
        # production's `dln_dx_hi[..., None]` is implicit here.
        pg_corr_x = (R_d * T_corner_hi * dln_dx_hi).astype(u_d_t.dtype)
        pg_corr_y_perp = (R_d * T_corner_hi * dln_dy_perp_hi).astype(v_d_t.dtype)
        if _hybrid:
            # grad_eta(ln p) = (B_full*p_s/p) * grad(ln p_s) — hybrid coord.
            p_full = pressure_from_hybrid(coord, p_s_t)        # (1, nl, nl, nlev)
            hf = coord.B_full * p_s_t[..., None] / p_full      # (1, nl, nl, nlev)
            hf_pad = scalar_body(hf[0], offs)[None]
            hf_corner = interp_center_to_corner(hf_pad, cdgrid, padded=hf_pad)
            pg_corr_x = pg_corr_x * hf_corner
            pg_corr_y_perp = pg_corr_y_perp * hf_corner

        # ---- (9) D-grid momentum tendencies (corner-located, pointwise) ----
        du_d_dt = zeta_corner * v_d_t - dB_dx - pg_corr_x      # (1, nl+1, nl+1, nlev)
        dv_d_dt = -zeta_corner * u_d_t - dB_dy_perp - pg_corr_y_perp
        return du_d_dt, dv_d_dt

    def stage(u_d, v_d, T, p_s, phis):
        # 4D state carries the trailing nlev (checked); p_s/phis are 2D cc.
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)), T=(T, (n, n, nlev)),
                      p_s=(p_s, (n, n)), phis=(phis, (n, n)))
        return _body(u_d, v_d, T, p_s, phis,
                     cosa_corner, dx_edge_y, dy_edge_x, area,
                     gc00, gc01, gc10, gc11, f_corner, offsets)

    return stage


# ---------------------------------------------------------------------------
# P-3D-continuity COMPOSED stage: the surface-pressure tendency dp_s/dt (step 10b
# of fv3_hydrostatic_tendencies).  Composes THREE local ops in ONE shard_map —
# dgrid_to_cgrid (D->C, within-face projection) -> cgrid_divergence (C-grid
# flux-form divergence; the staggered tile u_c/v_c carry the tile boundary faces,
# Pace layout, so NO halo) -> column-integrated divergence (per-column cumsum) ->
# dp_s/dt pointwise.  ALL LOCAL (NO in-stage halo, like the Bernoulli stage):
# dgrid_to_cgrid reads only within-face adjacent corners, cgrid_divergence reads a
# cell's own four bounding faces, the column sum is vertical (replicated per tile).
# Sigma: dp_s/dt = -p_s*D_total/(1-sigma_top) (compute_sigma_dot_and_total).
# Hybrid: dp_s/dt = -D_total_p/B_range (compute_mass_flux_hybrid).  The
# per-stage zero_mean_tendency (a GLOBAL reduction) is NOT part of this stage —
# it is applied downstream; the stage produces the pre-zero-mean tendency,
# bit-identical to the global pre-zero-mean dp_s/dt.  ``coord`` closed over.
# ---------------------------------------------------------------------------

def make_tiled_dp_s_dt_stage_2d(mesh, cdgrid, coord, n: int, kt: int, nlev: int):
    """Composed surface-pressure-tendency stage on a ``(6, kt, kt)`` mesh.
    ``stage(u_d, v_d, p_s) -> dp_s_dt``: u_d/v_d corner D-winds 4D
    ``(6,n+1,n+1,nlev)`` + p_s 2D cc ``(6,n,n)``, all FACE-REPLICATED; tile-sharded
    ``dp_s_dt`` ``(6,n,n)`` out (exact cc partition — NO shared face).  Bit-identical
    to the global FLUX-FORM continuity (primitive_eq_cdgrid sec 10b):
    ``dgrid_to_cgrid`` -> dp at C-grid faces (2-point average on the halo-1
    scalar pad) -> ``cgrid_divergence(dp·v)`` -> column sum.  ``coord`` sigma OR
    hybrid.  Excludes the downstream global zero_mean_tendency.
    """
    from legoesm.core.operators_cdgrid import (
        cgrid_divergence_local, cgrid_interp_cc_to_faces_local)
    from legoesm.grids.vertical import (
        HybridSigmaPressureCoordinate,
        dp_from_hybrid,
    )
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body
    grid = cdgrid.base
    # Fail-fast BEFORE mesh validation: the refusal must not depend on a
    # constructible device mesh (codex 2026-07-12).
    if grid.duogrid is not None:
        raise NotImplementedError(
            "make_tiled_dp_s_dt_stage_2d supports the orthogonal-rotation "
            "(non-duogrid) cube only: the in-stage scalar halo does not "
            "carry the duogrid kinked->extended remap, and the flux "
            "divergence here has no seam-flux synchronization (the global "
            "duogrid path uses cgrid_flux_divergence_sync).")
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_dp_s_dt_stage_2d: n={n} != cdgrid.n={cdgrid.n}")
    if getattr(coord, "n_levels", nlev) != nlev:
        raise ValueError(
            f"make_tiled_dp_s_dt_stage_2d: coord.n_levels={coord.n_levels} "
            f"!= nlev={nlev}")
    nl = n // kt
    _hybrid = isinstance(coord, HybridSigmaPressureCoordinate)
    if not _hybrid:
        _sigma_range = 1.0 - float(coord.sigma_half[0])   # static scalar

    cosa_u = cdgrid.cosa_u                          # (6, n+1, n)
    dy_edge_x = cdgrid.dy_edge_x                    # (6, n+1, n) — x-face length
    dx_edge_y = cdgrid.dx_edge_y                    # (6, n, n+1) — y-face length
    area = grid.area                               # (6, n, n)
    offsets = grid.halo_interp_offsets             # (6, 4, n)
    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)

    fo = P("face", None, None)                     # 2D-face metric / cc 2D
    fw = P("face", None, None, None)               # 4D winds
    co = P("face", "tile_i", "tile_j")             # 2D cc output

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fo, fo, fo, fo, fo,  # u_d,v_d,p_s,cosa_u,dye,dxe,area
                       P()),                        # offsets
             out_specs=co, check_vma=False)
    def _body(u_d, v_d, p_s, cu, dye, dxe, ar, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        # D->C (local within-face) -> dp at faces (in-stage halo-1 scalar pad
        # + shared 2-point average) -> C-grid FLUX-FORM divergence div(dp·v)
        # (local; the staggered u_c/v_c carry the tile boundary faces).
        u_c, v_c = dgrid_to_cgrid_tile_2d(u_d, v_d, cu, a_i, a_j, nl)
        p_s_t = _s(p_s, nl, nl)                      # (1, nl, nl)
        if _hybrid:
            dp_t = dp_from_hybrid(coord, p_s_t)      # (1, nl, nl, nlev)
        else:
            dp_t = p_s_t[..., None] * coord.dsigma.astype(p_s_t.dtype)
        dp_pad = scalar_body(dp_t[0], offs)[None]    # (1, nl+2, nl+2, nlev)
        dp_u, dp_v = cgrid_interp_cc_to_faces_local(dp_pad)
        div_dp = cgrid_divergence_local(
            dp_u * u_c, dp_v * v_c, _s(dye, nl + 1, nl), _s(dxe, nl, nl + 1),
            _s(ar, nl, nl))                          # (1, nl, nl, nlev)

        D_total_p = jnp.cumsum(div_dp, axis=-1)[..., -1:]   # (1, nl, nl, 1)
        if _hybrid:
            dp_s_dt = -D_total_p[..., 0] / coord.B_range
        else:
            dp_s_dt = -D_total_p[..., 0] / _sigma_range
        return dp_s_dt                               # (1, nl, nl)

    def stage(u_d, v_d, p_s):
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)), p_s=(p_s, (n, n)))
        return _body(u_d, v_d, p_s, cosa_u, dy_edge_x, dx_edge_y, area,
                     offsets)

    return stage


# ---------------------------------------------------------------------------
# P-3D-thermo COMPOSED stage: the temperature tendency dT/dt (step 11 of
# fv3_hydrostatic_tendencies, primitive_eq_cdgrid.py:856-871).  The advective +
# adiabatic core of the thermodynamic equation:
#   dT/dt = -(u_cell*dT/dx + v_cell*dT/dy)          [horizontal advection]
#           + vert_adv_T                            [upstream vertical stage]
#           + kappa*T*omega/p_adiab                 [adiabatic warming]
#           + kappa*T*(v . grad ln p_s) [* hybrid grad-eta factor]
# The ONLY horizontally-coupled work is the two CENTRED cc gradients dT and
# d ln(p_s) -> a 1-cell SCALAR halo on {T, ln_ps_3d} exchanged IN-STAGE via the
# ndim=4 make_tiled_pad_body (the momentum-stage primitive; bit-identical to the
# global pad_halo_4d).  u_cell/v_cell (dgrid_to_center_vector, within-face 4-pt),
# p_adiab (pressure_from_*(coord, p_s) per-column), and the adiabatic/hybrid
# pointwise terms are all cc-LOCAL.  omega and vert_adv_T are UPSTREAM stage
# outputs (omega carries the GLOBAL zero_mean_tendency baked into dp_s/dt; both
# per-column-local), so they enter as inputs.  The centred-gradient stencil is
# the shared gradient_{x,y}_3d_core (no re-derived numerics).
# ---------------------------------------------------------------------------

def _tiled_scalar_horiz_advect(field_t, u_cell, v_cell, dx_t, dy_t,
                               scalar_body, offs):
    """``-(u . grad field)`` for a cell-centre scalar (or packed tracer block) on
    one tile.

    THE shared horizontal scalar-advection: in-stage scalar halo
    (``scalar_body``, the ndim=4 ``make_tiled_pad_body``) -> centred cc gradients
    (``gradient_{x,y}_3d_core``) -> advective form.  Used by the thermodynamic
    stage (temperature), the single-tracer stage, AND the packed tracer stage so
    the identical numerics live in one place (no per-field copy-paste).

    ``field_t`` is a single tile, either ``(1, nl, nl, nlev)`` (a scalar: T or one
    tracer) or ``(1, nl, nl, nlev, n_tracers)`` (a packed tracer block).
    ``u_cell``/``v_cell`` are the cc winds ``(1, nl, nl, nlev)``; ``dx_t``/
    ``dy_t`` the tile cc metrics ``(1, nl, nl)``.  For the packed case the tracer
    axis is folded into the level axis for ONE halo + gradient pass (mirroring the
    serial ``advective_tracer_tendency`` ``q_flat`` reshape), then unfolded for the
    per-tracer ``-(u[...,None] . grad q)`` advect.  Returns the input's shape.
    """
    from legoesm.core.operators_3d import gradient_x_3d_core, gradient_y_3d_core
    packed = field_t.ndim == 5
    if packed:
        nlev, nt = field_t.shape[-2], field_t.shape[-1]
        f_flat = field_t.reshape(*field_t.shape[:3], nlev * nt)  # fold -> ndim 4
    else:
        f_flat = field_t
    f_pad = scalar_body(f_flat[0], offs)[None]            # (1, nl+2, nl+2, C)
    df_dx = gradient_x_3d_core(f_pad, dx_t)
    df_dy = gradient_y_3d_core(f_pad, dy_t)
    if packed:
        df_dx = df_dx.reshape(*field_t.shape)             # unfold -> ndim 5
        df_dy = df_dy.reshape(*field_t.shape)
        return -(u_cell[..., None] * df_dx + v_cell[..., None] * df_dy)
    return -(u_cell * df_dx + v_cell * df_dy)


def make_tiled_fv3_hydrostatic_thermo_stage_2d(mesh, cdgrid, coord, n: int,
                                              kt: int, nlev: int, *,
                                              p_floor: float):
    """Composed thermodynamic-tendency stage on a ``(6, kt, kt)`` mesh
    (axes ``("face","tile_i","tile_j")``).

    Returns ``stage(u_d, v_d, T, p_s, omega, vert_adv_T) -> dT_dt`` where the
    inputs are FACE-SHARDED, TILE-REPLICATED (``P("face",None,None,...)``):
    ``u_d``/``v_d`` D-grid CORNER winds ``(6, n+1, n+1, nlev)``; ``T``/``omega``/
    ``vert_adv_T`` cc ``(6, n, n, nlev)``; ``p_s`` cc ``(6, n, n)``.  The output
    ``dT_dt`` ``(6, n, n, nlev)`` is the EXACT cc partition
    (``P("face","tile_i","tile_j",None)`` — no shared face), gathered == the
    global dT/dt.

    Base cut (matches ``primitive_eq_cdgrid.py`` defaults): non-duogrid;
    A_h=0, hyperdiff_coeff=0, T_diss_coeff=0, no physics tendency — the advective
    + adiabatic core only.  ``coord`` (sigma OR hybrid; 1-D vertical arrays)
    closed over; the ``p_full`` core is dispatched on its type.  ``p_floor`` is
    the adiabatic-pressure floor (``CDGridPrimitiveEquationConfig.p_floor`` [Pa]).
    No new numerics — the centred gradient is ``gradient_{x,y}_3d_core``.
    """
    from legoesm import constants
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    from legoesm.core.operators_3d import gradient_x_3d_core, gradient_y_3d_core
    from legoesm.grids.vertical import (
        pressure_from_hybrid, pressure_from_sigma,
        HybridSigmaPressureCoordinate)
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_thermo_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_hydrostatic_thermo_stage_2d: base cut supports the "
            "orthogonal-rotation (non-duogrid) cube only; the in-stage scalar "
            "halo does not carry the duogrid kinked->extended remap.")
    # codex MED (momentum stage): a wrong singleton-level coord (n_levels=1)
    # would broadcast SILENTLY through pressure_from_*/coord.B_full* and produce
    # plausible-but-wrong tendencies (the state guard only checks the 4D nlev).
    if getattr(coord, "n_levels", nlev) != nlev:
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_thermo_stage_2d: coord.n_levels="
            f"{coord.n_levels} != nlev={nlev}")
    if not (float(p_floor) > 0.0):
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_thermo_stage_2d: p_floor must be a "
            f"positive pressure [Pa]; got {p_floor}")
    nl = n // kt
    kappa = constants.kappa
    _hybrid = isinstance(coord, HybridSigmaPressureCoordinate)
    _p_floor = float(p_floor)

    # Static cc metrics (face-sharded, tile-replicated -> sliced per tile).
    dx = grid.dx                                   # (6, n, n)
    dy = grid.dy                                   # (6, n, n)
    offsets = grid.halo_interp_offsets             # (6, 4, n) — non-duogrid

    # ndim=4 in-stage SCALAR halo body (same primitive as the momentum stage).
    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)

    fo = P("face", None, None)                     # 2D-face metric / cc 2D
    fw = P("face", None, None, None)               # 4D state / cc 4D
    co = P("face", "tile_i", "tile_j", None)       # 4D cc output (exact partition)

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fo, fw, fw)      # u_d,v_d,T,p_s,omega,vert_adv_T
                       + (fo, fo)                   # dx, dy
                       + (P(),),                    # offsets (replicated)
             out_specs=co, check_vma=False)
    def _body(u_d, v_d, T, p_s, omega, vert_adv_T, dx_, dy_, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            """Tile slice of a face-shard at (a_i, a_j) -> (1, si, sj[, C])."""
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        u_d_t = _s(u_d, nl + 1, nl + 1)            # (1, nl+1, nl+1, nlev) corner
        v_d_t = _s(v_d, nl + 1, nl + 1)
        T_t = _s(T, nl, nl)                        # (1, nl, nl, nlev) cc
        p_s_t = _s(p_s, nl, nl)                    # (1, nl, nl) cc
        omega_t = _s(omega, nl, nl)                # (1, nl, nl, nlev) cc
        vert_adv_T_t = _s(vert_adv_T, nl, nl)      # (1, nl, nl, nlev) cc
        dx_t = _s(dx_, nl, nl)                     # (1, nl, nl)
        dy_t = _s(dy_, nl, nl)

        # cc velocity (within-face 4-pt avg; NO halo — like the momentum stage).
        u_cell, v_cell = dgrid_to_center_vector(u_d_t, v_d_t)  # (1, nl, nl, nlev)

        # p_adiab from the coord (per-column cc-local; bit-identical to the
        # global p_full = pressure_from_*(coord, p_s) then the jnp.maximum floor).
        if _hybrid:
            p_full = pressure_from_hybrid(coord, p_s_t)        # (1, nl, nl, nlev)
        else:
            p_full = pressure_from_sigma(coord.sigma_full, p_s_t)
        p_adiab = jnp.maximum(p_full, _p_floor)

        ln_ps_3d = jnp.log(p_s_t)[..., None]                   # (1, nl, nl, 1)

        # ---- in-stage SCALAR halos {T, ln_ps_3d} (ndim=4 pad body) ----
        # Each is bit-identical to the global op's pad_halo_4d (the single-device
        # reference path the gate composes); the production packs T + ln_ps into
        # ONE collective — coalescing here is a perf-only change to the SAME pads.
        lnps_pad = scalar_body(ln_ps_3d[0], offs)[None]        # (1, nl+2, nl+2, 1)

        # ---- horizontal advection: -(u . grad T) (shared scalar advect) ----
        horiz_adv_T = _tiled_scalar_horiz_advect(
            T_t, u_cell, v_cell, dx_t, dy_t, scalar_body, offs)

        # ---- adiabatic: kappa*T*omega/p + kappa*T*(v . grad ln p_s) ----
        dln_ps_dx = gradient_x_3d_core(lnps_pad, dx_t)[..., 0]  # (1, nl, nl)
        dln_ps_dy = gradient_y_3d_core(lnps_pad, dy_t)[..., 0]
        adiabatic = kappa * T_t * omega_t / p_adiab
        v_dot_grad_lnps = (u_cell * dln_ps_dx[..., None]
                           + v_cell * dln_ps_dy[..., None])
        if _hybrid:
            # grad_eta(ln p) = (B_full*p_s/p) * grad(ln p_s) — hybrid coord.
            v_dot_grad_lnps = v_dot_grad_lnps * (
                coord.B_full * p_s_t[..., None] / p_adiab)
        adiabatic = adiabatic + kappa * T_t * v_dot_grad_lnps

        return horiz_adv_T + vert_adv_T_t + adiabatic          # (1, nl, nl, nlev)

    def stage(u_d, v_d, T, p_s, omega, vert_adv_T):
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)),
                      T=(T, (n, n, nlev)), p_s=(p_s, (n, n)),
                      omega=(omega, (n, n, nlev)),
                      vert_adv_T=(vert_adv_T, (n, n, nlev)))
        return _body(u_d, v_d, T, p_s, omega, vert_adv_T, dx, dy, offsets)

    return stage


def make_tiled_fv3_tracer_advection_stage_2d(mesh, cdgrid, n: int, kt: int,
                                             nlev: int):
    """Tiled advective tracer tendency on a ``(6, kt, kt)`` mesh
    (axes ``("face","tile_i","tile_j")``) — the cube-MOIST np>6 unlock.

    Returns ``stage(u_d, v_d, q, vert_adv_q) -> dq_dt`` for ONE cc tracer ``q``
    ``(6, n, n, nlev)``: the SAME advective horizontal transport the
    thermodynamic stage applies to temperature (``-(u . grad q)`` via the shared
    in-stage scalar halo + centred cc gradients), PLUS the per-column-local
    vertical advection ``vert_adv_q`` (an upstream cc-local stage output, exactly
    like ``vert_adv_T``).  Tiled analogue of the serial
    ``advective_tracer_tendency`` (FV3 cube moist transport), base case
    ``hyperdiff_coeff=0`` — q_v/q_c/q_r ride this instead of the replicated
    ``step_with_physics`` path that caps cube-moist at np<=6.

    Inputs FACE-SHARDED, TILE-REPLICATED (``P("face",None,None,None)``):
    ``u_d``/``v_d`` D-grid CORNER winds ``(6, n+1, n+1, nlev)``; ``q``/
    ``vert_adv_q`` cc ``(6, n, n, nlev)``.  Output ``dq_dt`` ``(6, n, n, nlev)``
    is the EXACT cc partition (``P("face","tile_i","tile_j",None)``), gathered ==
    the global ``dq/dt``.  No new numerics — reuses ``dgrid_to_center_vector`` +
    the shared ``_tiled_scalar_horiz_advect``.
    """
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_tracer_advection_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_tracer_advection_stage_2d: base cut supports the "
            "orthogonal-rotation (non-duogrid) cube only; the in-stage scalar "
            "halo does not carry the duogrid kinked->extended remap.")
    nl = n // kt
    dx = grid.dx                                   # (6, n, n)
    dy = grid.dy
    offsets = grid.halo_interp_offsets             # (6, 4, n) — non-duogrid
    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)

    fw = P("face", None, None, None)               # 4D state
    fo = P("face", None, None)                     # 2D cc metric
    co = P("face", "tile_i", "tile_j", None)       # 4D cc output (exact partition)

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fw)             # u_d, v_d, q, vert_adv_q
                       + (fo, fo)                   # dx, dy
                       + (P(),),                    # offsets (replicated)
             out_specs=co, check_vma=False)
    def _body(u_d, v_d, q, vert_adv_q, dx_, dy_, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        u_d_t = _s(u_d, nl + 1, nl + 1)            # (1, nl+1, nl+1, nlev) corner
        v_d_t = _s(v_d, nl + 1, nl + 1)
        q_t = _s(q, nl, nl)                        # (1, nl, nl, nlev) cc
        vadv_t = _s(vert_adv_q, nl, nl)            # (1, nl, nl, nlev) cc
        dx_t = _s(dx_, nl, nl)
        dy_t = _s(dy_, nl, nl)

        # cc velocity (within-face 4-pt avg; NO halo — like the thermo stage).
        u_cell, v_cell = dgrid_to_center_vector(u_d_t, v_d_t)
        horiz = _tiled_scalar_horiz_advect(
            q_t, u_cell, v_cell, dx_t, dy_t, scalar_body, offs)
        return horiz + vadv_t

    def stage(u_d, v_d, q, vert_adv_q):
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)),
                      q=(q, (n, n, nlev)),
                      vert_adv_q=(vert_adv_q, (n, n, nlev)))
        return _body(u_d, v_d, q, vert_adv_q, dx, dy, offsets)

    return stage


def make_tiled_fv3_tracer_pack_advection_stage_2d(mesh, cdgrid, n: int, kt: int,
                                                  nlev: int):
    """Tiled advective tendency for a PACKED tracer block on a ``(6, kt, kt)``
    mesh (axes ``("face","tile_i","tile_j")``) — increment 2 of the cube-MOIST
    np>6 step.

    Returns ``stage(u_d, v_d, q, vert_adv_q) -> dq_dt`` where ``q``/``vert_adv_q``
    are ``(6, n, n, nlev, n_tracers)`` (e.g. q_v/q_c/q_r): each tracer advected by
    ``-(u . grad q)`` with the tracer axis folded into the level axis for ONE
    in-stage scalar halo + gradient pass (mirroring serial
    ``advective_tracer_tendency``'s ``q_flat`` reshape), then per-column
    ``vert_adv_q`` added.  Same numerics as the single-tracer stage (the shared
    ``_tiled_scalar_horiz_advect`` packed branch); ``n_tracers`` is dynamic (the
    5-D specs carry any trailing size).  Output ``dq_dt``
    ``(6, n, n, nlev, n_tracers)`` is the EXACT cc partition
    (``P("face","tile_i","tile_j",None,None)``), gathered == the global dq/dt.
    """
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_tracer_pack_advection_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_tracer_pack_advection_stage_2d: base cut supports "
            "the orthogonal-rotation (non-duogrid) cube only.")
    nl = n // kt
    dx = grid.dx
    dy = grid.dy
    offsets = grid.halo_interp_offsets
    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)

    fw = P("face", None, None, None)               # 4D corner winds
    fw5 = P("face", None, None, None, None)         # 5D packed tracer block
    fo = P("face", None, None)                     # 2D cc metric
    co5 = P("face", "tile_i", "tile_j", None, None)  # 5D cc output (exact partition)

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw5, fw5)           # u_d, v_d, q, vert_adv_q
                       + (fo, fo)                   # dx, dy
                       + (P(),),                    # offsets (replicated)
             out_specs=co5, check_vma=False)
    def _body(u_d, v_d, q, vert_adv_q, dx_, dy_, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        u_d_t = _s(u_d, nl + 1, nl + 1)            # (1, nl+1, nl+1, nlev) corner
        v_d_t = _s(v_d, nl + 1, nl + 1)
        q_t = _s(q, nl, nl)                        # (1, nl, nl, nlev, nt) cc
        vadv_t = _s(vert_adv_q, nl, nl)            # (1, nl, nl, nlev, nt) cc
        dx_t = _s(dx_, nl, nl)
        dy_t = _s(dy_, nl, nl)

        u_cell, v_cell = dgrid_to_center_vector(u_d_t, v_d_t)  # (1, nl, nl, nlev)
        horiz = _tiled_scalar_horiz_advect(
            q_t, u_cell, v_cell, dx_t, dy_t, scalar_body, offs)  # packed branch
        return horiz + vadv_t

    def stage(u_d, v_d, q, vert_adv_q):
        if q.ndim != 5 or vert_adv_q.shape != q.shape:
            raise ValueError(
                "make_tiled_fv3_tracer_pack_advection_stage_2d: q and vert_adv_q "
                f"must be (6,n,n,nlev,n_tracers); got q={q.shape}, "
                f"vert_adv_q={vert_adv_q.shape}")
        nt = q.shape[-1]
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)),
                      q=(q, (n, n, nlev, nt)),
                      vert_adv_q=(vert_adv_q, (n, n, nlev, nt)))
        return _body(u_d, v_d, q, vert_adv_q, dx, dy, offsets)

    return stage


def make_tiled_fv3_moist_tracer_tendency_stage_2d(mesh, cdgrid, n: int, kt: int,
                                                  nlev: int, *,
                                                  column_tracer_physics_fn):
    """Tiled MOIST tracer tendency on a ``(6, kt, kt)`` mesh — increment 3 of the
    cube-MOIST np>6 step.

    Returns ``stage(u_d, v_d, T, p_s, q_pack, vert_adv_q) -> dq_pack`` where
    ``q_pack``/``vert_adv_q`` are ``(6, n, n, nlev, 3)`` with the tracer order
    ``[q_v, q_c, q_r]``.  Combines the sub-face tiled advection
    (``-(u . grad q) + vert_adv_q``, the increment-2 packed path) with an INJECTED
    per-tile column physics
    ``column_tracer_physics_fn(T_t, p_s_t, q_v_t, q_c_t, q_r_t) -> (dq_v, dq_c,
    dq_r)`` (each ``(1, nl, nl, nlev)``) added to the advected pack.  Dependency
    injection keeps this core stage PHYSICS-AGNOSTIC: the caller builds the fn
    from a column-local scheme (e.g.
    ``legoesm.atmosphere.forcing.idealized.kessler_forcing.kessler_column_tendencies`` with
    sigma_coord/dt/config closed over).  Warm-rain microphysics is column-local
    (no halo) so the per-tile call is bit-identical to a global apply.  Output
    ``dq_pack`` ``(6, n, n, nlev, 3)`` is the EXACT cc partition
    (``P("face","tile_i","tile_j",None,None)``), gathered == the global moist
    tracer tendency (dynamics + physics).
    """
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body

    if column_tracer_physics_fn is None:
        raise ValueError(
            "make_tiled_fv3_moist_tracer_tendency_stage_2d: "
            "column_tracer_physics_fn is required (inject a column-local physics, "
            "e.g. built from kessler_column_tendencies).")
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_moist_tracer_tendency_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_moist_tracer_tendency_stage_2d: base cut supports "
            "the orthogonal-rotation (non-duogrid) cube only.")
    nl = n // kt
    dx = grid.dx
    dy = grid.dy
    offsets = grid.halo_interp_offsets
    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    _phys = column_tracer_physics_fn

    fw = P("face", None, None, None)               # 4D corner winds / T
    fw5 = P("face", None, None, None, None)         # 5D packed tracer block
    fo = P("face", None, None)                     # 2D cc metric / p_s
    co5 = P("face", "tile_i", "tile_j", None, None)  # 5D cc output

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fo, fw5, fw5)   # u_d, v_d, T, p_s, q, vert_adv_q
                       + (fo, fo)                   # dx, dy
                       + (P(),),                    # offsets (replicated)
             out_specs=co5, check_vma=False)
    def _body(u_d, v_d, T, p_s, q, vert_adv_q, dx_, dy_, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        u_d_t = _s(u_d, nl + 1, nl + 1)            # (1, nl+1, nl+1, nlev) corner
        v_d_t = _s(v_d, nl + 1, nl + 1)
        T_t = _s(T, nl, nl)                        # (1, nl, nl, nlev) cc
        p_s_t = _s(p_s, nl, nl)                    # (1, nl, nl) cc
        q_t = _s(q, nl, nl)                        # (1, nl, nl, nlev, 3) cc
        vadv_t = _s(vert_adv_q, nl, nl)
        dx_t = _s(dx_, nl, nl)
        dy_t = _s(dy_, nl, nl)

        u_cell, v_cell = dgrid_to_center_vector(u_d_t, v_d_t)  # (1, nl, nl, nlev)
        adv = _tiled_scalar_horiz_advect(
            q_t, u_cell, v_cell, dx_t, dy_t, scalar_body, offs) + vadv_t
        # Injected per-tile column physics (column-local -> SPMD-local).
        dq_v, dq_c, dq_r = _phys(T_t, p_s_t, q_t[..., 0], q_t[..., 1], q_t[..., 2])
        phys_pack = jnp.stack([dq_v, dq_c, dq_r], axis=-1)  # (1, nl, nl, nlev, 3)
        return adv + phys_pack

    def stage(u_d, v_d, T, p_s, q, vert_adv_q):
        if q.ndim != 5 or q.shape[-1] != 3 or vert_adv_q.shape != q.shape:
            raise ValueError(
                "make_tiled_fv3_moist_tracer_tendency_stage_2d: q and vert_adv_q "
                "must be (6,n,n,nlev,3) with tracer order [q_v,q_c,q_r]; got "
                f"q={q.shape}, vert_adv_q={vert_adv_q.shape}")
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)),
                      T=(T, (n, n, nlev)), p_s=(p_s, (n, n)),
                      q=(q, (n, n, nlev, 3)),
                      vert_adv_q=(vert_adv_q, (n, n, nlev, 3)))
        return _body(u_d, v_d, T, p_s, q, vert_adv_q, dx, dy, offsets)

    return stage


# ---------------------------------------------------------------------------
# P-3D cc->D-grid VECTOR lift: tiled ``center_to_dgrid_vector`` (base case,
# use_fv3_a2b_ord4=False) — the cell-centre wind vector to D-grid CORNER lift.
# This is the section-12c ``_vert_adv_uv_d`` contribution
# (primitive_eq_cdgrid.py:957-977) that the standalone tiled momentum stage
# OMITS: the true base-case du_d_dt/dv_d_dt = momentum(307-479) +
# center_to_dgrid_vector(vert_adv_uv_cc) added UNCONDITIONALLY.  So the full
# hydrostatic capstone must add this lift to its momentum output.  Base case =
# a halo=1 ROTATING vector pad (the ndim=4 ``make_tiled_pad_vector_body``,
# pinned by test_tiled_pad_body's ndim=4-vector lane) + the 4-pt corner average
# (mirror of the SW capstone's corner-wind block, fv3_sw_tendencies lines
# 1800-1805).  Coord-FREE (pure geometry).  No new numerics.
# ---------------------------------------------------------------------------

def make_tiled_center_to_dgrid_vector_stage_2d(mesh, cdgrid, n: int, kt: int,
                                              nlev: int):
    """Tiled ``center_to_dgrid_vector`` (cc wind vector -> D-grid corners) on a
    ``(6, kt, kt)`` mesh (axes ``("face","tile_i","tile_j")``).

    Returns ``stage(u_cc, v_cc) -> (u_d, v_d)``: cc winds ``(6, n, n, nlev)``
    FACE-SHARDED, TILE-REPLICATED (``P("face",None,None,None)``); both outputs
    corner-staggered tiles ``P("face","tile_i","tile_j",None)``; gathered
    ``(6, kt*(nl+1), kt*(nl+1), nlev)`` reassemble lower-owns-shared (drop the
    duplicated shared corner face) to ``(6, n+1, n+1, nlev)`` — the same
    convention as the momentum stage's corner outputs.

    Base cut (``center_to_dgrid_vector`` default ``use_fv3_a2b_ord4=False``,
    operators_cdgrid.py:327-337): halo=1 ROTATING vector pad + the 4-pt corner
    average.  Non-duogrid orthogonal rotation only.  Coord-free.  No new
    numerics — the rotating halo is the shared ndim=4 ``make_tiled_pad_vector_body``
    and the average is the same stencil as the global op.
    """
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_vector_body

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_center_to_dgrid_vector_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_center_to_dgrid_vector_stage_2d: base cut supports the "
            "orthogonal-rotation (non-duogrid) cube only (use_fv3_a2b_ord4=False; "
            "the in-stage vector halo does not carry the duogrid h2 remap).")
    nl = n // kt

    cos_a, sin_a = grid.cos_angle, grid.sin_angle              # (6, n, n)
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded    # (6, n+2, n+2)
    offsets = grid.halo_interp_offsets                         # (6, 4, n)

    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=4, halo=1, with_offsets=True)

    fw = P("face", None, None, None)               # 4D cc winds
    fo = P("face", None, None)                     # 2D angle metrics
    cz = P("face", "tile_i", "tile_j", None)       # 4D corner outputs

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fo, fo, fo, fo, P()),  # u_cc,v_cc,ca,sa,cap,sap,offs
             out_specs=(cz, cz), check_vma=False)
    def _body(u_cc, v_cc, ca, sa, capf, sapf, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        u_cc_t = _s(u_cc, nl, nl)              # (1, nl, nl, nlev) cc
        v_cc_t = _s(v_cc, nl, nl)
        ca_t = _s(ca, nl, nl)[0]               # (nl, nl) interior angle
        sa_t = _s(sa, nl, nl)[0]
        cap_t = _s(capf, nl + 2, nl + 2)[0]    # (nl+2, nl+2) padded angle
        sap_t = _s(sapf, nl + 2, nl + 2)[0]

        # halo=1 ROTATING vector pad (ndim=4: cc tile -> (nl+2, nl+2, nlev)).
        u_pad, v_pad = vector_body(u_cc_t[0], v_cc_t[0], ca_t, sa_t,
                                   cap_t, sap_t, offs)
        u_pad = u_pad[None]                    # (1, nl+2, nl+2, nlev)
        v_pad = v_pad[None]

        # 4-pt corner average (mirror operators_cdgrid.center_to_dgrid_vector).
        u_d = 0.25 * (u_pad[:, :-1, :-1, :] + u_pad[:, 1:, :-1, :]
                      + u_pad[:, :-1, 1:, :] + u_pad[:, 1:, 1:, :])
        v_d = 0.25 * (v_pad[:, :-1, :-1, :] + v_pad[:, 1:, :-1, :]
                      + v_pad[:, :-1, 1:, :] + v_pad[:, 1:, 1:, :])
        return u_d, v_d                        # (1, nl+1, nl+1, nlev) corner

    def stage(u_cc, v_cc):
        _check_shapes(n, u_cc=(u_cc, (n, n, nlev)), v_cc=(v_cc, (n, n, nlev)))
        return _body(u_cc, v_cc, cos_a, sin_a, cap, sap, offsets)

    return stage


# ===========================================================================
# CAPSTONE: the FULL tiled fv3_hydrostatic_tendencies stage — the production
# cube 3D-PE dycore tendency (du_d_dt, dv_d_dt, dT_dt, dp_s_dt) on np=6*kt^2
# devices.  Composes ALL component stages in ONE shard_map (the 3D analogue of
# the SW capstone make_tiled_fv3_sw_tendencies_stage_2d):
#   continuity: dgrid_to_cgrid -> cgrid_divergence -> mass_flux/sigma_dot ->
#               dp_s/dt (RAW; base cut _apply_zero_mean_per_stage=False) -> omega
#   vertical:   vertical_advection of {interp_corner_to_center winds, T} (local)
#               -> vert_adv_uv_cc + vert_adv_T
#   momentum:   B=KE+Phi ; zeta ; [scalar halo {B,zeta,1/T,ln_ps,hf}] ;
#               zeta_corner ; A-L grad(B) ; PGF -> du/dv (eq 307-479)
#               + center_to_dgrid_vector(vert_adv_uv_cc) lift (vector halo;
#               the section-12c contribution at 976-977, ADDED unconditionally)
#   thermo:     [scalar halo {T}; ln_ps halo SHARED w/ momentum] horiz_adv(-u.gradT)
#               + adiabatic(kappa*T*omega/p + kappa*T*v.grad ln p_s) + vert_adv_T
#   dphis_dt = 0.
# Cross-stage flow is all per-column-LOCAL (continuity/vertical/omega read no
# horizontal halo); the ONLY collectives are the in-stage scalar halo (momentum
# {B,zeta,1/T,ln_ps,hf}+thermo {T}) and the vector halo (the lift).  NO global
# reduction: the base cut takes the fix_mass path (dp_s/dt used raw), so there is
# no per-stage zero_mean_tendency — the global-zero-mean variant (a psum over
# face+tile) is a documented follow-up.  The ln_ps scalar halo is SHARED between
# the momentum PGF (Arakawa-Lamb corner gradient) and the thermo adiabatic
# (centred cc gradient): in x64 the momentum's _pg_dt cast is a no-op, so the one
# padded ln_ps is bit-identical to both standalone stages' halos.  No new
# numerics — every op is the shared *_core / vertical.py / box-interp used by the
# already-gated component stages.  Base cut (primitive_eq_cdgrid.py defaults):
# div_damp=0, hyperdiff=0, corner_div_damp=0, A_h=0, T_diss=0, KE-heat off, no
# sponge, no physics, non-duogrid, use_fv3_a2b_zeta_corner=False.
# ===========================================================================

def _sponge_rate_from_config(coord, sponge_sigma: float,
                              sponge_tau_sec: float):
    """(nlev,) Rayleigh-sponge rate mirroring serial step 13, or ``None``
    when the sponge is off (either knob <= 0) — the byte-identical default
    for every pre-existing stage gate."""
    if sponge_tau_sec <= 0.0 or sponge_sigma <= 0.0:
        return None
    sigma_full = coord.sigma_full
    frac = jnp.clip((sponge_sigma - sigma_full) / sponge_sigma, 0.0, 1.0)
    return frac ** 2 / sponge_tau_sec


def _build_hydro_tile_tendency_fns(coord, cdgrid, nl: int, nlev: int,
                                   p_floor: float, scalar_body, vector_body,
                                   sponge_rate=None):
    """Build the SHARED per-tile hydrostatic tendency + metric-slicer closures
    used by BOTH ``make_tiled_fv3_hydrostatic_tendencies_stage_2d`` (called once)
    and ``make_tiled_fv3_hydrostatic_step_stage_2d`` (called once per SSP-RK3
    stage on the evolving tile-local state — no gather).  Single source of the
    cube hydrostatic tendency numerics (no duplication across the two factories);
    the body is the bit-identity-gated capstone tendency verbatim.  ``coord``
    sigma OR hybrid (cores dispatched on type); ``scalar_body``/``vector_body``
    the mesh-bound in-stage halo bodies; ``cdgrid`` for the box interps."""
    from legoesm import constants
    from legoesm.core.operators_cdgrid import (
        dgrid_to_center_vector, dgrid_vorticity_core, arakawa_lamb_gradient_core,
        interp_center_to_corner, interp_corner_to_center, cgrid_divergence_local,
        cgrid_interp_cc_to_faces_local)
    from legoesm.core.operators_3d import gradient_x_3d_core, gradient_y_3d_core
    from legoesm.core.precision import resolve_dtype
    from legoesm.grids.vertical import (
        compute_geopotential, compute_geopotential_hybrid, pressure_from_hybrid,
        pressure_from_sigma, dp_from_hybrid, compute_mass_flux_from_cumsum,
        compute_sigma_dot_from_cumsum,
        vertical_advection, vertical_advection_hybrid,
        compute_omega_hybrid, compute_pressure_velocity,
        HybridSigmaPressureCoordinate)
    R_d = constants.R_d
    kappa = constants.kappa
    _p_floor = float(p_floor)
    _hybrid = isinstance(coord, HybridSigmaPressureCoordinate)
    _geo = compute_geopotential_hybrid if _hybrid else compute_geopotential
    _sigma_range = None if _hybrid else 1.0 - float(coord.sigma_half[0])

    def _tile_tendency(u_d_t, v_d_t, T_t, p_s_t, phis_t,
                       cosa_c_t, dxe_t, dye_t, ar_t, gc, fco_t, cosau_t,
                       dx_t, dy_t, ca_t, sa_t, cap_t, sap_t, offs,
                       q_pack_t=None, column_physics_fn=None):
        # ---- cc winds: dgrid_to_center_vector for B/thermo (within-face, NO
        #      halo); interp_corner_to_center (batched) for vertical advection.
        u_cell, v_cell = dgrid_to_center_vector(u_d_t, v_d_t)  # (1, nl, nl, nlev)
        _uv_d = jnp.stack([u_d_t, v_d_t], axis=-1)
        _uv_cc = interp_corner_to_center(
            _uv_d.reshape(*_uv_d.shape[:-2], nlev * 2)
        ).reshape(1, nl, nl, nlev, 2)
        u_cc_va, v_cc_va = _uv_cc[..., 0], _uv_cc[..., 1]

        # ---- CONTINUITY: D->C (local) -> FLUX-FORM div(dp·v) -> dp_s/dt + vert ----
        # Mirrors the serial flux-form continuity (primitive_eq_cdgrid sec 10b):
        # dp at the C-grid faces via the shared 2-point average on the SAME
        # in-stage scalar halo the thermo/momentum pads use, then the exact
        # flux divergence.  NOT the advective div(v)·dp (differs when ∇p_s≠0).
        u_c, v_c = dgrid_to_cgrid_tile_2d(u_d_t, v_d_t, cosau_t, 0, 0, nl)
        if _hybrid:
            dp_t = dp_from_hybrid(coord, p_s_t)     # (1, nl, nl, nlev)
        else:
            dp_t = p_s_t[..., None] * coord.dsigma.astype(p_s_t.dtype)
        dp_pad = scalar_body(dp_t[0], offs)[None]   # (1, nl+2, nl+2, nlev)
        dp_u, dp_v = cgrid_interp_cc_to_faces_local(dp_pad)
        div_dp = cgrid_divergence_local(
            dp_u * u_c, dp_v * v_c, dye_t, dxe_t, ar_t)  # (1, nl, nl, nlev)
        _cumsum_dp = jnp.cumsum(div_dp, axis=-1)
        _D_total_p = _cumsum_dp[..., -1:]           # (1, nl, nl, 1) [Pa/s]
        if _hybrid:
            dp_s_dt = -_D_total_p[..., 0] / coord.B_range
            mass_flux = compute_mass_flux_from_cumsum(
                _cumsum_dp, _D_total_p, coord)
            # base cut: _apply_zero_mean_per_stage=False (fix_mass path) -> raw.
            omega = compute_omega_hybrid(mass_flux, p_s_t, dp_s_dt, coord)
            _vadv_drive = mass_flux
            _vadv = vertical_advection_hybrid
        else:
            dp_s_dt = -_D_total_p[..., 0] / _sigma_range
            sigma_dot = compute_sigma_dot_from_cumsum(
                _cumsum_dp, _D_total_p, p_s_t, coord)
            omega = compute_pressure_velocity(sigma_dot, p_s_t, dp_s_dt, coord)
            _vadv_drive = sigma_dot
            _vadv = vertical_advection

        # ---- VERTICAL advection of {u_cc_va, v_cc_va, T} (per-column local) ----
        _uvT_lead = jnp.stack([u_cc_va, v_cc_va, T_t], axis=0)   # (3,1,nl,nl,nlev)
        if _hybrid:
            _vadv_uvT = _vadv(_uvT_lead, _vadv_drive, p_s_t, coord)
        else:
            _vadv_uvT = _vadv(_uvT_lead, _vadv_drive, coord)
        vert_adv_u_cc, vert_adv_v_cc, vert_adv_T = (
            _vadv_uvT[0], _vadv_uvT[1], _vadv_uvT[2])

        # ---- p_adiab + ln_ps (cc-local) ----
        if _hybrid:
            p_full = pressure_from_hybrid(coord, p_s_t)         # (1, nl, nl, nlev)
        else:
            p_full = pressure_from_sigma(coord.sigma_full, p_s_t)
        p_adiab = jnp.maximum(p_full, _p_floor)
        ln_ps = jnp.log(p_s_t)
        _pg_dt = jnp.result_type(
            ln_ps.dtype, resolve_dtype("atm_pressure_gradient", "compute"))
        ln_ps_3d = ln_ps.astype(_pg_dt)[..., None]              # (1, nl, nl, 1)

        # ---- MOMENTUM (eq 307-479): B=KE+Phi ; zeta ; scalar halos ----
        Phi = _geo(T_t, p_s_t, coord, phis_t)
        B = 0.5 * (u_cell ** 2 + v_cell ** 2) + Phi            # (1, nl, nl, nlev)
        zeta = dgrid_vorticity_core(
            u_d_t, v_d_t, cosa_c_t, dxe_t, dye_t, ar_t)
        inv_T = 1.0 / T_t

        # In-stage SCALAR halos.  ln_ps SHARED by momentum PGF + thermo (x64: the
        # _pg_dt cast is a no-op, so this padded ln_ps == both standalone halos).
        B_pad = scalar_body(B[0], offs)[None]
        zeta_pad = scalar_body(zeta[0], offs)[None]
        invT_pad = scalar_body(inv_T[0], offs)[None]
        lnps_pad = scalar_body(ln_ps_3d[0], offs)[None]        # (1, nl+2, nl+2, 1)
        T_pad = scalar_body(T_t[0], offs)[None]                # (1, nl+2, nl+2, nlev)

        zeta_corner = (interp_center_to_corner(zeta_pad, cdgrid, padded=zeta_pad)
                       + fco_t[..., None])
        dB_dx, dB_dy_perp = arakawa_lamb_gradient_core(B_pad, *gc)
        dln_dx_hi, dln_dy_perp_hi = arakawa_lamb_gradient_core(lnps_pad, *gc)
        T_corner = 1.0 / interp_center_to_corner(invT_pad, cdgrid, padded=invT_pad)
        T_corner_hi = T_corner.astype(_pg_dt)
        pg_corr_x = (R_d * T_corner_hi * dln_dx_hi).astype(u_d_t.dtype)
        pg_corr_y_perp = (R_d * T_corner_hi * dln_dy_perp_hi).astype(v_d_t.dtype)
        if _hybrid:
            p_full_c = pressure_from_hybrid(coord, p_s_t)
            hf = coord.B_full * p_s_t[..., None] / p_full_c
            hf_pad = scalar_body(hf[0], offs)[None]
            hf_corner = interp_center_to_corner(hf_pad, cdgrid, padded=hf_pad)
            pg_corr_x = pg_corr_x * hf_corner
            pg_corr_y_perp = pg_corr_y_perp * hf_corner
        du_d_dt = zeta_corner * v_d_t - dB_dx - pg_corr_x      # (1, nl+1, nl+1, nlev)
        dv_d_dt = -zeta_corner * u_d_t - dB_dy_perp - pg_corr_y_perp

        # ---- vert_adv_uv_d lift (center_to_dgrid_vector: VECTOR halo + corner
        #      avg) ADDED to du/dv (primitive_eq_cdgrid.py:961-977) ----
        vau_pad, vav_pad = vector_body(
            vert_adv_u_cc[0], vert_adv_v_cc[0], ca_t, sa_t, cap_t, sap_t, offs)
        vau_pad = vau_pad[None]                    # (1, nl+2, nl+2, nlev)
        vav_pad = vav_pad[None]
        vau_d = 0.25 * (vau_pad[:, :-1, :-1, :] + vau_pad[:, 1:, :-1, :]
                        + vau_pad[:, :-1, 1:, :] + vau_pad[:, 1:, 1:, :])
        vav_d = 0.25 * (vav_pad[:, :-1, :-1, :] + vav_pad[:, 1:, :-1, :]
                        + vav_pad[:, :-1, 1:, :] + vav_pad[:, 1:, 1:, :])
        du_d_dt = du_d_dt + vau_d
        dv_d_dt = dv_d_dt + vav_d

        # ---- Upper-atmosphere Rayleigh sponge (D-grid, tile-local) ----
        # Mirrors serial fv3_hydrostatic_tendencies step 13 exactly:
        # ``d(u,v)/dt -= rate(sigma) * (u,v)`` with the (nlev,)-shaped rate
        # precomputed at factory build.  ``None`` (the default, and every
        # pre-existing stage gate) is byte-identical to the prior body;
        # the production adapter passes the model config's sponge so the
        # tiled step twins the DEFAULT serial step (sponge is default-ON:
        # sponge_tau_sec=3600, sponge_sigma=0.15 — measured 6.2e-6 u drift
        # without it, job 8689100).
        if sponge_rate is not None:
            du_d_dt = du_d_dt - sponge_rate * u_d_t
            dv_d_dt = dv_d_dt - sponge_rate * v_d_t

        # ---- THERMO dT/dt: horiz_adv + adiabatic + vert_adv_T ----
        dT_dx = gradient_x_3d_core(T_pad, dx_t)
        dT_dy = gradient_y_3d_core(T_pad, dy_t)
        horiz_adv_T = -(u_cell * dT_dx + v_cell * dT_dy)
        dln_ps_dx = gradient_x_3d_core(lnps_pad, dx_t)[..., 0]
        dln_ps_dy = gradient_y_3d_core(lnps_pad, dy_t)[..., 0]
        adiabatic = kappa * T_t * omega / p_adiab
        v_dot_grad_lnps = (u_cell * dln_ps_dx[..., None]
                           + v_cell * dln_ps_dy[..., None])
        if _hybrid:
            v_dot_grad_lnps = v_dot_grad_lnps * (
                coord.B_full * p_s_t[..., None] / p_adiab)
        adiabatic = adiabatic + kappa * T_t * v_dot_grad_lnps
        dT_dt = horiz_adv_T + vert_adv_T + adiabatic

        if q_pack_t is None:
            return du_d_dt, dv_d_dt, dT_dt, dp_s_dt

        # ---- MOIST tracer tendency + injected column physics (cube-MOIST np>6) ----
        # Tracers advect by the IDENTICAL horizontal (shared scalar advect, same
        # u_cell as T) + vertical (the SAME _vadv_drive = sigma_dot/mass_flux as
        # vert_adv_T) operators the serial fv3_hydrostatic_tendencies applies
        # (u_cell=dgrid_to_center_vector at :315; _tracer_vert_fn=vertical_advection
        # (·, sigma_dot) at :811).  Column physics (e.g. Kessler) is injected so
        # core stays physics-agnostic; its dT rides into dT_dt and its tracer rates
        # add to the advected pack (serial add primitive_eq_cdgrid.py:1188/1195).
        dq_horiz = _tiled_scalar_horiz_advect(
            q_pack_t, u_cell, v_cell, dx_t, dy_t, scalar_body, offs)
        q_lead = jnp.moveaxis(q_pack_t, -1, 0)            # (nt, 1, nl, nl, nlev)
        if _hybrid:
            vadv_q_lead = _vadv(q_lead, _vadv_drive, p_s_t, coord)
        else:
            vadv_q_lead = _vadv(q_lead, _vadv_drive, coord)
        vert_adv_q = jnp.moveaxis(vadv_q_lead, 0, -1)     # (1, nl, nl, nlev, nt)
        dT_k, dq_v, dq_c, dq_r = column_physics_fn(
            T_t, p_s_t, q_pack_t[..., 0], q_pack_t[..., 1], q_pack_t[..., 2])
        phys_pack = jnp.stack([dq_v, dq_c, dq_r], axis=-1)  # (1, nl, nl, nlev, 3)
        dq_pack = dq_horiz + vert_adv_q + phys_pack
        return du_d_dt, dv_d_dt, dT_dt + dT_k, dp_s_dt, dq_pack

    def _slice_metrics(_s, cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco,
                       cosau, dx_, dy_, ca, sa, capf, sapf):
        """Per-tile metric slices (constant across RK3 stages) — sliced from the
        FACE-SHARDED metric args (NEVER closed over the full (6,n,n) array; that
        broadcasts the output to face extent 6 — the codex U4a HIGH)."""
        return dict(
            cosa_c_t=_s(cosa_c, nl + 1, nl + 1),
            dxe_t=_s(dxe, nl, nl + 1), dye_t=_s(dye, nl + 1, nl),
            ar_t=_s(ar, nl, nl),
            gc=(_s(c00, nl + 1, nl + 1), _s(c01, nl + 1, nl + 1),
                _s(c10, nl + 1, nl + 1), _s(c11, nl + 1, nl + 1)),
            fco_t=_s(fco, nl + 1, nl + 1), cosau_t=_s(cosau, nl + 1, nl),
            dx_t=_s(dx_, nl, nl), dy_t=_s(dy_, nl, nl),
            ca_t=_s(ca, nl, nl)[0], sa_t=_s(sa, nl, nl)[0],
            cap_t=_s(capf, nl + 2, nl + 2)[0], sap_t=_s(sapf, nl + 2, nl + 2)[0])

    return _tile_tendency, _slice_metrics


def make_tiled_fv3_hydrostatic_tendencies_stage_2d(mesh, cdgrid, coord, n: int,
                                                  kt: int, nlev: int, *,
                                                  p_floor: float,
                                                  sponge_sigma: float = 0.0,
                                                  sponge_tau_sec: float = 0.0):
    """Full tiled ``fv3_hydrostatic_tendencies`` on a ``(6, kt, kt)`` mesh.

    Returns ``stage(u_d, v_d, T, p_s, phis) -> (du_d_dt, dv_d_dt, dT_dt,
    dp_s_dt)``.  State inputs FACE-SHARDED, TILE-REPLICATED: ``u_d``/``v_d``
    D-grid corner winds ``(6, n+1, n+1, nlev)``; ``T`` cc ``(6, n, n, nlev)``;
    ``p_s``/``phis`` cc ``(6, n, n)``.  Outputs: ``du_d_dt``/``dv_d_dt``
    corner-staggered tiles ``P("face","tile_i","tile_j",None)`` (gathered
    lower-owns-shared to ``(6, n+1, n+1, nlev)``); ``dT_dt`` cc
    ``P("face","tile_i","tile_j",None)`` -> ``(6, n, n, nlev)``; ``dp_s_dt`` cc
    ``P("face","tile_i","tile_j")`` -> ``(6, n, n)`` (both exact cc partition).
    ``dphis_dt`` is identically zero and not returned.

    ``coord`` (sigma OR hybrid) closed over; the mass-flux / vertical-advection /
    omega / geopotential cores are dispatched on its type.  ``p_floor`` =
    ``CDGridPrimitiveEquationConfig.p_floor`` [Pa] (the adiabatic 1/p floor).
    Base cut + halo/reduction structure: see the module comment above.
    """
    # The tendency numerics + their imports live in _build_hydro_tile_tendency_fns
    # (shared with the step stage); this factory only builds the halo bodies +
    # slices the face-sharded state/metrics and calls the shared tendency.
    from legoesm.parallel.cubesphere_exchange import (
        make_tiled_pad_body, make_tiled_pad_vector_body)

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_tendencies_stage_2d: n={n} != "
            f"cdgrid.n={cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_hydrostatic_tendencies_stage_2d: base cut supports "
            "the orthogonal-rotation (non-duogrid) cube only.")
    if getattr(coord, "n_levels", nlev) != nlev:
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_tendencies_stage_2d: coord.n_levels="
            f"{coord.n_levels} != nlev={nlev}")
    if not (float(p_floor) > 0.0):
        raise ValueError(
            f"make_tiled_fv3_hydrostatic_tendencies_stage_2d: p_floor must be a "
            f"positive pressure [Pa]; got {p_floor}")
    nl = n // kt

    # Static metrics (face-sharded, tile-replicated -> sliced per tile).
    cosa_corner = cdgrid.cosa_corner                          # (6, n+1, n+1)
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x  # (6,n,n+1)/(6,n+1,n)
    area = grid.area                                          # (6, n, n)
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01             # (6, n+1, n+1)
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    f_corner = cdgrid.f_corner                                # (6, n+1, n+1)
    cosa_u = cdgrid.cosa_u                                    # (6, n+1, n)
    dx, dy = grid.dx, grid.dy                                 # (6, n, n) cc
    cos_a, sin_a = grid.cos_angle, grid.sin_angle             # (6, n, n)
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded   # (6, n+2, n+2)
    offsets = grid.halo_interp_offsets                        # (6, 4, n)

    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=4, halo=1, with_offsets=True)

    fo = P("face", None, None)                     # 2D-face metric / cc 2D
    fw = P("face", None, None, None)               # 4D state
    cz = P("face", "tile_i", "tile_j", None)       # 4D corner / cc 4D out
    co = P("face", "tile_i", "tile_j")             # 2D cc out (dp_s_dt)

    # Single source of the tendency numerics (shared with the tiled STEP).
    _tile_tendency, _slice_metrics = _build_hydro_tile_tendency_fns(
        coord, cdgrid, nl, nlev, p_floor, scalar_body, vector_body,
        sponge_rate=_sponge_rate_from_config(coord, sponge_sigma,
                                             sponge_tau_sec))

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fo, fo)          # u_d, v_d, T, p_s, phis
                       + (fo,) * 4                  # cosa_corner, dxe, dye, area
                       + (fo,) * 4                  # gc00..gc11
                       + (fo, fo)                   # f_corner, cosa_u
                       + (fo, fo)                   # dx, dy (thermo)
                       + (fo,) * 4                  # cos_a, sin_a, cap, sap (lift)
                       + (P(),),                    # offsets
             out_specs=(cz, cz, cz, co), check_vma=False)
    def _body(u_d, v_d, T, p_s, phis,
              cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco, cosau,
              dx_, dy_, ca, sa, capf, sapf, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        m = _slice_metrics(_s, cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco,
                           cosau, dx_, dy_, ca, sa, capf, sapf)
        return _tile_tendency(
            _s(u_d, nl + 1, nl + 1), _s(v_d, nl + 1, nl + 1),
            _s(T, nl, nl), _s(p_s, nl, nl), _s(phis, nl, nl),
            m["cosa_c_t"], m["dxe_t"], m["dye_t"], m["ar_t"], m["gc"],
            m["fco_t"], m["cosau_t"], m["dx_t"], m["dy_t"],
            m["ca_t"], m["sa_t"], m["cap_t"], m["sap_t"], offs)

    def stage(u_d, v_d, T, p_s, phis):
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)), T=(T, (n, n, nlev)),
                      p_s=(p_s, (n, n)), phis=(phis, (n, n)))
        return _body(u_d, v_d, T, p_s, phis,
                     cosa_corner, dx_edge_y, dy_edge_x, area,
                     gc00, gc01, gc10, gc11, f_corner, cosa_u,
                     dx, dy, cos_a, sin_a, cap, sap, offsets)

    return stage


# ===========================================================================
# TILED STEP: the full SSP-RK3 time step wrapping the hydrostatic tendency
# capstone, in ONE shard_map with NO gather between RK3 stages.  The dominant
# compute (the tendency) runs sub-face-tiled on np=6*kt^2 devices; the 3 SSP-RK3
# stages + the 2 pointwise SSP combines run on the TILE-LOCAL state.  The
# corner-staggered D-grid winds carry a duplicated shared tile face that stays
# self-consistent across stages because adjacent tiles compute BIT-IDENTICAL
# shared-face du_d/dv_d (proven by the capstone gate's du/dv corner-overlap at
# 1e-10 — both tiles match the global there, hence each other).  Base cut: no
# post-step (sponge / damp_v / fix_ps_mass = increment-4); dphis=0 (phis const).
# Bit-identical (to RK3-accumulated reorder) to ssp_rk3_step(state, the base-cut
# fv3_hydrostatic_tendencies, dt) — NOT the full _step_fv3 (which also does the
# post-step _sync_dgrid_boundary).  Future-HW (np>6 anti-scales on Ginsburg);
# gated by bit-identity, not wall-clock.
# ===========================================================================

def _validate_tiled_step_factory_args(where, mesh, cdgrid, coord, n, kt,
                                      nlev, p_floor, dt):
    """Shared factory-entry validation for the tiled STEP builders (dry /
    moist / blocked) — one copy of the fail-loud guards (n%kt, mesh shape,
    cdgrid.n, duogrid refusal, coord levels, p_floor, dt)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(f"{where}: n={n} != cdgrid.n={cdgrid.n}")
    if cdgrid.base.duogrid is not None:
        raise ValueError(
            f"{where}: base cut supports the orthogonal-rotation "
            "(non-duogrid) cube only.")
    if getattr(coord, "n_levels", nlev) != nlev:
        raise ValueError(
            f"{where}: coord.n_levels={coord.n_levels} != nlev={nlev}")
    if not (float(p_floor) > 0.0):
        raise ValueError(
            f"{where}: p_floor must be a positive pressure [Pa]; got "
            f"{p_floor}")
    if not (float(dt) > 0.0):
        raise ValueError(
            f"{where}: dt must be a positive time step [s]; got {dt}")


def _ssp_rk3_tile_step(s0, F, dt):
    """SSP-RK3 (Shu-Osher; timestepping/ssp_rk3.py) on a tuple of tile-local
    state arrays.  ``F(s_tuple)`` returns the matching tuple of tendencies;
    pointwise tile-local combines (no halo — the halos live inside ``F``).
    Shared by the dry (4-tuple) and moist (5-tuple) tiled steps so the RK3
    weights/order live in ONE place (no per-arity copy).  Bit-identical to the
    prior explicit dry combine."""
    def _euler(s):
        f = F(s)
        return tuple(x + dt * fi for x, fi in zip(s, f))

    def _blend(a, s, b, e):
        return tuple(a * x + b * y for x, y in zip(s, e))

    s1 = _euler(s0)                                    # s0 + dt F(s0)
    s2 = _blend(0.75, s0, 0.25, _euler(s1))            # 3/4 s0 + 1/4 (s1 + dt F)
    s3 = _blend(1.0 / 3.0, s0, 2.0 / 3.0, _euler(s2))  # 1/3 s0 + 2/3 (s2 + dt F)
    return s3


def make_tiled_fv3_hydrostatic_step_stage_2d(mesh, cdgrid, coord, n: int,
                                            kt: int, nlev: int, *,
                                            p_floor: float, dt: float,
                                            sponge_sigma: float = 0.0,
                                            sponge_tau_sec: float = 0.0):
    """Full tiled SSP-RK3 STEP on a ``(6, kt, kt)`` mesh.

    Returns ``step(u_d, v_d, T, p_s, phis) -> (u_d, v_d, T, p_s)`` advanced one
    ``dt``.  State inputs FACE-SHARDED, TILE-REPLICATED (same layout as the
    tendency stage); outputs the STEPPED state in the same per-tile staggering
    (u_d/v_d corner ``P("face","tile_i","tile_j",None)``, T cc 4D, p_s cc 2D
    ``P("face","tile_i","tile_j")``).  ``dt`` [s] closed over (static — recompiles
    per dt, fine for a fixed-step run).  Reuses the SHARED ``_tile_tendency``
    builder (no duplicated numerics).  See the module comment above for the
    base cut + the no-gather self-consistency argument.
    """
    from legoesm.parallel.cubesphere_exchange import (
        make_tiled_pad_body, make_tiled_pad_vector_body)

    _validate_tiled_step_factory_args(
        "make_tiled_fv3_hydrostatic_step_stage_2d", mesh, cdgrid, coord, n,
        kt, nlev, p_floor, dt)
    grid = cdgrid.base
    nl = n // kt
    _dt = float(dt)

    # Static metrics (face-sharded, tile-replicated -> sliced per tile).
    cosa_corner = cdgrid.cosa_corner
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    area = grid.area
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    f_corner = cdgrid.f_corner
    cosa_u = cdgrid.cosa_u
    dx, dy = grid.dx, grid.dy
    cos_a, sin_a = grid.cos_angle, grid.sin_angle
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded
    offsets = grid.halo_interp_offsets

    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=4, halo=1, with_offsets=True)

    # Shared tendency numerics (same builder as the tendency stage — no dup).
    _tile_tendency, _slice_metrics = _build_hydro_tile_tendency_fns(
        coord, cdgrid, nl, nlev, p_floor, scalar_body, vector_body,
        sponge_rate=_sponge_rate_from_config(coord, sponge_sigma,
                                             sponge_tau_sec))

    fo = P("face", None, None)
    fw = P("face", None, None, None)
    cz = P("face", "tile_i", "tile_j", None)       # corner / cc 4D state out
    co = P("face", "tile_i", "tile_j")             # p_s 2D cc state out

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fo, fo)          # u_d, v_d, T, p_s, phis
                       + (fo,) * 4                  # cosa_corner, dxe, dye, area
                       + (fo,) * 4                  # gc00..gc11
                       + (fo, fo)                   # f_corner, cosa_u
                       + (fo, fo)                   # dx, dy
                       + (fo,) * 4                  # cos_a, sin_a, cap, sap
                       + (P(),),                    # offsets
             out_specs=(cz, cz, cz, co), check_vma=False)
    def _step_body(u_d, v_d, T, p_s, phis,
                   cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco, cosau,
                   dx_, dy_, ca, sa, capf, sapf, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        m = _slice_metrics(_s, cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco,
                           cosau, dx_, dy_, ca, sa, capf, sapf)
        mt = (m["cosa_c_t"], m["dxe_t"], m["dye_t"], m["ar_t"], m["gc"],
              m["fco_t"], m["cosau_t"], m["dx_t"], m["dy_t"],
              m["ca_t"], m["sa_t"], m["cap_t"], m["sap_t"], offs)

        phis_t = _s(phis, nl, nl)                  # constant (dphis=0)
        ud0 = _s(u_d, nl + 1, nl + 1)
        vd0 = _s(v_d, nl + 1, nl + 1)
        T0 = _s(T, nl, nl)
        ps0 = _s(p_s, nl, nl)

        def _F(s):
            return _tile_tendency(s[0], s[1], s[2], s[3], phis_t, *mt)

        # SSP-RK3 (shared _ssp_rk3_tile_step) — pointwise tile-local combines.
        return _ssp_rk3_tile_step((ud0, vd0, T0, ps0), _F, _dt)

    def step(u_d, v_d, T, p_s, phis):
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)), T=(T, (n, n, nlev)),
                      p_s=(p_s, (n, n)), phis=(phis, (n, n)))
        return _step_body(u_d, v_d, T, p_s, phis,
                          cosa_corner, dx_edge_y, dy_edge_x, area,
                          gc00, gc01, gc10, gc11, f_corner, cosa_u,
                          dx, dy, cos_a, sin_a, cap, sap, offsets)

    return step


def make_tiled_fv3_hydrostatic_moist_step_stage_2d(mesh, cdgrid, coord, n: int,
                                                  kt: int, nlev: int, *,
                                                  p_floor: float, dt: float,
                                                  column_physics_fn,
                                            sponge_sigma: float = 0.0,
                                            sponge_tau_sec: float = 0.0):
    """Full tiled MOIST SSP-RK3 STEP on a ``(6, kt, kt)`` mesh — increment 4, the
    cube-MOIST np>6 capstone.

    Returns ``step(u_d, v_d, T, p_s, phis, q_pack) -> (u_d, v_d, T, p_s, q_pack)``
    advanced one ``dt`` with ``q_pack`` ``(6, n, n, nlev, 3)`` (tracer order
    ``[q_v, q_c, q_r]``).  Threads the 5-tuple through the SAME SSP-RK3
    (``_ssp_rk3_tile_step``) and the SAME tracer-aware tendency
    (``_build_hydro_tile_tendency_fns``, ``q_pack_t``/``column_physics_fn`` given)
    the dry step uses — dynamics tracer advection (horizontal + the SAME
    ``sigma_dot``/``mass_flux`` vertical driver as T) + the INJECTED per-tile
    ``column_tracer_physics_fn(T_t, p_s_t, q_v_t, q_c_t, q_r_t) -> (dT, dq_v,
    dq_c, dq_r)`` (warm-rain; its dT rides into dT_dt).  A post-step
    ``jnp.maximum(q, 0)`` floor matches the serial tracer floor
    (primitive_eq_cdgrid.py:1758-1765).  Output staggering as the dry step plus
    ``q_pack`` cc 5-D ``P("face","tile_i","tile_j",None,None)``; gathered == the
    serial ``model.step_with_physics`` base cut.  sigma OR hybrid per ``coord``
    (the injected physics may restrict — Kessler is sigma-only).
    """
    from legoesm.parallel.cubesphere_exchange import (
        make_tiled_pad_body, make_tiled_pad_vector_body)

    if column_physics_fn is None:
        raise ValueError(
            "make_tiled_fv3_hydrostatic_moist_step_stage_2d: column_physics_fn "
            "is required (inject the column-local physics, e.g. built from "
            "kessler_column_tendencies, returning (dT, dq_v, dq_c, dq_r)).")
    _validate_tiled_step_factory_args(
        "make_tiled_fv3_hydrostatic_moist_step_stage_2d", mesh, cdgrid, coord,
        n, kt, nlev, p_floor, dt)
    grid = cdgrid.base
    nl = n // kt
    _dt = float(dt)
    _phys = column_physics_fn

    # Static metrics (same as the dry step — face-sharded, tile-replicated).
    cosa_corner = cdgrid.cosa_corner
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    area = grid.area
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    f_corner = cdgrid.f_corner
    cosa_u = cdgrid.cosa_u
    dx, dy = grid.dx, grid.dy
    cos_a, sin_a = grid.cos_angle, grid.sin_angle
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded
    offsets = grid.halo_interp_offsets

    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=4, halo=1, with_offsets=True)
    _tile_tendency, _slice_metrics = _build_hydro_tile_tendency_fns(
        coord, cdgrid, nl, nlev, p_floor, scalar_body, vector_body,
        sponge_rate=_sponge_rate_from_config(coord, sponge_sigma,
                                             sponge_tau_sec))

    fo = P("face", None, None)
    fw = P("face", None, None, None)
    fw5 = P("face", None, None, None, None)
    cz = P("face", "tile_i", "tile_j", None)       # corner / cc 4D state out
    co = P("face", "tile_i", "tile_j")             # p_s 2D cc state out
    cz5 = P("face", "tile_i", "tile_j", None, None)  # q_pack 5D cc state out

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fw, fo, fo, fw5)     # u_d, v_d, T, p_s, phis, q
                       + (fo,) * 4                  # cosa_corner, dxe, dye, area
                       + (fo,) * 4                  # gc00..gc11
                       + (fo, fo)                   # f_corner, cosa_u
                       + (fo, fo)                   # dx, dy
                       + (fo,) * 4                  # cos_a, sin_a, cap, sap
                       + (P(),),                    # offsets
             out_specs=(cz, cz, cz, co, cz5), check_vma=False)
    def _step_body(u_d, v_d, T, p_s, phis, q,
                   cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco, cosau,
                   dx_, dy_, ca, sa, capf, sapf, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        m = _slice_metrics(_s, cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco,
                           cosau, dx_, dy_, ca, sa, capf, sapf)
        mt = (m["cosa_c_t"], m["dxe_t"], m["dye_t"], m["ar_t"], m["gc"],
              m["fco_t"], m["cosau_t"], m["dx_t"], m["dy_t"],
              m["ca_t"], m["sa_t"], m["cap_t"], m["sap_t"], offs)

        phis_t = _s(phis, nl, nl)
        ud0 = _s(u_d, nl + 1, nl + 1)
        vd0 = _s(v_d, nl + 1, nl + 1)
        T0 = _s(T, nl, nl)
        ps0 = _s(p_s, nl, nl)
        q0 = _s(q, nl, nl)                          # (1, nl, nl, nlev, 3)

        def _F(s):
            return _tile_tendency(s[0], s[1], s[2], s[3], phis_t, *mt,
                                  q_pack_t=s[4], column_physics_fn=_phys)

        ud3, vd3, T3, ps3, q3 = _ssp_rk3_tile_step(
            (ud0, vd0, T0, ps0, q0), _F, _dt)
        q3 = jnp.maximum(q3, 0.0)                   # post-step tracer floor
        return ud3, vd3, T3, ps3, q3

    def step(u_d, v_d, T, p_s, phis, q):
        if q.ndim != 5 or q.shape[-1] != 3:
            raise ValueError(
                "make_tiled_fv3_hydrostatic_moist_step_stage_2d: q must be "
                f"(6,n,n,nlev,3) [q_v,q_c,q_r]; got {q.shape}")
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)), T=(T, (n, n, nlev)),
                      p_s=(p_s, (n, n)), phis=(phis, (n, n)),
                      q=(q, (n, n, nlev, 3)))
        return _step_body(u_d, v_d, T, p_s, phis, q,
                          cosa_corner, dx_edge_y, dy_edge_x, area,
                          gc00, gc01, gc10, gc11, f_corner, cosa_u,
                          dx, dy, cos_a, sin_a, cap, sap, offsets)

    return step


# ===========================================================================
# BLOCKED-I/O persistent tiled step (the np>6 PRODUCTION assembly).
#
# The dry/moist step stages above consume FACE-REPLICATED state and emit
# TILE-SHARDED state — a SINGLE-SHOT contract: feeding the output back in
# requires re-replicating the corner-staggered tiles, i.e. a full-cube
# all-gather per step, which negates the tiling (the bench adapter's
# documented limitation).  For a PRODUCTION multi-step run the state must
# STAY tile-sharded across steps: this section adds
#
#   * the BLOCKED layout: corner-staggered fields stored block-concatenated
#     per tile — tile (ti,tj) owns rows [ti*(nl+1):(ti+1)*(nl+1)] holding
#     global corners [ti*nl : ti*nl+nl+1], so adjacent tiles carry a
#     DUPLICATED shared face (global blocked shape (6, kt*(nl+1),
#     kt*(nl+1)[, nlev])); cc fields partition exactly (plain (6, n, n[,...])
#     resharded to P("face","tile_i","tile_j")).
#   * expand_corners_to_blocks — global corners -> blocked (the inverse of
#     the adapter's dedup_tiled_corners; both are pure slice/concat).
#   * make_tiled_fv3_hydrostatic_step_blocked_2d — the SAME RK3 body as the
#     step stages (shared _build_hydro_tile_tendency_fns/_ssp_rk3_tile_step;
#     zero duplicated numerics) with in_specs == out_specs, so
#     ``s = step(s)`` closes the loop with NO per-step gather/replicate.
#     Optionally applies the serial post-step dry-mass fixer IN-STAGE
#     (fix_mass=True: the telescoping fix_ps_mass(p_s_new, p_s_pre_step)
#     via the shared _tile_fix_ps_mass_delta psum — matching the serial
#     _step_fv3's use_conservation_fixer+fix_mass branch, whose
#     anchor_mass_to_initial path threads the per-call PRE-STEP mass under
#     an outer jit and therefore telescopes identically).
#
# Cross-step self-consistency of the duplicated shared faces follows from
# the within-step argument (module comment above): adjacent tiles compute
# bit-identical shared-face tendencies, the RK3 combines are pointwise, and
# the in-stage fixer adds the SAME global scalar to every tile — so state
# duplicates stay bit-identical for any number of steps.
# ===========================================================================


def expand_corners_to_blocks(t, kt: int, nl: int):
    """Global corner-staggered field -> BLOCKED per-tile layout.

    ``(F, n+1, n+1[, ...]) -> (F, kt*(nl+1), kt*(nl+1)[, ...])`` with
    ``n = kt*nl``: tile row-block ``ti`` = global corner rows
    ``[ti*nl : ti*nl + nl+1]`` (adjacent blocks DUPLICATE the shared
    staggered face).  Exact inverse of the adapter's
    ``dedup_tiled_corners`` (tiled_step_adapter.py) for any consistent
    blocked field.  Pure slice+concat — call once at loop entry (layout
    conversion), never per step."""
    if t.shape[1] != kt * nl + 1 or t.shape[2] != kt * nl + 1:
        raise ValueError(
            f"expand_corners_to_blocks: expected global corners "
            f"(F, {kt * nl + 1}, {kt * nl + 1}, ...) for kt={kt}, nl={nl}; "
            f"got {tuple(t.shape)}")
    rows = [t[:, i * nl: i * nl + nl + 1] for i in range(kt)]
    t = jnp.concatenate(rows, axis=1)
    cols = [t[:, :, j * nl: j * nl + nl + 1] for j in range(kt)]
    return jnp.concatenate(cols, axis=2)


def make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, coord, n: int, kt: int, nlev: int, *,
        p_floor: float, dt: float,
        sponge_sigma: float = 0.0, sponge_tau_sec: float = 0.0,
        column_physics_fn=None, fix_mass: bool = False):
    """Blocked-I/O tiled SSP-RK3 STEP on a ``(6, kt, kt)`` mesh — the
    PRODUCTION (closed-loop) variant of the step stages above.

    Contract (INPUT layout == OUTPUT layout — see the section comment):

    * dry (``column_physics_fn=None``):
      ``step(u_d, v_d, T, p_s, phis) -> (u_d, v_d, T, p_s)``
    * moist: ``step(u_d, v_d, T, p_s, phis, q_pack) -> (..., q_pack)`` with
      ``q_pack`` cc ``(6, n, n, nlev, 3)`` ([q_v, q_c, q_r]; per-tile
      column physics injected exactly like the moist step stage).

    with the corner-staggered ``u_d``/``v_d`` in the BLOCKED layout
    ``(6, kt*(nl+1), kt*(nl+1), nlev)`` sharded
    ``P("face","tile_i","tile_j",None)`` and the cc fields (``T``, ``p_s``,
    ``phis``, ``q_pack``) plain global shapes sharded over the same axes.
    ``fix_mass=True`` appends the serial post-step telescoping dry-mass
    fixer in-stage (one extra psum/step).  ``dt`` [s] static (closed over).
    """
    from legoesm.parallel.cubesphere_exchange import (
        make_tiled_pad_body, make_tiled_pad_vector_body)
    from legoesm.core.conservation import conservation_accumulator

    _validate_tiled_step_factory_args(
        "make_tiled_fv3_hydrostatic_step_blocked_2d", mesh, cdgrid, coord,
        n, kt, nlev, p_floor, dt)
    grid = cdgrid.base
    nl = n // kt
    _dt = float(dt)
    _phys = column_physics_fn
    _moist = _phys is not None
    _fix_mass = bool(fix_mass)
    acc = conservation_accumulator()

    # Static metrics (face-sharded, tile-replicated -> sliced per tile;
    # identical to the step stages).
    cosa_corner = cdgrid.cosa_corner
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    area = grid.area
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    f_corner = cdgrid.f_corner
    cosa_u = cdgrid.cosa_u
    dx, dy = grid.dx, grid.dy
    cos_a, sin_a = grid.cos_angle, grid.sin_angle
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded
    offsets = grid.halo_interp_offsets
    total_area = jnp.sum(area.astype(acc))   # global; closed over (fixer)

    scalar_body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=4, halo=1, with_offsets=True)
    _tile_tendency, _slice_metrics = _build_hydro_tile_tendency_fns(
        coord, cdgrid, nl, nlev, p_floor, scalar_body, vector_body,
        sponge_rate=_sponge_rate_from_config(coord, sponge_sigma,
                                             sponge_tau_sec))

    fo = P("face", None, None)
    cz = P("face", "tile_i", "tile_j", None)       # corner blocked / cc 4D
    co = P("face", "tile_i", "tile_j")             # cc 2D
    cz5 = P("face", "tile_i", "tile_j", None, None)  # q_pack cc 5D

    state_specs = (cz, cz, cz, co, co) + ((cz5,) if _moist else ())
    out_state_specs = (cz, cz, cz, co) + ((cz5,) if _moist else ())

    @partial(shard_map, mesh=mesh,
             in_specs=state_specs                   # u_d, v_d, T, p_s, phis[, q]
                       + (fo,) * 4                  # cosa_corner, dxe, dye, area
                       + (fo,) * 4                  # gc00..gc11
                       + (fo, fo)                   # f_corner, cosa_u
                       + (fo, fo)                   # dx, dy
                       + (fo,) * 4                  # cos_a, sin_a, cap, sap
                       + (P(),),                    # offsets
             out_specs=out_state_specs, check_vma=False)
    def _step_body(*args):
        if _moist:
            u_d, v_d, T, p_s, phis, q = args[:6]
            rest = args[6:]
        else:
            u_d, v_d, T, p_s, phis = args[:5]
            rest = args[5:]
        (cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco, cosau,
         dx_, dy_, ca, sa, capf, sapf, offs) = rest

        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        # METRICS are face-replicated -> sliced per tile (as in the step
        # stages); the STATE arrives already tile-local (blocked layout) —
        # no slicing, which is the whole point of the closed-loop contract.
        m = _slice_metrics(_s, cosa_c, dxe, dye, ar, c00, c01, c10, c11, fco,
                           cosau, dx_, dy_, ca, sa, capf, sapf)
        mt = (m["cosa_c_t"], m["dxe_t"], m["dye_t"], m["ar_t"], m["gc"],
              m["fco_t"], m["cosau_t"], m["dx_t"], m["dy_t"],
              m["ca_t"], m["sa_t"], m["cap_t"], m["sap_t"], offs)

        if _moist:
            def _F(s):
                return _tile_tendency(s[0], s[1], s[2], s[3], phis, *mt,
                                      q_pack_t=s[4], column_physics_fn=_phys)

            ud3, vd3, T3, ps3, q3 = _ssp_rk3_tile_step(
                (u_d, v_d, T, p_s, q), _F, _dt)
            q3 = jnp.maximum(q3, 0.0)          # post-step tracer floor
            out_rest = (q3,)
        else:
            def _F(s):
                return _tile_tendency(s[0], s[1], s[2], s[3], phis, *mt)

            ud3, vd3, T3, ps3 = _ssp_rk3_tile_step(
                (u_d, v_d, T, p_s), _F, _dt)
            out_rest = ()

        if _fix_mass:
            # Serial post-step fixer (use_conservation_fixer+fix_mass):
            # telescoping fix_ps_mass(new, pre-step) — shared delta-first
            # psum math (see _tile_fix_ps_mass_delta).
            ps3 = _tile_fix_ps_mass_delta(ps3, p_s, m["ar_t"], total_area,
                                          acc)
        return (ud3, vd3, T3, ps3) + out_rest

    blk = kt * (nl + 1)

    def step(u_d, v_d, T, p_s, phis, q=None):
        if _moist != (q is not None):
            raise ValueError(
                "make_tiled_fv3_hydrostatic_step_blocked_2d: q_pack must be "
                "passed iff column_physics_fn was given (moist contract); "
                f"got q={'set' if q is not None else 'None'} with "
                f"column_physics_fn={'set' if _moist else 'None'}.")
        _check_shapes(n, u_d=(u_d, (blk, blk, nlev)),
                      v_d=(v_d, (blk, blk, nlev)), T=(T, (n, n, nlev)),
                      p_s=(p_s, (n, n)), phis=(phis, (n, n)),
                      **({"q": (q, (n, n, nlev, 3))} if _moist else {}))
        args = (u_d, v_d, T, p_s, phis) + ((q,) if _moist else ())
        return _step_body(*args,
                          cosa_corner, dx_edge_y, dy_edge_x, area,
                          gc00, gc01, gc10, gc11, f_corner, cosa_u,
                          dx, dy, cos_a, sin_a, cap, sap, offsets)

    return step


def warmup_tiled_cube_comms(mesh, kt: int, *, force: bool = False) -> bool:
    """Deterministically prime EVERY NCCL communicator the closed-loop tiled
    cube step uses, BEFORE the first real step, so multi-process communicator
    init cannot deadlock (issue #921).

    The closed-loop blocked step
    (:func:`make_tiled_fv3_hydrostatic_step_blocked_2d` with ``fix_mass=True``,
    and the operator-split twin) issues, inside ONE compiled executable, BOTH
    the halo collective-permutes (``jax.lax.ppermute`` over the
    ``(face, tile_i, tile_j)`` mesh axes — the tiled halo cliques) AND the
    mass-fixer GLOBAL reduction (``jax.lax.psum`` over the SAME axes —
    :func:`_tile_fix_ps_mass_delta` / :func:`_tile_fix_ps_mass_target`).  NCCL
    communicator init is itself a collective over the clique; under the XLA/GPU
    defaults (latency-hiding scheduler + async collectives +
    ``nccl_comm_splitting``) the two clique KINDS can be scheduled for init in a
    DIFFERENT relative order on different ranks — rank A blocks initialising the
    reduction clique while rank B blocks initialising a permute clique — a
    cyclic wait that never converges (observed at np=24 on Derecho: late
    comm-init NCCL INFO, ZERO ``Init COMPLETE`` for ~85 min, walltime kill).
    The shipped single-shot lane (halo cliques only) and any single-process
    CPU-virtual run (no cross-process rendezvous) are immune, which is why the
    parity gate cannot catch this class of bug.

    This runs, in a FIXED rank-INDEPENDENT order with a cross-process barrier
    between, ONE tiny standalone zero-array executable per clique KIND:

    1. the halo ``ppermute`` cliques — every table the step's pad body issues
       (the edge-strip rounds + the guard-sliver rounds + the diagonal-corner
       rounds), which the shipped single-shot lane PROVES co-init cleanly on
       their own (89 permutes / 4 comms all reaching ``Init COMPLETE``); then
    2. the reduction ``psum`` clique.

    Because each executable contains a single clique kind and is driven to
    completion (``block_until_ready`` + ``sync_global_devices``) before the
    next, every NCCL communicator is established in isolation and in the same
    order on every rank.  XLA caches communicators process-globally by clique
    key (the participating device set / source-target pairs), so the subsequent
    mixed-clique step reuses the already-initialised comms and has no init left
    to race.  Same perms + same axes here as the step ⇒ the SAME cliques.

    No-op unless the run is genuinely multi-process (``jax.process_count() > 1``
    — the route-B one-process-per-GPU lane): single-device / single-process
    (CPU-virtual smoke, one GPU, or single-process multi-GPU) has no
    cross-process comm-init rendezvous and is left byte-for-byte unaffected.
    ``force=True`` runs it anyway (a test hook: exercises the warmup on a
    single-process CPU-virtual mesh to prove it touches the expected cliques
    without error — it cannot reproduce the cross-PROCESS NCCL race).

    Returns ``True`` if the warmup executed, ``False`` if it was skipped.
    """
    if jax.process_count() <= 1 and not force:
        return False

    from jax.sharding import NamedSharding
    from legoesm.parallel.cubesphere_exchange import (
        _get_tiled_tables, _tiled_diag_perms, _tiled_guard_perms,
    )

    AXES = ("face", "tile_i", "tile_j")
    if (int(kt) < 2 or tuple(mesh.devices.shape) != (6, kt, kt)
            or tuple(getattr(mesh, "axis_names", ())) != AXES):
        raise ValueError(
            f"warmup_tiled_cube_comms: mesh must be the tiled cube mesh — "
            f"axes {tuple(getattr(mesh, 'axis_names', ()))} == {AXES} and "
            f"devices.shape {tuple(mesh.devices.shape)} == (6, kt, kt)="
            f"(6, {kt}, {kt}) with kt>=2")

    # Every ppermute the blocked step's halo pad body issues, in the SAME table
    # order on every rank (the source-target pairs are host-side deterministic
    # — the cubesphere_exchange table builders): the edge-strip rounds, the
    # guard-sliver rounds, and the diagonal-corner rounds.  Same perms + same
    # AXES ⇒ the SAME NCCL cliques the step will reuse.
    tables = _get_tiled_tables(kt)
    perms = list(tables.perms)
    perms += list(_tiled_guard_perms(kt))
    perms += list(_tiled_diag_perms(kt))

    sh = NamedSharding(mesh, P(*AXES))
    dummy = jax.device_put(jnp.zeros((6, kt, kt), dtype=jnp.float32), sh)

    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
             check_vma=False)
    def _halo_warm(x):
        v = x.reshape((1,))
        acc = v
        for perm in perms:
            acc = acc + jax.lax.ppermute(v, AXES, perm)
        return acc.reshape((1, 1, 1))

    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
             check_vma=False)
    def _reduce_warm(x):
        v = x.reshape((1,))
        return (v + jax.lax.psum(v, axis_name=AXES)).reshape((1, 1, 1))

    def _barrier(tag):
        # A true cross-process rendezvous so no rank races ahead to the next
        # clique kind while a peer is still initialising the current one
        # (block_until_ready only proves the LOCAL device's stream drained).
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(tag)

    # (1) halo cliques ALONE (the single-shot lane proves they co-init), driven
    #     to completion, then a global barrier; (2) the reduction clique ALONE.
    jax.block_until_ready(_halo_warm(dummy))
    _barrier("tiled_cube_warmup_halo")
    jax.block_until_ready(_reduce_warm(dummy))
    _barrier("tiled_cube_warmup_reduce")
    return True


# ---------------------------------------------------------------------------
# Tiled GLOBAL reduction primitive: area-weighted zero_mean_tendency via psum.
# The FIRST global reduction in the cube tiled stages (all prior stages are
# halo-only) — the primitive the full tiled STEP needs for its mass-fixer and
# for the capstone's _apply_zero_mean_per_stage=True config path (the production
# default fix_mass=True path uses raw dp_s/dt, so the capstone's base cut omits
# it; this stage provides the alternative).  Bit-identical to the global
# core.conservation.zero_mean_tendency up to psum reduction-ORDER reordering;
# the conserved property sum(out*area)==0 holds to machine precision (the point).
# ---------------------------------------------------------------------------

def make_tiled_zero_mean_tendency_stage_2d(mesh, grid, n: int, kt: int):
    """Tiled area-weighted ``zero_mean_tendency`` on a ``(6, kt, kt)`` mesh.

    ``stage(tend) -> tend - sum(tend*area)/sum(area)``: a 2D cc tendency
    ``(6, n, n)`` FACE-REPLICATED in; tile-sharded ``P("face","tile_i","tile_j")``
    out (exact cc partition).  Computes the area-weighted global mean via a single
    ``jax.lax.psum`` over the (face, tile_i, tile_j) mesh axes — the first tiled
    GLOBAL collective in the cube stages (vs the halo-only tendency stages).  The
    global total area is closed over (a static reduction of the face-replicated
    ``grid.area``).  Accumulation in ``conservation_accumulator`` (f64 under x64).
    """
    from legoesm.core.conservation import conservation_accumulator

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(grid, "area", None) is None:
        raise ValueError(
            "make_tiled_zero_mean_tendency_stage_2d: grid must have .area")
    nl = n // kt
    acc = conservation_accumulator()
    area = grid.area                                   # (6, n, n)
    total_area = jnp.sum(area.astype(acc))             # global; closed over

    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo), out_specs=co,
             check_vma=False)
    def _body(tend, ar):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

        tend_t = _s(tend)                              # (1, nl, nl)
        ar_t = _s(ar)
        local_wsum = jnp.sum(tend_t.astype(acc) * ar_t.astype(acc))
        # global area-weighted sum across ALL 6*kt*kt tiles (the new collective).
        global_wsum = jax.lax.psum(
            local_wsum, axis_name=("face", "tile_i", "tile_j"))
        correction = global_wsum / total_area
        return (tend_t.astype(acc) - correction).astype(tend.dtype)

    def stage(tend):
        _check_shapes(n, tend=(tend, (n, n)))
        return _body(tend, area)

    return stage


# ---------------------------------------------------------------------------
# Tiled GLOBAL reduction: dry-mass fixer fix_ps_mass via psum.  The cube tiled
# STEP's post-RK3 mass conservation (default use_conservation_fixer+fix_mass) —
# the second tiled global reduction (after zero_mean), and a TIER-0 truth
# (conservation) op.  Two area-weighted global sums (psum) of p_s_old/p_s_new ->
# a uniform additive p_s correction so dry mass is conserved.  Bit-identical to
# core.conservation.fix_ps_mass up to psum reduction-ORDER reordering; conserves
# sum(out*area)==sum(p_s_old*area) to machine precision (the point).
# ---------------------------------------------------------------------------

def _tile_fix_ps_mass_delta(p_s_new_t, p_s_old_t, ar_t, total_area, acc):
    """DELTA-FIRST per-tile dry-mass fix (shared by the standalone fixer stage
    and the blocked step's in-stage fixer — ONE copy of the conservation math).

    Sums the per-cell ``(old-new)*area`` BEFORE the reduction (codex MAJOR on
    the original stage: two huge near-equal masses ~1e19 would catastrophically
    cancel in f32-storage / x64-off mode), ``psum``s the per-tile deltas over
    the ``(face, tile_i, tile_j)`` mesh axes, and applies the uniform additive
    correction.  ``total_area`` is the closed-over GLOBAL single-sum (matching
    the global op's ``_total_area`` — NOT a psummed local area, which would
    reorder the denominator off the global).  No cast-back (``fix_ps_mass``
    keeps the promoted dtype).  Must be called INSIDE a shard_map over the
    tiled mesh."""
    local_delta = jnp.sum(
        (p_s_old_t.astype(acc) - p_s_new_t.astype(acc)) * ar_t.astype(acc))
    g_delta = jax.lax.psum(
        local_delta, axis_name=("face", "tile_i", "tile_j"))
    return p_s_new_t + g_delta / total_area


def _tile_fix_ps_mass_target(p_s_new_t, target_mass, ar_t, total_area, acc):
    """TARGET-anchored per-tile dry-mass fix (the compiled-segment driver's
    ``fix_ps_mass_target`` semantics — the operator-split lane externalizes
    the fixer with a fixed t=0 target): uniform additive correction
    ``(target - psum(sum(p_s*area))) / total_area``.  Shares the psum axes /
    closed-over ``total_area`` doctrine of :func:`_tile_fix_ps_mass_delta`
    (see its docstring); a zero target disables the fix (the serial
    ``target_mass=0`` convention).  Must be called INSIDE a shard_map over
    the tiled mesh."""
    local_mass = jnp.sum(p_s_new_t.astype(acc) * ar_t.astype(acc))
    g_mass = jax.lax.psum(local_mass, axis_name=("face", "tile_i", "tile_j"))
    correction = (target_mass.astype(acc) - g_mass) / total_area
    return jnp.where(target_mass > 0, p_s_new_t + correction, p_s_new_t)


def make_tiled_fix_ps_mass_stage_2d(mesh, grid, n: int, kt: int):
    """Tiled ``fix_ps_mass`` (non-anchor dry-mass fixer) on a ``(6, kt, kt)`` mesh.

    ``stage(p_s_new, p_s_old) -> p_s_new + (mass_old - mass_new)/total_area`` with
    ``mass_* = sum(p_s_* * area)`` over ALL tiles via a single ``jax.lax.psum`` of
    the stacked per-tile (old, new) area-weighted sums.  Both inputs 2D cc
    ``(6, n, n)`` FACE-REPLICATED; tile-sharded ``P("face","tile_i","tile_j")``
    out.  Accumulation in ``conservation_accumulator`` (f64 under x64); the global
    total area is closed over (static reduction of face-replicated ``grid.area``).
    """
    from legoesm.core.conservation import conservation_accumulator

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(grid, "area", None) is None:
        raise ValueError(
            "make_tiled_fix_ps_mass_stage_2d: grid must have .area")
    nl = n // kt
    acc = conservation_accumulator()
    area = grid.area                                   # (6, n, n)
    total_area = jnp.sum(area.astype(acc))             # global; closed over

    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo), out_specs=co,
             check_vma=False)
    def _body(p_s_new, p_s_old, ar):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, nl, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, nl, axis=2)

        pn = _s(p_s_new)
        po = _s(p_s_old)
        art = _s(ar)
        # Delta-first math + psum live in the shared _tile_fix_ps_mass_delta
        # (also the blocked step's in-stage fixer) — see its docstring for
        # the codex-MAJOR cancellation rationale.
        return _tile_fix_ps_mass_delta(pn, po, art, total_area, acc)

    def stage(p_s_new, p_s_old):
        _check_shapes(n, p_s_new=(p_s_new, (n, n)), p_s_old=(p_s_old, (n, n)))
        return _body(p_s_new, p_s_old, area)

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


def make_tiled_interp_corner_to_center_stage_2d(mesh, n: int, kt: int,
                                                nlev: int | None = None):
    """Sharded ``interp_corner_to_center`` on a ``(6, kt, kt)`` mesh.
    ``stage(field_d) -> cc`` (face-replicated corner in; tile-sharded cc out,
    gathered ``(6, n, n)`` — exact cc partition, no shared face).

    ``nlev`` (3D PE dycore): when set, ``field_d`` is 4D ``(6, n+1, n+1, nlev)``
    and ``cc`` is 4D ``(6, n, n, nlev)`` — a trailing replicated vertical axis
    (the vertical is NOT tiled)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    fi = P("face", None, None, None) if nlev else P("face", None, None)
    co = (P("face", "tile_i", "tile_j", None) if nlev
          else P("face", "tile_i", "tile_j"))

    @partial(shard_map, mesh=mesh, in_specs=(fi,), out_specs=co,
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
    _check_tiled_mesh(mesh, n, kt)
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


def make_tiled_arakawa_lamb_gradient_stage_2d(mesh, cdgrid, n: int, kt: int,
                                              nlev: int | None = None):
    """Sharded default-path ``arakawa_lamb_gradient`` on a ``(6, kt, kt)`` mesh.
    ``stage(B_pad) -> (dB_dx, dB_dy_perp)`` where ``B_pad`` is the GLOBAL
    face-replicated ``pad_halo_auto(B)`` ``(6, n+2, n+2)``; the corner matrices
    are passed as face-sharded inputs (NOT closed over -> codex U4a HIGH).  Both
    outputs tile-sharded, gathered ``(6, kt*(nl+1), kt*(nl+1))`` reassembling
    lower-owns-shared to ``(6, n+1, n+1)``.

    ``nlev`` (3D PE dycore): when set, ``B_pad`` is 4D ``(6, n+2, n+2, nlev)``
    and the gradients are 4D ``(6, n+1, n+1, nlev)`` — a trailing replicated
    vertical axis; the corner matrices stay 2D-face (broadcast over nlev)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    nl = n // kt
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    fo = P("face", None, None)                 # 2D-face matrix
    co = P("face", "tile_i", "tile_j")
    fb = P("face", None, None, None) if nlev else fo            # B_pad
    cz = (P("face", "tile_i", "tile_j", None) if nlev else co)  # grad outputs

    @partial(shard_map, mesh=mesh, in_specs=(fb, fo, fo, fo, fo),
             out_specs=(cz, cz), check_vma=False)
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
    _check_tiled_mesh(mesh, n, kt)
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
    _check_tiled_mesh(mesh, n, kt)
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


# ---------------------------------------------------------------------------
# ASSEMBLY: the full tiled ``fv3_sw_tendencies`` MOMENTUM stage (du_d_dt,
# dv_d_dt).  This is the genuine np24 unlock the per-op kernels above were
# building toward.  Unlike a per-op kernel — which slices a GLOBAL
# face-replicated pre-pad of its (single) input — the assembly chains several
# ops whose halos are on IN-STAGE INTERMEDIATES (B from the Bernoulli fn; the
# cc winds u_cc/v_cc; the cc tendencies du_cc/dv_cc).  Those have no global
# pre-pad to slice, so the stage halo-exchanges them WITHIN the shard_map via
# the unwrapped tiled pad bodies (``make_tiled_pad_body`` scalar +
# ``make_tiled_pad_vector_body`` vector) — the SAME ppermute-over-(face,tile_i,
# tile_j) machinery the wrapped exchange uses, bit-exact to the serial
# ``pad_halo``/``pad_halo_vector`` at every cell incl. tile corners (the P3
# diagonal/sliver corner rounds; ``tests/parallel/test_tiled_pad_body.py``
# asserts array-equality vs serial).  Numerics come ONLY from the shared
# ``*_core`` ops (no dup).
#
# Scope of THIS increment — the BASE momentum path of fv3_sw_tendencies:
#   div_damp=0, hyperdiff_coeff=0, boundary_fix=False,
#   non-duogrid (orthogonal rotation) cube.
# The optional terms (div damp / hyperdiff / boundary smoothing) are deferred
# — each rides the same per-op kernels + one more
# in-stage halo and is its own increment.  The mass tendency dh_dt (PPM
# ``cgrid_mass_flux_divergence``) is the separately-tracked hardest op.
# ---------------------------------------------------------------------------

def make_tiled_fv3_sw_momentum_stage_2d(mesh, cdgrid, n: int, kt: int,
                                        g: float | None = None):
    """Build the sharded tiled ``fv3_sw_tendencies`` MOMENTUM stage on a
    ``(6, kt, kt)`` mesh (axes ``("face","tile_i","tile_j")``).

    Returns ``stage(h, u_d, v_d, h_s) -> (du_d_dt, dv_d_dt)`` where the four
    state inputs are FACE-SHARDED, TILE-REPLICATED (``P("face",None,None)``:
    each device holds its face's FULL field, replicated across that face's
    ``kt*kt`` tiles, so each tile ``dynamic_slice``s its own sub-window):
    ``h``/``h_s`` cc ``(6,n,n)``, ``u_d`` ``(6,n,n+1)``, ``v_d`` ``(6,n+1,n)``
    D-grid edge-midpoint winds.  Outputs are tile-sharded ``P("face","tile_i",
    "tile_j")``; gathered ``du_d_dt`` ``(6, kt*nl, kt*(nl+1))`` and ``dv_d_dt``
    ``(6, kt*(nl+1), kt*nl)`` reassemble lower-owns-shared (drop the duplicated
    shared staggered face) to ``(6, n, n+1)`` / ``(6, n+1, n)``.

    All static metrics are passed as face-sharded shard_map inputs (NEVER
    closed over a full ``(6,...)`` array — that broadcasts the output back to
    face extent 6, the codex U4a HIGH).  ``g`` defaults to ``constants.g`` to
    match ``fv3_sw_tendencies``.  Base cut is orthogonal-rotation only:
    a duogrid grid raises (the vector body does not yet carry the
    kinked->extended remap).
    """
    from legoesm.core.operators_cdgrid import (
        arakawa_lamb_gradient_core,
        dgrid_vorticity_core,
        fv3_d2cc,
        interp_corner_to_center,
    )
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body, make_tiled_pad_vector_body

    from legoesm import constants

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_sw_momentum_stage_2d: n={n} != cdgrid.n="
            f"{cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_sw_momentum_stage_2d: base cut supports the "
            "orthogonal-rotation (non-duogrid) cube only; the in-stage vector "
            "halo does not yet carry the duogrid kinked->extended remap.")
    g = constants.g if g is None else g
    nl = n // kt

    # Static metrics (face-sharded, tile-replicated -> sliced per tile).
    cosa_corner = cdgrid.cosa_corner               # (6, n+1, n+1)
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x  # (6,n,n+1)/(6,n+1,n)
    area = grid.area                               # (6, n, n)
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01  # (6, n+1, n+1)
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    cos_a, sin_a = grid.cos_angle, grid.sin_angle              # (6, n, n)
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded    # (6, n+2, n+2)
    f_cor = grid.f                                  # (6, n, n)
    offsets = grid.halo_interp_offsets             # (6, 4, n) — non-duogrid

    # Unwrapped in-stage halo bodies (built once; called inside the body).
    # ndim=3: global field rank (the per-device tile is 2-D).
    scalar_body = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=3, halo=1, with_offsets=True)

    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh,
             in_specs=(fo,) * 4        # h, u_d, v_d, h_s
                       + (fo,) * 5     # cosa_corner, dx_edge_y, dy_edge_x,
                                       #   area, f
                       + (fo,) * 4     # gc00..gc11
                       + (fo,) * 4     # cos_a, sin_a, cap, sap
                       + (P(),),       # offsets (replicated)
             out_specs=(co, co), check_vma=False)
    def _body(h, u_d, v_d, h_s,
              cosa_c, dxe, dye, ar, fco,
              c00, c01, c10, c11,
              ca, sa, capf, sapf, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            """Tile slice of a (1, A, B) face-shard at (a_i, a_j)."""
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        # ---- (a) cell-centre winds (D-grid edge-midpoint -> cc, LOCAL) ----
        # u_d (n,n+1): staggered +1 in j; v_d (n+1,n): staggered +1 in i.
        u_d_t = _s(u_d, nl, nl + 1)
        v_d_t = _s(v_d, nl + 1, nl)
        u_cc, v_cc = fv3_d2cc(u_d_t, v_d_t, None)        # (1, nl, nl) each

        # ---- (c) Bernoulli (pointwise) ----
        h_t = _s(h, nl, nl)
        h_s_t = _s(h_s, nl, nl)
        B = 0.5 * (u_cc ** 2 + v_cc ** 2) + g * (h_t + h_s_t)  # (1, nl, nl)

        # Per-tile rotation metrics (2-D, leading face axis stripped) for the
        # in-stage vector halos.
        ca_t = _s(ca, nl, nl)[0]
        sa_t = _s(sa, nl, nl)[0]
        cap_t = _s(capf, nl + 2, nl + 2)[0]       # tiled_padded_block (h=1)
        sap_t = _s(sapf, nl + 2, nl + 2)[0]

        def _vpad(u_1, v_1):
            up, vp = vector_body(u_1[0], v_1[0], ca_t, sa_t, cap_t, sap_t, offs)
            return up[None], vp[None]             # (1, nl+2, nl+2) each

        # ---- (d) Arakawa-Lamb gradient (in-stage SCALAR halo on B) ----
        B_pad = scalar_body(B[0], offs)[None]     # (1, nl+2, nl+2)
        dB_dx, dB_dy_perp = arakawa_lamb_gradient_core(
            B_pad, _s(c00, nl + 1, nl + 1), _s(c01, nl + 1, nl + 1),
            _s(c10, nl + 1, nl + 1), _s(c11, nl + 1, nl + 1))  # (1, nl+1, nl+1)

        # ---- (e) corner winds (in-stage VECTOR halo on u_cc, v_cc) ----
        u_cc_pad, v_cc_pad = _vpad(u_cc, v_cc)    # (1, nl+2, nl+2)
        u_corner = 0.25 * (
            u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
            + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])     # (1, nl+1, nl+1)
        v_corner = 0.25 * (
            v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
            + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])

        # ---- (f) absolute vorticity (cc; corner winds already +1) ----
        zeta = dgrid_vorticity_core(
            u_corner, v_corner, _s(cosa_c, nl + 1, nl + 1),
            _s(dxe, nl, nl + 1), _s(dye, nl + 1, nl), _s(ar, nl, nl))
        zeta_abs = zeta + _s(fco, nl, nl)         # (1, nl, nl)

        # ---- (g) cc momentum tendency ----
        dB_dx_cc = interp_corner_to_center(dB_dx)
        dB_dy_cc = interp_corner_to_center(dB_dy_perp)
        du_cc = zeta_abs * v_cc - dB_dx_cc        # (1, nl, nl)
        dv_cc = -zeta_abs * u_cc - dB_dy_cc

        # ---- (k) project cc tendency to D-grid (in-stage VECTOR halo) ----
        du_cc_pad, dv_cc_pad = _vpad(du_cc, dv_cc)
        du_d_dt = 0.5 * (du_cc_pad[:, 1:-1, :-1]
                         + du_cc_pad[:, 1:-1, 1:])     # (1, nl, nl+1)
        dv_d_dt = 0.5 * (dv_cc_pad[:, :-1, 1:-1]
                         + dv_cc_pad[:, 1:, 1:-1])     # (1, nl+1, nl)
        return du_d_dt, dv_d_dt

    def stage(h, u_d, v_d, h_s):
        _check_shapes(n, h=(h, (n, n)), u_d=(u_d, (n, n + 1)),
                      v_d=(v_d, (n + 1, n)), h_s=(h_s, (n, n)))
        return _body(h, u_d, v_d, h_s,
                     cosa_corner, dx_edge_y, dy_edge_x, area, f_cor,
                     gc00, gc01, gc10, gc11,
                     cos_a, sin_a, cap, sap, offsets)

    return stage


# ---------------------------------------------------------------------------
# MASS-PPM: the tiled `cgrid_mass_flux_divergence` height tendency (dh_dt) —
# the design-doc HARDEST op.  Unlike the momentum assembly (in-stage halos on
# intermediates), the PPM mass divergence tiles via the U3 DEEP-GLOBAL-PRE-PAD
# pattern (cf. `tiled_transport.py` ppm_transport_1d(external_halo, rd_prepadded)):
# `h` is a STAGE INPUT (cc height), so pre-pad it GLOBALLY (face-replicated)
# one ring deeper than the production halo=2 and slice the deep window per tile
# -> the per-tile PPM reconstruction is LOCAL (NO in-stage ppermute).  The cc
# winds u_c/v_c are stage inputs too (staggered slice, NO halo: flux divergence
# reads only a cell's own bounding faces, cf cgrid_divergence_local).
#
# The depth: PPM reconstruction of a tile-boundary cell (local -1) reads cells
# [-3..1]; in the GLOBAL face that -3 is the cube edge (the production op's
# internal `mode='edge'` pad supplies it), but at an INTERIOR tile cut -3 is a
# real neighbour -> the tile carries ONE ring deeper (halo_in=3).  The deep
# pad's outer ring is edge-extended from the halo=2 pad, so a FACE-edge tile
# reproduces the global's `mode='edge'` value exactly.  The shared
# `cgrid_ppm_fluxes_core(halo_in=3)` reuses the production numerics verbatim.
#
# Scope: base case (apply_fortran_xppm_boundary=False -> the `n_interior`
# face-edge override is OFF; non-duogrid -> NO `synchronize_cgrid_fluxes`).
# The Fortran xppm overrides + duogrid flux-sync are later increments.
# ---------------------------------------------------------------------------

def cgrid_mass_divergence_tile_2d(h_deep, u_c, v_c, dy, dx, area,
                                  a_i, a_j, nl: int):
    """Per-tile ``cgrid_mass_flux_divergence`` (base case).  ``h_deep``
    ``(F, n+6, n+6)`` = the GLOBAL deep pre-pad (halo=2 cross-face + 1 edge
    ring; FACE-REPLICATED); ``u_c`` ``(F, n+1, n)``, ``v_c`` ``(F, n, n+1)``
    C-grid winds; ``dy`` ``(F, n+1, n)`` (=dy_edge_x), ``dx`` ``(F, n, n+1)``
    (=dx_edge_y) face lengths; ``area`` ``(F, n, n)`` — all FACE-REPLICATED.
    The cc tile ``[a:a+nl]`` reads the ``nl+6`` deep h-window
    ``h_deep[a:a+nl+6]`` (global cells ``[a-3..a+nl+2]``) and the staggered
    wind/metric blocks.  Returns ``dh_dt`` ``(F, nl, nl)`` — cc cells
    partition exactly (no shared face, no trim)."""
    from legoesm.core.operators_cdgrid import cgrid_ppm_fluxes_core

    nfc = h_deep.shape[1]                       # n + 6
    nf = nfc - 6                                # n
    # Static guards (codex U3d/U4a pattern): a mis-sized input would make
    # dynamic_slice_in_dim silently CLAMP instead of failing.
    if h_deep.shape[1:] != (nf + 6, nf + 6):
        raise ValueError(
            f"cgrid_mass_divergence_tile_2d: h_deep must be the (n+6, n+6) "
            f"deep pre-pad; got {h_deep.shape[1:]}")
    if u_c.shape[1:] != (nf + 1, nf) or v_c.shape[1:] != (nf, nf + 1):
        raise ValueError(
            f"cgrid_mass_divergence_tile_2d: u_c/v_c must be C-grid "
            f"{(nf + 1, nf)}/{(nf, nf + 1)}; got u_c={u_c.shape[1:]}, "
            f"v_c={v_c.shape[1:]}")
    if dy.shape[1:] != (nf + 1, nf) or dx.shape[1:] != (nf, nf + 1):
        raise ValueError(
            f"cgrid_mass_divergence_tile_2d: dy/dx must be {(nf + 1, nf)}/"
            f"{(nf, nf + 1)}; got dy={dy.shape[1:]}, dx={dx.shape[1:]}")
    if area.shape[1:] != (nf, nf):
        raise ValueError(
            f"cgrid_mass_divergence_tile_2d: area must be {(nf, nf)}; got "
            f"{area.shape[1:]}")

    # Deep h-window: global cells [a-3..a+nl+2] live at deep indices
    # [a..a+nl+5] (global c at index c+3), so the slice start IS a (the +3
    # halo offset is absorbed by the deep-pad index origin).
    hw = jax.lax.dynamic_slice_in_dim(h_deep, a_i, nl + 6, axis=1)
    hw = jax.lax.dynamic_slice_in_dim(hw, a_j, nl + 6, axis=2)

    def _stag(arr, si, sj):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

    u_c_t = _stag(u_c, nl + 1, nl)
    v_c_t = _stag(v_c, nl, nl + 1)
    dy_t = _stag(dy, nl + 1, nl)
    dx_t = _stag(dx, nl, nl + 1)
    area_t = _stag(area, nl, nl)

    flux_x, flux_y = cgrid_ppm_fluxes_core(
        hw, u_c_t, v_c_t, dy_t, dx_t, nl, halo_in=3,
        effective_xppm_boundary=False)
    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    return -(net_x + net_y) / area_t


def make_tiled_cgrid_mass_divergence_stage_2d(mesh, cdgrid, n: int, kt: int):
    """Build the sharded base-case ``cgrid_mass_flux_divergence`` (dh_dt) stage
    on a ``(6, kt, kt)`` mesh.  Returns ``stage(h, u_c, v_c) -> dh_dt`` where
    ``h`` cc ``(6,n,n)``, ``u_c`` ``(6,n+1,n)``, ``v_c`` ``(6,n,n+1)`` are
    FACE-SHARDED, TILE-REPLICATED.  The deep pre-pad ``h_deep`` (the halo=2
    cross-face pad + 1 edge ring) is built GLOBALLY in ``stage`` (approach-C:
    a cheap cross-face pre-pad of the STAGE INPUT) and threaded as a
    face-sharded input; the staggered metrics likewise (NOT closed over -> the
    codex U4a face-broadcast HIGH).  Output tile-sharded, gathered ``(6,n,n)``
    (cc cells partition exactly).

    Base case only: non-duogrid (no ``synchronize_cgrid_fluxes``) and
    apply_fortran_xppm_boundary=False (the ``n_interior`` face-edge override is
    off; tiling a GLOBAL-index-keyed override is a later increment)."""
    import jax.numpy as jnp
    from legoesm.core.operators_cdgrid import pad_halo_auto_h2

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_cgrid_mass_divergence_stage_2d: n={n} != cdgrid.n="
            f"{cdgrid.n}")
    if cdgrid.base.duogrid is not None:
        raise ValueError(
            "make_tiled_cgrid_mass_divergence_stage_2d: base cut supports the "
            "non-duogrid cube only (duogrid needs synchronize_cgrid_fluxes, a "
            "later increment).")
    nl = n // kt
    dy_edge_x, dx_edge_y = cdgrid.dy_edge_x, cdgrid.dx_edge_y
    area = cdgrid.base.area
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo, fo, fo, fo),
             out_specs=co, check_vma=False)
    def _body(h_deep, u_c, v_c, dy, dx, ar):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return cgrid_mass_divergence_tile_2d(
            h_deep, u_c, v_c, dy, dx, ar, a_i, a_j, nl)

    def stage(h, u_c, v_c):
        _check_shapes(n, h=(h, (n, n)), u_c=(u_c, (n + 1, n)),
                      v_c=(v_c, (n, n + 1)))
        # GLOBAL deep pre-pad: halo=2 cross-face pad + 1 edge ring (matches the
        # production op's internal mode='edge' ghost at the cube edge), built
        # ONCE outside the shard_map (approach-C cross-face pre-pad).
        h_pad2 = pad_halo_auto_h2(h, cdgrid)            # (6, n+4, n+4)
        h_deep = jnp.pad(h_pad2, ((0, 0), (1, 1), (1, 1)), mode="edge")
        return _body(h_deep, u_c, v_c, dy_edge_x, dx_edge_y, area)

    return stage


# ---------------------------------------------------------------------------
# CAPSTONE: the FULL tiled `fv3_sw_tendencies` np24 stage = the SHIPPED momentum
# assembly + the SHIPPED mass divergence, COMPOSED.  The win of composing (vs
# running the two stages separately) is that the cc-wind VECTOR halo is computed
# ONCE and shared: the same `u_cc_pad`/`v_cc_pad` feeds BOTH `fv3_cc2c_core`
# (-> the C-grid winds u_c/v_c for the mass PPM divergence) AND the momentum
# corner winds (0.25 box).  So the full stage carries the SAME halo budget as
# the momentum stage alone: 1 scalar (B) + 2 vector (shared u_cc/v_cc; the
# du_cc/dv_cc tendency projection) in-stage halos, plus the GLOBAL deep-h
# pre-pad (mass, a stage input).  Every numeric is a shared `*_core` op already
# bit-identity-validated in isolation; this stage is the wiring + a full
# 3-output bit-identity gate.
#
# Base case only (same as the two halves): div_damp=0, hyperdiff=0,
# boundary_fix=False, fortran_*=False, non-duogrid.
# ---------------------------------------------------------------------------

def make_tiled_fv3_sw_tendencies_stage_2d(mesh, cdgrid, n: int, kt: int,
                                          g: float | None = None):
    """Build the FULL base-case tiled ``fv3_sw_tendencies`` stage on a
    ``(6, kt, kt)`` mesh.  Returns ``stage(h, u_d, v_d, h_s) -> (dh_dt,
    du_d_dt, dv_d_dt)`` (the three SW tendencies), FACE-SHARDED/TILE-REPLICATED
    inputs, tile-sharded outputs (``dh_dt`` gathers ``(6,n,n)`` exact;
    ``du_d_dt`` ``(6, kt*nl, kt*(nl+1))`` / ``dv_d_dt`` ``(6, kt*(nl+1), kt*nl)``
    reassemble lower-owns-shared to ``(6,n,n+1)``/``(6,n+1,n)``).

    Composes :func:`make_tiled_fv3_sw_momentum_stage_2d` (momentum) and
    :func:`make_tiled_cgrid_mass_divergence_stage_2d` (mass), SHARING the
    ``u_cc``/``v_cc`` vector halo.  All static metrics passed as face-sharded
    in_specs (never closed over -> the codex U4a face-broadcast HIGH).  ``g``
    defaults to ``constants.g``.  Base cut: non-duogrid orthogonal rotation."""
    import jax.numpy as jnp
    from legoesm.core.operators_cdgrid import (
        arakawa_lamb_gradient_core,
        cgrid_ppm_fluxes_core,
        dgrid_vorticity_core,
        fv3_cc2c_core,
        fv3_d2cc,
        interp_corner_to_center,
        pad_halo_auto_h2,
    )
    from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body, make_tiled_pad_vector_body

    from legoesm import constants

    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    _check_tiled_mesh(mesh, n, kt)
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_fv3_sw_tendencies_stage_2d: n={n} != cdgrid.n="
            f"{cdgrid.n}")
    grid = cdgrid.base
    if grid.duogrid is not None:
        raise ValueError(
            "make_tiled_fv3_sw_tendencies_stage_2d: base cut supports the "
            "non-duogrid orthogonal-rotation cube only.")
    g = constants.g if g is None else g
    nl = n // kt

    cosa_corner = cdgrid.cosa_corner
    dx_edge_y, dy_edge_x = cdgrid.dx_edge_y, cdgrid.dy_edge_x
    area = grid.area
    f_cor = grid.f
    cosa_u = cdgrid.cosa_u
    gc00, gc01 = cdgrid.grad_c00, cdgrid.grad_c01
    gc10, gc11 = cdgrid.grad_c10, cdgrid.grad_c11
    cos_a, sin_a = grid.cos_angle, grid.sin_angle
    cap, sap = grid.cos_angle_padded, grid.sin_angle_padded
    offsets = grid.halo_interp_offsets

    scalar_body = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)
    vector_body = make_tiled_pad_vector_body(
        mesh, ndim=3, halo=1, with_offsets=True)

    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh,
             in_specs=(fo,) * 4         # h_deep, u_d, v_d, h_s
                       + (fo,) * 6      # cosa_corner, dxe, dye, area, f, cosa_u
                       + (fo,) * 4      # gc00..gc11
                       + (fo,) * 4      # cos_a, sin_a, cap, sap
                       + (P(),),        # offsets
             out_specs=(co, co, co), check_vma=False)
    def _body(h_deep, u_d, v_d, h_s,
              cosa_c, dxe, dye, ar, fco, cosau,
              c00, c01, c10, c11,
              ca, sa, capf, sapf, offs):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        # ---- (a) cell-centre winds (LOCAL) ----
        u_cc, v_cc = fv3_d2cc(_s(u_d, nl, nl + 1), _s(v_d, nl + 1, nl), None)

        # ---- ONE shared cc-wind VECTOR halo (mass fv3_cc2c + momentum corners) ----
        ca_t = _s(ca, nl, nl)[0]
        sa_t = _s(sa, nl, nl)[0]
        cap_t = _s(capf, nl + 2, nl + 2)[0]
        sap_t = _s(sapf, nl + 2, nl + 2)[0]
        u_cc_p, v_cc_p = vector_body(
            u_cc[0], v_cc[0], ca_t, sa_t, cap_t, sap_t, offs)
        u_cc_pad = u_cc_p[None]            # (1, nl+2, nl+2)
        v_cc_pad = v_cc_p[None]

        # ---- (b) MASS: cc -> C winds, then PPM upwind divergence ----
        u_c, v_c = fv3_cc2c_core(u_cc_pad, v_cc_pad, _s(cosau, nl + 1, nl))
        hw = _s(h_deep, nl + 6, nl + 6)            # deep h window (1, nl+6, nl+6)
        dy_t = _s(dye, nl + 1, nl)
        dx_t = _s(dxe, nl, nl + 1)
        area_t = _s(ar, nl, nl)
        flux_x, flux_y = cgrid_ppm_fluxes_core(
            hw, u_c, v_c, dy_t, dx_t, nl, halo_in=3,
            effective_xppm_boundary=False)
        dh_dt = -((flux_x[:, 1:] - flux_x[:, :-1])
                  + (flux_y[:, :, 1:] - flux_y[:, :, :-1])) / area_t

        # ---- (c) Bernoulli (h from the deep window interior; pointwise) ----
        h_t = hw[:, 3:-3, 3:-3]                     # cc h tile (1, nl, nl)
        B = 0.5 * (u_cc ** 2 + v_cc ** 2) + g * (h_t + _s(h_s, nl, nl))

        # ---- (d) A-L gradient (in-stage SCALAR halo on B) ----
        B_pad = scalar_body(B[0], offs)[None]
        dB_dx, dB_dy_perp = arakawa_lamb_gradient_core(
            B_pad, _s(c00, nl + 1, nl + 1), _s(c01, nl + 1, nl + 1),
            _s(c10, nl + 1, nl + 1), _s(c11, nl + 1, nl + 1))

        # ---- (e) corner winds (REUSE the shared u_cc/v_cc vector halo) ----
        u_corner = 0.25 * (
            u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
            + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
        v_corner = 0.25 * (
            v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
            + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])

        # ---- (f) vorticity ----
        zeta = dgrid_vorticity_core(
            u_corner, v_corner, _s(cosa_c, nl + 1, nl + 1),
            _s(dxe, nl, nl + 1), _s(dye, nl + 1, nl), _s(ar, nl, nl))
        zeta_abs = zeta + _s(fco, nl, nl)

        # ---- (g) cc momentum tendency ----
        du_cc = zeta_abs * v_cc - interp_corner_to_center(dB_dx)
        dv_cc = -zeta_abs * u_cc - interp_corner_to_center(dB_dy_perp)

        # ---- (k) project to D-grid (in-stage VECTOR halo) ----
        du_cc_p, dv_cc_p = vector_body(
            du_cc[0], dv_cc[0], ca_t, sa_t, cap_t, sap_t, offs)
        du_d_dt = 0.5 * (du_cc_p[None][:, 1:-1, :-1]
                         + du_cc_p[None][:, 1:-1, 1:])
        dv_d_dt = 0.5 * (dv_cc_p[None][:, :-1, 1:-1]
                         + dv_cc_p[None][:, 1:, 1:-1])
        return dh_dt, du_d_dt, dv_d_dt

    def stage(h, u_d, v_d, h_s):
        _check_shapes(n, h=(h, (n, n)), u_d=(u_d, (n, n + 1)),
                      v_d=(v_d, (n + 1, n)), h_s=(h_s, (n, n)))
        # GLOBAL deep-h pre-pad (mass): halo=2 cross-face + 1 edge ring.
        h_deep = jnp.pad(pad_halo_auto_h2(h, cdgrid),
                         ((0, 0), (1, 1), (1, 1)), mode="edge")
        return _body(h_deep, u_d, v_d, h_s,
                     cosa_corner, dx_edge_y, dy_edge_x, area, f_cor, cosa_u,
                     gc00, gc01, gc10, gc11,
                     cos_a, sin_a, cap, sap, offsets)

    return stage


# ---------------------------------------------------------------------------
# Public re-exports of the shared tile-step building blocks for the
# OPERATOR-SPLIT tiled lane (driver/tiled_operator_split_step.py) — the
# no-private-cross-imports ratchet forbids importing the underscore names
# across modules; these aliases are the sanctioned surface.  Same objects,
# no wrappers: the tendency/RK3/fixer numerics stay single-source.
# ---------------------------------------------------------------------------
build_hydro_tile_tendency_fns = _build_hydro_tile_tendency_fns
sponge_rate_from_config = _sponge_rate_from_config
ssp_rk3_tile_step = _ssp_rk3_tile_step
tile_fix_ps_mass_target = _tile_fix_ps_mass_target
validate_tiled_step_factory_args = _validate_tiled_step_factory_args
