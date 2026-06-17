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
    to the global ``cgrid_divergence(dgrid_to_cgrid(u_d,v_d))`` -> continuity
    (all three ops cc/face-local).  ``coord`` sigma OR hybrid; the column-sum core
    is dispatched on its type.  Excludes the downstream global zero_mean_tendency.
    """
    from legoesm.core.operators_cdgrid import cgrid_divergence_local
    from legoesm.grids.vertical import (
        HybridSigmaPressureCoordinate,
        compute_mass_flux_hybrid,
        compute_sigma_dot_and_total,
    )
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    if getattr(cdgrid, "n", n) != n:
        raise ValueError(
            f"make_tiled_dp_s_dt_stage_2d: n={n} != cdgrid.n={cdgrid.n}")
    if getattr(coord, "n_levels", nlev) != nlev:
        raise ValueError(
            f"make_tiled_dp_s_dt_stage_2d: coord.n_levels={coord.n_levels} "
            f"!= nlev={nlev}")
    grid = cdgrid.base
    nl = n // kt
    _hybrid = isinstance(coord, HybridSigmaPressureCoordinate)
    if not _hybrid:
        _sigma_range = 1.0 - float(coord.sigma_half[0])   # static scalar

    cosa_u = cdgrid.cosa_u                          # (6, n+1, n)
    dy_edge_x = cdgrid.dy_edge_x                    # (6, n+1, n) — x-face length
    dx_edge_y = cdgrid.dx_edge_y                    # (6, n, n+1) — y-face length
    area = grid.area                               # (6, n, n)

    fo = P("face", None, None)                     # 2D-face metric / cc 2D
    fw = P("face", None, None, None)               # 4D winds
    co = P("face", "tile_i", "tile_j")             # 2D cc output

    @partial(shard_map, mesh=mesh,
             in_specs=(fw, fw, fo, fo, fo, fo, fo),  # u_d,v_d,p_s,cosa_u,dye,dxe,area
             out_specs=co, check_vma=False)
    def _body(u_d, v_d, p_s, cu, dye, dxe, ar):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl

        def _s(arr, si, sj):
            arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
            return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

        # D->C (local within-face) -> C-grid flux divergence (local; the
        # staggered u_c/v_c carry the tile boundary faces, no halo).
        u_c, v_c = dgrid_to_cgrid_tile_2d(u_d, v_d, cu, a_i, a_j, nl)
        div_v = cgrid_divergence_local(
            u_c, v_c, _s(dye, nl + 1, nl), _s(dxe, nl, nl + 1),
            _s(ar, nl, nl))                          # (1, nl, nl, nlev)

        p_s_t = _s(p_s, nl, nl)                      # (1, nl, nl)
        if _hybrid:
            _mf, D_total = compute_mass_flux_hybrid(div_v, p_s_t, coord)
            dp_s_dt = -D_total[..., 0] / coord.B_range
        else:
            _sd, D_total = compute_sigma_dot_and_total(div_v, coord)
            dp_s_dt = -p_s_t * D_total[..., 0] / _sigma_range
        return dp_s_dt                               # (1, nl, nl)

    def stage(u_d, v_d, p_s):
        _check_shapes(n, u_d=(u_d, (n + 1, n + 1, nlev)),
                      v_d=(v_d, (n + 1, n + 1, nlev)), p_s=(p_s, (n, n)))
        return _body(u_d, v_d, p_s, cosa_u, dy_edge_x, dx_edge_y, area)

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
#   fortran_vector_corner_fill=False, all fortran_* corner diagnostics False,
#   non-duogrid (orthogonal rotation) cube.
# The optional terms (div damp / hyperdiff / boundary smoothing / Fortran
# corner specials) are deferred — each rides the same per-op kernels + one more
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
