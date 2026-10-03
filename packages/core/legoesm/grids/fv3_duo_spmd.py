"""SPMD machinery for the duo six-face exchange (M3 ladder, PR #1656 line).

This module holds the pieces that let the certified duo halo exchange run
face-sharded with O(halo) communication.  First piece: the CROSS-FACE DEPTH
CENSUS -- the design's load-bearing number.

Measured (jobs 9493433-9493438, decode verified against
``_FlatLayout.idx`` by adversarial review): every cross-face READ in the
duo halo machinery lives in the strip scatters, at source depth-from-edge
<= 7 (the ng=4 geo lattice; A/B/C/D strips 5-6).  The k2e stencils, all
five corner-Lagrange modes, ``fill_corners``, the write-backs and the
edge blends are face-local once strips have landed.  A ring exchange that
gathers width-w full borders (w > census) therefore feeds the certified
flat tables verbatim.

The census is a BUILD-TIME GATE, not a constant: the depth depends on
``(n, ng, ngp)`` (codex MAJOR -- ``ngp`` is contextual), so the SPMD
exchange must call :func:`assert_ring_width_covers` on the actual tables
it will run with, and refuse a ring that does not cover them.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "blend_as_stencil",
    "build_tiled_barrier_split",
    "tiled_split_body",
    "tiled_vector_body",
    "tiled_me",
    "census_cross_face_depth",
    "assert_ring_width_covers",
    "build_ring_comm",
    "tile_spec",
    "build_tiled_ext_scalar_a_split",
    "build_tiled_ext_scalar_b_split",
    "build_tiled_fill_corners_agrid_split",
    "build_tiled_k2e_split",
    "build_tiled_corner_lagrange_split",
    "build_tiled_fill_corners_split",
    "build_tiled_ext_vector_splits",
    "make_tiled_from_split",
    "make_tiled_ext_scalar_sixface",
    "make_tiled_ext_scalar_sixface_allk",
    "make_tiled_fill_corners_agrid",
    "make_tiled_fill_corners",
    "make_tiled_k2e_remap_halo_rings",
    "make_tiled_corner_lagrange_fill",
    "make_tiled_ext_vector_dgrid_sixface",
    "make_tiled_ext_vector_cgrid_sixface",
    "make_tiled_ext_vector_dgrid_sixface_allk",
    "make_tiled_ext_vector_cgrid_sixface_allk",
    "make_tiled_ext_vector_sixface_from_splits",
    "to_blocked",
    "from_blocked",
]


# table attr -> layout attr, from the exchange fns' own pairings
# (fv3_duo_halos.py:1452-1520 and the builder at :1278-1430).
_TABLE_LAYOUTS = {
    "ex_a_strip": "lay_a", "ex_a_corner": "lay_a",
    "ex_a4_strip": "lay_a4", "ex_a4_corner": "lay_a4",
    "ex_b_strip": "lay_b",
    "ex_c_strip": "lay_c", "ex_c_corner": "lay_c",
    "ex_d_strip": "lay_d", "ex_d_corner": "lay_d",
    "k2e_a": "lay_a", "k2e_a4": "lay_a4", "k2e_b": "lay_b",
    "avg_c": "lay_fx", "avg_b": "lay_bb",
    "wr_d": "lay_wd", "wr_c": "lay_wc",
    "fc_bgrid_x": "lay_b", "fc_agrid_x": "lay_a", "fc_agrid_y": "lay_a",
    "fc_dgrid": "lay_d", "fc_cgrid": "lay_c", "fc_agrid_pair": "lay_ap",
}
_CORNER_LAYOUTS = {"a4": "lay_a4", "a3": "lay_a", "b3": "lay_b",
                   "du3": "lay_d", "dv3": "lay_d"}


def _decode_fij(lay, flat):
    """flat index -> ``(k, face, i, j)``, inverting ``_FlatLayout.idx``
    (``base + (face*m0 + i0)*m1 + j0``).  Shape-preserving.  The ONLY
    flat-index -> subscript meaning in this module: the census
    (:func:`_decode`) and the tiled table split both ride on it.
    Raises on an out-of-range index -- an index outside the layout has
    no meaning, and guessing one would mis-tile a table row."""
    shp = np.asarray(flat).shape
    flat = np.asarray(flat).ravel()
    if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= lay.total):
        raise ValueError(
            f"duo flat decode: index outside [0, {lay.total}) "
            f"(min {int(flat.min())}, max {int(flat.max())}) -- this "
            f"table does not belong to this layout")
    bases = np.asarray(list(lay.bases) + [lay.total])
    k = np.searchsorted(bases, flat, side="right") - 1
    face = np.empty_like(flat)
    ii = np.empty_like(flat)
    jj = np.empty_like(flat)
    for kk, (m0, m1) in enumerate(lay.shapes):
        m = k == kk
        if not m.any():
            continue
        rem = flat[m] - lay.bases[kk]
        face[m] = rem // (m0 * m1)
        r2 = rem % (m0 * m1)
        ii[m] = r2 // m1
        jj[m] = r2 % m1
    return (k.reshape(shp), face.reshape(shp),
            ii.reshape(shp), jj.reshape(shp))


def _decode(lay, flat):
    """flat index -> (face, depth-from-edge); see :func:`_decode_fij`."""
    k, face, ii, jj = _decode_fij(lay, flat)
    m0k = np.asarray([s[0] for s in lay.shapes])[k]
    m1k = np.asarray([s[1] for s in lay.shapes])[k]
    depth = np.minimum(np.minimum(ii, m0k - 1 - ii),
                       np.minimum(jj, m1k - 1 - jj))
    return face, depth


def _self_test(lay):
    """The decode must invert ``lay.idx`` exactly -- a wrong decode would
    bless a broken ring width, the worst failure mode (codex gate)."""
    rng = np.random.default_rng(0)
    for _ in range(64):
        k = int(rng.integers(len(lay.shapes)))
        m0, m1 = lay.shapes[k]
        face = int(rng.integers(6))
        i0, j0 = int(rng.integers(m0)), int(rng.integers(m1))
        f, d = _decode(lay, np.array([lay.idx(k, face, i0, j0)]))
        want_d = min(i0, m0 - 1 - i0, j0, m1 - 1 - j0)
        if int(f[0]) != face or int(d[0]) != want_d:
            raise AssertionError(
                f"duo SPMD census decode failed its self-test: "
                f"idx(k={k}, face={face}, i={i0}, j={j0}) decoded to "
                f"face={int(f[0])}, depth={int(d[0])} (want {face}, "
                f"{want_d})")


def _max_cross_depth(op, lay) -> int:
    """Max source depth over entries whose source face differs from the
    destination's.  -1 if the op never reads across faces."""
    if op is None:
        return -1
    if isinstance(op, (list, tuple)):
        return max((_max_cross_depth(o, lay) for o in op), default=-1)
    dst = getattr(op, "dst", None)
    if dst is None:
        raise TypeError(
            f"duo SPMD census: table op {type(op).__name__} has no 'dst' "
            f"-- a new op family the census does not understand; extend "
            f"it before sharding (fail closed, never guess)")
    dface, _ = _decode(lay, dst)
    worst = -1
    for a in ("src", "srcx", "srcy"):
        s = getattr(op, a, None)
        if s is None:
            continue
        sface, sdepth = _decode(lay, s)
        df = dface
        while df.ndim < sface.ndim:
            df = df[..., None]
        cross = sface != np.broadcast_to(df, sface.shape)
        if cross.any():
            worst = max(worst, int(sdepth[cross].max()))
    return worst


def census_cross_face_depth(tab) -> int:
    """Max depth-from-edge of any cross-face SOURCE cell over EVERY table
    in ``tab`` (a ``DuoHaloTables``).  Runs the decode self-test first.

    An unknown table attribute is a hard error, not a skip: a new op
    family added to the tables without extending this census could read
    deeper than the ring and silently consume zeros.
    """
    lays = {name: getattr(tab, name)
            for name in ("lay_a", "lay_a4", "lay_b", "lay_c", "lay_d",
                         "lay_fx", "lay_bb", "lay_wd", "lay_wc", "lay_ap")}
    for lay in lays.values():
        _self_test(lay)
    worst = -1
    for attr, layname in _TABLE_LAYOUTS.items():
        worst = max(worst,
                    _max_cross_depth(getattr(tab, attr), lays[layname]))
    for mode, layname in _CORNER_LAYOUTS.items():
        worst = max(worst,
                    _max_cross_depth(tab.corner[mode], lays[layname]))
    unknown = set(tab.corner) - set(_CORNER_LAYOUTS)
    if unknown:
        raise ValueError(
            f"duo SPMD census: corner modes {sorted(unknown)} have no "
            f"layout mapping -- extend _CORNER_LAYOUTS (fail closed)")
    return worst


def assert_ring_width_covers(tab, ring_width: int) -> int:
    """The build-time gate: refuse a ring narrower than the tables read.

    Returns the measured census so callers can log it.  ``ring_width``
    must EXCEED the census (>=, not >, would leave zero margin for the
    one-cell effective-footprint slack the design carries -- GLM's
    compounding-depth hole is closed by measurement at the composed
    level, but the raw gate keeps a +1 floor).
    """
    depth = census_cross_face_depth(tab)
    if ring_width <= depth:
        raise ValueError(
            f"duo SPMD ring width {ring_width} does not cover the "
            f"measured cross-face table depth {depth} for this "
            f"(n={tab.n}, ng={tab.ng}, ngp={tab.ngp}) -- the exchange "
            f"would silently read zeros. Widen the ring.")
    return depth


# ---------------------------------------------------------------------------
# ring exchange -- the O(halo) shard_map scalar exchange (v1.1)
# ---------------------------------------------------------------------------
#
# Wraps the certified ext_scalar_sixface_impl in a shard_map over a 'face' mesh
# axis: each device holds a contiguous block of faces, gathers only the
# width-w full BORDERS of every face (O(halo) -- the census above proves all
# cross-face reads land inside them), rebuilds a stack that is real where
# the tables read (borders everywhere + own faces in full), runs the
# certified flat tables VERBATIM, and keeps its own faces' slice.
#
# Scope (v1.1, dual-reviewed): scalar AND vector exchanges.  The vector
# flows need NO second mid-flow collective: an a4-depth-7 geo read maps
# to a c2l value at stepper depth 6 whose +-1 staggered u/v reads reach
# depth 5 -- inside the width-8 border -- so other faces' ring c2l
# values are recomputed locally from their gathered u/v rings, bitwise
# equal to owner-computed (codex-verified dependency arithmetic + the
# poison test).  Ceiling: the face axis is the only spatial shard (<= 6
# devices); v2 splits the tables into own-tile/neighbor-strip index sets.

def _border_mask(m: int, w: int) -> np.ndarray:
    """Boolean (m, m) mask of the width-w full border (corner blocks
    included -- the tables read diagonal corner cells)."""
    mask = np.zeros((m, m), dtype=bool)
    mask[:w, :] = True
    mask[-w:, :] = True
    mask[:, :w] = True
    mask[:, -w:] = True
    return mask


def make_ring_ext_scalar_sixface(tab, stag: str, mesh, *,
                                 ring_width: int = 8,
                                 poison: bool = False):
    """Build the shard_map'd scalar exchange for ``mesh`` (axis 'face').

    Returns ``fn(f6_local) -> f6_local`` operating on the LOCAL face block
    ``(6//d, m, m)`` of a global ``(6, m, m)`` array sharded ``P('face')``
    in certified face order (jax shards a (6,...) axis into contiguous
    blocks in axis order, so device i holds faces ``i*g..(i+1)*g-1`` --
    the face-order/mesh-order identity the design asserts).

    The certified tables run verbatim on a reconstructed stack; the
    census gate refuses a ring the tables outread.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    # the *_impl body, NOT the public dispatcher: with tab.ring_comm
    # set, the public name would recurse into this very shard_map
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_impl

    depth = assert_ring_width_covers(tab, ring_width)
    if stag == "A":
        m = tab.n + 2 * tab.ng
    elif stag == "B":
        m = tab.n + 2 * tab.ng + 1
    else:
        raise ValueError(
            f"make_ring_ext_scalar_sixface: stag {stag!r} (the certified "
            f"scalar exchange supports 'A' and 'B'; vector flows land "
            f"separately -- fail closed)")
    (ax_name, ax_size), = ((n, s) for n, s in
                          zip(mesh.axis_names, mesh.devices.shape))
    if 6 % ax_size != 0:
        raise ValueError(
            f"mesh axis {ax_name!r} has {ax_size} devices; 6 faces "
            f"require a divisor (1, 2, 3, 6)")
    g = 6 // ax_size                       # faces per device
    w = ring_width
    mask = _border_mask(m, w)
    # border cells enumerated ONCE, statically: (nb,) flat indices per face
    bidx = np.flatnonzero(mask.ravel())

    def _body(f6_local):
        # f6_local: (g, m, m) this device's faces, certified face order.
        # 1. own borders, fused into one payload
        borders = f6_local.reshape(g, m * m)[:, bidx]        # (g, nb)
        # 2. ONE all_gather of everyone's borders (O(halo)):
        #    (d, g, nb) -> (6, nb) in face order (axis order == face
        #    order under P('face') contiguous sharding).
        allb = jax.lax.all_gather(borders, ax_name)          # (d, g, nb)
        allb = allb.reshape(6, bidx.size)
        # 3. reconstruct: borders everywhere, own faces in full.  Zeros
        #    elsewhere -- the census proves the tables never read them,
        #    and the poison=True test PROVES the proof: NaN there instead
        #    of zeros, so any out-of-ring read corrupts the output
        #    loudly instead of silently contributing a plausible zero
        #    (GLM: zeros are indistinguishable from real data).
        fill = jnp.nan if poison else 0.0
        stack = jnp.full((6, m * m), fill, dtype=f6_local.dtype)
        stack = stack.at[:, bidx].set(allb)
        me = jax.lax.axis_index(ax_name)
        rows = me * g + jnp.arange(g)
        stack = stack.at[rows, :].set(f6_local.reshape(g, m * m))
        # 4. certified tables, verbatim
        out = ext_scalar_sixface_impl(stack.reshape(6, m, m), tab, stag)
        # 5. keep own faces
        return jax.lax.dynamic_slice_in_dim(out, me * g, g, axis=0)

    return shard_map(_body, mesh=mesh,
                     in_specs=P(ax_name), out_specs=P(ax_name),
                     check_rep=False), depth


def make_ring_ext_vector_sixface(tab, grid: str, mesh, *,
                                 ring_width: int = 8,
                                 poison: bool = False):
    """shard_map'd VECTOR exchange (D or C grid), O(halo) comm.

    ``grid``: ``"D"`` wraps ``ext_vector_dgrid_sixface_impl`` (u
    ``(6,ma,mb)``, v ``(6,mb,ma)``); ``"C"`` wraps
    ``ext_vector_cgrid_sixface_impl`` (uc
    ``(6,mb,ma)``, vc ``(6,ma,mb)``).  Returns ``(fn, depth)`` with
    ``fn(u_local, v_local) -> (u_local, v_local)``.

    ONE fused all_gather of both components' width-w borders replaces the
    design's second mid-flow collective: the composed flow's only
    cross-face reads are (1) the strip scatters (census depth <= 6) and
    (2) the geo-lattice exchange, whose sources are ``c2l`` values at a4
    depth <= 7 -- and ``c2l``/``pack_p1`` read only +-1 cell, so OTHER
    faces' ring c2l values are recomputed here from their gathered u/v
    rings, BITWISE equal to owner-computed (same inputs, same op).  The
    interior garbage of non-own faces never reaches an own-face output;
    ``poison=True`` (NaN fill) makes any violation loud, and the bitwise
    parity test is the proof.  Non-own-face compute is redundant by
    6/g -- a constant-factor cost on cheap diagnostic-grade ops, accepted
    at the <= 6-device ceiling and re-measured on the GPU ladder; the
    O(state) comm it replaces was the measured killer.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    # the *_impl bodies, NOT the public dispatchers (recursion guard,
    # same as the scalar ring)
    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface_impl,
        ext_vector_dgrid_sixface_impl,
    )

    depth = assert_ring_width_covers(tab, ring_width)
    ma = tab.n + 2 * tab.ng
    mb = ma + 1
    if grid == "D":
        shapes, flow = ((ma, mb), (mb, ma)), ext_vector_dgrid_sixface_impl
    elif grid == "C":
        shapes, flow = ((mb, ma), (ma, mb)), ext_vector_cgrid_sixface_impl
    else:
        raise ValueError(
            f"make_ring_ext_vector_sixface: grid {grid!r} (expected 'D' "
            f"or 'C')")
    (ax_name, ax_size), = ((n, s) for n, s in
                          zip(mesh.axis_names, mesh.devices.shape))
    if 6 % ax_size != 0:
        raise ValueError(
            f"mesh axis {ax_name!r} has {ax_size} devices; 6 faces "
            f"require a divisor (1, 2, 3, 6)")
    g = 6 // ax_size
    w = ring_width
    bidx = [np.flatnonzero(_border_mask_rect(s0, s1, w).ravel())
            for (s0, s1) in shapes]

    def _body(u_local, v_local):
        payloads = []
        for arr, (s0, s1), bi in zip((u_local, v_local), shapes, bidx):
            payloads.append(arr.reshape(g, s0 * s1)[:, bi])
        fused = jnp.concatenate(payloads, axis=1)         # (g, nb_u+nb_v)
        allb = jax.lax.all_gather(fused, ax_name)          # (d, g, nb)
        allb = allb.reshape(6, -1)
        me = jax.lax.axis_index(ax_name)
        rows = me * g + jnp.arange(g)
        fill = jnp.nan if poison else 0.0
        stacks = []
        off = 0
        for arr, (s0, s1), bi in zip((u_local, v_local), shapes, bidx):
            st = jnp.full((6, s0 * s1), fill, dtype=arr.dtype)
            st = st.at[:, bi].set(allb[:, off:off + bi.size])
            st = st.at[rows, :].set(arr.reshape(g, s0 * s1))
            stacks.append(st.reshape(6, s0, s1))
            off += bi.size
        u_out, v_out = flow(stacks[0], stacks[1], tab)
        return (jax.lax.dynamic_slice_in_dim(u_out, me * g, g, axis=0),
                jax.lax.dynamic_slice_in_dim(v_out, me * g, g, axis=0))

    return shard_map(_body, mesh=mesh,
                     in_specs=(P(ax_name), P(ax_name)),
                     out_specs=(P(ax_name), P(ax_name)),
                     check_rep=False), depth


def _border_mask_rect(m0: int, m1: int, w: int) -> np.ndarray:
    """Boolean (m0, m1) full-border mask, corner blocks included."""
    mask = np.zeros((m0, m1), dtype=bool)
    mask[:w, :] = True
    mask[-w:, :] = True
    mask[:, :w] = True
    mask[:, -w:] = True
    return mask


# ---------------------------------------------------------------------------
# k-batched ring exchanges (v2a) -- one collective per SITE, not per level
# ---------------------------------------------------------------------------
#
# The per-level ring closures above pay a fixed ~10 ms per call
# (shard_map region + all_gather launch + O(state) zero-stack rebuild +
# full table run); a step makes hundreds of them, which is the measured
# C192 2-GPU 50 s vs 6.6 s single / C384 190.9 s vs 23.5 s slowdown
# (jobs 9495469).  These `*_allk` factories take the whole (…, K)
# stack: own borders for ALL K ride in ONE payload, ONE all_gather, one
# sparse-stack rebuild with K on the trailing axis, and the certified
# per-level impl applied via ``jax.vmap`` over that trailing axis.  The
# static tables stay (6, m, m)-flat -- K is NEVER folded into the flat
# index space; vmap batches the gathers/scatters natively.

def make_ring_ext_scalar_sixface_allk(tab, stag: str, mesh, *,
                                      ring_width: int = 8,
                                      poison: bool = False):
    """k-batched twin of :func:`make_ring_ext_scalar_sixface`.

    Returns ``(fn, depth)`` with ``fn(f6k_local) -> f6k_local`` on the
    LOCAL block ``(6//d, m, m, K)`` of a global ``(6, m, m, K)`` array
    sharded ``P('face')`` (trailing K replicated per face, any static
    size -- callers may fold tracer x level into it).  Same census
    gate, same border mask, same zero/NaN fill as the per-level ring;
    the only new op is the vmap over the trailing axis.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    # the *_impl body, NOT the public dispatcher (recursion guard)
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_impl

    depth = assert_ring_width_covers(tab, ring_width)
    if stag == "A":
        m = tab.n + 2 * tab.ng
    elif stag == "B":
        m = tab.n + 2 * tab.ng + 1
    else:
        raise ValueError(
            f"make_ring_ext_scalar_sixface_allk: stag {stag!r} (the "
            f"certified scalar exchange supports 'A' and 'B' only -- "
            f"fail closed)")
    (ax_name, ax_size), = ((n, s) for n, s in
                          zip(mesh.axis_names, mesh.devices.shape))
    if 6 % ax_size != 0:
        raise ValueError(
            f"mesh axis {ax_name!r} has {ax_size} devices; 6 faces "
            f"require a divisor (1, 2, 3, 6)")
    g = 6 // ax_size
    mask = _border_mask(m, ring_width)
    bidx = np.flatnonzero(mask.ravel())

    def _body(f6k_local):
        # f6k_local: (g, m, m, K), this device's faces, certified order.
        kk = f6k_local.shape[-1]
        # 1. own borders for ALL K, one payload
        borders = f6k_local.reshape(g, m * m, kk)[:, bidx, :]  # (g, nb, K)
        # 2. ONE all_gather for the whole stack
        allb = jax.lax.all_gather(borders, ax_name)         # (d, g, nb, K)
        allb = allb.reshape(6, bidx.size, kk)
        # 3. reconstruct: borders everywhere, own faces in full; the
        #    census proves the tables never read the fill (poison=NaN
        #    makes any violation loud, as in the per-level ring).
        fill = jnp.nan if poison else 0.0
        stack = jnp.full((6, m * m, kk), fill, dtype=f6k_local.dtype)
        stack = stack.at[:, bidx, :].set(allb)
        me = jax.lax.axis_index(ax_name)
        rows = me * g + jnp.arange(g)
        stack = stack.at[rows].set(f6k_local.reshape(g, m * m, kk))
        # 4. certified tables verbatim, vmapped over trailing K
        out = jax.vmap(
            lambda f2: ext_scalar_sixface_impl(f2, tab, stag),
            in_axes=-1, out_axes=-1)(stack.reshape(6, m, m, kk))
        # 5. keep own faces
        return jax.lax.dynamic_slice_in_dim(out, me * g, g, axis=0)

    return shard_map(_body, mesh=mesh,
                     in_specs=P(ax_name), out_specs=P(ax_name),
                     check_rep=False), depth


def make_ring_ext_vector_sixface_allk(tab, grid: str, mesh, *,
                                      ring_width: int = 8,
                                      poison: bool = False):
    """k-batched twin of :func:`make_ring_ext_vector_sixface`.

    ``fn(u_local, v_local) -> (u_local, v_local)`` with trailing K on
    both components; both components' width-w borders for ALL K ride
    ONE fused all_gather, and the certified vector flow is vmapped over
    the trailing axis.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    # the *_impl bodies, NOT the public dispatchers (recursion guard)
    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface_impl,
        ext_vector_dgrid_sixface_impl,
    )

    depth = assert_ring_width_covers(tab, ring_width)
    ma = tab.n + 2 * tab.ng
    mb = ma + 1
    if grid == "D":
        shapes, flow = ((ma, mb), (mb, ma)), ext_vector_dgrid_sixface_impl
    elif grid == "C":
        shapes, flow = ((mb, ma), (ma, mb)), ext_vector_cgrid_sixface_impl
    else:
        raise ValueError(
            f"make_ring_ext_vector_sixface_allk: grid {grid!r} "
            f"(expected 'D' or 'C')")
    (ax_name, ax_size), = ((n, s) for n, s in
                          zip(mesh.axis_names, mesh.devices.shape))
    if 6 % ax_size != 0:
        raise ValueError(
            f"mesh axis {ax_name!r} has {ax_size} devices; 6 faces "
            f"require a divisor (1, 2, 3, 6)")
    g = 6 // ax_size
    bidx = [np.flatnonzero(_border_mask_rect(s0, s1, ring_width).ravel())
            for (s0, s1) in shapes]

    def _body(u_local, v_local):
        kk = u_local.shape[-1]
        payloads = []
        for arr, (s0, s1), bi in zip((u_local, v_local), shapes, bidx):
            payloads.append(arr.reshape(g, s0 * s1, kk)[:, bi, :])
        fused = jnp.concatenate(payloads, axis=1)   # (g, nb_u+nb_v, K)
        allb = jax.lax.all_gather(fused, ax_name)   # (d, g, nb, K)
        allb = allb.reshape(6, -1, kk)
        me = jax.lax.axis_index(ax_name)
        rows = me * g + jnp.arange(g)
        fill = jnp.nan if poison else 0.0
        stacks = []
        off = 0
        for arr, (s0, s1), bi in zip((u_local, v_local), shapes, bidx):
            st = jnp.full((6, s0 * s1, kk), fill, dtype=arr.dtype)
            st = st.at[:, bi, :].set(allb[:, off:off + bi.size, :])
            st = st.at[rows].set(arr.reshape(g, s0 * s1, kk))
            stacks.append(st.reshape(6, s0, s1, kk))
            off += bi.size
        u_out, v_out = jax.vmap(
            lambda u2, v2: flow(u2, v2, tab),
            in_axes=(-1, -1), out_axes=(-1, -1))(stacks[0], stacks[1])
        return (jax.lax.dynamic_slice_in_dim(u_out, me * g, g, axis=0),
                jax.lax.dynamic_slice_in_dim(v_out, me * g, g, axis=0))

    return shard_map(_body, mesh=mesh,
                     in_specs=(P(ax_name), P(ax_name)),
                     out_specs=(P(ax_name), P(ax_name)),
                     check_rep=False), depth


# ---------------------------------------------------------------------------
# ring_comm -- the bundle the halo dispatchers route through (M3 wiring)
# ---------------------------------------------------------------------------

class DuoRingComm:
    """The prebuilt ring exchanges for ONE ``(tab, mesh)`` pair.

    Rides on ``DuoHaloTables.ring_comm``; the tables are a STATIC jit
    argument, so this object is identity-hashable exactly like its host
    (two structurally identical bundles compile twice -- build one per
    context and reuse).  The closures are built up front (both scalar
    staggers and both vector grids) because ``stag``/grid are static at
    every call site; dispatch here is a trace-time dict lookup.
    """

    __slots__ = ("_scalar", "_dgrid", "_cgrid",
                 "_scalar_allk", "_dgrid_allk", "_cgrid_allk",
                 "ring_width", "depth", "mesh")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):                              # pragma: no cover
        return (f"DuoRingComm(ring_width={self.ring_width}, "
                f"depth={self.depth})")

    def ext_scalar(self, f6, stag: str):
        try:
            fn = self._scalar[stag]
        except KeyError:
            raise ValueError(
                f"DuoRingComm.ext_scalar: stagger {stag!r} not "
                f"implemented (the certified scalar exchange supports "
                f"'A' and 'B' only)") from None
        return fn(f6)

    def ext_vector_dgrid(self, u6, v6):
        return self._dgrid(u6, v6)

    def ext_vector_cgrid(self, uc6, vc6):
        return self._cgrid(uc6, vc6)

    def ext_scalar_allk(self, f6k, stag: str):
        try:
            fn = self._scalar_allk[stag]
        except KeyError:
            raise ValueError(
                f"DuoRingComm.ext_scalar_allk: stagger {stag!r} not "
                f"implemented (the certified scalar exchange supports "
                f"'A' and 'B' only)") from None
        return fn(f6k)

    def ext_vector_dgrid_allk(self, u6k, v6k):
        return self._dgrid_allk(u6k, v6k)

    def ext_vector_cgrid_allk(self, uc6k, vc6k):
        return self._cgrid_allk(uc6k, vc6k)


def build_ring_comm(tab, mesh, *, ring_width: int = 8) -> DuoRingComm:
    """Build the :class:`DuoRingComm` for ``tab`` on ``mesh``.

    Attach it as ``tab.ring_comm`` to route every step-side exchange
    (the three public ``ext_*_sixface`` dispatchers in
    ``fv3_duo_halos``) through the O(halo) shard_map ring.  Each
    ``make_ring_*`` factory re-runs the census gate against ``tab``, so
    a ring the tables outread is refused here, at build time.
    """
    import jax

    # Contract enforcement (codex MINOR): the advertised knob is a
    # single-'face'-axis mesh; accepting any axis name would let a
    # differently-purposed mesh slip in silently.
    if tuple(mesh.axis_names) != ("face",):
        raise ValueError(
            f"build_ring_comm: mesh axes {tuple(mesh.axis_names)} != "
            f"('face',) -- the ring runs on a single face axis only")
    # Multi-node fail-closed (GLM): a mesh built from local_devices() --
    # or before jax.distributed.initialize() -- is a valid SUBMESH of the
    # global world, and every process would then ring over its own local
    # faces with halos never crossing processes: deterministic wrong
    # answers, no crash. Under multiple processes, refuse any mesh that
    # does not span the world. (Single-process partial meshes -- e.g. 2
    # of 6 host devices in the parity tests -- are legitimate.)
    if jax.process_count() > 1 and mesh.size != jax.device_count():
        raise ValueError(
            f"build_ring_comm: mesh spans {mesh.size} of "
            f"{jax.device_count()} global devices under "
            f"{jax.process_count()} processes -- a partial-device ring "
            f"would exchange within each process only (halos never "
            f"cross processes). Build the mesh from jax.devices() "
            f"AFTER jax.distributed.initialize().")
    rc = DuoRingComm()
    rc.mesh = mesh          # for the caller-side identity assert
    scalar = {}
    scalar_allk = {}
    depth = -1
    for stag in ("A", "B"):
        scalar[stag], depth = make_ring_ext_scalar_sixface(
            tab, stag, mesh, ring_width=ring_width)
        scalar_allk[stag], _ = make_ring_ext_scalar_sixface_allk(
            tab, stag, mesh, ring_width=ring_width)
    rc._scalar = scalar
    rc._scalar_allk = scalar_allk
    rc._dgrid, _ = make_ring_ext_vector_sixface(
        tab, "D", mesh, ring_width=ring_width)
    rc._cgrid, _ = make_ring_ext_vector_sixface(
        tab, "C", mesh, ring_width=ring_width)
    rc._dgrid_allk, _ = make_ring_ext_vector_sixface_allk(
        tab, "D", mesh, ring_width=ring_width)
    rc._cgrid_allk, _ = make_ring_ext_vector_sixface_allk(
        tab, "C", mesh, ring_width=ring_width)
    rc.ring_width = int(ring_width)
    rc.depth = int(depth)
    return rc


# ---------------------------------------------------------------------------
# v2: build-time TABLE SPLIT + tiled exchange (M1: A scalar; M2: B/C/D)
# ---------------------------------------------------------------------------
#
# The ring above caps at 6 devices (whole faces).  The tiled port
# (docs/performance/scaling/duo_tiled_port_scope.md, pillars P1/P2) runs on a
# ('face', 'tile_i', 'tile_j') = (6, kt, kt) mesh.
#
# BLOCKED tile layout (P1, the cdgrid tiled lane's layout restated for the
# duo shapes -- tiled_production_cdgrid.py's expand_corners_to_blocks):
# with nl = (n + 2*ng) // kt, an axis of flat extent L = kt*nl + e
# (e = 0 cell axis, 1 node axis, 2 geo-lattice axis) is stored as kt
# per-tile windows of extent nl + e -- tile t holds flat rows
# [t*nl, t*nl + nl + e); adjacent tiles DUPLICATE the shared e rows, and
# every write to a duplicated slot is executed in EVERY storing tile from
# bit-identical inputs, so the copies never diverge (the cdgrid lane's
# coherence argument).  Blocked global shape: (6, kt*(nl+e0), kt*(nl+e1));
# kt=1 degenerates to the flat array.  :func:`to_blocked` /
# :func:`from_blocked` convert at the lane boundary (pure slice+concat).
#
# The split is INTEGER-ONLY: every certified table row is assigned to the
# tile(s) that STORE its destination cell, and every source index is
# resolved to a tile that stores it, using ONLY `_decode_fij` (the inverse
# of `_FlatLayout.idx`) and integer division.  The certified per-row
# WEIGHTS are BIT-COPIED (pure row indexing, hashed into the build) --
# never re-derived.  A row whose sources straddle several sender tiles
# becomes one ORDERED multi-source gather: the row's gathered values keep
# the certified source order, and stencil sums stay unrolled left-to-right
# (`_dot_static`'s order), never a dot/tree-reduce or scatter-accumulate.
#
# Remote sources are served by the NEAREST storing tile of the source
# (deterministic: clip the reader's tile coordinate into the source's
# storing range; canonical min(i//nl, kt-1) tile across faces).  With the
# census gate (table reach < nl) this keeps every within-face transfer
# inside the 8-WAY tile neighborhood -- checked mechanically, refused
# otherwise.  Cross-face sources route by index to the canonical storing
# tile of the certified cube neighbor.
#
# SIGN CHANNEL (M2): a phase entry may be ``(table, sign)`` -- the static
# caller ``mySign`` of the certified call site.  The runtime forms the
# effective weight EXACTLY as ``_apply_scatter`` does
# (``base * np.where(sgn_pow == 1, sign, 1.0)`` on static numpy, folded at
# trace time), and the build asserts byte-equality of that executed
# effective weight against the same expression on the certified records.
# Signed sites in the duo flow: the C/D strip exchanges carry base = +-1
# (the mpp NE-vector map derivative; sgn_pow = 0), the C/D vector-exchange
# corner scatters are applied with sign = -1.0 over sgn_pow in {0, 1}, and
# the standalone ``fill_corners_{dgrid,cgrid,agrid_pair}`` take the
# caller's sign.  All other tables are unsigned (base = 1, sgn_pow = 0).
#
# Exchange semantics per phase (the flat lane's own snapshot phases):
# senders pack values from their CURRENT tile state, one ppermute round
# per edge-coloring color (each device sends <= 1 and receives <= 1
# message per round -- ppermute's contract).  Receive buffers are
# NaN-filled scratch every phase; the build-time schedule proof guarantees
# every real slot is overwritten before use.
#
# Pad values (send index 0, place slot = scratch, dst = scratch,
# weight base=1/sgn_pow=0) are inert constants: padded rows write only the
# dropped scratch slot and padded transfers land only in the dropped
# scratch slot.  Fixed no-op fillers, not derived arithmetic.

_TILE_AXES = ("face", "tile_i", "tile_j")
_E_MAX = 2      # blocked per-axis extra: cell 0, node 1, geo lattice 2
_TILE_SPEC = None


def tile_spec():
    """THE blocked-array PartitionSpec ``P(*_TILE_AXES)`` -- ONE object,
    the single source of truth (GLM M3 close-out): every tiled
    shard_map's ``in_specs``/``out_specs`` and every caller placing a
    blocked array (tests, probes) take it from here, so the spec a
    blocked array is placed with can never silently drift from the spec
    the exchange partitions it by.  Built lazily (jax is not imported at
    module load) and cached, so identity (``is``) is assertable."""
    global _TILE_SPEC
    if _TILE_SPEC is None:
        from jax.sharding import PartitionSpec as P

        _TILE_SPEC = P(*_TILE_AXES)
    return _TILE_SPEC


class _TiledOp:
    """One certified table's rows for ALL tiles, ordered-gather form."""

    __slots__ = ("kind", "name", "sign", "dst_loc", "src_pos", "base",
                 "sgn_pow", "w", "srcx_pos", "wx", "srcy_pos", "wy")


class _TiledPhase:
    """One snapshot phase: its comm schedule + its split tables."""

    __slots__ = ("perms", "send_idx", "place_idx", "n_rbuf", "ops")


class DuoTiledSplit:
    """The built split for one pipeline at one ``kt`` (host numpy only).

    Two receipts: ``weights_sha256`` digests the CERTIFIED source-table
    weight bytes (bit-copy provenance), ``split_sha256`` digests the
    EXECUTED arrays (every per-op weight/index table + the comm
    schedule) as built -- :meth:`executed_sha256` recomputes it from the
    current arrays, so a post-build corruption is detectable.

    ``tiles`` carries the per-buffer tile extents ``(nl+e0, nl+e1)`` --
    the split OWNS its halo width (GLM M1 Q2): the runtime refuses any
    caller array whose blocked shape does not match ``kt * tiles``.
    """

    __slots__ = ("name", "kt", "nl", "m", "ndev", "depth",
                 "weights_sha256", "split_sha256", "phases",
                 "n_within_edges", "n_cross_edges",
                 "tiles", "sizes", "offs", "scr")

    def __repr__(self):                              # pragma: no cover
        return (f"DuoTiledSplit({self.name!r}, kt={self.kt}, "
                f"nl={self.nl}, phases={len(self.phases)}, "
                f"edges={self.n_within_edges}+{self.n_cross_edges}x)")

    def executed_sha256(self) -> str:
        """Digest of the EXECUTED split arrays (weights + indices +
        schedule), recomputed from the live numpy buffers.

        codex (close-out MAJOR): raw ``tobytes`` alone is not an array
        identity -- a reshape/reinterpret changes runtime semantics
        without changing the bytes.  Every array is hashed FRAMED as
        ``(name, dtype, shape, bytes)``, and the structural scalars
        (``n_rbuf``, phase count, tile extents, per-op sign) are folded
        in."""
        import hashlib

        h = hashlib.sha256()

        def _upd(name: str, a) -> None:
            a = np.asarray(a)
            h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())

        h.update(f"kt={self.kt}|nl={self.nl}|tiles={self.tiles}|".encode())
        h.update(f"phases={len(self.phases)}|".encode())
        for ph in self.phases:
            h.update(f"n_rbuf={ph.n_rbuf}|".encode())
            h.update(repr(ph.perms).encode())
            _upd("send_idx", ph.send_idx)
            _upd("place_idx", ph.place_idx)
            for op in ph.ops:
                h.update(f"{op.kind}|sign={op.sign!r}|".encode())
                _upd("dst_loc", op.dst_loc)
                if op.kind == "scatter":
                    arrs = (("src_pos", op.src_pos), ("base", op.base),
                            ("sgn_pow", op.sgn_pow))
                elif op.kind == "stencil":
                    arrs = (("src_pos", op.src_pos), ("w", op.w))
                else:
                    arrs = (("srcx_pos", op.srcx_pos), ("wx", op.wx),
                            ("srcy_pos", op.srcy_pos), ("wy", op.wy))
                for nm, a in arrs:
                    _upd(nm, a)
        return h.hexdigest()


def _op_kind(op) -> str:
    """Classify a certified table op by its record attributes (duck-typed:
    importing the private `_Scatter`/`_Stencil` classes cross-module is
    forbidden; the census above uses the same convention)."""
    if hasattr(op, "srcx"):
        return "stencil_pair"
    if hasattr(op, "w"):
        return "stencil"
    if hasattr(op, "base"):
        return "scatter"
    raise TypeError(
        f"duo tiled split: table op {type(op).__name__} has neither "
        f"'srcx', 'w' nor 'base' -- a new op family the split does not "
        f"understand; extend it before tiling (fail closed, never guess)")


def _color_edges(edges):
    """Greedy edge coloring into ppermute rounds: within a round every
    device sends at most one and receives at most one message (the
    ppermute contract).  Deterministic (sorted input, first-fit)."""
    rounds = []
    for s, d in sorted(edges):
        for snd, rcv, lst in rounds:
            if s not in snd and d not in rcv:
                snd.add(s)
                rcv.add(d)
                lst.append((s, d))
                break
        else:
            rounds.append(({s}, {d}, [(s, d)]))
    return [lst for _, _, lst in rounds]


def _norm_phase(tables):
    """Normalize a phase's entries to ``(op, static_sign)`` pairs."""
    out = []
    for ent in tables:
        if isinstance(ent, tuple) and len(ent) == 2 \
                and isinstance(ent[1], (int, float)):
            out.append((ent[0], float(ent[1])))
        else:
            out.append((ent, 1.0))
    return out


def _build_tiled_split(tab, lay, phase_tables, kt, name) -> DuoTiledSplit:
    """INTEGER-ONLY split of a sequence of snapshot phases over a
    (6, kt, kt) BLOCKED tiling of ``lay`` (any buffer count; per-axis
    blocked extras e in 0..2 -- see the section comment).

    Build-time completeness proof: every table row is claimed by EXACTLY
    the set of tiles that store its destination (claim count == storage
    multiplicity, asserted), every source resolves to a storing tile
    (decode raises out-of-range; the serving choice is checked to store
    what it sends), and every receive-buffer slot of every device is
    written by exactly one comm round entry.  Weights are bit-copied and
    hashed (``weights_sha256``); for signed scatters the EXECUTED
    effective weight is byte-compared against
    ``base * np.where(sgn_pow == 1, sign, 1.0)`` -- the exact expression
    ``_apply_scatter`` forms.  Raises on ragged tiling, on a tile
    narrower than the measured cross-face census, and on any within-face
    transfer outside the 8-way tile neighborhood.
    """
    import hashlib

    from legoesm.grids.fv3_native_halos import neighbor_tiles

    kt = int(kt)
    if kt < 1:
        raise ValueError(f"{name}: kt={kt} must be >= 1")
    ma = tab.n + 2 * tab.ng
    if ma % kt != 0:
        raise ValueError(
            f"{name}: face extent m={ma} does not tile evenly into "
            f"kt={kt} tiles (ragged tiles are a v-next decision made "
            f"explicitly, never silently)")
    nl = ma // kt
    tiles = []
    for (m0, m1) in lay.shapes:
        e0, e1 = m0 - kt * nl, m1 - kt * nl
        if not (0 <= e0 <= _E_MAX and 0 <= e1 <= _E_MAX):
            raise ValueError(
                f"{name}: buffer extent ({m0}, {m1}) is not kt*nl+e "
                f"with e in 0..{_E_MAX} for nl={nl}, kt={kt} -- ragged "
                f"tiles are a v-next decision made explicitly, never "
                f"silently")
        tiles.append((nl + e0, nl + e1))
    sizes = [t0 * t1 for (t0, t1) in tiles]
    offs = [int(o) for o in np.cumsum([0] + sizes)[:-1]]
    scr = int(sum(sizes))
    ndev = 6 * kt * kt
    # census width gate (G1c): a tile must be wider than the deepest
    # cross-face table read, so every cross-face source lands in the
    # tile row adjacent to its face edge, and every within-face reach
    # stays inside the 8-way neighborhood (reach < nl).
    # assert_ring_width_covers is the certified gate (nl must EXCEED the
    # census); it censuses ALL table families -- a superset of any
    # pipeline split here, so the guard is conservative.  Scope P5's v1
    # floor nl >= 8 follows from the measured census 7.
    depth = assert_ring_width_covers(tab, nl)
    _self_test(lay)          # the decode must invert idx before it tiles

    e0_arr = np.asarray([t0 - nl for (t0, _) in tiles])
    e1_arr = np.asarray([t1 - nl for (_, t1) in tiles])
    t1_arr = np.asarray([t1 for (_, t1) in tiles])
    off_arr = np.asarray(offs)

    def _rng(idx, e):
        """Storing-tile range [lo, hi] of flat index ``idx`` along one
        blocked axis with extra ``e`` (tile t stores [t*nl, t*nl+nl+e))."""
        hi = np.minimum(idx // nl, kt - 1)
        lo = np.maximum(-((-(idx - (nl + e - 1))) // nl), 0)
        return lo, hi

    def _dec_full(flat):
        k, f, i, j = _decode_fij(lay, np.asarray(flat, dtype=np.int64))
        lo_i, hi_i = _rng(i, e0_arr[k])
        lo_j, hi_j = _rng(j, e1_arr[k])
        return k, f, i, j, lo_i, hi_i, lo_j, hi_j

    def _local(k, i, j, ti, tj):
        """Local flat position of (buffer k, i, j) inside tile (ti, tj)'s
        blocked window (caller guarantees the tile stores it)."""
        return off_arr[k] + (i - ti * nl) * t1_arr[k] + (j - tj * nl)

    adj = {f: {nf - 1 for nf in neighbor_tiles(f + 1)} for f in range(6)}

    hasher = hashlib.sha256()
    phases = []
    n_within = 0
    n_cross = 0
    for raw_tables in phase_tables:
        tables = _norm_phase(raw_tables)
        # ---- decode rows once (integers only) -------------------------
        decoded = []
        for op, sign in tables:
            kind = _op_kind(op)
            dinfo = _dec_full(op.dst)
            dk, dfc, di, dj, dlo_i, dhi_i, dlo_j, dhi_j = dinfo
            mult = (dhi_i - dlo_i + 1) * (dhi_j - dlo_j + 1)
            nrow = int(np.asarray(op.dst).shape[0])
            srcs = {}
            names = ("srcx", "srcy") if kind == "stencil_pair" else ("src",)
            for a in names:
                sarr = np.asarray(getattr(op, a),
                                  dtype=np.int64).reshape(nrow, -1)
                srcs[a] = (sarr,) + _dec_full(sarr)
            decoded.append((kind, op, sign, nrow, dinfo, mult, srcs))

        # ---- pass 1: per-dev row claims + remote needs ----------------
        need = [dict() for _ in range(ndev)]    # dev -> {sdev: set(flat)}
        claims = []
        for kind, op, sign, nrow, dinfo, mult, srcs in decoded:
            dk, dfc, di, dj, dlo_i, dhi_i, dlo_j, dhi_j = dinfo
            per_dev_rows = []
            cnt = np.zeros(nrow, dtype=np.int64)
            for dev in range(ndev):
                fd = dev // (kt * kt)
                tid, tjd = (dev // kt) % kt, dev % kt
                cl = ((dfc == fd) & (dlo_i <= tid) & (tid <= dhi_i)
                      & (dlo_j <= tjd) & (tjd <= dhi_j))
                r = np.flatnonzero(cl)
                per_dev_rows.append(r)
                cnt[r] += 1
                if r.size == 0:
                    continue
                for a in srcs:
                    (sarr, sk, sfc, si, sj,
                     slo_i, shi_i, slo_j, shi_j) = srcs[a]
                    sa = sarr[r]
                    sfa, sia, sja = sfc[r], si[r], sj[r]
                    lia, hia = slo_i[r], shi_i[r]
                    lja, hja = slo_j[r], shi_j[r]
                    stored = ((sfa == fd) & (lia <= tid) & (tid <= hia)
                              & (lja <= tjd) & (tjd <= hja))
                    rem = ~stored
                    if not rem.any():
                        continue
                    same_face = sfa == fd
                    serve_ti = np.where(same_face, np.clip(tid, lia, hia),
                                        np.minimum(sia // nl, kt - 1))
                    serve_tj = np.where(same_face, np.clip(tjd, lja, hja),
                                        np.minimum(sja // nl, kt - 1))
                    wf = rem & same_face
                    if wf.any():
                        cheb = np.maximum(np.abs(serve_ti[wf] - tid),
                                          np.abs(serve_tj[wf] - tjd))
                        if int(cheb.max()) > 1:
                            raise ValueError(
                                f"{name}: a within-face source of "
                                f"{op.name} sits {int(cheb.max())} tiles "
                                f"from its reader -- outside the 8-way "
                                f"tile neighborhood; tile nl={nl} is too "
                                f"small for this table")
                    xf = rem & ~same_face
                    if xf.any():
                        # cross-face legality: the source face must be a
                        # certified cube neighbor of the destination's
                        # (neighbor_tiles is the oracle's own adjacency;
                        # the three faces at a cube corner are pairwise
                        # edge-adjacent, so corner third-face routing is
                        # covered).  A source on a non-adjacent face is a
                        # corrupted table, never scheduled silently.
                        for sf_ in np.unique(sfa[xf]):
                            if int(sf_) not in adj[fd]:
                                raise ValueError(
                                    f"{name}: {op.name} reads face "
                                    f"{int(sf_)} from face {fd}, which "
                                    f"are not cube-adjacent -- corrupted "
                                    f"table, refusing to schedule it")
                    sdev = (sfa * kt + serve_ti) * kt + serve_tj
                    for sd_, fl_ in zip(sdev[rem], sa[rem]):
                        need[dev].setdefault(int(sd_),
                                             set()).add(int(fl_))
            claims.append(per_dev_rows)
            # completeness: claim count == storage multiplicity of dst
            if not np.array_equal(cnt, mult):
                bad = int(np.flatnonzero(cnt != mult)[0])
                raise AssertionError(
                    f"{name}: row {bad} of {op.name} claimed by "
                    f"{int(cnt[bad])} tiles while its destination is "
                    f"stored by {int(mult[bad])} -- completeness proof "
                    f"failed")

        # ---- receive-buffer layout (deterministic slot order) ---------
        slot = [dict() for _ in range(ndev)]
        counts = np.zeros(ndev, dtype=np.int64)
        for dev in range(ndev):
            s = 0
            for sdv in sorted(need[dev]):
                for sfl in sorted(need[dev][sdv]):
                    slot[dev][(sdv, sfl)] = s
                    s += 1
            counts[dev] = s
        n_rbuf = int(counts.max()) if ndev else 0

        # ---- comm rounds (edge coloring -> ppermute schedule) ---------
        edges = sorted({(sdv, dev)
                        for dev in range(ndev) for sdv in need[dev]})
        for s_, d_ in edges:
            if s_ // (kt * kt) == d_ // (kt * kt):
                n_within += 1
            else:
                n_cross += 1
        rounds = _color_edges(edges)
        n_rounds = len(rounds)
        lmax = 1
        for lst in rounds:
            for s_, d_ in lst:
                lmax = max(lmax, len(need[d_][s_]))
        send_idx = np.zeros((ndev, max(n_rounds, 1), lmax), dtype=np.int32)
        place_idx = np.full((ndev, max(n_rounds, 1), lmax), n_rbuf,
                            dtype=np.int32)
        perms = []
        for c, lst in enumerate(rounds):
            perms.append(tuple((int(s_), int(d_)) for s_, d_ in lst))
            for s_, d_ in lst:
                flats = sorted(need[d_][s_])
                fk, ff, fi, fj, lo_i, hi_i, lo_j, hi_j = _dec_full(
                    np.asarray(flats, dtype=np.int64))
                sfd = s_ // (kt * kt)
                sti, stj = (s_ // kt) % kt, s_ % kt
                # a sender MUST store what it serves (mechanical)
                if ((ff != sfd).any() or (lo_i > sti).any()
                        or (sti > hi_i).any() or (lo_j > stj).any()
                        or (stj > hi_j).any()):
                    raise AssertionError(
                        f"{name}: comm schedule asks device {s_} to send "
                        f"a value it does not store")
                send_idx[s_, c, :len(flats)] = _local(fk, fi, fj, sti, stj)
                place_idx[d_, c, :len(flats)] = [slot[d_][(s_, f_)]
                                                 for f_ in flats]
        # schedule completeness: every rbuf slot of every device is
        # written by EXACTLY one round entry (pad = scratch slot n_rbuf)
        for dev in range(ndev):
            real = np.sort(place_idx[dev][place_idx[dev] != n_rbuf])
            if not np.array_equal(real, np.arange(counts[dev])):
                raise AssertionError(
                    f"{name}: comm schedule incomplete on device {dev} "
                    f"-- receive slots not covered exactly once")

        # ---- split each table's rows by storing tile ------------------
        ops_out = []
        for (kind, op, sign, nrow, dinfo, mult, srcs), per_dev_rows in zip(
                decoded, claims):
            dk, dfc, di, dj = dinfo[:4]
            rmax = max(1, max((r.size for r in per_dev_rows), default=1))

            def pos_of(dev, tid, tjd, fd, a, r):
                (sarr, sk, sfc, si, sj,
                 slo_i, shi_i, slo_j, shi_j) = srcs[a]
                sa, ska = sarr[r], sk[r]
                sfa, sia, sja = sfc[r], si[r], sj[r]
                lia, hia = slo_i[r], shi_i[r]
                lja, hja = slo_j[r], shi_j[r]
                stored = ((sfa == fd) & (lia <= tid) & (tid <= hia)
                          & (lja <= tjd) & (tjd <= hja))
                out = np.empty(sa.shape, dtype=np.int64)
                out[stored] = _local(ska[stored], sia[stored], sja[stored],
                                     tid, tjd)
                rem = ~stored
                if rem.any():
                    same_face = sfa == fd
                    serve_ti = np.where(same_face, np.clip(tid, lia, hia),
                                        np.minimum(sia // nl, kt - 1))
                    serve_tj = np.where(same_face, np.clip(tjd, lja, hja),
                                        np.minimum(sja // nl, kt - 1))
                    sdev = (sfa * kt + serve_ti) * kt + serve_tj
                    out[rem] = [scr + slot[dev][(int(sd_), int(fl_))]
                                for sd_, fl_ in zip(sdev[rem], sa[rem])]
                return out

            t = _TiledOp()
            t.kind, t.name, t.sign = kind, op.name, float(sign)
            t.dst_loc = np.full((ndev, rmax), scr, dtype=np.int32)
            if kind == "scatter":
                t.src_pos = np.zeros((ndev, rmax), dtype=np.int32)
                # the certified weight RECORD is carried verbatim
                # (bit-copied rows below); the effective weight is formed
                # at trace time by the runtime, exactly where and how
                # _apply_scatter forms it -- zero FP arithmetic here
                t.base = np.ones((ndev, rmax), dtype=np.float64)
                t.sgn_pow = np.zeros((ndev, rmax), dtype=np.int8)
            elif kind == "stencil":
                lw = int(np.asarray(op.src).shape[1])
                t.src_pos = np.zeros((ndev, rmax, lw), dtype=np.int32)
                t.w = np.zeros((ndev, rmax, lw), dtype=np.float64)
            else:
                lw = int(np.asarray(op.srcx).shape[1])
                t.srcx_pos = np.zeros((ndev, rmax, lw), dtype=np.int32)
                t.srcy_pos = np.zeros((ndev, rmax, lw), dtype=np.int32)
                t.wx = np.zeros((ndev, rmax, lw), dtype=np.float64)
                t.wy = np.zeros((ndev, rmax, lw), dtype=np.float64)
            for dev in range(ndev):
                r = per_dev_rows[dev]
                if r.size == 0:
                    continue
                fd = dev // (kt * kt)
                tid, tjd = (dev // kt) % kt, dev % kt
                nr = r.size
                t.dst_loc[dev, :nr] = _local(dk[r], di[r], dj[r], tid, tjd)
                if kind == "scatter":
                    t.src_pos[dev, :nr] = pos_of(dev, tid, tjd, fd,
                                                 "src", r)[:, 0]
                    t.base[dev, :nr] = np.asarray(op.base)[r]
                    t.sgn_pow[dev, :nr] = np.asarray(op.sgn_pow)[r]
                elif kind == "stencil":
                    t.src_pos[dev, :nr] = pos_of(dev, tid, tjd, fd,
                                                 "src", r)
                    t.w[dev, :nr] = np.asarray(op.w)[r]  # bit-copied rows
                else:
                    t.srcx_pos[dev, :nr] = pos_of(dev, tid, tjd, fd,
                                                  "srcx", r)
                    t.srcy_pos[dev, :nr] = pos_of(dev, tid, tjd, fd,
                                                  "srcy", r)
                    t.wx[dev, :nr] = np.asarray(op.wx)[r]  # bit-copied
                    t.wy[dev, :nr] = np.asarray(op.wy)[r]  # bit-copied
            # bit-copy PROOF (codex MAJOR): every executed weight slice
            # -- on EVERY storing tile, duplicates included -- must be
            # BYTE-equal to the certified rows it claims; together with
            # the claim-count assert above this covers every row exactly
            # once per storing tile AND bit-identity of the copies.
            if kind == "scatter":
                pairs = (("base", op.base), ("sgn_pow", op.sgn_pow))
            elif kind == "stencil":
                pairs = (("w", op.w),)
            else:
                pairs = (("wx", op.wx), ("wy", op.wy))
            for attr, source in pairs:
                source = np.asarray(source)
                for dev in range(ndev):
                    r = per_dev_rows[dev]
                    if r.size and (getattr(t, attr)[dev, :r.size].tobytes()
                                   != source[r].tobytes()):
                        raise AssertionError(
                            f"{name}: executed {attr} of {op.name} is not "
                            f"a bit copy of the certified table "
                            f"(device {dev})")
            if kind == "scatter":
                # the EXECUTED effective weight must equal what
                # _apply_scatter forms: base * where(sgn_pow == 1, sign, 1)
                # -- byte equality, per storing tile (review-binding)
                ref_eff = (np.asarray(op.base, dtype=np.float64)
                           * np.where(np.asarray(op.sgn_pow) == 1,
                                      float(sign), 1.0))
                for dev in range(ndev):
                    r = per_dev_rows[dev]
                    if not r.size:
                        continue
                    exe = (t.base[dev, :r.size]
                           * np.where(t.sgn_pow[dev, :r.size] == 1,
                                      float(t.sign), 1.0))
                    if exe.tobytes() != ref_eff[r].tobytes():
                        raise AssertionError(
                            f"{name}: executed effective weight of "
                            f"{op.name} diverges from _apply_scatter's "
                            f"base*sign^sgn_pow (device {dev})")
            # bit-copy receipt: the certified weight bytes, hashed
            if kind == "scatter":
                hasher.update(np.asarray(op.base).tobytes())
                hasher.update(np.asarray(op.sgn_pow).tobytes())
            elif kind == "stencil":
                hasher.update(np.asarray(op.w).tobytes())
            else:
                hasher.update(np.asarray(op.wx).tobytes())
                hasher.update(np.asarray(op.wy).tobytes())
            ops_out.append(t)

        ph = _TiledPhase()
        ph.perms = perms
        ph.send_idx = send_idx
        ph.place_idx = place_idx
        ph.n_rbuf = n_rbuf
        ph.ops = ops_out
        phases.append(ph)

    sp = DuoTiledSplit()
    sp.name, sp.kt, sp.nl, sp.m, sp.ndev = name, kt, nl, ma, ndev
    sp.depth = int(depth)
    sp.tiles = tuple(tiles)
    sp.sizes = tuple(sizes)
    sp.offs = tuple(offs)
    sp.scr = scr
    sp.weights_sha256 = hasher.hexdigest()
    sp.phases = phases
    sp.split_sha256 = sp.executed_sha256()
    sp.n_within_edges = n_within
    sp.n_cross_edges = n_cross
    return sp


# ---------------------------------------------------------------------------
# blocked layout converters (lane boundary; pure slice+concat, host numpy)
# ---------------------------------------------------------------------------

def to_blocked(x, kt: int, nl: int):
    """Flat ``(6, L0, L1, ...)`` -> BLOCKED ``(6, kt*(nl+e0), kt*(nl+e1),
    ...)`` with ``L = kt*nl + e``, ``e in 0..2`` (see the section
    comment).  Adjacent tiles DUPLICATE the shared e rows/cols.  kt=1 is
    the identity.  Call once at the lane boundary, never per step."""
    x = np.asarray(x)
    kt, nl = int(kt), int(nl)
    if x.ndim < 3 or x.shape[0] != 6:
        raise ValueError(
            f"to_blocked: expected (6, L0, L1, ...); got {x.shape}")
    e0, e1 = x.shape[1] - kt * nl, x.shape[2] - kt * nl
    if not (0 <= e0 <= _E_MAX and 0 <= e1 <= _E_MAX):
        raise ValueError(
            f"to_blocked: extents {x.shape[1:3]} are not kt*nl+e with e "
            f"in 0..{_E_MAX} for kt={kt}, nl={nl}")
    x = np.concatenate([x[:, i * nl:i * nl + nl + e0] for i in range(kt)],
                       axis=1)
    return np.concatenate([x[:, :, j * nl:j * nl + nl + e1]
                           for j in range(kt)], axis=2)


def from_blocked(xb, kt: int, nl: int):
    """Inverse of :func:`to_blocked` (canonical copies: each tile's first
    nl rows/cols; the last tile contributes its full window)."""
    xb = np.asarray(xb)
    kt, nl = int(kt), int(nl)
    if xb.ndim < 3 or xb.shape[0] != 6 or xb.shape[1] % kt \
            or xb.shape[2] % kt:
        raise ValueError(
            f"from_blocked: expected (6, kt*t0, kt*t1, ...); got "
            f"{xb.shape} for kt={kt}")
    t0, t1 = xb.shape[1] // kt, xb.shape[2] // kt
    if not (0 <= t0 - nl <= _E_MAX and 0 <= t1 - nl <= _E_MAX):
        raise ValueError(
            f"from_blocked: tile extents ({t0}, {t1}) are not nl+e with "
            f"e in 0..{_E_MAX} for nl={nl}")
    xb = np.concatenate(
        [xb[:, i * t0:i * t0 + (t0 if i == kt - 1 else nl)]
         for i in range(kt)], axis=1)
    return np.concatenate(
        [xb[:, :, j * t1:j * t1 + (t1 if j == kt - 1 else nl)]
         for j in range(kt)], axis=2)


def to_blocked_traced(x, kt: int, nl: int):
    """Traced (jnp) twin of :func:`to_blocked` -- same slices, same
    duplication of the shared e rows/cols, usable inside a jitted step.

    kt=1 returns ``x`` UNCHANGED (no graph perturbation on the kt=1
    bridge -- G0's bit-identity must not ride on an identity reshape
    surviving XLA).  Used by the tile-comm flat interface (M3): the full
    step's state stays FLAT until M4 tiles the kernels, so each tile-arm
    firing converts at its own boundary.  Pure slice+concat: every
    output element is a copy of an input element (bit-preserving)."""
    import jax.numpy as jnp

    kt, nl = int(kt), int(nl)
    if kt == 1:
        return x
    if x.ndim < 3 or x.shape[0] != 6:
        raise ValueError(
            f"to_blocked_traced: expected (6, L0, L1, ...); got {x.shape}")
    e0, e1 = x.shape[1] - kt * nl, x.shape[2] - kt * nl
    if not (0 <= e0 <= _E_MAX and 0 <= e1 <= _E_MAX):
        raise ValueError(
            f"to_blocked_traced: extents {x.shape[1:3]} are not kt*nl+e "
            f"with e in 0..{_E_MAX} for kt={kt}, nl={nl}")
    x = jnp.concatenate([x[:, i * nl:i * nl + nl + e0] for i in range(kt)],
                        axis=1)
    return jnp.concatenate([x[:, :, j * nl:j * nl + nl + e1]
                            for j in range(kt)], axis=2)


def from_blocked_traced(xb, kt: int, nl: int):
    """Traced (jnp) twin of :func:`from_blocked` (canonical copies --
    exactly the flat lane's slices); kt=1 returns ``xb`` unchanged."""
    import jax.numpy as jnp

    kt, nl = int(kt), int(nl)
    if kt == 1:
        return xb
    if xb.ndim < 3 or xb.shape[0] != 6 or xb.shape[1] % kt \
            or xb.shape[2] % kt:
        raise ValueError(
            f"from_blocked_traced: expected (6, kt*t0, kt*t1, ...); got "
            f"{xb.shape} for kt={kt}")
    t0, t1 = xb.shape[1] // kt, xb.shape[2] // kt
    if not (0 <= t0 - nl <= _E_MAX and 0 <= t1 - nl <= _E_MAX):
        raise ValueError(
            f"from_blocked_traced: tile extents ({t0}, {t1}) are not "
            f"nl+e with e in 0..{_E_MAX} for nl={nl}")
    xb = jnp.concatenate(
        [xb[:, i * t0:i * t0 + (t0 if i == kt - 1 else nl)]
         for i in range(kt)], axis=1)
    return jnp.concatenate(
        [xb[:, :, j * t1:j * t1 + (t1 if j == kt - 1 else nl)]
         for j in range(kt)], axis=2)


# ---------------------------------------------------------------------------
# tiled runtime -- one shard_map body per split (or composed pipeline)
# ---------------------------------------------------------------------------

def _apply_phase_ops(ph, me, work, cur, scr):
    """Apply one phase's split tables for device ``me`` (traced scalar
    under shard_map).

    ``work`` = state + landed receive buffer, ``cur`` = state; snapshot
    semantics: every op reads ``work``, writes into a fresh copy."""
    import jax.numpy as jnp

    kk = cur.shape[1]
    new = jnp.concatenate([cur, jnp.zeros((1, kk), cur.dtype)])
    for op in ph.ops:
        dl = jnp.asarray(op.dst_loc)[me]
        if op.kind == "scatter":
            # mirrors _apply_scatter VERBATIM: the effective weight is a
            # trace-time static numpy expression over the BIT-COPIED
            # base/sgn_pow records and the phase's static caller sign;
            # padded rows carry (base=1, sgn_pow=0) no-ops.
            eff = op.base * np.where(op.sgn_pow == 1, float(op.sign), 1.0)
            v = work[jnp.asarray(op.src_pos)[me]]
            vals = v * jnp.asarray(eff, cur.dtype)[me][:, None]
        elif op.kind == "stencil":
            # mirrors _apply_stencil/_dot_static: unrolled ordered sum,
            # left to right -- never a dot
            v = work[jnp.asarray(op.src_pos)[me]]
            w = jnp.asarray(op.w, cur.dtype)[me]
            vals = w[:, 0, None] * v[:, 0]
            for lev in range(1, w.shape[1]):
                vals = vals + w[:, lev, None] * v[:, lev]
        else:                            # stencil_pair
            vx_v = work[jnp.asarray(op.srcx_pos)[me]]
            wx = jnp.asarray(op.wx, cur.dtype)[me]
            vx = wx[:, 0, None] * vx_v[:, 0]
            for lev in range(1, wx.shape[1]):
                vx = vx + wx[:, lev, None] * vx_v[:, lev]
            vy_v = work[jnp.asarray(op.srcy_pos)[me]]
            wy = jnp.asarray(op.wy, cur.dtype)[me]
            vy = wy[:, 0, None] * vy_v[:, 0]
            for lev in range(1, wy.shape[1]):
                vy = vy + wy[:, lev, None] * vy_v[:, lev]
            vals = 0.5 * (vx + vy)
        new = new.at[dl].set(vals)
    return new[:scr]


def _run_phases(split, me, cur):
    """Run a split's phases on the local state ``cur`` (scr, K) inside a
    shard_map body.  Receive buffers are NaN-filled scratch every phase
    (an out-of-schedule read is loud, never a plausible zero); the
    build-time schedule proof guarantees every real slot is overwritten
    before use."""
    import jax
    import jax.numpy as jnp

    scr = split.scr
    kk = cur.shape[1]
    for ph in split.phases:
        nb = ph.n_rbuf
        rbuf = jnp.full((nb + 1, kk), jnp.nan, dtype=cur.dtype)
        if ph.perms:
            sidx = jnp.asarray(ph.send_idx)[me]
            pidx = jnp.asarray(ph.place_idx)[me]
            for c, perm in enumerate(ph.perms):
                recv = jax.lax.ppermute(cur[sidx[c]], _TILE_AXES, perm)
                rbuf = rbuf.at[pidx[c]].set(recv)
        work = jnp.concatenate([cur, rbuf[:nb]]) if nb else cur
        cur = _apply_phase_ops(ph, me, work, cur, scr)
    return cur


def _check_tile_mesh(mesh, kt, fname):
    import jax

    if tuple(mesh.axis_names) != _TILE_AXES:
        raise ValueError(
            f"{fname}: mesh axes {tuple(mesh.axis_names)} != {_TILE_AXES} "
            f"-- the tiled exchange runs on the (face, tile_i, tile_j) "
            f"mesh only")
    want = (6, kt, kt)
    if tuple(mesh.devices.shape) != want:
        raise ValueError(
            f"{fname}: mesh.devices.shape {tuple(mesh.devices.shape)} != "
            f"{want} -- the mesh tiling must match the split's kt (a "
            f"mismatch would exchange halos on a different tiling)")
    if jax.process_count() > 1 and mesh.size != jax.device_count():
        raise ValueError(
            f"{fname}: mesh spans {mesh.size} of {jax.device_count()} "
            f"global devices under {jax.process_count()} processes -- a "
            f"partial-device mesh would exchange within each process "
            f"only. Build the mesh from jax.devices() AFTER "
            f"jax.distributed.initialize().")


def _check_blocked_args(fname, arrs, blocked):
    """GLM M1 Q2 (width through the seam): the split CARRIES its required
    halo width -- per-buffer blocked tile extents kt*(nl+e) -- and any
    caller array with a narrower (or otherwise different) pad refuses
    here rather than exchanging on a wrong window."""
    seen = None
    trailing = None
    for a, bs in zip(arrs, blocked):
        if a.ndim < 3 or tuple(a.shape[:3]) != bs:
            raise ValueError(
                f"{fname}: expected blocked leading shape {bs} (the "
                f"split carries its required width, kt*(nl+e) per axis); "
                f"got {tuple(a.shape)} -- a narrower pad would starve "
                f"the exchange. Convert with to_blocked().")
        dt = a.dtype
        if dt not in (np.float32, np.float64):
            raise TypeError(
                f"{fname}: dtype {dt} (float32/float64 only)")
        if seen is None:
            seen = dt
        elif dt != seen:
            raise TypeError(
                f"{fname}: MIXED float dtypes ({seen} vs {dt}); the "
                f"exchange must be precision-uniform")
        if trailing is None:
            trailing = a.shape[3:]
        elif a.shape[3:] != trailing:
            raise ValueError(
                f"{fname}: trailing (K) axes differ across components "
                f"({trailing} vs {a.shape[3:]})")


def tiled_me():
    """This device's window index ``(face*kt + tile_i)*kt + tile_j`` inside a
    shard_map over the ``(face, tile_i, tile_j)`` mesh."""
    import jax

    kt = jax.lax.axis_size("tile_i")
    return ((jax.lax.axis_index("face") * kt
             + jax.lax.axis_index("tile_i")) * kt
            + jax.lax.axis_index("tile_j"))


def tiled_split_body(split: DuoTiledSplit):
    """The PER-DEVICE body of a split's tiled exchange: ``body(*locs) ->
    locs`` on this device's blocked tiles ``(t0, t1[, K...])``, one per
    layout buffer, with the split's ppermute schedule inside.  Public so
    a caller running its OWN shard_map over the same mesh (the window
    arm, fv3_duo_window_spmd) can fire the certified exchange on blocks
    it holds locally, without a nested shard_map or a global reshape.
    :func:`make_tiled_from_split` wraps exactly this body."""
    import jax.numpy as jnp

    tiles = split.tiles
    nbuf = len(tiles)
    sizes = split.sizes

    def _body(*locs):
        me = tiled_me()
        parts = [loc.reshape(t0 * t1, -1)
                 for loc, (t0, t1) in zip(locs, tiles)]
        cur = jnp.concatenate(parts) if nbuf > 1 else parts[0]
        cur = _run_phases(split, me, cur)
        outs = []
        off = 0
        for loc, sz in zip(locs, sizes):
            outs.append(cur[off:off + sz].reshape(loc.shape))
            off += sz
        return tuple(outs) if nbuf > 1 else outs[0]

    return _body


def make_tiled_from_split(split: DuoTiledSplit, mesh):
    """Build the shard_map'd tiled exchange for ``split`` on ``mesh``.

    ``mesh`` axes must be ``('face', 'tile_i', 'tile_j')`` with device
    shape ``(6, kt, kt)`` matching the split (the `_check_tiled_mesh`
    guard pattern, tiled_production_cdgrid.py:44 -- restated here, not
    imported: it is a private symbol of another module and its extents
    are the cdgrid lane's, not this layout's).

    Returns ``fn(*buffers) -> buffers`` on BLOCKED global arrays -- one
    per layout buffer, shapes ``(6, kt*t0, kt*t1[, K...])`` with
    ``(t0, t1) = split.tiles[k]`` (for a single e=0 square buffer this
    IS the flat ``(6, m, m)`` array, the M1 interface).  Trailing axes
    are batched natively (the ``*_allk`` arms).  The fn carries
    ``.split``, ``.block_shapes`` and ``.min_tile`` and REFUSES inputs
    whose blocked shapes disagree (width rides the split, not the
    caller).
    """
    from jax.experimental.shard_map import shard_map

    _check_tile_mesh(mesh, split.kt, "make_tiled_from_split")
    kt = split.kt
    tiles = split.tiles
    nbuf = len(tiles)
    blocked = tuple((6, kt * t0, kt * t1) for (t0, t1) in tiles)
    _body = tiled_split_body(split)

    spec = tile_spec()
    sm = shard_map(_body, mesh=mesh,
                   in_specs=(spec,) * nbuf if nbuf > 1 else spec,
                   out_specs=(spec,) * nbuf if nbuf > 1 else spec,
                   check_rep=False)

    def fn(*arrs):
        if len(arrs) != nbuf:
            raise ValueError(
                f"tiled exchange {split.name!r}: takes {nbuf} buffer "
                f"array(s), got {len(arrs)}")
        _check_blocked_args(f"tiled exchange {split.name!r}", arrs,
                            blocked)
        return sm(*arrs)

    fn.split = split
    fn.block_shapes = blocked
    fn.min_tile = split.depth + 1
    fn.spec = spec
    return fn


# ---------------------------------------------------------------------------
# scalar pipelines (A: M1; B: M2) + standalone table twins
# ---------------------------------------------------------------------------

def build_tiled_ext_scalar_a_split(tab, kt) -> DuoTiledSplit:
    """M1 split of the certified A-grid scalar ext pipeline.

    The phases are exactly the flat lane's own snapshot sequence
    (``ext_scalar_sixface_impl`` at ``stag="A"``): strip scatter ->
    corner scatter -> k2e ring stencil -> corner-Lagrange fill (whose
    directional + diagonal tables read ONE shared snapshot, as the flat
    lane does).
    """
    st_dir, st_dia = tab.corner["a3"]
    return _build_tiled_split(
        tab, tab.lay_a,
        [[tab.ex_a_strip], [tab.ex_a_corner], [tab.k2e_a],
         [st_dir, st_dia]],
        kt, "tiled_ext_scalar_sixface[A]")


def build_tiled_ext_scalar_b_split(tab, kt) -> DuoTiledSplit:
    """M2 split of the certified B-grid scalar ext pipeline.

    Flat sequence (``ext_scalar_sixface_impl`` at ``stag="B"``): strip
    scatter (B has NO corner scatter -- the NumPy lane leaves
    corner-diagonal regions untouched) -> k2e ring stencil -> b3 corner
    Lagrange.  B is node-staggered: the blocked interface is
    ``(6, kt*(nl+1), kt*(nl+1))`` (kt=1 == flat).
    """
    st_dir, st_dia = tab.corner["b3"]
    return _build_tiled_split(
        tab, tab.lay_b,
        [[tab.ex_b_strip], [tab.k2e_b], [st_dir, st_dia]],
        kt, "tiled_ext_scalar_sixface[B]")


class _StencilRec:
    """A stencil table record built HERE (dst / src / w / name), the same
    duck-typed surface ``_op_kind`` classifies as ``"stencil"``."""

    __slots__ = ("dst", "src", "w", "name")

    def __init__(self, dst, src, w, name):
        self.dst = np.asarray(dst, dtype=np.int32)
        self.src = np.asarray(src, dtype=np.int32)
        self.w = np.asarray(w, dtype=np.float64)
        self.name = name


def _remap_flat(lay_from, lay_to, flat, off):
    """Flat indices of ``lay_from`` -> the same (k, face, i+off, j+off)
    cells in ``lay_to`` (vectorised ``_FlatLayout.idx``)."""
    k, f, i, j = _decode_fij(lay_from, np.asarray(flat, dtype=np.int64))
    out = np.empty_like(k)
    for kk, (m0, m1) in enumerate(lay_to.shapes):
        m = k == kk
        ii, jj = i[m] + off, j[m] + off
        if m.any() and (ii.min() < 0 or ii.max() >= m0 or jj.min() < 0
                        or jj.max() >= m1):
            raise IndexError(f"padded remap: subscript outside array {kk} "
                             f"of shape (6, {m0}, {m1})")
        out[m] = lay_to.bases[kk] + (f[m] * m0 + ii) * m1 + jj
    return out.reshape(np.shape(flat))


def blend_as_stencil(bl, lay_from, lay_to, off, name):
    """A duo barrier blend ``out[dst] = 0.5*(in[dst] + sign*in[src])`` as
    a two-source STENCIL row ``0.5*in[dst] + (0.5*sign)*in[src]`` on the
    padded layout ``lay_to`` (indices shifted by ``off``).

    EXACT, not approximate: multiplying by 0.5 is exact for every
    NORMAL double (halving never overflows; it only rounds when the result
    is subnormal, i.e. |x| < 2^-1021), and rounding-to-nearest commutes
    with a power-of-two scaling as long as no operand or result is
    subnormal, so ``fl(0.5*fl(a + s*b)) == fl(fl(0.5*a) + fl(0.5*s*b))``
    for s = +-1.  The cancellation corner a == s*b gives +0 on both sides
    (RNE); any other pair of operands >= 1e-1 differs by at least one
    ulp, ~2^-56, far above the subnormal range -- the barrier operands
    are O(1e-1..1e5) winds and fluxes, and the argument, not the random
    sampling, closes that corner (GLM 2026-09-04).  FMA must be off (it
    is pinned on the tiled lane).  The tiled stencil runtime forms exactly
    the right-hand side (ordered left-to-right).  Pinned by
    ``test_barrier_blend_as_stencil_is_bitwise`` on the real tables.  The
    0.5*sign weights are formed here from the certified ``sign`` record
    (exact for +-1)."""
    dst = _remap_flat(lay_from, lay_to, bl.dst, off)
    src = _remap_flat(lay_from, lay_to, bl.src, off)
    sign = np.asarray(bl.sign, dtype=np.float64)
    if not np.all(np.abs(sign) == 1.0):
        raise ValueError(f"{name}: blend sign record is not +-1 "
                         f"(0.5*sign would round)")
    w = np.stack([np.full(dst.shape, 0.5), 0.5 * sign], axis=1)
    return _StencilRec(dst, np.stack([dst, src], axis=1), w, name)


def build_tiled_barrier_split(tab, kt, kind: str) -> DuoTiledSplit:
    """M4b-B: the edge-blend BARRIERS on the blocked layout.

    ``kind``: ``"bgrid"`` (barrier 2, ``tab.avg_b`` on the (npx, npx) B
    pair) or ``"cgrid"`` (barrier 1, ``tab.avg_c`` on the (npx, n)/(n, npx)
    flux pair; the allflux barrier is the same table batched over the
    selected slots).  The barrier operands are COMPUTE-domain arrays,
    which the blocked layout (``kt*nl + e`` extents) cannot hold, so the
    split is built on their PADDED twins (compute index + ng) and the
    window arm embeds the operands before firing.  The blend becomes an
    exact two-source stencil (:func:`blend_as_stencil`)."""
    from legoesm.grids.fv3_duo_halos import FlatLayout

    n, ng = int(tab.n), int(tab.ng)
    ma, mb = n + 2 * ng, n + 2 * ng + 1
    if kind == "bgrid":
        lay_from, lay_to, bl = tab.lay_bb, FlatLayout((mb, mb), (mb, mb)), tab.avg_b
    elif kind == "cgrid":
        lay_from, lay_to, bl = tab.lay_fx, FlatLayout((mb, ma), (ma, mb)), tab.avg_c
    else:
        raise ValueError(f"build_tiled_barrier_split: kind {kind!r} "
                         f"(expected 'bgrid' or 'cgrid' -- fail closed)")
    st = blend_as_stencil(bl, lay_from, lay_to, ng, f"barrier_{kind}")
    return _build_tiled_split(tab, lay_to, [[st]], kt,
                              f"tiled_barrier_{kind}")


def build_tiled_fill_corners_agrid_split(tab, kt, axis: str) -> DuoTiledSplit:
    """Split of a standalone A-grid ``fill_corners`` twin (single phase)."""
    if axis == "x":
        table = tab.fc_agrid_x
    elif axis == "y":
        table = tab.fc_agrid_y
    else:
        raise ValueError(
            f"build_tiled_fill_corners_agrid_split: axis {axis!r} "
            f"(expected 'x' or 'y' -- fail closed)")
    return _build_tiled_split(tab, tab.lay_a, [[table]], kt,
                              f"tiled_fill_corners_agrid_{axis}")


def build_tiled_k2e_split(tab, kt, stag: str,
                          ring: str = "stepper") -> DuoTiledSplit:
    """Standalone split of one k2e ring table (mirrors the flat
    ``k2e_remap_halo_rings`` dispatch, refusals included)."""
    if stag == "A" and ring == "stepper":
        lay, st = tab.lay_a, tab.k2e_a
    elif stag == "A" and ring == "geo":
        lay, st = tab.lay_a4, tab.k2e_a4
    elif stag == "B" and ring == "stepper":
        lay, st = tab.lay_b, tab.k2e_b
    elif stag in ("CX", "CY", "DX", "DY"):
        raise ValueError(
            f"build_tiled_k2e_split: stagger {stag!r} is not part of the "
            f"duo ext flow (upstream ext_scalar supports (0,0) and (1,1) "
            f"only; the C/D table families belong to the rejected "
            f"position-only vector remap)")
    else:
        raise ValueError(
            f"build_tiled_k2e_split: stagger {stag!r} / ring {ring!r} "
            f"unsupported")
    return _build_tiled_split(tab, lay, [[st]], kt,
                              f"tiled_k2e_remap_halo_rings[{stag},{ring}]")


def build_tiled_corner_lagrange_split(tab, kt, key: str) -> DuoTiledSplit:
    """Standalone split of one corner-Lagrange operator family (single
    phase: directional + diagonal read one shared snapshot, as flat)."""
    if key not in tab.corner:
        raise ValueError(
            f"build_tiled_corner_lagrange_split: key {key!r} (expected "
            f"one of {sorted(tab.corner)})")
    ma = tab.n + 2 * tab.ng
    mb = ma + 1
    m4 = tab.n + 2 * tab.ngp
    shp = {"a4": (m4, m4), "a3": (ma, ma), "b3": (mb, mb),
           "du3": (ma, mb), "dv3": (mb, ma)}[key]
    # single-array layout of this operator family, built via the layout
    # CLASS of an existing table (type(...) -- no private cross-module
    # import), with the exact shape the flat builder used
    lay = type(tab.lay_a)(shp)
    st_dir, st_dia = tab.corner[key]
    return _build_tiled_split(tab, lay, [[st_dir, st_dia]], kt,
                              f"tiled_corner_lagrange[{key}]")


def build_tiled_fill_corners_split(tab, kt, kind: str,
                                   sign: float = 1.0) -> DuoTiledSplit:
    """Standalone split of one ``fill_corners_*`` table.  ``sign`` is the
    caller's static ``mySign`` (baked into the split like the jit policy
    bakes it into the compile); it multiplies only the sgn_pow=1 records,
    exactly as ``_apply_scatter`` forms it."""
    reg = {"bgrid_x": (tab.lay_b, tab.fc_bgrid_x),
           "agrid_x": (tab.lay_a, tab.fc_agrid_x),
           "agrid_y": (tab.lay_a, tab.fc_agrid_y),
           "dgrid": (tab.lay_d, tab.fc_dgrid),
           "cgrid": (tab.lay_c, tab.fc_cgrid),
           "agrid_pair": (tab.lay_ap, tab.fc_agrid_pair)}
    if kind not in reg:
        raise ValueError(
            f"build_tiled_fill_corners_split: kind {kind!r} (expected one "
            f"of {sorted(reg)} -- fail closed)")
    lay, table = reg[kind]
    return _build_tiled_split(
        tab, lay, [[(table, float(sign))]], kt,
        f"tiled_fill_corners_{kind}[sign={float(sign)}]")


def make_tiled_ext_scalar_sixface(tab, mesh, stag: str = "A"):
    """Tiled twin of ``ext_scalar_sixface_impl`` -- stag "A" (M1, flat
    interface: e=0) or "B" (M2, blocked node interface).

    Returns ``(fn, split)``; ``kt`` is derived from the mesh shape and
    validated end to end by :func:`make_tiled_from_split`.
    """
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_ext_scalar_sixface: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    kt = int(mesh.devices.shape[1])
    if stag == "A":
        split = build_tiled_ext_scalar_a_split(tab, kt)
    elif stag == "B":
        split = build_tiled_ext_scalar_b_split(tab, kt)
    else:
        raise ValueError(
            f"make_tiled_ext_scalar_sixface: stag {stag!r} not "
            f"implemented (upstream ext_scalar supports (0,0) and (1,1) "
            f"only -- fail closed)")
    return make_tiled_from_split(split, mesh), split


def make_tiled_fill_corners_agrid(tab, mesh, axis: str):
    """Tiled twin of ``fill_corners_agrid_x``/``_y``; returns ``(fn, split)``."""
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_fill_corners_agrid: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    split = build_tiled_fill_corners_agrid_split(
        tab, int(mesh.devices.shape[1]), axis)
    return make_tiled_from_split(split, mesh), split


def make_tiled_k2e_remap_halo_rings(tab, mesh, stag: str,
                                    ring: str = "stepper"):
    """Tiled twin of ``k2e_remap_halo_rings``; returns ``(fn, split)``
    on the stagger's blocked array."""
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_k2e_remap_halo_rings: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    split = build_tiled_k2e_split(tab, int(mesh.devices.shape[1]), stag,
                                  ring)
    return make_tiled_from_split(split, mesh), split


def make_tiled_corner_lagrange_fill(tab, mesh, key: str):
    """Tiled twin of ``corner_lagrange_fill``; returns ``(fn, split)``."""
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_corner_lagrange_fill: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    split = build_tiled_corner_lagrange_split(
        tab, int(mesh.devices.shape[1]), key)
    return make_tiled_from_split(split, mesh), split


def make_tiled_fill_corners(tab, mesh, kind: str, sign: float = 1.0):
    """Tiled twin of the six ``fill_corners_*`` flat fns; the pair kinds
    (dgrid/cgrid/agrid_pair) take two blocked arrays and the caller's
    static ``mySign``.  Returns ``(fn, split)``."""
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_fill_corners: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    split = build_tiled_fill_corners_split(
        tab, int(mesh.devices.shape[1]), kind, sign)
    return make_tiled_from_split(split, mesh), split


def _allk_wrap(fn, nargs, fname):
    """The *_allk arm of a tiled fn: same schedule, trailing K batched in
    ONE firing (the runtime is K-native); refuses a K-less call so the
    dispatcher seam stays per-name explicit."""
    def allk(*arrs):
        for a in arrs[:nargs]:
            if a.ndim < 4:
                raise ValueError(
                    f"{fname}: expected a trailing K axis "
                    f"(got ndim={a.ndim}); use the per-level fn for "
                    f"single-level exchanges")
        return fn(*arrs)

    for at in ("split", "block_shapes", "min_tile", "spec"):
        if hasattr(fn, at):
            setattr(allk, at, getattr(fn, at))
    return allk


def make_tiled_ext_scalar_sixface_allk(tab, mesh, stag: str = "A"):
    """k-batched tile arm of ``ext_scalar_sixface_allk``; returns
    ``(fn, split)`` with ``fn(f6k_blocked)`` (trailing K required)."""
    fn, split = make_tiled_ext_scalar_sixface(tab, mesh, stag)
    return _allk_wrap(fn, 1, "make_tiled_ext_scalar_sixface_allk"), split


# ---------------------------------------------------------------------------
# composed C/D vector pipelines (M2) -- table phases + certified dense
# stages per tile
# ---------------------------------------------------------------------------
#
# The flat composed flows (ext_vector_{d,c}grid_sixface_impl) interleave
# TABLE stages (split exactly like the scalar pipelines) with three DENSE
# stages that are NOT tables and are therefore NOT re-derived as one
# (their weights are FP arithmetic over grid metrics -- composing them
# into a sparse table would re-derive certified numerics, forbidden):
#
#   * c2l (c2l_ord2_face / c2l_ord2_cgrid_face): reads only the +-1
#     staggered node the BLOCKED window already stores, so each tile
#     computes its own (nl, nl) cell block locally -- no communication;
#     the flat NaN window (valid Fortran is-1..ie+1 only) is reproduced
#     with a per-tile static mask (jnp.where selecting the SAME computed
#     value inside the window and the SAME NaN constant outside;
#     bit-identical, and gradients are not a gate on this parity lane).
#   * pack_p1: a pure index EMBEDDING (geo = stepper + (ngp-ng)); it IS
#     expressed as a synthetic weight-1.0 scatter phase so its cross-tile
#     data movement rides the proven comm machinery -- integer remap
#     only, nothing derived.
#   * a2d/a2c projection: pointwise in (i, j) given the 2-point edge
#     averages the blocked geo window (e=2) already stores; per-tile
#     local, mirroring the flat _dot_static 3-term unroll order.
#
# Per-tile METRIC blocks (dx/dy/amat/vlon4/vlat4 and the pre-sliced
# es4/ew4 projections) are BIT-COPIES of the certified tab arrays,
# sliced by pure indexing at build and hashed into the bundle receipt.
# Their shapes are ASSERTED at build against the layout contract (never
# inferred).

class _SynthScatter:
    """Duck-typed weight-1.0 scatter record for a pure index embedding
    (pack_p1).  Integer remap only -- base is exactly 1.0, sgn_pow 0;
    there is no derived arithmetic to bit-copy."""

    __slots__ = ("dst", "src", "base", "sgn_pow", "name")

    def __init__(self, dst, src, name):
        self.dst = np.asarray(dst, dtype=np.int64)
        self.src = np.asarray(src, dtype=np.int64)
        self.base = np.ones(self.dst.size, dtype=np.float64)
        self.sgn_pow = np.zeros(self.dst.size, dtype=np.int8)
        self.name = name


class _ShiftedOp:
    """Integer-remapped view of a certified table op: the SAME weight
    arrays (by reference -- the bit-copy proof compares against these
    very buffers), indices shifted by a flat-layout base offset so a
    single-array table can run as a phase of a multi-buffer layout."""

    def __init__(self, op, delta: int):
        self.name = op.name
        for a in ("dst", "src", "srcx", "srcy"):
            if hasattr(op, a):
                setattr(self, a,
                        np.asarray(getattr(op, a), dtype=np.int64)
                        + int(delta))
        for a in ("base", "sgn_pow", "w", "wx", "wy"):
            if hasattr(op, a):
                setattr(self, a, getattr(op, a))


def _pack_p1_op(tab, lay_pack) -> _SynthScatter:
    """Replay of ``pack_p1``'s embedding (geo slot (i+d, j+d) <- stepper
    (i, j), d = ngp - ng) as flat indices of the 2-buffer
    [geo | stepper] layout.  Slots outside the embedded block stay on
    the NaN-initialized geo buffer, exactly the flat NaN fill."""
    d = tab.ngp - tab.ng
    ma = tab.n + 2 * tab.ng
    dst, src = [], []
    for face in range(6):
        for i in range(ma):
            for j in range(ma):
                dst.append(lay_pack.idx(0, face, i + d, j + d))
                src.append(lay_pack.idx(1, face, i, j))
    return _SynthScatter(dst, src, "pack_p1.embed")


def _tile_blocks(arr, kt, nl, ax_i, ax_j, name):
    """Per-device blocked windows of a (6, ...) metric array -- pure
    slicing (bit-copies), device order (face*kt + ti)*kt + tj."""
    arr = np.asarray(arr)
    if arr.shape[0] != 6:
        raise ValueError(f"{name}: leading axis must be the 6 faces")
    L0, L1 = arr.shape[ax_i], arr.shape[ax_j]
    e0, e1 = L0 - kt * nl, L1 - kt * nl
    if not (0 <= e0 <= _E_MAX and 0 <= e1 <= _E_MAX):
        raise ValueError(
            f"{name}: tiled axes ({L0}, {L1}) are not kt*nl+e with e in "
            f"0..{_E_MAX} for kt={kt}, nl={nl}")
    out = []
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                sl = [slice(None)] * arr.ndim
                sl[0] = f
                sl[ax_i] = slice(ti * nl, ti * nl + nl + e0)
                sl[ax_j] = slice(tj * nl, tj * nl + nl + e1)
                out.append(arr[tuple(sl)])
    return np.stack(out, axis=0)


class DuoTiledVectorSplits:
    """The built composed C/D pipeline for one ``(tab, kt, grid)``:
    four sub-splits (exchange, geo lattice incl. the pack embedding,
    strip write-back, final corner fills) + the per-tile dense-stage
    metric blocks and the c2l window mask.

    ``weights_sha256`` chains the sub-splits' certified-weight receipts
    and the framed metric bit-copies; :meth:`executed_sha256` digests
    everything as built (framed), recomputable."""

    __slots__ = ("grid", "kt", "nl", "ndev", "depth", "n", "ng", "ngp",
                 "vector_corner", "split_ex", "split_geo", "split_wr",
                 "split_cn", "dense", "masks", "weights_sha256",
                 "split_sha256")

    def __repr__(self):                              # pragma: no cover
        return (f"DuoTiledVectorSplits({self.grid!r}, kt={self.kt}, "
                f"nl={self.nl})")

    def _sub_iter(self):
        return (("ex", self.split_ex), ("geo", self.split_geo),
                ("wr", self.split_wr), ("cn", self.split_cn))

    def executed_sha256(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(f"grid={self.grid}|kt={self.kt}|nl={self.nl}|"
                 f"corner={self.vector_corner}|".encode())
        for nm, sub in self._sub_iter():
            h.update(f"{nm}|".encode())
            if sub is not None:
                h.update(sub.executed_sha256().encode())
        for group, d in (("dense", self.dense), ("masks", self.masks)):
            for nm in sorted(d):
                a = np.asarray(d[nm])
                h.update(
                    f"{group}.{nm}|{a.dtype.str}|{a.shape}|".encode())
                h.update(a.tobytes())
        return h.hexdigest()


def build_tiled_ext_vector_splits(tab, kt, grid: str) -> DuoTiledVectorSplits:
    """Build the composed C/D vector pipeline splits (host numpy only).

    Phase structure mirrors the flat ``ext_vector_*_sixface_impl``
    stage order EXACTLY (P3 timing: nothing fused across stage
    boundaries, nothing applied early):

      1. exchange: [strips] then [corner scatter @ sign=-1.0]
         (D: DGRID_NE tables; C: CGRID_NE tables -- base carries the mpp
         NE-vector orientation +-1, sign=-1 rides sgn_pow rows only);
      2. dense c2l per tile (local; blocked windows carry the +-1 node);
      3. geo lattice on the [geo | stepper] layout, u and v batched on
         the trailing axis: [pack embed], [a4 strips], [a4 corners],
         [k2e a4], [a4 Lagrange dir+dia];
      4. dense a2d/a2c projection per tile (local);
      5. write-back: [wr_d]/[wr_c] on the 4-buffer layout;
      6. final corner Lagrange per component at its own staggering
         (D: u<-du3, v<-dv3; C: uc<-dv3, vc<-du3), skipped when
         ``tab.vector_corner == "a2d"`` exactly like the flat impl.
    """
    import hashlib

    kt = int(kt)
    if grid == "D":
        lay_pair = tab.lay_d
        strip, cornr = tab.ex_d_strip, tab.ex_d_corner
        wr_lay, wr = tab.lay_wd, tab.wr_d
        cn_keys = ("du3", "dv3")
    elif grid == "C":
        lay_pair = tab.lay_c
        strip, cornr = tab.ex_c_strip, tab.ex_c_corner
        wr_lay, wr = tab.lay_wc, tab.wr_c
        cn_keys = ("dv3", "du3")
    else:
        raise ValueError(
            f"build_tiled_ext_vector_splits: grid {grid!r} (expected 'D' "
            f"or 'C' -- fail closed)")
    ma = tab.n + 2 * tab.ng
    mb = ma + 1
    m4 = tab.n + 2 * tab.ngp

    split_ex = _build_tiled_split(
        tab, lay_pair, [[strip], [(cornr, -1.0)]], kt,
        f"tiled_ext_vector[{grid}].exchange")
    nl = split_ex.nl

    # geo stage: [geo | stepper-A] 2-buffer layout; layout CLASS via
    # type(...) (no private cross-module import), exact flat shapes
    lay_pack = type(tab.lay_a)((m4, m4), (ma, ma))
    a4_dir, a4_dia = tab.corner["a4"]
    split_geo = _build_tiled_split(
        tab, lay_pack,
        [[_pack_p1_op(tab, lay_pack)], [tab.ex_a4_strip],
         [tab.ex_a4_corner], [tab.k2e_a4], [a4_dir, a4_dia]],
        kt, f"tiled_ext_vector[{grid}].geo")

    split_wr = _build_tiled_split(
        tab, wr_lay, [[wr]], kt, f"tiled_ext_vector[{grid}].writeback")

    split_cn = None
    if tab.vector_corner == "lagrange":
        u_dir, u_dia = tab.corner[cn_keys[0]]
        v_dir, v_dia = tab.corner[cn_keys[1]]
        off = lay_pair.bases[1]
        split_cn = _build_tiled_split(
            tab, lay_pair,
            [[u_dir, u_dia],
             [_ShiftedOp(v_dir, off), _ShiftedOp(v_dia, off)]],
            kt, f"tiled_ext_vector[{grid}].corners")

    # dense-stage metric blocks -- shapes ASSERTED against the layout
    # contract (docstring items 2/4 of fv3_duo_halos), never inferred
    exp = {"dx": (6, ma, mb), "dy": (6, mb, ma),
           "amat": (6, 4, ma, ma), "vlon4": (6, m4, m4, 3),
           "vlat4": (6, m4, m4, 3), "es4": (6, m4, m4 + 1, 3, 2),
           "ew4": (6, m4 + 1, m4, 3, 2)}
    for nm, want in exp.items():
        got = tuple(np.asarray(getattr(tab, nm)).shape)
        if got != want:
            raise ValueError(
                f"build_tiled_ext_vector_splits: tab.{nm} has shape "
                f"{got}, layout contract says {want} -- refusing to "
                f"slice per-tile metrics off an unexpected layout")
    if grid == "D":
        pu = np.asarray(tab.es4)[:, :, 1:-1, :, 0]      # (6, m4, m4-1, 3)
        pv = np.asarray(tab.ew4)[:, 1:-1, :, :, 1]      # (6, m4-1, m4, 3)
    else:
        pu = np.asarray(tab.ew4)[:, 1:-1, :, :, 0]      # (6, m4-1, m4, 3)
        pv = np.asarray(tab.es4)[:, :, 1:-1, :, 1]      # (6, m4, m4-1, 3)
    dense = {
        "dx": _tile_blocks(tab.dx, kt, nl, 1, 2, "dx"),
        "dy": _tile_blocks(tab.dy, kt, nl, 1, 2, "dy"),
        "amat": _tile_blocks(tab.amat, kt, nl, 2, 3, "amat"),
        "vlon4": _tile_blocks(tab.vlon4, kt, nl, 1, 2, "vlon4"),
        "vlat4": _tile_blocks(tab.vlat4, kt, nl, 1, 2, "vlat4"),
        "pu": _tile_blocks(pu, kt, nl, 1, 2, "pu"),
        "pv": _tile_blocks(pv, kt, nl, 1, 2, "pv"),
    }
    # c2l valid window (Fortran is-1..ie+1 both axes) as a per-tile mask
    cell = np.zeros((ma, ma), dtype=bool)
    cell[tab.c2l_s:tab.c2l_e + 1, tab.c2l_s:tab.c2l_e + 1] = True
    masks = {"c2l": _tile_blocks(cell[None].repeat(6, axis=0), kt, nl,
                                 1, 2, "c2l_mask")}

    b = DuoTiledVectorSplits()
    b.grid, b.kt, b.nl, b.ndev = grid, kt, nl, split_ex.ndev
    b.depth = split_ex.depth
    b.n, b.ng, b.ngp = tab.n, tab.ng, tab.ngp
    b.vector_corner = tab.vector_corner
    b.split_ex, b.split_geo = split_ex, split_geo
    b.split_wr, b.split_cn = split_wr, split_cn
    b.dense, b.masks = dense, masks
    h = hashlib.sha256()
    for nm, sub in b._sub_iter():
        h.update(f"{nm}|".encode())
        if sub is not None:
            h.update(sub.weights_sha256.encode())
    for nm in sorted(dense):
        a = dense[nm]
        h.update(f"dense.{nm}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    b.weights_sha256 = h.hexdigest()
    b.split_sha256 = b.executed_sha256()
    return b


def _dense_c2l(bundle, me, u_t, v_t):
    """Per-tile c2l (D: c2l_ord2_face / C: c2l_ord2_cgrid_face), the flat
    arithmetic verbatim on the blocked windows; the flat NaN window is
    selected by the static per-tile mask.  ``me`` is a traced scalar
    under shard_map."""
    import jax.numpy as jnp

    dxb = jnp.asarray(bundle.dense["dx"])[me]        # (nl, nl+1)
    dyb = jnp.asarray(bundle.dense["dy"])[me]        # (nl+1, nl)
    am = jnp.asarray(bundle.dense["amat"])[me]       # (4, nl, nl)
    a11, a12, a21, a22 = am[0], am[1], am[2], am[3]
    if bundle.grid == "D":
        # wu = u*dx on the two bounding y-faces; covariant average
        wu_lo = u_t[:, :-1] * dxb[:, :-1, None]
        wu_hi = u_t[:, 1:] * dxb[:, 1:, None]
        u1 = 2.0 * (wu_lo + wu_hi) / (dxb[:, :-1, None]
                                      + dxb[:, 1:, None])
        wv_lo = v_t[:-1, :] * dyb[:-1, :, None]
        wv_hi = v_t[1:, :] * dyb[1:, :, None]
        v1 = 2.0 * (wv_lo + wv_hi) / (dyb[:-1, :, None]
                                      + dyb[1:, :, None])
    else:
        # wu = uc*dy on the two bounding x-faces (note the metric swap)
        wu_lo = u_t[:-1, :] * dyb[:-1, :, None]
        wu_hi = u_t[1:, :] * dyb[1:, :, None]
        u1 = 2.0 * (wu_lo + wu_hi) / (dyb[:-1, :, None]
                                      + dyb[1:, :, None])
        wv_lo = v_t[:, :-1] * dxb[:, :-1, None]
        wv_hi = v_t[:, 1:] * dxb[:, 1:, None]
        v1 = 2.0 * (wv_lo + wv_hi) / (dxb[:, :-1, None]
                                      + dxb[:, 1:, None])
    m = jnp.asarray(bundle.masks["c2l"])[me][..., None]
    nanv = jnp.full(u1.shape, jnp.nan, dtype=u1.dtype)
    ua = jnp.where(m, a11[..., None] * u1 + a12[..., None] * v1, nanv)
    va = jnp.where(m, a21[..., None] * u1 + a22[..., None] * v1, nanv)
    return ua, va


def _dense_project(bundle, me, ug, vg):
    """Per-tile a2d/a2c projection: the flat arithmetic verbatim on the
    blocked geo windows (pointwise given the 2-point averages the e=2
    window stores); the 3-term Cartesian dot stays the flat
    ``_dot_static`` left-to-right unroll."""
    import jax.numpy as jnp

    vlon = jnp.asarray(bundle.dense["vlon4"])[me]    # (nl+2, nl+2, 3)
    vlat = jnp.asarray(bundle.dense["vlat4"])[me]
    pu = jnp.asarray(bundle.dense["pu"])[me]
    pv = jnp.asarray(bundle.dense["pv"])[me]
    v3 = (ug[:, :, None, :] * vlon[..., None]
          + vg[:, :, None, :] * vlat[..., None])     # (t, t, 3, K)
    if bundle.grid == "D":
        ue = 0.5 * (v3[:, :-1] + v3[:, 1:])          # D-u slots (i, j-1/2)
        ve = 0.5 * (v3[:-1, :] + v3[1:, :])          # D-v slots (i-1/2, j)
    else:
        ue = 0.5 * (v3[:-1, :] + v3[1:, :])          # C-u slots (i-1/2, j)
        ve = 0.5 * (v3[:, :-1] + v3[:, 1:])          # C-v slots (i, j-1/2)
    out_u = pu[..., 0, None] * ue[:, :, 0]
    out_u = out_u + pu[..., 1, None] * ue[:, :, 1]
    out_u = out_u + pu[..., 2, None] * ue[:, :, 2]
    out_v = pv[..., 0, None] * ve[:, :, 0]
    out_v = out_v + pv[..., 1, None] * ve[:, :, 1]
    out_v = out_v + pv[..., 2, None] * ve[:, :, 2]
    return out_u, out_v


def make_tiled_ext_vector_sixface_from_splits(bundle: DuoTiledVectorSplits,
                                              mesh):
    """shard_map runtime of a prebuilt composed C/D bundle.

    Returns ``fn(u6b, v6b) -> (u6b, v6b)`` on BLOCKED global arrays --
    D: u ``(6, kt*nl, kt*(nl+1)[, K])``, v ``(6, kt*(nl+1), kt*nl[, K])``;
    C: the transposed pair.  Trailing K batched natively (the *_allk
    arm).  Width refusals ride the splits (GLM Q2).
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map

    _check_tile_mesh(mesh, bundle.kt,
                     "make_tiled_ext_vector_sixface_from_splits")
    kt = bundle.kt
    (tu0, tu1), (tv0, tv1) = bundle.split_ex.tiles
    blocked = ((6, kt * tu0, kt * tu1), (6, kt * tv0, kt * tv1))
    _body = tiled_vector_body(bundle)

    spec = tile_spec()
    sm = shard_map(_body, mesh=mesh, in_specs=(spec, spec),
                   out_specs=(spec, spec), check_rep=False)

    def fn(u6, v6):
        _check_blocked_args(
            f"tiled ext_vector[{bundle.grid}]", (u6, v6), blocked)
        return sm(u6, v6)

    fn.split = bundle
    fn.block_shapes = blocked
    fn.min_tile = bundle.depth + 1
    fn.spec = spec
    return fn


def tiled_vector_body(bundle: DuoTiledVectorSplits):
    """PER-DEVICE body of a composed C/D vector exchange (see
    :func:`tiled_split_body`): ``body(u_loc, v_loc) -> (u_loc, v_loc)``
    on this device's blocked tiles."""
    import jax.numpy as jnp

    nl = bundle.nl
    (tu0, tu1), (tv0, tv1) = bundle.split_ex.tiles
    su, sv = tu0 * tu1, tv0 * tv1
    (tg0, tg1) = bundle.split_geo.tiles[0]
    sg = tg0 * tg1
    sa = nl * nl

    def _body(u_loc, v_loc):
        me = tiled_me()
        u2 = u_loc.reshape(su, -1)
        v2 = v_loc.reshape(sv, -1)
        kk = u2.shape[1]
        # 1. tiled strip exchange + signed corner scatter
        cur = jnp.concatenate([u2, v2])
        cur = _run_phases(bundle.split_ex, me, cur)
        u_t = cur[:su].reshape(tu0, tu1, kk)
        v_t = cur[su:].reshape(tv0, tv1, kk)
        # 2. dense c2l (local)
        ua, va = _dense_c2l(bundle, me, u_t, v_t)
        # 3. geo lattice (pack embed + a4 pipeline), u/v batched on K
        step = jnp.concatenate([ua.reshape(sa, kk), va.reshape(sa, kk)],
                               axis=1)
        geo0 = jnp.full((sg, 2 * kk), jnp.nan, dtype=cur.dtype)
        curg = jnp.concatenate([geo0, step])
        curg = _run_phases(bundle.split_geo, me, curg)
        ug = curg[:sg, :kk].reshape(tg0, tg1, kk)
        vg = curg[:sg, kk:].reshape(tg0, tg1, kk)
        # 4. dense projection (local)
        p_u, p_v = _dense_project(bundle, me, ug, vg)
        # 5. tiled strip write-back
        curw = jnp.concatenate([u_t.reshape(su, kk), v_t.reshape(sv, kk),
                                p_u.reshape(-1, kk), p_v.reshape(-1, kk)])
        curw = _run_phases(bundle.split_wr, me, curw)
        u_f = curw[:su]
        v_f = curw[su:su + sv]
        # 6. final corner Lagrange per component (unless a2d variant)
        if bundle.split_cn is not None:
            curc = jnp.concatenate([u_f, v_f])
            curc = _run_phases(bundle.split_cn, me, curc)
            u_f = curc[:su]
            v_f = curc[su:]
        return (u_f.reshape(u_loc.shape), v_f.reshape(v_loc.shape))

    return _body


def make_tiled_ext_vector_dgrid_sixface(tab, mesh):
    """Tiled twin of ``ext_vector_dgrid_sixface_impl``; returns
    ``(fn, bundle)``."""
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_ext_vector_dgrid_sixface: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    bundle = build_tiled_ext_vector_splits(
        tab, int(mesh.devices.shape[1]), "D")
    return make_tiled_ext_vector_sixface_from_splits(bundle, mesh), bundle


def make_tiled_ext_vector_cgrid_sixface(tab, mesh):
    """Tiled twin of ``ext_vector_cgrid_sixface_impl``; returns
    ``(fn, bundle)``."""
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"make_tiled_ext_vector_cgrid_sixface: mesh.devices.ndim "
            f"{mesh.devices.ndim} != 3 (need the (6, kt, kt) tile mesh)")
    bundle = build_tiled_ext_vector_splits(
        tab, int(mesh.devices.shape[1]), "C")
    return make_tiled_ext_vector_sixface_from_splits(bundle, mesh), bundle


def make_tiled_ext_vector_dgrid_sixface_allk(tab, mesh):
    fn, b = make_tiled_ext_vector_dgrid_sixface(tab, mesh)
    return _allk_wrap(fn, 2, "make_tiled_ext_vector_dgrid_sixface_allk"), b


def make_tiled_ext_vector_cgrid_sixface_allk(tab, mesh):
    fn, b = make_tiled_ext_vector_cgrid_sixface(tab, mesh)
    return _allk_wrap(fn, 2, "make_tiled_ext_vector_cgrid_sixface_allk"), b


# ---------------------------------------------------------------------------
# tile_comm -- the bundle the halo dispatchers route through (M3 bridge)
# ---------------------------------------------------------------------------

def _tile_flat_iface(fn, kt: int, nl: int, mesh=None):
    """FLAT interface adapter for one tiled exchange fn.

    The full step's state stays FLAT six-face until M4 tiles the
    kernels, so the dispatcher seam hands FLAT arrays to the tile arm.
    kt=1: blocked == flat -- return ``fn`` itself (zero adaptation, the
    G0 bridge trace contains only the tile arm).  kt>1: convert
    flat -> blocked at entry and back at exit with the traced
    slice+concat twins (bit-preserving per element; the exchange keeps
    duplicated slots coherent, so the canonical read-back loses
    nothing).  One conversion pair PER FIRING -- firings are never
    fused or batched across names (scope P3 timing contract).

    SHARDING PIN (kt>1, measured defect + measured fix): with the
    converters unpinned, GSPMD propagates tile-axis shardings backward
    from the shard_map boundary into the FLAT-side kernels; the XLA CPU
    partitioner then produces NaN exactly on the internal shard-
    boundary cross of those [face,2,2]-split kernel intermediates
    (jobs 9577740/9589700/9591216: interior NaN at i=j=nl in uc/vc
    before the first C-vector exchange; the impl-return arm proved the
    NaN is KERNEL-born -- it persists even when only certified exchange
    values flow onward).  Pinning the wrapper's flat input and output
    to P('face') keeps every kernel on the face sharding and was
    measured to make the kt=2 step finite at the accepted lowering
    class (shadow4 mitig arm).  P('face') on the tile mesh REPLICATES
    flat state over the kt^2 tile devices -- that is the transitional
    pre-M4 bridge's existing contract (its out_shardings already do
    exactly this), not a new cost; M4's blocked tile-sharded state
    (P('face','tile_i','tile_j')) removes it."""
    if kt == 1:
        return fn

    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    face_sh = NamedSharding(mesh, P("face"))

    def _pin(x):
        # the pin matters only under jit (it steers the PARTITIONER);
        # eager callers (the G1 seam tests) pass concrete arrays and
        # need no annotation -- and eager with_sharding_constraint
        # semantics differ across jax versions, so skip it there.
        if isinstance(x, jax.core.Tracer):
            return jax.lax.with_sharding_constraint(x, face_sh)
        return x

    def flat(*arrs):
        outs = fn(*[to_blocked_traced(_pin(a), kt, nl) for a in arrs])
        if isinstance(outs, tuple):
            return tuple(_pin(from_blocked_traced(o, kt, nl))
                         for o in outs)
        return _pin(from_blocked_traced(outs, kt, nl))

    for at in ("split", "block_shapes", "min_tile", "spec"):
        if hasattr(fn, at):
            setattr(flat, at, getattr(fn, at))
    return flat


class DuoTileComm:
    """The prebuilt TILED exchanges for ONE ``(tab, mesh)`` pair (M3).

    The tile-mesh twin of :class:`DuoRingComm`: SAME method surface, so
    the six ``fv3_duo_halos`` dispatchers route through either bundle
    behind the same public names.  Rides on ``DuoHaloTables.tile_comm``;
    identity-hashable like its host (build one per context and reuse).
    All six arms (scalar A/B via one closure pair, D/C vector, and the
    K-batched variants) are built up front; ``splits`` maps arm name ->
    built split/bundle for instruments (write-set masks, receipts).

    Per-call halo width rides each SPLIT (``block_shapes``/``min_tile``,
    the M2 machinery): an arm refuses a caller array whose window does
    not match its own kt*(nl+e) extents, so a firing can never exchange
    on a narrower pad than its tables read.
    """

    __slots__ = ("_scalar", "_dgrid", "_cgrid",
                 "_scalar_allk", "_dgrid_allk", "_cgrid_allk",
                 "kt", "nl", "depth", "mesh", "splits")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):                              # pragma: no cover
        return f"DuoTileComm(kt={self.kt}, nl={self.nl}, depth={self.depth})"

    def ext_scalar(self, f6, stag: str):
        try:
            fn = self._scalar[stag]
        except KeyError:
            raise ValueError(
                f"DuoTileComm.ext_scalar: stagger {stag!r} not "
                f"implemented (the certified scalar exchange supports "
                f"'A' and 'B' only)") from None
        return fn(f6)

    def ext_vector_dgrid(self, u6, v6):
        return self._dgrid(u6, v6)

    def ext_vector_cgrid(self, uc6, vc6):
        return self._cgrid(uc6, vc6)

    def ext_scalar_allk(self, f6k, stag: str):
        try:
            fn = self._scalar_allk[stag]
        except KeyError:
            raise ValueError(
                f"DuoTileComm.ext_scalar_allk: stagger {stag!r} not "
                f"implemented (the certified scalar exchange supports "
                f"'A' and 'B' only)") from None
        return fn(f6k)

    def ext_vector_dgrid_allk(self, u6k, v6k):
        return self._dgrid_allk(u6k, v6k)

    def ext_vector_cgrid_allk(self, uc6k, vc6k):
        return self._cgrid_allk(uc6k, vc6k)


def build_tile_comm(tab, mesh) -> DuoTileComm:
    """Build the :class:`DuoTileComm` for ``tab`` on the (6, kt, kt)
    tile mesh.

    Attach it as ``tab.tile_comm`` to route every step-side exchange
    (all six ``ext_*_sixface[_allk]`` dispatchers in ``fv3_duo_halos``)
    through the M1/M2 tile arms.  kt is derived from the mesh; every
    split build re-runs the ALL-FAMILY census gate against its nl, so a
    window the tables outread is refused here, at build time.  Each
    K-batched arm shares its per-level arm's split (the tiled runtime is
    K-native -- one schedule, trailing K batched elementwise, no
    reassociation), wrapped with the K-required guard only.
    """
    if mesh.devices.ndim != 3:
        raise ValueError(
            f"build_tile_comm: mesh.devices.ndim {mesh.devices.ndim} != 3 "
            f"(need the (6, kt, kt) tile mesh)")
    kt = int(mesh.devices.shape[1])
    _check_tile_mesh(mesh, kt, "build_tile_comm")
    tc = DuoTileComm()
    tc.mesh = mesh
    scalar, scalar_allk, splits = {}, {}, {}
    for stag in ("A", "B"):
        fn, split = make_tiled_ext_scalar_sixface(tab, mesh, stag)
        flat = _tile_flat_iface(fn, kt, split.nl, mesh)
        scalar[stag] = flat
        scalar_allk[stag] = _allk_wrap(
            flat, 1, f"DuoTileComm.ext_scalar_allk[{stag}]")
        splits[f"scalar_{stag}"] = split
    tc._scalar = scalar
    tc._scalar_allk = scalar_allk
    dfn, dbundle = make_tiled_ext_vector_dgrid_sixface(tab, mesh)
    cfn, cbundle = make_tiled_ext_vector_cgrid_sixface(tab, mesh)
    tc._dgrid = _tile_flat_iface(dfn, kt, dbundle.nl, mesh)
    tc._cgrid = _tile_flat_iface(cfn, kt, cbundle.nl, mesh)
    tc._dgrid_allk = _allk_wrap(tc._dgrid, 2,
                                "DuoTileComm.ext_vector_dgrid_allk")
    tc._cgrid_allk = _allk_wrap(tc._cgrid, 2,
                                "DuoTileComm.ext_vector_cgrid_allk")
    splits["dgrid"] = dbundle
    splits["cgrid"] = cbundle
    tc.splits = splits
    tc.kt = kt
    tc.nl = int(splits["scalar_A"].nl)
    # the widest surface of the FULL step: every arm's census depth (the
    # census is ALL-family, so these agree; max kept as the binding record)
    tc.depth = max(int(s.depth) for s in splits.values())
    return tc
