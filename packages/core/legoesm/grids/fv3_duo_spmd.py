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
    "census_cross_face_depth",
    "assert_ring_width_covers",
    "build_ring_comm",
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


def _decode(lay, flat):
    """flat index -> (face, depth-from-edge), inverting ``_FlatLayout.idx``
    (``base + (face*m0 + i0)*m1 + j0``).  Shape-preserving."""
    shp = np.asarray(flat).shape
    flat = np.asarray(flat).ravel()
    bases = np.asarray(list(lay.bases) + [lay.total])
    k = np.searchsorted(bases, flat, side="right") - 1
    face = np.empty_like(flat)
    depth = np.empty_like(flat)
    for kk, (m0, m1) in enumerate(lay.shapes):
        m = k == kk
        if not m.any():
            continue
        rem = flat[m] - lay.bases[kk]
        face[m] = rem // (m0 * m1)
        r2 = rem % (m0 * m1)
        i, j = r2 // m1, r2 % m1
        depth[m] = np.minimum(np.minimum(i, m0 - 1 - i),
                              np.minimum(j, m1 - 1 - j))
    return face.reshape(shp), depth.reshape(shp)


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

    __slots__ = ("_scalar", "_dgrid", "_cgrid", "ring_width", "depth",
                 "mesh")

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
    depth = -1
    for stag in ("A", "B"):
        scalar[stag], depth = make_ring_ext_scalar_sixface(
            tab, stag, mesh, ring_width=ring_width)
    rc._scalar = scalar
    rc._dgrid, _ = make_ring_ext_vector_sixface(
        tab, "D", mesh, ring_width=ring_width)
    rc._cgrid, _ = make_ring_ext_vector_sixface(
        tab, "C", mesh, ring_width=ring_width)
    rc.ring_width = int(ring_width)
    rc.depth = int(depth)
    return rc
