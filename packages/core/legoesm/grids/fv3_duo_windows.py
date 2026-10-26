"""Sub-face WINDOWS for the DUO acoustic step (tiled port, M4).

The tiled port runs the CERTIFIED whole-face kernels unchanged on padded
sub-face windows.  A window is a ``(W, W)`` slab of one face's padded
array, ``W = nl + 2*pad`` with ``nl = n // kt`` the tile's own width and
``pad`` the seam pad depth; the kernel is told the window is a face of
size ``n_w = W - 2*ng`` with the usual ``ng`` halo.  Where a window side
is a REAL face edge the window is placed flush with the face's padded
array (its outer ``ng`` cells are the true cross-face ring); where it is
an intra-face SEAM the window boundary is ``pad`` cells inside the
neighbouring tile and the kernel's cube-edge/corner treatments there
act on cells the tile never publishes.

Two facts make this sound, both MEASURED on this lane:

* the intra-face reach of one acoustic substep with NO halo refresh is
  8 cells (static taint bound, tiled_m4_static_reach.py, job 9631177:
  C48/C96, hydrostatic and NH; the perturbation ladder reads 5..7), so a
  tile interior more than ``pad`` cells from its window boundary is
  unaffected by whatever the kernel does at that boundary, provided
  ``pad >= 8 + e`` where ``e`` is the depth of the kernel's own edge
  treatment at a fake edge.  ``pad`` is chosen by the BITWISE gate
  (``scripts/validate/fv3_native/tiled_m4_window_gate.py``), never by
  argument;
* with the seam pads refreshed ONCE per substep (at entry) the only
  data movement inside the substep is the cross-face exchange the
  whole-face step already fires, and each firing writes a STATIC set of
  cells (its write-set, censused once on random data); windows receive
  exactly that set, nothing else -- the pads stay as they were.

This module is the single-device ("windows as views") form: windows are
gathered from, and their owned interiors scattered back to, the flat
face-stacked state; every exchange runs the certified flat impl.  It is
the correctness gate for the design (G2/G3) and the layout the SPMD arm
will shard; it is not the fast path.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "WindowLayout",
    "build_window_layout",
    "window_axis_extent",
    "horizontal_axes",
    "gather_windows",
    "scatter_owned",
    "gather_masked",
    "build_window_ctx",
    "DuoWindowComm",
    "attach_window_comm",
]


class WindowLayout:
    """Static geometry of the ``6*kt*kt`` windows of one resolution.

    ``origins[w] = (face, oi, oj)``: the window's local (0, 0) sits at the
    face's padded index ``(oi, oj)``.  Interior tiles: ``o = ng + t*nl -
    pad``; a tile on a real face edge is placed flush with the padded
    array (``o = 0`` or ``o = m_a - W``), so a face-edge window boundary
    IS the face boundary.  ``kt == 1`` degenerates to ``W == m_a`` --
    the certified whole-face layout, byte for byte.
    """

    __slots__ = ("n", "ng", "kt", "pad", "nl", "W", "n_w", "m_a",
                 "origins", "nb", "partition")

    def __init__(self, n, ng, kt, pad, partition="compute"):
        """``partition`` -- which cells a tile OWNS along one axis:

        ``"compute"``  the compute domain split kt ways (``kt | n``); the
                       rings go to the edge tiles, a shared node to the
                       east/north tile.  The single-device gate's layout.
        ``"padded"``   the PADDED array split kt ways (``kt | m_a``),
                       block ``t`` = padded rows ``[t*nl, t*nl + nl + e)``
                       -- the M1-M3 tiled exchange's BLOCKED layout
                       (fv3_duo_spmd.to_blocked), so the SPMD arm can hand
                       its blocks to that certified exchange unchanged.  A
                       shared node (``e = 1``, padded index ``t*nl``) is
                       stored by both tiles and kept coherent by the
                       exchange; for a single owner the EAST/NORTH tile
                       (``t = f // nl``, the block that starts on it) is
                       canonical, as in the compute partition.
        """
        m_a = n + 2 * ng
        if partition == "compute":
            if n % kt:
                raise ValueError(f"face size n={n} is not divisible by "
                                 f"kt={kt}")
            nl = n // kt
        elif partition == "padded":
            if m_a % kt:
                raise ValueError(f"padded extent m_a={m_a} is not divisible "
                                 f"by kt={kt} (the blocked layout's rule)")
            nl = m_a // kt
        else:
            raise ValueError(f"partition {partition!r}: 'compute' or 'padded'")
        if kt == 1:
            # the whole face, exactly: W == m_a (the compute partition's
            # nl = n needs the ng ring back; the padded one already has it)
            pad = ng if partition == "compute" else 0
        elif pad < ng:
            raise ValueError(f"pad={pad} < ng={ng}: a window must carry at "
                             f"least the kernel's own halo")
        W = nl + 2 * pad
        if W > m_a:
            raise ValueError(
                f"window W={W} (nl={nl}, pad={pad}) exceeds the padded face "
                f"m_a={m_a}: kt={kt} cannot tile C{n} at this pad")
        self.n, self.ng, self.kt, self.pad, self.nl = n, ng, kt, pad, nl
        self.W, self.n_w, self.m_a = W, W - 2 * ng, m_a
        self.partition = partition
        origins = []
        for face in range(6):
            for ti in range(kt):
                for tj in range(kt):
                    origins.append((face, self._origin(ti),
                                    self._origin(tj)))
        self.origins = tuple(origins)
        self.nb = len(origins)

    def block_start(self, t):
        """Padded index of the first cell tile ``t`` owns along one axis."""
        if self.partition == "compute":
            return self.ng + t * self.nl
        return t * self.nl

    def _origin(self, t):
        o = self.block_start(t) - self.pad
        return int(min(max(o, 0), self.m_a - self.W))

    def owner(self, f_padded):
        """Tile index owning padded flat index ``f`` along one axis.
        compute: a node shared by two tiles belongs to the EAST/NORTH
        one, the last node and the rings to the edge tiles; padded: the
        block that STARTS on a shared node (east/north, ``f // nl``) is
        canonical, the last node goes to the last tile."""
        if self.partition == "compute":
            return int(min(max((f_padded - self.ng) // self.nl, 0),
                           self.kt - 1))
        return int(min(f_padded // self.nl, self.kt - 1))

    def __repr__(self):                                  # pragma: no cover
        return (f"WindowLayout(n={self.n}, ng={self.ng}, kt={self.kt}, "
                f"pad={self.pad}, W={self.W}, nb={self.nb}, "
                f"partition={self.partition!r})")


def build_window_layout(n: int, ng: int, kt: int, pad: int,
                        partition: str = "compute") -> WindowLayout:
    return WindowLayout(n, ng, kt, pad, partition)


def window_axis_extent(lay: WindowLayout, e_flat: int):
    """Window extent for a flat axis extent, or None if the axis is not
    a horizontal one.  Horizontal extents on this lane are compute-
    domain aligned with a symmetric halo, so ``e_w = e_f - (m_a - W)``;
    recognised extents run from the compute width ``n`` to the padded
    node width ``m_a + 1``."""
    if lay.n <= e_flat <= lay.m_a + 1:
        return e_flat - (lay.m_a - lay.W)
    return None


def _hslices(lay: WindowLayout, w: int, e0: int, e1: int):
    """(slice_i, slice_j) of window ``w`` inside a flat array whose
    horizontal extents are ``(e0, e1)``.  Compute-domain arrays (extent
    ``n``/``n+1``) start ``ng`` later than padded ones, and the window's
    own compute domain starts ``ng`` into the window, so the offset is
    the padded origin for every extent."""
    _, oi, oj = lay.origins[w]
    wi, wj = window_axis_extent(lay, e0), window_axis_extent(lay, e1)
    if wi is None or wj is None:
        raise ValueError(f"extents {(e0, e1)} are not horizontal on a C{lay.n}"
                         f" ng={lay.ng} face")
    return slice(oi, oi + wi), slice(oj, oj + wj)


def horizontal_axes(lay: WindowLayout, shape, lead):
    """``(ai, aj)`` -- the two horizontal axes of a stacked array whose
    leading extent is ``lead`` (6 = flat faces, ``lay.nb`` = windows), or
    None.  The lane's layouts are ``(., i, j, ...)`` and, for ``pe`` /
    ``peln`` only, ``(., i, k, j)`` (field_shape: load-bearing Fortran
    order).  A window array's extents are the flat ones shifted by
    ``m_a - W``, so the recognised range follows ``lead``."""
    shape = tuple(shape)
    if len(shape) < 3 or shape[0] != lead:
        return None
    shift = 0 if lead == 6 else (lay.m_a - lay.W)
    ok = [window_axis_extent(lay, e + shift) is not None for e in shape[1:4]]
    if len(shape) >= 4 and ok[0] and ok[1] and ok[2]:
        # (i, j, k) and (i, k, j) both fit: a vertical extent inside the
        # horizontal range (tiny toy faces).  Refuse rather than guess --
        # a wrong pick gathers the wrong slice silently (codex 2026-09-04).
        raise ValueError(
            f"horizontal_axes: shape {shape} is ambiguous between (i, j, k) "
            f"and (i, k, j) on a C{lay.n} ng={lay.ng} face (axis 2 extent "
            f"{shape[2]} is also a horizontal extent)")
    if ok[0] and ok[1]:
        return (1, 2)
    if len(shape) >= 4 and ok[0] and not ok[1] and ok[2]:
        return (1, 3)
    return None


def _fshape(lay: WindowLayout, shape, axes):
    out = list(shape)
    out[0] = 6
    for a in axes:
        out[a] = shape[a] + (lay.m_a - lay.W)
    return tuple(out)


def _index(lay: WindowLayout, w, shape_flat, axes):
    """Index tuple selecting window ``w`` inside a flat stacked array."""
    face, oi, oj = lay.origins[w]
    idx = [slice(None)] * len(shape_flat)
    idx[0] = face
    si, sj = _hslices(lay, w, shape_flat[axes[0]], shape_flat[axes[1]])
    idx[axes[0]], idx[axes[1]] = si, sj
    return tuple(idx)


def _expand(mask2, ndim, axes):
    """Broadcast a 2-D (i, j) mask onto the window's rank (no leading
    axis): axes ``axes[0]-1``/``axes[1]-1`` carry the mask."""
    shape = [1] * (ndim - 1)
    shape[axes[0] - 1] = mask2.shape[0]
    shape[axes[1] - 1] = mask2.shape[1]
    return mask2.reshape(shape)


def gather_windows(lay: WindowLayout, x6, xp=None):
    """``(6, ..i.., ..j.., ...)`` flat -> ``(nb, ...)`` windows (a pure
    index copy)."""
    if xp is None:
        import jax.numpy as xp
    axes = horizontal_axes(lay, x6.shape, 6)
    if axes is None:
        raise ValueError(f"gather_windows: {np.shape(x6)} is not a "
                         f"face-stacked horizontal array")
    return xp.stack([x6[_index(lay, w, x6.shape, axes)]
                     for w in range(lay.nb)], axis=0)


def _owner_mask(lay: WindowLayout, w: int, e0: int, e1: int):
    """Static bool ``(w0, w1)``: which window cells the tile OWNS (the
    cells it publishes back to the flat state)."""
    face, oi, oj = lay.origins[w]
    ti = (w % (lay.kt * lay.kt)) // lay.kt
    tj = w % lay.kt
    wi, wj = window_axis_extent(lay, e0), window_axis_extent(lay, e1)
    # an axis of extent e carries a symmetric halo of (m_a - e)/2, so
    # its index 0 sits (m_a - e + 1)//2 cells into the padded frame
    # (0 for m_a and m_a+1, ng for n and n+1, ng-1 for n+2, ...)
    shift_i = (lay.m_a - e0 + 1) // 2
    shift_j = (lay.m_a - e1 + 1) // 2
    mi = np.array([lay.owner(l + oi + shift_i) == ti for l in range(wi)])
    mj = np.array([lay.owner(l + oj + shift_j) == tj for l in range(wj)])
    return mi[:, None] & mj[None, :]


def scatter_owned(lay: WindowLayout, xw, xp=None):
    """``(nb, ...)`` windows -> ``(6, ...)`` flat, each flat cell written
    by exactly ONE window (its owner).  Every flat cell has an owner (the
    edge tiles own the rings), so the result is a complete array."""
    if xp is None:
        import jax.numpy as xp
    axes = horizontal_axes(lay, xw.shape, lay.nb)
    if axes is None:
        raise ValueError(f"scatter_owned: {np.shape(xw)} is not a "
                         f"window-stacked horizontal array")
    fshape = _fshape(lay, xw.shape, axes)
    out = xp.zeros(fshape, dtype=xw.dtype)
    for w in range(lay.nb):
        idx = _index(lay, w, fshape, axes)
        m = _expand(_owner_mask(lay, w, fshape[axes[0]], fshape[axes[1]]),
                    xw.ndim, axes)
        if xp is np:
            # host path (a gate reading SHARDED windows back: every
            # eager .at[].set on a sharded array is a 24-device
            # collective, job 9632092 timed out in exactly that)
            out[idx] = np.where(m, xw[w], out[idx])
        else:
            out = out.at[idx].set(xp.where(m, xw[w], out[idx]))
    return out


def gather_masked(lay: WindowLayout, xw, x6, mask6, xp=None):
    """Windows with the cells in ``mask6`` (static bool ``(6, ei, ej)`` in
    FLAT coordinates over the two horizontal axes) replaced from ``x6``;
    every other window cell is untouched.  This is how a mid-substep
    firing reaches a window: only the firing's write-set moves."""
    if xp is None:
        import jax.numpy as xp
    axes = horizontal_axes(lay, x6.shape, 6)
    full = np.ndim(mask6) == x6.ndim          # exact per-cell mask
    outs = []
    for w in range(lay.nb):
        idx = _index(lay, w, x6.shape, axes)
        face, _, _ = lay.origins[w]
        if full:
            m = np.asarray(mask6[idx], bool)
        else:
            m = np.asarray(mask6[face, idx[axes[0]], idx[axes[1]]], bool)
        if not m.any():
            outs.append(xw[w])
            continue
        outs.append(xp.where(m if full else _expand(m, xw.ndim, axes),
                             x6[idx], xw[w]))
    return xp.stack(outs, axis=0)


# ---------------------------------------------------------------------------
# window context (the kernels' static half)
# ---------------------------------------------------------------------------

def _window_metric(lay: WindowLayout, w: int, v):
    """Slice ONE gridstruct metric onto window ``w``: horizontal axes
    (the leading one or two whose extent is horizontal) are windowed,
    anything else (scalars, level tables, per-cell vectors) is kept."""
    a = np.asarray(v) if not hasattr(v, "shape") else v
    if a.ndim == 0:
        return a
    _, oi, oj = lay.origins[w]
    w0 = window_axis_extent(lay, a.shape[0])
    if a.ndim == 1:
        # 1-D edge factors (length n+1 = npx, compute-node aligned) and
        # any other horizontal 1-D table follow axis 0 == i
        return a[oi:oi + w0] if w0 is not None else a
    w1 = window_axis_extent(lay, a.shape[1])
    if w0 is not None and w1 is not None:
        return a[oi:oi + w0, oj:oj + w1]
    if w0 is not None:
        return a[oi:oi + w0]
    return a


def build_window_ctx(ctx, lay: WindowLayout):
    """A :class:`DuoStepperContext` whose 'faces' are the ``nb`` windows:
    ``n = n_w``, the per-window metric dicts sliced from the face's, the
    face's flags/da_min replicated per window, ``hs6`` windowed.  ``tab``
    is the FLAT table object (the window comm attached to it carries the
    layout); the kernels never index ``tab`` themselves."""
    from legoesm.core.fv3_duo_stepper import DuoStepperContext
    from legoesm.core.fv3_native_sw_core import Bounds

    out = DuoStepperContext.__new__(DuoStepperContext)
    out.n, out.ng = lay.n_w, lay.ng
    out.npx, out.m_a = lay.n_w + 1, lay.W
    out.bd = Bounds.single_tile(lay.n_w, lay.ng)
    out.tab = ctx.tab
    gs_w, fl_w = [], []
    for w, (face, _, _) in enumerate(lay.origins):
        gs_w.append({k: _window_metric(lay, w, v)
                     for k, v in ctx.gs6[face].items()})
        fl_w.append(ctx.flags6[face])
    out.gs6, out.flags6 = tuple(gs_w), tuple(fl_w)
    out.hs6 = gather_windows(lay, ctx.hs6)
    out.duogrid = ctx.duogrid
    out._batched_gs = None
    return out


# ---------------------------------------------------------------------------
# the comm bundle: certified flat exchanges seen through the windows
# ---------------------------------------------------------------------------

class DuoWindowComm:
    """Exchange bundle for window-stacked state (``tab.window_comm``).

    Same method surface as the ring/tile bundles; each method scatters
    the windows' OWNED cells to the flat state, runs the certified flat
    impl, and hands each window back exactly the impl's WRITE-SET cells
    (censused once per (name, shape) on random data).  ``refresh``
    rebuilds every window from the owners -- the once-per-substep seam
    refresh.  ``handles_barriers`` routes the two edge-blend barriers
    through the same path.
    """

    handles_barriers = True

    def __init__(self, lay: WindowLayout, tab, sharding=None):
        self.lay, self.tab = lay, tab
        self._writeset = {}
        # optional: pin every result to a window sharding, so that under
        # jit the kernels stay PARTITIONED across the exchange even though
        # the exchange itself goes through the flat state -- the
        # same-shape no-comm CONTROL arm of the SPMD gate (GLM 2026-09-04)
        self.sharding = sharding

    def _pin(self, x):
        import jax
        if self.sharding is not None and isinstance(x, jax.core.Tracer):
            return jax.lax.with_sharding_constraint(x, self.sharding)
        return x

    # -- write-set census --------------------------------------------------
    def _census(self, name, fn, shapes, dtypes, n_out):
        """Static bool masks (one per output, FULL rank -- exact per cell,
        so a slot-selective barrier refreshes only the slots it writes,
        codex 2026-09-04) of the cells ``fn`` writes, OR-ed over three
        random trials so an accidental equality cannot hide a written
        cell.  Keyed on the table parameters the write-set depends on."""
        tab = self.tab
        key = (name, tuple(shapes), getattr(tab, "nq", None),
               getattr(tab, "k2e_nord", None),
               getattr(tab, "vector_corner", None))
        if key in self._writeset:
            return self._writeset[key]
        import jax
        import jax.numpy as jnp
        rng = np.random.default_rng(0x5EED)
        masks = [np.zeros(s, bool) for s in shapes[:n_out]]
        # The census is called from INSIDE the step's trace (omnistaging
        # would stage these concrete ops as tracers and the mask could
        # not be read); evaluate it at compile time instead.
        with jax.ensure_compile_time_eval():
            for _ in range(3):
                ins = [jnp.asarray(rng.standard_normal(s), dtype=d)
                       for s, d in zip(shapes, dtypes)]
                outs = fn(*ins)
                outs = outs if isinstance(outs, (tuple, list)) else (outs,)
                for i, (o, x) in enumerate(zip(outs, ins)):
                    masks[i] |= np.asarray(o != x)   # NaN out = written
        self._writeset[key] = tuple(masks)
        return self._writeset[key]

    def _via_flat(self, name, fn, arrays, n_out=None):
        """Run the certified flat ``fn`` on the scattered windows; return
        the windows with only the write-set cells refreshed."""
        import jax.numpy as jnp
        n_out = len(arrays) if n_out is None else n_out
        flats = [scatter_owned(self.lay, a) for a in arrays]
        masks = self._census(name, fn, [tuple(f.shape) for f in flats],
                             [f.dtype for f in flats], n_out)
        outs = fn(*flats)
        outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        res = [self._pin(gather_masked(self.lay, a, o, m, jnp))
               for a, o, m in zip(arrays, outs, masks)]
        return res[0] if len(res) == 1 else tuple(res)

    # -- exchange surface ---------------------------------------------------
    def ext_scalar(self, fw, stag):
        from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_impl
        return self._via_flat(("ext_scalar", stag),
                              lambda f: ext_scalar_sixface_impl(f, self.tab,
                                                                stag),
                              [fw])

    def ext_scalar_allk(self, fwk, stag):
        from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_impl

        def fn(f6k):
            for k in range(f6k.shape[-1]):
                f6k = f6k.at[..., k].set(
                    ext_scalar_sixface_impl(f6k[..., k], self.tab, stag))
            return f6k
        return self._via_flat(("ext_scalar_allk", stag), fn, [fwk])

    def ext_vector_dgrid(self, uw, vw):
        from legoesm.grids.fv3_duo_halos import ext_vector_dgrid_sixface_impl
        return self._via_flat(
            "ext_vector_dgrid",
            lambda u, v: ext_vector_dgrid_sixface_impl(u, v, self.tab),
            [uw, vw])

    def ext_vector_dgrid_allk(self, uwk, vwk):
        from legoesm.grids.fv3_duo_halos import ext_vector_dgrid_sixface_impl

        def fn(u6k, v6k):
            for k in range(u6k.shape[-1]):
                uk, vk = ext_vector_dgrid_sixface_impl(u6k[..., k],
                                                       v6k[..., k], self.tab)
                u6k = u6k.at[..., k].set(uk)
                v6k = v6k.at[..., k].set(vk)
            return u6k, v6k
        return self._via_flat("ext_vector_dgrid_allk", fn, [uwk, vwk])

    def ext_vector_cgrid(self, ucw, vcw):
        from legoesm.grids.fv3_duo_halos import ext_vector_cgrid_sixface_impl
        return self._via_flat(
            "ext_vector_cgrid",
            lambda u, v: ext_vector_cgrid_sixface_impl(u, v, self.tab),
            [ucw, vcw])

    def ext_vector_cgrid_allk(self, ucwk, vcwk):
        from legoesm.grids.fv3_duo_halos import ext_vector_cgrid_sixface_impl

        def fn(u6k, v6k):
            for k in range(u6k.shape[-1]):
                uk, vk = ext_vector_cgrid_sixface_impl(u6k[..., k],
                                                       v6k[..., k], self.tab)
                u6k = u6k.at[..., k].set(uk)
                v6k = v6k.at[..., k].set(vk)
            return u6k, v6k
        return self._via_flat("ext_vector_cgrid_allk", fn, [ucwk, vcwk])

    # -- barriers -------------------------------------------------------------
    #  LEVEL AXIS (codex 2026-09-07): the callers now hand the barriers the
    #  WHOLE level stack in one call (M8-B).  The certified impls are
    #  per-level -- they flatten every cell axis into one index and blend
    #  through tables built for that layout -- so a level axis reaching the
    #  impl unmapped would land INSIDE the flattening and alias levels.
    #  The flat dispatchers vmap it; this single-device window bundle
    #  bypasses those dispatchers (it calls the impls directly), so it
    #  vmaps here.  ``k_axis`` is where the level axis sits per barrier:
    #  the bgrid/cgrid arrays carry it LAST, the allflux stacks carry it at
    #  axis 3 with the slot axis last (the oracle's allflux_x(i,j,k,iq)).
    def _barrier_via_flat(self, name, impl, arrays, k_axis, ndim_2d):
        import jax
        fn = impl
        if arrays[0].ndim > ndim_2d:
            fn = jax.vmap(impl, in_axes=k_axis, out_axes=k_axis)
        return self._via_flat(name, fn, arrays)

    def average_shared_edge_bgrid(self, xbw, ybw):
        from legoesm.grids.fv3_duo_halos import average_shared_edge_bgrid_impl
        return self._barrier_via_flat(
            "avg_bgrid",
            lambda x, y: average_shared_edge_bgrid_impl(x, y, self.tab),
            [xbw, ybw], k_axis=-1, ndim_2d=3)

    def average_shared_edge_cgrid(self, fxw, fyw):
        from legoesm.grids.fv3_duo_halos import average_shared_edge_cgrid_impl
        return self._barrier_via_flat(
            "avg_cgrid",
            lambda x, y: average_shared_edge_cgrid_impl(x, y, self.tab),
            [fxw, fyw], k_axis=-1, ndim_2d=3)

    def average_allflux_shared_edges(self, afxw, afyw):
        from legoesm.grids.fv3_duo_halos import (
            average_allflux_shared_edges_impl)
        # (nb, i, j, slot) is the per-level stack; (nb, i, j, km, slot) adds
        # the level axis at 3
        return self._barrier_via_flat(
            "avg_allflux",
            lambda x, y: average_allflux_shared_edges_impl(x, y, self.tab),
            [afxw, afyw], k_axis=3, ndim_2d=4)

    # -- the once-per-substep seam refresh --------------------------------------
    def refresh(self, bundle: dict) -> dict:
        """Every window-stacked horizontal array in ``bundle`` rebuilt from
        the owners' cells: the seam pads become the neighbours' current
        interiors.  Non-horizontal entries pass through."""
        import jax.numpy as jnp
        out = {}
        lay = self.lay
        for k, v in bundle.items():
            if (hasattr(v, "ndim")
                    and horizontal_axes(lay, v.shape, lay.nb) is not None):
                out[k] = self._pin(
                    gather_windows(lay, scatter_owned(lay, v, jnp), jnp))
            else:
                out[k] = v
        return out


def attach_window_comm(ctx, kt: int, pad: int, partition: str = "compute"):
    """Build the window layout for ``ctx`` (a whole-face
    :class:`DuoStepperContext`), attach a :class:`DuoWindowComm` to its
    tables and return ``(window_ctx, comm)``.  The flat ``ctx.tab`` is
    shared; the window ctx is what the phases run on."""
    lay = build_window_layout(ctx.n, ctx.ng, kt, pad, partition)
    comm = DuoWindowComm(lay, ctx.tab)
    ctx.tab.window_comm = comm
    return build_window_ctx(ctx, lay), comm
