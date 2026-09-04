"""SPMD arm of the DUO sub-face WINDOWS (tiled port, M4b).

The window layout of :mod:`fv3_duo_windows` sharded ONE WINDOW PER DEVICE
on the ``(face, tile_i, tile_j)`` mesh of the M1-M3 tiled exchange.  The
kernels are untouched: the phases' ``batched`` arm vmaps over the window
axis, which GSPMD partitions with zero communication.  Every exchange
firing is ONE shard_map over that mesh whose body does three local things
and two kinds of neighbour traffic:

1. cut this device's BLOCK out of its window (the padded partition:
   block ``t`` = padded rows ``[t*nl, t*nl+nl+e)``, the certified blocked
   layout of ``fv3_duo_spmd.to_blocked``) -- a dynamic slice, the offset
   depends on the tile's position because face-edge windows sit flush
   with the face;
2. run the certified tiled exchange BODY on the block
   (:func:`fv3_duo_spmd.tiled_split_body` / :func:`tiled_vector_body`:
   the same ppermute schedule and the same ordered arithmetic as M1-M3,
   bitwise);
3. write the block back and REFRESH THE PADS: two ppermute rounds (tile_i
   then tile_j) carrying ``2*pad`` rows/columns each way, so every seam
   pad -- including the wider pad of a flush edge window and the corner
   pads -- holds its neighbour's current block values.

The pad refresh after EVERY firing is the correctness-first form (fresher
pads than the single-device arm's once-per-substep refresh, hence equally
bitwise against the flat step); restricting it to each firing's write-set
is a later optimisation with its own gate.  The three edge-blend BARRIERS
are not yet tiled: they run on the flat state through the single-device
bundle's scatter/gather (O(state) communication) -- correct, slow, and
named as such (``barrier_mode``).  Requires ``2*pad <= nl + e`` (each
tile sends only its own block rows) and ``kt | m_a``.
"""
from __future__ import annotations

import numpy as np

from legoesm.grids.fv3_duo_windows import (
    DuoWindowComm, WindowLayout, build_window_ctx, build_window_layout,
    horizontal_axes, window_axis_extent)

__all__ = ["DuoWindowSpmdComm", "attach_window_spmd_comm", "window_sharding"]

_AXES = ("face", "tile_i", "tile_j")


def window_sharding(mesh):
    """NamedSharding placing a ``(nb, ...)`` window stack one window per
    device: the leading axis is split over the three mesh axes jointly, in
    mesh order, so device ``(f, ti, tj)`` holds window ``(f*kt+ti)*kt+tj``
    -- exactly :class:`WindowLayout`'s window order."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    return NamedSharding(mesh, P(_AXES))


class DuoWindowSpmdComm:
    """``tab.window_comm`` for window stacks sharded on the tile mesh."""

    handles_barriers = True
    barrier_mode = "flat-fallback"

    def __init__(self, lay: WindowLayout, tab, mesh):
        import jax
        from legoesm.grids.fv3_duo_spmd import (
            build_tiled_ext_scalar_a_split, build_tiled_ext_scalar_b_split,
            build_tiled_ext_vector_splits, tiled_split_body,
            tiled_vector_body)

        if lay.partition != "padded":
            raise ValueError("DuoWindowSpmdComm needs the 'padded' partition "
                             "(the blocked layout's kt | m_a rule)")
        if tuple(mesh.axis_names) != _AXES or tuple(
                mesh.devices.shape) != (6, lay.kt, lay.kt):
            raise ValueError(
                f"mesh axes/shape {tuple(mesh.axis_names)} "
                f"{tuple(mesh.devices.shape)} != {_AXES} (6, {lay.kt}, "
                f"{lay.kt})")
        if 2 * lay.pad > lay.nl:
            raise ValueError(
                f"2*pad = {2 * lay.pad} > nl = {lay.nl}: a flush edge window "
                f"needs 2*pad rows from ONE neighbour block; use a smaller "
                f"kt or pad at C{lay.n}")
        self.lay, self.tab, self.mesh = lay, tab, mesh
        self.sharding = window_sharding(mesh)
        kt = lay.kt
        self._bodies = {
            "A": tiled_split_body(build_tiled_ext_scalar_a_split(tab, kt)),
            "B": tiled_split_body(build_tiled_ext_scalar_b_split(tab, kt)),
            "D": tiled_vector_body(build_tiled_ext_vector_splits(tab, kt, "D")),
            "C": tiled_vector_body(build_tiled_ext_vector_splits(tab, kt, "C")),
        }
        self._flat = DuoWindowComm(lay, tab)     # the barriers' fallback
        self._jax = jax

    # ------------------------------------------------------------------
    # geometry inside the shard_map body (traced tile position)
    # ------------------------------------------------------------------
    def _b0(self, t):
        """Local index of this tile's block start along one axis (traced
        ``t``): ``block_start - origin`` with the flush clamp."""
        import jax.numpy as jnp
        lay = self.lay
        start = t * lay.nl
        origin = jnp.clip(start - lay.pad, 0, lay.m_a - lay.W)
        return start - origin

    def _pad_exchange(self, arrs):
        """Two ppermute rounds refreshing every seam pad of every array in
        ``arrs`` (local PADDED-extent windows ``(W+e0, W+e1, ...)``) from
        the neighbours' blocks: round 1 along tile_i (rows), round 2
        along tile_j (columns of the row-refreshed windows, so corners
        come along).  Each tile sends its first/last ``2*pad`` block
        rows; the receiver places them at the neighbour block's known
        position inside a ``2*pad``-extended copy of its window and
        slices the window back, so rows that fall outside the window (an
        interior tile needs only ``pad`` of them; an edge tile has no
        neighbour on one side and receives ppermute's zeros there) land
        in the discarded extension."""
        import jax
        import jax.numpy as jnp
        lay = self.lay
        P2 = 2 * lay.pad
        kt = lay.kt
        out = []
        for a in arrs:
            for axis_name, ax in (("tile_i", 0), ("tile_j", 1)):
                t = jax.lax.axis_index(axis_name)
                e = a.shape[ax] - lay.W            # 0 cell axis, 1 node axis
                b0 = self._b0(t)
                blk = lay.nl + e
                # a node axis (e = 1) stores the shared row in BOTH tiles:
                # the neighbour's pad starts after / ends before it
                lo = jax.lax.dynamic_slice_in_dim(a, b0 + e, P2, axis=ax)
                hi = jax.lax.dynamic_slice_in_dim(a, b0 + blk - e - P2, P2,
                                                  axis=ax)
                to_west = [(s, s - 1) for s in range(1, kt)]
                to_east = [(s, s + 1) for s in range(kt - 1)]
                from_east = jax.lax.ppermute(lo, axis_name, to_west)
                from_west = jax.lax.ppermute(hi, axis_name, to_east)
                widths = [(0, 0)] * a.ndim
                widths[ax] = (P2, P2)
                ext = jnp.pad(a, widths)
                # the east neighbour's block starts at my b0 + blk; the
                # west neighbour's last 2*pad rows end at my b0
                ext = jax.lax.dynamic_update_slice_in_dim(
                    ext, from_east, P2 + b0 + blk, axis=ax)
                ext = jax.lax.dynamic_update_slice_in_dim(
                    ext, from_west, b0, axis=ax)
                a = jax.lax.slice_in_dim(ext, P2, P2 + a.shape[ax], axis=ax)
            out.append(a)
        return out

    def _normalize(self, a):
        """Window array -> ``(padded-extent (W+e0, W+e1, ...) form,
        restore)``: an (i, k, j) array is transposed to (i, j, k), and a
        compute-domain array is embedded in the padded extent (its
        window compute index 0 sits ``ng`` into the window), so every
        firing sees the one blocked geometry.  ``restore`` undoes both."""
        import jax.numpy as jnp
        lay = self.lay
        axes = horizontal_axes(lay, (lay.nb,) + tuple(a.shape), lay.nb)
        if axes is None:
            raise ValueError(f"window array of shape {a.shape} has no "
                             f"horizontal axes on {lay}")
        swap = axes == (1, 3)
        if swap:
            a = jnp.swapaxes(a, 1, 2)          # (i, k, j) -> (i, j, k)
        e_w0, e_w1 = a.shape[0], a.shape[1]
        lo0 = (lay.W - e_w0 + 1) // 2 if e_w0 < lay.W else 0
        lo1 = (lay.W - e_w1 + 1) // 2 if e_w1 < lay.W else 0
        node0 = 1 if e_w0 in (lay.n_w + 1, lay.W + 1) else 0
        node1 = 1 if e_w1 in (lay.n_w + 1, lay.W + 1) else 0
        hi0 = lay.W + node0 - e_w0 - lo0
        hi1 = lay.W + node1 - e_w1 - lo1
        widths = [(lo0, hi0), (lo1, hi1)] + [(0, 0)] * (a.ndim - 2)
        if any(w != (0, 0) for w in widths):
            a = jnp.pad(a, widths)

        def restore(b):
            b = b[lo0:lo0 + e_w0, lo1:lo1 + e_w1]
            return jnp.swapaxes(b, 1, 2) if swap else b
        return a, restore

    def _firing(self, name, kind, arrays, body=None):
        """ONE shard_map: blocks out of the windows, the certified body,
        blocks back, pads refreshed.  ``body`` None = refresh only."""
        import jax
        from jax.experimental.shard_map import shard_map
        lay = self.lay
        spec = self.sharding.spec

        def _run(*locs):
            locs = [l[0] for l in locs]                     # (W0, W1, ...)
            norm = [self._normalize(a) for a in locs]
            locs = [a for a, _ in norm]
            ti = jax.lax.axis_index("tile_i")
            tj = jax.lax.axis_index("tile_j")
            if body is not None:
                bi, bj = self._b0(ti), self._b0(tj)
                blocks = []
                for a in locs:
                    e0, e1 = a.shape[0] - lay.W, a.shape[1] - lay.W
                    blk = jax.lax.dynamic_slice_in_dim(a, bi, lay.nl + e0, 0)
                    blk = jax.lax.dynamic_slice_in_dim(blk, bj, lay.nl + e1, 1)
                    blocks.append(blk)
                outs = body(*blocks)
                outs = list(outs) if isinstance(outs, tuple) else [outs]
                new = []
                for a, blk in zip(locs, outs):
                    strip = jax.lax.dynamic_slice_in_dim(a, bi, blk.shape[0],
                                                         0)
                    strip = jax.lax.dynamic_update_slice_in_dim(strip, blk,
                                                                bj, 1)
                    new.append(jax.lax.dynamic_update_slice_in_dim(
                        a, strip, bi, 0))
                locs = new
            locs = self._pad_exchange(locs)
            return tuple(restore(a)[None] for a, (_, restore)
                         in zip(locs, norm))

        n = len(arrays)
        sm = shard_map(_run, mesh=self.mesh, in_specs=(spec,) * n,
                       out_specs=(spec,) * n, check_rep=False)
        outs = sm(*[jax.lax.with_sharding_constraint(a, self.sharding)
                    if isinstance(a, jax.core.Tracer) else a
                    for a in arrays])
        return outs[0] if n == 1 else tuple(outs)

    # ------------------------------------------------------------------
    # exchange surface (window stacks in, window stacks out)
    # ------------------------------------------------------------------
    def ext_scalar(self, fw, stag):
        return self._firing("ext_scalar", stag, [fw], self._bodies[stag])

    def ext_scalar_allk(self, fwk, stag):
        return self._firing("ext_scalar_allk", stag, [fwk],
                            self._bodies[stag])

    def ext_vector_dgrid(self, uw, vw):
        return self._firing("ext_vector_dgrid", "D", [uw, vw],
                            self._bodies["D"])

    def ext_vector_dgrid_allk(self, uwk, vwk):
        return self._firing("ext_vector_dgrid_allk", "D", [uwk, vwk],
                            self._bodies["D"])

    def ext_vector_cgrid(self, ucw, vcw):
        return self._firing("ext_vector_cgrid", "C", [ucw, vcw],
                            self._bodies["C"])

    def ext_vector_cgrid_allk(self, ucwk, vcwk):
        return self._firing("ext_vector_cgrid_allk", "C", [ucwk, vcwk],
                            self._bodies["C"])

    # barriers: flat fallback (O(state) gather) until the blend op is tiled
    def average_shared_edge_bgrid(self, xbw, ybw):
        return self._flat.average_shared_edge_bgrid(xbw, ybw)

    def average_shared_edge_cgrid(self, fxw, fyw):
        return self._flat.average_shared_edge_cgrid(fxw, fyw)

    def average_allflux_shared_edges(self, afxw, afyw):
        return self._flat.average_allflux_shared_edges(afxw, afyw)

    def refresh(self, bundle: dict) -> dict:
        """Seam pads of every window-stacked padded array rebuilt from the
        neighbours' blocks (the two ppermute rounds, no exchange body)."""
        lay = self.lay
        keys = [k for k, v in bundle.items()
                if hasattr(v, "ndim")
                and horizontal_axes(lay, v.shape, lay.nb) is not None]
        if not keys:
            return dict(bundle)
        outs = self._firing("refresh", None, [bundle[k] for k in keys])
        outs = outs if isinstance(outs, tuple) else (outs,)
        out = dict(bundle)
        out.update(zip(keys, outs))
        return out


def attach_window_spmd_comm(ctx, mesh, pad: int):
    """Window ctx + SPMD comm for ``ctx`` on the ``(6, kt, kt)`` mesh
    (kt from the mesh; padded partition).  Returns ``(window_ctx,
    comm)``; the comm is attached as ``ctx.tab.window_comm``."""
    kt = int(mesh.devices.shape[1])
    lay = build_window_layout(ctx.n, ctx.ng, kt, pad, "padded")
    comm = DuoWindowSpmdComm(lay, ctx.tab, mesh)
    ctx.tab.window_comm = comm
    return build_window_ctx(ctx, lay), comm
