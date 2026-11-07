"""Window layout invariants for the DUO tiled port (fv3_duo_windows).

These are layout tests, not physics: every flat cell has exactly one
owner, a scatter of a gather is the identity, a masked gather touches
nothing outside its mask, and kt=1 is the whole face byte for byte.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

from legoesm.grids.fv3_duo_windows import (  # noqa: E402
    build_window_layout, gather_windows, scatter_owned, gather_masked,
    horizontal_axes, window_axis_extent)

N, NG = 24, 3


def _rand(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


@pytest.mark.parametrize("partition,kt,pad", [
    ("compute", 1, 3), ("compute", 2, 5), ("compute", 2, 8),
    ("compute", 3, 5), ("padded", 1, 99), ("padded", 2, 5), ("padded", 3, 5),
    ("padded", 5, 4)])
@pytest.mark.parametrize("extents", [(N + 2 * NG, N + 2 * NG),
                                     (N + 2 * NG + 1, N + 2 * NG),
                                     (N + 2 * NG, N + 2 * NG + 1),
                                     (N + 2 * NG + 1, N + 2 * NG + 1),
                                     (N, N), (N + 1, N), (N + 2, N + 2)])
def test_scatter_of_gather_is_identity(partition, kt, pad, extents):
    lay = build_window_layout(N, NG, kt, pad, partition)
    x = _rand((6,) + extents + (4,))
    xw = gather_windows(lay, x)
    assert xw.shape == (lay.nb, window_axis_extent(lay, extents[0]),
                        window_axis_extent(lay, extents[1]), 4)
    back = scatter_owned(lay, xw)
    assert np.array_equal(np.asarray(back), np.asarray(x))   # bitwise


@pytest.mark.parametrize("partition,kt", [("compute", 3), ("padded", 3),
                                          ("padded", 5)])
def test_every_flat_cell_has_exactly_one_owner(partition, kt):
    from legoesm.grids.fv3_duo_windows import _owner_mask, _index
    lay = build_window_layout(N, NG, kt, 5, partition)
    for extents in [(N + 2 * NG, N + 2 * NG + 1), (N + 1, N), (N + 2, N)]:
        count = np.zeros((6,) + extents, int)
        for w in range(lay.nb):
            idx = _index(lay, w, (6,) + extents, (1, 2))
            count[idx] += _owner_mask(lay, w, *extents)
        assert count.min() == 1 and count.max() == 1


def test_kt1_is_the_whole_face():
    lay = build_window_layout(N, NG, 1, 99)          # pad is forced to ng
    assert lay.W == N + 2 * NG and lay.nb == 6
    lay = build_window_layout(N, NG, 1, 99, "padded")   # pad forced to 0
    assert lay.pad == 0 and lay.W == N + 2 * NG and lay.nb == 6
    x = _rand((6, N + 2 * NG, N + 2 * NG + 1, 3))
    assert np.array_equal(np.asarray(gather_windows(lay, x)), np.asarray(x))


def test_gather_masked_touches_only_the_mask():
    lay = build_window_layout(N, NG, 2, 6)
    x = _rand((6, N + 2 * NG, N + 2 * NG, 2))
    xw = gather_windows(lay, x)
    y = x + 1.0
    mask = np.zeros((6, N + 2 * NG, N + 2 * NG), bool)
    mask[:, :NG, :] = True                           # the west ring
    out = np.asarray(gather_masked(lay, xw, y, mask))
    yw = np.asarray(gather_windows(lay, y))
    xw = np.asarray(xw)
    mw = np.asarray(gather_windows(lay, jnp.asarray(mask)[..., None]))[..., 0]
    assert np.array_equal(out[mw], yw[mw])
    assert np.array_equal(out[~mw], xw[~mw])
    assert mw.any() and (~mw).any()


def test_ikj_layout_round_trip():
    lay = build_window_layout(N, NG, 2, 5)
    x = _rand((6, N + 2, 7, N + 2))                  # pe: (i, k, j)
    assert horizontal_axes(lay, x.shape, 6) == (1, 3)
    xw = gather_windows(lay, x)
    assert xw.shape[2] == 7
    assert np.array_equal(np.asarray(scatter_owned(lay, xw)), np.asarray(x))


def test_window_too_wide_refused():
    with pytest.raises(ValueError, match="exceeds the padded face"):
        build_window_layout(N, NG, 2, 12)            # 12 + 24 > 30


def test_padded_partition_blocks_match_the_blocked_layout():
    """padded partition: block t = padded rows [t*nl, t*nl+nl+e) -- the
    fv3_duo_spmd.to_blocked layout; edge windows sit flush."""
    from legoesm.grids.fv3_duo_spmd import to_blocked
    lay = build_window_layout(N, NG, 3, 4, "padded")
    assert lay.nl == (N + 2 * NG) // 3 and lay.block_start(1) == lay.nl
    assert lay.origins[0][1] == 0 and lay.origins[-1][1] == lay.m_a - lay.W
    x = np.random.default_rng(1).standard_normal((6, N + 2 * NG + 1,
                                                  N + 2 * NG, 2))
    xb = to_blocked(x, 3, lay.nl)                     # (6, 3*(nl+1), 3*nl, 2)
    xw = np.asarray(gather_windows(lay, jnp.asarray(x)))
    for w, (face, oi, oj) in enumerate(lay.origins):
        ti, tj = (w % 9) // 3, w % 3
        blk = xb[face, ti * (lay.nl + 1):(ti + 1) * (lay.nl + 1),
                 tj * lay.nl:(tj + 1) * lay.nl]
        li, lj = ti * lay.nl - oi, tj * lay.nl - oj
        assert np.array_equal(xw[w, li:li + lay.nl + 1, lj:lj + lay.nl], blk)
    with pytest.raises(ValueError, match="not divisible"):
        build_window_layout(N, NG, 4, 4, "padded")   # 30 % 4


@pytest.mark.parametrize("kind", ["bgrid", "cgrid", "allflux"])
def test_window_barriers_with_a_level_axis_match_the_per_level_impl(kind):
    """K>1 parity for the single-device window bundle (codex 2026-09-07):
    the callers hand the barriers the WHOLE level stack in one call, and
    the certified impls are per-level (they flatten every cell axis into
    one index), so the bundle must map the level axis -- an unmapped one
    lands inside the flattening and aliases levels."""
    from legoesm.grids.fv3_duo_halos import (
        average_shared_edge_bgrid_impl, average_shared_edge_cgrid_impl,
        average_allflux_shared_edges_impl, build_jax_duo_halo_tables)
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_duo_windows import attach_window_comm
    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context

    km = 3
    grid = create_fv3_duo_grid(N)
    ctx = build_jax_duo_stepper_context(grid.ctx_np)
    wctx, comm = attach_window_comm(ctx, 2, 5, "padded")
    npx, n = N + 1, N
    rng = np.random.default_rng(5)
    if kind == "allflux":
        ns = 4 + int(ctx.tab.nq)
        a6 = jnp.asarray(rng.standard_normal((6, npx, n, km, ns)))
        b6 = jnp.asarray(rng.standard_normal((6, n, npx, km, ns)))
        impl, fn, kax = (average_allflux_shared_edges_impl,
                         comm.average_allflux_shared_edges, 3)
    else:
        i0, j0 = ((npx, npx) if kind == "bgrid" else (npx, n))
        i1, j1 = ((npx, npx) if kind == "bgrid" else (n, npx))
        a6 = jnp.asarray(rng.standard_normal((6, i0, j0, km)))
        b6 = jnp.asarray(rng.standard_normal((6, i1, j1, km)))
        impl = (average_shared_edge_bgrid_impl if kind == "bgrid"
                else average_shared_edge_cgrid_impl)
        fn = (comm.average_shared_edge_bgrid if kind == "bgrid"
              else comm.average_shared_edge_cgrid)
        kax = -1
    # reference: the certified impl, ONE LEVEL AT A TIME
    ra, rb = [], []
    for k in range(km):
        sa = (slice(None),) * 3 + (k,) if kax == 3 else (Ellipsis, k)
        x, y = impl(a6[sa], b6[sa], ctx.tab)
        ra.append(x)
        rb.append(y)
    ra = jnp.stack(ra, axis=kax if kax > 0 else -1)
    rb = jnp.stack(rb, axis=kax if kax > 0 else -1)
    from legoesm.grids.fv3_duo_windows import gather_windows, scatter_owned
    aw = gather_windows(comm.lay, a6)
    bw = gather_windows(comm.lay, b6)
    oa, ob = fn(aw, bw)
    for o, r, nm in ((oa, ra, "x"), (ob, rb, "y")):
        got = np.asarray(scatter_owned(comm.lay, np.asarray(o), np))
        want = np.asarray(r)
        bad = (got != want) & ~(np.isnan(got) & np.isnan(want))
        assert not bad.any(), f"{kind} {nm}: {bad.sum()} cells differ"


@pytest.mark.parametrize("nq", [1, 2])
def test_barrier_stacks_follow_the_tracer_count(nq):
    """nq>1 plumbing (2026-09-10): the halo tables, the d_sw1 flux stack
    and the barrier guard must agree on 4+nq slots, and the barrier must
    blend exactly the oracle's selection (iq==1, iq>=4: delp, pt and
    every tracer).  Default stays nq=1; nothing here changes a deck."""
    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_duo_halos import average_allflux_shared_edges

    grid = create_fv3_duo_grid(N)
    ctx = build_jax_duo_stepper_context(grid.ctx_np, nq=nq)
    assert int(ctx.tab.nq) == nq
    # dyn_core.F90:856 -- iq == 1 (delp), iq >= 4 (pt and the tracers)
    assert ctx.tab.allflux_slots.tolist() == [0] + list(range(3, 4 + nq))
    npx, n = N + 1, N
    ns = 4 + nq
    rng = np.random.default_rng(3)
    afx = jnp.asarray(rng.standard_normal((6, npx, n, ns)))
    afy = jnp.asarray(rng.standard_normal((6, n, npx, ns)))
    ox, oy = average_allflux_shared_edges(afx, afy, ctx.tab)
    assert ox.shape == afx.shape and oy.shape == afy.shape
    # the UNSELECTED slots (2 = w, 3 = q_con in the oracle's numbering)
    # come back byte-identical; the selected ones are blended somewhere
    unsel = [s for s in range(ns) if s not in ctx.tab.allflux_slots.tolist()]
    assert unsel, "every slot selected -- the exclusion is gone"
    for s in unsel:
        assert np.array_equal(np.asarray(ox[..., s]), np.asarray(afx[..., s]))
    assert not np.array_equal(np.asarray(ox[..., ctx.tab.allflux_slots[-1]]),
                              np.asarray(afx[..., ctx.tab.allflux_slots[-1]]))
    # a stack built for a DIFFERENT nq must be refused, not blended
    bad = jnp.asarray(rng.standard_normal((6, npx, n, ns + 1)))
    bady = jnp.asarray(rng.standard_normal((6, n, npx, ns + 1)))
    with pytest.raises(ValueError, match="slot axis"):
        average_allflux_shared_edges(bad, bady, ctx.tab)
