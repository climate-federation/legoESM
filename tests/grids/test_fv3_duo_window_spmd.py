"""SPMD window comm (fv3_duo_window_spmd) on 24 virtual CPU devices.

Run ALONE (the device count must be set before jax is imported):
    XLA_FLAGS=--xla_force_host_platform_device_count=24 pytest this_file

Each test compares one SPMD firing against the certified flat impl seen
through the single-device window bundle, bitwise, on random fields.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("XLA_FLAGS",
                      "--xla_force_host_platform_device_count=24")
jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh  # noqa: E402

N, NG, KM, KT, PAD = 24, 3, 5, 2, 4


@pytest.fixture(scope="module")
def setup():
    if jax.device_count() < 6 * KT * KT:
        pytest.skip(f"need {6 * KT * KT} devices (XLA_FLAGS "
                    f"--xla_force_host_platform_device_count)")
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_duo_windows import build_window_layout
    from legoesm.grids.fv3_duo_window_spmd import DuoWindowSpmdComm

    grid = create_fv3_duo_grid(N)
    model = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, hydrostatic=True))
    ctx = model._ctx_jax
    mesh = Mesh(np.array(jax.devices()[:6 * KT * KT]).reshape(6, KT, KT),
                ("face", "tile_i", "tile_j"))
    lay = build_window_layout(N, NG, KT, PAD, "padded")
    comm = DuoWindowSpmdComm(lay, ctx.tab, mesh)
    return ctx, lay, comm


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _windows(lay, comm, x6):
    from legoesm.grids.fv3_duo_windows import gather_windows
    return jax.device_put(gather_windows(lay, x6), comm.sharding)


def _bytes_equal(a, b):
    """strictly bitwise: np.array_equal treats +0.0 == -0.0 (codex)"""
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        a.view(np.uint8), b.view(np.uint8))


def _block_mask(lay, w, extents):
    """cells inside window ``w``'s BLOCK (owned + the shared node row a
    neighbour also stores) -- everything else is pad"""
    M = N + 2 * NG
    m = np.zeros(tuple(lay.W + (e - M) if e >= M else
                       lay.n_w + (e - N) for e in extents), bool)
    sl = []
    for ax, t in ((0, (w % (lay.kt ** 2)) // lay.kt), (1, w % lay.kt)):
        e_f = extents[ax]
        shift = (lay.m_a - e_f + 1) // 2
        e = e_f - (M if e_f >= M else N)
        lo = lay.block_start(t) - shift - lay.origins[w][1 + ax]
        sl.append(slice(max(lo, 0), min(lo + lay.nl + e, m.shape[ax])))
    m[sl[0], sl[1]] = True
    return m


def _refresh_check(lay, comm, packed):
    from legoesm.grids.fv3_duo_windows import gather_windows
    comm.pack_pad_refresh = packed
    M = N + 2 * NG
    block_mask = lambda w, extents: _block_mask(lay, w, extents)

    try:
        for extents in [(M, M), (M + 1, M), (M, M + 1), (N, N), (N + 1, N)]:
            bundle, want = {}, {}
            # two f64 arrays with DISTINCT values and, for "x2", the OTHER
            # node-axis extent: an equal-shape segment swap inside the f64
            # group or an offset slip between unequal slabs (GLM/codex
            # 2026-09-06) delivers the wrong array's slab and is caught
            ext2 = (extents[0] + (1 if extents[0] in (M, N) else -1),
                    extents[1])
            for name, dt, seed, ext in (("x", np.float64, 1, extents),
                                        ("x2", np.float64, 4, ext2),
                                        ("y", np.float32, 2, extents),
                                        ("z", np.int32, 3, extents)):
                x6 = _rand((6,) + ext + (KM,), seed)
                x6 = (x6 * 1000).astype(dt) if dt == np.int32 else x6.astype(dt)
                xw = np.asarray(gather_windows(lay, x6))
                poisoned = xw.copy()
                fill = np.nan if dt != np.int32 else -999999
                for w in range(lay.nb):
                    poisoned[w][~block_mask(w, ext)] = fill  # every pad
                bundle[name] = jax.device_put(jnp.asarray(poisoned),
                                              comm.sharding)
                want[name] = xw
            out = jax.jit(lambda b: comm.refresh(b))(bundle)
            for name, xw in want.items():
                o = np.asarray(out[name])
                assert _bytes_equal(o, xw), (packed, name, extents,
                                             (o != xw).sum())
    finally:
        comm.pack_pad_refresh = False


@pytest.mark.parametrize("packed", [False, True])
def test_refresh_fills_every_pad_from_the_neighbours(setup, packed):
    """Pads poisoned with NaN; after the 2-round refresh every window
    equals the flat array's window slab BYTE for byte (blocks untouched,
    pads = neighbours' blocks, corners included, edge windows' wider
    pads).  ``packed``: the M8-A one-message-per-direction refresh, which
    must deliver the same bytes to the same cells -- here on a bundle of
    two f64 arrays with distinct values and different node-axis extents
    (unequal slabs in ONE dtype group) plus f32 and int32."""
    ctx, lay, comm = setup
    _refresh_check(lay, comm, packed)


def test_packed_refresh_segment_rotation_is_caught(setup, monkeypatch):
    """Non-vacuity for the packed arm (codex 2026-09-06): rotating the
    packed vector by a third of its length before it is split (every
    segment then lands in the wrong array/offset) must FAIL the check
    above.  The unpacked arm's 3-D slabs are left alone."""
    import jax
    ctx, lay, comm = setup
    real = jax.lax.ppermute

    def rotated(x, axis_name, perm):
        y = real(x, axis_name, perm)
        return jnp.roll(y, y.shape[0] // 3) if y.ndim == 1 else y
    monkeypatch.setattr(jax.lax, "ppermute", rotated)
    with pytest.raises(AssertionError):
        _refresh_check(lay, comm, True)


@pytest.mark.parametrize("stag", ["A", "B"])
def test_ext_scalar_matches_flat_impl(setup, stag):
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_allk
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    assert ctx.tab.window_comm is None        # certified path below
    M = N + 2 * NG + (1 if stag == "B" else 0)
    x6 = _rand((6, M, M, KM), 2)
    ref = np.asarray(gather_windows(lay, ext_scalar_sixface_allk(
        x6, ctx.tab, stag)))
    out = jax.jit(lambda a: comm.ext_scalar_allk(a, stag))(
        _windows(lay, comm, x6))
    out = np.asarray(out)
    bad = (out != ref) & ~(np.isnan(out) & np.isnan(ref))
    assert not bad.any(), f"{bad.sum()} cells differ; first {np.argwhere(bad)[:5]}"


@pytest.mark.parametrize("grid", ["D", "C"])
def test_ext_vector_matches_flat_impl(setup, grid):
    from legoesm.grids.fv3_duo_halos import (
        ext_vector_dgrid_sixface_allk, ext_vector_cgrid_sixface_allk)
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    assert ctx.tab.window_comm is None
    M = N + 2 * NG
    if grid == "D":
        u6, v6 = _rand((6, M, M + 1, KM), 3), _rand((6, M + 1, M, KM), 4)
        ru, rv = ext_vector_dgrid_sixface_allk(u6, v6, ctx.tab)
        fn = comm.ext_vector_dgrid_allk
    else:
        u6, v6 = _rand((6, M + 1, M, KM), 5), _rand((6, M, M + 1, KM), 6)
        ru, rv = ext_vector_cgrid_sixface_allk(u6, v6, ctx.tab)
        fn = comm.ext_vector_cgrid_allk
    ou, ov = jax.jit(fn)(_windows(lay, comm, u6), _windows(lay, comm, v6))
    for o, r, nm in ((ou, ru, "u"), (ov, rv, "v")):
        o = np.asarray(o)
        r = np.asarray(gather_windows(lay, r))
        bad = (o != r) & ~(np.isnan(o) & np.isnan(r))
        assert not bad.any(), f"{nm}: {bad.sum()} cells differ"


def test_sabotaged_arms_fail(setup):
    """Non-vacuity (codex 2026-09-04): with the pad exchange replaced by
    an identity the refresh test's poison SURVIVES, and with the exchange
    body replaced by an identity the scalar firing DIFFERS from the flat
    impl -- so the two tests above cannot pass on a no-op."""
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_allk
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    M = N + 2 * NG
    x6 = _rand((6, M, M, KM), 7)
    xw = np.asarray(gather_windows(lay, x6))
    poisoned = xw.copy()
    poisoned[:, 0, :, :] = np.nan            # a pad row of every window
    orig = comm._pad_exchange
    comm._pad_exchange = lambda arrs, full=True: arrs
    try:
        out = np.asarray(jax.jit(lambda a: comm.refresh({"x": a})["x"])(
            jax.device_put(jnp.asarray(poisoned), comm.sharding)))
    finally:
        comm._pad_exchange = orig
    assert np.isnan(out).any(), "identity pad exchange still cleared the poison"

    ref = np.asarray(gather_windows(lay, ext_scalar_sixface_allk(
        x6, ctx.tab, "A")))
    body = comm._bodies["A"]
    comm._bodies["A"] = lambda blk: blk
    try:
        out = np.asarray(jax.jit(lambda a: comm.ext_scalar_allk(a, "A"))(
            _windows(lay, comm, x6)))
    finally:
        comm._bodies["A"] = body
    bad = (out != ref) & ~(np.isnan(out) & np.isnan(ref))
    assert bad.any(), "identity exchange body still matched the flat impl"


def test_barrier_blend_as_stencil_is_bitwise(setup):
    """The exact-arithmetic claim behind the tiled barriers, on the REAL
    tables: 0.5*(a + s*b) == 0.5*a + (0.5*s)*b bitwise (random fields,
    both blends), evaluated the way the two runtimes evaluate them."""
    from legoesm.grids.fv3_duo_halos import FlatLayout
    from legoesm.grids.fv3_duo_spmd import blend_as_stencil
    ctx, lay, comm = setup
    tab = ctx.tab
    rng = np.random.default_rng(11)
    for bl, lay_from in ((tab.avg_b, tab.lay_bb), (tab.avg_c, tab.lay_fx)):
        flat = rng.standard_normal(lay_from.total)
        st = blend_as_stencil(bl, lay_from, lay_from, 0, "t")   # no shift
        ref = flat.copy()
        ref[bl.dst] = 0.5 * (flat[bl.dst] + bl.sign * flat[bl.src])
        got = flat.copy()
        v = flat[st.src]                                     # (n, 2)
        got[st.dst] = st.w[:, 0] * v[:, 0] + st.w[:, 1] * v[:, 1]
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("kind", ["bgrid", "cgrid", "allflux"])
def test_barriers_match_flat_impl(setup, kind):
    from legoesm.grids.fv3_duo_halos import (
        average_shared_edge_bgrid, average_shared_edge_cgrid,
        average_allflux_shared_edges)
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    assert ctx.tab.window_comm is None
    npx = N + 1
    # the flat barriers are per-level 2-D (the 3-D phases loop k)
    if kind == "bgrid":
        a6, b6 = _rand((6, npx, npx), 8), _rand((6, npx, npx), 9)
        ra, rb = average_shared_edge_bgrid(a6, b6, ctx.tab)
        fn = comm.average_shared_edge_bgrid
    elif kind == "cgrid":
        a6, b6 = _rand((6, npx, N), 10), _rand((6, N, npx), 12)
        ra, rb = average_shared_edge_cgrid(a6, b6, ctx.tab)
        fn = comm.average_shared_edge_cgrid
    else:
        ns = 4 + int(ctx.tab.nq)
        a6, b6 = _rand((6, npx, N, ns), 13), _rand((6, N, npx, ns), 14)
        ra, rb = average_allflux_shared_edges(a6, b6, ctx.tab)
        fn = comm.average_allflux_shared_edges
    oa, ob = jax.jit(fn)(_windows(lay, comm, a6), _windows(lay, comm, b6))
    for o, r, nm in ((oa, ra, "x"), (ob, rb, "y")):
        o = np.asarray(o)
        r = np.asarray(gather_windows(lay, r))
        bad = (o != r) & ~(np.isnan(o) & np.isnan(r))
        assert not bad.any(), f"{kind} {nm}: {bad.sum()} cells differ"


def test_sabotaged_barrier_fails(setup):
    """Non-vacuity for the barrier parity test: a body that drops the
    sign (weights 0.5, 0.5 instead of 0.5, 0.5*sign) must DIFFER."""
    from legoesm.grids.fv3_duo_halos import average_shared_edge_bgrid
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    npx = N + 1
    a6, b6 = _rand((6, npx, npx), 21), _rand((6, npx, npx), 22)
    ra, rb = average_shared_edge_bgrid(a6, b6, ctx.tab)
    body = comm._bodies["avg_b"]
    comm._bodies["avg_b"] = lambda x, y: (x, y)          # no blend at all
    try:
        oa, ob = jax.jit(comm.average_shared_edge_bgrid)(
            _windows(lay, comm, a6), _windows(lay, comm, b6))
    finally:
        comm._bodies["avg_b"] = body
    ra = np.asarray(gather_windows(lay, ra))
    assert (np.asarray(oa) != ra).any(), "identity barrier still matched"


def test_band_refresh_touches_only_the_face_edge_bands(setup):
    """M8-C contract (codex 2026-09-06): with ``refresh_band = bw`` a
    firing's refresh rewrites ONLY the seam pads inside the four
    face-edge bands of depth ``bw`` (from the neighbours, byte for byte
    as the full refresh would) and leaves every other pad cell exactly
    as it was; the substep-entry refresh (body None) stays full."""
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    M = N + 2 * NG
    x6 = _rand((6, M, M, KM), 31)
    xw = np.asarray(gather_windows(lay, x6))
    poisoned = xw.copy()
    for w in range(lay.nb):
        poisoned[w][~_block_mask(lay, w, (M, M))] = -7.0   # every pad cell
    win = jax.device_put(jnp.asarray(poisoned), comm.sharding)
    # substep-entry refresh: full, in both modes
    comm.refresh_band = 4
    try:
        full = np.asarray(jax.jit(lambda a: comm.refresh({"x": a})["x"])(win))
        assert _bytes_equal(full, xw)
        # a FIRING's band refresh: drive _pad_exchange(full=False) through
        # the identity body so only the refresh acts
        band = np.asarray(jax.jit(lambda a: comm._firing(
            "probe", "A", [a], body=lambda blk: blk))(win))
    finally:
        comm.refresh_band = None
    bw = 4
    W = lay.W
    bandcol = np.zeros((W, W), bool)
    bandcol[:, :bw] = bandcol[:, -bw:] = True   # west/east column bands
    bandrow = np.zeros((W, W), bool)
    bandrow[:bw, :] = bandrow[-bw:, :] = True   # north/south row bands
    kt = lay.kt
    must = np.zeros(band.shape, bool)      # pads that MUST come back right
    anyb = np.zeros(band.shape, bool)      # cells a band round may touch
    for w in range(lay.nb):
        ti, tj = (w % (kt * kt)) // kt, w % kt
        blk = _block_mask(lay, w, (M, M))
        padrow = ~blk.any(axis=1)[:, None] & np.ones((1, W), bool)
        padcol = ~blk.any(axis=0)[None, :] & np.ones((W, 1), bool)
        # a band is a FACE-EDGE band only where this window is flush with
        # that face edge; elsewhere it is pad-on-pad (whatever the
        # neighbour's pad held), never consumed
        flushcol = np.zeros((W, W), bool)
        if tj == 0:
            flushcol[:, :bw] = True
        if tj == kt - 1:
            flushcol[:, -bw:] = True
        flushrow = np.zeros((W, W), bool)
        if ti == 0:
            flushrow[:bw, :] = True
        if ti == kt - 1:
            flushrow[-bw:, :] = True
        # round 1 rewrites the seam (pad) ROWS inside the column bands,
        # round 2 the seam (pad) COLUMNS inside the row bands
        must[w] = ((padrow & flushcol) | (padcol & flushrow))[:, :, None]
        anyb[w] = ((padrow & bandcol) | (padcol & bandrow))[:, :, None]
    assert _bytes_equal(band[must], xw[must])          # face-edge bands right
    assert _bytes_equal(band[~anyb], poisoned[~anyb])  # the rest untouched
    assert (band != full).any(), "band refresh is not narrower than full"
