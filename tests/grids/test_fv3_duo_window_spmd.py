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


def test_refresh_fills_every_pad_from_the_neighbours(setup):
    """Pads poisoned with NaN; after the 2-round refresh every window
    equals the flat array's window slab bitwise (blocks untouched, pads
    = neighbours' blocks, corners included, edge windows' wider pads)."""
    from legoesm.grids.fv3_duo_windows import gather_windows
    ctx, lay, comm = setup
    M = N + 2 * NG

    def block_mask(w, extents):
        """cells inside the tile's BLOCK (owned + the shared node row a
        neighbour also stores) -- everything else is pad"""
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

    for extents in [(M, M), (M + 1, M), (M, M + 1), (N, N), (N + 1, N)]:
        x6 = _rand((6,) + extents + (KM,), 1)
        xw = np.asarray(gather_windows(lay, x6))
        poisoned = xw.copy()
        for w in range(lay.nb):
            poisoned[w][~block_mask(w, extents)] = np.nan   # every pad cell
        out = jax.jit(lambda a: comm.refresh({"x": a})["x"])(
            jax.device_put(jnp.asarray(poisoned), comm.sharding))
        out = np.asarray(out)
        assert np.array_equal(out, xw), (extents, np.isnan(out).sum())


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
    comm._pad_exchange = lambda arrs: arrs
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
