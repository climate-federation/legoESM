"""Duo SPMD depth census: decode self-test, the measured census, the gate.

The census is the SPMD ring design's load-bearing number (every cross-face
read must land inside the gathered ring), so this file pins it and proves
the instrument can fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_duo_spmd import (
    _self_test,
    assert_ring_width_covers,
    census_cross_face_depth,
)

N, NG = 12, 3


@pytest.fixture(scope="module")
def duo_ctx():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


@pytest.fixture(scope="module")
def tab(duo_ctx):
    from legoesm.grids.fv3_duo_halos import build_jax_duo_halo_tables
    return build_jax_duo_halo_tables(duo_ctx["ectx"], duo_ctx["gs6"], nq=1)


def test_census_matches_the_measured_depth(tab):
    """C48 measured 7 (jobs 9493433-38); the builders' strip arithmetic is
    n-independent (codex-read: neighbor maps are affine, halo widths come
    from ng/ngp), so N=12 at the same ng=3/ngp=4 must census the same.
    A different number here is a FINDING about n-dependence, not a
    tolerance to relax."""
    assert census_cross_face_depth(tab) == 7


def test_gate_refuses_a_narrow_ring(tab):
    with pytest.raises(ValueError, match="does not cover"):
        assert_ring_width_covers(tab, 7)   # == depth: no margin, refused
    assert assert_ring_width_covers(tab, 8) == 7


def test_decode_self_test_can_fail(tab):
    """Synthetic violation (non-vacuity): a layout whose idx LIES must be
    caught by the self-test -- otherwise a broken decode could bless a
    broken ring width."""
    class _Lying:
        shapes = tab.lay_a.shapes
        bases = tab.lay_a.bases
        total = tab.lay_a.total

        @staticmethod
        def idx(k, face, i0, j0):
            return tab.lay_a.idx(k, face, i0, j0) + 1   # off by one

    with pytest.raises(AssertionError, match="self-test"):
        _self_test(_Lying())


def test_census_refuses_an_unknown_op_family(tab):
    """A table op without 'dst' must be a hard error, never a skip."""
    from legoesm.grids.fv3_duo_spmd import _max_cross_depth

    class _Alien:
        pass

    with pytest.raises(TypeError, match="fail closed"):
        _max_cross_depth(_Alien(), tab.lay_a)


# ---- ring exchange: bitwise parity vs the certified exchange + poison ----
#
# Needs >= 6 devices: the sbatch wrapper sets
# XLA_FLAGS=--xla_force_host_platform_device_count=6 BEFORE jax imports.

def _devices_or_skip(n):
    import jax
    if len(jax.devices()) < n:
        pytest.skip(f"needs {n} devices (set "
                    f"xla_force_host_platform_device_count)")
    return jax.devices()[:n]


@pytest.mark.parametrize("n_dev", [2, 3, 6])
@pytest.mark.parametrize("stag", ["A", "B"])
def test_ring_exchange_matches_certified_bitwise(tab, n_dev, stag):
    """The ring exchange copies the same VALUES the certified exchange
    reads (borders are copies, own faces are originals), then runs the
    SAME table program -- own-face outputs must be BITWISE equal.  A
    tolerance here would hide an index shift."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_scalar_sixface

    devs = _devices_or_skip(n_dev)
    m = tab.n + 2 * tab.ng + (1 if stag == "B" else 0)
    rng = np.random.default_rng(3)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ref = np.asarray(ext_scalar_sixface(f6, tab, stag))

    mesh = Mesh(np.array(devs), ("face",))
    fn, depth = make_ring_ext_scalar_sixface(tab, stag, mesh)
    assert depth == 7
    f6s = jax.device_put(f6, NamedSharding(mesh, P("face")))
    got = np.asarray(fn(f6s))
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"ring exchange diverged: |d|max="
        f"{np.abs(got - ref).max():.3e}")


@pytest.mark.parametrize("stag", ["A", "B"])
def test_ring_exchange_poison_proves_no_out_of_ring_read(tab, stag):
    """poison=True fills every non-border, non-own cell with NaN; the
    output must STILL match the certified exchange bitwise.  One NaN in
    the output = a read outside the census's ring -- the design's failure
    mode, made loud (GLM: zeros are indistinguishable from real data)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_scalar_sixface

    devs = _devices_or_skip(6)
    m = tab.n + 2 * tab.ng + (1 if stag == "B" else 0)
    rng = np.random.default_rng(4)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ref = np.asarray(ext_scalar_sixface(f6, tab, stag))

    mesh = Mesh(np.array(devs), ("face",))
    fn, _ = make_ring_ext_scalar_sixface(tab, stag, mesh, poison=True)
    f6s = jax.device_put(f6, NamedSharding(mesh, P("face")))
    got = np.asarray(fn(f6s))
    assert np.isfinite(got).all(), \
        "NaN reached the output: the tables read outside the ring"
    assert np.array_equal(got, ref)


# ---- vector ring exchange: bitwise parity + poison, D and C flows ----

def _vec_fixture(tab, grid, seed):
    ma = tab.n + 2 * tab.ng
    mb = ma + 1
    rng = np.random.default_rng(seed)
    if grid == "D":
        return (rng.standard_normal((6, ma, mb)),
                rng.standard_normal((6, mb, ma)))
    return (rng.standard_normal((6, mb, ma)),
            rng.standard_normal((6, ma, mb)))


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("n_dev", [2, 3, 6])
@pytest.mark.parametrize("grid", ["D", "C"])
def test_ring_vector_matches_certified_bitwise(tab, n_dev, grid, seed):
    """Full 7-step vector flow under the ring exchange must equal the
    certified single-device flow BITWISE on own faces: other faces' ring
    c2l values are recomputed from their gathered u/v rings (same inputs,
    same op).  NaN slots (c2l band edges, pack_p1) must match too --
    array_equal with equal_nan pins the whole layout."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface, ext_vector_dgrid_sixface)
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_vector_sixface

    devs = _devices_or_skip(n_dev)
    u6, v6 = map(jnp.asarray, _vec_fixture(tab, grid, seed))
    flow = (ext_vector_dgrid_sixface if grid == "D"
            else ext_vector_cgrid_sixface)
    ru, rv = flow(u6, v6, tab)
    ru, rv = np.asarray(ru), np.asarray(rv)

    mesh = Mesh(np.array(devs), ("face",))
    fn, depth = make_ring_ext_vector_sixface(tab, grid, mesh)
    assert depth == 7
    sh = NamedSharding(mesh, P("face"))
    gu, gv = fn(jax.device_put(u6, sh), jax.device_put(v6, sh))
    gu, gv = np.asarray(gu), np.asarray(gv)
    assert np.array_equal(gu, ru, equal_nan=True), (
        f"u diverged: |d|max={np.nanmax(np.abs(gu - ru)):.3e}")
    assert np.array_equal(gv, rv, equal_nan=True), (
        f"v diverged: |d|max={np.nanmax(np.abs(gv - rv)):.3e}")


@pytest.mark.parametrize("grid", ["D", "C"])
def test_ring_vector_poison_proves_ring_sufficiency(tab, grid):
    """poison=True: non-ring, non-own cells are NaN.  The composed flow's
    EFFECTIVE cross-face footprint (strips + c2l-feeding-geo, GLM's
    compounding-depth hole) must fit inside width 8 -- a NaN escaping
    into a certified-finite own-face cell fails array_equal loudly."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface, ext_vector_dgrid_sixface)
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_vector_sixface

    devs = _devices_or_skip(6)
    u6, v6 = map(jnp.asarray, _vec_fixture(tab, grid, 12))
    flow = (ext_vector_dgrid_sixface if grid == "D"
            else ext_vector_cgrid_sixface)
    ru, rv = flow(u6, v6, tab)

    mesh = Mesh(np.array(devs), ("face",))
    # Observability (codex): the comparison can only catch a wrong NaN
    # where the certified reference is FINITE -- pin that the reference
    # halo band genuinely is (its NaN slots are the structural c2l-band
    # ones only, a bounded count), so equal_nan cannot hide a defect.
    n_nan_u = int(np.isnan(np.asarray(ru)).sum())
    n_nan_v = int(np.isnan(np.asarray(rv)).sum())
    assert n_nan_u < np.asarray(ru).size // 4, "reference u mostly NaN"
    assert n_nan_v < np.asarray(rv).size // 4, "reference v mostly NaN"
    fn, _ = make_ring_ext_vector_sixface(tab, grid, mesh, poison=True)
    sh = NamedSharding(mesh, P("face"))
    gu, gv = fn(jax.device_put(u6, sh), jax.device_put(v6, sh))
    assert np.array_equal(np.asarray(gu), np.asarray(ru), equal_nan=True)
    assert np.array_equal(np.asarray(gv), np.asarray(rv), equal_nan=True)


def test_poison_instrument_can_fire(tab, monkeypatch):
    """NEGATIVE CONTROL (codex): with the border NARROWER than the tables
    read (mask width 4, gate bypassed by patching the mask builder), the
    poisoned exchange MUST diverge from the certified one -- proving the
    poison instrument can actually fail.  Without this, every poison
    pass above could be vacuous."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import legoesm.grids.fv3_duo_spmd as spmd
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface

    devs = _devices_or_skip(6)
    m = tab.n + 2 * tab.ng
    rng = np.random.default_rng(5)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ref = np.asarray(ext_scalar_sixface(f6, tab, "A"))

    real_mask = spmd._border_mask
    monkeypatch.setattr(spmd, "_border_mask",
                        lambda mm, w: real_mask(mm, 4))   # too narrow
    mesh = Mesh(np.array(devs), ("face",))
    fn, _ = spmd.make_ring_ext_scalar_sixface(tab, "A", mesh, poison=True)
    got = np.asarray(fn(jax.device_put(
        f6, NamedSharding(mesh, P("face")))))
    assert not np.array_equal(got, ref, equal_nan=True), (
        "narrow-border poison run still matched: the poison instrument "
        "cannot fire and every poison pass is vacuous")


# ---- M3 wiring: the public exchanges dispatch on tab.ring_comm ----

class _Sentinel(Exception):
    pass


def test_dispatch_default_path_is_the_impl(tab, monkeypatch):
    """With ring_comm=None (the builder's default) each public exchange
    must reach its *_impl body -- proven by a sentinel-raising impl, not
    by reading source."""
    import legoesm.grids.fv3_duo_halos as halos

    assert tab.ring_comm is None            # the builder's default
    for pub, impl in [
            (lambda: halos.ext_scalar_sixface(None, tab, "A"),
             "ext_scalar_sixface_impl"),
            (lambda: halos.ext_vector_dgrid_sixface(None, None, tab),
             "ext_vector_dgrid_sixface_impl"),
            (lambda: halos.ext_vector_cgrid_sixface(None, None, tab),
             "ext_vector_cgrid_sixface_impl")]:
        def _boom(*a, **k):
            raise _Sentinel
        monkeypatch.setattr(halos, impl, _boom)
        with pytest.raises(_Sentinel):
            pub()


def test_dispatch_routes_to_ring_comm(tab, monkeypatch):
    """With ring_comm set, each public exchange must call the matching
    ring method with the caller's arrays -- recorded on a fake ring, so
    no devices are needed."""
    import legoesm.grids.fv3_duo_halos as halos

    calls = []

    class _FakeRing:
        def ext_scalar(self, f6, stag):
            calls.append(("scalar", f6, stag))
            return "S"

        def ext_vector_dgrid(self, u6, v6):
            calls.append(("dgrid", u6, v6))
            return "D"

        def ext_vector_cgrid(self, uc6, vc6):
            calls.append(("cgrid", uc6, vc6))
            return "C"

    monkeypatch.setattr(tab, "ring_comm", _FakeRing())
    f, u, v = object(), object(), object()
    assert halos.ext_scalar_sixface(f, tab, "B") == "S"
    assert halos.ext_vector_dgrid_sixface(u, v, tab) == "D"
    assert halos.ext_vector_cgrid_sixface(u, v, tab) == "C"
    assert calls == [("scalar", f, "B"), ("dgrid", u, v),
                     ("cgrid", u, v)]


def test_ring_comm_refuses_unknown_stagger():
    """Dispatch hardening: an unknown stagger raises, never a silent
    default."""
    from legoesm.grids.fv3_duo_spmd import DuoRingComm

    rc = DuoRingComm()
    rc._scalar = {}
    with pytest.raises(ValueError, match="stagger"):
        rc.ext_scalar(None, "C")


def test_public_dispatch_runs_the_ring_no_recursion(tab, monkeypatch):
    """The behavioural recursion guard: build a real ring_comm, set it on
    tab, and call the PUBLIC exchanges on sharded arrays.  If the ring
    bodies routed back through the dispatchers, shard_map-inside-
    shard_map would raise -- so a bitwise-clean pass IS the guard test,
    for all three exchanges."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import legoesm.grids.fv3_duo_halos as halos
    from legoesm.grids.fv3_duo_spmd import build_ring_comm

    devs = _devices_or_skip(6)
    mesh = Mesh(np.array(devs), ("face",))
    sh = NamedSharding(mesh, P("face"))

    m = tab.n + 2 * tab.ng
    rng = np.random.default_rng(6)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ud, vd = map(jnp.asarray, _vec_fixture(tab, "D", 7))
    uc, vc = map(jnp.asarray, _vec_fixture(tab, "C", 8))

    # certified references from the impls (never dispatch)
    ref_s = np.asarray(halos.ext_scalar_sixface_impl(f6, tab, "A"))
    ref_du, ref_dv = (np.asarray(x) for x in
                      halos.ext_vector_dgrid_sixface_impl(ud, vd, tab))
    ref_cu, ref_cv = (np.asarray(x) for x in
                      halos.ext_vector_cgrid_sixface_impl(uc, vc, tab))

    monkeypatch.setattr(tab, "ring_comm", build_ring_comm(tab, mesh))
    got_s = np.asarray(halos.ext_scalar_sixface(
        jax.device_put(f6, sh), tab, "A"))
    got_du, got_dv = halos.ext_vector_dgrid_sixface(
        jax.device_put(ud, sh), jax.device_put(vd, sh), tab)
    got_cu, got_cv = halos.ext_vector_cgrid_sixface(
        jax.device_put(uc, sh), jax.device_put(vc, sh), tab)
    assert np.array_equal(got_s, ref_s)
    assert np.array_equal(np.asarray(got_du), ref_du, equal_nan=True)
    assert np.array_equal(np.asarray(got_dv), ref_dv, equal_nan=True)
    assert np.array_equal(np.asarray(got_cu), ref_cu, equal_nan=True)
    assert np.array_equal(np.asarray(got_cv), ref_cv, equal_nan=True)


def test_stepper_context_threads_the_mesh(duo_ctx):
    """build_jax_duo_stepper_context(spmd_mesh=...) attaches a ring_comm;
    the default builds ring_comm=None (certified path)."""
    import jax
    from jax.sharding import Mesh

    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context
    from legoesm.grids.fv3_duo_spmd import DuoRingComm

    jax.config.update("jax_enable_x64", True)   # the context's f64 gate
    devs = _devices_or_skip(2)
    ctx_default = build_jax_duo_stepper_context(duo_ctx)
    assert ctx_default.tab.ring_comm is None
    mesh = Mesh(np.array(devs), ("face",))
    ctx_ring = build_jax_duo_stepper_context(duo_ctx, spmd_mesh=mesh)
    assert isinstance(ctx_ring.tab.ring_comm, DuoRingComm)
    assert ctx_ring.tab.ring_comm.depth == 7


def test_model_knob_threads_the_mesh_without_mutating_the_bundle():
    """FV3DuoDynamicsModel(step_spmd_mesh=...) rebuilds a ring-enabled
    context; the SHARED grid bundle's certified context is untouched
    (mutating an identity-hashed static arg would leave a stale jit
    cache).  Construction only -- the sbatch probe steps it."""
    import jax
    from jax.sharding import Mesh

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_duo_spmd import DuoRingComm

    jax.config.update("jax_enable_x64", True)
    devs = _devices_or_skip(2)
    bundle = create_fv3_duo_grid(N)
    default = FV3DuoDynamicsModel(bundle, FV3DuoConfig())
    assert default._ctx_jax is bundle.ctx_jax
    mesh = Mesh(np.array(devs), ("face",))
    ring = FV3DuoDynamicsModel(bundle, FV3DuoConfig(),
                               step_spmd_mesh=mesh)
    assert isinstance(ring._ctx_jax.tab.ring_comm, DuoRingComm)
    assert ring._ctx_jax is not bundle.ctx_jax
    assert bundle.ctx_jax.tab.ring_comm is None


def test_build_ring_comm_refuses_wrong_axis_name(tab):
    """codex MINOR: the knob's contract is a single 'face' axis."""
    import jax
    from jax.sharding import Mesh
    from legoesm.grids.fv3_duo_spmd import build_ring_comm
    devs = _devices_or_skip(6)
    bad = Mesh(np.array(devs), ("tiles",))
    with pytest.raises(ValueError, match="face"):
        build_ring_comm(tab, bad)


# ---- v2a k-batched exchanges: certified relocation, ring parity, poison ----

K_BATCH = 3          # trailing batch size for the allk tests


def _scalar_stack_fixture(tab, stag, seed):
    m = tab.n + 2 * tab.ng + (1 if stag == "B" else 0)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, m, m, K_BATCH))


def _vec_stack_fixture(tab, grid, seed):
    u2, v2 = _vec_fixture(tab, grid, seed)
    rng = np.random.default_rng(seed + 1000)
    return (rng.standard_normal(u2.shape + (K_BATCH,)),
            rng.standard_normal(v2.shape + (K_BATCH,)))


# jit-vs-eager FMA contraction on the halo impl: measured max 3.0e-12
# rel over scalar A/B and vector C/D, old loop AND vmap alike (job
# 10134586); bound = measured x4.  NOT a batching tolerance: eager
# stays bitwise.
_JIT_CLASS_RTOL = 1.2e-11


def _assert_jit_class(got, ref):
    got = np.asarray(got); ref = np.asarray(ref)
    fill = np.abs(ref) >= 1.0e20
    assert np.array_equal(got[fill], ref[fill], equal_nan=True)
    np.testing.assert_allclose(got[~fill], ref[~fill],
                               rtol=_JIT_CLASS_RTOL, atol=0.0)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("stag", ["A", "B"])
def test_allk_certified_path_is_the_relocated_loop(tab, stag, jit):
    """ring_comm=None: ext_scalar_sixface_allk must equal the CALLER's
    former per-level loop BITWISE eager (the vmap over K replaced the
    loop 2026-09-29; the impl is elementwise in K, so an eager
    difference is a batching bug, not a tolerance).  Under jit, XLA's
    FMA contraction already moves the OLD loop off eager by up to
    3e-12 rel (job 10134586), so the jitted vmap is compared to the
    same EAGER loop reference at that class (jit-vmap vs eager loop
    measured 3.0e-12 rel max), not bitwise."""
    import jax
    import jax.numpy as jnp

    from legoesm.grids.fv3_duo_halos import (
        ext_scalar_sixface, ext_scalar_sixface_allk)

    assert tab.ring_comm is None
    f6k = jnp.asarray(_scalar_stack_fixture(tab, stag, 31))
    # the pre-batching caller loop, verbatim
    ref = f6k
    for k in range(K_BATCH):
        ref = ref.at[..., k].set(ext_scalar_sixface(ref[..., k], tab, stag))
    if jit:
        got = jax.jit(lambda f: ext_scalar_sixface_allk(f, tab, stag))(f6k)
        _assert_jit_class(got, ref)
        # and against the OLD loop compiled independently: jit-vmap vs
        # jit-loop measured 6.5e-13 rel max (job 10134586)
        def loop(f):
            for k in range(K_BATCH):
                f = f.at[..., k].set(ext_scalar_sixface(f[..., k], tab, stag))
            return f
        _assert_jit_class(got, jax.jit(loop)(f6k))
    else:
        got = ext_scalar_sixface_allk(f6k, tab, stag)
        assert np.array_equal(np.asarray(got), np.asarray(ref),
                              equal_nan=True)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("grid", ["D", "C"])
def test_allk_certified_vector_path_is_the_relocated_loop(tab, grid, jit):
    """Same relocation gate for both vector flows: bitwise eager; the
    jitted vmap against the same eager loop reference at the jit
    reassociation class (measured max 2.2e-12 rel, job 10134586, for
    BOTH the old loop and the vmap against eager)."""
    import jax
    import jax.numpy as jnp

    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface, ext_vector_cgrid_sixface_allk,
        ext_vector_dgrid_sixface, ext_vector_dgrid_sixface_allk)

    assert tab.ring_comm is None
    u6k, v6k = map(jnp.asarray, _vec_stack_fixture(tab, grid, 37))
    per_level = (ext_vector_dgrid_sixface if grid == "D"
                 else ext_vector_cgrid_sixface)
    allk = (ext_vector_dgrid_sixface_allk if grid == "D"
            else ext_vector_cgrid_sixface_allk)
    ru, rv = u6k, v6k
    for k in range(K_BATCH):
        uk, vk = per_level(ru[..., k], rv[..., k], tab)
        ru = ru.at[..., k].set(uk)
        rv = rv.at[..., k].set(vk)
    if jit:
        gu, gv = jax.jit(lambda u, v: allk(u, v, tab))(u6k, v6k)
        _assert_jit_class(gu, ru)
        _assert_jit_class(gv, rv)
        # and against the OLD loop compiled independently (jit-vmap vs
        # jit-loop measured 8.9e-13 rel max, job 10134586)
        def loop(u, v):
            for k in range(K_BATCH):
                uk, vk = per_level(u[..., k], v[..., k], tab)
                u = u.at[..., k].set(uk)
                v = v.at[..., k].set(vk)
            return u, v
        ju, jv = jax.jit(loop)(u6k, v6k)
        _assert_jit_class(gu, ju)
        _assert_jit_class(gv, jv)
    else:
        gu, gv = allk(u6k, v6k, tab)
        assert np.array_equal(np.asarray(gu), np.asarray(ru), equal_nan=True)
        assert np.array_equal(np.asarray(gv), np.asarray(rv), equal_nan=True)


@pytest.mark.parametrize("stag", ["A", "B"])
def test_ring_allk_scalar_matches_certified_bitwise(tab, stag):
    """The k-batched ring exchange (one all_gather for all K, certified
    tables vmapped over trailing K) must equal the certified allk path
    BITWISE on own faces."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_allk
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_scalar_sixface_allk

    devs = _devices_or_skip(6)
    f6k = jnp.asarray(_scalar_stack_fixture(tab, stag, 41))
    ref = np.asarray(ext_scalar_sixface_allk(f6k, tab, stag))

    mesh = Mesh(np.array(devs), ("face",))
    fn, depth = make_ring_ext_scalar_sixface_allk(tab, stag, mesh)
    assert depth == 7
    f6s = jax.device_put(f6k, NamedSharding(mesh, P("face")))
    got = np.asarray(fn(f6s))
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"k-batched ring diverged: |d|max={np.abs(got - ref).max():.3e}")


@pytest.mark.parametrize("grid", ["D", "C"])
def test_ring_allk_vector_matches_certified_bitwise(tab, grid):
    """k-batched vector ring vs the certified allk vector path, bitwise
    (equal_nan pins the structural c2l/pack_p1 NaN slots too)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface_allk, ext_vector_dgrid_sixface_allk)
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_vector_sixface_allk

    devs = _devices_or_skip(6)
    u6k, v6k = map(jnp.asarray, _vec_stack_fixture(tab, grid, 43))
    allk = (ext_vector_dgrid_sixface_allk if grid == "D"
            else ext_vector_cgrid_sixface_allk)
    ru, rv = allk(u6k, v6k, tab)
    ru, rv = np.asarray(ru), np.asarray(rv)

    mesh = Mesh(np.array(devs), ("face",))
    fn, depth = make_ring_ext_vector_sixface_allk(tab, grid, mesh)
    assert depth == 7
    sh = NamedSharding(mesh, P("face"))
    gu, gv = fn(jax.device_put(u6k, sh), jax.device_put(v6k, sh))
    assert np.array_equal(np.asarray(gu), ru, equal_nan=True)
    assert np.array_equal(np.asarray(gv), rv, equal_nan=True)


@pytest.mark.parametrize("stag", ["A", "B"])
def test_ring_allk_scalar_poison_proves_no_out_of_ring_read(tab, stag):
    """poison=True fills every non-border, non-own cell with NaN for ALL
    K; the batched output must still match the certified path bitwise --
    one NaN in the output is a read outside the ring, made loud."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_allk
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_scalar_sixface_allk

    devs = _devices_or_skip(6)
    f6k = jnp.asarray(_scalar_stack_fixture(tab, stag, 47))
    ref = np.asarray(ext_scalar_sixface_allk(f6k, tab, stag))

    mesh = Mesh(np.array(devs), ("face",))
    fn, _ = make_ring_ext_scalar_sixface_allk(tab, stag, mesh, poison=True)
    got = np.asarray(fn(jax.device_put(
        f6k, NamedSharding(mesh, P("face")))))
    assert np.isfinite(got).all(), \
        "NaN reached the output: the tables read outside the ring"
    assert np.array_equal(got, ref)


def test_allk_dispatch_routes_to_ring_comm(tab, monkeypatch):
    """With ring_comm set, the three allk publics must call the matching
    ring allk method with the caller's arrays (fake ring, no devices)."""
    import legoesm.grids.fv3_duo_halos as halos

    calls = []

    class _FakeRing:
        def ext_scalar_allk(self, f6k, stag):
            calls.append(("scalar", f6k, stag))
            return "S"

        def ext_vector_dgrid_allk(self, u6k, v6k):
            calls.append(("dgrid", u6k, v6k))
            return "D"

        def ext_vector_cgrid_allk(self, uc6k, vc6k):
            calls.append(("cgrid", uc6k, vc6k))
            return "C"

    monkeypatch.setattr(tab, "ring_comm", _FakeRing())
    f, u, v = object(), object(), object()
    assert halos.ext_scalar_sixface_allk(f, tab, "B") == "S"
    assert halos.ext_vector_dgrid_sixface_allk(u, v, tab) == "D"
    assert halos.ext_vector_cgrid_sixface_allk(u, v, tab) == "C"
    assert calls == [("scalar", f, "B"), ("dgrid", u, v),
                     ("cgrid", u, v)]


def test_ring_comm_allk_refuses_unknown_stagger():
    """Dispatch hardening for the batched scalar entry too."""
    from legoesm.grids.fv3_duo_spmd import DuoRingComm

    rc = DuoRingComm()
    rc._scalar_allk = {}
    with pytest.raises(ValueError, match="stagger"):
        rc.ext_scalar_allk(None, "C")


def test_build_ring_comm_wires_the_allk_closures(tab, monkeypatch):
    """build_ring_comm must attach working allk closures: the PUBLIC allk
    exchange on a sharded stack equals the certified allk path bitwise
    (also the behavioural no-recursion guard for the batched path)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import legoesm.grids.fv3_duo_halos as halos
    from legoesm.grids.fv3_duo_spmd import build_ring_comm

    devs = _devices_or_skip(6)
    mesh = Mesh(np.array(devs), ("face",))
    sh = NamedSharding(mesh, P("face"))

    f6k = jnp.asarray(_scalar_stack_fixture(tab, "A", 53))
    ud, vd = map(jnp.asarray, _vec_stack_fixture(tab, "D", 59))
    ref_s = np.asarray(halos.ext_scalar_sixface_allk(f6k, tab, "A"))
    ref_u, ref_v = (np.asarray(x) for x in
                    halos.ext_vector_dgrid_sixface_allk(ud, vd, tab))

    monkeypatch.setattr(tab, "ring_comm", build_ring_comm(tab, mesh))
    got_s = np.asarray(halos.ext_scalar_sixface_allk(
        jax.device_put(f6k, sh), tab, "A"))
    got_u, got_v = halos.ext_vector_dgrid_sixface_allk(
        jax.device_put(ud, sh), jax.device_put(vd, sh), tab)
    assert np.array_equal(got_s, ref_s)
    assert np.array_equal(np.asarray(got_u), ref_u, equal_nan=True)
    assert np.array_equal(np.asarray(got_v), ref_v, equal_nan=True)
