"""Multi-face-per-shard ppermute SPMD halo: bit-identity, AD, scan, HLO guard.

The multiface ppermute exchange (``cubesphere_exchange._make_exchange_
ppermute_multiface``) replaces the all_gather SPMD halo that provably
replicated ALL compute (HLO probe job 8456476: per-device FLOPs ratio 1.00 at
2 devices, all-gather results with full 6-face extent).  Each shard owns
``k = 6/n_devices`` contiguous faces; intra-shard edges are filled
shard-locally, cross-shard edges ride a static device-pair ppermute schedule.

This file is the codex-mandated test set (findings 1-5 of
``scripts/tmp/_codex_ppermute_multiface_out.txt``):

* static schedule properties (device-pair coloring, coverage, round counts);
* BIT identity vs the serial local pad for n_devices 1/2/3/6 x halo 1/2 x
  3D/4D x with/without ``interp_offsets``, on adversarial per-face ramps
  (anisotropic + cross-term, so any rotation/reversal/junction error flips
  bits — codex MAJOR 4 covers the 3-face-junction corner cells);
* ``grad(scan(step))`` AD smoke vs the serial gradient (ppermute transpose
  under shard_map inside lax.scan, jax 0.9.1);
* compiled-HLO assertion that the scan-runner hot path contains ZERO
  all-gather ops with full-cube face extent (codex BLOCKER 3), with both a
  synthetic-violation self-test and a real-program violation (the explicit
  all_gather diagnostic opt-in) proving the tripwire is non-vacuous.

Run with 6 host devices (2- and 3-device cases run on subsets):
    XLA_FLAGS="--xla_force_host_platform_device_count=6" \\
        JAX_ENABLE_X64=1 python -m pytest tests/parallel/test_ppermute_multiface.py
(parametrized cases skip individually when the forced device count is lower.)
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=6")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

if jax.device_count() < 2:
    pytest.skip(
        "needs >=2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=6)",
        allow_module_level=True,
    )

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from legoesm.grids import halo as halo_mod  # noqa: E402
from legoesm.grids.halo import (  # noqa: E402
    compute_halo_interp_offsets,
    compute_halo_interp_offsets_h2,
    pad_halo,
    pad_halo_4d,
    pad_halo_vector_4d,
    set_halo_backend,
)
from legoesm.parallel import cubesphere_exchange as cx  # noqa: E402

ALL_COUNTS = [1, 2, 3, 6]


def _mesh(n_devices):
    if jax.device_count() < n_devices:
        pytest.skip(f"needs {n_devices} devices, have {jax.device_count()}")
    return Mesh(
        np.array(jax.devices()[:n_devices]).reshape(n_devices), ("face",),
    )


@pytest.fixture(autouse=True)
def _restore_globals():
    """Snapshot+restore ALL process-global halo state after each test
    (same hardening as test_ppermute_halo_exchange.py: restore
    _use_ppermute, the backend string, both _spmd_mesh refs and the mpi
    topology, and clear the kernel cache)."""
    snap = (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
            halo_mod._spmd_mesh, halo_mod._mpi_topology,
            halo_mod.get_corner_fill_mode())
    yield
    (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
     halo_mod._spmd_mesh, halo_mod._mpi_topology) = snap[:5]
    halo_mod.set_corner_fill_mode(snap[5])
    cx._cache.clear()


def _ramp_3d(n):
    """Adversarial per-face ramp: face-distinct, i/j-anisotropic, with a
    cross term so any reversal / axis-swap / wrong-source-face error
    changes bits (incl. at the 3-face-junction corner cells)."""
    f = np.arange(6, dtype=np.float64)[:, None, None]
    i = np.arange(n, dtype=np.float64)[None, :, None]
    j = np.arange(n, dtype=np.float64)[None, None, :]
    return jnp.asarray(1000.0 * (f + 1.0) + 13.0 * i + 0.37 * j
                       + 0.001 * i * j)


def _ramp_4d(n, c):
    base = np.asarray(_ramp_3d(n))[..., None]
    ch = 77.7 * np.arange(c, dtype=np.float64)[None, None, None, :]
    return jnp.asarray(base + ch)


def _serial(fn, *args, **kw):
    set_halo_backend("local")
    return fn(*args, **kw)


# =======================================================================
# 1. Static schedule properties (codex finding 1: device-pair coloring)
# =======================================================================

@pytest.mark.parametrize(
    "k, expect_rounds, expect_slots",
    [(6, 0, 1), (3, 1, 8), (2, 2, 4), (1, 4, 1)],
)
def test_multiface_schedule_static_properties(k, expect_rounds, expect_slots):
    """Round counts hit the König optimum per layout; every directed
    device pair appears in exactly one round; each round is a partial
    permutation; every (face, edge) is covered exactly once by local
    fill or a ppermute slot.  (Coverage violations raise inside the
    builder; this test pins the schedule SHAPE so a regression that
    still 'covers' but degrades the schedule is caught.)"""
    t = cx._get_multiface_tables(k)
    n_dev = 6 // k
    assert t.faces_per_shard == k
    assert len(t.perms) == expect_rounds
    assert t.max_slots == expect_slots
    seen_pairs = []
    for rnd in t.perms:
        srcs = [s for s, _ in rnd]
        dsts = [d for _, d in rnd]
        assert len(set(srcs)) == len(srcs), f"duplicate sender in {rnd}"
        assert len(set(dsts)) == len(dsts), f"duplicate receiver in {rnd}"
        assert all(0 <= s < n_dev and 0 <= d < n_dev and s != d
                   for s, d in rnd)
        seen_pairs.extend(rnd)
    assert len(seen_pairs) == len(set(seen_pairs))
    # Reconstruct coverage from the static tables: local entries plus
    # receive targets must tile all 4k (face, edge) slots per device.
    sentinel = 4 * k
    for d in range(n_dev):
        local = set()
        for i in range(k):
            for e in range(4):
                g = d * k + i
                nf = cx._NBR_FACES[g][e]
                if nf // k == d:
                    local.add(i * 4 + e)
        recv = set()
        for r in range(len(t.perms)):
            for m in range(t.max_slots):
                tgt = int(t.recv_tgt[r, d, m])
                if tgt != sentinel:
                    assert tgt not in recv, "duplicate halo target"
                    recv.add(tgt)
        assert local | recv == set(range(4 * k))
        assert not (local & recv)


def test_n3_schedule_has_two_rounds_per_device_pair_coloring():
    """codex design point: at n_devices=3 every device has cross edges to
    BOTH peers (6 directed pairs on 3 devices), so one ppermute round
    cannot cover the graph — the coloring must produce 2 cycle rounds."""
    t = cx._get_multiface_tables(2)
    assert len(t.perms) == 2
    pairs = sorted(p for rnd in t.perms for p in rnd)
    assert pairs == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


# =======================================================================
# 2. Bit identity vs serial (codex findings 1, 2, 4)
# =======================================================================

@pytest.mark.parametrize("n_devices", ALL_COUNTS)
@pytest.mark.parametrize("halo", [1, 2])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_ramp_3d_bit_matches_serial(n_devices, halo, with_offsets):
    n = 8
    data = _ramp_3d(n)
    offsets = None
    if with_offsets:
        offsets = (compute_halo_interp_offsets(n) if halo == 1
                   else compute_halo_interp_offsets_h2(n))
    ref = _serial(pad_halo, data, halo=halo, interp_offsets=offsets)
    cx.set_ppermute_default(True)
    got = cx.explicit_pad_halo(data, _mesh(n_devices), halo=halo,
                               interp_offsets=offsets)
    assert got.shape == ref.shape == (6, n + 2 * halo, n + 2 * halo)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("n_devices", ALL_COUNTS)
@pytest.mark.parametrize("halo", [1, 2])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_ramp_4d_bit_matches_serial(n_devices, halo, with_offsets):
    n, c = 8, 3
    data = _ramp_4d(n, c)
    offsets = None
    if with_offsets:
        offsets = (compute_halo_interp_offsets(n) if halo == 1
                   else compute_halo_interp_offsets_h2(n))
    ref = _serial(pad_halo_4d, data, halo=halo, interp_offsets=offsets)
    cx.set_ppermute_default(True)
    got = cx.explicit_pad_halo_4d(data, _mesh(n_devices), halo=halo,
                                  interp_offsets=offsets)
    assert got.shape == ref.shape == (6, n + 2 * halo, n + 2 * halo, c)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("n_devices", [2, 3, 6])
@pytest.mark.parametrize("halo", [1, 2])
def test_random_4d_bit_matches_serial(n_devices, halo):
    """Random fields close the 'ramp happens to be flip-symmetric'
    loophole (it is not, but belt and braces)."""
    n, c = 6, 4
    data = jax.random.normal(jax.random.PRNGKey(11), (6, n, n, c))
    ref = _serial(pad_halo_4d, data, halo=halo)
    cx.set_ppermute_default(True)
    got = cx.explicit_pad_halo_4d(data, _mesh(n_devices), halo=halo)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("n_devices", [2, 3, 6])
@pytest.mark.parametrize("halo", [1, 2])
def test_cross_shard_junction_corners_bit_match(n_devices, halo):
    """codex MAJOR 4: the avg corner fill at 3-face junctions must be
    BIT-identical to serial whether the junction is shard-internal or
    cross-shard.  Full-array equality is asserted elsewhere; this test
    isolates the corner halo blocks so a junction regression reads as a
    corner failure, not a generic mismatch."""
    n = 8
    data = _ramp_3d(n)
    ref = np.asarray(_serial(pad_halo, data, halo=halo))
    cx.set_ppermute_default(True)
    got = np.asarray(cx.explicit_pad_halo(data, _mesh(n_devices), halo=halo))
    h = halo
    for f in range(6):
        for (si, sj) in ((slice(0, h), slice(0, h)),
                         (slice(0, h), slice(-h, None)),
                         (slice(-h, None), slice(0, h)),
                         (slice(-h, None), slice(-h, None))):
            np.testing.assert_array_equal(
                got[f][si, sj], ref[f][si, sj],
                err_msg=f"corner block mismatch face={f} halo={h} "
                        f"n_devices={n_devices}",
            )


@pytest.mark.parametrize("n_devices", [2, 3])
def test_vector_exchange_bit_matches_serial(n_devices):
    """Vector (u, v) packed exchange rides explicit_pad_halo_4d → the
    multiface kernel at 2/3 devices.  Identity rotation so components
    exchange like scalars."""
    n, nlev = 8, 3
    u = _ramp_4d(n, nlev)
    v = _ramp_4d(n, nlev) * 0.5 + 3.0
    cos = jnp.ones((6, n, n))
    sin = jnp.zeros((6, n, n))
    cos_p = jnp.ones((6, n + 2, n + 2))
    sin_p = jnp.zeros((6, n + 2, n + 2))
    ru, rv = _serial(pad_halo_vector_4d, u, v, cos, sin, cos_p, sin_p,
                     halo=1)
    cx.set_ppermute_default(True)
    gu, gv = cx.explicit_pad_halo_vector_4d(
        u, v, cos, sin, cos_p, sin_p, _mesh(n_devices), halo=1)
    np.testing.assert_array_equal(np.asarray(gu), np.asarray(ru))
    np.testing.assert_array_equal(np.asarray(gv), np.asarray(rv))


@pytest.mark.parametrize("n_devices", [2, 3])
def test_packed_pair_h2_bit_matches_serial(n_devices):
    """pad_halo_pair_h2 under the activated SPMD backend packs two
    fields into ONE halo=2 multiface ppermute collective."""
    n = 8
    q1 = _ramp_3d(n)
    q2 = _ramp_3d(n)[:, ::-1, :] * 1.7
    offsets = compute_halo_interp_offsets_h2(n)
    set_halo_backend("local")
    ref1 = np.asarray(pad_halo(q1, halo=2, interp_offsets=offsets))
    ref2 = np.asarray(pad_halo(q2, halo=2, interp_offsets=offsets))
    cx.activate_spmd_halo_backend(_mesh(n_devices), n=n, nlev=1)
    try:
        out1, out2 = halo_mod.pad_halo_pair_h2(q1, q2,
                                               interp_offsets=offsets)
        np.testing.assert_array_equal(np.asarray(out1), ref1)
        np.testing.assert_array_equal(np.asarray(out2), ref2)
    finally:
        cx.deactivate_spmd_halo_backend()


# =======================================================================
# 3. Routing, policy and cache keying (codex findings 1, 6)
# =======================================================================

@pytest.mark.parametrize("n_devices", [1, 2, 3, 6])
def test_activation_defaults_to_ppermute(n_devices):
    """The n_devices==6 forcing is gone: activation selects ppermute for
    every face-sharded count (all_gather only via explicit opt-in)."""
    cx.activate_spmd_halo_backend(_mesh(n_devices), n=8, nlev=2)
    try:
        assert cx._use_ppermute is True
        assert halo_mod.get_halo_backend() == "spmd"
    finally:
        cx.deactivate_spmd_halo_backend()


def test_activation_force_allgather_optin(monkeypatch):
    """Diagnostic opt-in: kwarg and env both route to all_gather; the
    env override also flips select_exchange_backend."""
    mesh = _mesh(2)
    cx.activate_spmd_halo_backend(mesh, n=8, force_allgather=True)
    try:
        assert cx._use_ppermute is False
    finally:
        cx.deactivate_spmd_halo_backend()
    monkeypatch.setenv("LEGOESM_SPMD_FORCE_ALLGATHER", "1")
    assert cx.select_exchange_backend(8, 1, 2) is False
    cx.activate_spmd_halo_backend(mesh, n=8)
    try:
        assert cx._use_ppermute is False
    finally:
        cx.deactivate_spmd_halo_backend()
    monkeypatch.delenv("LEGOESM_SPMD_FORCE_ALLGATHER")
    assert cx.select_exchange_backend(8, 1, 2) is True


def test_routing_emits_collective_permute_not_allgather():
    """The compiled multiface h1+h2 programs contain collective-permute
    and no all-gather; the one-face kernel still owns halo=1 at 6
    devices (cache key distinguishes the variants)."""
    n = 8
    data = _ramp_3d(n)
    mesh2 = _mesh(2)
    cx.set_ppermute_default(True)
    for halo in (1, 2):
        txt = jax.jit(
            lambda d, h=halo: cx.explicit_pad_halo(d, mesh2, halo=h)
        ).lower(data).compile().as_text()
        assert "collective-permute" in txt, f"halo={halo}: ppermute missing"
        assert "all-gather" not in txt, f"halo={halo}: all-gather present"
    # Variant resolution: one-face kernel reserved for h1@6dev only.
    assert cx._select_variant(True, 1, 6) == "ppermute_oneface"
    assert cx._select_variant(True, 2, 6) == "ppermute_multiface"
    assert cx._select_variant(True, 1, 2) == "ppermute_multiface"
    assert cx._select_variant(False, 1, 2) == "allgather"
    assert cx._select_variant(False, 2, 6) == "allgather_h2"


def test_cache_key_distinguishes_layout_and_variant():
    """codex MINOR 6: the cache key carries (variant, faces_per_shard);
    kernels built for different layouts/variants never collide."""
    mesh2, mesh3 = _mesh(2), _mesh(3)
    cx._cache.clear()
    k_a = cx._get_exchange(mesh2, 3, True, halo=1)
    k_b = cx._get_exchange(mesh3, 3, True, halo=1)
    k_c = cx._get_exchange(mesh2, 3, True, halo=2)
    k_d = cx._get_exchange(mesh2, 3, False, halo=1)
    assert len({id(k_a), id(k_b), id(k_c), id(k_d)}) == 4
    keys = list(cx._cache.keys())
    assert all(len(k) == 6 for k in keys)  # (id, ndim, variant, halo, offs, k)
    assert {k[2] for k in keys} == {"ppermute_multiface", "allgather"}
    assert {k[5] for k in keys} == {2, 3}


# =======================================================================
# 4. AD + scan legality (codex finding 5)
# =======================================================================

def _stencil_step(x, offsets_h1):
    """One halo-1 + halo-2 stencil step through the ACTIVE halo backend
    (pad_halo_4d dispatches to SPMD when activated)."""
    p1 = pad_halo_4d(x, halo=1, interp_offsets=offsets_h1)
    lap = (p1[:, 2:, 1:-1] + p1[:, :-2, 1:-1]
           + p1[:, 1:-1, 2:] + p1[:, 1:-1, :-2] - 4.0 * x)
    p2 = pad_halo_4d(x, halo=2)
    wide = p2[:, 4:, 2:-2] + p2[:, :-4, 2:-2] - 2.0 * x
    return x + 0.05 * lap - 0.01 * wide


@pytest.mark.parametrize("n_devices", [2, 3])
def test_grad_of_scanned_step_matches_serial(n_devices):
    """grad(scan(step)) through the multiface ppermute exchange:
    ppermute's transpose rule (inverse permutation) under shard_map
    inside lax.scan, jax 0.9.1.  Gradient must be finite and match the
    serial-backend gradient (the primal kernels are bit-identical; the
    cotangent path may legally reassociate, hence allclose not
    bit-equal)."""
    n, c, steps = 8, 2, 3
    x = _ramp_4d(n, c) / 1000.0
    offsets = compute_halo_interp_offsets(n)

    def loss(x0):
        def body(carry, _):
            return _stencil_step(carry, offsets), None
        y = jax.lax.scan(body, x0, None, length=steps)[0]
        return jnp.sum(y * y)

    set_halo_backend("local")
    g_serial = np.asarray(jax.grad(loss)(x))
    l_serial = float(loss(x))

    cx.activate_spmd_halo_backend(_mesh(n_devices), n=n, nlev=c)
    try:
        l_spmd = float(jax.jit(loss)(x))
        g_spmd = np.asarray(jax.jit(jax.grad(loss))(x))
    finally:
        cx.deactivate_spmd_halo_backend()

    assert np.isfinite(l_spmd)
    np.testing.assert_allclose(l_spmd, l_serial, rtol=1e-13)
    assert np.all(np.isfinite(g_spmd))
    np.testing.assert_allclose(g_spmd, g_serial, rtol=1e-12, atol=1e-12)


# =======================================================================
# 5. Compiled-HLO guard (codex BLOCKER 3) + non-vacuous self-tests
# =======================================================================

def _scan_runner_hlo(mesh, n, c, steps=4):
    """Lower + compile a sharded scan(step) runner — the same program
    shape the bench times — and return its optimized HLO text."""
    offsets = compute_halo_interp_offsets(n)
    sharding = NamedSharding(mesh, P("face", None, None, None))

    def run(x0):
        def body(carry, _):
            return _stencil_step(carry, offsets), None
        return jax.lax.scan(body, x0, None, length=steps)[0]

    x = jax.device_put(_ramp_4d(n, c) / 1000.0, sharding)
    runner = jax.jit(run, in_shardings=(sharding,), out_shardings=sharding)
    return runner.lower(x).compile().as_text()


@pytest.mark.parametrize("n_devices", [2, 3])
def test_scan_hot_path_has_zero_fullcube_allgathers(n_devices):
    """THE mechanical tripwire for the replication bug class: the
    compiled scan hot path under the ppermute backend must contain ZERO
    all-gather ops with full-cube face extent (and, on this pure
    halo+stencil step, zero all-gathers at all), while actually routing
    collective-permute."""
    n, c = 8, 2
    mesh = _mesh(n_devices)
    cx.activate_spmd_halo_backend(mesh, n=n, nlev=c)
    try:
        txt = _scan_runner_hlo(mesh, n, c)
    finally:
        cx.deactivate_spmd_halo_backend()
    assert cx.find_fullcube_allgathers(txt, n_faces=6, n=n) == []
    cx.assert_no_fullcube_allgather(txt, n=n, context="scan hot path")
    assert "collective-permute" in txt
    assert "all-gather" not in txt  # codex acceptance: hot-path count 0


def test_hlo_guard_flags_real_allgather_program():
    """Non-vacuous tripwire, REAL-program violation: the explicit
    all_gather diagnostic backend compiles to a program the guard MUST
    flag (this is exactly the replicating backend of probe 8456476)."""
    n, c = 8, 2
    mesh = _mesh(2)
    cx.activate_spmd_halo_backend(mesh, n=n, nlev=c, force_allgather=True)
    try:
        txt = _scan_runner_hlo(mesh, n, c)
    finally:
        cx.deactivate_spmd_halo_backend()
    bad = cx.find_fullcube_allgathers(txt, n_faces=6, n=n)
    assert bad, "guard failed to flag the known-replicating allgather backend"
    with pytest.raises(RuntimeError, match="full-cube face extent"):
        cx.assert_no_fullcube_allgather(txt, n=n, context="allgather opt-in")


def test_hlo_guard_synthetic_cases():
    """Synthetic-violation self-test (repo gate doctrine): the detector
    trips on doctored HLO lines and stays quiet on benign ones —
    including the codex-flagged false-pass surface (full-cube volume
    behind a reshaped/sliced result shape)."""
    flag = cx.find_fullcube_allgathers
    # Full face extent on a strips gather (the probe's smoking gun).
    assert flag("%ag = f32[6,4,24,8]{3,2,1,0} all-gather(f32[3,4,24,8] %x)")
    # Full cube field.
    assert flag("%ag.1 = f32[6,26,26,8]{3,2,1,0} all-gather(f32[3,26,26,8] %p)")
    # Async start op: tuple result carries the gathered [6,...] shape.
    assert flag("%ags = (f32[3,4,24]{2,1,0}, f32[6,4,24]{2,1,0}) "
                "all-gather-start(f32[3,4,24] %s)")
    # Async done op alone (codex r1 BLOCKER: the full result can be
    # visible only on the -done instruction; the guard must not depend
    # on the paired -start line surviving textual transformations).
    assert flag("%agd = f32[6,4,24,8]{3,2,1,0} all-gather-done(%ags)")
    assert flag("%agd.1 = f32[6,26,26,8]{3,2,1,0} all-gather-done("
                "(f32[3,26,26,8], f32[6,26,26,8]) %ags.1)")
    # Volume false-pass surface: face dim folded away but >= 6*n*n elems.
    assert flag("%ag.2 = f32[144,64]{1,0} all-gather(f32[72,64] %q)", n=24)
    # Benign: collective-permute is the expected op.
    assert not flag("%cp = f32[24,8]{1,0} collective-permute(f32[24,8] %s)")
    # Benign: small partial gather without face extent or cube volume.
    assert not flag("%ag.3 = f32[2,24,8]{2,1,0} all-gather(f32[1,24,8] %r)",
                    n=24)
    # Benign: 6-element scalar-stats gather (one element per face).
    assert not flag("%ag.4 = f32[6]{0} all-gather(f32[3] %t)")
    # Empty text.
    assert not flag("")


# =======================================================================
# 6. Serial / one-face behavior is preserved bit-for-bit
# =======================================================================

def test_six_device_h1_still_routes_oneface_kernel_bitmatch():
    """halo=1 at exactly 6 devices keeps the validated one-face kernel
    (lowest-risk path, codex sequencing item 2) and stays bit-identical
    to serial; the multiface kernel owns halo=2 at the same mesh."""
    n = 8
    mesh6 = _mesh(6)
    data = _ramp_3d(n)
    cx.set_ppermute_default(True)
    ref1 = _serial(pad_halo, data, halo=1)
    got1 = cx.explicit_pad_halo(data, mesh6, halo=1)
    np.testing.assert_array_equal(np.asarray(got1), np.asarray(ref1))
    ref2 = _serial(pad_halo, data, halo=2)
    got2 = cx.explicit_pad_halo(data, mesh6, halo=2)
    np.testing.assert_array_equal(np.asarray(got2), np.asarray(ref2))
    variants = {k[2] for k in cx._cache.keys()}
    assert "ppermute_oneface" in variants
    assert "ppermute_multiface" in variants


def test_single_device_multiface_is_pure_local():
    """n_devices=1: zero ppermute rounds; the kernel is a pure shard-
    local fill and must be bit-identical to serial (no collectives in
    the compiled program at all)."""
    n = 8
    mesh1 = _mesh(1)
    data = _ramp_3d(n)
    cx.set_ppermute_default(True)
    for halo in (1, 2):
        ref = _serial(pad_halo, data, halo=halo)
        got = cx.explicit_pad_halo(data, mesh1, halo=halo)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    txt = jax.jit(
        lambda d: cx.explicit_pad_halo(d, mesh1, halo=2)
    ).lower(data).compile().as_text()
    assert "all-gather" not in txt
    assert "collective-permute" not in txt


def teardown_module(_):
    cx.set_ppermute_default(False)
    cx._cache.clear()
    set_halo_backend("local")
