"""Data-parallel training tests (Stage: scale training).

The grad-average / loop tests need >=2 devices; run on a compute node with:
    srun --account=glab --time=0:15:00 env JAX_ENABLE_X64=1 \
        XLA_FLAGS=--xla_force_host_platform_device_count=2 <py> -m pytest \
        tests/ml/test_data_parallel.py -v
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from legoesm.training.data_parallel import (
    shard_samples,
    all_reduce_grad_mean,
    build_dp_value_and_grad,
    mpi_data_parallel_training_loop,
    data_parallel_value_and_grad,
    data_parallel_training_loop,
    mpi_abort_on_uncaught,
    _abort_multirank_job,
)


class _FakeComm:
    """Stand-in MPI comm recording Abort() calls, so the fail-fast rank-gate is
    testable without a real MPI runtime (#985)."""

    def __init__(self, size):
        self._size = size
        self.aborts = []

    def Get_size(self):
        return self._size

    def Abort(self, code):
        self.aborts.append(code)


def _quad_loss(w, x):
    """Simple differentiable stand-in for the real train step: sum((w*x)^2)."""
    return jnp.sum((w * x) ** 2)


# ---- Task 1: sample sharding ----

def test_shard_samples_balanced_and_disjoint():
    items = list(range(10))
    shards = [shard_samples(items, r, 3) for r in range(3)]
    assert all(len(s) == 3 for s in shards)              # 10//3 = 3, remainder dropped
    flat = [x for s in shards for x in s]
    assert len(set(flat)) == 9 and set(flat) <= set(items)   # disjoint, no dup
    assert shard_samples(items, 0, 1) == items           # single process = identity
    # contiguous, non-overlapping ordering
    assert shards[0] == [0, 1, 2] and shards[1] == [3, 4, 5] and shards[2] == [6, 7, 8]


# ---- Task 2: cross-device gradient average == serial batch-mean (THE gate) ----

def test_grad_average_matches_serial_mean():
    assert jax.device_count() >= 2, "run with --xla_force_host_platform_device_count=2"
    w = jnp.array([2.0, 3.0])
    xs = jnp.stack([jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])])   # (2 devices, 2)
    mean_loss, mean_grad = data_parallel_value_and_grad(_quad_loss, w, xs)

    g0 = jax.grad(_quad_loss)(w, xs[0])
    g1 = jax.grad(_quad_loss)(w, xs[1])
    l0 = _quad_loss(w, xs[0])
    l1 = _quad_loss(w, xs[1])
    assert np.allclose(np.asarray(mean_grad), np.asarray((g0 + g1) / 2), atol=1e-9)
    assert np.isclose(float(mean_loss), float((l0 + l1) / 2), atol=1e-9)


def test_grad_average_pytree_params():
    # params as a pytree (dict) -> averaging must be per-leaf
    assert jax.device_count() >= 2
    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}

    def loss(p, x):
        return jnp.sum((p["a"] * x) ** 2) + p["b"] * jnp.sum(x)

    xs = jnp.stack([jnp.array([1.0, 1.0]), jnp.array([2.0, 0.0])])
    _, mg = data_parallel_value_and_grad(loss, params, xs)
    g0 = jax.grad(loss)(params, xs[0])
    g1 = jax.grad(loss)(params, xs[1])
    assert np.allclose(np.asarray(mg["a"]), np.asarray((g0["a"] + g1["a"]) / 2), atol=1e-9)
    assert np.isclose(float(mg["b"]), float((g0["b"] + g1["b"]) / 2), atol=1e-9)


# ---- MPI cross-rank path: single-process (identity) unit tests ----

def test_all_reduce_grad_mean_single_process_identity():
    grad = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
    out = all_reduce_grad_mean(grad, 1)                  # num_processes=1 -> identity
    assert np.allclose(np.asarray(out["a"]), [1.0, 2.0])
    assert float(out["b"]) == 3.0


def test_grad_bucket_reconstructs_pytree(monkeypatch):
    """The dtype-bucketed reduction must reconstruct the EXACT pytree the
    per-leaf path produces — same structure, shapes, dtypes, and non-array
    passthrough — with a single packed reduction per dtype. mpi4jax is
    stubbed by an identity-sum (allreduce over a 1-rank world = the local
    value), so num_processes is spoofed >1 to exercise the bucket code
    while the 'reduction' is deterministic."""
    import legoesm.training.data_parallel as dp

    calls = {"n": 0}

    def fake_global_sum(x, comm=None):
        calls["n"] += 1
        return x * float(NPROC)          # so *(1/NPROC) mean == identity

    global NPROC
    NPROC = 3
    monkeypatch.setattr(dp, "all_reduce_grad_mean",
                        dp.all_reduce_grad_mean)  # keep ref
    monkeypatch.setattr("legoesm.parallel.reductions.global_sum_mpi",
                        fake_global_sum)

    grad = {
        "w32": jnp.ones((3, 4), dtype=jnp.float32),
        "b32": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "w64": jnp.arange(6.0, dtype=jnp.float64).reshape(2, 3),
        "pyf": 5.0,                                        # python float scalar
        "frozen": None,                                    # non-array leaf
        "count": 7,                                        # non-inexact int leaf
    }
    calls["n"] = 0
    bucketed = dp.all_reduce_grad_mean(grad, NPROC, bucket=True)
    bucket_calls = calls["n"]
    calls["n"] = 0
    perleaf = dp.all_reduce_grad_mean(grad, NPROC, bucket=False)
    perleaf_calls = calls["n"]

    # On the INEXACT leaves — including a python float, the only kinds a real
    # gradient pytree carries — bucketing equals the per-leaf path (mean ==
    # identity under the stubbed reduction here).
    for k in ("w32", "b32", "w64"):
        assert bucketed[k].dtype == grad[k].dtype
        assert np.allclose(np.asarray(bucketed[k]), np.asarray(perleaf[k]))
        assert bucketed[k].shape == grad[k].shape
    # python float 5.0 is REDUCED (arrayed), matching legacy — not skipped.
    assert np.allclose(float(np.asarray(bucketed["pyf"])), 5.0)
    assert np.allclose(np.asarray(bucketed["pyf"]), np.asarray(perleaf["pyf"]))
    assert bucketed["frozen"] is None
    # Non-inexact int counter passes through unchanged (legacy tree_map would
    # instead reduce+promote it int->float; bucketing is more correct here).
    assert bucketed["count"] == 7 and isinstance(bucketed["count"], int)
    # Bucketing: f32 group (w32,b32) + f64 group (w64, and pyf under x64) = 2
    # packed reductions. The legacy per-leaf tree_map issues one allreduce per
    # LEAF it visits — w32,b32,w64,pyf,count = 5 (it even reduces the int).
    # Bucketing strictly collapses the collective count.
    assert bucket_calls == 2
    assert perleaf_calls == 5
    assert bucket_calls < perleaf_calls


def test_mpi_loop_on_epoch_receives_current_params():
    # regression: on_epoch must get the CURRENT params (not the stale initial closure).
    w = jnp.array([2.0, 3.0])
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(w)

    def loss(p, x):
        return jnp.sum((p * x) ** 2)

    captured = []
    mpi_data_parallel_training_loop(
        loss, w, opt_state, optimizer, [jnp.array([1.0, 1.0])], n_epochs=3,
        num_processes=1, on_epoch=lambda e, l, params, ost: captured.append(np.asarray(params).copy()))
    assert len(captured) == 3
    assert not np.allclose(captured[0], captured[-1])       # params evolve across epochs
    assert not np.allclose(captured[-1], np.asarray(w))     # final != initial (not stale)


def test_mpi_loop_single_process_reduces_loss():
    w = jnp.array([2.0, 3.0])
    optimizer = optax.sgd(0.05)
    opt_state = optimizer.init(w)

    def loss(p, x):
        return jnp.sum((p * x) ** 2)

    samples = [jnp.array([1.0, 0.5]), jnp.array([0.5, 1.0])]
    params, _, hist = mpi_data_parallel_training_loop(
        loss, w, opt_state, optimizer, samples, n_epochs=5, num_processes=1)
    assert len(hist) == 5 and hist[-1] < hist[0]
    assert bool(jnp.all(jnp.isfinite(params)))


# ---- #1364: the rollout is traced/compiled ONCE per run, not once per step ----

def _scan_loss_with_trace_counter(counter, n_steps=8):
    """Production-shaped loss: a ``jax.checkpoint``-ed ``lax.scan`` rollout, the
    structure ``spectral_rollout`` builds and the one an EAGER path recompiles
    on every call.

    ``counter["n"]`` counts executions of the PYTHON body = TRACES (the body
    runs once per trace, never per call). That makes the gate below independent
    of JAX-internal compile counters, which move between versions.
    """
    def loss(p, x):
        counter["n"] += 1
        def step(s, _):
            return s * jnp.tanh(p) + x, None
        step_ck = jax.checkpoint(
            step, prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable)
        out, _ = jax.lax.scan(step_ck, x, None, length=n_steps)
        return jnp.sum(out ** 2)
    return loss


def test_eager_value_and_grad_traces_every_step_nonvacuity_control():
    """NON-VACUITY CONTROL for the #1364 gate below.

    Reproduces the PRE-FIX structure — a bare ``jax.value_and_grad`` called per
    step — and asserts it traces once PER STEP. If this ever reports 1, the
    trace counter is broken and the gate below would pass while proving
    nothing.
    """
    counter = {"n": 0}
    loss = _scan_loss_with_trace_counter(counter)
    w = jnp.array([0.5, 0.25])
    x = jnp.array([1.0, 0.5])
    n_calls = 5
    for _ in range(n_calls):
        jax.value_and_grad(loss)(w, x)
    assert counter["n"] == n_calls, (
        f"expected the un-jitted path to re-trace every call, got "
        f"{counter['n']} traces in {n_calls} calls")


def test_mpi_loop_traces_rollout_once_not_per_step():
    """#1364 REGRESSION GATE: host-RAM leak ~0.27 GB/step/rank at T106.

    The loop must jit the value+grad ONCE (``build_dp_value_and_grad``, hoisted
    above BOTH loops) so the rollout + reverse adjoint is traced and compiled a
    single time. With no jit — or with the jit rebuilt inside either loop — an
    eager ``lax.scan`` recompiles per step and retains each executable, which
    is what SIGKILLed the T106 4xA100 run at epoch 5/24.

    Fails loudly if the fix is reverted: the pre-fix structure traces
    ``n_epochs * len(samples)`` times (see the non-vacuity control above).
    """
    counter = {"n": 0}
    loss = _scan_loss_with_trace_counter(counter)
    w = jnp.array([0.5, 0.25])
    optimizer = optax.sgd(0.01)
    opt_state = optimizer.init(w)
    samples = [jnp.array([1.0, 0.5]), jnp.array([0.5, 1.0]), jnp.array([0.25, 0.75])]
    n_epochs = 4
    n_steps = n_epochs * len(samples)

    mpi_data_parallel_training_loop(
        loss, w, opt_state, optimizer, samples, n_epochs=n_epochs,
        num_processes=1)

    assert n_steps == 12                      # the loop really ran 12 steps
    assert counter["n"] == 1, (
        f"the rollout was traced {counter['n']} times over {n_steps} steps; "
        f"#1364 requires exactly 1 (jit hoisted outside both loops)")


def test_build_dp_value_and_grad_single_trace_and_matches_eager():
    """Direct test of the new public helper: one trace across many calls, and
    the same (loss, grad) an eager ``jax.value_and_grad`` produces."""
    counter = {"n": 0}
    loss = _scan_loss_with_trace_counter(counter)
    fn = build_dp_value_and_grad(loss)
    w = jnp.array([0.5, 0.25])
    samples = [jnp.array([1.0, 0.5]), jnp.array([0.5, 1.0]), jnp.array([2.0, 1.0])]

    outs = [fn(w, x) for x in samples] + [fn(w, samples[0])]
    assert fn.n_traces() == 1, f"retraced {fn.n_traces()} times over 4 calls"
    assert counter["n"] == 1

    for (l_jit, g_jit), x in zip(outs, samples + [samples[0]]):
        l_ref, g_ref = jax.value_and_grad(loss)(w, x)
        assert np.isclose(float(l_jit), float(l_ref), rtol=1e-10, atol=1e-12)
        assert np.allclose(np.asarray(g_jit), np.asarray(g_ref),
                           rtol=1e-10, atol=1e-12)


def test_device_put_immunizes_python_scalar_leaf():
    """#1286 fix A and the #1364 jit are COUPLED — document it so neither is
    removed in isolation.

    ``eqx.is_array(1.0) is False``: a per-sample-varying bare Python float leaf
    is frozen STATIC by filter_jit and retraces every step. The loop's
    ``jax.device_put(sample)`` arrays every leaf, Python scalars included, which
    removes the hazard. Both arms are asserted, so dropping the ``device_put``
    (or the jit) turns one of them red.
    """
    def loss(p, sample):
        x, scale = sample                 # `scale`: bare python float
        return jnp.sum((p * x) ** 2) * scale

    w = jnp.array([0.5, 0.25])
    raw = [(jnp.array([1.0, 0.5]), 1.0),
           (jnp.array([0.5, 1.0]), 2.0),
           (jnp.array([0.25, 0.75]), 3.0)]

    no_put = build_dp_value_and_grad(loss)
    for s in raw:
        no_put(w, s)
    assert no_put.n_traces() == len(raw), (
        "expected a varying PYTHON-scalar leaf to retrace once per call "
        "without device_put (the hazard this documents)")

    with_put = build_dp_value_and_grad(loss)
    for s in raw:
        with_put(w, jax.device_put(s))    # exactly what the loop does
    assert with_put.n_traces() == 1, (
        "device_put must array every leaf so nothing goes static; got "
        f"{with_put.n_traces()} traces")


def test_retrace_guard_fires_on_varying_sample_shape(caplog):
    """Synthetic-violation self-test for the #1364 retrace guard — proves it is
    not vacuous.

    A per-sample SHAPE change is what ``device_put`` cannot normalize (a ragged
    shard, a stray sample at another resolution); it recompiles every step just
    as #1364 did, so the loop must say so in the log.
    """
    import logging

    def loss(p, x):
        return jnp.sum(x ** 2) * jnp.sum(p)

    w = jnp.array([0.5, 0.25])
    optimizer = optax.sgd(0.01)
    opt_state = optimizer.init(w)
    samples = [jnp.ones(2), jnp.ones(3), jnp.ones(4)]   # varying shapes

    with caplog.at_level(logging.WARNING,
                         logger="legoesm.training.data_parallel"):
        mpi_data_parallel_training_loop(
            loss, w, opt_state, optimizer, samples, n_epochs=2, num_processes=1)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("RETRACE DETECTED" in m for m in msgs), \
        f"guard did not fire on varying shapes; records={msgs}"


def test_dp_jit_matches_eager_reference_math():
    """The #1364 jit must not change the MATH.

    Same params/samples/lr/order through the jitted loop and an explicitly
    EAGER reference. jit is semantics-preserving, but XLA may fuse differently
    from op-by-op dispatch, so this is a tight numerical tolerance rather than
    bit-identity.
    """
    counter = {"n": 0}
    w0 = jnp.array([0.5, 0.25])
    samples = [jnp.array([1.0, 0.5]), jnp.array([0.5, 1.0])]
    n_epochs = 3

    optimizer = optax.sgd(0.01)
    jit_params, _, jit_hist = mpi_data_parallel_training_loop(
        _scan_loss_with_trace_counter(counter), w0, optimizer.init(w0),
        optimizer, samples, n_epochs=n_epochs, num_processes=1)

    # explicit eager reference: no jit anywhere, same update order
    ref_loss = _scan_loss_with_trace_counter({"n": 0})
    ref_optimizer = optax.sgd(0.01)
    ref_params, ref_opt_state = w0, ref_optimizer.init(w0)
    ref_hist = []
    for _ in range(n_epochs):
        losses = []
        for x in samples:
            loss, grad = jax.value_and_grad(ref_loss)(ref_params, x)
            updates, ref_opt_state = ref_optimizer.update(
                grad, ref_opt_state, ref_params)
            ref_params = optax.apply_updates(ref_params, updates)
            losses.append(float(loss))
        ref_hist.append(sum(losses) / len(losses))

    assert np.allclose(np.asarray(jit_params), np.asarray(ref_params),
                       rtol=1e-10, atol=1e-12), (jit_params, ref_params)
    assert np.allclose(jit_hist, ref_hist, rtol=1e-10, atol=1e-12)


# ---- Task 3: training loop reduces the loss ----

def test_data_parallel_loop_reduces_loss():
    assert jax.device_count() >= 2
    w = jnp.array([2.0, 3.0])
    optimizer = optax.sgd(0.05)
    opt_state = optimizer.init(w)
    # one batch of 2 device-samples; 5 epochs
    xs = jnp.stack([jnp.array([1.0, 0.5]), jnp.array([0.5, 1.0])])
    params, opt_state, history = data_parallel_training_loop(
        _quad_loss, w, opt_state, optimizer, [xs], n_epochs=5)
    assert len(history) == 5
    assert history[-1] < history[0]                       # loss decreased
    assert bool(jnp.all(jnp.isfinite(params)))


# ---- fail-fast: MPI_Abort on a rank death (#985) ----

def test_abort_multirank_job_single_rank_noop():
    comm = _FakeComm(size=1)
    _abort_multirank_job(comm)
    assert comm.aborts == []          # single rank -> nothing to abort


def test_abort_multirank_job_multirank_aborts():
    comm = _FakeComm(size=4)
    _abort_multirank_job(comm, code=7)
    assert comm.aborts == [7]         # >1 rank -> abort the whole job


def test_mpi_abort_on_uncaught_multirank_aborts_then_reraises():
    comm = _FakeComm(size=4)

    @mpi_abort_on_uncaught(comm=comm)
    def boom():
        raise ValueError("rank died")

    with pytest.raises(ValueError, match="rank died"):
        boom()
    assert comm.aborts == [1]         # aborted BEFORE the exception propagated


def test_mpi_abort_on_uncaught_single_rank_passthrough():
    comm = _FakeComm(size=1)

    @mpi_abort_on_uncaught(comm=comm)
    def boom():
        raise ValueError("rank died")

    with pytest.raises(ValueError, match="rank died"):
        boom()
    assert comm.aborts == []          # single rank -> just propagate, no Abort


def test_mpi_abort_on_uncaught_success_no_abort():
    comm = _FakeComm(size=4)

    @mpi_abort_on_uncaught(comm=comm)
    def ok():
        return 42

    assert ok() == 42
    assert comm.aborts == []          # clean return -> never aborts
