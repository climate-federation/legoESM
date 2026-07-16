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
