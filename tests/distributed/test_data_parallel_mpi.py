"""Cross-RANK data-parallel gradient average (real MPI). Run:
    mpirun -np 2 <py> -m pytest tests/distributed/test_data_parallel_mpi.py -v
Validates that all_reduce_grad_mean averages a gradient pytree across ranks
(the primary multi-node data-parallel reduction). Skips unless nranks == 2.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.training.data_parallel import (
    all_reduce_grad_mean,
    build_dp_value_and_grad,
    mpi_data_parallel_train_step,
)


def _nranks():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except Exception:
        return 0, 1


def test_all_reduce_grad_mean_cross_rank():
    rank, nproc = _nranks()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    # rank 0 grad = [1,0], rank 1 grad = [0,1]  ->  mean = [0.5, 0.5] on BOTH ranks
    g = jnp.array([1.0, 0.0]) if rank == 0 else jnp.array([0.0, 1.0])
    out = all_reduce_grad_mean({"w": g}, nproc)
    assert np.allclose(np.asarray(out["w"]), [0.5, 0.5]), (rank, np.asarray(out["w"]))


def test_grad_bucket_equals_per_leaf_cross_rank():
    """The dtype-bucketed reduction (one allreduce per dtype) must be
    NUMERICALLY equal to the legacy per-leaf reduction under real MPI, on a
    multi-leaf mixed-dtype pytree (incl. a python-float leaf + a None leaf).
    Same dtype, same values reduced; the two paths can differ only in the
    last ULP through MPI's own reduction-tree non-associativity, so this
    asserts closeness, not bit-equality. Gate: the collective-count
    optimization did not change the averaged gradient."""
    rank, nproc = _nranks()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    key = jnp.array(rank + 1, dtype=jnp.float64)
    grad = {
        "a": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32) * (rank + 1),
        "b": jnp.arange(6.0, dtype=jnp.float64).reshape(2, 3) + key,
        "c": jnp.array(0.25, dtype=jnp.float32) * (rank + 1),
        "pyf": float(rank + 1),                       # python float leaf
        "frozen": None,
        "nested": {"d": jnp.ones((4,), dtype=jnp.float64) * (rank + 1)},
    }
    bucketed = all_reduce_grad_mean(grad, nproc, bucket=True)
    perleaf = all_reduce_grad_mean(grad, nproc, bucket=False)
    for path in (("a",), ("b",), ("c",), ("pyf",), ("nested", "d")):
        vb, vp = bucketed, perleaf
        for k in path:
            vb, vp = vb[k], vp[k]
        assert jnp.asarray(vb).dtype == jnp.asarray(vp).dtype
        np.testing.assert_allclose(
            np.asarray(vb), np.asarray(vp), rtol=1e-12, atol=0.0,
            err_msg=f"bucket != per-leaf at {path} on rank {rank}")
    # python-float leaf was actually reduced (mean of 1.0, 2.0 == 1.5)
    assert np.allclose(float(np.asarray(bucketed["pyf"])), 1.5)
    assert bucketed["frozen"] is None


def test_mpi_train_step_synced_across_ranks():
    rank, nproc = _nranks()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    import optax
    w = jnp.array([2.0, 3.0])
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(w)

    def loss(p, x):
        return jnp.sum((p * x) ** 2)

    # each rank a different sample -> averaged grad -> IDENTICAL updated params
    x = jnp.array([1.0, 0.0]) if rank == 0 else jnp.array([0.0, 1.0])
    # #1364: the step takes a PREBUILT value_and_grad (built once per run), not
    # a raw loss -- building it per step recompiles the rollout every step.
    params, _, _ = mpi_data_parallel_train_step(
        build_dp_value_and_grad(loss), w, opt_state, optimizer, x, nproc)
    # gather both ranks' params; they must match (synced replicas)
    from mpi4py import MPI
    allp = MPI.COMM_WORLD.allgather(np.asarray(params).tolist())
    assert np.allclose(allp[0], allp[1]), allp


def test_dp_gradient_equals_serial_mean_loss_gradient():
    """#1814, measured: the averaged 2-rank gradient equals the single-process
    gradient of the mean loss over both samples (not n_ranks x it).  The
    training loss is rank-local; the cross-rank sum runs AFTER jax.grad."""
    rank, nproc = _nranks()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    import jax

    w = jnp.array([2.0, -3.0, 0.5])
    xs = [jnp.array([1.0, 2.0, 0.0]), jnp.array([0.5, -1.0, 4.0])]

    def loss(p, x):
        return jnp.sum(jnp.sin(p * x) ** 2) + jnp.sum(p ** 2 * x)

    _, g = build_dp_value_and_grad(loss)(w, xs[rank])
    g = all_reduce_grad_mean(g, nproc)
    want = jax.grad(lambda p: 0.5 * (loss(p, xs[0]) + loss(p, xs[1])))(w)
    np.testing.assert_allclose(np.asarray(g), np.asarray(want), rtol=1e-12)


def test_one_poisoned_rank_skips_update_on_both_ranks():
    """Non-finite guard under real MPI: rank 0's sample produces a NaN
    gradient, rank 1's is finite. BOTH ranks must agree to skip (allreduce
    MIN on the flag), leave params AND optimizer state untouched, and
    return the NaN marker — a divergent decision would desync the replicas
    forever."""
    rank, nproc = _nranks()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    import jax
    import optax

    w = jnp.array([1.0, 2.0])
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(w)

    def loss(p, x):
        # x[0] == 0 -> sqrt(0) -> finite loss, NaN gradient.
        return jnp.sum((p * x) ** 2) + jnp.sqrt(jnp.sum(p ** 2) * x[0])

    x = jnp.array([0.0, 0.0]) if rank == 0 else jnp.array([1.0, 0.5])
    p2, o2, l2 = mpi_data_parallel_train_step(
        build_dp_value_and_grad(loss), w, opt_state, optimizer, x, nproc)
    # Marker + untouched state on EVERY rank, including the healthy one.
    assert float(l2) != float(l2), (rank, float(l2))
    np.testing.assert_array_equal(np.asarray(p2), np.asarray(w))
    for a, b in zip(jax.tree.leaves(o2), jax.tree.leaves(opt_state)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Cross-rank agreement on the decision (both skipped -> both markers).
    from mpi4py import MPI
    flags = MPI.COMM_WORLD.allgather(float(l2) != float(l2))
    assert flags == [True, True]
