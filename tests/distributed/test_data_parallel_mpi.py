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
