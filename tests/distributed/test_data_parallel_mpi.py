"""Cross-RANK data-parallel gradient average (real MPI). Run:
    mpirun -np 2 <py> -m pytest tests/distributed/test_data_parallel_mpi.py -v
Validates that all_reduce_grad_mean averages a gradient pytree across ranks
(the primary multi-node data-parallel reduction). Skips unless nranks == 2.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.training.data_parallel import all_reduce_grad_mean, mpi_data_parallel_train_step


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
    params, _, _ = mpi_data_parallel_train_step(loss, w, opt_state, optimizer, x, nproc)
    # gather both ranks' params; they must match (synced replicas)
    from mpi4py import MPI
    allp = MPI.COMM_WORLD.allgather(np.asarray(params).tolist())
    assert np.allclose(allp[0], allp[1]), allp
