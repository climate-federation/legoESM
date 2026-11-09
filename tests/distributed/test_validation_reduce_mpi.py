"""Held-out loss reduction with idle ranks (real MPI). Run:
    mpirun -np 2 <py> -m pytest tests/distributed/test_validation_reduce_mpi.py
User decision 2026-11-09: when the validation windows score on a subset of
the ranks, the idle ranks still join every collective and the mean is over
the real windows only. Skips unless nranks == 2.
"""
import pytest

from legoesm.training.data_parallel import reduce_validation_loss


def _rank_size():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except Exception:
        return 0, 1


def test_only_rank_0_scores_and_the_mean_is_over_its_windows():
    rank, nproc = _rank_size()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    # Rank 0 scored three windows (losses 1, 2, 3); rank 1 is idle. Three
    # windows on two ranks, so a mean over ranks (3.0) differs from the mean
    # over windows (2.0).
    if rank == 0:
        total, count, sums = 6.0, 3, {"mse_T": 1.5 + 1.5 + 3.0,
                                      "crps_q": 0.3 + 0.3 + 0.3}
    else:
        total, count, sums = 0.0, 0, {}
    loss, terms = reduce_validation_loss(total, count, sums, nproc)
    assert loss == pytest.approx(2.0), (rank, loss)
    assert terms == pytest.approx({"mse_T": 2.0, "crps_q": 0.3}), (rank, terms)


def test_no_rank_scores_gives_none_on_every_rank():
    rank, nproc = _rank_size()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    assert reduce_validation_loss(0.0, 0, {}, nproc) == (None, {})
