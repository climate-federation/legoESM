"""The pre-checkpoint finiteness verdict is the same on every rank.

Run::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_checkpoint_finite_mpi.py

Each rank holds only its own cells.  A NaN on one rank must stop every rank
at the checkpoint, or that rank breaks out of the time loop while its peers
enter the checkpoint's collectives and hang to walltime.
"""
from __future__ import annotations

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from tests.distributed.test_mpas_energy_tracker_mpi import _build  # noqa: E402

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2, reason="needs mpirun -n >= 2")


def test_a_nan_on_one_rank_fails_the_check_on_every_rank():
    d = _build(True)
    assert d._checkpoint_state_finite(), "control: clean state must pass"

    if d._voronoi_layout.rank == MPI.COMM_WORLD.Get_size() - 1:
        T = np.array(d.state.T.data)
        T[0, 0] = np.nan
        d.state = d.state._replace(T=d.state.T.replace(data=jnp.asarray(T)))

    assert not d._checkpoint_state_finite(), (
        f"rank {MPI.COMM_WORLD.Get_rank()} saw a finite state while another "
        "rank holds a NaN")
