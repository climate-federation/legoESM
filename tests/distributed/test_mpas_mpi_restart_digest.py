"""The MPAS column-lane restart is bitwise under a 2-rank cell partition.

Run::

    srun --mpi=pmix -n 2 python -m pytest tests/distributed/test_mpas_mpi_restart_digest.py

The single-rank gate (``tests/unit/test_column_lane_restart_digest.py``)
cannot see the partition: ``save_checkpoint`` gathers the owned cells to a
GLOBAL file and ``load_checkpoint`` scatters it back, so a rank-ordering or
partition drift between the writing and the reading run would surface only
here.  Each rank hashes its own cells (``digest_steps_r<rank>.log``); the
continuous and the resumed run must agree on every rank, and both runs must
have partitioned the mesh identically.

Skips loudly when mpi4jax/mpi4py are absent (conftest) or when launched on
one rank.  Preconditions raise ``RuntimeError`` (see the unit module) so only
the digest equality and the partition identity can fail the test."""
import numpy as np
import pytest

from tests.unit.test_column_lane_restart_digest import (
    _drive, _mismatch, _mpas_continuous, _parts, _digests, _require,
    _restarted_vs_continuous)
from tests.unit.test_mpas_multilayer_land_port import (
    _LS_KW, ONE_STEP_DAYS, THREE_STEPS_DAYS, TWO_STEPS_DAYS, _build_driver,
    _patch_land_loaders)

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

comm = MPI.COMM_WORLD
pytestmark = pytest.mark.skipif(
    comm.Get_size() < 2,
    reason="needs >= 2 MPI ranks (srun --mpi=pmix -n 2): one rank owns the "
           "whole mesh and the partition path never runs")


@pytest.fixture()
def shared_dir(tmp_path):
    """Rank 0's tmp_path on every rank: the checkpoint is written by rank 0
    and read by all, and both ranks' digest logs land in one directory."""
    return comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0)


def _owned(drv):
    part = drv._voronoi_layout.partition
    return np.asarray(part.local_cells[:part.n_owned_cells])


def test_two_rank_mpas_restart_matches_the_continuous_digest(
        shared_dir, monkeypatch):
    import pathlib
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")
    rank, root = comm.Get_rank(), pathlib.Path(shared_dir)
    dC = _mpas_continuous(root, "c", THREE_STEPS_DAYS, monkeypatch,
                          distributed=True)
    _require(dC._voronoi_layout is not None, "partition layout not built")
    _require(dC._voronoi_layout.n_ranks == comm.Get_size(), "rank count")
    dA = _mpas_continuous(root, "a", TWO_STEPS_DAYS, monkeypatch,
                          distributed=True)
    comm.Barrier()                       # rank 0 wrote the global checkpoint
    ckpt = sorted((root / "a").glob("checkpoint_day_*.npz"))[-1]
    _patch_land_loaders(monkeypatch)
    dB = _drive(_build_driver, str(root / "b"), ONE_STEP_DAYS, **_LS_KW,
                land_stress_from_land=True, distributed=True)
    step, day = _drive(dB.load_checkpoint, ckpt)
    _require(step == 2, step)
    _require(_drive(dB.run, start_step=step, start_day=day) == "COMPLETED",
             "b")
    comm.Barrier()                       # every rank's digest log is closed

    # Partition identity between the writing and the reading run, on this
    # rank and (via the gathered owned sets) on every rank.
    same_part = bool(np.array_equal(_owned(dC), _owned(dA))
                     and np.array_equal(_owned(dC), _owned(dB)))
    assert all(comm.allgather(same_part)), "cell partition drifted between runs"

    steps, bad = _restarted_vs_continuous(root / "c", step, root / "b",
                                          rank=rank)
    _require(steps == [3], steps)
    bad_any = comm.allgather(bad)
    assert not any(bad_any), "\n".join(
        f"rank {r}: " + _mismatch(root / "c", root / "b", b[0], rank=r)
        for r, b in enumerate(bad_any) if b)
    # Each rank's digest is rank-local: the two ranks hash different cells.
    assert _parts(_digests(root / "c", 0)[1])["state"] != \
        _parts(_digests(root / "c", 1)[1])["state"]
