"""#985 item 1: data-parallel gating for the AIMIP chunked spectral trainer.

The DP path activates ONLY when ``config.data_parallel`` is set AND a multi-rank
launcher yields ``nproc > 1``; every other case falls back to the serial fused
step (byte-identical to pre-#985).  The end-to-end replica-sync proof is the
real-MPI test ``tests/distributed/test_aimip_data_parallel_mpi.py``.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import legoesm.training.data_parallel as dp
import pytest
from legoesm.training.neural_gcm_spectral import _resolve_dp_context


def test_dp_off_when_flag_false(monkeypatch):
    # Flag off -> never DP, and mpi_rank_size is not even consulted.
    def _boom():
        raise AssertionError("mpi_rank_size must not be probed when the flag is off")

    monkeypatch.setattr(dp, "mpi_rank_size", _boom)
    assert _resolve_dp_context(SimpleNamespace(data_parallel=False)) == (False, 0, 1, None)
    assert _resolve_dp_context(SimpleNamespace()) == (False, 0, 1, None)  # missing field


def test_dp_off_when_single_rank(monkeypatch):
    # Flag on but no multi-rank launcher -> serial (identity), never a lone-rank
    # "data-parallel" that would skip the (absent) cross-rank average.
    monkeypatch.setattr(dp, "mpi_rank_size", lambda: (0, 1))
    assert _resolve_dp_context(SimpleNamespace(data_parallel=True)) == (False, 0, 1, None)


def test_dp_on_when_flag_and_multirank(monkeypatch):
    monkeypatch.setattr(dp, "mpi_rank_size", lambda: (3, 4))
    assert _resolve_dp_context(SimpleNamespace(data_parallel=True)) == (True, 3, 4, None)


def test_mpi_rank_size_no_launcher_is_single():
    # With no multi-rank launcher env, discovery returns (0, 1) so callers stay
    # on their serial path.
    for v in ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALNRANKS"):
        val = os.environ.get(v, "")
        if val.isdigit() and int(val) > 1:
            pytest.skip("running under a real multi-rank launcher")
    assert dp.mpi_rank_size() == (0, 1)
