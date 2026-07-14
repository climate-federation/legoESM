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


def test_dp_updates_per_epoch_sums_floors():
    # The LR schedule must be sized by the REAL per-rank update count under
    # sharding: sum_chunks floor(chunk_size / nproc), not the unsharded total
    # (#985 -- else the cosine schedule runs ~nproc x too long).
    from legoesm.training.neural_gcm_spectral import _dp_updates_per_epoch
    assert _dp_updates_per_epoch([10, 8, 9], 4) == 6   # 2 + 2 + 2
    assert _dp_updates_per_epoch([16, 16], 4) == 8
    assert _dp_updates_per_epoch([100], 1) == 100      # nproc==1 identity


def test_dp_updates_per_epoch_raises_on_undersized_chunk():
    # A chunk smaller than the world size would hand a rank an empty shard ->
    # zero updates for a whole epoch; fail fast instead of silently no-op'ing.
    from legoesm.training.neural_gcm_spectral import _dp_updates_per_epoch
    with pytest.raises(ValueError, match="world size"):
        _dp_updates_per_epoch([10, 3], 4)
    with pytest.raises(ValueError):
        _dp_updates_per_epoch([], 2)


def test_dp_chunk_sizes_requires_attribute_on_custom_loader():
    # DP shards PER CHUNK, so a multi-chunk loader must expose chunk_sizes or the
    # schedule is mis-sized and an undersized chunk is silently skipped (#985).
    from legoesm.training.neural_gcm_spectral import _dp_chunk_sizes

    assert _dp_chunk_sizes(None, 40) == [40]          # single in-memory pass = 1 chunk

    def _loader_with(sizes):
        def _l(start_chunk=0):
            yield from ()
        _l.chunk_sizes = sizes
        return _l

    assert _dp_chunk_sizes(_loader_with([5, 3]), 8) == [5, 3]

    def _loader_without(start_chunk=0):
        yield from ()

    with pytest.raises(ValueError, match="chunk_sizes"):
        _dp_chunk_sizes(_loader_without, 8)


def test_mpi_rank_size_no_launcher_is_single():
    # With no multi-rank launcher env, discovery returns (0, 1) so callers stay
    # on their serial path.
    for v in ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALNRANKS"):
        val = os.environ.get(v, "")
        if val.isdigit() and int(val) > 1:
            pytest.skip("running under a real multi-rank launcher")
    assert dp.mpi_rank_size() == (0, 1)
