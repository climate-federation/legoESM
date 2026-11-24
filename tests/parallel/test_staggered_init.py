"""legoesm.parallel.staggered_init: node-round staggering + heap release."""

from __future__ import annotations

import ctypes
import warnings

import pytest

from legoesm.parallel import staggered_init as si


class _FakeNode:
    def __init__(self, rank, size, log):
        self.rank, self.size, self.log = rank, size, log

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Barrier(self):
        self.log.append("barrier")

    def Free(self):
        self.log.append("free")


class _FakeComm:
    def __init__(self, rank, size, log):
        self.node = _FakeNode(rank, size, log)

    def Split_type(self, split_type):
        return self.node


@pytest.mark.parametrize("rank,size,k,calls_before", [
    (0, 20, 8, 0), (7, 20, 8, 0), (8, 20, 8, 1), (19, 20, 8, 2), (3, 4, 8, 0),
])
def test_rank_runs_in_its_round_and_every_rank_meets_every_barrier(
        monkeypatch, rank, size, k, calls_before):
    log = []
    monkeypatch.setattr(si, "release_freed_heap", lambda: log.append("trim"))
    out = si.staggered_first_call(lambda x: log.append("call") or x + 1, 41,
                                  comm=_FakeComm(rank, size, log),
                                  ranks_per_round=k)
    assert out == 42
    n_rounds = -(-size // k)
    assert log.count("barrier") == n_rounds and log[-1] == "free"
    i = log.index("call")
    assert log[:i].count("barrier") == calls_before
    assert log[i + 1] == "trim"  # heap returned before the next barrier


def test_serial_path_calls_then_trims(monkeypatch):
    log = []
    monkeypatch.setattr(si, "release_freed_heap", lambda: log.append("trim"))
    assert si.staggered_first_call(lambda: log.append("call") or 7) == 7
    assert log == ["call", "trim"]


@pytest.mark.parametrize("k", [None, 0])
def test_ranks_per_round_required_with_comm(k):
    with pytest.raises(ValueError, match="ranks_per_round"):
        si.staggered_first_call(lambda: 0, comm=_FakeComm(0, 2, []),
                                ranks_per_round=k)


def test_release_freed_heap_calls_malloc_trim_zero(monkeypatch):
    calls = []

    class _Trim:
        def __call__(self, pad):
            calls.append(pad)
            return 1

    class _Libc:
        malloc_trim = _Trim()

    monkeypatch.setattr(si.ctypes, "CDLL", lambda name: _Libc())
    assert si.release_freed_heap() is True
    assert calls == [0]


def test_release_freed_heap_runs_on_this_libc():
    if getattr(ctypes.CDLL(None), "malloc_trim", None) is None:
        pytest.skip("no malloc_trim in this C library")
    assert si.release_freed_heap() is True


def test_failing_rank_attends_all_barriers_then_raises():
    log = []

    def boom():
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        si.staggered_first_call(boom, comm=_FakeComm(0, 20, log),
                                ranks_per_round=8)
    assert log == ["barrier"] * 3 + ["free"]


def test_missing_malloc_trim_warns_once(monkeypatch):
    class _NoTrim:
        pass

    monkeypatch.setattr(si.ctypes, "CDLL", lambda name: _NoTrim())
    monkeypatch.setattr(si, "_warned_no_trim", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert si.release_freed_heap() is False
        assert si.release_freed_heap() is False
    assert sum("malloc_trim" in str(x.message) for x in w) == 1
