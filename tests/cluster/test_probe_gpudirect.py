"""Direct test for the GPU-direct probe's pure ring-neighbour logic.

The MPI/jax/mpi4jax imports live inside ``probe_gpudirect.main`` so the module
is import-safe on a plain host; only the pure ``expected_recv_value`` helper is
exercised here (the fabric behaviour itself needs a >=2-node GPU allocation)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_P = (Path(__file__).resolve().parents[2]
      / "scripts" / "cluster" / "scaling_derecho" / "probe_gpudirect.py")
_spec = importlib.util.spec_from_file_location("probe_gpudirect", _P)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_expected_recv_ring_8():
    # rank r receives from (r-1)%8, whose payload is (sender+1).
    assert probe.expected_recv_value(0, 8) == 8.0   # source 7 -> 8
    assert probe.expected_recv_value(1, 8) == 1.0   # source 0 -> 1
    assert probe.expected_recv_value(4, 8) == 4.0   # source 3 -> 4
    assert probe.expected_recv_value(7, 8) == 7.0   # source 6 -> 7


def test_expected_recv_wraps_at_zero():
    # The wrap-around rank (0) must receive from the last rank.
    assert probe.expected_recv_value(0, 2) == 2.0
    assert probe.expected_recv_value(1, 2) == 1.0


def test_expected_recv_rejects_nonpositive_nproc():
    with pytest.raises(ValueError):
        probe.expected_recv_value(0, 0)
