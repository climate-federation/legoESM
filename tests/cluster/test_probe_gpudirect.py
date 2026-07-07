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


def test_n_elements_rounds_and_floors():
    assert probe.n_elements(4) == 1          # the 4-byte scount=1 message
    assert probe.n_elements(4096) == 1024
    assert probe.n_elements(6) == 1          # rounds down, never below 1


def test_n_elements_rejects_too_small():
    with pytest.raises(ValueError):
        probe.n_elements(0)


def test_parse_args_defaults_and_escalation():
    d = probe.parse_args([])
    assert d.jit is False and d.iters == 1 and d.nbytes == 4
    e = probe.parse_args(["--jit", "--iters", "100", "--bytes", "65536"])
    assert e.jit is True and e.iters == 100 and e.nbytes == 65536
