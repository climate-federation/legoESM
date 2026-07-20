"""Direct tests for the ``--expect-devices`` anti-fake-scaling gate.

A scaling/census row is only meaningful if the run actually spanned the device
count it claims.  Rank count is NOT a device count: under route-A MPI each rank
builds its mesh from its own local devices, so ``-np 2`` can span four device
slots (both ranks inheriting ``CUDA_VISIBLE_DEVICES=0,1``) or one (both ranks
pinned to the same GPU) — and a ``world_size``-based check passes in both cases
while the recorded row is fake.  The gate therefore counts DISTINCT physical
devices, and these tests lock both failure modes so neither can be silently
weakened back into a rank-count check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD = (Path(__file__).resolve().parents[2]
        / "scripts" / "bench" / "run_scaling_diagnosis.py")
_spec = importlib.util.spec_from_file_location("run_scaling_diagnosis", _MOD)
rsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsd)

check = rsd.check_expected_devices


def _keys(*specs):
    """Build device identity keys from ``(host, cvd, device_id)`` triples."""
    return [tuple(s) for s in specs]


def test_matching_distinct_device_count_passes():
    # Two ranks, each pinned to its own GPU via CUDA_VISIBLE_DEVICES: the
    # local device id is 0 on BOTH (independent JAX processes), so the CVD
    # mask is what makes them distinct.
    keys = _keys(("node0", "0", 0), ("node0", "1", 0))
    assert check(keys, 2) is None


def test_single_device_run_passes():
    assert check(_keys(("node0", None, 0)), 1) is None


def test_multi_node_same_local_ids_are_distinct():
    # Same CVD and same local id on different hosts are different GPUs.
    keys = _keys(("node0", "0", 0), ("node1", "0", 0))
    assert check(keys, 2) is None


def test_oversubscribed_ranks_are_rejected():
    # THE trap: two ranks on one physical GPU. A world_size check would call
    # this a 2-device run; the distinct-count check must refuse it.
    keys = _keys(("node0", "0", 0), ("node0", "0", 0))
    err = check(keys, 2)
    assert err is not None
    assert "oversubscribed" in err
    assert "2 device slot(s) map to only 1 distinct" in err


def test_narrowed_rung_is_rejected():
    # The CUDA_VISIBLE_DEVICES-narrowing trap: a rung meant to span 6 devices
    # silently runs on 1.
    err = check(_keys(("node0", "0", 0)), 6)
    assert err is not None
    assert "--expect-devices 6" in err
    assert "1 distinct device(s)" in err


def test_extra_devices_are_rejected():
    # The opposite drift: a rank holding more GPUs than the rung intends.
    keys = _keys(("node0", "0,1", 0), ("node0", "0,1", 1))
    err = check(keys, 1)
    assert err is not None
    assert "2 distinct device(s)" in err


def test_detail_string_is_surfaced_for_the_operator():
    # The message must carry the diagnosis context, otherwise a failed rung
    # gives the operator nothing to act on.
    err = check(_keys(("node0", "0", 0)), 4, detail="world_size=1, CVD=0")
    assert "world_size=1, CVD=0" in err
