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
identity = rsd.device_identity_keys


def _keys(*specs):
    """Build device identity keys from ``(host, token)`` pairs."""
    return [tuple(s) for s in specs]


def _rank_keys(cvd, n_local=1, host="node0"):
    """Identity keys one MPI rank contributes, as the call site builds them."""
    return identity(range(n_local), cvd, host)


# --------------------------------------------------------------------------
# identity construction: a device id is CUDA_VISIBLE_DEVICES-RELATIVE, so the
# raw ordinal is not a physical identity.
# --------------------------------------------------------------------------


def test_ordinal_is_resolved_through_the_mask():
    # Rank sees one device at local position 0, but the mask says that is
    # physical GPU 3.
    assert _rank_keys("3") == [("node0", "3")]


def test_overlapping_masks_alias_to_the_same_physical_gpu():
    # THE codex P1 case: masks 0,1 and 1,2 both expose physical GPU 1. As bare
    # ordinals these look like distinct devices and fake a wider run; resolved
    # through the mask, GPU 1 collides exactly once as it should.
    rank_a = _rank_keys("0,1", n_local=2)   # -> GPUs 0, 1
    rank_b = _rank_keys("1,2", n_local=2)   # -> GPUs 1, 2
    assert rank_a == [("node0", "0"), ("node0", "1")]
    assert rank_b == [("node0", "1"), ("node0", "2")]
    # 4 slots, only 3 distinct GPUs -> must NOT pass as a 4-device run.
    err = check(rank_a + rank_b, 4)
    assert err is not None and "oversubscribed" in err
    # ...and the honest width (3) is likewise refused, because a slot is doubled.
    assert check(rank_a + rank_b, 3) is not None


def test_reordered_mask_is_the_same_device_set():
    assert set(_rank_keys("1,0", n_local=2)) == set(_rank_keys("0,1", n_local=2))


def test_uuid_mask_entries_pass_through():
    # CUDA_VISIBLE_DEVICES may hold UUIDs; those are already stable identities.
    assert _rank_keys("GPU-abc,GPU-def", n_local=2) == [
        ("node0", "GPU-abc"), ("node0", "GPU-def")]


def test_unmasked_devices_use_the_plain_ordinal():
    assert _rank_keys(None, n_local=2) == [("node0", "0"), ("node0", "1")]
    assert _rank_keys("", n_local=2) == [("node0", "0"), ("node0", "1")]


def test_short_mask_does_not_alias_slots():
    # A mask that does not describe the device list must still yield DISTINCT
    # tokens — aliasing here would invent a false oversubscription.
    keys = _rank_keys("0", n_local=3)
    assert len(set(keys)) == 3


def test_per_rank_masks_compose_into_a_distinct_set():
    # The normal cluster shape: one GPU per rank, bound by mask.
    keys = _rank_keys("0") + _rank_keys("1") + _rank_keys("2")
    assert check(keys, 3) is None


def test_matching_distinct_device_count_passes():
    # Two ranks, each pinned to its own GPU via CUDA_VISIBLE_DEVICES: the
    # local device id is 0 on BOTH (independent JAX processes), so the CVD
    # mask is what makes them distinct.
    keys = _keys(("node0", "0"), ("node0", "1"))
    assert check(keys, 2) is None


def test_single_device_run_passes():
    assert check(_keys(("node0", "0")), 1) is None


def test_multi_node_same_local_ids_are_distinct():
    # Same CVD and same local id on different hosts are different GPUs.
    keys = _keys(("node0", "0"), ("node1", "0"))
    assert check(keys, 2) is None


def test_oversubscribed_ranks_are_rejected():
    # THE trap: two ranks on one physical GPU. A world_size check would call
    # this a 2-device run; the distinct-count check must refuse it.
    keys = _keys(("node0", "0"), ("node0", "0"))
    err = check(keys, 2)
    assert err is not None
    assert "oversubscribed" in err
    assert "2 device slot(s) map to only 1 distinct" in err


def test_narrowed_rung_is_rejected():
    # The CUDA_VISIBLE_DEVICES-narrowing trap: a rung meant to span 6 devices
    # silently runs on 1.
    err = check(_keys(("node0", "0")), 6)
    assert err is not None
    assert "--expect-devices 6" in err
    assert "1 distinct device(s)" in err


def test_extra_devices_are_rejected():
    # The opposite drift: a rank holding more GPUs than the rung intends.
    keys = _keys(("node0", "0"), ("node0", "1"))
    err = check(keys, 1)
    assert err is not None
    assert "2 distinct device(s)" in err


def test_detail_string_is_surfaced_for_the_operator():
    # The message must carry the diagnosis context, otherwise a failed rung
    # gives the operator nothing to act on.
    err = check(_keys(("node0", "0")), 4, detail="world_size=1, CVD=0")
    assert "world_size=1, CVD=0" in err
