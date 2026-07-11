"""``run_omip_core2 --spmd-persistent-state`` flag validation + residency
state machine (scaling-M2).

Covers the pure combo validator ``_validate_spmd_persistent_state`` (fail-fast
BEFORE any device/data work), that ``main()`` actually wires it (the flag must
be registered on the parser AND the invalid combos must SystemExit with the
validator's message — an unregistered flag would SystemExit with argparse's
``unrecognized arguments`` code 2 instead), and the
``_PersistentStateResidency`` state machine: flag/counter bookkeeping across
the production interleavings (unforced snapshot cadence AND the
prognostic-ice / relative-winds FORCED per-step gather alternation) —
supplementary-review finding: as a ``main()`` closure this logic was
untestable and the forced-gather sequence was gated nowhere.

The loop-level parity gate (persistent-sharded loop == per-step global-wrapper
loop, gather counts) lives in
``tests/parallel/test_persistent_sharded_ocean_loop.py``.
"""
from __future__ import annotations

import sys

import pytest

from scripts.run.run_omip_core2 import (
    _PersistentStateResidency,
    _validate_spmd_persistent_state,
)


def test_flag_off_is_always_valid():
    # OFF must never raise, whatever the other knobs say (byte-identical path).
    _validate_spmd_persistent_state(False, 1, False)
    _validate_spmd_persistent_state(False, 8, True)


def test_valid_single_controller_combo_passes():
    _validate_spmd_persistent_state(True, 2, False)
    _validate_spmd_persistent_state(True, 8, False)


def test_requires_multi_gpu():
    with pytest.raises(SystemExit, match="requires --n-gpus > 1"):
        _validate_spmd_persistent_state(True, 1, False)
    with pytest.raises(SystemExit, match="requires --n-gpus > 1"):
        _validate_spmd_persistent_state(True, 0, False)


def test_refuses_distributed():
    with pytest.raises(SystemExit, match="single-controller only"):
        _validate_spmd_persistent_state(True, 4, True)


def test_main_registers_flag_and_wires_validation(monkeypatch):
    """Drive the REAL entry: ``--spmd-persistent-state`` alone must reach the
    validator (proving parser registration + main() wiring) and exit with ITS
    message — not argparse's ``unrecognized arguments`` (exit code 2)."""
    import scripts.run.run_omip_core2 as R

    monkeypatch.setattr(
        sys, "argv", ["run_omip_core2.py", "--spmd-persistent-state"])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert "requires --n-gpus > 1" in str(e.value.code)


def test_main_refuses_distributed_combo_before_bootstrap(monkeypatch):
    """--distributed + --spmd-persistent-state must fail in the early combo
    validation, BEFORE the jax.distributed bootstrap is attempted."""
    import scripts.run.run_omip_core2 as R

    monkeypatch.setattr(
        sys, "argv", ["run_omip_core2.py", "--spmd-persistent-state",
                      "--n-gpus", "2", "--distributed"])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert "single-controller only" in str(e.value.code)


# ---------------------------------------------------------------------------
# _PersistentStateResidency: the residency state machine.  Stub layout flips
# tag the payload so both the RETURNED object and the counters pin residency.
# ---------------------------------------------------------------------------

def _tracked_residency(enabled=True):
    res = _PersistentStateResidency(
        enabled,
        shard_fn=lambda st: ("sharded", st),
        gather_fn=lambda st: ("global", st[1] if isinstance(st, tuple) else st),
    )
    return res


def test_residency_disabled_is_pure_identity():
    """Flag OFF: both methods are exact identity no-ops in ANY order — the
    byte-identical default-path guarantee."""
    res = _PersistentStateResidency(False)          # no flip fns needed
    st = object()
    assert res.ensure_sharded(st) is st
    assert res.ensure_global(st) is st
    assert res.ensure_sharded(res.ensure_global(res.ensure_sharded(st))) is st
    assert (res.sharded, res.gathers, res.shards) == (False, 0, 0)
    # leaf-transfer counters start at zero too (honest-cost companion)
    assert (res.leaf_slice_pulls, res.leaf_slice_writes,
            res.leaf_full_gathers, res.leaf_full_uploads) == (0, 0, 0, 0)


def test_leaf_transfer_counters_are_unconditional_and_independent():
    """The LEAF host-transfer counters (codex batch4 HIGH: the per-step
    surface-slice pulls / full-3-D leaf round trips that REMAIN in the
    persistent lane) accumulate regardless of the residency flag and never
    touch the full-STATE flip bookkeeping — leaf transfers happen on every
    lane; the full-state counters track layout flips only."""
    # disabled lane: leaf counting still records (transfers happen anyway)
    res = _PersistentStateResidency(False)
    res.count_leaf_slice(pulls=2, writes=1)
    res.count_leaf_full(gathers=2, uploads=2)
    assert (res.leaf_slice_pulls, res.leaf_slice_writes) == (2, 1)
    assert (res.leaf_full_gathers, res.leaf_full_uploads) == (2, 2)
    assert (res.sharded, res.gathers, res.shards) == (False, 0, 0)

    # enabled lane: leaf counts accumulate across residency flips without
    # perturbing them (and vice versa)
    res = _tracked_residency()
    st = res.ensure_sharded("ic")
    res.count_leaf_slice(pulls=1)                  # forcing-builder pull
    res.count_leaf_slice(pulls=1, writes=1)        # SSS-restore pull+write
    st = res.ensure_global(st)
    res.count_leaf_full(gathers=2, uploads=2)      # WOA-nudge T,S round trip
    st = res.ensure_sharded(st)
    assert (res.shards, res.gathers) == (2, 1)     # flips unaffected by leaves
    assert (res.leaf_slice_pulls, res.leaf_slice_writes) == (2, 1)
    assert (res.leaf_full_gathers, res.leaf_full_uploads) == (2, 2)
    # keyword-only signature: positional use is a bug, refuse it
    with pytest.raises(TypeError):
        res.count_leaf_slice(1)
    with pytest.raises(TypeError):
        res.count_leaf_full(1)


def test_residency_enabled_requires_flip_fns():
    with pytest.raises(ValueError, match="requires both"):
        _PersistentStateResidency(True, shard_fn=None, gather_fn=None)
    with pytest.raises(ValueError, match="requires both"):
        _PersistentStateResidency(True, shard_fn=lambda s: s, gather_fn=None)


def test_residency_idempotent_and_counted():
    """Repeated ensure_* in the SAME residency must not re-flip or
    double-count (the lazy-re-shard contract at snapshot boundaries)."""
    res = _tracked_residency()
    st = res.ensure_sharded("state0")
    assert st == ("sharded", "state0") and res.sharded
    assert res.ensure_sharded(st) is st            # idempotent, uncounted
    assert (res.shards, res.gathers) == (1, 0)
    st = res.ensure_global(st)
    assert st == ("global", "state0") and not res.sharded
    assert res.ensure_global(st) is st             # idempotent, uncounted
    assert (res.shards, res.gathers) == (1, 1)


def test_residency_unforced_loop_with_snapshot_cadence():
    """The production UNFORCED interleaving over 10 steps with a snapshot at
    step 5: initial shard + post-snapshot lazy re-shard + final gather =>
    {shards: 2, gathers: 2}, zero per-step transfers (mirrors the loop-level
    parity gate's counts)."""
    res = _tracked_residency()
    n_steps, snap_step = 10, 5
    st = "ic"
    for k in range(1, n_steps + 1):
        st = res.ensure_sharded(st)                # pre-step (no-op once sharded)
        assert res.sharded                         # step consumes SHARDED state
        if k == snap_step:
            st = res.ensure_global(st)             # snapshot boundary
            assert not res.sharded
    st = res.ensure_global(st)                     # final I/O boundary
    assert not res.sharded
    assert (res.shards, res.gathers) == (2, 2)


def test_residency_forced_per_step_gather_alternation():
    """The FORCED lane (prognostic ice / relative winds): loop-top
    ensure_global EVERY step, then pre-step ensure_sharded — the alternation
    must count exactly one gather + one shard per step after the first, stay
    consistent (flag matches the object tag at every point), and never
    silently skip a flip."""
    res = _tracked_residency()
    n_steps = 7
    st = "ic"
    for _k in range(1, n_steps + 1):
        st = res.ensure_global(st)                 # loop-top forced gather
        assert not res.sharded                     # ice/wind read GLOBAL state
        st = res.ensure_sharded(st)                # re-shard for the step
        assert res.sharded and st[0] == "sharded"
    st = res.ensure_global(st)                     # final boundary
    # step 1's loop-top gather is a no-op (state starts global) => n_steps-1
    # forced gathers + the final one; one shard per step.
    assert res.shards == n_steps
    assert res.gathers == n_steps                  # (n_steps - 1) forced + final
    assert not res.sharded
