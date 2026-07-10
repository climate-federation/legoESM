"""``run_omip_core2 --spmd-persistent-state`` flag validation (scaling-M2).

Covers the pure combo validator ``_validate_spmd_persistent_state`` (fail-fast
BEFORE any device/data work) and that ``main()`` actually wires it: the flag
must be registered on the parser AND the invalid combos must SystemExit with
the validator's message (an unregistered flag would SystemExit with argparse's
``unrecognized arguments`` code 2 instead — asserted apart below).

The loop-level parity gate (persistent-sharded loop == per-step global-wrapper
loop, gather counts) lives in
``tests/parallel/test_persistent_sharded_ocean_loop.py``.
"""
from __future__ import annotations

import sys

import pytest

from scripts.run.run_omip_core2 import _validate_spmd_persistent_state


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
