"""Structural gates for the standing gradient-gate triage probe.

The probe itself is a MEASUREMENT tool -- running its five blocks needs
the full JAX lane and minutes of compute, which is what its sbatch is
for.  What is worth asserting cheaply, and what this file asserts, is
the property that makes its output trustworthy: every block
PRE-REGISTERS what would confirm and what would refute it, and no block
prints a verdict of its own.

That is not decoration.  This repo's two most expensive diagnostic
failures both came from a probe whose interpretation was baked into the
tool and then echoed back as evidence.
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib

import pytest

_PROBE = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "validate" / "fv3_gradient_gate_triage.py")


def _load():
    spec = importlib.util.spec_from_file_location("_fv3_gate_triage", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_file_exists():
    assert _PROBE.is_file(), _PROBE


def test_every_block_is_registered_and_callable():
    mod = _load()
    assert set(mod._BLOCKS) == {"P1", "P2", "P3", "P4", "P5"}
    for key, fn in mod._BLOCKS.items():
        assert callable(fn), key
        assert fn.__doc__, f"{key} has no docstring"


def test_every_block_pre_registers_confirm_and_refute():
    """The property the probe's credibility rests on.

    Each block calls ``_hdr(name, question, confirms, refutes)`` before
    it measures anything, so the log states the prediction ahead of the
    number.  Asserted by reading the SOURCE of each block, because a
    block that dropped the call would still run fine and would still
    print numbers -- it would just have stopped being falsifiable.
    """
    mod = _load()
    for key, fn in mod._BLOCKS.items():
        src = inspect.getsource(fn)
        assert "_hdr(" in src, f"{key} does not pre-register a prediction"
        head = src.split("_hdr(", 1)[1]
        # four arguments: name, question, confirms, refutes
        assert head.count(",") >= 3, f"{key}'s _hdr call is incomplete"


@pytest.mark.parametrize("banned", ["VERDICT", "PROVES",
                                    "CONFIRMED:", "REFUTED:"])
def test_no_block_prints_its_own_verdict(banned):
    """A probe must not interpret itself.

    ``CONFIRMS:``/``REFUTES:`` are the pre-registration and are passed
    to ``_hdr`` BEFORE the measurement, so they may say "=> X"; what is
    banned is a verdict on a ``print`` line, i.e. after the number is
    in hand.  A block that printed "the growth is REAL" would be putting
    the conclusion in the tool, which is how a confidently wrong claim
    gets manufactured in this repo and echoed back as evidence.

    Only ``print(`` lines are scanned, deliberately: scanning every
    string literal would flag the pre-registration itself, and a gate
    that fires on the thing it is meant to require is worse than none.
    """
    mod = _load()
    for key, fn in mod._BLOCKS.items():
        printed = [ln for ln in inspect.getsource(fn).splitlines()
                   if "print(" in ln]
        assert not any(banned in ln for ln in printed), (key, banned)
