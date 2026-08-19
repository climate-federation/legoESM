"""The gradient audit must never lose a run to a walltime kill again.

The turbulence category ran four hours and was killed at its limit with NOTHING
on disk, because results were written only after the whole run.  These pin the
three properties that fix depends on: it writes after every scheme, the file is
never observed half-written, and a truncated run is distinguishable from a
complete one.
"""

from __future__ import annotations

import json

import pytest

from scripts.validate.audit_param_gradients import _write_partial


class _V:
    """Minimal stand-in for ParamVerdict — `_write_partial` only asdict()s it."""

    def __init__(self, name):
        self.qualified_name = name


def _verdicts(n):
    from dataclasses import make_dataclass

    V = make_dataclass("V", [("qualified_name", str), ("verdict", str)])
    return [V(f"scheme.p{i}", "live") for i in range(n)]


def test_partial_write_is_readable_and_marked_incomplete(tmp_path):
    out = tmp_path / "cat.json"
    _write_partial(out, ["convection"], 30, 600.0, _verdicts(3), complete=False)
    d = json.loads(out.read_text())
    assert d["complete"] is False, (
        "a truncated run must be distinguishable from a full audit")
    assert d["n_verdicts"] == 3
    assert len(d["verdicts"]) == 3


def test_a_later_write_supersedes_an_earlier_one(tmp_path):
    out = tmp_path / "cat.json"
    _write_partial(out, ["convection"], 30, 600.0, _verdicts(2), complete=False)
    _write_partial(out, ["convection"], 30, 600.0, _verdicts(7), complete=True)
    d = json.loads(out.read_text())
    assert d["n_verdicts"] == 7
    assert d["complete"] is True


def test_no_temp_file_is_left_behind(tmp_path):
    """The write goes via a temp file and an atomic replace; a leftover .tmp
    would mean the replace did not happen and a reader could see a partial
    file."""
    out = tmp_path / "cat.json"
    _write_partial(out, ["convection"], 30, 600.0, _verdicts(4), complete=True)
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"
                 or p.name.endswith(".json.tmp")]
    assert not leftovers, f"temp file left behind: {leftovers}"
    assert out.exists()


def test_none_path_is_a_no_op(tmp_path):
    """`--json-out` is optional; the progress callback must not explode when it
    was not given."""
    _write_partial(None, ["convection"], 30, 600.0, _verdicts(1), complete=False)


def test_parent_directory_is_created(tmp_path):
    out = tmp_path / "nested" / "deeper" / "cat.json"
    _write_partial(out, ["turbulence"], 30, 600.0, _verdicts(1), complete=True)
    assert out.exists()


def test_the_audit_passes_a_progress_callback(tmp_path):
    """A guard on the wiring, not just the helper: `audit()` must accept and
    invoke `on_progress`, or the incremental write is dead code."""
    import inspect

    from scripts.validate import audit_param_gradients as A

    sig = inspect.signature(A.audit)
    assert "on_progress" in sig.parameters, (
        "audit() lost its on_progress hook; the incremental write would then "
        "never run and a walltime kill would again discard the whole run")
    src = inspect.getsource(A.audit)
    assert "on_progress(" in src, (
        "audit() accepts on_progress but never calls it")
    main_src = inspect.getsource(A.main)
    assert "on_progress=" in main_src and "_write_partial" in main_src, (
        "main() does not pass a progress callback, so nothing is written "
        "incrementally in a real run")
