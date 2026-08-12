"""``run_duo_stepper_w2.py`` must run the external-mode filter OFF.

Every Zenodo duo deck RESOLVES ``D_EXT = 0.000000000000000E+000`` --
`C48.sw.case2.alpha0.duo.hord8/rundir/logfile.000000.out:406`, and the
same for case6, case8 and nh.case-13. The decks' ``input.nml`` never
mentions ``d_ext``, and ``fv_arrays.F90`` carries two conflicting
declarations (``:392`` 0.0, ``:399`` 0.02), so the resolved echo is the
only authority.

Until 2026-08-06 this runner passed no value and inherited the stepper's
0.02, applying a divergence filter the oracle does not have -- directly
on the panel-seam mode the W2 imprint metric scores. Its siblings were
already right (`run_duo_stepper_w5.py:175`, `run_duo_stepper_modon.py:192`).

These tests pin the default AND prove the value is threaded, because a
default that never reaches the stepper would pass a value-only check
while changing nothing.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "fv3_native" / "run_duo_stepper_w2.py")


def _load():
    spec = importlib.util.spec_from_file_location("run_duo_stepper_w2",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w2 = _load()


class _Recorder:
    """Stands in for the duo stepper and records the d_ext it was given."""

    def __init__(self):
        self.d_ext_seen = []

    def install(self, monkeypatch):
        stub = types.ModuleType("legoesm.core.fv3_native_duo_stepper")

        def build_six_face_duo_context(n, ng, **kw):
            return {"n": n, "ng": ng}

        def w2_six_face_state(ctx):
            return [np.zeros((2, 2)) for _ in range(6)]

        def full_acoustic_step_sixface(ctx, states, dt, d_ext=0.02, **kw):
            self.d_ext_seen.append(d_ext)
            return states

        stub.build_six_face_duo_context = build_six_face_duo_context
        stub.w2_six_face_state = w2_six_face_state
        stub.full_acoustic_step_sixface = full_acoustic_step_sixface
        monkeypatch.setitem(sys.modules,
                            "legoesm.core.fv3_native_duo_stepper", stub)
        # main() also builds a lat-lon sampling map and samples v from the
        # real context; both are geometry, not the thing under test.
        monkeypatch.setattr(w2, "build_nearest_map",
                            lambda ctx: np.zeros(181 * 360, dtype=int))
        monkeypatch.setattr(w2, "geographic_va",
                            lambda ctx, states: [np.zeros(4)] * 6)
        return self


def _run(monkeypatch, tmp_path, extra_argv):
    rec = _Recorder().install(monkeypatch)
    out = tmp_path / "w2.npz"
    monkeypatch.setattr(
        sys, "argv",
        ["run_duo_stepper_w2.py", "--n", "8", "--dt", "43200",
         "--days", "1", "--out", str(out)] + extra_argv)
    w2.main()
    return rec, np.load(out, allow_pickle=True)


def test_default_d_ext_is_the_oracle_deck_value(monkeypatch, tmp_path):
    rec, z = _run(monkeypatch, tmp_path, [])
    assert rec.d_ext_seen, "the stepper was never called"
    assert set(rec.d_ext_seen) == {0.0}
    assert float(z["d_ext"]) == 0.0


def test_d_ext_is_actually_threaded_not_just_defaulted(monkeypatch, tmp_path):
    """Non-vacuity: if the flag were parsed but dropped, the previous test
    would still pass (the stepper's own default is 0.02, not 0.0 -- but a
    future refactor could change that). Prove a NON-default value arrives."""
    rec, z = _run(monkeypatch, tmp_path, ["--d-ext", "0.02"])
    assert set(rec.d_ext_seen) == {0.02}
    assert float(z["d_ext"]) == 0.02


def test_a_third_value_also_arrives(monkeypatch, tmp_path):
    """Two points rule out a hardcoded pair."""
    rec, _ = _run(monkeypatch, tmp_path, ["--d-ext", "0.005"])
    assert set(rec.d_ext_seen) == {0.005}


def test_npz_records_provenance(monkeypatch, tmp_path):
    """An artifact without its commit is not comparable to anything; this
    runner was the one duo runner not recording a SHA."""
    _, z = _run(monkeypatch, tmp_path, [])
    for key in ("d_ext", "git_sha", "dt", "n", "times_days", "v",
                "lat", "lon", "protocol"):
        assert key in z.files, f"npz is missing {key!r}"
    assert str(z["git_sha"]) != ""
    assert "d_ext=0.0" in str(z["protocol"])


def test_oracle_decks_really_resolve_d_ext_zero():
    """The claim this whole change rests on, checked against the shipped
    reference decks rather than repeated from a comment. Skipped off-box."""
    root = pathlib.Path(
        "/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/extracted/"
        "Code and simulations files")
    if not root.is_dir():
        pytest.skip("Zenodo reference decks not on this machine")
    decks = sorted(root.glob("*.duo.*/rundir/logfile.000000.out"))
    if not decks:
        pytest.skip("no duo deck logfiles found")
    seen = {}
    for log in decks:
        for line in log.read_text(errors="ignore").splitlines():
            if line.strip().upper().startswith("D_EXT"):
                seen[log.parts[-3]] = float(
                    line.split("=")[1].strip().rstrip(","))
                break
    assert seen, "no D_EXT line in any duo deck logfile"
    assert set(seen.values()) == {0.0}, f"decks disagree on D_EXT: {seen}"
