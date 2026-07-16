"""Gates for the Held-Suarez jet-strength floor (#1028) and the matrix
known-failures / XFAIL registry (#1029).

Pure-Python logic tests on the matrix-runner helpers — no dycore run, no x64
needed. Spec-load the script the same way the modon-catalog gate does so the
import stays sys.path-clean.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_matrix_module():
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    name = "_jet_known_failures_matrix_unit"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


mod = _load_matrix_module()


# --- #1028 jet-strength floor ------------------------------------------------

def test_jet_floor_fails_dead_full_run():
    """Full Held-Suarez run below the jet floor -> FAIL (the cd-grid cube case)."""
    ok, notes = mod._apply_jet_strength_floor(True, "n", 7.3, "held_suarez", 200.0)
    assert ok is False and "DEAD JET" in notes


def test_jet_floor_passes_healthy_full_run():
    """Healthy jet (spectral/latlon/ico reach 27-66 m/s) keeps PASS."""
    ok, notes = mod._apply_jet_strength_floor(True, "n", 41.0, "held_suarez", 200.0)
    assert ok is True and notes == "n"


def test_jet_floor_catches_nan():
    ok, _ = mod._apply_jet_strength_floor(True, "n", float("nan"), "held_suarez", 200.0)
    assert ok is False


def test_jet_floor_skips_quick_and_topo():
    """Quick runs (jet not spun up) and the topo case are NOT gated."""
    # quick run: too short to judge -> unchanged
    assert mod._apply_jet_strength_floor(True, "n", 5.0, "held_suarez", 30.0) == (True, "n")
    # topo case (may blow up first; #1029) -> unchanged even at full duration
    assert mod._apply_jet_strength_floor(True, "n", 5.0, "held_suarez_topo", 200.0) == (True, "n")


def test_jet_floor_idempotent_on_failed():
    """A run already FAILed upstream stays FAILed."""
    ok, _ = mod._apply_jet_strength_floor(False, "n", 41.0, "held_suarez", 200.0)
    assert ok is False


# --- #1029 known-failures registry ------------------------------------------

def _tc(case, grid, vert):
    return mod.TestCase("hydrostatic", case, grid, "72x144", vert, 200, 2)


def test_known_failure_registered_latlon_topo():
    assert ("held_suarez_topo", "latlon", "hybrid") in mod.KNOWN_FAILURES


_BLOWUP = "mass drift=nan, max|v|=nan; BLOWUP: state non-finite (NaN/Inf)"


def test_known_failure_waives_only_the_reproduced_blowup():
    """Full run: the tagged blow-up FAIL -> XFAIL; every other status/cause is
    NOT masked (codex rounds 1+2)."""
    tc = _tc("held_suarez_topo", "latlon", "hybrid")
    assert mod._apply_known_failure(tc, "FAIL", 200.0, _BLOWUP) == "XFAIL"
    # differently-caused full-run FAIL (no BLOWUP tag) stays red -> exit-gated
    assert mod._apply_known_failure(tc, "FAIL", 200.0, "mass drift=1e-3") == "FAIL"
    assert mod._apply_known_failure(tc, "ERROR", 200.0, _BLOWUP) == "ERROR"  # infra stays red
    assert mod._apply_known_failure(tc, "PASS", 200.0, "n") == "XPASS"       # fixed -> alert
    assert mod._apply_known_failure(tc, "SKIP", 200.0, _BLOWUP) == "SKIP"


def test_known_failure_short_run_reports_true_status():
    """Quick 2-day run never reaches the blow-up step -> no spurious XFAIL/XPASS."""
    tc = _tc("held_suarez_topo", "latlon", "hybrid")
    for s in ("PASS", "FAIL", "ERROR", "SKIP"):
        assert mod._apply_known_failure(tc, s, 2.0, _BLOWUP) == s


def test_known_failure_passthrough_for_unregistered():
    tc = _tc("held_suarez", "cubed_sphere", "hybrid")   # not registered
    for s in ("PASS", "FAIL", "ERROR", "SKIP"):
        assert mod._apply_known_failure(tc, s, 200.0, _BLOWUP) == s
