"""Direct unit test for the #1226 stp_MLF call-graph coverage gate
(``scripts/validate/ocean_fidelity/dino_1226/stpmlf_call_coverage.py``).

The gate enumerates every ``CALL`` in NEMO's DINO-override time-stepping
routine (``cfgs/DINO/MY_SRC/stpmlf.F90``) and forces a COVERED/WAIVED/
UNCOVERED disposition on each one -- this test proves the gate itself is not
vacuous (a synthetic unaccounted entry MUST fail it) and that the real,
checked-in CALL list currently passes clean, matching the oracle-fidelity
skill's Rule 1 ("coverage, not a checklist").
"""
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.dino_1226.stpmlf_call_coverage as mod
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_self_test_passes():
    """The module's own --self-test: the real CALL list validates clean AND
    a synthetic unaccounted entry is caught (non-vacuity proof)."""
    mod = _load_module()
    assert mod.main(["--self-test"]) == 0


def test_gate_passes_and_reports_zero_unaccounted(capsys):
    mod = _load_module()
    rc = mod.main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "COVERAGE GATE PASSED" in captured.out


def test_every_call_has_a_disposition():
    """Direct check on the data, independent of the CLI: _validate must
    return no errors for the shipped CALLS list."""
    mod = _load_module()
    errors = mod._validate(mod.CALLS)
    assert errors == []


def test_synthetic_unaccounted_entry_is_caught():
    """Non-vacuity: an entry with an empty disposition must fail _validate."""
    mod = _load_module()
    broken = mod.build_synthetic_violation()
    errors = mod._validate(broken)
    assert errors != []
    # the synthetic entry itself must be named in at least one error
    assert any("CALL_synthetic_unaccounted" in e for e in errors)


def test_uncovered_entries_are_all_ranked():
    """Every UNCOVERED disposition must carry a rank -- enforced by
    _validate, re-checked here directly on the data so a future edit that
    adds an unranked UNCOVERED entry fails this test even if main()'s
    exit-code plumbing is ever changed."""
    mod = _load_module()
    for c in mod.CALLS:
        if c.disposition == mod.UNCOVERED:
            assert c.rank is not None, f"UNCOVERED entry at line {c.line} " \
                f"({c.routine!r}) has no rank"


def test_zdf_mxl_turb_is_enumerated_and_covered():
    """Sanity anchor for Task B of the #1226 stp_MLF coverage work: the
    turbocline diagnostic call must appear in the table (it does execute
    every step) and be dispositioned COVERED (a gate row exists, even though
    that row is itself unmeasured -- see fidelity_bar_gate.py's own
    'zdf_mxl_turb' entry and the companion consumer-grep verdict)."""
    mod = _load_module()
    hits = [c for c in mod.CALLS if "zdf_mxl_turb" in c.routine]
    assert hits, "zdf_mxl_turb must be enumerated in the call list"
    assert all(c.disposition == mod.COVERED for c in hits)
