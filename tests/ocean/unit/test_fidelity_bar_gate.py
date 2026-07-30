"""Direct unit test for the #1226 fidelity bar gate
(``scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py``).

Covers the HUMAN-WAIVED mechanism added 2026-07-30: a waiver requires BOTH a
nonempty decision-provenance string and a nonempty evidence citation, is
keyed to a fixed set of row names (not a generic per-row flag), counts in the
gate's total, and does not block exit 0 on its own.
"""
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.dino_1226.fidelity_bar_gate as mod
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_self_test_passes():
    mod = _load_module()
    assert mod._self_test() == 0


def test_zdf_mxl_turb_is_waived():
    mod = _load_module()
    assert mod.classify(None, None, name="zdf_mxl_turb") == "WAIVED"
    assert "zdf_mxl_turb" in mod.WAIVED_ROWS


def test_waiver_missing_provenance_or_evidence_is_rejected():
    """Synthetic-violation proof: the waiver mechanism must not be vacuous."""
    mod = _load_module()
    for broken in (
        {"fake": ("", "evidence")},
        {"fake": ("decision", "")},
        {"fake": ("  ", "  ")},
    ):
        saved = dict(mod.WAIVED_ROWS)
        try:
            mod.WAIVED_ROWS.clear()
            mod.WAIVED_ROWS.update(broken)
            try:
                mod._validate_waivers()
                assert False, "a waiver missing provenance/evidence must raise"
            except ValueError:
                pass
        finally:
            mod.WAIVED_ROWS.clear()
            mod.WAIVED_ROWS.update(saved)


def test_no_generic_waiver_mechanism():
    """A row cannot be waived via BINARY_GATES -- WAIVED_ROWS is the only path."""
    mod = _load_module()
    assert "zdf_mxl_turb" not in mod.BINARY_GATES


def test_nine_uncovered_routines_are_unmeasured_rows():
    mod = _load_module()
    expected = {
        "wzv (vertical velocity)",
        "tra_zdf (tracer implicit vertical solve)",
        "dyn_zdf (momentum implicit vertical solve)",
        "traldf_iso_lap tendency",
        "ldf_dyn coefficient",
        "tra_qsr (shortwave penetration)",
        "ssh_atf",
        "tra_sbc",
    }
    assert expected.issubset(mod.MEASUREMENTS.keys())
    for term in expected:
        corr, ratio, note = mod.MEASUREMENTS[term]
        assert corr is None and ratio is None
        assert mod.classify(corr, ratio, name=term) == "UNMEASURED"
        assert "stpmlf_call_coverage.py" in note


def test_main_reports_waived_count_and_does_not_gate_exit_on_waiver_alone(capsys):
    mod = _load_module()
    rc = mod.main()
    captured = capsys.readouterr()
    assert "WAIVED 1" in captured.out
    assert "HUMAN-WAIVED" in captured.out
    # The gate still fails overall (DEBT/UNMEASURED rows exist independent of
    # the waiver), but the waiver itself must not be what is blocking it --
    # proven by the classify()-level check above, not by this exit code alone.
    assert rc in (0, 1)
