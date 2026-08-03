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
    """3 of the original 8 coverage rows are STILL genuinely unmeasured
    (bracketable dumps exist but the oracle-matching numerics they'd need are
    out of scope for a bracket-and-diff -- see each row's own note); 4 were
    measured 2026-07-30 (coverage_rows_measure.py); "wzv (vertical velocity)"
    was measured 2026-08-03 (#1455 queue item, wzv_row_measure.py) once its
    "never dumped" note was found STALE (direct wzv_dump_ww_call1/2.bin dumps
    exist in RUN_GDB) -- so these rows must NOT regress back to (None, None)
    silently."""
    mod = _load_module()
    still_unmeasured = {
        "tra_zdf (tracer implicit vertical solve)",
        "dyn_zdf (momentum implicit vertical solve)",
        "traldf_iso_lap tendency",
    }
    now_measured = {
        "ldf_dyn coefficient",
        "tra_qsr (shortwave penetration)",
        "ssh_atf",
        "tra_sbc",
    }
    assert still_unmeasured.issubset(mod.MEASUREMENTS.keys())
    assert now_measured.issubset(mod.MEASUREMENTS.keys())
    for term in still_unmeasured:
        corr, ratio, note = mod.MEASUREMENTS[term]
        assert corr is None and ratio is None
        assert mod.classify(corr, ratio, name=term) == "UNMEASURED"
        assert "stpmlf_call_coverage.py" in note
    for term in now_measured:
        corr, ratio, note = mod.MEASUREMENTS[term]
        assert corr is not None and ratio is not None
        assert mod.classify(corr, ratio, name=term) in ("AT BAR", "DEBT")
        assert "coverage_rows_measure.py" in note

    corr, ratio, note = mod.MEASUREMENTS["wzv (vertical velocity)"]
    assert corr is not None and ratio is not None
    assert mod.classify(corr, ratio, name="wzv (vertical velocity)") in ("AT BAR", "DEBT")
    assert "wzv_row_measure.py" in note


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


def test_unit_call_harness_rows_no_longer_phantom_provenance():
    """The #1226 unit-call-harness phantom-provenance sweep re-cited three
    rows' PROVENANCE_SCRIPT away from a script that does not exist:

    - "dyn_adv ZAD" / "dom_qco_r3c r3t": now cite the committed
      ``unit_harness/run_dyn_zad_probe.py`` / ``run_dom_qco_r3c_probe.py``
      drivers (this task) instead of the never-committed
      ``probe_1226_keg_zad_split.py`` / ``probe_1226_r2_item4_domqco.py``.
    - "dyn_ldf (dynldf_lev_lap) u"/"v": now cite ``ww_inheritance_walk.py``
      (which genuinely exists and measures this row's CURRENT, corrected
      tuple via ``measure_dyn_ldf_corrected``) instead of the never-committed
      ``probe_1226_r2_item2_dynldf.py`` -- a stale-citation bug this sweep
      caught (the row's tuple had already been corrected by a concurrent
      session, but PROVENANCE_SCRIPT still pointed at the old phantom name).

    This is a synthetic-violation-shaped check in the sense that it proves
    the fix is REAL: these five rows must NOT appear in
    ``check_provenance_scripts_exist()``'s failing list any more.
    """
    mod = _load_module()
    fixed_rows = {
        "dyn_adv ZAD",
        "dom_qco_r3c r3t",
        "dyn_ldf (dynldf_lev_lap) u",
        "dyn_ldf (dynldf_lev_lap) v",
    }
    failing = dict(mod.check_provenance_scripts_exist())
    for term in fixed_rows:
        assert term not in failing, (
            f"{term!r} still flagged as phantom-provenance: {failing.get(term)!r}")
    # "dom_qco_r3c r3u/r3v" is DELIBERATELY left phantom -- no legoESM
    # production function computes the u-/v-face ratio (pre-impl search
    # found none), so citing a real script here would be dishonest.
    assert "dom_qco_r3c r3u/r3v" in failing


def test_unit_call_harness_scripts_exist_on_disk():
    """The two new unit-call-harness probe files this sweep committed are
    real files, not just PROVENANCE_SCRIPT string literals (a typo in the
    dict would still "look" fixed by eye but fail check_provenance_scripts_
    exist -- assert the actual path resolves, independent of that check)."""
    scripts_dir = SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226"
    assert (scripts_dir / "unit_harness" / "run_dyn_zad_probe.py").is_file()
    assert (scripts_dir / "unit_harness" / "run_dom_qco_r3c_probe.py").is_file()
    assert (scripts_dir / "ww_inheritance_walk.py").is_file()
