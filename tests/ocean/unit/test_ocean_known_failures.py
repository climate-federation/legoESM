"""The ocean matrix's known-failure waiver must waive exactly one thing.

A waiver is the most dangerous construct in a test matrix: it converts a red
result into a green one by fiat, and if it is one character too broad it hides
the next real regression on the same case forever.  So every test here is
written to fail if the waiver widens.

The registry mirrors ``run_atmosphere_test_matrix.KNOWN_FAILURES`` (#1029)
rather than inventing a second convention: same key shape, same XFAIL/XPASS
semantics, same ``min_days`` guard.  It departs in one place, on review:
the waiver keys on a POSITIVE signature -- the recorded blow-up DAY falling
inside a band -- rather than on a substring of the failure notes.  This case
emits two different notes depending on which detector fires first, and no
substring covers both without also covering every unrelated regression.

The entry under test is ``eady_uniform / mpas_channel / 70km`` (#1609).
Centring the barotropic averaging window on ``t+dt`` restored the barotropic
mode to full forcing; the old half window applied only ~53% of the slow
baroclinic forcing.  That case was stable only because of the under-forcing.
See ``scripts/validate/ocean_fidelity/eady_channel_onset_structure.py`` for the
one-variable confirmation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MATRIX = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"


def _load_matrix():
    """Import the matrix script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_ocean_matrix_for_known_failures", _MATRIX)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_matrix()


#: The entry the registry used to hold, kept as the shape the machinery is
#: tested against. The registry itself is EMPTY -- the Eady channel passes
#: again -- but every guarantee below must still hold for the NEXT entry, so
#: the tests register this one temporarily rather than deleting themselves.
_ENTRY_SHAPE = {"issue": "#1609", "min_days": 68.0,
                "expect_day_range": (60.0, 75.0)}


@pytest.fixture(scope="module")
def entry_key():
    return ("eady_uniform", "mpas_channel", "70km")


@pytest.fixture
def registered(m, entry_key):
    """Temporarily register an entry so the waiver machinery can be tested."""
    m.KNOWN_FAILURES[entry_key] = dict(_ENTRY_SHAPE)
    try:
        yield entry_key
    finally:
        m.KNOWN_FAILURES.pop(entry_key, None)


_REMOVED_ENTRY = {
    "issue": "#1609", "min_days": 68.0, "expect_day_range": (60.0, 75.0)}


def _require_live(m, key):
    """Guard against the vacuity codex found.

    Every NEGATIVE test here asserts that some input is NOT waived. On an
    empty registry that is true for the boring reason -- ``_apply_known_failure``
    returns early when the key is unregistered -- so the test passes while
    proving nothing. Dropping the ``registered`` fixture from one of them was
    measured to leave it GREEN. This makes that failure mode impossible: the
    test now says out loud that the entry is live before asserting it is not
    applied."""
    assert key in m.KNOWN_FAILURES, (
        "no entry registered -- this negative test would pass vacuously; "
        "the `registered` fixture is missing")


def _case(m, key, **kw):
    case, grid, res = key
    return m.TestCase(case, grid, res, kw.pop("days", 200.0),
                      kw.pop("quick_days", 60.0))


def _out_dir(tmp_path, day):
    """A results.txt carrying a blow-up on ``day``, as the runner writes it."""
    d = tmp_path / "case"
    d.mkdir(exist_ok=True)
    (d / "results.txt").write_text(
        f"status: FAIL\nnotes: BLOWUP at step 19400 (day {day:.2f}), "
        f"reason: metric 5192.4 > threshold 100.0; last clean: "
        f"max_speed=1.9379m/s\n")
    return d


# ---------------------------------------------------------------------------
# The waiver applies where it should
# ---------------------------------------------------------------------------

def test_registered_fail_becomes_xfail(m, registered, entry_key, tmp_path):
    """The real case: a full-length run that blew up on the known day."""
    tc = _case(m, entry_key)
    d = _out_dir(tmp_path, 67.36)
    assert m._apply_known_failure(
        tc, "FAIL", 200.0, "Non-finite values in u", d) == "XFAIL"


def test_waiver_applies_at_exactly_min_days(m, registered, entry_key, tmp_path):
    """Boundary: ``days < min_days`` is the rejection test, so equality waives."""
    tc = _case(m, entry_key)
    min_days = m.KNOWN_FAILURES[entry_key]["min_days"]
    d = _out_dir(tmp_path, 67.36)
    assert m._apply_known_failure(tc, "FAIL", min_days, "boom", d) == "XFAIL"


def test_both_observed_failure_notes_are_waived(m, registered, entry_key, tmp_path):
    """The two detectors emit different notes; the day band covers both.

    This is why the waiver is NOT keyed on the notes text."""
    tc = _case(m, entry_key)
    d = _out_dir(tmp_path, 67.36)
    for note in ("Non-finite values in u",
                 "max_speed=1.9379m/s, T_drift=1.20e-12, Ld=107km, tau=5d"):
        assert m._apply_known_failure(tc, "FAIL", 200.0, note, d) == "XFAIL"


# ---------------------------------------------------------------------------
# The positive signature: a failure at the WRONG TIME is a different bug
# ---------------------------------------------------------------------------

def test_blowup_outside_the_day_band_is_not_waived(m, registered, entry_key, tmp_path):
    """An early blow-up cannot be this mechanism, whose onset is day 67.4."""
    _require_live(m, entry_key)
    tc = _case(m, entry_key)
    lo, hi = m.KNOWN_FAILURES[entry_key]["expect_day_range"]
    for day in (lo - 5.0, hi + 5.0, 1.0, 150.0):
        d = _out_dir(tmp_path, day)
        assert m._apply_known_failure(tc, "FAIL", 200.0, "boom", d) == "FAIL", day


def test_fail_with_no_recorded_blowup_is_not_waived(m, registered, entry_key, tmp_path):
    """Fail closed. A FAIL that never blew up is an unrelated regression."""
    _require_live(m, entry_key)
    tc = _case(m, entry_key)
    d = tmp_path / "nb"
    d.mkdir()
    (d / "results.txt").write_text(
        "status: FAIL\nnotes: growth rate 0.11 vs analytic 0.31\n")
    assert m._apply_known_failure(tc, "FAIL", 200.0, "rate", d) == "FAIL"


def test_missing_results_file_is_not_waived(m, registered, entry_key, tmp_path):
    tc = _case(m, entry_key)
    assert m._apply_known_failure(tc, "FAIL", 200.0, "boom",
                                  tmp_path / "absent") == "FAIL"
    assert m._apply_known_failure(tc, "FAIL", 200.0, "boom", None) == "FAIL"


def test_quick_lane_is_not_waived(m, registered, entry_key, tmp_path):
    """THE lane that must stay red.

    Onset is simulated day 67.4, so the 60-day --quick lane stops BEFORE the
    known blow-up and is expected to PASS on the Eady growth rate. Waiving it
    would paint an unrelated quick-lane bug green and destroy the cheapest
    early-warning lane for this case (GLM-5.2 review)."""
    _require_live(m, entry_key)
    _require_live(m, entry_key)
    tc = _case(m, entry_key)
    d = _out_dir(tmp_path, 40.0)
    assert m._apply_known_failure(tc, "FAIL", 60.0, "boom", d) == "FAIL"
    # ... and a quick-lane PASS must NOT be mislabelled a stale-waiver alarm.
    assert m._apply_known_failure(tc, "PASS", 60.0, "fine", d) == "PASS"


def test_min_days_exceeds_the_known_onset(m, registered, entry_key):
    """The invariant behind the two tests above, asserted directly."""
    e = m.KNOWN_FAILURES[entry_key]
    lo, _hi = e["expect_day_range"]
    assert e["min_days"] > 67.36, (
        "min_days must exceed the 67.36-day onset, else the quick lane -- "
        "which cannot reach the blow-up -- gets waived or alarms")
    assert lo <= 67.36 <= e["expect_day_range"][1]


# ---------------------------------------------------------------------------
# The waiver does NOT apply where it should not.  These are the tests that
# fail if someone widens the entry.
# ---------------------------------------------------------------------------

def test_short_run_is_not_waived(m, registered, entry_key, tmp_path):
    tc = _case(m, entry_key)
    min_days = m.KNOWN_FAILURES[entry_key]["min_days"]
    d = _out_dir(tmp_path, 67.36)
    assert m._apply_known_failure(
        tc, "FAIL", min_days - 1.0, "boom", d) == "FAIL"


def test_error_is_never_waived(m, registered, entry_key, tmp_path):
    """ERROR is infrastructure breakage, not the known physics failure."""
    _require_live(m, entry_key)
    _require_live(m, entry_key)
    tc = _case(m, entry_key)
    d = _out_dir(tmp_path, 67.36)
    assert m._apply_known_failure(
        tc, "ERROR", 200.0, "ImportError", d) == "ERROR"


def test_skip_passes_through(m, registered, entry_key, tmp_path):
    tc = _case(m, entry_key)
    d = _out_dir(tmp_path, 67.36)
    assert m._apply_known_failure(tc, "SKIP", 200.0, "no mesh", d) == "SKIP"


def test_other_cases_are_untouched(m, registered, tmp_path):
    """The waiver is keyed on all three of case, grid and resolution."""
    _require_live(m, registered)
    d = _out_dir(tmp_path, 67.36)
    for key in (("eady_uniform", "latlon_channel", "30x30"),
                ("phillips_two_layer", "mpas_channel", "70km"),
                ("eady_uniform", "mpas_channel", "35km")):
        tc = _case(m, key)
        assert m._apply_known_failure(
            tc, "FAIL", 200.0, "boom", d) == "FAIL", key


def test_registry_is_empty(m):
    """The registry holds nothing, and adding to it is a reviewed decision.

    Every waived case is a red result made green by fiat. If this goes red,
    someone added one -- check it carries a positive signature (a blow-up day
    band, not a notes substring) and a min_days above the onset, the two
    things review forced on the entry that used to live here."""
    assert m.KNOWN_FAILURES == {}


# ---------------------------------------------------------------------------
# A passing waived case must be LOUD, not silently green
# ---------------------------------------------------------------------------

def test_pass_becomes_xpass(m, registered, entry_key, tmp_path):
    """If the owed loop-closure fix lands, this entry must announce itself."""
    tc = _case(m, entry_key)
    d = _out_dir(tmp_path, 67.36)
    assert m._apply_known_failure(tc, "PASS", 200.0, "fine", d) == "XPASS"


def test_xpass_gates_the_exit_code(m):
    """A stale waiver hides the next regression, so XPASS must exit non-zero."""
    src = _MATRIX.read_text()
    assert "n_fail > 0 or n_error > 0 or n_xpass > 0" in src, (
        "the exit gate no longer fails on XPASS; a known-failure entry that "
        "has started passing would go unnoticed")


def test_xfail_does_not_gate_the_exit_code(m):
    """The whole point of the waiver: a waived failure must not exit non-zero."""
    src = _MATRIX.read_text()
    gate = src[src.index("if n_fail > 0 or n_error > 0"):]
    gate = gate[:gate.index("sys.exit(1)")]
    assert "n_xfail" not in gate


# ---------------------------------------------------------------------------
# The waiver has to be wired to something that runs
# ---------------------------------------------------------------------------

def test_waiver_is_applied_at_the_dispatcher(m):
    """A registry nothing calls is decoration.

    Anchored on the runner-dispatch line itself, not on a wrapper, so this
    fails if the call is dropped or moved off the executed path."""
    src = _MATRIX.read_text()
    anchor = "status, wall, notes = runner(tc, out_dir, days)"
    assert src.count(anchor) == 1
    tail = src[src.index(anchor):]
    tail = tail[:tail.index("record(tc, status, wall, notes)")]
    assert "_apply_known_failure(tc, status, days, notes," in tail
    assert "out_dir)" in tail, (
        "the waiver no longer receives out_dir, so it cannot read the "
        "blow-up day and its positive signature is inert")


def test_entry_is_a_real_matrix_case(m):
    """Every key must select a case the matrix actually enumerates.

    The registry is empty, so this checks the SHAPE the next entry has to
    satisfy: the key used by the entry that used to live here must still
    resolve to a real case. A typo in any of the three fields would produce a
    waiver that silently covers nothing.

    Written to be non-vacuous on an empty registry -- codex caught several
    sibling tests that quietly asserted nothing once the real entry went."""
    matrix = m._build_test_matrix()
    present = {(t.case, t.grid_type, t.resolution) for t in matrix}
    assert ("eady_uniform", "mpas_channel", "70km") in present, (
        "the case the removed waiver named is no longer in the matrix")
    for key in m.KNOWN_FAILURES:
        assert key in present, (
            f"KNOWN_FAILURES key {key} matches no case in the matrix")


# ---------------------------------------------------------------------------
# The blocker codex caught, and the class of bug it belongs to.
#
# The waiver remapped FAIL -> XFAIL at the dispatcher, but ``record`` kept its
# OWN status->icon map that knew only PASS/FAIL/ERROR/SKIP.  Recording an
# XFAIL raised KeyError, the caller's ``except`` turned it into ERROR, and the
# suite went MORE red than with no waiver at all.  Every test above passed,
# because none of them called ``record``.
# ---------------------------------------------------------------------------

def test_record_accepts_every_waived_status(m, registered, entry_key):
    """The regression test for the blocker: recording must not raise."""
    tc = _case(m, entry_key)
    before = len(m.ALL_RESULTS)
    try:
        for status in ("XFAIL", "XPASS"):
            m.record(tc, status, 1.0, "waived")
        assert [r["status"] for r in m.ALL_RESULTS[before:]] == ["XFAIL",
                                                                 "XPASS"]
    finally:
        del m.ALL_RESULTS[before:]


def test_every_status_the_waiver_can_emit_has_an_icon(m):
    """Enumerate from the waiver's own outputs rather than hardcoding."""
    for status in ("PASS", "FAIL", "ERROR", "SKIP", "XFAIL", "XPASS"):
        assert status in m.STATUS_ICONS, status


def test_there_is_exactly_one_status_icon_map(m):
    """Two maps is how the blocker happened; one map makes it unrepeatable.

    Anchored on the literal, so re-introducing a second inline map -- in
    ``record``, in the summary, or anywhere else -- fails here."""
    src = _MATRIX.read_text()
    assert src.count('"PASS": "  "') == 1, (
        "a second inline status->icon map has appeared; add the status to "
        "STATUS_ICONS instead, or the next new status will KeyError in "
        "whichever copy was forgotten")
    assert src.count("STATUS_ICONS[") >= 2, (
        "both record() and the summary must read the shared map")


def test_summary_json_carries_the_waived_counters(m):
    """A count-based consumer must be able to see a stale waiver (XPASS)."""
    src = _MATRIX.read_text()
    block = src[src.index('"n_pass": n_pass'):]
    block = block[:block.index("}")]
    assert '"n_xfail": n_xfail' in block and '"n_xpass": n_xpass' in block


def test_rerun_merge_carries_the_waived_counters():
    """The merged summary is a second consumer and must not drop them."""
    merge = _REPO / "scripts" / "data" / "merge_ocean_rerun.py"
    src = merge.read_text()
    assert '"n_xfail"' in src and '"n_xpass"' in src, (
        "merge_ocean_rerun drops XFAIL/XPASS, so a merged rerun would report "
        "a clean suite while a stale waiver went unaccounted for")


def test_malformed_blowup_marker_fails_closed(m, registered, entry_key, tmp_path):
    """A marker that matches the regex but is not a number must not ERROR.

    ``float()`` originally sat outside the try, so a corrupt results.txt
    raised instead of returning None (codex review)."""
    _require_live(m, entry_key)
    tc = _case(m, entry_key)
    d = tmp_path / "bad"
    d.mkdir()
    (d / "results.txt").write_text(
        "status: FAIL\nnotes: BLOWUP at step 19400 (day 1.2.3), reason: x\n")
    assert m._blowup_day_from_results(d) is None
    assert m._apply_known_failure(tc, "FAIL", 200.0, "boom", d) == "FAIL"
