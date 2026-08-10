"""``docs/ocean/fidelity/dino_step_wiring_generated.md`` must match a fresh
regeneration.

WHY: the hand-maintained ``dino_wiring_diagram.md`` went stale silently --
nothing failed when the Fortran, the gate, or legoESM moved underneath it, so
it kept asserting things that had stopped being true (no leapfrog/Asselin, a
unit Prandtl number, several rows "UNTRACED" that had since been measured).
This module is the tripwire that document never had.

SKIP POLICY: regenerating the doc needs the NEMO oracle checkout, which lives
outside the repo, so those tests skip when it is absent.  Everything that does
NOT need the oracle -- the normalise-scope guard, the bucket-split regression,
the gate-vocabulary check, and the whole ``_resolve_live`` truth table -- runs
unconditionally, so a machine without the oracle still gets real coverage of
the logic that decides what the document says.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GEN_DIR = os.path.join(REPO, "scripts/validate/ocean_fidelity/dino_1226")
GEN_PY = os.path.join(GEN_DIR, "gen_step_wiring.py")


def _load():
    spec = importlib.util.spec_from_file_location("gen_step_wiring", GEN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    """The generator module.  Importing it touches no oracle file."""
    if not os.path.exists(GEN_PY):
        pytest.skip(f"generator missing: {GEN_PY}")
    return _load()


@pytest.fixture(scope="module")
def oracle(gen):
    """Skip marker for the tests that actually read the NEMO checkout."""
    for path in (gen.STPMLF, gen.CPP_FCM, gen.OCEAN_OUTPUT):
        if not os.path.exists(path):
            pytest.skip(f"NEMO oracle checkout not present: {path}")
    return gen


# ---------------------------------------------------------------------------
# Oracle-independent: the logic that decides what the document says.
# ---------------------------------------------------------------------------
def test_resolve_live_truth_table(gen):
    """``_resolve_live`` is where a wrong LIVE/DEAD verdict would be born, so
    it gets a truth table rather than only end-to-end coverage.

    Each row is a real shape from ``stpmlf.F90``.  The two marked REGRESSION
    are bugs this table would have caught: a guard judged on its ``ln_*``
    names alone (dropping the rest of the conjunction -> FALSE LIVE), and
    ``.OR.`` resolved as ``.AND.`` (-> FALSE DEAD).
    """
    els = gen._ELSE_MARK
    cases = [
        # (cpp_dead, guards, switches, cpp_keys, expected verdict)
        ([], (), {}, set(), "LIVE"),
        ([], ("ln_traqsr",), {"ln_traqsr": True}, set(), "LIVE"),
        ([], ("ln_tradmp",), {"ln_tradmp": False}, set(), "DEAD"),
        # cpp-dead beats everything else
        (["defined key_top"], ("ln_traqsr",), {"ln_traqsr": True}, set(), "DEAD"),
        # ELSE branch runs on the negation (ldf_slp's shape)
        ([], (els + "ln_traldf_triad",), {"ln_traldf_triad": False}, set(), "LIVE"),
        ([], (els + "ln_traldf_triad",), {"ln_traldf_triad": True}, set(), "DEAD"),
        # explicit .NOT.
        ([], (".NOT.ln_tradmp",), {"ln_tradmp": False}, set(), "LIVE"),
        # REGRESSION (false LIVE): the non-ln_ conjunct must NOT be dropped.
        # stpmlf.F90:449 `IF( ln_zdfosm .AND. lrst_oce )`
        ([], ("ln_zdfosm .AND. lrst_oce",), {"ln_zdfosm": True}, set(),
         "UNRESOLVED"),
        # ...but a FALSE conjunct still proves DEAD, since the guard is an AND.
        ([], ("ln_zdfosm .AND. lrst_oce",), {"ln_zdfosm": False}, set(), "DEAD"),
        # REGRESSION (false DEAD): .OR. is not .AND.
        ([], ("ln_a .OR. ln_b",), {"ln_a": False, "ln_b": True}, set(),
         "UNRESOLVED"),
        # internal logicals ocean.output never prints
        ([], ("l_ldfslp",), {}, set(), "UNRESOLVED"),
        # cpp-derived logical: lk_linssh is FALSE when key_linssh is absent,
        # so `.NOT.lk_linssh` RUNS for DINO.
        ([], (".NOT.lk_linssh",), {}, set(), "LIVE"),
        ([], (".NOT.lk_linssh",), {}, {"key_linssh"}, "DEAD"),
        # arithmetic conditions are not resolvable
        ([], ("kstp == nit000",), {}, set(), "UNRESOLVED"),
        # De Morgan: the ELSE of `IF(a .AND. b)` runs on NOT(a) OR NOT(b).
        # Negating each conjunct inside the AND fold computed `NOT a AND NOT b`
        # and returned DEAD for a=T,b=F -- a branch that RUNS -- with a reason
        # string identical to the genuinely-DEAD case, so the two were
        # indistinguishable.  Both must now be UNRESOLVED.
        ([], (els + "ln_a .AND. ln_b",), {"ln_a": True, "ln_b": False}, set(),
         "UNRESOLVED"),
        ([], (els + "ln_a .AND. ln_b",), {"ln_a": True, "ln_b": True}, set(),
         "UNRESOLVED"),
    ]
    for cpp_dead, guards, switches, keys, want in cases:
        got, why = gen._resolve_live(cpp_dead, guards, switches, keys)
        assert got == want, f"{guards} with {switches}/{keys}: {got} ({why}), want {want}"


def test_bucket_split_does_not_swallow_digits(gen):
    """Regression test: the coverage bucket key was once split on
    ``[\\s(->]``, which is the character RANGE ``(``..``>`` and therefore
    matches digits -- silently filing ``bn2`` under ``bn`` and leaving the
    Brunt-Vaisala row UNMAPPED."""
    cov = gen.load_coverage()
    assert "bn2" in cov, sorted(cov)[:20]
    assert "bn" not in cov


def test_spec_table_is_read_from_section_1_only(gen):
    """``load_spec`` must scan only ``## 1. The step skeleton``.  The spec
    file holds other numbered tables (phases, P3 results) whose rows would
    otherwise be injected into the join."""
    spec = gen.load_spec()
    heads = set(spec)
    assert "zdf_phy" in heads and "tra_zdf" in heads
    assert "" not in heads, "inline (non-CALL) spec rows must not make a bucket"
    # A routine cell naming two routines indexes BOTH (spec row 8 is
    # "`eos`(Nbb, in-situ) + `ldf_slp`") -- indexing only the first left the
    # isoneutral-slope seam attached to the density call.
    assert "ldf_slp" in heads and "eos" in heads
    # The phase table's "P1"/"P2" rows must not have leaked in.
    assert not any(h.startswith("P") and h[1:].isdigit() for h in heads)
    # REGRESSION: an identifier inside PARENTHESES is a reference, not a
    # second routine.  Spec row 11 is "`ssh_nxt` (incl. `div_hor`)"; indexing
    # the parenthetical gave the div_hor call at line 302 ssh_nxt's seam
    # ("Nbb,Nnn -> Naa ssh") instead of row 20's ("Naa" / state-commit).
    assert [r["spec_row"] for r in spec["div_hor"]] == ["20"], spec["div_hor"]
    assert [r["spec_row"] for r in spec["ssh_nxt"]] == ["11"]
    # ...while the "+" shape of row 8 must still index BOTH routines.
    assert [r["spec_row"] for r in spec["ldf_slp"]] == ["8"]


def test_dead_branch_section_contains_no_live_call(gen):
    """REGRESSION: §3 is the DEAD-branch table, so a row whose verdict column
    reads ``LIVE`` must never appear in it.

    The document previously filed 14 executing calls (``eos``@246, the #1226
    debug dumps, ``dia_ar5``, ``dia_wri``, ``stp_ctl``) under a heading that
    said "Dead branches" -- a false statement about the oracle, in the one
    document whose entire purpose is not making false statements about the
    oracle.  They now live in §2, which says "Live, but waived".
    """
    doc = open(gen.OUT_MD).read()
    dead = doc.split("## 3. Not live")[1].split("## 4.")[0]
    live_rows = [ln for ln in dead.split("\n") if "| LIVE |" in ln]
    assert not live_rows, (
        f"{len(live_rows)} LIVE calls under the dead-branch heading: "
        f"{live_rows[:3]}")
    # ...and the waived-live section must not have simply swallowed them all
    # silently: it exists precisely so they stay visible.
    waived = doc.split("## 2. Live, but waived")[1].split("## 3.")[0]
    assert waived.count("| LIVE |") > 0, "the waived-live table is empty"
    # Every LINE NUMBER in the Fortran appears in exactly one of the three
    # tables -- nothing is dropped by the split, and nothing is double-filed.
    # (Row counts differ from call counts because a dispatch wrapper such as
    # `zdf_phy` emits one row per concrete routine.)
    chain = doc.split("## 1. Live physics chain")[1].split("## 2.")[0]
    def _lines(section, col):
        out = []
        for ln in section.split("\n"):
            if ln.startswith("| ") and ln.count("|") > 4:
                cell = ln.split("|")[col].strip()
                if cell.isdigit():
                    out.append(int(cell))
        return out
    seen = _lines(chain, 2) + _lines(waived, 1) + _lines(dead, 1)
    assert len(set(seen)) > 100, len(set(seen))
    overlap = (set(_lines(waived, 1)) | set(_lines(dead, 1))) & set(_lines(chain, 2))
    assert not overlap, f"lines filed in two sections at once: {sorted(overlap)}"


def test_normalise_only_blanks_the_two_volatile_fields(gen):
    """Guard the escape hatch: ``normalise`` must blank the timestamp and the
    commit and NOTHING else, or the byte comparison could be weakened into
    vacuity by widening that regex."""
    text = open(gen.OUT_MD).read()
    normalised = gen.normalise(text)
    changed = [(a, b) for a, b in zip(text.split("\n"), normalised.split("\n"))
               if a != b]
    assert len(changed) == 2, f"normalise() touched {len(changed)} lines, want 2"
    assert {b for _a, b in changed} == {
        "- generated-at: <normalised>", "- legoesm-commit: <normalised>"}
    # The oracle source hash must survive normalisation -- it is a real input.
    assert re.search(r"^- oracle-source-sha256: [0-9a-f]{64}$", normalised,
                     re.M), "the oracle sha256 must be compared, not blanked"


def test_every_status_cell_uses_the_gates_own_vocabulary(gen):
    """No hand-written status may appear.  Every verdict in the doc's status
    column must be one ``fidelity_bar_gate.classify()`` can emit, or an
    explicit gap marker."""
    sys.path.insert(0, GEN_DIR)
    gate = importlib.import_module("fidelity_bar_gate")
    allowed = {"AT BAR", "DEBT", "UNMEASURED", "WAIVED", "CEILING",
               "NEAR-CLASS", gen.UNMAPPED, gen.UNRESOLVED}
    emitted = {gate.classify(c, r, gen._per_elem(gate, n), n)
               for n, (c, r, _note) in gate.MEASUREMENTS.items()}
    assert emitted <= allowed, f"unknown classify() verdict: {emitted - allowed}"
    # Parse the §3 table and check each verdict cell, rather than substring-
    # searching the whole document (which can never fail).
    doc = open(gen.OUT_MD).read()
    section = doc.split("## 4. Measured status")[1].split("## 5.")[0]
    verdicts = {row.split("|")[3].strip()
                for row in section.split("\n")
                if row.startswith("|") and row.count("|") >= 5}
    verdicts -= {"classify()", "---"}
    assert verdicts, "no verdict cells parsed from section 3"
    assert verdicts <= allowed, f"hand-written status in doc: {verdicts - allowed}"


# ---------------------------------------------------------------------------
# Oracle-dependent.
# ---------------------------------------------------------------------------
def test_committed_doc_matches_regeneration(oracle):
    """The committed doc is byte-identical to a fresh run, except for the two
    deliberately volatile header fields, which ``normalise`` blanks on BOTH
    sides.  Everything else -- including the oracle sha256 -- is exact."""
    gen = oracle
    assert os.path.exists(gen.OUT_MD), (
        f"{gen.OUT_MD} is missing -- run gen_step_wiring.py")
    committed = gen.normalise(open(gen.OUT_MD).read())
    fresh = gen.normalise(gen.generate())
    assert committed == fresh, (
        "dino_step_wiring_generated.md is STALE. Re-run "
        "scripts/validate/ocean_fidelity/dino_1226/gen_step_wiring.py "
        "and commit the result.")


def test_generator_self_test_passes(oracle):
    """The generator ships its own synthetic-violation check."""
    assert oracle._self_test() == 0


def test_mutating_one_parsed_row_changes_the_document(oracle):
    """SYNTHETIC VIOLATION for the byte comparison above.

    A byte-identical assertion proves nothing if the generated text does not
    actually depend on the parse.  Mutate one parsed CALL and the committed
    doc must stop matching -- i.e. this test would CATCH the drift it exists
    to catch.
    """
    gen = oracle
    calls = gen.parse_stpmlf(gen.STPMLF)
    victim = next(c for c in calls if c.name == "dyn_vor")
    victim.name = "dyn_vor_MUTATED"
    mutated = gen.normalise(gen.render(*gen.build_rows(calls)))
    committed = gen.normalise(open(gen.OUT_MD).read())
    assert mutated != committed, (
        "mutating a parsed CALL left the document unchanged -- "
        "test_committed_doc_matches_regeneration is VACUOUS")
    assert "dyn_vor_MUTATED" in mutated


def test_reported_line_is_the_line_of_the_call_token(oracle):
    """REGRESSION: a CALL and its guard are one logical line but often two
    physical ones::

        IF( ln_dynspg_ts ) &                    ! 314
           &            CALL wzv( ... )         ! 315

    Reporting 314 made the generator tell the maintainers that
    ``stpmlf_call_coverage.py`` (which correctly records 315) was stale, at 8
    separate sites.  A tool that calls a correct human record wrong is worse
    than no tool.
    """
    gen = oracle
    lines = open(gen.STPMLF).read().split("\n")
    calls = gen.parse_stpmlf(gen.STPMLF)
    for c in calls:
        src = lines[c.line - 1]
        assert re.search(rf"\bCALL\s+{re.escape(c.name)}\b", src, re.I), (
            f"{c.name} reported at line {c.line}, but that line is {src!r}")
    # The specific continuation sites that were off by one.
    assert any(c.name == "wzv" and c.line == 315 for c in calls)
    assert any(c.name == "tra_asm_inc" and c.line == 386 for c in calls)


def test_parse_is_non_trivial(oracle):
    """A parser labelling everything LIVE would still produce a stable doc and
    pass the byte comparison, so assert the verdicts and the time-level
    extraction are genuinely exercised, with known-answer spot checks read
    from the oracle source."""
    gen = oracle
    calls = gen.parse_stpmlf(gen.STPMLF)
    assert {"LIVE", "DEAD"} <= {c.live for c in calls}
    sbc = next(c for c in calls if c.name == "sbc")
    assert sbc.live == "LIVE" and sbc.levels == ("Nbb", "Nnn"), sbc
    dmp = next(c for c in calls if c.name == "tra_dmp")
    assert dmp.live == "DEAD" and "ln_tradmp" in dmp.live_reason, dmp
    trc = next(c for c in calls if c.name == "trc_stp")
    assert trc.live == "DEAD" and "key_top" in trc.live_reason, trc
    # z-star thickness ratios are gated by `.NOT.lk_linssh`, a COMPILE-TIME
    # logical: key_linssh is absent from cpp_DINO.fcm, so they RUN.  Reporting
    # them UNRESOLVED filed the core of #1226 under "not live".
    qco = [c for c in calls if c.name == "dom_qco_r3c"]
    assert sum(c.live == "LIVE" for c in qco) == 3, [(c.line, c.live) for c in qco]


def test_else_branch_polarity_is_honoured(oracle):
    """``ldf_slp`` lives in the ELSE of ``IF(ln_traldf_triad)`` and DINO sets
    ln_traldf_triad=F, so the ELSE branch is the one that runs.

    Regression test for a real bug in this generator's first draft: ignoring
    the ELSE flipped the sense and reported live physics (the isoneutral-slope
    call feeding tra_ldf and GM) as DEAD.
    """
    gen = oracle
    calls = gen.parse_stpmlf(gen.STPMLF)
    triad = next(c for c in calls if c.name == "ldf_slp_triad")
    slp = next(c for c in calls if c.name == "ldf_slp")
    assert triad.live == "DEAD", "the IF branch must be dead (ln_traldf_triad=F)"
    # The ELSE branch must NOT inherit that DEAD verdict.  It resolves to
    # UNRESOLVED only because of the OUTER `l_ldfslp`, an internal logical
    # ocean.output does not print -- which is the honest answer.
    assert slp.live != "DEAD", (
        f"ldf_slp is in the ELSE of ln_traldf_triad=F so it RUNS; got "
        f"{slp.live} / {slp.live_reason}")
    assert "l_ldfslp" in slp.live_reason
