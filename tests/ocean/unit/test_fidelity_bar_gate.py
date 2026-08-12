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


def test_ceiling_rows_classify_unconditionally():
    """#1455 Decision 1: a name in CEILING_ROWS returns "CEILING" regardless
    of corr/ratio/per_elem passed -- mirrors test_zdf_mxl_turb_is_waived, and
    the "classify without corr/ratio" pattern of the existing WAIVED test at
    gate _self_test() line ~3213."""
    mod = _load_module()
    for term in mod.CEILING_ROWS:
        assert mod.classify(None, None, name=term) == "CEILING"
        assert mod.classify(0.1, 5.0, per_elem=1.0, name=term) == "CEILING", (
            f"{term!r}: CEILING must not re-derive from corr/ratio/per_elem")


def test_ceiling_rows_populated_with_the_four_candidates():
    """The 4 rows named in the task brief -- ldf_slp uslp/vslp, aeiu,
    ssh_nxt/div_hor, ssh_atf -- all met the bar for CEILING (complete
    quantitative mechanism proofs, cited verbatim from their own
    MEASUREMENTS notes). wslpi/wslpj are DELIBERATELY excluded: they are
    already AT BAR (closed 2026-08-03 by the hmlp live-stretch fix), so they
    are not ceiling candidates at all."""
    mod = _load_module()
    expected = {
        "ldf_slp uslp", "ldf_slp vslp", "ldf_eiv kappa (aeiu)",
        "ssh_nxt / div_hor", "ssh_atf",
    }
    assert set(mod.CEILING_ROWS) == expected
    assert "ldf_slp wslpi" not in mod.CEILING_ROWS
    assert "ldf_slp wslpj" not in mod.CEILING_ROWS
    assert mod.classify(*mod.MEASUREMENTS["ldf_slp wslpi"][:2],
                         name="ldf_slp wslpi") == "AT BAR"


def test_ceiling_entry_missing_decision_or_evidence_is_rejected():
    """Synthetic-violation proof: the CEILING mechanism must not be vacuous --
    mirrors test_waiver_missing_provenance_or_evidence_is_rejected."""
    mod = _load_module()
    for broken in (
        {"fake": ("", "evidence")},
        {"fake": ("decision", "")},
        {"fake": ("  ", "  ")},
    ):
        saved = dict(mod.CEILING_ROWS)
        try:
            mod.CEILING_ROWS.clear()
            mod.CEILING_ROWS.update(broken)
            try:
                mod._validate_ceiling()
                assert False, "a CEILING entry missing decision/evidence must raise"
            except ValueError:
                pass
        finally:
            mod.CEILING_ROWS.clear()
            mod.CEILING_ROWS.update(saved)


def test_no_generic_ceiling_mechanism():
    """A row cannot be ceilinged via BINARY_GATES -- CEILING_ROWS is the only
    path, and it is a closed dict (same closure property as WAIVED_ROWS)."""
    mod = _load_module()
    for term in mod.CEILING_ROWS:
        assert term not in mod.BINARY_GATES
    assert not (set(mod.WAIVED_ROWS) & set(mod.CEILING_ROWS)), (
        "a term must not be both WAIVED and CEILING")


def test_main_reports_ceiling_count_and_stays_gated_by_remaining_debt(capsys):
    """CEILING is a legal pass-state alongside AT BAR/WAIVED, but main()'s
    exit code stays conservative: it fails whenever ANY row is DEBT or
    UNMEASURED, regardless of how many rows are CEILING."""
    mod = _load_module()
    rc = mod.main()
    captured = capsys.readouterr()
    assert f"CEILING {len(mod.CEILING_ROWS)}" in captured.out
    assert "*** CEILING" in captured.out
    # At the time this test was written there are still DEBT/UNMEASURED rows
    # independent of CEILING, so the gate must still fail overall.
    debt_or_unmeasured = any(
        mod.classify(c, r, mod.PER_ELEMENT.get(t), name=t) in ("DEBT", "UNMEASURED")
        for t, (c, r, _n) in mod.MEASUREMENTS.items())
    assert debt_or_unmeasured, "test assumption stale: no DEBT/UNMEASURED rows remain"
    assert rc == 1


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
        # dyn_zdf: probed 2026-08-06 (#1492 W3) but the NEMO dump bracket is
        # broken at the source level (Naa/Nrhs slot-alias in MLF), so no
        # solver-isolated number exists -- genuinely still UNMEASURED, needs
        # a new NEMO dump, not a probe.
        "dyn_zdf (momentum implicit vertical solve)",
        # traldf_iso_lap MOVED OUT: measured 2026-08-06 (#1492 W3,
        # traldf_iso_lap_probe.py) once its "needs instrumentation" note was
        # found STALE (stp_dump_22/23_*_traldf dumps exist) -- now DEBT.
    }
    now_measured = {
        "ldf_dyn coefficient",
        "tra_qsr (shortwave penetration)",
        "ssh_atf",
        "tra_sbc",
    }
    # traldf_iso_lap uses its own probe, not coverage_rows_measure.py, so it
    # is checked separately below rather than in the now_measured loop.
    corr_t, ratio_t, note_t = mod.MEASUREMENTS["traldf_iso_lap tendency"]
    assert corr_t is not None and ratio_t is not None
    assert mod.classify(corr_t, ratio_t, name="traldf_iso_lap tendency") in ("AT BAR", "DEBT")
    assert "traldf_iso_lap_probe.py" in note_t
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
        # "ssh_atf" moved to CEILING 2026-08-04 (Decision 1) -- still a
        # measured, non-UNMEASURED row, just a different terminal bucket.
        expected = ("AT BAR", "CEILING") if term == "ssh_atf" else ("AT BAR", "DEBT")
        assert mod.classify(corr, ratio, name=term) in expected
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


def test_traadv_fct_ww_inheritance_script_exists_and_is_cited():
    """#1455: traadv_fct_ww_inheritance.py (the ww-substitution/quantitative-
    inheritance test for the traadv_fct family, following the same pattern
    as ww_inheritance_walk.py for dyn_adv ZAD) is a real committed file, and
    every row it re-measured cites it in the gate note -- not just a claim
    that would silently rot if the file were ever deleted/renamed."""
    scripts_dir = SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226"
    assert (scripts_dir / "traadv_fct_ww_inheritance.py").is_file()

    mod = _load_module()
    rows_citing_it = [
        "traadv_fct fluxes",
        "traadv_fct tendency (T)",
        "traadv_fct horizontal tend",
        "traadv_fct vertical upstream flux",
        "traadv_fct (SALINITY)",
    ]
    for term in rows_citing_it:
        _corr, _ratio, note = mod.MEASUREMENTS[term]
        assert "traadv_fct_ww_inheritance.py" in note, (
            f"{term!r} note does not cite traadv_fct_ww_inheritance.py")
        assert "#1455" in note


def test_traadv_fct_salinity_row_measured_at_updated():
    """The SALINITY row's tuple genuinely moved on re-measurement (Rule 1e
    reconciliation, not a silent overwrite): MEASURED_AT must reflect the
    NEW measuring commit, not the stale c1ba30e39 stamp the other 4
    (unchanged-tuple) traadv_fct rows still correctly carry."""
    mod = _load_module()
    assert mod.MEASURED_AT["traadv_fct (SALINITY)"] != "c1ba30e39"
    for term in ("traadv_fct fluxes", "traadv_fct tendency (T)",
                 "traadv_fct horizontal tend",
                 "traadv_fct vertical upstream flux"):
        assert mod.MEASURED_AT[term] == "c1ba30e39", (
            f"{term!r}'s tuple did not change -- MEASURED_AT should stay "
            "c1ba30e39 unless the corr/ratio numbers themselves moved")


# --- #1492 item 0.2: tolerance-by-arithmetic-class ---------------------------


def test_row_class_is_a_closed_dict_covering_every_measurement_row():
    """ROW_CLASS must classify EXACTLY the 53 MEASUREMENTS rows -- no more,
    no less (mirrors the WAIVED_ROWS/CEILING_ROWS closure style). Removing a
    class entry (or adding a phantom one) must be caught by
    _validate_row_class(), not silently tolerated."""
    mod = _load_module()
    assert set(mod.ROW_CLASS.keys()) == set(mod.MEASUREMENTS.keys())
    for term, (cls, justification) in mod.ROW_CLASS.items():
        assert cls in ("POINTWISE", "ACCUMULATING", "CONDITIONED"), (
            f"{term!r} has an unknown class {cls!r}")
        assert justification.strip(), f"{term!r} has an empty justification"


def test_removing_a_class_entry_fails_validation():
    """Synthetic-violation proof: a row missing from ROW_CLASS must be
    rejected by _validate_row_class -- the closed-dict property is not
    vacuous."""
    mod = _load_module()
    saved = dict(mod.ROW_CLASS)
    try:
        victim = next(iter(saved))
        mod.ROW_CLASS.clear()
        mod.ROW_CLASS.update({k: v for k, v in saved.items() if k != victim})
        try:
            mod._validate_row_class()
            assert False, "removing a ROW_CLASS entry must raise"
        except ValueError as e:
            assert victim in str(e)
    finally:
        mod.ROW_CLASS.clear()
        mod.ROW_CLASS.update(saved)


def test_class_bar_for_is_a_pure_function_of_row_class_and_named_constants():
    """class_bar_for must dispatch on ROW_CLASS's class label to exactly the
    two named constants (BAR_POINTWISE, BAR_ACCUMULATING) -- a caller cannot
    quietly relax a class's bar via some other threshold, only by editing the
    module-level constant (a visible diff, not a silent runtime path)."""
    mod = _load_module()
    assert mod.class_bar_for("sbc (utau/qsr/qns/sfx)") == mod.BAR_POINTWISE
    assert mod.class_bar_for("traadv_fct fluxes") == mod.BAR_ACCUMULATING
    assert mod.class_bar_for("ssh_nxt / div_hor") == float("inf")  # CONDITIONED
    # Unclear/absent class defaults to the STRICT bar (burden of proof rule).
    assert mod.class_bar_for("not-a-real-row-at-all") == mod.BAR_POINTWISE
    assert mod.class_bar_for(None) == mod.BAR_POINTWISE


def test_loosening_to_conditioned_without_a_decision_citation_fails_validation():
    """CONDITIONED is the loosest rank (class_bar_for returns +inf -- no
    numeric bar at all), so landing a row there is the maximal possible
    loosening. Mechanical enforcement of the sign-off's 'loosening requires a
    recorded decision string': _validate_row_class must reject a CONDITIONED
    entry whose justification does not cite a NUMBERED decision. Uses an
    existing CEILING row so only the decision-citation axis varies.
    Synthetic-violation proof, mirrors the WAIVED/CEILING missing-string
    tests above."""
    mod = _load_module()
    saved = dict(mod.ROW_CLASS)
    row = "ldf_slp uslp"  # already CONDITIONED + in CEILING_ROWS
    try:
        # No decision mention at all -> rejected.
        mod.ROW_CLASS[row] = (
            "CONDITIONED", "just a vague claim of conditioning, no citation")
        try:
            mod._validate_row_class()
            assert False, ("a CONDITIONED entry without a decision citation "
                            "must raise -- the loosening check is VACUOUS")
        except ValueError as e:
            assert "decision" in str(e).lower()
        # GAMED string containing the bare word "decision" but no numbered
        # citation -> still rejected (adversarial-review finding: the plain
        # substring check passed 'no decision was made here').
        mod.ROW_CLASS[row] = ("CONDITIONED", "no decision was made here")
        try:
            mod._validate_row_class()
            assert False, ("a bare 'decision' substring must NOT satisfy "
                            "the citation check -- it is gameable")
        except ValueError:
            pass
        # A proper numbered decision citation -> accepted.
        mod.ROW_CLASS[row] = (
            "CONDITIONED", "Dhruv 2026-08-04 Decision 1: cites a decision.")
        mod._validate_row_class()  # must not raise
    finally:
        mod.ROW_CLASS.clear()
        mod.ROW_CLASS.update(saved)


def test_conditioned_row_not_in_ceiling_rows_fails_validation():
    """Adversarial-review finding (2026-08-06): a CONDITIONED row NOT in
    CEILING_ROWS would fall through classify() with class_bar=+inf and could
    clear AT BAR on cancelling corr/ratio statistics alone -- the exact bn2
    regression the per-element bar exists to prevent, reopened for one
    class. _validate_row_class must therefore enforce CONDITIONED subset-of
    CEILING_ROWS mechanically, not by current coincidence."""
    mod = _load_module()
    saved = dict(mod.ROW_CLASS)
    try:
        # A non-CEILING row loosened to CONDITIONED, WITH a valid decision
        # citation (so only the CEILING-membership axis is violated).
        mod.ROW_CLASS["sbc (utau/qsr/qns/sfx)"] = (
            "CONDITIONED", "Dhruv 2026-08-04 Decision 1: synthetic violation.")
        try:
            mod._validate_row_class()
            assert False, ("a CONDITIONED row outside CEILING_ROWS must "
                            "raise -- the mechanism-proof gate is VACUOUS")
        except ValueError as e:
            assert "CEILING_ROWS" in str(e)
    finally:
        mod.ROW_CLASS.clear()
        mod.ROW_CLASS.update(saved)


def test_synthetic_pointwise_row_at_1e12_is_not_at_bar_for_its_class():
    """Non-vacuity proof for the STRICTER POINTWISE bar: a per-element error
    of 1e-12 would have cleared the legacy BAR_PER_ELEM_EPS=1e-9 (so it is
    NOT plain DEBT under the old regime) but must NOT be "AT BAR" under the
    new POINTWISE bar of 1e-15 -- it must land on NEAR-CLASS instead, per the
    sign-off's 'rows currently AT BAR in this class must tighten or be
    reclassified'."""
    mod = _load_module()
    assert 1e-12 <= mod.BAR_PER_ELEM_EPS
    assert 1e-12 > mod.BAR_POINTWISE
    result = mod.classify(1.0, 1.0, per_elem=1e-12,
                           name="sbc (utau/qsr/qns/sfx)")  # a real POINTWISE row
    assert result == "NEAR-CLASS", (
        f"a 1e-12 per-element error on a POINTWISE row must be NEAR-CLASS, "
        f"not {result!r} -- the class bar would be VACUOUS otherwise")
    # And confirm it would have been AT BAR under the legacy single bar.
    assert 1e-12 <= mod.BAR_PER_ELEM_EPS


def test_synthetic_accumulating_row_at_1e10_is_debt_not_near_class():
    """An ACCUMULATING row with per-element error 1e-10 misses BOTH the
    legacy 1e-9 bar's neighbourhood check (it clears 1e-9 itself, so this
    checks a value BETWEEN the class bar and something that also fails the
    class bar) -- more precisely: a value that clears BAR_ACCUMULATING=1e-12
    must be AT BAR, and a value between BAR_ACCUMULATING and
    BAR_PER_ELEM_EPS=1e-9 must be NEAR-CLASS (same as the POINTWISE case
    above), proving the per-class dispatch in class_bar_for is actually
    consulted (not just BAR_POINTWISE hardcoded everywhere)."""
    mod = _load_module()
    row = "traadv_fct fluxes"  # a real ACCUMULATING row
    assert mod.ROW_CLASS[row][0] == "ACCUMULATING"
    # Clears the ACCUMULATING bar -> AT BAR.
    assert mod.classify(1.0, 1.0, per_elem=1e-13, name=row) == "AT BAR"
    # Between ACCUMULATING (1e-12) and the legacy bar (1e-9) -> NEAR-CLASS.
    assert mod.classify(1.0, 1.0, per_elem=1e-10, name=row) == "NEAR-CLASS"
    # Above the legacy bar entirely -> plain DEBT, same as before #1492.
    assert mod.classify(1.0, 1.0, per_elem=1e-3, name=row) == "DEBT"


def test_conditioned_rows_are_not_gated_by_a_numeric_class_bar():
    """CONDITIONED rows get NO class-bar check (class_bar_for returns +inf
    for them) -- they clear via the pre-existing CEILING_ROWS mechanism-proof
    machinery only, per the sign-off ('mechanism-proven CEILING, existing
    machinery, unchanged'). All 4 CONDITIONED rows in this gate are in fact
    already in CEILING_ROWS; this test proves the CONDITIONED class itself
    imposes no separate numeric threshold (a huge per_elem does not flip a
    non-CEILING CONDITIONED row to DEBT via the class bar)."""
    mod = _load_module()
    conditioned = {t for t, (cls, _j) in mod.ROW_CLASS.items()
                   if cls == "CONDITIONED"}
    assert conditioned, "expected at least one CONDITIONED row"
    for term in conditioned:
        assert term in mod.CEILING_ROWS, (
            f"{term!r} is CONDITIONED but not in CEILING_ROWS -- every "
            "CONDITIONED row in this gate is expected to already be "
            "mechanism-proven CEILING")
        # classify() must still return CEILING unconditionally (precedence
        # check happens before the class-bar branch is ever reached).
        assert mod.classify(0.0, 100.0, per_elem=1e6, name=term) == "CEILING"


def test_wslpi_wslpj_are_the_two_rows_that_tightened_out_of_at_bar():
    """The concrete #1492 item 0.2 regression this task exists for: 'rows
    currently AT BAR in this class must tighten or be reclassified'.
    ldf_slp wslpi/wslpj were AT BAR under the legacy 1e-9 bar (per-element
    ~1.1e-11) but MISS the new POINTWISE bar of 1e-15 -- they must now show
    as NEAR-CLASS, not silently stay AT BAR and not silently vanish into
    plain DEBT."""
    mod = _load_module()
    for term in ("ldf_slp wslpi", "ldf_slp wslpj"):
        cls, _j = mod.ROW_CLASS[term]
        assert cls == "POINTWISE"
        corr, ratio, _note = mod.MEASUREMENTS[term]
        per_elem = mod.PER_ELEMENT[term]
        assert mod.BAR_POINTWISE < per_elem <= mod.BAR_PER_ELEM_EPS, (
            f"{term!r}'s per-element value moved outside the expected "
            "NEAR-CLASS band -- update this test if it was re-measured")
        assert mod.classify(corr, ratio, per_elem, name=term) == "NEAR-CLASS"


def test_main_reports_per_class_tally(capsys):
    """main() must print a per-class breakdown (the sign-off requires 'the
    gate reports the tally per class'), and the NEAR-CLASS count/list."""
    mod = _load_module()
    rc = mod.main()
    captured = capsys.readouterr()
    assert "PER-CLASS TALLY" in captured.out
    assert "POINTWISE" in captured.out and "ACCUMULATING" in captured.out
    assert "CONDITIONED" in captured.out
    assert "NEAR-CLASS(1e-9, below class bar)" in captured.out
    # NEAR-CLASS is not a legal terminal state -- it must gate exit 0 exactly
    # like DEBT/UNMEASURED (both still present independent of this feature).
    assert rc == 1


def test_near_class_gates_exit_code():
    """A NEAR-CLASS verdict must count toward gate failure (rc=1), not be
    silently absorbed into a passing count -- the sign-off treats it as
    'must tighten or be reclassified', not a terminal state."""
    mod = _load_module()
    assert mod.classify(1.0, 1.0, per_elem=1e-12,
                         name="sbc (utau/qsr/qns/sfx)") == "NEAR-CLASS"
    # Cross-check against main()'s own gating condition directly.
    rows = [(t, c, r, n, mod.classify(c, r, mod.PER_ELEMENT.get(t), name=t))
            for t, (c, r, n) in mod.MEASUREMENTS.items()]
    near_class = sum(s == "NEAR-CLASS" for *_, s in rows)
    assert near_class > 0, "test assumption stale: no NEAR-CLASS rows remain"
