"""The round-38 arms: the operand mapping, the dry slot, and the FMA probe.

Every test here is written so that it FAILS when the thing it checks is
removed.  Three of them are synthetic-violation checks in the sense the
campaign requires: they plant a defect and assert the arm goes red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(GATES))

import nemo_testcase_l2_gyre_round38_dry_slot_plant as D  # noqa: E402
import nemo_testcase_l2_gyre_round38_fma_flag as F  # noqa: E402
import nemo_testcase_l2_gyre_round38_matrix_operands as M  # noqa: E402


# --------------------------------------------------------------------------
# The mapping from NEMO's (i, j, k) to the model's (lat, lon, level)
# --------------------------------------------------------------------------
def _fake_record(jpi=9, jpj=8, jpk=6, halo=2):
    """A record shaped like the reader's output, with a UNIQUE value per cell.

    Unique values are the point: a transpose error, an axis swap or an off-by
    -one in the interior slice all survive a comparison of arrays whose
    entries repeat, and none of them survives this one.
    """
    ramp = np.arange(jpi * jpj * jpk, dtype=np.float64).reshape(
        (jpi, jpj, jpk))
    header = {"jpi": jpi, "jpj": jpj, "jpk": jpk, "jpkm1": jpk - 1,
              "ntsi": halo + 1, "ntei": jpi - halo,
              "ntsj": halo + 1, "ntej": jpj - halo}
    names = ("zwt_mix", "avt", "ah_wslp2", "e3t_Kaa", "e3w_Kmm", "tmask",
             "zwi", "zwd", "zws")
    arrays = {n: ramp + 1000.0 * i for i, n in enumerate(names)}
    arrays["rDt"] = 14400.0
    return {"header": header, "arrays": arrays}


def test_oracle_mapping_is_the_campaigns_own_transpose():
    """``_oracle`` must agree cell for cell with the phase-3 gate's ``_xyz``.

    ``_xyz`` reshapes the record's Fortran-order payload, crops the halo and
    transposes; ``_oracle`` slices the reader's already-shaped array.  They are
    two different routes to the same box, and this pins them together.
    """
    import nemo_testcase_l2_gyre_phase3_gate as P
    rec = _fake_record()
    h = rec["header"]
    got = M._oracle(rec)
    want = P._xyz(rec["arrays"]["e3t_Kaa"].ravel(order="F"),
                  h["jpi"], h["jpj"], h["jpk"])[..., :h["jpkm1"]]
    assert got["dz"].shape == want.shape
    np.testing.assert_array_equal(got["dz"], want)
    # the face arrays are the jk >= 2 slice, one shorter
    assert got["K"].shape == want.shape[:2] + (h["jpkm1"] - 1,)
    np.testing.assert_array_equal(
        got["K"], P._xyz(rec["arrays"]["zwt_mix"].ravel(order="F"),
                         h["jpi"], h["jpj"], h["jpk"])[..., 1:h["jpkm1"]])


def test_oracle_mapping_fails_on_a_transposed_record():
    """SYNTHETIC VIOLATION: swap i and j and the mapping must disagree."""
    import nemo_testcase_l2_gyre_phase3_gate as P
    rec = _fake_record(jpi=9, jpj=9)          # square, so the swap still fits
    h = rec["header"]
    want = P._xyz(rec["arrays"]["e3t_Kaa"].ravel(order="F"),
                  h["jpi"], h["jpj"], h["jpk"])[..., :h["jpkm1"]]
    rec["arrays"]["e3t_Kaa"] = np.ascontiguousarray(
        rec["arrays"]["e3t_Kaa"].transpose(1, 0, 2))
    assert not np.array_equal(M._oracle(rec)["dz"], want)


# --------------------------------------------------------------------------
# The row: wet cells score, dry cells are counted beside them
# --------------------------------------------------------------------------
def test_wet_row_scores_only_wet_cells_and_counts_the_dry_ones():
    oracle = np.ones((2, 3, 4))
    candidate = oracle.copy()
    wet = np.ones_like(oracle, dtype=bool)
    wet[..., -1] = False
    candidate[0, 0, -1] = 12345.0          # a DRY cell, must not score
    row = M._wet_row("t", oracle, candidate, wet)
    assert row["bit_unequal"] == 0 and row["status"] == "AT-BAR"
    assert row["dry_cells_unequal"] == 1
    assert row["n"] == int(wet.sum())


def test_wet_row_goes_red_on_one_wet_ulp():
    """SYNTHETIC VIOLATION: one ulp on a WET cell must leave AT-BAR."""
    oracle = np.full((2, 3, 4), 2.0)
    candidate = oracle.copy()
    candidate[1, 2, 0] = np.nextafter(2.0, np.inf)
    row = M._wet_row("t", oracle, candidate, np.ones_like(oracle, dtype=bool))
    assert row["bit_unequal"] == 1 and row["status"] != "AT-BAR"
    assert row["largest_difference_cell"] == [1, 2, 0]


# --------------------------------------------------------------------------
# The dry slot
# --------------------------------------------------------------------------
def _synthetic_column():
    wet = np.ones((1, 1, 6), dtype=bool)
    wet[0, 0, 2] = False
    return wet, np.full((1, 1, 6), 50.0)


def test_a_finite_dry_diagonal_never_reaches_a_wet_cell():
    wet, dz = _synthetic_column()
    rng = np.random.default_rng(7)
    rows = D._arms(wet, dz, 14400.0, rng)
    assert rows["dry_above_wet_cells"] == 1
    assert rows["finite"]["wet_cells_moved_or_nonfinite"] == 0
    assert rows["arm_informative"] is True


def test_nan_in_a_dry_diagonal_DOES_reach_a_wet_cell():
    """The identity is finite-only, and the gate must not overclaim it."""
    wet, dz = _synthetic_column()
    rows = D._arms(wet, dz, 14400.0, np.random.default_rng(7))
    assert rows["nan"]["wet_nonfinite"] > 0


def test_the_dry_slot_negative_control_fires():
    """SYNTHETIC VIOLATION: break the dry-interface coupling, wet must move."""
    wet, dz = _synthetic_column()
    rows = D._arms(wet, dz, 14400.0, np.random.default_rng(7))
    assert rows["negative_control"]["wet_cells_moved_or_nonfinite"] > 0


def test_a_land_column_card_is_reported_VACUOUS_not_passing():
    """Every dry cell a whole column: nothing wet is vertically adjacent."""
    wet = np.ones((1, 2, 4), dtype=bool)
    wet[0, 1, :] = False
    rows = D._arms(wet, np.full((1, 2, 4), 50.0), 14400.0,
                   np.random.default_rng(7))
    assert rows["dry_above_wet_cells"] == 0
    assert rows["wet_cells_vertically_adjacent_to_dry"] == 0
    assert rows["arm_informative"] is False


# --------------------------------------------------------------------------
# The FMA probe
# --------------------------------------------------------------------------
def test_the_fma_probes_two_references_actually_differ():
    """Without this the probe reports 'matches separate' for any flag."""
    import math
    a, b, c = F._triples()
    separate = c - a * b
    fused = np.array([math.fma(-x, y, z) for x, y, z in zip(a, b, c)])
    differ = int(np.count_nonzero(
        separate.view(np.uint64) != fused.view(np.uint64)))
    assert differ > F.N // 8, (
        f"only {differ} of {F.N} triples separate the two roundings; this "
        "probe would pass vacuously")


def test_the_fma_probe_carries_a_baseline_arm():
    """A probe with no unflagged arm cannot tell a cure from a no-op."""
    assert any(label == "baseline" and not flags
               for label, flags in F.CANDIDATES)


def test_every_plantable_operand_names_the_row_its_plant_must_move():
    """The first version of this asserted PLANT_OPERANDS against itself.

    It passed on any code at all.  The invariant that matters is that every
    plantable operand names a row the gate actually emits -- a plant pointing
    at a row that was renamed lands nowhere and reads as a pass.
    """
    import inspect
    assert M.PLANT_ROW == {
        "K": "operand.K", "K33": "operand.K33_fold",
        "dz": "operand.dz_after", "e3w": "operand.e3w_now",
        "wet": "operand.wet"}
    assert tuple(M.PLANT_ROW) == M.PLANT_OPERANDS
    emitted = inspect.getsource(M.run)
    for operand, row in M.PLANT_ROW.items():
        assert f'"{row}"' in emitted, (
            f"the plant for {operand!r} names row {row!r}, which this gate "
            "does not emit")


def test_the_plant_verdict_is_the_row_not_the_exit_status():
    """The gate's baseline is DEBT, so 'exited non-zero' proves nothing."""
    import inspect
    main = inspect.getsource(M.main)
    start = main.index("if args.plant:")
    # the branch ends where the NON-plant return begins; without that bound
    # the slice swallows it and this test reads the wrong statement
    plant_branch = main[start:main.index("return 0 if report", start)]
    assert 'report["plant_moved_its_own_row"]' in plant_branch
    assert 'report["status"] == "AT-BAR"' not in plant_branch


# --------------------------------------------------------------------------
# The rounds-33/34 bookkeeping, each pinned so it cannot silently return
# --------------------------------------------------------------------------
import nemo_testcase_l2_gyre_round21_admission as A  # noqa: E402
import nemo_testcase_l2_gyre_round35_trazdf_matrix as R35  # noqa: E402


def test_every_source_citation_names_the_card_it_was_read_from():
    """A bare ``MY_SRC/x.F90:123`` is not portable between cards.

    Red before round 38: every value was an unqualified path, and the comment
    above them claimed the writers sit at the same lines on every card.
    """
    unqualified = [magic for magic, where in A.SOURCES.items()
                   if "OMIP" not in where]
    assert unqualified == [], (
        f"these citations name no card: {unqualified}")


def test_the_zFw_waiver_does_not_publish_one_cards_line_numbers():
    reason = A.UNDEFINED_SLOTS[("NEMO_L1_TRANSP_1", "zFw")]["reason"]
    assert "Line numbers differ per card" in reason
    # the tank numbers must no longer stand alone as THE citation
    assert "on the L1 tank cards" in reason and "on the GYRE card" in reason


def test_both_self_describing_magics_have_a_writer_and_a_parser():
    for magic in A.SELF_DESCRIBING:
        assert magic in A.SOURCES, f"{magic} names no writer"
        assert magic in A.PARSERS, f"{magic} names no parser"


def test_a_pending_plant_survives_a_record_it_cannot_parse(tmp_path):
    """SYNTHETIC VIOLATION, from the other side.

    A byte-identical record whose SELF_DESCRIBING body this reader cannot
    decode used to raise BEFORE the plant landed, and run.sh would have read
    the crash as 'the plant turned the gate red'.  It must now be skipped
    loudly and the plant must go on to land somewhere.
    """
    baseline, candidate = tmp_path / "base", tmp_path / "cand"
    for directory in (baseline, candidate):
        directory.mkdir()
    magic = sorted(A.SELF_DESCRIBING)[0]
    # a header this reader accepts, followed by a body it cannot decode
    body = magic.ljust(16).encode("ascii") + b"\x00" * 4 * A.SELF_DESCRIBING[magic]
    body += b"\xff" * 64
    for directory in (baseline, candidate):
        (directory / "oracle_aaa_unreadable.bin").write_bytes(body)
    report = A.run(baseline, candidate, identical=(), plant_consumed=True)
    notes = [row for row in report["classified_changed_records"]
             if "plant_forced_open_but_unreadable" in row]
    assert notes, "the unreadable record was not reported"
    # A plant that lands NOWHERE is still a violation, and that is the point:
    # the run reports it instead of crashing.
    assert any("plant was never applied" in v for v in report["violations"])


def test_the_association_rows_are_inside_the_plant_signature():
    """Red before round 38: a plant that moves only an association row read
    as landing nowhere, exactly as the condition rows did before them."""
    report = {
        "calibration_rows": [], "given_inputs_rows": [], "clamp_rows": [],
        "condition_rows": [],
        "rhs_association_sensitivity": [
            {"name": "assoc.T", "status": "VALUE-AT-BAR",
             "absolute_max": 2.8e-14, "bit_unequal": 17}],
    }
    signature = R35._row_signature(report)
    assert signature["assoc.T"] == ("VALUE-AT-BAR", 2.8e-14, 17)
