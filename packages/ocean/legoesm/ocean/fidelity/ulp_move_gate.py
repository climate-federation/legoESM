"""How far a certified oracle-fidelity number is allowed to move, in ulps.

A refactor that only re-associates floating-point arithmetic -- deleting a
duplicated code path, reordering two summations, hoisting a common factor --
does not leave the certified numbers bit-identical.  It moves them by roundoff.
"Roundoff" is then a judgement call, and judgement drifts: 8e-11 relative looks
like nothing next to a 2.6e-7 residual, right up until the day it is 8e-9.

So the limit lives here, as a constant, and the comparison is mechanical:

    a certified row may move by at most ``MAX_ULP_MOVE`` float64 ulps of its
    own normalising scale, tracer (T/S) rows may not move at all, and the
    first-over-bar step and field set must not change.

Anything larger is not a re-association and must be reported as a behaviour
change, not waved through.

**What a row's "ulp" means here.**  Gate rows are normalised residuals:
``normalized_max_abs = max|candidate - oracle| / max(max|oracle|, 1)``.  If
every value in the state moved by at most N ulps, that quotient moves by at
most ``N * ulp(scale) / scale <= N * 2**-52``.  So one "row ulp" is
``2**-52`` for a normalised row, and ``2**-52 * scale`` for a row reported in
physical units (``absolute_max``, whose scale is its own
``reference_max_abs``).  This is a bound on the underlying field's movement,
which is the thing the bar is actually about -- not on the residual's own
mantissa, which is a near-cancellation and carries no such meaning.

**What this gate cannot see** (Rule 2 -- write the blind spot down):

* it reads REDUCTIONS, not fields.  Two states differing at a single cell by
  exactly the amount already attained elsewhere give identical rows.  The
  T/S prong is therefore "the tracer RESIDUAL did not move", which is
  necessary for bit-identity and NOT sufficient for it.  This is not
  hypothetical: over ten OVERFLOW-zps steps the T rows here were bit-identical
  while the T FIELD departed from kt=7 onward at five cells.  Run
  ``scripts/validate/ocean_fidelity/testcases/nemo_testcase_state_ulp_probe.py``
  before writing "the field is bit-identical"; this gate cannot support that
  sentence.
* the row scale is floored at 1 (``max(max|oracle|, 1)``), so for a field whose
  maximum is below 1 the bar admits MORE than ``MAX_ULP_MOVE`` ulps of the
  field, by the reciprocal of that maximum.  Measured on the stage sweeps: the
  largest admitted move was 1.5 ulps of its own field scale, and one row sat at
  3.0 -- i.e. this bar is loose in the admitting direction by up to ~4 decades
  on a small-magnitude field.  Tightening it requires the gates to emit each
  row's ``reference_max_abs`` (the stage sweep does, the trajectory gate does
  not) and a re-pin of the references.
* it compares only rows that both reports contain, under their own names, and
  fails closed on any name appearing in one and not the other.
* outside the rows it compares only ``status``, ``first_over_bar``,
  ``selectors`` and ``precision_policy``, and only when a key is present in
  both reports -- ``first_over_bar`` is absent from the stage-sweep schema, so
  that prong is INERT there and the result says so.
* it says nothing about rows produced by a PRIVATE ABLATION ARM whose meaning
  the change under test deliberately redefines.  Filter those out explicitly
  (``row_filter``) and say so in the receipt; do not let them fail silently.

Harness glue, not model code, same home and same reason as ``precision_gate``.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "ULP",
    "MAX_ULP_MOVE",
    "MOVABLE_ROW_KEYS",
    "certified_rows",
    "row_move_tolerance",
    "compare_gate_reports",
    "plant_ulp_move",
    "add_ulp_compare_arguments",
    "run_ulp_comparison",
]

#: One float64 ulp at unit scale (``numpy.spacing(1.0)``), spelled without a
#: numpy import so this module is importable from a bare gate script.
ULP = 2.0 ** -52

#: The bar.  Never relax it; a change that needs more is not a re-association.
MAX_ULP_MOVE = 2

#: Row keys that carry a MEASURED number and may therefore move.  Every other
#: key in a row (``status``, ``exact``, ``n``, ``bar``, ``reference_max_abs``,
#: prose) must match exactly.
MOVABLE_ROW_KEYS = ("normalized_max_abs", "absolute_max")

#: Rows whose name ends in one of these are tracer rows: bit-identical or bust.
TRACER_ROW_SUFFIXES = (".T", ".S")

#: Row keys that are a pure RESTATEMENT of the row's own residual rather than
#: an independent fact.  ``exact`` is ``normalized_max_abs == 0``, so a move
#: that the bar admits can legitimately flip it -- LOCK's kt=2 SSH residual
#: went 4.78e-28 -> 0.0 (0.000 ulp) and with it ``exact`` False -> True.
#: Comparing them directly would count one admitted move twice.  They are
#: instead checked for CONSISTENCY with the residual in each report (so a
#: future redefinition fails closed) and any flip is reported, not hidden.
DERIVED_ROW_KEYS = ("exact",)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_certified_row(node) -> bool:
    """A gate row: a dict naming itself and carrying a normalised residual."""
    return (
        isinstance(node, dict)
        and isinstance(node.get("name"), str)
        and _is_number(node.get("normalized_max_abs"))
    )


def certified_rows(
    report, *, row_filter: Callable[[str], bool] | None = None,
) -> dict[str, dict]:
    """``{row name: row}`` for every certified row anywhere in ``report``.

    Raises on a duplicated row name: the comparison is keyed by name, so a
    duplicate would silently compare one row twice and skip another.
    """
    found: dict[str, dict] = {}

    def walk(node) -> None:
        if _is_certified_row(node):
            name = node["name"]
            if row_filter is not None and not row_filter(name):
                return
            if name in found:
                raise ValueError(
                    f"duplicate certified row name {name!r}: this comparison "
                    "is keyed by name, so a duplicate would hide a row")
            found[name] = node
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(report)
    return found


def row_move_tolerance(row: dict, key: str, *, max_ulp: int = MAX_ULP_MOVE) -> float:
    """Allowed movement of ``row[key]``, in the units ``row[key]`` is in."""
    scale = 1.0
    if key == "absolute_max":
        # Reported in the field's own units, so the same relative bar is a
        # larger absolute number.  Rows without a reference scale fall back to
        # unit scale, which is the tighter of the two.
        reference = row.get("reference_max_abs")
        if _is_number(reference):
            scale = max(abs(float(reference)), 1.0)
    return max_ulp * ULP * scale


def _is_tracer_row(name: str) -> bool:
    return name.endswith(TRACER_ROW_SUFFIXES)


def compare_gate_reports(
    reference: dict, candidate: dict, *,
    max_ulp: int = MAX_ULP_MOVE,
    row_filter: Callable[[str], bool] | None = None,
) -> dict:
    """Machine check of the ulp bar between two gate JSONs.

    ``status`` is ``"PASS"`` only when every certified row moved at most
    ``max_ulp`` ulps of its own scale, every tracer row did not move at all,
    the row set and every non-measured row field are identical, and both
    ``first_over_bar`` and the report ``status`` are unchanged.
    """
    violations: list[str] = []
    reference_rows = certified_rows(reference, row_filter=row_filter)
    candidate_rows = certified_rows(candidate, row_filter=row_filter)

    if not reference_rows:
        violations.append(
            "the reference report contains no certified rows — a comparison "
            "that inspects nothing cannot fail, so this is a gate error")

    for name in sorted(set(reference_rows) - set(candidate_rows)):
        violations.append(f"{name}: present in the reference, missing from the candidate")
    for name in sorted(set(candidate_rows) - set(reference_rows)):
        violations.append(f"{name}: present in the candidate, missing from the reference")

    moves: list[dict] = []
    derived_changes: list[dict] = []
    for name in sorted(set(reference_rows) & set(candidate_rows)):
        before, after = reference_rows[name], candidate_rows[name]
        tracer = _is_tracer_row(name)
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                violations.append(
                    f"{name}: field {key!r} is in one report and not the other")
                continue
            old, new = before[key], after[key]
            if key in DERIVED_ROW_KEYS:
                for label, row in (("reference", before), ("candidate", after)):
                    residual = row.get("normalized_max_abs")
                    if bool(row[key]) != (_is_number(residual) and residual == 0.0):
                        violations.append(
                            f"{name}: the {label}'s {key!r}={row[key]!r} does not "
                            f"agree with its own normalized_max_abs={residual!r} — "
                            "this field is treated as a restatement of the "
                            "residual, and it no longer is")
                if old != new:
                    derived_changes.append({
                        "row": name, "field": key,
                        "reference": old, "candidate": new,
                        "note": ("derived from the residual, which moved within "
                                 "the bar; not counted a second time"),
                    })
                continue
            if key in MOVABLE_ROW_KEYS and _is_number(old) and _is_number(new):
                move = abs(float(new) - float(old))
                tolerance = 0.0 if tracer else row_move_tolerance(
                    after, key, max_ulp=max_ulp)
                ulps = move / row_move_tolerance(after, key, max_ulp=1)
                ok = move <= tolerance
                moves.append({
                    "row": name, "field": key,
                    "reference": float(old), "candidate": float(new),
                    "move": move, "ulps": ulps,
                    "tolerance": tolerance, "within_bar": ok,
                    "tracer_row_bit_identical_required": tracer,
                })
                if not ok:
                    violations.append(
                        f"{name}: {key} moved {move:.6e} "
                        f"({ulps:.3f} ulp) — the bar is "
                        + ("BIT-IDENTICAL (tracer row)" if tracer
                           else f"{max_ulp} ulp = {tolerance:.6e}"))
            elif old != new:
                violations.append(
                    f"{name}: field {key!r} changed {old!r} -> {new!r} "
                    "(only measured residuals may move at all)")

    # Report-level fields that must not move.  A key absent from BOTH reports
    # is INERT, not satisfied -- ``first_over_bar`` does not exist in the
    # stage-sweep schema, and a receipt must not claim a prong that never ran.
    checked_report_keys, inert_report_keys = [], []
    for key in ("first_over_bar", "status", "selectors", "precision_policy"):
        in_reference, in_candidate = key in reference, key in candidate
        if not in_reference and not in_candidate:
            inert_report_keys.append(key)
            continue
        checked_report_keys.append(key)
        old, new = reference.get(key, "<absent>"), candidate.get(key, "<absent>")
        if old != new:
            violations.append(f"{key} changed {old!r} -> {new!r}")

    return {
        "format": "legoesm-ocean-ulp-move-gate-v1",
        "status": "PASS" if not violations else "FAIL",
        "max_ulp_move": max_ulp,
        "ulp_at_unit_scale": ULP,
        "n_certified_rows_compared": len(set(reference_rows) & set(candidate_rows)),
        "row_filter_applied": row_filter is not None,
        "report_keys_checked": checked_report_keys,
        "report_keys_absent_from_both_so_unchecked": inert_report_keys,
        "first_over_bar": candidate.get("first_over_bar", "<absent>"),
        "largest_move_ulps": max((m["ulps"] for m in moves), default=0.0),
        "moves": [m for m in moves if m["move"] != 0.0],
        "derived_field_changes": derived_changes,
        "violations": violations,
    }


def plant_ulp_move(
    report: dict, n_ulps: float, *,
    row_filter: Callable[[str], bool] | None = None,
) -> dict:
    """Move ONE certified row by ``n_ulps`` — the gate's non-vacuity control.

    A planted move larger than the bar MUST turn ``compare_gate_reports`` red;
    if it does not, the comparison is inspecting nothing.  Mutates ``report``
    in place (callers hand it a freshly parsed copy) and returns it.
    """
    key = "normalized_max_abs"
    rows = certified_rows(report, row_filter=row_filter)
    # The target must be a NON-TRACER row (a tracer row's bar is bit-identity,
    # so any move there fails and the ulp bar is never exercised) with a
    # NONZERO residual (a zero-residual row carries ``exact=True``, and moving
    # it fails on the derived-field consistency check instead of on the move --
    # which made an earlier version of this control vacuous: it stayed red with
    # ``MOVABLE_ROW_KEYS`` emptied, i.e. with the bar deleted outright).
    name = next(
        (n for n in sorted(rows)
         if not _is_tracer_row(n) and float(rows[n].get(key, 0.0)) != 0.0),
        None)
    if name is None:
        raise ValueError(
            "nothing to plant into: no non-tracer certified row with a "
            "nonzero residual, so a plant could not exercise the ulp bar")
    row = rows[name]
    # No extra key is written: the plant must fail on the MOVE, not on a
    # schema difference the comparison would have caught anyway.
    row[key] = float(row[key]) + n_ulps * row_move_tolerance(row, key, max_ulp=1)
    return report


def add_ulp_compare_arguments(parser) -> None:
    """Wire ``--compare-to`` onto a gate script's ``ArgumentParser``.

    Lives here rather than in each gate so the two phase-3 gates share one
    implementation of the bar -- the same rule this module's own subject
    (a twice-written code path) exists to enforce.
    """
    parser.add_argument(
        "--compare-to", type=Path, metavar="REFERENCE.json",
        help=("after producing this run's report, compare it against a "
              "committed reference gate JSON under the ulp bar: every "
              f"certified row within {MAX_ULP_MOVE} float64 ulps of its own "
              "scale, tracer (T/S) rows bit-identical, first_over_bar and "
              "report status unchanged. Exit status then reflects the "
              "COMPARISON, not the gate's own AT-BAR/DEBT verdict."))
    parser.add_argument(
        "--compare-rows-matching", metavar="SUBSTRING",
        help=("restrict the comparison to certified rows whose name contains "
              "SUBSTRING (e.g. '.faithful.' to exclude private ablation arms "
              "whose definition the change under test deliberately redefines; "
              "say so in the receipt when you use it)"))
    parser.add_argument(
        "--compare-plant-ulps", type=float, metavar="N",
        help=("non-vacuity control: move ONE certified row by N ulps before "
              f"comparing. N > {MAX_ULP_MOVE} MUST make the comparison FAIL; "
              "if it passes, the comparison is inspecting nothing"))


def run_ulp_comparison(args, report: dict) -> dict:
    """Run the ``--compare-to`` comparison for a gate script's ``main``."""
    substring = getattr(args, "compare_rows_matching", None)
    row_filter = None if substring is None else (lambda name: substring in name)
    reference = json.loads(Path(args.compare_to).read_text())
    candidate = copy.deepcopy(report)
    planted = getattr(args, "compare_plant_ulps", None)
    if planted is not None:
        plant_ulp_move(candidate, planted, row_filter=row_filter)
    result = compare_gate_reports(reference, candidate, row_filter=row_filter)
    result["reference"] = str(args.compare_to)
    result["planted_ulp_move"] = planted
    # Record WHICH rows were compared, not just that a filter ran: a receipt
    # citing a filtered comparison has to be auditable from its own JSON.
    result["row_filter_substring"] = substring
    return result


def comparison_exit_code(result: dict) -> int:
    """0 pass, 1 the bar refused it, 2 the gate itself is broken.

    A requested plant that does NOT land is case 2, never case 0: a control
    that silently reports success is the failure mode this whole module is
    built to avoid.
    """
    planted = result.get("planted_ulp_move")
    if planted is not None and result["status"] == "PASS":
        return 2
    return 0 if result["status"] == "PASS" else 1
