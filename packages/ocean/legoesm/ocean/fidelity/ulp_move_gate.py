"""Oracle-relative admission gate for shared ocean-fidelity changes.

The certified object is the field, not its L-infinity reduction. Every gate
writes, for each scored boundary, the active-cell oracle values, candidate
values, and ``abs(candidate - oracle)`` residuals to a compressed NPZ sidecar.
``--compare-to`` compares those residuals cell by cell.

For every cell in a scored row, one ulp means
``numpy.spacing(max(max(abs(float64(oracle_row))), 1.0))``.  This is one
binary64 spacing at the row's campaign-normalization magnitude, floored at
one; it deliberately does not turn zeros or denormal oracle cells into an
unattainable subnormal-scale gate. A change fails when
``residual_after - residual_before > MAX_ULP_MOVE * row_scale_ulp`` at any cell.
Moves toward NEMO are free. It also fails when an AT-BAR row becomes DEBT or
when ``first_over_bar`` moves to an earlier time step. Movement from the old
legoesm field to the new legoESM field is still computed and disclosed, but is
not an admission criterion.

Compensating-error clause: A change is eligible to land only if the changed
operator is shown bit-exact given NEMO's own inputs on every card it touches.
If such a change makes a card worse against NEMO, that is a second error
exposed: the fix stays, the worsened row enters that card's register as debt
naming the boundary, and the next round walks it. Never reverted, never
waived silently, never a per-card switch.

Harness glue, not model code. The stage and trajectory gates share this one
implementation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp

__all__ = [
    "MAX_ULP_MOVE",
    "ResidualFieldRecorder",
    "capture_residual_fields",
    "record_residual_field",
    "write_residual_artifact",
    "load_residual_artifact",
    "certified_rows",
    "compare_gate_reports",
    "plant_cellwise_comparison",
    "plant_at_bar_to_debt",
    "add_ulp_compare_arguments",
    "run_ulp_comparison",
    "persist_ulp_comparison",
    "comparison_exit_code",
]

MAX_ULP_MOVE = 2
ARTIFACT_FORMAT = "legoesm-ocean-oracle-residual-fields-v1"
COMPARISON_FORMAT = "legoesm-ocean-oracle-relative-move-gate-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_certified_row(node) -> bool:
    return (
        isinstance(node, dict)
        and isinstance(node.get("name"), str)
        and _is_number(node.get("normalized_max_abs"))
    )


def certified_rows(
    report, *, row_filter: Callable[[str], bool] | None = None,
) -> dict[str, dict]:
    """Return every uniquely named scored row anywhere in a gate report."""
    found: dict[str, dict] = {}

    def walk(node) -> None:
        if _is_certified_row(node):
            name = node["name"]
            if row_filter is not None and not row_filter(name):
                return
            if name in found:
                raise ValueError(f"duplicate certified row name {name!r}")
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


class ResidualFieldRecorder:
    """In-memory active-cell payload collected by a gate's score function."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, np.ndarray]] = {}

    def record(self, name: str, oracle, candidate, mask) -> None:
        if name in self.rows:
            raise ValueError(f"duplicate residual field name {name!r}")
        oracle = np.asarray(oracle, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        active = np.asarray(mask, dtype=bool)
        if oracle.shape != candidate.shape or oracle.shape != active.shape:
            raise ValueError(f"{name}: residual field shape mismatch")
        oracle_active = np.ascontiguousarray(oracle[active], dtype=np.float64)
        candidate_active = np.ascontiguousarray(candidate[active], dtype=np.float64)
        self.rows[name] = {
            "oracle": oracle_active,
            "candidate": candidate_active,
            "residual": np.abs(candidate_active - oracle_active),
        }


_ACTIVE_RECORDER: ContextVar[ResidualFieldRecorder | None] = ContextVar(
    "ocean_fidelity_residual_recorder", default=None)


@contextmanager
def capture_residual_fields() -> Iterator[ResidualFieldRecorder]:
    """Capture all calls to :func:`record_residual_field` in this context."""
    recorder = ResidualFieldRecorder()
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def record_residual_field(name: str, oracle, candidate, mask) -> None:
    """Record a scored field when a gate main has enabled capture."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is not None:
        recorder.record(name, oracle, candidate, mask)


def write_residual_artifact(
    report: dict, report_path: Path, recorder: ResidualFieldRecorder,
) -> Path:
    """Write and register the compressed per-cell residual sidecar."""
    report_path = Path(report_path)
    sidecar = report_path.with_suffix(".residuals.npz")
    arrays: dict[str, np.ndarray] = {}
    rows: dict[str, dict] = {}
    for index, name in enumerate(sorted(recorder.rows)):
        prefix = f"r{index:05d}"
        payload = recorder.rows[name]
        rows[name] = {
            "oracle": f"{prefix}_oracle",
            "candidate": f"{prefix}_candidate",
            "residual": f"{prefix}_residual",
            "n": int(payload["oracle"].size),
        }
        for field in ("oracle", "candidate", "residual"):
            arrays[rows[name][field]] = payload[field]
    if not arrays:
        raise ValueError("cannot write an empty oracle-residual artifact")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sidecar, **arrays)
    digest = _sha256(sidecar)
    report["oracle_relative_residual_artifact"] = {
        "format": ARTIFACT_FORMAT,
        "path": sidecar.name,
        "sha256": digest,
        "n_rows": len(rows),
        "rows": rows,
    }
    report.setdefault("artifacts", {})[sidecar.name] = digest
    return sidecar


def load_residual_artifact(report: dict, report_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load a report's sidecar and fail closed on schema/hash mismatches."""
    meta = report.get("oracle_relative_residual_artifact")
    if not isinstance(meta, dict) or meta.get("format") != ARTIFACT_FORMAT:
        raise ValueError("report has no supported oracle-relative residual artifact")
    path = Path(meta["path"])
    if not path.is_absolute():
        path = Path(report_path).parent / path
    if not path.is_file():
        raise ValueError(f"missing oracle-relative residual artifact: {path}")
    actual = _sha256(path)
    if actual != meta.get("sha256"):
        raise ValueError(
            f"oracle-relative residual artifact hash mismatch: {actual} != {meta.get('sha256')}")
    result: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as data:
        for name, keys in meta.get("rows", {}).items():
            payload = {
                field: np.asarray(data[keys[field]], dtype=np.float64)
                for field in ("oracle", "candidate", "residual")
            }
            n = int(keys["n"])
            if any(values.ndim != 1 or values.size != n for values in payload.values()):
                raise ValueError(f"{name}: malformed residual field payload")
            expected = np.abs(payload["candidate"] - payload["oracle"])
            if not np.array_equal(payload["residual"], expected):
                raise ValueError(
                    f"{name}: persisted residual is inconsistent with candidate/oracle")
            result[name] = payload
    if len(result) != int(meta.get("n_rows", -1)):
        raise ValueError("residual artifact row count does not match its manifest")
    return result


def _first_over_bar_kt(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict) and isinstance(value.get("kt"), int):
        return value["kt"]
    raise ValueError(f"malformed first_over_bar value: {value!r}")


def _row_scale_ulp(oracle: np.ndarray) -> np.float64:
    """One float64 spacing at ``max(max(abs(oracle)), 1)`` for the row."""
    magnitude = float(np.max(np.abs(np.asarray(oracle, dtype=np.float64)), initial=0.0))
    return np.spacing(np.float64(max(magnitude, 1.0)))


def compare_gate_reports(
    reference: dict,
    candidate: dict,
    *,
    reference_fields: dict[str, dict[str, np.ndarray]],
    candidate_fields: dict[str, dict[str, np.ndarray]],
    max_ulp: int = MAX_ULP_MOVE,
    row_filter: Callable[[str], bool] | None = None,
) -> dict:
    """Compare before/after residual fields against their common NEMO oracle."""
    violations: list[str] = []
    before_rows = certified_rows(reference, row_filter=row_filter)
    after_rows = certified_rows(candidate, row_filter=row_filter)
    if not before_rows:
        violations.append("the reference report contains no certified rows")
    for name in sorted(set(before_rows) ^ set(after_rows)):
        side = "reference" if name in before_rows else "candidate"
        violations.append(f"{name}: present only in the {side} report")

    field_moves: list[dict] = []
    row_changes: list[dict] = []
    common = sorted(set(before_rows) & set(after_rows))
    for name in common:
        before, after = before_rows[name], after_rows[name]
        if name not in reference_fields or name not in candidate_fields:
            violations.append(f"{name}: missing per-cell residual field")
            continue
        old = reference_fields[name]
        new = candidate_fields[name]
        if any(old[key].shape != new[key].shape for key in ("oracle", "candidate", "residual")):
            violations.append(f"{name}: before/after residual field shapes differ")
            continue
        if not np.array_equal(old["oracle"], new["oracle"]):
            violations.append(f"{name}: NEMO oracle field changed between before and after")
            continue

        ulp = _row_scale_ulp(new["oracle"])
        degradation = new["residual"] - old["residual"]
        tolerance = max_ulp * ulp
        bad = degradation > tolerance
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            degradation_ulps = np.where(degradation > 0.0, degradation / ulp, 0.0)
            model_move_ulps = np.abs(new["candidate"] - old["candidate"]) / ulp
        move = {
            "row": name,
            "n_cells": int(new["oracle"].size),
            "row_scale_ulp": float(ulp),
            "n_worsened_cells": int(np.count_nonzero(degradation > 0.0)),
            "n_improved_cells": int(np.count_nonzero(degradation < 0.0)),
            "n_cells_worse_than_bar": int(np.count_nonzero(bad)),
            "max_oracle_residual_worsening": float(np.max(degradation, initial=0.0)),
            "max_oracle_residual_worsening_ulps": float(
                np.max(degradation_ulps, initial=0.0)),
            "max_previous_legoesm_field_move": float(
                np.max(np.abs(new["candidate"] - old["candidate"]), initial=0.0)),
            "max_previous_legoesm_field_move_in_row_scale_oracle_ulps": float(
                np.max(model_move_ulps, initial=0.0)),
        }
        field_moves.append(move)
        if np.any(bad):
            index = int(np.flatnonzero(bad)[0])
            violations.append(
                f"{name}: cell {index} worsened against NEMO by "
                f"{degradation[index]:.17e} = {degradation_ulps[index]:.3f} "
                "row-scale oracle ulp; "
                f"bar is {max_ulp} ulp")

        old_status, new_status = before.get("status"), after.get("status")
        if old_status != new_status:
            row_changes.append({"row": name, "reference": old_status, "candidate": new_status})
            if old_status == "AT-BAR" and new_status == "DEBT":
                violations.append(f"{name}: status crossed AT-BAR -> DEBT")

        # These are reductions derived from the persisted cell fields.  They
        # are disclosed in each report but cannot be an independent admission
        # criterion: the oracle-relative residual comparison above is the
        # certified object and deliberately sees compensation hidden by a
        # reduction.  Structural metadata (n, dtype, bar, staggering, etc.)
        # remains fail-closed below.
        ignored = {
            "normalized_max_abs",
            "absolute_max",
            "relative_max_abs",
            "n_unequal",
            "exact",
            "status",
        }
        for key in sorted((set(before) | set(after)) - ignored):
            if key not in before or key not in after or before.get(key) != after.get(key):
                violations.append(
                    f"{name}: non-residual field {key!r} changed "
                    f"{before.get(key, '<absent>')!r} -> {after.get(key, '<absent>')!r}")

    selected = set(common)
    for label, fields in (("reference", reference_fields), ("candidate", candidate_fields)):
        available = {name for name in fields if row_filter is None or row_filter(name)}
        for name in sorted(available - selected):
            violations.append(f"{name}: {label} residual artifact has an unscored extra row")

    checked_report_keys, inert_report_keys = [], []
    for key in ("selectors", "precision_policy"):
        if key not in reference and key not in candidate:
            inert_report_keys.append(key)
        else:
            checked_report_keys.append(key)
            if reference.get(key, "<absent>") != candidate.get(key, "<absent>"):
                violations.append(
                    f"{key} changed {reference.get(key, '<absent>')!r} -> "
                    f"{candidate.get(key, '<absent>')!r}")

    before_fob = reference.get("first_over_bar", "<absent>")
    after_fob = candidate.get("first_over_bar", "<absent>")
    if before_fob == "<absent>" and after_fob == "<absent>":
        inert_report_keys.append("first_over_bar")
    else:
        checked_report_keys.append("first_over_bar")
        try:
            old_kt = _first_over_bar_kt(None if before_fob == "<absent>" else before_fob)
            new_kt = _first_over_bar_kt(None if after_fob == "<absent>" else after_fob)
            if new_kt is not None and (old_kt is None or new_kt < old_kt):
                violations.append(f"first_over_bar moved earlier {before_fob!r} -> {after_fob!r}")
        except ValueError as error:
            violations.append(str(error))

    before_status, after_status = reference.get("status"), candidate.get("status")
    if before_status == "AT-BAR" and after_status == "DEBT":
        violations.append("report status crossed AT-BAR -> DEBT")
    report_status_change = None
    if before_status != after_status:
        report_status_change = {"reference": before_status, "candidate": after_status}

    return {
        "format": COMPARISON_FORMAT,
        "status": "PASS" if not violations else "FAIL",
        "criterion": "cellwise_oracle_relative",
        "max_ulp_worsening": max_ulp,
        "row_scale_ulp_definition": (
            "numpy.spacing(max(max(abs(float64(NEMO_row))), 1.0))"
        ),
        "n_certified_rows_compared": len(common),
        "row_filter_applied": row_filter is not None,
        "report_keys_checked": checked_report_keys,
        "report_keys_absent_from_both_so_unchecked": inert_report_keys,
        "first_over_bar_reference": before_fob,
        "first_over_bar_candidate": after_fob,
        "largest_oracle_residual_worsening_ulps": max(
            (item["max_oracle_residual_worsening_ulps"] for item in field_moves), default=0.0),
        "largest_previous_legoesm_field_move_in_row_scale_oracle_ulps": max(
            (item["max_previous_legoesm_field_move_in_row_scale_oracle_ulps"]
             for item in field_moves), default=0.0),
        "field_moves": field_moves,
        "row_status_changes": row_changes,
        "report_status_change": report_status_change,
        "violations": violations,
    }


def plant_cellwise_comparison(
    fields: dict[str, dict[str, np.ndarray]], mode: str,
    *, row_filter: Callable[[str], bool] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Plant one three-ulp worsening or one improvement in copied fields."""
    if mode not in {"worsen-3ulp", "improve"}:
        raise ValueError(f"unsupported cellwise plant {mode!r}")
    planted = {
        name: {key: np.array(value, copy=True) for key, value in payload.items()}
        for name, payload in fields.items()
    }
    names = [name for name in sorted(planted) if row_filter is None or row_filter(name)]
    for name in names:
        payload = planted[name]
        if mode == "improve":
            indices = np.flatnonzero(payload["residual"] > 0.0)
            if indices.size:
                index = int(indices[0])
                payload["candidate"][index] = payload["oracle"][index]
                payload["residual"][index] = 0.0
                return planted
        else:
            indices = np.flatnonzero(
                (payload["residual"] == 0.0)
                & np.isfinite(payload["oracle"]))
            if indices.size:
                index = int(indices[0])
                value = payload["oracle"][index] + (
                    (MAX_ULP_MOVE + 1) * _row_scale_ulp(payload["oracle"])
                )
                payload["candidate"][index] = value
                payload["residual"][index] = abs(value - payload["oracle"][index])
                return planted
    raise ValueError(f"nothing suitable for {mode!r} plant")


def plant_at_bar_to_debt(
    report: dict, *, row_filter: Callable[[str], bool] | None = None,
) -> dict:
    """Flip exactly one AT-BAR row to DEBT without touching its fields."""
    rows = certified_rows(report, row_filter=row_filter)
    name = next((name for name in sorted(rows) if rows[name].get("status") == "AT-BAR"), None)
    if name is None:
        raise ValueError("nothing to plant into: no AT-BAR certified row")
    rows[name]["status"] = "DEBT"
    return report


def add_ulp_compare_arguments(parser) -> None:
    parser.add_argument(
        "--compare-to", type=Path, metavar="BEFORE.json",
        help=("compare per-cell absolute residuals against the common NEMO oracle; "
              f"no cell may worsen by more than {MAX_ULP_MOVE} row-scale float64 ulps, "
              "no AT-BAR row may become DEBT, and first_over_bar may not move earlier"))
    parser.add_argument(
        "--compare-rows-matching", metavar="SUBSTRING",
        help="restrict comparison to scored row names containing SUBSTRING")
    parser.add_argument(
        "--compare-plant", choices=("worsen-3ulp", "improve", "at-bar-to-debt"),
        help="calibration control applied to one cell or row before comparison")
    parser.add_argument(
        "--comparison-output", type=Path, metavar="RESULT.json",
        help="persist the full oracle-relative comparison and its verdict line")


def run_ulp_comparison(args, report: dict) -> dict:
    """Load both hashed sidecars and apply the shared oracle-relative gate."""
    substring = getattr(args, "compare_rows_matching", None)
    row_filter = None if substring is None else (lambda name: substring in name)
    reference_path = Path(args.compare_to)
    candidate_path = getattr(args, "output", None)
    if candidate_path is None:
        raise ValueError("--compare-to requires --output so its residual sidecar is durable")
    reference = json.loads(reference_path.read_text())
    reference_fields = load_residual_artifact(reference, reference_path)
    candidate = copy.deepcopy(report)
    candidate_fields = load_residual_artifact(candidate, Path(candidate_path))
    plant = getattr(args, "compare_plant", None)
    if plant in {"worsen-3ulp", "improve"}:
        candidate_fields = plant_cellwise_comparison(
            candidate_fields, plant, row_filter=row_filter)
    elif plant == "at-bar-to-debt":
        plant_at_bar_to_debt(candidate, row_filter=row_filter)
    result = compare_gate_reports(
        reference, candidate,
        reference_fields=reference_fields,
        candidate_fields=candidate_fields,
        row_filter=row_filter)
    result["reference"] = str(reference_path)
    result["candidate"] = str(candidate_path)
    result["plant"] = plant
    result["row_filter_substring"] = substring
    # A comparison report is a MEASUREMENT and carries its provenance, exactly
    # as the gate reports it compares do.  Six of these were written unstamped:
    # a comparison taken on a dirty tree could not be told from one taken at a
    # commit, which is the whole point of the stamp ratchet.
    result["worktree"] = worktree_stamp()
    return result


def persist_ulp_comparison(args, result: dict) -> str:
    """Persist a comparison when requested and return its canonical verdict."""
    path = getattr(args, "comparison_output", None)
    if path is not None:
        Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return (
        f"ORACLE_RELATIVE_COMPARE {result['status']}: "
        f"rows={result['n_certified_rows_compared']} "
        f"max_worsening_ulps={result['largest_oracle_residual_worsening_ulps']:.17g} "
        f"first_over_bar={result['first_over_bar_reference']!r}->"
        f"{result['first_over_bar_candidate']!r} plant={result['plant']!r}")


def comparison_exit_code(result: dict) -> int:
    """Return 0 PASS, 1 refused/expected failing plant, 2 broken control."""
    plant = result.get("plant")
    expected = "PASS" if plant in {None, "improve"} else "FAIL"
    if plant is not None and result["status"] != expected:
        return 2
    return 0 if result["status"] == "PASS" else 1
