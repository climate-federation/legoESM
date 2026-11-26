#!/usr/bin/env python3
"""Round 332: rescore every current ORCA2 D96 ladder row.

This is deliberately a thin extension of the reviewed round-329 and round-309
measurement paths.  It changes no card and no integrator.  The only patched
function is the common score leaf, so the emitted rows additionally retain
the oracle scale and the preregistered bit/at-bar/over-bar classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round285_g4_d122_gate as r285,
    nemo_testcase_l4_orca2_round309_omt0_v3_d122_gate as r309,
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
)

BAR = 1.0e-15
PAIRS = (
    ("omt0", "independent"),
    ("rung0", "independent"),
    ("omt1", "independent"),
    ("omt2", "independent"),
    ("omt3", "independent"),
    ("omt4", "independent"),
    ("rung10", "given_nemo_entry"),
    ("rung10", "independent"),
)
CLASSES = ("BIT_EXACT", "AT_BAR_NOT_EXACT", "OVER_BAR")


class GateError(RuntimeError):
    """The round-332 score or aggregate is incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def score_with_bar(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """The campaign's strict bit score plus the preregistered row bar."""

    row = _ORIGINAL_SCORE(actual, expected)
    oracle_max = float(np.max(np.abs(expected)))
    scale = max(oracle_max, 1.0)
    normalized = float(row["max_abs"]) / scale
    if row["bit_identical"]:
        classification = "BIT_EXACT"
    elif normalized <= BAR:
        classification = "AT_BAR_NOT_EXACT"
    else:
        classification = "OVER_BAR"
    return {
        **row,
        "oracle_max_abs": oracle_max,
        "normalization_scale": scale,
        "normalized_max_abs": normalized,
        "bar": BAR,
        "classification": classification,
    }


_ORIGINAL_SCORE = rung10.score


def _measure_standard(rung: str, label: str, deck_root: Path,
                      twin_a: Path, expect_commit: str) -> dict:
    def select(base, *, candidate, rung):
        del candidate
        return r329.selected_card(base, arm="recordice", rung=rung)

    original_select, r285.selected_card = r285.selected_card, select
    original_score, rung10.score = rung10.score, score_with_bar
    try:
        result = r285.measure(
            rung, label, deck_root, twin_a, True, expect_commit)
    finally:
        rung10.score = original_score
        r285.selected_card = original_select
    return result


def _measure_omt0(deck_root: Path, twin_a: Path,
                  expect_commit: str) -> dict:
    original_score, rung10.score = rung10.score, score_with_bar
    try:
        return r309.measure_omt0("D122", deck_root, twin_a, expect_commit)
    finally:
        rung10.score = original_score


def measure(rung: str, label: str, deck_root: Path,
            twin_a: Path, expect_commit: str) -> dict:
    require((rung, label) in PAIRS, f"unsupported pair {(rung, label)}")
    result = (_measure_omt0(deck_root, twin_a, expect_commit)
              if rung == "omt0" else
              _measure_standard(rung, label, deck_root, twin_a, expect_commit))
    rows = result["ladder"]["rows"]
    expected = 160 if rung == "omt1" else 200
    require(len(rows) == expected, f"{rung}/{label}: incomplete ladder")
    require(all(row.get("classification") in CLASSES for row in rows),
            f"{rung}/{label}: row classification missing")
    counts = Counter(row["classification"] for row in rows)
    first = next(({
        "kt": row["kt"], "checkpoint": row["checkpoint"],
        "field": row["field"], "unequal": row["unequal"],
        "max_abs": row["max_abs"], "rms": row["rms"],
        "classification": row["classification"],
    } for row in rows if not row["bit_identical"]), None)
    return {
        **result,
        "format": "nemo-testcase-l4-orca2-round332-rescore-v1",
        "arm": "current-tip",
        "bar_definition": "max_abs / max(max_abs(oracle), 1) <= 1e-15",
        "classification_counts": {name: int(counts[name]) for name in CLASSES},
        "first_non_bit_row": first,
    }


def aggregate(paths: list[Path], table_out: Path | None) -> dict:
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    keys = [(doc["rung"], doc["label"]) for doc in docs]
    require(len(keys) == len(set(keys)), "duplicate pair in aggregate")
    require(set(keys) == set(PAIRS), f"pair set incomplete: {sorted(keys)}")
    commits = {doc["worktree"]["commit"] for doc in docs}
    require(len(commits) == 1, "measurements mix commits")
    rows = []
    first = {}
    for doc in docs:
        pair = f"{doc['rung']}/{doc['label']}"
        first[pair] = doc["first_non_bit_row"]
        for row in doc["ladder"]["rows"]:
            rows.append({"rung": doc["rung"], "label": doc["label"], **row})
    require(len(rows) == 1560, f"aggregate has {len(rows)} rows, expected 1560")
    counts = Counter(row["classification"] for row in rows)
    require(sum(counts.values()) == 1560, "classification census incomplete")
    if table_out is not None:
        columns = (
            "rung", "label", "kt", "checkpoint", "field", "max_abs", "rms",
            "unequal", "count", "oracle_max_abs", "normalization_scale",
            "normalized_max_abs", "classification",
        )
        table_out.parent.mkdir(parents=True, exist_ok=True)
        with table_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return {
        "format": "nemo-testcase-l4-orca2-round332-aggregate-v1",
        "status": "PASS_R332_FULL_RESCORE",
        "worktree_commit": next(iter(commits)),
        "row_count": len(rows),
        "classification_counts": {name: int(counts[name]) for name in CLASSES},
        "first_non_bit_by_pair": first,
        "table": str(table_out) if table_out is not None else None,
    }


def self_test(*, plant: bool = False) -> dict:
    exact = np.array([1.0], np.float64)
    signed = np.array([-0.0], np.float64)
    at_bar = np.array([1.0 + np.spacing(1.0)], np.float64)
    over = np.array([1.0 + 8.0e-15], np.float64)
    if plant:
        signed = np.array([0.0], np.float64)
    got = (
        score_with_bar(exact, exact)["classification"],
        score_with_bar(signed, np.array([0.0], np.float64))["classification"],
        score_with_bar(at_bar, exact)["classification"],
        score_with_bar(over, exact)["classification"],
    )
    require(got == ("BIT_EXACT", "AT_BAR_NOT_EXACT",
                    "AT_BAR_NOT_EXACT", "OVER_BAR"),
            f"classification control failed: {got}")
    return {"status": "PASS_R332_SCORE_CONTROL", "classes": got}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung", choices=tuple(sorted({p[0] for p in PAIRS})))
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--aggregate", nargs="+", type=Path)
    parser.add_argument("--table-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root,
                         args.twin_a, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(args.rung, args.label, args.deck_root,
                             args.twin_a, args.expect_commit)
        elif args.aggregate:
            result = aggregate(args.aggregate, args.table_out)
        elif args.self_test:
            result = self_test(plant=args.plant)
        else:
            raise GateError("no operation selected")
        require(not args.plant, "signed-zero plant stayed green")
    except (GateError, r309.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
