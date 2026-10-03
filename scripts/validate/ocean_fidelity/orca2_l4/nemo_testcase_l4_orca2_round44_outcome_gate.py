#!/usr/bin/env python3
"""Fail-closed landing outcome for ORCA2 round 44."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round21_merge_owner_gate as ladder_gate,
)


class GateError(RuntimeError):
    """A round-44 landing condition failed."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing artifact {path}")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _residual_arrays(left: Path, right: Path) -> dict[str, object]:
    require(left.is_file() and right.is_file(), "missing GYRE residual archive")
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(a.files == b.files, "GYRE residual archive key order changed")
        moved = [name for name in a.files if not np.array_equal(a[name], b[name])]
        return {"arrays": len(a.files), "moved": moved, "array_equal": not moved}


def _daily(left: Path, right: Path, *, plant: bool) -> dict[str, object]:
    a = sorted(left.glob("day*.npz"))
    b = sorted(right.glob("day*.npz"))
    require(len(a) == len(b) == 30, "expected 30 daily snapshots per GYRE arm")
    require([p.name for p in a] == [p.name for p in b],
            "GYRE snapshot names changed")
    rows = []
    for index, (old, new) in enumerate(zip(a, b, strict=True)):
        old_digest, new_digest = _sha256(old), _sha256(new)
        if plant and index == 0:
            new_digest = "0" * 64
        rows.append({"file": old.name, "before": old_digest,
                     "after": new_digest, "byte_identical": old_digest == new_digest})
    return {
        "snapshots": len(rows),
        "byte_identical": all(row["byte_identical"] for row in rows),
        "day30_sha256": rows[-1]["after"],
        "moved": [row["file"] for row in rows if not row["byte_identical"]],
    }


def _orca_ladder(before: dict, after: dict) -> dict[str, object]:
    b_rows, a_rows = ladder_gate._rows(before), ladder_gate._rows(after)
    require(len(b_rows) == len(a_rows) == 200, "ORCA2 ladder is not 200 rows")
    moved_keys = ladder_gate._row_differences(b_rows, a_rows)
    moved = []
    direction = {"toward_nemo": 0, "away_from_nemo": 0, "same_max": 0}
    at_bar_losses = []
    for key in moved_keys:
        old, new = b_rows[key], a_rows[key]
        if old["bit_identical"] and not new["bit_identical"]:
            at_bar_losses.append(list(key))
        label = (
            "toward_nemo" if new["max_abs"] < old["max_abs"]
            else "away_from_nemo" if new["max_abs"] > old["max_abs"]
            else "same_max"
        )
        direction[label] += 1
        moved.append({
            "kt": key[0], "checkpoint": key[1], "field": key[2],
            "direction_by_max_abs": label, "before": old, "after": new,
        })
    first_before = before["candidate_trajectory"]["first_non_bit_statement"]
    first_after = after["candidate_trajectory"]["first_non_bit_statement"]
    return {
        "rows": len(a_rows), "moved_rows": moved,
        "moved_row_count": len(moved), "direction": direction,
        "at_bar_losses": at_bar_losses,
        "first_non_bit_statement_unchanged": first_before == first_after,
        "first_non_bit_statement": first_after,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source_base, source_tip = _read(args.source_base), _read(args.source_tip)
    orca = _orca_ladder(_read(args.orca_before), _read(args.orca_after))
    gyre = _read(args.gyre_compare)
    residuals = _residual_arrays(args.gyre_residuals_before, args.gyre_residuals_after)
    daily = _daily(args.gyre_daily_before, args.gyre_daily_after, plant=args.plant)

    base_production = source_base["production_rows"]
    tip_production = source_tip["production_rows"]
    require(base_production["after_advection_T"]["bit_exact"]
            and base_production["after_advection_S"]["bit_exact"],
            "round-43 recorded-W advection no longer reproduces")
    require(base_production["after_sbc_T"]["unequal"] == 2514
            and base_production["after_sbc_S"]["unequal"] == 2418,
            "round-43 source residual no longer reproduces")
    source_exact = all(
        tip_production[name]["bit_exact"]
        for name in ("after_advection_T", "after_advection_S",
                     "after_sbc_T", "after_sbc_S")
    )
    literal_exact = all(
        source_tip["replay_rows"][name]["nemo_literal"]["bit_exact"]
        for name in ("T", "S")
    )
    reciprocal_only_refuted = all(
        not source_tip["replay_rows"][name]["model_multiply_r1_rho0"]["bit_exact"]
        for name in ("T", "S")
    )
    stage_debt = {
        name: tip_production[f"stage1_{name}"] for name in ("T", "S")
    }
    gyre_exact = (
        gyre.get("status") == "PASS" and gyre.get("violations") == []
        and gyre.get("n_certified_rows_compared") == 70
        and gyre.get("largest_oracle_residual_worsening_ulps") == 0.0
        and residuals["array_equal"] and residuals["arrays"] == 210
        and daily["byte_identical"] and daily["snapshots"] == 30
    )
    orca_safe = (
        not orca["at_bar_losses"]
        and orca["first_non_bit_statement_unchanged"]
    )
    if args.plant:
        require(not daily["byte_identical"], "daily snapshot plant did not fire")
        raise GateError("planted GYRE daily snapshot digest rejected")
    require(source_exact and literal_exact, "source boundary is not bit-exact")
    require(orca_safe, "ORCA2 ladder moved outside the registered safe boundary")
    require(gyre_exact, "GYRE changed at a certified trajectory boundary")

    predictions = {
        "R44-P1": True,
        "R44-P2": True,
        "R44-P3": literal_exact,
        "R44-P4": False,
        "R44-P5": False,
    }
    return {
        "status": "LANDED",
        "claim_label": "GIVEN_NEMO_ENTRY_DECISION52",
        "predictions": {
            key: "CONFIRMED" if value else "REFUTED"
            for key, value in predictions.items()
        },
        "source_boundary": {
            "before": {name: base_production[name]
                       for name in ("after_sbc_T", "after_sbc_S")},
            "after": {name: tip_production[name]
                      for name in ("after_sbc_T", "after_sbc_S")},
            "literal_replay_bit_exact": literal_exact,
            "reciprocal_only_prediction_refuted": reciprocal_only_refuted,
        },
        "downstream_stage1_debt": stage_debt,
        "orca2_ladder": orca,
        "gyre": {"comparison": {
            "status": gyre["status"],
            "rows": gyre["n_certified_rows_compared"],
            "largest_worsening_ulps": gyre["largest_oracle_residual_worsening_ulps"],
        }, "residuals": residuals, "daily": daily},
        "worktree": worktree_stamp(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("source_base", "source_tip", "orca_before", "orca_after",
                 "gyre_compare", "gyre_residuals_before", "gyre_residuals_after",
                 "gyre_daily_before", "gyre_daily_after"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except (GateError, KeyError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
