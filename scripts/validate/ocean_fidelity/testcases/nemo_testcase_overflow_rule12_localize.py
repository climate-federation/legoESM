#!/usr/bin/env python3
"""Localize cellwise Rule-12 movement in the OVERFLOW trajectory gate.

This diagnostic maps the active-cell ordering persisted by
``ulp_move_gate`` back to the model grid.  It deliberately does not infer an
operator from a reduction: the output inventories levels, columns, and
partial-cell bottoms, then records the first source operand that the current
state cannot reproduce.  Model coordinates are zero based ``(j, i, k)``;
NEMO coordinates include the two-point local halo and are Fortran one based.

The planted control removes one owned mask cell.  Payload/mask cardinality
must then disagree and the probe must exit nonzero, proving that a shifted
active-cell map cannot silently produce plausible coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.ulp_move_gate import (
    MAX_ULP_MOVE,
    compare_gate_reports,
    load_residual_artifact,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp


class ProbeError(RuntimeError):
    """Fail-closed diagnostic error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def overflow_masks() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the exact gate masks and each T column's last wet level."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("OVERFLOW-zps")
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    v = active & np.roll(active, -1, axis=0)
    u[:, -1] = False
    v[-1] = False
    bottom = np.max(
        np.where(active, np.arange(active.shape[-1]), -1), axis=-1)
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}, bottom


def _field_from_row(row: str) -> str:
    field = row.rsplit(".", 1)[-1]
    require(field in {"T", "S", "u", "v", "ssh"}, f"unknown row field: {row}")
    return field


def _coordinate(coord: np.ndarray) -> dict:
    j, i, *level = (int(value) for value in coord)
    result = {
        "model_zero_based": {"j": j, "i": i},
        "nemo_fortran_one_based": {"jj": j + 3, "ji": i + 3},
    }
    if level:
        result["model_zero_based"]["k"] = level[0]
        result["nemo_fortran_one_based"]["jk"] = level[0] + 1
    return result


def localize(before_path: Path, after_path: Path, *, plant: bool) -> dict:
    before = load_json(before_path)
    after = load_json(after_path)
    before_fields = load_residual_artifact(before, before_path)
    after_fields = load_residual_artifact(after, after_path)
    comparison = compare_gate_reports(
        before,
        after,
        reference_fields=before_fields,
        candidate_fields=after_fields,
    )
    flagged = [
        move for move in comparison["field_moves"]
        if move["n_cells_worse_than_bar"] > 0
    ]
    require(len(flagged) == 7, f"expected seven Rule-12 rows, found {len(flagged)}")

    masks, bottom_k = overflow_masks()
    if plant:
        first_field = _field_from_row(flagged[0]["row"])
        first_owned = tuple(np.argwhere(masks[first_field])[0])
        masks[first_field][first_owned] = False

    rows = []
    total_bottom = 0
    total_interior = 0
    for move in flagged:
        name = move["row"]
        field = _field_from_row(name)
        coords = np.argwhere(masks[field])
        old = before_fields[name]
        new = after_fields[name]
        require(
            coords.shape[0] == old["oracle"].size == new["oracle"].size,
            f"{name}: active-cell map does not match persisted payload",
        )
        require(np.array_equal(old["oracle"], new["oracle"]), f"{name}: oracle changed")
        ulp = np.spacing(max(float(np.max(np.abs(old["oracle"]))), 1.0))
        degradation = new["residual"] - old["residual"]
        bad = degradation > MAX_ULP_MOVE * ulp
        require(
            int(np.count_nonzero(bad)) == move["n_cells_worse_than_bar"],
            f"{name}: recomputed violation count differs",
        )
        bad_coords = coords[bad]
        max_index = int(np.argmax(degradation))
        levels: Counter[int] = Counter()
        columns: Counter[tuple[int, int]] = Counter()
        bottom_count = 0
        if field != "ssh":
            for j, i, k in bad_coords:
                levels[int(k)] += 1
                columns[(int(j), int(i))] += 1
                # A U face is owned only where both adjoining T columns are
                # active.  Its bottom is the shallower adjoining T bottom.
                if field == "u":
                    face_bottom = min(bottom_k[j, i], bottom_k[j, i + 1])
                elif field == "v":
                    face_bottom = min(bottom_k[j, i], bottom_k[j + 1, i])
                else:
                    face_bottom = bottom_k[j, i]
                bottom_count += int(k == face_bottom)
        else:
            for j, i in bad_coords:
                columns[(int(j), int(i))] += 1
        interior_count = int(bad_coords.shape[0]) - bottom_count
        if field != "ssh":
            total_bottom += bottom_count
            total_interior += interior_count
        rows.append({
            "row": name,
            "n_cells_worse_than_2_row_scale_ulp": int(bad_coords.shape[0]),
            "row_scale_ulp": float(ulp),
            "max_worsening_row_scale_ulp": float(np.max(degradation) / ulp),
            "first_bad": _coordinate(bad_coords[0]),
            "max_worsening": _coordinate(coords[max_index]),
            "levels_zero_based": dict(sorted(levels.items())),
            "columns_zero_based_j_i": [
                {"j": j, "i": i, "count": count}
                for (j, i), count in sorted(columns.items())
            ],
            "bottom_cells": bottom_count,
            "non_bottom_cells": interior_count,
        })

    require(
        all(item["first_bad"]["model_zero_based"]["j"] == 1 for item in rows),
        "the seven rows no longer lie on OVERFLOW's sole wet interior row",
    )
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-overflow-rule12-localization-v1",
        "regime": "CPU production JIT, fp64, scalar-libm",
        "reference_report": {"path": str(before_path), "sha256": sha256(before_path)},
        "candidate_report": {"path": str(after_path), "sha256": sha256(after_path)},
        "comparison_status": comparison["status"],
        "n_flagged_rows": len(rows),
        "rows": rows,
        "three_dimensional_violation_census": {
            "bottom": total_bottom,
            "non_bottom": total_interior,
            "classification": "REFUTED_PARTIAL_CELL_BOTTOM_CLUSTER",
        },
        "retracted_finding": {
            "classification": "REFUTED_AND_ILL_POSED",
            # Rule 11: the dead LABEL is kept next to what killed it, not
            # deleted.  Round 25 removed it outright, which is how a retracted
            # attribution gets rediscovered as if it were new.
            "struck_label": "CONFIRMED_STRUCTURAL_BOUNDARY_NUMERIC_VALUE_UNMEASURED",
            "struck_label_status": (
                "STRUCK 2026-09-05 (round 25, restored struck-in-place in "
                "round 26); it named stprk3_stg.F90:257-274 zub="
                "un_adv*r1_hu(Kmm)-uu_b(Kmm) as the boundary for the stage-2 "
                "and kt1 rows, which the source order below makes ill-posed"
            ),
            "dead_claim": (
                "persistent uu_b/vv_b(Kbb) was the first unavailable operand "
                "for the stage-2 and kt1 rows"
            ),
            "killed_by": (
                "stprk3.F90:186,195-207: stp_2D writes Naa before all stages; "
                "the stage-1 swap promotes it to Nnn, so stages 2-3 read this-step "
                "Nnn and never the carried Kbb pair"
            ),
            "live_scope": (
                "the carried Kbb pair can affect only a later step's external-mode "
                "seed (dynspg_ts.F90:484-500) and stage-1 S-21 operand "
                "(stprk3_stg.F90:257-274)"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        report = localize(args.before, args.after, plant=args.plant)
    except (OSError, ValueError, ProbeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
