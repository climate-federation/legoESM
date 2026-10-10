#!/usr/bin/env python3
"""Classify the first downstream consumer of ORCA2's exact vector pair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PLANTS = (
    "none", "boundary-v", "registry", "transport-mask", "materialization",
)
POST_REGISTRY = (
    "u", "v", "depth_u", "depth_v", "inverse_u", "inverse_v", "eta",
)
CHAIN = ("transport_v", "continuity_dv", "continuity_divergence", "after_ssh")


class GateError(RuntimeError):
    """The admitted downstream replay no longer supports its verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(pair: dict, association: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    pair = copy.deepcopy(pair)
    association = copy.deepcopy(association)
    registry = list(POST_REGISTRY)
    if plant == "boundary-v":
        association["post_association_rows"]["v"]["bit_exact"] = False
    elif plant == "registry":
        registry[0], registry[1] = registry[1], registry[0]
    elif plant == "transport-mask":
        association["transport_v_operand_split"]["rows"][
            "unmasked_transport_v"]["bit_exact"] = False
    elif plant == "materialization":
        association["materialized_v_arm_substep2"]["continuity_dv"][
            "operand_bit_exact"] = False

    require(pair.get("status") == "PASS_R215_OMT1_VECTOR_PAIR_REPLAY",
            "round-215 atomic pair is not admitted")
    require(association.get("status") == "MEASURED_R146_BOUNDARY_ASSOCIATION",
            "rank-complete association replay is not admitted")
    require(tuple(registry) == POST_REGISTRY,
            "seven-array association registry moved")
    require(set(association["post_association_rows"]) == set(POST_REGISTRY),
            "post-association field registry moved")
    require(all(row["bit_exact"] for row in association["passivity"].values()),
            "association observer is not passive")
    require(association["post_association_rows"]["v"]["bit_exact"],
            "complete association does not close post-LBC V")
    require(all(row["outside_allowed_cells"] == 0
                for row in association["boundary_scope_rows"].values()),
            "association changed an interior cell")

    split = association["transport_v_operand_split"]
    require(split["rows"]["unmasked_transport_v"]["bit_exact"],
            "NEMO's unmasked zhV statement does not close")
    require(not split["rows"]["production_transport_v"]["bit_exact"]
            and split["rows"]["production_transport_v"]["differing_cells"] == 68,
            "production zhV fold-row census moved")
    require(split["rows"]["masked_replay_vs_production"]["bit_exact"],
            "the extra compact V mask does not reproduce production")
    require(split["production_mismatch_model_vmask_zero_cells"] == 68
            and split["production_mismatch_model_vmask_diff_cells"] == 68,
            "the 68-cell zhV debt is not owned by the compact V mask")
    materialized = association["materialized_v_arm_substep2"]
    require(all(materialized[name]["operand_bit_exact"] for name in CHAIN),
            "unmasked materialized zhV does not close the continuity chain")

    first = association["control_first_non_bit"]
    require(first["substep"] == 2 and first["boundary"] == "continuity_dv"
            and first["differing_cells"] == 68,
            "first downstream non-bit boundary moved")
    return {
        "format": "nemo-testcase-l4-orca2-round216-downstream-v1",
        "status": "HELD_R216_DOWNSTREAM_TRANSPORT",
        "claim_label": "independent rung 0",
        "source_order": [
            "vector_v_update", "seven_array_association", "history_rotation",
            "substep2_midpoint_v", "substep2_zhV", "substep2_continuity_dv",
        ],
        "post_association_v": association["post_association_rows"]["v"],
        "first_downstream_non_bit": first,
        "transport_v": {
            "production": split["rows"]["production_transport_v"],
            "unmasked": split["rows"]["unmasked_transport_v"],
            "masked_replay": split["rows"]["masked_replay_vs_production"],
            "compact_mask_zero_cells":
                split["production_mismatch_model_vmask_zero_cells"],
            "materialized_chain": {
                name: materialized[name] for name in CHAIN
            },
        },
        "compiled_source": {
            "vector_update": (
                "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:669-682"),
            "association": (
                "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:747-756"),
            "transport_continuity": (
                "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:533-560"),
        },
        "predictions": {
            "R216-P1": "CONFIRMED",
            "R216-P2": "REFUTED_FIRST_DOWNSTREAM_ZHV",
            "R216-P3": "UNMEASURED_PREREQUISITE_R216-P2",
            "R216-P4": "UNMEASURED_PREREQUISITE_R216-P2",
            "R216-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
        "open": (
            "Re-score the indivisible vector pair + seven-array association + "
            "unmasked materialized zhV statement; do not land a partial half."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--association-report", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(
            json.loads(args.pair_report.read_text(encoding="utf-8")),
            json.loads(args.association_report.read_text(encoding="utf-8")),
            args.plant,
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    result["inputs"] = {
        "pair_report": str(args.pair_report),
        "pair_sha256": _sha256(args.pair_report),
        "association_report": str(args.association_report),
        "association_sha256": _sha256(args.association_report),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS HELD_R216_DOWNSTREAM_TRANSPORT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
