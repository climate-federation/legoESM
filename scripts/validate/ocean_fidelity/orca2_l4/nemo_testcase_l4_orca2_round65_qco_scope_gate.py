#!/usr/bin/env python3
"""Re-scope the returned QCO boundary to ORCA2's executing program."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp


SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90"
)
CITATIONS = {
    "momentum_stage_selector": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
        "stprk3_stg.f90:467-480"
    ),
    "tracer_qco_assignment": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
        "stprk3_stg.f90:670-681"
    ),
}
EXPECTED = {"T": (57141, 228641), "S": (57169, 228641)}


class GateError(RuntimeError):
    """A fail-closed round-65 condition was not met."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled_branches(path: Path) -> dict[str, object]:
    lines = path.read_text().splitlines()
    require(len(lines) >= 480, "compiled stage source is truncated")
    selected = lines[466:480]
    joined = "\n".join(selected)
    require("IF( ln_dynadv_vec .OR. lk_linssh ) THEN" in joined,
            "vector-stage predicate changed")
    require("uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt *" in joined,
            "vector U assignment changed")
    require("vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt *" in joined,
            "vector V assignment changed")
    require("ELSE" in joined and "1._wp + r3u(ji,jj,Kbb)" in joined,
            "thickness-weighted alternative changed")
    return {
        "source": str(path),
        "sha256": sha256(path),
        "vector_lines": [467, 470],
        "thickness_weighted_lines": [472, 480],
    }


def classify(scope: dict, qco: dict, *, source: Path = SOURCE,
             plant: str = "none") -> dict[str, object]:
    """Classify the boundary without duplicating either numerical scorer."""
    scope = json.loads(json.dumps(scope))
    qco = json.loads(json.dumps(qco))
    if plant == "scope":
        scope["card_scope"]["ORCA2-zps"][0] = "flux_form"
    elif plant == "qco-count":
        qco["replay_rows"]["T"]["production_stage1"]["unequal"] += 1

    require(scope.get("status") == "PASS", "card-scope probe did not pass")
    require(scope.get("disagreements") == [], "card-scope disagreements exist")
    card_scope = scope.get("card_scope", {})
    require(card_scope.get("ORCA2-zps", [None])[0] == "vector_invariant",
            "ORCA2 does not select vector-invariant momentum")
    require(card_scope.get("GYRE-zco", [None])[0] == "vector_invariant",
            "GYRE does not select vector-invariant momentum")
    require(card_scope.get("OVERFLOW-zps") == ["flux_form", "nemo_up3"],
            "OVERFLOW no longer selects flux-form UP3")
    require(card_scope.get("LOCK-zco") == ["flux_form", "nemo_up3"],
            "LOCK no longer selects flux-form UP3")

    branches = _compiled_branches(source)
    require(qco.get("status") == "PASS_MEASUREMENT_COMPLETE",
            "QCO scorer did not complete")
    require(qco.get("claim_label") == "GIVEN_NEMO_ENTRY_DECISION52",
            "QCO claim label changed")
    require(qco.get("execution", {}).get("dtype") == "float64",
            "QCO scorer is not fp64")

    rows = {}
    for tracer, (expected_unequal, expected_count) in EXPECTED.items():
        replay = qco["replay_rows"][tracer]
        production = replay["production_stage1"]
        require((production["unequal"], production["count"])
                == (expected_unequal, expected_count),
                f"{tracer} production row changed")
        for exact_name in ("numpy_literal", "jax_source_ordered"):
            exact = replay[exact_name]
            require(exact["bit_exact"] and exact["unequal"] == 0
                    and exact["count"] == expected_count,
                    f"{tracer} {exact_name} is not bit-exact")
        fused = replay["jax_fused"]
        require((fused["unequal"], fused["count"])
                == (expected_unequal, expected_count),
                f"{tracer} fused control did not retain the debt")
        rows[tracer] = {
            "production_unequal": expected_unequal,
            "count": expected_count,
            "literal_unequal": 0,
            "source_ordered_unequal": 0,
            "fused_control_unequal": expected_unequal,
        }

    return {
        "format": "nemo-testcase-l4-orca2-round65-qco-scope-v1",
        "status": "PASS_QCO_SCOPE",
        "routing_label": "independent",
        "numerical_label": "given NEMO's recorded entry",
        "compiled_branches": branches,
        "card_scope": card_scope,
        "round60_boundary_disposition": {
            "thickness_weighted_momentum_update": "NOT_EXECUTED_BY_ORCA2",
            "vector_velocity_update": "EXECUTED_BY_ORCA2",
        },
        "first_actual_orca2_nonbit_statement": {
            "statement": "tracer_qco_rk_assignment",
            "citation": CITATIONS["tracer_qco_assignment"],
            "rows": rows,
        },
        "landing": "HELD_BY_ROUND48_OVERFLOW_NONREGRESSION",
        "citations": CITATIONS,
    }


def run(scope_path: Path, qco_path: Path, *, source: Path = SOURCE,
        plant: str = "none") -> dict[str, object]:
    result = classify(
        json.loads(scope_path.read_text()),
        json.loads(qco_path.read_text()),
        source=source,
        plant=plant,
    )
    result["inputs"] = {
        "scope": {"path": str(scope_path), "sha256": sha256(scope_path)},
        "qco": {"path": str(qco_path), "sha256": sha256(qco_path)},
    }
    result["worktree"] = worktree_stamp()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-json", type=Path, required=True)
    parser.add_argument("--qco-json", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("none", "scope", "qco-count"),
                        default="none")
    args = parser.parse_args()
    try:
        report = run(args.scope_json, args.qco_json, source=args.source,
                     plant=args.plant)
    except (GateError, KeyError, OSError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_QCO_SCOPE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
