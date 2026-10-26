#!/usr/bin/env python3
"""Round 39: walk the ranked slow-forcing product and reduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round38_ranked_slow_forcing_gate as round38,
)


def run(deck_root: Path, boundary_root: Path, ranked_root: Path,
        json_out: Path | None, *, plant: bool = False) -> dict[str, object]:
    parent = round38.run(
        deck_root, boundary_root, ranked_root, None, plant=plant)
    arms = parent["one_variable_depth_arms"]
    first = parent["first_non_bit_rank1_disputed_source"]
    round38.require(
        first["boundary"] == "e3"
        and first["differing_cells"] == 1754
        and first["absolute_max"] == 0.027675060932892848
        and arms["parent_depth_vs_record"]["absolute_max"]
        == 7.356587026022005e-18
        and arms["recorded_e3_only_vs_record"]["absolute_max"]
        == 9.423577925168289e-11
        and arms["recorded_rhs_only_vs_record"]["absolute_max"]
        == 4.235164736271502e-22,
        "round-38 cancelling-pair boundary did not reproduce",
    )
    walk = parent["compiled_product_walk"]
    result = {
        "gate": "nemo_testcase_l4_orca2_round39_product_gate",
        "status": "HELD_AT_COMPILED_PRODUCT_OR_REDUCTION",
        "label": "given NEMO's entry",
        "round38_reproduced": True,
        "compiled_product_walk": walk,
        "scientific_plant": parent["scientific_plant"],
        "citations": parent["citations"],
        "single_statement_eligible": False,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--ranked-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.boundary_root, args.ranked_root,
                     args.json_out, plant=args.plant)
    except (round38.GateError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if args.plant else 3


if __name__ == "__main__":
    raise SystemExit(main())
