#!/usr/bin/env python3
"""Round 37: gate the measured native-area/stored-reciprocal ``r3f`` pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round32_readout_gate as r32,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round36_r3f_area_gate as r36,
)

EXPECTED_PARENT_MAX = r36.EXPECTED_PARENT_MAX
EXPECTED_ISOLATED_MAX = float.fromhex("0x1p-73")
EXPECTED_ISOLATED_UNEQUAL = 24


def capture(deck_root: Path, record_root: Path, *, plant: bool = False) -> dict:
    parent = r32.capture_ldf_replay(
        deck_root, record_root, use_shifted_r3f_area=True)
    reciprocal_only = r32.capture_ldf_replay(
        deck_root, record_root, r3f_reciprocal_order=True,
        use_shifted_r3f_area=True)
    area_only = r32.capture_ldf_replay(
        deck_root, record_root, use_native_r3f_area=True)
    pair = r32.capture_ldf_replay(deck_root, record_root, plant=plant)

    parent_reproduced = all(
        parent[name]["max_abs"] == expected
        for name, expected in EXPECTED_PARENT_MAX.items()
    )
    reciprocal_inert = all(
        reciprocal_only[name]["unequal"] == parent[name]["unequal"]
        and reciprocal_only[name]["max_abs"] == parent[name]["max_abs"]
        for name in ("u_momentum", "v_momentum")
    )
    area_residual_reproduced = all(
        area_only[name]["unequal"] == EXPECTED_ISOLATED_UNEQUAL
        and area_only[name]["max_abs"] == EXPECTED_ISOLATED_MAX
        for name in ("u_momentum", "v_momentum")
    )
    pair_exact = all(
        pair[name]["bit_identical"]
        for name in ("u_momentum", "v_momentum")
    )
    at_bar = (
        parent_reproduced
        and parent["status"] == "HELD"
        and reciprocal_inert
        and area_residual_reproduced
        and pair_exact
        and pair["status"] == "PASS"
    )
    return {
        "status": "AT_BAR" if at_bar else "DEBT",
        "claim_label": "given NEMO's entry (kt=2 recorded state)",
        "parent": parent,
        "stored_reciprocal_only": reciprocal_only,
        "native_f_area_only": area_only,
        "measured_pair": pair,
        "plant": plant,
        "citations": {
            "native_f_area": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "domhgr.f90:155-157"
            ),
            "r3f": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "domqco.f90:273-286"
            ),
            "consumer": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "dynldf_lev.f90:123"
            ),
        },
        "worktree": pair["worktree"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = capture(args.deck_root, args.record_root, plant=args.plant)
    except (r32.GateError, OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.plant:
        return 1 if result["status"] != "AT_BAR" else 3
    return 0 if result["status"] == "AT_BAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
