#!/usr/bin/env python3
"""Round 35: gate NEMO's stored-reciprocal arithmetic in live ``r3f``."""

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

EXPECTED_PARENT_MAX = {
    "u_momentum": 3.181628207426175e-09,
    "v_momentum": 2.9702048395431957e-09,
}


def capture(deck_root: Path, record_root: Path, *, plant: bool = False) -> dict:
    parent = r32.capture_ldf_replay(deck_root, record_root)
    arm = r32.capture_ldf_replay(
        deck_root,
        record_root,
        plant=plant,
        r3f_reciprocal_order=True,
    )
    parent_reproduced = all(
        parent[name]["max_abs"] == expected
        for name, expected in EXPECTED_PARENT_MAX.items()
    )
    arm_exact = all(
        arm[name]["bit_identical"]
        for name in ("u_momentum", "v_momentum")
    )
    at_bar = parent_reproduced and parent["status"] == "HELD" and arm_exact
    return {
        "status": "AT_BAR" if at_bar else "DEBT",
        "claim_label": "given NEMO's entry (kt=2 recorded state)",
        "parent": {
            "status": parent["status"],
            "u_momentum": parent["u_momentum"],
            "v_momentum": parent["v_momentum"],
        },
        "stored_reciprocal_arm": {
            "status": arm["status"],
            "u_momentum": arm["u_momentum"],
            "v_momentum": arm["v_momentum"],
        },
        "plant": plant,
        "citations": {
            "stored_reciprocal": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "domhgr.f90:154-156"
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
        "worktree": arm["worktree"],
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
