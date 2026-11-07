#!/usr/bin/env python3
"""Round 34: gate NEMO's carried F-column depth in the live r3f builder."""

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

EXPECTED_REPLAY_MAX = {
    "u_momentum": 3.181628207426175e-09,
    "v_momentum": 2.9702048395431957e-09,
}


def capture(deck_root: Path, record_root: Path, *, plant: bool = False) -> dict:
    hf0 = r32.capture_hf0(deck_root, record_root, plant=plant)
    carried = r32.capture_ldf_replay(
        deck_root, record_root, plant=plant, use_carried_hf0=False)

    replay_exact = all(
        carried[name]["max_abs"] == expected
        for name, expected in EXPECTED_REPLAY_MAX.items()
    )
    at_bar = (
        hf0["hf_0_carried_vs_compiled_construction"]["bit_identical"]
        and hf0["hf_0_carried_vs_reconstructed"]["unequal"] == 14324
        and hf0["hf_0_carried_vs_reconstructed"]["max_abs"]
        == 651.239260895305
        and replay_exact
        and carried["status"] == "HELD"
    )
    return {
        "status": "AT_BAR" if at_bar else "DEBT",
        "claim_label": "given NEMO's entry (kt=2 recorded state)",
        "hf_0": {
            "carried_vs_compiled_construction":
                hf0["hf_0_carried_vs_compiled_construction"],
            "carried_vs_reconstructed":
                hf0["hf_0_carried_vs_reconstructed"],
        },
        "production_replay": {
            "status": carried["status"],
            "u_momentum": carried["u_momentum"],
            "v_momentum": carried["v_momentum"],
        },
        "plant": plant,
        "citations": {
            "hf_0": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-215",
            "r3f": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286",
        },
        "worktree": carried["worktree"],
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
