#!/usr/bin/env python3
"""Admit the additions-only OMT-4 kt=1..10 frames and month boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round220_omt3_frame_record_gate as omt3_record,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round222_omt4_deck_gate as omt4_deck,
)


PLANTS = omt3_record.PLANTS
STEPS = omt3_record.STEPS
STAGES = omt3_record.STAGES
RANKS = omt3_record.RANKS
FIELDS = omt3_record.FIELDS


class GateError(RuntimeError):
    """The OMT-4 record violated a frozen admission predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def preflight() -> dict:
    inherited = omt3_record.preflight()
    return {
        **inherited,
        "format": "nemo-testcase-l4-orca2-round222-omt4-frame-preflight-v1",
        "status": "PASS_R222_OMT4_FRAME_PREFLIGHT",
    }


def admit(canonical: Path, calibration: Path, twin_a: Path, twin_b: Path,
          month: Path, plant: str = "none") -> dict:
    result = omt3_record.admit(
        canonical, calibration, twin_a, twin_b, month, plant,
        deck_gate=omt4_deck,
    )
    return {
        **result,
        "format": "nemo-testcase-l4-orca2-round222-omt4-record-v1",
        "status": "PASS_R222_OMT4_ENTRY_STAGE_AND_MONTH_RECORD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--month", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            require(args.plant == "none", "preflight does not accept a plant")
            result = preflight()
        else:
            require(all((args.candidate, args.calibration, args.twin_a,
                         args.twin_b, args.month)),
                    "admission requires candidate, calibration, twins and month")
            result = admit(
                args.candidate, args.calibration, args.twin_a, args.twin_b,
                args.month, args.plant,
            )
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt3_record.GateError,
            omt3_record.omt1_record.GateError,
            omt3_record.omt1_record.frame_gate.GateError,
            omt3_record.protocol.GateError, omt4_deck.GateError,
            OSError, UnicodeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
