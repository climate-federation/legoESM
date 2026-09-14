#!/usr/bin/env python3
"""Fail closed when the inherited stage-one record predates its barotropic target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
REQUIRED_SITE_FIELDS = {
    "baro_raw_u", "baro_raw_v", "baro_target_u", "baro_target_v",
    "baro_zub", "baro_zvb", "baro_final_u", "baro_final_v",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(args) -> dict[str, object]:
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-90 record audit worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-90 commit stamp mismatch")
    record = round46.read_stage(
        args.round46_root / "oracle_momstage_kt00000002_s1.bin")
    require(record["header"]["kt"] == 2 and record["header"]["stage"] == 1,
            "wrong inherited stage record")
    arrays = record["arrays"]
    wet_u = round46._owned3(arrays["umask"])[..., 0] > 0.5
    wet_v = round46._owned3(arrays["vmask"])[..., 0] > 0.5
    target_u = round46._owned2(arrays["uu_b_Kaa"]).copy()
    target_v = round46._owned2(arrays["vv_b_Kaa"]).copy()
    observed_u = int(np.count_nonzero(target_u[wet_u]))
    observed_v = int(np.count_nonzero(target_v[wet_v]))
    if args.plant:
        at = tuple(np.argwhere(wet_u)[0])
        target_u[at] = np.nextafter(target_u[at], np.inf)
        require(int(np.count_nonzero(target_u[wet_u])) != observed_u,
                "zero-target ULP plant was invisible")
        status = "PLANT_FIRED"
    else:
        require(observed_u == 0 and observed_v == 0,
                "inherited target is no longer the pre-solve zero snapshot")
        status = "STOPPED_FOR_RECORD"
    missing = sorted(REQUIRED_SITE_FIELDS - arrays.keys())
    require(missing, "inherited record unexpectedly gained correction-site fields")
    return {
        "format": "nemo-testcase-l2-gyre-round90-record-audit-v2",
        "status": status,
        "worktree": stamp,
        "record_header": record["header"],
        "wet_nonzero_entry_targets": {"u": observed_u, "v": observed_v},
        "missing_correction_site_fields": missing,
        "compiled_call_order": {
            "record_opens": "stp2d.f90:141-145",
            "external_target_is_interpolated": "stprk3_stg.f90:140-170",
            "target_is_consumed": "stprk3_stg.f90:734-760",
        },
        "record_request": (
            "Capture raw Kaa, current uu_b/vv_b(Kaa), zub/zvb, reference "
            "weights/reciprocals/masks, and final Kaa at the correction site."),
        "plant": args.plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--round46-root", type=Path,
                        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND90_BARO_RECORD", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
