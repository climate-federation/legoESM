#!/usr/bin/env python3
"""Admit rung 1 and close the Decision-80/83 NEMO-side hierarchy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round18_gate as rung1,
)

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy")
ADMISSION_SHA = {
    1: "f28561d93512774304cef16f6b1031c760156fb0e73385e472ed5da9f57c5b26",
    2: "90ba18953dac0082eceeb1e713d38e4a6f28f3cbbed0e4cf2f356e0e415dd04f",
    3: "591bce28aea5626ea009c674271061ae5f0d4c7959df2ef879bd9b36134fb4b0",
    4: "b337142222c0a0d9fc44539abc55c25c8df09ee5e2155022cbfc14676bb88089",
    5: "0c6543e08bef727b5a01096406f67fd7a41f33ca5f5cedbd36a0ebaf0e97e98a",
    6: "0cc13243c3fb100af19e048e4890b07138081d4d690477a44b828e8580cd34f2",
    7: "b5f80ea6fbf7f51d9871b72be4ee6eb44bf84d609e1d208dbc798e249a0a3b76",
    8: "4a7b0fb3f29cfd337c02a8e00f664b374c811ad86a359746127a6c1d90ccb9b6",
    9: "decbdcf68dc9cae3f5520d0a3a07dd918113e56279b00020774e656afc4fb924",
    10: "b77325d0877eb1e6ea6efea207ad0a04e696abe873ee935ea03c739d7e237e8a",
}
INVENTORY_SHA = {
    1: "5efea04fc4d6c6ebe34f87fb89c1f9ef928e44a12e23b329f1f1f978d3893e77",
    2: "372aaa3f15bf97e36b06a536be699e307599e6f28a234badca4fc5e4f20ae04b",
    3: "051c7a9fb4535cd94521e02f67c599fff5cd16e2730a13d9073f77cf0e88636d",
    4: "422289d037694835103dd6bc63b20c4b200b3b155543fba69c5da7d54ea7f247",
    5: "ccbfb3b085bb05dca476538c79054d2fdf001e6b65396e64f6bfef18b0446459",
    6: "55e969427055d4d438910cbdcdcf4754017a42f569dca4a416953669c696e801",
    7: "cc6f62c35052052d35697949416851fc80dd9bda35be19a04069a2a12311be45",
    8: "169c45f9e447307042d883434eb9c4b62620a7686ec19ba48069849d884ed979",
    9: "0ef284c3fcfc98a7c6815704cba4e986b3c0d51674387b14c77de386d77f3d2c",
    10: "0fa8bb461e6a5a42b5b013e4858bf86e4e130645d20e0c8eb47d1cb8c5fc13ea",
}
STATUS = {k: f"PASS_RUNG{k}_RECORD" for k in range(7, 11)} | {
    k: f"PASS_RUNG{k}_HAVTB0_RECORD" for k in range(1, 7)
}
PLANTS = (
    "none",
    "admission-pin",
    "inventory-pin",
    "status",
    "background-boundary",
    "rung1-payload",
    "superseded-missing",
)


class GateError(RuntimeError):
    """A frozen hierarchy-completion predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_worktree(report: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in report.items() if key != "worktree"}


def evaluate(*, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"invalid plant: {plant}")

    record = ROOT / "rung1/record"
    expected_commit = (record / "producer_commit.txt").read_text().strip()
    fresh = rung1.validate_record(record, expect_commit=expected_commit)
    canonical = json.loads((ROOT / "rung1/rung1_admission.json").read_text())
    if plant == "rung1-payload":
        canonical["frames"]["frames"] = 479
    require(
        _without_worktree(canonical) == fresh,
        "canonical rung-1 admission differs from fresh validation",
    )

    rows = []
    for level in range(1, 11):
        root = ROOT / f"rung{level}"
        admission_path = root / f"rung{level}_admission.json"
        inventory_path = root / "record/SHA256SUMS"
        admission_sha = sha256(admission_path)
        inventory_sha = sha256(inventory_path)
        if plant == "admission-pin" and level == 1:
            admission_sha = "0" * 64
        if plant == "inventory-pin" and level == 10:
            inventory_sha = "0" * 64
        require(admission_sha == ADMISSION_SHA[level], f"rung-{level} admission changed")
        require(inventory_sha == INVENTORY_SHA[level], f"rung-{level} inventory changed")

        admission = json.loads(admission_path.read_text())
        expected_status = STATUS[level]
        if plant == "status" and level == 5:
            expected_status = "PLANTED_WRONG_STATUS"
        require(admission.get("status") == expected_status, f"rung-{level} status changed")
        require(admission.get("claim_label") == "independent", f"rung-{level} label changed")
        require(admission.get("rung") == level, f"rung-{level} identity changed")
        require(admission.get("frames", {}).get("frames") == 480, f"rung-{level} frames changed")
        require(
            admission.get("month_products", {}).get("status") == "FINITE",
            f"rung-{level} month products changed",
        )
        require(
            admission.get("sha_inventory", {}).get("status") == "COMPLETE",
            f"rung-{level} inventory status changed",
        )

        exact_deck = root / "record/namelist_cfg.deck"
        if not exact_deck.exists():
            exact_deck = root / "record/namelist_cfg"
        values = rung1.legacy.namelist_values(exact_deck)
        expected_havtb = "0" if level <= 6 else "1"
        if plant == "background-boundary" and level == 6:
            expected_havtb = "1"
        actual_havtb = rung1.legacy._token(values["namzdf.nn_havtb"])
        require(actual_havtb == expected_havtb, f"rung-{level} nn_havtb changed")

        if level <= 6:
            superseded = root / "record_havtb1_superseded"
            superseded_admission = root / f"rung{level}_admission_havtb1_superseded.json"
            superseded_ok = superseded.is_dir() and superseded_admission.is_file()
            if plant == "superseded-missing" and level == 1:
                superseded_ok = False
            require(superseded_ok, f"rung-{level} superseded evidence is absent")

        rows.append(
            {
                "rung": level,
                "status": admission["status"],
                "frames": admission["frames"]["frames"],
                "nn_havtb": int(actual_havtb),
                "regular_files": admission["sha_inventory"]["regular_files"],
            }
        )

    return {
        "status": "PASS_ORCA2_NEMO_HIERARCHY_RUNGS_1_10",
        "claim_label": "independent",
        "rung1_fresh_validation": fresh["status"],
        "rung1_to_rung0_differences": fresh["main_rung0_semantic_differences"],
        "rungs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(plant=args.plant)
        report["worktree"] = worktree_stamp()
    except (RuntimeError, KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
