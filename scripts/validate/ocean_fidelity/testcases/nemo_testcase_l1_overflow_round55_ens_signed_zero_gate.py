#!/usr/bin/env python3
"""Walk the admitted OVERFLOW ENS signed-zero addition in source order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nemo_testcase_l1_overflow_round53_ens_operands_gate as r53_gate
import numpy as np
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l1_overflow_round50_pair_gate import GateError, read_record, require
from nemo_testcase_l1_overflow_round51_pair_gate import _nemo_owned
from nemo_testcase_phase3_trajectory_gate import expected_masks

RECORD_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round53/acquisition/oracle_overflow_ens_operands"
)
RECORD = "oracle_r53_ens_kt00000003_s2.bin"
PARENT = "oracle_r50_momentum_kt00000003_s2.bin"
PRODUCER_COMMIT = "39fedb35cbc4158338750449fb4a5d67f07102c6"
EXPECTED_FIRST = (1, 31, 0)
EXPECTED_CHANGED = 16_135
COMPILED = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90"
)


def _bits(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    require(value.dtype == np.dtype("float64"), "operand is not fp64")
    return value.view(np.uint64)


def _describe(value: np.float64) -> dict:
    scalar = np.asarray(value, dtype=np.float64)
    return {
        "value": float(scalar),
        "signbit": bool(np.signbit(scalar)),
        "bits_hex": f"0x{int(scalar.view(np.uint64)):016x}",
    }


def _analyze(
    operands: dict[str, np.ndarray],
    parent: dict[str, np.ndarray],
    active_u: np.ndarray,
    *,
    expected_first: tuple[int, int, int],
    expected_changed: int,
    plant: str | None,
) -> dict:
    names = (
        "zwz_prediv",
        "zwz_postdiv",
        "zuav",
        "zwz_pair_u",
        "product_u",
        "rhs_before_u",
        "rhs_after_u",
    )
    values = {name: np.asarray(operands[name], dtype=np.float64) for name in names}
    active = np.asarray(active_u, dtype=bool)
    shape = values["rhs_before_u"].shape
    require(active.shape == shape, f"active-u shape {active.shape} != {shape}")
    require(all(value.shape == shape for value in values.values()), "ENS operand shape drift")

    before_parent = np.asarray(parent["after_hpg_u"], dtype=np.float64)
    after_parent = np.asarray(parent["after_vor_u"], dtype=np.float64)
    require(before_parent.shape == after_parent.shape == shape, "round-50 endpoint shape drift")
    before_endpoint_unequal = active & (_bits(values["rhs_before_u"]) != _bits(before_parent))
    after_endpoint_unequal = active & (_bits(values["rhs_after_u"]) != _bits(after_parent))

    changed = active & (_bits(values["rhs_before_u"]) != _bits(values["rhs_after_u"]))
    require(changed.any(), "record contains no active-u accumulator transition")
    first = tuple(int(i) for i in np.argwhere(changed)[0])
    require(first == expected_first, f"first transition {first} != registered {expected_first}")

    product = values["product_u"].copy()
    after = values["rhs_after_u"].copy()
    if plant == "product_sign":
        require(
            product[first] == 0.0 and not np.signbit(product[first]),
            "product-sign plant target is not positive zero",
        )
        product[first] = np.copysign(np.float64(0.0), np.float64(-1.0))
    elif plant == "endpoint":
        after[first] = np.nextafter(after[first], np.float64(np.inf))
        require(after[first] != values["rhs_after_u"][first], "endpoint plant did not move")

    recomputed = np.add(values["rhs_before_u"], product)
    addition_unequal = active & (_bits(recomputed) != _bits(after))
    chain = (
        active
        & (values["rhs_before_u"] == 0.0)
        & np.signbit(values["rhs_before_u"])
        & (product == 0.0)
        & ~np.signbit(product)
        & (after == 0.0)
        & ~np.signbit(after)
    )
    trace = {name: _describe(values[name][first]) for name in names}
    trace["recomputed_addition"] = _describe(recomputed[first])
    trace["scored_after"] = _describe(after[first])

    baseline = plant is None
    p2 = (
        trace["rhs_before_u"]["bits_hex"] == "0x8000000000000000"
        and trace["product_u"]["bits_hex"] == "0x0000000000000000"
        and trace["rhs_after_u"]["bits_hex"] == "0x0000000000000000"
    )
    p3 = (
        int(np.count_nonzero(changed)) == expected_changed
        and int(np.count_nonzero(chain)) == expected_changed
        and not before_endpoint_unequal.any()
        and not after_endpoint_unequal.any()
        and not addition_unequal.any()
    )
    if baseline:
        status = "AT_BAR" if p2 and p3 else "DEBT"
    else:
        require(addition_unequal.any(), f"{plant} plant stayed green")
        status = "PLANTED_REFUSAL"

    return {
        "status": status,
        "plant": plant,
        "first_local_index": list(first),
        "active_u": int(np.count_nonzero(active)),
        "changed_accumulator_bits": int(np.count_nonzero(changed)),
        "negative_zero_plus_positive_zero_to_positive_zero": int(np.count_nonzero(chain)),
        "before_endpoint_unequal": int(np.count_nonzero(before_endpoint_unequal)),
        "after_endpoint_unequal": int(np.count_nonzero(after_endpoint_unequal)),
        "addition_unequal": int(np.count_nonzero(addition_unequal)),
        "first_trace": trace,
        "R55-P2": "CONFIRMED" if p2 else "REFUTED",
        "R55-P3": "CONFIRMED" if p3 else "REFUTED",
    }


def run(root: Path, expect_commit: str, plant: str | None) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "analysis worktree is dirty")
    require(
        stamp["commit"] == expect_commit, f"analysis commit {stamp['commit']} != {expect_commit}"
    )
    source = COMPILED.read_text()
    statement = "pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav * ( zwz(ji  ,jj-1) + zwz(ji,jj) )"
    require(statement in source, "compiled ENS accumulator statement drift")

    record = root / RECORD
    parent_path = root / PARENT
    admission = r53_gate.admit(record, parent_path, PRODUCER_COMMIT, None)
    require(admission["status"] == "AT_BAR", "round-53 record is not admitted")
    operands_raw = r53_gate.read_record(record)
    parent_raw = read_record(parent_path, "momentum", 2)
    operands = {name: _nemo_owned(value) for name, value in operands_raw["fields"].items()}
    parent = {name: _nemo_owned(value) for name, value in parent_raw["fields"].items()}
    card = build_nemo_testcase_card("OVERFLOW-zps")
    active_u = np.asarray(expected_masks(card)["u"], dtype=bool)
    analysis = _analyze(
        operands,
        parent,
        active_u,
        expected_first=EXPECTED_FIRST,
        expected_changed=EXPECTED_CHANGED,
        plant=plant,
    )
    origin = operands_raw["header"]["origin"]
    analysis["first_fortran_index"] = [
        origin[0] + EXPECTED_FIRST[1],
        origin[1] + EXPECTED_FIRST[0],
        EXPECTED_FIRST[2] + 1,
    ]
    analysis.update(
        {
            "format": "nemo-testcase-l1-overflow-round55-ens-sign-v1",
            "case": "OVERFLOW-zps",
            "kt": 3,
            "stage": 2,
            "claim_label": "given NEMO's recorded operands",
            "worktree": stamp,
            "record_root": str(root),
            "record_sha256": operands_raw["sha256"],
            "parent_sha256": parent_raw["sha256"],
            "record_admission": admission["status"],
            "compiled_source": str(COMPILED),
            "compiled_statement": statement,
            "R55-P1": "CONFIRMED",
            "R55-P4": "CONFIRMED" if plant else "NOT_RUN",
            "R55-P5": "UNMEASURED",
        }
    )
    return analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("product_sign", "endpoint"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.record_dir, args.expect_commit, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if args.plant else (0 if report["status"] == "AT_BAR" else 1)
    except (GateError, r53_gate.GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
