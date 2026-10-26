#!/usr/bin/env python3
"""Round-73 source-order walk of GYRE kt=2 stage-1 transport inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round54_tracer_decomposition as round54  # noqa: E402
import nemo_testcase_l2_gyre_round67_ldf_order as round67  # noqa: E402
import nemo_testcase_l2_gyre_round72_stage1_transport_gate as record_gate  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
RECORD_NAME = "oracle_rkstage1_transport_operands_kt00000002.bin"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def to_lego(array: np.ndarray, nlev: int | None = None) -> np.ndarray:
    """Map the owned NEMO i,j[,k] record layout to legoESM j,i[,k]."""
    value = np.ascontiguousarray(np.asarray(array).swapaxes(0, 1))
    return value if nlev is None else value[..., :nlev]


def comparison(candidate, oracle, active) -> dict:
    candidate = np.asarray(candidate)[active]
    oracle = np.asarray(oracle)[active]
    require(candidate.shape == oracle.shape, "comparison shapes differ")
    require(np.all(np.isfinite(candidate)), "candidate contains non-finite values")
    require(np.all(np.isfinite(oracle)), "oracle contains non-finite values")
    delta = candidate - oracle
    return {
        "bit_exact": bool(np.array_equal(
            candidate.view(np.uint64), oracle.view(np.uint64))),
        "differing_cells": int(np.count_nonzero(
            candidate.view(np.uint64) != oracle.view(np.uint64))),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "reference_max_abs": float(np.max(np.abs(oracle), initial=0.0)),
    }


def classify_first_u(rows: dict[str, dict]) -> str | None:
    """Return the first unequal compiled U-path boundary exposed this round."""
    for name in ("un_adv",):
        if not rows[name]["bit_exact"]:
            return name
    return None


def measure(args) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "round-73 producer worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "round-73 commit stamp mismatch")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")

    admission = json.loads(args.admission.read_text())
    require(admission["verdict"] == "PASS", "round-72 twin admission failed")
    require((admission["byte_identical_records"],
             len(admission["classified_changed_records"]),
             admission["admitted_difference_count"]) == (47, 20, 132),
            "round-72 admission census changed")
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(producer.lower() == args.expect_record_commit.lower(),
            "round-72 producer commit mismatch")
    record_path = args.record_root / RECORD_NAME
    stamp_parts = record_path.with_name(RECORD_NAME + ".stamp").read_text().split()
    require(stamp_parts == [record_gate.sha256(record_path), producer, RECORD_NAME],
            "round-72 record stamp mismatch")
    fields = record_gate._read(record_path)
    replay_rows = record_gate.validate(record_path)

    base, context = round72._capture_seeded_context(args)
    card, seeded, freshwater, surface, round66_trace = context
    trace, _ = round72._live_trace(card, seeded, freshwater, surface)
    identity = round67.pytree_exact_census(trace.state_after, round66_trace.state_after)
    require(identity["cells_unequal"] == 0,
            "WRITE-only live trace changed ordinary model state")

    masks = gate.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    live_un_adv = np.asarray(trace.barotropic_targets[2])[:, 1:]
    live_vn_adv = np.asarray(trace.barotropic_targets[3])[1:, :]
    oracle_un_adv = to_lego(fields["un_adv"])
    oracle_vn_adv = to_lego(fields["vn_adv"])
    active_u = masks["u"][..., 0]
    active_v = masks["v"][..., 0]
    require(np.any(live_un_adv[active_u] != 0.0),
            "null plant requires a nonzero live un_adv")
    if args.plant_null_un_adv:
        oracle_un_adv = np.array(live_un_adv, copy=True)

    u_rows = {"un_adv": comparison(live_un_adv, oracle_un_adv, active_u)}
    v_rows = {"vn_adv": comparison(live_vn_adv, oracle_vn_adv, active_v)}
    first_u = classify_first_u(u_rows)

    live_zfu = np.asarray(trace.stage_geometry[0][7])[:, 1:, :nlev]
    live_zfv = np.asarray(trace.stage_geometry[0][8])[1:, :, :nlev]
    zf_rows = {
        "zFu": round54.field_stats(
            live_zfu, to_lego(fields["zFu"], nlev), masks["u"]),
        "zFv": round54.field_stats(
            live_zfv, to_lego(fields["zFv"], nlev), masks["v"]),
    }
    round72_reproduced = bool(
        zf_rows["zFu"]["cells_unequal"] == 17400
        and zf_rows["zFu"]["max_abs"] == 0.8916110997497526
        and zf_rows["zFv"]["cells_unequal"] == 17100
        and zf_rows["zFv"]["max_abs"] == 0.7485346468365606)
    record_replay_exact = all(row["bit_exact"] for row in replay_rows.values())
    dtype_rows = {
        "record": str(np.asarray(fields["un_adv"]).dtype),
        "live_transport_average": str(live_un_adv.dtype),
        "live_geometry": str(live_zfu.dtype),
        "live_state": str(np.asarray(trace.stage_states[0][0]).dtype),
    }
    dtype_exact = all(value == "float64" for value in dtype_rows.values())
    prediction_confirmed = bool(
        record_replay_exact and identity["cells_unequal"] == 0
        and dtype_exact and round72_reproduced
        and first_u == "un_adv" and not v_rows["vn_adv"]["bit_exact"])
    return {
        "format": "nemo-testcase-l2-gyre-round73-stage1-transport-v1",
        "status": "CONFIRMED" if prediction_confirmed else "REFUTED",
        "worktree": stamp,
        "record_producer": producer,
        "record_sha256": record_gate.sha256(record_path),
        "record_admission": {"exact": 47, "total": 67, "changed": 20,
                             "admitted": 132},
        "record_replay": replay_rows,
        "dtype": dtype_rows,
        "ordinary_state_identity": identity,
        "u_rows": u_rows,
        "v_rows": v_rows,
        "first_non_bit_u": first_u,
        "downstream_u_rows": "UNREACHED_AFTER_FIRST_NON_BIT_INPUT",
        "zf_reproduction": zf_rows,
        "round72_zf_reproduced": round72_reproduced,
        "base_round66_status": base["status"],
        "plant_null_un_adv": args.plant_null_un_adv,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--record-root", type=Path, default=(
        ROOT / "round72/oracle_stage1_transport"))
    parser.add_argument("--admission", type=Path, default=(
        ROOT / "round73/round73_admission.json"))
    parser.add_argument("--plant-null-un-adv", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    row = report["u_rows"]["un_adv"]
    print(
        f"ROUND73 STAGE1 TRANSPORT {report['status']}: "
        f"first_u={report['first_non_bit_u']} "
        f"un_adv_max={row['absolute_max']:.12e}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
