#!/usr/bin/env python3
"""Round-65 exact association score for the admitted kt=2 TKE RHS row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_round54_tracer_decomposition as round54
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean.fidelity.provenance import worktree_stamp


def run(args) -> dict:
    stamp = worktree_stamp()
    round54.require(stamp["clean"], "producer worktree is dirty")
    round54.require(stamp["commit"].lower() == args.expect_commit.lower(),
                    "producer commit mismatch")
    producer = args.producer_commit.read_text().strip().lower()
    round54.require(producer == args.expect_record_commit.lower(),
                    "record producer mismatch")
    round54._verify_r63_stamp(args.record, args.record_stamp, producer)
    parsed = round54._read_r63_stream(args.record, kind="tke",
                                      plant=args.plant)
    calibration = round54._calibrate_r63_tke(parsed["arrays"])
    a = {
        name: jnp.asarray(np.ascontiguousarray(value.transpose(1, 0, 2)))
        if np.ndim(value) == 3 else jnp.asarray(value)
        for name, value in parsed["arrays"].items()
    }

    @jax.jit
    def score_values():
        strat = a["avt"] * a["rn2"]
        diss = a["zfact3"] * a["dissl"] * a["en_rhs_entry"]
        current = a["en_rhs_entry"] + a["rn_Dt"] * (
            a["shear"] - strat + diss) * a["wmask"]
        sr = nemo_source_round
        strat_literal = sr(a["avt"] * a["rn2"])
        diss_literal = sr(sr(a["zfact3"] * a["dissl"])
                          * a["en_rhs_entry"])
        bracket = sr(sr(a["shear"] - strat_literal) + diss_literal)
        literal = sr(a["en_rhs_entry"] + sr(
            sr(a["rn_Dt"] * bracket) * a["wmask"]))
        return strat, diss, current, strat_literal, diss_literal, literal

    values = jax.device_get(score_values())
    wet = np.asarray(a["wmask"]) > 0.5
    rows = {}
    for label, current_i, literal_i, oracle_name in (
        ("stratification", 0, 3, "strat_product"),
        ("dissipation", 1, 4, "diss_product"),
        ("rhs", 2, 5, "en_rhs_post"),
    ):
        oracle = np.asarray(a[oracle_name])
        rows[label] = {
            "current": round54.field_stats(values[current_i], oracle, wet),
            "compiled": round54.field_stats(values[literal_i], oracle, wet),
        }
    return {
        "format": "nemo-testcase-l2-gyre-round65-tke-rhs-v1",
        "worktree": stamp,
        "record_producer": producer,
        "calibration": calibration,
        "rows": rows,
        "statement": "zdftke.f90:428-431; dissipation subterm at :430",
        "plant": args.plant,
        "status": "PASS",
    }


def main(argv=None) -> int:
    root = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/oracle_krhs_split")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--record", type=Path,
                        default=root / "oracle_tke_rhs_split_kt00000002.bin")
    parser.add_argument("--record-stamp", type=Path,
                        default=root / "oracle_tke_rhs_split_kt00000002.bin.stamp")
    parser.add_argument("--producer-commit", type=Path,
                        default=root / "producer_commit.txt")
    parser.add_argument("--plant", choices=("ulp", "truncation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        print(f"GATE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"ROUND65 TKE RHS PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
