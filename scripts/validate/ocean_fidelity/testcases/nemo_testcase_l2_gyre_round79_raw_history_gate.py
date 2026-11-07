#!/usr/bin/env python3
"""Gate Decision 37's absolute AB3/AM4 history at the kt=2 U walk."""

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

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round76_uamid_gate as record_gate  # noqa: E402
import nemo_testcase_l2_gyre_round78_uamid_walk as round78  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(args) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64 libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-79 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "round-79 measurement commit mismatch")

    admission = json.loads(args.admission.read_text())
    require(admission["verdict"] == "PASS", "round-77 admission failed")
    require((admission["byte_identical_records"],
             len(admission["classified_changed_records"]),
             admission["admitted_difference_count"]) == (45, 24, 281),
            "round-77 inherited-record census changed")
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(producer.lower() == args.expect_record_commit.lower(),
            "round-77 producer commit mismatch")
    record_path = args.record_root / record_gate.RECORD
    require(round78._sha256(record_path) == round78.RECORD_SHA256,
            "round-77 U-midpoint digest changed")
    require(record_path.with_name(record_gate.RECORD + ".stamp").read_text().split()
            == [round78.RECORD_SHA256, producer, record_gate.RECORD],
            "round-77 U-midpoint stamp mismatch")
    fields = record_gate.read_record(record_path)
    replay = record_gate.validate_fields(fields)

    base, card, _seeded, trace = round78._context(args)
    masks = gate.expected_masks(card)
    active_u = masks["u"][..., 0]
    substeps = trace.substeps
    live = {
        "un_e": gate._trace_native(substeps["u_entry"], "u_entry"),
        "ub_e": gate._trace_native(substeps["u_history_b"], "u_history_b"),
        "ubb_e": gate._trace_native(substeps["u_history_bb"], "u_history_bb"),
        "ua_e": gate._trace_native(substeps["u_mid"], "u_mid"),
    }
    oracle = {name: np.array(fields[name], copy=True)
              for name in record_gate.FIELDS}
    if args.plant == "history-ulp":
        locations = np.argwhere(np.broadcast_to(active_u, oracle["ubb_e"].shape))
        require(locations.size > 0, "history plant requires a wet cell")
        location = tuple(locations[0])
        oracle["ubb_e"][location] = np.nextafter(
            oracle["ubb_e"][location], np.inf)

    live_coefficients = np.stack(
        [np.asarray(substeps[f"mid_weight_{i}"], dtype=np.float64)
         for i in (1, 2, 3)], axis=-1)
    scalar_mask = np.ones((), dtype=bool)
    rows = []
    for index in range(record_gate.N_CYCLE):
        rows.append({
            "substep": index + 1,
            "coefficient_1": round78.comparison(
                live_coefficients[index, 0], fields["coefficients"][index, 0],
                scalar_mask),
            "coefficient_2": round78.comparison(
                live_coefficients[index, 1], fields["coefficients"][index, 1],
                scalar_mask),
            "coefficient_3": round78.comparison(
                live_coefficients[index, 2], fields["coefficients"][index, 2],
                scalar_mask),
            **{name: round78.comparison(live[name][index], oracle[name][index],
                                        active_u)
               for name in record_gate.FIELDS},
        })
    first = round78.first_live_boundary(rows)

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (  # noqa: PLC0415
        nemo_literal_midpoint_extrapolation,
    )
    shared = np.asarray(jax.device_get(jax.jit(jax.vmap(
        nemo_literal_midpoint_extrapolation, in_axes=(0, 0, 0, 0)))(
            jnp.asarray(fields["coefficients"]), jnp.asarray(fields["un_e"]),
            jnp.asarray(fields["ub_e"]), jnp.asarray(fields["ubb_e"]))))
    shared_rows = [round78.comparison(shared[i], fields["ua_e"][i], active_u)
                   for i in range(record_gate.N_CYCLE)]
    shared_exact = all(row["bit_exact"] for row in shared_rows)
    first_row_exact = all(rows[0][name]["bit_exact"]
                          for name in round78.SOURCE_ORDER)
    confirmed = bool(args.plant == "none" and first_row_exact
                     and first is not None and first["substep"] == 2
                     and first["boundary"] == "un_e" and shared_exact
                     and all(row["bit_exact"] for row in replay)
                     and base["status"] == "MEASURED")
    plant_fires = bool(args.plant == "history-ulp" and first is not None
                       and first["substep"] == 1
                       and first["boundary"] == "ubb_e")
    return {
        "format": "nemo-testcase-l2-gyre-round79-raw-history-v1",
        "status": "CONFIRMED" if confirmed else "REFUTED",
        "worktree": stamp,
        "record_producer": producer,
        "record_sha256": round78.RECORD_SHA256,
        "record_replay_exact": all(row["bit_exact"] for row in replay),
        "base_round66_status": base["status"],
        "first_substep_all_bit_exact": first_row_exact,
        "first_live_u_non_bit_statement": first,
        "live_u_rows": rows,
        "shared_statement_on_oracle_inputs": {
            "all_bit_exact": shared_exact, "rows": shared_rows},
        "plant": args.plant,
        "history_ulp_plant_fires": plant_fires,
        "v_live_walk": "WITHHELD_UNTIL_U_OWNER",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--record-root", type=Path,
                        default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument("--admission", type=Path,
                        default=ROOT / "round77/oracle_uamid_kt2/round77_admission.json")
    parser.add_argument("--plant", choices=("none", "history-ulp"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"ROUND79 HISTORY PLANT {'FIRED' if report['history_ulp_plant_fires'] else 'STAYED_GREEN'}")
        return 1
    print(f"ROUND79 RAW HISTORY {report['status']}: first={report['first_live_u_non_bit_statement']}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
