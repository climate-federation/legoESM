#!/usr/bin/env python3
"""Compare the admitted kt=2 external step with the production-JIT trace."""

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
import nemo_testcase_l2_gyre_round78_uamid_walk as round78  # noqa: E402
import nemo_testcase_l2_gyre_round81_btstep_gate as round81  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
RECORD_COMMIT = "295a42edc9d9f45707a5349097e7a0183f57463c"

TRACE_KEYS = {
    "u_b": "u_history_b",
    "v_b": "v_history_b",
    "eta_b": "eta_history_b",
    "u_bb": "u_history_bb",
    "v_bb": "v_history_bb",
    "eta_bb": "eta_history_bb",
    "depth_u_mid": "transport_face_depth_u",
    "depth_v_mid": "transport_face_depth_v",
    "transport_u": "transport_metric_u",
    "transport_v": "transport_metric_v",
    "ssh_forcing": "continuity_forcing",
    "continuity_div": "continuity_divergence",
    "swap_u": "u_exit",
    "swap_v": "v_exit",
    "swap_eta": "eta_continuity",
}

SOURCE_ORDER = (
    "mid_coefficient_1", "mid_coefficient_2", "mid_coefficient_3",
    "u_entry", "v_entry", "eta_entry",
    "u_b", "v_b", "eta_b", "u_bb", "v_bb", "eta_bb",
    "u_mid", "v_mid", "eta_mid",
    "depth_u_mid", "depth_v_mid", "transport_u", "transport_v",
    "ssh_forcing", "continuity_div", "eta_continuity",
    "back_coefficient_0", "back_coefficient_1",
    "back_coefficient_2", "back_coefficient_3", "eta_pgf",
    "pgf_u", "pgf_v", "cor_u", "cor_v",
    "drag_coefficient_u", "drag_coefficient_v",
    "inverse_depth_u", "inverse_depth_v", "trd_u", "trd_v",
    "slow_u", "slow_v", "u_exit", "v_exit",
    "swap_u", "swap_v", "swap_eta",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def first_non_bit(rows: list[dict]) -> dict | None:
    """Return the first row in external-substep then compiled-source order."""
    for row in rows:
        for boundary in SOURCE_ORDER:
            result = row[boundary]
            if not result["bit_exact"]:
                return {"substep": row["substep"], "boundary": boundary, **result}
    return None


def _stagger(name: str) -> str:
    if name.startswith(("u_", "depth_u_")) or name.endswith("_u"):
        return "u"
    if name.startswith(("v_", "depth_v_")) or name.endswith("_v"):
        return "v"
    return "t"


def _native(trace, record_name: str) -> np.ndarray:
    key = TRACE_KEYS.get(record_name, record_name)
    return gate._trace_native(trace[key], key)


def _face_replay_from_live_cells(trace) -> tuple[np.ndarray, np.ndarray]:
    """Replay compiled dyn_drg_init's two face averages from its live input."""
    cell = np.asarray(trace["drag_coefficient_t"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    require(cell.ndim == 2, "live cell drag coefficient is not two-dimensional")
    u_inner = np.float64(0.5) * (np.roll(cell, 1, axis=1) + cell)
    u_full = np.concatenate([u_inner, u_inner[:, :1]], axis=1)
    v_inner = np.float64(0.5) * (cell[:-1, :] + cell[1:, :])
    v_full = np.pad(v_inner, ((1, 1), (0, 0)), mode="edge")
    return u_full[:, 1:], v_full[1:, :]


def _admit(args) -> dict:
    admission = json.loads(args.admission.read_text())
    require(admission["verdict"] == "PASS", "Round-81 twin admission failed")
    require(
        (
            admission["byte_identical_records"],
            len(admission["classified_changed_records"]),
            admission["admitted_difference_count"],
        ) == (46, 24, 264),
        "Round-81 inherited-record census changed",
    )
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(producer == RECORD_COMMIT, "Round-81 producer commit changed")
    record = args.record_root / round81.RECORD
    stamp = record.with_name(record.name + ".stamp")
    require(
        stamp.read_text().split() == [round81.sha256(record), producer, record.name],
        "Round-81 record stamp mismatch",
    )
    require(record.stat().st_size == round81.EXPECTED_SIZE, "Round-81 record size changed")
    fields = round81.read_record(record)
    require(
        all(
            result["bit_exact"]
            for row in round81.validate_fields(fields)
            for name, result in row.items()
            if name != "substep"
        ),
        "Round-81 arithmetic replay changed",
    )
    uamid = round81.compare_uamid(fields, args.uamid_root / round81.UAMID_RECORD)
    require(all(row["bit_exact"] for row in uamid.values()), "Round-77 U identity changed")
    return fields


def measure(args) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64 libm",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-82 measurement worktree is dirty")
    require(stamp["commit"] == args.expect_commit, "Round-82 measurement commit mismatch")

    fields = _admit(args)
    base, card, _seeded, captured = round78._context(args)
    require(base["status"] == "MEASURED", "inherited kt=2 seeded context changed")
    trace = captured.substeps
    masks = gate.expected_masks(card)
    active = {
        "t": masks["ssh"],
        "u": masks["u"][..., 0],
        "v": masks["v"][..., 0],
    }
    for name, expected in (
        ("t_mask", active["t"]),
        ("u_mask", active["u"]),
        ("v_mask", active["v"]),
    ):
        require(
            np.array_equal(fields[name] != 0.0, expected),
            f"{name} differs from the live native mask",
        )

    live_arrays = {
        name: np.asarray(_native(trace, name), dtype=np.float64)
        for name in round81.ARRAY_FIELDS
    }
    oracle_arrays = {
        name: np.array(fields[name], copy=True) for name in round81.ARRAY_FIELDS
    }
    live_mid = np.stack(
        [np.asarray(trace[f"mid_weight_{index}"], dtype=np.float64)
         for index in (1, 2, 3)], axis=-1)
    live_back = np.stack(
        [np.asarray(trace[f"back_weight_{index}"], dtype=np.float64)
         for index in (0, 1, 2, 3)], axis=-1)
    oracle_mid = np.array(fields["mid_coefficients"], copy=True)
    oracle_back = np.array(fields["back_coefficients"], copy=True)
    replay_drag_u, replay_drag_v = _face_replay_from_live_cells(trace)
    for name in round81.ARRAY_FIELDS:
        require(
            live_arrays[name].shape == oracle_arrays[name].shape,
            f"{name} live/oracle shape mismatch: "
            f"{live_arrays[name].shape} != {oracle_arrays[name].shape}",
        )
    require(
        live_mid.shape == oracle_mid.shape,
        f"mid coefficient shape mismatch: {live_mid.shape} != {oracle_mid.shape}",
    )
    require(
        live_back.shape == oracle_back.shape,
        f"back coefficient shape mismatch: {live_back.shape} != {oracle_back.shape}",
    )

    plant_detail = None
    if args.plant == "history-ulp":
        location = (0, *tuple(np.argwhere(active["u"])[0]))
        oracle_arrays["u_b"][location] = np.nextafter(
            oracle_arrays["u_b"][location], np.float64(np.inf))
        plant_detail = {"field": "u_b", "location": list(location)}
    elif args.plant == "slow-u-ulp":
        equal = (
            oracle_arrays["slow_u"].view(np.uint64)
            == live_arrays["slow_u"].view(np.uint64)
        ) & np.broadcast_to(active["u"], oracle_arrays["slow_u"].shape)
        require(np.any(equal), "slow-U ULP plant has no exact wet target")
        location = tuple(np.argwhere(equal)[0])
        oracle_arrays["slow_u"][location] = np.nextafter(
            oracle_arrays["slow_u"][location], np.float64(np.inf))
        plant_detail = {"field": "slow_u", "location": list(location)}
    elif args.plant == "null-slow-u":
        ordinary = round78.comparison(
            live_arrays["slow_u"][0], oracle_arrays["slow_u"][0], active["u"])
        require(not ordinary["bit_exact"], "null slow-U plant target is already exact")
        oracle_arrays["slow_u"][0] = live_arrays["slow_u"][0]
        plant_detail = {"field": "slow_u", "substep": 1}

    scalar_mask = np.ones((), dtype=bool)
    rows = []
    for substep in range(round81.N_CYCLE):
        row = {
            "substep": substep + 1,
            **{
                f"mid_coefficient_{index + 1}": round78.comparison(
                    live_mid[substep, index], oracle_mid[substep, index], scalar_mask)
                for index in range(3)
            },
            **{
                name: round78.comparison(
                    live_arrays[name][substep], oracle_arrays[name][substep],
                    active[_stagger(name)],
                )
                for name in round81.ARRAY_FIELDS[:19]
            },
            **{
                f"back_coefficient_{index}": round78.comparison(
                    live_back[substep, index], oracle_back[substep, index], scalar_mask)
                for index in range(4)
            },
            **{
                name: round78.comparison(
                    live_arrays[name][substep], oracle_arrays[name][substep],
                    active[_stagger(name)],
                )
                for name in round81.ARRAY_FIELDS[19:]
            },
        }
        rows.append(row)

    first = first_non_bit(rows)
    prediction_confirmed = bool(
        args.plant == "none"
        and first is not None
        and first["substep"] == 1
        and first["boundary"] in {"slow_u", "slow_v"}
    )
    plant_fires = False
    if args.plant == "history-ulp":
        plant_fires = bool(first and first["substep"] == 1 and first["boundary"] == "u_b")
    elif args.plant == "slow-u-ulp":
        location = tuple(plant_detail["location"])
        plant_fires = bool(
            oracle_arrays["slow_u"][location].view(np.uint64)
            != live_arrays["slow_u"][location].view(np.uint64)
        )
    elif args.plant == "null-slow-u":
        plant_fires = rows[0]["slow_u"]["bit_exact"]

    return {
        "format": "nemo-testcase-l2-gyre-round82-btstep-walk-v1",
        "status": "CONFIRMED" if prediction_confirmed else "REFUTED",
        "worktree": stamp,
        "execution_regime": "production-jit-cpu-fp64-x64-libm",
        "record_commit": RECORD_COMMIT,
        "record_sha256": round81.sha256(args.record_root / round81.RECORD),
        "record_size": round81.EXPECTED_SIZE,
        "admission_counts": {"exact": 46, "total": 70, "changed": 24, "admitted": 264},
        "dtype": {"record": str(fields["u_entry"].dtype),
                  "live": str(live_arrays["u_entry"].dtype)},
        "first_non_bit_statement": first,
        "drag_face_statement_on_live_cell_input": {
            "u": round78.comparison(
                replay_drag_u, oracle_arrays["drag_coefficient_u"][0], active["u"]),
            "v": round78.comparison(
                replay_drag_v, oracle_arrays["drag_coefficient_v"][0], active["v"]),
        },
        "rows": rows,
        "plant": args.plant,
        "plant_detail": plant_detail,
        "plant_fires": plant_fires,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument(
        "--record-root", type=Path,
        default=ROOT / "round81/oracle_btstep_kt2")
    parser.add_argument(
        "--uamid-root", type=Path,
        default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument(
        "--admission", type=Path,
        default=ROOT / "round81/oracle_btstep_kt2/round81_admission.json")
    parser.add_argument(
        "--plant", choices=("none", "history-ulp", "slow-u-ulp", "null-slow-u"),
        default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.plant != "none":
        print(
            f"ROUND82 {args.plant.upper()} PLANT "
            f"{'FIRED' if report['plant_fires'] else 'STAYED_GREEN'}"
        )
        return 1
    print(
        f"ROUND82 BTSTEP {report['status']}: "
        f"first={report['first_non_bit_statement']}"
    )
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
