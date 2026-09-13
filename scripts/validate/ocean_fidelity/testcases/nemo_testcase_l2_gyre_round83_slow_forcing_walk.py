#!/usr/bin/env python3
"""Walk the admitted GYRE kt=2 ``stp2d`` slow-forcing producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round16_slow_forcing as round16  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round78_uamid_walk as round78  # noqa: E402
import nemo_testcase_l2_gyre_round81_btstep_gate as round81  # noqa: E402
import nemo_testcase_l2_gyre_round82_btstep_walk as round82  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
ROUND64_PRODUCER = "3b3b045bd9e03b60330204e7590e4c4470b7a0ca"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def owned3(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2, :30]


def owned2(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2]


def native_u(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, 1:, ...]


def native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :, ...]


def bottom_value(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select NEMO's deepest wet level from an already face-staggered field."""
    require(values.shape == mask.shape and values.ndim == 3, "bad bottom-field shape")
    count = np.sum(mask, axis=-1, dtype=np.int64)
    require(np.all(count[mask.any(axis=-1)] > 0), "wet face has no wet level")
    index = np.maximum(count - 1, 0)[..., None]
    return np.take_along_axis(values, index, axis=-1)[..., 0]


def source_chain(
    rhs: np.ndarray,
    e3: np.ndarray,
    mask3: np.ndarray,
    reciprocal_ref: np.ndarray,
    inverse_depth: np.ndarray,
    drag_coefficient: np.ndarray,
    bottom_velocity: np.ndarray,
    barotropic_velocity: np.ndarray,
    rho_reciprocal: np.float64,
    stress: np.ndarray,
    coriolis: np.ndarray,
    mask2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay compiled stp2d/dynspg statements without reassociation."""
    depth = round16._source_sum(e3, rhs, mask3, reciprocal_ref)
    residual = bottom_velocity - barotropic_velocity
    drag_increment = (inverse_depth * drag_coefficient) * residual
    post_drag = depth + drag_increment
    wind_increment = (rho_reciprocal * stress) * inverse_depth
    post_wind = post_drag + wind_increment
    final = post_wind - coriolis * mask2
    return {
        "depth_mean": depth,
        "drag_increment": drag_increment,
        "post_drag": post_drag,
        "wind_increment": wind_increment,
        "post_wind": post_wind,
        "final": final,
    }


def comparison(candidate, oracle, active) -> dict[str, object]:
    return round78.comparison(
        np.asarray(candidate, dtype=np.float64),
        np.asarray(oracle, dtype=np.float64),
        np.asarray(active, dtype=bool),
    )


def _admit_round64(args) -> tuple[dict, dict]:
    admission = json.loads(args.round64_admission.read_text())
    require(admission.get("verdict") == "PASS", "Round-64 admission failed")
    require(
        (
            admission.get("byte_identical_records"),
            len(admission.get("classified_changed_records", [])),
            admission.get("admitted_difference_count"),
        ) == (43, 20, 132),
        "Round-64 inherited-record census changed",
    )
    producer = (args.round64_root / "producer_commit.txt").read_text().strip()
    require(producer == ROUND64_PRODUCER, "Round-64 producer commit changed")
    stage_path = args.round64_root / "oracle_momstage_kt00000002_s1.bin"
    original_stage = args.round46_root / stage_path.name
    require(stage_path.is_file() and original_stage.is_file(), "missing kt=2 stage record")
    classified = [
        row for row in admission["classified_changed_records"]
        if row["record"] == stage_path.name
    ]
    require(len(classified) == 1, "kt=2 stage record lacks one admission classification")
    stage_admission = classified[0]
    require(stage_admission["consumed_equal"], "kt=2 stage consumed projection differs")
    require(stage_admission["owned_field_differences"] == [],
            "kt=2 stage has an owned-field difference")
    require(
        stage_admission["reason_counts"] == {
            "halo": 7,
            "owned_defined_violation": 0,
            "owned_undefined_region": 0,
            "owned_undefined_slot": 0,
        },
        "kt=2 stage admission reasons changed",
    )
    used_fields = {
        "after_adv_u", "after_adv_v", "e3u_0", "e3v_0", "umask", "vmask",
        "u_Kmm", "v_Kmm", "uu_b_Kmm", "vv_b_Kmm", "has_ldf",
    }
    require(used_fields <= set(stage_admission["compared_fields"]),
            "a consumed kt=2 stage field is outside admission coverage")
    slow_path = args.round64_root / "oracle_slow_forcing_kt00000001.bin"
    stage = round46.read_stage(stage_path)
    slow = round16.read_slow_forcing(slow_path)
    require(stage["header"]["kt"] == 2 and stage["header"]["stage"] == 1,
            "wrong kt/stage record")
    require(stage["arrays"]["has_ldf"] == 1.0, "kt=2 RHS lacks final LDF row")
    return stage, slow


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-83 measurement worktree is dirty")
    require(stamp["commit"] == args.expect_commit, "Round-83 commit stamp mismatch")

    stage, static = _admit_round64(args)
    bt = round82._admit(args)
    base, card, seeded, captured = round78._context(args)
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    trace = captured.slow_forcing_operands
    substeps = captured.substeps
    masks = gate.expected_masks(card)
    active = {
        "u3": np.asarray(masks["u"], dtype=bool),
        "v3": np.asarray(masks["v"], dtype=bool),
    }
    active["u2"] = active["u3"][..., 0]
    active["v2"] = active["v3"][..., 0]

    stage_arrays = stage["arrays"]
    oracle = {
        "u": {
            "rhs": owned3(stage_arrays["after_adv_u"]),
            "e3": owned3(stage_arrays["e3u_0"]),
            "mask3": owned3(stage_arrays["umask"]),
            "reciprocal_ref": np.asarray(static["r1_hu0"], dtype=np.float64),
            "inverse_depth": np.asarray(bt["inverse_depth_u"][0], dtype=np.float64),
            "drag_coefficient": np.asarray(bt["drag_coefficient_u"][0], dtype=np.float64),
            "bottom_velocity": bottom_value(
                owned3(stage_arrays["u_Kmm"]), owned3(stage_arrays["umask"]) != 0.0),
            "barotropic_velocity": owned2(stage_arrays["uu_b_Kmm"]),
            "stress": None,
            "coriolis": np.asarray(bt["cor_u"][0], dtype=np.float64),
            "final": np.asarray(bt["slow_u"][0], dtype=np.float64),
        },
        "v": {
            "rhs": owned3(stage_arrays["after_adv_v"]),
            "e3": owned3(stage_arrays["e3v_0"]),
            "mask3": owned3(stage_arrays["vmask"]),
            "reciprocal_ref": np.asarray(static["r1_hv0"], dtype=np.float64),
            "inverse_depth": np.asarray(bt["inverse_depth_v"][0], dtype=np.float64),
            "drag_coefficient": np.asarray(bt["drag_coefficient_v"][0], dtype=np.float64),
            "bottom_velocity": bottom_value(
                owned3(stage_arrays["v_Kmm"]), owned3(stage_arrays["vmask"]) != 0.0),
            "barotropic_velocity": owned2(stage_arrays["vv_b_Kmm"]),
            "stress": None,
            "coriolis": np.asarray(bt["cor_v"][0], dtype=np.float64),
            "final": np.asarray(bt["slow_v"][0], dtype=np.float64),
        },
    }
    live = {
        "u": {
            "rhs": native_u(trace["du_dt"]),
            "e3": native_u(trace["h_u"]),
            "mask3": active["u3"].astype(np.float64),
            "reciprocal_ref": np.float64(1.0) / native_u(trace["H_u"]),
            "inverse_depth": gate._trace_native(substeps["inverse_depth_u"], "inverse_depth_u")[0],
            "drag_coefficient": gate._trace_native(
                substeps["drag_coefficient_u"], "drag_coefficient_u")[0],
            "bottom_velocity": bottom_value(native_u(seeded.u.data), active["u3"]),
            "barotropic_velocity": gate._trace_native(substeps["u_entry"], "u_entry")[0],
            "stress": native_u(trace["wind_tau_u"]),
            "coriolis": gate._trace_native(substeps["cor_u"], "cor_u")[0],
        },
        "v": {
            "rhs": native_v(trace["dv_dt"]),
            "e3": native_v(trace["h_v"]),
            "mask3": active["v3"].astype(np.float64),
            "reciprocal_ref": np.float64(1.0) / native_v(trace["H_v"]),
            "inverse_depth": gate._trace_native(substeps["inverse_depth_v"], "inverse_depth_v")[0],
            "drag_coefficient": gate._trace_native(
                substeps["drag_coefficient_v"], "drag_coefficient_v")[0],
            "bottom_velocity": bottom_value(native_v(seeded.v.data), active["v3"]),
            "barotropic_velocity": gate._trace_native(substeps["v_entry"], "v_entry")[0],
            "stress": native_v(trace["wind_tau_v"]),
            "coriolis": gate._trace_native(substeps["cor_v"], "cor_v")[0],
        },
    }
    # The inherited slow-forcing stream is kt=1; its stress is not silently
    # reused at kt=2.  No direct kt=2 stress record exists.  Use the live kt=2
    # value only as a cross-record calibration operand, and withhold a direct
    # wind-stress identity claim.  The replay must still recover NEMO's final
    # recorded forcing exactly or the joined record is refused.
    oracle["u"]["stress"] = live["u"]["stress"]
    oracle["v"]["stress"] = live["v"]["stress"]
    rho_reciprocal = np.float64(static["r1_rho0"])
    require(float(trace["wind_r1_rho0"]) == float(rho_reciprocal),
            "live/oracle density reciprocal differs")

    ordinary_plant_target = {
        "e3-ulp": comparison(live["u"]["e3"], oracle["u"]["e3"], active["u3"]),
        "rhs-ulp": comparison(live["u"]["rhs"], oracle["u"]["rhs"], active["u3"]),
        "final-ulp": comparison(
            source_chain(
                **{key: value for key, value in oracle["u"].items() if key != "final"},
                rho_reciprocal=rho_reciprocal,
                mask2=active["u2"].astype(np.float64),
            )["final"],
            oracle["u"]["final"],
            active["u2"],
        ),
    }

    plant_detail = None
    if args.plant == "e3-ulp":
        delta = np.where(
            active["u3"], np.abs(live["u"]["e3"] - oracle["u"]["e3"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["e3"] = np.array(oracle["u"]["e3"], copy=True)
        direction = np.inf if oracle["u"]["e3"][at] >= live["u"]["e3"][at] else -np.inf
        oracle["u"]["e3"][at] = np.nextafter(oracle["u"]["e3"][at], direction)
        plant_detail = {"field": "e3_u", "location": list(map(int, at))}
    elif args.plant == "rhs-ulp":
        delta = np.where(
            active["u3"], np.abs(live["u"]["rhs"] - oracle["u"]["rhs"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["rhs"] = np.array(oracle["u"]["rhs"], copy=True)
        direction = np.inf if oracle["u"]["rhs"][at] >= live["u"]["rhs"][at] else -np.inf
        oracle["u"]["rhs"][at] = np.nextafter(oracle["u"]["rhs"][at], direction)
        plant_detail = {"field": "rhs_u", "location": list(map(int, at))}
    elif args.plant == "final-ulp":
        replay = source_chain(
            **{key: value for key, value in oracle["u"].items() if key != "final"},
            rho_reciprocal=rho_reciprocal,
            mask2=active["u2"].astype(np.float64),
        )["final"]
        delta = np.where(
            active["u2"], np.abs(replay - oracle["u"]["final"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["final"] = np.array(oracle["u"]["final"], copy=True)
        direction = np.inf if oracle["u"]["final"][at] >= replay[at] else -np.inf
        oracle["u"]["final"][at] = np.nextafter(oracle["u"]["final"][at], direction)
        plant_detail = {"field": "slow_u", "location": list(map(int, at))}

    rows = {}
    cross_record = {}
    reference_geometry_arms = {}
    for face in ("u", "v"):
        active3 = active[f"{face}3"]
        active2 = active[f"{face}2"]
        for name in oracle[face]:
            if name == "final":
                continue
            require(np.asarray(oracle[face][name]).shape == np.asarray(live[face][name]).shape,
                    f"{face}.{name} live/oracle shape mismatch")
        oracle_chain = source_chain(
            **{key: value for key, value in oracle[face].items() if key != "final"},
            rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        live_chain = source_chain(
            **live[face], rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        reference_geometry_inputs = dict(live[face])
        for name in ("e3", "mask3", "reciprocal_ref"):
            reference_geometry_inputs[name] = oracle[face][name]
        reference_geometry_chain = source_chain(
            **reference_geometry_inputs,
            rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        cross_record[face] = comparison(oracle_chain["final"], oracle[face]["final"], active2)
        reference_geometry_arms[face] = {
            "depth_mean": comparison(
                reference_geometry_chain["depth_mean"], oracle_chain["depth_mean"], active2),
            "final_slow_forcing": comparison(
                reference_geometry_chain["final"], oracle[face]["final"], active2),
            "movement_from_current_depth_mean": comparison(
                reference_geometry_chain["depth_mean"], live_chain["depth_mean"], active2),
        }
        rows[face] = {
            "e3": comparison(live[face]["e3"], oracle[face]["e3"], active3),
            "mask": comparison(live[face]["mask3"], oracle[face]["mask3"], active3),
            "three_dimensional_rhs": comparison(live[face]["rhs"], oracle[face]["rhs"], active3),
            "reference_depth_reciprocal": comparison(
                live[face]["reciprocal_ref"], oracle[face]["reciprocal_ref"], active2),
            "depth_mean": comparison(live_chain["depth_mean"], oracle_chain["depth_mean"], active2),
            "drag_coefficient": comparison(
                live[face]["drag_coefficient"], oracle[face]["drag_coefficient"], active2),
            "inverse_depth": comparison(
                live[face]["inverse_depth"], oracle[face]["inverse_depth"], active2),
            "bottom_velocity": comparison(
                live[face]["bottom_velocity"], oracle[face]["bottom_velocity"], active2),
            "barotropic_velocity": comparison(
                live[face]["barotropic_velocity"], oracle[face]["barotropic_velocity"], active2),
            "post_drag": comparison(live_chain["post_drag"], oracle_chain["post_drag"], active2),
            "post_wind": comparison(live_chain["post_wind"], oracle_chain["post_wind"], active2),
            "coriolis": comparison(live[face]["coriolis"], oracle[face]["coriolis"], active2),
            "final_slow_forcing": comparison(live_chain["final"], oracle[face]["final"], active2),
            "oracle_rhs_substitution_final": comparison(
                oracle_chain["final"], oracle[face]["final"], active2),
        }

    cross_exact = all(row["bit_exact"] for row in cross_record.values())
    first = None
    source_order = (
        "e3", "mask", "three_dimensional_rhs", "reference_depth_reciprocal",
        "depth_mean", "drag_coefficient", "inverse_depth", "bottom_velocity",
        "barotropic_velocity", "post_drag", "post_wind",
        "coriolis", "final_slow_forcing",
    )
    for boundary in source_order:
        for face in ("u", "v"):
            if first is None and not rows[face][boundary]["bit_exact"]:
                first = {"boundary": boundary, "face": face, **rows[face][boundary]}

    confirmed = bool(
        args.plant == "none"
        and cross_exact
        and first is not None
        and first["boundary"] == "three_dimensional_rhs"
        and all(rows[face]["oracle_rhs_substitution_final"]["bit_exact"] for face in ("u", "v"))
    )
    planted_row = {
        "e3-ulp": rows["u"]["e3"],
        "rhs-ulp": rows["u"]["three_dimensional_rhs"],
        "final-ulp": cross_record["u"],
    }.get(args.plant)
    plant_fires = bool(
        args.plant != "none"
        and planted_row != ordinary_plant_target[args.plant]
    )
    status = "CONFIRMED" if confirmed else ("PLANT_FIRED" if plant_fires else "REFUTED")
    return {
        "format": "nemo-testcase-l2-gyre-round83-slow-forcing-walk-v1",
        "status": status,
        "worktree": stamp,
        "execution_regime": "production-jit-cpu-fp64-x64-libm",
        "round64_stage_sha256": sha256(args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
        "round81_btstep_sha256": round81.sha256(args.record_root / round81.RECORD),
        "cross_record_replay": cross_record,
        "kt2_wind_stress_identity": "WITHHELD_NO_DIRECT_RECORD",
        "reference_geometry_arm": reference_geometry_arms,
        "first_non_bit_statement": first,
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
    parser.add_argument("--round64-root", type=Path, default=ROOT / "round64/oracle_krhs_split")
    parser.add_argument("--round46-root", type=Path, default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--round64-admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    parser.add_argument("--record-root", type=Path, default=ROOT / "round81/oracle_btstep_kt2")
    parser.add_argument("--uamid-root", type=Path, default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument("--admission", type=Path,
                        default=ROOT / "round81/oracle_btstep_kt2/round81_admission.json")
    parser.add_argument("--plant", choices=("none", "e3-ulp", "rhs-ulp", "final-ulp"),
                        default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError, KeyError, ValueError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"ROUND83 {args.plant.upper()} {'FIRED' if report['plant_fires'] else 'STAYED_GREEN'}")
        return 1
    print(f"ROUND83 SLOW FORCING {report['status']}: first={report['first_non_bit_statement']}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
