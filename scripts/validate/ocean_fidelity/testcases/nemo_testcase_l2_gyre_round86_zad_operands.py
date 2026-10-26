#!/usr/bin/env python3
"""Walk GYRE kt=2 stage-1 ZAD operands against the admitted NEMO record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round41_dynadv_split as round41  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83  # noqa: E402
import nemo_testcase_l2_gyre_round84_rhs_walk as round84  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
OPERAND_ORDER = (
    "velocity_u", "velocity_v", "ww", "r3u", "r3v", "thickness_u",
    "thickness_v", "area_t", "reciprocal_area_u", "reciprocal_area_v",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _inject_owned(full: np.ndarray, owned: np.ndarray, *, nlev: int) -> np.ndarray:
    """Replace the admitted owned slab while retaining recorded halo values."""
    changed = np.array(full, copy=True)
    require(changed.ndim == 3 and owned.ndim == 3, "owned injection must be 3-D")
    require(changed[2:-2, 2:-2, :nlev].shape == owned[..., :nlev].shape,
            "owned injection shape mismatch")
    changed[2:-2, 2:-2, :nlev] = owned[..., :nlev]
    return changed


def _inject_owned2(full: np.ndarray, owned: np.ndarray) -> np.ndarray:
    changed = np.array(full, copy=True)
    require(changed.ndim == 2 and owned.ndim == 2, "owned injection must be 2-D")
    require(changed[2:-2, 2:-2].shape == owned.shape,
            "owned injection shape mismatch")
    changed[2:-2, 2:-2] = owned
    return changed


def _row(candidate, oracle, active) -> dict[str, object]:
    row = round84._row(candidate, oracle, active)
    row["dtype"] = str(np.asarray(candidate).dtype)
    return row


def _first_nonbit(rows: dict[str, dict[str, object]]) -> str | None:
    return next((name for name in OPERAND_ORDER if not rows[name]["bit_exact"]), None)


def _away_one_ulp(reference: np.ndarray, candidate: np.ndarray,
                  active: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    return round84._away_one_ulp(reference, candidate, active)


def _capture_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        expect_commit=args.expect_commit,
        expect_krhs_commit=args.expect_krhs_commit,
        output=args.prerequisite_output,
    )


def _replay_with(split: dict, replacements: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    replay = {name: np.array(value, copy=True) if isinstance(value, np.ndarray)
              else value for name, value in split.items()}
    replay.update(replacements)
    return round41._zad_replay(replay)


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-86 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-86 commit stamp mismatch")

    stage, _ = round83._admit_round64(args)
    arrays = stage["arrays"]
    split = round46._split_view(arrays)
    oracle_after = {
        "u": np.asarray(arrays["after_zad_u"]),
        "v": np.asarray(arrays["after_zad_v"]),
    }
    literal_after = dict(zip(("u", "v"), round41._zad_replay(split), strict=True))
    literal_calibration = {
        face: int(np.count_nonzero(literal_after[face] != oracle_after[face]))
        for face in ("u", "v")
    }
    require(all(value == 0 for value in literal_calibration.values()),
            f"given-input literal ZAD replay moved: {literal_calibration}")

    base, context = round72._capture_seeded_context(_capture_args(args))
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    card, seeded, _, _, trace = context
    parts = trace.operator_operands[0]
    masks = gate.expected_masks(card)
    active_u = np.asarray(masks["u"], dtype=bool)
    active_v = np.asarray(masks["v"], dtype=bool)
    active_t = np.asarray(masks["T"], dtype=bool)

    live = {
        "velocity_u": round83.native_u(parts["operand_velocity_u"]),
        "velocity_v": round83.native_v(parts["operand_velocity_v"]),
        "ww": np.asarray(parts["operand_zad_w"], dtype=np.float64)[..., :30],
        "thickness_u": round83.native_u(parts["operand_zad_h_u"]),
        "thickness_v": round83.native_v(parts["operand_zad_h_v"]),
        "area_t": np.asarray(card.recipe.grid.area_T, dtype=np.float64),
        "reciprocal_area_u": round83.native_u(
            np.float64(1.0) / (
                np.asarray(card.recipe.grid.dx_u, dtype=np.float64)
                * np.asarray(card.recipe.grid.dy_u, dtype=np.float64))),
        "reciprocal_area_v": round83.native_v(
            np.float64(1.0) / (
                np.asarray(card.recipe.grid.dx_v, dtype=np.float64)
                * np.asarray(card.recipe.grid.dy_v, dtype=np.float64))),
    }
    _, live_r3u, live_r3v = trace.stage_qco[0]
    live["r3u"] = round83.native_u(live_r3u)[..., 0]
    live["r3v"] = round83.native_v(live_r3v)[..., 0]
    oracle = {
        "velocity_u": round83.owned3(arrays["u_Kmm"]),
        "velocity_v": round83.owned3(arrays["v_Kmm"]),
        "ww": round83.owned3(arrays["ww"]),
        "r3u": round83.owned2(arrays["r3u_Kmm"]),
        "r3v": round83.owned2(arrays["r3v_Kmm"]),
        "thickness_u": round83.owned3(arrays["e3u_Kmm"]),
        "thickness_v": round83.owned3(arrays["e3v_Kmm"]),
        "area_t": round83.owned2(arrays["e1e2t"]),
        "reciprocal_area_u": round83.owned2(arrays["r1_e1e2u"]),
        "reciprocal_area_v": round83.owned2(arrays["r1_e1e2v"]),
    }
    active = {
        "velocity_u": active_u, "velocity_v": active_v,
        "ww": active_t, "r3u": active_u[..., 0], "r3v": active_v[..., 0],
        "thickness_u": active_u, "thickness_v": active_v,
        "area_t": active_t[..., 0],
        "reciprocal_area_u": active_u[..., 0],
        "reciprocal_area_v": active_v[..., 0],
    }

    ordinary_rows = {}
    for name in OPERAND_ORDER:
        shapes = (np.asarray(live[name]).shape, np.asarray(oracle[name]).shape,
                  np.asarray(active[name]).shape)
        require(shapes[0] == shapes[1] == shapes[2],
                f"{name} comparison shape mismatch: {shapes}")
        ordinary_rows[name] = _row(live[name], oracle[name], active[name])
    ordinary_rows["ww"]["scope"] = "owned t points, consumed levels 1:jpkm1"

    plant_detail = None
    if args.plant in ("ww-ulp", "thickness-ulp"):
        target_name = {
            "ww-ulp": "ww",
            "thickness-ulp": "thickness_u",
        }[args.plant]
        changed, at = _away_one_ulp(
            oracle[target_name], live[target_name], active[target_name])
        oracle[target_name] = changed
        plant_detail = {"field": target_name, "location": list(at)}

    rows = {
        name: _row(live[name], oracle[name], active[name])
        for name in OPERAND_ORDER
    }

    replacements = {
        "velocity": {
            "uu_Kmm": _inject_owned(split["uu_Kmm"], live["velocity_u"], nlev=30),
            "vv_Kmm": _inject_owned(split["vv_Kmm"], live["velocity_v"], nlev=30),
        },
        "ww": {"ww": _inject_owned(split["ww"], live["ww"], nlev=30)},
        "thickness": {
            "e3u_Kmm": _inject_owned(split["e3u_Kmm"], live["thickness_u"], nlev=30),
            "e3v_Kmm": _inject_owned(split["e3v_Kmm"], live["thickness_v"], nlev=30),
        },
        "metric": {
            "e1e2t": _inject_owned2(split["e1e2t"], live["area_t"]),
            "r1_e1e2u": _inject_owned2(
                split["r1_e1e2u"], live["reciprocal_area_u"]),
            "r1_e1e2v": _inject_owned2(
                split["r1_e1e2v"], live["reciprocal_area_v"]),
        },
    }
    replacements["all"] = {
        key: value for group in replacements.values() for key, value in group.items()
    }
    substitution_rows = {}
    for name, replacement in replacements.items():
        replay_u, replay_v = _replay_with(split, replacement)
        substitution_rows[name] = {
            "u": _row(round83.owned3(replay_u), round83.owned3(oracle_after["u"]), active_u),
            "v": _row(round83.owned3(replay_v), round83.owned3(oracle_after["v"]), active_v),
        }

    live_total = {
        "u": round83.native_u(parts["after_ldf_u"].data),
        "v": round83.native_v(parts["after_ldf_v"].data),
    }
    source_order = jax.device_get(jax.jit(round84.source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))
    source_order_native = {
        "u": round83.native_u(source_order["after_adv_u"]),
        "v": round83.native_v(source_order["after_adv_v"]),
    }
    ordinary_association = {
        face: _row(source_order_native[face], live_total[face],
                   active_u if face == "u" else active_v)
        for face in ("u", "v")
    }
    if args.plant == "association-ulp":
        live_total["u"], at = _away_one_ulp(
            live_total["u"], source_order_native["u"], active_u)
        plant_detail = {"field": "association_u", "location": list(at)}
    association = {
        face: _row(
            source_order_native[face],
            live_total[face], active_u if face == "u" else active_v)
        for face in ("u", "v")
    }

    first = _first_nonbit(rows)
    all_max = {face: substitution_rows["all"][face]["absolute_max"]
               for face in ("u", "v")}
    w_fraction = {
        face: (substitution_rows["ww"][face]["absolute_max"] / all_max[face]
               if all_max[face] else 0.0)
        for face in ("u", "v")
    }
    thickness_fraction = {
        face: (substitution_rows["thickness"][face]["absolute_max"] / all_max[face]
               if all_max[face] else 0.0)
        for face in ("u", "v")
    }
    confirmed = bool(
        args.plant is None
        and first == "ww"
        and all(value >= 0.9 for value in w_fraction.values())
        and all(value < 0.01 for value in thickness_fraction.values())
        and all(row["bit_exact"] for row in association.values())
    )
    plant_fired = None
    if args.plant:
        if args.plant == "association-ulp":
            target = association["u"]
            before = ordinary_association["u"]
        else:
            target_name = plant_detail["field"]
            target = rows[target_name]
            before = ordinary_rows[target_name]
        plant_fired = bool(
            target["absolute_max"] != before["absolute_max"]
            or target["differing_cells"] != before["differing_cells"])
        require(plant_fired, f"{args.plant} differential control was invisible")

    return {
        "format": "nemo-testcase-l2-gyre-round86-zad-operands-v1",
        "status": "PLANT_FIRED" if args.plant else (
            "CONFIRMED" if confirmed else "REFUTED"),
        "worktree": stamp,
        "record": {
            "path": str(args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
            "producer_commit": round83.ROUND64_PRODUCER,
            "admission": str(args.round64_admission),
        },
        "literal_given_input_unequal": literal_calibration,
        "operand_order": list(OPERAND_ORDER),
        "operand_rows": rows,
        "first_nonbit_operand": first,
        "substitution_rows": substitution_rows,
        "ww_fraction_of_all_live_operand_max": w_fraction,
        "thickness_fraction_of_all_live_operand_max": thickness_fraction,
        "source_order_to_live_association": association,
        "prediction_confirmed": confirmed,
        "plant": args.plant,
        "plant_detail": plant_detail,
        "plant_fired": plant_fired,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--round64-root", type=Path,
                        default=ROOT / "round64/oracle_krhs_split")
    parser.add_argument("--round46-root", type=Path,
                        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--round64-admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    parser.add_argument("--prerequisite-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=(
        "ww-ulp", "thickness-ulp", "association-ulp"))
    args = parser.parse_args(argv)
    report = measure(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND86_ZAD_OPERANDS", report["status"])
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
