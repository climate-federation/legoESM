#!/usr/bin/env python3
"""Prove the carried GYRE WS-RK3 Kaa SSH and its stage-one W/ZAD consumer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83  # noqa: E402
import nemo_testcase_l2_gyre_round84_rhs_walk as round84  # noqa: E402
import nemo_testcase_l2_gyre_round86_zad_operands as round86  # noqa: E402
import nemo_testcase_l2_gyre_round87_wzv_inputs as round87  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _confirmed(rows: dict[str, dict[str, object]]) -> bool:
    return bool(rows) and all(bool(row["bit_exact"]) for row in rows.values())


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision changed")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-88 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-88 commit stamp mismatch")

    stage, _ = round83._admit_round64(args)
    arrays = stage["arrays"]
    base, context = round72._capture_seeded_context(SimpleNamespace(
        expect_commit=args.expect_commit,
        expect_krhs_commit=args.expect_krhs_commit,
        output=args.prerequisite_output,
    ))
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    card, seeded, _, _, live_step = context
    require(seeded.eta_rk3_kaa is not None, "production state lacks Kaa scratch")

    oracle = round87._oracle_trace(arrays)
    oracle.update(round87._oracle_scalar_boundaries(arrays))
    masks = gate.expected_masks(card)
    active = {
        "u": np.asarray(masks["u"], bool),
        "v": np.asarray(masks["v"], bool),
        "t": np.asarray(masks["T"], bool),
    }
    active_w = round87._active("ww", active)
    scratch = np.asarray(seeded.eta_rk3_kaa.data)
    direct_w = np.asarray(jax.device_get(jax.jit(
        lambda: round87._direct_production_w(
            card, seeded, live_step, seeded.eta_rk3_kaa.data))()))
    captured_w = np.asarray(live_step.operator_operands[0]["operand_zad_w"])

    split = round46._split_view(arrays)
    zad_u, zad_v = round86._replay_with(
        split, {"ww": round86._inject_owned(
            split["ww"], captured_w[..., :30], nlev=30)})
    candidates = {
        "carried_kaa_ssh": scratch,
        "direct_shared_w": direct_w,
        "captured_stage1_w": captured_w,
        "zad_u": round83.owned3(zad_u),
        "zad_v": round83.owned3(zad_v),
    }
    references = {
        "carried_kaa_ssh": np.asarray(oracle["eta_kaa"]),
        "direct_shared_w": np.asarray(oracle["ww"]),
        "captured_stage1_w": np.asarray(oracle["ww"]),
        "zad_u": round83.owned3(np.asarray(arrays["after_zad_u"])),
        "zad_v": round83.owned3(np.asarray(arrays["after_zad_v"])),
    }
    actives = {
        "carried_kaa_ssh": active["t"][..., 0],
        "direct_shared_w": active_w,
        "captured_stage1_w": active_w,
        "zad_u": active["u"],
        "zad_v": active["v"],
    }

    plant_detail = None
    if args.plant:
        target = {
            "scratch-ulp": "carried_kaa_ssh",
            "w-ulp": "captured_stage1_w",
            "zad-u-ulp": "zad_u",
        }[args.plant]
        changed, at = round84._away_one_ulp(
            references[target], candidates[target], actives[target])
        references[target] = changed
        plant_detail = {"field": target, "location": list(at)}

    rows = {
        name: round87._row(candidates[name], references[name], actives[name],
                           name=name)
        for name in candidates
    }
    confirmed = _confirmed(rows)
    if args.plant:
        require(not confirmed, f"{args.plant} differential control was invisible")
        status = "PLANT_FIRED"
    else:
        status = "CONFIRMED" if confirmed else "REFUTED"
    return {
        "format": "nemo-testcase-l2-gyre-round88-kaa-scratch-v1",
        "status": status,
        "worktree": stamp,
        "round64_admission_status": "PASS",
        "rows": rows,
        "first_nonbit_statement": next(
            (name for name, row in rows.items() if not row["bit_exact"]), None),
        "prediction_confirmed": confirmed,
        "plant": args.plant,
        "plant_detail": plant_detail,
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
    parser.add_argument("--plant", choices=("scratch-ulp", "w-ulp", "zad-u-ulp"))
    args = parser.parse_args(argv)
    report = measure(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND88_KAA_SCRATCH", report["status"])
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
