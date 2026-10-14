#!/usr/bin/env python3
"""Measure one round-274 statement or compiled source unit on rung 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round269_g1_decision96_gate as decision96,
    nemo_testcase_l4_orca2_round274_g1_full_import_gate as round274,
)


class GateError(RuntimeError):
    """The private arm is incomplete or its protocol moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


MODEL = set(round274.MODEL_SELECTORS) | {"zdf_momentum_nemo_face_thickness"}
BAROTROPIC = set(round274.BAROTROPIC_SELECTORS)
COORDINATE = {"nemo_r3t_stored_product"}

ARMS: dict[str, tuple[str, ...]] = {
    "round275_landing": (
        "barotropic_een_ratio_fold_exchange",
        "barotropic_een_fe3mask_pivot_row",
        "nemo_r3t_stored_product",
        "rk3_stage_velocity_written_order",
        "rk3_stage_r3t_nemo_literal",
        "zdf_momentum_nemo_face_thickness",
    ),
    "barotropic_een_fold_pair": (
        "barotropic_een_ratio_fold_exchange",
        "barotropic_een_fe3mask_pivot_row",
    ),
    "barotropic_een_coefficients": (
        "barotropic_een_ratio_fold_exchange",
        "barotropic_een_fe3mask_pivot_row",
        "barotropic_een_ratio_written_order",
    ),
    "rk3_stage_update": (
        "rk3_stage1_rhs_nemo_literal", "rk3_stage_velocity_written_order",
    ),
    "een_stage_f": ("een_vorticity_written_order", "rk3_stage_r3f_nemo_literal"),
    "vertical_residual": ("zdf_barotropic_removal_nemo_literal",),
    "completed_stage_fold": ("rk3_fold_exchange_top_v_row",),
    "live_ratio_depth": (
        "nemo_r3t_stored_product", "rk3_stage_r3t_nemo_literal",
        "nemo_eos_gdept_rounded_product", "rk3_stage_r3uv_nemo_literal",
    ),
    "g2_zdf_divisors": ("zdf_momentum_nemo_face_thickness",),
}
for _name in sorted(MODEL | BAROTROPIC | COORDINATE):
    ARMS[f"only_{_name}"] = (_name,)


def arm_card(card, arm: str, *, plant: str = "none"):
    """Start from the safe round-274 control and select exactly ``arm``."""

    require(arm in ARMS, f"unknown arm {arm}")
    base = round274.legacy_card(round274.selected_card(card))
    expected = set(ARMS[arm])
    names = list(expected)
    if plant == "extra-selector":
        extra = "rk3_stage1_rhs_nemo_literal"
        if extra in names:
            extra = "een_vorticity_written_order"
        names.append(extra)
    unknown = set(names) - MODEL - BAROTROPIC - COORDINATE
    require(not unknown, f"arm has unknown selectors: {sorted(unknown)}")
    config = base.recipe.model_config._replace(
        **{name: True for name in names if name in MODEL},
        barotropic=base.recipe.model_config.barotropic._replace(
            **{name: True for name in names if name in BAROTROPIC}),
    )
    z_coord = base.recipe.z_coord._replace(
        nemo_r3t_stored_product="nemo_r3t_stored_product" in names)
    selected = [name for name in MODEL if getattr(config, name)]
    selected += [name for name in BAROTROPIC if getattr(config.barotropic, name)]
    selected += (["nemo_r3t_stored_product"]
                 if z_coord.nemo_r3t_stored_product else [])
    require(set(selected) == expected,
            f"selected fields differ from arm: {sorted(selected)} != {sorted(expected)}")
    return base._replace(recipe=base.recipe._replace(
        model_config=config, z_coord=z_coord))


def measure(deck_root: Path, twin_a: Path, baseline_report: Path,
            arm: str, expect_commit: str, plant: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    require(baseline.get("status") == "PASS_R269_G1_LEGACY_CONTROL",
            "baseline control is not admitted")
    require(baseline.get("rung") == "rung0"
            and baseline.get("label") == "independent",
            "round 275 requires the independent rung-0 baseline")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-275 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-275 measurement requires production JIT on CPU")

    card = arm_card(decision96._selected_card("rung0", deck_root), arm,
                    plant=plant)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    state = rung0.bridge_entry(card, entry)
    ladder = omt1._run_ladder(
        card, twin_a, state, "independent", steps=tuple(range(1, 11)))
    ladder.update({
        "format": "nemo-testcase-l4-orca2-round275-source-arm-v1",
        "status": "PASS_RUNG0_TEN_STEP_LADDER",
        "worktree": stamp,
        "arm": arm,
        "selected_statements": list(ARMS[arm]),
        "execution": "production-jit-cpu-fp64-x64-libm",
    })
    require(ladder["row_count"] == 200, "arm ladder is incomplete")
    return ladder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("none", "extra-selector"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list_arms:
        print(json.dumps(ARMS, indent=2, sort_keys=True))
        return 0
    try:
        require(args.measure, "--measure is required")
        require(all((args.arm, args.deck_root, args.twin_a,
                     args.baseline_report, args.expect_commit)),
                "measurement arguments are incomplete")
        result = measure(args.deck_root, args.twin_a, args.baseline_report,
                         args.arm, args.expect_commit, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, round274.GateError, OSError, KeyError, TypeError,
            ValueError, StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print("STATUS PASS_R275_SOURCE_ARM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
