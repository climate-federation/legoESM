#!/usr/bin/env python3
"""Decision-96 gate for the complete certified G1 statement import."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
)


class GateError(RuntimeError):
    """The full G1 import is incomplete or outside Decision 96."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


MODEL_SELECTORS = (
    "rk3_stage1_rhs_nemo_literal",
    "rk3_stage_velocity_written_order",
    "een_vorticity_written_order",
    "zdf_barotropic_removal_nemo_literal",
    "rk3_fold_exchange_top_v_row",
    "rk3_stage_r3f_nemo_literal",
    "rk3_stage_r3t_nemo_literal",
    "nemo_eos_gdept_rounded_product",
    "rk3_stage_r3uv_nemo_literal",
)
BAROTROPIC_SELECTORS = (
    "barotropic_een_ratio_fold_exchange",
    "barotropic_een_fe3mask_pivot_row",
    "barotropic_een_ratio_written_order",
)


def selected_card(card):
    """Private arm selecting all thirteen certified G1 statements."""

    cfg = card.recipe.model_config._replace(
        **{name: True for name in MODEL_SELECTORS},
        barotropic=card.recipe.model_config.barotropic._replace(
            **{name: True for name in BAROTROPIC_SELECTORS}),
    )
    z_coord = card.recipe.z_coord._replace(nemo_r3t_stored_product=True)
    return card._replace(recipe=card.recipe._replace(
        model_config=cfg, z_coord=z_coord))


def legacy_card(card, *, plant: str = "none"):
    """Disable only the thirteen round-274 statements on the current tree."""

    cfg = card.recipe.model_config
    cfg = cfg._replace(
        **{name: False for name in MODEL_SELECTORS},
        barotropic=cfg.barotropic._replace(
            **{name: False for name in BAROTROPIC_SELECTORS}),
    )
    z_coord = card.recipe.z_coord._replace(nemo_r3t_stored_product=False)
    if plant == "retained-selector":
        cfg = cfg._replace(rk3_stage_r3f_nemo_literal=True)
    retained = [name for name in MODEL_SELECTORS if getattr(cfg, name)]
    retained += [name for name in BAROTROPIC_SELECTORS
                 if getattr(cfg.barotropic, name)]
    retained += (["nemo_r3t_stored_product"]
                 if z_coord.nemo_r3t_stored_product else [])
    require(not retained, f"legacy control retained selectors: {retained}")
    return card._replace(recipe=card.recipe._replace(
        model_config=cfg, z_coord=z_coord))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(rung: str, deck_root: Path, twin_a: Path, label: str,
            candidate_report: Path, expect_commit: str, plant: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(rung in prior.MODULES, f"unknown rung {rung}")
    require(label in prior.LABELS, f"unknown label {label}")
    candidate = json.loads(candidate_report.read_text(encoding="utf-8"))
    require(candidate.get("status") == prior.STATUSES[rung],
            "candidate ladder is not admitted")
    candidate_ladder = candidate if rung == "rung0" else candidate.get(label)
    require(candidate_ladder is not None, "candidate ladder label is absent")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-274 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-274 measurement requires production JIT on CPU")

    selected = selected_card(prior._selected_card(rung, deck_root))
    card = legacy_card(selected, plant=plant)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    state = (rung0.bridge_entry(card, entry)
             if rung == "rung0" or label == "given_nemo_entry"
             else card.recipe.initial_state)
    row_count = int(candidate_ladder["row_count"])
    require(row_count % 20 == 0, "candidate row count is incomplete")
    steps = tuple(range(1, row_count // 20 + 1))
    ladder = omt1._run_ladder(card, twin_a, state, label, steps=steps)
    return {
        "format": "nemo-testcase-l4-orca2-round274-g1-control-v1",
        "status": "PASS_R269_G1_LEGACY_CONTROL",
        "rung": rung,
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "candidate_report": {
            "path": str(candidate_report), "sha256": _sha256(candidate_report),
        },
        "disabled_statements": [*MODEL_SELECTORS, *BAROTROPIC_SELECTORS,
                                "nemo_r3t_stored_product"],
        "baseline": ladder,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung", choices=tuple(prior.MODULES))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--label", choices=prior.LABELS)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=(
        "none", "retained-selector", "exact-loss"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.deck_root, args.twin_a, args.label,
                         args.candidate_report, args.expect_commit)),
                    "measurement arguments are incomplete")
            require(args.plant in ("none", "retained-selector"),
                    "bad measurement plant")
            result = measure(args.rung, args.deck_root, args.twin_a, args.label,
                             args.candidate_report, args.expect_commit,
                             args.plant)
        else:
            require(args.classify is not None,
                    "classification inputs are required")
            require(args.plant in ("none", "exact-loss"),
                    "bad classification plant")
            result = prior.classify(*args.classify, plant=args.plant)
            result["format"] = "nemo-testcase-l4-orca2-round274-g1-decision96-v1"
            result["status"] = "PASS_R274_G1_FULL_IMPORT_DECISION96"
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, prior.GateError, OSError, KeyError, TypeError,
            ValueError, StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
