#!/usr/bin/env python3
"""Split the current rung-0 fold-row HPG in compiled source order."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate as r42,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round181_hpg_component_walk as r181,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round183_hpg_fold_walk as r183,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round180_hpg1_acquisition import (  # noqa: E402
    check_record as component_record,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round181_hpg_fold_acquisition import (  # noqa: E402
    check_record as fold_record,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)

FLOOR = np.float64(2.0e-10)
COMPONENT_ORDER = ("zhpj", "zvap", "pvv")
PRIMITIVE_ORDER = (
    "north_e3w", "north_rhd", "current_e3w", "current_rhd", "r1_e2v",
)
PLANTS = (
    "none", "admission", "target", "source-order", "record-identity",
    "component-selector", "primitive-selector", "dead-arm",
)


class GateError(RuntimeError):
    """The admitted HPG evidence no longer supports the classification."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first(rows: dict[str, dict[str, object]], order) -> dict[str, object] | None:
    for name in order:
        if not rows[name]["at_floor"]:
            return {"boundary": name, **rows[name]}
    return None


def _candidate_north_rhd(rhd: np.ndarray, grid) -> np.ndarray:
    """Reproduce the production T-fold neighbour selected for V HPG."""

    from legoesm.grids.operators_latlon_cgrid import fold_ghost_source_T

    fold = getattr(grid, "fold", None)
    require(fold is not None and bool(fold.is_active), "ORCA2 fold is inactive")
    source = np.asarray(fold_ghost_source_T(np.asarray(rhd), fold)[0])
    return source[np.asarray(fold.perm_T)]


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "admission":
        report["admission"]["fold"] = "FAIL"
    elif plant == "target":
        report["target"]["cells"] -= 1
    elif plant == "source-order":
        report["component_order"][0:2] = reversed(report["component_order"][0:2])
    elif plant == "record-identity":
        report["record_component_identity"]["bit_exact"] = False
    elif plant == "component-selector":
        report["first_component"] = {"boundary": "pvv"}
    elif plant == "primitive-selector":
        report["first_primitive"] = {"boundary": "r1_e2v"}
    elif plant == "dead-arm":
        report["resolved_branch"]["zps_hde_executed"] = True

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(report["admission"] == {
        "fold": "PASS_R181_HPG_FOLD_ADMISSION",
        "component": "PASS_R180_HPG1_ADMISSION",
    }, "HPG records are not admitted")
    require(tuple(report["component_order"]) == COMPONENT_ORDER,
            "component source order moved")
    require(tuple(report["primitive_order"]) == PRIMITIVE_ORDER,
            "primitive source order moved")
    require(report["target"] == {
        "cells": 68, "row": 147, "wet_levels": 1319,
    }, f"registered fold target moved: {report['target']}")
    require(report["record_component_identity"]["bit_exact"],
            "recorded pvv is not recorded zhpj + zvap")
    require(report["record_component_identity"]["at_floor"],
            "recorded component identity left the floor")
    require(report["resolved_branch"] == {
        "name": "hpg_sco", "zps_hde_executed": False,
    }, "resolved HPG branch moved")

    first_component = _first(report["component_rows"], COMPONENT_ORDER)
    first_primitive = _first(report["primitive_rows"], PRIMITIVE_ORDER)
    require(first_component == report["first_component"],
            "first component selector moved")
    require(first_primitive == report["first_primitive"],
            "first primitive selector moved")

    if first_component is None:
        status = "HELD_R240_HPG_EXONERATED_REATTRIBUTE_R239"
    elif first_component["boundary"] == "zhpj":
        require(first_primitive is not None,
                "non-bit zhpj has no non-bit primitive")
        status = "HELD_R240_FIRST_HPG_COMPONENT_ZHPJ"
    elif first_component["boundary"] == "zvap":
        status = "STOPPED_FOR_RECORD_R240_ZVAP_LIVE_DEPTH"
    else:
        raise GateError("pvv differs although both source components are at floor")

    report["predictions"] = {
        "R240-P1": "CONFIRMED",
        "R240-P2": (
            "CONFIRMED_FIRST_ZVAP"
            if first_component is not None and first_component["boundary"] == "zvap"
            else "REFUTED"
        ),
        "R240-P3": (
            "UNMEASURED_WITH_SPEC"
            if first_component is not None and first_component["boundary"] == "zvap"
            else "NOT_ACTIVATED"
        ),
        "R240-P4": "NOT_ACTIVATED",
        "R240-P5": "NOT_ACTIVATED",
        "R240-P6": "PENDING_DOCUMENTARY_GATES",
    }
    report["status"] = status
    return report


def measure(
    deck_root: Path,
    fold_root: Path,
    component_root: Path,
    baseline_root: Path,
    static_root: Path,
    expect_commit: str,
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-240 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-240 measurement requires production JIT on CPU")

    fold_admission = fold_record.run(fold_root, baseline_root, "none")
    component_admission = component_record.run(component_root, baseline_root)
    fold_records = [fold_record.read_record(
        fold_root / f"oracle_r181_hpgfold_rank{rank:04d}_kt00000001.bin")
        for rank in (0, 1)]
    component_records = [component_record.read_record(
        component_root / f"oracle_r180_hpg1_rank{rank:04d}_kt00000001.bin")
        for rank in (0, 1)]

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    model.prime_step_caches(state)
    candidate_inputs = r42._candidate_inputs(model, state)
    candidate_literal = r42._literal_from_inputs(
        candidate_inputs, card.recipe.model_config.g, grid=card.recipe.grid)

    oracle_static, _ = r170.assemble_record(static_root)
    oracle_vmask = np.asarray(oracle_static["vmask"])[..., :30]
    target = np.zeros(oracle_vmask.shape[:2], dtype=bool)
    target[-1] = np.any(oracle_vmask[-1] != 0.0, axis=-1)
    contributing = np.broadcast_to(target[..., None], oracle_vmask.shape) & (
        oracle_vmask != 0.0)
    target_summary = {
        "cells": int(np.count_nonzero(target)),
        "row": int(np.flatnonzero(np.any(target, axis=1))[0]),
        "wet_levels": int(np.count_nonzero(contributing)),
    }

    oracle_components = {
        "zhpj": r181._assemble(component_records, "zhpi_v")[..., :30],
        "zvap": r181._assemble(component_records, "zuap_v")[..., :30],
        "pvv": r181._assemble(component_records, "sum_v")[..., :30],
    }
    candidate_components = {
        "zhpj": r83.native_v(candidate_literal["zhpi_v"]),
        "zvap": r83.native_v(candidate_literal["zuap_v"]),
        "pvv": r83.native_v(candidate_literal["sum_v"]),
    }
    component_rows = {
        name: r181._score(candidate_components[name], oracle_components[name],
                          contributing)
        for name in COMPONENT_ORDER
    }
    component_identity = r181._score(
        oracle_components["zhpj"] + oracle_components["zvap"],
        oracle_components["pvv"], contributing)

    fold = {name: r181._assemble(fold_records, name)[..., :30]
            for name in fold_record.NAMES}
    oracle_metric = r181._assemble(component_records, "r1_e2v")[..., 0]
    north_rhd = _candidate_north_rhd(candidate_inputs["rhd"], card.recipe.grid)
    candidate_primitive = {
        "north_e3w": np.roll(candidate_inputs["e3w"], -1, axis=0),
        "north_rhd": np.broadcast_to(north_rhd[None, ...],
                                      candidate_inputs["rhd"].shape),
        "current_e3w": candidate_inputs["e3w"],
        "current_rhd": candidate_inputs["rhd"],
    }
    oracle_primitive = {
        "north_e3w": fold["north_e3w"],
        "north_rhd": fold["north_rhd"],
        "current_e3w": fold["current_e3w"],
        "current_rhd": fold["current_rhd"],
    }
    primitive_rows = {
        name: r181._score(candidate_primitive[name], oracle_primitive[name],
                          contributing)
        for name in PRIMITIVE_ORDER[:-1]
    }
    primitive_rows["r1_e2v"] = r181._score(
        candidate_inputs["r1_e2v"], oracle_metric, target)

    return classify({
        "format": "nemo-testcase-l4-orca2-round240-hpg-split-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR),
        "worktree": stamp,
        "admission": {
            "fold": fold_admission["status"],
            "component": component_admission["status"],
        },
        "component_order": list(COMPONENT_ORDER),
        "primitive_order": list(PRIMITIVE_ORDER),
        "target": target_summary,
        "record_component_identity": component_identity,
        "component_rows": component_rows,
        "first_component": _first(component_rows, COMPONENT_ORDER),
        "primitive_rows": primitive_rows,
        "first_primitive": _first(primitive_rows, PRIMITIVE_ORDER),
        "resolved_branch": {"name": "hpg_sco", "zps_hde_executed": False},
        "compiled_source": {
            "surface": "ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-427",
            "interior": "ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-463",
            "dispatch": "ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:117-123",
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--fold-root", type=Path)
    parser.add_argument("--component-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            required = (args.deck_root, args.fold_root, args.component_root,
                        args.baseline_root, args.static_root, args.expect_commit)
            require(all(value is not None for value in required),
                    "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.fold_root, args.component_root,
                args.baseline_root, args.static_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            fold_record.Refusal, component_record.Refusal,
            rung0.GateError, r170.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
