#!/usr/bin/env python3
"""Continue the rung-0 step-36 walk after its finite transport operands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round131_step16_walk_gate as prior,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round133_step36_operand_gate as operands_gate,
)


BOUNDARY_ORDER = (
    "stage1_after_advection",
    "stage1_after_sbc",
    "stage1_tracer",
    "stage2_tracer",
    "stage3_advection_content",
    "pre_implicit_content",
    "pre_implicit_concentration",
    "returned_state",
)
FIELDS = prior.FIELDS
TARGET = operands_gate.TARGET
PLANTS = ("none", "operand-replay", "passivity")


class GateError(RuntimeError):
    """The downstream step-36 walk violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_boundary(report: dict[str, object]) -> str | None:
    return next((name for name in BOUNDARY_ORDER
                 if report["boundaries"][name]["nonfinite_total"]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "operand-replay":
        report["operand_replay"]["stage1_metric_transport"] = 1
    elif plant == "passivity":
        report["ordinary_repeat_state_equal"]["T"] = False

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "downstream walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("boundary_order", ())) == BOUNDARY_ORDER,
            "downstream boundary order changed")
    require(tuple(report.get("operand_replay", ()))
            == operands_gate.OPERAND_ORDER,
            "operand replay order changed")
    require(all(int(value) == 0
                for value in report["operand_replay"].values()),
            "the committed finite-operand result did not replay")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in FIELDS},
        "ordinary step-36 repeat changed bits")

    for name in BOUNDARY_ORDER:
        row = report["boundaries"][name]
        require(sum(int(value) for value in row["nonfinite"].values())
                == int(row["nonfinite_total"]),
                f"{name}: non-finite census disagrees")
        require((row["first_nonfinite"] is None)
                == (row["nonfinite_total"] == 0),
                f"{name}: first-nonfinite/census disagreement")
        if name != "returned_state":
            require(row.get("support") == "active_t_cells"
                    and int(row.get("active_count", 0)) > 0,
                    f"{name}: tracer support is not the active T-cell mask")
            require(set(row.get("nonfinite_unscored", ())) == {"T", "S"},
                    f"{name}: dry-cell census is incomplete")
    first = _first_boundary(report)
    require(first == report.get("first_nonfinite_boundary"),
            "first downstream non-finite boundary is not source ordered")
    require(report["boundaries"]["returned_state"]["first_nonfinite"] == {
        "field": "T", "index": list(TARGET), "value": "nan"},
        "round-132 step-36 failure did not reproduce")

    predicted = "stage3_advection_content"
    first_index = BOUNDARY_ORDER.index(first) if first is not None else None
    later_all_nonfinite = (
        first_index is not None
        and all(report["boundaries"][name]["nonfinite_total"]
                for name in BOUNDARY_ORDER[first_index:]))
    predictions = {
        "R133-P6": {
            "status": "CONFIRMED" if first == predicted else "REFUTED",
            "predicted": predicted, "observed": first,
        },
        "R133-P7": {
            "status": "CONFIRMED" if later_all_nonfinite else "REFUTED",
            "observed": [name for name in BOUNDARY_ORDER
                         if report["boundaries"][name]["nonfinite_total"]],
        },
    }
    return {**report, "status": "PASS_ROUND133_STEP36_DOWNSTREAM_WALK",
            "prediction_ledger": predictions}


def _operand_replay(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text())
    require(raw.get("status") == "PASS_ROUND133_STEP36_OPERAND_WALK",
            "operand report is not admitted")
    require(raw.get("first_nonfinite_operand") is None,
            "operand report no longer says every operand is finite")
    return {
        name: int(raw["operands"][name]["nonfinite_total"])
        for name in operands_gate.OPERAND_ORDER
    }


def measure(deck_root: Path, operand_report: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    operand_replay = _operand_replay(operand_report)
    require(all(value == 0 for value in operand_replay.values()),
            "operand report contains a non-finite row")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-133 continuation worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-133 continuation commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "downstream walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    freshwater, surface = ladder._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))

    def model(hooks=None):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=(hooks or _NEMOWSRK3TestHooks()))

    ordinary_model = model()
    state = card.recipe.initial_state
    started = time.time()
    for step in range(1, 36):
        state = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        require(prior.boundary_summary(prior._state_arrays(state))[
                    "nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"STEP36_DOWNSTREAM_PROGRESS step={step}/35 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state35 = state
    returned = jax.device_get(ordinary_model.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    returned_repeat = jax.device_get(ordinary_model.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    def exposed(hooks):
        return jax.device_get(model(hooks).step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    after_advection1 = exposed(_NEMOWSRK3TestHooks(
        expose_tracer_stage1_boundary="after_advection"))
    after_sbc1 = exposed(_NEMOWSRK3TestHooks(
        expose_tracer_stage1_boundary="after_sbc"))
    stage1 = exposed(_NEMOWSRK3TestHooks(expose_tracer_stage=1))
    stage2 = exposed(_NEMOWSRK3TestHooks(expose_tracer_stage=2))
    adv_content = exposed(_NEMOWSRK3TestHooks(
        expose_stage3_advection_content=True))
    pre_content = exposed(_NEMOWSRK3TestHooks(
        expose_pre_implicit_content=True))
    pre_concentration = exposed(_NEMOWSRK3TestHooks(
        expose_pre_implicit_state=True))

    active_t = np.asarray(card.recipe.z_coord.is_active, dtype=bool)

    def tracer_summary(state_value):
        raw = {
            "T": np.asarray(state_value.T.data),
            "S": np.asarray(state_value.S.data),
        }
        row = prior.boundary_summary({
            name: np.where(active_t, values, 0.0)
            for name, values in raw.items()})
        row["support"] = "active_t_cells"
        row["active_count"] = int(np.count_nonzero(active_t))
        row["nonfinite_unscored"] = {
            name: int(np.count_nonzero(~np.isfinite(values[~active_t])))
            for name, values in raw.items()
        }
        return row

    boundaries = {
        "stage1_after_advection": tracer_summary(after_advection1),
        "stage1_after_sbc": tracer_summary(after_sbc1),
        "stage1_tracer": tracer_summary(stage1),
        "stage2_tracer": tracer_summary(stage2),
        "stage3_advection_content": tracer_summary(adv_content),
        "pre_implicit_content": tracer_summary(pre_content),
        "pre_implicit_concentration": tracer_summary(pre_concentration),
        "returned_state": prior.boundary_summary(prior._state_arrays(returned)),
    }
    first = next((name for name in BOUNDARY_ORDER
                  if boundaries[name]["nonfinite_total"]), None)
    return {
        "format": "nemo-testcase-l4-orca2-round133-step36-downstream-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "unmeasured_features": list(card.unmeasured_features),
        "operand_report": str(operand_report),
        "operand_replay": operand_replay,
        "boundary_order": list(BOUNDARY_ORDER),
        "boundaries": boundaries,
        "first_nonfinite_boundary": first,
        "ordinary_repeat_state_equal": prior.state_bit_rows(
            returned_repeat, returned),
        "instrument_limit": (
            "existing write-only tracer slots expose the source-ordered "
            "boundaries; the ordinary step is separately repeated bitwise"),
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "stage_order": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227",
            "stage1_transport":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:265-284",
            "stage1_advection":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:633-645",
            "vertical_solve":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:97-107",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--operand-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.operand_report is None
                    and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.operand_report is not None
                    and args.expect_commit,
                    "runtime mode requires deck, operand report, and commit")
            raw = measure(args.deck_root, args.operand_report, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, operands_gate.GateError, prior.GateError,
            rung0.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND133_STEP36_DOWNSTREAM_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
