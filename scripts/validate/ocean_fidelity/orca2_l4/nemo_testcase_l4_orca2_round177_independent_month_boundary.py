#!/usr/bin/env python3
"""Measure rung 0 from its own corrected state to its first non-finite step."""

from __future__ import annotations

import argparse
import copy
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
    nemo_testcase_l4_orca2_round130_rung0_month_gate as month,
)

STEPS = 240
EXPECTED_BOUNDARY = {"step": 36, "field": "T", "index": [86, 159, 0]}
PLANTS = ("none", "claim-label", "initial-entry", "boundary-step", "boundary-cell")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Validate protocol invariants while retaining a changed boundary."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "claim-label":
        report["claim_label"] = "given NEMO's entry"
    elif plant == "initial-entry":
        report["initial_entry"]["T"]["bit_exact"] = False
    elif plant == "boundary-step":
        report["first_nonfinite"]["step"] = 0
    elif plant == "boundary-cell":
        report["first_nonfinite"]["index"] = [0]

    require(report.get("claim_label") == "independent",
            "month claim is not independent")
    require(report.get("initial_mode") == "card_own_state"
            and report.get("decision52_bridge") is None,
            "month used a recorded-entry bridge")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report.get("unmeasured_features") == ["linear_implicit_bottom_drag"],
            "rung-0 unmeasured-feature registry moved")
    initial = report.get("initial_entry", {})
    require(tuple(initial) == month.FIELDS
            and all(initial[name]["bit_exact"]
                    and int(initial[name]["unequal"]) == 0
                    for name in month.FIELDS),
            "corrected independent entry is not bit-exact")

    boundary = report.get("first_nonfinite")
    refusal = report.get("runtime_refusal")
    completed = int(report.get("steps_completed", -1))
    require(not (boundary is not None and refusal is not None),
            "month has two terminal boundaries")
    if boundary is None and refusal is None:
        require(completed == STEPS, "finite run stopped before step 240")
    elif boundary is not None:
        require(1 <= int(boundary["step"]) <= STEPS,
                "first non-finite step is outside the month")
        require(boundary["field"] in month.FIELDS,
                "first non-finite field is unregistered")
        require(len(boundary["index"]) in (2, 3),
                "first non-finite cell has the wrong rank")
        require(completed == int(boundary["step"]),
                "completed-step count disagrees with first non-finite boundary")
    else:
        require(1 <= int(refusal["step"]) <= STEPS,
                "runtime-refusal step is outside the month")
        require(completed == int(refusal["step"]) - 1,
                "completed-step count disagrees with runtime refusal")
        require("raw-mesh e3w_int must contain only finite values > 0"
                in str(refusal["message"]),
                "runtime refusal is not the registered live-thickness guard")

    observed = None if boundary is None else {
        "step": int(boundary["step"]),
        "field": str(boundary["field"]),
        "index": [int(value) for value in boundary["index"]],
    }
    report["prediction_ledger"] = {
        "R177-P1": "CONFIRMED",
        "R177-P4": ("CONFIRMED" if observed == EXPECTED_BOUNDARY else "REFUTED"),
        "R177-P6": "CONFIRMED",
    }
    report["predicted_first_nonfinite"] = EXPECTED_BOUNDARY
    report["status"] = (
        "MEASURED_R177_FIRST_NONFINITE" if boundary is not None
        else ("MEASURED_R177_RUNTIME_REFUSAL" if refusal is not None
              else "MEASURED_R177_MONTH_COMPLETE"))
    return report


def measure(deck_root: Path, frame_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-177 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "month requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    oracle_entry = rung0.assemble_frame(frame_root, 1, 0)
    candidate_entry = rung0.candidate_fields(state)
    initial = {}
    for name in month.FIELDS:
        left = np.asarray(candidate_entry[name], dtype=np.float64)
        right = np.asarray(oracle_entry[name], dtype=np.float64)
        initial[name] = {
            "bit_exact": bool(np.array_equal(left, right)),
            # The campaign's frozen exact predicate is np.array_equal; keep
            # its signed-zero convention in the companion census too.
            "unequal": int(np.count_nonzero(left != right)),
        }

    zero = jnp.zeros_like(state.eta.data, dtype=jnp.float64)
    freshwater = FreshwaterForcing(zero, zero, zero, zero, zero)
    surface = OceanSurfaceForcing(
        sw_down=zero, q_net=zero, tau_x=zero, tau_y=zero,
        freshwater=zero, salt_flux=zero, taum=zero,
        tau_i_native=zero, tau_j_native=zero,
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    boundary = None
    refusal = None
    started = time.time()
    completed = 0
    for step in range(1, STEPS + 1):
        try:
            pending = model.step(
                state, card.dt_s,
                freshwater=freshwater, surface_forcing=surface)
            # JAX dispatch is asynchronous.  Synchronise every state leaf so a
            # pure-callback guard is attributed to the step that executed it,
            # not a later dispatch that happened to observe the error token.
            state = jax.device_get(jax.block_until_ready(pending))
        except ValueError as error:
            refusal = {"step": step, "message": str(error)}
            break
        completed = step
        found = month.first_nonfinite(rung0.candidate_fields(state))
        if found is not None:
            boundary = {"step": step, **found}
            break
        if step % 10 == 0:
            print(f"MONTH_PROGRESS step={step}/{STEPS} "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    return classify({
        "format": "nemo-testcase-l4-orca2-round177-month-boundary-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "unmeasured_features": list(card.unmeasured_features),
        "initial_entry": initial,
        "steps_completed": completed,
        "first_nonfinite": boundary,
        "runtime_refusal": refusal,
        "wall_seconds": time.time() - started,
        "worktree": stamp,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.deck_root, args.frame_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
