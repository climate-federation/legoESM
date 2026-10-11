#!/usr/bin/env python3
"""Measure OMT-4 independently through its admitted 96-step month record."""

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
    nemo_testcase_l4_orca2_round130_rung0_month_gate as month,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)

STEPS = 96
PLANTS = ("none", "claim-label", "initial-entry", "boundary-step")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    report = json.loads(json.dumps(report))
    require(plant in PLANTS, f"unknown plant {plant}")
    if plant == "claim-label":
        report["claim_label"] = "given NEMO's entry"
    elif plant == "initial-entry":
        report["initial_entry"]["T"]["bit_exact"] = False
    elif plant == "boundary-step":
        report["runtime_refusal"] = {"step": 0, "message": "plant"}

    require(report.get("claim_label") == "independent",
            "OMT-4 month claim is not independent")
    require(report.get("initial_mode") == "card_own_state"
            and report.get("decision52_bridge") is None,
            "OMT-4 month used a recorded-entry bridge")
    initial = report.get("initial_entry", {})
    require(tuple(initial) == month.FIELDS
            and all(initial[name]["bit_exact"]
                    and int(initial[name]["unequal"]) == 0
                    for name in month.FIELDS),
            "OMT-4 independent entry is not bit-exact")
    boundary = report.get("first_nonfinite")
    refusal = report.get("runtime_refusal")
    completed = int(report.get("steps_completed", -1))
    require(not (boundary is not None and refusal is not None),
            "OMT-4 month has two terminal boundaries")
    if boundary is None and refusal is None:
        require(completed == STEPS, "OMT-4 stopped before step 96")
        status = "PASS_R237_OMT4_MONTH_COMPLETE"
    elif boundary is not None:
        require(1 <= int(boundary["step"]) <= STEPS,
                "OMT-4 non-finite boundary is outside the protocol")
        require(completed == int(boundary["step"]),
                "OMT-4 non-finite boundary disagrees with completion")
        status = "MEASURED_R237_OMT4_NONFINITE_BOUNDARY"
    else:
        require(1 <= int(refusal["step"]) <= STEPS,
                "OMT-4 refusal boundary is outside the protocol")
        require(completed == int(refusal["step"]) - 1,
                "OMT-4 refusal boundary disagrees with completion")
        status = "MEASURED_R237_OMT4_RUNTIME_REFUSAL"
    report["status"] = status
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
            "OMT-4 month requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "OMT-4 month requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    state = card.recipe.initial_state
    oracle_entry = rung0.assemble_frame(frame_root, 1, 0)
    candidate_entry = rung0.candidate_fields(state)
    initial = {}
    for name in month.FIELDS:
        left = np.asarray(candidate_entry[name], dtype=np.float64)
        right = np.asarray(oracle_entry[name], dtype=np.float64)
        initial[name] = {
            "bit_exact": bool(np.array_equal(left, right)),
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
    completed = 0
    started = time.time()
    for step in range(1, STEPS + 1):
        try:
            pending = model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
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
        "format": "nemo-testcase-l4-orca2-round237-omt4-month-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
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
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
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
