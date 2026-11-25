#!/usr/bin/env python3
"""Measure the resolved-stage-mask / completed-stage-V-fold atomic pair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)

FIELDS = ("T", "S", "u", "v", "ssh")
ARMS = ("resolved-mask-only", "resolved-mask-plus-top-v")
PLANTS = ("none", "one-bit", "missing-single-failure", "fake-pair-pass")
EXPECTED_REFUSAL = "raw-mesh e3w_int must contain only finite values > 0"
FLOOR = np.float64(2.0e-10)


class GateError(RuntimeError):
    """A registered arm, stage boundary, or non-vacuity control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score_boundary(card, state, expected, kt: int, stage: int) -> dict:
    actual = rung0.candidate_fields(state)
    comparison = rung0.ladder.compare_fields(actual, expected)
    for field in FIELDS:
        require(bool(np.all(np.isfinite(actual[field]))),
                f"kt={kt} stage={stage} {field}: candidate is non-finite")
        require(bool(np.all(np.isfinite(expected[field]))),
                f"kt={kt} stage={stage} {field}: oracle is non-finite")
    return {"kt": kt, "stage": stage, **comparison}


def _nonfinite_fields(state) -> dict[str, dict[str, object]]:
    values = rung0.candidate_fields(state)
    result = {}
    for field in FIELDS:
        finite = np.isfinite(values[field])
        if not bool(np.all(finite)):
            first = tuple(int(index) for index in np.argwhere(~finite)[0])
            result[field] = {
                "count": int(np.count_nonzero(~finite)),
                "first_index": list(first),
            }
    return result


def _arm_card(deck_root: Path, arm: str):
    require(arm in ARMS, f"unknown arm {arm!r}")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    cfg = card.recipe.model_config
    require(not cfg.rk3_fold_exchange_top_v_row,
            "production rung-0 card unexpectedly selects top-V exchange")
    if arm == "resolved-mask-plus-top-v":
        card = card._replace(recipe=card.recipe._replace(
            model_config=cfg._replace(rk3_fold_exchange_top_v_row=True)))
    return card


def measure(deck_root: Path, record_root: Path, arm: str,
            expect_commit: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-288 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-288 measurement requires production JIT on CPU")

    admission = ladder.frames.admit(
        record_root, ladder.EXPECTED_PRODUCER, None)
    require(admission["record_count"] == 80,
            "rung-0 frame record census moved")
    card = _arm_card(deck_root, arm)
    state = card.recipe.initial_state
    freshwater, surface = ladder._zero_forcing((148, 180))
    stage_models = {
        stage: LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_stage=stage, expose_tracer_stage=stage))
        for stage in (1, 2)
    }
    final_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    boundaries: list[dict] = []
    terminal = None
    for kt in range(1, 11):
        boundaries.append(_score_boundary(
            card, state, rung0.assemble_frame(record_root, kt, 0), kt, 0))
        candidates = []
        for stage in (1, 2):
            try:
                stage_state = jax.device_get(
                    stage_models[stage].step(
                        state, card.dt_s, freshwater=freshwater,
                        surface_forcing=surface))
                nonfinite = _nonfinite_fields(stage_state)
                if nonfinite:
                    terminal = {
                        "kt": kt, "stage": stage,
                        "error": "completed stage contains non-finite values",
                        "nonfinite_fields": nonfinite,
                    }
                    break
                candidates.append((stage, stage_state))
            except Exception as error:  # the registered fail-closed boundary
                terminal = {"kt": kt, "stage": stage, "error": str(error)}
                break
        if terminal is None:
            try:
                state_after = jax.device_get(final_model.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
                candidates.append((3, state_after))
            except Exception as error:  # the registered fail-closed boundary
                terminal = {"kt": kt, "stage": 3, "error": str(error)}
                state_after = None
        else:
            state_after = None
            try:
                jax.device_get(final_model.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
            except Exception as error:
                terminal["ordinary_error"] = str(error)
        for stage, stage_state in candidates:
            boundaries.append(_score_boundary(
                card, stage_state,
                rung0.assemble_frame(record_root, kt, stage), kt, stage))
        print(f"PROGRESS round288 arm={arm} kt={kt} "
              f"stages={[stage for stage, _ in candidates]}",
              file=sys.stderr, flush=True)
        if terminal is not None:
            break
        state = state_after

    one = np.asarray([1.0], np.float64)
    one_bit = rung0.ladder.score(one, np.nextafter(one, np.inf))
    return {
        "format": "nemo-testcase-l4-orca2-round288-stage-table-v1",
        "status": "PASS_R288_STAGE_TABLE",
        "claim_label": "independent_rung0",
        "arm": arm,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "admission": {"status": admission["status"],
                      "record_count": admission["record_count"]},
        "compiled_sources": {
            "mask": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232",
            "update": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480",
            "association": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820",
        },
        "boundaries": boundaries,
        "terminal": terminal,
        "one_bit_control": one_bit,
    }


def _row(report: dict, kt: int, stage: int, field: str) -> dict:
    boundary = next(row for row in report["boundaries"]
                    if (row["kt"], row["stage"]) == (kt, stage))
    return boundary["rows"][field]


def _growth(report: dict) -> list[dict]:
    rows = []
    previous = None
    for boundary in report["boundaries"]:
        maximum = max(float(boundary["rows"][field]["max_abs"])
                      for field in FIELDS)
        factor = (None if previous is None else
                  float("inf") if previous <= FLOOR and maximum > FLOOR else
                  maximum / max(previous, float(FLOOR)))
        rows.append({"kt": boundary["kt"], "stage": boundary["stage"],
                     "max_abs": maximum, "growth_factor": factor})
        previous = maximum
    return rows


def classify(control: dict, single: dict, pair: dict,
             plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    control, single, pair = map(copy.deepcopy, (control, single, pair))
    if plant == "one-bit":
        pair["one_bit_control"]["unequal"] = 0
    elif plant == "missing-single-failure":
        single["terminal"] = None
    elif plant == "fake-pair-pass":
        pair["terminal"] = None
        pair["boundaries"] = single["boundaries"]

    require(control["status"] == "PASS_R276_SOURCE_LADDER"
            and control["arm"] == "control"
            and control["rung"] == "rung0"
            and control["label"] == "independent"
            and control["ladder"]["row_count"] == 200,
            "production control no longer completes 200 rows")
    for report, arm in ((single, ARMS[0]), (pair, ARMS[1])):
        require(report["status"] == "PASS_R288_STAGE_TABLE"
                and report["claim_label"] == "independent_rung0"
                and report["arm"] == arm, f"{arm}: report identity moved")
        require(report["admission"]["record_count"] == 80,
                f"{arm}: frame admission moved")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{arm}: one-bit control did not fire")

    single_error = " ".join((
        single["terminal"].get("error", "") if single["terminal"] else "",
        single["terminal"].get("ordinary_error", "")
        if single["terminal"] else "",
    ))
    require(single["terminal"] is not None
            and single["terminal"]["kt"] == 4
            and EXPECTED_REFUSAL in single_error,
            "resolved-mask single-arm refusal moved")
    pair_completed = pair["terminal"] is None and any(
        row["kt"] == 10 and row["stage"] == 3 for row in pair["boundaries"])
    pair_stage1_v = _row(pair, 1, 1, "v")
    single_stage1_v = _row(single, 1, 1, "v")
    pair_toward = float(pair_stage1_v["rms"]) < float(single_stage1_v["rms"])
    growth = _growth(pair)
    first_growth = next((row for row in growth
                         if row["growth_factor"] is not None
                         and row["growth_factor"] > 10.0), None)
    exact_losses = []
    first_pair_effect = None
    control_rows = {(row["kt"], row["checkpoint"], row["field"]): row
                    for row in control["ladder"]["rows"]}
    single_boundaries = {
        (row["kt"], row["stage"]): row for row in single["boundaries"]
    }
    for boundary in pair["boundaries"]:
        checkpoint = "entry" if boundary["stage"] == 0 else f"stage{boundary['stage']}"
        for field in FIELDS:
            old = control_rows.get((boundary["kt"], checkpoint, field))
            new = boundary["rows"][field]
            if (boundary["stage"] > 0 and old and old["bit_identical"]
                    and not new["bit_identical"]):
                exact_losses.append({"kt": boundary["kt"],
                                     "checkpoint": checkpoint,
                                     "field": field})
            single_boundary = single_boundaries.get(
                (boundary["kt"], boundary["stage"]))
            if single_boundary is not None and first_pair_effect is None:
                prior = single_boundary["rows"][field]
                signature = ("unequal", "max_abs", "rms")
                if any(prior[name] != new[name] for name in signature):
                    first_pair_effect = {
                        "kt": boundary["kt"], "checkpoint": checkpoint,
                        "field": field, "single": prior, "pair": new,
                    }
    result = {
        "format": "nemo-testcase-l4-orca2-round288-pair-classification-v1",
        "status": ("QUALIFIED_R288_FULL_GATES" if pair_completed and pair_toward
                   and not exact_losses else "HELD_R288_PAIR"),
        "claim_label": "independent_rung0",
        "control_rows": 200,
        "single_terminal": single["terminal"],
        "pair_terminal": pair["terminal"],
        "pair_completed_kt10": pair_completed,
        "kt1_stage1_v": {"single": single_stage1_v, "pair": pair_stage1_v,
                         "pair_toward": pair_toward},
        "exact_row_losses": exact_losses,
        "first_pair_effect": first_pair_effect,
        "growth": growth,
        "first_over_10x_growth": first_growth,
        "prediction_ledger": {
            "R288-P1": "CONFIRMED",
            "R288-P2": "CONFIRMED" if pair_completed else "REFUTED",
            "R288-P3": ("CONFIRMED" if pair_toward and not exact_losses
                        else "REFUTED"),
            "R288-P4": ("READY_FOR_FULL_GATES" if pair_completed and pair_toward
                        and not exact_losses else "OFF_CLAIM"),
            "R288-P5": "CONFIRMED",
        },
    }
    require(plant == "none", f"{plant} plant stayed green")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=3, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.record_root, args.arm,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.record_root, args.arm,
                             args.expect_commit)
        else:
            require(args.classify is not None, "classification inputs missing")
            documents = [json.loads(path.read_text(encoding="utf-8"))
                         for path in args.classify]
            result = classify(*documents, plant=args.plant)
    except (GateError, ladder.GateError, rung0.GateError, OSError, KeyError,
            TypeError, ValueError, IndexError, StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
