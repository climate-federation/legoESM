#!/usr/bin/env python3
"""Split OMT-4 kt=8 substep-3 midpoint SSH across recorded operands."""

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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round253_kt8_boundary_gate as r253,
)

LABELS = ("independent", "given_nemo_entry")
OPERANDS = ("sshn_e", "sshb_e", "sshbb_e", "za1", "za2", "za3")
PLANTS = (
    "none", "label-coverage", "south-refutation", "source-order",
    "nemo-replay", "candidate-replay", "first-nonbit", "one-bit",
)


class GateError(RuntimeError):
    """The midpoint operand split cannot support its frozen claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-254 midpoint split requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-254 midpoint split requires production JIT on CPU")
    return stamp


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _midpoint(coefficients, now, before, before_before) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_midpoint_extrapolation,
    )

    return np.asarray(jax.device_get(nemo_literal_midpoint_extrapolation(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(now, dtype=jnp.float64),
        jnp.asarray(before, dtype=jnp.float64),
        jnp.asarray(before_before, dtype=jnp.float64),
    )), np.float64)


def _first_nonbit(rows: dict[str, dict[str, object]]) -> str | None:
    return next((name for name in OPERANDS if rows[name]["unequal"]), None)


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            south_report: Path, label: str, expect_commit: str) -> dict:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSBarotropicTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    south = json.loads(south_report.read_text())
    require(south["label"] == label, "south audit claim label moved")
    require(south["status"] == "PASS_R254_SOUTH_ASSOCIATION_MEASUREMENT"
            and south["south_changed_fields"] == [],
            "south association hypothesis was not refuted")

    oracle, census = r97.assemble_record(
        spg_root, prefix="oracle_r166_spg", expected_kt=8,
        expected_magic="NEMO_L4_R166SPG")
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    complete = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r253._hooks(card))
    observer = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r253._hooks(card, substeps=True))

    arm = r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
    with arm:
        for kt in range(1, 8):
            state = jax.device_get(complete.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
            print(f"PROGRESS round254-midpoint {label} kt={kt}",
                  file=sys.stderr, flush=True)
        trace = jax.device_get(observer.step(
            state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    require(isinstance(trace, _NEMOWSBarotropicTrace),
            "complete-unit substep trace return type moved")

    substep = 2  # zero-based substep 3, round 253's first invalid writer.
    candidate = {
        "sshn_e": np.asarray(trace.substeps["eta_entry"][substep], np.float64),
        "sshb_e": np.asarray(trace.substeps["eta_history_b"][substep], np.float64),
        "sshbb_e": np.asarray(trace.substeps["eta_history_bb"][substep], np.float64),
        "za1": np.asarray(trace.substeps["mid_weight_1"][substep], np.float64),
        "za2": np.asarray(trace.substeps["mid_weight_2"][substep], np.float64),
        "za3": np.asarray(trace.substeps["mid_weight_3"][substep], np.float64),
    }
    # NEMO rotates after each completed substep at dynspg_ts.f90:780-792.
    # Therefore the substep-3 histories are j002 exit, j001 exit, and the
    # entry NOW field respectively; the record carries all three explicitly.
    coefficients = np.asarray(oracle["j003_ext_coef"], np.float64)
    nemo = {
        "sshn_e": np.asarray(oracle["j002_ssha_e"], np.float64),
        "sshb_e": np.asarray(oracle["j001_ssha_e"], np.float64),
        "sshbb_e": np.asarray(oracle["i000_sshn_e"], np.float64),
        "za1": coefficients[0], "za2": coefficients[1],
        "za3": coefficients[2],
    }
    rows = {name: _score(candidate[name], nemo[name]) for name in OPERANDS}
    candidate_coefficients = np.asarray(
        [candidate["za1"], candidate["za2"], candidate["za3"]], np.float64)
    candidate_replay = _midpoint(
        candidate_coefficients, candidate["sshn_e"], candidate["sshb_e"],
        candidate["sshbb_e"])
    nemo_replay = _midpoint(
        coefficients, nemo["sshn_e"], nemo["sshb_e"], nemo["sshbb_e"])
    candidate_output = np.asarray(trace.substeps["eta_mid"][substep], np.float64)
    nemo_output = np.asarray(oracle["j003_sshp2_mid"], np.float64)

    substitutions = {}
    for name in OPERANDS:
        trial = dict(candidate)
        trial[name] = nemo[name]
        trial_coefficients = np.asarray(
            [trial["za1"], trial["za2"], trial["za3"]], np.float64)
        trial_output = _midpoint(
            trial_coefficients, trial["sshn_e"], trial["sshb_e"],
            trial["sshbb_e"])
        substitutions[name] = _score(trial_output, nemo_output)

    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round254-midpoint-operands-v1",
        "status": "PASS_R254_MIDPOINT_OPERAND_SPLIT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved",
        "worktree": stamp,
        "south_hypothesis": "REFUTED_ZERO_SOUTH_CHANGES",
        "record_census": census,
        "source_order": list(OPERANDS),
        "operand_rows": rows,
        "first_nonbit_operand": _first_nonbit(rows),
        "candidate_replay": _score(candidate_replay, candidate_output),
        "nemo_replay": _score(nemo_replay, nemo_output),
        "candidate_output_vs_nemo": _score(candidate_output, nemo_output),
        "one_variable_substitutions": substitutions,
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
        "compiled_source": {
            "midpoint_coefficients": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:455-468",
            "midpoint_ssh": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:489-493",
            "history_rotation": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:780-792",
            "candidate_midpoint": "barotropic_latlon_cgrid.py:2006-2025,4039-4054",
        },
    }


def classify(reports: list[dict[str, object]], plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "south-refutation":
        independent["south_hypothesis"] = "CONFIRMED"
    elif plant == "source-order":
        independent["source_order"].reverse()
    elif plant == "nemo-replay":
        independent["nemo_replay"]["unequal"] = 1
    elif plant == "candidate-replay":
        independent["candidate_replay"]["unequal"] = 1
    elif plant == "first-nonbit":
        independent["first_nonbit_operand"] = None
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    firsts = set()
    summaries = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R254_MIDPOINT_OPERAND_SPLIT",
                f"{label}: measurement status moved")
        require(report["south_hypothesis"] == "REFUTED_ZERO_SOUTH_CHANGES",
                f"{label}: south hypothesis disposition moved")
        require(report["operand_route"] == "production-d114-resolved",
                f"{label}: stale operand route")
        require(report["source_order"] == list(OPERANDS),
                f"{label}: midpoint operand order moved")
        require(report["candidate_replay"]["unequal"] == 0,
                f"{label}: candidate midpoint self-replay is not exact")
        require(report["nemo_replay"]["unequal"] == 0,
                f"{label}: NEMO midpoint replay is not exact")
        first = report["first_nonbit_operand"]
        require(first in OPERANDS, f"{label}: no first non-bit operand")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control went vacuous")
        firsts.add(first)
        summaries[label] = {
            "first_nonbit_operand": first,
            "operand_rows": report["operand_rows"],
            "candidate_output_vs_nemo": report["candidate_output_vs_nemo"],
            "one_variable_substitutions": report["one_variable_substitutions"],
        }
    require(len(firsts) == 1,
            "claim labels disagree on the first non-bit midpoint operand")
    return {
        "format": "nemo-testcase-l4-orca2-round254-classification-v1",
        "status": "HELD_R254_MIDPOINT_OPERAND_NAMED",
        "south_hypothesis": "REFUTED_ZERO_SOUTH_CHANGES",
        "first_nonbit_operand": next(iter(firsts)),
        "summaries": summaries,
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--south-report", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path, action="append", default=[])
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.south_report, args.label, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.south_report, args.label, args.expect_commit)
        else:
            require(len(args.report_in) == 2,
                    "classification requires exactly two reports")
            result = classify(
                [json.loads(path.read_text()) for path in args.report_in],
                plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError,
            rung0.GateError, r97.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
