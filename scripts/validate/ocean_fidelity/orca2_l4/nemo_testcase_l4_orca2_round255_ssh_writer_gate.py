#!/usr/bin/env python3
"""Split OMT-4 kt=8 substep-2 SSH across writer boundaries and operands."""

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
BOUNDARIES = ("continuity_output", "post_association", "history_rotation")
OPERANDS = (
    "sshn_e", "zhU", "zhV", "r1_e1e2t", "zhdiv", "ssh_frc",
    "rDt_e", "ssmask",
)
PLANTS = (
    "none", "label-coverage", "round254-dependency", "source-order",
    "nemo-replay", "candidate-replay", "boundary-identity",
    "first-boundary", "first-operand", "one-bit",
)


class GateError(RuntimeError):
    """The substep-2 SSH split cannot support its frozen claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-255 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-255 measurement requires production JIT on CPU")
    return stamp


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _first_nonbit(order: tuple[str, ...], rows: dict[str, dict[str, object]]):
    return next((name for name in order if rows[name]["unequal"]), None)


def _continuity_terms(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replay dynspg_ts.f90:550-558 in its explicit fp64 associations."""
    zh_u = np.asarray(inputs["zhU"], np.float64)
    zh_v = np.asarray(inputs["zhV"], np.float64)
    du = np.subtract(zh_u, np.roll(zh_u, 1, axis=1))
    south = np.concatenate((np.zeros_like(zh_v[:1]), zh_v[:-1]), axis=0)
    dv = np.subtract(zh_v, south)
    flux_sum = np.add(du, dv)
    divergence = np.multiply(flux_sum, inputs["r1_e1e2t"])
    rhs = np.add(inputs["ssh_frc"], divergence)
    increment = np.multiply(inputs["rDt_e"], rhs)
    raw = np.subtract(inputs["sshn_e"], increment)
    output = np.multiply(raw, inputs["ssmask"])
    return {
        "du": du, "dv": dv, "zhdiv": divergence, "rhs": rhs,
        "increment": increment, "raw_output": raw, "output": output,
    }


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            midpoint_report: Path, label: str, expect_commit: str) -> dict:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSBarotropicTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    midpoint = json.loads(midpoint_report.read_text())
    require(midpoint["label"] == label, "round-254 claim label moved")
    require(midpoint["status"] == "PASS_R254_MIDPOINT_OPERAND_SPLIT"
            and midpoint["first_nonbit_operand"] == "sshn_e",
            "round-254 midpoint dependency moved")

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
    # The boundary-association hook continues through the kt=8 stage and hits
    # the already-registered live-W guard.  Reuse round 254's passive pattern:
    # return the association-off substep trace, then replay the one seven-array
    # statement offline below.
    off_hooks = r253._hooks(card, substeps=True)._replace(
        barotropic_atomic_fold_unit=False,
        barotropic_external_mode_association=False,
    )
    observer = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=off_hooks)
    arm = r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
    with arm:
        for kt in range(1, 8):
            state = jax.device_get(complete.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
            print(f"PROGRESS round255 {label} kt={kt}",
                  file=sys.stderr, flush=True)
        trace = jax.device_get(observer.step(
            state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    require(isinstance(trace, _NEMOWSBarotropicTrace),
            "complete-unit substep trace return type moved")

    substep = 1  # zero-based substep 2; its exit is substep 3's sshn_e.
    target = np.asarray(oracle["j002_ssha_e"], np.float64)
    before_association = {
        name: np.asarray(trace.substeps[trace_name][substep], np.float64)
        for name, trace_name in (
            ("u", "u_exit"), ("v", "v_exit"),
            ("depth_u", "face_depth_u_exit"),
            ("depth_v", "face_depth_v_exit"),
            ("inverse_u", "r1_face_depth_u_exit"),
            ("inverse_v", "r1_face_depth_v_exit"),
            ("eta", "eta_continuity"),
        )
    }
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )
    post = jax.device_get(_nemo_external_mode_boundary_association(
        *(before_association[name] for name in (
            "u", "v", "depth_u", "depth_v", "inverse_u", "inverse_v",
            "eta")),
        card.recipe.grid,
    ))
    boundary_values = {
        "continuity_output": np.asarray(
            trace.substeps["eta_continuity"][substep], np.float64),
        "post_association": np.asarray(post[6], np.float64),
        "history_rotation": np.asarray(
            trace.substeps["eta_entry"][substep + 1], np.float64),
    }
    boundary_rows = {
        name: _score(value, target) for name, value in boundary_values.items()
    }
    boundary_identity = {
        "continuity_to_association": _score(
            boundary_values["continuity_output"],
            boundary_values["post_association"]),
        "association_to_rotation": _score(
            boundary_values["post_association"],
            boundary_values["history_rotation"]),
    }

    raw_area = np.asarray(card.recipe.z_coord.nemo_e1e2t, np.float64)
    require(raw_area.shape == target.shape and np.all(raw_area > 0.0),
            "resolved raw T-cell area is absent or invalid")
    mask = np.asarray(card.recipe.initial_state.land_mask.data, np.float64)
    candidate_inputs = {
        "sshn_e": np.asarray(trace.substeps["eta_entry"][substep], np.float64),
        "zhU": r97._native_u(trace.substeps["transport_metric_u"][substep]),
        "zhV": r97._native_v(trace.substeps["transport_metric_v"][substep]),
        "r1_e1e2t": np.reciprocal(raw_area),
        "ssh_frc": np.asarray(
            trace.substeps["continuity_forcing"][substep], np.float64),
        "rDt_e": np.asarray(card.dt_s / int(oracle["i000_entry_sc"][3]),
                             np.float64),
        "ssmask": mask,
    }
    reference_inputs = {
        "sshn_e": np.asarray(oracle["j001_ssha_e"], np.float64),
        "zhU": np.asarray(oracle["j002_zhU"], np.float64),
        "zhV": np.asarray(oracle["j002_zhV"], np.float64),
        "r1_e1e2t": np.reciprocal(raw_area),
        "ssh_frc": np.asarray(oracle["i000_ssh_frc"], np.float64),
        "rDt_e": np.asarray(oracle["i000_entry_sc"][0], np.float64),
        "ssmask": mask,
    }
    candidate_terms = _continuity_terms(candidate_inputs)
    reference_terms = _continuity_terms(reference_inputs)
    candidate_inputs["zhdiv"] = np.asarray(
        trace.substeps["continuity_divergence"][substep], np.float64)
    reference_inputs["zhdiv"] = reference_terms["zhdiv"]
    operand_rows = {
        name: _score(candidate_inputs[name], reference_inputs[name])
        for name in OPERANDS
    }

    substitutions = {}
    for name in OPERANDS:
        if name == "zhdiv":
            continue
        trial = dict(candidate_inputs)
        trial[name] = reference_inputs[name]
        substitutions[name] = _score(_continuity_terms(trial)["output"], target)

    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round255-ssh-writer-v1",
        "status": "PASS_R255_SSH_WRITER_SPLIT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved",
        "worktree": stamp,
        "round254_dependency": "HELD_R254_SSHN_E_FIRST_NONBIT",
        "record_census": census,
        "boundary_order": list(BOUNDARIES),
        "boundary_rows": boundary_rows,
        "boundary_identity": boundary_identity,
        "first_nonbit_boundary": _first_nonbit(BOUNDARIES, boundary_rows),
        "operand_order": list(OPERANDS),
        "operand_rows": operand_rows,
        "first_nonbit_operand": _first_nonbit(OPERANDS, operand_rows),
        "candidate_divergence_replay": _score(
            candidate_terms["zhdiv"], candidate_inputs["zhdiv"]),
        "candidate_output_replay": _score(
            candidate_terms["output"], boundary_values["continuity_output"]),
        "nemo_output_replay": _score(reference_terms["output"], target),
        "one_variable_substitutions": substitutions,
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
        "resolved_values": {
            "dtype": str(target.dtype),
            "substep_count": int(oracle["i000_entry_sc"][3]),
            "candidate_rDt_e": float(candidate_inputs["rDt_e"]),
            "nemo_rDt_e": float(reference_inputs["rDt_e"]),
        },
        "compiled_source": {
            "continuity": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:550-558",
            "association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739",
            "history_rotation": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:781-793",
            "candidate": "barotropic_latlon_cgrid.py:2050-2096,2336-2374,2403-2408",
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
    if plant == "round254-dependency":
        independent["round254_dependency"] = "MOVED"
    elif plant == "source-order":
        independent["operand_order"].reverse()
    elif plant == "nemo-replay":
        independent["nemo_output_replay"]["unequal"] = 1
    elif plant == "candidate-replay":
        independent["candidate_output_replay"]["unequal"] = 1
    elif plant == "boundary-identity":
        independent["boundary_identity"]["association_to_rotation"]["unequal"] = 1
    elif plant == "first-boundary":
        independent["first_nonbit_boundary"] = "post_association"
    elif plant == "first-operand":
        independent["first_nonbit_operand"] = "zhU"
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summaries = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R255_SSH_WRITER_SPLIT",
                f"{label}: measurement status moved")
        require(report["round254_dependency"] == "HELD_R254_SSHN_E_FIRST_NONBIT",
                f"{label}: round-254 dependency moved")
        require(report["operand_route"] == "production-d114-resolved",
                f"{label}: stale operand route")
        require(report["boundary_order"] == list(BOUNDARIES),
                f"{label}: boundary order moved")
        require(report["operand_order"] == list(OPERANDS),
                f"{label}: operand order moved")
        require(report["candidate_divergence_replay"]["unequal"] == 0,
                f"{label}: candidate divergence replay is not exact")
        require(report["candidate_output_replay"]["unequal"] == 0,
                f"{label}: candidate output replay is not exact")
        require(report["nemo_output_replay"]["unequal"] == 0,
                f"{label}: NEMO output replay is not exact")
        require(all(row["unequal"] == 0
                    for row in report["boundary_identity"].values()),
                f"{label}: association or rotation changes current SSH")
        require(report["first_nonbit_boundary"] == "continuity_output",
                f"{label}: first non-bit boundary moved")
        require(report["first_nonbit_operand"] == "sshn_e",
                f"{label}: first non-bit continuity operand moved")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control went vacuous")
        summaries[label] = {
            "boundary_rows": report["boundary_rows"],
            "operand_rows": report["operand_rows"],
            "one_variable_substitutions": report["one_variable_substitutions"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round255-classification-v1",
        "status": "HELD_R255_INCOMING_SSH_OWNS_CONTINUITY",
        "first_nonbit_boundary": "continuity_output",
        "first_nonbit_operand": "sshn_e",
        "summaries": summaries,
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"),
                        default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--midpoint-report", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path, action="append", default=[])
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.midpoint_report, args.label, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.midpoint_report, args.label, args.expect_commit)
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
