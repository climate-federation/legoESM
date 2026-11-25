#!/usr/bin/env python3
"""Split OMT-4's kt=7 stage-1 tracer transport one component at a time."""

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
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round298_stage7_writer as r298,
)

LABELS = r298.LABELS
COMPONENTS = ("trp_u", "trp_v", "trp_w")
FLOOR = r298.FLOOR
PLANTS = (
    "none", "admission", "label-coverage", "component-order", "passivity",
    "replay", "candidate-control", "full-control", "first-component",
    "one-bit",
)


class GateError(RuntimeError):
    """The admitted record, passive replay, or frozen classifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _component_geometry(candidate, oracle, component: str):
    """Replace one native NEMO transport without changing the other two."""
    require(component in COMPONENTS, f"unknown component {component!r}")
    result = list(candidate)
    if component == "trp_u":
        result[0], result[7] = oracle[0], oracle[7]
    elif component == "trp_v":
        result[1], result[8] = oracle[1], oracle[8]
    else:
        result[2] = oracle[2]
    return tuple(result)


def _transport_qco_replays(card, state, trace, candidate_geometry,
                           oracle_geometry):
    """Replay literal CEN2/QCO with eager binary64 statement boundaries."""
    active = np.asarray(card.recipe.z_coord.is_active, bool)
    h_stage = np.asarray(candidate_geometry[3], np.float64)
    q0 = np.asarray(1.0 + trace.stage_qco[0][0], np.float64)
    q1 = np.asarray(1.0 + trace.stage_qco[1][0], np.float64)
    dt = np.float64(card.dt_s / 3.0)
    area = np.asarray(card.recipe.grid.area_T, np.float64)[..., None]
    base = (np.asarray(state.T.data, np.float64),
            np.asarray(state.S.data, np.float64))
    perm_t = np.asarray(card.recipe.grid.fold.perm_T, np.int64)

    def source(value):
        return np.array(value, dtype=np.float64, copy=True, order="C")

    def u_sum(tracer):
        core = source(np.roll(tracer, 1, axis=1) + tracer)
        return np.concatenate([core, core[:, :1]], axis=1)

    def v_sum(tracer):
        result = np.zeros((tracer.shape[0] + 1, *tracer.shape[1:]), np.float64)
        result[1:-1] = source(tracer[:-1] + tracer[1:])
        north = source(tracer[-2:-1] + tracer[-1:])[:, perm_t]
        result[-1] = north[0]
        return result

    def rhs(tracer, p_u, p_v, p_w):
        flux_u = source(source(np.float64(0.5) * p_u) * u_sum(tracer))
        flux_v = source(source(np.float64(0.5) * p_v) * v_sum(tracer))
        delta_u = source(flux_u[:, 1:] - flux_u[:, :-1])
        delta_v = source(flux_v[1:] - flux_v[:-1])
        horizontal = source(delta_u + delta_v)
        inverse_area = source(np.float64(1.0) / area)
        horizontal_scaled = source(horizontal * inverse_area)
        result = source(-source(horizontal_scaled / h_stage))
        tracer_w_sum = source(tracer[..., :-1] + tracer[..., 1:])
        wmask = source(active[..., 1:] * active[..., :-1])
        flux_w_inner = source(
            source(source(np.float64(0.5) * p_w[..., 1:-1]) * tracer_w_sum)
            * wmask)
        flux_w = np.pad(flux_w_inner, ((0, 0), (0, 0), (1, 1)))
        vertical_difference = source(flux_w[..., :-1] - flux_w[..., 1:])
        vertical_scaled = source(vertical_difference * inverse_area)
        return source(result - source(vertical_scaled / h_stage))

    def one(p_u, p_v, p_w):
        outputs = []
        for tracer in base:
            tendency = rhs(tracer, p_u, p_v, p_w)
            if card.recipe.model_config.rk3_stage_tracer_written_order:
                first = source(q0[..., None] * tracer)
                second = source(source(dt * q0[..., None]) * tendency)
                out = source(source(first + second) / q1[..., None])
            else:
                out = ((q0[..., None] * tracer
                        + dt * q0[..., None] * tendency) / q1[..., None])
            outputs.append(np.where(active, out, tracer))
        return tuple(outputs)

    def metric_triplet(geometry):
        return (np.asarray(geometry[7], np.float64),
                np.asarray(geometry[8], np.float64),
                np.asarray(geometry[2], np.float64) * area)

    geometries = {
        "candidate": candidate_geometry, "full_nemo": oracle_geometry,
        **{name: _component_geometry(
            candidate_geometry, oracle_geometry, name) for name in COMPONENTS},
    }
    return {name: one(*metric_triplet(geometry))
            for name, geometry in geometries.items()}


def _measure_label(deck_root: Path, frame_root: Path, oracle: dict,
                   label: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )

    require(label in LABELS, f"unknown label {label}")
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = omt4.rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else omt4.rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True))
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True, live=True))
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        for _ in range(6):
            state = jax.device_get(ordinary_model.step(
                state, card.dt_s, freshwater=forcing[0],
                surface_forcing=forcing[1]))
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=forcing[0],
            surface_forcing=forcing[1]))
        trace = jax.device_get(traced_model.step(
            state, card.dt_s, freshwater=forcing[0],
            surface_forcing=forcing[1]))
    require(isinstance(trace, _NEMOWSLiveOperandTrace),
            "live trace return type moved")
    passivity = passive._ordinary_state_equal(trace.state_after, ordinary)

    active_t = np.asarray(card.recipe.z_coord.is_active, bool)
    g0 = trace.stage_geometry[0]
    oracle_geometry = r298._oracle_geometry(g0, oracle, card.recipe.grid)
    candidate = r298._tracer_replay(card, state, trace, g0)
    direct = _transport_qco_replays(
        card, state, trace, g0, oracle_geometry)
    oracle_qco = {
        "T": np.asarray(oracle["qco_t"][..., :30], np.float64),
        "S": np.asarray(oracle["qco_s"][..., :30], np.float64),
    }
    frame = omt4.rung0.assemble_frame(frame_root, 7, 0)
    entry_rows = {
        name: r298._regions(
            np.asarray(getattr(state, name).data, np.float64),
            np.asarray(frame[name], np.float64), active_t)
        for name in ("T", "S")
    }

    def score_pair(values):
        return {
            "T": r298._regions(values[0], oracle_qco["T"], active_t),
            "S": r298._regions(values[1], oracle_qco["S"], active_t),
        }

    replay_check = {
        "T": r298._regions(
            candidate[0], np.asarray(trace.stage_outputs[0][2]), active_t),
        "S": r298._regions(
            candidate[1], np.asarray(trace.stage_outputs[0][3]), active_t),
    }
    direct_check = {
        "T": r298._regions(direct["candidate"][0], candidate[0], active_t),
        "S": r298._regions(direct["candidate"][1], candidate[1], active_t),
    }
    return {
        "status": "PASS_R299_LABEL_MEASUREMENT", "label": label,
        "passivity": passivity, "entry": entry_rows,
        "candidate_replay": replay_check, "direct_replay": direct_check,
        "candidate": score_pair(direct["candidate"]),
        "full_nemo": score_pair(direct["full_nemo"]),
        "component_arms": {name: score_pair(direct[name])
                           for name in COMPONENTS},
        "subnormal_flush_control": r298._score(
            np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
            np.asarray([True])),
        "one_bit_control": r298._score(
            np.asarray([1.0]), np.asarray([np.nextafter(1.0, 2.0)]),
            np.asarray([True])),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    for row in report.get("labels", ()):
        row["subnormal_flush_control"] = r298._score(
            np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
            np.asarray([True]))
        row["one_bit_control"] = r298._score(
            np.asarray([1.0]), np.asarray([np.nextafter(1.0, 2.0)]),
            np.asarray([True]))
    if plant == "admission":
        report["admission"]["status"] = "BROKEN"
    elif plant == "label-coverage":
        report["labels"].pop()
    elif plant == "component-order":
        report["component_order"][0], report["component_order"][1] = (
            report["component_order"][1], report["component_order"][0])
    elif plant == "passivity":
        report["labels"][0]["passivity"]["S"] = False
    elif plant == "replay":
        report["labels"][0]["candidate_replay"]["S"]["complete"]["max_abs"] = 1.0
    elif plant == "candidate-control":
        report["labels"][0]["candidate"]["S"]["fold_band"]["max_abs"] = 0.0
    elif plant == "full-control":
        report["labels"][0]["full_nemo"]["S"]["fold_band"]["max_abs"] = 1.0e9
    elif plant == "first-component":
        report["labels"][0]["component_arms"]["trp_u"]["S"]["fold_band"]["max_abs"] = 0.0
    elif plant == "one-bit":
        report["labels"][0]["one_bit_control"]["unequal"] = 0

    require(report["admission"]["status"] ==
            "PASS_R297_OMT4_KT7_STAGE1_ADMISSION", "admission moved")
    require(tuple(report["component_order"]) == COMPONENTS,
            "component order moved")
    by_label = {row["label"]: row for row in report["labels"]}
    require(set(by_label) == set(LABELS), "both labels are required")
    first_by_label = {}
    summaries = {}
    for label in LABELS:
        row = by_label[label]
        require(row["status"] == "PASS_R299_LABEL_MEASUREMENT",
                f"{label}: measurement status moved")
        require(all(row["passivity"].values()), f"{label}: trace moved state")
        require(all(item["complete"]["max_abs"] <= FLOOR
                    for item in row["candidate_replay"].values()),
                f"{label}: offline replay missed production")
        require(all(item["complete"]["max_abs"] <= FLOOR
                    for item in row["direct_replay"].values()),
                f"{label}: direct transport replay missed the trusted replay")
        require(row["one_bit_control"]["unequal"] == 1,
                f"{label}: normal one-bit control did not fire")
        sub = row["subnormal_flush_control"]
        require(sub["raw_unequal"] == 1 and sub["subnormal_matches"] == 1
                and sub["unequal"] == 0,
                f"{label}: Decision-127 control moved")
        prior = row["entry"]["S"]["fold_band"]["max_abs"]
        threshold = 10.0 * prior
        candidate = row["candidate"]["S"]["fold_band"]["max_abs"]
        full = row["full_nemo"]["S"]["fold_band"]["max_abs"]
        require(candidate > threshold,
                f"{label}: candidate no longer reproduces the growth")
        require(full <= threshold,
                f"{label}: full NEMO triplet no longer closes the growth")
        causal = [name for name in COMPONENTS
                  if row["component_arms"][name]["S"]["fold_band"]["max_abs"]
                  <= threshold]
        first = causal[0] if causal else "none"
        first_by_label[label] = first
        summaries[label] = {
            "entry_fold_S_max": prior, "threshold": threshold,
            "candidate_fold_S_max": candidate,
            "full_nemo_fold_S_max": full,
            "component_fold_S_max": {
                name: row["component_arms"][name]["S"]["fold_band"]["max_abs"]
                for name in COMPONENTS
            },
            "causal_components": causal, "first_causal_component": first,
        }
    require(len(set(first_by_label.values())) == 1,
            "labels disagree on the first causal component")
    first = next(iter(first_by_label.values()))
    report["classification"] = {
        "first_causal_component": first, "labels": summaries,
        "leaf_record_required": first in COMPONENTS,
        "prediction_ledger": {
            "R299-P1": "CONFIRMED", "R299-P2": "CONFIRMED",
            "R299-P3": "CONFIRMED" if first == "trp_v" else "REFUTED",
            "R299-P4": "CONFIRMED", "R299-P5": "CONFIRMED_RECORD_GAP",
            "R299-P6": "CONFIRMED_MEASUREMENT_ONLY",
        },
    }
    report["status"] = f"STOPPED_FOR_RECORD_R299_FIRST_COMPONENT_{first.upper()}"
    require(plant == "none", f"{plant} plant stayed green")
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-299 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 299 requires production JIT on CPU")
    oracle, admission = r298.assemble_record(record_root, baseline_root)
    labels = [_measure_label(deck_root, frame_root, oracle, label)
              for label in LABELS]
    return {
        "format": "nemo-testcase-l4-orca2-round299-transport-component-v1",
        "status": "PASS_R299_MEASUREMENT", "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm plus offline component replay",
        "admission": admission, "component_order": list(COMPONENTS),
        "labels": labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.baseline_root, args.expect_commit)),
                    "measurement arguments are incomplete")
            report = measure(args.deck_root, args.frame_root, args.record_root,
                             args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            report = json.loads(args.report_in.read_text(encoding="utf-8"))
        if args.raw_output:
            args.raw_output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        result = classify(report, args.plant)
    except (GateError, r298.GateError, OSError, KeyError, TypeError, ValueError,
            IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
