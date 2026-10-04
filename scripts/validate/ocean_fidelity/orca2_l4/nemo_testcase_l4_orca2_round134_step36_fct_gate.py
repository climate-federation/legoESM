#!/usr/bin/env python3
"""Name rung-0's first non-finite arithmetic boundary inside stage-3 FCT."""

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
    nemo_testcase_l4_orca2_round133_step36_downstream_gate as downstream,
)


TRACE_GROUPS = (
    ("first_upstream_flux", ("first_u", "first_v", "first_w")),
    ("first_upstream_divergence", ("first_div",)),
    ("midpoint_predictor", ("midpoint",)),
    ("averaged_upstream_flux", ("average_u", "average_v", "average_w")),
    ("upstream_divergence", ("upstream_div",)),
    ("rhs_after_upstream", ("rhs_after_up",)),
    ("antidiffusive_flux", ("anti_pre_u", "anti_pre_v", "anti_pre_w")),
    ("limiter_coefficient", ("coef_u", "coef_v", "coef_w")),
    ("limited_antidiffusive_flux", ("anti_post_u", "anti_post_v", "anti_post_w")),
    ("final_corrected_divergence", ("final_div",)),
    ("final_divisor", ("divisor",)),
    ("final_rhs", ("rhs_final",)),
    ("caller_advection_content", ("caller_content",)),
)
GROUP_ORDER = tuple(name for name, _ in TRACE_GROUPS)
TRACE_FIELD_ORDER = tuple(field for _, fields in TRACE_GROUPS[:-1] for field in fields)
TARGET = downstream.TARGET
FIELDS = prior.FIELDS
PLANTS = ("none", "source-order", "passivity", "support")


class GateError(RuntimeError):
    """The FCT arithmetic walk violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_nonfinite(groups: dict[str, object], key: str) -> str | None:
    return next((name for name in GROUP_ORDER if groups[name][key]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "source-order":
        report["first_nonfinite_group"] = "first_upstream_flux"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "support":
        report["groups"]["antidiffusive_flux"]["active_count"] += 1

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "FCT walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("trace_field_order", ())) == TRACE_FIELD_ORDER,
            "FCT trace field order changed")
    require(tuple(report.get("group_order", ())) == GROUP_ORDER,
            "FCT group order changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in FIELDS},
        "ordinary step-36 repeat changed bits")
    require(report.get("observer_state_equal") == {
        name: True for name in FIELDS},
        "write-only FCT observer moved the ordinary step")
    require(report.get("duplicate_calls_equal") is True,
            "duplicate production FCT calls disagree")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": list(TARGET), "value": "nan"},
        "round-133 step-36 failure did not reproduce")
    require(report.get("downstream_replay") == {
        "T": 132, "S": 134, "target": list(TARGET)},
        "round-133 stage-3 advection boundary did not replay")

    for group, fields in TRACE_GROUPS:
        row = report["groups"][group]
        require(tuple(row["fields"]) == fields,
                f"{group}: field registry changed")
        require(int(row["nonfinite_total"]) == sum(
            int(value) for value in row["nonfinite"].values()),
            f"{group}: non-finite census disagrees")
        require(int(row["target_nonfinite_total"]) == sum(
            int(value) for value in row["target_nonfinite"].values()),
            f"{group}: target census disagrees")
        expected_active = sum(int(row["support_count"][name]) for name in fields)
        require(int(row["active_count"]) == expected_active,
                f"{group}: active support census disagrees")

    first = _first_nonfinite(report["groups"], "nonfinite_total")
    first_target = _first_nonfinite(report["groups"], "target_nonfinite_total")
    require(first == report.get("first_nonfinite_group"),
            "first active-support non-finite group is not source ordered")
    require(first_target == report.get("first_target_nonfinite_group"),
            "first target non-finite group is not source ordered")
    require(first is not None and first_target is not None,
            "FCT walk did not reach the committed non-finite boundary")

    first_index = GROUP_ORDER.index(first)
    later_all = all(report["groups"][name]["nonfinite_total"]
                    for name in GROUP_ORDER[first_index:])
    predictions = {
        "R134-P1": {"status": "CONFIRMED",
                     "observed": report["returned_first_nonfinite"]},
        "R134-P2": {
            "status": "CONFIRMED" if first == "antidiffusive_flux" else "REFUTED",
            "predicted": "antidiffusive_flux", "observed": first,
        },
        "R134-P3": {
            "status": ("CONFIRMED" if first_target == "antidiffusive_flux"
                       else "REFUTED"),
            "predicted": "antidiffusive_flux", "observed": first_target,
        },
        "R134-P4": {
            "status": "CONFIRMED" if later_all else "REFUTED",
            "observed": [name for name in GROUP_ORDER[first_index:]
                         if report["groups"][name]["nonfinite_total"]],
        },
        "R134-P5": {"status": "CONFIRMED", "observed": "bit-identical"},
        "R134-P6": {"status": "CONFIRMED", "observed": "measurement-only"},
    }
    return {**report, "status": "PASS_ROUND134_STEP36_FCT_WALK",
            "prediction_ledger": predictions}


def _support_masks(active: np.ndarray) -> dict[str, np.ndarray]:
    u_int = active | np.roll(active, 1, axis=1)
    u = np.concatenate([u_int, u_int[:, :1, :]], axis=1)
    v = np.zeros((active.shape[0] + 1, active.shape[1], active.shape[2]), bool)
    v[0] = active[0]
    v[-1] = active[-1]
    v[1:-1] = active[:-1] | active[1:]
    w = np.zeros((*active.shape[:-1], active.shape[-1] + 1), bool)
    w[..., 0] = active[..., 0]
    w[..., -1] = active[..., -1]
    w[..., 1:-1] = active[..., :-1] | active[..., 1:]
    return {"u": u, "v": v, "w": w, "cell": active}


def _kind(name: str) -> str:
    if name.endswith("_u"):
        return "u"
    if name.endswith("_v"):
        return "v"
    if name.endswith("_w"):
        return "w"
    return "cell"


def _target_values(values: np.ndarray, kind: str) -> tuple[np.ndarray, list[list[int]]]:
    j, i, k = TARGET
    if kind == "u":
        indices = ((j, i, k), (j, i + 1, k))
    elif kind == "v":
        indices = ((j, i, k), (j + 1, i, k))
    elif kind == "w":
        indices = ((j, i, k), (j, i, k + 1))
    else:
        indices = ((j, i, k),)
    return np.asarray([values[index] for index in indices]), [list(index) for index in indices]


def _field_summary(name: str, values: np.ndarray,
                   supports: dict[str, np.ndarray]) -> dict[str, object]:
    values = np.asarray(values)
    require(values.dtype == np.dtype(np.float64), f"{name}: trace is not fp64")
    kind = _kind(name)
    support = supports[kind]
    require(values.shape == support.shape,
            f"{name}: shape {values.shape} != {kind} support {support.shape}")
    selected = values[support]
    bad = np.argwhere((~np.isfinite(values)) & support)
    target_values, target_indices = _target_values(values, kind)
    return {
        "kind": kind,
        "shape": list(values.shape),
        "support_count": int(np.count_nonzero(support)),
        "nonfinite": int(np.count_nonzero(~np.isfinite(selected))),
        "first_nonfinite": (list(map(int, bad[0])) if bad.size else None),
        "finite_max_abs": (float(np.max(np.abs(selected[np.isfinite(selected)])))
                           if bool(np.isfinite(selected).any()) else None),
        "target_indices": target_indices,
        "target_values": [str(value) for value in target_values],
        "target_nonfinite": int(np.count_nonzero(~np.isfinite(target_values))),
    }


def _group_summaries(traces: dict[str, dict[str, np.ndarray]],
                     supports: dict[str, np.ndarray],
                     caller: dict[str, object]) -> dict[str, object]:
    groups = {}
    for group, fields in TRACE_GROUPS[:-1]:
        summaries = {
            tracer: {name: _field_summary(name, traces[tracer][name], supports)
                     for name in fields}
            for tracer in ("T", "S")
        }
        nonfinite = {
            tracer: sum(row["nonfinite"] for row in summaries[tracer].values())
            for tracer in ("T", "S")
        }
        target_nonfinite = {
            tracer: sum(row["target_nonfinite"]
                        for row in summaries[tracer].values())
            for tracer in ("T", "S")
        }
        support_count = {
            name: summaries["T"][name]["support_count"] * 2 for name in fields}
        groups[group] = {
            "fields": list(fields), "details": summaries,
            "support_count": support_count,
            "active_count": sum(support_count.values()),
            "nonfinite": nonfinite, "nonfinite_total": sum(nonfinite.values()),
            "target_nonfinite": target_nonfinite,
            "target_nonfinite_total": sum(target_nonfinite.values()),
        }
    caller_nonfinite = {name: int(caller["nonfinite"][name])
                        for name in ("T", "S")}
    caller_target = {name: int(caller["target_nonfinite"][name])
                     for name in ("T", "S")}
    groups["caller_advection_content"] = {
        "fields": ["caller_content"],
        "support_count": {"caller_content": int(caller["active_count"]) * 2},
        "active_count": int(caller["active_count"]) * 2,
        "nonfinite": caller_nonfinite,
        "nonfinite_total": sum(caller_nonfinite.values()),
        "target_nonfinite": caller_target,
        "target_nonfinite_total": 1,
        "details": {"replayed_round133": True},
    }
    return groups


def _load_downstream(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text())
    require(raw.get("status") == "PASS_ROUND133_STEP36_DOWNSTREAM_WALK",
            "round-133 downstream report is not admitted")
    row = raw["boundaries"]["stage3_advection_content"]
    require(row["nonfinite"] == {"T": 132, "S": 134},
            "round-133 stage-3 advection census moved")
    require(raw["boundaries"]["returned_state"]["first_nonfinite"] == {
        "field": "T", "index": list(TARGET), "value": "nan"},
        "round-133 returned boundary moved")
    return row


def measure(deck_root: Path, downstream_report: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import advection as advection_module
    from legoesm.ocean.advection import NEMO_FCT_TRACE_FIELDS
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-134 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-134 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "FCT walk requires production JIT on CPU")
    require(tuple(NEMO_FCT_TRACE_FIELDS) == TRACE_FIELD_ORDER,
            "production FCT trace registry changed")

    caller = _load_downstream(downstream_report)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    freshwater, surface = ladder._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))

    def model():
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    ordinary = model()
    state = card.recipe.initial_state
    started = time.time()
    for step in range(1, 36):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        require(prior.boundary_summary(prior._state_arrays(state))[
                    "nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"ROUND134_FCT_PROGRESS step={step}/35 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state35 = state
    returned = jax.device_get(ordinary.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    returned_repeat = jax.device_get(ordinary.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    real_fct = advection_module.fct_tracer_advection
    calls: list[tuple[np.ndarray, ...]] = []

    def sink(*values):
        calls.append(tuple(np.asarray(value) for value in values))

    def capture(*values, **kwargs):
        require(kwargs.get("low_order_predictor") == "nemo_rk3_two_step",
                "observed FCT call is not the compiled two-step program")
        div_h, div_w, trace = real_fct(
            *values, **kwargs, return_nemo_trace=True)
        jax.debug.callback(sink, *trace, ordered=True)
        return div_h, div_w

    observed_model = model()
    advection_module.fct_tracer_advection = capture
    try:
        observed = jax.device_get(observed_model.step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
        jax.effects_barrier()
    finally:
        advection_module.fct_tracer_advection = real_fct

    content_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_stage3_advection_content=True))
    content_state = jax.device_get(content_model.step(
        state35, card.dt_s, freshwater=freshwater,
        surface_forcing=surface))

    require(len(calls) >= 2 and len(calls) % 2 == 0,
            f"production step observed {len(calls)} FCT calls")
    require(all(len(call) == len(TRACE_FIELD_ORDER) for call in calls),
            "production FCT trace arity changed")
    duplicate_equal = True
    for index, duplicate in enumerate(calls[2:], 2):
        original = calls[index % 2]
        duplicate_equal &= all(np.array_equal(
            np.ascontiguousarray(left).view(np.uint64),
            np.ascontiguousarray(right).view(np.uint64))
            for left, right in zip(duplicate, original, strict=True))
    traces = {
        tracer: dict(zip(TRACE_FIELD_ORDER, calls[index], strict=True))
        for index, tracer in enumerate(("T", "S"))
    }
    supports = _support_masks(
        np.asarray(card.recipe.z_coord.is_active, dtype=bool))
    active = supports["cell"]
    content_arrays = {
        "T": np.asarray(content_state.T.data),
        "S": np.asarray(content_state.S.data),
    }
    content_summary = prior.boundary_summary({
        name: np.where(active, values, 0.0)
        for name, values in content_arrays.items()})
    j, i, k = TARGET
    content_summary["active_count"] = int(np.count_nonzero(active))
    content_summary["target_nonfinite"] = {
        name: int(not np.isfinite(values[j, i, k]))
        for name, values in content_arrays.items()
    }
    require(content_summary["nonfinite"] == caller["nonfinite"],
            "live stage-3 content disagrees with the round-133 census")
    groups = _group_summaries(traces, supports, content_summary)
    first = _first_nonfinite(groups, "nonfinite_total")
    first_target = _first_nonfinite(groups, "target_nonfinite_total")
    returned_summary = prior.boundary_summary(prior._state_arrays(returned))
    return {
        "format": "nemo-testcase-l4-orca2-round134-step36-fct-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "unmeasured_features": list(card.unmeasured_features),
        "trace_field_order": list(TRACE_FIELD_ORDER),
        "group_order": list(GROUP_ORDER),
        "groups": groups,
        "first_nonfinite_group": first,
        "first_target_nonfinite_group": first_target,
        "returned_first_nonfinite": returned_summary["first_nonfinite"],
        "ordinary_repeat_state_equal": prior.state_bit_rows(
            returned_repeat, returned),
        "observer_state_equal": prior.state_bit_rows(observed, returned),
        "observed_fct_call_count": len(calls),
        "duplicate_calls_equal": bool(duplicate_equal),
        "downstream_report": str(downstream_report),
        "downstream_replay": {
            "T": int(caller["nonfinite"]["T"]),
            "S": int(caller["nonfinite"]["S"]),
            "target": list(TARGET),
        },
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "upstream": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-610",
            "antidiffusive": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-200,260-280",
            "limiter": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:314-316,743-938",
            "final_rhs": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:318-330",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--downstream-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.downstream_report is None
                    and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.downstream_report is not None
                    and args.expect_commit,
                    "runtime mode requires deck, downstream report, and commit")
            raw = measure(args.deck_root, args.downstream_report, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, downstream.GateError, prior.GateError, rung0.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND134_STEP36_FCT_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
