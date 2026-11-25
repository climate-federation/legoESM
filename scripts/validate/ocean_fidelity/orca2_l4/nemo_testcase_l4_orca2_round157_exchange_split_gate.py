#!/usr/bin/env python3
"""Split ORCA2's external-mode U/V association into cyclic and fold operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round156_association_growth_gate as round156,
)


COMPONENTS = ("u_cyclic", "u_fold", "v_cyclic", "v_fold")
PLANTS = ("none", "comparison-bit", "signed-zero", "selector", "overlap",
          "composition")
STATE_FIELDS = ("T", "S", "u", "v", "ssh", "uu_b", "vv_b")


class GateError(RuntimeError):
    """The exchange split or one of its controls is incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def validate_selection(*, complete: bool = False, field: str = "",
                       component: str = "") -> None:
    require(component in ("", *COMPONENTS),
            f"unknown association component {component!r}")
    require(sum(bool(value) for value in (complete, field, component)) <= 1,
            "complete, field, and component arms overlap")


def run_plant(plant: str) -> None:
    if plant == "comparison-bit":
        row = round156.exact_pair_row(
            np.nextafter(np.array([1.0]), np.array([np.inf])),
            np.array([1.0]))
        require(row["differing_cells"] == 1 and not row["bit_exact"],
                "comparison-bit plant stayed green")
        raise GateError("comparison-bit plant fired")
    if plant == "signed-zero":
        row = round156.exact_pair_row(
            np.array([0.0]), np.array([-0.0]))
        require(row["differing_cells"] == 1
                and row["maximum_absolute"] == 0.0,
                "signed-zero plant stayed green")
        raise GateError("signed-zero plant fired")
    if plant == "selector":
        validate_selection(component="plausible")
        raise GateError("selector plant stayed green")
    if plant == "overlap":
        validate_selection(field="u", component="u_fold")
        raise GateError("overlap plant stayed green")


def _hooks(reference_depth, *, component: str = "", stage: int = 0,
           expose: bool = False):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    validate_selection(component=component)
    return _NEMOWSRK3TestHooks(
        expose_barotropic_substeps=expose,
        expose_barotropic_boundary_association=expose,
        expose_momentum_stage=stage,
        expose_tracer_stage=stage,
        barotropic_external_mode_association_component=component,
        barotropic_reference_face_depth_override=(
            reference_depth if component else None),
        barotropic_unmasked_v_transport=bool(component),
        barotropic_materialize_v_transport=bool(component),
    )


def _run(card, state, reference_depth, freshwater, surface, *,
         component: str = "", stage: int = 0, expose: bool = False):
    import jax

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(
            reference_depth, component=component, stage=stage, expose=expose),
    )
    return jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))


def _state(value):
    return value.state_after if hasattr(value, "state_after") else value


def _compare_states(candidate, reference) -> dict[str, dict[str, object]]:
    candidate_fields = round156.state_fields(_state(candidate))
    reference_fields = round156.state_fields(_state(reference))
    return {
        name: round156.exact_pair_row(
            candidate_fields[name], reference_fields[name])
        for name in STATE_FIELDS
    }


def _first_nonbit(rows: list[dict[str, object]]) -> dict[str, object] | None:
    for row in rows:
        if not row["comparison"]["bit_exact"]:
            return row
    return None


def _component_walk(component: str, candidate_trace, production_trace) -> dict:
    face = component[0]
    post_key = f"boundary_post_{face}"
    pre_key = f"{face}_exit"
    transport_key = f"transport_metric_{face}"
    continuity_key = f"continuity_d{face}"
    association_rows = []
    consumer_rows = []
    for index in range(65):
        association_rows.append({
            "substep": index + 1,
            "boundary": f"{face}_after_association",
            "comparison": round156.exact_pair_row(
                candidate_trace[post_key][index],
                production_trace[pre_key][index]),
        })
        if index == 64:
            continue
        consumer = index + 1
        for boundary, key in (
            (f"{face}_transport", transport_key),
            (f"continuity_d{face}", continuity_key),
            ("after_ssh", "eta_continuity"),
        ):
            consumer_rows.append({
                "association_substep": index + 1,
                "consumer_substep": consumer + 1,
                "boundary": boundary,
                "comparison": round156.exact_pair_row(
                    candidate_trace[key][consumer],
                    production_trace[key][consumer]),
            })
    return {
        "first_association_difference": _first_nonbit(association_rows),
        "first_consumer_difference": _first_nonbit(consumer_rows),
        "association_rows": association_rows,
        "consumer_rows": consumer_rows,
    }


def _composition_rows(trace, grid, *, plant: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )

    keys = (
        "u_exit", "v_exit", "face_depth_u_exit", "face_depth_v_exit",
        "r1_face_depth_u_exit", "r1_face_depth_v_exit", "eta_exit",
    )
    composed_u = []
    full_u = []
    composed_v = []
    full_v = []
    for index in range(65):
        operands = tuple(jnp.asarray(trace[key][index]) for key in keys)
        full = _nemo_external_mode_boundary_association(
            *operands, grid, component="")
        u_cyclic = _nemo_external_mode_boundary_association(
            *operands, grid, component="u_cyclic")
        u_both = _nemo_external_mode_boundary_association(
            *u_cyclic, grid, component="u_fold")
        v_cyclic = _nemo_external_mode_boundary_association(
            *operands, grid, component="v_cyclic")
        v_both = _nemo_external_mode_boundary_association(
            *v_cyclic, grid, component="v_fold")
        composed_u.append(np.asarray(jax.device_get(u_both[0])))
        full_u.append(np.asarray(jax.device_get(full[0])))
        composed_v.append(np.asarray(jax.device_get(v_both[1])))
        full_v.append(np.asarray(jax.device_get(full[1])))
    composed_u = np.stack(composed_u)
    full_u = np.stack(full_u)
    composed_v = np.stack(composed_v)
    full_v = np.stack(full_v)
    if plant == "composition":
        composed_v = np.array(composed_v, copy=True)
        composed_v.flat[0] = np.nextafter(
            composed_v.flat[0], np.float64(np.inf))
    rows = {
        "u_cyclic_then_fold": round156.exact_pair_row(composed_u, full_u),
        "v_cyclic_then_fold": round156.exact_pair_row(composed_v, full_v),
    }
    if plant == "composition":
        require(rows["v_cyclic_then_fold"]["differing_cells"] == 1,
                "composition plant stayed green")
        raise GateError("composition plant fired")
    require(all(row["bit_exact"] for row in rows.values()),
            "component composition differs from the complete field image")
    return rows


def _setup_policy() -> None:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-157 gate requires production JIT on CPU")


def measure_prelude(deck_root: Path, record_root: Path, *, plant: str) -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    if plant not in ("none", "composition"):
        run_plant(plant)
    _setup_policy()
    (card, reference_depth, _entry, initial_state,
     freshwater, surface) = round156._setup(deck_root, record_root)
    ordinary = _run(
        card, initial_state, reference_depth, freshwater, surface)
    observed = _run(
        card, initial_state, reference_depth, freshwater, surface,
        expose=True)
    passivity = _compare_states(observed, ordinary)
    require(all(row["bit_exact"] for row in passivity.values()),
            "association observer moved the production state")

    production_stage = _run(
        card, initial_state, reference_depth, freshwater, surface,
        stage=1, expose=True)
    require(production_stage.substeps["eta_entry"].shape[0] == 65,
            "production trace does not contain 65 external substeps")
    composition = _composition_rows(
        production_stage.substeps, card.recipe.grid, plant=plant)
    return {
        "status": "MEASURED_R157_EXCHANGE_SPLIT_PRELUDE",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "source_order": list(COMPONENTS),
        "passivity": passivity,
        "composition": composition,
    }


def measure_component(deck_root: Path, record_root: Path,
                      component: str) -> dict:
    validate_selection(component=component)
    require(bool(component), "component mode requires a component")
    _setup_policy()
    (card, reference_depth, _entry, initial_state,
     freshwater, surface) = round156._setup(deck_root, record_root)
    production_stage = _run(
        card, initial_state, reference_depth, freshwater, surface,
        stage=1, expose=True)
    candidate = _run(
        card, initial_state, reference_depth, freshwater, surface,
        component=component, stage=1, expose=True)
    return {
        "status": "MEASURED_R157_EXCHANGE_COMPONENT",
        "claim_label": "independent",
        "component": component,
        "walk": _component_walk(
            component, candidate.substeps, production_stage.substeps),
        "stage1_state": _compare_states(candidate, production_stage),
    }


def aggregate(prelude: dict, reports: list[dict]) -> dict:
    require(prelude.get("status") == "MEASURED_R157_EXCHANGE_SPLIT_PRELUDE",
            "invalid round-157 prelude")
    require(tuple(report.get("component") for report in reports) == COMPONENTS,
            "component reports are missing or out of source order")
    components = {
        report["component"]: {
            "walk": report["walk"],
            "stage1_state": report["stage1_state"],
        }
        for report in reports
    }

    u_cyclic_moves = (
        components["u_cyclic"]["walk"]["first_association_difference"]
        is not None)
    u_fold_moves = (
        components["u_fold"]["walk"]["first_association_difference"]
        is not None)
    v_cyclic_exact = (
        components["v_cyclic"]["walk"]["first_association_difference"]
        is None
        and all(row["bit_exact"] for row in
                components["v_cyclic"]["stage1_state"].values()))
    v_fold_t = components["v_fold"]["stage1_state"]["T"]
    expected_v_max = np.float64(0.1774972822409795)
    v_max = np.float64(v_fold_t["maximum_absolute"])
    v_fold_matches = (
        not v_fold_t["bit_exact"]
        and abs(v_max - expected_v_max) <= abs(np.spacing(expected_v_max)))
    return {
        "status": "MEASURED_R157_EXCHANGE_SPLIT",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "source_order": list(COMPONENTS),
        "passivity": prelude["passivity"],
        "composition": prelude["composition"],
        "components": components,
        "predictions": {
            "R157-P1": "CONFIRMED",
            "R157-P2": "CONFIRMED" if u_cyclic_moves else "REFUTED",
            "R157-P3": "CONFIRMED" if u_fold_moves else "REFUTED",
            "R157-P4": "CONFIRMED" if v_cyclic_exact else "REFUTED",
            "R157-P5": "CONFIRMED" if v_fold_matches else "REFUTED",
            "R157-P6": "CONFIRMED",
        },
        "choices": {"asked": [], "unasked": []},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument(
        "--mode", choices=("prelude", "component", "aggregate"),
        default="prelude")
    parser.add_argument("--component", choices=COMPONENTS)
    parser.add_argument("--prelude", type=Path)
    parser.add_argument("--component-report", type=Path, action="append",
                        default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "prelude":
            result = measure_prelude(
                args.deck_root, args.record_root, plant=args.plant)
        elif args.mode == "component":
            require(args.plant == "none", "component mode does not take a plant")
            require(args.component is not None,
                    "component mode requires --component")
            result = measure_component(
                args.deck_root, args.record_root, args.component)
        else:
            require(args.plant == "none", "aggregate mode does not take a plant")
            require(args.prelude is not None, "aggregate mode requires --prelude")
            require(len(args.component_report) == len(COMPONENTS),
                    "aggregate mode requires four --component-report paths")
            result = aggregate(
                json.loads(args.prelude.read_text()),
                [json.loads(path.read_text())
                 for path in args.component_report])
    except (GateError, round156.GateError, OSError, ValueError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
