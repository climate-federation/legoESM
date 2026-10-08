#!/usr/bin/env python3
"""Locate the first non-finite kt=8 external-substep boundary passively."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round165_vertical_boundary_gate as r165,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)

EXPECTED_ERROR = r165.EXPECTED_ERROR
PLANTS = ("none", "terminal", "passivity", "source-order", "record-support")

# dynspg_ts.f90:503-790, stopping at each already-materialised trace boundary.
SOURCE_ORDER = (
    ("entry_ssh", "eta_entry"),
    ("entry_u", "u_entry"),
    ("entry_v", "v_entry"),
    ("mid_ssh", "eta_mid"),
    ("mid_u", "u_mid"),
    ("mid_v", "v_mid"),
    ("mid_depth_u", "transport_face_depth_u"),
    ("mid_depth_v", "transport_face_depth_v"),
    ("transport_u", "transport_metric_u"),
    ("transport_v", "transport_metric_v"),
    ("continuity_du", "continuity_du"),
    ("continuity_dv", "continuity_dv"),
    ("continuity_divergence", "continuity_divergence"),
    ("continuity_rhs", "continuity_rhs"),
    ("continuity_increment", "continuity_increment"),
    ("after_ssh", "eta_continuity"),
    ("face_depth_u_exit", "face_depth_u_exit"),
    ("face_depth_v_exit", "face_depth_v_exit"),
    ("inverse_depth_u_exit", "r1_face_depth_u_exit"),
    ("inverse_depth_v_exit", "r1_face_depth_v_exit"),
    ("back_ssh", "eta_pgf"),
    ("pressure_u", "pgf_u"),
    ("pressure_v", "pgf_v"),
    ("coriolis_u", "cor_u"),
    ("coriolis_v", "cor_v"),
    ("trend_u", "trd_u"),
    ("trend_v", "trd_v"),
    ("exit_u", "u_exit"),
    ("exit_v", "v_exit"),
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _state_rows(actual, expected) -> dict[str, bool]:
    a = rung0.candidate_fields(actual)
    b = rung0.candidate_fields(expected)
    return {name: bool(np.array_equal(a[name], b[name])) for name in a}


def _first_nonfinite(frame: dict[str, object]) -> dict[str, object] | None:
    counts = np.asarray(frame["invalid_counts"])
    flats = np.asarray(frame["first_flat_indices"])
    values = np.asarray(frame["first_invalid_values"])
    for substep in range(counts.shape[1]):
        for row, (boundary, trace_name) in enumerate(SOURCE_ORDER):
            if counts[row, substep]:
                shape = tuple(frame["shapes"][trace_name])
                return {
                    "substep": substep + 1,
                    "boundary": boundary,
                    "trace_name": trace_name,
                    "invalid_count": int(counts[row, substep]),
                    "flat_index": int(flats[row, substep]),
                    "index": list(np.unravel_index(int(flats[row, substep]), shape)),
                    "value": float(values[row, substep]),
                }
    return None


def _install_trace_wrapper(frames: list[dict[str, object]]):
    """Request the existing trace, publish scalars, return ordinary outputs."""
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    original = model_module.barotropic_substeps_latlon_cgrid

    def wrapped(*args, **kwargs):
        require(not kwargs.get("_nemo_substep_trace_test_hook", False),
                "round-166 wrapper cannot wrap an already traced call")
        traced_kwargs = dict(kwargs)
        traced_kwargs["_nemo_substep_trace_test_hook"] = True
        state_after, averages, trace = original(*args, **traced_kwargs)
        counts = []
        flats = []
        bad_values = []
        shapes: dict[str, list[int]] = {}
        for _boundary, trace_name in SOURCE_ORDER:
            values = jnp.asarray(trace[trace_name])
            shapes[trace_name] = list(values.shape[1:])
            flat = values.reshape((values.shape[0], -1))
            invalid = ~jnp.isfinite(flat)
            first = jnp.argmax(invalid, axis=1)
            counts.append(jnp.count_nonzero(invalid, axis=1))
            flats.append(first)
            bad_values.append(jnp.take_along_axis(
                flat, first[:, None], axis=1)[:, 0])

        def capture(invalid_counts, first_flat_indices, first_invalid_values):
            frames.append({
                "invalid_counts": np.asarray(invalid_counts).tolist(),
                "first_flat_indices": np.asarray(first_flat_indices).tolist(),
                "first_invalid_values": np.asarray(first_invalid_values).tolist(),
                "shapes": shapes,
            })

        jax.debug.callback(
            capture, jnp.stack(counts), jnp.stack(flats),
            jnp.stack(bad_values), ordered=True)
        return state_after, averages

    model_module.barotropic_substeps_latlon_cgrid = wrapped
    return model_module, original


def _hooks(card, *, expose_stage: int = 0):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    return _NEMOWSRK3TestHooks(
        expose_momentum_stage=expose_stage,
        expose_tracer_stage=expose_stage,
        barotropic_external_mode_association=True,
        barotropic_reference_face_depth_override=(
            rung0.ladder.build_reference_depth_override(card)),
        barotropic_unmasked_v_transport=True,
        barotropic_materialize_v_transport=True,
    )


def _record_inventory(root: Path) -> list[str]:
    candidates = []
    for path in root.rglob("*kt00000008*.bin"):
        lowered = path.name.lower()
        if any(token in lowered for token in ("spg", "substep", "btstep")):
            candidates.append(str(path))
    return sorted(candidates)


def measure(deck_root: Path, record_root: Path, search_root: Path) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "measurement requires production JIT on CPU")
    admission = rung0.frames.admit(record_root, ladder.EXPECTED_PRODUCER, None)
    require(admission["record_count"] == 80, "rung-0 frame admission changed")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    entry = rung0.assemble_frame(record_root, 1, 0)
    initial = rung0.bridge_entry(card, entry)
    freshwater, surface = ladder._zero_forcing(entry["ssh"].shape)

    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card))
    expected_states = []
    state = initial
    unobserved_error = None
    for kt in range(1, 9):
        try:
            state = jax.device_get(ordinary.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        except Exception as error:
            unobserved_error = str(error)
            require(kt == 8, f"unobserved arm refused early at kt={kt}")
            break
        expected_states.append(state)
    require(unobserved_error is not None and EXPECTED_ERROR in unobserved_error,
            "unobserved complete arm did not reproduce the kt=8 refusal")

    # The already-established host-side stage exposure must still reach both
    # kt=8 boundaries from the last completed state.
    stage_exposed = []
    for stage in (1, 2):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_hooks(card, expose_stage=stage))
        stage_exposed.append(bool(np.all(np.isnan(np.asarray(jax.device_get(
            model.step(expected_states[-1], card.dt_s, freshwater=freshwater,
                       surface_forcing=surface)).eta.data)))))

    frames: list[dict[str, object]] = []
    model_module, original = _install_trace_wrapper(frames)
    try:
        observed = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_hooks(card))
        state = initial
        passivity = []
        observed_error = None
        for kt in range(1, 9):
            try:
                state = jax.device_get(observed.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
            except Exception as error:
                observed_error = str(error)
                require(kt == 8, f"observed arm refused early at kt={kt}")
                break
            rows = _state_rows(state, expected_states[kt - 1])
            passivity.append({"kt": kt, "fields": rows,
                              "bit_exact": all(rows.values())})
    finally:
        model_module.barotropic_substeps_latlon_cgrid = original
    require(observed_error is not None and EXPECTED_ERROR in observed_error,
            "observed complete arm did not reproduce the kt=8 refusal")
    require(len(frames) == 8, f"expected 8 external traces, got {len(frames)}")
    first = _first_nonfinite(frames[-1])
    return {
        "format": "nemo-testcase-l4-orca2-round166-external-substep-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "private_arm": {
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "unobserved_terminal": {"completed_kt": 7, "kt8_stage3": False,
                                "error": EXPECTED_ERROR},
        "observed_terminal": {"completed_kt": 7, "kt8_stage3": False,
                              "error": EXPECTED_ERROR},
        "kt8_stages12_exposed": stage_exposed,
        "completed_checkpoint_passivity": passivity,
        "trace_call_count": len(frames),
        "trace_source_order": [name for name, _ in SOURCE_ORDER],
        "kt8_first_nonfinite": first,
        "kt8_record_candidates": _record_inventory(search_root),
        "worktree": worktree_stamp(),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "terminal":
        report["observed_terminal"]["kt8_stage3"] = True
    elif plant == "passivity":
        report["completed_checkpoint_passivity"][0]["bit_exact"] = False
    elif plant == "source-order":
        report["trace_source_order"][0], report["trace_source_order"][1] = (
            report["trace_source_order"][1], report["trace_source_order"][0])
    elif plant == "record-support":
        report["kt8_record_candidates"] = ["planted.bin"]

    terminal = {"completed_kt": 7, "kt8_stage3": False,
                "error": EXPECTED_ERROR}
    require(report["unobserved_terminal"] == terminal,
            "unobserved terminal changed")
    require(report["observed_terminal"] == terminal,
            "passive observer changed the terminal")
    require(report["kt8_stages12_exposed"] == [True, True],
            "kt=8 stage-1/2 exposure changed")
    require(len(report["completed_checkpoint_passivity"]) == 7 and all(
        row["bit_exact"] for row in report["completed_checkpoint_passivity"]),
        "passive trace changed a completed checkpoint")
    require(tuple(report["trace_source_order"]) == tuple(
        name for name, _ in SOURCE_ORDER), "source order changed")
    require(report["kt8_first_nonfinite"] is not None,
            "kt=8 trace has no non-finite boundary")
    require(not report["kt8_record_candidates"],
            "an existing kt=8 external-substep record requires admission")
    report["status"] = "PASS_ROUND166_EXTERNAL_SUBSTEP_BOUNDARY"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--record-search-root", type=Path)
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            raw = json.loads(args.report_in.read_text())
        else:
            require(all((args.deck_root, args.record_root,
                         args.record_search_root)),
                    "measurement requires deck, record and search roots")
            raw = measure(args.deck_root, args.record_root,
                          args.record_search_root)
            if args.json_out:
                args.json_out.write_text(
                    json.dumps(raw, indent=2, sort_keys=True) + "\n")
        report = classify(raw, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND166_EXTERNAL_SUBSTEP_BOUNDARY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
