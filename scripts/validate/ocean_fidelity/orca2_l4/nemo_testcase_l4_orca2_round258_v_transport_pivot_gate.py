#!/usr/bin/env python3
"""Split current-tree OMT-4 stage-1 pivot V transport into written operands."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round257_tracer_writer_gate as r257,
    nemo_testcase_l4_orca2_stage1_transport_gate as transport,
)

LABELS = r249.LABELS
OPERANDS = r232.OPERANDS
PLANTS = (
    "none", "label-coverage", "source-order", "support", "passivity",
    "reproduction", "candidate-replay", "first-owner", "mask-closure",
    "round247-reconciliation", "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-258 provenance, operand, or causal predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _physical(values) -> np.ndarray:
    return np.asarray(values, np.float64)[-1, :90, :30]


def _rank0(values) -> np.ndarray:
    return r232._rank0(np.asarray(values, np.float64))


def _to_3d(values, shape) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if values.ndim == 2:
        values = np.broadcast_to(values[..., None], shape)
    require(values.shape == shape, f"operand shape moved: {values.shape} != {shape}")
    return values


def _first_valued(rows: dict[str, dict[str, object]]) -> str | None:
    return next((name for name in OPERANDS if rows[name]["max_abs"] != 0.0), None)


@contextlib.contextmanager
def _resolved_stage_vmask_arm(resolved_vmask):
    """Route only the stage transport's V mask to D114's resolved source."""

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    original = model_module._nemo_ws_stage_transport

    def armed(*args, **kwargs):
        require("v_mask_3d" in kwargs, "stage transport V-mask argument moved")
        kwargs["v_mask_3d"] = resolved_vmask
        return original(*args, **kwargs)

    model_module._nemo_ws_stage_transport = armed
    try:
        yield
    finally:
        model_module._nemo_ws_stage_transport = original


def _run(card, state, forcing, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    return model.step(
        state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1])


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            round247_json: Path, label: str, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_stage_corrected_velocity,
        _nemo_ws_qco_stage_faces,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_resolved_mesh_operands,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    hooks = r249._hooks(card, True)
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    live_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True, live=True),
    )
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
        traced = jax.device_get(live_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live-stage trace return type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live-stage trace moved the completed state")

    grid, z_coord = card.recipe.grid, card.recipe.z_coord
    eta = jnp.asarray(state.eta.data)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(eta), state.H_bathy.data, z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    compact_u, compact_v = compute_face_masks_3d(z_coord.is_active, grid)
    compact_u = jnp.asarray(compact_u, dtype=eta.dtype)
    compact_v = jnp.asarray(compact_v, dtype=eta.dtype)
    resolved = nemo_qco_resolved_mesh_operands(
        z_coord, grid, compact_u, compact_v, eta.dtype, h_ref.shape[-1])
    resolved_v = jnp.concatenate(
        [jnp.zeros_like(resolved.vmask3[:1]), resolved.vmask3], axis=0)

    def correction(stage_v, transport_average, vv_b):
        _, _, _, _, _, r1_hv = _nemo_ws_qco_stage_faces(
            eta, h_ref, compact_u, compact_v, grid, z_coord=z_coord,
            include_reciprocals=True)
        corrected, zvb = _nemo_stage_corrected_velocity(
            stage_v, transport_average, r1_hv, vv_b, compact_v,
            return_correction=True)
        return corrected, zvb, r1_hv

    corrected_v, zvb, r1_hv = jax.device_get(jax.jit(correction)(
        jnp.asarray(state.v.data),
        jnp.asarray(traced.barotropic_targets[3]),
        jnp.asarray(state.vv_b.data),
    ))
    geometry = traced.stage_geometry[0]
    candidate = {
        "e1v": _rank0(grid.dx_v),
        "e3v": _rank0(geometry[5]),
        "vv": _rank0(state.v.data),
        "zvb": _rank0(zvb),
        "vmask": _rank0(compact_v),
    }
    candidate_zfv = _rank0(geometry[8])
    replay_zfv = np.asarray(jax.device_get(jax.jit(r232._source_replay)(
        jnp.asarray(candidate["e1v"]), jnp.asarray(candidate["e3v"]),
        jnp.asarray(candidate["vv"]), jnp.asarray(candidate["zvb"]),
        jnp.asarray(candidate["vmask"]),
    )), np.float64)
    candidate_replay = _score(replay_zfv, candidate_zfv)

    record_path = record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    oracle = transport.read_record(record_path)
    expected = {
        "e1v": np.asarray(oracle["e1v"], np.float64),
        "e3v": np.asarray(oracle["e3v"], np.float64)[..., :-1],
        "vv": np.asarray(oracle["vv"], np.float64)[..., :-1],
        "zvb": np.asarray(oracle["zvb"], np.float64),
        "vmask": np.asarray(oracle["vmask"], np.float64)[..., :-1],
    }
    expected_zfv = np.asarray(oracle["zFv"], np.float64)[..., :-1]
    oracle_replay = np.asarray(jax.device_get(jax.jit(r232._source_replay)(
        *(jnp.asarray(expected[name]) for name in OPERANDS)
    )), np.float64)
    oracle_replay_score = _score(oracle_replay, expected_zfv)

    bit_unequal = candidate_zfv.view(np.uint64) != expected_zfv.view(np.uint64)
    pivot = np.zeros_like(bit_unequal, dtype=bool)
    pivot[-1, :, :] = True
    debt_support = pivot & bit_unequal & np.isfinite(candidate_zfv) & np.isfinite(expected_zfv)
    require(np.count_nonzero(debt_support) == 668,
            "round-257 pivot zFv support did not reproduce")
    rows = {
        name: r232._exact_masked(
            _to_3d(candidate[name], expected_zfv.shape),
            _to_3d(expected[name], expected_zfv.shape), debt_support)
        for name in OPERANDS
    }
    first = _first_valued(rows)

    resolved_candidate_vmask = _rank0(resolved_v)
    mask_arm_zfv = np.asarray(jax.device_get(jax.jit(r232._source_replay)(
        jnp.asarray(candidate["e1v"]), jnp.asarray(candidate["e3v"]),
        jnp.asarray(candidate["vv"]), jnp.asarray(candidate["zvb"]),
        jnp.asarray(resolved_candidate_vmask),
    )), np.float64)
    mask_arm = {
        "vmask": r232._exact_masked(
            resolved_candidate_vmask, expected["vmask"], pivot),
        "zFv": r232._exact_masked(mask_arm_zfv, expected_zfv, pivot),
    }

    oracle_stage = rung0.assemble_frame(frame_root, 1, 1)
    baseline_stage = {
        "T": _score(_physical(traced.stage_outputs[0][2]),
                    _physical(oracle_stage["T"])),
        "S": _score(_physical(traced.stage_outputs[0][3]),
                    _physical(oracle_stage["S"])),
    }
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        with _resolved_stage_vmask_arm(resolved_v):
            arm_stage = jax.device_get(_run(
                card, state, forcing, hooks._replace(expose_tracer_stage=1)))
    arm_stage_score = {
        "T": _score(_physical(arm_stage.T.data), _physical(oracle_stage["T"])),
        "S": _score(_physical(arm_stage.S.data), _physical(oracle_stage["S"])),
    }

    historical = json.loads(round247_json.read_text(encoding="utf-8"))
    reconciliation = {
        "current_arm": "current private unit plus resolved-stage-vmask arm",
        "round257_arm": "current private unit without resolved-stage-vmask arm",
        "round247_arm": "resolved-stage-mask private arm",
        "support": "rank-0 native pivot row and NEMO-live stage-1 slots",
        "record_sha256": transport.sha256(record_path),
        "round247_record_sha256": historical["record"]["sha256"],
        "round247_vmask_unequal": historical["operand_rows"]["vmask"]["unequal"],
        "round247_zfv_unequal": historical["zFv_signature"]["unequal"],
        "current_vmask_unequal": mask_arm["vmask"]["unequal"],
        "current_zfv_unequal": mask_arm["zFv"]["unequal"],
    }
    one = np.asarray([1.0], np.float64)
    one_bit = _score(one, np.nextafter(one, np.inf))
    return {
        "format": "nemo-testcase-l4-orca2-round258-v-pivot-v1",
        "status": "PASS_R258_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-geometry-compact-stage-mask",
        "private_unit": True,
        "passivity": passivity,
        "operand_order": list(OPERANDS),
        "pivot_zfv_support": int(np.count_nonzero(debt_support)),
        "operand_rows": rows,
        "first_valued_operand": first,
        "candidate_replay": candidate_replay,
        "oracle_replay": oracle_replay_score,
        "baseline_stage": baseline_stage,
        "mask_arm": mask_arm,
        "mask_arm_stage": arm_stage_score,
        "round247_reconciliation": reconciliation,
        "one_bit_control": one_bit,
        "diagnostic": {
            "corrected_v_vs_trace": _score(corrected_v, geometry[10]),
            "r1_hv_finite": bool(np.all(np.isfinite(r1_hv))),
        },
        "record": {"path": str(record_path), "sha256": transport.sha256(record_path)},
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {row["label"]: row for row in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "source-order":
        independent["operand_order"][1:3] = reversed(independent["operand_order"][1:3])
    elif plant == "support":
        independent["pivot_zfv_support"] = 667
    elif plant == "passivity":
        independent["passivity"][next(iter(independent["passivity"]))] = False
    elif plant == "reproduction":
        independent["baseline_stage"]["S"]["max_abs"] = 0.0
    elif plant == "candidate-replay":
        independent["candidate_replay"]["unequal"] = 1
    elif plant == "first-owner":
        independent["operand_rows"]["e3v"]["max_abs"] = 1.0
    elif plant == "mask-closure":
        independent["mask_arm"]["zFv"]["unequal"] = 1
    elif plant == "round247-reconciliation":
        independent["round247_reconciliation"]["round247_record_sha256"] = "bad"
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    labels = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R258_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-geometry-compact-stage-mask"
                and report["private_unit"] is True,
                f"{label}: production route moved")
        require(all(report["passivity"].values()), f"{label}: trace is not passive")
        require(report["operand_order"] == list(OPERANDS),
                f"{label}: compiled operand order moved")
        require(report["pivot_zfv_support"] == 668,
                f"{label}: pivot zFv support moved")
        require(report["candidate_replay"]["unequal"] == 0
                and report["oracle_replay"]["unequal"] == 0,
                f"{label}: product replay missed an endpoint")
        require(report["baseline_stage"]["T"]["max_abs"] == 0.17735225873138316
                and report["baseline_stage"]["S"]["max_abs"] == 3.283688696376231,
                f"{label}: round-257 stage boundary did not reproduce")
        first = _first_valued(report["operand_rows"])
        require(first == report["first_valued_operand"],
                f"{label}: first-owner field is inconsistent")
        require(first == "vmask", f"{label}: predicted V-mask owner was refuted")
        require(all(report["operand_rows"][name]["max_abs"] == 0.0
                    for name in OPERANDS[:-1]),
                f"{label}: an earlier written operand is non-bit in value")
        require(report["mask_arm"]["vmask"]["unequal"] == 0
                and report["mask_arm"]["zFv"]["unequal"] == 0,
                f"{label}: resolved V mask did not close the transport")
        require(all(report["mask_arm_stage"][name]["max_abs"] <= float(r249.FLOOR)
                    for name in ("T", "S")),
                f"{label}: resolved V mask did not close the tracer stage")
        recon = report["round247_reconciliation"]
        require(recon["record_sha256"] == recon["round247_record_sha256"]
                and recon["round247_vmask_unequal"] == 0
                and recon["round247_zfv_unequal"] == 0
                and recon["current_vmask_unequal"] == 0
                and recon["current_zfv_unequal"] == 0,
                f"{label}: round-247 resolved-arm result did not reproduce")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        labels[label] = {
            "first_valued_operand": first,
            "operand_rows": report["operand_rows"],
            "baseline_stage": report["baseline_stage"],
            "mask_arm": report["mask_arm"],
            "mask_arm_stage": report["mask_arm_stage"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round258-classification-v1",
        "status": "HELD_R258_STAGE_VMASK_OWNER_NAMED",
        "first_valued_operand": "vmask",
        "labels": labels,
        "predictions": {
            "R258-P1": "CONFIRMED",
            "R258-P2": "CONFIRMED",
            "R258-P3": "CONFIRMED",
            "R258-P4": "CONFIRMED_PRIVATE_ARM",
            "R258-P5": "CONFIRMED",
            "R258-P6": "HELD_D116_ATOMIC_UNIT",
            "R258-P7": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--round247-json", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            required = (args.deck_root, args.frame_root, args.record_root,
                        args.round247_json, args.label, args.expect_commit)
            require(all(value is not None for value in required),
                    "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.record_root,
                args.round247_json, args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification JSONs are required")
            result = classify(
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.classify], plant=args.plant)
    except (GateError, r232.GateError, r249.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
