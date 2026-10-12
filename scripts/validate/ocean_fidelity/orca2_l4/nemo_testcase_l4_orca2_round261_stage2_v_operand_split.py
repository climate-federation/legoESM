#!/usr/bin/env python3
"""Split OMT-4's residual kt=1 stage-2 ``zFv`` into written operands."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round260_stage2_transport_split as r260,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_prognostic_barotropic_state_gate import (  # noqa: E402
    read_bt_pair,
)
from nemo_testcase_l2_gyre_round11_composition import (  # noqa: E402
    read_stage2_composition,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean  # noqa: E402

LABELS = r249.LABELS
OPERANDS = r232.OPERANDS
CORRECTION_INPUTS = r232.CORRECTION_INPUTS
NX, NY, NZ = r260.NX, r260.NY, r260.NZ
FLOOR = np.float64(2.0e-10)
PLANTS = (
    "none", "label-coverage", "source-order", "record-identity",
    "candidate-replay", "oracle-replay", "round260-reproduction",
    "first-owner", "substitution-closure", "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-261 operand or control predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_v(values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    require(values.shape[:2] == (149, 180), f"redundant V shape moved {values.shape}")
    return values[1:, :90, ...]


def _native_t(values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    require(values.shape[0] == 148 and values.shape[1] in (90, 180),
            f"native T shape moved {values.shape}")
    return values[:, :90, ...]


def _to_3d(values, shape) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if values.ndim == 2:
        values = np.broadcast_to(values[..., None], shape)
    require(values.shape == shape, f"operand shape moved {values.shape} != {shape}")
    return values


def _score(left, right, support) -> dict[str, object]:
    return r232._exact_masked(
        np.asarray(left, np.float64), np.asarray(right, np.float64),
        np.asarray(support, bool),
    )


def _first_valued(rows: dict[str, dict[str, object]]) -> str | None:
    return next((name for name in OPERANDS if rows[name]["max_abs"] > FLOOR), None)


def _run_model(card, state, forcing, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    ).step(state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1])


def measure(
    deck_root: Path, frame_root: Path, record_root: Path, prior_report: Path,
    label: str, expect_commit: str,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSLiveOperandTrace,
        _nemo_stage_corrected_velocity,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_resolved_mesh_operands,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    prior = json.loads(prior_report.read_text(encoding="utf-8"))
    require(prior["label"] == label
            and prior["status"] == "PASS_R260_TRANSPORT_SPLIT_MEASUREMENT",
            "round-260 labelled report moved")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    stage1 = rung0.assemble_frame(frame_root, 1, 1)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    stage1_u_full = np.concatenate([stage1["u"][:, -1:, :], stage1["u"]], axis=1)
    stage1_v_full = np.concatenate([stage1["v"][-1:, :, :], stage1["v"]], axis=0)
    hooks = r249._hooks(card, True, live=True)._replace(
        stage_entry_override=(
            2, stage1_u_full, stage1_v_full, stage1["T"], stage1["S"],
            stage1["ssh"],
        ),
    )
    ordinary_hooks = r249._hooks(card, True)._replace(
        stage_entry_override=hooks.stage_entry_override)
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        ordinary = jax.device_get(_run_model(
            card, state, forcing, ordinary_hooks))
        traced = jax.device_get(_run_model(card, state, forcing, hooks))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live trace return type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live trace moved the completed state")

    z_coord, grid = card.recipe.z_coord, card.recipe.grid
    dtype = jnp.asarray(state.eta.data).dtype
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(dtype)
    compact_u, compact_v = compute_face_masks_3d(z_coord.is_active, grid)
    compact_u = jnp.asarray(compact_u, dtype=dtype)
    compact_v = jnp.asarray(compact_v, dtype=dtype)
    resolved = nemo_qco_resolved_mesh_operands(
        z_coord, grid, compact_u, compact_v, dtype, h_ref.shape[-1])
    resolved_v = jnp.concatenate(
        [jnp.zeros_like(resolved.vmask3[:1]), resolved.vmask3], axis=0)

    eta_stage = jnp.asarray(stage1["ssh"])
    _, _, _, _, _, candidate_r1_hv_full = (
        _nemo_ws_qco_stage_faces(
            eta_stage, h_ref, compact_u, compact_v, grid,
            z_coord=z_coord, include_reciprocals=True,
        ))
    geometry = traced.stage_geometry[1]
    candidate_v = jnp.asarray(traced.stage_states[1][1])
    candidate_corrected, candidate_zvb = jax.device_get(jax.jit(
        lambda v, avg, inv, baro, mask: _nemo_stage_corrected_velocity(
            v, avg, inv, baro, mask, return_correction=True)
    )(
        candidate_v,
        jnp.asarray(traced.barotropic_targets[3]),
        candidate_r1_hv_full,
        jnp.asarray(traced.barotropic_targets[1]),
        resolved_v,
    ))
    require(np.array_equal(
        np.asarray(candidate_corrected), np.asarray(geometry[10])),
        "candidate correction replay moved")

    composition_path = record_root / "oracle_rkstage2_operands_kt00000001.bin"
    advmean_path = record_root / "oracle_bt_advmean_operands_kt00000001.bin"
    bt_path = record_root / "oracle_bt_frames_kt00000001.bin"
    tracer_path = record_root / "oracle_rktracer_operands_kt00000001_s2.bin"
    composition = read_stage2_composition(
        composition_path, expected_dims=(NX, NY, NZ))
    advmean = read_advmean(
        advmean_path, expected_dims=(NX, NY), expected_ncycle=65)
    bt = read_bt_pair(bt_path)
    tracer = r260._read_tracer(tracer_path)

    # domqco.f90:213-217 computes r3v at the stage-1 SSH, then
    # stprk3_stg.f90:168-170 associates that V-point scalar across the fold.
    # Replaying without the association compares the candidate to itself on
    # the pivot row and cannot split live e3v from the downstream zvb.
    raw = z_coord.nemo_een_barotropic
    require(raw is not None and grid.fold is not None,
            "ORCA2 raw face geometry or north-fold descriptor is missing")
    area_t = jnp.asarray(z_coord.nemo_e1e2t, dtype=dtype)
    area_v = jnp.asarray(raw.e1v * raw.e2v, dtype=dtype)
    raw_vmask = jnp.asarray(raw.vmask, dtype=dtype)
    surface_vmask = jnp.max(raw_vmask, axis=-1)
    raw_hv = jnp.asarray(raw.hv_0, dtype=dtype)
    r1_hv0 = surface_vmask / (raw_hv + 1.0 - surface_vmask)
    r1_area_v = jnp.where(area_v > 0.0, 1.0 / area_v, 0.0)
    eta_native = jnp.asarray(stage1["ssh"], dtype=dtype)
    north_eta = jnp.concatenate(
        [eta_native[1:], jnp.zeros_like(eta_native[:1])], axis=0)
    north_area = jnp.concatenate(
        [area_t[1:], jnp.zeros_like(area_t[:1])], axis=0)
    r3v = nemo_source_round(
        nemo_source_round(
            nemo_source_round(
                jnp.float64(0.5) * nemo_source_round(
                    nemo_source_round(area_t * eta_native)
                    + nemo_source_round(north_area * north_eta)))
            * r1_hv0)
        * r1_area_v)
    r3v = r3v.at[-1].set(r3v[-2, jnp.asarray(grid.fold.perm_v)])
    one_plus_r3v = nemo_source_round(1.0 + r3v[..., None] * raw_vmask)
    expected_e3v = nemo_source_round(
        jnp.asarray(raw.e3v_0, dtype=dtype) * one_plus_r3v)
    expected_r1_hv = np.asarray(
        nemo_source_round(r1_hv0 / (1.0 + r3v))[:, :90], np.float64)
    expected_zvb = np.asarray(nemo_source_round(
        nemo_source_round(jnp.asarray(advmean["post_lbc_v"][:, :90])
                          * jnp.asarray(expected_r1_hv))
        - jnp.asarray(bt["vv_b"][:, :90])), np.float64)
    expected = {
        "e1v": np.asarray(raw.e1v[:, :90], np.float64),
        "e3v": np.asarray(expected_e3v[:, :90, :30], np.float64),
        "vv": _native_t(composition["Kmm_v"])[..., :30],
        "zvb": expected_zvb,
        "vmask": np.asarray(raw.vmask[:, :90, :30], np.float64),
    }
    candidate = {
        "e1v": _native_v(grid.dx_v),
        "e3v": _native_v(geometry[5]),
        "vv": _native_v(candidate_v),
        "zvb": _native_v(candidate_zvb),
        "vmask": _native_v(resolved_v),
    }
    expected_zfv = np.asarray(tracer["zFv"][..., :30], np.float64)
    candidate_zfv = _native_v(geometry[8])
    shape = expected_zfv.shape
    require(candidate_zfv.shape == shape, "candidate zFv shape moved")
    fold = np.zeros(shape, dtype=bool)
    fold[-3:] = True
    live = np.asarray(expected["vmask"], bool)
    live_fold = fold & live
    support = {
        "e1v": np.any(live_fold, axis=-1),
        "e3v": live_fold,
        "vv": live_fold,
        "zvb": np.any(live_fold, axis=-1),
        "vmask": fold,
    }
    rows = {
        name: _score(candidate[name], expected[name], support[name])
        for name in OPERANDS
    }

    candidate_replay = np.asarray(jax.device_get(jax.jit(r232._source_replay)(
        *(jnp.asarray(candidate[name]) for name in OPERANDS))), np.float64)
    oracle_replay = np.asarray(jax.device_get(jax.jit(r232._source_replay)(
        *(jnp.asarray(expected[name]) for name in OPERANDS))), np.float64)
    replay = {
        "candidate": _score(candidate_replay, candidate_zfv, fold),
        "oracle": _score(oracle_replay, expected_zfv, fold),
    }
    substitutions = {}
    cumulative = dict(candidate)
    for name in OPERANDS:
        cumulative[name] = expected[name]
        replayed = np.asarray(jax.device_get(jax.jit(r232._source_replay)(
            *(jnp.asarray(cumulative[item]) for item in OPERANDS))), np.float64)
        substitutions[name] = _score(replayed, expected_zfv, fold)

    candidate_inputs = {
        "vn_adv": _native_v(traced.barotropic_targets[3]),
        "r1_hv": _native_v(candidate_r1_hv_full),
        "vv_b": _native_v(traced.barotropic_targets[1]),
    }
    expected_inputs = {
        "vn_adv": np.asarray(advmean["post_lbc_v"][:, :90], np.float64),
        "r1_hv": expected_r1_hv,
        "vv_b": np.asarray(bt["vv_b"][:, :90], np.float64),
    }
    correction_rows = {
        name: _score(candidate_inputs[name], expected_inputs[name],
                     np.any(live_fold, axis=-1))
        for name in CORRECTION_INPUTS
    }
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round261-stage2-v-operands-v1",
        "status": "PASS_R261_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": passivity,
        "operand_order": list(OPERANDS),
        "correction_input_order": list(CORRECTION_INPUTS),
        "operand_rows": rows,
        "correction_input_rows": correction_rows,
        "first_valued_operand": _first_valued(rows),
        "zFv": _score(candidate_zfv, expected_zfv, fold),
        "replay": replay,
        "cumulative_substitutions": substitutions,
        "round260": prior["transport_rows"]["zFv"],
        "one_bit_control": r249._score(one, np.nextafter(one, np.inf)),
        "records": {
            path.name: _sha256(path)
            for path in (composition_path, advmean_path, bt_path, tracer_path)
        },
        "compiled_source": {
            "correction": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279",
            "transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285",
            "live_geometry": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:207-217",
            "stage_ratio_fold": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "source-order":
        independent["operand_order"][1:3] = reversed(
            independent["operand_order"][1:3])
    elif plant == "record-identity":
        independent["records"][next(iter(independent["records"]))] = "bad"
    elif plant == "candidate-replay":
        independent["replay"]["candidate"]["unequal"] = 1
    elif plant == "oracle-replay":
        independent["replay"]["oracle"]["unequal"] = 1
    elif plant == "round260-reproduction":
        independent["zFv"]["max_abs"] = 1.0
    elif plant == "first-owner":
        independent["first_valued_operand"] = "e1v"
    elif plant == "substitution-closure":
        for row in independent["cumulative_substitutions"].values():
            row["unequal"] = 1
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summaries = {}
    first_by_label = {}
    closure_by_label = {}
    reference_records = by_label[LABELS[0]]["records"]
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R261_MEASUREMENT",
                f"{label}: status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-complete-private-unit",
                f"{label}: provenance moved")
        require(all(report["passivity"].values()), f"{label}: trace is not passive")
        require(report["operand_order"] == list(OPERANDS),
                f"{label}: source order moved")
        require(report["correction_input_order"] == list(CORRECTION_INPUTS),
                f"{label}: correction source order moved")
        require(report["records"] == reference_records,
                f"{label}: record identity moved")
        require(report["replay"]["candidate"]["unequal"] == 0,
                f"{label}: candidate replay is not exact")
        require(report["replay"]["oracle"]["unequal"] == 0,
                f"{label}: oracle replay is not exact")
        require(report["zFv"]["unequal"] == 849
                and report["zFv"]["max_abs"] == 0.7981690085143782,
                f"{label}: round-260 zFv boundary moved")
        require(report["round260"]["unequal"] == 849
                and report["round260"]["max_abs"] == 0.7981690085143782,
                f"{label}: prior report moved")
        first = _first_valued(report["operand_rows"])
        require(first == report["first_valued_operand"],
                f"{label}: first-owner classification is inconsistent")
        require(first is not None, f"{label}: no valued operand owns zFv")
        closing = next((name for name in OPERANDS
                        if report["cumulative_substitutions"][name]["unequal"] == 0),
                       None)
        require(closing is not None, f"{label}: all-oracle replay did not close")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        first_by_label[label] = first
        closure_by_label[label] = closing
        summaries[label] = {
            "operand_rows": report["operand_rows"],
            "correction_input_rows": report["correction_input_rows"],
            "zFv": report["zFv"],
            "cumulative_substitutions": report["cumulative_substitutions"],
        }
    require(len(set(first_by_label.values())) == 1,
            "claim labels disagree on the first valued operand")
    require(len(set(closure_by_label.values())) == 1,
            "claim labels disagree on the closing substitution")
    first = first_by_label[LABELS[0]]
    closing = closure_by_label[LABELS[0]]
    closing_index = OPERANDS.index(closing)
    return {
        "format": "nemo-testcase-l4-orca2-round261-stage2-v-classification-v1",
        "status": "PASS_R261_STAGE2_V_OPERAND_SPLIT",
        "first_valued_operand": first,
        "closing_operand": closing,
        "closing_unit": list(OPERANDS[:closing_index + 1]),
        "labels": summaries,
        "predictions": {
            "R261-P1": "CONFIRMED",
            "R261-P2": "CONFIRMED",
            "R261-P3": "CONFIRMED" if first == "zvb" else "REFUTED",
            "R261-P4": "CONFIRMED" if first == closing else "CANCELLING_UNIT",
            "R261-P5": "NOT_REACHED",
            "R261-P6": "NOT_REACHED",
            "R261-P7": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.prior_report, args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.record_root,
                args.prior_report, args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification reports are required")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, r260.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
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
