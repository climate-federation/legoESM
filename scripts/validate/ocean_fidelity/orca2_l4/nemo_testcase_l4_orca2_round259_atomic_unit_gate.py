#!/usr/bin/env python3
"""Gate round 259's complete OMT-4 fold unit through kt=10."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_stage1_transport_gate as transport,
)

LABELS = r249.LABELS
PLANTS = (
    "none", "label-coverage", "production-route", "passivity",
    "mask-closure", "transport-closure", "kt1-floor", "growth",
    "completion", "one-bit",
)


class GateError(RuntimeError):
    """A frozen complete-unit predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right, support) -> dict[str, object]:
    return r232._exact_masked(
        np.asarray(left, np.float64), np.asarray(right, np.float64), support)


def _pivot_transport(deck_root: Path, frame_root: Path, record_root: Path,
                     label: str, expect_commit: str) -> dict[str, object]:
    """Score the executing resolved stage mask and its written V product."""

    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands

    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True))
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True, live=True))
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
        traced = jax.device_get(traced_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live-stage trace return type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live-stage trace moved completed state")

    z_coord, grid = card.recipe.z_coord, card.recipe.grid
    compact_u, compact_v = compute_face_masks_3d(z_coord.is_active, grid)
    dtype = jnp.asarray(state.eta.data).dtype
    compact_u = jnp.asarray(compact_u, dtype=dtype)
    compact_v = jnp.asarray(compact_v, dtype=dtype)
    resolved = nemo_qco_resolved_mesh_operands(
        z_coord, grid, compact_u, compact_v, dtype, state.T.data.shape[-1])
    resolved_v = np.asarray(jnp.concatenate(
        [jnp.zeros_like(resolved.vmask3[:1]), resolved.vmask3], axis=0),
        np.float64)[1:]
    compact_native = np.asarray(compact_v, np.float64)[1:]

    record_path = record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    oracle = transport.read_record(record_path)
    expected_mask = np.asarray(oracle["vmask"], np.float64)[..., :-1]
    expected_zfv = np.asarray(oracle["zFv"], np.float64)[..., :-1]
    candidate_zfv = np.asarray(traced.stage_geometry[0][8], np.float64)[1:]
    support = ((compact_native.view(np.uint64) != expected_mask.view(np.uint64))
               & np.isfinite(expected_mask) & np.isfinite(expected_zfv))
    require(int(np.count_nonzero(support)) == 668,
            "pivot-row stage-mask support moved")
    return {
        "format": "nemo-testcase-l4-orca2-round259-pivot-v1",
        "status": "PASS_R259_PIVOT_TRANSPORT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-stage-vmask",
        "private_unit": True,
        "passivity": passivity,
        "pivot_support": int(np.count_nonzero(support)),
        "vmask": _score(resolved_v, expected_mask, support),
        "zFv": _score(candidate_zfv, expected_zfv, support),
    }


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            label: str, expect_commit: str) -> dict[str, object]:
    stage = r249.measure_stages(
        deck_root, frame_root, label, True, "resolved-candidate",
        expect_commit, max_kt=10)
    pivot = _pivot_transport(
        deck_root, frame_root, record_root, label, expect_commit)
    value = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round259-measurement-v1",
        "status": "PASS_R259_MEASUREMENT",
        "label": label,
        "stage_growth": stage,
        "pivot_transport": pivot,
        "one_bit_control": r249._score(value, np.nextafter(value, np.inf)),
    }


def _first_boundary(report, kt: int, stage: int):
    rows = [row for row in report["boundaries"]
            if (row["kt"], row["stage"]) == (kt, stage)]
    require(len(rows) == 1, f"missing unique kt={kt} stage={stage} boundary")
    return rows[0]


def classify(reports: list[dict[str, object]], *,
             plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "production-route":
        independent["pivot_transport"]["operand_route"] = "compact"
    elif plant == "passivity":
        independent["pivot_transport"]["passivity"]["T"] = False
    elif plant == "mask-closure":
        independent["pivot_transport"]["vmask"]["unequal"] = 1
    elif plant == "transport-closure":
        independent["pivot_transport"]["zFv"]["unequal"] = 1
    elif plant == "kt1-floor":
        _first_boundary(independent["stage_growth"], 1, 1)["fields"]["S"][
            "fold_band"]["max_abs"] = 1.0
    elif plant == "growth":
        _first_boundary(independent["stage_growth"], 2, 1)["fields"]["S"][
            "fold_band"]["max_abs"] = 1.0e6
    elif plant == "completion":
        independent["stage_growth"]["boundaries"] = independent[
            "stage_growth"]["boundaries"][:-1]
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summaries = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R259_MEASUREMENT",
                f"{label}: measurement status moved")
        pivot = report["pivot_transport"]
        require(pivot["status"] == "PASS_R259_PIVOT_TRANSPORT"
                and pivot["execution"] == "production-jit-cpu-fp64-x64-libm"
                and pivot["operand_route"] ==
                "production-d114-resolved-stage-vmask"
                and pivot["private_unit"] is True,
                f"{label}: private production route moved")
        require(all(pivot["passivity"].values()),
                f"{label}: stage observer is not passive")
        require(pivot["pivot_support"] == 668,
                f"{label}: pivot support moved")
        require(pivot["vmask"]["unequal"] == 0,
                f"{label}: resolved stage V mask is non-bit")
        require(pivot["zFv"]["unequal"] == 0,
                f"{label}: stage V transport is non-bit")

        stage = report["stage_growth"]
        require(stage["status"] == "PASS_R249_STAGE_TABLE"
                and stage["unit"] is True
                and stage["operand_route"] == "resolved-candidate",
                f"{label}: atomic stage route moved")
        expected_order = [(kt, substage) for kt in range(1, 11)
                          for substage in range(4)]
        actual_order = [(row["kt"], row["stage"])
                        for row in stage["boundaries"]]
        require(actual_order == expected_order,
                f"{label}: kt=1..10 completion coverage moved")
        require(len(stage["passivity"]) == 10
                and all(all(row["fields"].values())
                        for row in stage["passivity"]),
                f"{label}: completed-state passivity moved")
        kt1 = _first_boundary(stage, 1, 1)
        for field in ("T", "S"):
            require(kt1["fields"][field]["fold_band"]["max_abs"]
                    <= float(r249.FLOOR),
                    f"{label}: kt=1 stage-1 {field} is above the floor")
        fold_growth = r249._growth_rows(stage["boundaries"], "fold_band")
        interior_growth = r249._growth_rows(stage["boundaries"], "interior")
        offenders = [row for row in (*fold_growth, *interior_growth)
                     if row["previous_max_abs"] > float(r249.FLOOR)
                     and row["over_10x"]]
        if offenders:
            raise GateError(
                f"{label}: greater-than-10x growth survives at {offenders[0]}")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summaries[label] = {
            "pivot_transport": pivot,
            "kt1_stage1": {
                field: kt1["fields"][field]["fold_band"]
                for field in ("T", "S")},
            "growth": {"fold_band": fold_growth,
                       "interior": interior_growth},
            "last_boundary": actual_order[-1],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round259-classification-v1",
        "status": "PASS_R259_COMPLETE_UNIT_KT10",
        "labels": summaries,
        "predictions": {
            "R259-P1": "CONFIRMED",
            "R259-P2": "CONFIRMED",
            "R259-P3": "CONFIRMED_NO_POST_FLOOR_10X_GROWTH",
            "R259-P4": "CONFIRMED_KT10_BOTH_LABELS",
            "R259-P5": "PENDING_DECISION96_CENSUS",
            "R259-P6": "PENDING_FULL_LANDING_GATES",
            "R259-P7": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
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
                        args.label, args.expect_commit)
            require(all(value is not None for value in required),
                    "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.record_root,
                args.label, args.expect_commit)
        else:
            require(args.classify is not None,
                    "classification JSONs are required")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify], plant=args.plant)
    except (GateError, r249.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
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
