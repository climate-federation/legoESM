#!/usr/bin/env python3
"""Audit the private seven-array association on ORCA2's closed south row."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round253_spg8_acquisition import (  # noqa: E402
    check_record as admission,
)

LABELS = ("independent", "given_nemo_entry")
PLANTS = (
    "none", "label-coverage", "admission", "source-order", "south-support",
    "old-behavior", "north-nonvacuity", "one-bit",
)
FIELDS = (
    ("u", "u_exit", "j001_ua_new", "u"),
    ("v", "v_exit", "j001_va_new", "v"),
    ("depth_u", "face_depth_u_exit", "j001_hu_e", "u"),
    ("depth_v", "face_depth_v_exit", "j001_hv_e", "v"),
    ("inverse_u", "r1_face_depth_u_exit", "j001_hur_e", "u"),
    ("inverse_v", "r1_face_depth_v_exit", "j001_hvr_e", "v"),
    ("eta", "eta_continuity", "j001_ssha_e", "t"),
)


class GateError(RuntimeError):
    """The frozen south-row causal predicates moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-254 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-254 measurement requires production JIT on CPU")
    return stamp


def _native(values, face: str) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if face == "u":
        return values[:, 1:]
    if face == "v":
        return values[1:]
    return values


def _model(values, face: str) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if face == "u":
        return r97._to_model_u(values)
    if face == "v":
        return r97._to_model_v(values)
    return values


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _range(array: np.ndarray) -> dict[str, object]:
    array = np.asarray(array, np.float64)
    return {
        "shape": list(array.shape),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "nonzero": int(np.count_nonzero(array)),
        "nonfinite": int(np.count_nonzero(~np.isfinite(array))),
    }


def _offline_association(trace: dict[str, object], card) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )

    before = {
        name: np.asarray(trace[trace_name][0], np.float64)
        for name, trace_name, _oracle_name, _face in FIELDS
    }
    post = jax.device_get(_nemo_external_mode_boundary_association(
        *(jnp.asarray(before[name], dtype=jnp.float64) for name, *_ in FIELDS),
        card.recipe.grid,
    ))
    return {name: np.asarray(value, np.float64)
            for (name, *_), value in zip(FIELDS, post)}


def _south_rows(before: dict[str, np.ndarray], after: dict[str, np.ndarray],
                oracle: dict[str, np.ndarray]) -> dict[str, object]:
    rows = {}
    for name, _trace_name, oracle_name, face in FIELDS:
        pre = before[name]
        post = after[name]
        nemo = _model(oracle[oracle_name], face)
        require(pre.shape == post.shape == nemo.shape,
                f"{name}: model/oracle association shape moved")
        rows[name] = {
            "off_rows_0_2": _range(pre[:3]),
            "on_rows_0_2": _range(post[:3]),
            "nemo_rows_0_2": _range(nemo[:3]),
            "on_vs_off_row0": _score(post[0], pre[0]),
            "off_vs_nemo_row0": _score(pre[0], nemo[0]),
            "on_vs_nemo_row0": _score(post[0], nemo[0]),
            "on_vs_off_rows_0_2": _score(post[:3], pre[:3]),
            "on_vs_off_north_rows": _score(post[-2:], pre[-2:]),
        }
    return rows


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            baseline_root: Path, label: str, expect_commit: str) -> dict:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSBarotropicTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    admitted = admission.run(spg_root, baseline_root, "none")
    require(admitted["status"] == "PASS_R253_OMT4_SPG8_ADMISSION",
            "round-253 record admission moved")
    oracle, record_census = r97.assemble_record(
        spg_root, prefix="oracle_r166_spg", expected_kt=8,
        expected_magic="NEMO_L4_R166SPG")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))

    complete_hooks = r253._hooks(card)
    complete = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=complete_hooks)
    # Hold every other part of the private unit fixed and omit only the
    # seven-array association.  The returned trace is the already-established
    # passive substep side output; the association replay below is offline.
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
            print(f"PROGRESS round254 {label} kt={kt}", file=sys.stderr, flush=True)
        trace = jax.device_get(observer.step(
            state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    require(isinstance(trace, _NEMOWSBarotropicTrace),
            "association-off substep trace return type moved")
    before = {
        name: np.asarray(trace.substeps[trace_name][0], np.float64)
        for name, trace_name, _oracle_name, _face in FIELDS
    }
    after = _offline_association(trace.substeps, card)
    rows = _south_rows(before, after, oracle)
    source_order = [name for name, *_ in FIELDS]
    changed = [name for name in source_order
               if rows[name]["on_vs_off_row0"]["unequal"]]
    north_changed = sum(
        rows[name]["on_vs_off_north_rows"]["unequal"]
        for name in source_order)
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round254-south-association-v1",
        "status": "PASS_R254_SOUTH_ASSOCIATION_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved",
        "worktree": stamp,
        "admission": admitted["status"],
        "record_census": record_census,
        "source_order": source_order,
        "south_rows": rows,
        "south_changed_fields": changed,
        "north_changed_cells": north_changed,
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
        "compiled_source": {
            "association_call": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:730-741",
            "north_only_t_pivot": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:584-700",
            "candidate_association": "barotropic_latlon_cgrid.py:688-767",
            "candidate_padding": "operators_latlon_cgrid.py:178-208,304-323",
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
    if plant == "admission":
        independent["admission"] = "REFUSE"
    elif plant == "source-order":
        independent["source_order"].reverse()
    elif plant == "south-support":
        independent["south_changed_fields"] = ["depth_v"]
    elif plant == "old-behavior":
        independent["south_rows"]["depth_v"]["on_vs_off_row0"]["unequal"] = 1
    elif plant == "north-nonvacuity":
        independent["north_changed_cells"] = 0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    signatures = set()
    summaries = {}
    expected_order = [name for name, *_ in FIELDS]
    for label, report in by_label.items():
        require(report["status"] == "PASS_R254_SOUTH_ASSOCIATION_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["admission"] == "PASS_R253_OMT4_SPG8_ADMISSION",
                f"{label}: admitted kt=8 record moved")
        require(report["operand_route"] == "production-d114-resolved",
                f"{label}: stale operand route")
        require(report["source_order"] == expected_order,
                f"{label}: association source order moved")
        changed = report["south_changed_fields"]
        measured = [name for name in expected_order
                    if report["south_rows"][name]["on_vs_off_row0"]["unequal"]]
        require(changed == measured,
                f"{label}: south-row changed-field registry disagrees with data")
        require(report["north_changed_cells"] > 0,
                f"{label}: association north-fold control went vacuous")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control went vacuous")
        signatures.add(tuple(changed))
        summaries[label] = {
            "south_changed_fields": changed,
            "south_rows": report["south_rows"],
            "north_changed_cells": report["north_changed_cells"],
        }
    require(len(signatures) == 1,
            "claim labels disagree on the south-row writer")
    changed = list(next(iter(signatures)))
    return {
        "format": "nemo-testcase-l4-orca2-round254-south-classification-v1",
        "status": ("CONFIRMED_R254_SOUTH_ASSOCIATION_WRITER" if changed
                   else "REFUTED_R254_SOUTH_ASSOCIATION_HYPOTHESIS"),
        "first_changed_fields": changed,
        "summaries": summaries,
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path, action="append", default=[])
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.baseline_root, args.label, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.label, args.expect_commit)
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
