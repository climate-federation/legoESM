#!/usr/bin/env python3
"""Capture and classify exact ORCA2 kt=1 stage/return arrays for round 22."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

FIELDS = ("T", "S", "u", "v", "ssh")
CHECKPOINTS = ("entry", "stage1", "stage2", "stage3", "returned")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_imports(repo_root: Path):
    """Import the gate and model from the explicitly selected clean tree."""
    roots = (
        repo_root / "scripts/validate/ocean_fidelity/orca2_l4",
        repo_root,
        repo_root / "packages/core",
        repo_root / "packages/ocean",
    )
    for root in reversed(roots):
        sys.path.insert(0, str(root))
    gate = importlib.import_module(
        "nemo_testcase_l4_orca2_round1_ladder_gate")
    return gate


def _install_controls(owner: str) -> None:
    if owner == "none":
        return
    control = importlib.import_module(
        "nemo_testcase_l4_orca2_round21_merge_owner_control")
    if owner in ("both", "all"):
        fold = importlib.import_module(
            "nemo_testcase_l4_orca2_merge_gyre_fold_layout_control")
        fold.install_parent_layout()
        control.install_bridge_control()
    if owner == "all":
        control.install_legacy_ldf_routing()


def _score_exact(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    require(left.shape == right.shape, f"shape mismatch {left.shape} != {right.shape}")
    require(left.dtype == right.dtype == np.dtype(np.float64),
            f"dtype mismatch {left.dtype} != {right.dtype}")
    unequal = left.view(np.uint64) != right.view(np.uint64)
    count = int(np.count_nonzero(unequal))
    return {
        "bit_identical": count == 0,
        "unequal": count,
        "count": int(left.size),
        "max_abs": float(np.max(np.abs(left[unequal] - right[unequal])))
        if count else 0.0,
        "first_unequal_index": (
            [int(value) for value in np.argwhere(unequal)[0]] if count else None),
    }


def capture(
    repo_root: Path,
    deck_root: Path,
    record_root: Path,
    owner: str,
    output_npz: Path,
    *,
    plant: bool = False,
) -> dict[str, object]:
    gate = _target_imports(repo_root)
    _install_controls(owner)

    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _, card = gate.card_fields(deck_root)
    oracle_entry1 = gate.assemble_state_fields(record_root, 1, stage=None)
    state = card.recipe.initial_state
    state = state._replace(
        eta=state.eta.replace(
            data=jnp.asarray(oracle_entry1["ssh"], dtype=jnp.float64)))
    surface_fields = gate.assemble_surface_fields(record_root, 1)
    freshwater, surface = gate._surface_forcings(
        card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    trace = model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface)

    candidates = {"entry": gate._candidate_fields(state)}
    for stage in (1, 2, 3):
        candidates[f"stage{stage}"] = gate._stage_candidate_fields(
            trace.stage_outputs[stage - 1])
    candidates["returned"] = gate._candidate_fields(trace.state_after)

    oracle_rows: dict[str, dict[str, object]] = {}
    oracle_rows["entry"] = gate.compare_fields(candidates["entry"], oracle_entry1)["rows"]
    for stage in (1, 2, 3):
        oracle = gate.read_state_frame(
            record_root / f"oracle_stage_kt00000001_s{stage}.bin",
            kt=1, stage=stage)
        oracle_rows[f"stage{stage}"] = gate.compare_fields(
            gate._rank0_fields(candidates[f"stage{stage}"]), oracle)["rows"]
    oracle_entry2 = gate.assemble_state_fields(record_root, 2, stage=None)
    oracle_rows["returned"] = gate.compare_fields(
        candidates["returned"], oracle_entry2)["rows"]
    oracle_stage3 = gate.read_state_frame(
        record_root / "oracle_stage_kt00000001_s3.bin", kt=1, stage=3)
    oracle_transition = gate.compare_fields(
        oracle_stage3, gate._rank0_fields(oracle_entry2))["rows"]

    arrays = {
        f"candidate_{checkpoint}_{field}": np.ascontiguousarray(
            candidates[checkpoint][field], dtype=np.float64)
        for checkpoint in CHECKPOINTS for field in FIELDS
    }
    if plant:
        key = "candidate_returned_u"
        planted = arrays[key].copy()
        planted.flat[0] = np.nextafter(planted.flat[0], np.float64(np.inf))
        arrays[key] = planted
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)

    removed_calls = None
    if owner == "all":
        method = LatLonCGridOceanModel.tendencies
        removed_calls = int(method.removed_calls[0])
        require(removed_calls > 0,
                "legacy LDF routing control removed no live operand bundle")
    return {
        "status": "CAPTURED",
        "claim_label": "INDEPENDENT_WITH_DECISION52_SSH",
        "worktree": worktree_stamp(repo=repo_root),
        "repo_root": str(repo_root),
        "owner_control": owner,
        "precision_policy": "fp64-libm",
        "jax_backend": jax.default_backend(),
        "candidate_rows": oracle_rows,
        "oracle_stage3_to_kt2_entry": oracle_transition,
        "legacy_ldf_removed_trace_calls": removed_calls,
        "plant": plant,
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256(output_npz),
    }


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _ladder_row(document: dict, kt: int, checkpoint: str, field: str) -> dict:
    for item in document["candidate_trajectory"]["checkpoints"]:
        if int(item["kt"]) == kt and item["checkpoint"] == checkpoint:
            return item["rows"][field]
    raise GateError(f"missing ladder row {kt}/{checkpoint}/{field}")


def _capture_reproduces(capture_doc: dict, ladder_doc: dict) -> bool:
    for checkpoint in ("entry", "stage1", "stage2", "stage3"):
        for field in FIELDS:
            if (capture_doc["candidate_rows"][checkpoint][field]
                    != _ladder_row(ladder_doc, 1, checkpoint, field)):
                return False
    for field in FIELDS:
        if (capture_doc["candidate_rows"]["returned"][field]
                != _ladder_row(ladder_doc, 2, "entry", field)):
            return False
    return True


def analyze(
    old_npz: Path,
    combined_npz: Path,
    all_npz: Path,
    plant_npz: Path,
    old_capture: dict,
    combined_capture: dict,
    all_capture: dict,
    old_ladder: dict,
    combined_ladder: dict,
    restored_ladder: dict,
) -> dict[str, object]:
    old = _load_arrays(old_npz)
    combined = _load_arrays(combined_npz)
    all_control = _load_arrays(all_npz)
    plant = _load_arrays(plant_npz)
    require(set(old) == set(combined) == set(all_control) == set(plant),
            "snapshot key sets differ")

    order = [f"candidate_{checkpoint}_{field}"
             for checkpoint in CHECKPOINTS for field in FIELDS]
    combined_differences = {
        key: _score_exact(combined[key], old[key]) for key in order
        if not np.array_equal(combined[key], old[key])
    }
    first_key = next((key for key in order if key in combined_differences), None)
    restored_differences = {
        key: _score_exact(all_control[key], old[key]) for key in order
        if not np.array_equal(all_control[key], old[key])
    }
    control_movement = {
        key: _score_exact(all_control[key], combined[key]) for key in order
        if not np.array_equal(all_control[key], combined[key])
    }
    plant_differences = {
        key: _score_exact(plant[key], all_control[key]) for key in order
        if not np.array_equal(plant[key], all_control[key])
    }
    plant_cells = sum(int(row["unequal"]) for row in plant_differences.values())

    oracle_transition_exact = all(
        row["bit_identical"]
        for row in combined_capture["oracle_stage3_to_kt2_entry"].values())
    p1 = (_capture_reproduces(old_capture, old_ladder)
          and _capture_reproduces(combined_capture, combined_ladder))
    p2 = first_key in ("candidate_stage1_u", "candidate_stage1_v")
    p3 = not restored_differences and all(
        _ladder_row(restored_ladder, kt, checkpoint, field)
        == _ladder_row(old_ladder, kt, checkpoint, field)
        for kt in range(1, 11)
        for checkpoint in ("entry", "stage1", "stage2", "stage3")
        for field in FIELDS)
    p4 = oracle_transition_exact
    p5 = bool(control_movement) and len(plant_differences) == 1 and plant_cells == 1
    predictions = {
        "R22-P1": "CONFIRMED" if p1 else "REFUTED",
        "R22-P2": "CONFIRMED" if p2 else "REFUTED",
        "R22-P3": "CONFIRMED" if p3 else "REFUTED",
        "R22-P4": "CONFIRMED" if p4 else "REFUTED",
        "R22-P5": "CONFIRMED" if p5 else "REFUTED",
    }
    recertified = all((p1, p3, p4, p5))
    return {
        "status": "RECERTIFIED" if recertified else "HELD_RESIDUAL_OWNER",
        "claim_label": "INDEPENDENT_WITH_DECISION52_SSH",
        "first_exact_combined_difference": (
            {"key": first_key, **combined_differences[first_key]}
            if first_key else None),
        "combined_exact_arrays_different": len(combined_differences),
        "combined_exact_difference_rows": combined_differences,
        "legacy_ldf_exact_arrays_different_from_old": len(restored_differences),
        "legacy_ldf_residual_rows": restored_differences,
        "legacy_ldf_control_arrays_moved": len(control_movement),
        "plant_rows": plant_differences,
        "plant_cells": plant_cells,
        "oracle_stage3_to_kt2_entry_bit_identical": oracle_transition_exact,
        "full_ladder_restored_to_round20": p3,
        "predictions": predictions,
        "decision58_eligible_next_round": recertified,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--repo-root", type=Path, required=True)
    capture_parser.add_argument("--deck-root", type=Path, required=True)
    capture_parser.add_argument("--record-root", type=Path, required=True)
    capture_parser.add_argument(
        "--owner", choices=("none", "both", "all"), required=True)
    capture_parser.add_argument("--output-npz", type=Path, required=True)
    capture_parser.add_argument("--json-out", type=Path, required=True)
    capture_parser.add_argument("--plant", action="store_true")

    analyze_parser = subparsers.add_parser("analyze")
    for name in ("old", "combined", "all", "plant"):
        analyze_parser.add_argument(f"--{name}-npz", type=Path, required=True)
    for name in ("old", "combined", "all"):
        analyze_parser.add_argument(f"--{name}-capture", type=Path, required=True)
    analyze_parser.add_argument("--old-ladder", type=Path, required=True)
    analyze_parser.add_argument("--combined-ladder", type=Path, required=True)
    analyze_parser.add_argument("--restored-ladder", type=Path, required=True)
    analyze_parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "capture":
            report = capture(
                args.repo_root.resolve(), args.deck_root, args.record_root,
                args.owner, args.output_npz, plant=args.plant)
            exit_code = 0
        else:
            report = analyze(
                args.old_npz, args.combined_npz, args.all_npz, args.plant_npz,
                json.loads(args.old_capture.read_text()),
                json.loads(args.combined_capture.read_text()),
                json.loads(args.all_capture.read_text()),
                json.loads(args.old_ladder.read_text()),
                json.loads(args.combined_ladder.read_text()),
                json.loads(args.restored_ladder.read_text()),
            )
            exit_code = 0 if report["status"] == "RECERTIFIED" else 2
    except (GateError, OSError, ValueError, KeyError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered)
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
