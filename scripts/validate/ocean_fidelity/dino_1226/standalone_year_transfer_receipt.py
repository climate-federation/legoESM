#!/usr/bin/env python3
"""Bind the frozen T1 standalone-year prediction against NEMO year 1.

This is a read-only scorer. It accepts the standalone ``--snap-final``
artifact, the independent NEMO ``kt=11520`` restart, and the canonical DINO
mesh. Horizontal alignment is shape-driven and fail-closed: a standalone
snapshot may carry either NEMO's complete 199x52 construction frame or the
legacy two-ring-stripped core. The dry terminal NEMO ``jpk`` level is removed
in both cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np


SCHEMA = "dino_standalone_year_transfer_receipt_v1"
SESSION_ID = "01a053d4-8e9f-7212-bbdb-19ba2d64e140"
STEPS = 11520
DAY = 360
CONFIRM_SST_RMS_DEGC = 0.02
REFUTE_SST_RMS_DEGC = 0.10


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_provenance() -> tuple[str, list[str]]:
    env = os.environ.copy()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True, env=env).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True, text=True, capture_output=True, env=env).stdout.splitlines()
    return sha, dirty


def _yxz(value: Any) -> np.ndarray:
    array = np.asarray(value).squeeze()
    if array.ndim != 3:
        raise ValueError(f"expected a three-dimensional NEMO field, got {array.shape}")
    return np.moveaxis(array, 0, -1)


def align_nemo_to_standalone(
        standalone_shape: tuple[int, int, int], nemo: np.ndarray,
        mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Align NEMO T-frame data to a full-ring or two-ring-stripped snapshot."""
    if nemo.shape != mask.shape:
        raise ValueError(f"NEMO field shape {nemo.shape} != mask shape {mask.shape}")
    if np.any(mask[..., -1]):
        raise ValueError("NEMO terminal jpk is not uniformly dry")
    ny, nx, nz = nemo.shape
    full = (ny, nx, nz - 1)
    core = (ny - 4, nx - 4, nz - 1)
    if standalone_shape == full:
        return nemo[..., :-1], mask[..., :-1], "full_nemo_construction_frame"
    if standalone_shape == core:
        return (nemo[2:-2, 2:-2, :-1], mask[2:-2, 2:-2, :-1],
                "two_ring_stripped_physical_core")
    raise ValueError(
        f"standalone shape {standalone_shape} is neither full {full} nor core {core}")


def align_nemo_surface(nemo: np.ndarray, mode: str) -> np.ndarray:
    if nemo.ndim != 2:
        raise ValueError(f"expected a two-dimensional surface field, got {nemo.shape}")
    if mode == "full_nemo_construction_frame":
        return nemo
    if mode == "two_ring_stripped_physical_core":
        return nemo[2:-2, 2:-2]
    raise ValueError(f"unknown frame-alignment mode {mode!r}")


def statistics(delta: np.ndarray, wet: np.ndarray) -> dict[str, float | int]:
    if delta.shape != wet.shape:
        raise ValueError(f"delta shape {delta.shape} != wet shape {wet.shape}")
    values = np.asarray(delta, dtype=np.float64)[np.asarray(wet, dtype=bool)]
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("scored population is empty or non-finite")
    return {
        "n": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))),
        "bias": float(np.mean(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def classify(sst_rms: float) -> str:
    if sst_rms <= CONFIRM_SST_RMS_DEGC:
        return "CONFIRM"
    if sst_rms >= REFUTE_SST_RMS_DEGC:
        return "REFUTE"
    return "INCONCLUSIVE"


def self_test() -> dict[str, str]:
    nemo = np.arange(6 * 8 * 3, dtype=np.float64).reshape(6, 8, 3)
    mask = np.ones_like(nemo, dtype=bool)
    mask[..., -1] = False
    full, full_mask, full_mode = align_nemo_to_standalone(
        (6, 8, 2), nemo, mask)
    core, core_mask, core_mode = align_nemo_to_standalone(
        (2, 4, 2), nemo, mask)
    np.testing.assert_array_equal(full, nemo[..., :-1])
    np.testing.assert_array_equal(core, nemo[2:-2, 2:-2, :-1])
    assert full_mask.all() and core_mask.all()
    assert full_mode == "full_nemo_construction_frame"
    assert core_mode == "two_ring_stripped_physical_core"
    try:
        align_nemo_to_standalone((5, 5, 2), nemo, mask)
    except ValueError:
        shape_plant = "FIRED"
    else:
        raise RuntimeError("planted unsupported frame did not fail")
    if classify(0.01) != "CONFIRM" or classify(0.39) != "REFUTE":
        raise RuntimeError("frozen prediction classifier controls did not fire")
    return {
        "full_frame_alignment": "PASS",
        "core_frame_alignment": "PASS",
        "unsupported_frame_plant": shape_plant,
        "prediction_classifier_plants": "FIRED",
    }


def run(args: argparse.Namespace) -> int:
    producer, dirty = git_provenance()
    if dirty:
        raise SystemExit("REFUSING dirty tracked producer: " + "; ".join(dirty))
    standalone_dir = args.standalone_dir.resolve()
    nemo_run = args.nemo_run.resolve()
    mesh = args.mesh.resolve()
    manifest_path = standalone_dir / "manifest.json"
    initial_path = standalone_dir / "initial_receipt.json"
    reducer_path = standalone_dir / "reducer_convention_receipt.json"
    snapshot_path = standalone_dir / "snapshots" / "day_0360.npz"
    restart_path = nemo_run / "DINO_00011520_restart.nc"
    inputs = [manifest_path, initial_path, reducer_path, snapshot_path,
              restart_path, mesh]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing T1 receipt inputs: " + ", ".join(missing))

    manifest = json.loads(manifest_path.read_text())
    reducer_receipt = json.loads(reducer_path.read_text())
    required_manifest = {
        "member": 0,
        "steps_completed": STEPS,
        "compute_dtype": "float64",
        "storage_dtype": "float64",
        "snap_final_requested": True,
        "final_snapshot_day": DAY,
        "final_snapshot_present": True,
        "twin_start_mode": "standalone",
        "recipe": "nemo_dino_kamm_mlf",
    }
    drift = {name: {"actual": manifest.get(name), "expected": expected}
             for name, expected in required_manifest.items()
             if manifest.get(name) != expected}
    if drift:
        raise SystemExit(f"standalone manifest contract drift: {drift}")
    if (Path(reducer_receipt.get("reducer_mesh", "")).resolve() != mesh
            or reducer_receipt.get("reducer_mesh_sha256") != file_sha256(mesh)):
        raise SystemExit(
            "standalone reducer receipt does not identify the supplied "
            "canonical mesh path and hash")
    capture = next(
        (row for row in manifest.get("captures", [])
         if row.get("day") == DAY and row.get("step") == STEPS), None)
    if capture is None or capture.get("snapshot_sha256") != file_sha256(snapshot_path):
        raise SystemExit("day-360 final snapshot is absent or hash-mismatched")

    with np.load(snapshot_path) as lego, nc.Dataset(restart_path) as restart, \
            nc.Dataset(mesh) as mesh_data:
        lego_t = np.asarray(lego["T"], dtype=np.float64)
        lego_s = np.asarray(lego["S"], dtype=np.float64)
        lego_eta = np.asarray(lego["eta"], dtype=np.float64)
        nemo_t = _yxz(restart["tn"][:])
        nemo_s = _yxz(restart["sn"][:])
        nemo_eta = np.asarray(restart["sshn"][:], dtype=np.float64).squeeze()
        nemo_mask = _yxz(mesh_data["tmask"][0]) > 0.5
        aligned_t, wet, frame_mode = align_nemo_to_standalone(
            lego_t.shape, nemo_t, nemo_mask)
        aligned_s, wet_s, frame_mode_s = align_nemo_to_standalone(
            lego_s.shape, nemo_s, nemo_mask)
        if frame_mode_s != frame_mode or not np.array_equal(wet_s, wet):
            raise SystemExit("T/S frame alignment disagrees")
        aligned_eta = align_nemo_surface(nemo_eta, frame_mode)
        if lego_eta.shape != aligned_eta.shape:
            raise SystemExit(
                f"standalone eta shape {lego_eta.shape} != aligned NEMO "
                f"shape {aligned_eta.shape}")
        surface_wet = wet[..., 0]
        scores = {
            "sst_degC": statistics(lego_t[..., 0] - aligned_t[..., 0],
                                    surface_wet),
            "ssh_m": statistics(lego_eta - aligned_eta, surface_wet),
            "T_3d_degC": statistics(lego_t - aligned_t, wet),
            "S_3d_PSU": statistics(lego_s - aligned_s, wet),
        }

    outcome = classify(float(scores["sst_degC"]["rms"]))
    artifact = {
        "schema": SCHEMA,
        "session_id": SESSION_ID,
        "producer_git_sha": producer,
        "producer_dirty_tracked_files": dirty,
        "outcome": outcome,
        "claim_scope": {
            "confirmed": "one-year cold-start transfer discriminator",
            "not_claimed": "six-member 20-year climate-family equivalence",
            "runner_manifest_claim_admissible": manifest.get("claim_admissible"),
            "reason": "the runner reserves claim_admissible for its 20-year schedule",
        },
        "frozen_prediction": {
            "text": "independent-year SST RMS 0.39 C collapses toward ~0.01 C",
            "confirm_sst_rms_degC_lte": CONFIRM_SST_RMS_DEGC,
            "refute_sst_rms_degC_gte": REFUTE_SST_RMS_DEGC,
        },
        "frame_alignment": {
            "mode": frame_mode,
            "standalone_T_shape": list(lego_t.shape),
            "nemo_T_shape_before_alignment": list(nemo_t.shape),
            "operation": "retain full horizontal construction frame; drop dry terminal jpk",
        },
        "scores": scores,
        "four_rung_ladder": [
            {"rung": "twin_at_day180", "sst_rms_degC": 0.005,
             "scope": "restart-bridged twin"},
            {"rung": "bridged_at_step2_to_day360", "sst_rms_degC": 0.0104,
             "ssh_rms_mm": 0.29,
             "scope": "359-day free run; daily endpoint offset 0.94 day"},
            {"rung": "standalone_before_repairs_day360", "sst_rms_degC": 0.39,
             "T_3d_bias_degC": 0.045,
             "scope": "fully independent from rest; pre-fix"},
            {"rung": "standalone_after_repairs_day360",
             "sst_rms_degC": scores["sst_degC"]["rms"],
             "ssh_rms_mm": float(scores["ssh_m"]["rms"]) * 1000.0,
             "T_3d_rms_degC": scores["T_3d_degC"]["rms"],
             "T_3d_bias_degC": scores["T_3d_degC"]["bias"],
             "S_3d_rms_PSU": scores["S_3d_PSU"]["rms"],
             "scope": "fully independent from rest; member 0; fixed"},
        ],
        "repairs": [
            "initialization geometry: Mercator latitude, stepped depth/mask, construction frame",
            "initialization profile arithmetic: scalar libm tanh, source association, live anchors",
            "coupled degenerate-Euler FCT: collapsed MLF horizontal/vertical rate paths",
        ],
        "controls": self_test(),
        "inputs": {str(path): {"bytes": path.stat().st_size,
                               "sha256": file_sha256(path)}
                   for path in inputs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"OUTCOME={outcome}")
    print(f"FRAME_ALIGNMENT={frame_mode}")
    print(f"SST_RMS_DEGC={scores['sst_degC']['rms']:.17g}")
    print(f"SST_MAX_DEGC={scores['sst_degC']['max_abs']:.17g}")
    print(f"SSH_RMS_MM={float(scores['ssh_m']['rms']) * 1000.0:.17g}")
    print(f"T3_RMS_DEGC={scores['T_3d_degC']['rms']:.17g}")
    print(f"T3_BIAS_DEGC={scores['T_3d_degC']['bias']:.17g}")
    print(f"S3_RMS_PSU={scores['S_3d_PSU']['rms']:.17g}")
    print(f"WROTE={args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--standalone-dir", type=Path, required=True)
    parser.add_argument("--nemo-run", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
