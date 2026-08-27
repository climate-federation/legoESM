#!/usr/bin/env python3
"""Fail-closed verifier for NEMO's named DINO jpdyn_tau restart slot."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

NI = 52
NJ = 199
JPK = 36


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _load_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f8")
    expected = NI * NJ
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} float64 values, got {values.size}")
    return values.reshape(NJ, NI)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--restart", required=True)
    parser.add_argument("--rdt-seconds", required=True, type=float)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    restart = run_dir / args.restart
    script = Path(__file__).resolve()
    inputs = {
        "restart": restart,
        "u_reference": run_dir / "trddyn_dump_utrd_tau.bin",
        "v_reference": run_dir / "trddyn_dump_vtrd_tau.bin",
        "u_pre": run_dir / "zdf_dump_u1_prestress.bin",
        "u_post": run_dir / "zdf_dump_u1_poststress.bin",
        "v_pre": run_dir / "zdf_dump_v1_prestress.bin",
        "v_post": run_dir / "zdf_dump_v1_poststress.bin",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    print("PROVENANCE_BEGIN")
    print(f"git_sha={_git_sha()}")
    print(f"script={script}")
    print(f"script_sha256={_sha256(script)}")
    print(f"run_dir={run_dir}")
    print(f"rdt_seconds={args.rdt_seconds:.17g}")
    for name, path in inputs.items():
        print(f"{name}={path}")
        print(f"{name}_sha256={_sha256(path)}")
    print("PROVENANCE_END")

    with Dataset(restart) as dataset:
        u_slot = np.asarray(dataset.variables["utrd_tau"][0], dtype=np.float64)
        v_slot = np.asarray(dataset.variables["vtrd_tau"][0], dtype=np.float64)
    expected_shape = (JPK, NJ, NI)
    if u_slot.shape != expected_shape or v_slot.shape != expected_shape:
        raise ValueError(
            f"restart slot shape mismatch: u={u_slot.shape}, v={v_slot.shape}, "
            f"expected={expected_shape}"
        )

    u_reference = _load_bin(inputs["u_reference"])
    v_reference = _load_bin(inputs["v_reference"])
    u_bracket = _load_bin(inputs["u_post"]) - _load_bin(inputs["u_pre"])
    v_bracket = _load_bin(inputs["v_post"]) - _load_bin(inputs["v_pre"])
    u_applied = u_bracket / args.rdt_seconds
    v_applied = v_bracket / args.rdt_seconds
    u_storage_residual = u_slot[0] - u_reference
    v_storage_residual = v_slot[0] - v_reference

    eps_bar = 8.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(u_reference)))
    )
    u_nonzero_count = int(np.count_nonzero(u_slot[0]))
    u_reference_nonzero_count = int(np.count_nonzero(u_reference))
    u_lower_max = float(np.max(np.abs(u_slot[1:])))
    v_slot_max = float(np.max(np.abs(v_slot)))
    v_reference_max = float(np.max(np.abs(v_reference)))
    u_storage_max = float(np.max(np.abs(u_storage_residual)))
    v_storage_max = float(np.max(np.abs(v_storage_residual)))
    reference_mask = u_reference != 0.0
    if np.any(reference_mask):
        u_storage_norm_err = _rms(u_storage_residual[reference_mask]) / _rms(
            u_reference[reference_mask]
        )
    else:
        u_storage_norm_err = float("inf")

    source_mask = (u_reference != 0.0) | (u_applied != 0.0)
    if np.any(source_mask) and _rms(u_applied[source_mask]) > 0.0:
        u_named_to_applied_rms_ratio = _rms(u_reference[source_mask]) / _rms(
            u_applied[source_mask]
        )
        u_named_vs_applied_norm_err = _rms(
            u_reference[source_mask] - u_applied[source_mask]
        ) / _rms(u_applied[source_mask])
        u_named_applied_corr = float(
            np.corrcoef(u_reference[source_mask], u_applied[source_mask])[0, 1]
        )
    else:
        u_named_to_applied_rms_ratio = float("nan")
        u_named_vs_applied_norm_err = float("inf")
        u_named_applied_corr = float("nan")

    lower_ok = u_lower_max == 0.0 and float(np.max(np.abs(v_slot[1:]))) == 0.0
    meridional_ok = v_slot_max == 0.0 and v_reference_max == 0.0
    storage_ok = u_storage_max <= eps_bar and v_storage_max <= eps_bar

    lower_plant = u_slot.copy()
    lower_plant[1, 1, 1] = np.nextafter(0.0, 1.0)
    lower_plant_fires = float(np.max(np.abs(lower_plant[1:]))) != 0.0
    top_plant = u_slot[0].copy()
    top_plant[1, 1] += 2.0 * eps_bar
    top_plant_error = float(np.max(np.abs(top_plant - u_reference)))
    top_plant_fires = top_plant_error > eps_bar

    print("RETRACTION=prior applied-increment/rDt hook was not NEMO_named_utrd_tau")
    print(f"u_slot_nonzero_count={u_nonzero_count}")
    print(f"u_reference_nonzero_count={u_reference_nonzero_count}")
    print(f"u_lower_max_abs={u_lower_max:.17e}")
    print(f"v_slot_max_abs={v_slot_max:.17e}")
    print(f"v_reference_max_abs={v_reference_max:.17e}")
    print(f"u_storage_max_abs={u_storage_max:.17e}")
    print(f"v_storage_max_abs={v_storage_max:.17e}")
    print(f"u_storage_normalized_error={u_storage_norm_err:.17e}")
    print(f"u_named_to_applied_rms_ratio={u_named_to_applied_rms_ratio:.17e}")
    print(f"u_named_vs_applied_normalized_error={u_named_vs_applied_norm_err:.17e}")
    print(f"u_named_vs_applied_correlation={u_named_applied_corr:.17e}")
    print(f"v_applied_max_abs={float(np.max(np.abs(v_applied))):.17e}")
    print(f"rounding_bar={eps_bar:.17e}")
    print(f"CONTROL_lower_level_plant_fires={lower_plant_fires}")
    print(f"CONTROL_top_storage_plant_fires={top_plant_fires}")
    if not lower_plant_fires or not top_plant_fires:
        print("CLASSIFICATION=INVALID_CONTROL_FAILURE")
        return 3

    if u_named_vs_applied_norm_err >= 0.25:
        print("SOURCE_DISTINCTION=CONFIRMED_NAMED_AND_APPLIED_DIFFER")
    elif u_named_vs_applied_norm_err <= 0.05:
        print("SOURCE_DISTINCTION=REFUTED_NAMED_AND_APPLIED_MATCH")
    else:
        print("SOURCE_DISTINCTION=UNRESOLVED")

    if u_nonzero_count == 0 and u_reference_nonzero_count > 0:
        classification = "REFUTED_UNWIRED"
    elif not lower_ok or not meridional_ok or u_storage_norm_err >= 0.05:
        classification = "REFUTED_WRONG_SOURCE"
    elif u_nonzero_count > 0 and storage_ok:
        classification = "CONFIRMED_NAMED_TAU_SLOT_WIRED"
    else:
        classification = "UNRESOLVED"
    print(f"CLASSIFICATION={classification}")
    print("PLACEMENT_OWNERSHIP=UNRESOLVED")
    return 0 if classification == "CONFIRMED_NAMED_TAU_SLOT_WIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
