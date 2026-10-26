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
    parser.add_argument(
        "--mesh-mask", type=Path,
        default=Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
                     "cfgs/DINO/RUN_SEQDUMP_D180_1R/mesh_mask.nc"),
    )
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
        "mesh_mask": args.mesh_mask.resolve(),
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
    with Dataset(inputs["mesh_mask"]) as dataset:
        u_mask_raw = np.asarray(dataset.variables["umask"][0], dtype=np.float64)
    if u_mask_raw.shape == (JPK, NJ, NI):
        u_wet = u_mask_raw[0] > 0.5
    elif u_mask_raw.shape == (NJ, NI, JPK):
        u_wet = u_mask_raw[..., 0] > 0.5
    else:
        raise ValueError(f"unexpected umask shape {u_mask_raw.shape}")
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

    wet_nonzero = u_wet & (u_applied != 0.0)
    if np.any(wet_nonzero) and _rms(u_applied[wet_nonzero]) > 0.0:
        u_named_to_applied_rms_ratio = _rms(u_reference[wet_nonzero]) / _rms(
            u_applied[wet_nonzero]
        )
        u_named_applied_corr = float(np.corrcoef(
            u_reference[wet_nonzero], u_applied[wet_nonzero])[0, 1])
        pointwise_ratio = u_reference[wet_nonzero] / u_applied[wet_nonzero]
        pointwise_ratio_mean = float(np.mean(pointwise_ratio))
        pointwise_ratio_mean_error = abs(pointwise_ratio_mean - 2.0)
        pointwise_ratio_std = float(np.std(pointwise_ratio))
        pointwise_ratio_max_error = float(np.max(np.abs(pointwise_ratio - 2.0)))
        identity_ok = (pointwise_ratio_mean_error <= 1.0e-8
                       and pointwise_ratio_std <= 1.0e-8)
        planted_ratio = pointwise_ratio.copy()
        planted_ratio[0] += 1.0e-3
        ratio_std_plant_fires = float(np.std(planted_ratio)) > 1.0e-8
        uniform_shifted_ratio = pointwise_ratio + 1.0
        ratio_mean_plant_fires = abs(
            float(np.mean(uniform_shifted_ratio)) - 2.0) > 1.0e-8
    else:
        u_named_to_applied_rms_ratio = float("nan")
        u_named_applied_corr = float("nan")
        pointwise_ratio_mean = float("nan")
        pointwise_ratio_mean_error = float("inf")
        pointwise_ratio_std = float("inf")
        pointwise_ratio_max_error = float("inf")
        identity_ok = False
        ratio_std_plant_fires = False
        ratio_mean_plant_fires = False

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
    print("RETRACTION=CONFIRMED_NAMED_AND_APPLIED_DIFFER_2.0966766111")
    print("RETRACTION_REASON=zDt_2_equals_rDt_over_2_plus_324_dry_coastal_"
          "faces_from_union_mask_and_unreachable_REFUTE_gate")
    print("CHECK_SCOPE=named_slot_and_reference_are_two_output_paths_from_"
          "one_in_memory_array_not_independent_evidence")
    print(f"u_slot_nonzero_count={u_nonzero_count}")
    print(f"u_reference_nonzero_count={u_reference_nonzero_count}")
    print(f"u_lower_max_abs={u_lower_max:.17e}")
    print(f"v_slot_max_abs={v_slot_max:.17e}")
    print(f"v_reference_max_abs={v_reference_max:.17e}")
    print(f"u_storage_max_abs={u_storage_max:.17e}")
    print(f"v_storage_max_abs={v_storage_max:.17e}")
    print(f"u_storage_normalized_error={u_storage_norm_err:.17e}")
    print(f"u_named_to_applied_rms_ratio={u_named_to_applied_rms_ratio:.17e}")
    print(f"u_named_vs_applied_correlation={u_named_applied_corr:.17e}")
    print(f"u_wet_nonzero_count={int(wet_nonzero.sum())}")
    print(f"u_wet_pointwise_ratio_mean={pointwise_ratio_mean:.17e}")
    print(f"u_wet_pointwise_ratio_mean_error_from_2="
          f"{pointwise_ratio_mean_error:.17e}")
    print(f"u_wet_pointwise_ratio_std={pointwise_ratio_std:.17e}")
    print(f"u_wet_pointwise_ratio_max_error_from_2="
          f"{pointwise_ratio_max_error:.17e}")
    print(f"v_applied_max_abs={float(np.max(np.abs(v_applied))):.17e}")
    print(f"rounding_bar={eps_bar:.17e}")
    print(f"CONTROL_lower_level_plant_fires={lower_plant_fires}")
    print(f"CONTROL_top_storage_plant_fires={top_plant_fires}")
    print(f"CONTROL_wet_ratio_std_plant_fires={ratio_std_plant_fires}")
    print(f"CONTROL_wet_ratio_mean_plant_fires={ratio_mean_plant_fires}")
    if (not lower_plant_fires or not top_plant_fires
            or not ratio_std_plant_fires or not ratio_mean_plant_fires):
        print("CLASSIFICATION=INVALID_CONTROL_FAILURE")
        return 3

    print("WET_ARITHMETIC_IDENTITY=" + (
        "CONFIRMED_NAMED_IS_TWICE_APPLIED_MEAN_AND_STD_LE_1E-8" if identity_ok
        else "REFUTED_OR_UNRESOLVED"))

    if u_nonzero_count == 0 and u_reference_nonzero_count > 0:
        classification = "REFUTED_UNWIRED"
    elif not lower_ok or not meridional_ok or u_storage_norm_err >= 0.05:
        classification = "REFUTED_WRONG_SOURCE"
    elif u_nonzero_count > 0 and storage_ok:
        classification = "CHECKED_CLEAN_WRITE_ONLY"
    else:
        classification = "UNRESOLVED"
    print(f"CLASSIFICATION={classification}")
    print("PLACEMENT_OWNERSHIP=UNRESOLVED")
    return 0 if classification == "CHECKED_CLEAN_WRITE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
