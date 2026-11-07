#!/usr/bin/env python3
"""Re-score registered matrix PE gates with legacy and current diagnostics.

Pre-registration (B4, 2026-09-19)
---------------------------------
Measurements: ``PE_rel_final`` and the emitted PASS/FAIL verdict for the
registered quick MPAS LOCK_EXCHANGE and lat-lon OVERFLOW cases, each run twice
from the same repaired code: once with the pre-``6cb7419da`` modular PE
diagnostic and once with the current shared diagnostic.

CONFIRM the scoring change if either pair differs.  REFUTE it if both pairs are
bit-identical.  Emit ``DECISION_NEEDED`` if either acceptance verdict flips.
No gate threshold, timestep, duration, resolution, or model configuration is
changed by this probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_DIR = _REPO_ROOT / "scripts" / "matrix"
if str(_MATRIX_DIR) not in sys.path:
    sys.path.insert(0, str(_MATRIX_DIR))

from legoesm import constants  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from ocean_test_matrix import config, experiments  # noqa: E402
from ocean_test_matrix.testcase import TEST_MATRIX  # noqa: E402


def _legacy_reference_thickness_pe(state, grid_type, grid, z_coord) -> float:
    """Pre-6cb7419da modular ``_compute_rpe`` source transcription."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d

        temperature = np.asarray(
            sh_synthesis_3d(grid, state.T_hat.data), dtype=np.float64)
        salinity = np.asarray(
            sh_synthesis_3d(grid, state.S_hat.data), dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)
    elif grid_type == "mpas":
        temperature = np.asarray(state.T.data, dtype=np.float64)
        salinity = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.areaCell, dtype=np.float64)
    else:
        temperature = np.asarray(state.T.data, dtype=np.float64)
        salinity = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    from legoesm.ocean.eos import linear_eos

    rho = np.asarray(
        linear_eos(
            jnp.asarray(temperature),
            jnp.asarray(salinity),
            jnp.zeros_like(jnp.asarray(temperature)),
            rho_ref=constants.rho_ocean,
            alpha_T=2.0e-4,
            beta_S=0.0,
            T_ref=15.0,
        ),
        dtype=np.float64,
    )
    spatial_shape = temperature.shape[:-1]
    area_bc = area.reshape(spatial_shape)
    pe = 0.0
    for k in range(len(z_full)):
        pe += float(np.nansum(
            rho[..., k] * z_full[k] * dz[k] * area_bc))
    return config._G_EARTH * pe


def _registered_case(case: str, grid_type: str):
    matches = [
        tc for tc in TEST_MATRIX
        if tc.case == case and tc.grid_type == grid_type
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"B4_CASE_RESOLUTION_ERROR: expected one registered "
            f"{case}/{grid_type} case, found {len(matches)}")
    return matches[0]


def _read_pe_score(result_path: Path) -> float:
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PE_rel_final:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"B4_SCORE_MISSING: {result_path}")


def _assert_identical_trajectory(output_root: Path, case: str) -> None:
    """Prove the read-only scorer swap did not change physical snapshots."""
    legacy_path = (
        output_root / "legacy_reference_unmasked" / case
        / "snapshots_native.npz"
    )
    shared_path = (
        output_root / "shared_wet_live" / case / "snapshots_native.npz"
    )
    with np.load(legacy_path) as legacy, np.load(shared_path) as shared:
        if legacy.files != shared.files:
            raise RuntimeError(
                f"B4_TRAJECTORY_KEYS_CHANGED: {case}: "
                f"{legacy.files!r} != {shared.files!r}")
        n_keys = len(legacy.files)
        changed = [
            key for key in legacy.files
            if not np.array_equal(legacy[key], shared[key], equal_nan=True)
        ]
    if changed:
        raise RuntimeError(
            f"B4_TRAJECTORY_CHANGED: {case}: {', '.join(changed)}")
    print(
        f"B4_TRAJECTORY_IDENTICAL case={case} keys={n_keys}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"B4_OUTPUT_EXISTS: {args.output_root}")
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"B4_CPU_REQUIRED: backend is {jax.default_backend()!r}")
    set_policy(PrecisionPolicy.fp64())

    cases = (
        ("lock_exchange", "mpas", experiments.run_lock_exchange),
        ("overflow", "latlon", experiments.run_overflow),
    )
    current_scorer = experiments._compute_rpe
    records = []
    for scorer_name, scorer in (
        ("legacy_reference_unmasked", _legacy_reference_thickness_pe),
        ("shared_wet_live", current_scorer),
    ):
        experiments._compute_rpe = scorer
        for case, grid_type, runner in cases:
            tc = _registered_case(case, grid_type)
            output_dir = args.output_root / scorer_name / case
            status, wall, notes = runner(tc, output_dir, tc.quick_days)
            record = {
                "case": case,
                "days": tc.quick_days,
                "grid_type": grid_type,
                "notes": notes,
                "pe_rel_final": _read_pe_score(output_dir / "results.txt"),
                "resolution": tc.resolution,
                "scorer": scorer_name,
                "status": status,
                "wall_seconds": wall,
            }
            records.append(record)
            print("B4_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    experiments._compute_rpe = current_scorer

    by_case = {}
    for record in records:
        by_case.setdefault(record["case"], {})[record["scorer"]] = record
    for case in by_case:
        _assert_identical_trajectory(args.output_root, case)
    flips = [
        case for case, pair in by_case.items()
        if pair["legacy_reference_unmasked"]["status"]
        != pair["shared_wet_live"]["status"]
    ]
    print("B4_VERDICT CONFIRM")
    print("DECISION_NEEDED: " + (", ".join(flips) if flips else "NONE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
