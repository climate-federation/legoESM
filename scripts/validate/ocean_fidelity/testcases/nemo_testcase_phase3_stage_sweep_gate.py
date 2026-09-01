#!/usr/bin/env python3
"""Scaling-first WS-RK3 stage owner controls for the NEMO testcase lane.

The public testcase cards are never changed here.  Each arm is a private,
one-variable test hook, and every stage row compares NEMO's instantaneous Kaa
velocity with legoESM's instantaneous prognostic U-face velocity at that same
stage.  Final rows use the next Nbb entry state.  Owner labels are derived only
after the arm movement is compared with the faithful residual scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


BAR = 1.0e-15
ROOTS = {
    "LOCK_EXCHANGE-zco": Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"),
    "OVERFLOW-zps": Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10"),
}
DIMS = {
    "LOCK_EXCHANGE-zco": (134, 7, 21),
    "OVERFLOW-zps": (206, 7, 101),
}
EXPECTED_LEVELS = {
    1: {"Kaa": 3},
    2: {"Kaa": 2},
    3: {"Kaa": 3},
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_stage(path: Path, case: str, expected_stage: int) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kaa, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *DIMS[case], 2, 64), f"{path}: bad header {header}"
    )
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    require(
        (kt, stage, kaa) == (1, expected_stage, EXPECTED_LEVELS[expected_stage]["Kaa"]),
        f"{path}: wrong time-level registry {header}",
    )
    return {
        "kt": kt,
        "stage": stage,
        "Kaa": kaa,
        "T": _xyz(values[:count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
    }


def read_entry(path: Path, case: str) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nbb, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *DIMS[case], 2, 64), f"{path}: bad header {header}"
    )
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require((kt, nbb) == (2, 3), f"{path}: expected kt=2/Nbb=3")
    return {
        "T": _xyz(values[:count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
    }


def score(name: str, oracle, candidate, mask, *, plant=False, quantity="u") -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate {candidate.dtype}")
    require(bool(active.any()), f"{name}: empty mask")
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: non-finite")
    absolute = float(np.max(np.abs(candidate[active] - oracle[active])))
    reference = float(np.max(np.abs(oracle[active])))
    normalized = absolute / max(reference, 1.0)
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "normalized_max_abs": normalized,
        "absolute_max": absolute,
        "reference_max_abs": reference,
        "bar": BAR,
        "n": int(active.sum()),
        "frame": (
            "instantaneous_prognostic_Kaa" if quantity == "u" else "instantaneous_tracer_Nbb"
        ),
        "staggering_and_reduction": (
            "oracle uu(:,:,:,Kaa) and legoESM's exposed WS-RK3 stage u are "
            "both instantaneous 3-D C-grid U-face velocities; both are scored "
            "on the same wet U-face mask by elementwise L-infinity, with no "
            "vertical or substep-time reduction"
            if quantity == "u"
            else "oracle ts(:,:,:,temperature,Nbb) and legoESM T are both "
            "instantaneous 3-D T-point tracers; both are scored on the same "
            "wet T-cell mask by elementwise L-infinity, with no vertical or "
            "time reduction"
        ),
    }


def movement(faithful, control, mask) -> dict:
    faithful = np.asarray(faithful, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    value = float(np.max(np.abs(control[active] - faithful[active])))
    return {"absolute_max": value}


def expected_masks(card) -> dict:
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    return {"T": active, "u": u}


def classify_arm(
    faithful_row: dict, control_row: dict, arm_movement: dict, *, improving: bool
) -> dict:
    residual = faithful_row["absolute_max"]
    moved = arm_movement["absolute_max"]
    ratio = moved / residual if residual else float("inf")
    clears = control_row["normalized_max_abs"] <= BAR
    if clears:
        classification = "CONFIRMED"
    elif improving and ratio >= 0.1:
        classification = "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
    else:
        classification = "REFUTED_AS_PRIMARY_OWNER"
    return {
        "classification": classification,
        "faithful_absolute_error": residual,
        "control_absolute_error": control_row["absolute_max"],
        "arm_movement_absolute": moved,
        "movement_over_faithful_residual": ratio,
        "improves_residual": improving,
        "clears_bar": clears,
    }


def run(case: str, root: Path, *, plant_stage=False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(case)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    artifacts = {}
    oracle_stages = {}
    for stage in (1, 2, 3):
        path = root / f"oracle_stage_kt00000001_s{stage}.bin"
        require(path.is_file(), f"missing {path}")
        oracle_stages[stage] = read_stage(path, case, stage)
        artifacts[path.name] = sha256(path)
    entry_path = root / "oracle_step_entry_kt00000002.bin"
    require(entry_path.is_file(), f"missing {entry_path}")
    oracle_entry = read_entry(entry_path, case)
    artifacts[entry_path.name] = sha256(entry_path)

    if case == "OVERFLOW-zps":
        arms = {
            "faithful": _NEMOWSRK3TestHooks(),
            "omit_stage_primary_velocity_correction": _NEMOWSRK3TestHooks(
                stage_barotropic_correction=False
            ),
        }
    else:
        arms = {
            "faithful": _NEMOWSRK3TestHooks(),
            "omit_stage_primary_velocity_correction": _NEMOWSRK3TestHooks(
                stage_barotropic_correction=False
            ),
            "omit_momentum_transport_reconcile": _NEMOWSRK3TestHooks(
                momentum_transport_reconcile=False
            ),
        }

    states = {}
    stage_states = {}
    rows = []
    for arm, hooks in arms.items():
        states[arm] = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            _nemo_ws_test_hooks=hooks,
        ).step(card.recipe.initial_state, dt=card.dt_s)
        stage_states[arm] = {}
        for stage in (1, 2, 3):
            if stage == 3:
                stage_state = states[arm]
            else:
                stage_state = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=hooks._replace(expose_momentum_stage=stage),
                ).step(card.recipe.initial_state, dt=card.dt_s)
            stage_states[arm][stage] = stage_state
            row = score(
                f"{case}.kt1.stage{stage}.{arm}.instantaneous_u",
                oracle_stages[stage]["u"][..., :nlev],
                np.asarray(stage_state.u.data)[:, 1:, :],
                masks["u"],
                plant=plant_stage and arm == "faithful" and stage == 1,
            )
            row["verdict"] = arm == "faithful"
            rows.append(row)

    faithful_u = score(
        f"{case}.kt2.faithful.instantaneous_u",
        oracle_entry["u"][..., :nlev],
        np.asarray(states["faithful"].u.data)[:, 1:, :],
        masks["u"],
    )
    faithful_u["frame"] = "instantaneous_prognostic_Nbb"
    rows.append(faithful_u)
    faithful_T = score(
        f"{case}.kt2.faithful.T",
        oracle_entry["T"][..., :nlev],
        np.asarray(states["faithful"].T.data),
        masks["T"],
        quantity="T",
    )
    faithful_T["frame"] = "instantaneous_tracer_Nbb"
    rows.append(faithful_T)

    arm_results = {}
    for arm in arms:
        if arm == "faithful":
            continue
        control_u = score(
            f"{case}.kt2.{arm}.instantaneous_u",
            oracle_entry["u"][..., :nlev],
            np.asarray(states[arm].u.data)[:, 1:, :],
            masks["u"],
        )
        control_u.update({"verdict": False, "frame": "instantaneous_prognostic_Nbb"})
        rows.append(control_u)
        control_T = score(
            f"{case}.kt2.{arm}.T",
            oracle_entry["T"][..., :nlev],
            np.asarray(states[arm].T.data),
            masks["T"],
            quantity="T",
        )
        control_T.update({"verdict": False, "frame": "instantaneous_tracer_Nbb"})
        rows.append(control_T)
        target = "T" if case == "OVERFLOW-zps" else "u"
        faithful_row = faithful_T if target == "T" else faithful_u
        control_row = control_T if target == "T" else control_u
        faithful_field = (
            np.asarray(states["faithful"].T.data)
            if target == "T"
            else np.asarray(states["faithful"].u.data)[:, 1:, :]
        )
        control_field = (
            np.asarray(states[arm].T.data)
            if target == "T"
            else np.asarray(states[arm].u.data)[:, 1:, :]
        )
        arm_move = movement(faithful_field, control_field, masks[target])
        improving = control_row["absolute_max"] < faithful_row["absolute_max"]
        arm_results[arm] = classify_arm(faithful_row, control_row, arm_move, improving=improving)
        arm_results[arm]["target"] = target
        arm_results[arm]["one_variable"] = (
            "stage_barotropic_correction" if "primary" in arm else "momentum_transport_reconcile"
        )

    if case == "LOCK_EXCHANGE-zco" and not any(
        result["clears_bar"] for result in arm_results.values()
    ):
        ownership = {
            "classification": "UNMEASURED_AFTER_TWO_ARMS",
            "reason": (
                "the instantaneous-u tail resisted both registered "
                "one-variable arms; no further owner is assigned this round"
            ),
            "arms": arm_results,
        }
    else:
        faithful_stage1 = next(
            row for row in rows if row["name"] == f"{case}.kt1.stage1.faithful.instantaneous_u"
        )
        control_stage1 = next(
            row
            for row in rows
            if row["name"]
            == f"{case}.kt1.stage1.omit_stage_primary_velocity_correction.instantaneous_u"
        )
        ownership = {
            "classification": arm_results["omit_stage_primary_velocity_correction"][
                "classification"
            ],
            "scheme_component_requirement": "CONFIRMED_REQUIRED",
            "direct_stage1_faithful_absolute_error_m_s": faithful_stage1["absolute_max"],
            "direct_stage1_omission_absolute_error_m_s": control_stage1["absolute_max"],
            "cancellation_warning": (
                "omission improves the final T residual but makes the direct "
                "instantaneous stage-1 u comparison much worse; the T movement "
                "is scale-compatible evidence for a residual contribution, not "
                "evidence that NEMO's required correction should be removed"
            ),
            "arms": arm_results,
        }

    failed = [row["name"] for row in rows if row["status"] == "DEBT" and row.get("verdict", True)]
    return {
        "format": "nemo-testcase-l1-phase3-stage-sweep-v1",
        "case": case,
        "status": "AT-BAR" if not failed else "DEBT",
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "candidate_dtypes": {
            "T": str(np.asarray(states["faithful"].T.data).dtype),
            "u": str(np.asarray(states["faithful"].u.data).dtype),
        },
        "oracle_root": str(root),
        "time_level_registry": {
            "stage1": "Kaa=3 after stprk3_stg.F90:433-446",
            "stage2": "Kaa=2 after stprk3_stg.F90:433-446",
            "stage3": "Kaa=3 after stprk3_stg.F90:433-446",
            "next_entry": "kt=2 Nbb=3 from stprk3.F90 step-entry instrument",
        },
        "rows": rows,
        "failed_rows": failed,
        "ownership": ownership,
        "artifacts": artifacts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(ROOTS))
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-stage", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.case, args.oracle_root or ROOTS[args.case], plant_stage=args.plant_stage)
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
