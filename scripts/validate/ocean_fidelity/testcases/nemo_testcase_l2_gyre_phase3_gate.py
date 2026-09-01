#!/usr/bin/env python3
"""Fail-closed GYRE whole-step WS-RK3 trajectory certification gate.

This is lane 2's composition of lane 1 machinery: the binary record format,
the central time-level registry, fp64 pointwise scoring, scaling-first private
one-variable arms, first-over-bar retention, and the shared log-log growth
instrument.  A scientific DEBT is an expected exit status, never a harness
failure and never permission to assign an unsupported owner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

BAR = 1.0e-15
CASE = "GYRE-zco"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10")
DIMS = (36, 26, 31)
LEVELS = {1: {"Kaa": 3, "Kmm": 1}, 2: {"Kaa": 2, "Kmm": 3}, 3: {"Kaa": 3, "Kmm": 2}}


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


def _registered(path: Path, expected: str) -> str:
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    observed = time_level_for_dump(path.name)
    require(observed == expected, f"{path}: registry {observed!r}, expected {expected!r}")
    return observed


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_entry(path: Path) -> dict:
    level = _registered(path, "before")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nbb, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require((version, nx, ny, nz, ntr, bits) == (1, *DIMS, 2, 64), f"{path}: bad header")
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")

    def xy(block):
        return block.reshape((nx, ny), order="F")[2:-2, 2:-2].T

    return {
        "kt": kt,
        "Nbb": nbb,
        "registry_level": level,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count : 2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count : 4 * count], nx, ny, nz),
        "ssh": xy(values[4 * count :]),
    }


def read_stage(path: Path, expected_stage: int) -> dict:
    level = _registered(path, "after")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kaa, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic")
    require((version, nx, ny, nz, ntr, bits) == (1, *DIMS, 2, 64), f"{path}: bad header")
    require(
        (kt, stage, kaa) == (1, expected_stage, LEVELS[expected_stage]["Kaa"]),
        f"{path}: wrong Kaa",
    )
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    return {
        "stage": stage,
        "Kaa": kaa,
        "registry_level": level,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count : 2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count : 4 * count], nx, ny, nz),
    }


def read_transport(path: Path, expected_stage: int) -> dict:
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_TRANSP_1", f"{path}: bad magic")
    require((version, nx, ny, nz, bits) == (1, *DIMS, 64), f"{path}: bad header")
    require(
        (kt, stage, kmm) == (1, expected_stage, LEVELS[expected_stage]["Kmm"]),
        f"{path}: wrong Kmm",
    )
    require(values.size == 3 * nx * ny * nz, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {"stage": stage, "Kmm": kmm, "registry_level": level}


def read_rhs(path: Path) -> dict:
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nrhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic")
    require((version, kt, nrhs, nx, ny, nz, bits) == (1, 1, 3, *DIMS, 64), f"{path}: bad header")
    require(values.size == 2 * nx * ny * nz, f"{path}: bad payload")
    return {"Nrhs": nrhs, "registry_level": level}


def read_bt(path: Path, expected_kt: int) -> dict:
    level = _registered(path, "after")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kaa, nx, ny, bits = header
    require(magic == "NEMO_L1_BTFRM_1", f"{path}: bad magic")
    expected_kaa = 3 if expected_kt % 2 else 1
    require(
        (version, kt, kaa, nx, ny, bits) == (1, expected_kt, expected_kaa, DIMS[0], DIMS[1], 64),
        f"{path}: bad header",
    )
    require(values.size == 4 * nx * ny, f"{path}: bad payload")
    return {"kt": kt, "Kaa": kaa, "registry_level": level}


def expected_masks(card) -> dict:
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    v = active & np.roll(active, -1, axis=0)
    u[:, -1] = False
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}


def lego_fields(state) -> dict:
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data)[:, 1:, :],
        "v": np.asarray(state.v.data)[1:, :, :],
        "ssh": np.asarray(state.eta.data),
    }


def score(name: str, oracle, candidate, mask, *, plant=False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is {candidate.dtype}")
    require(bool(active.any()), f"{name}: empty mask")
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: non-finite candidate")
    absolute = float(np.max(np.abs(candidate[active] - oracle[active])))
    reference = float(np.max(np.abs(oracle[active])))
    normalized = absolute / max(reference, 1.0)
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "absolute_max": absolute,
        "reference_max_abs": reference,
        "normalized_max_abs": normalized,
        "bar": BAR,
        "n": int(active.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }


def _mark_kt1_uninformative(row: dict, field: str, kt: int) -> dict:
    if kt == 1 and field in {"u", "v", "ssh"} and row["status"] == "AT-BAR":
        row["status"] = "UNINFORMATIVE"
        row["reason"] = {
            "u": (
                "at-rest initial U is identically zero; exactness cannot "
                "exercise momentum evolution"
            ),
            "v": (
                "at-rest initial V is identically zero; exactness cannot "
                "exercise momentum evolution"
            ),
            "ssh": (
                "at-rest initial SSH is identically zero; exactness cannot "
                "exercise barotropic evolution"
            ),
        }[field]
    return row


def _surface_forcings(card, state, kt: int):
    import jax.numpy as jnp
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_boundary_condition
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    sbc = gyre_surface_boundary_condition(card, kt * card.dt_s)
    sst = state.T.data[..., 0]
    # usrdef_sbc.F90:109-120,138-145: qns+qsr is the Haney term plus EMP
    # heat content, evaluated once from the entering Kbb/Nbb SST.
    q_total = -40.0 * (sst - sbc.t_star_c) - sbc.emp_kg_m2_s * sst * NEMO_CONSTANTS_CONFIG.c_sw
    surface = OceanSurfaceForcing(
        sw_down=sbc.qsr_w_m2,
        q_net=q_total,
        # OceanSurfaceForcing is atmospheric convention; the production core
        # applies the ocean reaction. usrdef_sbc utau/vtau are already ocean.
        tau_x=-sbc.utau_pa,
        tau_y=-sbc.vtau_pa,
    )
    zeros = jnp.zeros_like(sbc.emp_kg_m2_s)
    freshwater = FreshwaterForcing(
        precip=zeros,
        evap=sbc.emp_kg_m2_s,
        runoff=zeros,
        ice_fw=zeros,
        restoring=zeros,
    )
    return freshwater, surface


def _trajectory_growth(steps: list[dict]) -> dict:
    path = Path(__file__).with_name("nemo_testcase_phase3_trajectory_gate.py")
    spec = importlib.util.spec_from_file_location("lane1_phase3_trajectory", path)
    require(spec is not None and spec.loader is not None, "cannot load lane-1 growth instrument")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.characterize_growth(steps)


def validate_one_variable_arms(arms: dict, *, plant=False) -> list[dict]:
    rows = []
    for name, manifest in arms.items():
        changed = list(manifest["changed_operands"])
        if plant and name == "omit_stage_barotropic_correction":
            changed.append("second_illegal_operand")
        ok = len(changed) == 1
        rows.append(
            {
                "name": f"arm_manifest.{name}",
                "status": "VERIFIED" if ok else "DEBT",
                "changed_operands": changed,
            }
        )
    return rows


def run(
    root: Path,
    *,
    max_step=10,
    plant_state=False,
    plant_registry=False,
    plant_arm=False,
) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(1 <= max_step <= 10, "max_step must be in 1..10")

    card = build_nemo_testcase_card(CASE)
    # NEMO QCO represents E-P through changing volume, with sfx=0
    # (usrdef_sbc.F90:138-145).  The fix_eta_drift projection is legoESM's
    # required source-inclusive realization of that real-freshwater contract.
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels

    artifacts = {}
    stages = {}
    transports = {}
    for stage in (1, 2, 3):
        stage_path = root / f"oracle_stage_kt00000001_s{stage}.bin"
        transport_path = root / f"oracle_transport_kt00000001_s{stage}.bin"
        require(stage_path.is_file() and transport_path.is_file(), f"missing stage {stage} records")
        stages[stage] = read_stage(stage_path, stage)
        transports[stage] = read_transport(transport_path, stage)
        artifacts[stage_path.name] = sha256(stage_path)
        artifacts[transport_path.name] = sha256(transport_path)
    rhs_path = root / "oracle_rhs_kt00000001.bin"
    rhs = read_rhs(rhs_path)
    artifacts[rhs_path.name] = sha256(rhs_path)
    bt = {}
    for kt in range(1, max_step + 1):
        path = root / f"oracle_bt_frames_kt{kt:08d}.bin"
        bt[kt] = read_bt(path, kt)
        artifacts[path.name] = sha256(path)

    registry_rows = []
    for stage in (1, 2, 3):
        observed_kmm = transports[stage]["Kmm"]
        if plant_registry and stage == 1:
            observed_kmm = 2
        ok = stages[stage]["Kaa"] == LEVELS[stage]["Kaa"] and observed_kmm == LEVELS[stage]["Kmm"]
        registry_rows.append(
            {
                "name": f"GYRE.kt1.stage{stage}.Kaa_Kmm_registry",
                "status": "VERIFIED" if ok else "DEBT",
                "observed": {"Kaa": stages[stage]["Kaa"], "Kmm": observed_kmm},
                "expected": LEVELS[stage],
            }
        )

    state = card.recipe.initial_state
    steps = []
    first_over_bar = None
    exact_prefix = True
    faithful_kt2 = None
    for kt in range(1, max_step + 1):
        path = root / f"oracle_step_entry_kt{kt:08d}.bin"
        oracle = read_entry(path)
        expected_nbb = 1 if kt % 2 else 3
        require(
            (oracle["kt"], oracle["Nbb"]) == (kt, expected_nbb),
            f"{path}: wrong Nbb",
        )
        artifacts[path.name] = sha256(path)
        candidate = lego_fields(state)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = oracle[field] if field == "ssh" else oracle[field][..., :nlev]
            row = score(
                f"{CASE}.kt{kt}.before.{field}",
                reference,
                candidate[field],
                masks[field],
                plant=plant_state and kt == 1 and field == "T",
            )
            rows.append(_mark_kt1_uninformative(row, field, kt))
        over = [row["name"].rsplit(".", 1)[-1] for row in rows if row["status"] == "DEBT"]
        exact_here = all(row["exact"] for row in rows)
        steps.append(
            {
                "kt": kt,
                "Nbb": oracle["Nbb"],
                "exact_prefix_entering": exact_prefix,
                "exact_at_step": exact_here,
                "rows": rows,
            }
        )
        exact_prefix = exact_prefix and exact_here
        if over and first_over_bar is None:
            first_over_bar = {"kt": kt, "fields": over}
        if kt < max_step:
            freshwater, surface = _surface_forcings(card, state, kt)
            state = model.step(state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
            if kt == 1:
                faithful_kt2 = state

    require(faithful_kt2 is not None or max_step == 1, "kt2 candidate was not produced")

    stage_rows = []
    if max_step >= 2:
        freshwater0, surface0 = _surface_forcings(card, card.recipe.initial_state, 1)
        for stage in (1, 2, 3):
            if stage == 3:
                stage_state = faithful_kt2
            else:
                stage_state = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    cfg,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_momentum_stage=stage),
                ).step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    freshwater=freshwater0,
                    surface_forcing=surface0,
                )
            fields = lego_fields(stage_state)
            for velocity in ("u", "v"):
                row = score(
                    f"{CASE}.kt1.stage{stage}.{velocity}",
                    stages[stage][velocity][..., :nlev],
                    fields[velocity],
                    masks[velocity],
                )
                row["frame"] = "instantaneous_prognostic_Kaa"
                stage_rows.append(row)
            for tracer in ("T", "S"):
                stage_rows.append(
                    {
                        "name": f"{CASE}.kt1.stage{stage}.{tracer}",
                        "status": "UNMEASURED",
                        "reason": (
                            "NEMO Kaa tracer artifact is present and registered, "
                            "but production legoESM exposes only momentum stage "
                            "states; no private tracer-stage surrogate is scored"
                        ),
                    }
                )

    arm_manifest = {
        "omit_stage_barotropic_correction": {
            "changed_operands": ["stage_barotropic_correction"],
            "hooks": _NEMOWSRK3TestHooks(stage_barotropic_correction=False),
        },
        "omit_momentum_transport_reconcile": {
            "changed_operands": ["momentum_transport_reconcile"],
            "hooks": _NEMOWSRK3TestHooks(momentum_transport_reconcile=False),
        },
        "omit_surface_boundary_forcing": {
            "changed_operands": ["surface_forcing"],
            "hooks": _NEMOWSRK3TestHooks(),
        },
        "omit_freshwater_forcing": {
            "changed_operands": ["freshwater"],
            "hooks": _NEMOWSRK3TestHooks(),
        },
    }
    arm_rows = validate_one_variable_arms(arm_manifest, plant=plant_arm)
    arms = {}
    if max_step >= 2:
        oracle2 = read_entry(root / "oracle_step_entry_kt00000002.bin")
        faithful_fields = lego_fields(faithful_kt2)
        for name, manifest in arm_manifest.items():
            arm_model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, cfg, _nemo_ws_test_hooks=manifest["hooks"]
            )
            if name == "omit_surface_boundary_forcing":
                control = arm_model.step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    freshwater=freshwater0,
                )
            elif name == "omit_freshwater_forcing":
                control = arm_model.step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    surface_forcing=surface0,
                )
            else:
                control = arm_model.step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    freshwater=freshwater0,
                    surface_forcing=surface0,
                )
            control_fields = lego_fields(control)
            field_rows = {}
            movements = []
            residuals = []
            improves = []
            for field in ("T", "S", "u", "v", "ssh"):
                reference = oracle2[field] if field == "ssh" else oracle2[field][..., :nlev]
                row = score(
                    f"{CASE}.kt2.arm.{name}.{field}",
                    reference,
                    control_fields[field],
                    masks[field],
                )
                field_rows[field] = row
                active = masks[field]
                movement_abs = float(
                    np.max(np.abs(control_fields[field][active] - faithful_fields[field][active]))
                )
                scale = max(float(np.max(np.abs(reference[active]))), 1.0)
                movements.append(movement_abs / scale)
                faithful_abs = (
                    float(np.max(np.abs(faithful_fields[field][active] - reference[active])))
                    / scale
                )
                residuals.append(faithful_abs)
                improves.append(row["normalized_max_abs"] < faithful_abs)
            residual = max(residuals)
            movement = max(movements)
            clears = all(row["status"] == "AT-BAR" for row in field_rows.values())
            if clears:
                label = "CONFIRMED_OWNER"
            elif movement < 0.1 * residual or not any(improves):
                label = "REFUTED_AS_PRIMARY_OWNER"
            else:
                label = "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
            arms[name] = {
                "classification": "DIAGNOSTIC_ONE_VARIABLE_ARM",
                "changed_operand": manifest["changed_operands"][0],
                "scaling_check_before_owner_label": True,
                "faithful_worst_normalized_residual": residual,
                "arm_worst_normalized_movement": movement,
                "movement_over_faithful_residual": (
                    movement / residual if residual else float("inf")
                ),
                "owner_label": label,
                "field_rows": field_rows,
            }

    failed_controls = [row["name"] for row in registry_rows + arm_rows if row["status"] == "DEBT"]
    require(not failed_controls, f"planted/control failure: {failed_controls}")
    status = "AT-BAR" if first_over_bar is None else "DEBT"
    return {
        "format": "nemo-testcase-l2-gyre-phase3-v1",
        "case": CASE,
        "status": status,
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "oracle_root": str(root),
        "max_step": max_step,
        "continue_after_first": True,
        "first_over_bar": first_over_bar,
        "selectors": {
            "eos": cfg.eos,
            "eos_depth": cfg.eos_depth,
            "momentum_time_integrator": cfg.momentum_time_integrator,
            "tracer_time_integrator": cfg.tracer_time_integrator,
            "rk3_ws_scheme_identity": "nemo_kmm+two_step_fct+stage_correction+transport_reconcile",
            "pgf": cfg.pgf_scheme,
            "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
            "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
            "freshwater_closure": cfg.freshwater_closure,
        },
        "full_stage_program": {
            "oracle_stage_headers": {
                str(stage): {
                    "Kaa": stages[stage]["Kaa"],
                    "Kmm": transports[stage]["Kmm"],
                }
                for stage in (1, 2, 3)
            },
            "rhs": rhs,
            "barotropic_frames": bt,
            "momentum_rows": stage_rows,
            "tracer_stage_numerics": "UNMEASURED",
            "barotropic_frame_numerics": "UNMEASURED",
        },
        "registry_rows": registry_rows,
        "arm_manifest_rows": arm_rows,
        "one_variable_arms": arms,
        "owner_verdict": "UNMEASURED_AFTER_REGISTERED_ARMS",
        "growth_characterization": _trajectory_growth(steps),
        "steps": steps,
        "artifacts": artifacts,
        "unmeasured": [
            "numerical T/S agreement at internal Kaa stages",
            "numerical barotropic primary/advecting-frame agreement",
            "a two-sided source-isolated owner for the first over-bar whole-step row",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=ROOT)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-state", action="store_true")
    parser.add_argument("--plant-registry", action="store_true")
    parser.add_argument("--plant-arm", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.oracle_root,
            max_step=args.max_step,
            plant_state=args.plant_state,
            plant_registry=args.plant_registry,
            plant_arm=args.plant_arm,
        )
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
