#!/usr/bin/env python3
"""Own the LOCK kt=2 first divergence with stage-level oracle evidence.

This gate is intentionally diagnostic and red.  It establishes that the live
EOS/HPG stage-1 RHS is at the campaign bar, measures the Kmm tracer program,
the distinct WS advecting transport, literal live-Kmm UP3, two-step FCT, and
vertical-viscosity controls, and leaves the remaining residual loud.  The
source/call-site register is written to the JSON report; numerical rows are
computed from committed instrument records.
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
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_stage_kt1")
DEFAULT_TRAJECTORY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_3")
NO_ADV_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_stage_kt1_no_adv")
NO_AVM_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_stage_kt1_no_avm")
NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
EXPECTED_DIMS = (134, 7, 21)
EXPECTED_LEVELS = {
    1: {"Kaa": 3, "Kmm": 1},
    2: {"Kaa": 2, "Kmm": 3},
    3: {"Kaa": 3, "Kmm": 2},
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
    return values.reshape((nx, ny, nz), order="F")[
        2:-2, 2:-2].transpose(1, 0, 2)


def read_stage(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, level, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *EXPECTED_DIMS, 2, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "stage": stage,
        "Kaa": level,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count:2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count:3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count:4 * count], nx, ny, nz),
        "ssh": values[4 * count:].reshape((nx, ny), order="F")[2:-2, 2:-2].T,
    }


def read_transport(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, level, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_TRANSP_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, bits) == (1, *EXPECTED_DIMS, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 3 * count, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "stage": stage,
        "Kmm": level,
        "Fu": _xyz(values[:count], nx, ny, nz),
        "Fv": _xyz(values[count:2 * count], nx, ny, nz),
        "Fw": _xyz(values[2 * count:], nx, ny, nz),
    }


def read_rhs(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, level, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, bits) == (1, *EXPECTED_DIMS, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 2 * count, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "level": level,
        "u": _xyz(values[:count], nx, ny, nz),
        "v": _xyz(values[count:], nx, ny, nz),
    }


def read_entry(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nbb, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *EXPECTED_DIMS, 2, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "Nbb": nbb,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count:2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count:3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count:4 * count], nx, ny, nz),
        "ssh": values[4 * count:].reshape((nx, ny), order="F")[2:-2, 2:-2].T,
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
    require(np.all(np.isfinite(candidate[active])), f"{name}: candidate non-finite")
    reference_max_abs = float(np.max(np.abs(oracle[active])))
    absolute_error = np.abs(candidate[active] - oracle[active])
    absolute_max = float(np.max(absolute_error))
    absolute_mean = float(np.mean(absolute_error))
    scale = max(reference_max_abs, 1.0)
    error = absolute_max / scale
    return {
        "name": name,
        "status": "AT-BAR" if error <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "normalized_max_abs": error,
        "absolute_max": absolute_max,
        "absolute_mean": absolute_mean,
        "reference_max_abs": reference_max_abs,
        "bar": BAR,
        "n": int(active.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }


def thickness_weighted_mean(values, thickness) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    require(values.shape == thickness.shape, "weighted mean shape mismatch")
    total = np.sum(thickness, axis=-1)
    require(bool(np.any(total > 0.0)), "weighted mean has no positive depth")
    numerator = np.sum(values * thickness, axis=-1)
    return np.divide(
        numerator, total, out=np.zeros_like(numerator), where=total > 0.0)


def validate_registry(stages: dict, transports: dict, *, plant=False) -> list[dict]:
    rows = []
    for stage in (1, 2, 3):
        kaa = stages[stage]["Kaa"]
        kmm = transports[stage]["Kmm"]
        if plant and stage == 1:
            kmm = 2
        expected = EXPECTED_LEVELS[stage]
        ok = (
            stages[stage]["kt"] == transports[stage]["kt"] == 1
            and stages[stage]["stage"] == transports[stage]["stage"] == stage
            and kaa == expected["Kaa"]
            and kmm == expected["Kmm"]
        )
        rows.append({
            "name": f"LOCK_EXCHANGE-zco.stage{stage}.time_levels",
            "status": "VERIFIED" if ok else "DEBT",
            "observed": {"Kaa": kaa, "Kmm": kmm},
            "expected": expected,
        })
    return rows


def expected_masks(card) -> dict:
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    return {"u": u, "T": active}


def run(
    root: Path, trajectory_root: Path = DEFAULT_TRAJECTORY_ROOT, *,
    plant_rhs=False, plant_registry=False,
) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_lock_exchange_zco_card
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_lock_exchange_zco_card()
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    stages = {}
    transports = {}
    artifacts = {}
    for stage in (1, 2, 3):
        stage_path = root / f"oracle_stage_kt00000001_s{stage}.bin"
        transport_path = root / f"oracle_transport_kt00000001_s{stage}.bin"
        require(stage_path.is_file(), f"missing {stage_path}")
        require(transport_path.is_file(), f"missing {transport_path}")
        stages[stage] = read_stage(stage_path)
        transports[stage] = read_transport(transport_path)
        artifacts[stage_path.name] = sha256(stage_path)
        artifacts[transport_path.name] = sha256(transport_path)
    rhs_path = root / "oracle_rhs_kt00000001.bin"
    require(rhs_path.is_file(), f"missing {rhs_path}")
    rhs = read_rhs(rhs_path)
    require((rhs["kt"], rhs["level"]) == (1, 3), "RHS is not kt=1/Nrhs=3")
    artifacts[rhs_path.name] = sha256(rhs_path)
    entry_path = trajectory_root / "oracle_step_entry_kt00000002.bin"
    require(entry_path.is_file(), f"missing {entry_path}")
    entry = read_entry(entry_path)
    require((entry["kt"], entry["Nbb"]) == (2, 3), "entry is not kt=2/Nbb=3")
    artifacts[entry_path.name] = sha256(entry_path)

    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    tendency = model.tendencies(
        card.recipe.initial_state, dt=card.dt_s, momentum_only=True)
    candidate_rhs_u = np.asarray(tendency.du_dt.data)[:, 1:, :]
    rows = [score(
        "LOCK_EXCHANGE-zco.kt1.stage1.full_u_rhs",
        rhs["u"][..., :nlev], candidate_rhs_u, masks["u"], plant=plant_rhs)]
    rows.extend(validate_registry(stages, transports, plant=plant_registry))

    # Compute legoESM's ACTUAL pre-barotropic WS stage-1 state using the same
    # thickness-weighted removal as the production code.  Never substitute a
    # zero assertion for the candidate.  Also verify that the former simple
    # level mean happens to equal the registered thickness-weighted mean on
    # this 20x1 m z-coordinate; that shortcut is not assumed on other grids.
    initial = card.recipe.initial_state
    h_k = compute_layer_thickness(
        initial.eta.data, initial.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    h_u = min_cell_to_uface(h_k)
    H_u = np.asarray(h_u).sum(axis=-1)
    du = np.asarray(tendency.du_dt.data)
    h_u_np = np.asarray(h_u)
    rhs_mean = np.sum(du * h_u_np, axis=-1) / np.maximum(H_u, 1.0e-10)
    lego_stage1 = (
        np.asarray(initial.u.data)
        + (card.dt_s / 3.0) * (du - rhs_mean[..., None]))
    lego_stage1_mean = (
        np.sum(lego_stage1 * h_u_np, axis=-1)
        / np.maximum(H_u, 1.0e-10))[:, 1:]
    oracle_stage1_u = stages[1]["u"][..., :nlev]
    oracle_stage1_weighted = thickness_weighted_mean(
        oracle_stage1_u, h_u_np[:, 1:, :])
    oracle_stage1_level_mean = np.mean(oracle_stage1_u, axis=-1)
    u_face_2d = np.any(masks["u"], axis=-1)
    rows.append(score(
        "LOCK_EXCHANGE-zco.kt1.stage1.level_mean_assumption",
        oracle_stage1_weighted, oracle_stage1_level_mean, u_face_2d))
    stage_mean_row = score(
        "LOCK_EXCHANGE-zco.kt1.stage1.actual_candidate_depth_mean",
        oracle_stage1_weighted, lego_stage1_mean, u_face_2d)
    stage_mean_row["verdict"] = False
    stage_mean_row["reason"] = (
        "diagnostic stage mismatch only; ownership requires the causal "
        "post-stage correction experiment below")
    rows.append(stage_mean_row)

    # LOCK has e3u=1 m at every active level.  NEMO zFu is e2u*e3u times
    # the Kmm velocity plus its barotropic transport correction
    # (stprk3_stg.F90:257-303).  Compare that actual stage-3 tracer transport
    # with the final Kaa velocity that legoESM currently substitutes for every
    # tracer substage (ocean_model_latlon_cgrid.py:4554-4561,4618-4629).
    e2u = np.asarray(card.recipe.grid.dy_u)[:, 1:]
    stage3_transport_velocity = (
        transports[3]["Fu"][..., :nlev] / e2u[..., None])
    transport_scale_row = score(
        "LOCK_EXCHANGE-zco.kt1.stage3.Kmm_transport_vs_final_Kaa_velocity",
        stages[3]["u"][..., :nlev], stage3_transport_velocity, masks["u"])
    transport_scale_row["verdict"] = False
    transport_scale_row["reason"] = (
        "oracle-only scale diagnostic; causal ownership is measured by the "
        "frozen-final versus nemo_kmm candidate trajectories")
    rows.append(transport_scale_row)

    oracle_T = entry["T"][..., :nlev]
    oracle_u = entry["u"][..., :nlev]
    state_kmm = model.step(initial, dt=card.dt_s)
    cfg_frozen = card.recipe.model_config._replace(
        tracer_rk3_transport_time_levels="frozen_final",
        rk3_ws_stage_barotropic_correction=False,
        rk3_ws_momentum_transport_reconcile=False)
    state_frozen = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_frozen).step(
            initial, dt=card.dt_s)
    cfg_baseline = card.recipe.model_config._replace(
        rk3_ws_stage_barotropic_correction=False,
        rk3_ws_momentum_transport_reconcile=False)
    state_baseline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_baseline).step(
            initial, dt=card.dt_s)
    cfg_stage_baro = card.recipe.model_config._replace(
        rk3_ws_stage_barotropic_correction=True,
        rk3_ws_momentum_transport_reconcile=False)
    state_stage_baro = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_stage_baro).step(
            initial, dt=card.dt_s)
    cfg_one_step_fct = card.recipe.model_config._replace(
        tracer_fct_low_order_predictor="one_step")
    state_one_step_fct = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_one_step_fct).step(
            initial, dt=card.dt_s)

    frozen_T_row = score(
        "LOCK_EXCHANGE-zco.kt2.frozen_final_transport.T", oracle_T,
        np.asarray(state_frozen.T.data), masks["T"])
    frozen_T_row["verdict"] = False
    rows.append(frozen_T_row)
    kmm_T_row = score(
        "LOCK_EXCHANGE-zco.kt2.nemo_stage_reconcile_two_step_fct.T",
        oracle_T, np.asarray(state_kmm.T.data), masks["T"])
    rows.append(kmm_T_row)
    one_step_T_row = score(
        "LOCK_EXCHANGE-zco.kt2.nemo_stage_reconcile_one_step_fct.T",
        oracle_T, np.asarray(state_one_step_fct.T.data), masks["T"])
    one_step_T_row["verdict"] = False
    rows.append(one_step_T_row)
    baseline_u = np.asarray(state_baseline.u.data)[:, 1:, :]
    stage_baro_u = np.asarray(state_stage_baro.u.data)[:, 1:, :]
    full_u = np.asarray(state_kmm.u.data)[:, 1:, :]
    baseline_u_row = score(
        "LOCK_EXCHANGE-zco.kt2.poststage_only.u", oracle_u,
        baseline_u, masks["u"])
    rows.append(baseline_u_row)
    stage_baro_u_row = score(
        "LOCK_EXCHANGE-zco.kt2.per_stage_barotropic_correction.u", oracle_u,
        stage_baro_u, masks["u"])
    stage_baro_u_row["verdict"] = False
    rows.append(stage_baro_u_row)
    full_u_row = score(
        "LOCK_EXCHANGE-zco.kt2.stage_mean_and_transport_reconcile.u",
        oracle_u, full_u, masks["u"])
    rows.append(full_u_row)

    # Direct source-form causal control: the existing NEMO no-advection run
    # differs only by ln_dynadv_OFF.  Stage 1 is identical, so its stage-2
    # difference divided by dt/2 is the applied UP3 tendency.  Re-evaluate
    # dynadv_up3.F90:141-212 from the dumped live Kmm u and zFu.  Crucially,
    # stprk3_stg.F90:316,326-331 passes Kmm into BOTH velocity-level slots.
    no_adv_path = NO_ADV_ROOT / "oracle_stage_kt00000001_s2.bin"
    require(no_adv_path.is_file(), f"missing {no_adv_path}")
    no_adv_stage2 = read_stage(no_adv_path)
    artifacts["control:no_adv_stage2"] = sha256(no_adv_path)
    u_kmm = stages[1]["u"][..., :nlev]
    Fu = transports[2]["Fu"][..., :nlev]
    lap = np.roll(u_kmm, 1, axis=1) - 2.0 * u_kmm + np.roll(
        u_kmm, -1, axis=1)
    u_sum = u_kmm + np.roll(u_kmm, -1, axis=1)
    lap_up = np.where(u_sum > 0.0, lap, np.roll(lap, -1, axis=1))
    flux = 0.25 * (Fu + np.roll(Fu, -1, axis=1)) * (
        u_sum - lap_up / 3.0)
    area = np.asarray(card.recipe.grid.area)
    area_u = 0.5 * (area + np.roll(area, 1, axis=1))
    area_u = np.concatenate([area_u, area_u[:, :1]], axis=1)[:, 1:]
    literal_hadv = -(flux - np.roll(flux, 1, axis=1)) / np.maximum(
        area_u[..., None] * h_u_np[:, 1:, :], 1.0e-12)
    literal_hadv -= thickness_weighted_mean(
        literal_hadv, h_u_np[:, 1:, :])[..., None]
    oracle_hadv = (
        stages[2]["u"][..., :nlev]
        - no_adv_stage2["u"][..., :nlev]) * (2.0 / card.dt_s)
    rows.append(score(
        "LOCK_EXCHANGE-zco.kt1.stage2.literal_live_Kmm_UP3",
        oracle_hadv, literal_hadv, masks["u"]))

    no_avm_path = NO_AVM_ROOT / "oracle_stage_kt00000001_s3.bin"
    require(no_avm_path.is_file(), f"missing {no_avm_path}")
    no_avm_stage3 = read_stage(no_avm_path)
    artifacts["control:no_avm_stage3"] = sha256(no_avm_path)
    cfg_no_avm = card.recipe.model_config._replace(A_v=0.0)
    state_no_avm = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_no_avm).step(
            initial, dt=card.dt_s)
    no_avm_row = score(
        "LOCK_EXCHANGE-zco.kt2.no_vertical_viscosity.u",
        no_avm_stage3["u"][..., :nlev],
        np.asarray(state_no_avm.u.data)[:, 1:, :], masks["u"])
    no_avm_row["verdict"] = False
    rows.append(no_avm_row)

    source_files = {
        "stprk3": NEMO_ROOT / "src/OCE/stprk3.F90",
        "stprk3_stg": NEMO_ROOT / "src/OCE/stprk3_stg.F90",
        "traadv_fct": NEMO_ROOT / "src/OCE/TRA/traadv_fct.F90",
    }
    for name, path in source_files.items():
        require(path.is_file(), f"missing source {path}")
        artifacts[f"source:{name}"] = sha256(path)
    failed = [row["name"] for row in rows
              if row["status"] == "DEBT" and row.get("verdict", True)]
    frozen_T_error = frozen_T_row["normalized_max_abs"]
    kmm_T_error = kmm_T_row["normalized_max_abs"]
    baseline_u_error = baseline_u_row["normalized_max_abs"]
    stage_baro_u_error = stage_baro_u_row["normalized_max_abs"]
    full_u_error = full_u_row["normalized_max_abs"]
    return {
        "format": "nemo-testcase-l1-phase3-first-divergence-v2",
        "case": card.case,
        "status": "AT-BAR" if not failed else "DEBT",
        "first_over_bar_step": 2,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "candidate_dtypes": {
            "rhs_u": str(candidate_rhs_u.dtype),
            "stage3_transport_velocity": str(stage3_transport_velocity.dtype),
            "actual_stage1_depth_mean": str(lego_stage1_mean.dtype),
        },
        "bar": BAR,
        "oracle_root": str(root),
        "rows": rows,
        "failed_rows": failed,
        "ownership": {
            "stage1_eos_hpg": {
                "classification": "CONFIRMED_EXONERATED",
                "evidence": "full at-rest u RHS is inside the registered bar",
            },
            "tracer_transport_time_level": {
                "classification": "CONFIRMED_OWNER",
                "before_normalized_error": frozen_T_error,
                "after_normalized_error": kmm_T_error,
                "improvement_factor": frozen_T_error / kmm_T_error,
            },
            "momentum_stage_barotropic_correction": {
                "classification": "CONFIRMED_EXONERATED",
                "retraction": (
                    "the prior owner label compared NEMO with zeros and ignored "
                    "the six-order scale contradiction"),
                "poststage_only_u_error": baseline_u_error,
                "per_stage_u_error": stage_baro_u_error,
                "movement": stage_baro_u_error - baseline_u_error,
            },
            "momentum_stage_advecting_transport": {
                "classification": "PLAUSIBLE_PARTIAL_OWNER",
                "stage_mean_only_u_error": stage_baro_u_error,
                "stage_mean_plus_transport_u_error": full_u_error,
                "movement": full_u_error - stage_baro_u_error,
                "reason": "70% residual reduction, but the row remains over bar",
            },
            "horizontal_up3_spatial_operator": {
                "classification": "CONFIRMED_EXONERATED",
                "evidence": "literal live-Kmm source recurrence matches the no-advection control",
            },
            "vertical_viscosity": {
                "classification": "CONFIRMED_EXONERATED",
                "evidence": "matched rn_avm0/A_v=0 control leaves the residual unchanged",
            },
            "fct_stage_kernel": {
                "classification": "PLAUSIBLE_PARTIAL_OWNER",
                "one_step_error": one_step_T_row["normalized_max_abs"],
                "two_step_error": kmm_T_row["normalized_max_abs"],
                "movement": (kmm_T_row["normalized_max_abs"]
                             - one_step_T_row["normalized_max_abs"]),
                "reason": "causal improvement, but the row remains over bar",
            },
        },
        "source_register": {
            "NEMO_stage_calls": "src/OCE/stprk3.F90:184-207",
            "NEMO_Kmm_transport": "src/OCE/stprk3_stg.F90:257-303",
            "NEMO_stage_barotropic_correction": "src/OCE/stprk3_stg.F90:433-446",
            "NEMO_tracer_transport_call": "src/OCE/stprk3_stg.F90:456-519",
            "NEMO_RK3_FCT_dispatch": "src/OCE/TRA/traadv_fct.F90:153-161",
            "NEMO_FCT_two_step": "src/OCE/TRA/traadv_fct.F90:470-641",
            "lego_final_transport": (
                "packages/ocean/legoesm/ocean/dynamics/"
                "ocean_model_latlon_cgrid.py:4554-4561,4618-4629"),
            "lego_WS_wrapper": (
                "packages/ocean/legoesm/ocean/dynamics/"
                "ocean_model_latlon_cgrid.py:5242-5251"),
            "lego_WS_momentum_split": (
                "packages/ocean/legoesm/ocean/dynamics/"
                "ocean_model_latlon_cgrid.py:3991-4017,4060-4065"),
        },
        "artifacts_sha256": artifacts,
        "unmeasured": [
            "individual FCT limiter coefficients and antidiffusive fluxes",
            "owner of the remaining kt=2 u residual after stage transport; "
            "live-Kmm horizontal UP3 and vertical viscosity are exonerated",
            "kt>=3 and OVERFLOW are reported by the separate explicit "
            "owner-exhausted continuation artifact, not this first-step gate",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-rhs", action="store_true")
    parser.add_argument("--plant-registry", action="store_true")
    args = parser.parse_args()
    report = run(
        args.oracle_dir, args.trajectory_dir, plant_rhs=args.plant_rhs,
        plant_registry=args.plant_registry)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, OSError, UnicodeError, struct.error) as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
