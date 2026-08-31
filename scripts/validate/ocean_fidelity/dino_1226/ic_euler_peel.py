"""Committed CPU peel of DINO's analytic IC and first Euler bootstrap.

The frozen protocol is ``PREREG_dino_ic_euler_peel.md``.  This probe reports
mechanical rows and the first over-bar row; scientific interpretation lives in
the result document, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

import hpg_tendency_compare as hpg
import kamm_twin_90d as twin
import standalone_20y as standalone
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing,
    dino_S_profile_1d,
    dino_T_profile_1d,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)


SCHEMA = "dino_ic_euler_peel_v1"
BAR = 1.0e-15
RUNTIME_SHAPE = (203, 56)
RUNTIME_HALO = 2
STANDALONE_CORE = 2
SESSION_ID = "01a053d4-8e9f-7212-bbdb-19ba2d64e140"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {value.shape} != {expected}")


def _ordered_float64(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where((bits & sign) != 0, ~bits, bits ^ sign)


def diff_row(name: str, actual: np.ndarray, expected: np.ndarray,
             active: np.ndarray, *, bar: float = BAR) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require_shape(f"{name}.actual", actual, expected.shape)
    require_shape(f"{name}.active", active, expected.shape)
    if not np.isfinite(actual[active]).all() or not np.isfinite(expected[active]).all():
        raise ValueError(f"{name} contains non-finite active values")
    delta = np.abs(actual - expected)
    selected = delta[active]
    mismatch = selected > bar
    exact_mismatch = selected != 0.0
    indices = np.argwhere(active & (delta > bar))
    first = None
    if indices.size:
        index = tuple(int(i) for i in indices[0])
        first = {
            "index": list(index),
            "actual": float(actual[index]),
            "expected": float(expected[index]),
            "abs": float(delta[index]),
        }
    ordered_actual = _ordered_float64(actual[active])
    ordered_expected = _ordered_float64(expected[active])
    ulps = np.maximum(ordered_actual, ordered_expected) - np.minimum(
        ordered_actual, ordered_expected)
    return {
        "name": name,
        "bar": float(bar),
        "n": int(selected.size),
        "max_abs": float(selected.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(selected * selected))) if selected.size else 0.0,
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "exact_mismatch_count": int(np.count_nonzero(exact_mismatch)),
        "max_ulp": int(ulps.max(initial=0)),
        "first_over_bar": first,
        "status": "PASS" if not np.any(mismatch) else "OVER_BAR",
    }


def _broadcast_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.asarray(mask, dtype=bool), shape)


def _mesh_core(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        return array[STANDALONE_CORE:-STANDALONE_CORE,
                     STANDALONE_CORE:-STANDALONE_CORE, :-1]
    if array.ndim == 2:
        return array[STANDALONE_CORE:-STANDALONE_CORE,
                     STANDALONE_CORE:-STANDALONE_CORE]
    raise ValueError(f"unsupported mesh rank {array.ndim}")


def _runtime_dump(path: Path) -> np.ndarray:
    array = hpg.read_dump(
        str(path), 35, RUNTIME_SHAPE[0], RUNTIME_SHAPE[1])
    return array[RUNTIME_HALO + STANDALONE_CORE:
                 -RUNTIME_HALO - STANDALONE_CORE,
                 RUNTIME_HALO + STANDALONE_CORE:
                 -RUNTIME_HALO - STANDALONE_CORE, :]


def _runtime_ssh(path: Path) -> np.ndarray:
    array = np.fromfile(path, dtype=np.float64)
    expected = RUNTIME_SHAPE[0] * RUNTIME_SHAPE[1]
    if array.size != expected:
        raise ValueError(f"{path}: {array.size} values != {expected}")
    array = array.reshape(RUNTIME_SHAPE)
    edge = RUNTIME_HALO + STANDALONE_CORE
    return array[edge:-edge, edge:-edge]


def _initial_rows(grid, geometry, z_coord, state, cfg) -> tuple[
        list[dict[str, Any]], np.ndarray, np.ndarray]:
    if grid.gdept_0 is None:
        raise ValueError("mesh_mask.nc lacks gdept_0")
    nemo_t, nemo_s = hpg.nemo_istate_case4(
        grid.gdept_0, grid.gphit, grid.tmask)
    nemo_t = _mesh_core(nemo_t)
    nemo_s = _mesh_core(nemo_s)
    mask = _mesh_core(grid.tmask).astype(bool)
    lego_mask = (np.asarray(z_coord.is_active, dtype=bool)
                 & _broadcast_mask(
                     np.asarray(state.land_mask.data) > 0.5, mask.shape))
    lego_depth = np.asarray(z_coord.nemo_gdept_0)
    lego_lat = np.asarray(geometry.native_lat_T_deg)
    rows = [
        diff_row("input_wet_mask", lego_mask.astype(np.float64),
                 mask.astype(np.float64), np.ones(mask.shape, dtype=bool), bar=0.0),
        diff_row("input_latitude_deg", lego_lat, _mesh_core(grid.gphit),
                 np.ones(lego_lat.shape, dtype=bool), bar=0.0),
        diff_row("input_t_depth_m", lego_depth, _mesh_core(grid.gdept_0),
                 mask, bar=0.0),
    ]

    nemo_depth = _mesh_core(grid.gdept_0)
    source_t_profile, source_s_profile = hpg.nemo_istate_profiles_1d(nemo_depth)
    lego_t_profile = np.asarray(dino_T_profile_1d(nemo_depth))
    lego_s_profile = np.asarray(dino_S_profile_1d(nemo_depth))
    rows.extend([
        diff_row("common_depth_T_profile", lego_t_profile, source_t_profile, mask),
        diff_row("common_depth_S_profile", lego_s_profile, source_s_profile, mask),
        diff_row("resolved_T", np.asarray(state.T.data), nemo_t, mask),
        diff_row("resolved_S", np.asarray(state.S.data), nemo_s, mask),
    ])

    full_source_t, full_source_s = hpg.nemo_istate_profiles_1d(grid.gdept_0)
    phi_max = float(np.max(grid.gphit))
    t_bot = float(np.min(full_source_t + 100.0 * (1.0 - grid.tmask)))
    s_bot = float(np.min(full_source_s + 100.0 * (1.0 - grid.tmask)))
    source_lego_depth, source_lego_depth_s = hpg.nemo_istate_case4(
        lego_depth, lego_lat, lego_mask.astype(np.float64),
        phi_max_deg=phi_max, t_bot=t_bot, s_bot=s_bot)
    source_lego_anchors, source_lego_anchors_s = hpg.nemo_istate_case4(
        lego_depth, lego_lat, lego_mask.astype(np.float64))
    legacy_t_prof, legacy_s_prof = hpg.nemo_istate_profiles_1d(lego_depth)
    source_legacy, source_legacy_s = hpg.nemo_istate_case4(
        lego_depth, lego_lat, lego_mask.astype(np.float64),
        phi_max_deg=float(cfg.lat_max_deg),
        t_bot=float(legacy_t_prof[..., -1].min()),
        s_bot=float(legacy_s_prof[..., -1].min()))
    rows.extend([
        diff_row("substitute_nemo_depth_T", source_lego_depth, nemo_t, mask),
        diff_row("substitute_nemo_depth_S", source_lego_depth_s, nemo_s, mask),
        diff_row("substitute_nemo_anchors_T", source_lego_anchors,
                 np.asarray(state.T.data), mask),
        diff_row("substitute_nemo_anchors_S", source_lego_anchors_s,
                 np.asarray(state.S.data), mask),
        diff_row("source_order_legacy_T", source_legacy,
                 np.asarray(state.T.data), mask),
        diff_row("source_order_legacy_S", source_legacy_s,
                 np.asarray(state.S.data), mask),
    ])
    return rows, nemo_t, nemo_s


def _step_rows(run_kt2: Path, grid, cfg, z_coord, state, model,
               forcing, step_forcing, nemo_t: np.ndarray,
               nemo_s: np.ndarray) -> list[dict[str, Any]]:
    common = state._replace(
        T=state.T.replace(data=nemo_t),
        S=state.S.replace(data=nemo_s))
    unchanged = [name for name in common._fields
                 if name not in {"T", "S"}
                 and getattr(common, name) is not getattr(state, name)]
    if unchanged:
        raise ValueError("common-IC replacement changed non-T/S fields: "
                         + ", ".join(unchanged))
    dyn = jax.jit(lambda st, rate: model.step(
        st, standalone.DT_SECONDS, surface_forcing=step_forcing,
        external_tracer_rate=rate))
    common, rate = apply_dino_lat_lon_surface_forcing(
        common, forcing, z_coord, cfg, standalone.DT_SECONDS,
        t_seconds=standalone.DT_SECONDS, return_rate=True)
    step1 = dyn(common, rate)
    mask_t = _mesh_core(grid.tmask).astype(bool)
    mask_u = _mesh_core(grid.umask).astype(bool)
    mask_v = _mesh_core(grid.vmask).astype(bool)
    rows = [
        diff_row("euler_T_after_trazdf", np.asarray(step1.T.data),
                 _runtime_dump(run_kt2 / "stp_dump_21_trazdf_tem.bin"), mask_t),
        diff_row("euler_S_after_trazdf", np.asarray(step1.S.data),
                 _runtime_dump(run_kt2 / "stp_dump_21_trazdf_sal.bin"), mask_t),
        diff_row("euler_U_after_corrector", np.asarray(step1.u.data)[:, 1:, :],
                 _runtime_dump(run_kt2 / "baro_dump_u_after.bin"), mask_u),
        diff_row("euler_V_after_corrector", np.asarray(step1.v.data)[1:, :, :],
                 _runtime_dump(run_kt2 / "baro_dump_v_after.bin"), mask_v),
        diff_row("euler_SSH_after_split", np.asarray(step1.eta.data),
                 _runtime_ssh(run_kt2 / "spg_dump_pssh_final.bin"),
                 mask_t[..., 0]),
    ]
    step1, rate = apply_dino_lat_lon_surface_forcing(
        step1, forcing, z_coord, cfg, standalone.DT_SECONDS,
        t_seconds=2 * standalone.DT_SECONDS, return_rate=True)
    step2 = dyn(step1, rate)
    before = read_nemo_restart_before(
        str(run_kt2 / "DINO_00000002_restart.nc"), nn_hls=0)
    rows.extend([
        diff_row("filtered_step1_T_carry", np.asarray(step2.T_before.data),
                 _mesh_core(before.T), mask_t),
        diff_row("filtered_step1_S_carry", np.asarray(step2.S_before.data),
                 _mesh_core(before.S), mask_t),
        diff_row("filtered_step1_U_carry",
                 np.asarray(step2.u_before.data)[:, 1:, :],
                 _mesh_core(before.u), mask_u),
        diff_row("filtered_step1_V_carry",
                 np.asarray(step2.v_before.data)[1:, :, :],
                 _mesh_core(before.v), mask_v),
        diff_row("filtered_step1_SSH_carry", np.asarray(step2.eta_before.data),
                 _mesh_core(before.ssh), mask_t[..., 0]),
    ])
    return rows


def self_test() -> dict[str, str]:
    active = np.ones((2, 2), dtype=bool)
    base = np.arange(4, dtype=np.float64).reshape(2, 2)
    planted = base.copy()
    planted[1, 0] += 2.0 * BAR
    row = diff_row("planted", planted, base, active)
    if row["status"] != "OVER_BAR" or row["first_over_bar"]["index"] != [1, 0]:
        raise RuntimeError("planted field mismatch did not fire")
    try:
        diff_row("wrong_core", base[:, :-1], base, active)
    except ValueError:
        shape = "FIRED"
    else:
        raise RuntimeError("planted wrong core did not fire")
    return {"field_mismatch_plant": "FIRED", "wrong_core_plant": shape}


def run(args: argparse.Namespace) -> int:
    if os.environ.get("JAX_PLATFORM_NAME") != "cpu":
        raise SystemExit("CPU-only peel requires explicit JAX_PLATFORM_NAME=cpu")
    producer, dirty = standalone.git_provenance()
    if dirty:
        raise SystemExit("REFUSING dirty tracked producer")
    run_kt2 = args.run_kt2.resolve()
    run_y1 = args.run_traj_y1.resolve()
    inputs = [
        args.lego_kt2.resolve(), args.lego_d10.resolve(),
        run_kt2 / "mesh_mask.nc", run_kt2 / "DINO_00000002_restart.nc",
        run_kt2 / "namelist_cfg", run_y1 / "namelist_cfg",
        run_y1 / "DINO_00011520_restart.nc",
        run_y1 / "DINO_1y_00010701_00011230_grid_T_0000.nc",
    ] + [run_kt2 / name for name in (
        "stp_dump_21_trazdf_tem.bin", "stp_dump_21_trazdf_sal.bin",
        "baro_dump_u_after.bin", "baro_dump_v_after.bin",
        "spg_dump_pssh_final.bin")]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing peel inputs: " + ", ".join(missing))

    controls = self_test()
    set_policy(PrecisionPolicy.fp64())
    grid = read_nemo_mesh_mask(str(run_kt2 / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(
        str(run_kt2 / "DINO_00000002_restart.nc"), nn_hls=0)
    bridge = twin.bridge_nemo_to_legoesm_topo(
        grid, restart, periodic_i=True, full_step=True, e3t_mode="both",
        vface_zonal_metric_evaluation="nemo_vpoint",
        coriolis_placement="cell_average")
    twin.verify_day0_matches_restart(
        bridge.state, restart, bridge.land_mask, min_restart_speed=0.0)
    controls["actual_kt2_restart_equality_and_nonzero"] = "PASS"

    (cfg, geometry, z_coord, state, _model_cfg, model, forcing,
     step_forcing, _perturbation) = standalone.build_standalone(0)
    ic_rows, nemo_t, nemo_s = _initial_rows(
        grid, geometry, z_coord, state, cfg)
    step_rows = _step_rows(
        run_kt2, grid, cfg, z_coord, state, model, forcing, step_forcing,
        nemo_t, nemo_s)
    ordered = ic_rows + step_rows
    first_over_bar = next(
        (row["name"] for row in ordered if row["status"] == "OVER_BAR"), None)
    artifact = {
        "schema": SCHEMA,
        "session_id": SESSION_ID,
        "producer_git_sha": producer,
        "producer_dirty_tracked_files": dirty,
        "controls": controls,
        "bars": {"evaluated_fp64_abs": BAR, "discrete": 0.0},
        "first_over_bar": first_over_bar,
        "rows": ordered,
        "inputs": {str(path): {"bytes": path.stat().st_size,
                               "sha256": file_sha256(path)}
                   for path in inputs},
        "ladder_receipt": {
            "source": "user_reported_2026-08-30",
            "day180_twin_sst_rms_degC": 0.005,
            "kt2_bridge_free_days": 359,
            "kt2_daily_endpoint_sst_rms_degC": 0.0104,
            "kt2_daily_endpoint_ssh_rms_mm": 0.29,
            "kt2_endpoint_offset_days": 0.94,
            "independent_sst_rms_degC": 0.39,
            "kt2_claim_scope": "TWIN_CLASS_DAILY_ENDPOINT_WITH_OFFSET",
            "bridge_artifact_producer":
                "7c0f121baed6bc57243e849127040865dda2d597",
            "bridge_artifact_dirty_tracked_files": 1,
            "dirty_override": "LEGOESM_ALLOW_DIRTY=1",
            "temporary_restart_guard_override": "DINO_TWIN_MIN_SPEED",
            "first_d10_mid_run_edit_refusal": "USER_REPORTED_NO_HASH",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for row in ordered:
        print(f"ROW {row['name']} {row['status']} max={row['max_abs']:.17e} "
              f"n={row['mismatch_count']}")
    print(f"FIRST_OVER_BAR={first_over_bar}")
    print(f"WROTE={args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-kt2", type=Path, required=True)
    parser.add_argument("--run-traj-y1", type=Path, required=True)
    parser.add_argument("--lego-kt2", type=Path, required=True)
    parser.add_argument("--lego-d10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
