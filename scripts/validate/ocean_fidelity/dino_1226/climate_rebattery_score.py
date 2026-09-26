#!/usr/bin/env python3
"""Score the registered current-default DINO climate re-battery."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HERE))

MLD_AUDIT_SHA256 = (
    "cf1bcffb4ee7994bd4eb433f6b3fa3ba5ac0f2610463b9bb3133ea0c1762dc14")
NEMO_WALL_SHA256 = (
    "52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a")
HISTORICAL_BASIN_GAP = -0.9519122331848315
BASIN_FLOOR = 0.06173656216045926
MLD_CONFIRM_MAX = 11.2397455
MLD_REFUTE_MIN = 20.2775

CURRENT_DEFAULTS = {
    "bridge_tke": True,
    "barotropic_continuity_evaluation": "nemo_literal",
    "vface_zonal_metric_evaluation": "nemo_vpoint",
    "barotropic_transport_accumulation_evaluation": "nemo_literal",
    "barotropic_seed_evaluation": "nemo_literal",
    "barotropic_een_coefficient_evaluation": "nemo_literal",
    "barotropic_pgf_evaluation": "nemo_literal",
    "zad_qco_evaluation": "nemo_literal",
    "wzv_call2_evaluation": "nemo_literal",
    "dino_wind_profile_evaluation": "nemo_literal",
    "tke_preclosure_coeff_source": "carried_previous_step",
    "tke_shear_evaluation_stage": "step_entry",
    "tke_shear_metric_source": "nemo_qco_live_face",
    "tke_n2_evaluation_stage": "step_entry",
    "tke_matrix_evaluation": "nemo_literal",
    "tke_solver_evaluation": "nemo_literal",
    "tke_etau_exponential_evaluation": "nemo_literal",
    "tke_htau_evaluation": "nemo_literal",
    "tke_mxl_raw_evaluation": "nemo_literal",
    "tke_langmuir_evaluation": "nemo_literal",
    "zdf_implicit_solver_evaluation": "nemo_literal",
    "gm_redi_slope_n2_evaluation": "carried_step_entry",
    "gm_redi_slope_prd_geometry_stage": "before_step",
    "gm_redi_slope_prd_evaluation": "nemo_literal",
    "gm_redi_slope_metric_evaluation": "nemo_reciprocal",
    "gm_redi_slope_face_thickness_evaluation": "nemo_qco_live",
    "gm_redi_flux_face_thickness_evaluation": "nemo_qco_live",
    "gm_redi_horizontal_evaluation": "nemo_metric_literal",
    "gm_redi_vertical_skew_evaluation": "nemo_literal",
    "gm_redi_a33_evaluation": "nemo_literal",
    "gm_redi_w_slope_stage_evaluation": "nemo_post_slope_pair",
    "gm_redi_slope_depth_evaluation": "nemo_qco_live_literal",
    "gm_treguier_vertical_reduction_evaluation": "nemo_left",
    "gm_treguier_sqrt_evaluation": "nemo_forward_exact",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    require(key in data.files, f"missing artifact stamp {key}")
    value = np.asarray(data[key])
    require(value.shape == (), f"artifact stamp {key} is not scalar")
    return value.item()


def mld_verdict(rms_m: float) -> str:
    if rms_m <= MLD_CONFIRM_MAX:
        return "CONFIRM"
    if rms_m >= MLD_REFUTE_MIN:
        return "REFUTE"
    return "PARTIAL/INDETERMINATE"


def basin_verdict(current_gap: float, current_acceptance_5x: bool) -> str:
    delta = current_gap - HISTORICAL_BASIN_GAP
    if abs(delta) <= 2.0 * BASIN_FLOOR:
        return "UNRESOLVED/FLOOR"
    if not current_acceptance_5x:
        return "INVALID_CURRENT"
    response = delta / abs(HISTORICAL_BASIN_GAP)
    if response >= 0.10:
        return "CONFIRMED"
    if response <= 0.02:
        return "REFUTED"
    return "UNRESOLVED"


def wall_verdict(ratio: float, wall_share: float) -> str:
    if ratio <= 1.25 and wall_share <= 0.17:
        return "CONFIRMED"
    if ratio >= 2.30 and wall_share >= 0.38:
        return "REFUTED"
    return "UNRESOLVED"


def classifier_controls() -> dict[str, bool]:
    require(mld_verdict(MLD_CONFIRM_MAX) == "CONFIRM", "MLD confirm plant")
    require(mld_verdict(MLD_REFUTE_MIN) == "REFUTE", "MLD refute plant")
    require(mld_verdict(0.5 * (MLD_CONFIRM_MAX + MLD_REFUTE_MIN))
            == "PARTIAL/INDETERMINATE", "MLD partial plant")
    require(basin_verdict(HISTORICAL_BASIN_GAP, True)
            == "UNRESOLVED/FLOOR", "basin floor plant")
    require(basin_verdict(0.0, True) == "CONFIRMED", "basin confirm plant")
    require(basin_verdict(HISTORICAL_BASIN_GAP - 3 * BASIN_FLOOR, True)
            == "REFUTED", "basin refute plant")
    require(basin_verdict(0.0, False) == "INVALID_CURRENT",
            "basin invalid plant")
    require(wall_verdict(1.0, 0.1) == "CONFIRMED", "wall confirm plant")
    require(wall_verdict(3.0, 0.5) == "REFUTED", "wall refute plant")
    require(wall_verdict(1.5, 0.2) == "UNRESOLVED", "wall unresolved plant")
    return {"mld_branches": True, "basin_branches": True,
            "wall_branches": True}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_arm(path: Path, producer: str, session: str, days: int,
                 snap_days: list[int], save_step_eta: bool) -> dict[str, object]:
    require(path.is_file(), f"missing arm {path}")
    with np.load(path, allow_pickle=False) as data:
        require(str(scalar(data, "producer_git_sha")) == producer,
                f"wrong producer: {path}")
        require(int(scalar(data, "producer_dirty_tracked_files")) == 0,
                f"dirty producer: {path}")
        require(str(scalar(data, "codex_session_id")) == session,
                f"wrong session: {path}")
        require(bool(scalar(data, "stable")), f"unstable arm: {path}")
        require(str(scalar(data, "control_dtype")) == "float64",
                f"non-fp64 control: {path}")
        require(str(scalar(data, "nemo_ladder_mode")) == "both",
                f"wrong NEMO ladder: {path}")
        require(str(scalar(data, "twin_start_mode")) == "bridged",
                f"wrong start mode: {path}")
        require(str(scalar(data, "bridge_before_stress_stagger")) == "T",
                f"wrong stress stagger: {path}")
        require(float(scalar(data, "seasonal_t0_seconds")) ==
                float(scalar(data, "seasonal_t0_reference_seconds")),
                f"wrong seasonal clock: {path}")
        config = json.loads(str(scalar(data, "run_config")))
        require(config.get("n_days") == days, f"wrong duration: {path}")
        require(config.get("snap_days") == snap_days, f"wrong snapshots: {path}")
        require(config.get("save_step_eta") is save_step_eta,
                f"wrong step-eta selection: {path}")
        for key, expected in CURRENT_DEFAULTS.items():
            require(key in config, f"missing current-default stamp {key}: {path}")
            require(config[key] == expected,
                    f"off-default {key}={config[key]!r}, expected {expected!r}")
        if days == 360:
            require(config.get("save_3d") is True, f"3-D snapshots disabled: {path}")
            storage = json.loads(str(scalar(data, "storage_dtypes")))
            for key in ("T3d", "S3d", "eta3d", "u3d", "v3d"):
                require(storage.get(key) == "float64",
                        f"{path}: {key} is not fp64 storage")
            for day in snap_days:
                for field in ("T", "S", "eta", "u", "v"):
                    key = f"{field}3d_day{day}"
                    require(key in data.files, f"missing {key}: {path}")
                    require(np.asarray(data[key]).dtype == np.float64,
                            f"{key} not fp64: {path}")
        else:
            require(int(scalar(data, "capture_every_steps")) == 1,
                    f"wrong eta cadence: {path}")
            require(np.asarray(data["eta"]).shape[0] == 160,
                    f"wrong eta sample count: {path}")
            require(np.asarray(data["eta"]).dtype == np.float64,
                    f"eta is not fp64: {path}")
        return {"config": config, "initial_state_sha256":
                str(scalar(data, "initial_state_sha256")), "sha256": sha256(path)}


def require_pair_identity(a_path: Path, b_path: Path, keys: tuple[str, ...]) -> None:
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for key in keys:
            require(key in a.files and key in b.files, f"pair missing {key}")
            require(np.array_equal(np.asarray(a[key]), np.asarray(b[key])),
                    f"epoch duplicate differs at {key}")


def acceptance_score(path: Path, gate) -> dict[str, object]:
    candidate = gate.load_candidate(str(path), day=90)
    wet = gate.A.tmask & (candidate["land_mask"] > 0.5)[:, :, None]
    nemo = gate.load_nemo_day90()
    candidate_metrics = gate.metrics(candidate, wet)
    nemo_metrics = gate.metrics(nemo, wet)
    rows = gate.classify(candidate_metrics, nemo_metrics, 5)
    return {"passes": int(sum(ok for *_, ok in rows)),
            "fails": int(sum(not ok for *_, ok in rows)),
            "rows": {key: {"candidate": float(cand), "nemo": float(ref),
                           "absolute_error": float(diff),
                           "threshold_5x": float(threshold), "pass": bool(ok)}
                     for key, cand, ref, diff, threshold, ok in rows}}


def score_mld(path: Path, audit) -> dict[str, object]:
    grid = audit.read_nemo_mesh_mask(str(audit.MESH), nn_hls=0)
    initial = audit.read_nemo_restart(str(audit.restart_paths(0)[0]), nn_hls=0)
    cfg = dataclasses.replace(
        audit.dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    bridge = audit.bridge_nemo_to_legoesm_topo(
        grid, initial, periodic_i=True, full_step=True, omega=cfg.omega,
        e3t_mode="both")
    model_config, _ = audit.dino_lat_lon_model_config(bridge.geometry, cfg)
    z_coord = bridge.z_coord
    mask = jnp.asarray(bridge.state.land_mask.data, dtype=jnp.float64)
    h_bathy = jnp.asarray(bridge.state.H_bathy.data, dtype=jnp.float64)
    eos_fn = audit.make_eos_fn(model_config.eos, model_config.eos_linear)
    active = audit._nemo_native_active_3d(mask, z_coord, h_bathy, jnp.float64)
    pattern = str(audit.RUN / f"DINO_{audit.KT0 + audit.STEPS_PER_DAY * 90:08d}_restart*.nc")
    raw = audit.REBUILD(pattern, ["tn", "sn", "sshn"])
    nemo_hml, nemo_base = audit.compute_mld(
        np.moveaxis(raw["tn"], 0, -1), np.moveaxis(raw["sn"], 0, -1),
        raw["sshn"], mask, h_bathy, z_coord, eos_fn, model_config, active)
    weights = np.asarray(grid.e1t * grid.e2t, dtype=np.float64)
    nemo_wet = np.asarray(grid.tmask[..., 0]) > 0.5
    audit.global_map_gate(nemo_hml, nemo_base, weights, nemo_wet,
                          int(np.asarray(z_coord.dz_ref).size), "NEMO day90")
    with np.load(path, allow_pickle=False) as data:
        hml, base = audit.compute_mld(
            np.asarray(data["T3d_day90"], dtype=np.float64),
            np.asarray(data["S3d_day90"], dtype=np.float64),
            np.asarray(data["eta3d_day90"], dtype=np.float64),
            mask, h_bathy, z_coord, eos_fn, model_config, active)
        common_wet = nemo_wet & (np.asarray(data["land_mask"]) > 0.5)
    require(np.array_equal(common_wet, nemo_wet), "MLD wet mask differs")
    audit.global_map_gate(hml, base, weights, common_wet,
                          int(np.asarray(z_coord.dz_ref).size), "current faithful")
    return audit.region_metrics(hml, nemo_hml, weights, common_wet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--climate-a", type=Path, required=True)
    parser.add_argument("--climate-b", type=Path, required=True)
    parser.add_argument("--wall-a", type=Path, required=True)
    parser.add_argument("--wall-b", type=Path, required=True)
    parser.add_argument("--nemo-wall", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(jax.default_backend() == "cpu" and bool(jax.config.x64_enabled),
            "scorer requires CPU/JAX fp64")
    require(not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip(), "scorer checkout has tracked changes")
    audit_path = HERE / "mld_climate_audit.py"
    require(sha256(audit_path) == MLD_AUDIT_SHA256, "MLD audit probe SHA mismatch")
    require(sha256(args.nemo_wall) == NEMO_WALL_SHA256,
            "NEMO wall comparator SHA mismatch")

    climate_a = validate_arm(args.climate_a, args.producer_commit,
                             args.session_id, 360, [0, 90, 360], False)
    climate_b = validate_arm(args.climate_b, args.producer_commit,
                             args.session_id, 360, [0, 90, 360], False)
    wall_a = validate_arm(args.wall_a, args.producer_commit,
                          args.session_id, 5, [], True)
    wall_b = validate_arm(args.wall_b, args.producer_commit,
                          args.session_id, 5, [], True)
    require(climate_a["config"] == climate_b["config"], "climate configs differ")
    require(wall_a["config"] == wall_b["config"], "wall configs differ")
    require(climate_a["initial_state_sha256"] == climate_b["initial_state_sha256"]
            == wall_a["initial_state_sha256"] == wall_b["initial_state_sha256"],
            "initial states differ across battery")
    climate_keys = ("land_mask",) + tuple(
        f"{field}3d_day{day}" for day in (0, 90, 360)
        for field in ("T", "S", "eta", "u", "v"))
    require_pair_identity(args.climate_a, args.climate_b, climate_keys)
    require_pair_identity(args.wall_a, args.wall_b,
                          ("land_mask", "eta", "t_seconds"))

    import acceptance_gate_90d as gate
    import eta_flicker_decay as flicker
    import tcarry_basin_reverdict as basin
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    audit = load_module(audit_path, "_rebattery_mld_audit")

    mld = score_mld(args.climate_a, audit)
    acceptance = acceptance_score(args.climate_a, gate)
    candidate = gate.load_candidate(str(args.climate_a), day=360)
    nemo = basin._nemo(360)
    basin._reducer_plants(nemo)
    nemo_sv, _ = basin._reduce(nemo)
    current_sv, current_rows_abs = basin._reduce(candidate)
    _, nemo_rows_abs = basin._reduce(nemo)
    current_gap = float(current_sv - nemo_sv)
    current_rows = np.asarray(current_rows_abs - nemo_rows_abs).tolist()

    wall_json = args.output.with_name(args.output.stem + "_wall_detail.json")
    wall = flicker.analyse(
        str(args.nemo_wall), str(args.wall_a), str(wall_json),
        flicker.ewt.MESH_MASK)
    require(wall["self_check"]["land_poison_identical"], "wall land plant failed")
    recovery = wall["self_check"]["plant_large"]["recovery_ratio"]
    require(0.95 <= recovery <= 1.05, "wall amplitude plant failed")
    ratio = float(wall["regions"]["all"]["ratio_lego_over_nemo"]["first8"])
    wall_share = float(wall["wall_share"]["legoESM"]["first8"]["wall"])

    result = {
        "schema": "dino-current-default-climate-rebattery-v1",
        "producer_commit": args.producer_commit,
        "session_id": args.session_id,
        "epoch_duplicate_identity": True,
        "arms": {"climate_a": climate_a, "climate_b": climate_b,
                 "wall_a": wall_a, "wall_b": wall_b},
        "mld": {"regions": mld,
                "southern_basin_rms_m": mld["basin"]["rms_difference_m"],
                "verdict": mld_verdict(mld["basin"]["rms_difference_m"]),
                "bands_m": {"confirm_max": MLD_CONFIRM_MAX,
                            "refute_min": MLD_REFUTE_MIN}},
        "basin_day360": {"nemo_sv": float(nemo_sv),
                         "current_gap_sv": current_gap,
                         "current_rows_sv": current_rows,
                         "historical_gap_sv": HISTORICAL_BASIN_GAP,
                         "delta_sv": current_gap - HISTORICAL_BASIN_GAP,
                         "floor_sv": BASIN_FLOOR,
                         "acceptance_5x": acceptance,
                         "verdict": basin_verdict(
                             current_gap, acceptance["passes"] == 5)},
        "wall_flicker": {"ratio_first8": ratio,
                         "wall_share_first8": wall_share,
                         "detail_path": str(wall_json),
                         "detail_sha256": sha256(wall_json),
                         "verdict": wall_verdict(ratio, wall_share)},
        "classifier_controls": classifier_controls(),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mld": result["mld"]["verdict"],
                      "basin_day360": result["basin_day360"]["verdict"],
                      "wall_flicker": result["wall_flicker"]["verdict"]},
                     sort_keys=True))
    print(f"wrote={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
