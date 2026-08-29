#!/usr/bin/env python
"""Score the registered ZDF faithful/control climate arms, CPU/fp64 only.

This is a composition receipt, not a new MLD transcription.  It imports the
post-review MLD audit probe supplied by ``--mld-audit-probe`` and calls that
probe's production-criterion and area-reduction functions.  The five
acceptance metrics come directly from ``acceptance_gate_90d.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HERE))
import acceptance_gate_90d as gate  # noqa: E402

BASELINE_M = 22.479491
LEGACY_TOL_M = 0.001
CONFIRM_MAX_M = 11.2397455
REFUTE_MIN_M = 20.2775
FLOORS = dict(gate.FLOORS)
EXPECTED_AUDIT_PROBE_SHA256 = (
    "cf1bcffb4ee7994bd4eb433f6b3fa3ba5ac0f2610463b9bb3133ea0c1762dc14"
)
EXPECTED_AUDIT_PREREG_SHA256 = (
    "7c4679398d3d2f029a5cfab1fbe4bbbdb970f305ec0b7d488cd3f62fb5afbfd0"
)
EXPECTED_AUDIT_ARTIFACT_SHA256 = (
    "0f00f2c5bad331a871fbb2237aaddc75e789619ff0230c76a9fdd67b16657624"
)
EXPECTED_PRODUCER_HEAD = "e4ab87b422f89bdbf46ddb4aaf0e9db892ec4443"
EXPECTED_FAITHFUL = {
    "bridge_tke": True,
    "tke_preclosure_coeff_source": "carried_previous_step",
    "tke_shear_evaluation_stage": "step_entry",
    "tke_shear_metric_source": "nemo_qco_live_face",
    "dino_wind_profile_evaluation": "nemo_literal",
    "tke_n2_evaluation_stage": "step_entry",
    "tke_matrix_evaluation": "nemo_literal",
    "tke_solver_evaluation": "nemo_literal",
    "tke_etau_exponential_evaluation": "nemo_literal",
    "tke_htau_evaluation": "nemo_literal",
    "tke_mxl_raw_evaluation": "nemo_literal",
    "tke_langmuir_evaluation": "nemo_literal",
    "gm_redi_slope_n2_evaluation": "carried_step_entry",
    "gm_redi_slope_prd_evaluation": "nemo_literal",
    "gm_redi_slope_metric_evaluation": "nemo_reciprocal",
    "gm_redi_slope_face_thickness_evaluation": "nemo_qco_live",
    "gm_redi_slope_depth_evaluation": "nemo_qco_live_literal",
    "zdf_implicit_solver_evaluation": "nemo_literal",
}
EXPECTED_LEGACY = {
    "bridge_tke": True,
    "tke_preclosure_coeff_source": "current_subiteration",
    "tke_shear_evaluation_stage": "implicit_solve_state",
    "tke_shear_metric_source": "tpoint_jacobian",
    "dino_wind_profile_evaluation": "factored_smoothstep",
    "tke_n2_evaluation_stage": "implicit_solve_state",
    "tke_matrix_evaluation": "factored",
    "tke_solver_evaluation": "shared_thomas",
    "tke_etau_exponential_evaluation": "jax_expression",
    "tke_htau_evaluation": "jax_expression",
    "tke_mxl_raw_evaluation": "factored",
    "tke_langmuir_evaluation": "vectorized",
    "gm_redi_slope_n2_evaluation": "recompute",
    "gm_redi_slope_prd_evaluation": "density_roundtrip",
    "gm_redi_slope_metric_evaluation": "division",
    "gm_redi_slope_face_thickness_evaluation": "static_face",
    "gm_redi_slope_depth_evaluation": "legacy_jacobian_t_surface",
    "zdf_implicit_solver_evaluation": "shared_thomas",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scalar(data: np.lib.npyio.NpzFile, key: str):
    require(key in data.files, f"missing arm stamp {key}")
    value = data[key]
    require(value.shape == (), f"arm stamp {key} is not scalar")
    return value.item()


def validate_arm(path: Path, log: Path, expected_sha: str,
                 expected_log_sha: str,
                 expected_config: dict[str, object]) -> dict[str, object]:
    require(path.is_file() and log.is_file(), f"missing arm receipt: {path}, {log}")
    require(sha256(path) == expected_sha, f"arm SHA mismatch: {path}")
    require(sha256(log) == expected_log_sha, f"log SHA mismatch: {log}")
    with np.load(path, allow_pickle=False) as data:
        require(bool(scalar(data, "stable")), f"unstable arm: {path}")
        require(int(scalar(data, "blew_up_at_step")) == -1,
                f"arm blew up: {path}")
        require(str(scalar(data, "control_dtype")) == "float64",
                f"non-fp64 arm: {path}")
        require(str(scalar(data, "nemo_ladder_mode")) == "both",
                f"off-claim ladder: {path}")
        require(str(scalar(data, "twin_start_mode")) == "bridged",
                f"off-claim start: {path}")
        require(float(scalar(data, "seasonal_t0_seconds")) ==
                float(scalar(data, "seasonal_t0_reference_seconds")),
                f"off-season arm: {path}")
        run_config = json.loads(str(scalar(data, "run_config")))
        for key, expected in expected_config.items():
            require(run_config.get(key) == expected,
                    f"{path}: selector {key}={run_config.get(key)!r}, "
                    f"expected {expected!r}")
        ladder_sha = str(scalar(data, "vertical_ladder_sha256"))
        require(re.fullmatch(r"[0-9a-f]{64}", ladder_sha) is not None,
                f"bad ladder hash: {path}")
        land_mask = np.asarray(data["land_mask"], dtype=np.float64)
    text = log.read_text(errors="replace")
    match = re.search(
        r"^PROVENANCE: HEAD=([0-9a-f]{40}) dirty_tracked_files=([0-9]+)$",
        text, re.MULTILINE)
    require(match is not None, f"missing producer provenance: {log}")
    require(match.group(1) == EXPECTED_PRODUCER_HEAD and int(match.group(2)) == 0,
            f"wrong/dirty producer HEAD: {log}")
    require("DONE nsteps=2880 STABLE=True" in text,
            f"missing stable 2880-step completion: {log}")
    return {
        "path": str(path), "sha256": expected_sha,
        "log": str(log), "log_sha256": expected_log_sha,
        "producer_head": match.group(1), "dirty_tracked_files": 0,
        "ladder_sha256": ladder_sha, "run_config": run_config,
        "registered_selector_count": len(expected_config),
        "registered_selectors_exact": True,
        "land_mask": land_mask,
    }


def validate_acceptance_receipt(path: Path, expected_sha: str,
                                candidate: Path | None) -> dict[str, object]:
    require(path.is_file(), f"missing acceptance receipt {path}")
    require(sha256(path) == expected_sha,
            f"acceptance receipt SHA mismatch: {path}")
    text = path.read_text(errors="replace")
    if candidate is None:
        require("SELF-TEST PASS" in text and "PASS 0 | FAIL 5" in text,
                "acceptance self-test receipt did not fire")
        kind = "self_test"
    else:
        require(f"candidate: {candidate}" in text,
                f"wrong candidate in acceptance receipt {path}")
        require("GATE 90D-TWIN: PASS 5 | FAIL 0 | level 5x | total 5" in text,
                f"acceptance receipt is not certified 5/5: {path}")
        kind = "candidate_5x"
    return {"path": str(path), "sha256": expected_sha, "kind": kind}


def decision(faithful_rms: float, legacy_rms: float,
             faithful_errors: dict[str, float],
             legacy_errors: dict[str, float],
             faithful_passes: int, legacy_passes: int) -> tuple[str, dict[str, object]]:
    legacy_ok = abs(legacy_rms - BASELINE_M) <= LEGACY_TOL_M
    worsening = {
        key: faithful_errors[key] - legacy_errors[key] for key in gate.KEYS
    }
    do_no_harm = all(worsening[key] <= FLOORS[key] for key in gate.KEYS)
    tally_ok = faithful_passes >= legacy_passes
    surface_improvement = any(
        legacy_errors[key] - faithful_errors[key] >= FLOORS[key]
        for key in ("smax", "smean")
    )
    conditions = {
        "legacy_baseline_ok": legacy_ok,
        "acceptance_one_floor_do_no_harm": do_no_harm,
        "acceptance_worsening_faithful_minus_legacy": worsening,
        "five_x_pass_tally_not_decreased": tally_ok,
        "southern_surface_density_improves_one_floor": surface_improvement,
    }
    if (faithful_rms >= REFUTE_MIN_M or not legacy_ok or not do_no_harm
            or not tally_ok):
        verdict = "REFUTE"
    elif (faithful_rms <= CONFIRM_MAX_M and surface_improvement):
        verdict = "CONFIRM"
    else:
        verdict = "PARTIAL/INDETERMINATE"
    return verdict, conditions


def decision_controls() -> dict[str, bool]:
    zero = {key: 0.0 for key in gate.KEYS}
    legacy = dict(zero)
    legacy["smax"] = 2.0 * FLOORS["smax"]
    faithful = dict(zero)
    faithful["smax"] = FLOORS["smax"]
    confirm, _ = decision(0.0, BASELINE_M, faithful, legacy, 5, 5)
    require(confirm == "CONFIRM", "planted CONFIRM did not confirm")
    refute_rms, _ = decision(REFUTE_MIN_M, BASELINE_M, zero, zero, 5, 5)
    require(refute_rms == "REFUTE", "planted RMS REFUTE did not refute")
    harmful = dict(zero)
    harmful["acc"] = FLOORS["acc"] + 1.0e-12
    refute_harm, _ = decision(0.0, BASELINE_M, harmful, zero, 5, 5)
    require(refute_harm == "REFUTE", "planted do-no-harm failure did not refute")
    return {
        "planted_confirm_fired": True,
        "planted_rms_refute_fired": True,
        "planted_do_no_harm_refute_fired": True,
    }


def acceptance_score(path: Path) -> dict[str, object]:
    candidate = gate.load_candidate(str(path))
    wet = gate.A.tmask & (candidate["land_mask"] > 0.5)[:, :, None]
    nemo = gate.load_nemo_day90()
    candidate_metrics = gate.metrics(candidate, wet)
    nemo_metrics = gate.metrics(nemo, wet)
    rows = gate.classify(candidate_metrics, nemo_metrics, 5)
    return {
        "candidate": candidate_metrics,
        "nemo": nemo_metrics,
        "absolute_errors": {key: float(diff) for key, _, _, diff, _, _ in rows},
        "thresholds_5x": {key: float(threshold)
                          for key, _, _, _, threshold, _ in rows},
        "passes_5x": int(sum(ok for *_, ok in rows)),
        "fails_5x": int(sum(not ok for *_, ok in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithful", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--faithful-log", type=Path, required=True)
    parser.add_argument("--legacy-log", type=Path, required=True)
    parser.add_argument("--faithful-sha256", required=True)
    parser.add_argument("--legacy-sha256", required=True)
    parser.add_argument("--faithful-log-sha256", required=True)
    parser.add_argument("--legacy-log-sha256", required=True)
    parser.add_argument("--mld-audit-probe", type=Path, required=True)
    parser.add_argument("--mld-audit-prereg", type=Path, required=True)
    parser.add_argument("--published-mld-artifact", type=Path, required=True)
    parser.add_argument("--acceptance-selftest-log", type=Path, required=True)
    parser.add_argument("--acceptance-selftest-log-sha256", required=True)
    parser.add_argument("--faithful-acceptance-log", type=Path, required=True)
    parser.add_argument("--faithful-acceptance-log-sha256", required=True)
    parser.add_argument("--legacy-acceptance-log", type=Path, required=True)
    parser.add_argument("--legacy-acceptance-log-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(jax.default_backend() == "cpu", "scorer must run on CPU")
    require(bool(jax.config.x64_enabled), "scorer requires JAX x64")
    set_policy(PrecisionPolicy.fp64())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip()
    require(not status, f"tracked worktree is dirty:\n{status}")
    require(sha256(args.mld_audit_probe) == EXPECTED_AUDIT_PROBE_SHA256,
            "MLD audit probe SHA mismatch")
    require(sha256(args.mld_audit_prereg) == EXPECTED_AUDIT_PREREG_SHA256,
            "MLD audit prereg SHA mismatch")
    require(sha256(args.published_mld_artifact) ==
            EXPECTED_AUDIT_ARTIFACT_SHA256,
            "published MLD audit artifact SHA mismatch")
    published = json.loads(args.published_mld_artifact.read_text())
    published_baseline = float(
        published["stats"]["basin_legacy"]["90"]["basin"]["rms_difference_m"])
    require(abs(published_baseline - BASELINE_M) <= 0.5e-6,
            "published artifact does not support the rounded frozen baseline")
    audit = load_module(args.mld_audit_probe, "_zdf_bound_mld_audit")

    arms = {
        "faithful": validate_arm(
            args.faithful, args.faithful_log, args.faithful_sha256,
            args.faithful_log_sha256, EXPECTED_FAITHFUL),
        "legacy": validate_arm(
            args.legacy, args.legacy_log, args.legacy_sha256,
            args.legacy_log_sha256, EXPECTED_LEGACY),
    }
    require(arms["faithful"]["ladder_sha256"] == arms["legacy"]["ladder_sha256"],
            "arm vertical ladders differ")

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
    active_3d = audit._nemo_native_active_3d(
        mask, z_coord, h_bathy, jnp.float64)
    ladder_sha = audit.VERTICAL_LADDER_SHA256(z_coord)
    require(ladder_sha == arms["faithful"]["ladder_sha256"],
            "saved arm ladder differs from rebuilt audit ladder")

    day = 90
    pattern = str(audit.RUN / f"DINO_{audit.KT0 + audit.STEPS_PER_DAY * day:08d}_restart*.nc")
    raw = audit.REBUILD(pattern, ["tn", "sn", "sshn"])
    nemo_hml, nemo_base = audit.compute_mld(
        np.moveaxis(raw["tn"], 0, -1), np.moveaxis(raw["sn"], 0, -1),
        raw["sshn"], mask, h_bathy, z_coord, eos_fn, model_config, active_3d)
    weights = np.asarray(grid.e1t * grid.e2t, dtype=np.float64)
    nemo_wet = np.asarray(grid.tmask[..., 0]) > 0.5
    audit.global_map_gate(nemo_hml, nemo_base, weights, nemo_wet,
                          int(np.asarray(z_coord.dz_ref).size), "NEMO day90")

    mld_scores: dict[str, object] = {}
    for name, path in (("faithful", args.faithful), ("legacy", args.legacy)):
        with np.load(path, allow_pickle=False) as data:
            hml, base = audit.compute_mld(
                np.asarray(data["T3d_day90"], dtype=np.float64),
                np.asarray(data["S3d_day90"], dtype=np.float64),
                np.asarray(data["eta3d_day90"], dtype=np.float64),
                mask, h_bathy, z_coord, eos_fn, model_config, active_3d)
            common_wet = nemo_wet & (np.asarray(data["land_mask"]) > 0.5)
        require(np.array_equal(common_wet, nemo_wet),
                f"{name}: surface wet mask differs from NEMO")
        audit.global_map_gate(hml, base, weights, common_wet,
                              int(np.asarray(z_coord.dz_ref).size), name)
        mld_scores[name] = audit.region_metrics(
            hml, nemo_hml, weights, common_wet)

    mld_controls = {
        "synthetic": audit.synthetic_controls(
            float(model_config.g), float(model_config.rho_0),
            float(model_config.gm_redi.mld_rho_c)),
        "reduction_poison": audit.reduction_poison_control(
            weights, nemo_wet),
    }
    acceptance = {
        "faithful": acceptance_score(args.faithful),
        "legacy": acceptance_score(args.legacy),
    }
    acceptance_receipts = {
        "self_test": validate_acceptance_receipt(
            args.acceptance_selftest_log,
            args.acceptance_selftest_log_sha256, None),
        "faithful": validate_acceptance_receipt(
            args.faithful_acceptance_log,
            args.faithful_acceptance_log_sha256, args.faithful),
        "legacy": validate_acceptance_receipt(
            args.legacy_acceptance_log,
            args.legacy_acceptance_log_sha256, args.legacy),
    }
    faithful_rms = mld_scores["faithful"]["basin"]["rms_difference_m"]
    legacy_rms = mld_scores["legacy"]["basin"]["rms_difference_m"]
    verdict, conditions = decision(
        faithful_rms, legacy_rms,
        acceptance["faithful"]["absolute_errors"],
        acceptance["legacy"]["absolute_errors"],
        acceptance["faithful"]["passes_5x"],
        acceptance["legacy"]["passes_5x"])

    provenance_paths = [
        Path(__file__).resolve(), args.faithful, args.legacy,
        args.faithful_log, args.legacy_log, args.mld_audit_probe,
        args.mld_audit_prereg, Path(gate.__file__).resolve(), audit.MESH,
        args.published_mld_artifact, args.acceptance_selftest_log,
        args.faithful_acceptance_log, args.legacy_acceptance_log,
        *audit.restart_paths(90),
    ]
    result = {
        "schema": "dino-zdf-climate-score-v1",
        "verdict": verdict,
        "registered_bands_m": {
            "baseline": BASELINE_M, "legacy_tolerance": LEGACY_TOL_M,
            "confirm_max": CONFIRM_MAX_M, "refute_min": REFUTE_MIN_M,
            "published_unrounded_baseline": published_baseline,
        },
        "scope": audit.__doc__.splitlines()[0],
        "mld_day90": mld_scores,
        "acceptance_gate_5x": acceptance,
        "acceptance_receipts": acceptance_receipts,
        "registered_conditions": conditions,
        "controls": {
            "mld_audit": mld_controls,
            "decision": decision_controls(),
        },
        "arms": {
            name: {key: value for key, value in meta.items()
                   if key != "land_mask"}
            for name, meta in arms.items()
        },
        "provenance": {
            "repo_head": head, "dirty_tracked_files": 0,
            "jax_backend": jax.default_backend(),
            "jax_x64": bool(jax.config.x64_enabled),
            "vertical_ladder_sha256": ladder_sha,
            "input_sha256": {
                str(path): sha256(path) for path in sorted(set(provenance_paths))
            },
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CONTROLS: PASS")
    print(f"MLD faithful basin RMS: {faithful_rms:.9f} m")
    print(f"MLD legacy basin RMS: {legacy_rms:.9f} m ")
    for name in ("faithful", "legacy"):
        score = acceptance[name]
        print(f"ACCEPTANCE {name}: PASS {score['passes_5x']} | "
              f"FAIL {score['fails_5x']} | level 5x")
    print("DO_NO_HARM: " +
          ("PASS" if conditions["acceptance_one_floor_do_no_harm"] else "FAIL"))
    print("SOUTHERN_SURFACE_IMPROVEMENT: " +
          ("PASS" if conditions["southern_surface_density_improves_one_floor"]
           else "FAIL"))
    print(f"REGISTERED VERDICT: {verdict}")
    print(f"RESULT: {args.output}")
    print(f"RESULT_SHA256: {sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
