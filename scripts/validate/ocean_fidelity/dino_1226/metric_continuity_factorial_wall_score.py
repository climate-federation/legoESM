#!/usr/bin/env python3
"""Frozen 2x2 wall-flicker score for metric x continuity association."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import eta_flicker_decay as F
import tcarry_basin_reverdict as T
from zdf_stream_bracket import sha256

NEMO_SHA256 = "52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a"
ARMS = {
    "legacy_generic": ("legacy_tracer_midpoint", "generic"),
    "legacy_literal": ("legacy_tracer_midpoint", "nemo_literal"),
    "nemo_generic": ("nemo_vpoint", "generic"),
    "nemo_literal": ("nemo_vpoint", "nemo_literal"),
}


def _scalar(z, key):
    if key not in z.files:
        raise SystemExit(f"STOP missing artifact stamp {key}")
    return np.asarray(z[key]).item()


def _validate(path, producer, metric_selector, association_selector):
    with np.load(path) as z:
        if _scalar(z, "producer_git_sha") != producer \
                or int(_scalar(z, "producer_dirty_tracked_files")) != 0 \
                or not bool(_scalar(z, "stable")) \
                or _scalar(z, "control_dtype") != "float64" \
                or _scalar(z, "nemo_ladder_mode") != "both":
            raise SystemExit(f"STOP {path} provenance/runtime receipt")
        session = os.environ.get("CODEX_SESSION_ID")
        if not session or _scalar(z, "codex_session_id") != session:
            raise SystemExit(f"STOP {path} session receipt")
        cfg = json.loads(str(_scalar(z, "run_config")))
        stress_receipt = {
            key: _scalar(z, key) for key in (
                "twin_start_mode", "bridge_before_stress_stagger",
                "bridge_before_stress_reconstruction_seconds",
                "bridge_before_stress_sha256")
        }
        stress_errors = T.tpoint_stress_receipt_errors(stress_receipt, cfg)
        if stress_errors:
            raise SystemExit(
                f"STOP {path} start/stress receipt:\n  "
                + "\n  ".join(stress_errors))
        if cfg.get("vface_zonal_metric_evaluation") != metric_selector \
                or cfg.get("barotropic_continuity_evaluation") != association_selector \
                or cfg.get("n_days") != 5 or not cfg.get("save_step_eta") \
                or cfg.get("save_3d") or cfg.get("snap_days") != []:
            raise SystemExit(f"STOP {path} selector/day config")
        if int(_scalar(z, "capture_every_steps")) != 1 \
                or float(_scalar(z, "dt_seconds")) != 2700.0 \
                or np.asarray(z["eta"]).shape[0] != 160 \
                or np.asarray(z["eta"]).dtype != np.float64:
            raise SystemExit(f"STOP {path} per-step eta contract")
        return cfg, str(_scalar(z, "initial_state_sha256")), stress_receipt


def _bit_identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape \
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def _verdict(control, candidate):
    c_ratio, c_wall = control["ratio_first8"], control["wall_share_first8"]
    x_ratio, x_wall = candidate["ratio_first8"], candidate["wall_share_first8"]
    control_valid = 2.60 <= c_ratio <= 3.18 and 0.38 <= c_wall <= 0.59
    if not control_valid:
        return "INVALID_CONTROL", False
    if x_ratio <= 1.25 and x_wall <= 0.17:
        return "CONFIRMED", True
    if x_ratio >= 2.30 and x_wall >= 0.38:
        return "REFUTED", True
    return "UNRESOLVED", True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nemo", required=True)
    for arm in ARMS:
        p.add_argument(f"--{arm.replace('_', '-')}", required=True)
    p.add_argument("--producer-commit", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True):
        raise SystemExit("STOP dirty scorer checkout")
    if sha256(Path(args.nemo)) != NEMO_SHA256:
        raise SystemExit("STOP certified NEMO comparator SHA mismatch")

    paths = {arm: getattr(args, arm) for arm in ARMS}
    validated = {arm: _validate(paths[arm], args.producer_commit, *selectors)
                 for arm, selectors in ARMS.items()}
    configs = {arm: item[0] for arm, item in validated.items()}
    initial_hashes = {arm: item[1] for arm, item in validated.items()}
    T.check_tpoint_stress_receipt_plants(
        validated["legacy_generic"][2], configs["legacy_generic"])
    if len(set(initial_hashes.values())) != 1:
        raise SystemExit(f"STOP factorial initial-state hashes differ: {initial_hashes}")
    stripped = []
    for cfg in configs.values():
        cfg = dict(cfg)
        cfg.pop("vface_zonal_metric_evaluation")
        cfg.pop("barotropic_continuity_evaluation")
        stripped.append(cfg)
    if any(cfg != stripped[0] for cfg in stripped[1:]):
        raise SystemExit("STOP factorial configs differ beyond two selectors")
    with np.load(paths["legacy_generic"]) as control_npz:
        for arm in list(ARMS)[1:]:
            with np.load(paths[arm]) as candidate_npz:
                if not _bit_identical(
                        control_npz["land_mask"], candidate_npz["land_mask"]):
                    raise SystemExit(f"STOP {arm} land mask differs")

    out = Path(args.out)
    scores = {}
    for arm, path in paths.items():
        detail = out.with_name(out.stem + f"_{arm}.json")
        rec = F.analyse(args.nemo, path, str(detail), F.ewt.MESH_MASK)
        if not rec["self_check"]["land_poison_identical"] \
                or not (0.95 <= rec["self_check"]["plant_large"]["recovery_ratio"] <= 1.05):
            raise SystemExit(f"STOP {arm} flicker controls")
        scores[arm] = {
            "ratio_first8": rec["regions"]["all"]["ratio_lego_over_nemo"]["first8"],
            "wall_share_first8": rec["wall_share"]["legoESM"]["first8"]["wall"],
        }
    verdict, control_valid = _verdict(
        scores["legacy_generic"], scores["nemo_literal"])
    effects = {
        "metric_at_generic": {
            key: scores["nemo_generic"][key] - scores["legacy_generic"][key]
            for key in scores["legacy_generic"]},
        "association_at_legacy_metric": {
            key: scores["legacy_literal"][key] - scores["legacy_generic"][key]
            for key in scores["legacy_generic"]},
        "interaction": {
            key: (scores["nemo_literal"][key] - scores["nemo_generic"][key]
                  - scores["legacy_literal"][key] + scores["legacy_generic"][key])
            for key in scores["legacy_generic"]},
        "combined": {
            key: scores["nemo_literal"][key] - scores["legacy_generic"][key]
            for key in scores["legacy_generic"]},
    }
    result = {
        "schema": "metric-continuity-wall-score-v2-factorial",
        "producer_commit": args.producer_commit,
        "session_id": os.environ["CODEX_SESSION_ID"],
        "initial_state_sha256": next(iter(initial_hashes.values())),
        "arm_sha256": {arm: sha256(Path(path)) for arm, path in paths.items()},
        "nemo_sha256": NEMO_SHA256,
        "scores": scores,
        "effects": effects,
        "headline_contrast": "nemo_literal minus legacy_generic",
        "control_valid": control_valid,
        "headline_verdict": verdict,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={out} sha256={sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
