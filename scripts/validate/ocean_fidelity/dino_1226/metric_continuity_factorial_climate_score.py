#!/usr/bin/env python3
"""Frozen 2x2 basin score for V-face metric x continuity association."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import acceptance_gate_90d as G
import acc_thermal_wind as A
import tcarry_basin_reverdict as T


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
                or int(_scalar(z, "producer_dirty_tracked_files")) != 0:
            raise SystemExit(f"STOP {path} producer receipt")
        if not bool(_scalar(z, "stable")) \
                or _scalar(z, "control_dtype") != "float64" \
                or _scalar(z, "nemo_ladder_mode") != "both" \
                or _scalar(z, "twin_start_mode") != "BRIDGED_BEFORE" \
                or _scalar(z, "bridge_before_stress_stagger") != "T":
            raise SystemExit(f"STOP {path} runtime/stagger receipt")
        cfg = json.loads(str(_scalar(z, "run_config")))
        if cfg.get("vface_zonal_metric_evaluation") != metric_selector \
                or cfg.get("barotropic_continuity_evaluation") != association_selector \
                or cfg.get("n_days") != 360:
            raise SystemExit(f"STOP {path} selector/day stamps")
        return cfg


def _metric_safety(nemo, control, candidate):
    wet = A.tmask
    nm = G.metrics(nemo, wet)
    cm = G.metrics(control, wet)
    xm = G.metrics(candidate, wet)
    rows, safe = {}, True
    for key in G.KEYS:
        control_abs = abs(cm[key] - nm[key])
        candidate_abs = abs(xm[key] - nm[key])
        regression = candidate_abs - control_abs
        passed = bool(regression <= G.FLOORS[key])
        safe &= passed
        rows[key] = {
            "nemo": nm[key], "control": cm[key], "candidate": xm[key],
            "control_abs_gap": control_abs,
            "candidate_abs_gap": candidate_abs,
            "gap_regression": regression,
            "allowed_regression": G.FLOORS[key], "safe": passed,
        }
    return rows, safe


def main() -> int:
    p = argparse.ArgumentParser()
    for arm in ARMS:
        p.add_argument(f"--{arm.replace('_', '-')}", required=True)
    p.add_argument("--producer-commit", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True)
    if dirty:
        raise SystemExit("STOP dirty scorer checkout")

    paths = {arm: getattr(args, arm) for arm in ARMS}
    configs = {
        arm: _validate(paths[arm], args.producer_commit, *selectors)
        for arm, selectors in ARMS.items()
    }
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
                T._check_day0(control_npz, candidate_npz)
                if not T._bit_identical(
                        control_npz["land_mask"], candidate_npz["land_mask"]):
                    raise SystemExit(f"STOP {arm} land mask differs")

    fields = {arm: G.load_candidate(path, day=360)
              for arm, path in paths.items()}
    nemo = T._nemo(360)
    T._reducer_plants(nemo)
    nemo_g, nemo_rows = T._reduce(nemo)
    basin, row_gaps = {}, {}
    for arm, field in fields.items():
        absolute, rows = T._reduce(field)
        basin[arm] = absolute - nemo_g
        row_gaps[arm] = (rows - nemo_rows).tolist()

    baseline = basin["legacy_generic"]
    if abs(baseline - T.BASELINE[360]) > 2.0 * T.FLOOR[360]:
        raise SystemExit(
            "STOP legacy/generic control does not reproduce frozen day-360 "
            f"basin epoch: got {baseline:.17g}, expected "
            f"{T.BASELINE[360]:.17g} within {2.0*T.FLOOR[360]:.17g} Sv")

    effects = {
        "metric_at_generic_sv": basin["nemo_generic"] - basin["legacy_generic"],
        "association_at_legacy_metric_sv": (
            basin["legacy_literal"] - basin["legacy_generic"]),
        "association_at_nemo_metric_sv": (
            basin["nemo_literal"] - basin["nemo_generic"]),
        "metric_at_literal_sv": basin["nemo_literal"] - basin["legacy_literal"],
        "interaction_sv": (basin["nemo_literal"] - basin["nemo_generic"]
                           - basin["legacy_literal"] + basin["legacy_generic"]),
        "combined_sv": basin["nemo_literal"] - basin["legacy_generic"],
    }
    metric_rows, safe = _metric_safety(
        nemo, fields["legacy_generic"], fields["nemo_literal"])
    verdict = T.classify(effects["combined_sv"], baseline, T.FLOOR[360], safe)
    plants = {
        "confirm": T.classify(0.11 * abs(baseline), baseline, 0.0, True),
        "refute": T.classify(0.01 * abs(baseline), baseline, 0.0, True),
        "floor": T.classify(0.5 * T.FLOOR[360], baseline,
                            T.FLOOR[360], True),
    }
    if plants != {"confirm": "CONFIRMED", "refute": "REFUTED",
                  "floor": "UNRESOLVED/FLOOR"}:
        raise SystemExit(f"STOP classifier plants failed: {plants}")
    result = {
        "schema": "metric-continuity-climate-score-v2-factorial",
        "producer_commit": args.producer_commit,
        "arm_sha256": {arm: T.sha256(path) for arm, path in paths.items()},
        "nemo_absolute_sv": nemo_g,
        "basin_gap_sv": basin,
        "effects": effects,
        "effect_response_fraction": {
            key: value / abs(baseline) for key, value in effects.items()},
        "floor_sv": T.FLOOR[360],
        "frozen_baseline_sv": T.BASELINE[360],
        "headline_contrast": "nemo_literal minus legacy_generic",
        "headline_verdict": verdict,
        "rows_sv": row_gaps,
        "combined_compensation_metrics": metric_rows,
        "classifier_plants": plants,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={args.out} sha256={T.sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
