#!/usr/bin/env python3
"""Paired basin score for generic versus NEMO-literal QCO continuity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import acceptance_gate_90d as G
import acc_thermal_wind as A
import tcarry_basin_reverdict as T


def _scalar(z, key):
    if key not in z.files:
        raise SystemExit(f"STOP missing artifact stamp {key}")
    return np.asarray(z[key]).item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("generic")
    p.add_argument("literal")
    p.add_argument("--producer-commit", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True)
    if dirty:
        raise SystemExit("STOP dirty scorer checkout")

    with np.load(args.generic) as generic_npz, np.load(args.literal) as literal_npz:
        for label, z in (("generic", generic_npz), ("literal", literal_npz)):
            if _scalar(z, "producer_git_sha") != args.producer_commit \
                    or int(_scalar(z, "producer_dirty_tracked_files")) != 0:
                raise SystemExit(f"STOP {label} producer receipt")
            if not bool(_scalar(z, "stable")) \
                    or _scalar(z, "control_dtype") != "float64" \
                    or _scalar(z, "nemo_ladder_mode") != "both" \
                    or _scalar(z, "twin_start_mode") != "BRIDGED_BEFORE" \
                    or _scalar(z, "bridge_before_stress_stagger") != "T":
                raise SystemExit(f"STOP {label} runtime/stagger receipt")
        gcfg = json.loads(str(_scalar(generic_npz, "run_config")))
        lcfg = json.loads(str(_scalar(literal_npz, "run_config")))
        if gcfg.get("barotropic_continuity_evaluation") != "generic" \
                or lcfg.get("barotropic_continuity_evaluation") != "nemo_literal":
            raise SystemExit("STOP selector stamps")
        g_other, l_other = dict(gcfg), dict(lcfg)
        g_other.pop("barotropic_continuity_evaluation")
        l_other.pop("barotropic_continuity_evaluation")
        if g_other != l_other or g_other.get("n_days") != 360:
            raise SystemExit("STOP paired run configs differ beyond selector")
        T._check_day0(generic_npz, literal_npz)
        if not T._bit_identical(generic_npz["land_mask"], literal_npz["land_mask"]):
            raise SystemExit("STOP paired land masks differ")

    generic = G.load_candidate(args.generic, day=360)
    literal = G.load_candidate(args.literal, day=360)
    nemo = T._nemo(360)
    T._reducer_plants(nemo)
    nemo_g, nemo_rows = T._reduce(nemo)
    generic_g, generic_rows_abs = T._reduce(generic)
    literal_g, literal_rows_abs = T._reduce(literal)
    generic_gap, literal_gap = generic_g - nemo_g, literal_g - nemo_g
    delta = literal_gap - generic_gap
    if abs(generic_gap - T.BASELINE[360]) > 2.0 * T.FLOOR[360]:
        raise SystemExit(
            "STOP generic control does not reproduce the frozen day-360 "
            f"basin epoch: got {generic_gap:.17g}, expected "
            f"{T.BASELINE[360]:.17g} within {2.0*T.FLOOR[360]:.17g} Sv")

    wet = A.tmask
    nm, gm, lm = G.metrics(nemo, wet), G.metrics(generic, wet), G.metrics(literal, wet)
    metric_rows, safe = {}, True
    for key in G.KEYS:
        generic_abs = abs(gm[key] - nm[key])
        literal_abs = abs(lm[key] - nm[key])
        regression = literal_abs - generic_abs
        passed = bool(regression <= G.FLOORS[key])
        safe &= passed
        metric_rows[key] = {
            "nemo": nm[key], "generic": gm[key], "nemo_literal": lm[key],
            "generic_abs_gap": generic_abs, "literal_abs_gap": literal_abs,
            "gap_regression": regression,
            "allowed_regression": G.FLOORS[key], "safe": passed,
        }
    verdict = T.classify(delta, generic_gap, T.FLOOR[360], safe)
    # Prove all three substantive classifier branches remain reachable.
    plants = {
        "confirm": T.classify(0.11 * abs(generic_gap), generic_gap, 0.0, True),
        "refute": T.classify(0.01 * abs(generic_gap), generic_gap, 0.0, True),
        "floor": T.classify(0.5 * T.FLOOR[360], generic_gap, T.FLOOR[360], True),
    }
    if plants != {"confirm": "CONFIRMED", "refute": "REFUTED",
                  "floor": "UNRESOLVED/FLOOR"}:
        raise SystemExit(f"STOP classifier plants failed: {plants}")
    result = {
        "schema": "metric-continuity-climate-score-v1",
        "producer_commit": args.producer_commit,
        "generic_sha256": T.sha256(args.generic),
        "literal_sha256": T.sha256(args.literal),
        "nemo_absolute_sv": nemo_g,
        "generic_gap_sv": generic_gap,
        "literal_gap_sv": literal_gap,
        "literal_minus_generic_sv": delta,
        "response_fraction": delta / abs(generic_gap),
        "floor_sv": T.FLOOR[360],
        "frozen_baseline_sv": T.BASELINE[360],
        "verdict": verdict,
        "generic_rows_sv": (generic_rows_abs - nemo_rows).tolist(),
        "literal_rows_sv": (literal_rows_abs - nemo_rows).tolist(),
        "metrics": metric_rows,
        "classifier_plants": plants,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={args.out} sha256={T.sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
