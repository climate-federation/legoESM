#!/usr/bin/env python3
"""Gate 6.25 km truth against a 3.125 km dg50-restricted sensitivity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_l1_swg_dg_climate import (  # noqa: E402
    diagnose,
    filtered_npz_provider,
    sha256,
)

ADEQUACY_RELATIVE_BAR = 0.10
DAYS_PER_YEAR = 365


def adequacy_metrics(candidate: dict[str, np.ndarray],
                     sensitivity: dict[str, np.ndarray],
                     *, basin_width_m: float) -> dict:
    """Return the four pre-registered candidate-versus-sensitivity scores."""
    if not np.isfinite(basin_width_m) or basin_width_m <= 0.0:
        raise ValueError("basin width must be positive finite")
    ke_candidate = float(np.mean(candidate["basin_mean_ke_m2_s2"]))
    ke_sensitivity = float(np.mean(sensitivity["basin_mean_ke_m2_s2"]))
    eke_candidate = float(np.mean(candidate["basin_mean_eke_m2_s2"]))
    eke_sensitivity = float(np.mean(sensitivity["basin_mean_eke_m2_s2"]))
    jet_candidate = np.asarray(candidate["jet_y_m"])
    jet_sensitivity = np.asarray(sensitivity["jet_y_m"])
    paired = np.isfinite(jet_candidate) & np.isfinite(jet_sensitivity)
    if not np.any(paired) or ke_sensitivity <= 0.0 or eke_sensitivity <= 0.0:
        raise ValueError("reference adequacy denominator is unavailable")
    spectrum_candidate = np.asarray(
        candidate["mean_kinetic_energy_spectrum_m3_s2"])[1:]
    spectrum_sensitivity = np.asarray(
        sensitivity["mean_kinetic_energy_spectrum_m3_s2"])[1:]
    spectrum_denominator = np.sum(np.abs(spectrum_sensitivity), dtype=np.float64)
    if spectrum_denominator <= 0.0:
        raise ValueError("reference spectrum denominator is zero")
    scores = {
        "kinetic_energy_relative_error": abs(
            ke_candidate - ke_sensitivity) / ke_sensitivity,
        "eddy_kinetic_energy_relative_error": abs(
            eke_candidate - eke_sensitivity) / eke_sensitivity,
        "jet_position_rmse_over_basin_width": float(np.sqrt(np.mean(
            (jet_candidate[paired] - jet_sensitivity[paired]) ** 2,
            dtype=np.float64)) / basin_width_m),
        "ke_spectrum_relative_l1_error": float(np.sum(np.abs(
            spectrum_candidate - spectrum_sensitivity), dtype=np.float64)
            / spectrum_denominator),
    }
    return {
        "scores": scores,
        "bar": ADEQUACY_RELATIVE_BAR,
        "passes": {name: value <= ADEQUACY_RELATIVE_BAR
                   for name, value in scores.items()},
        "adequate": all(value <= ADEQUACY_RELATIVE_BAR
                        for value in scores.values()),
    }


def compare_references(candidate_root: Path, sensitivity_root: Path,
                       *, years: int) -> dict:
    """Compare identical complete years from rest on the common dg50 grid."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
    )

    if type(years) is not int or years <= 0:
        raise ValueError("years must be a positive integer")
    records = []
    for year in range(1, years + 1):
        start_day = (year - 1) * DAYS_PER_YEAR + 1
        providers = (
            filtered_npz_provider(
                candidate_root, DAYS_PER_YEAR, start_day=start_day,
                name="6.25 km truth"),
            filtered_npz_provider(
                sensitivity_root, DAYS_PER_YEAR, start_day=start_day,
                name="3.125 km sensitivity"),
        )
        diagnosed = [diagnose(
            provider, days=DAYS_PER_YEAR,
            card=build_swg_card(SWG_DG50_NAMELIST),
            namelist=SWG_DG50_NAMELIST) for provider in providers]
        records.append({
            "year": year,
            "days": [start_day, start_day + DAYS_PER_YEAR - 1],
            **adequacy_metrics(
                diagnosed[0], diagnosed[1],
                basin_width_m=SWG_DG50_NAMELIST.rn_domsiz_m),
        })
    adequate = all(record["adequate"] for record in records)
    return {
        "schema": "swg-online-reference-adequacy-v1",
        "candidate_root": str(candidate_root.resolve()),
        "sensitivity_root": str(sensitivity_root.resolve()),
        "candidate_manifest_sha256": sha256(candidate_root / "manifest.json"),
        "sensitivity_manifest_sha256": sha256(
            sensitivity_root / "manifest.json"),
        "years": records,
        "rule": (
            "6.25 km is adequate only when all yearly dg50-restricted KE, EKE, "
            "jet-position, and KE-spectrum scores are <=10%"),
        "adequate": adequate,
        "decision_needed": not adequate,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--years", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_references(
            args.candidate_root, args.sensitivity_root, years=args.years)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    print(f"PASS measured={args.years} adequate={report['adequate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
