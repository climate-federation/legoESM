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
from nemo_testcase_l1_swg_online_predictability import (  # noqa: E402
    rotational_velocity_from_vorticity,
)

ADEQUACY_RELATIVE_BAR = 0.10
EARLY_PATTERN_CORRELATION_BAR = 0.99
EARLY_DAYS = 8
DAYS_PER_YEAR = 365


def adequacy_metrics(candidate: dict[str, np.ndarray],
                     sensitivity: dict[str, np.ndarray],
                     *, basin_width_m: float,
                     uncertainty: dict[str, float] | None = None) -> dict:
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
    wavenumber = np.asarray(sensitivity["spectrum_wavenumber_m-1"])
    if not np.array_equal(np.asarray(candidate["spectrum_wavenumber_m-1"]),
                          wavenumber):
        raise ValueError("reference spectra use different wavenumber bins")
    # Drop only a zero-wavenumber (mean) bin.  The pre-registered sine-transform
    # spectrum has none: its first bin is the gravest basin mode.
    resolved = wavenumber > 0.0
    spectrum_candidate = np.asarray(
        candidate["mean_kinetic_energy_spectrum_m3_s2"])[resolved]
    spectrum_sensitivity = np.asarray(
        sensitivity["mean_kinetic_energy_spectrum_m3_s2"])[resolved]
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
    bars = {name: max(ADEQUACY_RELATIVE_BAR,
                      0.0 if uncertainty is None else uncertainty[name])
            for name in scores}
    return {
        "scores": scores,
        "bars": bars,
        "sampling_uncertainty": uncertainty,
        "passes": {name: value <= bars[name]
                   for name, value in scores.items()},
        "adequate": all(value <= bars[name]
                        for name, value in scores.items()),
    }


def early_path_metrics(candidate: dict[str, np.ndarray],
                       sensitivity: dict[str, np.ndarray],
                       wet: np.ndarray) -> dict:
    """Compare common-path truncation error through the first eight days."""
    wet = np.asarray(wet, dtype=bool)
    records = {}
    for name in ("ssh", "relative_vorticity", "rotational_velocity"):
        left = np.asarray(candidate[name], dtype=np.float64)
        right = np.asarray(sensitivity[name], dtype=np.float64)
        if left.shape != right.shape or left.shape[0] < EARLY_DAYS:
            raise ValueError(f"early {name} arrays are incomplete")
        if name == "rotational_velocity":
            left_flat = left[:EARLY_DAYS, :, wet]
            right_flat = right[:EARLY_DAYS, :, wet]
        else:
            left_flat = left[:EARLY_DAYS, wet]
            right_flat = right[:EARLY_DAYS, wet]
        rms = np.sqrt(np.mean((left_flat - right_flat) ** 2, axis=1))
        denominator = np.sqrt(np.mean(right_flat**2, axis=1))
        if np.any(denominator <= 0.0):
            raise ValueError(f"early {name} normalization is zero")
        nrms = rms / denominator
        correlation = np.asarray([
            np.corrcoef(left_flat[day].ravel(), right_flat[day].ravel())[0, 1]
            for day in range(EARLY_DAYS)])
        if not np.all(np.isfinite(correlation)):
            raise ValueError(f"early {name} pattern correlation is undefined")
        crossings = {}
        for bar in (0.01, 0.10):
            hit = np.flatnonzero(nrms > bar)
            crossings[str(bar)] = None if hit.size == 0 else int(hit[0] + 1)
        records[name] = {
            "normalized_rms_by_day": nrms.tolist(),
            "pattern_correlation_by_day": correlation.tolist(),
            "first_day_above": crossings,
            "passes": bool(np.all(nrms <= ADEQUACY_RELATIVE_BAR)
                           and np.all(correlation >= EARLY_PATTERN_CORRELATION_BAR)),
        }
    return {
        "days": EARLY_DAYS,
        "normalized_rms_bar": ADEQUACY_RELATIVE_BAR,
        "pattern_correlation_bar": EARLY_PATTERN_CORRELATION_BAR,
        "fields": records,
        "adequate": all(item["passes"] for item in records.values()),
    }


def _load_early_fields(root: Path) -> dict[str, np.ndarray]:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        restricted_relative_vorticity,
    )

    manifest = json.loads((root / "manifest.json").read_text())
    pins = manifest.get("output_daily_sha256", manifest.get("daily_sha256", {}))
    geometry = cgrid_geometry(SWG_DG50_NAMELIST)
    fields = {name: [] for name in (
        "ssh", "relative_vorticity", "rotational_velocity")}
    for day in range(1, EARLY_DAYS + 1):
        path = root / f"day{day:04d}.npz"
        if sha256(path) != pins.get(path.name):
            raise ValueError(f"early-path digest differs: {path}")
        with np.load(path, allow_pickle=False) as saved:
            native = {"T": np.asarray(saved["ssh_m"]),
                      "U": np.asarray(saved["u_m_s"]),
                      "V": np.asarray(saved["v_m_s"])}
        fields["ssh"].append(native["T"])
        zeta = restricted_relative_vorticity(native, geometry)
        fields["relative_vorticity"].append(zeta)
        fields["rotational_velocity"].append(np.stack(
            rotational_velocity_from_vorticity(
                zeta, geometry["mask"]["T"],
                float(geometry["spacing_m"]))))
    return {name: np.stack(value) for name, value in fields.items()}


def _year_to_year_uncertainty(years: list[dict[str, np.ndarray]],
                              basin_width_m: float) -> dict[str, float]:
    """Maximum adjacent 6.25-km annual difference in each gate metric."""
    pairs = [adequacy_metrics(years[index], years[index + 1],
                              basin_width_m=basin_width_m)["scores"]
             for index in range(len(years) - 1)]
    return {name: max(pair[name] for pair in pairs) for name in pairs[0]}


def compare_references(candidate_root: Path, sensitivity_root: Path,
                       *, years: int) -> dict:
    """Compare the early common path and one-year statistics with uncertainty."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
    )

    if years != 1:
        raise ValueError("the registered sensitivity acquisition is exactly one year")
    card = build_swg_card(SWG_DG50_NAMELIST)
    # The dg50 coordinates, sine-transform spectrum and jet live only on the
    # explicitly selected online path; the registered default reads the
    # 6.25 km coordinate table and would under-report a jet shift eightfold.
    candidate_years = []
    sensitivity_year = None
    for year in range(1, 4):
        start_day = (year - 1) * DAYS_PER_YEAR + 1
        provider = filtered_npz_provider(
            candidate_root, DAYS_PER_YEAR, start_day=start_day,
            name=f"6.25 km truth year {year}")
        candidate_years.append(diagnose(
            provider, days=DAYS_PER_YEAR, card=card,
            namelist=SWG_DG50_NAMELIST, diagnostics="online"))
        if year == 1:
            sensitivity = filtered_npz_provider(
                sensitivity_root, DAYS_PER_YEAR, start_day=1,
                name="3.125 km sensitivity year 1")
            if sensitivity.sampling != provider.sampling:
                raise ValueError(
                    f"sampling mismatch: candidate {provider.sampling}, "
                    f"sensitivity {sensitivity.sampling}")
            sensitivity_year = diagnose(
                sensitivity, days=DAYS_PER_YEAR, card=card,
                namelist=SWG_DG50_NAMELIST, diagnostics="online")
    uncertainty = _year_to_year_uncertainty(
        candidate_years, SWG_DG50_NAMELIST.rn_domsiz_m)
    annual = adequacy_metrics(
        candidate_years[0], sensitivity_year,
        basin_width_m=SWG_DG50_NAMELIST.rn_domsiz_m,
        uncertainty=uncertainty)
    wet = np.asarray(card.recipe.land_mask, dtype=bool)
    early = early_path_metrics(
        _load_early_fields(candidate_root),
        _load_early_fields(sensitivity_root), wet)
    adequate = bool(early["adequate"] and annual["adequate"])
    return {
        "schema": "swg-online-reference-adequacy-v1",
        "candidate_root": str(candidate_root.resolve()),
        "sensitivity_root": str(sensitivity_root.resolve()),
        "candidate_manifest_sha256": sha256(candidate_root / "manifest.json"),
        "sensitivity_manifest_sha256": sha256(
            sensitivity_root / "manifest.json"),
        "early_common_path": early,
        "year1_statistics": annual,
        "rule": (
            "6.25 km is adequate only when days 1-8 have NRMS<=10% and "
            "correlation>=0.99 for SSH/vorticity/rotational velocity, and "
            "year-1 KE, EKE, "
            "jet and DST-spectrum differences are <=max(10%, measured 6.25-km "
            "year-to-year variability)"),
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
    print(
        f"{'PASS' if report['adequate'] else 'FAIL'} "
        f"measured={args.years} adequate={report['adequate']}")
    return 0 if report["adequate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
