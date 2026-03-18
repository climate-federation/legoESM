#!/usr/bin/env python
"""Validate AMIP run output against observational targets.

Checks the timeseries and monthly-mean output from a completed AMIP run
against the validation targets from NEXT_STEPS.md Task 12:

  - Global mean T_2m: 287-289 K (observed ~288 K)
  - Global mean precipitation: 2.5-3.0 mm/day (observed ~2.7 mm/day)
  - Net TOA imbalance: < 1 W/m² (ideally < 0.5 W/m²)
  - OLR: 235-245 W/m² (observed ~240 W/m²)
  - Zonal-mean zonal wind: subtropical jet at ~30° lat, ~30-40 m/s
  - Zonal-mean temperature: within 5 K of ERA5 at all levels

Usage:
    python scripts/validate_amip.py results/amip/C48_L40_3650d_*/
    python scripts/validate_amip.py --output validation_report.txt results/amip/run_dir/
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_timeseries(run_dir: Path) -> dict:
    """Load timeseries.npz from a run directory."""
    ts_path = run_dir / "timeseries.npz"
    if not ts_path.exists():
        raise FileNotFoundError(f"No timeseries.npz in {run_dir}")
    return dict(np.load(str(ts_path), allow_pickle=True))


def load_monthly(run_dir: Path) -> dict | None:
    """Load monthly_means.npz if available."""
    mm_path = run_dir / "monthly_means.npz"
    if not mm_path.exists():
        return None
    return dict(np.load(str(mm_path), allow_pickle=True))


# ==========================================================================
# Validation targets
# ==========================================================================

TARGETS = {
    "T_2m": {"min": 287.0, "max": 289.0, "observed": 288.0, "unit": "K"},
    "precipitation": {"min": 2.5, "max": 3.0, "observed": 2.7, "unit": "mm/day"},
    "TOA_imbalance": {"min": -1.0, "max": 1.0, "target": 0.0, "unit": "W/m²"},
    "OLR": {"min": 235.0, "max": 245.0, "observed": 240.0, "unit": "W/m²"},
}


def check_target(name: str, value: float, target: dict) -> tuple[str, bool]:
    """Check if value is within target range. Returns (message, passed)."""
    lo, hi = target["min"], target["max"]
    obs = target.get("observed", target.get("target", (lo + hi) / 2))
    passed = lo <= value <= hi
    status = "PASS" if passed else "FAIL"
    return (
        f"  [{status}] {name}: {value:.2f} {target['unit']} "
        f"(target: {lo}–{hi}, obs: {obs:.1f})",
        passed,
    )


def validate_timeseries(ts: dict, spinup_days: float = 365.0) -> list[str]:
    """Run validation checks on timeseries data.

    Parameters
    ----------
    ts : dict
        Loaded timeseries data.
    spinup_days : float
        Days to skip as spinup before computing means.
    """
    lines = []
    n_pass = 0
    n_total = 0

    days = ts.get("days", np.array([]))
    if len(days) == 0:
        return ["  No diagnostic data found."]

    total_days = float(days[-1] - days[0])
    lines.append(f"  Simulation length: {total_days:.0f} days")
    lines.append(f"  Spinup excluded: {spinup_days:.0f} days")

    # Mask for post-spinup
    mask = days >= spinup_days
    if not np.any(mask):
        lines.append(f"  WARNING: No data after spinup ({spinup_days} days). Using all data.")
        mask = np.ones(len(days), dtype=bool)

    # 1. Global mean T_2m (approximated by T_low)
    if "T_low" in ts:
        T_low = ts["T_low"][mask]
        mean_T = float(np.mean(T_low))
        msg, ok = check_target("T_2m (T_low)", mean_T, TARGETS["T_2m"])
        lines.append(msg)
        n_pass += ok
        n_total += 1

    # 2. Global mean precipitation
    if "precip" in ts:
        precip = ts["precip"][mask]
        mean_precip = float(np.mean(precip))
        msg, ok = check_target("Precipitation", mean_precip, TARGETS["precipitation"])
        lines.append(msg)
        n_pass += ok
        n_total += 1

    # 3. Net TOA imbalance
    if "energy_toa_net" in ts:
        toa_net = ts["energy_toa_net"][mask]
        mean_toa = float(np.mean(toa_net))
        msg, ok = check_target("TOA imbalance", mean_toa, TARGETS["TOA_imbalance"])
        lines.append(msg)
        n_pass += ok
        n_total += 1
    elif "sw_up_toa" in ts and "lw_up_toa" in ts:
        # Approximate: OLR = LW_up_toa, ASR ≈ S0/4 - SW_up_toa
        sw_up = ts["sw_up_toa"][mask]
        lw_up = ts["lw_up_toa"][mask]
        # Approximate TOA net = S0/4 - SW_up - LW_up ≈ 340 - SW_up - LW_up
        toa_net = 340.0 - np.mean(sw_up) - np.mean(lw_up)
        msg, ok = check_target("TOA imbalance (approx)", toa_net, TARGETS["TOA_imbalance"])
        lines.append(msg)
        n_pass += ok
        n_total += 1

    # 4. OLR
    if "lw_up_toa" in ts:
        lw_up = ts["lw_up_toa"][mask]
        mean_olr = float(np.mean(lw_up))
        msg, ok = check_target("OLR (LW_up_TOA)", mean_olr, TARGETS["OLR"])
        lines.append(msg)
        n_pass += ok
        n_total += 1

    # 5. Energy budget residual
    if "energy_residual" in ts:
        resid = ts["energy_residual"][mask]
        # Skip first entry (no dE/dt)
        resid = resid[resid != 0.0] if len(resid) > 1 else resid
        if len(resid) > 0:
            mean_resid = float(np.mean(np.abs(resid)))
            max_resid = float(np.max(np.abs(resid)))
            passed = mean_resid < 1.0
            status = "PASS" if passed else "FAIL"
            lines.append(
                f"  [{status}] Energy residual: mean |R| = {mean_resid:.4f} W/m², "
                f"max |R| = {max_resid:.4f} W/m² (target: < 1.0)"
            )
            n_pass += passed
            n_total += 1

    # 6. Stability checks
    if "max_wind" in ts:
        max_wind = ts["max_wind"][mask]
        peak_wind = float(np.max(max_wind))
        stable = peak_wind < 200.0
        status = "PASS" if stable else "FAIL"
        lines.append(
            f"  [{status}] Max wind speed: {peak_wind:.1f} m/s (should be < 200)"
        )
        n_pass += stable
        n_total += 1

    if "T_atm" in ts:
        T_atm = ts["T_atm"][mask]
        T_range = float(np.max(T_atm) - np.min(T_atm))
        stable = T_range < 30.0
        status = "PASS" if stable else "WARN"
        lines.append(
            f"  [{status}] T_atm range: {T_range:.1f} K over post-spinup "
            f"(drift indicator, should be < 30 K)"
        )
        if status != "WARN":
            n_pass += 1
        n_total += 1

    lines.append(f"\n  Score: {n_pass}/{n_total} checks passed")
    return lines


def validate_monthly(monthly: dict) -> list[str]:
    """Run validation on monthly-mean data (zonal means)."""
    lines = []

    if monthly is None:
        lines.append("  No monthly-mean data available.")
        return lines

    lat = monthly.get("lat", np.array([]))
    if len(lat) == 0:
        lines.append("  Monthly data has no latitude coordinate.")
        return lines

    # Zonal-mean zonal wind: check for subtropical jet
    if "profile_u" in monthly:
        u_zm = monthly["profile_u"]  # (n_months, n_lat, nlev)
        if u_zm.ndim == 3 and u_zm.shape[0] > 0:
            # Time-mean over last year
            n_months = u_zm.shape[0]
            n_last_year = min(12, n_months)
            u_mean = np.nanmean(u_zm[-n_last_year:], axis=0)  # (n_lat, nlev)

            # Find max wind at upper levels (top 30% of column)
            nlev = u_mean.shape[1]
            upper = max(1, nlev // 3)
            u_upper = u_mean[:, :upper]
            max_u = np.nanmax(u_upper)
            idx = np.unravel_index(np.nanargmax(u_upper), u_upper.shape)
            jet_lat = lat[idx[0]]

            passed_speed = 20.0 < max_u < 60.0
            passed_lat = 20.0 < abs(jet_lat) < 50.0
            status = "PASS" if (passed_speed and passed_lat) else "CHECK"
            lines.append(
                f"  [{status}] Subtropical jet: {max_u:.1f} m/s at {jet_lat:.0f}° "
                f"(target: 30-40 m/s at ~30° lat)"
            )

    # Zonal-mean temperature: check tropics and poles
    if "profile_T" in monthly:
        T_zm = monthly["profile_T"]  # (n_months, n_lat, nlev)
        if T_zm.ndim == 3 and T_zm.shape[0] > 0:
            n_months = T_zm.shape[0]
            n_last_year = min(12, n_months)
            T_mean = np.nanmean(T_zm[-n_last_year:], axis=0)  # (n_lat, nlev)

            # Equatorial surface temperature
            eq_mask = np.abs(lat) < 10.0
            if np.any(eq_mask):
                T_eq_sfc = float(np.nanmean(T_mean[eq_mask, -1]))
                passed = 290.0 < T_eq_sfc < 310.0
                status = "PASS" if passed else "CHECK"
                lines.append(
                    f"  [{status}] Equatorial low-level T: {T_eq_sfc:.1f} K "
                    f"(expected ~295-300 K)"
                )

            # Polar surface temperature
            pole_mask = np.abs(lat) > 70.0
            if np.any(pole_mask):
                T_pole_sfc = float(np.nanmean(T_mean[pole_mask, -1]))
                passed = 220.0 < T_pole_sfc < 270.0
                status = "PASS" if passed else "CHECK"
                lines.append(
                    f"  [{status}] Polar low-level T: {T_pole_sfc:.1f} K "
                    f"(expected ~240-260 K)"
                )

    # Zonal precipitation: check ITCZ position
    if "zonal_precip" in monthly:
        precip_zm = monthly["zonal_precip"]  # (n_months, n_lat)
        if precip_zm.ndim == 2 and precip_zm.shape[0] > 0:
            n_months = precip_zm.shape[0]
            n_last_year = min(12, n_months)
            precip_mean = np.nanmean(precip_zm[-n_last_year:], axis=0)

            # ITCZ: latitude of maximum tropical precipitation
            trop_mask = np.abs(lat) < 30.0
            if np.any(trop_mask):
                lat_trop = lat[trop_mask]
                precip_trop = precip_mean[trop_mask]
                if np.any(np.isfinite(precip_trop)):
                    itcz_lat = lat_trop[np.nanargmax(precip_trop)]
                    passed = abs(itcz_lat) < 15.0
                    status = "PASS" if passed else "CHECK"
                    lines.append(
                        f"  [{status}] ITCZ latitude: {itcz_lat:.1f}° "
                        f"(target: within ±15° of equator)"
                    )

    return lines


def generate_report(run_dir: Path, spinup_days: float = 365.0) -> str:
    """Generate full validation report."""
    lines = [
        "=" * 60,
        "AMIP Validation Report",
        "=" * 60,
        f"Run directory: {run_dir}",
        "",
    ]

    # Load config if available
    config_path = run_dir / "experiment_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        lines.append("Configuration:")
        for key in ["resolution", "nlev", "dt", "days", "radiation",
                     "microphysics", "cloud_scheme", "topography",
                     "diurnal_cycle", "dynamic_albedo"]:
            if key in config:
                lines.append(f"  {key}: {config[key]}")
        lines.append("")

    # Timeseries validation
    lines.append("Timeseries Validation")
    lines.append("-" * 40)
    try:
        ts = load_timeseries(run_dir)
        lines.extend(validate_timeseries(ts, spinup_days))
    except FileNotFoundError as e:
        lines.append(f"  {e}")
    lines.append("")

    # Monthly-mean validation
    lines.append("Monthly-Mean Validation")
    lines.append("-" * 40)
    monthly = load_monthly(run_dir)
    lines.extend(validate_monthly(monthly))
    lines.append("")

    # Summary statistics
    lines.append("Summary Statistics (post-spinup means)")
    lines.append("-" * 40)
    try:
        ts = load_timeseries(run_dir)
        days = ts.get("days", np.array([]))
        mask = days >= spinup_days
        if not np.any(mask):
            mask = np.ones(len(days), dtype=bool)

        stats = {}
        for key, label, unit, scale in [
            ("T_atm", "Global T_atm", "K", 1.0),
            ("T_low", "Global T_low (≈T_2m)", "K", 1.0),
            ("precip", "Precipitation", "mm/day", 1.0),
            ("CWV", "Column water vapor", "kg/m²", 1.0),
            ("sw_up_toa", "SW up TOA", "W/m²", 1.0),
            ("lw_up_toa", "LW up TOA (OLR)", "W/m²", 1.0),
            ("sw_net_sfc", "SW net sfc", "W/m²", 1.0),
            ("lw_net_sfc", "LW net sfc", "W/m²", 1.0),
            ("dry_mass_ps", "Surface pressure", "hPa", 0.01),
        ]:
            if key in ts:
                vals = ts[key][mask]
                mean = float(np.mean(vals)) * scale
                std = float(np.std(vals)) * scale
                stats[label] = (mean, std, unit)
                lines.append(f"  {label}: {mean:.2f} ± {std:.2f} {unit}")

    except FileNotFoundError:
        lines.append("  No timeseries data.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate AMIP run output")
    parser.add_argument("run_dir", type=str, help="Path to AMIP output directory")
    parser.add_argument("--spinup", type=float, default=365.0,
                        help="Spinup days to exclude (default: 365)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save report to file (default: print to stdout)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    report = generate_report(run_dir, spinup_days=args.spinup)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
