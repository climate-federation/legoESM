#!/usr/bin/env python3
"""ORCA2 Round 342: month-by-month ORCA2X metric card comparison against Decision-185 reference bars.

Reuses the Decision-160 ORCA2X long-run metric card (`nemo_testcase_l4_orca2x_round1_pilot_metrics.py`
and `nemo_testcase_l4_orca2x_round1_longrun.py`) without re-implementing any metric physics.
Applies the metric card month by month across the 12 calendar months of the 365-day
no-leap year (`[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]` days) for both:
  1. The NEMO reproducibility twin comparison (Twin B vs Twin A, when both are present), and
  2. The legoESM standalone K0 candidate vs NEMO Arm A comparison.

Every metric is reported next to its Decision-185 REFERENCE bar:
  - Monthly RMS SST and depth-band T (`0-100m`, `100-1000m`, `>1000m`): `0.01 K`
  - Monthly RMS SSS and depth-band S (`0-100m`, `100-1000m`, `>1000m`): `0.002 psu`
  - Monthly RMS SSH: `1 mm` (`0.001 m`)
  - Surface KE: `1%` (`0.01`) of NEMO monthly mean
  - Barotropic streamfunction psi: RMS `0.1 Sv` and spatial correlation `>= 0.999`
  - Drake Passage transport: `0.5 Sv`
  - Mixed-layer depth (MLD, Delta sigma_0 = 0.01 kg m^-3): RMS `1 m` (`1.0 m`)

All bars are REFERENCE ONLY (`bar_mode = "DECISION_185_REFERENCE_ONLY"`, `verdict = None`,
`pass_fail_assigned = False` — no pass/fail is computed or emitted).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

import jax  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean import bulk_flux_omip as bulk  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round339_common as common,
    nemo_testcase_l4_orca2_round340_standalone_gate as r340_s4,
    nemo_testcase_l4_orca2x_round1_longrun as longrun,
    nemo_testcase_l4_orca2x_round1_pilot_metrics as pilot_metrics,
)

require = common.require


class K0MLDEvaluator(pilot_metrics.MLDEvaluator):
    """Subclass of `pilot_metrics.MLDEvaluator` initialized from the standalone K0 card."""

    def __init__(self, record: Path):
        jax.config.update("jax_enable_x64", True)
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        k0_card = bulk.build_nemo_orca2_k0_ocean_card(record)
        recipe = k0_card.ocean.recipe
        model = LatLonCGridOceanModel(
            recipe.grid,
            recipe.z_coord,
            recipe.model_config,
            iwm_forcing=recipe.iwm_forcing,
        )
        self.model = model
        self.state = recipe.initial_state
        cfg = recipe.model_config
        self.g = float(cfg.constants.g)
        self.rho0 = float(cfg.constants.rho_0)
        self._bundle = jax.jit(lambda s: model._tke_step_entry_n2_bundle(s).rn2b)

STEPS_PER_DAY = pilot_metrics.STEPS_PER_DAY
FIELDS = pilot_metrics.FIELDS
NOLEAP_MONTHS: tuple[tuple[int, str, int], ...] = (
    (1, "M01_Jan", 31),
    (2, "M02_Feb", 28),
    (3, "M03_Mar", 31),
    (4, "M04_Apr", 30),
    (5, "M05_May", 31),
    (6, "M06_Jun", 30),
    (7, "M07_Jul", 31),
    (8, "M08_Aug", 31),
    (9, "M09_Sep", 30),
    (10, "M10_Oct", 31),
    (11, "M11_Nov", 30),
    (12, "M12_Dec", 31),
)

DECISION_185_REFERENCE_BARS: dict[str, dict[str, Any]] = {
    "SST": {"metric": "rms", "reference_bar": 0.01, "units": "K", "note": "DECISION 185 reference only"},
    "T_0-100m": {"metric": "rms", "reference_bar": 0.01, "units": "K", "note": "DECISION 185 reference only"},
    "T_100-1000m": {"metric": "rms", "reference_bar": 0.01, "units": "K", "note": "DECISION 185 reference only"},
    "T_>1000m": {"metric": "rms", "reference_bar": 0.01, "units": "K", "note": "DECISION 185 reference only"},
    "SSS": {"metric": "rms", "reference_bar": 0.002, "units": "psu", "note": "DECISION 185 reference only"},
    "S_0-100m": {"metric": "rms", "reference_bar": 0.002, "units": "psu", "note": "DECISION 185 reference only"},
    "S_100-1000m": {"metric": "rms", "reference_bar": 0.002, "units": "psu", "note": "DECISION 185 reference only"},
    "S_>1000m": {"metric": "rms", "reference_bar": 0.002, "units": "psu", "note": "DECISION 185 reference only"},
    "SSH": {"metric": "rms", "reference_bar": 0.001, "units": "m (1 mm)", "note": "DECISION 185 reference only"},
    "KE": {
        "metric": "relative_to_nemo_mean",
        "reference_bar_fraction": 0.01,
        "units": "m^2 s^-2 (1% of NEMO mean)",
        "note": "DECISION 185 reference only",
    },
    "PSI": {
        "metric": "rms_and_pattern_corr",
        "reference_bar_rms_sv": 0.1,
        "reference_bar_pattern_corr_min": 0.999,
        "units": "Sv / dimensionless",
        "note": "DECISION 185 reference only",
    },
    "DRAKE": {
        "metric": "abs_transport_diff_sv",
        "reference_bar": 0.5,
        "units": "Sv",
        "note": "DECISION 185 reference only",
    },
    "MLD": {"metric": "rms", "reference_bar": 1.0, "units": "m", "note": "DECISION 185 reference only"},
    "ICE_SST": {"metric": "rms", "reference_bar": 0.01, "units": "K", "note": "DECISION 185 reference only"},
    "ICE_SSS": {"metric": "rms", "reference_bar": 0.002, "units": "psu", "note": "DECISION 185 reference only"},
}


def calendar_month_day_ranges(max_days: int = 365) -> list[dict[str, Any]]:
    """Return the 12 no-leap calendar month windows intersected with `1..max_days`."""
    require(1 <= max_days <= 365, f"max_days must be in 1..365, got {max_days}")
    windows: list[dict[str, Any]] = []
    cursor = 1
    for month_idx, label, length in NOLEAP_MONTHS:
        start_day = cursor
        end_day = cursor + length - 1
        cursor = end_day + 1
        if start_day > max_days:
            windows.append({
                "month": month_idx,
                "label": label,
                "calendar_days": length,
                "start_day": start_day,
                "end_day": end_day,
                "available_days": [],
                "complete_month": False,
            })
            continue
        actual_end = min(end_day, max_days)
        days = list(range(start_day, actual_end + 1))
        windows.append({
            "month": month_idx,
            "label": label,
            "calendar_days": length,
            "start_day": start_day,
            "end_day": end_day,
            "available_days": days,
            "complete_month": len(days) == length,
        })
    return windows


def _mean_fields(states: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    require(bool(states), "cannot average empty state list")
    return {
        name: np.mean([st[name] for st in states], axis=0)
        for name in FIELDS
    }


def _attach_decision185_reference(
    monthly_summary: dict[str, Any],
    mean_field_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Attach Decision-185 reference bars next to each monthly metric (REFERENCE ONLY)."""
    out: dict[str, Any] = {}
    for key in (
        "SST", "T_0-100m", "T_100-1000m", "T_>1000m",
        "SSS", "S_0-100m", "S_100-1000m", "S_>1000m",
        "SSH", "MLD", "ICE_SST", "ICE_SSS",
    ):
        if key not in monthly_summary:
            continue
        ref = DECISION_185_REFERENCE_BARS[key]
        out[key] = {
            "daily_mean_bias": monthly_summary[key]["bias"],
            "daily_mean_rms": monthly_summary[key]["rms"],
            "monthly_mean_field_bias": mean_field_metrics[key]["bias"],
            "monthly_mean_field_rms": mean_field_metrics[key]["rms"],
            "decision185_reference_bar_rms": ref["reference_bar"],
            "units": ref["units"],
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }

    if "KE" in monthly_summary:
        ke_m = mean_field_metrics["KE"]
        nemo_ke = float(ke_m["nemo_mean"])
        cand_ke = float(ke_m["cand_mean"])
        rel_bias = abs(cand_ke - nemo_ke) / nemo_ke if nemo_ke > 0.0 else 0.0
        rel_rms = float(ke_m["rms"]) / nemo_ke if nemo_ke > 0.0 else 0.0
        out["KE"] = {
            "nemo_mean": nemo_ke,
            "cand_mean": cand_ke,
            "daily_mean_bias": monthly_summary["KE"]["bias"],
            "daily_mean_rms": monthly_summary["KE"]["rms"],
            "monthly_mean_field_bias": ke_m["bias"],
            "monthly_mean_field_rms": ke_m["rms"],
            "relative_mean_diff_fraction": rel_bias,
            "relative_rms_fraction": rel_rms,
            "decision185_reference_bar_fraction": 0.01,
            "decision185_reference_bar_abs": 0.01 * nemo_ke,
            "units": "m^2 s^-2",
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }

    if "PSI" in monthly_summary:
        psi_m = mean_field_metrics["PSI"]
        out["PSI"] = {
            "daily_mean_bias_sv": monthly_summary["PSI"]["bias"],
            "daily_mean_rms_sv": monthly_summary["PSI"]["rms"],
            "daily_mean_pattern_corr": monthly_summary["PSI"]["pattern_corr"],
            "monthly_mean_field_bias_sv": psi_m["bias"],
            "monthly_mean_field_rms_sv": psi_m["rms"],
            "monthly_mean_field_pattern_corr": psi_m["pattern_corr"],
            "decision185_reference_bar_rms_sv": 0.1,
            "decision185_reference_bar_pattern_corr_min": 0.999,
            "units": "Sv / dimensionless",
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }

    if "DRAKE" in monthly_summary:
        dr = monthly_summary["DRAKE"]
        diff_sv = float(dr["cand_mean_sv"] - dr["nemo_mean_sv"])
        out["DRAKE"] = {
            "nemo_mean_sv": dr["nemo_mean_sv"],
            "cand_mean_sv": dr["cand_mean_sv"],
            "nemo_std_sv": dr["nemo_std_sv"],
            "cand_std_sv": dr["cand_std_sv"],
            "mean_diff_sv": diff_sv,
            "abs_mean_diff_sv": abs(diff_sv),
            "decision185_reference_bar_sv": 0.5,
            "units": "Sv",
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }

    return out


def evaluate_month_by_month(
    *,
    nemo_root: Path,
    candidate_root: Path,
    days: int,
    candidate_is_nemo_restart: bool = False,
    ice_cover_path: Path = r340_s4.read_decision171_cover.__defaults__[0]
    if r340_s4.read_decision171_cover.__defaults__
    else Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
        "ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
    ),
    skip_mld: bool = False,
) -> dict[str, Any]:
    """Evaluate the ORCA2X metric card daily and month by month next to Decision-185 reference bars."""
    require(1 <= days <= 365, f"days must be in 1..365, got {days}")
    geo = pilot_metrics.static_geometry(nemo_root)
    mld = None if skip_mld else K0MLDEvaluator(nemo_root)
    cover = r340_s4.read_decision171_cover(ice_cover_path)

    read_cand = (
        pilot_metrics.read_nemo_day
        if candidate_is_nemo_restart
        else pilot_metrics.read_candidate_day
    )

    cand_day1 = read_cand(candidate_root, 1)
    nemo_day1 = pilot_metrics.read_nemo_day(nemo_root, 1)
    planted = pilot_metrics.planted_controls(cand_day1, nemo_day1, geo, cover)

    daily_rows: list[dict[str, Any]] = []
    daily_maps: list[dict[str, np.ndarray]] = []
    cand_states_by_day: dict[int, dict[str, np.ndarray]] = {}
    nemo_states_by_day: dict[int, dict[str, np.ndarray]] = {}

    for day in range(1, days + 1):
        cand = cand_day1 if day == 1 else read_cand(candidate_root, day)
        nemo = nemo_day1 if day == 1 else pilot_metrics.read_nemo_day(nemo_root, day)
        row, mp = pilot_metrics.day_metrics(cand, nemo, geo, mld, cover)
        daily_rows.append({"day": day, "kt": day * STEPS_PER_DAY, **row})
        daily_maps.append(mp)
        cand_states_by_day[day] = cand
        nemo_states_by_day[day] = nemo

    months_out: list[dict[str, Any]] = []
    for window in calendar_month_day_ranges(days):
        avail = window["available_days"]
        if not avail:
            months_out.append({
                **window,
                "status": "PENDING_ACQUISITION",
                "bar_mode": "DECISION_185_REFERENCE_ONLY",
                "verdict": None,
                "metrics": None,
            })
            continue
        m_rows = [daily_rows[d - 1] for d in avail]
        m_maps = [daily_maps[d - 1] for d in avail]
        m_summary = pilot_metrics.monthly(m_rows, m_maps, geo)
        cand_mean_state = _mean_fields([cand_states_by_day[d] for d in avail])
        nemo_mean_state = _mean_fields([nemo_states_by_day[d] for d in avail])
        mean_field_row, _ = pilot_metrics.day_metrics(
            cand_mean_state, nemo_mean_state, geo, mld, cover
        )
        metrics_with_bars = _attach_decision185_reference(m_summary, mean_field_row)
        months_out.append({
            **window,
            "status": "EVALUATED_COMPLETE_MONTH" if window["complete_month"] else "EVALUATED_PARTIAL_MONTH",
            "bar_mode": "DECISION_185_REFERENCE_ONLY",
            "verdict": None,
            "metrics": metrics_with_bars,
        })

    overall_summary = pilot_metrics.monthly(daily_rows, daily_maps, geo)
    overall_cand_mean = _mean_fields([cand_states_by_day[d] for d in range(1, days + 1)])
    overall_nemo_mean = _mean_fields([nemo_states_by_day[d] for d in range(1, days + 1)])
    overall_mean_row, _ = pilot_metrics.day_metrics(
        overall_cand_mean, overall_nemo_mean, geo, mld, cover
    )
    overall_with_bars = _attach_decision185_reference(overall_summary, overall_mean_row)

    return {
        "format": "orca2-r342-s4-year-metrics-v1",
        "round": "orca2_r342",
        "bias_convention": "candidate - NEMO",
        "bar_mode": "DECISION_185_REFERENCE_ONLY",
        "verdict": None,
        "pass_fail_assigned": False,
        "days_evaluated": days,
        "nemo_root": str(nemo_root),
        "candidate_root": str(candidate_root),
        "candidate_is_nemo_restart": candidate_is_nemo_restart,
        "decision185_reference_bars": DECISION_185_REFERENCE_BARS,
        "planted_controls": planted,
        "nlb10": int(geo["nlb10"]),
        "mld_constants": None if mld is None else {
            "g": mld.g,
            "rho0": mld.rho0,
            "rho_c": pilot_metrics.ZDFMXL_RHO_C,
        },
        "overall_window_metrics": overall_with_bars,
        "months": months_out,
        "daily": daily_rows,
        "worktree": worktree_stamp(allow_dirty=True),
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise Markdown table of month-by-month metrics next to Decision-185 reference bars."""
    lines = [
        "# ORCA2 Round 342 — Standalone K0 Month-by-Month ORCA2X Metric Card Report",
        "",
        f"- **Bar mode:** `{report['bar_mode']}` (`verdict = {report['verdict']}`, `pass_fail_assigned = {report['pass_fail_assigned']}`)",
        f"- **Bias convention:** `{report['bias_convention']}`",
        f"- **NEMO root:** `{report['nemo_root']}`",
        f"- **Candidate root:** `{report['candidate_root']}`",
        f"- **Days evaluated:** `{report['days_evaluated']}`",
        "",
        "## Decision-185 Reference Bars (Reference Only — No Pass/Fail)",
        "",
        "| Metric | Statistic | Decision-185 Reference Bar | Units |",
        "|---|---|---:|---|",
        "| `SST` / `T_0-100m` / `T_100-1000m` / `T_>1000m` | Monthly RMS | `0.01` | `K` |",
        "| `SSS` / `S_0-100m` / `S_100-1000m` / `S_>1000m` | Monthly RMS | `0.002` | `psu` |",
        "| `SSH` | Monthly RMS | `0.001` (`1 mm`) | `m` |",
        "| `KE` (surface) | Relative diff / RMS vs NEMO mean | `0.01` (`1%`) | fraction of NEMO mean |",
        "| `PSI` (barotropic streamfunction) | Monthly RMS / spatial corr | `0.1 Sv` / `>= 0.999` | `Sv` / dimensionless |",
        "| `DRAKE` (Drake Passage transport) | `|cand - NEMO|` monthly mean | `0.5` | `Sv` |",
        "| `MLD` (`Delta sigma_0 = 0.01 kg m^-3`) | Monthly RMS | `1.0` | `m` |",
        "",
        "## Month-by-Month Summary (Evaluated & Pending Months 1–12)",
        "",
        "| Month | Days | Status | SST RMS (`0.01 K`) | SSS RMS (`0.002 psu`) | SSH RMS (`0.001 m`) | KE rel RMS (`0.01`) | PSI RMS (`0.1 Sv`) | PSI corr (`0.999`) | Drake `|diff|` (`0.5 Sv`) | MLD RMS (`1.0 m`) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in report["months"]:
        day_span = f"{m['start_day']}..{m['end_day']}"
        if m["metrics"] is None:
            lines.append(
                f"| `{m['label']}` | `{day_span}` | `{m['status']}` | — | — | — | — | — | — | — | — |"
            )
            continue
        mt = m["metrics"]
        mld_str = (
            f"{mt['MLD']['monthly_mean_field_rms']:.4e}"
            if "MLD" in mt
            else "skipped"
        )
        lines.append(
            f"| `{m['label']}` | `{day_span}` (`{len(m['available_days'])}d`) | `{m['status']}` "
            f"| `{mt['SST']['monthly_mean_field_rms']:.4e}` "
            f"| `{mt['SSS']['monthly_mean_field_rms']:.4e}` "
            f"| `{mt['SSH']['monthly_mean_field_rms']:.4e}` "
            f"| `{mt['KE']['relative_rms_fraction']:.4e}` "
            f"| `{mt['PSI']['monthly_mean_field_rms_sv']:.4e}` "
            f"| `{mt['PSI']['monthly_mean_field_pattern_corr']:.6f}` "
            f"| `{mt['DRAKE']['abs_mean_diff_sv']:.4e}` "
            f"| `{mld_str}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--nemo-twin-b-root", type=Path, default=None,
                        help="Optional NEMO Twin B directory to score NEMO Twin B vs Twin A alongside candidate vs NEMO")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--skip-mld", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cand_report = evaluate_month_by_month(
        nemo_root=args.nemo_root,
        candidate_root=args.candidate_root,
        days=args.days,
        candidate_is_nemo_restart=False,
        skip_mld=args.skip_mld,
    )
    (args.output_dir / "round342_s4_candidate_vs_nemo_metrics.json").write_text(
        json.dumps(cand_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "round342_s4_candidate_vs_nemo_metrics.md").write_text(
        format_markdown_report(cand_report),
        encoding="utf-8",
    )

    if args.nemo_twin_b_root is not None:
        twin_report = evaluate_month_by_month(
            nemo_root=args.nemo_root,
            candidate_root=args.nemo_twin_b_root,
            days=args.days,
            candidate_is_nemo_restart=True,
            skip_mld=args.skip_mld,
        )
        (args.output_dir / "round342_s4_nemo_twin_b_vs_a_metrics.json").write_text(
            json.dumps(twin_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "round342_s4_nemo_twin_b_vs_a_metrics.md").write_text(
            format_markdown_report(twin_report),
            encoding="utf-8",
        )

    print(json.dumps({
        "status": "EVALUATED_REFERENCE_ONLY",
        "bar_mode": cand_report["bar_mode"],
        "verdict": cand_report["verdict"],
        "days_evaluated": cand_report["days_evaluated"],
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
