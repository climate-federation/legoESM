#!/usr/bin/env python3
"""ORCA2 Round 342: month-by-month ORCA2X metric card comparison against Decision-185 reference bars.

Reuses the Decision-160 ORCA2X long-run metric card (`nemo_testcase_l4_orca2x_round1_pilot_metrics.py`
and `nemo_testcase_l4_orca2x_round1_longrun.py`) verbatim without re-implementing any metric physics
and without averaging state fields before computing nonlinear metrics (`KE`, `PSI`, `MLD`).

Sampling schedule (Decision 160 card, identical for both arms):
  - Saved states are the instantaneous completed end-of-day states every `nn_stock = 8` steps
    (`kt = 8, 16, 24, ..., 2920` for Days `d = 1..365` at `rn_Dt = 10800` s):
      * NEMO arm (`orca2_k0_year1_{a,b}_np2`): two-rank daily restarts
        `ORCA2_{8*d:08d}_restart_{0000,0001}.nc` holding `tn, sn, un, vn, sshn` (`Kbb` after Stage-3 swap).
      * legoESM arm: daily snapshots `candidate_step{8*d:08d}.npz` holding `T, S, u, v, ssh`.
  - On each saved daily state `d`, `pilot_metrics.day_metrics(cand, nemo, geo, mld, cover)`
    evaluates all linear and nonlinear metrics (`SST`, `SSS`, `SSH`, depth-band `T`/`S`,
    `KE`, `PSI`, `DRAKE`, `MLD`, `ICE_SST`, `ICE_SSS`) and daily difference maps.
  - Within each of the 12 no-leap calendar months (`[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]`
    days), `pilot_metrics.monthly(m_rows, m_maps, geo)` averages the per-saved-state metrics
    across the saved days in that month, identically for:
      1. The NEMO reproducibility twin comparison (`Twin B vs Twin A`), and
      2. The legoESM standalone K0 candidate vs NEMO Arm A comparison.

Every monthly metric is reported next to its Decision-185 REFERENCE bar:
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
SAMPLING_DESCRIPTION = (
    "Instantaneous completed end-of-day states saved every 8 steps "
    "(kt = 8, 16, ..., 2920 for days d = 1..365 at rn_Dt = 10800 s): "
    "NEMO two-rank restarts ORCA2_{8*d:08d}_restart_{0000,0001}.nc (nn_stock = 8, Kbb after Stage-3 swap) "
    "and legoESM snapshots candidate_step{8*d:08d}.npz (save_stride = 8). "
    "Each metric is evaluated on each saved daily state via pilot_metrics.day_metrics and then averaged "
    "across the saved daily states in each month via pilot_metrics.monthly (never on state-averaged fields)."
)
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
                "saved_kts": [],
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
            "saved_kts": [d * STEPS_PER_DAY for d in days],
            "complete_month": len(days) == length,
        })
    return windows


def _augment_day_single_arm_stats(
    row: dict[str, Any],
    cand: dict[str, np.ndarray],
    nemo: dict[str, np.ndarray],
    geo: dict[str, np.ndarray],
    cover: np.ndarray,
    mp: dict[str, np.ndarray],
) -> None:
    """Attach per-day single-arm NEMO and candidate values using Decision-160 reduction weights."""
    area = geo["area_t"]
    w_surf = area * geo["tmask"][..., 0]
    w_surf_sum = float(w_surf.sum())
    for key, name in (("SST", "T"), ("SSS", "S")):
        row[key]["nemo_mean"] = float((w_surf * nemo[name][..., 0]).sum() / w_surf_sum)
        row[key]["cand_mean"] = float((w_surf * cand[name][..., 0]).sum() / w_surf_sum)

    n_ssh = pilot_metrics.bias_rms(nemo["ssh"], area)
    c_ssh = pilot_metrics.bias_rms(cand["ssh"], area)
    row["SSH"]["nemo_mean"] = n_ssh["bias"]
    row["SSH"]["cand_mean"] = c_ssh["bias"]
    row["SSH"]["nemo_rms"] = n_ssh["rms"]
    row["SSH"]["cand_rms"] = c_ssh["rms"]

    live_n = pilot_metrics.live_scales(geo, nemo["ssh"])
    vol = (
        area[..., None]
        * geo["e3t_0"]
        * (1.0 + live_n["r3t"][..., None] * geo["tmask"])
        * geo["tmask"]
    )
    fracs = pilot_metrics.band_fractions(geo, live_n["r3t"])
    for var in ("T", "S"):
        for bname, frac in zip(pilot_metrics.BAND_NAMES, fracs):
            w_b = vol * frac
            w_b_sum = float(w_b.sum())
            row[f"{var}_{bname}"]["nemo_mean"] = float((w_b * nemo[var]).sum() / w_b_sum)
            row[f"{var}_{bname}"]["cand_mean"] = float((w_b * cand[var]).sum() / w_b_sum)

    psi_n = mp["PSI_nemo"]
    psi_c = psi_n + mp["PSI"]
    w_psi = geo["psi_weight"]
    n_psi = pilot_metrics.bias_rms(psi_n, w_psi)
    c_psi = pilot_metrics.bias_rms(psi_c, w_psi)
    row["PSI"]["nemo_mean_sv"] = n_psi["bias"]
    row["PSI"]["cand_mean_sv"] = c_psi["bias"]
    row["PSI"]["nemo_rms_sv"] = n_psi["rms"]
    row["PSI"]["cand_rms_sv"] = c_psi["rms"]

    drake_diff = float(row["DRAKE"]["cand_sv"] - row["DRAKE"]["nemo_sv"])
    row["DRAKE"]["diff_sv"] = drake_diff
    row["DRAKE"]["abs_diff_sv"] = abs(drake_diff)

    wi = area * (cover > 0.0) * geo["tmask"][..., 0]
    wi_sum = float(wi.sum())
    row["ICE_SST"]["nemo_mean"] = float((wi * nemo["T"][..., 0]).sum() / wi_sum)
    row["ICE_SST"]["cand_mean"] = float((wi * cand["T"][..., 0]).sum() / wi_sum)
    row["ICE_SSS"]["nemo_mean"] = float((wi * nemo["S"][..., 0]).sum() / wi_sum)
    row["ICE_SSS"]["cand_mean"] = float((wi * cand["S"][..., 0]).sum() / wi_sum)


def _attach_decision185_reference(
    monthly_summary: dict[str, Any],
    daily_rows_in_window: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach Decision-185 reference bars to `pilot_metrics.monthly` outputs (REFERENCE ONLY).

    Every metric comes from `pilot_metrics.day_metrics` evaluated on each saved daily state
    and averaged over the window by `pilot_metrics.monthly` — never from state-averaged fields.
    """
    require(bool(daily_rows_in_window), "cannot attach reference bars to empty daily row window")
    out: dict[str, Any] = {}
    for key in (
        "SST", "T_0-100m", "T_100-1000m", "T_>1000m",
        "SSS", "S_0-100m", "S_100-1000m", "S_>1000m",
        "SSH", "MLD", "ICE_SST", "ICE_SSS",
    ):
        if key not in monthly_summary:
            continue
        ref = DECISION_185_REFERENCE_BARS[key]
        entry: dict[str, Any] = {
            "bias": float(monthly_summary[key]["bias"]),
            "rms": float(monthly_summary[key]["rms"]),
            "daily_mean_bias": float(monthly_summary[key]["bias"]),
            "daily_mean_rms": float(monthly_summary[key]["rms"]),
            "decision185_reference_bar_rms": ref["reference_bar"],
            "units": ref["units"],
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }
        map_key = f"{key}_monthly_mean_map"
        if map_key in monthly_summary:
            entry["monthly_mean_difference_map"] = {
                "bias": float(monthly_summary[map_key]["bias"]),
                "rms": float(monthly_summary[map_key]["rms"]),
            }
        if key in daily_rows_in_window[0] and "nemo_mean" in daily_rows_in_window[0][key]:
            entry["nemo_mean"] = float(np.mean([r[key]["nemo_mean"] for r in daily_rows_in_window]))
            entry["cand_mean"] = float(np.mean([r[key]["cand_mean"] for r in daily_rows_in_window]))
        if key in daily_rows_in_window[0] and "nemo_rms" in daily_rows_in_window[0][key]:
            entry["nemo_rms"] = float(np.mean([r[key]["nemo_rms"] for r in daily_rows_in_window]))
            entry["cand_rms"] = float(np.mean([r[key]["cand_rms"] for r in daily_rows_in_window]))
        out[key] = entry

    if "KE" in monthly_summary:
        nemo_ke = float(np.mean([r["KE"]["nemo_mean"] for r in daily_rows_in_window]))
        cand_ke = float(np.mean([r["KE"]["cand_mean"] for r in daily_rows_in_window]))
        ke_bias = float(monthly_summary["KE"]["bias"])
        ke_rms = float(monthly_summary["KE"]["rms"])
        rel_bias = abs(cand_ke - nemo_ke) / nemo_ke if nemo_ke > 0.0 else 0.0
        rel_rms = ke_rms / nemo_ke if nemo_ke > 0.0 else 0.0
        out["KE"] = {
            "nemo_mean": nemo_ke,
            "cand_mean": cand_ke,
            "bias": ke_bias,
            "rms": ke_rms,
            "daily_mean_bias": ke_bias,
            "daily_mean_rms": ke_rms,
            "relative_mean_diff_fraction": rel_bias,
            "relative_rms_fraction": rel_rms,
            "decision185_reference_bar_fraction": 0.01,
            "decision185_reference_bar_abs": 0.01 * nemo_ke,
            "units": "m^2 s^-2",
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }

    if "PSI" in monthly_summary:
        psi_entry: dict[str, Any] = {
            "bias_sv": float(monthly_summary["PSI"]["bias"]),
            "rms_sv": float(monthly_summary["PSI"]["rms"]),
            "pattern_corr": float(monthly_summary["PSI"]["pattern_corr"]),
            "daily_mean_bias_sv": float(monthly_summary["PSI"]["bias"]),
            "daily_mean_rms_sv": float(monthly_summary["PSI"]["rms"]),
            "daily_mean_pattern_corr": float(monthly_summary["PSI"]["pattern_corr"]),
            "decision185_reference_bar_rms_sv": 0.1,
            "decision185_reference_bar_pattern_corr_min": 0.999,
            "units": "Sv / dimensionless",
            "bar_mode": "REFERENCE_ONLY",
            "verdict": None,
        }
        if "PSI" in daily_rows_in_window[0] and "nemo_rms_sv" in daily_rows_in_window[0]["PSI"]:
            psi_entry["nemo_mean_sv"] = float(np.mean([r["PSI"]["nemo_mean_sv"] for r in daily_rows_in_window]))
            psi_entry["cand_mean_sv"] = float(np.mean([r["PSI"]["cand_mean_sv"] for r in daily_rows_in_window]))
            psi_entry["nemo_rms_sv"] = float(np.mean([r["PSI"]["nemo_rms_sv"] for r in daily_rows_in_window]))
            psi_entry["cand_rms_sv"] = float(np.mean([r["PSI"]["cand_rms_sv"] for r in daily_rows_in_window]))
        if "PSI_monthly_mean_map" in monthly_summary:
            psi_entry["monthly_mean_difference_map"] = {
                "bias_sv": float(monthly_summary["PSI_monthly_mean_map"]["bias"]),
                "rms_sv": float(monthly_summary["PSI_monthly_mean_map"]["rms"]),
            }
        out["PSI"] = psi_entry

    if "DRAKE" in monthly_summary:
        dr = monthly_summary["DRAKE"]
        diff_sv = float(dr["cand_mean_sv"] - dr["nemo_mean_sv"])
        daily_diffs = [float(r["DRAKE"]["cand_sv"] - r["DRAKE"]["nemo_sv"]) for r in daily_rows_in_window]
        out["DRAKE"] = {
            "nemo_mean_sv": float(dr["nemo_mean_sv"]),
            "cand_mean_sv": float(dr["cand_mean_sv"]),
            "nemo_std_sv": float(dr["nemo_std_sv"]),
            "cand_std_sv": float(dr["cand_std_sv"]),
            "mean_diff_sv": diff_sv,
            "abs_mean_diff_sv": abs(diff_sv),
            "daily_mean_abs_diff_sv": float(np.mean(np.abs(daily_diffs))),
            "daily_rms_diff_sv": float(np.sqrt(np.mean(np.square(daily_diffs)))),
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
    geo: dict[str, np.ndarray] | None = None,
    mld: K0MLDEvaluator | None = None,
    cover: np.ndarray | None = None,
    plots_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the ORCA2X metric card daily and month by month next to Decision-185 reference bars."""
    require(1 <= days <= 365, f"days must be in 1..365, got {days}")
    if geo is None:
        geo = pilot_metrics.static_geometry(nemo_root)
    if not skip_mld and mld is None:
        mld = K0MLDEvaluator(nemo_root)
    if cover is None:
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

    for day in range(1, days + 1):
        cand = cand_day1 if day == 1 else read_cand(candidate_root, day)
        nemo = nemo_day1 if day == 1 else pilot_metrics.read_nemo_day(nemo_root, day)
        row, mp = pilot_metrics.day_metrics(cand, nemo, geo, mld, cover)
        _augment_day_single_arm_stats(row, cand, nemo, geo, cover, mp)
        daily_rows.append({"day": day, "kt": day * STEPS_PER_DAY, **row})
        daily_maps.append(mp)

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
        metrics_with_bars = _attach_decision185_reference(m_summary, m_rows)
        months_out.append({
            **window,
            "status": "EVALUATED_COMPLETE_MONTH" if window["complete_month"] else "EVALUATED_PARTIAL_MONTH",
            "bar_mode": "DECISION_185_REFERENCE_ONLY",
            "verdict": None,
            "pilot_monthly": m_summary,
            "metrics": metrics_with_bars,
        })

    overall_summary = pilot_metrics.monthly(daily_rows, daily_maps, geo)
    overall_with_bars = _attach_decision185_reference(overall_summary, daily_rows)

    plot_files: list[str] = []
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_files = pilot_metrics.plots(daily_rows, daily_maps, geo, plots_dir)

    return {
        "format": "orca2-r342-s4-year-metrics-v2",
        "round": "orca2_r342",
        "bias_convention": "candidate - NEMO",
        "sampling": SAMPLING_DESCRIPTION,
        "steps_per_day": STEPS_PER_DAY,
        "saved_kts_evaluated": [d * STEPS_PER_DAY for d in range(1, days + 1)],
        "state_averaging_before_nonlinear_metrics": False,
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
        "overall_pilot_monthly": overall_summary,
        "overall_window_metrics": overall_with_bars,
        "months": months_out,
        "daily": daily_rows,
        "plots": plot_files,
        "worktree": worktree_stamp(allow_dirty=False),
    }


def write_daily_sst_rms_curve_figure_and_data(
    cand_report: dict[str, Any],
    twin_report: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Write the day-by-day SST RMS difference curve figure and JSON data."""
    import hashlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    daily_c = cand_report["daily"]
    twin_by_day = (
        {r["day"]: r for r in twin_report["daily"]}
        if twin_report is not None
        else {}
    )
    curve_rows: list[dict[str, Any]] = []
    for r in daily_c:
        d = int(r["day"])
        tr = twin_by_day.get(d)
        curve_rows.append({
            "day": d,
            "kt": int(r["kt"]),
            "nemo_mean_sst_degc": float(r["SST"]["nemo_mean"]),
            "legoesm_mean_sst_degc": float(r["SST"]["cand_mean"]),
            "sst_bias_k": float(r["SST"]["bias"]),
            "sst_rms_k": float(r["SST"]["rms"]),
            "twin_b_vs_a_sst_rms_k": None if tr is None else float(tr["SST"]["rms"]),
            "decision185_reference_bar_rms_k": 0.01,
        })

    days = [row["day"] for row in curve_rows]
    rms_vals = [row["sst_rms_k"] for row in curve_rows]
    bias_vals = [row["sst_bias_k"] for row in curve_rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(days, rms_vals, color="tab:blue", lw=1.5, label="legoESM vs NEMO SST RMS [K]")
    axes[0].plot(days, [abs(b) for b in bias_vals], color="tab:cyan", lw=1.0, ls="--", label="|legoESM - NEMO SST bias| [K]")
    if twin_report is not None:
        axes[0].plot(
            days,
            [row["twin_b_vs_a_sst_rms_k"] for row in curve_rows],
            color="tab:green",
            lw=1.0,
            ls=":",
            label="NEMO Twin B vs Twin A SST RMS [K]",
        )
    axes[0].axhline(0.01, color="tab:red", lw=1.0, ls="--", label="Decision-185 ref bar (0.01 K, reference only)")
    axes[0].set_ylabel("SST difference [K] (linear)")
    axes[0].set_title("ORCA2 Round 342 Standalone K0 — Day-by-Day SST RMS Difference Curve")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    floor = 1e-16
    rms_log = [max(v, floor) for v in rms_vals]
    axes[1].semilogy(days, rms_log, color="tab:blue", lw=1.5, label="max(SST RMS, 1e-16) [K]")
    axes[1].axhline(0.01, color="tab:red", lw=1.0, ls="--", label="Decision-185 ref bar (0.01 K)")
    axes[1].set_xlabel("Day (kt = 8 * day)")
    axes[1].set_ylabel("SST RMS [K] (log scale)")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fig_path = output_dir / "sst_rms_daily_curve.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    nonzero_days = [row for row in curve_rows if row["sst_rms_k"] > 0.0]
    above_1e12 = [row for row in curve_rows if row["sst_rms_k"] >= 1e-12]
    above_1e6 = [row for row in curve_rows if row["sst_rms_k"] >= 1e-6]
    above_1e3 = [row for row in curve_rows if row["sst_rms_k"] >= 1e-3]
    above_ref = [row for row in curve_rows if row["sst_rms_k"] >= 0.01]

    fig_sha = hashlib.sha256(fig_path.read_bytes()).hexdigest()
    payload = {
        "format": "orca2-r342-daily-sst-rms-curve-v1",
        "bar_mode": "DECISION_185_REFERENCE_ONLY",
        "verdict": None,
        "days_evaluated": len(curve_rows),
        "figure_path": str(fig_path),
        "figure_sha256": fig_sha,
        "separation_milestones": {
            "last_exact_zero_day": max((row["day"] for row in curve_rows if row["sst_rms_k"] == 0.0), default=None),
            "first_nonzero_day": nonzero_days[0]["day"] if nonzero_days else None,
            "first_nonzero_kt": nonzero_days[0]["kt"] if nonzero_days else None,
            "first_nonzero_sst_rms_k": nonzero_days[0]["sst_rms_k"] if nonzero_days else None,
            "first_day_ge_1e-12_k": above_1e12[0]["day"] if above_1e12 else None,
            "first_day_ge_1e-6_k": above_1e6[0]["day"] if above_1e6 else None,
            "first_day_ge_1e-3_k": above_1e3[0]["day"] if above_1e3 else None,
            "first_day_ge_0.01_k_ref_bar": above_ref[0]["day"] if above_ref else None,
            "max_sst_rms_k": max(rms_vals) if rms_vals else None,
            "max_sst_rms_day": curve_rows[int(np.argmax(rms_vals))]["day"] if rms_vals else None,
            "day365_sst_rms_k": curve_rows[-1]["sst_rms_k"] if curve_rows else None,
        },
        "daily_curve": curve_rows,
    }
    curve_json_path = output_dir / "round342_s4_daily_sst_rms_curve.json"
    curve_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


METRIC_TABLE_ORDER: tuple[str, ...] = (
    "SST",
    "T_0-100m",
    "T_100-1000m",
    "T_>1000m",
    "SSS",
    "S_0-100m",
    "S_100-1000m",
    "S_>1000m",
    "SSH",
    "KE",
    "PSI",
    "DRAKE",
    "MLD",
    "ICE_SST",
    "ICE_SSS",
)


def _format_metric_row(
    month_label: str,
    metric_key: str,
    mt: dict[str, Any],
    twin_mt: dict[str, Any] | None,
) -> str:
    """Format one row of the month-by-month, metric-by-metric table."""
    entry = mt[metric_key]
    t_entry = twin_mt[metric_key] if (twin_mt is not None and metric_key in twin_mt) else None

    if metric_key == "KE":
        nemo_str = f"`{entry['nemo_mean']:.6e} m^2 s^-2` (mean)"
        cand_str = f"`{entry['cand_mean']:.6e} m^2 s^-2` (mean)"
        diff_str = (
            f"bias `{entry['daily_mean_bias']:+.4e}`, RMS `{entry['daily_mean_rms']:.4e} m^2 s^-2` "
            f"(rel RMS `{entry['relative_rms_fraction']:.4e}`, rel `|mean diff|` `{entry['relative_mean_diff_fraction']:.4e}`)"
        )
        twin_str = (
            f"rel RMS `{t_entry['relative_rms_fraction']:.4e}` (RMS `{t_entry['daily_mean_rms']:.4e}`)"
            if t_entry is not None
            else "—"
        )
        bar_str = f"`0.01` (`1%` of NEMO mean = `{entry['decision185_reference_bar_abs']:.4e} m^2 s^-2`; REFERENCE ONLY)"
    elif metric_key == "PSI":
        nemo_str = (
            f"`{entry['nemo_rms_sv']:.4f} Sv` (spatial RMS; mean `{entry['nemo_mean_sv']:+.4f} Sv`)"
            if "nemo_rms_sv" in entry
            else "—"
        )
        cand_str = (
            f"`{entry['cand_rms_sv']:.4f} Sv` (spatial RMS; mean `{entry['cand_mean_sv']:+.4f} Sv`)"
            if "cand_rms_sv" in entry
            else "—"
        )
        diff_str = (
            f"bias `{entry['daily_mean_bias_sv']:+.4e} Sv`, RMS `{entry['daily_mean_rms_sv']:.4e} Sv`, "
            f"corr `{entry['daily_mean_pattern_corr']:.6f}`"
        )
        twin_str = (
            f"RMS `{t_entry['daily_mean_rms_sv']:.4e} Sv`, corr `{t_entry['daily_mean_pattern_corr']:.6f}`"
            if t_entry is not None
            else "—"
        )
        bar_str = "`0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY)"
    elif metric_key == "DRAKE":
        nemo_str = f"`{entry['nemo_mean_sv']:.6f} Sv` (std `{entry['nemo_std_sv']:.4f} Sv`)"
        cand_str = f"`{entry['cand_mean_sv']:.6f} Sv` (std `{entry['cand_std_sv']:.4f} Sv`)"
        diff_str = (
            f"`cand - NEMO = {entry['mean_diff_sv']:+.4e} Sv`, "
            f"`|mean diff| = {entry['abs_mean_diff_sv']:.4e} Sv`"
        )
        twin_str = (
            f"`|mean diff| = {t_entry['abs_mean_diff_sv']:.4e} Sv`"
            if t_entry is not None
            else "—"
        )
        bar_str = "`0.5 Sv` (REFERENCE ONLY)"
    elif metric_key == "SSH":
        nemo_str = (
            f"`{entry['nemo_mean']:+.6e} m` (mean; spatial RMS `{entry['nemo_rms']:.6f} m`)"
            if "nemo_mean" in entry and "nemo_rms" in entry
            else "—"
        )
        cand_str = (
            f"`{entry['cand_mean']:+.6e} m` (mean; spatial RMS `{entry['cand_rms']:.6f} m`)"
            if "cand_mean" in entry and "cand_rms" in entry
            else "—"
        )
        diff_str = f"bias `{entry['daily_mean_bias']:+.4e} m`, RMS `{entry['daily_mean_rms']:.4e} m`"
        twin_str = (
            f"RMS `{t_entry['daily_mean_rms']:.4e} m`"
            if t_entry is not None
            else "—"
        )
        bar_str = "`0.001 m` (`1 mm` RMS; REFERENCE ONLY)"
    else:
        units = entry["units"]
        mean_units = "degC" if units == "K" else units
        nemo_str = (
            f"`{entry['nemo_mean']:.6f} {mean_units}` (mean)"
            if "nemo_mean" in entry
            else "—"
        )
        cand_str = (
            f"`{entry['cand_mean']:.6f} {mean_units}` (mean)"
            if "cand_mean" in entry
            else "—"
        )
        diff_str = f"bias `{entry['daily_mean_bias']:+.4e} {units}`, RMS `{entry['daily_mean_rms']:.4e} {units}`"
        twin_str = (
            f"RMS `{t_entry['daily_mean_rms']:.4e} {units}`"
            if t_entry is not None
            else "—"
        )
        bar_str = f"`{entry['decision185_reference_bar_rms']} {units}` RMS (REFERENCE ONLY)"

    return f"| `{month_label}` | `{metric_key}` | {nemo_str} | {cand_str} | {diff_str} | {twin_str} | {bar_str} |"


def format_markdown_report(
    report: dict[str, Any],
    twin_report: dict[str, Any] | None = None,
) -> str:
    """Render a concise Markdown table of month-by-month metrics next to Decision-185 reference bars."""
    lines = [
        "# ORCA2 Round 342 — Standalone K0 Month-by-Month ORCA2X Metric Card Report",
        "",
        f"- **Bar mode:** `{report['bar_mode']}` (`verdict = {report['verdict']}`, `pass_fail_assigned = {report['pass_fail_assigned']}`)",
        f"- **Bias convention:** `{report['bias_convention']}`",
        f"- **Sampling:** {report['sampling']}",
        f"- **State averaging before nonlinear metrics:** `{report['state_averaging_before_nonlinear_metrics']}`",
        f"- **NEMO root:** `{report['nemo_root']}`",
        f"- **Candidate root:** `{report['candidate_root']}`",
        f"- **Days evaluated:** `{report['days_evaluated']}` (`kt = {report['saved_kts_evaluated'][0]}..{report['saved_kts_evaluated'][-1]}`)",
        "",
        "## Decision-185 Reference Bars (Reference Only — No Pass/Fail)",
        "",
        "| Metric | Statistic (mean of daily saved-state metrics) | Decision-185 Reference Bar | Units |",
        "|---|---|---:|---|",
        "| `SST` / `T_0-100m` / `T_100-1000m` / `T_>1000m` | Monthly mean of daily RMS | `0.01` | `K` |",
        "| `SSS` / `S_0-100m` / `S_100-1000m` / `S_>1000m` | Monthly mean of daily RMS | `0.002` | `psu` |",
        "| `SSH` | Monthly mean of daily RMS | `0.001` (`1 mm`) | `m` |",
        "| `KE` (surface) | Monthly mean of daily RMS / NEMO mean | `0.01` (`1%`) | fraction of NEMO mean |",
        "| `PSI` (barotropic streamfunction) | Monthly mean of daily RMS / spatial corr | `0.1 Sv` / `>= 0.999` | `Sv` / dimensionless |",
        "| `DRAKE` (Drake Passage transport) | `|cand - NEMO|` monthly mean of daily transport | `0.5` | `Sv` |",
        "| `MLD` (`Delta sigma_0 = 0.01 kg m^-3`) | Monthly mean of daily RMS | `1.0` | `m` |",
        "",
        "## Month-by-Month Summary (Evaluated & Pending Months 1–12)",
        "",
        "| Month | Days (`kt`) | Status | SST RMS (`0.01 K`) | SSS RMS (`0.002 psu`) | SSH RMS (`0.001 m`) | KE rel RMS (`0.01`) | PSI RMS (`0.1 Sv`) | PSI corr (`0.999`) | Drake `|diff|` (`0.5 Sv`) | MLD RMS (`1.0 m`) |",
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
        kts = m["saved_kts"]
        kt_span = f"kt={kts[0]}..{kts[-1]}"
        mld_str = (
            f"{mt['MLD']['daily_mean_rms']:.4e}"
            if "MLD" in mt
            else "skipped"
        )
        lines.append(
            f"| `{m['label']}` | `{day_span}` (`{len(m['available_days'])}d`, `{kt_span}`) | `{m['status']}` "
            f"| `{mt['SST']['daily_mean_rms']:.4e}` "
            f"| `{mt['SSS']['daily_mean_rms']:.4e}` "
            f"| `{mt['SSH']['daily_mean_rms']:.4e}` "
            f"| `{mt['KE']['relative_rms_fraction']:.4e}` "
            f"| `{mt['PSI']['daily_mean_rms_sv']:.4e}` "
            f"| `{mt['PSI']['daily_mean_pattern_corr']:.6f}` "
            f"| `{mt['DRAKE']['abs_mean_diff_sv']:.4e}` "
            f"| `{mld_str}` |"
        )
    lines.extend([
        "",
        "## Month-by-Month and Metric-by-Metric Table (NEMO Value, legoESM Value, Difference, Reproducibility Control, Decision-185 Reference Bar)",
        "",
        "| Month | Metric | NEMO Value | legoESM Value | Difference (`legoESM - NEMO`) | NEMO Twin B vs Twin A Control | Decision-185 Reference Bar (REFERENCE ONLY) |",
        "|---|---|---:|---:|---|---|---|",
    ])
    twin_months = (
        {m["label"]: m["metrics"] for m in twin_report["months"]}
        if twin_report is not None
        else {}
    )
    for m in report["months"]:
        if m["metrics"] is None:
            continue
        mt = m["metrics"]
        t_mt = twin_months.get(m["label"])
        for key in METRIC_TABLE_ORDER:
            if key not in mt:
                continue
            lines.append(_format_metric_row(m["label"], key, mt, t_mt))

    if "overall_window_metrics" in report and report["overall_window_metrics"]:
        overall_mt = report["overall_window_metrics"]
        overall_twin_mt = twin_report.get("overall_window_metrics") if twin_report is not None else None
        for key in METRIC_TABLE_ORDER:
            if key not in overall_mt:
                continue
            lines.append(_format_metric_row(f"ALL_{report['days_evaluated']}d", key, overall_mt, overall_twin_mt))

    lines.extend([
        "",
        "## Day-by-Day SST RMS Difference Curve",
        "",
        "| Day | `kt` | NEMO Mean SST (`degC`) | legoESM Mean SST (`degC`) | SST Bias (`K`) | SST RMS (`K`) | NEMO Twin B vs A SST RMS (`K`) | Decision-185 Reference Bar (`K`, Reference Only) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    twin_daily = (
        {int(r["day"]): r for r in twin_report["daily"]}
        if twin_report is not None
        else {}
    )
    for r in report["daily"]:
        d = int(r["day"])
        tr = twin_daily.get(d)
        t_rms = f"`{tr['SST']['rms']:.4e}`" if tr is not None else "—"
        n_sst = f"`{r['SST']['nemo_mean']:.6f}`" if "nemo_mean" in r["SST"] else "—"
        c_sst = f"`{r['SST']['cand_mean']:.6f}`" if "cand_mean" in r["SST"] else "—"
        lines.append(
            f"| `{d}` | `{r['kt']}` | {n_sst} | {c_sst} | `{r['SST']['bias']:+.4e}` | `{r['SST']['rms']:.4e}` | {t_rms} | `0.01` |"
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
    geo = pilot_metrics.static_geometry(args.nemo_root)
    mld = None if args.skip_mld else K0MLDEvaluator(args.nemo_root)
    ice_cover_path = (
        r340_s4.read_decision171_cover.__defaults__[0]
        if r340_s4.read_decision171_cover.__defaults__
        else Path(
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
            "ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
        )
    )
    cover = r340_s4.read_decision171_cover(ice_cover_path)

    cand_report = evaluate_month_by_month(
        nemo_root=args.nemo_root,
        candidate_root=args.candidate_root,
        days=args.days,
        candidate_is_nemo_restart=False,
        ice_cover_path=ice_cover_path,
        skip_mld=args.skip_mld,
        geo=geo,
        mld=mld,
        cover=cover,
        plots_dir=args.output_dir / "candidate_vs_nemo_plots",
    )

    twin_report: dict[str, Any] | None = None
    if args.nemo_twin_b_root is not None:
        twin_report = evaluate_month_by_month(
            nemo_root=args.nemo_root,
            candidate_root=args.nemo_twin_b_root,
            days=args.days,
            candidate_is_nemo_restart=True,
            ice_cover_path=ice_cover_path,
            skip_mld=args.skip_mld,
            geo=geo,
            mld=mld,
            cover=cover,
            plots_dir=args.output_dir / "twin_b_vs_a_plots",
        )
        (args.output_dir / "round342_s4_nemo_twin_b_vs_a_metrics.json").write_text(
            json.dumps(twin_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "round342_s4_nemo_twin_b_vs_a_metrics.md").write_text(
            format_markdown_report(twin_report),
            encoding="utf-8",
        )

    curve_payload = write_daily_sst_rms_curve_figure_and_data(
        cand_report=cand_report,
        twin_report=twin_report,
        output_dir=args.output_dir,
    )
    cand_report["daily_sst_rms_curve_summary"] = curve_payload["separation_milestones"]
    cand_report["daily_sst_rms_curve_figure"] = {
        "path": curve_payload["figure_path"],
        "sha256": curve_payload["figure_sha256"],
    }

    (args.output_dir / "round342_s4_candidate_vs_nemo_metrics.json").write_text(
        json.dumps(cand_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "round342_s4_candidate_vs_nemo_metrics.md").write_text(
        format_markdown_report(cand_report, twin_report=twin_report),
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
