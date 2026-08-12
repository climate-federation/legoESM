#!/usr/bin/env python
"""Per-scheme SCM RCE convection tuning against the plane-CRM RCEMIP reference.

For EVERY convection scheme (not just the campaign winner) this driver runs

1. the a-priori (default-parameter) SCM RCE equilibrium, and
2. a bounded derivative-free tuning of that scheme's ``__param_spec__``
   tunables (same machinery as ``run_scm_rce_campaign.tune_category_winner``),
   followed by the a-posteriori (tuned) equilibrium,

then writes a summary RMSE table (a priori vs tuned, normalized scores plus
physical-unit RMSE), per-scheme profile plots (CRM reference black, a priori
red dashed, a posteriori red solid), combined multi-panel figures, and JSON
records suitable for aggregation across parallel per-scheme invocations.

All reference extraction, SCM evaluation, scoring and tuning numerics are
reused from ``scripts/run/run_scm_rce_campaign.py`` and
``legoesm.training.scm_rce_metrics`` — nothing is re-derived here.

Run on CPU, for example:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/run/run_scm_rce_convection_tuning.py

Parallel orchestration: launch one process per scheme with
``--schemes <name>`` sharing the same ``--outdir`` (each writes its own
``scheme_<name>.json``), then run once with ``--aggregate-only`` to build the
table and figures.  Use ``--quick`` for the smoke path used by the unit test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from legoesm.training.scm_rce_metrics import weighted_rmse

from scripts.run import run_scm_rce_campaign as campaign


DEFAULT_OUTDIR = Path("results/scm_rce_convection_tuning")
CONVECTION_SCHEMES = campaign.SCHEME_SWEEPS["convection"]
QUICK_SCHEMES = campaign.QUICK_SCHEME_SWEEPS["convection"]
KG_KG_TO_G_KG = 1_000.0
M_PER_KM = 1_000.0

CRM_COLOR = "black"
SCM_COLOR = "#d62728"


# Physical-unit RMSE lives in the campaign module so this driver and the
# intercomparison driver cannot drift apart on the weights or the reference
# arrays (CLAUDE.md: no duplicate numerics across drivers).
_physical_rmse = campaign.physical_profile_rmse


def _run_kwargs(args) -> dict[str, Any]:
    return {
        "days": args.days,
        "dt": args.dt,
        "analysis_days": args.analysis_days,
        "require_equilibrium": args.require_equilibrium,
        "require_realism": args.require_realism,
        "equil_T_tol_K": args.equil_T_tol_K,
        "equil_qv_tol": args.equil_qv_tol,
        "equil_qcond_tol": args.equil_qcond_tol,
        "scm_microphysics_substeps": args.scm_microphysics_substeps,
        "scm_convection_substeps": args.scm_convection_substeps,
        "surface_wind_m_s": args.surface_wind_m_s,
        "coriolis_s_inv": args.coriolis_s_inv,
        "large_scale_forcing": args.large_scale_forcing,
    }


def run_one_scheme(
    scheme: str,
    scheme_index: int,
    ref: campaign.ReferenceProfiles,
    cache: dict[str, campaign.RunDiagnostics],
    args,
) -> dict[str, Any]:
    cfg = campaign.make_physics_config(
        radiation=args.radiation,
        radiation_update_interval_steps=args.radiation_update_interval_steps,
        convection=scheme,
    )
    kwargs = _run_kwargs(args)
    print(f"[a-priori] convection={scheme}")
    a_priori = campaign.run_cached(
        cache, cfg, ref, label=f"apriori:{scheme}", **kwargs,
    )
    print(
        f"          -> {a_priori.status} score={a_priori.score:.6g} "
        f"{a_priori.reason.splitlines()[0] if a_priori.reason else ''}"
    )
    if args.skip_tuning:
        tuned_cfg, records, tuned = cfg, [], a_priori
    else:
        print(f"[tune]     convection={scheme} ({args.tune_evals} evals)")
        tuned_cfg, records, tuned = campaign.tune_category_winner(
            "convection",
            cfg,
            ref,
            cache,
            tune_evals=args.tune_evals,
            seed=args.tune_seed + scheme_index,
            **kwargs,
        )
        print(
            f"          -> {len(records)} params, tuned score={tuned.score:.6g} "
            f"({tuned.status})"
        )
    return {
        "scheme": scheme,
        "a_priori": asdict(a_priori),
        "tuned": asdict(tuned),
        "a_priori_physical_rmse": _physical_rmse(ref, a_priori),
        "tuned_physical_rmse": _physical_rmse(ref, tuned),
        "tune_records": [asdict(r) for r in records],
        "tuned_convection_config": campaign._to_jsonable(
            getattr(tuned_cfg, "convection"),
        ),
    }


# ----------------------------- aggregation ----------------------------- #


def _fmt(value: float, spec: str = ".4g") -> str:
    if value is None or math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, spec)


_CSV_FIELDS = [
    "scheme", "n_params_tuned",
    "apriori_status", "tuned_status",
    "apriori_score", "tuned_score", "score_improvement_pct",
    "apriori_T_rmse_K", "tuned_T_rmse_K",
    "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
    "apriori_qcond_rmse_g_kg", "tuned_qcond_rmse_g_kg",
    "apriori_T_rmse_norm", "tuned_T_rmse_norm",
    "apriori_qv_rmse_norm", "tuned_qv_rmse_norm",
    "apriori_cloud_rmse_norm", "tuned_cloud_rmse_norm",
    "apriori_precip_mm_day", "tuned_precip_mm_day", "crm_precip_mm_day",
    "apriori_reason", "tuned_reason",
]


def _table_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for res in results:
        ap = res["a_priori"]
        tu = res["tuned"]
        ap_phys = res["a_priori_physical_rmse"]
        tu_phys = res["tuned_physical_rmse"]
        ap_score = float(ap["score"])
        tu_score = float(tu["score"])
        if math.isfinite(ap_score) and math.isfinite(tu_score) and ap_score > 0.0:
            improvement = 100.0 * (ap_score - tu_score) / ap_score
        else:
            improvement = float("nan")
        rows.append(
            {
                "scheme": res["scheme"],
                "n_params_tuned": len(res["tune_records"]),
                "apriori_status": ap["status"],
                "tuned_status": tu["status"],
                "apriori_score": ap_score,
                "tuned_score": tu_score,
                "score_improvement_pct": improvement,
                "apriori_T_rmse_K": ap_phys["T_rmse_K"],
                "tuned_T_rmse_K": tu_phys["T_rmse_K"],
                "apriori_qv_rmse_g_kg": ap_phys["qv_rmse_kg_kg"] * KG_KG_TO_G_KG,
                "tuned_qv_rmse_g_kg": tu_phys["qv_rmse_kg_kg"] * KG_KG_TO_G_KG,
                "apriori_qcond_rmse_g_kg": (
                    ap_phys["qcond_rmse_kg_kg"] * KG_KG_TO_G_KG
                ),
                "tuned_qcond_rmse_g_kg": tu_phys["qcond_rmse_kg_kg"] * KG_KG_TO_G_KG,
                "apriori_T_rmse_norm": float(ap["T_rmse"]),
                "tuned_T_rmse_norm": float(tu["T_rmse"]),
                "apriori_qv_rmse_norm": float(ap["qv_rmse"]),
                "tuned_qv_rmse_norm": float(tu["qv_rmse"]),
                "apriori_cloud_rmse_norm": float(ap["cloud_rmse"]),
                "tuned_cloud_rmse_norm": float(tu["cloud_rmse"]),
                "apriori_precip_mm_day": float(ap["precip_mm_day"]),
                "tuned_precip_mm_day": float(tu["precip_mm_day"]),
                "crm_precip_mm_day": float(ap["precip_ref_mm_day"]),
                "apriori_reason": ap["reason"].replace("\n", " "),
                "tuned_reason": tu["reason"].replace("\n", " "),
            }
        )
    # Match the campaign's status-aware ordering: an equilibrated/realistic
    # ("ok") scheme always ranks ahead of a "failed" one, whatever its score.
    status_rank = {"ok": 0, "failed": 1, "crashed": 2}
    return sorted(
        rows,
        key=lambda r: (
            status_rank.get(r["tuned_status"], 3),
            r["tuned_score"]
            if math.isfinite(r["tuned_score"])
            else float("inf"),
            r["scheme"],
        ),
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_md(
    path: Path,
    rows: list[dict[str, Any]],
    ref: campaign.ReferenceProfiles,
    args,
) -> None:
    lines = [
        "# SCM RCE convection-scheme tuning vs CRM",
        "",
        f"CRM reference: `{args.reference_dir}` "
        f"(last {len(ref.files_used)} daily volumes; "
        f"CRM precip {ref.precip_ref_mm_day:.3f} mm/day).",
        f"SCM: radiation `{args.radiation}`, days {args.days:.4g}, "
        f"dt {args.dt:.4g} s, analysis window {args.analysis_days:.4g} d, "
        f"tune evals {args.tune_evals} "
        f"(bounded search within `__param_spec__` extended-tier bounds).",
        "`score` = combined normalized RMSE (mass-weighted, std-normalized "
        "T/qv/cloud profiles + precipitation term). Physical RMSEs are "
        "mass-weighted vertical RMSE vs the CRM profile.",
        "",
        "| scheme | tuned params | a-priori score | tuned score | Δ% | "
        "a-priori T-RMSE [K] | tuned T-RMSE [K] | "
        "a-priori qv-RMSE [g/kg] | tuned qv-RMSE [g/kg] | "
        "a-priori P [mm/d] | tuned P [mm/d] | CRM P [mm/d] | status (a-priori → tuned) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scheme']} | {r['n_params_tuned']} "
            f"| {_fmt(r['apriori_score'])} | {_fmt(r['tuned_score'])} "
            f"| {_fmt(r['score_improvement_pct'], '.1f')} "
            f"| {_fmt(r['apriori_T_rmse_K'])} | {_fmt(r['tuned_T_rmse_K'])} "
            f"| {_fmt(r['apriori_qv_rmse_g_kg'])} | {_fmt(r['tuned_qv_rmse_g_kg'])} "
            f"| {_fmt(r['apriori_precip_mm_day'])} "
            f"| {_fmt(r['tuned_precip_mm_day'])} "
            f"| {_fmt(r['crm_precip_mm_day'])} "
            f"| {r['apriori_status']} → {r['tuned_status']} |"
        )
    lines += ["", "## Tuned parameters", ""]
    lines.append("| scheme | parameter | default | tuned | bounds | units |")
    lines.append("|---|---|---:|---:|---|---|")
    path.write_text("\n".join(lines) + "\n")


def _append_tuned_params_md(
    path: Path, results: list[dict[str, Any]],
) -> None:
    lines = []
    for res in results:
        for rec in res["tune_records"]:
            lines.append(
                f"| {res['scheme']} | {rec['parameter']} "
                f"| {_fmt(rec['default'], '.6g')} | {_fmt(rec['tuned'], '.6g')} "
                f"| [{_fmt(rec['lower'], '.6g')}, {_fmt(rec['upper'], '.6g')}] "
                f"| {rec['units']} |"
            )
    with path.open("a") as f:
        f.write("\n".join(lines) + "\n")


def _plot_scheme_profiles(
    path: Path,
    ref: campaign.ReferenceProfiles,
    result: dict[str, Any],
) -> None:
    a_priori = result["a_priori"]
    tuned = result["tuned"]
    if not a_priori["T_profile"] and not tuned["T_profile"]:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        print(f"[plot] skipped {path}: {exc}")
        return
    z_km = np.asarray(ref.z_m, dtype=float) / M_PER_KM
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.0))
    panels = [
        ("T [K]", np.asarray(ref.T_ref), "T_profile", 1.0),
        ("q_v [g/kg]", np.asarray(ref.qv_ref) * KG_KG_TO_G_KG, "qv_profile",
         KG_KG_TO_G_KG),
        ("condensate [g/kg]", np.asarray(ref.qcond_ref) * KG_KG_TO_G_KG,
         "qcond_profile", KG_KG_TO_G_KG),
    ]
    for ax, (label, ref_prof, key, scale) in zip(axes[:3], panels):
        ax.plot(ref_prof, z_km, color=CRM_COLOR, lw=2.2, label="CRM")
        if a_priori[key]:
            ax.plot(
                np.asarray(a_priori[key]) * scale, z_km,
                color=SCM_COLOR, lw=1.8, ls="--", label="SCM a priori",
            )
        if tuned[key]:
            ax.plot(
                np.asarray(tuned[key]) * scale, z_km,
                color=SCM_COLOR, lw=2.2, ls="-", label="SCM a posteriori",
            )
        ax.set_xlabel(label)
        ax.set_ylim(0.0, float(np.nanmax(z_km)))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z [km]")
    axes[0].legend(loc="best", fontsize=9)
    pax = axes[3]
    bars = [
        ("CRM", float(a_priori["precip_ref_mm_day"]), CRM_COLOR, "-"),
        ("a priori", float(a_priori["precip_mm_day"]), "none", "--"),
        ("a posteriori", float(tuned["precip_mm_day"]), SCM_COLOR, "-"),
    ]
    for i, (label, value, facecolor, _ls) in enumerate(bars):
        pax.bar(
            i, value,
            color=facecolor if facecolor != "none" else "white",
            edgecolor=CRM_COLOR if i == 0 else SCM_COLOR,
            linestyle="--" if i == 1 else "-",
            linewidth=1.8,
            width=0.65,
        )
    pax.set_xticks(range(len(bars)))
    pax.set_xticklabels([b[0] for b in bars], rotation=20)
    pax.set_ylabel("surface precip [mm/day]")
    pax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"convection={result['scheme']}: "
        f"a-priori score {_fmt(float(a_priori['score']))} → "
        f"tuned {_fmt(float(tuned['score']))}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_combined(
    path: Path,
    ref: campaign.ReferenceProfiles,
    results: list[dict[str, Any]],
    key: str,
    xlabel: str,
    scale: float,
) -> None:
    plottable = [
        r for r in results if r["a_priori"][key] or r["tuned"][key]
    ]
    if not plottable:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        print(f"[plot] skipped {path}: {exc}")
        return
    z_km = np.asarray(ref.z_m, dtype=float) / M_PER_KM
    if key == "T_profile":
        ref_prof = np.asarray(ref.T_ref) * scale
    elif key == "qv_profile":
        ref_prof = np.asarray(ref.qv_ref) * scale
    else:
        ref_prof = np.asarray(ref.qcond_ref) * scale
    ncols = min(5, len(plottable))
    nrows = (len(plottable) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.1 * ncols, 3.6 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, res in zip(axes.flat, plottable):
        ax.set_visible(True)
        ax.plot(ref_prof, z_km, color=CRM_COLOR, lw=1.8, label="CRM")
        if res["a_priori"][key]:
            ax.plot(
                np.asarray(res["a_priori"][key]) * scale, z_km,
                color=SCM_COLOR, lw=1.4, ls="--", label="a priori",
            )
        if res["tuned"][key]:
            ax.plot(
                np.asarray(res["tuned"][key]) * scale, z_km,
                color=SCM_COLOR, lw=1.8, ls="-", label="a posteriori",
            )
        ax.set_title(res["scheme"], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, float(np.nanmax(z_km)))
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(xlabel)
    for row in axes:
        if row[0].get_visible():
            row[0].set_ylabel("z [km]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(
        f"SCM RCE convection schemes vs CRM — {xlabel} "
        "(red dashed: a priori, red solid: a posteriori)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_score_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    """Grouped a-priori vs tuned bar chart, one pair per scheme (finite scores)."""
    finite = [r for r in rows if math.isfinite(r["tuned_score"])]
    if not finite:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        print(f"[plot] skipped {path}: {exc}")
        return
    schemes = [r["scheme"] for r in finite]
    apriori = [r["apriori_score"] if math.isfinite(r["apriori_score"]) else np.nan
               for r in finite]
    tuned = [r["tuned_score"] for r in finite]
    x = np.arange(len(finite))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(finite)), 4.6))
    ax.bar(x - width / 2, apriori, width, label="a priori", color="#9ecae1")
    ax.bar(x + width / 2, tuned, width, label="a posteriori (tuned)",
           color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=40, ha="right")
    ax.set_ylabel("combined normalized RMSE vs RCEMIP1")
    ax.set_title("SCM RCE convection: a priori vs tuned score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def aggregate(
    outdir: Path,
    ref: campaign.ReferenceProfiles,
    results: list[dict[str, Any]],
    args,
) -> None:
    rows = _table_rows(results)
    _write_summary_csv(outdir / "summary_table.csv", rows)
    _write_summary_md(outdir / "summary.md", rows, ref, args)
    _append_tuned_params_md(outdir / "summary.md", results)
    _plot_score_comparison(outdir / "score_apriori_vs_tuned.png", rows)
    for res in results:
        _plot_scheme_profiles(
            outdir / f"profiles_{res['scheme']}.png", ref, res,
        )
    _plot_combined(
        outdir / "profiles_all_T.png", ref, results,
        "T_profile", "T [K]", 1.0,
    )
    _plot_combined(
        outdir / "profiles_all_qv.png", ref, results,
        "qv_profile", "q_v [g/kg]", KG_KG_TO_G_KG,
    )
    _plot_combined(
        outdir / "profiles_all_qcond.png", ref, results,
        "qcond_profile", "condensate [g/kg]", KG_KG_TO_G_KG,
    )
    print(f"[aggregate] wrote table + plots for {len(results)} schemes")


def _parse_schemes(raw: str, quick: bool) -> list[str]:
    if raw == "all":
        return list(QUICK_SCHEMES if quick else CONVECTION_SCHEMES)
    selected = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in selected if s not in CONVECTION_SCHEMES]
    if unknown:
        raise SystemExit(
            f"Unknown convection schemes {unknown}; "
            f"known = {sorted(CONVECTION_SCHEMES)}"
        )
    if not selected:
        raise SystemExit("--schemes selected no convection scheme")
    return selected


def main(argv: list[str] | None = None) -> int:
    campaign._require_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir", type=Path, default=campaign.DEFAULT_REFERENCE_DIR,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--days", type=float, default=campaign.DEFAULT_DAYS)
    parser.add_argument("--dt", type=float, default=campaign.DEFAULT_DT_S)
    parser.add_argument(
        "--analysis-days", type=float, default=campaign.DEFAULT_ANALYSIS_DAYS,
    )
    parser.add_argument(
        "--last-reference-files",
        type=int,
        default=campaign.DEFAULT_LAST_REFERENCE_FILES,
    )
    parser.add_argument(
        "--schemes",
        default="all",
        help=(
            "Comma-separated convection schemes, or 'all' "
            f"({', '.join(CONVECTION_SCHEMES)})."
        ),
    )
    parser.add_argument("--tune-evals", type=int, default=32)
    parser.add_argument("--tune-seed", type=int, default=20260703)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument(
        "--radiation",
        choices=["rrtmgp", "gray"],
        default=campaign.BASELINE_SCHEMES["radiation"],
    )
    parser.add_argument(
        "--radiation-update-interval-steps", type=int, default=None,
    )
    parser.add_argument(
        "--scm-microphysics-substeps",
        type=int,
        default=campaign.DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
    )
    parser.add_argument(
        "--scm-convection-substeps",
        type=int,
        default=campaign.DEFAULT_SCM_CONVECTION_SUBSTEPS,
    )
    parser.add_argument(
        "--surface-wind-m-s",
        type=float,
        default=campaign.DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
    )
    parser.add_argument(
        "--coriolis-s-inv",
        type=float,
        default=campaign.DEFAULT_SCM_RCE_CORIOLIS_S_INV,
    )
    parser.add_argument(
        "--large-scale-forcing",
        choices=campaign.SCM_RCE_LARGE_SCALE_FORCING_CHOICES,
        default=campaign.DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
    )
    parser.add_argument(
        "--equil-T-tol-K", type=float, default=campaign.EQUIL_T_TOL_K,
    )
    parser.add_argument(
        "--equil-qv-tol", type=float, default=campaign.EQUIL_QV_TOL,
    )
    parser.add_argument(
        "--equil-qcond-tol", type=float, default=campaign.EQUIL_QCOND_TOL,
    )
    parser.add_argument(
        "--tuning-objective",
        choices=["rmse", "gated"],
        default="rmse",
        help=(
            "Acceptance criterion for a tuned trial. 'rmse' (default) accepts "
            "any finite lower-RMSE profile match — the right objective for "
            "'tune each scheme to best match the CRM', since most SCM RCE "
            "equilibria fail the strict realism/equilibrium gates (the SCM "
            "tropopause is warmer than the gate's 175-210 K window) which would "
            "otherwise block all tuning. 'gated' reproduces the campaign "
            "behavior: only realism+equilibrium-passing trials are accepted. "
            "The per-scheme realism diagnostics (cold point, moist-adiabat "
            "deviation) are recorded either way."
        ),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Skip all SCM runs; rebuild the summary table and plots from the "
            "scheme_<name>.json records already present in --outdir."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "With --aggregate-only: aggregate a partial scheme set instead of "
            "failing when some scheme_<name>.json records are absent."
        ),
    )
    args = parser.parse_args(argv)
    if args.scm_microphysics_substeps < 1:
        raise SystemExit("--scm-microphysics-substeps must be a positive integer")
    if args.scm_convection_substeps < 1:
        raise SystemExit("--scm-convection-substeps must be a positive integer")
    if args.surface_wind_m_s < 0.0:
        raise SystemExit("--surface-wind-m-s must be non-negative")
    if args.tune_evals < 1:
        raise SystemExit("--tune-evals must be a positive integer")
    if args.quick:
        args.days = min(args.days, campaign.QUICK_DAYS)
        args.analysis_days = min(args.analysis_days, args.days)
        args.tune_evals = min(args.tune_evals, campaign.QUICK_TUNE_EVALS)
        args.radiation = "gray"
    if args.radiation_update_interval_steps is None:
        args.radiation_update_interval_steps = (
            campaign.DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS
            if args.radiation == "rrtmgp"
            else 1
        )
    if args.radiation_update_interval_steps < 1:
        raise SystemExit("--radiation-update-interval-steps must be positive")
    # 'rmse' objective (default): tune purely on profile RMSE so the strict
    # realism/equilibrium gates do not veto every lower-RMSE trial (the SCM RCE
    # tropopause is warmer than the gate window). 'gated' keeps the campaign's
    # realism+equilibrium acceptance. Quick smoke never gates.
    gated = (not args.quick) and args.tuning_objective == "gated"
    args.require_equilibrium = gated
    args.require_realism = gated
    args.outdir.mkdir(parents=True, exist_ok=True)

    schemes = _parse_schemes(args.schemes, args.quick)
    ref = campaign.build_reference_profiles(
        args.reference_dir,
        args.last_reference_files,
        precip_analysis_days=args.analysis_days,
    )

    if args.aggregate_only:
        results = []
        missing = []
        for scheme in schemes:
            path = args.outdir / f"scheme_{scheme}.json"
            if not path.exists():
                missing.append(scheme)
                continue
            results.append(json.loads(path.read_text()))
        if missing and not args.allow_missing:
            raise SystemExit(
                f"--aggregate-only is missing scheme records for {missing} in "
                f"{args.outdir}; rerun those schemes or pass --allow-missing "
                "to aggregate the partial set."
            )
        if missing:
            print(f"[aggregate] WARNING: aggregating without {missing}")
        if not results:
            raise SystemExit(
                f"--aggregate-only found no scheme_<name>.json in {args.outdir}"
            )
        aggregate(args.outdir, ref, results, args)
        return 0

    cache: dict[str, campaign.RunDiagnostics] = {}
    results = []
    for scheme in schemes:
        scheme_index = CONVECTION_SCHEMES.index(scheme)
        result = run_one_scheme(scheme, scheme_index, ref, cache, args)
        path = args.outdir / f"scheme_{scheme}.json"
        path.write_text(
            json.dumps(campaign._to_jsonable(result), indent=2, sort_keys=True)
            + "\n"
        )
        results.append(result)
    aggregate(args.outdir, ref, results, args)
    print(f"[done] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
