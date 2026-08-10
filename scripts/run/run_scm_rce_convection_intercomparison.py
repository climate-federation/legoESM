"""Single-column (SCM) RCE **convection-scheme** intercomparison vs the plane CRM.

For every convection scheme in the campaign sweep this driver measures how well
an SCM RCE column reproduces the RCEMIP1 plane-CRM reference, both **a priori**
(scheme defaults) and **a posteriori** (after derivative-free tuning of the
scheme's ``tunable_tier<=extended`` parameters against the CRM profiles).

It is a thin orchestration layer over ``run_scm_rce_campaign`` — the CRM
reference extraction, the JIT SCM RCE evaluation, the normalized profile score
and the derivative-free tuner are all reused verbatim (see CLAUDE.md: "reuse
scripts/run/run_scm_rce_campaign.py for CRM reference extraction / SCM
evaluation"; no RCE profile numerics are re-derived here).

Outputs (under ``--outdir``, default ``results/scm_rce_convection_intercomparison``):

* ``summary.md``            — a-priori vs tuned RMSE table + tuned parameters,
* ``intercomparison.csv``   — machine-readable per-scheme metrics,
* ``tuned_parameters.json`` — per-scheme tuned overrides (recommended, never
  mutates production ``*Config`` defaults),
* ``profiles_<scheme>.png`` — CRM (black), a-priori (red **dashed**), tuned
  (red **solid**) T / q_v / condensate profiles,
* ``profiles_all_convection.png`` — every scheme in one grid.

Tuning objective is the combined profile+precip score against the CRM (lower is
better). The realism/equilibrium gate is *reported* per scheme but not used to
reject tuning trials, so every scheme is tuned toward the best CRM match the
user asked for; unphysical equilibria are flagged in the ``realism`` column
rather than silently hidden.

The schemes do not share a compensating-subsidence kernel by default, so an
as-shipped ranking partly measures the KERNEL rather than the scheme.
``--subsidence-solve`` selects the arm (via the campaign's shared
``apply_subsidence_solve_override`` selector); run it twice into distinct
``--outdir`` and read the PRIMARY arm for scheme physics.

Example (GPU) — the two arms::

    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/run/run_scm_rce_convection_intercomparison.py \
        --tune-evals 48 --subsidence-solve implicit_flux \
        --outdir results/scm_rce_convection_intercomparison_matched   # PRIMARY

    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/run/run_scm_rce_convection_intercomparison.py \
        --tune-evals 48 --subsidence-solve as_shipped \
        --outdir results/scm_rce_convection_intercomparison_shipped   # SECONDARY
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_campaign():
    """Import the campaign module as the shared SCM-RCE numerics library.

    The campaign lives in ``scripts/run`` as an executable driver, not an
    installed package; load it by path and register it in ``sys.modules`` so its
    module-level ``@dataclass`` definitions resolve.
    """
    path = Path(__file__).resolve().parent / "run_scm_rce_campaign.py"
    spec = importlib.util.spec_from_file_location("scm_rce_campaign", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


camp = _load_campaign()

DEFAULT_OUTDIR = Path("results/scm_rce_convection_intercomparison")
# Convection schemes to intercompare (the campaign's convection sweep).
CONVECTION_SCHEMES = camp.SCHEME_SWEEPS["convection"]


@dataclass
class SchemeResult:
    scheme: str
    prior: "camp.RunDiagnostics"
    tuned: "camp.RunDiagnostics"
    records: list  # list[camp.TuneRecord]
    # Which vertical-transport kernel ARM produced these numbers, and what the
    # shared selector actually did for THIS scheme.  Carried on every result and
    # written into the JSON / CSV / summary so a score can never be read without
    # knowing whether the schemes were kernel-matched (a cross-arm comparison is
    # a confound, not a result).
    subsidence_solve: str = "as_shipped"
    subsidence_solve_status: str = ""


# --------------------------------------------------------------------------- #
# Per-scheme persistence (each scheme runs in its own process; recompilation of
# the static-config SCM graph on every tune eval makes expensive-compile schemes
# — zhang_mcfarlane, emanuel, … — slow, so schemes are parallelized across cores
# and results checkpointed to disk so a killed/parallel run is resumable).
# --------------------------------------------------------------------------- #
from dataclasses import asdict  # noqa: E402


def _scheme_json_path(outdir: Path, scheme: str) -> Path:
    return outdir / f"scheme_{scheme}.json"


# Config fields that change a scheme's numerical result.  A checkpoint written
# under one signature must not be silently reused under another — the classic
# footgun is a ``--quick`` smoke checkpoint (0.05 day, 2 tune evals) reused in a
# full-length run and merged as if it were a real result.  Guards skip + merge.
#
# ``subsidence_solve`` is in here for the same reason and is the sharpest case:
# the PRIMARY (matched-kernel) and SECONDARY (as-shipped) arms are two different
# experiments, so reusing one arm's per-scheme checkpoint in the other would
# silently FABRICATE that arm's number.  Removing it from this tuple is a
# correctness regression, gated by
# tests/unit/test_scm_rce_convection_intercomparison_cli.py.
_SIGNATURE_FIELDS = (
    "days", "dt", "analysis_days", "tune_evals", "tune_seed", "radiation",
    "radiation_update_interval_steps", "large_scale_forcing",
    "surface_wind_m_s", "coriolis_s_inv",
    "scm_microphysics_substeps", "scm_convection_substeps",
    "subsidence_solve",
    "reference_dir", "last_reference_files",
    # The column's saturation treatment is part of the experiment: the ice
    # super-saturation allowance only exists in the ice-capable schemes, and
    # the in-scheme liquid guard changes the condensation rate every step.  A
    # checkpoint from one setting must not be reused under another.
    "microphysics", "hard_saturation_adjustment",
)


def _run_signature(args) -> dict:
    # Path -> str so the signature is JSON-serializable; every other field the
    # argparse namespace stores here is already a JSON scalar/list.  Includes the
    # tune seed and the reference selection — different seed or CRM reference is a
    # different experiment and must invalidate a checkpoint.
    def _norm(v):
        return str(v) if isinstance(v, Path) else v
    return {k: _norm(getattr(args, k)) for k in _SIGNATURE_FIELDS}


def _checkpoint_signature(path: Path) -> dict:
    try:
        return json.loads(path.read_text()).get("signature", {}) or {}
    except (OSError, ValueError):
        return {}


def _signature_compatible(ckpt_sig: dict, run_sig: dict) -> bool:
    # Legacy/unstamped checkpoints ({}) are grandfathered in; a STAMPED
    # signature that differs is the mismatch we reject.
    return (not ckpt_sig) or ckpt_sig == run_sig


def save_scheme_result(outdir: Path, res: SchemeResult, signature: dict | None = None) -> None:
    payload = {
        "scheme": res.scheme,
        "signature": signature or {},
        "subsidence_solve": res.subsidence_solve,
        "subsidence_solve_status": res.subsidence_solve_status,
        "prior": asdict(res.prior),
        "tuned": asdict(res.tuned),
        "records": [asdict(r) for r in res.records],
    }
    # Checkpoints use Python's extended JSON (NaN/Infinity permitted) so a
    # crashed scheme's non-finite score round-trips back through
    # load_scheme_result; they are Python-self-consumed, not strict JSON.
    _scheme_json_path(outdir, res.scheme).write_text(json.dumps(payload, indent=2))


def load_scheme_result(path: Path) -> SchemeResult:
    payload = json.loads(path.read_text())
    return SchemeResult(
        scheme=payload["scheme"],
        prior=camp.RunDiagnostics(**payload["prior"]),
        tuned=camp.RunDiagnostics(**payload["tuned"]),
        records=[camp.TuneRecord(**r) for r in payload["records"]],
        # Pre-arm checkpoints predate the flag; they were necessarily run with
        # the shipped defaults, so that is the honest label for them.  The empty
        # status distinguishes "never stamped" from a stamped "as_shipped".
        subsidence_solve=payload.get("subsidence_solve", "as_shipped"),
        subsidence_solve_status=payload.get("subsidence_solve_status", ""),
    )


def load_all_scheme_results(outdir: Path, schemes) -> list[SchemeResult]:
    # Merge aggregates EVERY checkpoint present (its documented job); the
    # signature guard lives on the compute-path skip, which recomputes a
    # mismatched checkpoint before it can reach the merge.
    results = []
    for scheme in schemes:
        path = _scheme_json_path(outdir, scheme)
        if path.exists():
            results.append(load_scheme_result(path))
    return results


def evaluate_scheme(
    scheme: str,
    ref,
    *,
    days: float,
    dt: float,
    analysis_days: float,
    tune_evals: int,
    seed: int,
    scm_microphysics_substeps: int,
    scm_convection_substeps: int,
    surface_wind_m_s: float,
    coriolis_s_inv: float,
    large_scale_forcing: str,
    radiation: str,
    radiation_update_interval_steps: int,
    subsidence_solve: str = "as_shipped",
    microphysics: str = camp.BASELINE_SCHEMES["microphysics"],
    hard_saturation_adjustment: bool = False,
) -> SchemeResult:
    """A-priori run + derivative-free tuning for one convection scheme.

    The realism/equilibrium gate is disabled (``require_*=False``) so the tuner
    minimizes the CRM profile score freely; realism diagnostics are still
    computed and reported.

    ``subsidence_solve`` selects the vertical-transport kernel arm via the shared
    ``camp.apply_subsidence_solve_override`` selector (never re-implemented
    here).  It is applied to ``base_cfg`` IMMEDIATELY, before the a-priori run
    and before tuning, so BOTH see the same kernel — applying it later would
    tune under one kernel and report under another.
    """
    cache: dict[str, "camp.RunDiagnostics"] = {}
    base_cfg = camp.make_physics_config(
        radiation=radiation,
        radiation_update_interval_steps=radiation_update_interval_steps,
        convection=scheme,
        microphysics=microphysics,
        hard_saturation_adjustment=hard_saturation_adjustment,
    )
    base_cfg, solve_status = camp.apply_subsidence_solve_override(
        base_cfg, subsidence_solve, category="convection")
    common = dict(
        days=days,
        dt=dt,
        analysis_days=analysis_days,
        require_equilibrium=False,
        require_realism=False,
        equil_T_tol_K=camp.EQUIL_T_TOL_K,
        equil_qv_tol=camp.EQUIL_QV_TOL,
        equil_qcond_tol=camp.EQUIL_QCOND_TOL,
        scm_microphysics_substeps=scm_microphysics_substeps,
        scm_convection_substeps=scm_convection_substeps,
        surface_wind_m_s=surface_wind_m_s,
        coriolis_s_inv=coriolis_s_inv,
        large_scale_forcing=large_scale_forcing,
    )
    prior = camp.run_cached(cache, base_cfg, ref, label=f"prior:{scheme}", **common)
    _best_cfg, records, tuned = camp.tune_category_winner(
        "convection",
        base_cfg,
        ref,
        cache,
        tune_evals=tune_evals,
        seed=seed,
        **common,
    )
    return SchemeResult(
        scheme=scheme, prior=prior, tuned=tuned, records=records,
        subsidence_solve=subsidence_solve, subsidence_solve_status=solve_status,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float, prec: str = ".4g") -> str:
    return "inf" if not np.isfinite(x) else format(float(x), prec)


def _physical_verdict(run) -> str:
    """Physical/unphysical verdict from the always-computed realism diagnostics.

    Mirrors ``run_scm_rce``'s equilibrium+realism gate — same campaign threshold
    constants (reused by name, no re-derived numerics) and same NaN-fails-closed
    semantics as ``realism_reasons_from_diagnostics``. We evaluate it here because
    the tuning runs use ``require_realism=False`` (so the tuner minimizes the CRM
    score freely), which leaves ``run.realism_status`` as ``not_checked`` even
    though the underlying moist-adiabat / cold-point / drift diagnostics are still
    populated.

    Covers every gate condition whose scalar is stored on ``RunDiagnostics``:
    equilibrium drift (T/qv/qcond), moist-adiabat mean+max deviation, and
    cold-point T+z. The one gate condition it cannot see is the free-troposphere
    level count (``n_free_trop_levels >= MIN_FREE_TROP_LEVELS = 3``), which is not
    a stored field — negligible for a 30-level RCE column but noted for honesty:
    a ``physical`` verdict here means "physical modulo the level-count check".
    """
    if not np.isfinite(run.score):
        return "nonfinite"
    # Honor the campaign's own hard-failure status FIRST: a run the campaign
    # marked failed (negative q_v/condensate, out-of-range T, invalid precip,
    # non-finite profile — see run_scm_rce_campaign's realism gate) must never
    # be reported "physical" just because the stored realism scalars happen to
    # pass.  ``status == "ok"`` means the campaign's hard checks passed.
    if run.status != "ok":
        return "unphysical"
    reasons = (
        run.drift_T_rmse_K > camp.EQUIL_T_TOL_K
        or run.drift_qv_rmse > camp.EQUIL_QV_TOL
        or run.drift_qcond_rmse > camp.EQUIL_QCOND_TOL
        or not (run.moist_adiabat_mean_abs_K <= camp.MADIAB_MEAN_TOL_K)
        or not (run.moist_adiabat_max_abs_K <= camp.MADIAB_MAX_TOL_K)
        or not (camp.COLD_POINT_MIN_K <= run.cold_point_T_K <= camp.COLD_POINT_MAX_K)
        or not (camp.TROP_MIN_Z_KM <= run.cold_point_z_km <= camp.TROP_MAX_Z_KM)
    )
    return "unphysical" if reasons else "physical"


CSV_FIELDS = (
    "scheme",
    # Kernel arm, immediately after the scheme name: every downstream reader of
    # this CSV sees which arm produced the row before it sees any metric.
    "subsidence_solve", "subsidence_solve_status",
    "prior_score", "prior_T_rmse", "prior_qv_rmse", "prior_cloud_rmse",
    "prior_precip_rmse", "prior_precip_mm_day", "prior_verdict",
    "tuned_score", "tuned_T_rmse", "tuned_qv_rmse", "tuned_cloud_rmse",
    "tuned_precip_rmse", "tuned_precip_mm_day", "tuned_verdict",
    "score_improvement_pct", "crm_precip_mm_day", "n_tuned_params",
    "tuned_drift_T_K", "tuned_madiab_mean_K", "tuned_cold_point_T_K",
    "tuned_cold_point_z_km",
    # PHYSICAL-unit RMSE (K, g/kg).  The scores above are normalised by the
    # reference's mass-weighted standard deviation — commensurable for the
    # optimiser, uninterpretable in a figure caption.  Column names match what
    # scripts/plot/plot_scm_rce_convection_paper.py reads.
    "apriori_T_rmse_K", "tuned_T_rmse_K",
    "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
    "apriori_qcond_rmse_g_kg", "tuned_qcond_rmse_g_kg",
)

KG_KG_TO_G_KG = 1_000.0


def _row(res: SchemeResult, ref=None) -> dict:
    p, t = res.prior, res.tuned
    impr = (
        100.0 * (p.score - t.score) / p.score
        if np.isfinite(p.score) and p.score > 0 and np.isfinite(t.score)
        else float("nan")
    )
    if ref is None:
        # No reference in scope (unit tests of the row shape): the physical
        # columns are NaN rather than absent, so the CSV header never changes
        # shape between call sites.
        nan = float("nan")
        phys = {k: nan for k in (
            "apriori_T_rmse_K", "tuned_T_rmse_K",
            "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
            "apriori_qcond_rmse_g_kg", "tuned_qcond_rmse_g_kg")}
    else:
        pr = camp.physical_profile_rmse(ref, p)
        tr = camp.physical_profile_rmse(ref, t)
        phys = {
            "apriori_T_rmse_K": pr["T_rmse_K"],
            "tuned_T_rmse_K": tr["T_rmse_K"],
            "apriori_qv_rmse_g_kg": pr["qv_rmse_kg_kg"] * KG_KG_TO_G_KG,
            "tuned_qv_rmse_g_kg": tr["qv_rmse_kg_kg"] * KG_KG_TO_G_KG,
            "apriori_qcond_rmse_g_kg": pr["qcond_rmse_kg_kg"] * KG_KG_TO_G_KG,
            "tuned_qcond_rmse_g_kg": tr["qcond_rmse_kg_kg"] * KG_KG_TO_G_KG,
        }
    return {
        **phys,
        "scheme": res.scheme,
        "subsidence_solve": res.subsidence_solve,
        "subsidence_solve_status": res.subsidence_solve_status,
        "prior_score": p.score, "prior_T_rmse": p.T_rmse,
        "prior_qv_rmse": p.qv_rmse, "prior_cloud_rmse": p.cloud_rmse,
        "prior_precip_rmse": p.precip_rmse, "prior_precip_mm_day": p.precip_mm_day,
        "prior_verdict": _physical_verdict(p),
        "tuned_score": t.score, "tuned_T_rmse": t.T_rmse,
        "tuned_qv_rmse": t.qv_rmse, "tuned_cloud_rmse": t.cloud_rmse,
        "tuned_precip_rmse": t.precip_rmse, "tuned_precip_mm_day": t.precip_mm_day,
        "tuned_verdict": _physical_verdict(t),
        "score_improvement_pct": impr, "crm_precip_mm_day": p.precip_ref_mm_day,
        "n_tuned_params": len(res.records),
        "tuned_drift_T_K": t.drift_T_rmse_K,
        "tuned_madiab_mean_K": t.moist_adiabat_mean_abs_K,
        "tuned_cold_point_T_K": t.cold_point_T_K,
        "tuned_cold_point_z_km": t.cold_point_z_km,
    }


def write_csv(path: Path, results: list[SchemeResult], ref=None) -> None:
    """Machine-readable per-scheme metrics.

    ``ref`` is what turns the physical-unit columns from NaN into numbers, so
    the merge stage passes it; a caller that only wants the normalised scores
    may omit it.
    """
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for res in sorted(results, key=lambda r: r.tuned.score):
            writer.writerow(_row(res, ref))


def _finite_or_none(x: float):
    # A crashed scheme can carry a non-finite score; null keeps this advertised
    # artifact valid for strict JSON consumers (jq) while losing no real value.
    return x if np.isfinite(x) else None


def write_tuned_parameters(path: Path, results: list[SchemeResult]) -> None:
    payload = {}
    for res in results:
        if not res.records:
            continue
        payload[res.scheme] = {
            rec.scheme_key + "." + rec.parameter: {
                "default": rec.default, "tuned": rec.tuned,
                "bounds": [rec.lower, rec.upper], "units": rec.units,
                "score_default": _finite_or_none(rec.score_default),
                "score_tuned": _finite_or_none(rec.score_tuned),
            }
            for rec in res.records
        }
    # allow_nan=False: tuned_parameters.json is an externally consumed artifact,
    # not self-reloaded checkpoint state, so it must be strict JSON.
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


# One blurb per kernel arm, stated at the TOP of the summary so no table in this
# file can be read without knowing which arm produced it.
_ARM_BLURB = {
    "implicit_flux": (
        "**PRIMARY (matched-kernel) arm.** Every convection scheme that owns a "
        "compensating-subsidence mass-flux kernel is forced onto the "
        "conservative `implicit_flux` solve, so a score gap between two such "
        "schemes measures the SCHEME, not the transport kernel."
    ),
    "as_shipped": (
        "**SECONDARY (as-shipped) arm.** Every scheme keeps its own shipped "
        "`subsidence_solve` default — this is what users get today. Because "
        "Bechtold/EDMF/Kain-Fritsch ship `implicit_flux` while "
        "Tiedtke/Zhang-McFarlane/mass_flux ship the leaky `advective` solve, "
        "part of any score gap here measures the KERNEL rather than the scheme; "
        "read the PRIMARY arm for scheme physics."
    ),
    "advective": (
        "**Symmetric control arm.** Every kernel-capable scheme is forced onto "
        "the leaky `advective` solve, bounding the kernel's contribution from "
        "the other direction."
    ),
}
_MIXED_ARM_BLURB = (
    "**WARNING — CONFOUNDED TABLE.** These rows were NOT produced under one "
    "common kernel arm, so the ranking mixes scheme physics with transport-"
    "kernel differences and must NOT be read as a scheme intercomparison. "
    "Re-run each arm into its own `--outdir`."
)


def write_summary(path: Path, ref, results: list[SchemeResult], meta: dict) -> None:
    ordered = sorted(results, key=lambda r: r.tuned.score)
    lines: list[str] = []
    lines.append("# SCM RCE Convection-Scheme Intercomparison vs CRM (RCEMIP1)\n")
    lines.append(
        "Each convection scheme is run in the SCM RCE column and scored against "
        "the plane-CRM reference (horizontal/time mean of the last "
        f"{meta['last_reference_files']} CRM 3-D daily volumes). Metrics are the "
        "std-normalized, mass-weighted profile RMSE (T, q_v, condensate) plus a "
        "surface-precip term; **score** is their combination (lower = closer to "
        "CRM). *A priori* = scheme defaults; *tuned* = after "
        f"{meta['tune_evals']}-evaluation derivative-free tuning of the scheme's "
        "extended-tier parameters against the CRM profiles.\n"
    )
    lines.append(
        f"SCM: radiation `{meta['radiation']}`, fixed SST 300 K, dt {meta['dt']:.0f} s, "
        f"{meta['days']:.0f} d ({meta['analysis_days']:.0f} d analysis window), "
        f"surface wind {meta['surface_wind_m_s']:.1f} m/s, large-scale forcing "
        f"`{meta['large_scale_forcing']}`. CRM equilibrium surface precip "
        f"{ref.precip_ref_mm_day:.3g} mm/day.\n"
    )
    # Kernel arm, stated before any number. Derived from the RESULTS (not from
    # meta) so a merged outdir that accidentally mixes arms is reported as mixed
    # rather than mislabelled with the current invocation's flag.
    arms = sorted({r.subsidence_solve for r in results})
    single_arm = len(arms) == 1
    arm_label = arms[0] if single_arm else "MIXED(" + ",".join(arms) + ")"
    lines.append(
        f"Kernel arm: `--subsidence-solve {arm_label}`. "
        + (_ARM_BLURB[arm_label] if single_arm and arm_label in _ARM_BLURB
           else _MIXED_ARM_BLURB)
        + " The per-scheme `kernel` column below reports what the shared "
        "override actually did: `forced:<scheme>=<solve>` was kernel-matched, "
        "while `not_applicable:<scheme>` has no such knob (sbm/dca/kuo have no "
        "mass-flux kernel; emanuel's shipped buoyancy-sorting path never calls "
        "it) and so was **not** kernel-matched in either arm.\n"
    )
    lines.append(
        "> The realism/equilibrium gate is **reported** (`verdict` column, derived "
        "from the campaign's own moist-adiabat / cold-point / equilibrium-drift "
        "thresholds) but not used to reject tuning trials, so each scheme is tuned "
        "to the best CRM match; `unphysical` marks a good-RMSE but non-RCE column.\n"
    )
    lines.append("## A priori vs tuned RMSE\n")
    lines.append(
        "| rank | scheme | kernel | score (prior→tuned) | T RMSE (p→t) | "
        "qv RMSE (p→t) | cloud RMSE (p→t) | precip mm/d (p→t) | Δscore % | "
        "verdict (p→t) | cold-pt T,z (tuned) | #params |"
    )
    lines.append("|---:|---|---|---|---|---|---|---|---:|---|---|---:|")
    for i, res in enumerate(ordered, 1):
        p, t = res.prior, res.tuned
        row = _row(res, ref)
        lines.append(
            f"| {i} | {res.scheme} "
            f"| {res.subsidence_solve_status or res.subsidence_solve} "
            f"| {_fmt(p.score)}→{_fmt(t.score)} "
            f"| {_fmt(p.T_rmse)}→{_fmt(t.T_rmse)} "
            f"| {_fmt(p.qv_rmse)}→{_fmt(t.qv_rmse)} "
            f"| {_fmt(p.cloud_rmse)}→{_fmt(t.cloud_rmse)} "
            f"| {_fmt(p.precip_mm_day, '.3g')}→{_fmt(t.precip_mm_day, '.3g')} "
            f"| {_fmt(row['score_improvement_pct'], '.1f')} "
            f"| {row['prior_verdict']}→{row['tuned_verdict']} "
            f"| {_fmt(t.cold_point_T_K, '.0f')} K, {_fmt(t.cold_point_z_km, '.1f')} km "
            f"| {len(res.records)} |"
        )
    lines.append(f"\nCRM reference surface precip: {ref.precip_ref_mm_day:.3g} mm/day.\n")
    lines.append(
        "Verdict thresholds (campaign RCE realism gate): equilibrium drift_T ≤ "
        f"{camp.EQUIL_T_TOL_K:g} K, |T−T_moist| mean ≤ {camp.MADIAB_MEAN_TOL_K:g} K / "
        f"max ≤ {camp.MADIAB_MAX_TOL_K:g} K, cold point in "
        f"[{camp.COLD_POINT_MIN_K:g}, {camp.COLD_POINT_MAX_K:g}] K at "
        f"[{camp.TROP_MIN_Z_KM:g}, {camp.TROP_MAX_Z_KM:g}] km.\n"
    )

    lines.append("## Tuned parameters\n")
    for res in ordered:
        if not res.records:
            lines.append(f"- **{res.scheme}**: no extended-tier tunable parameters.")
            continue
        lines.append(f"- **{res.scheme}** (score {_fmt(res.prior.score)}→{_fmt(res.tuned.score)}):")
        for rec in res.records:
            lines.append(
                f"  - `{rec.scheme_key}.{rec.parameter}` "
                f"{_fmt(rec.default)} → {_fmt(rec.tuned)} {rec.units} "
                f"(bounds [{_fmt(rec.lower)}, {_fmt(rec.upper)}])"
            )
    lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Plotting — CRM (black), a-priori (red dashed), tuned (red solid)
# --------------------------------------------------------------------------- #
_CRM_COLOR = "#000000"
_RED = "#d62728"
_M_PER_KM = camp.M_PER_KM
_KG_TO_G = camp.MSE_KJ_TO_J  # kg/kg -> g/kg is also x1000


def _panels(ref, run):
    T = np.asarray(run.T_profile)
    qv = np.asarray(run.qv_profile) * _KG_TO_G
    qc = np.asarray(run.qcond_profile) * _KG_TO_G
    return [
        ("T [K]", ref.T_ref, T),
        ("q$_v$ [g/kg]", ref.qv_ref * _KG_TO_G, qv),
        ("condensate [g/kg]", ref.qcond_ref * _KG_TO_G, qc),
    ]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_scheme(path: Path, ref, res: SchemeResult) -> None:
    if not res.prior.T_profile:
        return
    plt = _plt()
    z_km = ref.z_m / _M_PER_KM
    ztop = float(np.nanmax(z_km))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0))
    prior_panels = _panels(ref, res.prior)
    tuned_panels = _panels(ref, res.tuned)
    for ax, (label, ref_prof, prior_prof), (_l, _r, tuned_prof) in zip(
        axes, prior_panels, tuned_panels
    ):
        ax.plot(ref_prof, z_km, color=_CRM_COLOR, lw=2.2, label="CRM (RCEMIP1)")
        ax.plot(prior_prof, z_km, color=_RED, lw=1.8, ls="--", label="SCM a priori")
        ax.plot(tuned_prof, z_km, color=_RED, lw=2.0, ls="-", label="SCM tuned")
        ax.set_xlabel(label)
        ax.set_ylim(0.0, ztop)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z [km]")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"convection = {res.scheme}   "
        f"score {_fmt(res.prior.score)} → {_fmt(res.tuned.score)}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all(path: Path, ref, results: list[SchemeResult]) -> None:
    # Only plot schemes that actually produced profiles: a crashed scheme has
    # empty T/qv/qcond profiles and would raise "x and y must have same first
    # dimension" in the shared-axes montage.  Filter EVERY result, not just the
    # first, since a valid scheme can sort ahead of a crashed one.
    plottable = [r for r in results if r.prior.T_profile and r.tuned.T_profile]
    ordered = sorted(plottable, key=lambda r: r.tuned.score)
    if not ordered:
        return
    plt = _plt()
    z_km = ref.z_m / _M_PER_KM
    ztop = float(np.nanmax(z_km))
    n = len(ordered)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 3.1 * n), squeeze=False)
    for i, res in enumerate(ordered):
        prior_panels = _panels(ref, res.prior)
        tuned_panels = _panels(ref, res.tuned)
        for j, ((label, ref_prof, prior_prof), (_l, _r, tuned_prof)) in enumerate(
            zip(prior_panels, tuned_panels)
        ):
            ax = axes[i][j]
            ax.plot(ref_prof, z_km, color=_CRM_COLOR, lw=2.0, label="CRM")
            ax.plot(prior_prof, z_km, color=_RED, lw=1.6, ls="--", label="a priori")
            ax.plot(tuned_prof, z_km, color=_RED, lw=1.8, ls="-", label="tuned")
            ax.set_ylim(0.0, ztop)
            ax.grid(alpha=0.2)
            if i == n - 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel(f"{res.scheme}\nz [km]", fontsize=9)
        axes[i][2].text(
            1.02, 0.5,
            f"score\n{_fmt(res.prior.score)}\n→ {_fmt(res.tuned.score)}",
            transform=axes[i][2].transAxes, fontsize=8, va="center",
        )
    axes[0][0].legend(loc="best", fontsize=7)
    fig.suptitle("SCM RCE convection intercomparison vs CRM — a priori (dashed) vs tuned (solid)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, exposed so tests exercise the REAL parser (a hand-rolled
    namespace would keep passing after a flag is renamed or dropped)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=camp.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--days", type=float, default=camp.DEFAULT_DAYS)
    parser.add_argument("--dt", type=float, default=camp.DEFAULT_DT_S)
    parser.add_argument("--analysis-days", type=float, default=camp.DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--last-reference-files", type=int,
                        default=camp.DEFAULT_LAST_REFERENCE_FILES)
    parser.add_argument("--tune-evals", type=int, default=48)
    parser.add_argument("--tune-seed", type=int, default=20260705)
    parser.add_argument(
        "--radiation", default="rrtmgp", choices=("rrtmgp", "gray"),
        help=(
            "SCM radiation. Default `rrtmgp` is the RCEMIP standard (broadband) and "
            "matches the `rrtmgp`-default plane CRM (run_rcemip_plane.py); the CRM "
            "reference MUST be regenerated with rrtmgp for an apples-to-apples "
            "stratosphere. `gray` is a fast approximation that under-drives "
            "convection and leaves a warm cold point."
        ),
    )
    parser.add_argument("--radiation-update-interval-steps", type=int, default=None)
    parser.add_argument(
        "--microphysics", default=camp.BASELINE_SCHEMES["microphysics"],
        choices=camp.SCHEME_SWEEPS["microphysics"],
        help=(
            "SCM microphysics, held FIXED across every convection scheme. The "
            "default `kessler` is WARM-RAIN ONLY: it carries no ice, so the "
            "IFS/SAM homogeneous-freezing ice-super-saturation allowance is "
            "inert and the upper troposphere is biased for every scheme "
            "alike. Use `morrison` (SAM M2005 flavor) to score the cold point "
            "against an ice-carrying CRM reference."
        ),
    )
    parser.add_argument(
        "--hard-saturation-adjustment", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the in-scheme iterated saturation adjustment (the IFS "
            "'no liquid super-saturation' half) for the selected microphysics. "
            "Default off = the smooth-sigmoid path, which leaves a few percent "
            "standing super-saturation. sdm/fast_sbm reject the flag."
        ),
    )
    parser.add_argument("--schemes", default=None,
                        help="comma-separated convection scheme subset")
    parser.add_argument("--scm-microphysics-substeps", type=int,
                        default=camp.DEFAULT_SCM_MICROPHYSICS_SUBSTEPS)
    parser.add_argument("--scm-convection-substeps", type=int,
                        default=camp.DEFAULT_SCM_CONVECTION_SUBSTEPS)
    parser.add_argument("--surface-wind-m-s", type=float,
                        default=camp.DEFAULT_SCM_RCE_SURFACE_WIND_M_S)
    parser.add_argument("--coriolis-s-inv", type=float,
                        default=camp.DEFAULT_SCM_RCE_CORIOLIS_S_INV)
    parser.add_argument("--large-scale-forcing", default=camp.DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
                        choices=camp.SCM_RCE_LARGE_SCALE_FORCING_CHOICES)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run (short, 2 schemes, few evals)")
    parser.add_argument("--merge-only", action="store_true",
                        help="skip running; build summary/CSV/combined plot from "
                             "existing scheme_*.json checkpoints")
    parser.add_argument("--no-merge", action="store_true",
                        help="run schemes + write per-scheme json/png but skip the "
                             "aggregate summary/CSV/combined plot (for parallel workers)")
    parser.add_argument("--force", action="store_true",
                        help="re-run schemes even if a scheme_*.json checkpoint exists")
    parser.add_argument(
        "--subsidence-solve", default="as_shipped",
        choices=list(camp.SUBSIDENCE_SOLVE_MODES),
        help=(
            "Vertical-transport kernel ARM. Convection schemes do not share a "
            "compensating-subsidence kernel by default — Bechtold/EDMF/"
            "Kain-Fritsch ship the conservative `implicit_flux` solve while "
            "Tiedtke/Zhang-McFarlane/mass_flux ship the leaky `advective` one — "
            "so an as-shipped ranking partly measures the KERNEL, not the "
            "scheme. Run the campaign TWICE into distinct --outdir: "
            "`implicit_flux` = PRIMARY arm, every kernel-capable scheme forced "
            "onto the conservative solve (isolates scheme physics); "
            "`as_shipped` (default) = SECONDARY arm, every scheme keeps its own "
            "shipped default (what users get today); `advective` = symmetric "
            "control. Schemes with no such knob (sbm/dca/kuo, and emanuel whose "
            "shipped path bypasses the kernel) are reported "
            "`not_applicable:<scheme>`, never silently skipped. The arm is part "
            "of the checkpoint signature, so one arm's result can never be "
            "reused for another."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.radiation_update_interval_steps is None:
        args.radiation_update_interval_steps = (
            camp.DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS if args.radiation == "rrtmgp" else 1
        )

    if args.schemes:
        selected = tuple(s.strip() for s in args.schemes.split(",") if s.strip())
        unknown = [s for s in selected if s not in CONVECTION_SCHEMES]
        if unknown:
            raise SystemExit(f"Unknown convection schemes {unknown}; "
                             f"known={list(CONVECTION_SCHEMES)}")
        schemes = selected
    else:
        schemes = CONVECTION_SCHEMES
    if args.quick:
        args.days = min(args.days, 0.05)
        args.analysis_days = min(args.analysis_days, args.days)
        args.tune_evals = min(args.tune_evals, 2)
        schemes = schemes[:2]

    args.outdir.mkdir(parents=True, exist_ok=True)
    meta_path = args.outdir / "run_meta.json"

    # In --merge-only we aggregate checkpoints from a PRIOR run: rebuild the CRM
    # reference from that run's recorded config (run_meta.json), not the current
    # CLI defaults, so the summary/plots compare against the same target.
    saved_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if args.merge_only and saved_meta:
        args.analysis_days = saved_meta.get("analysis_days", args.analysis_days)
        args.last_reference_files = saved_meta.get(
            "last_reference_files", args.last_reference_files)
        if "reference_dir" in saved_meta:
            args.reference_dir = Path(saved_meta["reference_dir"])

    ref = camp.build_reference_profiles(
        args.reference_dir, args.last_reference_files,
        precip_analysis_days=args.analysis_days,
    )

    meta = dict(
        radiation=args.radiation, dt=args.dt, days=args.days,
        analysis_days=args.analysis_days, tune_evals=args.tune_evals,
        surface_wind_m_s=args.surface_wind_m_s,
        large_scale_forcing=args.large_scale_forcing,
        last_reference_files=args.last_reference_files,
        reference_dir=str(args.reference_dir),
        subsidence_solve=args.subsidence_solve,
        microphysics=args.microphysics,
        hard_saturation_adjustment=args.hard_saturation_adjustment,
        scm_microphysics_substeps=args.scm_microphysics_substeps,
        scm_convection_substeps=args.scm_convection_substeps,
    )
    if args.merge_only and saved_meta:
        meta = {**meta, **saved_meta}

    run_sig = _run_signature(args)
    if not args.merge_only:
        for scheme in schemes:
            ckpt = _scheme_json_path(args.outdir, scheme)
            if ckpt.exists() and not args.force:
                if _signature_compatible(_checkpoint_signature(ckpt), run_sig):
                    print(f"[scheme] convection={scheme} — checkpoint exists, skip", flush=True)
                    continue
                print(f"[scheme] convection={scheme} — checkpoint config mismatch "
                      f"(e.g. --quick vs full), recomputing", flush=True)
            print(f"[scheme] convection={scheme} ...", flush=True)
            res = evaluate_scheme(
                scheme, ref,
                days=args.days, dt=args.dt, analysis_days=args.analysis_days,
                tune_evals=args.tune_evals, seed=args.tune_seed,
                scm_microphysics_substeps=args.scm_microphysics_substeps,
                scm_convection_substeps=args.scm_convection_substeps,
                surface_wind_m_s=args.surface_wind_m_s,
                coriolis_s_inv=args.coriolis_s_inv,
                large_scale_forcing=args.large_scale_forcing,
                radiation=args.radiation,
                radiation_update_interval_steps=args.radiation_update_interval_steps,
                subsidence_solve=args.subsidence_solve,
                microphysics=args.microphysics,
                hard_saturation_adjustment=args.hard_saturation_adjustment,
            )
            save_scheme_result(args.outdir, res, run_sig)  # checkpoint before plotting
            plot_scheme(args.outdir / f"profiles_{scheme}.png", ref, res)
            print(f"    prior score={_fmt(res.prior.score)} "
                  f"-> tuned score={_fmt(res.tuned.score)} "
                  f"({len(res.records)} params) "
                  f"[kernel {res.subsidence_solve_status}]", flush=True)
        # ATOMIC: the campaign runs one process per scheme against a shared
        # --outdir, so several finish at once and write this same file. The
        # bytes are identical (nothing scheme-specific is in `meta`), but two
        # interleaved write_text calls can still leave a truncated file, and
        # --merge-only reads it to rebuild the reference. Write-then-rename is
        # atomic within a directory on POSIX.
        _tmp = meta_path.with_suffix(f".json.{os.getpid()}.tmp")
        _tmp.write_text(json.dumps(meta, indent=2))
        _tmp.replace(meta_path)

    if args.no_merge:
        print(f"[done] ran {len(schemes)} scheme(s); merge skipped (--no-merge)")
        return 0

    # Merge: aggregate every scheme checkpoint present in outdir (covers schemes
    # run by parallel worker processes, not just this invocation's subset).
    results = load_all_scheme_results(args.outdir, CONVECTION_SCHEMES)
    if not results:
        raise SystemExit(f"No scheme_*.json checkpoints found in {args.outdir}")
    write_csv(args.outdir / "intercomparison.csv", results, ref)
    # Second copy under the name the paper figure script reads, so the
    # figures are built from THIS table rather than a hand-copied one.
    write_csv(args.outdir / "summary_table.csv", results, ref)
    write_tuned_parameters(args.outdir / "tuned_parameters.json", results)
    write_summary(args.outdir / "summary.md", ref, results, meta)
    plot_all(args.outdir / "profiles_all_convection.png", ref, results)
    print(f"[done] merged {len(results)} scheme(s) -> {args.outdir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
