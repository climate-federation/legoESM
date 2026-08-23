#!/usr/bin/env python
"""Q1b MEASURED diagnostic-flux margin: best-tuned nonlocal vs best-tuned local.

Quantifies LES_SUITE.md §7 Q1(b) in *diagnostic* scoring: at a driven CBL mean state,
evaluate each closure's diagnosed heat flux ``⟨w'θ'⟩`` (``scm_runner.diagnostic_scheme_flux``,
which imposes the LES surface θ-flux on the closure) and score it against the LES total
flux (``score.diagnostic_flux_score``). A LOCAL down-gradient closure carries ≈0 flux
through the well-mixed layer (``F=−Kh·∂θ/∂z`` with ``∂θ/∂z≈0``) however it is tuned, while a
NONLOCAL closure's counter-gradient carries the surface flux up — so best-tuned nonlocal
beats best-tuned local by a large, tuning-robust margin. The margin's σ_LES error bar is the
spread over the LES SGS closures {lasd, smagorinsky, vreman}; the result is significant when
it exceeds that spread (D7 gate).

Best-tuned = the Q2 tier-1 tuned params (the per-closure ``results/les_suite/tuned/*__df.json``
fit on the lasd artifact); a closure with no tuned JSON falls back to its default config
(logged). Scope: the harness
needs enough mean wind to carry the bulk surface flux, so a free-convective (Ug≈0) or stable
(negative-flux) artifact raises inside ``diagnostic_scheme_flux`` and is reported as skipped —
the SHEARED CBL is the calibratable vehicle (the free-convective structural ceiling is Q1a).

Usage::

    python scripts/validate/les_suite/q1b_diagnostic_margin.py \\
        --artifacts-dir results/les_suite/artifacts \\
        --tuned-dir results/les_suite/tuned \\
        --case-prefix cbl_nieuwstadt --suffix ug8 \\
        --output results/les_suite/q1b_diagnostic_margin.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.atmosphere.les_suite.bridge import diagnostic_truth, load_artifact
from legoesm.atmosphere.les_suite.scm_runner import diagnostic_scheme_flux
from legoesm.atmosphere.les_suite.score import diagnostic_flux_score
from legoesm.atmosphere.physics import TurbulenceConfig

# The four K-closures that expose TurbulenceOutput.wtheta_flux, split by transport type.
LOCAL_SCHEMES = ("smagorinsky", "louis")
NONLOCAL_SCHEMES = ("holtslag_boville", "ysu")
SGS_VARIANTS = ("lasd", "smagorinsky", "vreman")


def _tuned_config(
    tuned_dir: Path, case_prefix: str, suffix: str, scheme: str
) -> tuple[TurbulenceConfig, bool]:
    """Best-tuned TurbulenceConfig for a scheme (tuned on the lasd artifact), or default.

    Returns ``(config, had_tuned)``. The lazy import of the tuner's single apply-site keeps
    this a plain validator that reuses the SAME override-splice as production tuning (no
    re-derivation of the param-application logic).
    """
    base = TurbulenceConfig(scheme=scheme)
    path = tuned_dir / f"{case_prefix}__lasd__{suffix}__{scheme}__df.json"
    if not path.exists():
        return base, False
    sys.path.insert(0, "scripts/run")
    from tune_scm_to_les import apply_overrides_to_base  # noqa: PLC0415

    overrides = json.loads(path.read_text()).get("best_overrides", {})
    return apply_overrides_to_base(base, overrides), True


def measure_margin(
    artifacts_dir: Path,
    tuned_dir: Path,
    case_prefix: str,
    suffix: str,
    *,
    nlev: int = 32,
    use_tuned: bool = True,
) -> dict:
    """Best-tuned diagnostic-flux margin across the SGS variants of one sheared case.

    One tuned SCM (params fit on the lasd artifact) scored against each LES SGS truth — the
    controlled comparison (one closure config, several LES truths for the σ_LES bar).
    """
    cfgs: dict[str, TurbulenceConfig] = {}
    had_tuned: dict[str, bool] = {}
    for scheme in LOCAL_SCHEMES + NONLOCAL_SCHEMES:
        if use_tuned:
            cfgs[scheme], had_tuned[scheme] = _tuned_config(
                tuned_dir, case_prefix, suffix, scheme)
        else:
            cfgs[scheme], had_tuned[scheme] = TurbulenceConfig(scheme=scheme), False

    per_sgs: list[dict] = []
    for sgs in SGS_VARIANTS:
        path = artifacts_dir / f"{case_prefix}__{sgs}__{suffix}.npz"
        if not path.exists():
            continue
        art = load_artifact(path)
        truth = diagnostic_truth(art)
        rmse: dict[str, float] = {}
        skipped: dict[str, str] = {}
        for scheme, cfg in cfgs.items():
            try:
                flux = diagnostic_scheme_flux(art, cfg, nlev=nlev)
                rmse[scheme] = float(diagnostic_flux_score(truth, flux).wtheta_rmse)
            except ValueError as exc:  # uncalibratable (free-conv / stable) → report, skip
                skipped[scheme] = str(exc).split(" — ")[0]
        if not (all(s in rmse for s in LOCAL_SCHEMES)
                and all(s in rmse for s in NONLOCAL_SCHEMES)):
            per_sgs.append({"sgs": sgs, "rmse": rmse, "skipped": skipped})
            continue
        local_worst = max(rmse[s] for s in LOCAL_SCHEMES)
        nonlocal_best = min(rmse[s] for s in NONLOCAL_SCHEMES)
        per_sgs.append({
            "sgs": sgs,
            "rmse": rmse,
            "local_worst": local_worst,
            "nonlocal_best": nonlocal_best,
            "margin": local_worst - nonlocal_best,
        })

    margins = [r["margin"] for r in per_sgs if "margin" in r]
    summary = {
        "case_prefix": case_prefix,
        "suffix": suffix,
        "use_tuned": use_tuned,
        "had_tuned": had_tuned,
        "per_sgs": per_sgs,
        "n_sgs_scored": len(margins),
    }
    if margins:
        m = np.asarray(margins)
        summary.update({
            "margin_mean": float(m.mean()),
            "margin_std_sigma_les": float(m.std(ddof=0)),
            "margin_min": float(m.min()),
            # significant when the smallest margin still clears the σ_LES spread.
            "significant": bool(m.min() > m.std(ddof=0)),
        })
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", type=Path,
                   default=Path("results/les_suite/artifacts"))
    p.add_argument("--tuned-dir", type=Path, default=Path("results/les_suite/tuned"))
    p.add_argument("--case-prefix", default="cbl_nieuwstadt")
    p.add_argument("--suffix", default="ug8",
                   help="artifact suffix identifying the sheared case (e.g. ug8)")
    p.add_argument("--nlev", type=int, default=32)
    p.add_argument("--untuned", action="store_true",
                   help="use default (untuned) closures instead of the Q2 tuned params")
    p.add_argument("--output", type=Path,
                   default=Path("results/les_suite/q1b_diagnostic_margin.json"))
    args = p.parse_args(argv)

    if not args.artifacts_dir.exists():
        print(f"error: {args.artifacts_dir} does not exist", file=sys.stderr)
        return 2
    res = measure_margin(
        args.artifacts_dir, args.tuned_dir, args.case_prefix, args.suffix,
        nlev=args.nlev, use_tuned=not args.untuned)
    if res["n_sgs_scored"] == 0:
        print(f"error: no scorable {args.case_prefix}__*__{args.suffix} artifacts in "
              f"{args.artifacts_dir} (all SGS variants missing or uncalibratable)",
              file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    tuned_tag = "default" if args.untuned else "best-tuned"
    print(f"Q1b diagnostic-flux margin ({tuned_tag}) — {args.case_prefix} {args.suffix}")
    print(f"  {'LES-sgs':<12} {'smag':>7} {'louis':>7} | {'hb':>7} {'ysu':>7} | margin")
    for r in res["per_sgs"]:
        rm = r["rmse"]
        cell = lambda s: f"{rm[s]:.3f}" if s in rm else "  skip"  # noqa: E731
        margin = f"{r['margin']:+.3f}" if "margin" in r else "   n/a"
        print(f"  {r['sgs']:<12} {cell('smagorinsky'):>7} {cell('louis'):>7} | "
              f"{cell('holtslag_boville'):>7} {cell('ysu'):>7} | {margin}")
    if "margin_mean" in res:
        print(f"\n  margin = {res['margin_mean']:.3f} ± {res['margin_std_sigma_les']:.3f} "
              f"(σ_LES), min {res['margin_min']:.3f} → "
              f"{'SIGNIFICANT' if res['significant'] else 'NOT significant'} vs σ_LES")
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
