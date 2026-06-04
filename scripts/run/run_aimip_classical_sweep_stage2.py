#!/usr/bin/env python
"""Plan stage-2 of the AIMIP classical physics combinatorial sweep.

Stage-2 strategy
----------------
After stage-1 (one-at-a-time, 27 runs) finishes, parse each combo's
scorecard, rank schemes per dimension, pick the top-2 per dimension,
and run the full Cartesian product of winners (2^4 = 16 runs).  This
captures cross-scheme interactions that the OAT stage cannot see.

Selection metric
----------------
The default ranking is the eval ``loss['mean']`` from each combo
scorecard — that's the same composite objective used in training
(MSE + bias + grid-CRPS + spec-CRPS, level-weighted, multi-day) so
it's the natural "lower is better" composite.  Use ``--metric`` to
override (``T_rmse``, ``T_bias_abs``, etc.) for sensitivity checks.

Inputs
------
Reads stage-1 scorecards from
``results/aimip_classical_sweep_stage1/combo_*/aimip_scorecard.json``
(written by ``run_aimip.py`` for each variant).

Outputs (idempotent)
--------------------
- ``config/aimip/sweep/stage2/<combo_name>/{suite,variant_classical}.yaml``
- ``config/aimip/sweep/stage2/manifest.json``
- ``config/aimip/sweep/stage2/stage1_winners.json``  (audit trail)
- ``scripts/run/_aimip_sweep_stage2_runner.sbatch``

Launching
---------
::

    sbatch --array=0-15 scripts/run/_aimip_sweep_stage2_runner.sbatch

After stage-2, identify the single best classical combo (lowest eval
loss) for the final T106 head-to-head against column_nn and sfno.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


# Same shared baseline as stage-1 (T21 + RRTMGP + multi-day + bias +
# spec-CRPS).  Stage-2 inherits all settings except the four scheme
# choices, which are filled per combo from the stage-1 winners.
BASELINE_REL = Path("config/aimip/sweep/stage1/baseline_classical_t21_rrtmgp.yaml")
STAGE1_RESULTS_REL = Path("results/aimip_classical_sweep_stage1")
STAGE2_DIR_REL = Path("config/aimip/sweep/stage2")
STAGE2_RESULTS_REL = Path("results/aimip_classical_sweep_stage2")

DIMENSIONS = (
    "aimip_convection",
    "aimip_turbulence",
    "aimip_gwd",
    "aimip_microphysics",
)
SHORT_DIM = {
    "aimip_convection": "conv",
    "aimip_turbulence": "turb",
    "aimip_gwd": "gwd",
    "aimip_microphysics": "micro",
}
BASELINE_SCHEMES = {
    "aimip_convection": "tiedtke",
    "aimip_turbulence": "louis",
    "aimip_gwd": "mcfarlane",
    "aimip_microphysics": "sundqvist",
}


def _load_stage1_manifest(repo_root: Path) -> dict:
    path = repo_root / "config/aimip/sweep/stage1/manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run scripts/run/run_aimip_classical_sweep_stage1.py first."
        )
    return json.loads(path.read_text())


def _score_combo(scorecard: dict, metric: str) -> float:
    """Extract the ranking metric from a per-combo scorecard.

    Default ``metric='loss'`` reads
    ``variants.classical.eval_metrics.loss.mean`` — the composite
    objective the model was trained against.  Lower is better.
    """
    try:
        em = scorecard["variants"]["classical"]["eval_metrics"]
    except KeyError as exc:
        raise KeyError(
            f"scorecard missing variants.classical.eval_metrics ({exc})"
        )
    if metric == "loss":
        return float(em["loss"]["mean"])
    if metric == "T_rmse":
        return float(em["rmse"]["T"]["mean"])
    if metric == "T_bias_abs":
        return float(abs(em["bias"]["T"]["mean"]))
    if metric == "composite":
        # Composite: T_RMSE + 2 * |T_bias| (heavy bias penalty) + 0.5 * (u_RMSE + v_RMSE)
        T_rmse = float(em["rmse"]["T"]["mean"])
        T_bias = float(abs(em["bias"]["T"]["mean"]))
        u_rmse = float(em["rmse"]["u"]["mean"])
        v_rmse = float(em["rmse"]["v"]["mean"])
        return T_rmse + 2.0 * T_bias + 0.5 * (u_rmse + v_rmse)
    raise ValueError(f"Unknown metric {metric!r}")


def _collect_stage1_scores(
    repo_root: Path, stage1_manifest: dict, metric: str,
) -> tuple[dict, dict, dict]:
    """Read every stage-1 scorecard, return:

    - ``baseline_score`` : scalar score from the baseline combo
    - ``scores_by_dim`` : dim -> [{scheme, score, combo_name}, ...] (incl baseline scheme)
    - ``missing`` : combo_name -> error message for any unreadable scorecard
    """
    baseline_score = None
    scores_by_dim: dict[str, list[dict]] = {d: [] for d in DIMENSIONS}
    missing: dict[str, str] = {}

    # Stage-1 manifest carries the (dim, scheme, combo_name) tuple per run.
    # Sort baseline first so we can fold it into every dimension's list.
    for combo in stage1_manifest["combos"]:
        sc_path = (
            repo_root / STAGE1_RESULTS_REL / combo["name"] / "aimip_scorecard.json"
        )
        if not sc_path.exists():
            missing[combo["name"]] = f"scorecard not found: {sc_path}"
            continue
        try:
            sc = json.loads(sc_path.read_text())
            score = _score_combo(sc, metric)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            missing[combo["name"]] = repr(exc)
            continue

        if combo["dim"] == "baseline":
            baseline_score = score
            continue
        scores_by_dim[combo["dim"]].append({
            "scheme": combo["scheme"],
            "score": score,
            "combo_name": combo["name"],
        })

    # Fold baseline into each dim's ranking — baseline scheme is in
    # competition with the OAT alternatives for the top-2 slots.
    if baseline_score is not None:
        for dim, scheme in BASELINE_SCHEMES.items():
            scores_by_dim[dim].append({
                "scheme": scheme,
                "score": baseline_score,
                "combo_name": "combo_baseline",
            })
    return baseline_score, scores_by_dim, missing


def _pick_top_k(scores_by_dim: dict, k: int = 2) -> dict[str, list[dict]]:
    """Return the top-k schemes per dimension (lowest score wins)."""
    winners: dict[str, list[dict]] = {}
    for dim, entries in scores_by_dim.items():
        sorted_entries = sorted(entries, key=lambda e: e["score"])
        # Deduplicate by scheme (baseline scheme might appear with the
        # baseline score AND from its own OAT run if we ever add one)
        seen: set[str] = set()
        dedup: list[dict] = []
        for e in sorted_entries:
            if e["scheme"] in seen:
                continue
            seen.add(e["scheme"])
            dedup.append(e)
            if len(dedup) >= k:
                break
        winners[dim] = dedup
    return winners


def _cartesian(winners: dict[str, list[dict]]) -> list[dict]:
    """Build the full Cartesian product of winners across dimensions."""
    from itertools import product
    keys = list(DIMENSIONS)
    lists = [[(d, e["scheme"]) for e in winners[d]] for d in keys]
    combos: list[dict] = []
    for picks in product(*lists):
        overrides = {dim: scheme for (dim, scheme) in picks}
        # Combo name: conv-X_turb-Y_gwd-Z_micro-W
        name_parts = [
            f"{SHORT_DIM[dim]}-{overrides[dim]}" for dim in keys
        ]
        combo_name = "combo_" + "_".join(name_parts)
        combos.append({
            "name": combo_name,
            "overrides": overrides,
        })
    return combos


def _write_combo(repo_root: Path, combo_name: str, overrides: dict) -> str:
    combo_dir = repo_root / STAGE2_DIR_REL / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)
    suite_yaml = {
        "base": str(BASELINE_REL),
        "variants": ["classical"],
        "output_dir": str(STAGE2_RESULTS_REL / combo_name),
    }
    overlay_yaml = dict(overrides)
    with (combo_dir / "suite.yaml").open("w") as fh:
        yaml.safe_dump(suite_yaml, fh, sort_keys=False)
    with (combo_dir / "variant_classical.yaml").open("w") as fh:
        yaml.safe_dump(overlay_yaml, fh, sort_keys=False)
    return str(STAGE2_DIR_REL / combo_name / "suite.yaml")


def _write_runner_sbatch(repo_root: Path, manifest_rel: Path) -> Path:
    path = repo_root / "scripts" / "run" / "_aimip_sweep_stage2_runner.sbatch"
    content = f"""#!/bin/bash
#SBATCH --job-name=aimip_sweep_s2
#SBATCH --output=results/aimip_classical_sweep_stage2/slurm_logs/sweep_s2_%A_%a.out
#SBATCH --error=results/aimip_classical_sweep_stage2/slurm_logs/sweep_s2_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Stage-2 SLURM array runner.  Same shape as stage-1: picks one combo
# from the stage-2 manifest by SLURM_ARRAY_TASK_ID, dispatches
# run_aimip.py with --variants classical.

set -euo pipefail

# Hard-coded repo root: SLURM copies the script to a temp dir before
# executing, so $(dirname (dirname (readlink -f $0))) lands in
# /var/spool/slurmd/.../job_NNN rather than the repo.
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

MANIFEST="{manifest_rel}"
SUITE_PATH=$(/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c "
import json, sys
m = json.load(open('${{MANIFEST}}'))
print(m['combos'][int(sys.argv[1])]['suite'])
" "${{SLURM_ARRAY_TASK_ID}}")

mkdir -p results/aimip_classical_sweep_stage2/slurm_logs

export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:${{PYTHONPATH:-}}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

echo "[$(date)] task ${{SLURM_ARRAY_TASK_ID}} -> suite=${{SUITE_PATH}}"
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \\
    scripts/run/run_aimip.py \\
    --suite "${{SUITE_PATH}}" \\
    --variants classical
echo "[$(date)] task ${{SLURM_ARRAY_TASK_ID}} done"
"""
    path.write_text(content)
    path.chmod(0o755)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path("/burg-archive/glab/users/pg2328/legoESM"),
    )
    parser.add_argument(
        "--metric", type=str, default="loss",
        choices=["loss", "T_rmse", "T_bias_abs", "composite"],
        help="Scoring metric for picking stage-1 winners (default: 'loss').",
    )
    parser.add_argument(
        "--top-k", type=int, default=2,
        help="Number of winners per dimension to promote to stage-2 (default 2).",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Proceed even if some stage-1 scorecards are missing/invalid.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    stage1_manifest = _load_stage1_manifest(repo_root)

    baseline_score, scores_by_dim, missing = _collect_stage1_scores(
        repo_root, stage1_manifest, args.metric,
    )

    if missing:
        msg = f"Missing/invalid stage-1 scorecards ({len(missing)}):"
        for name, err in missing.items():
            msg += f"\n  {name}: {err}"
        if not args.allow_incomplete:
            print(msg, file=sys.stderr)
            print(
                "\nPass --allow-incomplete to proceed with partial results.",
                file=sys.stderr,
            )
            sys.exit(2)
        print("WARNING: " + msg, file=sys.stderr)

    if baseline_score is None:
        print("ERROR: baseline scorecard not readable.", file=sys.stderr)
        sys.exit(2)

    winners = _pick_top_k(scores_by_dim, k=args.top_k)

    print(f"Stage-1 metric: {args.metric}")
    print(f"Baseline score: {baseline_score:.6f}")
    print()
    print("Top-{} per dimension:".format(args.top_k))
    for dim in DIMENSIONS:
        print(f"  {SHORT_DIM[dim]:6s}:", end="")
        for e in winners[dim]:
            print(f"  {e['scheme']:<22s} ({e['score']:.4f})", end="")
        print()
    print()

    # Materialize stage-2 combos.
    stage2_combos = _cartesian(winners)
    print(f"Stage-2 Cartesian product: {len(stage2_combos)} combos")
    for combo in stage2_combos:
        combo["suite"] = _write_combo(repo_root, combo["name"], combo["overrides"])

    stage2_manifest = {
        "stage": 2,
        "metric": args.metric,
        "top_k": args.top_k,
        "winners": {dim: winners[dim] for dim in DIMENSIONS},
        "baseline_score": baseline_score,
        "n_combos": len(stage2_combos),
        "combos": stage2_combos,
        "results_dir": str(STAGE2_RESULTS_REL),
    }
    manifest_path = repo_root / STAGE2_DIR_REL / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(stage2_manifest, indent=2) + "\n")

    winners_audit = repo_root / STAGE2_DIR_REL / "stage1_winners.json"
    winners_audit.write_text(json.dumps({
        "metric": args.metric,
        "baseline_score": baseline_score,
        "all_scores": {dim: scores_by_dim[dim] for dim in DIMENSIONS},
        "winners": winners,
    }, indent=2) + "\n")

    runner_sbatch = _write_runner_sbatch(
        repo_root,
        manifest_rel=Path(STAGE2_DIR_REL) / "manifest.json",
    )

    print()
    print(f"Manifest: {manifest_path.relative_to(repo_root)}")
    print(f"Winners audit: {winners_audit.relative_to(repo_root)}")
    print(f"Runner sbatch: {runner_sbatch.relative_to(repo_root)}")
    print()
    print(f"Launch with:")
    print(f"  sbatch --array=0-{len(stage2_combos) - 1} scripts/run/_aimip_sweep_stage2_runner.sbatch")


if __name__ == "__main__":
    main()
