#!/usr/bin/env python
"""Pick the best classical physics combo from stage-2, build the T106
3-way head-to-head (classical / column_nn / sfno_physics) configs +
sbatch launchers.

Selection
---------
Reads ``results/aimip_classical_sweep_stage2/combo_*/aimip_scorecard.json``,
picks the combo with the lowest eval ``loss['mean']`` (same metric the
training objective optimises).  Records the choice and runner-up
candidates in ``config/aimip/t106_headtohead/winner.json``.

Configs written
---------------
``config/aimip/t106_headtohead/``
  base.yaml                          — shared T106 + RRTMGP + multi-day base
  suite.yaml                         — lists [classical, column_nn, sfno_physics]
  variant_classical.yaml             — winning physics-scheme overrides
  variant_column_nn.yaml             — column_nn-specific knobs
  variant_sfno_physics.yaml          — sfno-specific knobs

Launchers written (one per variant for max parallelism)
-------------------------------------------------------
``scripts/run/run_aimip_headtohead_t106_classical.sbatch``
``scripts/run/run_aimip_headtohead_t106_column_nn.sbatch``
``scripts/run/run_aimip_headtohead_t106_sfno_physics.sbatch``

Submit any subset; they're independent and target distinct GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


STAGE2_MANIFEST_REL = Path("config/aimip/sweep/stage2/manifest.json")
STAGE2_RESULTS_REL = Path("results/aimip_classical_sweep_stage2")
HEADTOHEAD_DIR_REL = Path("config/aimip/t106_headtohead")
HEADTOHEAD_RESULTS_REL = Path("results/aimip_t106_headtohead")


def _load_winner(repo_root: Path, metric: str) -> tuple[dict, list[dict]]:
    """Return (winner_combo, ranking) from the stage-2 scorecards."""
    manifest_path = repo_root / STAGE2_MANIFEST_REL
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing stage-2 manifest at {manifest_path}.  Run "
            "scripts/run/run_aimip_classical_sweep_stage2.py first."
        )
    manifest = json.loads(manifest_path.read_text())

    ranked: list[dict] = []
    for combo in manifest["combos"]:
        sc_path = (
            repo_root / STAGE2_RESULTS_REL / combo["name"] / "aimip_scorecard.json"
        )
        if not sc_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        try:
            em = sc["variants"]["classical"]["eval_metrics"]
        except KeyError:
            continue
        if metric == "loss":
            score = float(em["loss"]["mean"])
        elif metric == "composite":
            T_rmse = float(em["rmse"]["T"]["mean"])
            T_bias = float(abs(em["bias"]["T"]["mean"]))
            u_rmse = float(em["rmse"]["u"]["mean"])
            v_rmse = float(em["rmse"]["v"]["mean"])
            score = T_rmse + 2.0 * T_bias + 0.5 * (u_rmse + v_rmse)
        else:
            raise ValueError(f"Unknown metric {metric!r}")
        ranked.append({
            "name": combo["name"],
            "overrides": combo["overrides"],
            "score": score,
            "T_rmse": float(em["rmse"]["T"]["mean"]),
            "T_bias": float(em["bias"]["T"]["mean"]),
        })

    if not ranked:
        raise RuntimeError(
            f"No stage-2 scorecards found under {STAGE2_RESULTS_REL}.  "
            "Wait for stage-2 jobs to finish first."
        )
    ranked.sort(key=lambda e: e["score"])
    return ranked[0], ranked


def _write_base_yaml(repo_root: Path, winning_overrides: dict) -> Path:
    """Write the T106 + RRTMGP + multi-day base config (winner-aware)."""
    base = {
        "era5_zarr": (
            "gs://weatherbench2/datasets/era5/"
            "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        ),
        "cache_dir": ".cache/aimip_era5",
        "train_windows": [
            [2015, 0, 3], [2015, 180, 3], [2016, 0, 3], [2016, 180, 3],
        ],
        "eval_windows": [[2017, 0, 3], [2017, 180, 3]],
        "train_year": 2015,
        "eval_year": 2017,
        "train_period": ["2015-01-01", "2016-12-31"],
        "eval_period": ["2017-01-01", "2017-12-31"],
        "n_train_days": 12,
        "n_eval_days": 6,
        "aimip_grid": "gaussian",
        "n_max": 106,
        "nlev": 8,
        "dt": 400.0,
        "aimip_n_epochs": 4,
        "aimip_patience": 2,
        "aimip_min_delta": 1.0e-3,
        "aimip_lr": 3.0e-4,
        "aimip_warmup": 10,
        "aimip_weight_decay": 1.0e-5,
        "aimip_grad_clip": 1.0,
        "aimip_optimizer": "adamw",
        # User instruction: RRTMGP everywhere.
        "aimip_radiation": "rrtmgp",
        "aimip_rad_update_interval": 12,
        "aimip_spatial_surface": False,
        "aimip_spatial_init_std": 0.0,
        "aimip_spatial_seed": 0,
        "aimip_spatial_lr_scale": 1.0,
        "aimip_rollout_days": 1,
        "aimip_rollout_hours": 6,
        # Winning classical schemes (column_nn / sfno variants override
        # the classical scheme keys but inherit everything else).
        "aimip_convection": winning_overrides["aimip_convection"],
        "aimip_turbulence": winning_overrides["aimip_turbulence"],
        "aimip_gwd": winning_overrides["aimip_gwd"],
        "aimip_microphysics": winning_overrides["aimip_microphysics"],
        "aimip_cloud": "xu_randall",
        "loss": {
            "w_T": 1.0, "w_u": 0.5, "w_v": 0.5, "w_q": 0.3, "w_ps": 0.3,
            "w_bias_T": 100.0, "w_bias_u": 1.0, "w_bias_v": 1.0, "w_bias_ps": 1.0,
            "w_crps_T": 0.5, "w_crps_u": 0.25, "w_crps_v": 0.25,
            "w_crps_q": 0.5, "w_crps_ps": 0.15,
            "w_spec_crps_T": 0.5, "w_spec_crps_u": 0.25,
            "w_spec_crps_v": 0.25, "w_spec_crps_q": 0.5,
            "multi_step_hours": [24, 48, 72],
            "multi_step_weights": [1.0, 0.7, 0.5],
            "spectral_weight": 0.0,
            "level_weighting": "pressure",
            "normalize_by_scale": True,
            "T_scale": 30.0, "wind_scale": 20.0,
            "q_scale": 5.0e-3, "ps_scale": 1000.0,
        },
    }
    path = repo_root / HEADTOHEAD_DIR_REL / "base.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(base, fh, sort_keys=False)
    return path


def _write_variants(repo_root: Path) -> None:
    """Write the 3 variant overlays (empty for classical since the winner
    is baked into base; explicit knobs for column_nn / sfno_physics)."""
    out_dir = repo_root / HEADTOHEAD_DIR_REL

    suite_yaml = {
        "base": str(HEADTOHEAD_DIR_REL / "base.yaml"),
        "variants": ["classical", "column_nn", "sfno_physics"],
        "output_dir": str(HEADTOHEAD_RESULTS_REL),
    }
    with (out_dir / "suite.yaml").open("w") as fh:
        yaml.safe_dump(suite_yaml, fh, sort_keys=False)

    # Classical: empty (uses base scheme keys + AIMIPClassicalParams defaults)
    with (out_dir / "variant_classical.yaml").open("w") as fh:
        yaml.safe_dump({}, fh)

    # Column NN: AIMIP-style MLP.  Matches the production multistep suite.
    with (out_dir / "variant_column_nn.yaml").open("w") as fh:
        yaml.safe_dump({
            "column_nn_hidden_dim": 256,
            "column_nn_n_layers": 4,
        }, fh, sort_keys=False)

    # SFNO: 34M parameter model (matches the AIMIP scorecard SFNO entry).
    with (out_dir / "variant_sfno_physics.yaml").open("w") as fh:
        yaml.safe_dump({
            "sfno_embed_dim": 32,
            "sfno_n_blocks": 2,
            "aimip_optimizer": "muon_partitioned",
        }, fh, sort_keys=False)


def _write_sbatch(repo_root: Path, variant: str, walltime_h: int = 14) -> Path:
    path = repo_root / "scripts" / f"run_aimip_headtohead_t106_{variant}.sbatch"
    content = f"""#!/bin/bash
#SBATCH --job-name=h2h_{variant}
#SBATCH --output={HEADTOHEAD_RESULTS_REL}/slurm_logs/{variant}_%j.out
#SBATCH --error={HEADTOHEAD_RESULTS_REL}/slurm_logs/{variant}_%j.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time={walltime_h:02d}:00:00

# T106 head-to-head: trains the {variant} variant with the new
# multi-day + bias + spectral-CRPS loss and RRTMGP radiation.
# Configs: {HEADTOHEAD_DIR_REL}/(base.yaml + suite.yaml + variant_{variant}.yaml)

set -euo pipefail
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

mkdir -p {HEADTOHEAD_RESULTS_REL}/slurm_logs

export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:${{PYTHONPATH:-}}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \\
    scripts/run/run_aimip.py \\
    --suite {HEADTOHEAD_DIR_REL}/suite.yaml \\
    --variants {variant}
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
        "--metric", default="loss", choices=["loss", "composite"],
        help="Metric for picking the stage-2 winner (default: 'loss').",
    )
    parser.add_argument(
        "--require-results", action="store_true",
        help=(
            "Fail if stage-2 scorecards are not yet available.  Without "
            "this flag, the script will still write configs using the "
            "stage-2 manifest's first combo as a placeholder so you can "
            "review file layout / sbatch contents ahead of time."
        ),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    try:
        winner, ranking = _load_winner(repo_root, args.metric)
        print(f"Winner ({args.metric}): {winner['name']}")
        print(f"  schemes: {winner['overrides']}")
        print(f"  score:   {winner['score']:.6f}")
        print(f"  T_rmse:  {winner['T_rmse']:.4f} K, T_bias: {winner['T_bias']:+.4f} K")
        print()
        print(f"Stage-2 ranking (top 5):")
        for r in ranking[:5]:
            print(
                f"  {r['name']:60s} score={r['score']:.4f} "
                f"T_rmse={r['T_rmse']:.3f} T_bias={r['T_bias']:+.3f}"
            )
        overrides = winner["overrides"]
    except (FileNotFoundError, RuntimeError) as exc:
        if args.require_results:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        print(
            f"WARNING: {exc}\n"
            "Using placeholder overrides (tiedtke+louis+mcfarlane+sundqvist) "
            "so configs/sbatch can be reviewed.  Re-run after stage-2 "
            "finishes to bake in real winner.",
            file=sys.stderr,
        )
        overrides = {
            "aimip_convection": "tiedtke",
            "aimip_turbulence": "louis",
            "aimip_gwd": "mcfarlane",
            "aimip_microphysics": "sundqvist",
        }
        ranking = []

    base_path = _write_base_yaml(repo_root, overrides)
    _write_variants(repo_root)
    sbatch_paths = [
        _write_sbatch(repo_root, "classical"),
        _write_sbatch(repo_root, "column_nn"),
        _write_sbatch(repo_root, "sfno_physics"),
    ]

    # Audit trail.
    winner_audit = {
        "metric": args.metric,
        "winner": overrides,
        "ranking_top5": ranking[:5] if ranking else None,
        "config_dir": str(HEADTOHEAD_DIR_REL),
        "results_dir": str(HEADTOHEAD_RESULTS_REL),
    }
    (repo_root / HEADTOHEAD_DIR_REL / "winner.json").write_text(
        json.dumps(winner_audit, indent=2) + "\n"
    )

    print()
    print(f"Base config:  {base_path.relative_to(repo_root)}")
    print(f"Sbatch files written:")
    for p in sbatch_paths:
        print(f"  {p.relative_to(repo_root)}")
    print()
    print("Launch (parallel, 3 separate GPUs):")
    for p in sbatch_paths:
        print(f"  sbatch {p.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
