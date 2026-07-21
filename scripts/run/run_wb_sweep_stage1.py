#!/usr/bin/env python
"""Plan Stage-1 of the WeatherBench classical-physics OAT sweep (T63).

One-at-a-time (OAT) variation around the baseline
``tiedtke + louis + mcfarlane + sundqvist + xu_randall`` at T63 + RRTMGP,
training each swapped scheme's parameters on ERA5 with a multi-step FORECAST
objective (RMSE + CRPS over chained 6h rollouts). A CURATED 2-3 alternatives
per family are screened (not the full menu); ALL 5 physics categories stay
ACTIVE (no ``none`` control -- standing user directive).

Shares the planner core with ``run_aimip_classical_sweep_stage1.py`` via
``legoesm.training.sweep_planner`` (D6); only the paths, the curated
SWEEP_SPACE, and the runner sbatch (worktree root + walltime) live here.

Writes (idempotent, submits no jobs):
  config/wb/sweep/stage1/combo_<name>/suite.yaml
  config/wb/sweep/stage1/combo_<name>/variant_classical.yaml
  config/wb/sweep/stage1/manifest.json
  scripts/run/_wb_sweep_stage1_runner.sbatch

Launch after this runs:
    sbatch --array=0-<N> scripts/run/_wb_sweep_stage1_runner.sbatch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# packages/* namespace roots (the sweep planner core lives in
# legoesm.training.sweep_planner; PYTHONPATH may not carry them when this
# planner runs bare on a login node).
_REPO = Path(__file__).resolve().parents[2]
for _d in sorted((_REPO / "packages").glob("*/")):
    if (_d / "legoesm").is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from legoesm.training.sweep_planner import (  # noqa: E402
    build_oat_combos,
    validate_sweep_baseline,
    write_sweep_plan,
)

# Baseline scheme choices (the AIMIP classical recipe).
BASELINE = {
    "aimip_convection": "tiedtke",
    "aimip_turbulence": "louis",
    "aimip_gwd": "mcfarlane",
    "aimip_microphysics": "sundqvist",
    "aimip_cloud": "xu_randall",   # explicit 5th family so the manifest/analysis
                                   # sees the cloud OAT diff as xu_randall->sundqvist
}

# CURATED OAT alternatives per family (excludes the baseline value, added
# separately as combo_baseline). All 5 categories stay active -- no "none".
SWEEP_SPACE = {
    "aimip_convection": ["bechtold", "edmf"],          # + baseline tiedtke
    "aimip_turbulence": ["ysu", "edmf"],               # + baseline louis
    "aimip_gwd": ["hines"],                            # + baseline mcfarlane
    "aimip_microphysics": ["kessler", "morrison", "thompson"],  # + baseline sundqvist
    "aimip_cloud": ["sundqvist"],                      # + baseline xu_randall
}

WORKTREE = "/burg-archive/glab/users/pg2328/legoESM_wbforecast"
BASELINE_REL = Path("config/wb/sweep/stage1/baseline_classical_t63_rrtmgp.yaml")
SWEEP_DIR_REL = Path("config/wb/sweep/stage1")
RESULTS_DIR_REL = Path("results/wb_sweep_stage1")


def _write_runner_sbatch(repo_root: Path, manifest_rel: Path) -> Path:
    path = repo_root / "scripts" / "run" / "_wb_sweep_stage1_runner.sbatch"
    content = f"""#!/bin/bash
#SBATCH --job-name=wb_sweep_s1
#SBATCH --output=results/wb_sweep_stage1/slurm_logs/sweep_s1_%A_%a.out
#SBATCH --error=results/wb_sweep_stage1/slurm_logs/sweep_s1_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# SLURM array entry: each task trains one combo from the Stage-1 manifest at
# T63 + RRTMGP with the multi-step forecast loss.
#     sbatch --array=0-N scripts/run/_wb_sweep_stage1_runner.sbatch
# where N+1 = number of combos in the manifest.

set -euo pipefail

WORKTREE="{WORKTREE}"
cd "$WORKTREE"
# worktree-pinned env (jn2808 conda python + worktree packages on PYTHONPATH)
source "$WORKTREE/scripts/cluster/wb_forecast/env.sh"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

MANIFEST="{manifest_rel}"
SUITE_PATH=$("$WB_PYTHON" -c "
import json, sys
m = json.load(open('${{MANIFEST}}'))
print(m['combos'][int(sys.argv[1])]['suite'])
" "${{SLURM_ARRAY_TASK_ID}}")

mkdir -p results/wb_sweep_stage1/slurm_logs

echo "[$(date)] task ${{SLURM_ARRAY_TASK_ID}} -> suite=${{SUITE_PATH}}"
"$WB_PYTHON" scripts/run/run_aimip.py \\
    --suite "${{SUITE_PATH}}" \\
    --variants classical \\
    --resume
echo "[$(date)] task ${{SLURM_ARRAY_TASK_ID}} done"
"""
    path.write_text(content)
    path.chmod(0o755)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(WORKTREE))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_sweep_baseline(repo_root, BASELINE_REL)

    combos = build_oat_combos(BASELINE, SWEEP_SPACE)
    manifest_path = write_sweep_plan(
        repo_root,
        campaign="weatherbench",
        combos=combos,
        sweep_dir_rel=SWEEP_DIR_REL,
        results_dir_rel=RESULTS_DIR_REL,
        baseline=BASELINE,
        baseline_rel=BASELINE_REL,
    )

    runner_sbatch = _write_runner_sbatch(
        repo_root, manifest_rel=Path(SWEEP_DIR_REL) / "manifest.json")

    print(f"Wrote {len(combos)} combo configs under {SWEEP_DIR_REL}/")
    print(f"Manifest: {manifest_path.relative_to(repo_root)}")
    print(f"Runner sbatch: {runner_sbatch.relative_to(repo_root)}")
    print()
    print("Sweep summary:")
    for i, combo in enumerate(combos):
        print(f"  [{i:2d}] {combo['name']:35s} -> {combo['scheme']}")
    print()
    print("Launch with:")
    print(f"  sbatch --array=0-{len(combos) - 1} scripts/run/_wb_sweep_stage1_runner.sbatch")


if __name__ == "__main__":
    main()
