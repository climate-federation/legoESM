#!/usr/bin/env python
"""Plan Stage-1 of the WeatherBench classical-physics OAT sweep (T63).

One-at-a-time (OAT) variation around the baseline
``tiedtke + louis + mcfarlane + sundqvist + xu_randall`` at T63 + RRTMGP,
training each swapped scheme's parameters on ERA5 with a multi-step FORECAST
objective (RMSE + CRPS over chained 6h rollouts). A CURATED 2-3 alternatives
per family are screened (not the full menu); ALL 5 physics categories stay
ACTIVE (no ``none`` control -- standing user directive).

Cloned from ``run_aimip_classical_sweep_stage1.py``; only the paths, the curated
SWEEP_SPACE, and the runner (worktree root + walltime) differ. The per-family
swap logic is identical and grid-agnostic.

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
import json
from pathlib import Path

import yaml

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


def _combo_name(dim: str, scheme: str) -> str:
    short_dim = {
        "aimip_convection": "conv",
        "aimip_turbulence": "turb",
        "aimip_gwd": "gwd",
        "aimip_microphysics": "micro",
        "aimip_cloud": "cloud",
    }[dim]
    return f"combo_{short_dim}_{scheme}"


def _write_combo(repo_root: Path, combo_name: str, overrides: dict, *,
                 baseline_rel: Path, results_dir_rel: Path) -> None:
    combo_dir = repo_root / SWEEP_DIR_REL / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)
    suite_yaml = {
        "base": str(baseline_rel),
        "variants": ["classical"],
        "output_dir": str(results_dir_rel / combo_name),
    }
    with (combo_dir / "suite.yaml").open("w") as fh:
        yaml.safe_dump(suite_yaml, fh, sort_keys=False)
    with (combo_dir / "variant_classical.yaml").open("w") as fh:
        yaml.safe_dump(dict(overrides), fh, sort_keys=False)


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
    baseline_path = repo_root / BASELINE_REL
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline: {baseline_path}")

    combos: list[dict] = []
    combos.append({
        "name": "combo_baseline",
        "dim": "baseline",
        "scheme": "tiedtke+louis+mcfarlane+sundqvist+xu_randall",
        "overrides": dict(BASELINE),
    })
    for dim, alts in SWEEP_SPACE.items():
        for alt in alts:
            overrides = dict(BASELINE)
            overrides[dim] = alt
            combos.append({
                "name": _combo_name(dim, alt),
                "dim": dim,
                "scheme": alt,
                "overrides": overrides,
            })

    for combo in combos:
        _write_combo(
            repo_root, combo["name"], combo["overrides"],
            baseline_rel=BASELINE_REL, results_dir_rel=RESULTS_DIR_REL,
        )
        combo["suite"] = str(SWEEP_DIR_REL / combo["name"] / "suite.yaml")

    manifest = {
        "stage": 1,
        "campaign": "weatherbench",
        "baseline": dict(BASELINE),
        "baseline_yaml": str(BASELINE_REL),
        "results_dir": str(RESULTS_DIR_REL),
        "n_combos": len(combos),
        "combos": combos,
    }
    manifest_path = repo_root / SWEEP_DIR_REL / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # SLURM opens #SBATCH --output/--error before the job body runs, so the log
    # dir must exist at submission time.
    (repo_root / RESULTS_DIR_REL / "slurm_logs").mkdir(parents=True, exist_ok=True)

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
