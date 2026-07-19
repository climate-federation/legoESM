#!/usr/bin/env python
"""Plan stage-1 of the AIMIP classical physics combinatorial sweep.

Stage-1 strategy
----------------
One-at-a-time (OAT) variation around the baseline
``tiedtke + louis + mcfarlane + sundqvist + rrtmgp + xu_randall`` at
T21.  For each of the four dimensions (convection, turbulence, gwd,
microphysics) we run every alternative scheme exactly once while
holding the other three at the baseline.  Total = 28 runs (the
baseline itself is run once and shared across dimensions).

Per-run cost at T21 + RRTMGP + 3-day rollout (4 epochs, 48 samples):
- ~40k dycore steps total per run
- ~1-3h on a single RTX 8000 (rough estimate from the existing T21
  multistep smoke runs, scaled for the longer 3-day rollout)
- 28 runs -> ~28-84 GPU-h total

Outputs
-------
This script writes (idempotent):
  config/aimip/sweep/stage1/combo_<name>/suite.yaml
  config/aimip/sweep/stage1/combo_<name>/variant_classical.yaml
  config/aimip/sweep/stage1/manifest.json
  scripts/run/_aimip_sweep_stage1_runner.sbatch
  scripts/run_aimip_classical_sweep_stage1.sbatch

Launching
---------
After this script runs (writes configs only, no jobs submitted):
::

    sbatch --array=0-27 scripts/run/_aimip_sweep_stage1_runner.sbatch

The array job index selects one combo from the manifest and runs
``scripts/run/run_aimip.py --suite <combo>/suite.yaml --variants classical``
under JAX/CUDA on a single GPU.

After stage-1 finishes, parse the per-combo scorecards with
``scripts/run/run_aimip_classical_sweep_stage2.py`` (TBD) which picks
top-2 per dimension and launches the 16-run full-Cartesian stage-2
sweep.
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


# Baseline scheme choices (matches the production AIMIP classical recipe).
BASELINE = {
    "aimip_convection": "tiedtke",
    "aimip_turbulence": "louis",
    "aimip_gwd": "mcfarlane",
    "aimip_microphysics": "sundqvist",
}

# OAT alternatives per dimension.  Each list excludes the baseline (which
# is added separately as a single "combo_baseline" run).  Excludes
# ml_emulator on GWD + microphysics because they're not part of the
# physics-comparison sweep -- the head-to-head includes column_nn + sfno
# as their own variants.  Excludes "none" on convection because the
# dynamics-only baseline is a degenerate case for an ESM physics
# comparison.
# NO "none" in any category — user directive: always keep ALL 5
# parameterization categories ACTIVE; the sweep compares real schemes only
# (radiation is fixed to RRTMGP, not a swept axis).
SWEEP_SPACE = {
    "aimip_convection": [
        "sbm", "dca", "kuo", "mass_flux", "edmf",
        "zhang_mcfarlane", "kain_fritsch", "emanuel", "bechtold",
    ],  # 9 alternatives + baseline tiedtke
    "aimip_turbulence": [
        "tke", "smagorinsky", "clubb_lite",
        "holtslag_boville", "ysu", "edmf",
    ],  # 6 alternatives + baseline louis
    "aimip_gwd": [
        "lindzen", "rayleigh", "hines", "prognostic_spectral",
    ],  # 4 alternatives + baseline mcfarlane (skip ml_emulator)
    "aimip_microphysics": [
        "kessler", "seifert_beheng", "morrison", "thompson",
    ],  # 4 alternatives + baseline sundqvist (skip ml_emulator)
    "aimip_cloud": [
        "sundqvist",
    ],  # 1 alternative + baseline xu_randall ('resolved' needs explicit
        # condensate micro -> invalid with the diagnostic sundqvist micro)
}

BASELINE_REL = Path("config/aimip/sweep/stage1/baseline_classical_t21_rrtmgp.yaml")
SWEEP_DIR_REL = Path("config/aimip/sweep/stage1")
RESULTS_DIR_REL = Path("results/aimip_classical_sweep_stage1")


def _write_runner_sbatch(repo_root: Path, manifest_rel: Path) -> Path:
    """Write the SLURM array runner that picks one combo from the manifest."""
    path = repo_root / "scripts" / "run" / "_aimip_sweep_stage1_runner.sbatch"
    content = f"""#!/bin/bash
#SBATCH --job-name=aimip_sweep_s1
#SBATCH --output=results/aimip_classical_sweep_stage1/slurm_logs/sweep_s1_%A_%a.out
#SBATCH --error=results/aimip_classical_sweep_stage1/slurm_logs/sweep_s1_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# SLURM array entry: each task picks one combo from the stage-1 manifest
# (config/aimip/sweep/stage1/manifest.json) and launches run_aimip.py
# with --variants classical on the corresponding suite YAML.
#
# Submit with:
#     sbatch --array=0-N scripts/run/_aimip_sweep_stage1_runner.sbatch
# where N+1 = number of combos in the manifest (currently 28; counted
# at submission time).

set -euo pipefail

# Hard-coded repo root: SLURM copies the script to a temp dir before
# executing, so "$(dirname "$(dirname "$(readlink -f "$0")")")" lands
# in /var/spool/slurmd/.../job_NNN rather than the repo.
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

MANIFEST="{manifest_rel}"
SUITE_PATH=$(/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c "
import json, sys
m = json.load(open('${{MANIFEST}}'))
print(m['combos'][int(sys.argv[1])]['suite'])
" "${{SLURM_ARRAY_TASK_ID}}")

mkdir -p results/aimip_classical_sweep_stage1/slurm_logs

# Post-merge layout is packages/* (PEP420 namespace), NOT src/. Source the
# shared env helper so PYTHONPATH points at packages/* — the old hardcoded
# src/ path silently broke every array task (ModuleNotFoundError: legoesm).
source "$REPO_ROOT/scripts/cluster/scaling_ginsburg/_env.sh"
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
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_sweep_baseline(repo_root, BASELINE_REL)

    # "combo_baseline" goes first so the baseline scorecard is index 0;
    # the OAT alternatives follow (shared planner core, D6).
    combos = build_oat_combos(BASELINE, SWEEP_SPACE)
    manifest_path = write_sweep_plan(
        repo_root,
        campaign="aimip",
        combos=combos,
        sweep_dir_rel=SWEEP_DIR_REL,
        results_dir_rel=RESULTS_DIR_REL,
        baseline=BASELINE,
        baseline_rel=BASELINE_REL,
    )

    runner_sbatch = _write_runner_sbatch(
        repo_root,
        manifest_rel=Path(SWEEP_DIR_REL) / "manifest.json",
    )

    print(f"Wrote {len(combos)} combo configs under {SWEEP_DIR_REL}/")
    print(f"Manifest: {manifest_path.relative_to(repo_root)}")
    print(f"Runner sbatch: {runner_sbatch.relative_to(repo_root)}")
    print()
    print("Sweep summary:")
    for i, combo in enumerate(combos):
        print(f"  [{i:2d}] {combo['name']:35s} -> {combo['scheme']}")
    print()
    print(f"Launch with:")
    print(f"  sbatch --array=0-{len(combos) - 1} scripts/run/_aimip_sweep_stage1_runner.sbatch")


if __name__ == "__main__":
    main()
