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
import json
from pathlib import Path

import yaml


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


def _combo_name(dim: str, scheme: str) -> str:
    """Per-combo dirname -- short, ASCII, sortable."""
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
    """Write the per-combo suite.yaml + variant_classical.yaml pair."""
    combo_dir = repo_root / SWEEP_DIR_REL / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)

    suite_yaml = {
        "base": str(baseline_rel),
        "variants": ["classical"],
        "output_dir": str(results_dir_rel / combo_name),
    }
    overlay_yaml = dict(overrides)  # keys are aimip_* scheme literals

    with (combo_dir / "suite.yaml").open("w") as fh:
        yaml.safe_dump(suite_yaml, fh, sort_keys=False)
    with (combo_dir / "variant_classical.yaml").open("w") as fh:
        yaml.safe_dump(overlay_yaml, fh, sort_keys=False)


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
    baseline_path = repo_root / BASELINE_REL
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline: {baseline_path}")

    # Build the combo list.  "combo_baseline" goes first so the
    # baseline scorecard is index 0; the OAT alternatives follow.
    combos: list[dict] = []
    combos.append({
        "name": "combo_baseline",
        "dim": "baseline",
        "scheme": "tiedtke+louis+mcfarlane+sundqvist",
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

    # Write per-combo config pairs.
    for combo in combos:
        _write_combo(
            repo_root, combo["name"], combo["overrides"],
            baseline_rel=BASELINE_REL,
            results_dir_rel=RESULTS_DIR_REL,
        )
        combo["suite"] = str(SWEEP_DIR_REL / combo["name"] / "suite.yaml")

    manifest = {
        "stage": 1,
        "baseline": dict(BASELINE),
        "baseline_yaml": str(BASELINE_REL),
        "results_dir": str(RESULTS_DIR_REL),
        "n_combos": len(combos),
        "combos": combos,
    }
    manifest_path = repo_root / SWEEP_DIR_REL / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

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
