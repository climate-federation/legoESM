#!/bin/bash
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

MANIFEST="config/aimip/sweep/stage1/manifest.json"
SUITE_PATH=$(/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c "
import json, sys
m = json.load(open('${MANIFEST}'))
print(m['combos'][int(sys.argv[1])]['suite'])
" "${SLURM_ARRAY_TASK_ID}")

mkdir -p results/aimip_classical_sweep_stage1/slurm_logs

# Post-merge layout is packages/* (PEP420 namespace), NOT src/. Source the
# shared env helper so PYTHONPATH points at packages/* — the old hardcoded
# src/ path silently broke every array task (ModuleNotFoundError: legoesm).
source "$REPO_ROOT/scripts/cluster/scaling_ginsburg/_env.sh"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} -> suite=${SUITE_PATH}"
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    scripts/run/run_aimip.py \
    --suite "${SUITE_PATH}" \
    --variants classical
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
