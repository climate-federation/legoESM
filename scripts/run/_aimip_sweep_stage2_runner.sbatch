#!/bin/bash
#SBATCH --job-name=aimip_sweep_s2
#SBATCH --output=results/aimip_classical_sweep_stage2/slurm_logs/sweep_s2_%A_%a.out
#SBATCH --error=results/aimip_classical_sweep_stage2/slurm_logs/sweep_s2_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Stage-2 SLURM array runner.  Same shape as stage-1: picks one combo
# from the stage-2 manifest by SLURM_ARRAY_TASK_ID, dispatches
# run_aimip.py with --variants classical.

set -euo pipefail

# Hard-coded repo root: SLURM copies the script to a temp dir before
# executing, so $(dirname (dirname (readlink -f $0))) lands in
# /var/spool/slurmd/.../job_NNN rather than the repo.
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

MANIFEST="config/aimip/sweep/stage2/manifest.json"
SUITE_PATH=$(/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c "
import json, sys
m = json.load(open('${MANIFEST}'))
print(m['combos'][int(sys.argv[1])]['suite'])
" "${SLURM_ARRAY_TASK_ID}")

mkdir -p results/aimip_classical_sweep_stage2/slurm_logs

export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:${PYTHONPATH:-}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} -> suite=${SUITE_PATH}"
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    scripts/run/run_aimip.py \
    --suite "${SUITE_PATH}" \
    --variants classical
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
