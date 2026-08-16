#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=grad_audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/grad_audit_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY scripts/validate/audit_param_gradients.py \
  --categories ${CATEGORIES:-all} \
  --json-out $REPO/results/param_grad_audit/atm_audit.json \
  2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -120
echo "RC=${PIPESTATUS[0]}"
echo ALLDONE
