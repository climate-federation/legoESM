#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=grad_audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/grad_audit_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# ONE CATEGORY PER ARRAY TASK.  The single-job form ran 2h37 at 10.7% CPU on a
# contended node with its output fully buffered behind `| tail`, so nothing was
# observable and a failure in the last category would have discarded the first
# four.  Per-category tasks land partial results, isolate failures, and can be
# re-run individually.
# ---------------------------------------------------------------------------
WT=${WT:?}; PIN=${PIN:?}
REPO=/burg-archive/glab/users/pg2328/legoESM
CATS=(convection turbulence microphysics radiation gravity_wave_drag)
CAT=${CATS[$SLURM_ARRAY_TASK_ID]}
cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)  category: $CAT"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
mkdir -p $REPO/results/param_grad_audit
# -u so progress is visible while it runs; no `| tail` swallowing the stream.
JAX_ENABLE_X64=1 $PY -u scripts/validate/audit_param_gradients.py \
  --categories $CAT \
  --json-out $REPO/results/param_grad_audit/${CAT}.json 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
RC=${PIPESTATUS[0]}
echo "AUDIT_STATUS=$([ $RC -eq 0 ] && echo OK || echo FAILED) (category=$CAT rc=$RC)"
echo ALLDONE
exit $RC
