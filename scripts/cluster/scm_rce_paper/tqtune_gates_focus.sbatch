#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_gf
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_gf_%j.out
set +e
# Focused re-run of the tests under repair, with FULL tracebacks (the main gate
# job tails its output, which is why the first failure text was unreadable).
WT=${WT:?}; PIN=${PIN:?}
cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY -m pytest -q ${TESTS:?} 2>&1 | tail -120
echo "RC=${PIPESTATUS[0]}"
echo ALLDONE
