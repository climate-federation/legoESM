#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gust_gates
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/gust_gates_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA  host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY -m pytest ${GATES:-tests/core/unit/test_convective_gust_wind.py} \
  -x -q 2>&1 | tail -25
echo "RC=${PIPESTATUS[0]}"
echo ALLDONE
