#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=param_grad
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/param_grad_%j.out
set +e
# ---------------------------------------------------------------------------
# Is every declared tunable parameter reachable by a tuner AND differentiable?
#
# One reverse-mode pass per (scheme, column) yields the gradient for every
# parameter of that scheme at once, so the whole registry is affordable; the
# finite-difference sweep runs ONLY for parameters whose gradient came back
# exactly zero, which is what separates "read but the derivative is severed"
# from "not read at all".
#
# fp64 is not optional here: a true-but-tiny gradient can underflow to exactly
# 0.0 in fp32 and would be published as a DEAD parameter.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha>}
REPO=/burg-archive/glab/users/pg2328/legoESM
OUT=${OUT:-$REPO/results/param_audit}
CATS=${CATS:-all}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}
mkdir -p "$OUT"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export MPLBACKEND=Agg
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ audit gates ############"
$PY -m pytest -q $WT/tests/unit/test_audit_param_gradients.py 2>&1 | tail -20
RC=${PIPESTATUS[0]}; echo "GATES_RC=$RC"
[ $RC -eq 0 ] || { echo "ABORT: audit gates red"; exit $RC; }

echo; echo "############ registry-wide gradient audit ############"
$PY $WT/scripts/validate/audit_param_gradients.py \
  --categories $CATS --json-out $OUT/param_gradients_$CATS.json 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader\|CUDA_ERROR_NO_DEVICE\|xla_bridge\|plugin_module\|_check_cuda_versions\|cuda_device_count\|Jax plugin configuration error\|Traceback (most recent call last)\|^  File \|^    "
echo "AUDIT_RC=${PIPESTATUS[0]}"

echo; echo "COMPLETED"
