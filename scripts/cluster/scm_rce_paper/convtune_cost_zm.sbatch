#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_zmcost
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_zmcost_%j.out
set +e
# ---------------------------------------------------------------------------
# WHY zhang_mcfarlane cost 1114 s per 100-day tuning evaluation when the other
# nine cost 220-312 s (job 9376354).  Node contention is already refuted by the
# accounting: ZM's co-tenant on g235 exited after 6 minutes, so ZM had the node
# essentially to itself and was still the slowest.
#
# The structural reading is that ZM's dilute-parcel CAPE nests a 20-iteration
# Newton solve inside a 2x-unrolled loop inside a 74-level scan (8,880
# `_moist_entropy` evaluations per call), and that this is expensive in a
# SINGLE COLUMN specifically because there is no vector width to amortize it.
# That reading is a hypothesis until it survives a perturbation test, which is
# what this job runs: an ON/OFF control (use_dilute_cape), a SCALING control
# (_NEWTON_ITERS), and a VECTOR-WIDTH sweep (ncol = 1, 64, 1024).
#
# The unit tests run FIRST because the scaling control is the kind that lies
# silently: a monkeypatch that fails to retrace yields a flat scaling that
# reads as a refutation.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
OUT=${OUT:-$REPO/results/scm_rce_convtune/zm_cost}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
echo "cpus: ${SLURM_CPUS_PER_TASK:-?}"

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR" "$OUT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ bench gates ############"
JAX_ENABLE_X64=1 $PY -m pytest -q \
  $WT/tests/unit/test_bench_scm_convection_cost.py 2>&1 | tail -20
TEST_RC=${PIPESTATUS[0]}
echo "TEST_RC=$TEST_RC"
[ $TEST_RC -eq 0 ] || { echo "ABORT: bench gates red — its controls are not trustworthy"; exit $TEST_RC; }

echo; echo "############ cost bench ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/bench/bench_scm_convection_cost.py \
  --nlev 74 --dt 600 --repeats 12 --ncols 1 64 1024 \
  --newton-iters 5 10 20 \
  --out $OUT/zm_cost.json 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -80
BENCH_RC=${PIPESTATUS[0]}
echo "BENCH_RC=$BENCH_RC"
echo ALLDONE
