#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_merge_%j.out
set +e
# ---------------------------------------------------------------------------
# Merge one seed's per-scheme checkpoints into the table + figures, then -- if
# BOTH seeds are present -- report the seed-to-seed disagreement, which is the
# honest statement about how much of the ranking is search rather than physics.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OBJECTIVE=${OBJECTIVE:-thermo}
PARAM_SET=${PARAM_SET:-physical}
SEEDS=${SEEDS:-"20260815 20260816"}
# Protocol tag, part of the results directory since the nonrotating (f0)
# correction: a corrected run and a pinned one are NOT comparable term by term,
# so they cannot share a directory or a merged table.
PROTO_TAG=${PROTO_TAG:-}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
echo "host: $(hostname)  date: $(date -Is)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }

export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR" || { echo "PREFLIGHT FAIL: cannot create TMPDIR"; exit 89; }
trap 'rm -rf /burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}' EXIT

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RC_ALL=0
for SEED in $SEEDS; do
  if [ -n "$PROTO_TAG" ]; then
    OUT=$REPO/results/scm_rce_tqtune/arm_${OBJECTIVE}_${PARAM_SET}_${PROTO_TAG}_seed${SEED}
  else
    OUT=$REPO/results/scm_rce_tqtune/arm_${OBJECTIVE}_${PARAM_SET}_seed${SEED}
  fi
  echo "=== MERGE seed=$SEED  $OUT ==="
  [ -d "$OUT" ] || { echo "SKIP: $OUT does not exist"; continue; }
  JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
    --reference-dir $REF --outdir $OUT --merge-only \
    2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -40
  RC=${PIPESTATUS[0]}
  echo "MERGE_RC(seed=$SEED)=$RC"
  [ $RC -eq 0 ] || RC_ALL=$RC
  CSV=$(ls $OUT/*.csv 2>/dev/null | head -1)
  if [ -n "$CSV" ]; then
    for HUM in logq rh qv; do
      JAX_ENABLE_X64=1 $PY $WT/scripts/plot/plot_scm_rce_tq_ranking.py "$CSV" \
        --out $OUT/tq_ranking_${HUM}.png --humidity $HUM --log-x \
        --suptitle "SCM-RCE convection tuned to CRM T and humidity (seed $SEED${PROTO_TAG:+, $PROTO_TAG})" \
        2>&1 | tail -2
    done
  fi
done

# --- seed-to-seed disagreement --------------------------------------------
# Reported, never hidden: a uniform search over 20-26 parameters cannot be
# called converged, and the size of the rank churn between two seeds is the
# measurement that says how much of the ranking to believe.
#
# SKIPPED, not failed, for a single seed.  The reporter correctly refuses fewer
# than two seeds, and a merge of one complete seed is a legitimate deliverable
# in its own right -- letting that exit status mark the whole job FAILED would
# hide a successful merge behind a step that was never applicable.
N_SEEDS=$(echo $SEEDS | wc -w)
if [ "$N_SEEDS" -lt 2 ]; then
  echo "=== SEED AGREEMENT: SKIPPED ($N_SEEDS seed) ==="
  echo "Needs two independent seeds; run again when the second completes."
  echo "MERGE_STATUS=$([ $RC_ALL -eq 0 ] && echo OK || echo FAILED)"
  echo ALLDONE
  exit $RC_ALL
fi
echo "=== SEED AGREEMENT ==="
JAX_ENABLE_X64=1 $PY $WT/scripts/validate/scm_rce_tq_seed_agreement.py \
  --arm-dir $REPO/results/scm_rce_tqtune \
  --objective $OBJECTIVE --param-set $PARAM_SET --seeds $SEEDS \
  ${PROTO_TAG:+--arm-suffix $PROTO_TAG} \
  --out $REPO/results/scm_rce_tqtune/seed_agreement.json 2>&1 | tail -40
ARC=${PIPESTATUS[0]}
echo "SEED_AGREEMENT_RC=$ARC"
[ $ARC -eq 0 ] || RC_ALL=$ARC
echo "MERGE_STATUS=$([ $RC_ALL -eq 0 ] && echo OK || echo FAILED)"
echo ALLDONE
exit $RC_ALL
