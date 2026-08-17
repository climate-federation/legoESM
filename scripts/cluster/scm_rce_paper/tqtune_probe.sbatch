#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_probe_%j.out
set +e
# ---------------------------------------------------------------------------
# The two measurements the T/q tuning objective depends on, plus the probe's
# own controls.  Gates run FIRST: an instrument whose controls have not passed
# is not allowed to print a number that will be quoted.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=${REF:-$REPO/results/rcemip_ref_sam300}
OUT=${OUT:-$REPO/results/scm_rce_tqtune/objective_probe.json}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
echo "host: $(hostname)  date: $(date -Is)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
[ -d "$REF/snapshots3d" ] || { echo "PREFLIGHT FAIL: no reference at $REF"; exit 94; }

export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR" || { echo "PREFLIGHT FAIL: cannot create TMPDIR"; exit 89; }
trap 'rm -rf /burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}' EXIT

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== GATES ==="
JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/unit/test_scm_rce_tq_objective_probe.py 2>&1 | tail -5
GATE_RC=${PIPESTATUS[0]}
[ $GATE_RC -eq 0 ] || { echo "GATES FAILED rc=$GATE_RC - probe output NOT trustworthy"; exit 95; }

echo "=== PROBE ==="
JAX_ENABLE_X64=1 $PY scripts/validate/scm_rce_tq_objective_probe.py \
  --reference-dir $REF \
  --last-reference-files 5 \
  --out $OUT 2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
RC=${PIPESTATUS[0]}
echo "PROBE_STATUS=$([ $RC -eq 0 ] && echo OK || echo FAILED) (rc=$RC)"
echo ALLDONE
exit $RC
