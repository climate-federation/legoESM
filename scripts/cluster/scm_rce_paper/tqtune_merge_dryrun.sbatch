#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_dry
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_dry_%j.out
set +e
# ---------------------------------------------------------------------------
# Exercise the REPORTING chain (merge -> CSV -> summary -> ranking figure) on
# whatever scheme checkpoints exist so far, into a SCRATCH outdir.  The point is
# that the final merge should not be the first time this code runs on real data;
# --allow-partial is required and correct here because the table IS partial.
# Writes nowhere near the production arm directories.
# ---------------------------------------------------------------------------
WT=${WT:?}; PIN=${PIN:?}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
SRC=${SRC:-$REPO/results/scm_rce_tqtune/arm_thermo_physical_seed20260815}
OUT=${OUT:-$REPO/results/scm_rce_tqtune/_dryrun}

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

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$SRC"/scheme_*.json "$SRC"/run_meta.json "$OUT"/ 2>/dev/null
echo "copied $(ls $OUT/scheme_*.json 2>/dev/null | wc -l) checkpoint(s)"

JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT --merge-only --allow-partial \
  2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -30
echo "MERGE_RC=${PIPESTATUS[0]}"

CSV=$(ls $OUT/*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] || { echo "DRYRUN FAILED: no CSV"; exit 96; }
for HUM in logq rh qv; do
  JAX_ENABLE_X64=1 $PY $WT/scripts/plot/plot_scm_rce_tq_ranking.py "$CSV" \
    --out $OUT/tq_ranking_${HUM}.png --humidity $HUM --log-x \
    --suptitle "DRY RUN - partial table" 2>&1 | tail -2
done
echo "=== artifacts ==="
ls -la $OUT/*.png $OUT/*.csv $OUT/*.md 2>/dev/null | awk '{print $5, $9}'
echo ALLDONE
