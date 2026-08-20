#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_mrg
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_merge_%j.out
set +e
# Merges the per-scheme arm-(a)/(b) checkpoints into summary.md,
# intercomparison.csv, tuned_parameters.json and the figures.  arm (c) reads
# tuned_parameters.json from here, so this MUST run before it.
WT=/burg/glab/users/pg2328/scmrce_wt_campaign
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
# Arms of 9311347 died with `LLVM ERROR: Unable to allocate section memory!`
# after 0.5-3.7 h — the XLA CPU JIT's allocator, not the model, which is why
# raising --mem alone did not help. A persistent cache reuses compilations.
export JAX_COMPILATION_CACHE_DIR=/burg/glab/users/pg2328/.jax_cache_scmrce
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ARM=${ARM:-as_shipped}
REF=${REF:-/burg-archive/glab/users/pg2328/legoESM/results/rcemip_ref_sam300}
# OUTBASE lets the substep-control arm merge ITS OWN directory rather than
# the default one -- merging the wrong outdir would silently publish the
# default-substep results under the control's name.
OUTBASE=${OUTBASE:-arms_ab}
OUT=/burg-archive/glab/users/pg2328/legoESM/results/scm_rce_paper/${OUTBASE}_${ARM}
mkdir -p "$OUT"
# COMPLETENESS GATE. run_scm_rce_convection_intercomparison.py --merge-only
# collects whatever scheme_*.json checkpoints happen to exist and raises only
# when there are ZERO of them. So a run where 3 of 10 schemes aborted still
# produces a clean-looking ranking of the 7 survivors, with nothing in
# summary.md saying which are missing -- and since merge is chained afterany on
# the arms (deliberately, so one dead scheme does not sink the round), that is
# the DEFAULT path, not an edge case. A ranking silently computed over a subset
# is worse than no ranking: it is publishable and wrong.
SCHEMES="sbm dca kuo mass_flux edmf zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold"
MISSING=""
for s in $SCHEMES; do
  [ -s "$OUT/scheme_${s}.json" ] || MISSING="$MISSING $s"
done
if [ -n "$MISSING" ]; then
  echo "INCOMPLETE: missing scheme checkpoints:$MISSING"
  if [ "${ALLOW_PARTIAL:-0}" != "1" ]; then
    echo "STAGE_STATUS=FAILED (incomplete scheme set; rerun the failed arms, or"
    echo "  set ALLOW_PARTIAL=1 to publish a table explicitly labelled partial)"
    echo ALLDONE; exit 5
  fi
  echo "ALLOW_PARTIAL=1 -- publishing a PARTIAL ranking on purpose."
fi
echo "KERNEL_ARM=$ARM  OUT=$OUT"
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT --subsidence-solve $ARM \
  --days 100 --dt 600 --analysis-days 5 --last-reference-files 5 \
  --radiation rrtmgp --tune-evals 48 --tune-seed 20260705 \
  --scm-convection-substeps ${SUBSTEPS:-10} \
  --merge-only > "$OUT/.merge.log" 2>&1
RC=$?
tail -60 "$OUT/.merge.log"
echo "merge rc=$RC"
# Capture RC from the merge itself. A previous revision of this block read
# `RC=$?` AFTER the `sed` that prints the table, so a failed merge with a
# readable stale summary.md still reported STAGE_STATUS=OK -- the same
# "status of the last echo" defect this stage was meant to fix.
if [ "$RC" -ne 0 ]; then echo "STAGE_STATUS=FAILED merge rc=$RC"; echo ALLDONE; exit "$RC"; fi
if [ ! -s "$OUT/summary.md" ]; then
  echo "STAGE_STATUS=FAILED (merge rc=0 but $OUT/summary.md missing/empty)"; echo ALLDONE; exit 3
fi
if [ -n "$MISSING" ]; then
  # The job log rotates and gets lost; summary.md is what gets read months
  # later. Stamp the omission into the artifact, at the TOP where it cannot be
  # scrolled past.
  { echo "> **PARTIAL RANKING.** Missing scheme checkpoints:$MISSING"
    echo "> Published with ALLOW_PARTIAL=1. Do not compare against a full-set run."
    echo
    cat "$OUT/summary.md"; } > "$OUT/summary.md.tmp" && mv -f "$OUT/summary.md.tmp" "$OUT/summary.md"
fi
echo "=== ranking table ==="; sed -n '1,80p' "$OUT/summary.md"
echo "STAGE_STATUS=OK"
echo ALLDONE
