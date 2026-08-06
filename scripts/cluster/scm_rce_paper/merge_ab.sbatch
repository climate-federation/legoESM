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
echo "KERNEL_ARM=$ARM  OUT=$OUT"
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT --subsidence-solve $ARM \
  --days 100 --dt 600 --analysis-days 5 --last-reference-files 5 \
  --radiation rrtmgp --tune-evals 48 --tune-seed 20260705 \
  --scm-convection-substeps ${SUBSTEPS:-10} \
  --merge-only 2>&1 | tail -60
echo "merge rc=${PIPESTATUS[0]}"
echo "=== ranking table ==="; sed -n '1,80p' $OUT/summary.md 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then echo "STAGE_STATUS=FAILED rc=$RC"; echo ALLDONE; exit "$RC"; fi
echo "STAGE_STATUS=OK"
echo ALLDONE
