#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=scmrce_ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_ab_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# Arms (a) LITERATURE DEFAULTS and (b) DERIVATIVE-FREE TUNING, for all 10
# convection schemes, under ONE kernel arm (passed in via ARM).
#
# run_scm_rce_convection_intercomparison.py produces BOTH arms in a single
# pass per scheme: an "a priori" run at the shipped *Config defaults and an
# "a posteriori" run after derivative-free tuning of the scheme's
# tunable_tier<=extended parameters against the same CRM profiles.
#
# CONTROLLED COMPARISON: every flag below except --subsidence-solve is held
# byte-identical across the two kernel arms and across all 10 schemes.  The
# reference dir, --days, --dt, --analysis-days, --last-reference-files,
# --radiation and the tuner seed/budget are all pinned here, never defaulted,
# so a later re-read of the log shows the exact protocol.
# ---------------------------------------------------------------------------
WT=/burg/glab/users/pg2328/scmrce_wt_campaign
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
# Arms of 9311347 died with `LLVM ERROR: Unable to allocate section memory!`
# after 0.5-3.7 h — the XLA CPU JIT's allocator, not the model, which is why
# raising --mem alone did not help. --tune-evals 48 drives many distinct
# compilations per scheme. A persistent cache reuses them across evals + arms.
export JAX_COMPILATION_CACHE_DIR=/burg/glab/users/pg2328/.jax_cache_scmrce
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export OMP_NUM_THREADS=4
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

SCHEMES=(sbm dca kuo mass_flux edmf zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold)
SCHEME=${SCHEMES[$SLURM_ARRAY_TASK_ID]}
ARM=${ARM:-as_shipped}
REF=${REF:-/burg-archive/glab/users/pg2328/legoESM/results/rcemip_ref_sam300}
OUT=/burg-archive/glab/users/pg2328/legoESM/results/scm_rce_paper/arms_ab_${ARM}

echo "SCHEME=$SCHEME  KERNEL_ARM=$ARM"
echo "REFERENCE=$REF"
ls $REF/snapshots3d/ | tail -6

JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF \
  --outdir $OUT \
  --schemes $SCHEME \
  --subsidence-solve $ARM \
  --days 100 \
  --dt 600 \
  --analysis-days 5 \
  --last-reference-files 5 \
  --radiation rrtmgp \
  --tune-evals 48 \
  --tune-seed 20260705 \
  --no-merge 2>&1
RC=$?
echo "rc=$RC"
# FAIL LOUD. Arms 9284304/9311117/9311347 all aborted (rc=134) while SLURM
# recorded COMPLETED/0:0, because this wrapper ended on `echo ALLDONE` and
# returned ITS status. That makes every downstream `afterok` a lie.
if [ "$RC" -ne 0 ]; then
  echo "ARM_STATUS=FAILED (scheme=$SCHEME arm=$ARM rc=$RC)"; echo ALLDONE; exit "$RC"
fi
echo "ARM_STATUS=OK (scheme=$SCHEME arm=$ARM)"
echo ALLDONE
