#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=scmrce_ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-9
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
# XLA CPU JIT memory. Every arm of 9311347 died with
#   LLVM ERROR: Unable to allocate section memory!
# after 0.5-3.7 h — the COMPILER's allocator exhausting host RAM, not the
# model. Raising --mem does not help: --tune-evals 48 drives many distinct
# compilations per scheme and LLVM section memory accumulates across them.
# A persistent cache makes repeat shapes reuse compiled code instead of
# recompiling, across evals AND across arms.
export JAX_COMPILATION_CACHE_DIR=/burg/glab/users/pg2328/.jax_cache_scmrce
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Cap the compiler's own parallelism; each worker holds its own LLVM arena.
export XLA_FLAGS="--xla_cpu_enable_fast_math=false"
export OMP_NUM_THREADS=4
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

SCHEMES=(sbm dca kuo mass_flux edmf zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold)
SCHEME=${SCHEMES[$SLURM_ARRAY_TASK_ID]}
ARM=${ARM:-as_shipped}
REF=/burg-archive/glab/users/pg2328/legoESM/results/rcemip1_n128_ocean
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
echo "rc=$?"
echo ALLDONE
