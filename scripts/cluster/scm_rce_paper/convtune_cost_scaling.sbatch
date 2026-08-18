#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=convtune_cost
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_cost_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# TWO POINTS, ONE LINE.  Per-evaluation cost of an SCM-RCE run splits into a
# fixed compile intercept and a per-step slope; a single measurement cannot
# separate them, and the production budget depends on the slope alone (the
# 100-day arm amortises the intercept over 14 400 steps).
#
# Task 0: --days 2   (288 steps)      Task 1: --days 10  (1440 steps)
# Everything else identical, INCLUDING --tune-evals, so the difference is
# purely the extra steps.  Wall time and peak RSS come from /usr/bin/time -v.
#
# Read: slope [s/step] = (wall_10 - wall_2) / ((1440 - 288) * n_configs)
#       intercept [s]  = wall_2 / n_configs - 288 * slope
# with n_configs = the number of DISTINCT configs evaluated (1 a-priori + the
# tuner's candidates, the first of which is the defaults and hits the cache).
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
DAYS_LIST=(2 10)
DAYS=${DAYS_LIST[$SLURM_ARRAY_TASK_ID]}
OUT=$REPO/results/scm_rce_convtune/_cost_d${DAYS}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "COST POINT: days=$DAYS (steps=$(( DAYS * 144 ))) scheme=bechtold evals=2"
mkdir -p $OUT
/usr/bin/time -v env JAX_ENABLE_X64=1 $PY \
  $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT --schemes bechtold \
  --subsidence-solve implicit_flux --microphysics morrison \
  --hard-saturation-adjustment \
  --days $DAYS --dt 600 --analysis-days 1 --last-reference-files 5 \
  --radiation rrtmgp --scm-convection-substeps 10 \
  --scm-microphysics-substeps 30 \
  --tune-evals 2 --tune-seed 20260810 --force --no-merge 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" \
  | grep -E "scheme|prior|Elapsed|Maximum resident|Exit status|rc="
echo "COST_RC=${PIPESTATUS[0]}"
echo ALLDONE
