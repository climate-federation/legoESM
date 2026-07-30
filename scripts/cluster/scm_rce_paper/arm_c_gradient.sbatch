#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=scmrce_c
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_c_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# Arm (c) GRADIENT TUNING, per convection scheme, under ONE kernel arm.
#
# Initialised from the derivative-free arm's tuned_parameters.json for the
# SAME kernel arm (never the other one -- that would cross-contaminate the two
# ranking tables).  Every protocol flag is pinned identically to arms_ab.sbatch;
# only the optimiser differs, which is the whole point of the arm.
#
# Known structural limits, reported rather than hidden:
#   * dca has ONE tier<=2 trainable parameter (cape_threshold) and kuo has
#     two, so their gradient arms are very weak and may trip the
#     nonzero-gradient preflight. That is a result, not a failure to hide.
#   * dca cannot reach the 7 nested AhmedNeelinDCAConfig parameters
#     (campaign._tunable_subconfig unwraps only CLUBB).
#   * bechtold REQUIRES --bechtold-policy train_deterministic, otherwise its
#     scheme key is discarded and the job would train nothing; the driver now
#     exits loudly instead of producing an empty bechtold-labelled result.
# ---------------------------------------------------------------------------
WT=/burg/glab/users/pg2328/scmrce_wt_campaign
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

SCHEMES=(sbm dca kuo mass_flux edmf zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold)
SCHEME=${SCHEMES[$SLURM_ARRAY_TASK_ID]}
ARM=${ARM:-as_shipped}
REF=/burg-archive/glab/users/pg2328/legoESM/results/rcemip1_n128_ocean
AB=/burg-archive/glab/users/pg2328/legoESM/results/scm_rce_paper/arms_ab_${ARM}
OUT=/burg-archive/glab/users/pg2328/legoESM/results/scm_rce_paper/arm_c_${ARM}/${SCHEME}

BPOL=auto
if [ "$SCHEME" = "bechtold" ]; then BPOL=train_deterministic; fi

echo "SCHEME=$SCHEME  KERNEL_ARM=$ARM  bechtold_policy=$BPOL"
echo "init tuned_parameters from: $AB/tuned_parameters.json"

JAX_ENABLE_X64=1 $PY $WT/scripts/run/train_scm_rce_params.py \
  --reference-dir $REF \
  --tuned-parameters $AB/tuned_parameters.json \
  --campaign-dir $AB \
  --outdir $OUT \
  --convection $SCHEME \
  --subsidence-solve $ARM \
  --bechtold-policy $BPOL \
  --days 100 \
  --dt 600 \
  --analysis-days 5 \
  --radiation rrtmgp 2>&1
echo "rc=$?"
echo ALLDONE
