#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=blqv_disc
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/blqv_disc_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY -u scripts/validate/scm_rce_blqv_discriminator.py \
  --scheme ${SCHEME:-mass_flux} \
  --arms ${ARMS:-control,entrainment,coare3,hb} \
  --out $REPO/results/scm_rce_tqtune/blqv_discriminator_${TAG:-${SCHEME:-mass_flux}}.json \
  2>&1 | grep --line-buffered -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader\|cuInit\|CUDA"
echo "RC=${PIPESTATUS[0]}"
echo ALLDONE
