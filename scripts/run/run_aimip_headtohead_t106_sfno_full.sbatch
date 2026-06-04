#!/bin/bash
#SBATCH --job-name=h2h_sfno_full
#SBATCH --output=results/aimip_t106_headtohead/slurm_logs/sfno_full_%j.out
#SBATCH --error=results/aimip_t106_headtohead/slurm_logs/sfno_full_%j.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 head-to-head: trains the sfno_full variant -- SFNO as a full
# atmospheric emulator (no dycore, no physics tendency) via
# ``train_sfno_full_spectral`` (dt_sfno=6 h macro step matching the
# standard SFNO/FourCastNet cadence).  Same SFNO arch as the
# sfno_physics variant (embed_dim=32, n_blocks=2) so model size is
# directly comparable; only the integration path differs.
#
# Configs: config/aimip/t106_headtohead/(base.yaml + suite.yaml + variant_sfno_full.yaml)
#
# 72 h single-shot walltime; chain machinery left inert via MAX_LINKS=1.
# See run_aimip_headtohead_t106_classical.sbatch for the rationale.

set -euo pipefail
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

VARIANT="sfno_full"
SUITE="config/aimip/t106_headtohead/suite.yaml"
OUT_DIR="${REPO_ROOT}/results/aimip_t106_headtohead"
SCORECARD="${OUT_DIR}/aimip_scorecard.json"
LOG_DIR="${OUT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

: "${LINK:=1}"
: "${MAX_LINKS:=1}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME="${HOME:-/burg-archive/home/pg2328}"

# Dedicated cache: sfno_full's HLO graph is different from
# sfno_physics (no dycore in the rollout), so do not share.
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.cache/jax_compile_cache_h2h_${VARIANT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "[h2h-${VARIANT}] $(date)  link=${LINK}/${MAX_LINKS}  host=$(hostname)  job=${SLURM_JOB_ID:-<no-slurm>}"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    -u scripts/run/run_aimip.py \
    --suite "${SUITE}" \
    --variants "${VARIANT}" \
    --resume
RC=$?
echo "[h2h-${VARIANT}] link=${LINK} python exit=${RC}  $(date)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    VARIANT_DONE="false"
    if [[ -s "${SCORECARD}" ]]; then
        if /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c "
import json, sys
sc = json.load(open('${SCORECARD}'))
sys.exit(0 if (sc.get('variants') or {}).get('${VARIANT}') else 1)
        " 2>/dev/null; then
            VARIANT_DONE="true"
        fi
    fi

    if [[ "${VARIANT_DONE}" == "true" ]]; then
        echo "[h2h-${VARIANT}] scorecard has '${VARIANT}' entry; chain complete."
    elif (( LINK >= MAX_LINKS )); then
        echo "[h2h-${VARIANT}] reached MAX_LINKS=${MAX_LINKS}; chain stopped without completing."
    else
        NEXT=$((LINK + 1))
        echo "[h2h-${VARIANT}] scorecard missing '${VARIANT}'; queueing link ${NEXT} ..."
        NEXT_JOB=$(LINK="${NEXT}" MAX_LINKS="${MAX_LINKS}" sbatch \
            --dependency=afterany:"${SLURM_JOB_ID}" \
            --export=ALL,LINK="${NEXT}",MAX_LINKS="${MAX_LINKS}" \
            "$0" | awk '{print $NF}')
        echo "[h2h-${VARIANT}] submitted link ${NEXT} as job ${NEXT_JOB}"
    fi
fi

exit "${RC}"
