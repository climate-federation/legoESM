#!/bin/bash
#SBATCH --job-name=h2h_classical
#SBATCH --output=results/aimip_t106_headtohead/slurm_logs/classical_%j.out
#SBATCH --error=results/aimip_t106_headtohead/slurm_logs/classical_%j.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 head-to-head: trains the classical variant with the new
# multi-day + bias + spectral-CRPS loss and RRTMGP radiation.
# Configs: config/aimip/t106_headtohead/(base.yaml + suite.yaml + variant_classical.yaml)
#
# 72 h single-shot walltime (chain disabled)
# ------------------------------------------
# Runs ``scripts/run_aimip.py --resume`` once: scans
# ``results/aimip_t106_headtohead/classical/`` for the latest
# ``epoch_NNNN.eqx`` and continues training from epoch+1.  72 h is
# enough headroom for the full 4-epoch run including the one-time
# RRTMGP/dycore JIT compile (prior 12 h links never finished compile
# + training for the classical variant).
#
# The chain-resubmit machinery below is left in place but inert with
# MAX_LINKS=1 (link 1 hits the cap and exits without queueing link 2).
# To re-enable chained resubmits, set MAX_LINKS>=2 in the env or here.
# The per-variant JAX compile cache is still useful for ``--resume``
# reruns after manual cancellation.

set -euo pipefail
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

VARIANT="classical"
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

# Per-variant JIT compile cache (RRTMGP + spectral PE rollout graph is
# ~hours to build at T106 L8 with multi-step supervision).  Cached so
# any post-cancellation ``--resume`` skips the RRTMGP/dycore JIT graph
# build and goes straight to training steps.  Per-variant dir prevents
# collision with the column_nn / sfno_full graphs which have different
# HLO shapes.
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.cache/jax_compile_cache_h2h_${VARIANT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "[h2h-${VARIANT}] $(date)  link=${LINK}/${MAX_LINKS}  host=$(hostname)  job=${SLURM_JOB_ID:-<no-slurm>}"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    -u scripts/run_aimip.py \
    --suite "${SUITE}" \
    --variants "${VARIANT}" \
    --resume
RC=$?
echo "[h2h-${VARIANT}] link=${LINK} python exit=${RC}  $(date)"

# Chain logic: stop when the scorecard contains a non-empty entry for
# this variant (training+eval complete); otherwise submit the next
# link via --dependency=afterany so it picks up regardless of how we
# exited (success OR walltime).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    VARIANT_DONE="false"
    if [[ -s "${SCORECARD}" ]]; then
        # ``python -c`` parse: returns 0 (done) when variants.${VARIANT}
        # exists in the scorecard JSON.  Anything else => keep chaining.
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
