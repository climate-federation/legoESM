#!/bin/bash
# Submit learned WeatherBench v9 arms, data-parallel, one job per named arm.
# Arms: ngcm, ngcm_nolearnedembed, sfno, sfno_nolearnedembed.
#
#   WB_REPO=<frozen checkout of the commit to train>  \
#   WB_ROOT=<directory holding one run directory per arm>  \
#     scripts/cluster/levante/submit_wb_v9.sh ngcm [sfno ...]
#
# GPUs per arm sized from the GPU ladder (2026-11-04) for ~10-11.5 h of wall
# per run at n_epochs 4 (one whole epoch per curriculum stage 6/12/24/72 h):
#   NeuralGCM 32 GPUs (8 nodes), SFNO 88 GPUs (22 nodes; 80 left ~5% margin).
# A run that hits the 12 h limit is continued by resubmitting the same line
# (train_wb.slurm resumes from the newest complete checkpoint).
set -euo pipefail
: "${WB_REPO:?set WB_REPO}" "${WB_ROOT:?set WB_ROOT}"
cd "$WB_REPO"
submit() {   # name deck mode nodes
  sbatch --job-name="wb_v9_$1" --nodes="$4" --time=12:00:00 \
    --output="$WB_ROOT/$1/slurm_%j.out" \
    --export=ALL,LEGOESM_REPO="$WB_REPO",WB_CONFIG="config/wb/campaign/$2",WB_MODE="$3",WB_OUT="$WB_ROOT/$1" \
    scripts/cluster/levante/train_wb.slurm
}
[ $# -ge 1 ] || { echo "name the arm(s) to submit" >&2; exit 2; }
for arm in "$@"; do
  mkdir -p "$WB_ROOT/$arm"
  case "$arm" in
    ngcm)                submit ngcm                spectral_t63_ngcm_v9.yaml                neural_gcm 8 ;;
    ngcm_nolearnedembed) submit ngcm_nolearnedembed spectral_t63_ngcm_nolearnedembed_v9.yaml neural_gcm 8 ;;
    sfno)                submit sfno                spectral_t63_sfno_v9.yaml                sfno       22 ;;
    sfno_nolearnedembed) submit sfno_nolearnedembed spectral_t63_sfno_nolearnedembed_v9.yaml sfno       22 ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
done
