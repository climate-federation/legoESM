#!/bin/bash
#SBATCH --job-name=ensoprov
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ensoprov_%j.out
#
# THE EQUATORIAL PACIFIC IS THE BIGGEST ERROR IN THE MODEL and the latitude
# bands were hiding it: Nino3 SST bias +3.19 C (corr 0.530) against a global
# rmse of 1.139 and a 23S-23N band bias of +0.63. A cold-tongue error and an
# off-equatorial error of opposite sign cancel in the band average.
#
# TWO QUESTIONS, both answerable from snapshots already on disk:
#
#  1. DOES IT PREDATE THE N2 CHANGE? Score xgrid_trp_d30, which runs the OLD
#     card (insitu N2, alpha_tke=30, gaspar_sqrt2e). NOT a one-variable
#     control -- it is the degraded cross-grid config and differs in several
#     fields -- so it can answer "does the cold-tongue bias exist without the
#     N2 change" but NOT "how much of it the N2 change caused". The matched
#     insitu control (9412975) answers that.
#
#  2. HOW FAST DOES IT GROW? Score the signed arm at day 10 and day 20. A bias
#     already saturated by day 10 is an initialisation/forcing problem; one
#     growing steadily to day 30 is a circulation spin-down.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
echo "HEAD: $(git -C $ROOT log --oneline -1)"
score () {  # $1=snapshot $2=label
  [ -f "$1" ] || { echo "[skip] missing $2: $1"; return 0; }
  echo; echo "########## $2"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$1" \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "$2" --output-dir $ROOT/results/omip_nemo/enso_$2 2>&1 \
    | grep -aE "^\[SST\]|nino|eq_pacific|tropics_23S"
}
R=$ROOT/results/omip_nemo
score $R/xgrid_trp_d30/snapshot_final.npz            oldcard_trp_d30
score $R/nemolev_trp_signedn2_d30/snapshot_day0010.npz signed_d10
score $R/nemolev_trp_signedn2_d30/snapshot_day0020.npz signed_d20
echo; echo "=== DONE"
