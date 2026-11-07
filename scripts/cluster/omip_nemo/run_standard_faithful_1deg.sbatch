#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=std1deg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/std1deg_%j.out
#
# ============================================================================
# THE STANDARD NEMO-FAITHFUL 1-DEGREE TRIPOLE CONFIGURATION (user-declared
# standard, 2026-08-25).  This is the FLAG-IDENTICAL card of the validated
# 180-day drift run (results/omip_nemo/nemolev_trp_long180, job 9471858) --
# run length is parameterized (+0.05 d overshoot, see YEARS below), so a
# DAYS=180 invocation is 10 steps shorter than long180's 180.017 d,
# which against NEMO's matched GATEWAY cold start scored:
#   d30: SST rmse 0.515 C (bias +0.01) SSS rmse 0.464  nino3 +1.42
#   d60: SST rmse 0.495 C (bias +0.00) SSS rmse 0.617  nino3 +1.43
#   d90: SST rmse 0.456 C (bias -0.01) SSS rmse 0.697  nino3 +0.70  EXCELLENT
# The equatorial warm bias HALVES between d60 and d90: a bounded, closing
# 1-degree spin-up transient, not a drift.
#
# The corrected (_MOD) CORE-II forcing is the DEFAULT resolver (gated below by
# tests/ocean/unit/test_core2_corrected_cache_default.py).  Any change to a
# flag on this card is a NEW configuration and must be A/B'd against this one
# (one variable per run) -- never edited in place.
#
# Usage: sbatch run_standard_faithful_1deg.sbatch <pinned-sha> [days] [name]
#   days defaults to 90 (the GATEWAY-scored window); name defaults to
#   std1deg_d<days>.  Output: results/omip_nemo/nemolev_trp_<name>.
# ============================================================================
set -uo pipefail
PIN=${1:?usage: sbatch run_standard_faithful_1deg.sbatch <pinned-sha> [days] [name]}
DAYS=${2:-90}
NAME=${3:-std1deg_d${DAYS}}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)  DAYS=$DAYS  NAME=$NAME"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_${NAME}
# +0.05 day overshoot: the driver SKIPS the cadence snapshot at the final
# step (codex HIGH: DAYS=90 with exact-length run wrote only snapshot_final,
# so the advertised d90 self-score never ran).  The overshoot makes every
# 15-day cadence snapshot, day DAYS included, land as a normal cadence write.
YEARS=$($PY -c "print((${DAYS}+0.05)/365.0)")

echo "=== gate: corrected forcing is the default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years $YEARS --snapshot-every-days 15 --diag-every-days 5 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
echo "=== GATEWAY-matched scores (5-day means; only to day 90) ==="
score_fail=0
for pair in "0030 5" "0060 11" "0090 17"; do
  set -- $pair
  [ $((10#$1)) -le "$DAYS" ] || continue
  SNAP=$OUT/snapshot_day$1.npz
  # A missing in-range snapshot is FATAL (codex: silent continue hid the
  # endpoint-snapshot bug), not skipped.
  [ -f "$SNAP" ] || { echo "FATAL: expected $SNAP missing"; score_fail=1; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label ${NAME}_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_${NAME}_d$1 2>&1 \
    | grep -E "nino3_5S5N|verdict|\[SST\]|\[SSS\]" | head -6
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: scorer failed for day $1"; score_fail=1; }
done
[ $score_fail -eq 0 ] || { echo "[STANDARD-SCORE-FAIL]"; exit 1; }
echo DONE
