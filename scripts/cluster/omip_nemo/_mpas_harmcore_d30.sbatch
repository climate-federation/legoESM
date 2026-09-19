#!/bin/bash
#SBATCH --job-name=mpas_harmcore
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasmod_%j.out
#
# THE SECOND GRID ON THE CORRECTED FORCING (three-grid agreement check).
# Tripole (0.515 SST rmse, nino3 +1.42) and FESOM2 (1.016, nino3 +0.17) have
# GATEWAY-matched day-30 scores on the corrected (_MOD) CORE-II cache, which
# is now the DEFAULT resolver; every existing MPAS arm predates it.  This arm
# runs MPAS level 7 on the maximal card MPAS supports (the driver hard-errors
# on --dm2dc/--isf/--bbl-adv/--sw-rgb-chl/--iwm/--nemo-monthly-init and the
# --tke-* knobs -- see _xgrid_mpas.sbatch), scores day 30 vs GATEWAY rec 5,
# and its numbers carry that reduced-card caveat with them.
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_harmcore_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/mpas_harmcore_d30

echo "=== gate: corrected forcing is the default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

# codex review fixes (job 9499087): --years overshoots to 30.05 d so the
# day-30 cadence snapshot is actually written (0.0821355 y = 29.979 d wrote
# only snapshot_final); --adaptive-implicit-vertadv/--momentum-rk3 DROPPED
# (the MPAS builder silently ignores both -- carrying them would claim
# physics the arm does not run); the four --tke-* knobs ADDED (legal on MPAS
# under --mpas-vmix tke since the tke-card wiring; matches the standard
# card's closure exactly).
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid mpas \
  --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --dt 150 --nemo-vertical --partial-cell \
  --mle --mle-ce 0.06 \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --years 0.0823288 --snapshot-every-days 30 \
  --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --ice-ocean-heat-coeff 70 \
  --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --ice-init $NEMOIN/Ice_initialization.nc \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
echo "=== day 30 vs GATEWAY rec 5 ==="
[ -f $OUT/snapshot_day0030.npz ] || { echo "FATAL: day-30 snapshot missing"; exit 1; }
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot $OUT/snapshot_day0030.npz \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 \
  --sss-faithful --grid-label mpas_harmcore_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_gw_mpas_harmcore_d30 2>&1 | tail -38
echo DONE
