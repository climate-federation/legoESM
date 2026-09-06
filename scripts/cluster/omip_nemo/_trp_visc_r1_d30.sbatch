#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=trpviscr1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpviscr1_%j.out
# VISCOSITY LADDER RUNG 1 (harmonization_plan_2026-09-05.md P1, codex/GLM
# reviewed design): ONE variable vs the unified tripole card = the LATITUDINAL
# SHAPE of the Laplacian background taken from ORCA1's own eddy_viscosity_3D.nc
# (1000 m2/s at the equator -> 20000 mid-latitude), applied as ratios to
# --A-h 2e4 so that the schedule's A_h=1e4 keeps the mid-latitude background
# UNCHANGED (1e4) and only the equatorial band drops (~500 at 0N).  The
# Smagorinsky ramp, dt, and every other flag are byte-identical to
# _trp_unified_180d.sbatch; 30 days on the short partition as TWO 15-day legs
# (0.4 steps/s = 6.5 h per leg; a single 30-day leg overran 12 h) chained
# with --restart-save/--restart-from (leg 2 = sbatch --dependency=afterok).
#
# WHY. Day-30 undercurrent probe (job 9648737) on the unified arm: thermocline
# depth matches NEMO (Z20 gap -5 m, tilt 102%) but the undercurrent core is
# half of NEMO's (200E 0.289 vs 0.496; 220E 0.183 vs 0.546 m/s) under a flat
# 1e4 background that is 10x NEMO's at the equator.  Earlier arms with the
# file profile (Aug 22-23) reached 0.35/0.45 and halved the cold-tongue
# tendency; this rung re-measures that on the CURRENT card.
#
# PRE-REGISTERED (day 30 vs GATEWAY rec 5, same instrument, ctl = trp_unified180):
#   EUC core 200E  ctl 0.289  NEMO 0.496  CONFIRM >= 0.40   REFUTE < 0.32
#   EUC core 220E  ctl 0.183  NEMO 0.546  CONFIRM >= 0.35
#   nino3 bias     ctl +1.30              CONFIRM <= +1.05
#   SST rmse       ctl 0.51               tripwire > 0.56 ; SH-mid bias > +0.5
#   stability      max|u| < 2.5 m/s through day 30 (Smag-off twin ran away at 2.5)
set -uo pipefail
PIN=${1:?usage: sbatch _trp_visc_r1_d30.sbatch <pinned-sha> [leg: 1|2]}
LEG=${2:-1}
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/trp_viscr1_d30

echo "=== gate ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --A-h 2e4 --A-h-profile-file $NEMOIN/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc \
  --visc-schedule '0:1e4:3.0,8:1e4:1.5,16:1e4:0.5' \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --years $([ "$LEG" = 1 ] && echo 0.0411 || echo 0.0822) --snapshot-every-days 15 --diag-every-days 5 \
  --restart-save $OUT/restart_leg$LEG.npz $([ "$LEG" = 2 ] && echo "--restart-from $OUT/restart_leg1.npz") \
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

[ "$LEG" = 2 ] || { echo "leg 1 done (day 15); scoring runs in leg 2"; exit 0; }
echo "=== nino3 trajectory (matched NEMO to day 90; our shape beyond) ==="
for pair in "0030 5"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label trp_viscr1_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_trpviscr1_d$1 2>&1 \
    | grep -E "nino3_5S5N|\[SST\].*rmse" | head -3
done

echo "=== undercurrent / thermocline vs GATEWAY (same instrument as job 9648737) ==="
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $OUT/snapshot_day0030.npz --nemo-gridt "$GW_T" --nemo-t-recs 5:6 \
  --label trp_viscr1_d30 --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --meridional-lon 240,260 \
  --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc 2>&1 | grep -vE "Warning|warn" | tail -120
echo DONE
