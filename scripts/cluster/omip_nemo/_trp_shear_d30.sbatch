#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=trpshear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpshear_%j.out
# ONE VARIABLE against trp_base2_d30: the TKE SHEAR PRODUCTION discretisation.
#   --tke-shear-production  squared_centered (card default)  ->  nemo_face_native_now2
# This is NEMO's own zdf_sh2: face-native velocity differences, the product
# taken at NOW^2, and production doubled adjacent to coasts.  The variant is
# FIXED by the oracle, not chosen: ORCA1 builds with key_RK3 (cpp_ORCA1.fcm),
# which is the "_now2" family.  Our own code calls this "the LAST unclosed gap
# on this card".  User approved a shear arm 2026-09-08 after the mixing-length
# arm was cancelled for having the wrong expected sign.
#
# WHY SHEAR (the chain the evidence now supports, each link measured):
#   Our undercurrent is weak (core 0.20 m/s at 220E against the oracle's 0.55).
#   Weak shear means little TKE production below the mixed layer.  Our mixing at
#   20-60 m is orders weaker than the oracle's, and our Prandtl number there is
#   11.4 against its 1.23 -- i.e. our Richardson number is high because the
#   shear is missing, and our own tracer limiter then suppresses heat mixing
#   hardest exactly there.  Cold water at 20 m never mixes up; heat stays above.
#   The oracle reaches that mixing with a SHORTER length scale than ours, so its
#   turbulence must be fed by production, not by geometry.
#
# PRIOR, stated so it is not forgotten: an earlier face-native shear arm was a
# NULL on the vertical velocity, the meridional transport and the Pacific index.
# It never measured the diffusivities or the Prandtl number, which is what this
# arm asks.  A null here too would retire the shear hypothesis.
#
# PRE-REGISTERED, day 30 vs GATEWAY rec 5, written BEFORE the run:
#   PRIMARY   20-60 m tracer diffusivity rises from 2.60e-07 by at least 10x.
#             Below 2x = the shear discretisation is NOT the operator, refuted.
#   CO-GATE   20-60 m Prandtl number falls from 11.4 toward the oracle's 1.2.
#             A primary hit WITHOUT this is suspect: deeper mixing lowers the
#             stratification, which lowers Ri and unclamps the limiter, so the
#             gate can be passed without any real gain in turbulence (GLM).
#   SECONDARY undercurrent core at 220E rises from 0.20 m/s toward 0.55;
#             T(0.5)-T(20) in the cold tongue falls from 3.80 K.
#   GUARDRAIL SST rmse must not degrade by more than 0.05, nino3 stays <= +1.10,
#             no polar SST regression beyond 0.1.  A primary hit bought with a
#             guardrail breach is NOT an adoption.
set -uo pipefail
PIN=${1:?usage: sbatch _trp_base2_d30.sbatch <pinned-sha> [leg: 1|2]}
LEG=${2:-1}
ROOT=/burg-archive/glab/users/pg2328/_evd_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/trp_shear_d30

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
  --tke-shear-production nemo_face_native_now2 --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 --kprofile-snapshots \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

[ "$LEG" = 2 ] || { cp -f $OUT/snapshot_final.npz $OUT/snapshot_day0015.npz; echo "leg 1 done (day 15); state copied to snapshot_day0015.npz; scoring runs in leg 2"; exit 0; }
echo "=== nino3 trajectory (matched NEMO to day 90; our shape beyond) ==="
for pair in "0030 5"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label trp_shear_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_trpshear_d$1 2>&1 \
    | grep -E "nino3_5S5N|\[SST\].*rmse" | head -3
done

echo "=== undercurrent / thermocline vs GATEWAY (same instrument as job 9648737) ==="
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $OUT/snapshot_day0030.npz --nemo-gridt "$GW_T" --nemo-t-recs 5:6 \
  --label trp_shear_d30 --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --meridional-lon 240,260 \
  --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc 2>&1 | grep -vE "Warning|warn" | tail -120
echo DONE
