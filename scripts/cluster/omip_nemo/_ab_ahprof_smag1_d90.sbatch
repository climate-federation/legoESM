#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ahsmag1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=44:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ahsmag1_%j.out
#
# ============================================================================
# THE LEVER IS C_smag, NOT A_h.  Measured, not assumed.
#
# The A_h ramp arm (9441856) moved day-30 nino3 by 0.07 C against its control
# (+3.12 vs +3.19) even though the ramp was demonstrably active.  Running the
# model's OWN smagorinsky_viscosity_cgrid on both d30 snapshots (job 9444633)
# shows why -- area-weighted, |lat| < 1:
#
#   A_smag 5.763e4  vs prescribed 5079   ->  ratio 11.35
#
# so the flow-adaptive term, not the prescribed field, sets the viscosity the
# momentum equation sees.  Total at the equator:
#
#   card : 1e5   background + 3.04e4 smag = 1.30e5
#   ramp : 5079  background + 5.76e4 smag = 6.27e4
#
# an intended 20x reduction delivered as 2.1x.  Worse, the closure REFILLS
# what the background gives up: lowering A_h sharpens the shear, which raises
# A_smag from 3.04e4 to 5.76e4.  That is why the Smag-OFF twin restored the
# equatorial circulation and then went unstable, while Smag-ON did nothing.
#
# A_smag scales as C^2, so C = 3.0/sqrt(11.35) = 0.89 would put the adaptive
# term near NEMO's prescribed field.  This arm takes C_smag_lap = 1.0, the
# round number just above it, keeping the ramp identical.  ONE VARIABLE vs
# 9441856.
#
# PRE-REGISTERED, against the ramp arm (ahpsmag, C_smag 3.0) at MATCHED days:
#   A_smag/prescribed at |lat|<1, d30: CONFIRMED if it falls to <= 2 (C^2
#     scaling predicts 11.35/9 = 1.26 before the shear response, so a value
#     between 1 and 2 means the closure is no longer in charge). If it stays
#     above 5, the shear response cancels the coefficient change and the
#     Smagorinsky form itself -- not its coefficient -- is the problem.
#   nino3 d30 [ramp +3.12, ctl +3.19]: CONFIRMED <= +2.5, PARTIAL +2.5..+3.0.
#   max|u| : the STABILITY RISK. The Smag-off twin doubled every ~15 days and
#     reached 11.53 by d65. CONFIRMED stable if < 2.5 through d90; REFUTED if
#     it exceeds 4, which would mean C=1.0 is below the stability floor for
#     this dycore and the backstop must come from somewhere else (higher-order
#     or biharmonic friction) rather than from a larger Laplacian.
#   SH_midlat SST: report either way; the Smag-off twin cost +0.79 there.
# ============================================================================
set -uo pipefail
ARM=ahsmag1
VISC_FLAGS="--A-h 20000 --C-smag-lap 1.0 --A-h-profile-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc"
PIN=${1:?usage: sbatch _ab_ahprofile_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_ahsmag1_d90

echo "=== gate: unit tests for the reduction semantics + CLI wiring ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_polar_cap_boost.py \
  tests/unit/test_run_omip_core2_eq_visc_shaping.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate tests failed rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  $VISC_FLAGS --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.2466 --snapshot-every-days 10 --diag-every-days 5 \
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
# codex 9430935 MAJOR-4: never score a failed or snapshot-less arm -- a stale
# or missing snapshot scored silently looks like a completed arm.
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
EXP_GATEWAY_V=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc
SNAP=$OUT/snapshot_day0090.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot at $SNAP -- not scoring a partial arm"; exit 1; }
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 3 \
  --label ahpsmag_d90 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 17:18 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$EXP_GATEWAY_V" 2>&1 | tail -140
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label ahpsmag_d90 \
  --output-dir $ROOT/results/omip_nemo/compare_ahpsmag_d90_m03 2>&1 | tail -40
echo "=== compare rc=$?"
