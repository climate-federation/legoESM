#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=mxlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mxlab_%j.out
#
# ============================================================================
# ONE VARIABLE: the mixing length NEMO's ORCA1 namelist actually selects.
#
# ORCA1's namelist_cfg sets `nn_mxl = 2` (EXP00/namelist_cfg, &namzdf_tke),
# which is an OVERRIDE of the reference default of 3.  Our production card has
# been running `--tke-mxl-choice 3`, and this driver's own documentation says
# choice 4 is NEMO's nn_mxl=2.  So production has been running a mixing length
# the oracle does not use.  That is a concrete faithfulness deviation, not a
# tuning knob, and this pair measures what it costs.
#
# Everything else is the control card verbatim from the modfrc_d30 run
# manifest (corrected CORE-II forcing, NEMO's own 75 levels, the zdftke card).
# The two arms differ in ONE argument.
#
# SECOND THING THIS BUYS, at no extra GPU cost: --kprofile-snapshots stores the
# closure's own viscosity and diffusivity in every snapshot.  NEMO publishes
# avm and avt in its five-day output; ours have never been persisted, so every
# statement about our equatorial mixing has been inferred from TKE and the
# stratification rather than measured against the quantity the oracle
# publishes.  The oracle's side is already measured (job 9467236): in the cold
# tongue, 200-260E, |lat|<=2, NEMO's closure-active layer (5-65 m) has
#   Prandtl median 1.211 (range 1.115-1.616)   avt median 1.57e-3 m2/s
#
# PRE-REGISTERED, fixed before launch.
#
# (1) THE PRANDTL HYPOTHESIS.  The standing account of the tripole's cold
#     tongue is that the closure mixes momentum too well and heat too poorly:
#     the undercurrent is four times too weak while the pressure gradient
#     driving it is right, and the thermocline is 2.3x too sharp at the right
#     depth.  One number produces both.  Measured in the SAME box, the SAME
#     5-65 m window, the SAME per-column median:
#       CONFIRMED  our Pr >= 2.4 (2x NEMO) AND our avt <= 7.9e-4 (half NEMO)
#       REFUTED    Pr within 0.8-1.5x of 1.211 AND avt within 0.5-2x of
#                  1.57e-3 -- the closure's split is right and the cold-tongue
#                  tendency error is NOT a Prandtl defect, which retires the
#                  campaign's leading hypothesis
#       PARTIAL    anything else; say which half moved
#
# (2) THE MIXING-LENGTH A/B, judged on the TENDENCY, not on a day-30 value.
#     The tripole's cold tongue is a growing drift (nino3 +0.55/+0.93/+1.42 at
#     days 10/20/30) while NEMO and FESOM both settle, so a day-30 number is a
#     rate caught mid-flight.  The discriminating statistic is the SST change
#     from day 10 to day 30 at 220-240E, |lat|<=2, where NEMO cools 1.67 C,
#     FESOM cools like NEMO, and we WARM 0.74 C:
#       CONFIRMED  choice 4 moves that tendency toward NEMO by >= 0.5 C
#       REFUTED    it moves it by < 0.2 C -- the deviation is real but not
#                  what drives the cold tongue, and it should be fixed anyway
#                  for faithfulness while the search moves elsewhere
#     Report the global SST rmse either way: a tropical win bought with a
#     global loss is a trade, not a fix.
#
# WHY THIS COSTS GPU HOURS AND CANNOT BE DONE OFFLINE.  The diffusivities are
# built inside the step and no snapshot on disk carries them, so there is no
# existing artifact to read; and a mixing length changes the integration, so
# its effect is not a rescoring.  Both questions are answered by the same two
# runs, which is why they are one launch.
#
# The dump is diagnostic-only -- one extra tendency evaluation per snapshot,
# no prognostic field touched -- so arm A is also a check on that: its SST
# scores must reproduce the existing modfrc_d30 numbers (rmse 0.515, nino3
# +1.42 at day 30 against GATEWAY rec 5).
# ============================================================================
set -uo pipefail
MXL=${1:?usage: sbatch _ab_mxl_nemo_faithful_d30.sbatch <3|4> <pinned-sha>}
PIN=${2:?usage: sbatch _ab_mxl_nemo_faithful_d30.sbatch <3|4> <pinned-sha>}
case "$MXL" in 3|4) ;; *) echo "FATAL: mxl choice must be 3 or 4"; exit 1;; esac
ARM=mxl${MXL}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_${ARM}_d30

echo "=== gate: the snapshot K dump and the probe that reads it ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/unit/test_snapshot_kprofiles.py \
  tests/ocean/unit/test_equatorial_diffusivity_oracle.py \
  tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate tests failed rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0822 --snapshot-every-days 10 --diag-every-days 5 \
  --kprofile-snapshots \
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
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice $MXL --tke-prognostic \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
GW_W=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
export JAX_PLATFORMS=cpu

# The MATCHED reference: NEMO's own cold start from the same January state.
# rec 1 = days 6-10, rec 3 = days 16-20, rec 5 = days 26-30.
for pair in "0010 1" "0020 3" "0030 5"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || SNAP=$OUT/snapshot_final.npz
  [ -f "$SNAP" ] || { echo "no snapshot for day $1"; continue; }
  echo "=== ${ARM} day $1 vs GATEWAY rec $2 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label ${ARM}_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_${ARM}_d$1_gw 2>&1 | tail -32
done

echo "=== ${ARM} equatorial diffusivity vs NEMO's published avm/avt (day 30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW_W" --rec 5 \
  --legoesm-snapshot $OUT/snapshot_day0030.npz 2>&1 | tail -90
echo "=== eqdiff rc=$?"

echo "=== ${ARM} equatorial thermocline structure (day 30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot $OUT/snapshot_day0030.npz --label ${ARM} \
  --nemo-gridt "$GW_T" --nemo-rec 5 2>&1 | tail -40
echo DONE
