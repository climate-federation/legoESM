#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ahsmod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=44:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ahsmod_%j.out
#
# ============================================================================
# THE FOURTH CORNER: corrected CORE-II forcing x faithful lateral viscosity.
#
# Two independent faithfulness corrections have each been measured alone, and
# they fix different halves of the same equatorial defect.  This arm runs both
# and completes a 2x2 factorial, so every cell differs from two neighbours by
# exactly one variable.
#
#                          raw CORE-II winds     corrected (_MOD) winds
#   flat A_h + Smag 3.0    signedn2_d30          modfrc_d30
#   ORCA1 A_h file + 1.0   ahsmag1_d90           THIS ARM
#
# WHAT EACH NEIGHBOUR MEASURED, day 30 unless noted, vs NEMO ORCA1:
#   signedn2 (control)  nino3 +3.19, global SST rmse 1.139, EUC 200E ~0.34
#   modfrc              nino3 +2.05, global SST rmse 0.928, EUC 200E  0.116
#                       -- the cold tongue responded, the undercurrent did NOT
#   ahsmag1             global SST rmse 1.132 at day 30; at day 90 the EUC at
#                       240E is 0.926 against NEMO's 1.018, i.e. essentially
#                       right, and the run is stable for the full 90 days
#
# So the corrected winds are the cold tongue's lever and the ORCA1 viscosity
# file is the undercurrent's.  Neither alone gives both.  The open question is
# whether they compose or interfere: stronger trades put more momentum into a
# far less viscous equatorial band, which is exactly the combination neither
# previous run tested.
#
# WHY THIS IS NOT TUNING.  NEMO's namelist reads the Large & Yeager _MOD
# fields, and nn_ahm_ijk_t=-30 reads eddy_viscosity_3D.nc (20000 m2/s at
# midlatitudes ramping to 1000 at the equator).  Both settings here are the
# oracle's own; the flat A_h and the raw winds were our approximations.
#
# PRE-REGISTERED, fixed before launch.  Day 30 against NEMO month 1, scored
# with the same instrument and the same box definitions as modfrc_d30:
#   nino3 SST bias : CONFIRMED if <= +1.2 (the two fixes compose)
#                    PARTIAL   +1.2 to +1.8
#                    REFUTED   >= +2.0 -- the viscosity file adds nothing once
#                    the forcing is right, and the remaining cold-tongue bias
#                    is neither a momentum-source nor a dissipation defect.
#   EUC 200E core  : must be >= 0.30 m/s (NEMO 0.496).  REFUTED below 0.20,
#                    which would mean the corrected winds actively destroy the
#                    undercurrent the viscosity file restores.
#   global SST rmse: must be <= 0.93 (modfrc 0.928).  A tropical win bought
#                    with a global loss is a trade, not a fix.
#   SH_midlat SST  : report either way.  Smag 3.0 -> 1.0 warmed this band by
#                    +0.79 C in the Smag-off twin, so it is the known cost.
#   max|u|         : stability tripwire, must stay under 2.5 m/s through day
#                    90.  The Smag-off ahprof twin grew monotonically to 6.21.
#
# WALLTIME 44 h for a ~24 h run: the ahsmag1 twin took 34 h and a timeout
# costs the day-90 half of the answer.
# ============================================================================
set -uo pipefail
ARM=ahsmod
VISC_FLAGS="--A-h 20000 --C-smag-lap 1.0 --A-h-profile-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc"
PIN=${1:?usage: sbatch _ab_ahprof_smag1_modfrc_d90.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_ahsmod_d90

echo "=== gate: unit tests for the reduction semantics, CLI wiring, forcing default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_polar_cap_boost.py \
  tests/ocean/unit/test_core2_corrected_cache_default.py \
  tests/unit/test_run_omip_core2_eq_visc_shaping.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate tests failed rc=$rc"; exit 1; }

# The corrected cache is the default since 24a25b286, so this arm passes no
# --forcing-path; the run's own "[forcing] CORE-II NYF cache:" line records
# which cache actually resolved, with the nino3 wind speed that identifies it
# (~6.28 corrected, ~4.74 raw).  Verify the LOG, never the flag.
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
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
GV=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc
export JAX_PLATFORMS=cpu

# Day 30 is the pre-registered cell of the factorial -- it is the only day all
# four corners have in common, so it is the one that may be compared.
SNAP30=$OUT/snapshot_day0030.npz
[ -f "$SNAP30" ] || { echo "FATAL: no day-30 snapshot at $SNAP30"; exit 1; }

echo "=== score day 30: Z20 + upwelling + EUC (NEMO 5-day recs 5:6 = days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP30" --nemo-gridt "$GT" --nemo-month 1 \
  --label ${ARM}_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$GV" 2>&1 | tail -140
echo "=== eqtherm d30 rc=$?"

echo "=== score day 30: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP30" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label ${ARM}_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_${ARM}_d0030_m01 2>&1 | tail -40
echo "=== compare d30 rc=$?"

SNAP90=$OUT/snapshot_day0090.npz
if [ -f "$SNAP90" ]; then
  echo "=== score day 90: Z20 + upwelling + EUC (NEMO recs 17:18) ==="
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP90" --nemo-gridt "$GT" --nemo-month 3 \
    --label ${ARM}_d90 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 17:18 \
    --nemo-ufile "$GU" --meridional-lon 240,260 \
    --nemo-vfile "$GV" 2>&1 | tail -140
  echo "=== eqtherm d90 rc=$?"
  echo "=== score day 90: full skill vs NEMO m3 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP90" \
    --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label ${ARM}_d90 \
    --output-dir $ROOT/results/omip_nemo/compare_${ARM}_d0090_m03 2>&1 | tail -40
  echo "=== compare d90 rc=$?"
else
  echo "NOTE: no day-90 snapshot -- day-30 half of the answer stands, day-90 does not"
fi
echo DONE
