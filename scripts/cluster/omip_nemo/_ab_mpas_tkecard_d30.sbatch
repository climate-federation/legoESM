#!/bin/bash
#SBATCH --job-name=mpastke
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpastke_%j.out
#
# ============================================================================
# MPAS ON THE SAME TKE CARD THE TRIPOLE RUNS.  ONE VARIABLE vs xgrid_mpas_d30.
#
# WHAT CHANGED, and only this: the four zdftke card knobs the tripole has been
# running -- the NEMO Dirichlet surface-TKE boundary condition (nn_bc_surf=1),
# mixing-length choice 3, the prognostic en integration, and the veros_sqrte
# diffusivity convention -- now reach the MPAS grid.  They were rejected off
# the tripole until 4ad4a3bc7, so every previous cross-grid comparison had the
# tripole on the faithful card and MPAS on the builder defaults: a difference
# in the CLOSURE reported as a difference between GRIDS.
#
# The surface boundary condition is the largest single lever this campaign has
# measured.  On the tripole it shipped as PR #1264; ported into fesom-jax it
# took that grid's global SST error from 1.555 to 1.160 and its Southern-Ocean
# warm bias from +1.76 to +0.45.  MPAS has never had it.
#
# STILL NOT MATCHED to the NEMO-faithful tripole card, and it must be said
# with every number this produces: --dm2dc, --isf, --bbl-adv, --sw-rgb-chl,
# --iwm and --nemo-monthly-init are wired for the structured grids only
# (verified in run_omip_core2.py, not inferred from a comment), so MPAS runs
# without diurnal shortwave, ice-shelf melt, bottom-boundary-layer advection,
# RGB chlorophyll attenuation, wave-driven interior mixing, and NEMO's January
# initial state.  This arm closes the CLOSURE gap, not those six.
#
# PRE-REGISTERED, day 30, scored against NEMO's MATCHED cold start
# (RUN_GATEWAY 5-day grid_T record 5 = days 26-30), not the old five-year
# January climatology.  The tripole on corrected forcing scores 0.515 global
# SST RMSE and +1.42 nino3 there.
#   global SST RMSE : CONFIRMED if it falls below the previous MPAS card's
#                     value by more than 0.05 C.  REFUTED if it rises.
#   Antarctic and SH-midlat SST bias : the Dirichlet BC's signature on the
#                     other two grids was a large warm-side correction here.
#                     CONFIRMED if both move toward zero.
#   Antarctic mixed layer : deepens on the other two grids (FESOM 19.9 -> 41.0
#                     against NEMO's 34.3).  Report either way.
#   cross-grid gap  : the point of the arm.  Report MPAS-minus-tripole for
#                     global SST RMSE and each band, and state the six
#                     unmatched flags next to it.
# ============================================================================
#
# BOTH ARMS RUN AT THIS PIN.  The previous MPAS result exists, but it was
# produced at a different commit from a dirty tree, so it is not a control for
# anything (codex CRITICAL, 2026-08-22).  Pass "card" or "nocard" as the second
# argument; the two differ in the four knobs and in nothing else.
set -uo pipefail
PIN=${1:?usage: sbatch _ab_mpas_tkecard_d30.sbatch <pinned-sha> card|nocard}
ARM=${2:?usage: sbatch _ab_mpas_tkecard_d30.sbatch <pinned-sha> card|nocard}
case "$ARM" in
  card)   KNOBS="--tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic --tke-kappa-convention veros_sqrte" ;;
  nocard) KNOBS="" ;;
  *) echo "FATAL: arm must be 'card' or 'nocard', got '$ARM'"; exit 1 ;;
esac
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
# A dirty MODEL tree makes the run unreproducible; dirty cluster scripts do
# not, since the scheduler already copied this file.  Check the code only.
DIRTY=$(git status --porcelain -uno -- packages src scripts/run scripts/validate)
[ -z "$DIRTY" ] || { echo "FATAL: model tree dirty:"; echo "$DIRTY"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository
OUT=$R/results/omip_nemo/nemolev_mpas_tke_${ARM}_d30

echo "=== gate: the card knobs must actually reach the MPAS config ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_tripole_vmix_cli.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate tests failed rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  --grid mpas --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke $KNOBS \
  --dt 150 --nemo-vertical --partial-cell --adaptive-implicit-vertadv \
  --momentum-rk3 --geothermal --runoff --sss-restore \
  --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo \
  --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0821918 --snapshot-every-days 10 --diag-every-days 5 \
  --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --woa-init \
  --woa-t $R/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s $R/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NIN/input_fields/sss_climatology_for_restoring.nc \
  --ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc \
  --ice-ocean-heat-coeff 70 \
  --output $OUT 2>&1 | tail -30
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot"; exit 1; }
export JAX_PLATFORMS=cpu

echo "=== score day 30 vs NEMO's MATCHED cold start (GATEWAY rec 5) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label mpas_tke_${ARM}_d30 \
  --output-dir $R/results/omip_nemo/compare_gw_mpas_tke_${ARM}_d30 2>&1 | tail -40
echo "=== compare rc=$?"
echo DONE
