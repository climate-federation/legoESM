#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=iceab30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/iceab30_%j.out
#
# ============================================================================
# CARD REVERT TEST: is --ice-ocean-heat-coeff 70 too large?  ONE VARIABLE.
#
# WHY. The production card runs 70 because a 2026-07-28 help-string claims the
# model default 20 sits "3-5x below NEMO's u*-dependent MIZ exchange".  That
# claim is now contradicted by measurement.  Southern-Ocean ice-volume
# tendency south of 45S, true cell area, rho_ice=917, days 3-5 mean, against
# NEMO's runoff+ice residual of 11.85e-6 kg/m2/s (empmr - emp_oce, recs 5:6):
#
#     coeff 70 (card)     33.7   2.84x NEMO
#     coeff 20 (default)  21.7   1.83x NEMO
#     coeff  0            14.9   1.25x NEMO
#
# Going 70 -> 0 removes 86% of the excess, so the coefficient is the DOMINANT
# term (an earlier extrapolation of mine said otherwise and was refuted by the
# coeff=0 arm).  At 20 we still melt 1.8x too fast, i.e. the July claim has
# the SIGN of the error backwards.
#
# WHAT THIS ARM ADDS. The 5-day probes measured the melt rate only. Melt is
# not the deliverable -- SST, SSS and MLD are -- and less melt means less
# freshwater, which could WORSEN the Southern-Ocean surface salinity that is
# already an open defect. This runs 30 days so the consequence is visible.
#
# PRE-REGISTERED, day 30 vs NEMO m1, coeff 20 against the card's 70:
#   SH sea-ice area: the card melts SH ice to near zero by d30. CONFIRMED if
#     coeff 20 retains visibly more ice; REFUTED if the two are within 5%,
#     which would mean the d5 melt difference does not survive to d30.
#   SO (S of 45S) SST bias: CONFIRMED improvement if |bias| falls; the melt
#     excess should have been cooling the band via latent heat.
#   SO (S of 45S) SSS bias: THE RISK, and it cuts against the change. Less
#     melt = less freshwater = saltier. Report the signed change either way;
#     if SSS degrades by more than SST improves, the coefficient is a
#     trade, not a fix, and the card should NOT be reverted on this evidence.
#   nino3 / global RMSE: must not move materially (a polar coefficient has no
#     business changing the tropics). Any large move = confound, investigate
#     before believing the polar numbers.
#
# NOT A TUNING TARGET. The melt instrument is an inventory tendency: it
# includes ice exported across 45S and excludes snow melt. Do not tune the
# coefficient to make 33.7 equal 11.85.
# ============================================================================
set -uo pipefail
ARM=rfw
VISC_FLAGS="--C-smag-lap 3.0"
PIN=${1:?usage: sbatch _ab_icecoef_d30.sbatch <pinned-sha> <coeff>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
# Codex 9442707 MAJOR: every coefficient arm wrote the SAME directory, so a
# second arm silently overwrote the first one's snapshots and the comparison
# would have been scored against whichever ran last. Key the output on the
# coefficient under test.
OUT=$ROOT/results/omip_nemo/nemolev_trp_iceab${2:?coeff}_d30

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
  --years 0.0822 --snapshot-every-days 1 --diag-every-days 5 \
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
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff ${2:?coeff} \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | grep --line-buffered -E "fwbudget|\[setup\]|\[diag\]|Error|error" | cat
run_rc=$?
echo "=== run rc=$run_rc"
# codex 9430935 MAJOR-4: never score a failed or snapshot-less arm -- a stale
# or missing snapshot scored silently looks like a completed arm.
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
EXP_GATEWAY_V=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: $SNAP missing -- not scoring"; exit 1; }

echo "=== score: Southern-Ocean ice volume tendency (the quantity under test) ==="
JAX_PLATFORMS=cpu $PY scripts/validate/ocean_fidelity/so_ice_volume_tendency.py \
  --run-dir "$OUT" --days 6 --label "icecoef=${2}" 2>&1 | tail -14
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 \
  --label icecoef${2}_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$EXP_GATEWAY_V" 2>&1 | tail -140
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label icecoef${2}_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_icecoef${2}_d30_m01 2>&1 | tail -40
echo "=== compare rc=$?"
