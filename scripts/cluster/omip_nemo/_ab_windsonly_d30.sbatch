#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=wndonly
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/wndonly_%j.out
#
# ============================================================================
# WAS IT THE WIND?  The arm that separates the forcing correction's channels.
#
# Moving to the fields NEMO reads took the cold-tongue bias from +2.56 to
# +1.42 C and global SST error from 0.750 to 0.515, and that was reported as
# evidence that our equatorial trades had been a third too weak.  The
# adversarial review refused the attribution, correctly: the corrected cache
# changes more than the wind.  Measured on the caches themselves, area
# weighted, whole year:
#
#     channel        raw        corrected
#     wind speed    6.206        6.742 m/s   (+8.6% global, +33% in nino3)
#     air temp    287.107      287.495 K
#     humidity   0.010373     0.010090 kg/kg
#     shortwave   189.346      181.815 W/m2   (-7.5)
#     longwave    343.339      343.103 W/m2
#     precip      identical to 6 figures
#
# A 7.5 W/m2 global shortwave reduction is not a small term next to a tropical
# warm bias, and the undercurrent -- which the wind story predicts must
# respond -- did not.  So the wind explanation is PLAUSIBLE, not measured.
#
# THE DECOMPOSITION.  Three arms, each one variable from its neighbour:
#     control      raw winds,       raw thermodynamics
#     THIS ARM     CORRECTED winds, raw thermodynamics
#     modfrc       corrected winds, CORRECTED thermodynamics
# control -> this arm is the wind alone.  This arm -> modfrc is the
# temperature, humidity and radiation alone.  They must sum to the total.
#
# The forcing cache is verified winds-only before it is used, by comparing the
# same statistic across all three caches rather than against a remembered
# number (_verify_nyf_variants.sbatch).
#
# PRE-REGISTERED, day 30 against NEMO's matched cold start (GATEWAY record 5),
# where the control scores nino3 +2.56 and modfrc +1.42:
#   nino3 bias : WIND-DOMINATED if this arm lands below +1.7, i.e. it captures
#                at least three quarters of the 1.14 C total.
#                RADIATION-DOMINATED if it stays above +2.2, i.e. under a
#                quarter -- in which case the cold-tongue fix is a shortwave
#                result, the wind story is wrong, and the next lever is
#                radiative rather than dynamical.
#                MIXED in between, and then both channels get named.
#   EUC 200E   : the wind story predicts the undercurrent responds to wind and
#                not to radiation.  Report it either way; a wind-only arm that
#                reproduces modfrc's collapsed undercurrent (0.116 against the
#                control's 0.163) confirms the winds caused it.
#   global SST : report, so a channel that helps the tropics and hurts
#                elsewhere is visible.
#
# NOT A PRODUCTION CONFIG.  This cache is not what NEMO reads; it exists to
# attribute, and no fidelity verdict may be scored from it.
# ============================================================================
set -uo pipefail
ARM=windsonly
VISC_FLAGS="--C-smag-lap 3.0"
PIN=${1:?usage: sbatch _ab_modforcing_d30.sbatch <pinned-sha>}
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
OUT=$ROOT/results/omip_nemo/nemolev_trp_windsonly_d30

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
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 --forcing-path /burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_windsonly \
  --output $OUT 2>&1 | grep --line-buffered -E "fwbudget|\[setup\]|\[diag\]|Error|error" | cat
run_rc=$?
echo "=== run rc=$run_rc"
# codex 9430935 MAJOR-4: never score a failed or snapshot-less arm -- a stale
# or missing snapshot scored silently looks like a completed arm.
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
EXP_GATEWAY_V=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: $SNAP missing -- not scoring"; exit 1; }

echo "=== score: Southern-Ocean ice volume tendency (the quantity under test) ==="
JAX_PLATFORMS=cpu $PY scripts/validate/ocean_fidelity/so_ice_volume_tendency.py \
  --run-dir "$OUT" --days 6 --label "windsonly" 2>&1 | tail -14
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GW_T" --nemo-t-recs 5:6 \
  --label windsonly_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$EXP_GATEWAY_V" 2>&1 | tail -140
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO matched cold start (GATEWAY rec 5) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label windsonly_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_gw_windsonly_d30 2>&1 | tail -40
echo "=== compare rc=$?"
