#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=icecoef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/icecoef_%j.out
#
# ============================================================================
# ICE-OCEAN HEAT COEFFICIENT: one variable, 5 days. usage: sbatch <sha> <coeff>
#
# THE TENSION THIS RESOLVES. The card runs --ice-ocean-heat-coeff 70 because
# a 2026-07-28 probe (quoted in the flag's own help) measured the default 20
# as 3-5x BELOW NEMO's u*-dependent marginal-ice-zone exchange. But the
# freshwater budget measured TODAY (probe 9442075, two independent
# instruments) says our SO melt is 52-63e-6 kg/m2/s against NEMO's ENTIRE
# runoff+ice budget of 16.67e-6 -- i.e. ~3.2x too FAST, sustained d2-d5,
# melting out 11% of the SO ice volume in 5 days.
# Both cannot be right as stated, and the older number is PROSE until
# re-measured, so this runs the same 5-day melt diagnostic at a given coeff.
# READ: the [fwbudget-bands] antarctic 'ice=' column and the d1..d5 ice
# volume trajectory, against NEMO's 16.67e-6 total.
#   coeff 20 (model default) melt ~= 16-20e-6  => the 70 is the cause, revert
#   coeff 20 still >> 16.67                    => the coefficient is NOT the
#                                                 lever; look at SW under ice
#                                                 or the ice thermo albedo
#
# 2026-08-20 RESULT, and a correction to the units above. Both sides of that
# comparison were weighted by cos(lat) on a curvilinear mesh, and our side
# converted ice with 1000 kg/m3 instead of rho_ice=917. On true area, with
# the right density (commit 036832cdb):
#   NEMO runoff+ice residual S of 45S : 11.85e-6   (not 16.67)
#   coeff 70, days 3-5                : 33.7e-6    (2.8x NEMO)
#   coeff 20, days 3-5                : 21.7e-6    (1.8x NEMO)
# So NEITHER preregistered branch fired cleanly: a 3.5x cut in the
# coefficient buys 36% less ice loss -- a real but SUB-LINEAR lever that
# does not reach NEMO. Linear extrapolation (0.236e-6 per unit coefficient)
# says even coeff=0 leaves ~19e-6, still 1.6x NEMO, which would mean a heat
# source OTHER than the basal turbulent exchange dominates our melt.
#
# That extrapolation across a nonlinear closure is PLAUSIBLE, not measured,
# so coeff=0 is run to settle it. PRE-REGISTERED, days 3-5 mean:
#   >= 15e-6  => CONFIRMED: the exchange coefficient cannot close the gap;
#                the dominant term is elsewhere (SW under ice, ice albedo,
#                or a too-warm mixed layer). Stop tuning the coefficient.
#   <= 12e-6  => REFUTED: the extrapolation was wrong, the coefficient IS
#                the whole story, and the card's 70 is simply too large.
#   12-15e-6  => inconclusive; the lever is real but partial, and the
#                remaining gap is inside the instrument's known caveats
#                (inventory tendency includes export, excludes snow).
# ============================================================================
set -uo pipefail
ARM=rfw
VISC_FLAGS="--C-smag-lap 3.0"
PIN=${1:?usage: sbatch _ice_heatcoeff_probe.sbatch <pinned-sha> <coeff>}
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
OUT=$ROOT/results/omip_nemo/nemolev_trp_icecoef${2:?coeff}_probe

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
  --years 0.0164 --snapshot-every-days 1 --diag-every-days 5 \
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
# This is a 1-DAY diagnostic run: the whole deliverable is the step-1
# [fwbudget-bands] print above, so there is no day-30 snapshot to score and
# the inherited scoring gate does not apply.
echo "=== probe complete; the [fwbudget-bands] lines above ARE the result ==="
exit 0
SNAP=$OUT/snapshot_day0030.npz
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 \
  --label rfw_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$EXP_GATEWAY_V" 2>&1 | tail -140
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label rfw_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_rfw_d30_m01 2>&1 | tail -40
echo "=== compare rc=$?"
