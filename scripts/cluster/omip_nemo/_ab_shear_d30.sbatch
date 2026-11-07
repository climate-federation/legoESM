#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=shear30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/shear30_%j.out
#
# ============================================================================
# FACE-NATIVE SHEAR PRODUCTION — the last unclosed knob on the ORCA1 card,
# now promoted from fidelity item to PRIME SUSPECT by measurement:
#   * 10 m Ekman drift at 240E is 3-4x too weak on BOTH sides of the equator
#     (ours +0.016..0.07 vs NEMO +0.11..0.20 at 1-4N; matched NCAR stress,
#     vfac=0 both, stress deposited into the same 1 m top cell) => the
#     divergence that upwells the cold tongue never forms (job 9433989);
#   * identical in the eqonly arm => NOT lateral viscosity;
#   * our equatorial Prandtl sits on its 10.0 ceiling on 83% of interfaces
#     where NEMO's median is 1.85, from Ri measured 1.43x too high — and Ri
#     is inflated exactly when SHEAR is under-measured. squared_centered
#     collapses face shears to T points (mean-of-gradients <= gradient-of-
#     means); NEMO's zdfsh2 is face-native NOW x BEFORE.
# ONE variable vs nemolev_trp_signedn2_d30: --tke-shear-production
# nemo_face_native.
# PRE-REGISTERED, day 30:
#   v(10 m) at 240E, +1..4N mean [ctl ~+0.02..0.07; NEMO +0.11..0.20]:
#     CONFIRMED >= +0.08 mean; REFUTED unchanged within 30%.
#   180-80W w-box ratio [0.42]: CONFIRMED > 0.7.
#   nino3 m1 bias [+3.19]: improvement expected; direction only (the SST
#     response lags the circulation, per the eqonly arm).
# ============================================================================
set -uo pipefail
ARM=shear
VISC_FLAGS="--C-smag-lap 3.0"
PIN=${1:?usage: sbatch _ab_shear_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_shear_d30

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
  --years 0.0822 --snapshot-every-days 10 --diag-every-days 5 \
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
  --tke-kappa-convention veros_sqrte --tke-shear-production nemo_face_native_nbb2 --sw-rgb-chl --ice-ocean-heat-coeff 70 \
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
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot at $SNAP -- not scoring a partial arm"; exit 1; }
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 \
  --label shear_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$EXP_GATEWAY_V" 2>&1 | tail -140
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label shear_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_shear_d30_m01 2>&1 | tail -40
echo "=== compare rc=$?"
