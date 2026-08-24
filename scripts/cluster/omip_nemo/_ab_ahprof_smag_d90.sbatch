#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ahpsmag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=44:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ahpsmag_%j.out
#
# ============================================================================
# FAITHFUL A_h RAMP **WITH** THE SMAGORINSKY BACKSTOP (the corrected config).
# WHY THIS EXISTS: the Smag-OFF twin (9435841) restored the equatorial
# circulation at d20 (upwelling -12 -> +32e-6 m/s, nino3 +2.17 vs ctl +2.50,
# growth arrested) but then RAN AWAY on the equator -- max|u| 1.46 (d5),
# 1.90 (d20), 2.50 (d35), 3.65 (d50), 11.53 (d65), doubling every ~15 d at
# lat~0, lon 222-230E, with the global mean SST starting to fall. Cancelled
# at d65; its d10..d60 snapshots are kept.
# THE ONE CHANGE vs that run: --C-smag-lap 3.0 (the flow-adaptive term the
# eqonly arm KEPT, and which stayed stable at max|u| 1.27). The prescribed
# NEMO ahm ramp is unchanged, so this tests exactly whether the background
# viscosity can be NEMO-faithful while our own adaptive term supplies the
# stability NEMO gets from its 3-D field + different numerics.
# PRE-REGISTERED, day 90 vs NEMO m3 (ctl signedn2_d90 bracketed):
#   max|u| : CONFIRMED stays < 2.5 through d90 (no doubling trend);
#            REFUTED if it exceeds 4 -- then the ramp itself is the problem
#            and the equatorial floor must be raised above the file's 1000.
#   nino3 [+2.78]: CONFIRMED <= +1.5; PARTIAL +1.5..+2.5.
#   EUC 200E [ctl 0.22; NEMO 0.82]: CONFIRMED >= 0.6.
#   SH_midlat SST: Smag is BACK, so the +0.79 Smag-off warm cost should not
#            appear; report it either way.
# ============================================================================
set -uo pipefail
ARM=ahprof
VISC_FLAGS="--A-h 20000 --C-smag-lap 3.0 --A-h-profile-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc"
PIN=${1:?usage: sbatch _ab_ahprofile_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_ahpsmag_d90

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
