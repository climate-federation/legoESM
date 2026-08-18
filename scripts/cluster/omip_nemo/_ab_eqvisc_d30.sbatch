#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=eqvisc30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqvisc30_%j.out
#
# ============================================================================
# EQUATORIAL VISCOSITY vs THE MISSING EUC — usage: sbatch ..._ab_eqvisc_d30.sbatch {nemo|eqonly}
#
# THE CHAIN (jobs 9430332 + 9430829, matched-window 5-day wo/uo):
#   EUC 3-4x too weak (200E d90: ours 0.22 vs NEMO 0.82 m/s; barely spins up
#   d30->d90 while NEMO doubles) -> ascent at 120W ABSENT/REVERSED (ours
#   -5.0e-6 m/s where NEMO +21.9e-6; 180-80W box ratio 0.28) -> Z20 +26-29 m
#   too deep at 120W -> nino3 +2.78 C (d90), the model's largest SST error.
#
# THE CONFIG GAP (read from both configs, not inferred): NEMO ORCA1 reads its
# momentum viscosity from eddy_viscosity_3D.nc (nn_ahm_ijk_t=-30): 20000 m2/s
# midlatitude REDUCED to 1000 m2/s within ~2 deg of the equator (the classic
# EUC-preserving reduction; ramp ~5-10 deg). Our card runs A_h=100000
# EVERYWHERE (no lat scaling) + Laplacian Smagorinsky C=3 — shear-adaptive,
# so LARGEST exactly in the EUC shear. ~100x NEMO at the equator.
#
# TWO ARMS, each ONE treatment vs control nemolev_trp_signedn2_d30 (9409057):
#   nemo   : the full NEMO-like lateral-viscosity block
#            A_h 1e5 -> 2e4, eq reduction 0.05 (sigma 7 deg), Smag-lap 3 -> 0.
#            Treatment = the viscosity BLOCK as a unit; attribution is to the
#            block, not any single knob.
#   eqonly : ONLY the equatorial reduction added to the card (A_h 1e5 and
#            Smag C=3 kept). Isolates the equatorial shaping alone.
#
# PRE-REGISTERED, day 30 (control values in brackets):
#   EUC max at 200E   [0.163; NEMO 0.496 m/s]: CONFIRMED > 0.30; REFUTED
#                     unchanged within +-20%.
#   180-80W w ratio   [0.42]: CONFIRMED > 0.7; REFUTED <= 0.5.
#   nino3 SST bias m1 [+3.19 C]: CONFIRMED <= +2.2; REFUTED >= +2.9.
#   WATCH: max|u| runaway / non-finite (the historical reason the viscosity
#   was raised). A blowup REFUTES the arm as-shipped and names the stability
#   constraint the faithful config must solve differently (NEMO holds 1000
#   m2/s at the equator stably with the same dt class).
# WHY A GPU RUN AND NOT OFFLINE: the question is the time-integrated momentum
# response (EUC spin-up d0->d30); no saved checkpoint contains the counter-
# factual trajectory. ~12 h at 0.39 steps/s measured on the d90 twin.
# ============================================================================
set -uo pipefail
ARM=${1:?usage: sbatch _ab_eqvisc_d30.sbatch nemo|eqonly}
case "$ARM" in
  nemo)   VISC_FLAGS="--A-h 20000 --A-h-eq-boost 0.05 --A-h-eq-sigma-deg 7.0 --C-smag-lap 0.0" ;;
  eqonly) VISC_FLAGS="--C-smag-lap 3.0 --A-h-eq-boost 0.05 --A-h-eq-sigma-deg 7.0" ;;
  *) echo "FATAL: unknown arm $ARM"; exit 1 ;;
esac
PIN=${2:?usage: sbatch _ab_eqvisc_d30.sbatch <arm> <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_eqvisc_${ARM}_d30

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
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -25
echo "=== run rc=$?"

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || SNAP=$OUT/snapshot_final.npz
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 \
  --label eqvisc_${ARM}_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 200,240 2>&1 | tail -60
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label eqvisc_${ARM}_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_eqvisc_${ARM}_d30_m01 2>&1 | tail -40
echo "=== compare rc=$?"
