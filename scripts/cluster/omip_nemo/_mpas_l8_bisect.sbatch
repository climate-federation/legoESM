#!/bin/bash
#SBATCH --job-name=l8bisect
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/l8bisect_%j.out
# LEVEL-8 BLOWUP BISECT. The ico8 real-FW card blows up from rest x5/step from
# step 5 (max|u| 0.74 -> 3750 m/s by step 10; first at the Black Sea at ~800 m,
# then the Alboran Sea at ~1200 m). Mesh quality, barotropic CFL (0.67, same as
# level 7 at its dt), the IC and the t=0 geometry are all exonerated
# (probe_mpas_e3w_geometry.py). ONE operator removed per arm, 30 steps each,
# per-step diag. PRE-REGISTERED: the first arm whose step-12 max|u| stays
# < 1 m/s names the operator; if none does, the seed is in the base dynamics
# (partial-cell PGF on the steeper level-8 slopes) and the next arm is
# --no-partial-cell / bathymetry smoothing.
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_l8_bisect.sbatch <pinned-sha>}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
R=$ROOT/results/omip_nemo
BASE="--grid mpas --mpas-level 8 --mpas-lloyd 20 --nemo-vertical
  --dm2dc --sw-rgb-chl --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5
  --sss-restore-bound-mmday 4 
  --years 0.0000685 --snapshot-every-days 1
  --freshwater-closure real_freshwater --sss-restore-channel water_flux
  --sss-restore-normalization live_s
  --diag-every-days 0.0009 --tracer-advection superbee --runoff-depth-nemo-ini
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc
"
NEMOIC="--nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc"
WOA="--woa-init"
ICE="--sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --ice-lead-freeze-source nemo_qlead --ice-ocean-heat-coeff 70 --ice-init $NEMOIN/Ice_initialization.nc"
TKE="--mpas-vmix tke --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic --tke-kappa-convention veros_sqrte"
ISF="--isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc"
IWM="--iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc"
run_arm () {  # name, extra flags
  name=$1; shift
  OUT=$R/l8bisect_$name; rm -rf $OUT
  echo; echo "########## ARM $name"
  JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py $BASE "$@" --output $OUT 2>&1 | grep -E "rc=|Error|must contain|BLOWUP|\[run\]" | tail -5
  echo "--- $name per-step max|u| (step,max_abs_u,lat,lon,lev):"
  cut -d, -f1,5,7,8,9 $OUT/diag_timeseries.csv 2>/dev/null | sed -n '2,32p' | tr '\n' ' ' ; echo
}
# LADDER 8 -- THE OPERATOR. Per-term dump on the step-6 state (probe 9635536):
# visc = -0.31 m/s^2 at the max edge (x75 s = -23 m/s per step on u=+3.7 =>
# the observed -19.6), every other term <= 1e-3. A_h/Smag-off still blew, so
# the offender is the third viscous component: K_zeta_bih = 1e14 m4/s, the
# EXPLICIT biharmonic vorticity filter of the NEMO-match recipe, sized for the
# 58 km level-7 mesh and NOT scaled with dx. Explicit biharmonic stability
# number K dt (6/dv^2)^2: level 7 (dv 29 km, dt 150) ~0.8 stable; level 8
# (dv 14 km, dt 75) ~6 => sign flip, growth |1-6| ~ 5/step = MEASURED.
# ONE VARIABLE: K_zeta_bih scaled by (dx8/dx7)^3 = 1/8 -> 1.25e13.
CTL="--dt 75 --partial-cell --mle --mle-ce 0.06 $TKE $ISF $IWM --bbl-adv $WOA $NEMOIC $ICE"
run_arm kzeta_dx3  $CTL --K-zeta-bih 1.25e13
run_arm kzeta_0    $CTL --K-zeta-bih 0
echo DONE
