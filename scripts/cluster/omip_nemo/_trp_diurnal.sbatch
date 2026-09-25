#!/bin/bash
#SBATCH --job-name=diurn
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/diurn_%j.out
#
# THE NIGHT-TIME MEASUREMENT the convection arm could not make.  Two model
# days from the SAME day-15 restart, hourly box-mean temperature at eight
# depths, run twice: EVD=1 reproduces the arm, EVD=0 its control.  The arm's
# 30-day null was read at 00Z = mid-afternoon local in the cold tongue, where
# nothing fires in either configuration; this resolves the whole daily cycle,
# so "does nightly convection erode the warm layer" becomes a direct reading
# rather than an inference from one hour.
#   EVD=1 sbatch _trp_diurnal.sbatch <sha>     (the scheme on)
#   EVD=0 sbatch _trp_diurnal.sbatch <sha>     (its control)
# BOTH restart from the SAME day-15 state, which is what makes this a clean
# pair.  The restart guard refuses that for the no-convection side, correctly:
# a silent configuration difference makes a chained leg meaningless.  This is
# the legitimate exception -- one physics switch flipped, same state -- so it
# is taken explicitly with --restart-branch-from-different-config, which says
# so in the log and in the manifest.
set -uo pipefail
PIN=${1:?usage: EVD=0|1 sbatch _trp_diurnal.sbatch <pinned-sha>}
EVD=${EVD:?set EVD=0 or EVD=1}
LEG=2
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/trp_diurnal_evd${EVD}

echo "=== gate ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --A-h 2e4 --A-h-profile-file $NEMOIN/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc \
  --visc-schedule '0:1e4:3.0,8:1e4:1.5,16:1e4:0.5' \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --years 0.0466 --snapshot-every-days 2 --diag-every-days 1 \
  --restart-from /burg-archive/glab/users/pg2328/_evd_wt/results/omip_nemo/trp_evd_d30/restart_leg1.npz \
  $([ "$EVD" = 0 ] && echo "--restart-branch-from-different-config") \
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
  --gateway-transports --mle --mle-ce 0.06 --kprofile-snapshots \
  $([ "$EVD" = 1 ] && echo "--convection enhanced_diffusion --convection-K-conv 100 --convection-K-bg 0 --convection-n2-mode nemo_bn2 --convection-n2-eos teos10 --convection-trigger hard --convection-n2-threshold=-1e-12") \
  --evd-occupancy-every-hours 1 \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }
echo "=== hourly box profile (the deliverable): $OUT/evd_occupancy.csv ==="
head -1 $OUT/evd_occupancy.csv
tail -n 50 $OUT/evd_occupancy.csv
echo DONE
