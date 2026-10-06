#!/bin/bash
#SBATCH --job-name=orca1_faith
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/orca1_faith_%j.out
#
# THE ORCA1-FAITHFUL PRODUCTION CARD, REBUILT.
#
# The first version of this card was written FROM SCRATCH with about twenty
# flags and it went 100% NaN in velocity within a day. The cause was not the
# adopted physics: the card simply was not a working configuration. The
# known-good baseline (_trp_base2_d30.sbatch) carries about forty flags --
# the equatorial viscosity shape and profile, the Smagorinsky settings, the
# NEMO vertical coordinate, partial cells, the SMC03 pressure gradient,
# min-levels, TKE vertical mixing, bottom drag, the restoring stack, the
# geothermal and internal-wave forcing, the monthly IC -- and omitting any of
# that is not "a simpler run", it is a different and unstable model.
#
# So this card is now the BASELINE'S OWN FLAG BLOCK, verbatim, PLUS the four
# adopted sets appended at the end. That is the only honest way to claim the
# only difference is the adoption.
#
# THIS IS A NEW BASELINE, NOT AN ARM: the adopted sets move together by user
# authorisation, so the result is NOT attributable term by term against the
# pre-adoption baseline, and no single lever may be credited for a difference.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
# TWO LEGS, as the baseline did: ~0.70 h per simulated day means 30 days is
# ~21 h, well past the 12 h partition cap. --years is the ABSOLUTE target, so
# leg 2 asks for the FULL 30 days and resumes from leg 1's restart.
DAYS=${DAYS:-30}
LEG=${LEG:-1}
HALF=$($PY -c "print(${DAYS}/2)")
TARGET=$([ "$LEG" = 1 ] && echo $HALF || echo $DAYS)
YEARS=$($PY -c "print(${TARGET}/365.0)")
# trp_orca1faith_d30 is the pre-mask baseline every 2026-09-13/14 measurement
# references; the masked card writes beside it, never over it.
# trp_orca1faithmask_d30 = mask only (leg 1 to day 15, 9769155); this card now
# also carries NEMO's initial-condition time blend, so it writes beside both.
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithtint_d${DAYS}}
echo "DAYS=$DAYS LEG=$LEG TARGET=$TARGET YEARS=$YEARS OUT=$OUT"
# The run's stdout used to go straight into "tail -40", which buffers the whole
# leg and then discards all but the last forty lines.  Everything the driver
# says about what it RESOLVED -- the A_h profile it built, the viscosity each
# schedule segment applies, the forcing paths -- fires in the first minutes and
# was therefore unreadable, so the half-viscosity defect ran a whole campaign
# with its own log line printing the correct diagnosis into a pipe nobody could
# read.  Keep the full log on disk beside the output; tail still goes to the
# SLURM file so the job summary is unchanged.
mkdir -p $OUT
# NEMO runs no Smagorinsky at all -- its lateral momentum viscosity is read
# entirely from eddy_viscosity_3D.nc (nn_ahm_ijk_t = -30) -- while this card
# adds one and ramps it down mid-run.  The two knobs below let an arm turn
# that term off without touching any other flag; the DEFAULTS ARE THE CARD'S
# OWN VALUES, so an unparameterised submission runs exactly what it always
# ran.  Both columns must move together: --C-smag-lap sets the value the
# schedule's first segment inherits before day 0.
CSMAG=${CSMAG:-3.0}
VSCHED=${VSCHED:-0:1e4:3.0,8:1e4:1.5,16:1e4:0.5}
echo "CSMAG=$CSMAG VSCHED=$VSCHED"

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap $CSMAG --smag-cfl-safety 0.125 \
  --A-h 2e4 --A-h-profile-file $NEMOIN/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc \
  --visc-schedule "$VSCHED" \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --barotropic-slow-forcing-depth-evaluation min_rule_live `# user 2026-10-05: old method; NEMO average is ~99% inert under implicit CN (needs split-explicit + carried mode)` \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --years $YEARS --snapshot-every-days 5 --diag-every-days 1 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  `# ---- initial condition = what NEMO starts from: fldread ln_tint puts the` \
  `# 1 January state at 0.5*December + 0.5*January of the monthly climatology ----` \
  --nemo-init-tint \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --tke-langmuir-rhs-coupling nemo_pre_solve `# user 2026-10-06: NEMO Langmuir/dissipation RHS order (zdftke.F90:367,419)` \
  --gateway-transports --mle --mle-ce 0.06 --kprofile-snapshots \
  `# ---- THE FOUR ADOPTED SETS, and nothing else, appended ----` \
  --ice-freeze-scheme unesco \
  --gm-treguier --gm-kappa-min 0 \
  --gm-slope-scheme nemo_iso_lap --gm-slope-positions nemo_native \
  --gm-bolus-advection through_fct --gm-msc-stabilize \
  --redi-coefficient nemo21 --redi-aht0 900 \
  --ice-exchange nemo_si3 --ice-snow bulk --ice-snow-flooding on \
  --forcing-time-interp linear --chl-remap nemo_scrip \
  `# ---- wet domain = the oracle's: the Great Lakes, Caspian and Lake Victoria` \
  `# (207 cells wet in mesh_mask.nc, land in NEMO's domain_cfg) become land ----` \
  --nemo-domain-mask \
  `# EOS stays wright: --eos nemo_teos10 is REFUSED because NEMO's TEOS-10` \
  `# takes Conservative Temperature and ABSOLUTE salinity while our tracers` \
  `# are potential temperature and PRACTICAL salinity.` \
  --restart-save $OUT/restart_leg$LEG.npz \
  $([ "$LEG" = 2 ] && echo "--restart-from $OUT/restart_leg1.npz") \
  --output $OUT 2>&1 | tee $OUT/run.log | tail -40
echo "=== card exit: $? ==="
echo "OUT=$OUT"
echo DONE
