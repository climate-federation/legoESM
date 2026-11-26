#!/bin/bash
#SBATCH --job-name=mpasmixharm
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasmixharm_%j.out
#
# MPAS WITH THE LATERAL-MIXING FLAGS IT CAN ACTUALLY HONOUR.
#
# A manifest audit on 2026-09-20 found the MPAS OMIP run passing NONE of the
# lateral-mixing or GM/Redi flags -- not different values, absent -- so it ran
# library defaults (A_h 100000, Smagorinsky 0.33) while the tripole ran the
# adopted NEMO-faithful card (A_h 20000 times NEMO's own 3-D profile,
# Smagorinsky 3.0). Every cross-grid "MPAS confirms" statement in this campaign
# compared two different physical configurations.
#
# ONLY FOUR OF THE TWELVE FLAGS ARE WIRED ON THIS LANE, and that was measured
# rather than assumed: build_mpas_ocean's signature accepts A_h, C_smag_lap,
# gm_treguier and gm_kappa_min and nothing else from the stack. The other eight
# now REFUSE on MPAS instead of being silently discarded (same commit), so this
# card cannot quietly under-deliver the way the last one did.
#
# NOT PASSED, AND WHY -- each verified in the source, not inferred:
#   --A-h-profile-file        hard-refused off tripole by design; NEMO's 3-D
#                             viscosity profile is built on eORCA nominal
#                             latitude rows and has no MPAS analogue.
#   --visc-schedule           tripole-only ramp.
#   --redi-coefficient        )
#   --redi-aht0               ) read only inside build_tripole.
#   --gm-slope-scheme         )
#   --gm-slope-positions      ) GM operator flags, already hard-refused
#   --gm-bolus-advection      ) off tripole before tonight.
#   --gm-msc-stabilize        )
#   --momentum-rk3            )
#   --adaptive-implicit-vertadv ) wired on tripole and fesom, not MPAS.
#   --min-levels              )
#   --smag-cfl-safety         )
#   --kprofile-snapshots      the MPAS model exposes no diagnose_vertical_K,
#                             so _kprofiles prints SKIPPED and writes nothing.
#   --gateway-transports      SUPPORTED_APP_GRIDS is ("tripole","latlon").
#
# THIS IS A NEW BASELINE, DELIBERATELY MULTI-CHANGE, AND IT IS LABELLED AS SUCH.
# Four physics knobs move at once, so no output is attributable to any one of
# them and this run must NEVER be compared term-by-term against the old MPAS
# run. Its purpose is a cross-grid comparison whose two sides are finally on
# the same lateral mixing, not an attribution experiment.
#
# PRE-REGISTERED READING, written before submission, against the tripole at
# matched days (5/10/15/20/25/30) and the NEMO record D/5-1:
#   CLOSER   every SST band moves toward the tripole's value relative to the
#            old MPAS run, and nino3 in particular. Then the cross-grid gap
#            was configuration, and the remaining gap is grid.
#   NEUTRAL  bands move by less than the old MPAS-vs-tripole spread. Then the
#            lateral mixing was not what separated the grids, and the four
#            knobs are not the explanation for the cross-grid disagreement.
#   WORSE    bands move AWAY from the tripole. That is a real result about the
#            MPAS lane's response to NEMO-faithful viscosity and is reported as
#            such, not quietly discarded.
#   BLOWUP   non-finite before day 30. A_h dropping 100000 -> 20000 on an
#            icosahedral mesh at dt=150 with only a 0.33 -> 3.0 Smagorinsky
#            increase to compensate is a real stability risk, and this band is
#            the reason the run is watched rather than assumed.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
echo "dirty: $(git status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/mpas_mixharm_d30
mkdir -p $OUT

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  --grid mpas \
  --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke --K-zeta-bih 0 \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --A-h 2e4 --C-smag-lap 3.0 --gm-treguier --gm-kappa-min 0 \
  --dt 150 --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --ew-cyclic-overlap --mle --mle-ce 0.06 --dm2dc --sw-rgb-chl \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --ice-lead-freeze-source nemo_qlead \
  --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --bbl-adv --years 0.0821917808219178 \
  --snapshot-every-days 5 --diag-every-days 1 \
  --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --ice-ocean-heat-coeff 70 --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
                      $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --nemo-init-tint \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --ice-init $NEMOIN/Ice_initialization.nc \
  --ice-freeze-scheme unesco --ice-exchange nemo_si3 --ice-snow bulk \
  --ice-snow-flooding on --forcing-time-interp linear --nemo-domain-mask \
  --restart-save $OUT/restart.npz \
  --output $OUT 2>&1 | tee $OUT/run.log | tail -40
rc=${PIPESTATUS[0]}
echo "=== rc=$rc ==="
exit $rc
