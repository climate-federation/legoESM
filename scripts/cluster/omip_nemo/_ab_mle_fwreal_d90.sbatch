#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=mle90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mle90_%j.out
#
# ONE-VARIABLE A/B: Fox-Kemper mixed-layer-eddy restratification ON.
# Control = nemolev_trp_fwreal_d90 (already run, already scored).  This command
# is that run's command_line, copied from its run_manifest.json, plus --mle and
# a different --output.  Nothing else differs, and it runs from the SAME pinned
# worktree so the model code is identical too.
#
# WHY THIS RUN EXISTS
# arctic_deep_convection_columns.py localised the largest remaining Arctic
# defect on NATIVE eORCA1 cells (the meshes are identical, verified to
# max|dlat| = 1.4e-14 deg, so there is no interpolation in that measurement).
# Of 6225 Arctic columns at day 90 vs NEMO March, 291 are GENUINELY deeper than
# 500 m against NEMO's 84 -- a 3.5x excess that is the SAME in the
# virtual-salt-flux control (271), i.e. it is not caused by the freshwater
# closure.  Those columns sit in deep basins (median H 2152 m), are mostly
# ICE-FREE (0.19 ice fraction), are all statically unstable, and are NOT where
# the Arctic salinity bias lives (median dSSS +0.08 vs +0.80 elsewhere).  By
# position they are the Greenland/Norwegian Seas and the Labrador Sea, reaching
# 1800-2600 m where NEMO holds 20-160 m.
#
# NEMO ORCA1 runs `ln_mle = .true.` (EXP00/namelist_cfg, namtra_mle) and takes
# the namelist_ref defaults, which our MLEConfig already mirrors (ce=0.06,
# lat_ref 20 deg, rho_c_mle 0.01, mld_uv "min").  Every arm scored so far ran
# with MLE=off.  Restratifying deep convective mixed layers in exactly these
# basins is what the parameterisation is FOR, so its absence is the leading
# candidate -- but that is PLAUSIBLE, not established, which is what this arm
# tests.
#
# PRE-REGISTERED PREDICTION (per the compute rule; recorded BEFORE the run)
#  (a) THE NUMBER: the count of GENUINE-deep Arctic columns (>500 m, not
#      bottom-limited) at day 90, from
#      scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py, and
#      the median MLD of the Greenland-Sea and Labrador-Sea column sets.
#  (b) CONFIRMS: genuine-deep count falls from 291 toward NEMO's 84, and the
#      Greenland/Labrador medians drop from ~2000 m to O(100 m).  REFUTES: the
#      count stays within noise of 291 (say > 250), or the deep columns simply
#      move rather than shoal.  A count that FALLS BELOW ~84 would mean MLE is
#      over-restratifying and is also a failure, not a success.
#  (c) WHY NO CHEAPER TEST: MLE is a bolus overturning that changes the model
#      trajectory from the first step; there is no offline manipulation of a
#      saved checkpoint that reproduces 90 days of altered restratification.
#      The column diagnostic itself is already offline and has been run.
#
# WATCH FOR: MLE shoaling the mixed layer EVERYWHERE, not just at the runaway
# sites.  The global and Antarctic MLD bands must be re-scored too -- a fix that
# trades an Arctic error for a Southern Ocean one is not a fix.
#
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_fw_fix_wt
cd $WT || { echo "FATAL: pinned worktree $WT missing"; exit 1; }
echo "=== git SHA (pinned worktree): $(git rev-parse HEAD) ==="
git status --porcelain | head -5
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nemolev_trp_mle_d90
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository

# --- PRE-FLIGHT: stat every input FILE, not its directory ---------------------
# The first submission (job 9285423) died in seconds on
#   FileNotFoundError: data/grids/eORCA1.2_mesh_mask.nc
# because the pre-flight was `ls -d $WT/data && echo "data OK"` -- that proves a
# DIRECTORY exists, not the FILE that matters, so it passed while proving
# nothing.  A pinned worktree checks out only git-TRACKED data; grids/, woa18/,
# woce_gouretski/, clm/ and dino_paper/ are gitignored runtime data and are
# absent until symlinked.  Fail here, loudly, rather than after allocation.
_missing=0
for f in \
  "$WT/data/grids/eORCA1.2_mesh_mask.nc" \
  "/burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc" \
  "/burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc" \
  "$NEMOIN/input_fields/zdfiwm_forcing_TRA.nc" \
  "$NEMOIN/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc" \
  "$NEMOIN/input_fields/sss_climatology_for_restoring.nc" \
  "$NEMOIN/initial_conditions/woce_temp_monthly_init_4p2.nc" \
  "$NEMOIN/initial_conditions/woce_salt_monthly_init_4p2.nc" \
  "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc" \
; do
  if [ -f "$f" ]; then echo "[preflight] OK   $f"
  else echo "[preflight] MISSING $f"; _missing=$((_missing+1)); fi
done
if [ "$_missing" -ne 0 ]; then
  echo "FATAL: $_missing input file(s) missing -- refusing to consume the slot."
  exit 2
fi
echo "[preflight] all inputs present"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  --grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap \
  --nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv \
  --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo \
  --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.2466 --snapshot-every-days 30 --diag-every-days 5 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $NEMOIN/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $NEMOIN/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init $NEMOIN/initial_conditions/woce_temp_monthly_init_4p2.nc \
                      $NEMOIN/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $NEMOIN/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke \
  --ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic --sw-rgb-chl \
  --ice-ocean-heat-coeff 70 \
  --gateway-transports --freshwater-closure real_freshwater --mle --mle-ce 0.06 \
  --output $OUT
rc=$?
echo "=== run rc=$rc ==="
echo "--- transports.txt (gateway rows are the deliverable) ---"
tail -30 $OUT/transports.txt 2>/dev/null || echo "(no transports.txt)"
exit $rc
