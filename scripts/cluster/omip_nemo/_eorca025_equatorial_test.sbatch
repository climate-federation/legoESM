#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=e025eq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/e025eq_%j.out
#
# DOES 1/4-DEGREE RESOLUTION FIX THE EQUATORIAL COLD TONGUE?
# The closure is faithful (proven); the residual +1.42 K nino3 is hypothesised
# to be the 1-degree resolution limit -- at 1 degree the equatorial undercurrent
# is under-resolved (our EUC is 4x too weak), which starves the cold supply.
# eORCA025 refines the equatorial band to ~1/9 degree meridionally, which should
# resolve the EUC.  The TEST: does the EUC strengthen toward observed (~1 m/s)
# and the cold tongue cool, relative to our 1-degree run?
#
# NO MATCHED NEMO 1/4-DEGREE COLD START EXISTS, so this is NOT scored against a
# NEMO oracle.  The comparison is INTERNAL (our 1/4 vs our 1 degree) plus the
# observed equatorial climatology (EUC core ~1 m/s at 140W; cold tongue SST).
#
# FRAGILITY: the 1/4-degree cold start has blown up before (e025_tripole_mld,
# day 2) when --momentum-rk3 was dropped.  This keeps RK3 + the proven
# cold-start guards (--balanced-init --freeze-floor) from the runnable e025
# config, and starts at dt=75 (half the 1-degree dt) for the finer grid's CFL.
#
# OPEN DESIGN QUESTIONS FOR REVIEW (codex + GLM) BEFORE THIS IS TRUSTED:
#   1. Does the FULL 75-level faithful card (iwm/isf/bbl/mle/tke/prognostic-ice)
#      fit eORCA025 on ONE GPU, or must levels be reduced / physics dropped /
#      SPMD used?  The prior runnable e025 config used --nlev 20 (reduced) and
#      NONE of the faithful physics -- so this card is UNPROVEN at 1/4 degree.
#   2. Is dt=75 stable for this card at 1/4 degree, or does the finer grid +
#      full physics need smaller dt?
#   3. Is an INTERNAL 1/4-vs-1 comparison valid given the forcing is the same
#      corrected CORE-II but the meshes differ -- what confounds (land mask,
#      bathymetry, coastline) must be controlled?
#   4. Is 30 days long enough for the EUC to spin up at 1/4 degree?
# DO NOT LAUNCH until the reviews resolve 1-2 (feasibility) at minimum.
set -uo pipefail
PIN=${1:?usage: sbatch _eorca025_equatorial_test.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_e025_eq

# NOTE: --nemo-vertical (full 75 levels) is the faithful choice but may OOM at
# 1/4 degree on one GPU; the review must confirm before this runs.  --fp32 is
# available as the GPU-memory fallback (the driver documents "eORCA025 fits"
# under it).  Left OFF here pending the memory-fit review.
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA025_mesh_mask.nc --dt 75 --min-levels 2 \
  --nemo-vertical --balanced-init --freeze-floor \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0822 --snapshot-every-days 10 --diag-every-days 2 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -30
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
echo "=== equatorial structure: EUC + thermocline, 1/4 vs our 1 degree ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot $OUT/snapshot_day0030.npz --label e025_d30 \
  --snapshot $ROOT/results/omip_nemo/nemolev_trp_kmean_d30/snapshot_day0030.npz \
  --label deg1_d30 \
  --nemo-gridt /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc \
  --nemo-rec 5 2>&1 | tail -40
echo DONE
