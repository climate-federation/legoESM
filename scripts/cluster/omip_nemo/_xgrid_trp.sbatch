#!/bin/bash
#SBATCH --job-name=xg_trp
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/xg_trp_%j.out
#
# CROSS-GRID MATCHED PAIR (tripole vs MPAS) -- the user's standing ask:
# "excellent agreement between our two grids".
#
# WHY THIS SHAPE.  The two grids CANNOT share NEMO's January IC: the driver
# raises SystemExit on --nemo-monthly-init for MPAS ("wired for the structured
# grids ... MPAS keeps its own IC path", run_omip_core2.py:5242).  So a
# NEMO-faithful matched pair is impossible without new code.  Instead BOTH arms
# use the SAME annual-WOA IC (--woa-init), which makes them match EACH OTHER
# exactly -- which IS the cross-grid question.  Neither arm then matches NEMO's
# IC, and the driver's own comment says the annual IC "leaves Arctic shelves
# ~1-3 PSU salty vs NEMO's January state": that penalty now applies EQUALLY to
# both arms, so it cancels in the grid-vs-grid difference.  Do NOT compare these
# numbers against the NEMO-faithful tripole runs.
#
# WHAT STAYS MATCHED: IC, dt, days, vmix closure (tke) + every TKE knob,
# tracer advection, runoff/SSS-restoring, ice, bottom drag, geothermal, iwm.
# WHAT CANNOT MATCH (grid-intrinsic, list them with every result):
#   tripole: --mesh eORCA1.2, --ew-cyclic-overlap, --C-smag-lap, --pgf-scheme
#   MPAS   : --mpas-level 7, --mpas-lloyd 20, --mpas-vmix (not --tripole-vmix)
#
# SPLIT into one job PER ARM so they run in PARALLEL: sequential was
# ~11.5h + ~11.5h = 23h against a 24h walltime, and the second arm's
# only artifact lands at the very end.  Walltime now 30h per arm.
# MXL: set to the value the 9332440/9332441 A/B endorses (default 4 = NEMO
# nn_mxl=2, ORCA1's namelist value).  BOTH arms use the SAME value.
# ★MATCHED-TKE FIX (2026-08-07). MPAS `--mpas-vmix tke` calls
# build_tripole_vmix_config("tke", iwm=None) with NO knob arguments
# (run_omip_core2.py:4924), and the four --tke-* knobs are TRIPOLE-ONLY
# by design (:1904, a codex HIGH) -- passing them on MPAS is a hard
# error. So the ONLY way both grids run the SAME TKE closure is for
# BOTH to take the ORCA1 card defaults: no --tke-* knobs, and NO --iwm
# (MPAS is given iwm=None). The first attempt had the tripole arm on
# --iwm + 3 knobs while MPAS could take none -- not a matched pair.
# MPAS-INCOMPATIBLE FLAGS STRIPPED (2026-08-07). Discovered one crash at a
# time, then enumerated from the guards in run_omip_core2.py: --dm2dc,
# --isf, --bbl-adv, --sw-rgb-chl are ALL "wired for tripole/latlon" and
# hard-error on MPAS, as are the four --tke-* knobs. MPAS also gets
# iwm=None. A matched pair must therefore run WITHOUT all of them on BOTH
# grids -- i.e. the tripole arm is deliberately DEGRADED from its
# NEMO-faithful config. That is the real cost of cross-grid matching and
# must be stated with every number from this pair.
set -uo pipefail
MXL=${MXL:-4}
WT=/burg-archive/glab/users/pg2328/_mxl_ab_wt
PIN=d0702a46c7ffbe15f48a5ead6b49daf8fcc9eca9
cd $WT || exit 1
[ "$(git -C $WT rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: pin drift"; exit 1; }
[ -r "$WT/data/grids/eORCA1.2_mesh_mask.nc" ] || { echo "FATAL: data/ not linked"; exit 1; }
echo "=== pin $PIN  MXL=$MXL ==="
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM
NIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository
# IDENTICAL on both arms -- the annual WOA IC is the ONLY IC both grids accept
IC="--woa-init --woa-t $R/data/woce_gouretski/woce_t_woa57_annual.nc \
--woa-s $R/data/woce_gouretski/woce_s_woa57_annual.nc \
--sss-restore-file $NIN/input_fields/sss_climatology_for_restoring.nc \
--ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc"
COMMON="--dt 150 --nemo-vertical --partial-cell --adaptive-implicit-vertadv \
--momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
--sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
--prognostic-ice-dynamics free_drift --years 0.0821355 --snapshot-every-days 30 \
--diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini \
--river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic\
--ice-ocean-heat-coeff 70 --gateway-transports"
TKE=""   # card defaults on BOTH grids -- see MATCHED-TKE FIX above
echo "########## ARM 1: TRIPOLE"
$PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --min-levels 2 --C-smag-lap 3.0 \
  --smag-cfl-safety 0.125 --ew-cyclic-overlap --pgf-scheme smc03 \
  $COMMON $IC $TKE --tripole-vmix tke \
  --output $R/results/omip_nemo/xgrid_trp_mxl${MXL}_d30
rc1=$?; echo "=== tripole rc=$rc1 ==="
echo "=== rc=$rc1 ==="
exit $rc1
