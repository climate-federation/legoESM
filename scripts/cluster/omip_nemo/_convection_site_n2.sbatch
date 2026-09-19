#!/bin/bash
#SBATCH --job-name=convn2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/convn2_%j.out
#
# ARE THE DEEP-CONVECTION SITES PRECONDITIONED THE SAME WAY?
#
# WHAT IS AND IS NOT TESTABLE AT DAY 30. Deep-convection ONSET is a seasonal
# event; a 30-day run cannot show it and nothing here claims to. What a
# 30-day run CAN show is whether the water column at the three classical
# sites is stratified the way NEMO's is -- the PRECONDITIONING. If our
# stratification is already too weak or too strong at day 30, every later
# convection difference is downstream of that, and if it matches, convection
# differences would have to come from the forcing or the closure instead.
#
# REUSES the existing Arctic N2 probe rather than adding a third stratification
# instrument (there are already MLD instruments in mld_basin_vs_nemo.py and
# the scorecard). That probe's load-bearing property is that BOTH SIDES go
# through the model's OWN compute_buoyancy_frequency_nemo_bn2, NEMO's
# eosbn2.F90 bn2_t -- so the instrument is identical on both sides and the
# ours-vs-NEMO contrast is robust even though the absolute N2 depends on the
# S-EOS coefficient set. It gained named site boxes for this; with no --site
# it still runs the Arctic band exactly as before.
#
# SITES, CHOSEN AND STATED rather than defaulted, following the 26N/30S/140W
# precedent. Two of the three WRAP the prime meridian, which is the case that
# returns an empty box instead of an error if handled carelessly, so the
# probe prints a column count for every site and refuses an empty one:
#   weddell    72S-62S, 300E-30E   the Weddell gyre and polynya sector
#   labrador   55N-65N, 300E-315E  the Labrador Sea (60W-45W)
#   greenland  70N-78N, 350E-15E   the Greenland Sea (10W-15E)
# Depth cut moved to 500 m: the probe's 120 m default suits the Arctic
# halocline it was built for, and convective preconditioning is deeper.
#
# TWO SAMPLING CAVEATS, stated here rather than discovered in the numbers.
# Ours is an INSTANTANEOUS day-30 state; NEMO's is a JANUARY MONTHLY MEAN, so
# this is the snapshot-versus-mean mismatch that has burned this campaign
# before, and it flatters whichever side has less sub-monthly variance. And
# this probe is NATIVE-GRID TRIPOLE ONLY -- MPAS is an unstructured node
# cloud with a different state shape and is NOT covered. Do not report this
# as three-grid agreement.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
R=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1
echo "=== HEAD $(git -C $ROOT log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

SNAP=$R/results/omip_nemo/trp_orca1faithtint_d30/snapshot_final.npz
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
MESH=$R/data/grids/eORCA1.2_mesh_mask.nc
for f in "$SNAP" "$GT" "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

$PY -u scripts/validate/ocean_fidelity/arctic_n2_compare.py \
  --snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 --mesh-mask "$MESH" \
  --max-depth-m 500 \
  --site="weddell:-72,-62:300,30" \
  --site="labrador:55,65:300,315" \
  --site="greenland:70,78:350,15"
echo "=== rc=$? ==="
