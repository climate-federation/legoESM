#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=n2ts
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:35:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n2ts_%j.out
#
# IS THE EQUATORIAL SURFACE OVER-STRATIFICATION STANDING OR A DAY-30 ARTIFACT?
# The day-30 snapshot showed our N2 2.5x NEMO's in 5-65 m, which starves the
# surface mixing and would warm the cold tongue.  But a prior session found a
# day-30-ONLY N2 spike that recovered by day 40 (cause unknown).  The kmean arm
# dumped a snapshot EVERY day, so days 26-30 settle standing-vs-transient
# WITHOUT a new run.  A ratio steady across 26-30 is a real bias; a ratio that
# spikes only at 30 is the known artifact and must not drive a fix.
# Reads existing snapshots, runs no model.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW_W=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
D=$ROOT/results/omip_nemo/nemolev_trp_kmean_d30
# NEMO rec 5 = days 26-30 (a 5-day mean); compare each of our days 26-30
# against it, so the trend is ours, not a moving reference.
for day in 0026 0027 0028 0029 0030; do
  SNAP=$D/snapshot_day${day}.npz
  [ -f "$SNAP" ] || { echo "day $day: MISSING"; continue; }
  echo "########## our day $day vs NEMO rec 5 ##########"
  $PY -u scripts/validate/ocean_fidelity/equatorial_n2_vs_nemo.py \
    --snapshot "$SNAP" --nemo-gridw "$GW_W" --rec 5 --depth-max 100 2>&1 \
    | grep -E "closure-active layer|below 100 m" | head -3
done
echo DONE
