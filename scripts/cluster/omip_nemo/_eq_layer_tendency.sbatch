#!/bin/bash
#SBATCH --job-name=eq_layers
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eq_layers_%j.out
# Slab heat-content tendency (days 15 -> 30) in the equatorial band, all three
# unified grids vs GATEWAY recs 2 (days 11-15) -> 5 (days 26-30): does NEMO's
# cold-tongue cooling come from vertical redistribution (0-50 m cools while
# 50-150 m warms) or net loss (every slab cools)?  Same per-grid area weights.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY -m pytest -q tests/ocean/unit/test_equatorial_thermocline_structure.py 2>&1 | tail -2
# ARMS="label=arm_dir ..." (default: the rung-1 pair); every arm needs snapshot_day0015 and 0030.
ARMS=${ARMS:-"ctl=trp_unified180 rung1=trp_viscr1_d30"}
ARGS=""
for a in $ARMS; do
  ARGS="$ARGS --snapshot $R/${a#*=}/snapshot_day0030.npz --label ${a%%=*} --snapshot-early $R/${a#*=}/snapshot_day0015.npz"
done
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py $ARGS \
  --nemo-gridt $GT --nemo-rec 5 --nemo-rec-early 2 --isopycnal 2>&1 | grep -vE "Warning|warn|^\s+[a-zA-Z_]+ = np\.|^Jax|Traceback|File |RuntimeError|^\s*$"
echo "=== rc=$?"
echo DONE
