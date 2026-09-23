#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=plotvisc
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/plotvisc_%j.out

# SST/SSS/MLD maps + zonal means for the viscosity arm, at every day that has
# a matched partner in the Smag-OFF twin (ahprof), so the pair differs in ONE
# variable (--C-smag-lap 3.0) and the maps are directly comparable.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc

for ARM in ahpsmag ahprof; do
  for DAY in 0020 0025; do
    SNAP=$ROOT/results/omip_nemo/nemolev_trp_${ARM}_d90/snapshot_day${DAY}.npz
    [ -f "$SNAP" ] || { echo "skip $ARM d$DAY (no snapshot)"; continue; }
    OUT=$ROOT/results/omip_nemo/compare_${ARM}_d${DAY}_m01
    echo "=== $ARM day $DAY -> $OUT ==="
    $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
      --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
      --grid-label ${ARM}_d${DAY} --output-dir "$OUT" 2>&1 | tail -25
    echo "=== rc=$?"
  done
done
echo "=== done ==="
