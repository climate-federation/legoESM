#!/bin/bash
#SBATCH --job-name=grid3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/grid3_%j.out
# CPU-only work while every GPU node is held by the September maintenance
# reservation. The standing question is whether our three grids AGREE with each
# other as well as with the oracle, at day 30 of the same baseline.
#   pair 1: tripole vs MPAS  (the comparator takes exactly two of ours)
#   pair 2: tripole vs FESOM
# Same oracle file, same regridder, same masking on both, so the two answers are
# comparable to each other.
set -u
ROOT=/burg-archive/glab/users/pg2328/_evd_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
echo "HEAD: $(git log --oneline -1)"
for pair in "mpas MPAS" "fesom FESOM2"; do
  set -- $pair
  echo; echo "########## tripole vs $2, day 30 vs GATEWAY rec 5 ##########"
  JAX_PLATFORMS=cpu $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/trp_base2_d30/snapshot_day0030.npz \
    --mpas    $R/$1_base2_d30/snapshot_day0030.npz \
    --label-tripole tripole --label-mpas "$2" \
    --nemo-gridt "$GT" --nemo-time-idx 5 --res-deg 1.0 \
    --out-dir $R/threegrid_d30_$1 2>&1 | grep -vE "Warning|warn" | tail -60
  echo "rc=$?"
done
echo DONE
