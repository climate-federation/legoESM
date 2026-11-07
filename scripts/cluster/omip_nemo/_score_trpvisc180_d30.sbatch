#!/bin/bash
#SBATCH --job-name=score_trp180_d30
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_trp180_%j.out
# Early scoring of the running 180d MPAS arm's day-30 snapshot: replicates the
# old-SHA fullharm d30 arm (SST rmse 0.491, bias +0.005, nino3 +1.00) at the
# new SHA. Divergence beyond ~0.02 C flags the TKE-refactor merge as non-inert.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY scripts/validate/compare_omip_nemo.py \
  --legoesm-snapshot $ROOT/results/omip_nemo/trp_visc180/snapshot_day0030.npz \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 \
  --sss-faithful --grid-label trp_visc180_d30early \
  --output-dir $ROOT/results/omip_nemo/compare_trpvisc180_d0030 2>&1 | tail -38
echo DONE
