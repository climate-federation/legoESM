#!/bin/bash
#SBATCH --job-name=score_fh180_d60
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_fh180_%j.out
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
  --legoesm-snapshot $ROOT/results/omip_nemo/mpas_fullharm_180d/snapshot_day0060.npz \
  --nemo-gridt "$GW_T" --nemo-time-idx 11 --freeze-clamp-C -1.8 \
  --sss-faithful --grid-label mpas_fullharm180_d60early \
  --output-dir $ROOT/results/omip_nemo/compare_mpasfullharm180_d0060 2>&1 | tail -38
echo DONE
