#!/bin/bash
#SBATCH --job-name=pair_stmean5
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pair_stmean5_%j.out
# Full pair scorecard at day 5 vs NEMO record 0, instantaneous AND window
# mean, same snapshot: the integration run codex could not execute, and the
# first window-matched scorecard of every region (SST/SSS/MLD bands).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
S=$R/trp_statemean_d30/snapshot_day0005.npz
for MODE in "" "--use-mean-fields"; do
  echo "##### MODE: ${MODE:-instantaneous}"
  $PY -u scripts/validate/compare_omip_nemo.py --legoesm-snapshot $S --nemo-gridt $N --nemo-time-idx 0 \
      --output-dir $R/compare_statemean_d5${MODE:+_mean} --grid-label tripole $MODE 2>&1 | grep -vi warning
  echo "=== mode exit: ${PIPESTATUS[0]} ==="
done
echo "=== card exit: $? ==="
