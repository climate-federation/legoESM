#!/bin/bash
#SBATCH --job-name=scoreuni
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scoreuni_%j.out
#
# The previous MPAS baseline (mpas_unified_180d) at day 30, scored with the
# SAME compare command the base2 card runs, so base2's SSS 0.517 is judged
# against a like-for-like number rather than a table entry of unknown flags.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_base2_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$ROOT/results/omip_nemo/mpas_unified_180d/snapshot_day0030.npz
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 \
  --sss-faithful --grid-label mpas_unified_d0030 \
  --output-dir $ROOT/results/omip_nemo/compare_mpasunified_d0030 2>&1 | tail -38
echo DONE
