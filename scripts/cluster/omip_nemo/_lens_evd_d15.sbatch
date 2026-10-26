#!/bin/bash
#SBATCH --job-name=lensevd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lensevd_%j.out
# The primary endpoint at the FIRST matched point (day 15), scored the moment
# leg 1 lands rather than waiting for the day-30 leg: EVD arm vs its control,
# same 00Z sampling, NEMO record 2 as the reference.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
B=/burg-archive/glab/users/pg2328/_base2_wt/results/omip_nemo/trp_base2_d30
E=/burg-archive/glab/users/pg2328/_evd_wt/results/omip_nemo/trp_evd_d30
for box in "220 240" "240 260"; do
  set -- $box
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
    --snapshot $B/snapshot_day0015.npz --label base2 \
    --snapshot $E/snapshot_day0015.npz --label evd \
    --nemo-gridt $GT --nemo-rec 2 --surface-profile \
    --surface-lon-lo $1 --surface-lon-hi $2 2>&1 | sed -n '/=== Near-surface/,$p'
done
echo DONE
