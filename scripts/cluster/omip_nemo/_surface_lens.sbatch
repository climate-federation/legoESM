#!/bin/bash
#SBATCH --job-name=lens
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lens_%j.out
#
# Warm-lens probe: box-mean T at common depths over the top 50 m of the cold
# tongue, all three base2 grids vs NEMO, day 15 (rec 2) and day 30 (rec 5).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY -m pytest -q tests/ocean/unit/test_equatorial_thermocline_structure.py 2>&1 | tail -2
ARMS=${ARMS:-"tripole=trp_base2_d30 mpas=mpas_base2_d30 fesom=fesom_base2_d30"}
ARGS=""
for a in $ARMS; do
  ARGS="$ARGS --snapshot $R/${a#*=}/snapshot_day0030.npz --label ${a%%=*} --snapshot-early $R/${a#*=}/snapshot_day0015.npz"
done
for box in "220 240" "240 260" "200 220" "150 170"; do
  set -- $box
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py $ARGS \
    --nemo-gridt $GT --nemo-rec 5 --nemo-rec-early 2 --surface-profile \
    --surface-lon-lo $1 --surface-lon-hi $2 2>&1 | sed -n '/=== Near-surface/,$p'
done
echo DONE
