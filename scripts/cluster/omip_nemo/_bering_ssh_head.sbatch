#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=bering
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/bering_%j.out
#
# BUDGET BEFORE HYPOTHESES. Five candidate causes for the 0.698 Sv Bering
# reversal have been eliminated by reading and none found, which means guesses
# are being ranked. The sea-level head across the strait is the leading-order
# control on its throughflow, so measuring it BISECTS the problem: a head that
# differs in sign says the forcing is wrong and the cause is upstream; a head
# that agrees says the response is wrong and the cause is local to the channel.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RES=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithtint_d30
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
# The SAME mesh the validated gateway card uses (_nemo_arctic_gateways.sbatch
# line 47). NOT an `ls` of the run directory: `ls` happily lists BROKEN
# symlinks, and every mesh-shaped name in those directories is one -- the same
# trap that failed the internal-wave card earlier today.
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc
echo "MESH=$MESH"

for f in "$RES/snapshot_day0030.npz" "$MESH" "$GW/ORCA1_5d_20000101_20000331_grid_T.nc"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

$PY -u scripts/validate/ocean_fidelity/bering_ssh_head.py \
  --snapshot "$RES/snapshot_day0030.npz" \
  --mesh "$MESH" \
  --nemo-gridt "$GW/ORCA1_5d_20000101_20000331_grid_T.nc" \
  --time-idx 5 \
  --out "$ROOT/omip_nemo/bering_ssh_head.json" 2>&1 | tail -30
echo "rc=${PIPESTATUS[0]}"
