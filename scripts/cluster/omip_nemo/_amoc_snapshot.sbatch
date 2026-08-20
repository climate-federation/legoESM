#!/bin/bash
#SBATCH --job-name=amocsnap
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/amocsnap_%j.out
#
# Discriminate: is the -0.13 Sv day-90 AMOC an instrument artifact of the
# in-run diagnostic, or a real missing overturning?  NEMO's amoc_core on OUR
# snapshots; NEMO matched-window truth is 19.7 Sv (d26-30) / 15.9 (d86-90).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
DC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/domain_cfg.nc

$PY -u scripts/validate/ocean_fidelity/amoc_from_snapshot.py \
  --snapshot $R/nemolev_trp_gwcorr_d90/snapshot_day0030.npz \
             $R/nemolev_trp_gwcorr_d90/snapshot_day0090.npz \
  --domain-cfg "$DC"
rc=$?
echo "=== rc=$rc ==="
exit $rc
