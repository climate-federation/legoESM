#!/bin/bash
#SBATCH --job-name=icevol
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/icevol_%j.out
# GLM 2026-09-04 hidden-heat-sink check: d90 ice volume vs NEMO SI3 restart.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM; cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1
R=$REPO/results/omip_nemo
$PY -u scripts/validate/ocean_fidelity/ice_volume_vs_nemo_gateway.py \
  --nemo-ice-tiles "$N/EXP00/RUN_GATEWAY/ORCA1_00002160_restart_ice_00*.nc" \
  --nemo-domain-cfg $N/INPUTS/orca1_inputs/data_repository/input_fields/domain_cfg.nc \
  --mesh-mask $REPO/data/grids/eORCA1.2_mesh_mask.nc \
  --nemo-gridt $N/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc \
  --arm "unified d90=$R/trp_unified180/snapshot_day0090.npz" \
  --arm "pre-fix visc180=$R/trp_visc180/snapshot_day0090.npz" \
  --arm "realFW d30=$R/trp_fwnemo_d30/snapshot_day0030.npz"
echo "rc=$?"
