#!/bin/bash
#SBATCH --job-name=eqdiff
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqdiff_%j.out
#
# NEMO's own equatorial Prandtl profile, and the MPAS twin that the arm script
# refused to score because a run ending exactly on a snapshot day writes
# snapshot_final rather than snapshot_day0030.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc

echo "############ NEMO equatorial diffusivities, cold-tongue box ############"
$PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW" --rec 5 --lon-west 200 --lon-east 260
echo "### rc=$?"

echo
echo "############ the same, WEST Pacific, as a within-oracle control ############"
echo "If the Prandtl profile is the same in the warm pool as in the cold"
echo "tongue, it is a property of the closure rather than of the regime, and a"
echo "cold-tongue-only comparison would be reading a constant."
$PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW" --rec 5 --lon-west 140 --lon-east 180
echo "### rc=$?"
echo DONE
