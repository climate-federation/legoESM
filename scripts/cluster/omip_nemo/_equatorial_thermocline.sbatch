#!/bin/bash
#SBATCH --job-name=eqtherm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqtherm_%j.out
#
# DISCRIMINATOR for the +3.19 C Nino3 cold-tongue bias: thermocline/upwelling
# versus surface flux. Z20 and its east-west tilt separate them.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $ROOT/results/omip_nemo/nemolev_trp_signedn2_d30/snapshot_day0030.npz \
  --nemo-gridt $GT --nemo-month 1 --label signedn2_d30 2>&1 | tail -40
echo "=== rc=$?"
