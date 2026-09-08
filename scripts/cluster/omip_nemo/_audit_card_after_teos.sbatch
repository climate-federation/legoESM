#!/bin/bash
#SBATCH --job-name=cardaudit
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cardaudit_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/run"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  scripts/validate/ocean_fidelity/audit_orca1_tke_card.py \
  --out-json /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/card_audit_teos.json 2>&1
