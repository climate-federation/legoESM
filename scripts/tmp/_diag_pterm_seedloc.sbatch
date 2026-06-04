#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pterm_seed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pseed_%j.out

# Localise the SEED of the OMIP cold-start blowup: the per-term diag showed
# KE_PGF_v (equatorial meridional pressure gradient) seeds the runaway and
# vertadv amplifies it. This probe reports the (lat,lon,level) of each top
# term over the first 5 steps -> is the spurious PGF at the SURFACE (k~0, IC/
# flood-fill artifact) or DEEP (k large, partial-cell PGF over topography)?
# and at a specific lon (a localized IC artifact)?
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### tripole, NO forcing, 5 steps, seed localisation"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 5 --dt 600 --top-n 6 2>&1
echo "[rc]=$?"
echo; echo "=== DONE ==="; date
