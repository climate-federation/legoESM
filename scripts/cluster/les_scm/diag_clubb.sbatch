#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbdiag
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubbdiag_%j.out
set +e
: "${C_WT:?set C_WT}"
WT=$C_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
for case in cbl gabls1; do
  echo "=================== $case / clubb (prognostic) ==================="
  $PY scripts/validate/diag_clubb_surface_flux.py --case $case --scheme clubb \
      --steps 30 --print-every 1 2>&1 | tail -45
  echo "rc=${PIPESTATUS[0]}"
  echo "=================== $case / clubb_lite (control) ==================="
  $PY scripts/validate/diag_clubb_surface_flux.py --case $case --scheme clubb_lite \
      --steps 30 --print-every 10 2>&1 | tail -12
  echo "rc=${PIPESTATUS[0]}"
done
echo "DONE $(date +%T)"
