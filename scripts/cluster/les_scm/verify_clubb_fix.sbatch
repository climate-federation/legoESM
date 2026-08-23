#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbfix
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubbfix_%j.out
set +e
: "${C_WT:?set C_WT}"
WT=$C_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== variance-floor unit tests ==============="
$PY -m pytest -q tests/atmosphere/test_clubb_variance_floor.py 2>&1 | tail -20
echo "unit rc=${PIPESTATUS[0]}"

echo "=============== the column that blew up ==============="
# BEFORE: max|dT| grew linearly at 4.49 K/step and the pack max sat at 3690.
for case in cbl gabls1; do
  echo "--------- $case ---------"
  $PY scripts/validate/diag_clubb_surface_flux.py --case $case --scheme clubb \
      --steps 30 --print-every 5 2>&1 | tail -14
  echo "rc=${PIPESTATUS[0]}"
done

echo "=============== CLUBB regression suite ==============="
# The floor and the seed are both on the shared prognostic path, so every
# CLUBB test is in the blast radius, not only the one written for the fix.
$PY -m pytest -q tests/ -k "clubb" 2>&1 | tail -20
echo "clubb suite rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
