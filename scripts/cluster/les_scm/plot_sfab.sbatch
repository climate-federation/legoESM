#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=plotab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/plotab_%j.out
set +e
: "${P_WT:?set P_WT}"
WT=$P_WT
cd $WT || exit 1
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
for arm in fluxon fluxoff; do
  O=$R/scm_les_turbulence/sfab_$arm
  [ -d "$O" ] || continue
  for c in bomex rico gabls1 cbl ekman; do
    $PY scripts/plot/plot_scm_les_turbulence.py "$O" --case "$c" \
        --out "$O/profiles_vs_les_$c.png" && echo "  $arm/$c"
  done
done
echo "DONE $(date +%T)"
