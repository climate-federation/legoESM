#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=diag_llice
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/diag_llice_%j.out
# Localize the latlon sea-ice thickness runaway (ll7/ll9): polar-singularity
# convergence vs coastal.  CPU-only, non-blocking (does NOT touch the queued
# GPU runs mpas10/ll_ship1head).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
echo "=== HOST ==="; hostname; date
for snap in $R/nemolev_ll7_woaic/snapshot_day0090.npz $R/nemolev_ll9_kpppair/snapshot_day0030.npz $R/nemolev_ll9_kpppair/snapshot_day0020.npz; do
  if [ -f "$snap" ]; then
    echo; echo "########## $snap ##########"
    $PY scripts/tmp/_diag_latlon_ice_runaway.py "$snap" 2>&1 | tail -45
  fi
done
echo "=== DONE ==="; date
