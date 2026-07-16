#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=zenodo_duogrid
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/fetch_%j.log
# Procure the AUTHORITATIVE duo-grid FV3 oracle (Mouallem 2023, JAMES):
# Zenodo 8327578 "Implementation of the novel Duo-Grid in GFDL's FV3 —
# Code and simulations files" (3.6 GB).  Used as the phase-4 cross-grid
# battery oracle (idealized 2D/3D grid-imprinting cases) + the definitive
# fv_duogrid.F90 / c_sw / d_sw to re-pin phases 3-4 against.
set -euo pipefail
DEST=/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo
cd "$DEST"
URL="https://zenodo.org/records/8327578/files/Code%20and%20new%20simulations%20files.zip?download=1"
echo "[$(date)] downloading Zenodo 8327578 (3.6 GB) ..."
wget -q --show-progress -O duogrid_fv3.zip "$URL" || { echo "WGET FAILED (compute node may lack internet)"; exit 1; }
echo "[$(date)] download complete: $(du -h duogrid_fv3.zip | cut -f1)"
echo "[$(date)] listing archive (code paths first) ..."
unzip -l duogrid_fv3.zip | grep -iE "duogrid|sw_core|dyn_core|tp_core|test_cases|\.F90|README" | head -60 > archive_code_listing.txt || true
echo "[$(date)] extracting Fortran source + READMEs (defer the heavy sim data) ..."
unzip -o -q duogrid_fv3.zip "*.F90" "*.f90" "*README*" "*.namelist" "*input.nml*" -d extracted 2>/dev/null || \
  unzip -o -q duogrid_fv3.zip -d extracted   # fallback: full extract
echo "[$(date)] fv_duogrid.F90 present:"; find extracted -iname "fv_duogrid*.F90" 2>/dev/null | head
echo "[$(date)] DONE"
touch "$DEST/FETCH_COMPLETE"
