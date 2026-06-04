#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pcell_fix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pcell_%j.out

# DECISIVE TEST of the partial-cell-coordinate fix.
# Root cause (per-term diag 8106208): OMIP runs used the plain sigma-like z*
# coord (J=(eta+H)/H_max, all levels stretched to local depth), with the
# Adcroft/SMC03 partial-cell PGF correction GATED OFF -> spurious bottom
# meridional PGF over steep equatorial topography -> blowup by step ~10.
#
# Fix = OceanPartialCellCoordinate (z-level + partial steps, NEMO-faithful)
# with thin-cell snapping (the documented-stable run_omip_single path).
#
#   ARM 1: --partial-cell, adcroft PGF -> seed shrunk? blowup gone?
#   ARM 2: --partial-cell, smc03 PGF   -> now should DIFFER from adcroft
#                                         (correction active); compare seed.
#   ARM 3 (baseline reminder): plain z* (no --partial-cell) blew up step 10
#          (already have from job 8106208) -- not rerun here.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### ARM 1: tripole --partial-cell, adcroft PGF, 16 steps"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/tmp/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 16 --dt 600 --top-n 5 --partial-cell --pgf-scheme adcroft 2>&1
echo "[arm1_rc]=$?"

echo; echo "### ARM 2: tripole --partial-cell, smc03 PGF, 16 steps"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/tmp/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 16 --dt 600 --top-n 5 --partial-cell --pgf-scheme smc03 2>&1
echo "[arm2_rc]=$?"

echo; echo "=== DONE ==="; date
