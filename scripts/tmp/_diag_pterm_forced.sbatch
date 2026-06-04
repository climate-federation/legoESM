#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pterm_fc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ptermfc_%j.out

# Localise the FORCED equatorial instability that survives the partial-cell fix.
# Clean validation 8106781 (--partial-cell + CORE-II forcing) hit 227 m/s @ -1.3N
# by day 0.5. Per-term WITH forcing + partial-cell, 40 steps, to see which dv/du
# term drives the equatorial blowup (residual PGF? vortcor? surface-forcing phys?).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### per-term, --partial-cell --forcing, 40 steps"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 40 --dt 600 --top-n 6 --partial-cell --forcing 2>&1 \
  | grep -vE "Constant folding|This isn|If you|reduce-window|slow_operation|^$"
echo "[rc]=${PIPESTATUS[0]}"
echo; echo "=== DONE ==="; date
