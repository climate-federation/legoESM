#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_pterm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pterm_%j.out

# Per-term momentum-tendency localisation of the OMIP WOA cold-start blowup.
# ARM A (primary): NO surface forcing -> isolate the pure cold-start
#                  geostrophic adjustment from the CORE-II applicator.
# ARM B (control): WITH CORE-II forcing -> confirm the forcing is not the
#                  trigger (prior work: forcing ramp did not change day-1 blowup).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### ARM A: tripole, NO forcing (pure cold-start dynamics)"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 16 --dt 600 --top-n 5 2>&1
echo "[armA_rc]=$?"

echo; echo "### ARM B: tripole, WITH CORE-II forcing"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 16 --dt 600 --top-n 5 --forcing 2>&1
echo "[armB_rc]=$?"

echo; echo "=== DONE ==="; date
