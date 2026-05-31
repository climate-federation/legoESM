#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_c2trip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:55:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/c2trip_%j.out

# legoESM eORCA1 tripole, 4-yr CORE-II NYF spinup (dt=600 validated stable).
# ~2.53 h/yr -> ~10.1h (fits short 12h). Yearly snapshots -> compare to NEMO
# year-N. Self-aborts on non-finite (writes a blowup snapshot).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi --query-gpu=name --format=csv,noheader 2>&1|head -1; date
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run_omip_core2.py \
  --years 4 --dt 600 --diag-every-days 30 \
  --output results/omip_nemo/legoesm_tripole 2>&1 | tail -80
echo "c2trip_rc=${PIPESTATUS[0]}"
echo "=== snapshots ==="; ls -la results/omip_nemo/legoesm_tripole/ 2>/dev/null
echo "=== DONE ==="
