#!/bin/bash
#SBATCH --job-name=bgc_omip_test
#SBATCH --partition=gpu
#SBATCH --account=bd1179_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoESM/logs/bgc_omip_%j.out
#SBATCH --error=/work/bd1083/b309178/diffESM/legoESM/logs/bgc_omip_%j.err

set -e

cd /work/bd1083/b309178/diffESM/legoESM
mkdir -p logs

echo "=== Step 1: Applying BGC patches ==="
python patch_state.py
python patch_ocean_model.py
python patch_run_omip.py

echo "=== Step 2: Verifying ==="
grep -n "biogeo" src/legoesm/ocean/state.py
grep -n "BGC source" src/legoesm/ocean/dynamics/ocean_model.py

echo "=== Step 3: BGC smoke test ==="
source /sw/spack-levante/mambaforge-23.11.0-0-Linux-x86_64-befbel/etc/profile.d/conda.sh
conda activate diffesm

JAX_ENABLE_X64=1 python scripts/run_omip.py \
    --quick --grid cubed_sphere --biogeo npzd \
    --pco2-atm 400.0 --output results/omip_bgc

echo "=== Results ==="
cat results/omip_bgc/cubed_sphere/C24/results.json
echo "DONE"
EOF

