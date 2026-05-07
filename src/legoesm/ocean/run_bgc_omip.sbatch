#!/bin/bash
#SBATCH --job-name=bgc_omip_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=b384036
#SBATCH --output=/work/bd1083/b309178/diffESM/legoESM/logs/bgc_omip_%j.out
#SBATCH --error=/work/bd1083/b309178/diffESM/legoESM/logs/bgc_omip_%j.err

set -e

cd /work/bd1083/b309178/diffESM/legoESM

# Create logs directory
mkdir -p logs

echo "============================================"
echo "Step 1: Applying BGC patches"
echo "============================================"

python patch_state.py
python patch_ocean_model.py
python patch_run_omip.py

echo ""
echo "============================================"
echo "Step 2: Verifying patches"
echo "============================================"

echo "--- state.py ---"
grep -n "biogeo" src/legoesm/ocean/state.py

echo "--- ocean_model.py ---"
grep -n "BGC source" src/legoesm/ocean/dynamics/ocean_model.py

echo "--- run_omip.py ---"
grep -n "biogeo" scripts/run_omip.py | head -5

echo ""
echo "============================================"
echo "Step 3: Running BGC smoke test (30 days)"
echo "============================================"

source /sw/spack-levante/mambaforge-23.11.0-0-Linux-x86_64-befbel/etc/profile.d/conda.sh
conda activate diffesm

JAX_ENABLE_X64=1 python scripts/run_omip.py \
    --quick \
    --grid cubed_sphere \
    --biogeo npzd \
    --pco2-atm 400.0 \
    --output results/omip_bgc

echo ""
echo "============================================"
echo "Step 4: Results"
echo "============================================"
cat results/omip_bgc/cubed_sphere/C24/results.json
echo ""
echo "Timeseries (first 3 lines):"
head -3 results/omip_bgc/cubed_sphere/C24/timeseries.csv

echo "DONE"
