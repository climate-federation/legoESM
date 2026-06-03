#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_woa
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/woa_%j.out

# OMIP-faithful step D: stage WOA18 1deg annual T/S (NCEI public) for legoESM
# initial conditions + obs reference. Independent of the NEMO build.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
echo "=== HOST ==="; hostname; date
bash scripts/download_omip_data.sh 2>&1 | tail -20
echo "woa_rc=$?"
echo "=== staged ==="; ls -la data/woa18/ 2>&1
echo "=== DONE ==="
