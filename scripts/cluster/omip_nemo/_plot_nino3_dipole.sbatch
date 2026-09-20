#!/bin/bash
#SBATCH --job-name=n3plot
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n3plot_%j.out
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY scripts/plot/plot_layer_bias_growth.py \
    --report $MAIN/results/omip_nemo/nino3_dipole_growth.json \
    --out $MAIN/results/omip_nemo/nino3_dipole_growth.png \
    --title "nino3 (5S-5N, 150W-90W): our layer-mean T minus NEMO, days 5-30"
echo "=== rc=$? ==="
