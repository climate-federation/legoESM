#!/bin/bash
# Multi-layer land + DifferLand carbon equilibrium spin-up across climate
# pixels, with the physical-realism / carbon-allocation audit.
#
#   sbatch scripts/cluster/land_carbon/run_equilibrium.sbatch [YEARS] [DT]
#
# Outputs land under results/land_carbon_equilibrium/ (scorecard.json,
# report.md, *.png).  Tiny per-column columns -> CPU is the right backend
# (the whole nested scan compiles to a single on-device program per pixel).
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=lc_equil
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/tmp/%x_%j.out

set -uo pipefail
cd /burg-archive/glab/users/pg2328/legoESM

YEARS=${1:-80}
DT=${2:-3600}

export PYTHONPATH=$(ls -d /burg-archive/glab/users/pg2328/legoESM/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== node: $(hostname)  time: $(date)  years=$YEARS dt=$DT ==="

/usr/bin/time -v $PYJ scripts/validate/land_carbon_equilibrium.py \
  --years "$YEARS" --dt "$DT" --n-layers 10 --soil-depth 3.0 \
  --output results/land_carbon_equilibrium 2>&1
echo "EXIT=$?  done: $(date)"
