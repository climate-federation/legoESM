#!/bin/bash
# Task 6: 8-pixel (7 biomes + larch) no-regression + realism gate for the arctic
# carbon-productivity fix.  Runs the DifferLand equilibrium harness flags-OFF
# (baseline) then flags-ON (NSC gate + cold-deciduous dormancy) at IDENTICAL
# pixels/spin config -- the only change is the two flags (controlled comparison).
# Criteria: temperate/tropical equilibria UNCHANGED; boreal/tundra/larch reach
# NONZERO biomass with flags ON (they decline with OFF).
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=arctic_rescue_val
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/lc_rebase/scripts/tmp/%x_%j.out
set -uo pipefail
WORKDIR=/burg-archive/glab/users/pg2328/lc_rebase; cd "$WORKDIR"
export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
SPIN="--spinup-years 120 --verify-years 30 --dt 3600 --n-layers 10 --soil-depth 3.0"
echo "=== node $(hostname)  $(date)  OFF (baseline) ==="
/usr/bin/time -v $PYJ scripts/validate/land_carbon_equilibrium.py $SPIN \
  --output results/arctic_rescue_off 2>&1
echo "=== $(date)  ON (nsc + cold-deciduous) ==="
/usr/bin/time -v $PYJ scripts/validate/land_carbon_equilibrium.py $SPIN \
  --nsc-gated-respiration --cold-deciduous-dormancy \
  --output results/arctic_rescue_on 2>&1
echo "EXIT=$?  done $(date)"
