#!/bin/bash
# Post-build QC for the global land-carbon IC: re-integrate each archetype from
# the MAPPED equilibrium vs a COLD start and confirm total-C drift -> 0 (starts
# NEAR equilibrium), + per-PFT SOC / live-biomass in published biome ranges.
# INDICATIVE archetype-level QC (map_to_grid is linear but a mixed cell re-
# integrates nonlinearly -> necessary, not per-cell sufficient).  Reads
# archetypes.npz from the Stage-2 build.  Compute node (JITs the coupled step).
#
#   # chain after Stage 2:
#   sbatch --dependency=afterok:<carbon_ic_jobid> \
#       scripts/cluster/land_carbon/validate_global_carbon_ic.sbatch
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_ic_val
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/lc_rebase/scripts/tmp/%x_%j.out

set -uo pipefail
WORKDIR=/burg-archive/glab/users/pg2328/lc_rebase
cd "$WORKDIR"

export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
# Reuse the build's coupled-land compile cache -> validator compiles in seconds.
export JAX_COMPILATION_CACHE_DIR="$WORKDIR/.xla_cache_carbon_ic"
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

ARCH="$WORKDIR/results/global_carbon_ic/archetypes.npz"
echo "=== node: $(hostname)  time: $(date)  validate global carbon IC ==="
test -f "$ARCH" || { echo "MISSING $ARCH (run build_global_carbon_ic.sbatch first)"; exit 2; }

/usr/bin/time -v $PYJ scripts/validate/global_carbon_ic_map.py \
  --archetypes "$ARCH" --years 3 \
  --output "$WORKDIR/results/global_carbon_ic_validation" 2>&1
echo "EXIT=$?  done: $(date)"
