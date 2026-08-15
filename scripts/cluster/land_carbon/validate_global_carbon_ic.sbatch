#!/bin/bash
# Post-build QC for the global land-carbon IC: re-integrate each archetype from
# the MAPPED equilibrium vs a COLD start and confirm total-C drift -> 0 (starts
# NEAR equilibrium), + per-PFT SOC / live-biomass in published biome ranges.
# INDICATIVE archetype-level QC (map_to_grid is linear but a mixed cell re-
# integrates nonlinearly -> necessary, not per-cell sufficient).  Reads
# archetypes.npz from the Stage-2 build.  Compute node (JITs the coupled step).
#
#   # chain after Stage 2 (submit from the repo root):
#   sbatch --dependency=afterok:<carbon_ic_jobid> \
#       scripts/cluster/land_carbon/validate_global_carbon_ic.sbatch
#
# WORKDIR defaults to SLURM_SUBMIT_DIR (override WORKDIR/PYJ/ARCH/OUTDIR in the
# environment).  Also the QC gate for a DOWNLOADED IC: point ARCH at the
# archetypes.npz from the `land-carbon-ic-v1` release to reproduce its drift
# numbers on your own tree.  See docs/land/global_carbon_ic_spinup.md.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_ic_val
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/tmp/%x_%j.out

set -uo pipefail
# sbatch runs the script from a SPOOL copy, so BASH_SOURCE is useless here;
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from (the repo root).
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -d "$WORKDIR/packages" || {
    echo "WORKDIR=$WORKDIR is not a legoESM checkout (no packages/); submit from the repo root or set WORKDIR"; exit 2; }
cd "$WORKDIR"

export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
# Reuse the build's coupled-land compile cache -> validator compiles in seconds.
export JAX_COMPILATION_CACHE_DIR="${CCACHE:-$WORKDIR/.xla_cache_carbon_ic}"
PYJ="${PYJ:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}"

ARCH="${ARCH:-$WORKDIR/results/global_carbon_ic/archetypes.npz}"
OUTDIR="${OUTDIR:-$WORKDIR/results/global_carbon_ic_validation}"
echo "=== node: $(hostname)  time: $(date)  validate global carbon IC ==="
test -f "$ARCH" || { echo "MISSING $ARCH (run build_global_carbon_ic.sbatch first)"; exit 2; }

/usr/bin/time -v $PYJ scripts/validate/global_carbon_ic_map.py \
  --archetypes "$ARCH" --years 3 \
  --output "$OUTDIR" 2>&1
echo "EXIT=$?  done: $(date)"
