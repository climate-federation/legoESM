#!/bin/bash
# Stage-2 spin-up as a PARALLEL ARRAY over archetype groups: each task spins one
# group into the per-group equilibrium cache (--only-groups), then a plain
# build_global_carbon_ic.sbatch run assembles the map from the cache in seconds.
# One two-leaf group takes ~4 h on 16 CPU cores; 21 groups serially would not
# fit a 72 h job and a timeout used to lose everything.
#
#   CIC_OUTDIR=... CIC_EXTRA_ARGS="--stomatal-model medlyn" \
#     sbatch --array=0-20 scripts/cluster/land_carbon/build_global_carbon_ic_groups.sbatch
#   then: sbatch --dependency=afterok:<array job id> scripts/cluster/land_carbon/build_global_carbon_ic.sbatch
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cic_group
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/tmp/%x_%A_%a.out
set -uo pipefail
WORKDIR="${LEGOESM_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -d "$WORKDIR/packages" || { echo "WORKDIR=$WORKDIR is not a legoESM checkout"; exit 2; }
cd "$WORKDIR"
export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ="${LEGOESM_PYTHON:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}"
SURF="${CIC_SURF:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
CLIM="${CIC_CLIM:-$WORKDIR/results/land_forcing/era5_land_forcing_clim_2deg.nc}"
OUTDIR="${CIC_OUTDIR:-$WORKDIR/results/global_carbon_ic}"
CCACHE="${CIC_CCACHE:-$WORKDIR/.xla_cache_carbon_ic}"
ECACHE="${CIC_ECACHE:-$WORKDIR/.carbon_equilibrium_cache}"
echo "=== node: $(hostname) time: $(date) group ${SLURM_ARRAY_TASK_ID} ==="
echo "provenance: $WORKDIR @ $(git rev-parse --short HEAD 2>/dev/null || echo NO-GIT) dirty=$(git status --porcelain 2>/dev/null | wc -l) files"
$PYJ scripts/data/build_global_carbon_ic.py \
  --surfdata-preset clm5_surfdata --surf-path "$SURF" \
  --climatology "$CLIM" \
  --output "$OUTDIR" \
  --compilation-cache-dir "$CCACHE" \
  --equilibrium-cache-dir "$ECACHE" \
  --only-groups "$SLURM_ARRAY_TASK_ID" ${CIC_EXTRA_ARGS:-} 2>&1
RC=$?
echo "EXIT=$RC  done: $(date)"
exit $RC
