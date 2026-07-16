#!/bin/bash
# Stage 2 of the SCIENCE-GRADE global land-carbon IC: build the finidat
# (global_carbon_ic.npz = per-cell 8-pool CarbonState + per-cell permafrost phi)
# from real CLM5 cover/soil (native 1.9x2.5 grid) + the Stage-1 ERA5 climatology.
# Samples the (PFT x climate) space, spins each archetype to a verified
# semi-analytic soil-C equilibrium (Xia 2012), and maps the cover-weighted mix
# onto the grid -- so a coupled DifferLand run starts AT equilibrium.
#
#   # chain after Stage 1:
#   J1=$(sbatch --parsable scripts/cluster/land_carbon/build_land_forcing_climatology.sbatch)
#   sbatch --dependency=afterok:$J1 scripts/cluster/land_carbon/build_global_carbon_ic.sbatch
#
# CPU backend: equilibrate_archetypes cold-compiles ONE coupled-land+carbon XLA
# graph per (is_woody, is_evergreen, soil_class) group (~19 min/group); GPU x64
# is compile-bound (>50 min, no result).  The persistent compile cache + the
# deterministic equilibrium RESULT cache make a re-run / resume cheap.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/lc_rebase/scripts/tmp/%x_%j.out

set -uo pipefail
WORKDIR=/burg-archive/glab/users/pg2328/lc_rebase
cd "$WORKDIR"

export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# Raw CLM5 surfdata is staged in the SHARED tree (read-only data, not the code);
# the code runs from this lc_rebase worktree (land/carbon-global-init branch).
SURF=/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc
CLIM="$WORKDIR/results/land_forcing/era5_land_forcing_clim_2deg.nc"
OUTDIR="$WORKDIR/results/global_carbon_ic"
CCACHE="$WORKDIR/.xla_cache_carbon_ic"          # persistent XLA compile cache
ECACHE="$WORKDIR/.carbon_equilibrium_cache"     # deterministic (eq,qc) result cache

echo "=== node: $(hostname)  time: $(date)  global carbon IC (science-grade climate) ==="
test -f "$SURF" || { echo "MISSING surfdata $SURF"; exit 2; }
test -f "$CLIM" || { echo "MISSING Stage-1 climatology $CLIM (run build_land_forcing_climatology.sbatch first)"; exit 2; }

# clm5_surfdata preset -> native 1.9x2.5 grid (no cover regrid); the ERA5
# climatology is conservatively regridded onto that grid by the driver.
# Default --clim-*-var (tas/pr/rsds/netrad) match the Stage-1 output exactly.
/usr/bin/time -v $PYJ scripts/data/build_global_carbon_ic.py \
  --surfdata-preset clm5_surfdata --surf-path "$SURF" \
  --climatology "$CLIM" \
  --output "$OUTDIR" \
  --compilation-cache-dir "$CCACHE" \
  --equilibrium-cache-dir "$ECACHE" 2>&1
echo "EXIT=$?  done: $(date)"
