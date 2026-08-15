#!/bin/bash
# Stage 2 of the SCIENCE-GRADE global land-carbon IC: build the finidat
# (global_carbon_ic.npz = per-cell 8-pool CarbonState + per-cell permafrost phi)
# from real CLM5 cover/soil (native 1.9x2.5 grid) + the Stage-1 ERA5 climatology.
# Samples the (PFT x climate) space, spins each archetype to a verified
# semi-analytic soil-C equilibrium (Xia 2012), and maps the cover-weighted mix
# onto the grid -- so a coupled DifferLand run starts AT equilibrium.
#
#   # chain after Stage 1 (submit from the repo root):
#   mkdir -p scripts/tmp   # SLURM opens --output before the job starts
#   J1=$(sbatch --parsable scripts/cluster/land_carbon/build_land_forcing_climatology.sbatch)
#   sbatch --dependency=afterok:$J1 scripts/cluster/land_carbon/build_global_carbon_ic.sbatch
#
# WORKDIR defaults to SLURM_SUBMIT_DIR, so this runs in any worktree (override
# WORKDIR/PYJ/SURF/CLIM/OUTDIR in the environment to point elsewhere).  The
# prebuilt product of this recipe is published as the `land-carbon-ic-v1` release
# -- see docs/land/global_carbon_ic_spinup.md before spending the 7 h.
#
# CPU backend: equilibrate_archetypes cold-compiles ONE coupled-land+carbon XLA
# graph per (is_woody, is_evergreen, is_cold_deciduous, soil_class) group
# (~19 min/group); GPU x64
# is compile-bound (>50 min, no result).  The persistent compile cache + the
# deterministic equilibrium RESULT cache make a re-run / resume cheap.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/tmp/%x_%j.out

set -uo pipefail
# sbatch runs the script from a SPOOL copy, so BASH_SOURCE is useless here;
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from (the repo root).
# The override is LEGOESM_WORKDIR, not WORKDIR: sbatch exports the caller's whole
# environment, and a generic inherited WORKDIR would silently outrank the
# submission directory and run a DIFFERENT checkout.
WORKDIR="${LEGOESM_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -d "$WORKDIR/packages" || {
    echo "WORKDIR=$WORKDIR is not a legoESM checkout (no packages/); submit from the repo root or set LEGOESM_WORKDIR"; exit 2; }
cd "$WORKDIR"

export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ="${LEGOESM_PYTHON:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}"

# Raw CLM5 surfdata is staged read-only in the shared tree (data, not code), so
# it is an absolute default rather than a WORKDIR-relative path.
SURF="${CIC_SURF:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
CLIM="${CIC_CLIM:-$WORKDIR/results/land_forcing/era5_land_forcing_clim_2deg.nc}"
OUTDIR="${CIC_OUTDIR:-$WORKDIR/results/global_carbon_ic}"
CCACHE="${CIC_CCACHE:-$WORKDIR/.xla_cache_carbon_ic}"          # persistent XLA compile cache
ECACHE="${CIC_ECACHE:-$WORKDIR/.carbon_equilibrium_cache}"     # deterministic (eq,qc) result cache

echo "=== node: $(hostname)  time: $(date)  global carbon IC (science-grade climate) ==="
# The guard above only proves "a legoESM-like tree" -- stamp WHICH tree, so the
# product is attributable to a commit rather than to a path.
echo "provenance: $WORKDIR @ $(git rev-parse --short HEAD 2>/dev/null || echo NO-GIT) dirty=$(git status --porcelain 2>/dev/null | wc -l) files"
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
# EXIT the build's status, not echo's: this script is the afterok gate for the
# QC stage, and a trailing echo would make every failed build "succeed".
RC=$?
echo "EXIT=$RC  done: $(date)"
exit $RC
