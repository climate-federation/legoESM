#!/bin/bash
# Stage 1 of the SCIENCE-GRADE global land-carbon IC: assemble a real ERA5
# monthly land-forcing climatology (tas/pr/rsds/netrad) from the public
# ARCO-ERA5 store, so build_global_carbon_ic.py --climatology reads an OBSERVED
# 2-D climate instead of the zonal --climate-from-latitude DEMONSTRATION
# (which collapses desert vs rainforest at one latitude).
#
#   mkdir -p scripts/tmp   # SLURM opens --output before the job starts
#   sbatch scripts/cluster/land_carbon/build_land_forcing_climatology.sbatch
#
# Submit FROM THE REPO ROOT of the checkout you want to run: WORKDIR defaults to
# SLURM_SUBMIT_DIR, so this runs in any worktree (override WORKDIR/PYJ/OUT in the
# environment to point elsewhere).  See docs/land/global_carbon_ic_spinup.md.
#
# Streams gs://gcp-public-data-arco-era5 via gcsfs anon (Ginsburg compute nodes
# have outbound internet).  ~5184 hourly steps (6y x 12m x 3d x 24h) -> (12,lat,
# lon) monthly means; a networked, multi-GB job -> compute node only.
# Output: results/land_forcing/era5_land_forcing_clim_2deg.nc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=land_clim
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
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

# packages/* federation layout (PEP420); the builder imports legoesm.constants only.
export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
PYJ="${LEGOESM_PYTHON:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}"

OUT="${CIC_CLIM:-$WORKDIR/results/land_forcing/era5_land_forcing_clim_2deg.nc}"
mkdir -p "$(dirname "$OUT")"

echo "=== node: $(hostname)  time: $(date)  ARCO-ERA5 land-forcing climatology ==="
echo "provenance: $WORKDIR @ $(git rev-parse --short HEAD 2>/dev/null || echo NO-GIT) dirty=$(git status --porcelain 2>/dev/null | wc -l) files"
/usr/bin/time -v $PYJ scripts/data/build_land_forcing_climatology.py \
  --out "$OUT" \
  --stride 8 --years 2015 2016 2017 2018 2019 2020 2>&1
# EXIT the builder's status, not echo's: Stage 2 chains on afterok, and a
# trailing echo would make every failed build "succeed".
RC=$?
echo "EXIT=$RC  done: $(date)"
exit $RC
