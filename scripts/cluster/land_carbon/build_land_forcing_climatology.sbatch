#!/bin/bash
# Stage 1 of the SCIENCE-GRADE global land-carbon IC: assemble a real ERA5
# monthly land-forcing climatology (tas/pr/rsds/netrad) from the public
# ARCO-ERA5 store, so build_global_carbon_ic.py --climatology reads an OBSERVED
# 2-D climate instead of the zonal --climate-from-latitude DEMONSTRATION
# (which collapses desert vs rainforest at one latitude).
#
#   sbatch scripts/cluster/land_carbon/build_land_forcing_climatology.sbatch
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
#SBATCH --output=/burg-archive/glab/users/pg2328/lc_rebase/scripts/tmp/%x_%j.out

set -uo pipefail
WORKDIR=/burg-archive/glab/users/pg2328/lc_rebase
cd "$WORKDIR"

# packages/* federation layout (PEP420); the builder imports legoesm.constants only.
export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

OUT="$WORKDIR/results/land_forcing/era5_land_forcing_clim_2deg.nc"

echo "=== node: $(hostname)  time: $(date)  ARCO-ERA5 land-forcing climatology ==="
/usr/bin/time -v $PYJ scripts/data/build_land_forcing_climatology.py \
  --out "$OUT" \
  --stride 8 --years 2015 2016 2017 2018 2019 2020 2>&1
echo "EXIT=$?  done: $(date)"
