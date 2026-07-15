#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=build_sif_obs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/carbon_calibration/build_sif_obs_%j.out
# =============================================================================
# Build the gridded satellite-SIF --sif-obs NetCDF for the Stage-B SIF carbon
# calibration (scripts/data/build_sif_observations.py).
#
# Streams the Caltech TROPOMI SIF740 gridded product over anonymous FTP
# (ftp://fluo.gps.caltech.edu/data/tropomi/gridded/SIF740), reduces it to a
# growing-season climatology, regrids onto the surfdata cover grid, converts the
# 740 nm radiance to the model's emitted photon-flux units [umol/m2/s], and writes
# an nc that load_gridded_sif (train_carbon_params.py --with-sif --sif-obs) accepts.
# NETWORK job -> compute node only (login-node policy).
#
# Override via env, e.g.:
#   YEARS="2018 2019 2020 2021" RES=0_2deg \
#     sbatch scripts/cluster/carbon_calibration/build_sif_observations.sbatch
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration results/land_forcing

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CLM5="${CLM5:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
OUT="${OUT:-results/land_forcing/tropomi_sif740_growingseason_2deg.nc}"
YEARS="${YEARS:-2018 2019 2020}"
RES="${RES:-1deg}"                 # source-file resolution (1deg is a ~6 MB/year download)
SRC_VAR="${SRC_VAR:-sif_dc}"       # daily-corrected SIF (productivity-representative)
AGG="${AGG:-growing_season}"       # growing_season | annual

echo "=== node $(hostname) $(date) ==="
echo "=== build_sif_observations: years=$YEARS res=$RES var=$SRC_VAR agg=$AGG -> $OUT ==="
$PY scripts/data/build_sif_observations.py \
    --surf-path "$CLM5" \
    --out "$OUT" \
    --years $YEARS \
    --resolution "$RES" \
    --source-var "$SRC_VAR" \
    --aggregate "$AGG"
RC=$?
echo "=== build_sif_observations exit=$RC ==="
if [ "$RC" -eq 0 ]; then
    echo "SIF_OBS_OK: wrote $OUT"
    ls -la "$OUT"
else
    echo "SIF_OBS_FAIL"
fi
