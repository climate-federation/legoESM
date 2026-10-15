#!/usr/bin/env bash
# legoESM amip_sota on Levante: the shared paths (amip_production.sh) with this
# deck's ERA5 start file -- 37 pressure levels, 1-1000 hPa, 1979-01-01 00Z, 1 deg,
# built with cdo from /pool/data/ERA5 (user decision 2026-10-15).  The shared
# default points at the 21-level Ginsburg file, which is not on Levante.
#   source config/amip/amip_sota.sh
#   "${PY}" -u scripts/run/run_amip.py --config config/amip/amip_sota.yaml \
#       "${AMIP_PATH_FLAGS[@]}" --days "${DAYS}" --output "${OUTDIR}"
: "${ERA5_IC:=/scratch/b/b309178/era5_ic_1979-01-01.zarr}"
source "$(dirname "${BASH_SOURCE[0]}")/amip_production.sh"
