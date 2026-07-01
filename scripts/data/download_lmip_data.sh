#!/usr/bin/env bash
# Stage the input data for a biophysics-only LMIP run (run_lmip_biophys.py).
#
# Idempotent: re-running skips items already staged; pass --force to redo.
# Modelled on scripts/data/download_omip_data.sh.  All paths land under
# ``data/`` which is gitignored.
#
# Two inputs, staged differently because their sources differ:
#
#   data/crujra/    — CRU-JRA (TRENDY c2023) CLM forcing (Solr/Prec/TPQWL),
#                     SYMLINKED from the glade copy (large; no copy/download).
#                     Files: <prefix>.<stream>.<year><suffix>.nc, defaults
#                     prefix=clmforc.TRENDY.c2023_0.5x0.5, suffix=_cdf5.
#                     Source dir: --crujra-src / $LEGOESM_CRUJRA_SRC.
#
#   data/legoesm_surfdata_soil_0p25.nc — 0.25deg HWSD soil INTERMEDIATE,
#                     DOWNLOADED from Zenodo (record 21087689).  This is soil
#                     ONLY; it must be combined with a CLM5 surfdata (PFT/LAI/
#                     cover) via build_legoesm_surfdata.py --skip-hwsd to produce
#                     the full surfdata the driver reads (the "regrid" step,
#                     printed at the end).
#
# Usage:
#   ./scripts/data/download_lmip_data.sh                       # both, year 2022
#   ./scripts/data/download_lmip_data.sh --year 2021 --year 2022
#   ./scripts/data/download_lmip_data.sh --crujra-only
#   ./scripts/data/download_lmip_data.sh --soil-only --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
CRUJRA_DIR="$DATA_DIR/crujra"

# --- defaults / config ---
PREFIX="clmforc.TRENDY.c2023_0.5x0.5"        # CLM datm stream filename prefix
SUFFIX="_cdf5"                                # after-year suffix on the glade files
SOIL_NAME="legoesm_surfdata_soil_0p25.nc"    # 0.25deg soil intermediate (Zenodo)
SOIL_URL="${LEGOESM_SOIL_URL:-https://zenodo.org/records/21087689/files/legoesm_surfdata_soil_0p25.nc}"
CRUJRA_SRC="${LEGOESM_CRUJRA_SRC:-/glade/campaign/cesm/cesmdata/inputdata/atm/datm7/atm_forcing.datm7.CRUJRA.0.5d.c2023/TRENDY_cdf5}"

FORCE=0
DO_CRUJRA=1
DO_SOIL=1
declare -a YEARS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)        FORCE=1 ;;
        --crujra-only)  DO_SOIL=0 ;;
        --soil-only)    DO_CRUJRA=0 ;;
        --crujra-src)   CRUJRA_SRC="$2"; shift ;;
        --soil-url)     SOIL_URL="$2"; shift ;;
        --prefix)       PREFIX="$2"; shift ;;
        --suffix)       SUFFIX="$2"; shift ;;
        --year)         YEARS+=("$2"); shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ ${#YEARS[@]} -eq 0 ]]; then YEARS=(2022); fi

mkdir -p "$DATA_DIR"

verify_netcdf() {
    local path="$1" magic
    magic=$(head -c 4 "$path" | od -An -c | tr -d ' \n')
    case "$magic" in
        CDF001|CDF\\001|CDF002|211HDF) return 0 ;;
        *) echo "  [warn] $path does not look like NetCDF (magic=$magic)" >&2; return 1 ;;
    esac
}

# --- CRU-JRA: symlink the three CLM streams per requested year ---
if [[ $DO_CRUJRA -eq 1 ]]; then
    if [[ ! -d "$CRUJRA_SRC" ]]; then
        echo "ERROR: CRU-JRA source dir not found: $CRUJRA_SRC" >&2
        echo "       set --crujra-src <glade dir> or \$LEGOESM_CRUJRA_SRC" >&2
        exit 3
    fi
    mkdir -p "$CRUJRA_DIR"
    echo "=== CRU-JRA ($PREFIX ... $SUFFIX) symlink ${YEARS[*]} -> $CRUJRA_DIR ==="
    for year in "${YEARS[@]}"; do
        for stream in Solr Prec TPQWL; do
            fname="${PREFIX}.${stream}.${year}${SUFFIX}.nc"
            src="$CRUJRA_SRC/$fname"
            dest="$CRUJRA_DIR/$fname"
            if [[ ! -e "$src" ]]; then
                echo "  [err]  source missing: $src" >&2; exit 4
            fi
            if [[ -L "$dest" || -e "$dest" ]]; then
                if [[ $FORCE -eq 1 ]]; then rm -f "$dest"; else
                    echo "  [skip] $fname"; continue; fi
            fi
            ln -s "$src" "$dest"
            echo "  [link] $fname"
        done
        verify_netcdf "$CRUJRA_DIR/${PREFIX}.TPQWL.${year}${SUFFIX}.nc" || true
    done
fi

# --- soil intermediate: download from Zenodo ---
if [[ $DO_SOIL -eq 1 ]]; then
    dest="$DATA_DIR/$SOIL_NAME"
    if [[ -s "$dest" && $FORCE -ne 1 ]]; then
        echo "=== soil: [skip] $dest already present ($(du -h "$dest" | cut -f1)) ==="
    else
        echo "=== soil intermediate download -> $dest ==="
        echo "  [get]  $SOIL_URL"
        curl --fail --location --retry 3 --retry-delay 5 \
             --show-error --silent --output "$dest.partial" "$SOIL_URL"
        mv "$dest.partial" "$dest"
        echo "  [done] $dest ($(du -h "$dest" | cut -f1))"
        verify_netcdf "$dest" || true
    fi
fi

# --- next-step hint: build the full surfdata, then run ---
cat <<EOF

LMIP inputs staged under $DATA_DIR.

NEXT (the "regrid" step): combine the 0.25deg soil with a CLM5 surfdata
(PFT/LAI/cover) to build the full surfdata the driver reads. Point --clm-surfdata
at a CLM5 surfdata NetCDF on glade (e.g. under
/glade/campaign/cesm/cesmdata/inputdata/lnd/clm2/surfdata_esmf/...):

  python scripts/data/build_legoesm_surfdata.py --skip-hwsd \\
      --intermediate data/$SOIL_NAME \\
      --clm-surfdata /glade/.../surfdata_*.nc \\
      --out data/legoesm_surfdata.nc

THEN run (~2 deg, 10 model days):

  JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\
      --surfdata data/legoesm_surfdata.nc \\
      --forcing-dir data/crujra --year ${YEARS[0]} \\
      --grid-type latlon --resolution 90 --dt 3600 --n-steps 240 \\
      --start-doy 196 --output \$SCRATCH/lmip_biophys_${YEARS[0]}
EOF
