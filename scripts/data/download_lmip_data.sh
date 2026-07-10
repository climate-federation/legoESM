#!/usr/bin/env bash
# Stage the input data for a biophysics-only LMIP run (run_lmip_biophys.py).
#
# Idempotent: re-running skips items already staged; pass --force to redo.
# Modelled on scripts/data/download_omip_data.sh.  All paths land under
# ``data/`` which is gitignored.
#
# Two inputs, staged differently because their sources differ:
#
#   data/crujra/    — CRU-JRA v2.5 CLM forcing (Solr/Prec/TPQWL), SYMLINKED from
#                     the glade copy (large; no copy/download).  Files:
#                     <prefix>.<stream>.<year><suffix>.nc; defaults match the
#                     glade three_stream archive (year 1920 available).
#                     Source dir: --crujra-src / $LEGOESM_CRUJRA_SRC.
#
#   data/legoesm_surfdata_c250617.nc — full harmonized surfdata (soil + CLM5
#                     PFT/LAI/cover), DOWNLOADED from Zenodo (record 21087964).
#                     Regridded to the model grid at run time by the driver.
#
# Usage:
#   ./scripts/data/download_lmip_data.sh                       # both, year 1920
#   ./scripts/data/download_lmip_data.sh --year 1919 --year 1920
#   ./scripts/data/download_lmip_data.sh --crujra-only
#   ./scripts/data/download_lmip_data.sh --surfdata-only --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
CRUJRA_DIR="$DATA_DIR/crujra"

# --- defaults / config ---
PREFIX="clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5"   # glade three_stream naming
SUFFIX=""                                                        # after-year suffix (none)
SURFDATA_NAME="legoesm_surfdata_c250617.nc"                      # full harmonized surfdata
SURFDATA_URL="${LEGOESM_SURFDATA_URL:-https://zenodo.org/records/21087964/files/legoesm_surfdata_c250617.nc}"
CRUJRA_SRC="${LEGOESM_CRUJRA_SRC:-/glade/campaign/cesm/cesmdata/inputdata/atm/datm7/atm_forcing.datm7.CRUJRA.0.5d.c20260129/three_stream}"

FORCE=0
DO_CRUJRA=1
DO_SURFDATA=1
declare -a YEARS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=1 ;;
        --crujra-only)   DO_SURFDATA=0 ;;
        --surfdata-only) DO_CRUJRA=0 ;;
        --crujra-src)    CRUJRA_SRC="$2"; shift ;;
        --surfdata-url)  SURFDATA_URL="$2"; shift ;;
        --prefix)        PREFIX="$2"; shift ;;
        --suffix)        SUFFIX="$2"; shift ;;
        --year)          YEARS+=("$2"); shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ ${#YEARS[@]} -eq 0 ]]; then YEARS=(1920); fi

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
    echo "=== CRU-JRA ($PREFIX) symlink ${YEARS[*]} -> $CRUJRA_DIR ==="
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

# --- surfdata: download the full harmonized NetCDF from Zenodo ---
if [[ $DO_SURFDATA -eq 1 ]]; then
    dest="$DATA_DIR/$SURFDATA_NAME"
    if [[ -s "$dest" && $FORCE -ne 1 ]]; then
        echo "=== surfdata: [skip] $dest already present ($(du -h "$dest" | cut -f1)) ==="
    else
        echo "=== surfdata download -> $dest ==="
        echo "  [get]  $SURFDATA_URL"
        curl --fail --location --retry 3 --retry-delay 5 \
             --show-error --silent --output "$dest.partial" "$SURFDATA_URL"
        mv "$dest.partial" "$dest"
        echo "  [done] $dest ($(du -h "$dest" | cut -f1))"
        verify_netcdf "$dest" || true
    fi
fi

cat <<EOF

LMIP inputs staged under $DATA_DIR. Run (~2 deg, 10 model days):

  JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\
      --surfdata data/$SURFDATA_NAME \\
      --forcing-dir data/crujra --year ${YEARS[0]} \\
      --grid-type latlon --resolution 90 --dt 3600 --n-steps 240 \\
      --start-doy 196 --output \$SCRATCH/lmip_biophys_${YEARS[0]}
EOF
