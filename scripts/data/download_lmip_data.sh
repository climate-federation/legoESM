#!/usr/bin/env bash
# Stage the input data for a biophysics-only LMIP run (run_lmip_biophys.py).
#
# Idempotent: re-running skips items already staged; pass --force to redo.
# Modelled on scripts/data/download_omip_data.sh.  All paths land under
# ``data/`` which is gitignored.
#
# Two inputs, staged differently because their sources differ:
#
#   data/crujra/    — CRU-JRA v2.5 CLM forcing (Solr/Prec/TPQWL), SYMLINKED
#                     from an existing glade copy (it is large; no copy/download).
#                     Source dir via --crujra-src or $LEGOESM_CRUJRA_SRC.
#
#   data/legoesm_surfdata_c250617.nc — harmonized surfdata, DOWNLOADED from a
#                     hosted URL via --surfdata-url or $LEGOESM_SURFDATA_URL
#                     (it is a 31 MB build product, not in git).
#
# Usage:
#   # both (default), single year 2023:
#   LEGOESM_CRUJRA_SRC=/glade/.../crujra \
#   LEGOESM_SURFDATA_URL=https://.../legoesm_surfdata_c250617.nc \
#       ./scripts/data/download_lmip_data.sh
#
#   ./scripts/data/download_lmip_data.sh --crujra-src /glade/.../crujra --year 2021 --year 2022
#   ./scripts/data/download_lmip_data.sh --surfdata-only --surfdata-url https://.../sd.nc
#   ./scripts/data/download_lmip_data.sh --crujra-only --crujra-src /glade/.../crujra --force

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
CRUJRA_DIR="$DATA_DIR/crujra"

# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------
PREFIX="clmforc.CRUJRAv2.5_0.5x0.5"          # CLM datm stream filename prefix
SURFDATA_NAME="legoesm_surfdata_c250617.nc"  # current harmonized surfdata
CRUJRA_SRC="${LEGOESM_CRUJRA_SRC:-}"
SURFDATA_URL="${LEGOESM_SURFDATA_URL:-}"

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
        --year)          YEARS+=("$2"); shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done
# Default to a single year if none requested.
if [[ ${#YEARS[@]} -eq 0 ]]; then YEARS=(2023); fi

mkdir -p "$DATA_DIR"

verify_netcdf() {
    # First bytes of a NetCDF file: "CDF\001/2" (classic) or "\211HDF" (NetCDF-4).
    local path="$1" magic
    magic=$(head -c 4 "$path" | od -An -c | tr -d ' \n')
    case "$magic" in
        CDF001|CDF\\001|CDF002|211HDF) return 0 ;;
        *) echo "  [warn] $path does not look like NetCDF (magic=$magic)" >&2; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# CRU-JRA — symlink the three CLM streams for each requested year.
# ---------------------------------------------------------------------------
if [[ $DO_CRUJRA -eq 1 ]]; then
    if [[ -z "$CRUJRA_SRC" ]]; then
        echo "ERROR: CRU-JRA source dir not set. Pass --crujra-src <glade dir> or" >&2
        echo "       export LEGOESM_CRUJRA_SRC=/glade/.../crujra" >&2
        exit 3
    fi
    if [[ ! -d "$CRUJRA_SRC" ]]; then
        echo "ERROR: CRU-JRA source dir not found: $CRUJRA_SRC" >&2; exit 3
    fi
    mkdir -p "$CRUJRA_DIR"
    echo "=== CRU-JRA ($PREFIX) symlink ${YEARS[*]} -> $CRUJRA_DIR ==="
    for year in "${YEARS[@]}"; do
        for stream in Solr Prec TPQWL; do
            fname="${PREFIX}.${stream}.${year}.nc"
            src="$CRUJRA_SRC/$fname"
            dest="$CRUJRA_DIR/$fname"
            if [[ ! -e "$src" ]]; then
                echo "  [err]  source missing: $src" >&2; exit 4
            fi
            if [[ -L "$dest" || -e "$dest" ]]; then
                if [[ $FORCE -eq 1 ]]; then rm -f "$dest"; else
                    echo "  [skip] $dest"; continue; fi
            fi
            ln -s "$src" "$dest"
            echo "  [link] $fname -> $src"
        done
        verify_netcdf "$CRUJRA_DIR/${PREFIX}.TPQWL.${year}.nc" || true
    done
fi

# ---------------------------------------------------------------------------
# Surfdata — download the hosted harmonized NetCDF.
# ---------------------------------------------------------------------------
if [[ $DO_SURFDATA -eq 1 ]]; then
    dest="$DATA_DIR/$SURFDATA_NAME"
    if [[ -s "$dest" && $FORCE -ne 1 ]]; then
        echo "=== surfdata: [skip] $dest already present ($(du -h "$dest" | cut -f1)) ==="
    else
        if [[ -z "$SURFDATA_URL" ]]; then
            echo "ERROR: surfdata URL not set. Pass --surfdata-url <url> or" >&2
            echo "       export LEGOESM_SURFDATA_URL=https://.../$SURFDATA_NAME" >&2
            exit 5
        fi
        echo "=== surfdata download -> $dest ==="
        echo "  [get]  $SURFDATA_URL"
        curl --fail --location --retry 3 --retry-delay 5 \
             --show-error --silent --output "$dest.partial" "$SURFDATA_URL"
        mv "$dest.partial" "$dest"
        echo "  [done] $dest ($(du -h "$dest" | cut -f1))"
        verify_netcdf "$dest" || true
    fi
fi

# ---------------------------------------------------------------------------
# Summary + ready-to-run hint
# ---------------------------------------------------------------------------
echo ""
echo "LMIP data staged under $DATA_DIR. Run (one year, ~2 deg, 10 model days):"
echo ""
echo "  JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\"
echo "      --surfdata data/$SURFDATA_NAME \\"
echo "      --forcing-dir data/crujra --year ${YEARS[0]} \\"
echo "      --grid-type latlon --resolution 90 --dt 3600 --n-steps 240 \\"
echo "      --start-doy 196 --output \$SCRATCH/lmip_biophys_${YEARS[0]}"
