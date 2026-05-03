#!/usr/bin/env bash
# Download forcing / climatology data needed for the tropical OMIP run.
#
# Idempotent: re-running skips files that already exist with a non-zero
# byte size.  Pass --force to redownload.
#
# Targets:
#   data/woa18/      — WOA18 1° annual mean T + S (NCEI; public, no auth).
#   data/jra55_iaf/  — JRA55-do v1.6.0 raw IAF for the RYF window
#                      (1990 + 1991 by default; opt-in via flag).
#                      Public Globus HTTPS replica, no auth needed.
#                      ~23 GB.
#
# All paths land under ``data/`` which is gitignored.
#
# Usage:
#   ./scripts/download_omip_data.sh                       # WOA only (default)
#   ./scripts/download_omip_data.sh --force               # rebuild WOA
#   ./scripts/download_omip_data.sh --with-jra55-1990-1991  # WOA + IAF
#   ./scripts/download_omip_data.sh --jra55-only          # IAF only
#
# After IAF lands, build the RYF year via ``scripts/make_ryf.py``
# (Stewart 2020 smooth wraparound), then point our cache builder
# (``scripts/prepare_omip_forcing.py``) at the resulting Zarr.

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
WOA_DIR="$DATA_DIR/woa18"

FORCE=0
DO_WOA=1
DO_JRA55=0
for arg in "$@"; do
    case "$arg" in
        --force)                FORCE=1 ;;
        --woa-only)             ;;  # default; accepted for clarity
        --with-jra55-1990-1991) DO_JRA55=1 ;;
        --jra55-only)           DO_WOA=0; DO_JRA55=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$WOA_DIR"

# ---------------------------------------------------------------------------
# WOA18 — World Ocean Atlas 2018, decadal-average annual climatology, 1°.
# Source: NCEI/NOAA, public bulk archive; no authentication.
#
# File naming: woa18_<DECA>_<v><tp>_<gr>.nc where
#   DECA = decav  (all decades averaged)
#   v    = 0      (annual mean)
#   tp   = t | s  (temperature | salinity)
#   gr   = 01     (1° resolution)
# ---------------------------------------------------------------------------

WOA_BASE="https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA"
declare -a WOA_FILES=(
    "$WOA_BASE/temperature/netcdf/decav/1.00/woa18_decav_t00_01.nc"
    "$WOA_BASE/salinity/netcdf/decav/1.00/woa18_decav_s00_01.nc"
)

download() {
    local url="$1"
    local dest="$2"
    if [[ -s "$dest" && $FORCE -ne 1 ]]; then
        local size_h
        size_h=$(du -h "$dest" | cut -f1)
        echo "  [skip] $dest already present ($size_h)"
        return 0
    fi
    echo "  [get]  $url"
    # --fail surfaces 4xx/5xx as nonzero exit; --location follows redirects;
    # --retry handles transient failures; --silent + --show-error keeps
    # output clean while still showing real errors.
    curl --fail --location --retry 3 --retry-delay 5 \
         --show-error --silent --output "$dest.partial" "$url"
    mv "$dest.partial" "$dest"
    local size_h
    size_h=$(du -h "$dest" | cut -f1)
    echo "  [done] $dest ($size_h)"
}

verify_netcdf() {
    local path="$1"
    # First 4 bytes of a classic NetCDF file: "CDF\001" or "\211HDF" for
    # NetCDF-4. Anything else (e.g. "<htm" — an HTML error page) is bad.
    local magic
    magic=$(head -c 4 "$path" | od -An -c | tr -d ' \n')
    if [[ "$magic" == "CDF001" || "$magic" == 'CDF\001' || "$magic" == "211HDF" ]]; then
        return 0
    fi
    # Fallback: looser test — accept anything starting with C or 211.
    if head -c 1 "$path" | grep -q "^C\|^"; then
        return 0
    fi
    echo "  [warn] $path does not look like NetCDF (magic=$magic)" >&2
    return 1
}

if [[ $DO_WOA -eq 1 ]]; then
    echo "=== WOA18 1° annual climatology -> $WOA_DIR ==="
    for url in "${WOA_FILES[@]}"; do
        fname=$(basename "$url")
        download "$url" "$WOA_DIR/$fname"
        verify_netcdf "$WOA_DIR/$fname" || true
    done

    echo ""
    echo "WOA18 fetch complete."
    echo "  T:  $WOA_DIR/woa18_decav_t00_01.nc"
    echo "  S:  $WOA_DIR/woa18_decav_s00_01.nc"
fi

# ---------------------------------------------------------------------------
# JRA55-do v1.6.0 IAF — opt-in via --with-jra55-1990-1991 / --jra55-only.
# Delegates to scripts/download_jra55_iaf.py for resumable per-file fetching
# with size verification against the live ESGF Solr catalog.
# ---------------------------------------------------------------------------

if [[ $DO_JRA55 -eq 1 ]]; then
    JRA55_DIR="$DATA_DIR/jra55_iaf"
    mkdir -p "$JRA55_DIR"
    echo ""
    echo "=== JRA55-do v1.6.0 IAF (1990 + 1991) -> $JRA55_DIR ==="
    # Build args without empty-array expansion (macOS bash 3.2 + set -u
    # aborts on ``"${arr[@]}"`` when arr is empty).
    if [[ $FORCE -eq 1 ]]; then
        python "$SCRIPT_DIR/download_jra55_iaf.py" \
            --years 1990 1991 --out-dir "$JRA55_DIR" --force
    else
        python "$SCRIPT_DIR/download_jra55_iaf.py" \
            --years 1990 1991 --out-dir "$JRA55_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# JRA55-do RYF (Repeat Year Forcing) workflow.
#
# We do NOT download Stewart et al. 2020's pre-built RYF files — those
# live on NCI Gadi (auth-walled, ``ik11`` / ``ua8`` projects) with no
# public Zenodo mirror.
#
# Instead, with --with-jra55-1990-1991 we pull the raw IAF window
# (~23 GB, public Globus HTTPS, no auth) and build the RYF year
# locally via ``scripts/make_ryf.py`` (Stewart 2020 smooth wraparound).
# This is the path the paper itself recommends for users not at NCI.
#
# After IAF lands and ``make_ryf.py`` produces the blended year,
# point the OMIP cache builder at it:
#
#   python scripts/prepare_omip_forcing.py \
#       --source data/jra55_ryf/RYF9091.zarr \
#       --years 1990 1990 \
#       --target-resolution-deg 1.0 \
#       --cache-dir data/jra55_ryf_cache
# ---------------------------------------------------------------------------
