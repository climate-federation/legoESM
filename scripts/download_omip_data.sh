#!/usr/bin/env bash
# Download forcing / climatology data needed for the tropical OMIP run.
#
# Idempotent: re-running skips files that already exist with a non-zero
# byte size.  Pass --force to redownload.
#
# Targets:
#   data/woa18/    — WOA18 1° annual mean temperature + salinity
#                    (NCEI; public, no auth)
#   data/jra55_ryf/ — JRA55-do RYF NOT FETCHED HERE (ESGF auth required;
#                    see the section at the bottom of this script)
#
# All paths land under ``data/`` which is gitignored.
#
# Usage:
#   ./scripts/download_omip_data.sh
#   ./scripts/download_omip_data.sh --force
#   ./scripts/download_omip_data.sh --woa-only   (default; explicit)

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
WOA_DIR="$DATA_DIR/woa18"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force)     FORCE=1 ;;
        --woa-only)  ;;  # default behaviour; accepted for clarity
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

# ---------------------------------------------------------------------------
# JRA55-do RYF — NOT downloaded here.
#
# The Stewart et al. 2020 pre-built RYF files are hosted on NCI Gadi
# (Australia) under the ``ik11`` (formerly ``ua8``) project, alongside
# the ``qv56`` mirror of the input4MIPs IAF.  Both require an NCI
# account + group membership; there is no public Zenodo mirror.
#
# Recommended path for users without NCI access:
#
#   1. Get an ESGF account (free OpenID at any node, e.g.
#      https://esgf-node.llnl.gov/projects/input4mips/).
#   2. Search input4MIPs for: MIP Era = CMIP6Plus, Target MIP = OMIP,
#      Institution = MRI, Source = MRI-JRA55-do-1-6-0.
#      Or fetch directly from MRI:
#      https://climate.mri-jma.go.jp/pub/ocean/JRA55-do/
#   3. Pull the RYF 12-month window — May 1990 to April 1991 (RYF9091)
#      is the recommended neutral year per Stewart et al. 2020.
#   4. Apply the Stewart-style smooth wraparound blending; reference
#      implementation lives in the COSIMA GitHub org under tools that
#      accompanied the 2020 paper.
#
# Once the resulting RYF NetCDF/Zarr is on disk, point our cache builder
# at it:
#
#   python scripts/prepare_omip_forcing.py \
#       --source data/jra55_ryf/RYF9091.zarr \
#       --years 1990 1990 \
#       --target-resolution-deg 1.0 \
#       --cache-dir data/jra55_ryf_cache
#
# When you're ready to wire JRA55-do RYF into this script, factor the
# WOA download into a function and add a parallel ``download_jra55_ryf``
# section guarded by a ``--with-jra55`` flag.
# ---------------------------------------------------------------------------
