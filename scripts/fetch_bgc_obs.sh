#!/usr/bin/env bash
# Download WOA23 (nutrients, O2, T/S) and GLODAPv2 (DIC, ALK) for BGC init + calibration.
# Run on a Levante LOGIN node (compute nodes have no internet). Downloads only, no compute.
#   bash scripts/fetch_bgc_obs.sh [DEST]      default DEST=/work/bb1596/b384036/data/bgc_obs
# Re-runnable: wget -c resumes partial files and skips complete ones.
set -euo pipefail

DEST="${1:-/work/bb1596/b384036/data/bgc_obs}"
WOA_BASE="https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA"
GLODAP_MAPPED="https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0162565/mapped/GLODAPv2.2016b_MappedClimatologies.tar.gz"
GLODAP_BOTTLE="https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_Merged_Master_File.csv.zip"

fetch() {  # fetch URL OUTDIR
    mkdir -p "$2"
    wget -q -c --tries=3 --timeout=60 -P "$2" "$1" || echo "FAILED: $1" | tee -a "$DEST/failed.txt"
}

mkdir -p "$DEST"
: > "$DEST/failed.txt"

# --- WOA23, 1 degree --------------------------------------------------------
# Nutrients/O2: period "all". Index 00 = annual (full depth), 01-12 = monthly (to 800 m).
# n=nitrate, p=phosphate, i=silicate, o=dissolved oxygen
for pair in nitrate:n phosphate:p silicate:i oxygen:o; do
    var="${pair%%:*}"; code="${pair##*:}"
    for tt in 00 01 02 03 04 05 06 07 08 09 10 11 12; do
        fetch "$WOA_BASE/$var/netcdf/all/1.00/woa23_all_${code}${tt}_01.nc" "$DEST/woa23/$var"
    done
done
# T/S: decadal average 1971-2020 ("decav"), annual + monthly
for pair in temperature:t salinity:s; do
    var="${pair%%:*}"; code="${pair##*:}"
    for tt in 00 01 02 03 04 05 06 07 08 09 10 11 12; do
        fetch "$WOA_BASE/$var/netcdf/decav/1.00/woa23_decav_${code}${tt}_01.nc" "$DEST/woa23/$var"
    done
done

# --- GLODAPv2 ----------------------------------------------------------------
# Mapped 1-degree climatology (DIC, TAlk, nutrients, O2): only published for v2.2016b.
fetch "$GLODAP_MAPPED" "$DEST/glodapv2"
if [[ -f "$DEST/glodapv2/$(basename "$GLODAP_MAPPED")" ]]; then
    tar -xzf "$DEST/glodapv2/$(basename "$GLODAP_MAPPED")" -C "$DEST/glodapv2"
fi
# Latest bottle data (validation at stations / sections).
fetch "$GLODAP_BOTTLE" "$DEST/glodapv2"

echo "Done. Files in $DEST:"
find "$DEST" -type f | sort
if [[ -s "$DEST/failed.txt" ]]; then
    echo "Some downloads failed (see $DEST/failed.txt) -- URL may have moved; check the NCEI/glodap.info index pages."
    exit 1
fi
