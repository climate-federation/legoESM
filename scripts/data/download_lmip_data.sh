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
#   data/legoesm_surfdata_c260716.nc — full harmonized surfdata (soil + CLM5
#                     PFT/LAI/cover), DOWNLOADED from Zenodo.  Regridded to the
#                     model grid at run time by the driver.  Bump SURFDATA_NAME/URL
#                     to the current dated build; override with --surfdata-url or
#                     $LEGOESM_SURFDATA_URL.
#
#   data/lmip_soil_ic/restart_1985_d000h00.npz — the spun-up 2 deg soil-state IC
#                     (10-yr 1975-1984 calibrated spin-up end state), DOWNLOADED
#                     from Zenodo.  Warm-start with run_lmip_biophys.py
#                     --restart-from; usable ONLY on the 2 deg / 10-layer grid it
#                     was built on (the loader refuses anything else).  See
#                     docs/land/lmip_biophys_soil_ic_spinup.md.  Override with
#                     --soil-ic-url or $LEGOESM_SOIL_IC_URL.
#
# Usage:
#   ./scripts/data/download_lmip_data.sh                       # all three, year 1920
#   ./scripts/data/download_lmip_data.sh --year 1919 --year 1920
#   ./scripts/data/download_lmip_data.sh --crujra-only
#   ./scripts/data/download_lmip_data.sh --surfdata-only --force
#   ./scripts/data/download_lmip_data.sh --soil-ic-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
CRUJRA_DIR="$DATA_DIR/crujra"

# --- defaults / config ---
PREFIX="clmforc.CRUJRAv2.5_0.5x0.5"                              # public CESM inputdata naming
SUFFIX=""                                                        # after-year suffix (none)
SURFDATA_NAME="legoesm_surfdata_c260716.nc"                      # current dated build
# Zenodo concept DOI 10.5281/zenodo.21087963 (latest = record 21401647); bump the
# record + name for a new dated build.  Override with --surfdata-url or $LEGOESM_SURFDATA_URL.
SURFDATA_URL="${LEGOESM_SURFDATA_URL:-https://zenodo.org/records/21401647/files/legoesm_surfdata_c260716.nc}"
SOIL_IC_NAME="restart_1985_d000h00.npz"
# Spun-up soil-state IC — Zenodo concept DOI 10.5281/zenodo.21986852 (latest =
# record 21986853, version DOI 10.5281/zenodo.21986853).  md5 in
# docs/land/lmip_biophys_soil_ic_spinup.md; bump record + md5 together.
SOIL_IC_URL="${LEGOESM_SOIL_IC_URL:-https://zenodo.org/records/21986853/files/restart_1985_d000h00.npz}"
SOIL_IC_MD5="e619b555cf6bd018a19ac8a52f77b52a"
# Env-override = different build: drop the md5 pin (magic check still applies).
if [[ -n "${LEGOESM_SOIL_IC_URL:-}" ]]; then SOIL_IC_MD5=""; fi
CRUJRA_SRC="${LEGOESM_CRUJRA_SRC:-/glade/campaign/cesm/cesmdata/inputdata/atm/datm7/atm_forcing.datm7.CRUJRA.0.5d.c20260129/three_stream}"
# Public CESM inputdata mirror of the same three CLM datm streams, 1901-2023, no
# credentials.  Used when the glade directory is not mounted -- which is every
# machine that is not NCAR's, and is why staging used to be impossible off-site.
# ~10 GB per year (Solr 1.4, Prec 1.4, TPQWL 7.2), so years are opt-in one at a
# time rather than a default sweep.
CRUJRA_URL="${LEGOESM_CRUJRA_URL:-https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/datm7/atm_forcing.datm7.CRUJRA.0.5d.c20241231/three_stream}"

FORCE=0
DO_CRUJRA=1
DO_SURFDATA=1
DO_SOIL_IC=1
declare -a YEARS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=1 ;;
        --crujra-only)   DO_SURFDATA=0; DO_SOIL_IC=0 ;;
        --surfdata-only) DO_CRUJRA=0; DO_SOIL_IC=0 ;;
        --soil-ic-only)  DO_CRUJRA=0; DO_SURFDATA=0 ;;
        --crujra-src)    CRUJRA_SRC="$2"; shift ;;
        --crujra-url)    CRUJRA_URL="$2"; shift ;;
        --surfdata-url)  SURFDATA_URL="$2"; shift ;;
        # A non-default URL is a different build: drop the md5 pin (magic
        # check still applies).
        --soil-ic-url)   SOIL_IC_URL="$2"; SOIL_IC_MD5=""; shift ;;
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
    mkdir -p "$CRUJRA_DIR"
    if [[ ! -d "$CRUJRA_SRC" ]]; then
        # Not at NCAR: fetch the same three streams over HTTP instead of
        # symlinking them.  Each file is checked for the NetCDF magic before it
        # counts as staged, so a truncated transfer or an HTML error page is
        # never left behind looking like data.
        echo "=== CRU-JRA ($PREFIX) download ${YEARS[*]} -> $CRUJRA_DIR ==="
        echo "    source: $CRUJRA_URL"
        for year in "${YEARS[@]}"; do
            for stream in Solr Prec TPQWL; do
                fname="${PREFIX}.${stream}.${year}${SUFFIX}.nc"
                dest="$CRUJRA_DIR/$fname"
                if [[ -s "$dest" && $FORCE -ne 1 ]] && verify_netcdf "$dest" 2>/dev/null; then
                    echo "  [skip] $fname ($(du -h "$dest" | cut -f1))"; continue
                fi
                echo "  [get]  $fname"
                if ! curl -fL --retry 3 --retry-delay 5 -C - -o "$dest.part" \
                        "$CRUJRA_URL/$fname"; then
                    echo "  [err]  download failed: $CRUJRA_URL/$fname" >&2
                    rm -f "$dest.part"; exit 4
                fi
                mv -f "$dest.part" "$dest"
                verify_netcdf "$dest" || { echo "  [err] not NetCDF: $dest" >&2; exit 4; }
                echo "  [ok]   $fname ($(du -h "$dest" | cut -f1))"
            done
        done
        DO_CRUJRA=0          # done via HTTP; skip the symlink branch below
    fi
fi

if [[ $DO_CRUJRA -eq 1 ]]; then
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

# --- soil-state IC: download the spun-up 2 deg land restart from Zenodo ---
verify_soil_ic() {
    # .npz is a zip: magic "PK".  Catches truncated transfers / HTML error pages.
    local path="$1"
    if [[ "$(head -c 2 "$path")" != "PK" ]]; then
        echo "  [err]  $path is not an .npz (zip magic missing)" >&2; return 1
    fi
    # md5 pinned to the published Zenodo build (md5sum on Linux, md5 on macOS);
    # empty pin (custom URL) = magic check only.
    [[ -z "$SOIL_IC_MD5" ]] && return 0
    local got=""
    if command -v md5sum >/dev/null 2>&1; then got=$(md5sum "$path" | cut -d' ' -f1)
    elif command -v md5 >/dev/null 2>&1; then got=$(md5 -q "$path")
    else echo "  [warn] no md5 tool; checksum not verified" >&2; return 0; fi
    if [[ "$got" != "$SOIL_IC_MD5" ]]; then
        echo "  [err]  md5 mismatch for $path: got $got want $SOIL_IC_MD5" >&2
        return 1
    fi
}

if [[ $DO_SOIL_IC -eq 1 ]]; then
    SOIL_IC_DIR="$DATA_DIR/lmip_soil_ic"
    mkdir -p "$SOIL_IC_DIR"
    dest="$SOIL_IC_DIR/$SOIL_IC_NAME"
    if [[ -s "$dest" && $FORCE -ne 1 ]] && verify_soil_ic "$dest"; then
        echo "=== soil IC: [skip] $dest already present ($(du -h "$dest" | cut -f1)) ==="
    else
        echo "=== soil IC download -> $dest ==="
        echo "  [get]  $SOIL_IC_URL"
        curl --fail --location --retry 3 --retry-delay 5 \
             --show-error --silent --output "$dest.partial" "$SOIL_IC_URL"
        verify_soil_ic "$dest.partial" || { rm -f "$dest.partial"; exit 4; }
        mv "$dest.partial" "$dest"
        echo "  [done] $dest ($(du -h "$dest" | cut -f1))"
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
