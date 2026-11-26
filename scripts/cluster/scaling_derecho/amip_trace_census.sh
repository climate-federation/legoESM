#!/bin/bash
# Windowed rank-0 profiler trace + HLO dump of the MPAS AMIP production step,
# for scripts/bench/bytes_census.py (bytes written / MPI calls per step).
# Run inside a PBS allocation of NN nodes (16 ranks x 8 cores per node):
#   amip_trace_census.sh NN OUTDIR
# Env (required): AMIP_INPUTS (dir with the deck's forcing files), CESM_INPUTDATA,
#   ETOPO.  Optional: TRACE_FIRST (48) TRACE_COUNT (2) TRACE_WINDOWS (16),
#   MPI_SHIM_DIR (prepended to LD_LIBRARY_PATH when mpi4py needs a newer
#   cray-mpich for cross-node CXI init).  The run lasts the deck's 1 day; the
#   traces cover loop steps TRACE_FIRST .. TRACE_FIRST + COUNT*WINDOWS - 1.
# Window edges drain the model state on the traced rank only and span whole
# driver-loop iterations (land included); the 2026-10-26 numbers were taken
# with an uncommitted wrapper that drained every live array on every rank
# around the compiled step calls, so edge effects can differ slightly.
# Then: python scripts/bench/bytes_census.py OUTDIR/hlo OUTDIR/trace/rank0 $((COUNT*WINDOWS))
set -uo pipefail
NN=$1; OUT=$(realpath -m "$2"); NR=$((NN * 16))
[ -e "$OUT/trace" ] && { echo "refusing: $OUT/trace exists (old windows would mix into the census)" >&2; exit 2; }
mkdir -p "$OUT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAX_PLATFORMS=cpu
source "$HERE/_env.sh"
# _env.sh carries on after a failed activation: a path must be the active
# prefix exactly, a name must be its last component.
_p=$(realpath -m "${CONDA_PREFIX:-/}")
case "$CONDA_ENV" in
  */*) [ "$_p" = "$(realpath -m "$CONDA_ENV")" ] ;;
  *) [ "${_p##*/}" = "$CONDA_ENV" ] ;;
esac || { echo "conda env ${CONDA_ENV} not active (CONDA_PREFIX=${CONDA_PREFIX:-})" >&2; exit 2; }
[ -n "${MPI_SHIM_DIR:-}" ] && export LD_LIBRARY_PATH="$MPI_SHIM_DIR:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
AI=${AMIP_INPUTS:?}; CI=${CESM_INPUTDATA:?}
export ERA5_IC=$AI/era5_ic_1979-01-01.zarr ETOPO=${ETOPO:?}
export SST=$CI/atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_1850_2017_c180117.nc; export SIC=$SST AMIP_SST_OFFSET=273.15
export SOLAR=$AI/solar_cmip6_14band_from_cesm_c171031.nc GHG=$AI/ghg_cmip6_from_cesm_GHG_CMIP-1-2-0.nc
export OZONE=$AI/ozone_cam6_plev_1978-2014.nc VOLCANIC=$AI/volcanic_cmip6_from_cesm_1978-2014.nc
export AEROSOL=$AI/aeropt_kinne_sw_b14_fin_1979-2014_rast.nc; export AEROSOL_CCN=$AEROSOL
export CLM_SURFDATA=$AI/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000_c170706.nc AMIP_SSO=$AI/sso_stdh_2deg_from1deg.nc
export AMIP_LAND_MASK=$AI/sftlf_clm_1.9x2.5.nc AMIP_SURFDATA=$AI/legoesm_surfdata_c260716.nc
cd "$REPO"; source config/amip/amip_production.sh
AMIP_PATH_FLAGS+=( --sst-var SST_cpl --sic-var ice_cov --sic-scale 1.0 )
echo "commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) nodes=$NN ranks=$NR $(date -u)"
export LEGOESM_JIT_CACHE_DIR="" PYTHONUNBUFFERED=1
export LEGOESM_TRACE_STEPS="${TRACE_FIRST:-48}:${TRACE_COUNT:-2}:$OUT/trace"
export LEGOESM_TRACE_WINDOWS="${TRACE_WINDOWS:-16}" LEGOESM_TRACE_RANKS=0
FLAGS=(--config config/amip/amip_production.yaml "${AMIP_PATH_FLAGS[@]}" --resolution 6
       --days 1 --checkpoint-days 1 --output "$OUT" --distributed --distributed-mode mpi)
# HLO dump on the traced rank only (every rank dumping would write NR copies).
mpiexec -n "$NR" --ppn 16 --depth 8 --cpu-bind depth bash -c '
  [ "${PMI_RANK:-${PALS_RANKID:-}}" = 0 ] && export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to='"$OUT"'/hlo --xla_dump_hlo_as_text"
  exec "$0" "$@"' "$PY" -u scripts/run/run_amip.py "${FLAGS[@]}"
rc=$?; echo "exit $rc $(date -u)"; exit $rc
