#!/usr/bin/env bash
# Build the merged 1979-2014 transient forcing files for the full 36-year
# CMIP6 AMIP production run (see the 36-yr campaign discussion, 2026-07-03).
#
# The chain runners historically pinned CYCLIC single-year forcing for three
# channels (ozone file ending 1999, Kinne aerosol + volcanic fixed at 1979) —
# fine for short 1979 runs, wrong for a 36-year transient: El Chichon (1982)
# and Pinatubo (1991) would be absent entirely.
#
# Sources (all on the Levante pool, read-only):
#   ozone    input4MIPs vmro3 1950-1999 + 2000-2014  -> cdo mergetime+seldate
#   aerosol  Kinne aeropt_kinne_sw_b14_fin_YYYY_rast per-year files.  Each
#            carries a PLACEHOLDER "days since 2000-01-01" 12-month axis, so a
#            naive mergetime would collapse; each year is re-stamped to its
#            real year (cdo -r settaxis) before merging.  The merged file has
#            432 CF-anchored months -> legoesm.forcing.external dispatches it
#            interannually (_interp_monthly_noncyclic).
#   volcanic bc_aeropt_cmip6_volc per-year files.  These use an ABSOLUTE
#            "month starting from 1850 01" axis (1979 -> 1549 ...), so a plain
#            concat along 'month' is already monotonic; xarray concat keeps
#            the schema _load_volcanic_extinction_anchored expects.
#
# Output: ${OUT} (on /work — persistent; scratch purges mid-campaign).
# Idempotent: existing outputs are skipped unless FORCE=1.

set -euo pipefail

POOL=/pool/data/ICON/grids/public/mpim/common
OUT=${OUT:-/work/bd1083/b309178/diffESM/legoesm_ap/data/forcing/cmip6_1979-2014}
WORK=${WORK:-/scratch/b/b309178/tmp/forcing_build}
Y0=1979
Y1=2014

mkdir -p "${OUT}" "${WORK}"

echo "[1/3] ozone: merge 1950-1999 + 2000-2014, select ${Y0}-${Y1}"
OZONE_OUT="${OUT}/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_${Y0}01-${Y1}12.nc"
if [[ -s "${OZONE_OUT}" && "${FORCE:-0}" != "1" ]]; then
    echo "  exists, skip: ${OZONE_OUT}"
else
    cdo -O seldate,${Y0}-01-01,${Y1}-12-31 \
        -mergetime \
        "${POOL}/ozone_cmip6_forcing/historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_195001-199912.nc" \
        "${POOL}/ozone_cmip6_forcing/historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_200001-201412.nc" \
        "${OZONE_OUT}"
fi

echo "[2/3] Kinne aerosol: restamp per-year placeholder axes -> mergetime"
KINNE_OUT="${OUT}/aeropt_kinne_sw_b14_fin_${Y0}-${Y1}_rast.nc"
if [[ -s "${KINNE_OUT}" && "${FORCE:-0}" != "1" ]]; then
    echo "  exists, skip: ${KINNE_OUT}"
else
    rm -f "${WORK}"/kinne_*.nc
    for Y in $(seq ${Y0} ${Y1}); do
        SRC="${POOL}/aerosol_kinne/aeropt_kinne_sw_b14_fin_${Y}_rast.nc"
        [[ -e "${SRC}" ]] || { echo "ERROR: missing ${SRC}"; exit 1; }
        # -r: relative time axis ("days since Y-01-01") on mid-month stamps —
        # the CF anchor the repo reader needs for interannual dispatch.
        cdo -s -r settaxis,${Y}-01-16,12:00:00,1mon "${SRC}" "${WORK}/kinne_${Y}.nc"
        echo -n "."
    done
    echo
    cdo -O mergetime "${WORK}"/kinne_*.nc "${KINNE_OUT}"
    rm -f "${WORK}"/kinne_*.nc
fi

echo "[3/3] volcanic: concat absolute-month per-year files (xarray)"
VOLC_OUT="${OUT}/bc_aeropt_cmip6_volc_lw_b16_sw_b14_${Y0}-${Y1}.nc"
if [[ -s "${VOLC_OUT}" && "${FORCE:-0}" != "1" ]]; then
    echo "  exists, skip: ${VOLC_OUT}"
else
    POOL="${POOL}" VOLC_OUT="${VOLC_OUT}" Y0="${Y0}" Y1="${Y1}" python - <<'PYEOF'
import os
import xarray as xr

pool, out = os.environ["POOL"], os.environ["VOLC_OUT"]
y0, y1 = int(os.environ["Y0"]), int(os.environ["Y1"])
paths = [f"{pool}/aerosol_volcanic_cmip6/bc_aeropt_cmip6_volc_lw_b16_sw_b14_{y}.nc"
         for y in range(y0, y1 + 1)]
dsets = [xr.open_dataset(p, decode_times=False) for p in paths]
merged = xr.concat(dsets, dim="month", data_vars="minimal", coords="minimal",
                   compat="override")
# monotonicity guard: absolute "month starting from 1850 01" axis
m = merged["month"].values
assert (m[1:] > m[:-1]).all(), "volcanic month axis not monotonic after concat"
assert m.size == 12 * (y1 - y0 + 1), f"expected {12*(y1-y0+1)} months, got {m.size}"
merged.to_netcdf(out)
print(f"  wrote {out}: months {m[0]:.0f}..{m[-1]:.0f} ({m.size} records)")
PYEOF
fi

echo "DONE. Outputs in ${OUT}:"
ls -la "${OUT}"
