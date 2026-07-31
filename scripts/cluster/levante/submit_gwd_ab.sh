#!/usr/bin/env bash
# GWD attribution A/B: does the low-level momentum sink explain the missing
# trades/westerlies (and hence the hfls deficit)?
#
# Diagnosis measured on century8's day-80 state (fixed-state, CONFIRMED):
#   Hines non-oro GWD  0.156 Pa column sink, 55% deposited below 1 km,
#                      2.4 h spin-down at the lowest level -- because the
#                      2.0 m/s launch amplitude exceeds sigma_sat = N/m_star
#                      (1.25 m/s at 140 m) over 78.5% of the area, so the
#                      wave breaks AT the launch level instead of aloft.
#   McFarlane oro GWD  0.036 Pa, 85% below 3 km, applied over OCEAN too
#                      (VoronoiMesh carries neither subgrid_topo_stddev nor
#                      land_frac, so h_topo_col is structurally None on MPAS
#                      and the scalar 500 m pseudo-mountain is planet-wide).
#   Louis turbulence   0.121 Pa (the physical sink, for scale)
# The EQUILIBRIUM response is what this A/B tests -- the fixed-state numbers
# bound the tendency, they do not prove the circulation recovers.
#
# Controlled comparison: every arm starts from the SAME pinned day-80
# checkpoint (COPIED into each arm dir, never a newest-wins pointer while
# century8 is still advancing) and runs to the same absolute TARGET_DAYS
# with byte-identical EXTRA except the single GWD variable under test.
set -euo pipefail

ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs
REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
SRC="${ROOT}/century8"
PIN="${SRC}/checkpoint_day_0080.npz"
TARGET_DAYS=110          # absolute: 80 (pinned) + 30

[[ -f "${PIN}" ]] || { echo "missing pinned checkpoint ${PIN}" >&2; exit 2; }

# Base EXTRA = century8's, verbatim, minus the --gravity-wave-drag token and
# minus --params (each arm gets its own copy so a century8 rewrite cannot
# confound the comparison).
BASE_EXTRA="--grid-type voronoi --discretization mpas --resolution 5 --nlev 30 \
--dt 75 --vertical-coord sigma --convection bechtold --rad-update-steps 48 \
--hard-saturation-adjustment --hard-sat-ice-curve --homogeneous-ice-nucleation \
--mpas-ice-skin-prognostic --mpas-conservative-tracer-clamp \
--no-convective-cloud --aerosol-ccn --sst-offset 273.15 \
--no-use-multilayer-land --no-surface-tiled --checkpoint-days 5 --diag-days 1 \
--mpas-land-lapse-k-per-km 6.5 --mpas-land-beta 0.6 \
--mpas-qv-smooth-del2-m2s 2e5 --start-year 1923"

submit_arm () {
    local name="$1" gwd_flags="$2"
    local dir="${ROOT}/${name}"
    mkdir -p "${dir}"
    cp -f "${PIN}" "${dir}/checkpoint_day_0080.npz"
    cp -f "${SRC}/params.yaml" "${dir}/params.yaml"
    # A stale manifest from a prior attempt would trip the config-hash guard.
    rm -f "${dir}/run_manifest.json"
    local extra="${BASE_EXTRA} ${gwd_flags} --params ${dir}/params.yaml"
    echo "=== ${name}: ${gwd_flags}"
    NAME="${name}" TARGET_DAYS="${TARGET_DAYS}" CENTURY_DECK=1 START_YEAR=1923 \
    EXTRA="${extra}" \
    sbatch --job-name="${name}" --time=03:30:00 \
           "${REPO}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch"
}

# Arm A is the CONTROL: identical to century8's physics, rerun in its own
# directory so the comparison is arm-vs-arm at the same protocol (never
# arm-vs-century8, whose checkpoint cadence and manifest differ).
submit_arm gwdA_base      "--gravity-wave-drag mcfarlane+hines"
submit_arm gwdB_nohines   "--gravity-wave-drag mcfarlane"
submit_arm gwdC_off       "--gravity-wave-drag none"
# Below the 1.06 m/s tropospheric saturation threshold measured on the state,
# so the wave should propagate rather than break at the launch level.
submit_arm gwdD_hinesweak "--gravity-wave-drag mcfarlane+hines --hines-total-rms-wind 0.8"

echo
echo "4 arms submitted, all from ${PIN} to absolute day ${TARGET_DAYS}."
echo "Compare: lowest-level |U|, zonal-mean u (trades/westerlies), hfls."
