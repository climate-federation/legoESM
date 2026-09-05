#!/usr/bin/env bash
# Cloud-cover / invisible-ice A/B: four 5-day paired arms off the res6 AMIP
# run's pinned day-80 state (PR #1714 pre-registration, 2026-09-05).
#
# Measured on rhebc90_r6 checkpoints with scripts/validate/amip_bias/
# cloud_layers.py (offline, CONFIRMED):
#   * 90 % of the prognostic cloud-ice mass sits in layers the sundqvist cover
#     scheme calls clear, because RH is measured against LIQUID saturation; all
#     of it is ice-saturated (RH_ice >= 0.85), 95 % at RH_ice >= 1.
#   * two redundant gates then remove it from the radiation: the fsd=1
#     two_region factor on a cf-floored in-cloud tau, and the cf=0 subcolumn
#     mask.  Solver input: 1.9 of 32.6 g/m2.  The pre-#1519 optics (overlap
#     none, chi=1) saw all of it.
#   * mixed_phase saturation alone makes the ice visible but the ITCZ
#     overcast aloft (high cover 8 -> 99.5 %); xu_randall (condensate-aware
#     cover) with liquid saturation makes 68 % of the ice visible at 27 % ITCZ
#     high cover (offline snapshot numbers; the arms measure the response).
#
# Controlled comparison: every arm starts from the SAME pinned checkpoint,
# COPIED into its own directory (never a newest-wins pointer), the params
# file copied likewise, byte-identical EXTRA except the single variable
# under test, same absolute TARGET_DAYS.  Score with
#   scripts/validate/amip_bias/window_diff.py --ctl cld_ctl --d0 80 --d1 85 cld_*
# and scripts/validate/amip_bias/cloud_layers.py <arm> --days 85.
set -euo pipefail

ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}
SRC="${ROOT}/rhebc90_r6"
PIN="${SRC}/checkpoint_day_0080.npz"
PIN_ACCUM="${SRC}/cmor_accum_day_0080.npz"
PARAMS="${REPO}/config/amip/params/w7_eps_hi_tuning.yaml"   # what rhebc90_r6 ran
TARGET_DAYS=85          # absolute: 80 (pinned) + 5; window stays inside March

for f in "${PIN}" "${PIN_ACCUM}" "${PARAMS}"; do
    [[ -f "${f}" ]] || { echo "missing ${f}" >&2; exit 2; }
done

# rhebc90_r6's EXTRA verbatim (from its slurm log), minus --params (each arm
# gets its own copy) and with checkpoints every day so cloud_layers.py can
# read the whole window.
BASE_EXTRA="--resolution 6 --checkpoint-days 1 \
--subgrid-orography-file /work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc \
--land-ic ${REPO}/data/lmip_soil_ic/soil_ic_mpas6.npz"

submit_arm () {
    local name="$1" flags="$2"
    local dir="${ROOT}/${name}"
    mkdir -p "${dir}"
    cp -f "${PIN}" "${dir}/checkpoint_day_0080.npz"
    cp -f "${PIN_ACCUM}" "${dir}/cmor_accum_day_0080.npz"
    cp -f "${PARAMS}" "${dir}/params.yaml"
    rm -f "${dir}/run_manifest.json"
    local extra="${BASE_EXTRA} ${flags} --params ${dir}/params.yaml"
    echo "=== ${name}: ${flags:-<control>}"
    NAME="${name}" TARGET_DAYS="${TARGET_DAYS}" EXTRA="${extra}" \
    sbatch --job-name="${name}" --time=04:00:00 \
           "${REPO}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch"
}

# A  control: production cloud path on the current tree (catches code drift)
submit_arm cld_ctl        ""
# B  cover sees ice saturation (PR #1597 lever, default OFF)
submit_arm cld_mixed      "--cloud-saturation-scheme mixed_phase"
# C  condensate-aware cover, liquid saturation unchanged
submit_arm cld_xurandall  "--clouds xu_randall"
# D  pre-#1519 radiative path: ice fully visible, no inhomogeneity, no overlap
submit_arm cld_pre1519    "--cloud-vertical-overlap-optics none --cloud-optics-inhomogeneity constant --cloud-inhomogeneity-factor 1.0"
