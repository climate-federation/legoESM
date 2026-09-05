#!/usr/bin/env bash
# Cloud-cover / invisible-ice A/B: 5-day paired arms (4 + a control twin) off the res6 AMIP
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
# Absolute 80 (pinned) + 6.  The scored window is days 81-85; the run must go
# ONE day past it because the driver deletes the sidecar of the FINAL day at
# clean finalization (model_driver: cmor_accum_day_<final>.npz unlinked once
# the NetCDF is written), so a run ending at 85 would leave window_diff.py
# nothing to read (codex review).  Day 85's sidecar is then an ordinary
# per-checkpoint sidecar and survives.
TARGET_DAYS=86

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
    # Refuse a pre-existing arm directory: the chain resumes from the GREATEST
    # checkpoint it finds, so a stale day-86 file would make the arm exit
    # "already at target" without running (codex review).  Delete it by hand
    # if a relaunch is intended.
    if [[ -e "${dir}" ]]; then
        echo "arm dir exists: ${dir} -- refusing (stale checkpoints would be resumed)" >&2
        exit 3
    fi
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
# A' identical twin of the control: measures the paired 5-day noise floor and
#    restart determinism (GLM review: no threshold is sized without it; a
#    twin difference != 0 voids the pairing of every other arm)
submit_arm cld_ctl_twin   ""
# B  cover sees ice saturation (PR #1597 lever, default OFF)
submit_arm cld_mixed      "--cloud-saturation-scheme mixed_phase"
# C  condensate-aware cover, liquid saturation unchanged
submit_arm cld_xurandall  "--clouds xu_randall"
# D  pre-#1519 radiative path: ice fully visible, no inhomogeneity, no overlap.
#    TWO optics switches at once, on purpose: offline each alone leaves the
#    ice invisible (2.6 / 2.7 g/m2 vs 2.1) so only the joint arm measures the
#    radiative weight of the hidden ice.  Hypothesis is "the pre-#1519 path
#    gives the hidden ice radiative weight", not "overlap matters".
#    SPLIT_D=1 adds the two single-switch arms (predicted nulls).
submit_arm cld_pre1519    "--cloud-vertical-overlap-optics none --cloud-optics-inhomogeneity constant --cloud-inhomogeneity-factor 1.0"
if [[ "${SPLIT_D:-0}" == "1" ]]; then
    submit_arm cld_overlapnone "--cloud-vertical-overlap-optics none"
    submit_arm cld_chi1        "--cloud-optics-inhomogeneity constant --cloud-inhomogeneity-factor 1.0"
fi
