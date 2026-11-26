#!/usr/bin/env bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --job-name=mpas_seg_sched
#SBATCH --output=/scratch/b/b384341/mpas_segment_schedule/%x-%a-%A.out
# Receipt for the checkpoint-scheduled differentiable MPAS segment
# (legoesm.driver.mpas_diff_step.mpas_segment, schedule=...): the production
# deck (config/amip/amip_production.yaml) at --resolution 3, 128 steps, one
# A100 80 GB per arm (GPU over CPU: the CPU allocation is spent, user
# 2026-11-04), one process per loop form so the peak is per arm:
#   array 0: none      (Python-unrolled loop, per-step remat; the baseline)
#   array 1: uniform   (fixed-length loop, per-step remat)
#   array 2: binomial  (Treeverse, K = 12 checkpoints)
# Pre-registered: FD/AD-1 per arm (confirm |.| < 1e-4 as the 16-step probe;
# refute if an arm's AD differs from 'none' beyond roundoff), PEAK_GIB per arm
# (confirm binomial < uniform < none; refute binomial >= none), and
# SCHED_VS_UNROLLED forward max|diff| at roundoff.
set -uo pipefail
: "${REPO:=/home/b/b384341/projects/legoESM}"
# Smoke overrides (e.g. RES=2 N_STEPS=4 K=2 sbatch --array=2 ...); defaults are the receipt.
: "${RES:=3}"
: "${N_STEPS:=128}"
: "${K:=12}"
# RRTMGP in fp64 (the probe's diagnostic switch), as the 16-step receipt of
# PR #1938 ran: the FD check is the point and fp32 roundoff swamps it.
: "${RAD_FP64:=1}"
: "${PY:=${REPO}/.venv/bin/python}"
# amip_production.sh pins two paths this account cannot read; the surface data
# is the same file in a sibling checkout, the IC is the 37-level 1 deg ERA5
# 1979-01-01 store amip_production.sh names as the Levante-available product
# (the SOTA deck's IC) instead of the pinned gproxy_L21 derivative of it --
# a recorded deviation (user 2026-11-04): the receipt certifies memory and
# AD-vs-FD, not climate.
: "${AMIP_SURFDATA:=/work/bb1596/b309178/diffESM/legoesm_ap/data/legoesm_surfdata_c260716.nc}"
: "${ERA5_IC:=/work/bb1596/b309178/diffESM/legoesm_ap_main/data/era5_ic_1979-01-01.zarr}"
export AMIP_SURFDATA ERA5_IC
# The deck's land IC is the res-6 spun-up state; the strict loader refuses a
# column-count mismatch, so for RES != 6 it is regridded once with the deck's
# own tool (nearest land neighbour) into this scratch tree:
#   scripts/data/regrid_land_ic.py --source <res6 npz> --surfdata $AMIP_SURFDATA
#     --target-grid mpas --target-resolution $RES --source-soil-column 10,3.0,2.0
#     --out $LAND_IC
: "${LAND_IC:=/scratch/b/b384341/mpas_segment_schedule/data/lmip_soil_ic/soil_ic_mpas${RES}_era5T_19790101.npz}"
source "${REPO}/config/amip/amip_production.sh"
amip_require_forcing_files || exit 3

# Arms as "schedule:n_steps" words; the default is the receipt.  Bisection of
# the 128-step non-finite gradient (job 28729306, both scheduled arms):
#   ARM_LIST="uniform:32 uniform:48 uniform:64 uniform:96 none:48" sbatch --array=0-4 ...
: "${ARM_LIST:=none:${N_STEPS} uniform:${N_STEPS} binomial:${N_STEPS}}"
read -r -a ARMS <<< "${ARM_LIST}"
ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
S="${ARM%%:*}"
N_STEPS="${ARM##*:}"
EXTRA=()
[[ "${S}" == "binomial" ]] && EXTRA=(--checkpoints "${K}")
[[ "${RAD_FP64}" == "1" ]] && EXTRA+=(--rad-fp64)
# STASH_GRADS=1: per-step adjoint record (ADJOINT lines; unrolled loop only)
: "${STASH_GRADS:=0}"
[[ "${STASH_GRADS}" == "1" ]] && EXTRA+=(--stash-grads)
# STAGE: the probe stage (step = the segment receipt); PROBE_FLAGS: further
# probe-side flags, e.g. "--at-step 16 --physics-variant norad --x-tracers q_v,q_i"
: "${STAGE:=step}"
: "${PROBE_FLAGS:=}"
EXTRA+=(${PROBE_FLAGS})
# Extra run_amip flags appended after the deck's (an explicit flag beats the
# deck's YAML default), e.g. PROBE_EXTRA="--checkpoint-days 0" so a window
# that is not a multiple of the radiation cadence is accepted (the driver only
# constrains runs that write restart checkpoints).
: "${PROBE_EXTRA:=}"
# Code under test defaults to the checkout holding this script's REPO; set
# PYTHONPATH to a worktree's packages/*:src to run another commit on the same venv.
# FORCE_CPU=1: run the JAX CPU backend on whatever node this lands on (e.g. a
# GPU-partition node billed to the GPU account): hide the GPU from CUDA and
# pin JAX to cpu, so neither JAX nor legoesm.runtime.backend sees a device.
: "${FORCE_CPU:=0}"
if [[ "${FORCE_CPU}" == "1" ]]; then
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1
fi
cd "${REPO}"
echo "force_cpu=${FORCE_CPU} jax_platforms=${JAX_PLATFORMS:-} gpus=${CUDA_VISIBLE_DEVICES:-none} repo=${REPO} pythonpath=${PYTHONPATH:-} extra=${PROBE_EXTRA} stash=${STASH_GRADS} stage=${STAGE} flags=${PROBE_FLAGS} arm=${S} res=${RES} n_steps=${N_STEPS} K=${K} rad_fp64=${RAD_FP64} tridiag=${LEGOESM_TRIDIAG:-default} host=$(hostname) date=$(date -Is) ERA5_IC=${ERA5_IC}"
JAX_ENABLE_X64=1 \
  "${PY}" -u scripts/validate/production_ad_map.py --stage "${STAGE}" --resolution "${RES}" \
  --n-steps "${N_STEPS}" --schedule "${S}" "${EXTRA[@]}" -- "${AMIP_PATH_FLAGS[@]}" \
  --land-ic "${LAND_IC}" ${PROBE_EXTRA}
rc=$?
echo "exit=${rc} arm=${S}"
exit "${rc}"
