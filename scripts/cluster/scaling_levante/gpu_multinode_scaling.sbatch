#!/bin/bash -l
#SBATCH --job-name=legoesm_gpu_multinode
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=legoesm_gpu_multinode.%j.log
# ===========================================================================
# MULTI-NODE GPU scaling on Levante (2x GPU nodes, 4x A100-80 each, IB HDR).
#
# Three jax.distributed + NCCL lanes (1 process per GPU; SLURM auto-detected
# by jax.distributed.initialize(); NCCL native IB-verbs — env defaults in
# _env.sh: NCCL_IB_HCA=mlx5, NCCL_SOCKET_IFNAME=ib0, NCCL_NET_GDR_LEVEL=PHB):
#
#   Lane A (RUN_NCCL=1)   cube cs-spmd, 6 procs (3 GPUs/node, one face/GPU)
#   Lane C (RUN_LATLON=1) atm lat-lon lat-band multihost, 8 procs — the
#                         gap-#0 lane (native ppermute halos replace the
#                         route-A mpi4jax >=2-node ceiling)
#   Lane D (RUN_OCEAN=1)  ocean lat-band multihost, 8 procs
#   Lane E (RUN_MPAS=1)   icosahedral MPAS multicontroller, 6 procs
#   Lane T (RUN_TUNE=1, default 0) comm-tuning A/B ladder (fused-halo /
#                         XLA CP-combining + pipelined p2p / PGLE); outputs
#                         under _ab_tuning/, excluded from the curves
#                         (cell-partition reorder + ppermute halo)
#
# First run: NCCL_DEBUG=INFO must show `NET/IB` (not `NET/Socket`) — a wrong
# bootstrap interface (NCCL_SOCKET_IFNAME) is the #1 IB-cluster timeout cause.
#
# Plain `legoesm-gpu` env — jax.distributed ONLY, mpi4jax never armed
# (mixed stacks deadlock; the route-A cross-node lane lives in
# gpu_moist_scaling.sbatch with NODES=2).
#
# Account: export SBATCH_ACCOUNT=<project> (or sbatch -A <project> ...).
# SUBMIT:
#   sbatch scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if   [ -f "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
elif [ -f "${SUBMIT_DIR}/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
else
    echo "ERROR: cannot find _env.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 1
fi

export JAX_PLATFORMS=cuda
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

RUN_NCCL="${RUN_NCCL:-1}"
RUN_LATLON="${RUN_LATLON:-1}"
RUN_OCEAN="${RUN_OCEAN:-1}"
RUN_MPAS="${RUN_MPAS:-1}"
# Lat-band lanes C/D task count (8 = 2 nodes; 16 = sbatch --nodes=4).
NP_LL="${NP_LL:-8}"
# Lane E device count; >6 pads the mesh via --reorder-for (even split).
NP_MPAS="${NP_MPAS:-6}"
# Slurm job STEPS do not inherit the job's GPU allocation on all Slurm
# versions/configs — without an explicit step gres some tasks see zero
# devices (jax: "no supported devices found for platform CUDA"). Make every
# GPU visible to every task; the PIN block below does the per-task pinning.
STEP_GPU_OPTS="${STEP_GPU_OPTS:---gpus-per-node=4 --gpu-bind=none}"

# The NP knobs must fit the allocation (codex: NP_LL=16 on the default
# 2-node header would just fail in srun) — validate up front, loudly.
NODES_HAVE="${SLURM_JOB_NUM_NODES:-1}"
need=0
[ "$RUN_NCCL" = "1" ] && [ 2 -gt "$need" ] && need=2
if [ "$RUN_LATLON" = "1" ] || [ "$RUN_OCEAN" = "1" ]; then
    n=$(( (NP_LL + 3) / 4 )); [ "$n" -gt "$need" ] && need=$n
fi
# Lane T is hard-coded to np=8 (atm/ocean) + np=6 cube — a fixed 2-node
# requirement independent of NP_LL (codex round-2 finding 3).
if [ "${RUN_TUNE:-0}" = "1" ] && [ 2 -gt "$need" ]; then need=2; fi
if [ "$RUN_MPAS" = "1" ]; then
    tpn=3; [ $((NP_MPAS % 4)) -eq 0 ] && tpn=4
    n=$(( (NP_MPAS + tpn - 1) / tpn )); [ "$n" -gt "$need" ] && need=$n
fi
if [ "$NODES_HAVE" -lt "$need" ]; then
    echo "ERROR: enabled lanes need $need nodes (NP_LL=$NP_LL NP_MPAS=$NP_MPAS" \
         "cube=$RUN_NCCL) but the job has $NODES_HAVE — resubmit with" \
         "sbatch --nodes=$need." >&2
    exit 2
fi
MODE="${MODE:-strong}"
PRECISION="${PRECISION:-float32}"
CS_RES="${CS_RES:-96}"
LL_RES="${LL_RES:-512}"
OC_RES="${OC_RES:-288}"
ICO_LEVEL="${ICO_LEVEL:-7}"   # lane E: L7 = 163842 cells -> ~27k cells/GPU at np=6
PHYSICS="${PHYSICS:-moist}"
NLEV="${NLEV:-26}"
OC_NLEV="${OC_NLEV:-20}"
N_WARMUP="${N_WARMUP:-3}"
N_TIMING="${N_TIMING:-30}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"   # first run: check for NET/IB

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/gpu_multinode_${STAMP}_j${SLURM_JOB_ID:-local}}"
mkdir -p "$OUTDIR"

echo "=== HOSTS ==="; scontrol show hostnames "$SLURM_JOB_NODELIST"; date
echo "=== CONFIG ==="
echo "lanes: cube=$RUN_NCCL latlon=$RUN_LATLON ocean=$RUN_OCEAN mpas=$RUN_MPAS | mode=$MODE prec=$PRECISION C$CS_RES LL$LL_RES OC$OC_RES ICO-L$ICO_LEVEL"
echo "outdir=$OUTDIR"

# One GPU per task — DO NOT pre-pin CUDA_VISIBLE_DEVICES here. Under SLURM
# the benches call bare jax.distributed.initialize(), whose SLURM detection
# derives local_device_ids=[SLURM_LOCALID]; pinning CVD to one device
# renumbers it to index 0, so every localid!=0 task asks for a device index
# it cannot see -> "no supported devices found for platform CUDA"
# (observed job 26445837/26447612: preflight nvidia-smi fine, jax dead on
# all nonzero localids). The Slurm-native contract is: all node GPUs
# visible (STEP_GPU_OPTS --gpu-bind=none) + jax picks the LOCALID-th
# device per process. The Derecho twin pins CVD instead because its PALS
# path initializes with local_device_ids=[0] (the #693 shim convention) —
# the two conventions must not be mixed.
PIN='exec "$@"'

rc_all=0

# Per-task GPU-visibility preflight, step-capped at 2 min (diagnoses
# step-gres/binding failures in-log instead of burning lanes on "no
# supported devices found"; nvidia-smi visibility — not a JAX device
# check — because THAT was the observed failure mode).
echo "=== PREFLIGHT: per-task GPU visibility (STEP_GPU_OPTS='$STEP_GPU_OPTS') ==="
# shellcheck disable=SC2086
srun --ntasks="$NP_LL" --ntasks-per-node=4 --time=2 $STEP_GPU_OPTS bash -c "$PIN" _ \
    bash -c 'echo "task=$SLURM_PROCID host=$(hostname -s) CVD=${CUDA_VISIBLE_DEVICES:-UNSET} ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)"' \
    | sort || echo "PREFLIGHT srun failed rc=$?"

if [ "$RUN_NCCL" = "1" ]; then
    echo "=== LANE A: cube cs-spmd np=6 (2 nodes x 3 GPUs) ==="
    # shellcheck disable=SC2086
    srun --ntasks=6 --ntasks-per-node=3 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
        "$PY" scripts/bench/run_cpu_mpi_scaling.py \
        --grid cubed-sphere --cs-spmd --device gpu \
        --physics "$PHYSICS" \
        --mode "$MODE" --resolution "$CS_RES" \
        --precision "$PRECISION" --n-levels "$NLEV" \
        --n-warmup "$N_WARMUP" --n-timing "$N_TIMING" \
        --output-dir "$OUTDIR/nccl_cs_spmd"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "LANE A FAILED rc=$rc"; rc_all=$rc; }
fi

# Gap-#0 lane: native ppermute band halos over NCCL replace the route-A
# mpi4jax >=2-node ceiling. Gate: tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py.
if [ "$RUN_LATLON" = "1" ]; then
    echo "=== LANE C: atm latlon multihost np=$NP_LL (LL$LL_RES) ==="
    # shellcheck disable=SC2086
    srun --ntasks="$NP_LL" --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
        "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
        --multicontroller --n-devices "$NP_LL" \
        --mode "$MODE" --n-lat "$LL_RES" --n-lon $((2 * LL_RES)) \
        --nlev "$NLEV" \
        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
        --out "$OUTDIR/latlon_multihost/spmd_scaling.jsonl"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "LANE C FAILED rc=$rc"; rc_all=$rc; }
fi

# Gate: tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py.
if [ "$RUN_OCEAN" = "1" ]; then
    echo "=== LANE D: ocean latlon multihost np=$NP_LL (LL$OC_RES) ==="
    # shellcheck disable=SC2086
    srun --ntasks="$NP_LL" --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
        "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
        --multicontroller --n-devices "$NP_LL" \
        --mode "$MODE" --n-lat "$OC_RES" --n-lon $((2 * OC_RES)) \
        --nlev "$OC_NLEV" \
        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
        --out "$OUTDIR/ocean_multicontroller/ocean_spmd_scaling.jsonl"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "LANE D FAILED rc=$rc"; rc_all=$rc; }
fi

# Lane E: icosahedral/MPAS multicontroller (6 procs = 2 nodes x 3 GPUs) —
# cell-partition reorder + ppermute halo (make_voronoi_sharded_step) over
# jax.distributed. np=6: nCells = 10*4^L+2 admits 1/2/3/6 even splits at
# every level. Smoke gate (parity+conservation) first, then the timed case.
# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
if [ "$RUN_MPAS" = "1" ]; then
    echo "=== LANE E: icosahedral MPAS multicontroller np=$NP_MPAS (L$ICO_LEVEL) ==="
    # >6 devices: 10*4^L+2 = 2*odd has no even split beyond 6 — pad the mesh
    # to the target count via --reorder-for (identical padded mesh per ladder).
    E_TPN=3; [ $((NP_MPAS % 4)) -eq 0 ] && E_TPN=4
    E_REORDER=""
    [ "$NP_MPAS" -gt 6 ] && E_REORDER="--reorder-for $NP_MPAS"
    # shellcheck disable=SC2086
    srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$NP_MPAS" \
        --subdivision 4 --nlev 8 --steps 4 --warmup 1 \
        --partition-method sfc $E_REORDER \
        --parity-gate --check-conservation \
        --out "$OUTDIR/mpas_multicontroller/smoke.jsonl" \
    && srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$NP_MPAS" \
        --subdivision "$ICO_LEVEL" --nlev "$NLEV" \
        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
        --partition-method sfc $E_REORDER \
        --out "$OUTDIR/mpas_multicontroller/mpas_spmd_scaling.jsonl"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "LANE E FAILED rc=$rc"; rc_all=$rc; }
fi

# Lane T (RUN_TUNE=1): comm-tuning A/B ladder — Derecho twin (see
# gpu_multinode_scaling.pbs lane T + docs/performance/scaling/
# spmd_message_census_2026-07-08.md). Arms: base (control, same allocation),
# fused (LEGOESM_LATLON_SPMD_FUSED_HALO=1 — bit-identical multi-pad packing,
# trace receipt atm 41->29 CPs/step), xla (collective-permute combining +
# pipelined p2p), pgle (profile-guided latency estimation; recompiles after
# the profiling runs — bench-jit-safe, not for AOT jobs). Outputs land under
# _ab_tuning/ which aggregate_bcw_scaling SKIPS (A/B receipts never join the
# scaling curves); compare arms via steady_median_ms / sypd per JSONL row.
RUN_TUNE="${RUN_TUNE:-0}"
if [ "$RUN_TUNE" = "1" ]; then
    echo "=== LANE T: comm-tuning A/B (latlon atm np=8; ocean np=8; cube np=6) ==="
    _XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
    for ARM in base fused xla pgle; do
        echo "--- lane T arm=$ARM: atm latlon LL$LL_RES np=8 ---"
        (
            case "$ARM" in
              fused) export LEGOESM_LATLON_SPMD_FUSED_HALO=1 ;;
              xla)   export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}" ;;
              pgle)  export JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3 ;;
            esac
            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
                "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
                --multicontroller --n-devices 8 \
                --mode strong --n-lat "$LL_RES" --n-lon $((2 * LL_RES)) \
                --nlev "$NLEV" \
                --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
                --out "$OUTDIR/_ab_tuning/latlon_${ARM}.jsonl"
        )
        rc=$?
        [ "$rc" -ne 0 ] && { echo "LANE T latlon arm=$ARM FAILED rc=$rc"; rc_all=$rc; }
    done
    for ARM in base fused; do
        echo "--- lane T arm=$ARM: ocean latlon LL$OC_RES np=8 ---"
        (
            [ "$ARM" = fused ] && export LEGOESM_LATLON_SPMD_FUSED_HALO=1
            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
                "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
                --multicontroller --n-devices 8 \
                --mode strong --n-lat "$OC_RES" --n-lon $((2 * OC_RES)) \
                --nlev "$OC_NLEV" \
                --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
                --out "$OUTDIR/_ab_tuning/ocean_${ARM}.jsonl"
        )
        rc=$?
        [ "$rc" -ne 0 ] && { echo "LANE T ocean arm=$ARM FAILED rc=$rc"; rc_all=$rc; }
    done
    # Cube arms: base vs xla — the census's 46-CPs/step lane (Derecho twin).
    for ARM in base xla; do
        echo "--- lane T arm=$ARM: cube cs-spmd C$CS_RES np=6 ---"
        (
            [ "$ARM" = xla ] && export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"
            srun --ntasks=6 --ntasks-per-node=3 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
                "$PY" scripts/bench/run_cpu_mpi_scaling.py \
                --grid cubed-sphere --cs-spmd --device gpu \
                --physics "$PHYSICS" \
                --mode strong --resolution "$CS_RES" \
                --precision "$PRECISION" --n-levels "$NLEV" \
                --n-warmup "$N_WARMUP" --n-timing "$N_TIMING" \
                --output-dir "$OUTDIR/_ab_tuning/cube_${ARM}"
        )
        rc=$?
        [ "$rc" -ne 0 ] && { echo "LANE T cube arm=$ARM FAILED rc=$rc"; rc_all=$rc; }
    done
fi

echo "=== RESULTS ==="
find "$OUTDIR" \( -name "*.csv" -o -name "*.json" -o -name "*.jsonl" \) | head -20
echo "=== DONE rc=$rc_all ==="
exit "$rc_all"
