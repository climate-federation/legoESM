#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=nature_ladder_%j.log

# Nature-figure scaling ladders (2026-09-16): strong + weak, f32 + f64, one
# JSONL receipt per arm, every grid on the SAME shard_map SPMD benches for
# CPU (gloo) and GPU (NCCL).  Arms whose node need exceeds the allocation
# are skipped, so the same script is submitted at several node counts:
#
#   MATRIX=atm_gpu   sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 8|32|48 ...
#   MATRIX=atm_cpu   sbatch -p compute -A bd1179 -N 16 --time=08:00:00 ...
#   MATRIX=ocean_gpu / ocean_cpu likewise.
#
# Existing receipts that this ladder does NOT repeat (reuse decision):
# atm lat-lon LL2048/LL4096 GPU strong f32/f64 1-128 (Aug 2026,
# /work/bd1083/b309178/diffESM/scaling_receipts/ladder2n).  The MPAS s9/s10
# GPU strong series from the same campaign ran at 8 NCCL channels and is NOT
# reused: both MPAS lanes now pin 32 channels (2026-09-21 sweep at s9:
# 5.57 -> 4.65 ms at 128 GPUs, 7.70 -> 7.26 at 32; atm_mpas_nccl_sweep.sbatch)
# and the plotter refuses MPAS GPU rows measured at any other channel count.
#
# Weak arms of the MPAS-atm and cube lanes are strong-bench runs at a larger
# mesh per device count (those benches have no --mode); the receipt then says
# mode=strong and the "_weak_" tag in the FILENAME is what marks them — the
# plotter (plot_nature_scaling.py) reads the mode from the filename first.
#
# Per-lane env is pinned per arm (the adopted production comm recipe), never
# inherited between arms.  Precision = JAX_ENABLE_X64; every bench
# set_policy()s from it so the f64 arm carries real f64 state.

set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
MATRIX="${MATRIX:?set MATRIX=atm_gpu|atm_cpu|ocean_gpu|ocean_cpu}"
BACKEND="${MATRIX##*_}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
if [[ "$BACKEND" == gpu ]]; then
  export JAX_PLATFORMS=cuda,cpu
  PPN=4; ARM_TIMEOUT="${ARM_TIMEOUT:-1200}"
  # Each GPU rank gets its full 64-thread host share (A/B gpu_cpubind_ab,
  # job 27586798, 32 GPUs x2: ocean-MPAS 43.25 -> 42.62 ms, atm-MPAS 7.36
  # -> 7.23/7.39, i.e. -1.5% / within spread); receipts stamp cpu_affinity.
  SRUN_DEV=(--gpus-per-node=4 --gpu-bind=none --cpu-bind=cores
            --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))")
else
  export JAX_PLATFORMS=cpu
  # Overridable: the 256- and 512-rank CPU arms spend most of their wall
  # clock building and partitioning the mesh on every rank, and silently
  # died at the fixed 2400 s.
  PPN="${PPN:-8}"; ARM_TIMEOUT="${ARM_TIMEOUT:-2400}"
  # Each rank gets its full share of the node's hardware threads.  Without
  # --cpus-per-task srun binds every rank to ONE core (affinity = 2 hw
  # threads): every CPU ladder before 2026-09-21 ran that way, 7.4x slower
  # per rank than the 32-core share (s7 atm, 4 ranks/node: 1270 -> 172 ms).
  # 4 ranks/node measured best on the interactive node (4x32: 172, 8x16:
  # 195, 16x8: 199, 2x64: 211 ms); every CPU submission passes PPN=4.
  # SLURM_CPUS_ON_NODE is the batch step's per-node allocation (whole node
  # under --exclusive); no fallback, a wrong divisor must fail loudly.
  SRUN_DEV=(--cpu-bind=cores --distribution=block:cyclic
            --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))")
fi
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"   # $REPO, $PY, PYTHONPATH
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/nature_${MATRIX}_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"
MAX_ND="${MAX_ND:-$((NODES * PPN))}"
MIN_ND="${MIN_ND:-1}"
STEPS="${STEPS:-60}"; WARMUP="${WARMUP:-10}"
echo "matrix=$MATRIX backend=$BACKEND nodes=$NODES ppn=$PPN nd=[$MIN_ND,$MAX_ND] outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

# run_arm <lane> <mode> <res> <nd> <x64>
#   lane: latlon | mpas | cube | tripole | omps    res: latlon rows / mpas level / cube N:kt
run_arm() {
  local lane="$1" mode="$2" res="$3" nd="$4" x64="$5"
  # LANES filters this submission to a subset of lanes, so a lane whose
  # per-rank footprint forces a different rank density can be re-run at that
  # density WITHOUT re-running (and so changing the density of) the others.
  if [[ -n "${LANES:-}" && " $LANES " != *" $lane "* ]]; then return 0; fi
  # MODES likewise: the strong arms need a lower rank density than the weak
  # ones (per-rank memory grows with rank count on this lane), and a curve
  # whose points were measured at different densities is not a curve.
  if [[ -n "${MODES:-}" && " $MODES " != *" $mode "* ]]; then return 0; fi
  local nn=$(( (nd + PPN - 1) / PPN ))
  if (( nn > NODES || nd > MAX_ND || nd < MIN_ND )); then
    echo "[skip] ${lane}_${mode}_${res} d${nd} (needs ${nn} nodes / outside [$MIN_ND,$MAX_ND])"
    return 0
  fi
  local tag="${lane}_${mode}_${res//:/k}_d${nd}_x${x64}"
  if grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null; then echo "[have] $tag"; return 0; fi
  rm -f "$OUTDIR/${tag}.jsonl"
  export JAX_ENABLE_X64="$x64"
  unset NCCL_P2P_NET_CHUNKSIZE NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS \
        LEGOESM_LATLON_PACKED_EXCHANGE LEGOESM_LATLON_SPMD_FUSED_HALO LEGOESM_MESH_CACHE_DIR \
        LEGOESM_OCEAN_MIXED LEGOESM_VMIX_F32_SOLVE LEGOESM_BAROCLINIC_F32 LEGOESM_ENABLE_TF32
  local bench args=(--multicontroller --n-devices "$nd")
  case "$lane" in
    latlon)
      export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
      export LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1
      bench=scripts/bench/bench_atm_latlon_spmd_scaling.py
      if [[ "$mode" == strong ]]; then
        args+=(--mode strong --n-lat "$res" --n-lon $((2 * res)))
      else
        args+=(--mode weak --nlat-per-dev "$res" --n-lon "$NLON_WEAK")
      fi
      args+=(--nlev 26 --steps "$STEPS" --warmup "$WARMUP") ;;
    mpas)
      export NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=131072
      export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
      bench=scripts/bench/bench_mpas_spmd_scaling.py
      args+=(--subdivision "$res" --nlev 26 --lloyd 0 --wide-halo --partition-method sfc
             --steps "$STEPS" --warmup "$WARMUP") ;;
    cube)
      if (( ${res%:*} % ${res#*:} != 0 )); then echo "[skip] $tag: kt does not divide N"; return 0; fi
      export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
      bench=scripts/bench/bench_cube_tiled_step_scaling.py
      # the tiled bench has no --n-devices: the count is 6*kt^2 by construction
      args=(--multicontroller --kt "${res#*:}" --resolution "${res%:*}" --nlev 26
            --steps "$STEPS" --warmup "$WARMUP") ;;
    tripole)
      export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
      bench=scripts/bench/bench_ocean_latlon_spmd_scaling.py
      args+=(--tripole --nlev 75)
      if [[ "$mode" == strong ]]; then
        args+=(--mode strong --n-lat "$res" --n-lon "$NLON_TRIPOLE")
      else
        args+=(--mode weak --nlat-per-dev "$res" --n-lon "$NLON_TRIPOLE")
      fi
      args+=(--steps "$STEPS" --warmup "$WARMUP") ;;
    omps)
      export NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=131072
      export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
      bench=scripts/bench/bench_ocean_mpas_spmd_scaling.py
      args+=(--nlev 40 --lloyd 0 --block-steps 20 --blocks 3)
      if [[ "$mode" == strong ]]; then args+=(--subdivision "$res")
      else args+=(--mode weak --cells-per-device "$res"); fi ;;
    *) echo "unknown lane $lane"; return 1 ;;
  esac
  args+=(--out "$OUTDIR/${tag}.jsonl")
  echo "=== arm $tag nd=$nd nn=$nn $(date '+%F %T')"
  if ! timeout --kill-after=30 "$ARM_TIMEOUT" srun --export=ALL --nodes="$nn" --ntasks="$nd" \
         --ntasks-per-node="$PPN" "${SRUN_DEV[@]}" --kill-on-bad-exit=1 \
         "$PY" "$bench" "${args[@]}" < /dev/null; then
    echo "!!! FAILED $tag (continuing)"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}

ARMS=()
add() {  # add <lane> <mode> <res> <nd...>   (both precisions)
  local lane="$1" mode="$2" res="$3"; shift 3
  local nd x64
  for nd in "$@"; do for x64 in 0 1; do ARMS+=("$lane" "$mode" "$res" "$nd" "$x64"); done; done
}
P2_GPU="1 2 4 8 16 32 64 128 192"
P2_CPU="1 2 4 8 16 32 64 128 256 512"
case "$MATRIX" in
  atm_gpu)
    NLON_WEAK=8192
    add latlon weak   32   $P2_GPU                # 32 rows x 8192 x 26 per GPU (= LL4096@128 share)
    add mpas   strong 9    $P2_GPU
    add mpas   strong 10   2 4 8 16 32 64 128 192
    add mpas   weak 7 2;  add mpas weak 8 8;  add mpas weak 9 32;  add mpas weak 10 128   # ~81k cells/GPU
    add cube   strong 768:2 24; add cube strong 768:3 54
    add cube   strong 768:4 96                    # kt must divide 768: {2,3,4,6,8}; 216 > GPU cap
    add cube   weak 192:2 24; add cube weak 288:3 54; add cube weak 384:4 96
    ;;
  atm_cpu)
    NLON_WEAK=8192
    add latlon strong 4096 32 64 128 256 512
    add latlon weak   8    $P2_CPU                # 8 rows x 8192 x 26 per rank (= LL4096@512 share)
    add mpas   strong 9    16 32 64 128 256 512
    add mpas   weak 5 2; add mpas weak 6 8; add mpas weak 7 32; add mpas weak 8 128; add mpas weak 9 512
    add cube   strong 768:2 24; add cube strong 768:3 54; add cube strong 768:4 96
    add cube   strong 768:6 216; add cube strong 768:8 384   # kt must divide 768
    add cube   weak 192:2 24; add cube weak 288:3 54; add cube weak 384:4 96; add cube weak 768:8 384
    ;;
  ocean_gpu)
    NLON_TRIPOLE=4352                             # ORCA12-class: 3072 x 4352 x 75
    add tripole strong 3072 8 16 32 64 128 192
    add tripole weak   24   $P2_GPU
    add omps    strong 9    $P2_GPU               # ico9 2.6M cells (~15 km) x 40 levels
    # Twice as fine: ico10 is 10.5M cells (~7.5 km, near 1/15 degree). Its
    # mesh is already cached. Four devices carry the same per-device load as
    # one device does at ico9, which is a row that ran, so the ladder starts
    # there rather than at one.
    add omps    strong 10   4 8 16 32 64 128 192
    add omps    weak 20480 2 8 32 128             # level-quantized: s6@2, s7@8, s8@32, s9@128
    ;;
  ocean_cpu)
    NLON_TRIPOLE=4352
    add tripole strong 3072 32 64 128 256 512
    add tripole weak   6    $P2_CPU
    add omps    strong 9    16 32 64 128 256 512
    add omps    weak 5120   2 8 32 128 512         # s5@2 ... s9@512
    ;;
  *) echo "unknown MATRIX $MATRIX"; exit 2 ;;
esac

n=$(( ${#ARMS[@]} / 5 ))
echo "### $n arms"
for ((i = 0; i < n; i++)); do s=$((i * 5)); run_arm "${ARMS[@]:s:5}"; done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
