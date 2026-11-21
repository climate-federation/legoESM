#!/usr/bin/env bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=gdr_halo
#SBATCH --output=/scratch/b/b384341/amip_gpu_profile/gdr/%x-%j.out
# GPU-direct halo exchange, phase 0 (user 2026-11-17: route A, 2 degree):
# the PRODUCTION batched MPAS state exchange (edge winds + T + p_s + 10
# tracers, 32 levels, float64; the production checkpoint carries 10 tracers)
# at MPAS level 5 (~2 deg, user choice), 4 ranks on one node, three arms on
# the SAME node so only one thing changes between neighbours:
#   A  host-staged, production stack  (.venv, OpenMPI 4.1.2 without CUDA)
#   B  host-staged, HPC-X 2.20        (.venv-gdr, MPI4JAX_USE_CUDA_MPI=0)
#   C  GPU-direct,  HPC-X 2.20        (.venv-gdr, MPI4JAX_USE_CUDA_MPI=1)
# A launches like production (srun, default MPI plugin).  HPC-X cannot start
# under srun here (probe job 29077883: MPI_Init / "PML ucx cannot be selected"
# with every srun MPI plugin) but runs under its own mpirun, so B and C use
# HPC-X mpirun with one GPU per local rank; B and C differ ONLY in
# MPI4JAX_USE_CUDA_MPI.
# Pre-registered: C at least 2x faster per exchange than A -> phase 1;
# C less than 20 % faster than A, or C crashes -> drop route A.  Digests of
# the exchanged arrays must be identical in A, B and C (the exchange copies).
set -uo pipefail
: "${REPO:=/home/b/b384341/projects/legoESM-perf}"
: "${LEVEL:=5}"
: "${N_ITERS:=50}"
OUT=/scratch/b/b384341/amip_gpu_profile/gdr/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
cd "$REPO"
PKG="$REPO/packages"
export PYTHONPATH="$REPO/src:$PKG/core:$PKG/atmosphere:$PKG/coupler:$PKG/tools:$PKG/ml:$PKG/ocean:$PKG/land:$PKG/ice:$REPO"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPI4JAX_NO_WARN_JAX_VERSION=1
unset SLURM_CPUS_PER_TASK
BENCH=( scripts/bench/bench_halo_exchange.py --grid voronoi --payload state
        --level "$LEVEL" --nlev 32 --n-tracers 10 --n-warmup 5 --n-iters "$N_ITERS" )
SRUN=( srun --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4
       --gpu-bind=map_gpu:0,1,2,3 --cpus-per-task=8 --kill-on-bad-exit=1 )
WRAP="$OUT/gpu_local_rank.sh"
printf '#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}\nexec "$@"\n' > "$WRAP"
chmod +x "$WRAP"
MPIRUN=( mpirun -np 4 --bind-to none -x PYTHONPATH -x JAX_PLATFORMS -x JAX_ENABLE_X64
         -x XLA_PYTHON_CLIENT_PREALLOCATE -x MPI4JAX_NO_WARN_JAX_VERSION
         -x MPI4JAX_USE_CUDA_MPI -x LD_LIBRARY_PATH -x PATH -x OPAL_PREFIX "$WRAP" )
nvidia-smi -L | head -4; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
nvidia-smi topo -m 2>/dev/null | head -6

run_arm() {  # name python cuda_mpi launcher(srun|mpirun)
  local name=$1 py=$2 cmpi=$3 how=$4
  local -a L; if [[ $how == srun ]]; then L=( "${SRUN[@]}" ); else L=( "${MPIRUN[@]}" ); fi
  echo "=== arm $name: $py MPI4JAX_USE_CUDA_MPI=$cmpi via $how $(date -u)"
  export MPI4JAX_USE_CUDA_MPI=$cmpi
  "${L[@]}" "$py" -u "${BENCH[@]}" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "arm $name exit=$rc"; tail -5 "$OUT/$name.log"
  return $rc
}

( export LD_LIBRARY_PATH="/sw/spack-levante/openmpi-4.1.2-mnmady/lib:/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  run_arm A "$REPO/.venv/bin/python" 0 srun )
( source "$REPO/scripts/cluster/levante/hpcx_gdr_env.sh"
  run_arm B "$REPO/.venv-gdr/bin/python" 0 mpirun
  run_arm C "$REPO/.venv-gdr/bin/python" 1 mpirun )

"$REPO/.venv/bin/python" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); rows = {}
for arm in "ABC":
    p = out / f"{arm}.json"
    rows[arm] = json.loads(p.read_text()) if p.exists() else None
print("arm  ms/exchange  per-rank medians                cuda_built  library")
for arm, r in rows.items():
    if r is None:
        print(f"{arm}    FAILED"); continue
    e = r["exchanges"]["state"]
    print(f"{arm}    {e['time_per_exchange_ms']:9.3f}  {[round(x,3) for x in e['per_rank_median_ms']]}"
          f"  {r['mpi4jax_cuda_built']}  {r['mpi_library'][:40]}")
ok = [r for r in rows.values() if r]
if len(ok) > 1:
    same = all(r["per_rank_digest"] == ok[0]["per_rank_digest"] for r in ok)
    print("digests identical across arms:", same)
if rows["A"] and rows["C"]:
    a = rows["A"]["exchanges"]["state"]["time_per_exchange_ms"]
    c = rows["C"]["exchanges"]["state"]["time_per_exchange_ms"]
    print(f"speed-up C vs A: {a / c:.2f}x  (pre-registered: >=2 go, <1.25 drop)")
PY
