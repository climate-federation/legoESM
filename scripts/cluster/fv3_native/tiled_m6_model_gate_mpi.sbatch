#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m6_mpi
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_mpi_gate_%j.log
# M6 on REAL RANKS: one window per process, XLA:CPU collectives=mpi
# (MPItrampoline -> MPIwrapper -> OpenMPI), jax.distributed via SLURM.
# Usage: sbatch --nodes=N --ntasks=6*kt*kt --export=ALL,KT=..,PAD=..,REF=<npz> this
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
WRAPLIB=/burg-archive/glab/users/pg2328/mpiwrapper/install/lib64/libmpiwrapper.so
cd "$WT" || exit 1
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 JAX_LOG_COMPILES=1
export LEGOESM_JIT_CACHE_DIR=""
export XLA_FLAGS="--xla_cpu_max_isa=AVX"
# task/cgroup without affinity: every rank sees all 32 cores, so XLA:CPU's
# intra-op (Eigen) pool is 32 threads PER RANK -- 8x32 spinning threads on
# 32 cores starve MPI progress (job 9654131: 12 ms per halo ppermute vs
# 0.11 ms bare MPI on 54 ranks).  BIND=1 (default) pins each rank to its
# --cpus-per-task cores with taskset, which XLA's pool size follows.
BIND=${BIND:-1}
LAUNCH=""; [ "$BIND" = 1 ] && LAUNCH="$WT/scripts/cluster/fv3_native/bind_rank.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
source /etc/profile.d/modules.sh
module load openmpi/gcc/64/4.1.7a1
for v in $(env | sed -n 's/^\(MPITRAMPOLINE_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=${COLL:-mpi}
[ "$JAX_CPU_COLLECTIVES_IMPLEMENTATION" = mpi ] && export MPITRAMPOLINE_LIB="$WRAPLIB" MPITRAMPOLINE_VERBOSE=1
N=${N:-48}; KM=${KM:-10}; KT=${KT:-2}; PAD=${PAD:-11}; NHFLAG=""; [ "${NH:-0}" = 1 ] && NHFLAG="--nh"
echo "host: $(hostname) bind=$BIND cpus/task=${SLURM_CPUS_PER_TASK:-?} ranks=$SLURM_NTASKS nodes=$SLURM_JOB_NUM_NODES sha: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l) coll=$JAX_CPU_COLLECTIVES_IMPLEMENTATION"
# PREFLIGHT (2026-09-05): nodes g[195-201] carried a 40 ms neighbour
# ppermute (0.11 ms elsewhere) and three rows ran 150x slow on them.  One
# bare ppermute round trip per allocation; a slow fabric refuses the row.
PREOUT=$(srun --mpi=pmix $LAUNCH "$PY" scripts/bench/bench_fv3_duo_spmd_scaling.py --mode bare --n-timed 30 2>&1)
PRE=$(echo "$PREOUT" | grep -m1 "ONE neighbour ppermute")
echo "preflight: ${PRE:-NO RESULT} (nodes $SLURM_JOB_NODELIST)"
echo "$PREOUT" | grep -m1 "BARE-SLOWEST"
US=$(echo "$PRE" | grep -oE "= [0-9.]+ us" | grep -oE "[0-9.]+")
if [ -z "$US" ] || [ "$(echo "$US > ${PREFLIGHT_MAX_US:-1000}" | bc)" = 1 ]; then
  echo "PREFLIGHT REFUSED: neighbour ppermute ${US:-?} us > ${PREFLIGHT_MAX_US:-1000} us on $SLURM_JOB_NODELIST"; echo "SRUN_RC=99"; echo DONE m6_mpi_gate; exit 99
fi
echo "===== M6 MPI gate C$N km=$KM ${NHFLAG:-hydro} kt=$KT pad=$PAD ranks=$SLURM_NTASKS ref=${REF:-none} ====="
srun --mpi=pmix --kill-on-bad-exit=0 $LAUNCH "$PY" scripts/validate/fv3_native/tiled_m6_model_gate.py --n "$N" --km "$KM" $NHFLAG --kt "$KT" --pad "$PAD" --n-split "${NSPLIT:-3}" --steps "${STEPS:-2}" --distributed ${REF:+--ref-npz $REF} ${GATE_EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^$|RuntimeError: jaxlib|MPItrampoline\]|MPIwrapper|Open MPI v|Deprecat|shard_map import"
RC=${PIPESTATUS[0]}
echo "SRUN_RC=$RC"
echo DONE m6_mpi_gate
exit $RC
