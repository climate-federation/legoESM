#!/bin/bash
#SBATCH --job-name=profile_jit_compile
#SBATCH --output=results/profile_jit_compile/slurm_%j.out
#SBATCH --error=results/profile_jit_compile/slurm_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Task #25 profiling driver — measure the JIT compile time and jaxpr
# equation count of the lat-lon C-grid PE step for two SSP-RK3
# variants:
#
#   ssp_rk3       — unrolled 3-stage explicit (legacy default)
#   ssp_rk3_scan  — same coefficients folded into a 3-iter ``lax.scan``
#
# Repeats the measurement at np=1, np=2, np=4 (serial + 2-rank + 4-rank
# MPI) on a small grid (n_lat=16, n_lon=32, nlev=4) so the profiling
# itself fits the 2-h smoke walltime even if a single trace is slow.
#
# Why this exists:
# Smokes 8063948 + 8063949 BOTH timed out at 3 h walltime with the
# 4-rank lat-lon C-grid step still JIT-compiling.  Documented
# expectation was ~25 min per rank-count; actual is >2.5 h.  The
# hypothesis being tested here is that the SSP-RK3's 3× unrolling
# of ``tendency_fn`` is the dominant compile-time cost.

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/profile_jit_compile

PY="${HOME}/.venvs/legoesm-mpi/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /etc/profile.d/modules.sh 2>/dev/null || true
module load openmpi/gcc/64/4.1.7a1 2>/dev/null || true

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null; then
    MPI_LAUNCH=(srun --mpi=pmix --cpus-per-task=1)
else
    MPI_LAUNCH=(mpirun)
fi

echo "========================================"
echo "  JIT compile profile (Task #25)"
echo "  Job ID: ${SLURM_JOB_ID:-<no-slurm>}"
echo "  Python: $(${PY} --version)"
echo "========================================"

# ----------------------------------------------------------------------
# Phase 1: serial profile (np=1, no MPI).  Smoke-fast feedback on
# whether the scan-fold actually reduces jaxpr.eqns + compile time.
# ----------------------------------------------------------------------
echo
echo "=== Phase 1: np=1 (serial) ==="
for INT in ssp_rk3 ssp_rk3_scan; do
    echo
    echo "--- integrator=${INT}, tracers=0 (dry, polar filter off) ---"
    "${PY}" "${REPO_ROOT}/scripts/bench/jit_profile/profile_step_jit_compile.py" \
        --n-lat 16 --n-lon 32 --nlev 4 --tracers 0 \
        --integrator "${INT}"
    echo
    echo "--- integrator=${INT}, tracers=3 (q_v/q_c/q_r), polar filter on ---"
    "${PY}" "${REPO_ROOT}/scripts/bench/jit_profile/profile_step_jit_compile.py" \
        --n-lat 16 --n-lon 32 --nlev 4 --tracers 3 \
        --use-polar-filter --integrator "${INT}"
done

# ----------------------------------------------------------------------
# Phase 2: MPI profile (np=2).  This is where the bloat really shows
# up — at 2 ranks the inline RK3 was already taking ~25 min to compile
# the dry step.  If the scan variant cuts that to <5 min the
# hypothesis is confirmed.
# ----------------------------------------------------------------------
for NP in 2 4; do
    echo
    echo "=== Phase: np=${NP} (MPI) ==="
    for INT in ssp_rk3 ssp_rk3_scan; do
        echo
        echo "--- np=${NP}, integrator=${INT}, tracers=0 ---"
        "${MPI_LAUNCH[@]}" -n ${NP} \
            "${PY}" "${REPO_ROOT}/scripts/bench/jit_profile/profile_step_jit_compile.py" \
            --n-lat 16 --n-lon 32 --nlev 4 --tracers 0 \
            --integrator "${INT}"
    done
done

echo
echo "========================================"
echo "  Profile complete"
echo "  Date: $(date -u)"
echo "========================================"
