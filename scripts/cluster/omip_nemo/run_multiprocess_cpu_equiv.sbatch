#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=spmd_mpequiv
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/spmd_mpequiv_%j.out

# VALIDATION GATE (CLAUDE.md "single-rank -> smallest distributed"): the multi-
# PROCESS jax.distributed equivalence test for the lat-band SPMD ocean step.  Two
# MPI processes x 2 CPU devices each = 4 GLOBAL devices spanning the two processes.
# The bootstrap initializes jax.distributed (rank-0 hostname coordinator); the
# gathered lat-band SPMD step must match the serial single-device step to the FP-
# reassociation floor (~1e-4), AND the gather must be replicated bitwise across
# both processes.  CPU-only (no GPU) -- proves correctness cheaply before the
# 4-GPU smoke.  No mpi4jax needed (the SPMD halo is pure-JAX ppermute/psum).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/multinode
cd "$WT"
# Run the WORKTREE's edited packages (the env's editable install points at a
# different checkout), so PREPEND every package's namespace root.
export PYTHONPATH="$WT/packages/core:$WT/packages/ocean:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/tools:$PYTHONPATH"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
# The conda env is not activated on the compute node, so mpirun is not on PATH;
# use the env's own mpirun (Open MPI 5, the one mpi4py is built against).
MPIRUN=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/mpirun
echo "=== HOST ==="; hostname; date
echo "=== mpi4py / jax versions ==="
$PY -c "import mpi4py,jax; print('mpi4py',mpi4py.__version__,'jax',jax.__version__)"

# One process per task; each forces 2 CPU devices -> 4 global across np=2.  Force
# the CPU platform (no GPU on this short-CPU job) and x64 (ocean is fp64).  Run the
# STANDALONE validation script (NOT pytest): the root tests/conftest.py initializes
# the XLA backend at collection, before jax.distributed.initialize() could run.

# Run one variant under mpirun, capture the REAL rc (the script LAND-reduces all
# ranks), tee to a log, and REQUIRE a real [PASS] (a [SKIP] would mean the 4-device
# setup did not materialize -> treat as failure, not green).  Returns 0 only on
# [PASS] + rc 0.  No `set -e`/`pipefail` reliance: we capture PIPESTATUS explicitly.
run_variant () {  # $1 = label, $2.. = extra args to the script
  local label="$1"; shift
  local log="/tmp/_mpequiv_${label}_$SLURM_JOB_ID.log"
  echo "=== ${label} multi-process equivalence ==="
  "$MPIRUN" -np 2 env \
    JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    JAX_ENABLE_X64=1 \
    $PY scripts/validate/validate_latlon_ocean_spmd_multiprocess.py "$@" 2>&1 | tee "$log" | tail -40
  local rc=${PIPESTATUS[0]}
  echo "${label}_rc=${rc}"
  if [ "$rc" -ne 0 ] || ! grep -q "\[PASS\]" "$log"; then
    echo "[GATE-FAIL] ${label}: rc=${rc}, [PASS] present=$(grep -c '\[PASS\]' "$log")"
    rm -f "$log"
    return 1
  fi
  rm -f "$log"
  return 0
}

rc_all=0
run_variant "latlon"  || rc_all=1
run_variant "tripole" --tripole || rc_all=1
echo "(rc 0 = PASS on every rank for BOTH variants; non-zero = a gate failed)"
echo "mpequiv_overall_rc=${rc_all}"
echo "=== DONE ==="
exit ${rc_all}
