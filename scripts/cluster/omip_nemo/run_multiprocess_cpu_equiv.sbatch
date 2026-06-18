#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=spmd_mpequiv
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# CPU multi-device shard_map compile of the full ocean step is heavy (~20 min per
# variant x 2 variants x 2 phases); give it generous walltime.
#SBATCH --time=02:00:00
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

SCRIPT=scripts/validate/validate_latlon_ocean_spmd_multiprocess.py

# TWO PHASES per variant (the SERIAL reference cannot run under jax.distributed --
# its eta-floor reduction would hit batch_allreduce_mpi -> mpi4jax, which is NOT
# installed; the SPMD path is pure-JAX psum).  Phase 1 (single process, 4 CPU
# devices, NO jax.distributed): compute the serial reference + single-process
# 4-device SPMD, assert equal, save the reference.  Phase 2 (mpirun -np 2, 4 global
# devices): run the MULTI-process SPMD step and compare to the saved reference +
# assert the gather is replicated bitwise across processes.  Each phase tees to a
# log, captures the REAL rc through the pipe, and REQUIRES a real [PASS] (a [SKIP]
# means the device setup did not materialize -> treated as failure, not green).
run_variant () {  # $1 = label, $2.. = extra script args (e.g. --tripole)
  local label="$1"; shift
  local ref="/tmp/_spmd_ref_${label}_$SLURM_JOB_ID.npz"
  local rlog="/tmp/_mpequiv_ref_${label}_$SLURM_JOB_ID.log"
  local mlog="/tmp/_mpequiv_mp_${label}_$SLURM_JOB_ID.log"

  echo "--- [${label} phase 1/2] single-process serial + 4-device SPMD reference ---"
  env JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    JAX_ENABLE_X64=1 \
    $PY "$SCRIPT" --phase ref --out "$ref" "$@" 2>&1 | tee "$rlog" | tail -25
  local rc1=${PIPESTATUS[0]}

  echo "--- [${label} phase 2/2] multi-process (mpirun -np 2) SPMD vs reference ---"
  "$MPIRUN" -np 2 env JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
    $PY "$SCRIPT" --phase mp --ref "$ref" "$@" 2>&1 | tee "$mlog" | tail -25
  local rc2=${PIPESTATUS[0]}

  echo "${label}_ref_rc=${rc1}  ${label}_mp_rc=${rc2}"
  local bad=0
  if [ "$rc1" -ne 0 ] || ! grep -q "\[PASS ref\]" "$rlog"; then bad=1; fi
  if [ "$rc2" -ne 0 ] || ! grep -q "\[PASS\]" "$mlog"; then bad=1; fi
  if [ "$bad" -ne 0 ]; then
    echo "[GATE-FAIL] ${label}: ref_rc=${rc1}, mp_rc=${rc2}"
  fi
  rm -f "$ref" "$rlog" "$mlog"
  return $bad
}

rc_all=0
run_variant "latlon"  || rc_all=1
run_variant "tripole" --tripole || rc_all=1
# implicit_cn (the eORCA025 smoke solver): exercises the barotropic PCG + mass-
# conservation reduction SPMD psum path (a different reduction site than the
# explicit-substep eta-floor one the two above hit).
run_variant "tripole_implicitcn" --tripole --implicit-cn || rc_all=1
echo "(rc 0 = both phases PASS for ALL variants; non-zero = a gate failed)"
echo "mpequiv_overall_rc=${rc_all}"
echo "=== DONE ==="
exit ${rc_all}
