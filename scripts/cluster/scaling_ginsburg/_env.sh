# Shared environment for Ginsburg scaling jobs (sourced by sbatch scripts).
REPO=/burg-archive/glab/users/pg2328/legoESM
# Repo root FIRST so `import tests` (test_cases ICs reused by the benchmarks)
# resolves regardless of the submitting shell's PYTHONPATH — do not rely on the
# session default being inherited via --export=ALL.
PP="$REPO:$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
# Disable the persistent JIT cache so compile_time_s is a true cold compile
# (the runtime reads LEGOESM_JIT_CACHE_DIR; empty string = off).
export LEGOESM_JIT_CACHE_DIR=""
# Per-job XLA/ptxas temp dir.  XLA writes PTX to $TMPDIR during subprocess
# compilation; the default shared /local races across co-located jobs on a
# node -> "Check failed: DeleteFile(ptx_path) NOT_FOUND /local/tempfile-*"
# crashes mid-compile (killed an n_lat=48 T106 after epoch-0).  A job-private
# TMPDIR removes the collision.  Prefer SLURM's job-scoped scratch.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  export TMPDIR="$SLURM_TMPDIR"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  export TMPDIR="/local/${USER}_xla_${SLURM_JOB_ID}"
  mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/tmp"
fi
PY_GPU=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
PY_MPI="$HOME/.venvs/legoesm-mpi/bin/python"
