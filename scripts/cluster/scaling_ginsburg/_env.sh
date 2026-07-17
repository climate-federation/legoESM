# Shared environment for Ginsburg scaling jobs (sourced by sbatch scripts).
# LEGOESM_REPO overrides the checkout the job runs from (pinned worktrees —
# the shared-checkout long-run hazard: concurrent sessions mutate model code
# mid-run). Default = the historical shared tree.
REPO="${LEGOESM_REPO:-/burg-archive/glab/users/pg2328/legoESM}"
# Repo root FIRST so `import tests` (test_cases ICs reused by the benchmarks)
# resolves regardless of the submitting shell's PYTHONPATH — do not rely on the
# session default being inherited via --export=ALL.
PP="$REPO:$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
# Persistent JIT cache: OFF by default (so scaling benchmarks see a true cold
# compile), but HONOR a caller-provided value — a submit that pre-exports
# LEGOESM_JIT_CACHE_DIR=<dir> keeps it, so expensive one-time compiles (e.g. the
# ~6 h classical rrtmgp+AD training step) cache to disk and reuse across
# epochs/resumes/reruns. The runtime reads LEGOESM_JIT_CACHE_DIR; ""=off.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-}"
# Per-job XLA/ptxas temp dir.  XLA writes PTX to $TMPDIR during subprocess
# compilation.  TWO failure modes seen on the shared nodes: (a) the default
# /local races across co-located jobs ("DeleteFile NOT_FOUND /local/tempfile")
# and (b) /local is small and FILLS during the huge rrtmgp PTX ("ptxas fatal:
# Could not open output file /local/...") — this killed the classical variant
# while column_nn/sfno (smaller compiles) survived.  Use a roomy, fast,
# job-private RAM-backed dir (/dev/shm) with /local then /tmp as fallbacks.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  for _cand in "/dev/shm/${USER}_xla_${SLURM_JOB_ID}" \
               "${SLURM_TMPDIR:-}" \
               "/local/${USER}_xla_${SLURM_JOB_ID}"; do
    [ -z "$_cand" ] && continue
    if mkdir -p "$_cand" 2>/dev/null && [ -w "$_cand" ]; then
      export TMPDIR="$_cand"; break
    fi
  done
  : "${TMPDIR:=/tmp}"; export TMPDIR
fi
PY_GPU=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
PY_MPI="$HOME/.venvs/legoesm-mpi/bin/python"
