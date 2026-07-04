#!/usr/bin/env bash
# Shared environment for WeatherBench-campaign SLURM jobs.
#
# No pg2328 venv exists; reuse jn2808's conda env and prepend the WORKTREE's
# package namespace roots so `import legoesm` / `import evaluations` resolve to
# THIS worktree's code (not jn2808's editable install). See memory python-env.
WB_WORKTREE="/burg-archive/glab/users/pg2328/legoESM_wbforecast"
WB_PYTHON="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

_pp="${WB_WORKTREE}"
for _d in "${WB_WORKTREE}"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export WB_PYTHON
export WB_WORKTREE
# match the AIMIP runner env (suppress mpi4jax warning; disable persistent JIT cache)
export MPI4JAX_NO_WARN_JAX_VERSION=1
export LEGOESM_JIT_CACHE_DIR=""
