#!/usr/bin/env bash
# Shared environment for unified WB+AIMIP training jobs — MAIN repo checkout.
#
# No pg2328 venv exists; reuse jn2808's conda env and prepend THIS checkout's
# package namespace roots so `import legoesm` / `import evaluations` resolve
# here (see memory python-env). Mirrors scripts/cluster/wb_forecast/env.sh,
# which pins the legoESM_wbforecast worktree instead.
UT_REPO="${UT_REPO:-/burg-archive/glab/users/pg2328/legoESM}"
UT_PYTHON="${UT_PYTHON:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}"

_pp="${UT_REPO}"
for _d in "${UT_REPO}"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export UT_PYTHON
export UT_REPO
export MPI4JAX_NO_WARN_JAX_VERSION=1
export LEGOESM_JIT_CACHE_DIR=""
