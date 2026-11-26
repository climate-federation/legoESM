#!/usr/bin/env bash
set -euo pipefail

RANK="${OMPI_COMM_WORLD_RANK:?Open MPI rank is required}"
# Rank r pins itself to the r-th CPU of the operator's list, as legoESM is pinned with taskset.
# Open MPI 5 --cpu-set/--bind-to core binds every rank to the whole set (and refuses np=40 across sockets).
IFS=, read -ra CPUS <<<"${ROUND349_CPU_IDS:?operator CPU list is required}"
taskset -cp "${CPUS[RANK]}" $$ >/dev/null
grep '^Cpus_allowed_list:' /proc/self/status >"binding_rank${RANK}.txt"
"${GNU_TIME_BIN:-/usr/bin/time}" \
  -f $'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kib=%M' \
  -o "rank_time${RANK}.txt" \
  ./nemo
