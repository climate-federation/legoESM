#!/usr/bin/env bash
set -euo pipefail

RANK="${OMPI_COMM_WORLD_RANK:?Open MPI rank is required}"
grep '^Cpus_allowed_list:' /proc/self/status >"binding_rank${RANK}.txt"
/usr/bin/time \
  -f $'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kib=%M' \
  -o "rank_time${RANK}.txt" \
  ./nemo
