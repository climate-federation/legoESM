#!/usr/bin/env bash
# Run the legoESM distributed (MPI) test suite reliably.
#
# WHY per-file mpirun: each distributed test file uses its own comm topology
# (latlon partition, cubed-sphere face split, ...) and ``mpi4jax.sendrecv``.
# Running MULTIPLE files in ONE ``mpirun`` session desyncs the ranks — a
# leftover/unmatched request from one file trips MPICH's ``dtp_ != NULL``
# assertion (ch3u_request.c) or deadlocks the next file. Each file passes in
# isolation, so we launch a FRESH ``mpirun`` per file (fresh MPI_Init/Finalize),
# guarded by a HARD wall-clock ``timeout`` (pytest's signal-timeout can't
# interrupt a blocked MPI C call).
#
# Usage:
#   bash scripts/experiment/run_mpi_tests.sh [np] [per_file_timeout_s]
#   bash scripts/experiment/run_mpi_tests.sh 2 300
set -u

NP="${1:-2}"
TMO="${2:-300}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source scripts/experiment/setup_mpi_local.sh env

PY=".venv-mpi/bin/python"
declare -a FILES
mapfile -t FILES < <(ls tests/distributed/test_*.py | sort)

pass=0; fail=0; hang=0
declare -a FAILED
for f in "${FILES[@]}"; do
  printf '%-58s ' "$(basename "$f")"
  log="/tmp/mpitest_$(basename "$f" .py).log"
  timeout -s KILL "$TMO" mpirun -np "$NP" "$PY" -m pytest "$f" \
      -q -p no:cacheprovider --timeout=$((TMO - 30)) > "$log" 2>&1
  rc=$?
  if [[ $rc -eq 137 || $rc -eq 124 ]]; then
    echo "HANG/TIMEOUT (rc=$rc)"; hang=$((hang+1)); FAILED+=("$f [hang]")
  elif [[ $rc -eq 0 ]]; then
    res=$(grep -aoE '[0-9]+ passed[a-z, 0-9]*' "$log" | tail -1)
    echo "PASS  ${res:-ok}"; pass=$((pass+1))
  else
    res=$(grep -aoE '[0-9]+ (passed|failed|error)[a-z, 0-9]*' "$log" | tail -1)
    echo "FAIL (rc=$rc) ${res:-}"; fail=$((fail+1)); FAILED+=("$f")
  fi
done

echo "------------------------------------------------------------"
echo "MPI suite (np=$NP): $pass passed-files, $fail failed-files, $hang hung-files"
if (( fail + hang > 0 )); then
  printf '  problem: %s\n' "${FAILED[@]}"
  exit 1
fi
echo "All distributed test files pass under MPI."
