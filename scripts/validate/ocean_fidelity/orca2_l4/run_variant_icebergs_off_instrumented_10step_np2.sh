#!/usr/bin/env bash
set -uo pipefail

readonly EXPECTED_DIR=/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_instrumented_10step_np2
readonly EXPECTED_BINARY_SHA256=b31fc33edd3109a41640f9fb59f915d90a52f1f0509fde7c33254cf46b896f28
readonly EXPECTED_DECK_MANIFEST_SHA256=e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5

RUN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [[ "$RUN_DIR" != "$EXPECTED_DIR" ]]; then
  printf 'REFUSE: expected run directory %s; resolved %s\n' "$EXPECTED_DIR" "$RUN_DIR" >&2
  exit 64
fi
cd "$RUN_DIR" || exit 65

for output in ocean.output time.step run.user.stdout.log run.user.time.log; do
  if [[ -e "$output" ]]; then
    printf 'REFUSE: output already exists: %s/%s\n' "$RUN_DIR" "$output" >&2
    exit 66
  fi
done
if compgen -G 'ORCA2_*restart*.nc' >/dev/null || compgen -G 'oracle_*.bin' >/dev/null; then
  printf 'REFUSE: restart or oracle records already exist in %s\n' "$RUN_DIR" >&2
  exit 67
fi

printf '%s  %s\n' "$EXPECTED_BINARY_SHA256" nemo | sha256sum -c - || exit 68
printf '%s  %s\n' "$EXPECTED_DECK_MANIFEST_SHA256" deck_files.sha256 | sha256sum -c - || exit 68
printf '%s  %s\n' "$EXPECTED_INPUT_MANIFEST_SHA256" input_files.sha256 | sha256sum -c - || exit 68
sha256sum -c deck_files.sha256 || exit 68
sha256sum -c input_files.sha256 || exit 68

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
{ time mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } 2>>run.user.time.log
pipe_rc=("${PIPESTATUS[@]}")
mpi_rc=${pipe_rc[0]}
tee_rc=${pipe_rc[1]:-0}
printf 'MPIRUN_RC=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
if (( mpi_rc == 0 )); then
  printf 'RUN DONE\n' >>run.user.time.log
else
  printf 'RUN FAILED\n' >>run.user.time.log
fi
(( mpi_rc != 0 )) && exit "$mpi_rc"
(( tee_rc != 0 )) && exit "$tee_rc"
exit 0
