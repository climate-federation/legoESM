#!/usr/bin/env bash
set -uo pipefail

readonly EXPECTED_DIR=/data/abyssal/dbalwada/nemo-testcases-l4/runs/uninstrumented_30day_np2
readonly EXPECTED_BINARY_SHA256=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
readonly EXPECTED_DECK_MANIFEST_SHA256=4f8c480d03061ddd44218b0913dabc901fcaa2741b59bcc7d169730ded54d4db
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
if compgen -G 'ORCA2_*restart*.nc' >/dev/null; then
  printf 'REFUSE: restart files already exist in %s\n' "$RUN_DIR" >&2
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
# GNU time is not installed on this host; bash's time keyword writes the same three fields.
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
