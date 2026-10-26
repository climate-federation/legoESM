# Sourced only by the round-336 preflight's generated-driver flow tests.
# The caller has already redirected EVIDENCE/TARGET_RUN into a temporary root.
r336_stub_driver_flow() {
  local role=${R336_STUB_DRIVER_ROLE:-unknown}
  local stub=${R336_STUB_NEMO:-}
  local scenario=${R336_STUB_NEMO_SCENARIO:-success}
  local canonical retained producer binary_sha rc lines

  [[ -n ${R336_STUB_FLOW_ROOT:-} && -d $R336_STUB_FLOW_ROOT ]] \
    || refuse 'stub driver flow root is absent'
  [[ -n $stub && -x $stub ]] || refuse 'stub NEMO is absent or not executable'
  mkdir -p "$EVIDENCE"

  canonical=$TARGET_RUN
  if declare -p DEFAULT_TARGET_RUN >/dev/null 2>&1; then
    canonical=$DEFAULT_TARGET_RUN
  fi

  if [[ $MODE == --admit-existing ]]; then
    retained=${R328_EXISTING_RUN:-}
    [[ -n $retained && -d $retained ]] || refuse 'stub retained run is absent'
    [[ $(dirname "$retained") == "$EVIDENCE/refused" ]] \
      || refuse 'stub retained run is outside refused/'
    [[ ! -e $canonical ]] || refuse 'stub canonical admission target already exists'
    [[ -x $retained/nemo ]] || refuse 'stub retained run lacks NEMO'
    [[ -f $retained/producer_commit.txt && -f $retained/nemo.sha256 ]] \
      || refuse 'stub retained run lacks identity manifests'
    producer=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
      "$retained/producer_commit.txt")
    [[ $producer =~ ^[0-9a-f]{40}$ ]] || refuse 'stub producer manifest is malformed'
    git -C "$REPO" merge-base --is-ancestor "$producer" HEAD \
      || refuse 'stub producer is not an ancestor'
    binary_sha=$(awk 'NF == 2 && $2 == "nemo" && $1 ~ /^[0-9a-f]{64}$/ {print $1}' \
      "$retained/nemo.sha256")
    [[ $binary_sha =~ ^[0-9a-f]{64}$ ]] || refuse 'stub binary manifest is malformed'
    [[ $(sha256sum "$retained/nemo" | awk '{print $1}') == "$binary_sha" ]] \
      || refuse 'stub binary digest changed'
    mv -- "$retained" "$canonical"
    printf 'R336_STUB_DRIVER_ADMIT_PASS role=%s path=%s\n' "$role" "$canonical"
    exit 0
  fi

  [[ $MODE == --run ]] || refuse 'stub driver flow requires --run or --admit-existing'
  [[ ! -e $canonical ]] || refuse 'stub run target already exists'
  mkdir "$canonical"
  cp -- "$stub" "$canonical/nemo"
  chmod 700 "$canonical/nemo"
  git -C "$REPO" rev-parse HEAD >"$canonical/producer_commit.txt"
  ( cd "$canonical" && sha256sum nemo >nemo.sha256 )
  if declare -p retention_armed >/dev/null 2>&1; then retention_armed=1; fi
  if declare -p created_run >/dev/null 2>&1; then created_run=1; fi

  set +e
  ( cd "$canonical" && R336_STUB_NEMO_SCENARIO=$scenario \
      ./nemo >run.user.stdout.log 2>&1 )
  rc=$?
  set -e
  lines=$(grep -nF 'E R R O R' "$canonical/ocean.output" 2>/dev/null || true)
  [[ -z $lines ]] || {
    printf 'NEMO_ERROR_LINES_BEGIN %s\n%s\nNEMO_ERROR_LINES_END\n' \
      "$canonical/ocean.output" "$lines" >&2
    refuse 'stub NEMO ocean.output contains E R R O R'
  }
  [[ $rc -eq 0 ]] || refuse "stub NEMO exited $rc"
  grep -F 'STOP 0' "$canonical/run.user.stdout.log" >/dev/null \
    || refuse 'stub NEMO did not report STOP 0'
  if declare -p run_completed >/dev/null 2>&1; then run_completed=1; fi
  if declare -p retention_armed >/dev/null 2>&1; then retention_armed=0; fi
  printf 'R336_STUB_DRIVER_SUCCESS_PASS role=%s path=%s\n' "$role" "$canonical"
  exit 0
}
