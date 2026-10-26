# Sourced only by the round-336 preflight's generated-driver flow tests.
# It replaces the build product with a stub NEMO; it never replaces the driver.
r336_stub_prepare_target() {
  local target=$1
  [[ -n ${R336_STUB_FLOW_ROOT:-} && -d $R336_STUB_FLOW_ROOT ]] \
    || refuse 'stub driver flow root is absent'
  [[ $target == "$R336_STUB_FLOW_ROOT"/* ]] \
    || refuse 'stub target build escaped its flow root'
  mkdir -p "$target/EXP00" "$target/MY_SRC"
}

r336_stub_finish_target() {
  local target=$1 template=${2:-${R336_STUB_BUILD_TEMPLATE:-}} stub=${R336_STUB_NEMO:-}
  [[ -d $template && -d $template/ppsrc && -d $template/obj ]] \
    || refuse 'stub build template is absent'
  [[ -x $stub ]] || refuse 'stub NEMO is absent or not executable'
  mkdir -p "$target/BLD/bin"
  cp -a "$template/." "$target/BLD/"
  # The inherited preflight deliberately leaves its same-program icevar plant
  # in dry_bld after proving that it refuses. Restore that one planted file
  # before using the otherwise-clean dry build as the makenemo stand-in.
  cp -- "$SOURCE_ROOT/BLD/ppsrc/nemo/icevar.f90" \
    "$target/BLD/ppsrc/nemo/icevar.f90"
  cp -- "$stub" "$target/BLD/bin/nemo.exe"
  chmod 700 "$target/BLD/bin/nemo.exe"
}

r336_stub_admit() {
  local run=$1 producer binary_sha scenario
  [[ $run == "$R336_STUB_FLOW_ROOT"/* && -d $run ]] \
    || refuse 'stub admission run escaped its flow root'
  [[ -x $run/nemo && -f $run/producer_commit.txt && -f $run/nemo.sha256 ]] \
    || refuse 'stub admission lacks its executable identity manifests'
  producer=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
    "$run/producer_commit.txt")
  [[ $producer =~ ^[0-9a-f]{40}$ ]] || refuse 'stub producer manifest is malformed'
  git -C "$REPO" cat-file -e "$producer^{commit}" 2>/dev/null \
    || refuse 'stub producer is not a local commit'
  git -C "$REPO" merge-base --is-ancestor "$producer" HEAD \
    || refuse 'stub producer is not an ancestor'
  binary_sha=$(awk 'NF == 2 && $2 == "nemo" && $1 ~ /^[0-9a-f]{64}$/ {print $1}' \
    "$run/nemo.sha256")
  [[ $binary_sha =~ ^[0-9a-f]{64}$ ]] || refuse 'stub binary manifest is malformed'
  [[ $(sha256sum "$run/nemo" | awk '{print $1}') == "$binary_sha" ]] \
    || refuse 'stub binary digest changed'
  [[ -f $run/.r336-stub-scenario ]] || refuse 'stub NEMO did not leave its scenario stamp'
  scenario=$(cat "$run/.r336-stub-scenario")
  case "$scenario" in success|retention) ;; *) refuse 'stub scenario is not admissible' ;; esac
  printf 'R336_STUB_NEMO_ADMISSION_PASS scenario=%s path=%s\n' "$scenario" "$run"
}

r336_check_driver_body() {
  local driver=$1 role=$2 last
  [[ -s $driver ]] || return 1
  grep -F 'case "$MODE" in' "$driver" >/dev/null || return 1
  grep -F 'cleanup() {' "$driver" >/dev/null || return 1
  case "$role" in
    parent)
      grep -F '"$PARENT" >"$ADAPTED"' "$driver" >/dev/null || return 1
      grep -F 'synthetic in-place admission did not pass' "$driver" >/dev/null || return 1
      grep -F '"$ADAPTED" "$MODE"' "$driver" >/dev/null || return 1
      grep -F 'ORCA2_ROUND333_TRALDF_ACQUISITION_PASS' "$driver" >/dev/null || return 1
      ;;
    inner)
      grep -F 'if [[ "$MODE" == --admit-existing ]]' "$driver" >/dev/null || return 1
      grep -F './makenemo -r "$REFERENCE_CFG"' "$driver" >/dev/null || return 1
      grep -F './makenemo -n "$TARGET_CFG"' "$driver" >/dev/null || return 1
      grep -F 'mkdir "$TARGET_RUN"' "$driver" >/dev/null || return 1
      grep -F 'mpirun -np 2 --oversubscribe ./nemo' "$driver" >/dev/null || return 1
      grep -F 'run_completed=1' "$driver" >/dev/null || return 1
      last=$(awk 'NF {line=$0} END {print line}' "$driver")
      [[ $last == admit ]] || return 1
      ;;
    *) return 1 ;;
  esac
}
