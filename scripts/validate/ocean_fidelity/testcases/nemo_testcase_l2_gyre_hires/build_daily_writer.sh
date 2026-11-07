#!/usr/bin/env bash
set -Eeuo pipefail

# Operator-only builder.  Read-only preflight is the default; --run creates
# the dedicated NEMO copy.  The math flavour is explicit so a vector oracle
# can never be mistaken for the certified scalar-libm oracle.
do_run=0
math_flavour=vector
while [[ $# -gt 0 ]]; do
  case $1 in
    --run) do_run=1 ;;
    --math-flavour)
      [[ $# -ge 2 ]] || { printf 'REFUSE_USAGE: --math-flavour needs a value\n' >&2; exit 64; }
      math_flavour=$2
      shift
      ;;
    *) printf 'REFUSE_USAGE: expected [--math-flavour vector|scalar] [--run]\n' >&2; exit 64 ;;
  esac
  shift
done
case $math_flavour in
  vector)
    readonly SOURCE_NAME=GYRE_OMIP_L2_P3
    readonly TARGET_NAME=GYRE_OMIP_L2_P3_DAILY
    readonly ARCH=conda
    ;;
  scalar)
    readonly SOURCE_NAME=GYRE_OMIP_L2_P3_SMW
    readonly TARGET_NAME=GYRE_OMIP_L2_P3_SMW_DAILY
    readonly ARCH=conda-scalarmath
    ;;
  *) printf 'REFUSE_MATH_FLAVOUR: expected vector or scalar\n' >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly CREATE_REFERENCE=GYRE_OMIP_L2_P3
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/$SOURCE_NAME
readonly TARGET_CFG=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly PATCH=$here/gyre_p3_daily_writer.patch
readonly SOURCE_SHA=11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee
readonly PATCHED_SHA=a3cc1e37ab362f62bc5aba8b3f4ae0a051a5665ea1491c2043220dd5edd41d83

[[ -f $PATCH && -f $SOURCE_CFG/MY_SRC/stprk3.F90 ]] || {
  printf 'REFUSE_MISSING_INPUT: P3 source or daily patch\n' >&2; exit 67; }
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $SOURCE_SHA ]] || {
  printf 'REFUSE_SOURCE_IDENTITY: stprk3.F90 changed\n' >&2; exit 68; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$SOURCE_CFG/MY_SRC/stprk3.F90" "$dry/"
patch -s --fuzz=0 -d "$dry" -p1 <"$PATCH"
[[ $(sha256sum "$dry/stprk3.F90" | awk '{print $1}') == $PATCHED_SHA ]] || {
  printf 'REFUSE_PATCH_IDENTITY: daily patch result changed\n' >&2; exit 68; }
grep -q 'MOD( kstp - nit000, 24 ) == 0' "$dry/stprk3.F90" || {
  printf 'REFUSE_PATCH_PROGRAM: daily cadence missing\n' >&2; exit 68; }
printf 'PREFLIGHT: source=%s create_reference=%s target=%s math_flavour=%s arch=%s patch=%s\n' \
  "$SOURCE_NAME" "$CREATE_REFERENCE" "$TARGET_NAME" "$math_flavour" "$ARCH" "$PATCH"
if [[ $do_run -ne 1 ]]; then
  printf 'WOULD_BUILD: %s --math-flavour %s --run\nPREFLIGHT_OK\n' \
    "$0" "$math_flavour"
  exit 0
fi

[[ ! -e $TARGET_CFG ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$TARGET_CFG" >&2; exit 69; }
cd "$NEMO_ROOT"
# ``makenemo -r`` resolves only a registered reference/demo config.  The
# scalar source is a work config, so asking makenemo to clone it leaves
# REF_CMP empty and emits ``dirname: missing operand``.  Create from the
# shipped registered P3 reference, then install the selected source's MY_SRC.
./makenemo -r "$CREATE_REFERENCE" -n "$TARGET_NAME" -d "OCE" -m "$ARCH" -j 0
cp -r "$SOURCE_CFG/MY_SRC/." "$TARGET_CFG/MY_SRC/"
patch -s --fuzz=0 -d "$TARGET_CFG/MY_SRC" -p1 <"$PATCH"
[[ $(sha256sum "$TARGET_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $PATCHED_SHA ]] || {
  printf 'REFUSE_PATCH_IDENTITY: installed daily source changed\n' >&2; exit 68; }
readonly WORK_CFGS=$NEMO_ROOT/cfgs/work_cfgs.txt
[[ -f $WORK_CFGS ]] || {
  printf 'REFUSE_MISSING_INPUT: %s\n' "$WORK_CFGS" >&2; exit 67; }
if ! grep -qE "^${TARGET_NAME}[[:space:]]+OCE([[:space:]]|$)" "$WORK_CFGS"; then
  printf '%s OCE\n' "$TARGET_NAME" >>"$WORK_CFGS"
fi
./makenemo -n "$TARGET_NAME" -m "$ARCH" -j 8
[[ $(sha256sum "$TARGET_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $PATCHED_SHA ]] || {
  printf 'REFUSE_BUILD_OVERWROTE_PATCH: daily MY_SRC changed\n' >&2; exit 70; }
grep -q 'MOD( kstp - nit000, 24 ) == 0' \
  "$TARGET_CFG/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: daily cadence is absent\n' >&2; exit 70; }
grep -q 'LANE1_STEP_ENTRY_DUMP' "$TARGET_CFG/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: entry writer is absent\n' >&2; exit 70; }
if [[ $math_flavour == scalar ]] && nm -D "$TARGET_CFG/BLD/bin/nemo.exe" | grep -q '_ZGV'; then
  printf 'REFUSE_COMPILED_MATH: vector-math symbol in scalar writer\n' >&2; exit 70
fi
printf 'BUILD_DONE: %s\n' "$TARGET_CFG/BLD/bin/nemo.exe"
