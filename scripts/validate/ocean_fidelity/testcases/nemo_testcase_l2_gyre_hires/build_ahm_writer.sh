#!/usr/bin/env bash
set -Eeuo pipefail

# Operator-only builder.  The default validates the pristine P3 source and the
# patch without touching NEMO; --run creates and compiles the dedicated copy.
do_run=0
[[ $# -eq 0 ]] || [[ $# -eq 1 && $1 == --run ]] || {
  printf 'REFUSE_USAGE: expected [--run]\n' >&2; exit 64; }
[[ $# -eq 0 ]] || do_run=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/GYRE_OMIP_L2_P3
readonly TARGET_NAME=GYRE_OMIP_L2_P3_AHM
readonly TARGET_CFG=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly PATCH=$here/gyre_p3_ahm_writer.patch

[[ -f $PATCH && -f $SOURCE_CFG/MY_SRC/stprk3.F90 && -f $SOURCE_CFG/MY_SRC/stprk3_stg.F90 ]] || {
  printf 'REFUSE_MISSING_INPUT: P3 source or AHM patch\n' >&2; exit 67; }
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == 11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee ]] || {
  printf 'REFUSE_SOURCE_IDENTITY: stprk3.F90 changed\n' >&2; exit 68; }
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3_stg.F90" | awk '{print $1}') == 4d3493de0e837654e961a30ac4684801d87b4226ed9e8c226922db780d699afe ]] || {
  printf 'REFUSE_SOURCE_IDENTITY: stprk3_stg.F90 changed\n' >&2; exit 68; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$SOURCE_CFG/MY_SRC/stprk3.F90" "$SOURCE_CFG/MY_SRC/stprk3_stg.F90" "$dry/"
patch -s --fuzz=0 -d "$dry" -p1 <"$PATCH"
[[ $(sha256sum "$dry/stprk3.F90" | awk '{print $1}') == b2b73a7899dfd06c9eaf8106d0c332d0d51f93a3e8bad1c44b2d58353caa6a9a \
   && $(sha256sum "$dry/stprk3_stg.F90" | awk '{print $1}') == f65546e500ccd0b4d84e9641eeeb11d7e9160fe36a1339c1a2426425322d07e7 ]] || {
  printf 'REFUSE_PATCH_IDENTITY: AHM patch result changed\n' >&2; exit 68; }
printf 'PREFLIGHT: source=GYRE_OMIP_L2_P3 target=%s patch=%s\n' "$TARGET_NAME" "$PATCH"
if [[ $do_run -ne 1 ]]; then
  printf 'WOULD_BUILD: %s --run\nPREFLIGHT_OK\n' "$0"
  exit 0
fi

[[ ! -e $TARGET_CFG ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$TARGET_CFG" >&2; exit 69; }
cd "$NEMO_ROOT"
./makenemo -r GYRE_OMIP_L2_P3 -n "$TARGET_NAME" -d "OCE" -m conda -j 0
patch -s --fuzz=0 -d "$TARGET_CFG/MY_SRC" -p1 <"$PATCH"
[[ $(sha256sum "$TARGET_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == b2b73a7899dfd06c9eaf8106d0c332d0d51f93a3e8bad1c44b2d58353caa6a9a \
   && $(sha256sum "$TARGET_CFG/MY_SRC/stprk3_stg.F90" | awk '{print $1}') == f65546e500ccd0b4d84e9641eeeb11d7e9160fe36a1339c1a2426425322d07e7 ]] || {
  printf 'REFUSE_PATCH_IDENTITY: installed AHM source changed\n' >&2; exit 68; }
# makenemo -j 0 creates the copy but does not register it.  Register exactly
# the component list used for creation; a bare subsequent -n otherwise prints
# usage and never compiles.
readonly WORK_CFGS=$NEMO_ROOT/cfgs/work_cfgs.txt
[[ -f $WORK_CFGS ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$WORK_CFGS" >&2; exit 67; }
if ! grep -qE "^${TARGET_NAME}[[:space:]]+OCE([[:space:]]|$)" "$WORK_CFGS"; then
  printf '%s OCE\n' "$TARGET_NAME" >>"$WORK_CFGS"
fi
./makenemo -n "$TARGET_NAME" -m conda -j 8
[[ $(sha256sum "$TARGET_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == b2b73a7899dfd06c9eaf8106d0c332d0d51f93a3e8bad1c44b2d58353caa6a9a \
   && $(sha256sum "$TARGET_CFG/MY_SRC/stprk3_stg.F90" | awk '{print $1}') == f65546e500ccd0b4d84e9641eeeb11d7e9160fe36a1339c1a2426425322d07e7 ]] || {
  printf 'REFUSE_BUILD_OVERWROTE_PATCH: AHM MY_SRC changed\n' >&2; exit 70; }
for token in NEMO_L4_AHMTF__1 NEMO_L4_DYNLDF_1; do
  grep -q "$token" "$TARGET_CFG/BLD/ppsrc/nemo/stprk3.f90" \
    "$TARGET_CFG/BLD/ppsrc/nemo/stprk3_stg.f90" || {
    printf 'REFUSE_COMPILED_WRITER: missing %s\n' "$token" >&2; exit 70; }
done
printf 'BUILD_DONE: %s\n' "$TARGET_CFG/BLD/bin/nemo.exe"
