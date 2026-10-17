#!/usr/bin/env bash
set -Eeuo pipefail

do_run=0
[[ $# -eq 0 ]] || [[ $# -eq 1 && $1 == --run ]] || {
  printf 'REFUSE_USAGE: expected [--run]\n' >&2; exit 64; }
[[ $# -eq 0 ]] || do_run=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$here/deck.sh"
gyre_hires_select_deck r4_smag "$here"
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/GYRE_OMIP_L2_P3_AHM
readonly WRITER=$SOURCE_CFG/BLD/bin/nemo.exe
readonly BASE_EXP00=$NEMO_ROOT/cfgs/GYRE_OMIP_L2_P3/EXP00
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round4/oracle_gyre_r4_smag_ahm}
readonly RANKS=${RANKS:-1}
[[ $RANKS == 1 ]] || {
  printf 'REFUSE_RANKS: AHM dumps are guarded by lwp and are not gathered; RANKS must be 1\n' >&2
  exit 64
}

repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)
for path in "$BASE_DECK" "$DECK_PATCH" "$WRITER" "$BASE_EXP00/namelist_ref" \
  "$SOURCE_CFG/MY_SRC/stprk3.F90" "$SOURCE_CFG/MY_SRC/stprk3_stg.F90"; do
  [[ -f $path ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == b2b73a7899dfd06c9eaf8106d0c332d0d51f93a3e8bad1c44b2d58353caa6a9a \
   && $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3_stg.F90" | awk '{print $1}') == f65546e500ccd0b4d84e9641eeeb11d7e9160fe36a1339c1a2426425322d07e7 ]] || {
  printf 'REFUSE_WRITER_SOURCE: AHM sources differ\n' >&2; exit 68; }
for token in NEMO_L4_AHMTF__1 NEMO_L4_DYNLDF_1; do
  grep -q "$token" "$SOURCE_CFG/BLD/ppsrc/nemo/stprk3.f90" \
    "$SOURCE_CFG/BLD/ppsrc/nemo/stprk3_stg.f90" || {
    printf 'REFUSE_COMPILED_WRITER: missing %s\n' "$token" >&2; exit 68; }
done

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$BASE_DECK" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK_PATCH"
gyre_hires_verify_deck r4_smag "$dry/namelist_cfg"
writer_hash=$(sha256sum "$WRITER" | awk '{print $1}')
printf 'PREFLIGHT: deck=r4_smag writer=GYRE_OMIP_L2_P3_AHM evidence=%s writer_sha256=%s commit=%s\n' \
  "$EVIDENCE" "$writer_hash" "$COMMIT"
if [[ $do_run -ne 1 ]]; then
  printf 'WOULD_RUN: EVIDENCE=%s RANKS=%s %s --run\nPREFLIGHT_OK\n' "$EVIDENCE" "$RANKS" "$0"
  exit 0
fi

[[ ! -e $EVIDENCE ]] || { printf 'REFUSE_EVIDENCE_EXISTS: %s\n' "$EVIDENCE" >&2; exit 69; }
mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/run"
cp "$dry/namelist_cfg" "$BASE_EXP00/namelist_ref" "$EVIDENCE/run/"
for xml in "$BASE_EXP00"/*.xml; do [[ ! -e $xml ]] || cp "$xml" "$EVIDENCE/run/"; done
cp "$WRITER" "$EVIDENCE/run/nemo"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
gyre_hires_write_resolved_namelist "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/manifest/resolved_namelist_lines.txt"
sha256sum "$EVIDENCE/run/nemo" "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/run/namelist_ref" >"$EVIDENCE/manifest/run.sha256"
(
  cd "$EVIDENCE/run"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np "$RANKS" --oversubscribe ./nemo >run.user.log 2>&1
)
for kt in 2 3; do
  for kind in ahm dynldf; do
    record=$(printf '%s/run/oracle_%s_kt%08d.bin' "$EVIDENCE" "$kind" "$kt")
    [[ -s $record ]] || { printf 'REFUSE_MISSING_RECORD: %s\n' "$record" >&2; exit 70; }
  done
done
printf 'ACQUISITION_DONE: %s (r4_smag AHM/dynldf discriminator, kt=2..3)\n' "$EVIDENCE"
