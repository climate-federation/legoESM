#!/usr/bin/env bash
set -Eeuo pipefail

# Operator-run only: one passive, post-call replay record at developed step 1080.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round186 qsr acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_YRPERT
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R186QSRWALK
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/nemo_seed0
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round186/oracle_qsr_walk
readonly RESTART_0180=GYRE_OMIP_L2_P3_00000180_restart.nc
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly SHA_0180=853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6
readonly SHA_1080=6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976
readonly SOURCE_BINARY_SHA=578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a
readonly RECORD_BYTES=1174060

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly STP_PATCH=$here/stprk3_stg_round186.patch
readonly QSR_PATCH=$here/traqsr_round186.patch
readonly NML_PATCH=$here/namelist_cfg_round186.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round186_qsr_walk.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round186.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$STP_PATCH" "$QSR_PATCH" "$NML_PATCH" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$SOURCE_ROOT/WORK/traqsr.F90" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
  "$SOURCE_RUN/$RESTART_0180" "$SOURCE_RUN/$RESTART_1080"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing source-card input %s\n' "$path" >&2
    exit 64
  fi
done
if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: new target already exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
fi
digest=$(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}')
if [[ "$digest" != "$SOURCE_BINARY_SHA" ]] || ! cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo"; then
  printf 'REFUSE: frozen source binary identity failed: %s\n' "$digest" >&2
  exit 65
fi
for pair in "$RESTART_0180:$SHA_0180" "$RESTART_1080:$SHA_1080"; do
  name=${pair%%:*}; expected=${pair#*:}
  digest=$(sha256sum "$SOURCE_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]]; then
    printf 'REFUSE: source restart %s is %s, expected %s\n' "$name" "$digest" "$expected" >&2
    exit 65
  fi
done
for pattern in 'number of the last time step.*nn_itend *= *2160' \
  'ocean time step.*rn_Dt *= *14400' \
  'Light penetration in temperature Eq.*ln_traqsr *= *T' \
  '2 band.*light penetration.*ln_qsr_2bd *= *T'; do
  if ! grep -Eq "$pattern" "$SOURCE_RUN/ocean.output"; then
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  fi
done
for patch_file in "$STP_PATCH" "$QSR_PATCH"; do
  removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$patch_file")
  if [[ "$removed" -ne 0 ]]; then
    printf 'REFUSE: additive writer patch removes %s lines: %s\n' "$removed" "$patch_file" >&2
    exit 66
  fi
done

dry=$(mktemp -d /tmp/gyre-r186-qsr-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
cp "$SOURCE_ROOT/WORK/traqsr.F90" "$dry/traqsr.F90"
cp "$SOURCE_RUN/namelist_cfg" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/stprk3_stg.F90" <"$STP_PATCH"
patch -s --fuzz=0 "$dry/traqsr.F90" <"$QSR_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$NML_PATCH"
for needle in 'CALL tra_qsr' 'CALL qsr_2BD_l2_replay' \
  "r186_magic = 'NEMO_L2_R186QSR1'" 'gdepw_1d, qsr, r3t(:,:,Kmm)' \
  'ts(:,:,:,jp_tem,Krhs)-r186_qsr_before(:,:,:)'; do
  if ! grep -Fq "$needle" "$dry/stprk3_stg.F90"; then
    printf 'REFUSE: dry stage writer lacks %s\n' "$needle" >&2
    exit 66
  fi
done
for needle in 'PUBLIC   qsr_2BD_l2_replay' 'SUBROUTINE qsr_2BD_l2_replay' \
  'pinc(ji,jj,jk) = ( pbefore(ji,jj,jk) + qsr(ji,jj)' \
  'DO jk = nk0+1, nkV'; do
  if ! grep -Fq "$needle" "$dry/traqsr.F90"; then
    printf 'REFUSE: dry qsr replay lacks %s\n' "$needle" >&2
    exit 66
  fi
done
if ! grep -Eq '^[[:space:]]*nn_itend *= *1080' "$dry/namelist_cfg"; then
  printf 'REFUSE: dry namelist does not end at step 1080\n' >&2
  exit 66
fi

syntax=$(mktemp -d /tmp/gyre-r186-qsr-syntax.XXXXXX)
printf 'temporary syntax-proof directory (retained): %s\n' "$syntax"
for source in traqsr stprk3_stg; do
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
    -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$dry/$source.F90" -o "$syntax/$source.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
    -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/$source.f90"
done
printf 'SYNTAX_PROOF_PASS traqsr.f90 stprk3_stg.f90\n'

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has under 2 GB free\n' "$mount" >&2
    exit 67
  fi
done

manifest=$(mktemp -d /tmp/gyre-r186-qsr-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$STP_PATCH" "$QSR_PATCH" "$NML_PATCH" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/traqsr.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/WORK/traqsr.F90" "$TARGET_ROOT/MY_SRC/traqsr.F90"
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STP_PATCH"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/traqsr.F90" <"$QSR_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
if [[ ! -x "$BINARY" ]]; then
  printf 'REFUSE: target binary is missing\n' >&2
  exit 68
fi
for pair in "stprk3_stg.f90:CALL qsr_2BD_l2_replay" \
            "traqsr.f90:SUBROUTINE qsr_2BD_l2_replay"; do
  file=${pair%%:*}; needle=${pair#*:}
  if ! grep -Fq "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file"; then
    printf 'REFUSE: compiled %s lacks %s\n' "$file" "$needle" >&2
    exit 68
  fi
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in acquisition binary\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  if [[ ! -f "$SOURCE_RUN/$name" ]]; then
    printf 'REFUSE: missing prepared input %s\n' "$name" >&2
    exit 68
  fi
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
patch -s --fuzz=0 "$TARGET_RUN/namelist_cfg" <"$NML_PATCH"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"

(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  SECONDS=0
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    printf 'REFUSE: NEMO acquisition process failed\n' >&2
    exit 69
  fi
  if ! grep -Fxq 'STOP 0' run.user.stdout.log; then
    printf 'REFUSE: NEMO run did not terminate with STOP 0\n' >&2
    exit 69
  fi
  printf 'wall_seconds=%s\nNEMO_DONE\n' "$SECONDS" >>run.user.time.log
)

record=$TARGET_RUN/oracle_qsr_walk_kt00001080.bin
if [[ ! -f "$record" ]]; then
  printf 'REFUSE: NEMO omitted the step-1080 qsr record\n' >&2
  exit 70
fi
bytes=$(stat -c %s "$record")
if [[ "$bytes" -ne "$RECORD_BYTES" ]]; then
  printf 'REFUSE: qsr record has %s bytes, expected %s\n' "$bytes" "$RECORD_BYTES" >&2
  exit 70
fi
for pair in "$RESTART_0180:$SHA_0180" "$RESTART_1080:$SHA_1080"; do
  name=${pair%%:*}; expected=${pair#*:}
  if [[ ! -f "$TARGET_RUN/$name" ]]; then
    printf 'REFUSE: target run omitted passive-control restart %s\n' "$name" >&2
    exit 71
  fi
  digest=$(sha256sum "$TARGET_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]] || ! cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; then
    printf 'REFUSE: instrument perturbed restart %s: %s, expected %s\n' "$name" "$digest" "$expected" >&2
    exit 71
  fi
done
digest=$(sha256sum "$record" | awk '{print $1}')
printf '%s %s %s\n' "$digest" "$COMMIT" "$(basename "$record")" >"$TARGET_RUN/qsr_record.stamp"

"$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
  --json "$TARGET_RUN/round186_qsr_admission.json"
if "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
     --plant actual-increment-ulp >"$TARGET_RUN/round186_qsr_plant.log" 2>&1; then
  printf 'REFUSE: qsr ULP plant stayed green\n' >&2
  exit 72
fi
if ! grep -Fq 'STATUS PLANT-FIRED: actual-increment-ulp' "$TARGET_RUN/round186_qsr_plant.log"; then
  printf 'REFUSE: qsr ULP plant exited without its marker\n' >&2
  exit 72
fi
sha256sum "$record" "$TARGET_RUN/qsr_record.stamp" \
  "$TARGET_RUN/round186_qsr_admission.json" "$TARGET_RUN/round186_qsr_plant.log" \
  >"$TARGET_RUN/round186_outputs.sha256"
printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$TARGET_RUN/run.user.time.log"
printf 'ROUND186_QSR_RECORD_READY %s\n' "$TARGET_RUN"
