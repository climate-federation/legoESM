#!/usr/bin/env bash
# Operator-executed rank-complete ORCA2 rung-0 kt=8 external-substep record.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-166 SPG8 acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout|--plant-content) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout|--plant-content]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R96SPG
readonly TARGET_CFG=ORCA2_OMIP_L4_R166SPG8
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round96/acquisition/orca2_rung0_spgts_ranked_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round166/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_spgts_kt8_ranked_10step_np2
readonly SOURCE_DYNSPG_SHA=7182eb82238833842dbd287486dfa68cf7120941a2e985dd0d0d8a3dccf2f3be
readonly SOURCE_WRITER_SHA=7e9a2c2af4bd33d1ed3a60a35e1d1d58c0d785aac9700f82b9f08c68b3976971
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=7a65083c8f9a625a394844c9bfb0fcb927d666f34136a010de33f28685bab7f6
readonly SOURCE_NML_SHA=5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8
readonly DECK_MANIFEST_SHA=0e40688deddd7a22f8a6a7105ebc623c80e335d7bea5d88a80702bf447148312
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly WRITER=$here/l4_r95_spgts_frames.F90
readonly GATE=$here/../nemo_testcase_l4_orca2_round95_spgts_acquisition/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round166.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
for path in "$0" "$WRITER" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
pin "$SOURCE_DYNSPG_SHA" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" 'source dynspg_ts'
pin "$SOURCE_WRITER_SHA" "$SOURCE_ROOT/MY_SRC/l4_r95_spgts_frames.F90" 'source writer'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$SOURCE_NML_SHA" "$SOURCE_RUN/namelist_cfg" 'source namelist'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$EVIDENCE"
bash -n "$0"
check_layout() {
  local source=$1
  [[ "$(grep -Fc 'CALL r95_spg_open' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r95_spg_w2('ssha_e'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r95_spg_w2('hur_e'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r95_spg_close' "$source")" -eq 1 ]]
}
check_layout "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" || {
  printf 'REFUSE: source SPG writer layout is incomplete\n' >&2; exit 66;
}
grep -Fq 'IF(kt /= nit000 + 7) RETURN' "$WRITER" || {
  printf 'REFUSE: writer does not select kt=8\n' >&2; exit 66;
}
grep -Fq 'oracle_r166_spg_rank' "$WRITER" || {
  printf 'REFUSE: writer lacks per-rank target name\n' >&2; exit 66;
}
if [[ "$MODE" == --plant-layout ]]; then
  printf 'STATUS PLANT-FIRED layout\n'; exit 69
fi

dry=$(mktemp -d /tmp/orca2-r166-spg8.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$WRITER" -o "$dry/l4_r95_spgts_frames.f90"
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$dry" \
  "$dry/l4_r95_spgts_frames.f90" -o "$dry/l4_r95_spgts_frames.o"
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" -o "$dry/dynspg_ts.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$dry" \
  -I "$SOURCE_ROOT/BLD/inc" -J "$dry" "$dry/dynspg_ts.f90"
printf 'SYNTAX_PROOF_PASS round166 writer and inherited dynspg_ts\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND166_SPG8_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

verify_manifest() {
  local manifest=$1 path name count digest
  [[ -f "$manifest" ]] || { printf 'REFUSE: missing producer content manifest\n' >&2; exit 70; }
  for path in "$0" "$WRITER" "$GATE" "$PREREG"; do
    name=$(basename "$path")
    count=$(awk -v name="$name" '{n=split($2,p,"/"); if(p[n]==name)c++} END{print c+0}' "$manifest")
    [[ "$count" -eq 1 ]] || { printf 'REFUSE: producer manifest count for %s is %s\n' "$name" "$count" >&2; exit 70; }
    digest=$(awk -v name="$name" '{n=split($2,p,"/"); if(p[n]==name)print $1}' "$manifest")
    pin "$digest" "$path" "recorded producer content $name"
  done
}

admit() {
  local plant record expected_stamp
  verify_manifest "$TARGET_RUN/toolchain.sha256"
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r166_spg_rank????_kt00000008.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank kt=8 SPG records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r166_spg_rank????_kt00000008.bin; do
    expected_stamp="$(sha256sum "$record" | awk '{print $1}') $(sha256sum "$WRITER" | awk '{print $1}') $(basename "$record")"
    [[ "$(cat "$record.stamp")" == "$expected_stamp" ]] || {
      printf 'REFUSE: record stamp moved: %s\n' "$record" >&2; exit 70;
    }
  done
  for plant in header field-name field-dims truncation missing-frame swapped-rank restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
      --expected-kt 8 --prefix oracle_r166_spg --magic NEMO_L4_R166SPG \
      --plant "$plant" >"$TARGET_RUN/round166_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round166_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
    --expected-kt 8 --prefix oracle_r166_spg --magic NEMO_L4_R166SPG \
    --output "$TARGET_RUN/round166_spg8_admission.json"
  printf 'ORCA2_ROUND166_SPG8_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --plant-content ]]; then
  plant_manifest=$dry/toolchain.sha256
  sha256sum "$0" "$WRITER" "$GATE" "$PREREG" >"$plant_manifest"
  sed -i '1s/^[0-9a-f]/z/' "$plant_manifest"
  if (verify_manifest "$plant_manifest") >/dev/null 2>&1; then
    printf 'REFUSE: content plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED content\n'; exit 69
fi
if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_RUN/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r95_spgts_frames.F90"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || {
  printf 'REFUSE: compiled writer layout moved\n' >&2; exit 69;
}

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$TARGET_RUN/"
cp "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" "$TARGET_RUN/compiled_dynspg_ts.f90"
sha256sum "$0" "$WRITER" "$GATE" "$PREREG" \
  "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" >"$TARGET_RUN/toolchain.sha256"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
(
  cd "$TARGET_RUN"
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]] || {
    printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 69;
  }
  grep -Fxq 'STOP 0' run.user.stdout.log || {
    printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69;
  }
  for record in oracle_r166_spg_rank????_kt00000008.bin; do
    printf '%s %s %s\n' "$(sha256sum "$record" | awk '{print $1}')" \
      "$(sha256sum "$WRITER" | awk '{print $1}')" "$record" >"$record.stamp"
  done
)
admit
