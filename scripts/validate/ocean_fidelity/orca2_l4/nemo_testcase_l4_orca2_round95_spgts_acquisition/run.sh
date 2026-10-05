#!/usr/bin/env bash
# Operator-executed ORCA2 hierarchy rung-0 split-explicit substep acquisition.
# Round 96 repairs round 95's deck-staging provenance defect while retaining
# the already preregistered self-describing R95SPG record format.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-96 SPG acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout|--plant-source-deck|--plant-toolchain) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout|--plant-source-deck|--plant-toolchain]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R93SLOW
readonly TARGET_CFG=ORCA2_OMIP_L4_R96SPG
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round93/acquisition/orca2_rung0_slow_ranked_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round96/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_spgts_ranked_10step_np2
readonly SOURCE_DYNSPG_SHA=c1542a517a40627aeea099c0955728cda85eb49b3ec5aa9d98068f371dbcb74c
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=1c8c4a2df513e9b035889c9861571639c94a7161d46a05e2d8e4d12959c21809
readonly SOURCE_NML_SHA=d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c
readonly HARMONIZED_NML_SHA=5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8
readonly DECK_MANIFEST_SHA=92f2a73eeb3b9989b6f390519e3f0cab900b122194259f2988cad672e1819677
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/dynspg_ts_round95.patch
readonly DECISION83_PATCH=$here/namelist_rung0_decision83.patch
readonly WRITER=$here/l4_r95_spgts_frames.F90
readonly GATE=$here/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round96.md

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
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$PATCH" "$DECISION83_PATCH" "$WRITER" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$SOURCE_RUN/ocean.output"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_DYNSPG_SHA" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" 'source dynspg_ts'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$SOURCE_NML_SHA" "$SOURCE_RUN/namelist_cfg" 'source rung-0 namelist'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Eq 'Vector form: 2nd order centered scheme.*ln_dynadv_vec *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not resolve vector-invariant C2\n' >&2; exit 65;
}
grep -Eq 'Free surface with time splitting.*ln_dynspg_ts *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not resolve split-explicit free surface\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE"
bash -n "$0"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66; }
dry=$(mktemp -d /tmp/orca2-r95-spgts.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
cp "$WRITER" "$dry/l4_r95_spgts_frames.F90"
cp "$SOURCE_RUN/namelist_cfg" "$dry/namelist_cfg"
if [[ "$MODE" == --plant-source-deck ]]; then
  cp "$SOURCE_ROOT/EXP00/namelist_cfg" "$dry/namelist_cfg"
  if [[ "$(sha256sum "$dry/namelist_cfg" | awk '{print $1}')" == "$SOURCE_NML_SHA" ]]; then
    printf 'REFUSE: source-deck plant stayed green\n' >&2
    exit 69
  fi
  printf 'STATUS PLANT-FIRED source-deck\n'
  exit 69
fi
patch -s --fuzz=0 -p0 -d "$dry" <"$PATCH"
patch -s --fuzz=0 -p0 -d "$dry" <"$DECISION83_PATCH"
pin "$HARMONIZED_NML_SHA" "$dry/namelist_cfg" 'Decision-83 rung-0 namelist'

check_layout() {
  local source=$1
  [[ "$(grep -Fc 'CALL r95_spg_open' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r95_spg_w2('ssha_e'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r95_spg_w2('cor_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r95_spg_w2('trd_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r95_spg_w2('ua_new'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r95_spg_close' "$source")" -eq 1 ]]
}
check_layout "$dry/dynspg_ts.F90" || { printf 'REFUSE: SPG writer layout is incomplete\n' >&2; exit 66; }
if [[ "$MODE" == --plant-layout ]]; then
  sed -i "/CALL r95_spg_w2('cor_u'/d" "$dry/dynspg_ts.F90"
  if check_layout "$dry/dynspg_ts.F90"; then printf 'REFUSE: layout plant stayed green\n' >&2; exit 69; fi
  printf 'STATUS PLANT-FIRED layout\n'
  exit 69
fi

cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/l4_r95_spgts_frames.F90" -o "$dry/l4_r95_spgts_frames.f90"
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$dry" \
  "$dry/l4_r95_spgts_frames.f90" -o "$dry/l4_r95_spgts_frames.o"
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/dynspg_ts.F90" -o "$dry/dynspg_ts.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$dry" -I "$SOURCE_ROOT/BLD/inc" -J "$dry" "$dry/dynspg_ts.f90"
printf 'SYNTAX_PROOF_PASS l4_r95_spgts_frames.f90 dynspg_ts.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND96_RUNG0_SPGTS_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

verify_recorded_tools() {
  local recorded=$1 manifest=${2:-$TARGET_RUN/toolchain.sha256} path name matches recorded_digest
  [[ "$recorded" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'REFUSE: malformed producer token %s\n' "$recorded" >&2; exit 70;
  }
  [[ -f "$manifest" && "$(wc -l <"$manifest")" -eq 6 ]] || {
    printf 'REFUSE: producer content manifest is missing or has wrong cardinality\n' >&2; exit 70;
  }
  for path in "$PATCH" "$DECISION83_PATCH" "$WRITER" "$GATE" "$PREREG" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90"; do
    name=$(basename "$path")
    matches=$(awk -v name="$name" '
      { n = split($2, parts, "/"); if (parts[n] == name) count++ }
      END { print count + 0 }
    ' "$manifest")
    [[ "$matches" -eq 1 ]] || {
      printf 'REFUSE: producer content manifest has %s entries for %s\n' "$matches" "$name" >&2; exit 70;
    }
    recorded_digest=$(awk -v name="$name" '
      { n = split($2, parts, "/"); if (parts[n] == name) print $1 }
    ' "$manifest")
    [[ "$recorded_digest" =~ ^[0-9a-f]{64}$ ]] || {
      printf 'REFUSE: malformed producer content digest for %s\n' "$name" >&2; exit 70;
    }
    pin "$recorded_digest" "$path" "recorded producer content $name"
  done
}

if [[ "$MODE" == --plant-toolchain ]]; then
  plant_manifest=$dry/toolchain.sha256
  cp "$TARGET_RUN/toolchain.sha256" "$plant_manifest"
  sed -i '1s/^[0-9a-f]/z/' "$plant_manifest"
  if (verify_recorded_tools "$COMMIT" "$plant_manifest") >/dev/null 2>&1; then
    printf 'REFUSE: producer-content plant stayed green\n' >&2
    exit 69
  fi
  printf 'STATUS PLANT-FIRED toolchain\n'
  exit 69
fi

admit() {
  local plant record expected_stamp recorded
  recorded=$(cat "$TARGET_RUN/producer_commit.txt")
  verify_recorded_tools "$recorded"
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r95_spg_rank????_kt00000001.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank SPG records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r95_spg_rank????_kt00000001.bin; do
    expected_stamp="$(sha256sum "$record" | awk '{print $1}') $recorded $(basename "$record")"
    [[ "$(cat "$record.stamp")" == "$expected_stamp" ]] || {
      printf 'REFUSE: record stamp moved: %s\n' "$record" >&2; exit 70;
    }
  done
  for plant in header field-name field-dims truncation missing-frame swapped-rank restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" --plant "$plant" \
      >"$TARGET_RUN/round96_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round96_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
    --output "$TARGET_RUN/round96_spgts_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r95_spg_rank*.bin oracle_r95_spg_rank*.bin.stamp \
    ORCA2_000000??_restart_????.nc round96_*_plant.log round96_spgts_admission.json \
    >round96_outputs.sha256)
  printf 'ORCA2_ROUND96_RUNG0_SPGTS_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_ROOT/BLD/bin/nemo.exe" && -x "$TARGET_RUN/nemo" ]] || {
    printf 'REFUSE: existing target lacks binary\n' >&2; exit 68;
  }
  cmp -s "$TARGET_ROOT/BLD/bin/nemo.exe" "$TARGET_RUN/nemo" || {
    printf 'REFUSE: staged and built binaries differ\n' >&2; exit 68;
  }
  check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || {
    printf 'REFUSE: compiled writer layout moved\n' >&2; exit 68;
  }
  pin "$HARMONIZED_NML_SHA" "$TARGET_RUN/namelist_cfg" 'existing Decision-83 namelist'
  (cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
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

manifest=$(mktemp -d /tmp/orca2-r95-spgts-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PATCH" "$DECISION83_PATCH" "$WRITER" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
# The config directory is only a source-code container.  Its EXP00 namelist is
# the shipped deck, not the admitted rung-0 run deck.  Round 95 patched that
# stale copy and refused.  Stage and pin the actual admitted deck first.
cp "$SOURCE_RUN/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
pin "$SOURCE_NML_SHA" "$TARGET_ROOT/EXP00/namelist_cfg" 'staged source rung-0 namelist'
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/EXP00" <"$DECISION83_PATCH"
pin "$HARMONIZED_NML_SHA" "$TARGET_ROOT/EXP00/namelist_cfg" 'built Decision-83 namelist'
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r95_spgts_frames.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cmp -s "$dry/dynspg_ts.F90" "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" || {
  printf 'REFUSE: target source differs from syntax-proved source\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || { printf 'REFUSE: compiled writer layout is incomplete\n' >&2; exit 69; }
if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector-math symbol present\n' >&2; exit 69; fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$TARGET_RUN/source_deck_files.sha256"
cp "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
(cd "$TARGET_RUN" && sha256sum -c source_deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
patch -s --fuzz=0 -p0 -d "$TARGET_RUN" <"$DECISION83_PATCH"
pin "$HARMONIZED_NML_SHA" "$TARGET_RUN/namelist_cfg" 'run Decision-83 namelist'
(cd "$TARGET_RUN" && while read -r _ name; do sha256sum "$name"; done <source_deck_files.sha256 >deck_files.sha256)
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" "$TARGET_RUN/compiled_dynspg_ts.f90"
cp "$manifest"/* "$TARGET_RUN/"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}; tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nWALL_SECONDS=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || { printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2; exit 69; }
  grep -Fxq 'STOP 0' run.user.stdout.log || { printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69; }
  for record in oracle_r95_spg_rank????_kt00000001.bin; do
    printf '%s %s %s\n' "$(sha256sum "$record" | awk '{print $1}')" "$COMMIT" "$record" >"$record.stamp"
  done
  printf 'RUN DONE\n' >>run.user.time.log
)
admit
