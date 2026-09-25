#!/usr/bin/env bash
# ORCA2 round-17 acquisition: direct per-rank barotropic halo operands.
# USER-EXECUTED ONLY. The default --run mode invokes makenemo and mpirun.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-17 ORCA2 halo acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --preflight-only|--run) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R17HALO
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round17/acquisition
readonly RUN_A=$EVIDENCE/orca1ice_bt_halo_ranked_np2
readonly EXPECTED_SOURCE_SHA256=c1542a517a40627aeea099c0955728cda85eb49b3ec5aa9d98068f371dbcb74c
readonly EXPECTED_CPP_SHA256=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly EXPECTED_BASELINE_BINARY_SHA256=0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630
readonly EXPECTED_DECK_MANIFEST_SHA256=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly EXPECTED_PATCH_SHA256=6494994c2fed29c94ec47aaff2d790255889e62a73faa3f01490452d1265ace9
readonly RECORD_BYTES=$((16 + 7 * 4 + 2 * (4 + 5 * 94 * 152 * 8)))

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly PATCH=$here/halo_writer.patch

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

preflight() {
  local dry
  pin "$EXPECTED_PATCH_SHA256" "$PATCH" 'halo writer patch'
  pin "$EXPECTED_SOURCE_SHA256" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" 'source dynspg_ts'
  pin "$EXPECTED_CPP_SHA256" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
  pin "$EXPECTED_BASELINE_BINARY_SHA256" "$SOURCE_RUN/nemo" 'baseline binary'
  pin "$EXPECTED_DECK_MANIFEST_SHA256" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
  pin "$EXPECTED_INPUT_MANIFEST_SHA256" "$SOURCE_RUN/input_files.sha256" 'input manifest'
  dry=$(mktemp -d /tmp/orca2-r17-halo.XXXXXX)
  cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
  (cd "$dry" && patch -p0 --dry-run <"$PATCH" >/dev/null) || {
    printf 'REFUSE: halo writer patch no longer applies to pinned source\n' >&2
    exit 66
  }
  command -v makenemo >/dev/null 2>&1 || [[ -x "$NEMO_ROOT/makenemo" ]] || {
    printf 'REFUSE: makenemo is unavailable\n' >&2; exit 67;
  }
  command -v mpirun >/dev/null 2>&1 || {
    printf 'REFUSE: mpirun is unavailable\n' >&2; exit 67;
  }
  printf 'ORCA2_ROUND17_HALO_PREFLIGHT_READY %s\n' "$RUN_A"
}

stage_run() {
  local digest name
  mkdir "$RUN_A"
  while read -r digest name; do
    [[ -f "$SOURCE_RUN/$name" || -L "$SOURCE_RUN/$name" ]] || {
      printf 'REFUSE: deck manifest source absent: %s\n' "$name" >&2; exit 65;
    }
    cp -a "$SOURCE_RUN/$name" "$RUN_A/$name"
  done <"$SOURCE_RUN/deck_files.sha256"
  while read -r digest name; do
    [[ -f "$SOURCE_RUN/$name" || -L "$SOURCE_RUN/$name" ]] || {
      printf 'REFUSE: input manifest source absent: %s\n' "$name" >&2; exit 65;
    }
    cp -a "$SOURCE_RUN/$name" "$RUN_A/$name"
  done <"$SOURCE_RUN/input_files.sha256"
  cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$RUN_A/"
  cp "$BINARY" "$RUN_A/nemo"
  cp "$COMPILED" "$RUN_A/compiled_dynspg_ts.f90"
  (cd "$RUN_A" && sha256sum -c deck_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged deck differs\n' >&2; exit 68;
  }
  (cd "$RUN_A" && sha256sum -c input_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged inputs differ\n' >&2; exit 68;
  }
  sha256sum "$RUN_A/nemo" >"$RUN_A/binary.sha256"
  sha256sum "$RUN_A/compiled_dynspg_ts.f90" >"$RUN_A/compiled_source.sha256"
  sha256sum "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$PATCH" >"$RUN_A/acquisition_sources.sha256"
}

run_one() {
  local mpi_rc tee_rc
  (
    cd "$RUN_A"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    mpi_rc=${pipe_rc[0]}
    tee_rc=${pipe_rc[1]:-0}
    printf 'MPIRUN_RC=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
    [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || {
      printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2
      exit 68
    }
    printf 'RUN DONE\n' >>run.user.time.log
  )
}

admit() {
  local rank file got
  for rank in 0000 0001; do
    file=$RUN_A/oracle_bt_halo_operands_kt00000001_r${rank}.bin
    got=$(stat -c %s "$file" 2>/dev/null || echo 0)
    [[ "$got" == "$RECORD_BYTES" ]] || {
      printf 'REFUSE: rank %s halo stream is %s bytes, expected %s\n' \
        "$rank" "$got" "$RECORD_BYTES" >&2
      exit 69
    }
  done
  [[ "$(grep -c 'LANE4_BT_HALO_DUMP' "$RUN_A/ocean.output")" -eq 2 ]] || {
    printf 'REFUSE: ocean.output does not carry two ranked halo dump markers\n' >&2
    exit 69
  }
  (cd "$RUN_A" && sha256sum oracle_bt_halo_operands_kt00000001_r*.bin >round17_halo_outputs.sha256)
  printf 'ORCA2_ROUND17_HALO_ACQUISITION_PASS %s\n' "$RUN_A"
}

preflight
[[ "$MODE" == --run ]] || exit 0

[[ ! -e "$TARGET_ROOT" && ! -e "$RUN_A" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 64;
}
[[ -d "$(dirname "$EVIDENCE")" && ! -L "$(dirname "$EVIDENCE")" ]] || {
  printf 'REFUSE: round-17 evidence parent is absent or a symlink\n' >&2; exit 64;
}
mkdir -p "$EVIDENCE"
[[ ! -L "$EVIDENCE" ]] || { printf 'REFUSE: evidence directory is a symlink\n' >&2; exit 64; }

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not create isolated target\n' >&2; exit 68;
}
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
(cd "$TARGET_ROOT/MY_SRC" && patch -p0 <"$PATCH") || {
  printf 'REFUSE: halo writer patch failed\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not build isolated target\n' >&2; exit 68;
}
[[ -x "$BINARY" ]] || { printf 'REFUSE: target build produced no executable\n' >&2; exit 68; }
for marker in NEMO_L4_BTHALO1 oracle_bt_halo_operands_kt LANE4_BT_HALO_DUMP; do
  grep -Fq "$marker" "$COMPILED" || {
    printf 'REFUSE: compiled source lacks marker: %s\n' "$marker" >&2; exit 68;
  }
done
grep -Fq "WRITE(l4_halo_unit) zhU, e2u, ua_e, zhup2_e, ssumask" "$COMPILED" || {
  printf 'REFUSE: compiled source lacks direct halo operand write\n' >&2; exit 68;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in acquisition binary\n' >&2; exit 68
fi

stage_run
run_one
admit
