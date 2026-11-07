#!/usr/bin/env bash
# ORCA2 round-20 acquisition: the complete slow-forcing construction on both ranks.
# USER-EXECUTED ONLY. The default --run mode invokes makenemo and mpirun.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-20 ORCA2 ranked slow-forcing acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --preflight-only|--admit-existing|--run) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--admit-existing|--run]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R19PREX
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round19/acquisition/orca1ice_u_preexchange_ranked_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round20/acquisition
readonly RUN_A=$EVIDENCE/orca1ice_slow_forcing_ranked_np2
readonly EXPECTED_SOURCE_SHA256=552bda13b973699cf7c383d659dfc476271f4167b7a9df2eada6961b23581763
readonly EXPECTED_CPP_SHA256=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly EXPECTED_BASELINE_BINARY_SHA256=5a424f9af5c7b7a492f9d9a577056ad77cc142b1922be1aa82399b8d658c8112
readonly EXPECTED_DECK_MANIFEST_SHA256=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly EXPECTED_PATCH_SHA256=942d3ef35633148dd6e076245dac1e2940df0b2783900637cd4ff042ce496f07
readonly RECORD_BYTES=22814424

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly PATCH=$here/ranked_slow_forcing_writer.patch

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

preflight() {
  pin "$EXPECTED_PATCH_SHA256" "$PATCH" 'ranked slow-forcing writer patch'
  pin "$EXPECTED_SOURCE_SHA256" "$SOURCE_ROOT/MY_SRC/stp2d.F90" 'source stp2d'
  pin "$EXPECTED_CPP_SHA256" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
  pin "$EXPECTED_BASELINE_BINARY_SHA256" "$SOURCE_RUN/nemo" 'baseline binary'
  pin "$EXPECTED_DECK_MANIFEST_SHA256" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
  pin "$EXPECTED_INPUT_MANIFEST_SHA256" "$SOURCE_RUN/input_files.sha256" 'input manifest'
  patch --dry-run --fuzz=0 -p0 -d "$SOURCE_ROOT/MY_SRC" <"$PATCH" >/dev/null || {
    printf 'REFUSE: ranked slow-forcing writer patch no longer applies exactly\n' >&2
    exit 66
  }
  command -v makenemo >/dev/null 2>&1 || [[ -x "$NEMO_ROOT/makenemo" ]] || {
    printf 'REFUSE: makenemo is unavailable\n' >&2; exit 67;
  }
  command -v mpirun >/dev/null 2>&1 || {
    printf 'REFUSE: mpirun is unavailable\n' >&2; exit 67;
  }
  printf 'ORCA2_ROUND20_SLOW_RANK_PREFLIGHT_READY %s\n' "$RUN_A"
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
  cp "$COMPILED" "$RUN_A/compiled_stp2d.f90"
  (cd "$RUN_A" && sha256sum -c deck_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged deck differs\n' >&2; exit 68;
  }
  (cd "$RUN_A" && sha256sum -c input_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged inputs differ\n' >&2; exit 68;
  }
  sha256sum "$RUN_A/nemo" >"$RUN_A/binary.sha256"
  sha256sum "$RUN_A/compiled_stp2d.f90" >"$RUN_A/compiled_source.sha256"
  sha256sum "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$PATCH" >"$RUN_A/acquisition_sources.sha256"
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

admit_rank() {
  local file=$1 rank=$2
  /usr/bin/python3 - "$file" "$rank" "$RECORD_BYTES" <<'PY'
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
rank = int(sys.argv[2])
expected_bytes = int(sys.argv[3])
if not path.is_file() or path.stat().st_size != expected_bytes:
    raise SystemExit(f"REFUSE: rank {rank} stream has wrong or absent byte count: {path}")
with path.open("rb") as stream:
    magic = stream.read(16).decode("ascii").rstrip()
    header = struct.unpack("=9i", stream.read(36))
expected = (1, 1, 1, 3, rank, 94, 152, 31, 64)
if magic != "NEMO_L4_SLOW_R1" or header != expected:
    raise SystemExit(f"REFUSE: rank {rank} slow-forcing header changed: {magic!r} {header}")
PY
}

admit() {
  local rank file
  for rank in 0000 0001; do
    file=$RUN_A/oracle_slow_forcing_ranked_kt00000001_r${rank}.bin
    admit_rank "$file" "$((10#$rank))"
  done
  [[ "$(sha256sum "$RUN_A"/oracle_slow_forcing_ranked_kt00000001_r*.bin | awk '{print $1}' | sort -u | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: ranked slow-forcing streams are byte-identical\n' >&2; exit 69;
  }
  grep -Fxq 'MPIRUN_RC=0' "$RUN_A/run.user.time.log" || {
    printf 'REFUSE: completed run does not carry MPIRUN_RC=0\n' >&2; exit 69;
  }
  grep -Fxq 'RUN DONE' "$RUN_A/run.user.time.log" || {
    printf 'REFUSE: completed run does not carry RUN DONE\n' >&2; exit 69;
  }
  grep -Fxq 'STOP 0' "$RUN_A/run.user.stdout.log" || {
    printf 'REFUSE: completed run does not carry STOP 0\n' >&2; exit 69;
  }
  [[ "$(grep -c 'LANE2_SLOW_FORCING_DUMP.*oracle_slow_forcing_ranked_kt00000001_r0000.bin' "$RUN_A/ocean.output")" -eq 1 ]] || {
    printf 'REFUSE: ocean.output does not carry the unique rank-0 marker\n' >&2; exit 69;
  }
  [[ "$(grep -c 'LANE2_SLOW_FORCING_DUMP.*oracle_slow_forcing_ranked_kt00000001_r0001.bin' "$RUN_A/run.user.stdout.log")" -eq 1 ]] || {
    printf 'REFUSE: captured MPI stdout does not carry the unique rank-1 marker\n' >&2; exit 69;
  }
  (cd "$RUN_A" && sha256sum oracle_slow_forcing_ranked_kt00000001_r*.bin >round20_slow_forcing_outputs.sha256)
  printf 'ORCA2_ROUND20_SLOW_RANK_ACQUISITION_PASS %s\n' "$RUN_A"
}

preflight
if [[ "$MODE" == --admit-existing ]]; then
  admit
  exit 0
fi
[[ "$MODE" == --run ]] || exit 0

[[ ! -e "$TARGET_ROOT" && ! -e "$RUN_A" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 64;
}
[[ -d "$(dirname "$EVIDENCE")" && ! -L "$(dirname "$EVIDENCE")" ]] || {
  printf 'REFUSE: round-20 evidence parent is absent or a symlink\n' >&2; exit 64;
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
(cd "$TARGET_ROOT/MY_SRC" && patch --fuzz=0 -p0 <"$PATCH") || {
  printf 'REFUSE: ranked slow-forcing writer patch failed\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not build isolated target\n' >&2; exit 68;
}
[[ -x "$BINARY" ]] || { printf 'REFUSE: target build produced no executable\n' >&2; exit 68; }
for marker in NEMO_L4_SLOW_R1 oracle_slow_forcing_ranked_kt LANE2_SLOW_FORCING_DUMP; do
  grep -Fq "$marker" "$COMPILED" || {
    printf 'REFUSE: compiled source lacks marker: %s\n' "$marker" >&2; exit 68;
  }
done
grep -Fq "WRITE(l2_slow_unit) l4_canon_3d(e3u_3d,'U')" "$COMPILED" || {
  printf 'REFUSE: compiled source lacks the direct vertical-average operand payload\n' >&2; exit 68;
}
grep -Fq "WRITE(l2_slow_unit) l4_canon_2d(Ue_rhs,'U')" "$COMPILED" || {
  printf 'REFUSE: compiled source lacks the direct slow-forcing boundary payload\n' >&2; exit 68;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in acquisition binary\n' >&2; exit 68
fi

stage_run
run_one
admit
