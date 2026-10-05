#!/usr/bin/env bash
# ORCA2 round-8 acquisition: the OTHER HALF of the tripolar fold row.
#
# WHY.  The EEN operand writer that produced every existing record is guarded
# by `lwp` and writes an unranked filename, so only rank zero's ninety owned
# longitudes were ever recorded.  Round 8 gated the vertex thickness on the
# fold row bit-exactly over those ninety -- whose fold SOURCES all lie in rank
# one's half, so the exchange across the rank boundary is exercised in one
# direction -- and left rank one's ninety DESTINATIONS unmeasured.  This
# acquisition removes the rank guard and tags each file with the rank, so the
# two halves together cover all one hundred and eighty longitudes.
#
# USER-EXECUTED ONLY.  Default mode is --run, which invokes makenemo and
# mpirun.  Every check below is fail-closed and prints a named REFUSE line.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-8 ORCA2 EEN per-rank acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --preflight-only|--run) ;;
  *)
    printf 'REFUSE: usage: %s [--preflight-only|--run]\n' "$0" >&2
    exit 64
    ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly TARGET_CFG=ORCA2_OMIP_L4_R8EENRANK
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90
readonly SOURCE_MY_SRC=/data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_5.0.2_phase2n_bbl/cfgs/ORCA2_OMIP_L4/MY_SRC
readonly SOURCE_CPP=/data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_5.0.2_phase2n_bbl/cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm
readonly BASELINE=/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round8/acquisition
readonly RUN_A=$EVIDENCE/een_per_rank_a_np2
readonly EXPECTED_PATCH_SHA256=b0c5d762b63fdb575e1795bbe0476537a57e37b6f97e65f29a7d7c7675331496
readonly EXPECTED_SOURCE_DYNSPG_SHA256=3430c945df0d8e2dbdb7fe6f23599bcf24af19f81d4c2a16555c10c7453456c1
readonly EXPECTED_BASELINE_BINARY_SHA256=8ba6120d4c54c31caaefb595d87da43ef1bb38890208b36d6ca98da71a950a29
readonly EXPECTED_DECK_MANIFEST_SHA256=e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly EXPECTED_CPP_SHA256=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
# 16-byte magic + ten 4-byte header integers + jpi*jpj*jpk doubles, per field.
readonly SINGLE_FIELD_BYTES=$((16 + 40 + 94 * 152 * 31 * 8))
readonly EIGHT_FIELD_BYTES=$((16 + 40 + 8 * 94 * 152 * 31 * 8))

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly PATCH=$here/een_per_rank_writer.patch

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || {
    printf 'REFUSE: missing pinned %s: %s\n' "$label" "$path" >&2
    exit 65
  }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2
    exit 66
  }
}

preflight() {
  local dry
  pin "$EXPECTED_PATCH_SHA256" "$PATCH" 'per-rank writer patch'
  pin "$EXPECTED_SOURCE_DYNSPG_SHA256" "$SOURCE_MY_SRC/dynspg_ts.F90" 'source writer'
  pin "$EXPECTED_CPP_SHA256" "$SOURCE_CPP" 'source cpp keys'
  pin "$EXPECTED_BASELINE_BINARY_SHA256" "$BASELINE/nemo" 'baseline binary'
  pin "$EXPECTED_DECK_MANIFEST_SHA256" "$BASELINE/deck_files.sha256" 'deck manifest'
  pin "$EXPECTED_INPUT_MANIFEST_SHA256" "$BASELINE/input_files.sha256" 'input manifest'
  dry=$(mktemp -d)
  cp "$SOURCE_MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
  (cd "$dry" && patch -p0 --dry-run <"$PATCH" >/dev/null) || {
    printf 'REFUSE: per-rank writer patch no longer applies to the pinned source\n' >&2
    rm -rf "$dry"
    exit 66
  }
  rm -rf "$dry"
  command -v makenemo >/dev/null 2>&1 || [[ -x "$NEMO_ROOT/makenemo" ]] || {
    printf 'REFUSE: makenemo is not available\n' >&2
    exit 67
  }
  command -v mpirun >/dev/null || {
    printf 'REFUSE: mpirun is not available\n' >&2
    exit 67
  }
  printf 'ORCA2_ROUND8_EEN_RANK_PREFLIGHT_READY %s\n' "$RUN_A"
}

stage_run() {
  local target=$1 digest name
  mkdir -p "$target"
  while read -r digest name; do
    [[ -f "$BASELINE/$name" || -L "$BASELINE/$name" ]] || {
      printf 'REFUSE: deck manifest source absent: %s\n' "$name" >&2
      exit 65
    }
    cp -a "$BASELINE/$name" "$target/$name"
  done <"$BASELINE/deck_files.sha256"
  while read -r digest name; do
    [[ -f "$BASELINE/$name" || -L "$BASELINE/$name" ]] || {
      printf 'REFUSE: input manifest source absent: %s\n' "$name" >&2
      exit 65
    }
    cp -a "$BASELINE/$name" "$target/$name"
  done <"$BASELINE/input_files.sha256"
  cp "$BASELINE/deck_files.sha256" "$BASELINE/input_files.sha256" "$target/"
  cp "$BINARY" "$target/nemo"
  cp "$COMPILED" "$target/compiled_dynspg_ts.f90"
  (cd "$target" && sha256sum -c deck_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged deck differs in %s\n' "$target" >&2
    exit 66
  }
  (cd "$target" && sha256sum -c input_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged inputs differ in %s\n' "$target" >&2
    exit 66
  }
  sha256sum "$target/nemo" >"$target/binary.sha256"
  sha256sum "$target/compiled_dynspg_ts.f90" >"$target/compiled_source.sha256"
}

run_one() {
  local run=$1 mpi_rc tee_rc
  (
    cd "$run"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    mpi_rc=${pipe_rc[0]}
    tee_rc=${pipe_rc[1]:-0}
    printf 'MPIRUN_RC=%d\nRUN_FINISHED_UTC=%s\n' \
      "$mpi_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
    if [[ "$mpi_rc" -ne 0 || "$tee_rc" -ne 0 ]]; then
      printf 'RUN FAILED\n' >>run.user.time.log
      printf 'REFUSE: NEMO run failed in %s (mpi=%s tee=%s)\n' "$run" "$mpi_rc" "$tee_rc" >&2
      exit 68
    fi
    printf 'RUN DONE\n' >>run.user.time.log
  )
}

admit() {
  local rank stem want got
  for rank in 0000 0001; do
    for stem in e3f0vor e3fvor q; do
      want=$SINGLE_FIELD_BYTES
      got=$(stat -c %s "$RUN_A/oracle_een_${stem}_kt00000001_r${rank}.bin" 2>/dev/null || echo 0)
      [[ "$got" == "$want" ]] || {
        printf 'REFUSE: %s rank %s is %s bytes, expected %s\n' "$stem" "$rank" "$got" "$want" >&2
        exit 69
      }
    done
    got=$(stat -c %s "$RUN_A/oracle_een_zpvo_kt00000001_r${rank}.bin" 2>/dev/null || echo 0)
    [[ "$got" == "$EIGHT_FIELD_BYTES" ]] || {
      printf 'REFUSE: zpvo rank %s is %s bytes, expected %s\n' "$rank" "$got" "$EIGHT_FIELD_BYTES" >&2
      exit 69
    }
  done
  if compgen -G "$RUN_A/oracle_een_e3f0vor_kt00000001.bin" >/dev/null; then
    printf 'REFUSE: an UNRANKED EEN record was written; the rank guard is still in place\n' >&2
    exit 69
  fi
  printf 'ORCA2_ROUND8_EEN_RANK_ACQUISITION_PASS %s\n' "$RUN_A"
}

preflight
if [[ "$MODE" == --preflight-only ]]; then
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$RUN_A" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2
  exit 64
}
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: operator must create the non-symlink evidence directory first: %s\n' \
    "$EVIDENCE" >&2
  exit 64
}

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not create the isolated target\n' >&2
  exit 68
}
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_MY_SRC" -maxdepth 1 -type f -print0 | sort -z)
cp "$SOURCE_CPP" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
(cd "$TARGET_ROOT/MY_SRC" && patch -p0 <"$PATCH") || {
  printf 'REFUSE: per-rank writer patch failed\n' >&2
  exit 68
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not build the isolated target\n' >&2
  exit 68
}
[[ -x "$BINARY" ]] || {
  printf 'REFUSE: target build produced no executable\n' >&2
  exit 68
}
for marker in \
  'll_l4_een_dump = kt == nit000' \
  '"_r",I4.4,".bin")'"'"') kt, narea - 1'; do
  grep -Fq "$marker" "$COMPILED" || {
    printf 'REFUSE: compiled source lacks marker: %s\n' "$marker" >&2
    exit 68
  }
done
if grep -Fq 'll_l4_een_dump = lwp' "$COMPILED"; then
  printf 'REFUSE: compiled source retained the rank-zero-only EEN writer\n' >&2
  exit 68
fi
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in the acquisition binary\n' >&2
  exit 68
fi

stage_run "$RUN_A"
sha256sum "$SOURCE_MY_SRC/dynspg_ts.F90" "$PATCH" >"$RUN_A/acquisition_sources.sha256"
run_one "$RUN_A"
admit
