#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 TKE boundary acquisition failed unexpectedly at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

verify_exp00_copy() {
  local source_root=$1
  local target_root=$2
  local source target name

  for source in "$source_root"/*; do
    name=$(basename "$source")
    target=$target_root/$name
    if [[ -L "$source" ]]; then
      if [[ ! -L "$target" ]] || \
          [[ "$(readlink "$source")" != "$(readlink "$target")" ]]; then
        printf 'REFUSE: file-by-file EXP00 copy differs for %s\n' "$name" >&2
        exit 68
      fi
    elif [[ -f "$source" ]]; then
      if [[ ! -f "$target" || -L "$target" ]] || ! cmp -s "$source" "$target"; then
        printf 'REFUSE: file-by-file EXP00 copy differs for %s\n' "$name" >&2
        exit 68
      fi
    else
      printf 'REFUSE: file-by-file EXP00 copy differs for %s\n' "$name" >&2
      exit 68
    fi
  done
}

# USER-EXECUTED ACQUISITION ONLY. --run invokes makenemo and mpirun;
# --finalize only admits an already completed run and never invokes either.
# The construction follows the round-56/59/64/101 source-card clone pattern:
# clone the registered NEMO reference, copy the current ORCA2 source card file
# by file, apply only a WRITE-only instrument, prove the resulting Fortran
# syntax, and admit output only when every inherited stream is byte-identical.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly FAILED_TARGET_CFG=ORCA2_OMIP_L4_P2VBND
readonly TARGET_CFG=ORCA2_OMIP_L4_P2VBND_R2
readonly BASELINE_RUN=/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2/oracle_phase2v_tke_boundary_np2
readonly RECORD=oracle_tke_boundary_kt00000002.bin
readonly EXPECTED_BASELINE_STREAMS=101
# Of the 101 native streams the Phase-2v A reference run holds, seven come from
# instrumentation that no longer exists in the ORCA2_OMIP_L4 source card this
# acquisition clones: the reference executable still contains those writer
# format strings and the candidate executable contains none of them.  A passive
# rebuild of the current card can therefore never reproduce them, and requiring
# them measures the source card's history rather than this build's passivity.
# They are pinned BY NAME so a stream that disappears for any other reason
# still refuses, and so that any of them reappearing changes the compared count
# and refuses too.
readonly -a EXPECTED_ABSENT_STREAMS=(
  oracle_bbl_diffusive_kt00000001.bin
  oracle_een_e3f0vor_kt00000001.bin
  oracle_een_e3fvor_kt00000001.bin
  oracle_een_q_kt00000001.bin
  oracle_een_zpvo_kt00000001.bin
  oracle_zdf_sh2_operands_kt00000001.bin
  oracle_zdf_sh2_operands_kt00000002.bin
)
readonly EXPECTED_ABSENT_STREAM_COUNT=7
# The 101 inherited streams minus the seven the current source card can no
# longer emit.  This is the count that must be compared and identical.
readonly EXPECTED_COMPARED_STREAMS=94
# The 94 comparable inherited streams plus the one new boundary record.
readonly EXPECTED_TARGET_STREAMS=95

# EXCLUSION, and why it is not a weakening of the passivity proof.  The stage-1
# transport stream holds three 94x152x31 double-precision fields -- zFu, zFv
# and zFw -- written back to back after a 48-byte header.  zFw is allocated in
# stprk3_stg.F90 but is assigned only inside the flux-form branch there; this
# deck runs vector-invariant momentum advection (namelist ln_dynadv_vec =
# .true., resolved in ocean.output as "Vector form: 2nd order centered scheme
# ln_dynadv_vec = T"), so that branch never executes and the dump writes an
# allocated-but-never-initialised buffer.  Those bytes are leftover heap and
# track the executable's memory layout rather than the model state: the
# Phase-2v A reference happens to show zeros while this candidate shows values
# at the 270 K sea-ice initialisation temperature (rn_tsu_ini / rn_tmi_ini /
# rn_tms_ini), and other historical builds show the same 270s.  A field dumped
# before it is ever assigned cannot carry a passivity signal, so ONLY that
# third field is excluded.  The header and the two real transports zFu and zFv
# are still byte-compared, and both files must still be exactly the registered
# length, so neither a truncated nor an extended stream can hide in the
# excluded tail.
readonly PARTIAL_STREAM=oracle_transport_kt00000001_s1.bin
readonly PARTIAL_STREAM_BYTES=$((16 + 8 * 4 + 3 * 94 * 152 * 31 * 8))
readonly PARTIAL_COMPARED_BYTES=$((16 + 8 * 4 + 2 * 94 * 152 * 31 * 8))
readonly EXPECTED_SIZE=$((16 + 15 * 4 + 3 * 4 + 94 * 152 * 31 * 8))
readonly EXPECTED_KT=2
readonly EXPECTED_BASELINE_ORACLE_MANIFEST_SHA256=4fbffaed98202a052059c503a637bc2f8d5e57c28e7345f5bfce5b62320708a9
readonly PHASE2V_REVISION=b7ce08cc8afa5cf377922abf198cf1794fab8a73
readonly PHASE2V_PATCH_PATH=scripts/validate/ocean_fidelity/orca2_l4/phase2v_tke_walk_writer.patch
readonly PHASE2V_PATCH_SHA256=e317142ae3e23c626dd00a7aac53ca3ff81eddee9aa3df0d8d1e3249b105db54
readonly EXPECTED_DECK_MANIFEST_SHA256=e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--finalize) ;;
  *)
    printf 'REFUSE: usage: %s [--run|--preflight-only|--finalize]\n' "$0" >&2
    exit 64
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly FAILED_TARGET_ROOT=$NEMO_ROOT/cfgs/$FAILED_TARGET_CFG
readonly CANONICAL=$NEMO_ROOT/src/OCE/ZDF/zdftke.F90
readonly WRITER=$here/l2_orca2_tke_boundary.F90
readonly BOUNDARY_PATCH=$here/zdftke_orca2_boundary.patch
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED_ZDF=$TARGET_ROOT/BLD/ppsrc/nemo/zdftke.f90
readonly COMPILED_WRITER=$TARGET_ROOT/BLD/ppsrc/nemo/l2_orca2_tke_boundary.f90
readonly ADMISSION_JSON=orca2_tke_boundary_admission.json
readonly DIGEST_MANIFEST=orca2_tke_boundary_outputs.sha256
readonly REFERENCE_MANIFEST=orca2_tke_boundary_references.sha256

stream_is_expected_absent() {
  local candidate=$1 name
  for name in "${EXPECTED_ABSENT_STREAMS[@]}"; do
    if [[ "$name" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

# Byte-compare one inherited stream between reference and candidate, returning
# cmp's own status: 0 identical, 1 differing, greater than 1 unreadable.  Only
# the stage-1 transport stream is handled specially, and only by stopping the
# comparison before its uninitialised third field; see the EXCLUSION note above.
# A stage-1 transport stream of the wrong length returns the distinct status 3
# so that the length requirement -- the check that stops a short or padded file
# hiding bytes in the excluded tail -- reports as its own named condition rather
# than as an ordinary differing stream.
compare_inherited_stream() {
  local name=$1 status=0
  if [[ "$name" == "$PARTIAL_STREAM" ]]; then
    if [[ "$(stat -c %s "$BASELINE_RUN/$name")" -ne "$PARTIAL_STREAM_BYTES" ]] \
        || [[ "$(stat -c %s "$TARGET_RUN/$name")" -ne "$PARTIAL_STREAM_BYTES" ]]; then
      return 3
    fi
    cmp -s -n "$PARTIAL_COMPARED_BYTES" \
      "$BASELINE_RUN/$name" "$TARGET_RUN/$name" || status=$?
    return "$status"
  fi
  cmp -s "$BASELINE_RUN/$name" "$TARGET_RUN/$name" || status=$?
  return "$status"
}

validate_record_schema() {
  if ! python3 - "$TARGET_RUN/$RECORD" <<'PY'
import pathlib
import struct
import sys

record = pathlib.Path(sys.argv[1])
with record.open("rb") as handle:
    magic = handle.read(16).decode("ascii").rstrip()
    header = struct.unpack("=15i", handle.read(15 * 4))
    extents = struct.unpack("=3i", handle.read(3 * 4))
    payload = handle.read()
expected_header = (1, 2, 3, 3, 94, 152, 31, 64, 1, 0,
                   94 * 152 * 31, 1, 1, 3, 1)
if magic != "NEMO_L4_TKEB_1" or header != expected_header:
    raise SystemExit(1)
if extents != (94, 152, 31) or len(payload) != 94 * 152 * 31 * 8:
    raise SystemExit(1)
if record.stat().st_size != 3543512:
    raise SystemExit(1)
PY
  then
    printf 'REFUSE: native en_after_boundaries frame failed exact schema/EOF validation\n' >&2
    exit 66
  fi
}

finalize_existing() {
  local path resolved_target expected_binary recorded_binary manifest_extra
  local reference_hash
  local built_binary copied_binary record_size record_digest producer
  local baseline_reference_digest reference_manifest_digest finalize_exit
  local marker_count=0 expected_marker_count=0 marker_kt marker_line
  local compared=0 identical=0 cmp_status name verdict
  local audit_tmp admission_tmp digest_tmp reference_output_tmp
  local -a ocean_outputs=() baseline_streams=() target_streams=()
  local -a marker_outputs=() missing_streams=() differing_streams=()
  local -a extra_streams=() absent_streams=() sized_streams=()

  [[ -d "$TARGET_RUN" && ! -L "$TARGET_RUN" ]] || {
    printf 'REFUSE: finalization run directory does not match registered target: %s\n' \
      "$TARGET_RUN" >&2
    exit 64
  }
  resolved_target=$(CDPATH= cd -- "$TARGET_RUN" && pwd -P)
  [[ "$resolved_target" == "$TARGET_RUN" ]] || {
    printf 'REFUSE: finalization run directory does not match registered target: %s\n' \
      "$resolved_target" >&2
    exit 64
  }
  for path in "${TARGET_RUN}/${RECORD}.stamp" \
      "$TARGET_RUN/$ADMISSION_JSON" "$TARGET_RUN/$DIGEST_MANIFEST" \
      "$TARGET_RUN/$REFERENCE_MANIFEST"; do
    [[ ! -d "$path" || -L "$path" ]] || {
      printf 'REFUSE: finalization artifact path is not a replaceable file: %s\n' \
        "$path" >&2
      exit 64
    }
    rm -f -- "$path"
  done
  for path in "$BINARY" "$TARGET_RUN/nemo" "$TARGET_RUN/binary.sha256" \
      "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/$RECORD" \
      "$TARGET_RUN/run.user.stdout.log" "$TARGET_RUN/run.user.time.log" \
      "$TARGET_RUN/time.step"; do
    [[ -f "$path" ]] || {
      printf 'REFUSE: existing finalization target lacks %s\n' "$path" >&2
      exit 64
    }
  done

  if ! read -r expected_binary recorded_binary manifest_extra \
      <"$TARGET_RUN/binary.sha256"; then
    printf 'REFUSE: existing binary manifest is not the registered one-line target manifest\n' >&2
    exit 66
  fi
  [[ "$(wc -l <"$TARGET_RUN/binary.sha256")" -eq 1 \
      && "$expected_binary" =~ ^[0-9a-f]{64}$ && -z "${manifest_extra:-}" \
      && "$recorded_binary" == "$BINARY" ]] || {
    printf 'REFUSE: existing binary manifest is not the registered one-line target manifest\n' >&2
    exit 66
  }
  built_binary=$(sha256sum "$BINARY" | awk '{print $1}')
  copied_binary=$(sha256sum "$TARGET_RUN/nemo" | awk '{print $1}')
  [[ "$built_binary" == "$expected_binary" \
      && "$copied_binary" == "$expected_binary" ]] || {
    printf 'REFUSE: existing target binary digest does not match binary manifest\n' >&2
    exit 66
  }

  record_size=$(stat -c %s "$TARGET_RUN/$RECORD")
  [[ "$record_size" -eq "$EXPECTED_SIZE" ]] || {
    printf 'REFUSE: existing record size %s does not match registered manifest size %s\n' \
      "$record_size" "$EXPECTED_SIZE" >&2
    exit 66
  }
  validate_record_schema

  mapfile -d '' -t ocean_outputs < <(
    find "$TARGET_RUN" -maxdepth 1 -type f \
      \( -name 'ocean.output' -o -name 'ocean.output.*' \
         -o -name 'ocean.output_*' \) -print0 | sort -z
  )
  [[ "${#ocean_outputs[@]}" -gt 0 ]] || {
    printf 'REFUSE: existing target has no ocean.output rank files\n' >&2
    exit 66
  }
  for path in "${ocean_outputs[@]}"; do
    while IFS= read -r marker_line; do
      marker_count=$((marker_count + 1))
      marker_outputs+=("$(basename "$path")")
      if [[ "$marker_line" =~ ^[[:space:]]*ORCA2_TKE_BOUNDARY_DUMP[[:space:]]+([0-9]+)([[:space:]]|$) ]]; then
        marker_kt=${BASH_REMATCH[1]}
        if [[ "$marker_kt" -eq "$EXPECTED_KT" ]]; then
          expected_marker_count=$((expected_marker_count + 1))
        fi
      fi
    done < <(grep -E '^[[:space:]]*ORCA2_TKE_BOUNDARY_DUMP([[:space:]]|$)' \
      "$path" || true)
  done
  [[ "$marker_count" -eq 1 && "$expected_marker_count" -eq 1 ]] || {
    printf 'REFUSE: ocean.output boundary marker mismatch; expected kt=%s, found markers=%s matching=%s\n' \
      "$EXPECTED_KT" "$marker_count" "$expected_marker_count" >&2
    exit 66
  }
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || {
    printf 'REFUSE: candidate stdout lacks STOP 0\n' >&2
    exit 66
  }
  grep -Fxq 'MPIRUN_RC=0' "$TARGET_RUN/run.user.time.log" || {
    printf 'REFUSE: candidate run lacks MPIRUN_RC=0\n' >&2
    exit 66
  }
  grep -Fxq 'RUN DONE' "$TARGET_RUN/run.user.time.log" || {
    printf 'REFUSE: candidate run lacks RUN DONE\n' >&2
    exit 66
  }
  [[ "$(tr -d '[:space:]' <"$TARGET_RUN/time.step")" == 10 ]] || {
    printf 'REFUSE: candidate run did not finish time step 10\n' >&2
    exit 66
  }

  mapfile -t baseline_streams < <(
    find "$BASELINE_RUN" -maxdepth 1 -type f -name 'oracle_*.bin' \
      -printf '%f\n' | sort
  )
  [[ "${#baseline_streams[@]}" -eq "$EXPECTED_BASELINE_STREAMS" ]] || {
    printf 'REFUSE: Phase-2v A manifest has %s native streams, expected %s\n' \
      "${#baseline_streams[@]}" "$EXPECTED_BASELINE_STREAMS" >&2
    exit 65
  }
  audit_tmp=$(mktemp -d /tmp/orca2-tke-boundary-finalize.XXXXXXXX)
  (
    cd "$BASELINE_RUN"
    for name in "${baseline_streams[@]}"; do
      sha256sum "$name"
    done
  ) >"$audit_tmp/baseline-oracle-streams.sha256"
  baseline_reference_digest=$( \
    sha256sum "$audit_tmp/baseline-oracle-streams.sha256" | \
    awk '{print $1}')
  [[ "$baseline_reference_digest" == \
      "$EXPECTED_BASELINE_ORACLE_MANIFEST_SHA256" ]] || {
    rm -rf -- "$audit_tmp"
    printf 'REFUSE: Phase-2v A oracle-stream manifest digest changed\n' >&2
    exit 65
  }
  while read -r reference_hash name; do
    printf '%s  %s/%s\n' "$reference_hash" "$BASELINE_RUN" "$name"
  done <"$audit_tmp/baseline-oracle-streams.sha256" \
    >"$audit_tmp/$REFERENCE_MANIFEST"
  sha256sum "$BINARY" >>"$audit_tmp/$REFERENCE_MANIFEST"
  reference_manifest_digest=$(sha256sum "$audit_tmp/$REFERENCE_MANIFEST" | \
    awk '{print $1}')
  mapfile -t target_streams < <(
    find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_*.bin' \
      -printf '%f\n' | sort
  )
  for name in "${baseline_streams[@]}"; do
    if [[ ! -f "$TARGET_RUN/$name" ]]; then
      # A stream the current source card can no longer emit is recorded and
      # skipped; anything else absent is a genuine missing stream and refuses.
      if stream_is_expected_absent "$name"; then
        absent_streams+=("$name")
      else
        missing_streams+=("$name")
      fi
      continue
    fi
    compared=$((compared + 1))
    cmp_status=0
    compare_inherited_stream "$name" || cmp_status=$?
    if [[ "$cmp_status" -eq 0 ]]; then
      identical=$((identical + 1))
    elif [[ "$cmp_status" -eq 3 ]]; then
      sized_streams+=("$name")
    else
      [[ "$cmp_status" -eq 1 ]] || {
        rm -rf -- "$audit_tmp"
        printf 'REFUSE: byte comparison failed to read inherited stream %s\n' \
          "$name" >&2
        exit 69
      }
      differing_streams+=("$name")
    fi
  done
  for name in "${target_streams[@]}"; do
    if [[ "$name" != "$RECORD" && ! -f "$BASELINE_RUN/$name" ]]; then
      extra_streams+=("$name")
    fi
  done

  verdict=PASS
  finalize_exit=0
  if [[ "${#target_streams[@]}" -ne "$EXPECTED_TARGET_STREAMS" \
      || "${#missing_streams[@]}" -ne 0 \
      || "${#absent_streams[@]}" -ne "$EXPECTED_ABSENT_STREAM_COUNT" \
      || "${#differing_streams[@]}" -ne 0 \
      || "${#sized_streams[@]}" -ne 0 \
      || "${#extra_streams[@]}" -ne 0 \
      || "$compared" -ne "$EXPECTED_COMPARED_STREAMS" \
      || "$identical" -ne "$EXPECTED_COMPARED_STREAMS" ]]; then
    verdict=REFUSE
    finalize_exit=69
  fi

  producer=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  [[ "$producer" =~ ^[0-9a-f]{40}$ ]] || {
    rm -rf -- "$audit_tmp"
    printf 'REFUSE: existing target producer manifest is malformed\n' >&2
    exit 66
  }
  record_digest=$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$record_digest" "$producer" "$RECORD" \
    >"$TARGET_RUN/$RECORD.stamp"

  reference_output_tmp=$(mktemp \
    "$TARGET_RUN/.orca2-tke-boundary-references.XXXXXXXX")
  cp "$audit_tmp/$REFERENCE_MANIFEST" "$reference_output_tmp"
  mv "$reference_output_tmp" "$TARGET_RUN/$REFERENCE_MANIFEST"
  printf '%s\n' "${missing_streams[@]:-}" >"$audit_tmp/missing"
  printf '%s\n' "${absent_streams[@]:-}" >"$audit_tmp/absent"
  printf '%s\n' "${differing_streams[@]:-}" >"$audit_tmp/differing"
  printf '%s\n' "${sized_streams[@]:-}" >"$audit_tmp/sized"
  printf '%s\n' "${extra_streams[@]:-}" >"$audit_tmp/extra"
  printf '%s\n' "${marker_outputs[@]:-}" >"$audit_tmp/marker_outputs"
  admission_tmp=$(mktemp "$TARGET_RUN/.orca2-tke-boundary-admission.XXXXXXXX")
  python3 - "$admission_tmp" "$audit_tmp" "$verdict" "$BASELINE_RUN" \
      "$TARGET_RUN" "$producer" "$expected_binary" "$record_digest" \
      "$record_size" "$EXPECTED_KT" "$marker_count" "$finalize_exit" \
      "$REFERENCE_MANIFEST" "$reference_manifest_digest" \
      "$baseline_reference_digest" \
      "${#ocean_outputs[@]}" "$EXPECTED_BASELINE_STREAMS" \
      "$EXPECTED_TARGET_STREAMS" "${#target_streams[@]}" "$compared" \
      "$identical" "$EXPECTED_COMPARED_STREAMS" \
      "$EXPECTED_ABSENT_STREAM_COUNT" "$PARTIAL_STREAM" \
      "$PARTIAL_STREAM_BYTES" "$PARTIAL_COMPARED_BYTES" <<'PY'
import json
import pathlib
import sys

(
    output, audit_root, verdict, baseline, candidate, producer,
    binary_sha256, record_sha256, record_size, expected_kt, marker_count,
    exit_code, reference_manifest, reference_manifest_sha256,
    baseline_oracle_manifest_sha256, output_count, expected_baseline,
    expected_target, target_count, compared, identical, expected_compared,
    expected_absent_count, partial_stream, partial_stream_bytes,
    partial_compared_bytes,
) = sys.argv[1:]
audit_root = pathlib.Path(audit_root)

def lines(name):
    return [line for line in (audit_root / name).read_text().splitlines()
            if line]

report = {
    "format": "nemo-orca2-tke-boundary-admission-v1",
    "verdict": verdict,
    "exit_code": int(exit_code),
    "baseline": baseline,
    "candidate": candidate,
    "producer_commit": producer,
    "binary_sha256": binary_sha256,
    "references": {
        "manifest": reference_manifest,
        "sha256": reference_manifest_sha256,
        "baseline_oracle_streams_sha256": baseline_oracle_manifest_sha256,
    },
    "record": {
        "name": "oracle_tke_boundary_kt00000002.bin",
        "size_bytes": int(record_size),
        "sha256": record_sha256,
        "schema_eof_valid": True,
    },
    "marker": {
        "expected_kt": int(expected_kt),
        "matches": int(marker_count),
        "ocean_output_files_scanned": int(output_count),
        "files": lines("marker_outputs"),
    },
    "passivity": {
        "expected_baseline_streams": int(expected_baseline),
        "expected_compared_streams": int(expected_compared),
        "expected_target_streams": int(expected_target),
        "target_streams": int(target_count),
        "streams_compared": int(compared),
        "byte_identical": int(identical),
        "missing": lines("missing"),
        "differing": lines("differing"),
        "wrong_length": lines("sized"),
        "unregistered_extra": lines("extra"),
        "expected_absent": {
            "count": int(expected_absent_count),
            "reason": (
                "writer instrumentation absent from the current ORCA2_OMIP_L4 "
                "source card; present only in the reference executable"
            ),
            "streams": lines("absent"),
        },
        "partial_comparison": {
            "compared_bytes": int(partial_compared_bytes),
            "excluded_field": "zFw",
            "reason": (
                "zFw is dumped before assignment under ln_dynadv_vec = .true. "
                "and holds uninitialised heap; header, zFu and zFv are still "
                "byte-compared and the full length is still required"
            ),
            "required_bytes": int(partial_stream_bytes),
            "stream": partial_stream,
        },
    },
}
pathlib.Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
  mv "$admission_tmp" "$TARGET_RUN/$ADMISSION_JSON"

  digest_tmp=$(mktemp "$TARGET_RUN/.orca2-tke-boundary-outputs.XXXXXXXX")
  (
    cd "$TARGET_RUN"
    {
      printf '%s\0' nemo binary.sha256 producer_commit.txt \
        "$RECORD.stamp" "$ADMISSION_JSON" "$REFERENCE_MANIFEST" \
        run.user.stdout.log run.user.time.log time.step
      printf '%s\0' "${target_streams[@]}"
      for path in "${ocean_outputs[@]}"; do
        printf '%s\0' "$(basename "$path")"
      done
    } | sort -zu | xargs -0 sha256sum >"$digest_tmp"
    cat "$REFERENCE_MANIFEST" >>"$digest_tmp"
  )
  mv "$digest_tmp" "$TARGET_RUN/$DIGEST_MANIFEST"

  printf 'ORCA2_TKE_BOUNDARY_REFERENCES PASS baseline_sha256=%s manifest_sha256=%s\n' \
    "$baseline_reference_digest" "$reference_manifest_digest"
  printf 'ORCA2_TKE_BOUNDARY_MARKER PASS kt=%s output_files=%s markers=%s\n' \
    "$EXPECTED_KT" "${#ocean_outputs[@]}" "$marker_count"
  printf 'ORCA2_TKE_BOUNDARY_RECORD PASS bytes=%s sha256=%s\n' \
    "$record_size" "$record_digest"
  printf 'ORCA2_TKE_BOUNDARY_PASSIVITY %s expected=%s compared=%s identical=%s missing=%s absent=%s/%s differing=%s wrong_length=%s extra=%s target=%s/%s\n' \
    "$verdict" "$EXPECTED_COMPARED_STREAMS" "$compared" "$identical" \
    "${#missing_streams[@]}" "${#absent_streams[@]}" \
    "$EXPECTED_ABSENT_STREAM_COUNT" "${#differing_streams[@]}" \
    "${#sized_streams[@]}" \
    "${#extra_streams[@]}" "${#target_streams[@]}" \
    "$EXPECTED_TARGET_STREAMS"
  printf 'ORCA2_TKE_BOUNDARY_PARTIAL %s compared_bytes=%s of %s excluded_field=zFw\n' \
    "$PARTIAL_STREAM" "$PARTIAL_COMPARED_BYTES" "$PARTIAL_STREAM_BYTES"
  for name in "${absent_streams[@]}"; do
    printf 'ORCA2_TKE_BOUNDARY_ABSENT %s\n' "$name"
  done
  for name in "${missing_streams[@]}"; do
    printf 'ORCA2_TKE_BOUNDARY_MISSING %s\n' "$name"
  done
  for name in "${differing_streams[@]}"; do
    printf 'ORCA2_TKE_BOUNDARY_DIFFERS %s\n' "$name"
  done
  for name in "${sized_streams[@]}"; do
    printf 'ORCA2_TKE_BOUNDARY_WRONG_LENGTH %s\n' "$name"
  done
  for name in "${extra_streams[@]}"; do
    printf 'ORCA2_TKE_BOUNDARY_EXTRA %s\n' "$name"
  done
  printf 'ORCA2_TKE_BOUNDARY_ADMISSION %s %s\n' \
    "$verdict" "$TARGET_RUN/$ADMISSION_JSON"
  printf 'ORCA2_TKE_BOUNDARY_DIGESTS READY %s\n' \
    "$TARGET_RUN/$DIGEST_MANIFEST"
  rm -rf -- "$audit_tmp"
  if [[ "$verdict" != PASS ]]; then
    for name in "${sized_streams[@]}"; do
      printf 'REFUSE: inherited stream %s is not the registered %s bytes\n' \
        "$name" "$PARTIAL_STREAM_BYTES" >&2
    done
    printf 'REFUSE: write-only passivity failed; expected=%s compared=%s identical=%s missing=%s absent=%s/%s differing=%s extra=%s\n' \
      "$EXPECTED_COMPARED_STREAMS" "$compared" "$identical" \
      "${#missing_streams[@]}" "${#absent_streams[@]}" \
      "$EXPECTED_ABSENT_STREAM_COUNT" "${#differing_streams[@]}" \
      "${#extra_streams[@]}" >&2
    exit "$finalize_exit"
  fi
  printf 'ORCA2_TKE_BOUNDARY_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --finalize ]]; then
  finalize_existing
  exit 0
fi

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
}
readonly COMMIT=$(git rev-parse HEAD)

for path in "$CANONICAL" "$WRITER" "$BOUNDARY_PATCH" \
    "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$BASELINE_RUN/deck_files.sha256" \
    "$BASELINE_RUN/input_files.sha256" "$BASELINE_RUN/ocean.output" \
    "$BASELINE_RUN/run.user.time.log" "$BASELINE_RUN/time.step"; do
  [[ -f "$path" ]] || {
    printf 'REFUSE: missing required acquisition input %s\n' "$path" >&2
    exit 64
  }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || {
  printf 'REFUSE: registered source configuration is incomplete: %s\n' "$SOURCE_ROOT" >&2
  exit 64
}
[[ -d "$NEMO_ROOT/cfgs/$REFERENCE_CFG" ]] || {
  printf 'REFUSE: registered reference configuration is absent: %s\n' "$REFERENCE_CFG" >&2
  exit 64
}
[[ "$EXPECTED_SIZE" -eq 3543512 ]] || {
  printf 'REFUSE: native boundary-frame byte arithmetic moved from 3543512\n' >&2
  exit 66
}

if ! printf '%s  %s\n' "$EXPECTED_DECK_MANIFEST_SHA256" \
    "$BASELINE_RUN/deck_files.sha256" | sha256sum -c - >/dev/null; then
  printf 'REFUSE: Phase-2v A deck manifest digest changed\n' >&2
  exit 65
fi
if ! printf '%s  %s\n' "$EXPECTED_INPUT_MANIFEST_SHA256" \
    "$BASELINE_RUN/input_files.sha256" | sha256sum -c - >/dev/null; then
  printf 'REFUSE: Phase-2v A input manifest digest changed\n' >&2
  exit 65
fi
if ! (cd "$BASELINE_RUN" && sha256sum -c deck_files.sha256 >/dev/null); then
  printf 'REFUSE: Phase-2v A deck files no longer verify\n' >&2
  exit 65
fi
if ! (cd "$BASELINE_RUN" && sha256sum -c input_files.sha256 >/dev/null); then
  printf 'REFUSE: Phase-2v A input files no longer verify\n' >&2
  exit 65
fi
grep -Fxq 'MPIRUN_RC=0' "$BASELINE_RUN/run.user.time.log" || {
  printf 'REFUSE: Phase-2v A lacks MPIRUN_RC=0\n' >&2
  exit 65
}
grep -Fxq 'RUN DONE' "$BASELINE_RUN/run.user.time.log" || {
  printf 'REFUSE: Phase-2v A lacks RUN DONE\n' >&2
  exit 65
}
[[ "$(tr -d '[:space:]' <"$BASELINE_RUN/time.step")" == 10 ]] || {
  printf 'REFUSE: Phase-2v A did not finish time step 10\n' >&2
  exit 65
}
mapfile -t baseline_streams < <(
  find "$BASELINE_RUN" -maxdepth 1 -type f -name 'oracle_*.bin' \
    -printf '%f\n' | sort
)
[[ "${#baseline_streams[@]}" -eq "$EXPECTED_BASELINE_STREAMS" ]] || {
  printf 'REFUSE: Phase-2v A has %s native streams, expected %s\n' \
    "${#baseline_streams[@]}" "$EXPECTED_BASELINE_STREAMS" >&2
  exit 65
}
for pattern in \
    'jpni *= *2' \
    'jpnj *= *1' \
    'number of the first time step.*nn_it000 *= *1' \
    'number of the last time step.*nn_itend *= *10' \
    'Tiling.*ln_tile *= *F' \
    'prandl number flag.*nn_pdl *= *1' \
    'mixing length type.*nn_mxl *= *3' \
    'Langmuir cells parametrization.*ln_lc *= *T' \
    'test param. to add tke induced by wind.*nn_etau *= *1' \
    'langmuir & surface wave breaking under ice.*nn_eice *= *1'; do
  grep -Eq "$pattern" "$BASELINE_RUN/ocean.output" || {
    printf 'REFUSE: Phase-2v A lacks resolved configuration row %s\n' "$pattern" >&2
    exit 65
  }
done

dry=$(mktemp -d /tmp/orca2-tke-boundary-source.XXXXXXXX)
readonly DRY_PHASE2V=$dry/zdftke.phase2v.F90
readonly DRY_BOUNDARY=$dry/zdftke.boundary.F90
readonly HISTORICAL_PATCH=$dry/phase2v_tke_walk_writer.patch
if ! git show "$PHASE2V_REVISION:$PHASE2V_PATCH_PATH" >"$HISTORICAL_PATCH"; then
  printf 'REFUSE: cannot extract the registered Phase-2v source patch\n' >&2
  exit 66
fi
[[ "$(sha256sum "$HISTORICAL_PATCH" | awk '{print $1}')" == \
    "$PHASE2V_PATCH_SHA256" ]] || {
  printf 'REFUSE: registered Phase-2v source patch digest changed\n' >&2
  exit 66
}
cp "$CANONICAL" "$DRY_PHASE2V"
if ! patch -s "$DRY_PHASE2V" <"$HISTORICAL_PATCH"; then
  printf 'REFUSE: Phase-2v patch no longer applies to canonical zdftke.F90\n' >&2
  exit 66
fi
cp "$DRY_PHASE2V" "$DRY_BOUNDARY"
if ! patch -s "$DRY_BOUNDARY" <"$BOUNDARY_PATCH"; then
  printf 'REFUSE: native boundary patch no longer applies to Phase-2v zdftke.F90\n' >&2
  exit 66
fi
removed_lines=$(diff "$DRY_PHASE2V" "$DRY_BOUNDARY" | grep '^<' || true)
if [[ -n "$removed_lines" ]]; then
  printf 'REFUSE: native boundary patch removes or replaces a Phase-2v source line\n' >&2
  exit 66
fi
if grep -Eq '^-[^-]' "$BOUNDARY_PATCH"; then
  printf 'REFUSE: native boundary patch contains a removed source line\n' >&2
  exit 66
fi
for call in o2b_begin o2b_after_boundaries_row o2b_finish; do
  [[ "$(grep -c "CALL $call" "$DRY_BOUNDARY")" -eq 1 ]] || {
    printf 'REFUSE: patched zdftke has the wrong call count for %s\n' "$call" >&2
    exit 66
  }
done
if ! python3 - "$DRY_BOUNDARY" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
capture = "CALL o2b_after_boundaries_row( jj, en )"
langmuir = "IF( ln_lc ) THEN"
if source.index(capture) > source.index(langmuir):
    raise SystemExit(1)
between = source[source.index(capture) + len(capture) : source.index(langmuir)]
if between.strip():
    raise SystemExit(1)
PY
then
  printf 'REFUSE: native frame is not immediately after boundaries and before Langmuir\n' >&2
  exit 66
fi
grep -Fq "CHARACTER(LEN=16), PARAMETER :: o2b_magic = 'NEMO_L4_TKEB_1'" \
    "$WRITER" || {
  printf 'REFUSE: boundary writer magic/layout declaration moved\n' >&2
  exit 66
}
grep -Fq 'INTENT(in) :: p_en' "$WRITER" || {
  printf 'REFUSE: boundary writer no longer treats model TKE as read-only\n' >&2
  exit 66
}

syntax=$(mktemp -d /tmp/orca2-tke-boundary-syntax.XXXXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 \
    -P -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
if ! preprocess "$WRITER" "$syntax/l2_orca2_tke_boundary.f90"; then
  printf 'REFUSE: preprocessing the native boundary writer failed\n' >&2
  exit 66
fi
if ! preprocess "$DRY_BOUNDARY" "$syntax/zdftke.f90"; then
  printf 'REFUSE: preprocessing patched zdftke failed\n' >&2
  exit 66
fi
if ! "$FC" -fsyntax-only -ffree-line-length-none \
    -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" \
    "$syntax/l2_orca2_tke_boundary.f90"; then
  printf 'REFUSE: native boundary writer failed gfortran -fsyntax-only\n' >&2
  exit 66
fi
if ! "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
    -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/zdftke.f90"; then
  printf 'REFUSE: patched zdftke failed gfortran -fsyntax-only\n' >&2
  exit 66
fi

if [[ "$MODE" == --preflight-only ]]; then
  if [[ -d "$FAILED_TARGET_ROOT/EXP00" ]]; then
    # The failed first attempt copied this card after building its reference
    # clone. Verify the corrected link-aware comparison against that copy, but
    # never reuse its pre-instrumentation executable.
    verify_exp00_copy "$SOURCE_ROOT/EXP00" "$FAILED_TARGET_ROOT/EXP00"
    printf 'ORCA2_TKE_BOUNDARY_EXISTING_EXP00_COPY_READY %s\n' \
      "$FAILED_TARGET_ROOT"
  fi
  printf 'ORCA2_TKE_BOUNDARY_PREFLIGHT_READY %s\n' "$BASELINE_RUN"
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: new target already exists: %s or %s\n' \
    "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  }
done

manifest=$(mktemp -d /tmp/orca2-tke-boundary-manifest.XXXXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -maxdepth 2 \( -type f -o -type l \) -print0 \
    | sort -z | xargs -0 sha256sum
) >"$manifest/source_card.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$CANONICAL" "$HISTORICAL_PATCH" \
  "$WRITER" "$BOUNDARY_PATCH" "$BASELINE_RUN/ocean.output" \
  "$BASELINE_RUN/deck_files.sha256" "$BASELINE_RUN/input_files.sha256" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
if ! ./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath \
    del_key 'key_xios'; then
  printf 'REFUSE: makenemo could not clone the registered reference configuration\n' >&2
  exit 68
fi
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$DRY_BOUNDARY" "$TARGET_ROOT/MY_SRC/zdftke.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_orca2_tke_boundary.F90"
verify_exp00_copy "$SOURCE_ROOT/EXP00" "$TARGET_ROOT/EXP00"
for source in "$SOURCE_ROOT"/MY_SRC/*; do
  name=$(basename "$source")
  if ! cmp -s "$source" "$TARGET_ROOT/MY_SRC/$name"; then
    printf 'REFUSE: file-by-file MY_SRC copy differs for %s\n' "$name" >&2
    exit 68
  fi
done
if ! cmp -s "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
    "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"; then
  printf 'REFUSE: copied registered CPP source card differs\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
if ! ./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'; then
  printf 'REFUSE: makenemo could not build the new ORCA2 boundary target\n' >&2
  exit 68
fi
[[ -x "$BINARY" ]] || {
  printf 'REFUSE: new ORCA2 boundary target produced no executable\n' >&2
  exit 68
}
for pair in \
    "$COMPILED_ZDF:CALL o2b_begin" \
    "$COMPILED_ZDF:CALL o2b_after_boundaries_row" \
    "$COMPILED_ZDF:CALL o2b_finish" \
    "$COMPILED_WRITER:NEMO_L4_TKEB_1"; do
  file=${pair%%:*}
  needle=${pair#*:}
  grep -Fq "$needle" "$file" || {
    printf 'REFUSE: compiled source lacks registered write-only marker %s\n' "$needle" >&2
    exit 68
  }
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in the acquisition binary\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
while read -r _ name; do
  [[ -f "$BASELINE_RUN/$name" || -L "$BASELINE_RUN/$name" ]] || {
    printf 'REFUSE: Phase-2v A deck manifest names missing file %s\n' "$name" >&2
    exit 65
  }
  cp -a "$BASELINE_RUN/$name" "$TARGET_RUN/$name"
done <"$BASELINE_RUN/deck_files.sha256"
while read -r _ name; do
  [[ -f "$BASELINE_RUN/$name" || -L "$BASELINE_RUN/$name" ]] || {
    printf 'REFUSE: Phase-2v A input manifest names missing file %s\n' "$name" >&2
    exit 65
  }
  cp -a "$BASELINE_RUN/$name" "$TARGET_RUN/$name"
done <"$BASELINE_RUN/input_files.sha256"
cp "$BASELINE_RUN/deck_files.sha256" "$BASELINE_RUN/input_files.sha256" \
  "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
if ! (cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null); then
  printf 'REFUSE: staged candidate deck is not byte-identical to Phase-2v A\n' >&2
  exit 68
fi
if ! (cd "$TARGET_RUN" && sha256sum -c input_files.sha256 >/dev/null); then
  printf 'REFUSE: staged candidate inputs are not byte-identical to Phase-2v A\n' >&2
  exit 68
fi

(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  set +e
  { time mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}
  tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nRUN_FINISHED_UTC=%s\n' \
    "$mpi_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  if [[ "$mpi_rc" -ne 0 ]]; then
    printf 'RUN FAILED\n' >>run.user.time.log
    printf 'REFUSE: np2 NEMO acquisition exited %s\n' "$mpi_rc" >&2
    exit "$mpi_rc"
  fi
  if [[ "$tee_rc" -ne 0 ]]; then
    printf 'RUN FAILED\n' >>run.user.time.log
    printf 'REFUSE: acquisition stdout capture exited %s\n' "$tee_rc" >&2
    exit "$tee_rc"
  fi
  printf 'RUN DONE\n' >>run.user.time.log
)

finalize_existing
