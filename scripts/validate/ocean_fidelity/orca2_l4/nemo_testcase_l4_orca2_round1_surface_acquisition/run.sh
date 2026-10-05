#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-5 ORCA2 surface/entry acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

# USER-EXECUTED ACQUISITION ONLY.  No argument or --run invokes makenemo and
# mpirun.  Agents must pass --preflight-only explicitly and never use --run.
readonly MODE=${1:---run}
case "$MODE" in
  --preflight-only|--run|--finalize) ;;
  *)
    printf 'REFUSE: usage: %s [--preflight-only|--run|--finalize]\n' "$0" >&2
    exit 64
    ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED_STPRK3=$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90
readonly COMPILED_ICEISTATE=$TARGET_ROOT/BLD/ppsrc/nemo/iceistate.f90
readonly BASELINE=/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_a_10step_np2
readonly FROZEN_ICE=/data/abyssal/dbalwada/nemo-testcases-l4/build/phase2x_orca1ice/MY_SRC_final
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition
readonly RUN_A=$EVIDENCE/orca1ice_surface_entry_every_step_a_np2
readonly RUN_B=$EVIDENCE/orca1ice_surface_entry_every_step_b_np2
readonly EXPECTED_BASELINE_STREAMS=116
readonly EXPECTED_INHERITED_STREAMS=107
readonly EXPECTED_TARGET_STREAMS=136
readonly EXPECTED_DECK_MANIFEST_SHA256=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly EXPECTED_BASELINE_BINARY_SHA256=8e40bf0b595eabba1bd428775a45e87f3f42a778333374cbbfb26eeb6c685869
readonly EXPECTED_STPRK3_SHA256=9d0318fda246ef1ed3df50172b9d72a5e0b5df879b9a661b38622c9078078989
readonly EXPECTED_PATCH_SHA256=877fcb02c77aa8f7f6b5ebb7845ee8ca0f26d0be9d437b0f10c3cef7c7803b9c
readonly EXPECTED_CPP_SHA256=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly EXPECTED_ARCH_SHA256=132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561
readonly PARTIAL_STREAM=oracle_transport_kt00000001_s1.bin
readonly PARTIAL_STREAM_BYTES=$((16 + 8 * 4 + 3 * 94 * 152 * 31 * 8))
readonly PARTIAL_COMPARED_BYTES=$((16 + 8 * 4 + 2 * 94 * 152 * 31 * 8))
readonly -a EXPECTED_ABSENT_STREAMS=(
  oracle_bbl_diffusive_kt00000001.bin
  oracle_een_e3f0vor_kt00000001.bin
  oracle_een_e3fvor_kt00000001.bin
  oracle_een_q_kt00000001.bin
  oracle_een_zpvo_kt00000001.bin
  oracle_si3_bulk_operands.bin
  oracle_tke_walk_kt00000002.bin
  oracle_zdf_sh2_operands_kt00000001.bin
  oracle_zdf_sh2_operands_kt00000002.bin
)

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/stprk3_surface_every_step.patch
readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly PYTHONPATH_VALUE=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src:$REPO

is_expected_absent() {
  local candidate=$1 item
  for item in "${EXPECTED_ABSENT_STREAMS[@]}"; do
    [[ "$candidate" == "$item" ]] && return 0
  done
  return 1
}

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
  local count dry
  [[ -d "$BASELINE" && ! -L "$BASELINE" ]] || {
    printf 'REFUSE: pinned ORCA1-ice baseline missing\n' >&2
    exit 65
  }
  pin "$EXPECTED_DECK_MANIFEST_SHA256" "$BASELINE/deck_files.sha256" 'deck manifest'
  pin "$EXPECTED_INPUT_MANIFEST_SHA256" "$BASELINE/input_files.sha256" 'input manifest'
  pin "$EXPECTED_BASELINE_BINARY_SHA256" "$BASELINE/nemo" 'baseline binary'
  pin "$EXPECTED_PATCH_SHA256" "$PATCH" 'surface writer patch'
  pin "$EXPECTED_CPP_SHA256" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'CPP card'
  pin "$EXPECTED_ARCH_SHA256" "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" 'compiler card'
  pin "$EXPECTED_STPRK3_SHA256" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'source stprk3'
  pin 0ff0fc32c0845f23e093a2ad7957da89a3eef13ab1dc741005edc30e0785aa24 \
    "$FROZEN_ICE/icedyn_adv_pra.F90" 'Phase-2x icedyn_adv_pra'
  pin bf61835a80faaa38a49c49644a11a16fb45be1049caed8aa55cb224f62f5d7b1 \
    "$FROZEN_ICE/icedyn_rhg_evp.F90" 'Phase-2x icedyn_rhg_evp'
  pin b66823d02f35e60c9fa315c73ba9d51235b5bb88aa45247d0ed7142cb1d64bdf \
    "$FROZEN_ICE/icethd.F90" 'Phase-2x icethd'
  count=$(find "$BASELINE" -maxdepth 1 -type f -name 'oracle_*.bin' | wc -l)
  [[ "$count" -eq "$EXPECTED_BASELINE_STREAMS" ]] || {
    printf 'REFUSE: pinned baseline has %s oracle streams, expected %s\n' \
      "$count" "$EXPECTED_BASELINE_STREAMS" >&2
    exit 66
  }
  (cd "$BASELINE" && sha256sum -c deck_files.sha256 >/dev/null) || {
    printf 'REFUSE: pinned baseline deck manifest does not bind\n' >&2
    exit 66
  }
  (cd "$BASELINE" && sha256sum -c input_files.sha256 >/dev/null) || {
    printf 'REFUSE: pinned baseline input manifest does not bind\n' >&2
    exit 66
  }
  dry=$(mktemp -d /tmp/orca2-r1-surface-preflight.XXXXXXXX)
  cp "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$dry/stprk3.F90"
  (cd "$dry" && patch --dry-run -p0 <"$PATCH" >/dev/null) || {
    printf 'REFUSE: surface writer patch no longer applies to pinned stprk3\n' >&2
    exit 66
  }
  printf 'ORCA2_ROUND1_SURFACE_PREFLIGHT_READY %s\n' "$BASELINE"
}

compare_inherited() {
  local reference=$1 candidate=$2 name=$3 status=0
  if [[ "$name" == "$PARTIAL_STREAM" ]]; then
    [[ "$(stat -c %s "$reference/$name")" -eq "$PARTIAL_STREAM_BYTES" ]] || return 3
    [[ "$(stat -c %s "$candidate/$name")" -eq "$PARTIAL_STREAM_BYTES" ]] || return 3
    cmp -s -n "$PARTIAL_COMPARED_BYTES" "$reference/$name" "$candidate/$name" || status=$?
    return "$status"
  fi
  cmp -s "$reference/$name" "$candidate/$name" || status=$?
  return "$status"
}

validate_surface_schema() {
  local run=$1
  "$PYTHON" - "$run" <<'PY'
import pathlib
import struct
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
nx, ny, ntr, nclasses, halo = 94, 152, 2, 10, 2
classes = (20, 12, 1, 2)
reduced = (nx - 2 * halo) * (ny - 2 * halo)
halo1 = (nx - 2 * (halo - 1)) * (ny - 2 * (halo - 1))
count = classes[0] * nx * ny + (classes[1] + classes[3] * ntr) * reduced + classes[2] * halo1
for rank in range(2):
    for kt in range(1, 11):
        name = (f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
                f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin")
        path = root / name
        if not path.is_file():
            raise SystemExit(f"missing {path.name}")
        with path.open("rb") as handle:
            magic = handle.read(16).decode("ascii").rstrip()
            header = struct.unpack("=13i", handle.read(52))
            values = np.fromfile(handle, dtype=np.float64)
        level = 1 if kt % 2 else 3
        wanted = (1, kt, level, nx, ny, ntr, nclasses, halo, *classes, 64)
        if magic != "NEMO_L4_SBCIN_1" or header != wanted:
            raise SystemExit(f"bad schema {path.name}: {magic!r} {header}")
        if values.size != count or not np.isfinite(values).all():
            raise SystemExit(f"bad payload/EOF {path.name}: {values.size}/{count}")
PY
}

validate_entry_schema() {
  local run=$1
  "$PYTHON" - "$run" <<'PY'
import pathlib
import struct
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
nx, ny, nz, ntr = 94, 152, 31, 2
count = 4 * nx * ny * nz + nx * ny
for rank in range(2):
    for kt in range(1, 11):
        name = (f"oracle_step_entry_kt{kt:08d}.bin" if rank == 0 else
                f"oracle_step_entry_rank{rank:04d}_kt{kt:08d}.bin")
        path = root / name
        if not path.is_file():
            raise SystemExit(f"missing {path.name}")
        with path.open("rb") as handle:
            magic = handle.read(16).decode("ascii").rstrip()
            header = struct.unpack("=8i", handle.read(32))
            values = np.fromfile(handle, dtype=np.float64)
        level = 1 if kt % 2 else 3
        wanted = (1, kt, level, nx, ny, nz, ntr, 64)
        if magic != "NEMO_L1_ENTRY_1" or header != wanted:
            raise SystemExit(f"bad schema {path.name}: {magic!r} {header}")
        if values.size != count or not np.isfinite(values).all():
            raise SystemExit(f"bad payload/EOF {path.name}: {values.size}/{count}")
PY
}

ordinary_identity() {
  local reference=$1 candidate=$2
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" - "$reference" "$candidate" <<'PY'
import pathlib
import sys
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2y_orca1ice_admission_gate as admission,
)
admission._ordinary_identity(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
PY
}

finalize() {
  local count_a count_b inherited=0 name kt cmp_status
  [[ -d "$RUN_A" && ! -L "$RUN_A" && -d "$RUN_B" && ! -L "$RUN_B" ]] || {
    printf 'REFUSE: twin acquisition directories are absent or symlinks\n' >&2
    exit 65
  }
  for run in "$RUN_A" "$RUN_B"; do
    [[ "$(tr -d '[:space:]' <"$run/time.step")" == 10 ]] || {
      printf 'REFUSE: %s did not finish kt=10\n' "$run" >&2
      exit 66
    }
    grep -Fq 'RUN DONE' "$run/run.user.time.log" || {
      printf 'REFUSE: %s lacks RUN DONE\n' "$run" >&2
      exit 66
    }
    [[ "$(grep -c 'LANE4_OCEAN_SURFACE_INPUT_DUMP' "$run/ocean.output")" -eq 10 ]] || {
      printf 'REFUSE: %s lacks exactly ten surface dump notices\n' "$run" >&2
      exit 66
    }
    validate_surface_schema "$run" || {
      printf 'REFUSE: %s surface frame schema failed\n' "$run" >&2
      exit 66
    }
    validate_entry_schema "$run" || {
      printf 'REFUSE: %s entry frame schema failed\n' "$run" >&2
      exit 66
    }
  done
  count_a=$(find "$RUN_A" -maxdepth 1 -type f -name 'oracle_*.bin' | wc -l)
  count_b=$(find "$RUN_B" -maxdepth 1 -type f -name 'oracle_*.bin' | wc -l)
  [[ "$count_a" -eq "$EXPECTED_TARGET_STREAMS" && "$count_b" -eq "$EXPECTED_TARGET_STREAMS" ]] || {
    printf 'REFUSE: target stream counts are A=%s B=%s, expected %s each\n' \
      "$count_a" "$count_b" "$EXPECTED_TARGET_STREAMS" >&2
    exit 66
  }
  while IFS= read -r path; do
    name=$(basename "$path")
    if is_expected_absent "$name"; then
      [[ ! -e "$RUN_A/$name" && ! -e "$RUN_B/$name" ]] || {
        printf 'REFUSE: historically absent writer unexpectedly returned: %s\n' "$name" >&2
        exit 66
      }
      continue
    fi
    [[ -f "$RUN_A/$name" && -f "$RUN_B/$name" ]] || {
      printf 'REFUSE: inherited stream missing from target twins: %s\n' "$name" >&2
      exit 66
    }
    cmp -s "$RUN_A/$name" "$RUN_B/$name" || {
      printf 'REFUSE: target twins differ: %s\n' "$name" >&2
      exit 66
    }
    compare_inherited "$BASELINE" "$RUN_A" "$name" || cmp_status=$?
    cmp_status=${cmp_status:-0}
    [[ "$cmp_status" -eq 0 ]] || {
      printf 'REFUSE: inherited passivity differs for %s (status %s)\n' "$name" "$cmp_status" >&2
      exit 66
    }
    inherited=$((inherited + 1))
    cmp_status=0
  done < <(find "$BASELINE" -maxdepth 1 -type f -name 'oracle_*.bin' | sort)
  [[ "$inherited" -eq "$EXPECTED_INHERITED_STREAMS" ]] || {
    printf 'REFUSE: compared %s inherited streams, expected %s\n' \
      "$inherited" "$EXPECTED_INHERITED_STREAMS" >&2
    exit 66
  }
  for kt in $(seq 2 10); do
    name=$(printf 'oracle_ocean_surface_input_kt%08d.bin' "$kt")
    cmp -s "$RUN_A/$name" "$RUN_B/$name" || {
      printf 'REFUSE: target twins differ: %s\n' "$name" >&2
      exit 66
    }
  done
  for kt in $(seq 1 10); do
    name=$(printf 'oracle_ocean_surface_input_rank0001_kt%08d.bin' "$kt")
    cmp -s "$RUN_A/$name" "$RUN_B/$name" || {
      printf 'REFUSE: target twins differ: %s\n' "$name" >&2
      exit 66
    }
  done
  for kt in $(seq 1 10); do
    name=$(printf 'oracle_step_entry_rank0001_kt%08d.bin' "$kt")
    cmp -s "$RUN_A/$name" "$RUN_B/$name" || {
      printf 'REFUSE: target twins differ: %s\n' "$name" >&2
      exit 66
    }
  done
  ordinary_identity "$BASELINE" "$RUN_A" || {
    printf 'REFUSE: A ordinary outputs are not passive to the pinned root\n' >&2
    exit 66
  }
  ordinary_identity "$BASELINE" "$RUN_B" || {
    printf 'REFUSE: B ordinary outputs are not passive to the pinned root\n' >&2
    exit 66
  }
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" - "$RUN_A" "$RUN_B" "$inherited" <<'PY'
import hashlib
import json
import pathlib
import sys

from legoesm.ocean.fidelity.provenance import worktree_stamp

a, b = map(pathlib.Path, sys.argv[1:3])
inherited = int(sys.argv[3])
rows = []
for path in sorted(a.glob("oracle_ocean_surface_input*.bin")):
    other = b / path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != hashlib.sha256(other.read_bytes()).hexdigest():
        raise SystemExit(f"twin digest mismatch: {path.name}")
    rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
entry_rows = []
for path in sorted(a.glob("oracle_step_entry_rank*.bin")):
    other = b / path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != hashlib.sha256(other.read_bytes()).hexdigest():
        raise SystemExit(f"twin digest mismatch: {path.name}")
    entry_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
producer = {}
for name in ("nemo", "compiled_iceistate.f90", "compiled_stprk3.f90"):
    digest = hashlib.sha256((a / name).read_bytes()).hexdigest()
    if digest != hashlib.sha256((b / name).read_bytes()).hexdigest():
        raise SystemExit(f"twin producer differs: {name}")
    producer[name] = digest
result = {
    "worktree": worktree_stamp(),
    "status": "PASS",
    "root_label": "VARIANT_ORACLE_ORCA1ICE_ROUND5_FULL_ENTRY_INPUTS",
    "inherited_streams_passive": inherited,
    "new_surface_frames": 19,
    "surface_frames_total": len(rows),
    "twin_surface_frames_raw_exact": True,
    "surface_frames": rows,
    "new_rank1_entry_frames": len(entry_rows),
    "entry_frames_total": 10 + len(entry_rows),
    "twin_rank1_entry_frames_raw_exact": True,
    "rank1_entry_frames": entry_rows,
    "producer_sha256": producer,
}
rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
(a / "round1_surface_admission.json").write_text(rendered)
print(rendered, end="")
PY
  printf 'ORCA2_ROUND5_FULL_ENTRY_ACQUISITION_PASS %s %s\n' "$RUN_A" "$RUN_B"
}

stage_run() {
  local target=$1 digest name
  mkdir "$target"
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
  cp "$COMPILED_ICEISTATE" "$target/compiled_iceistate.f90"
  cp "$COMPILED_STPRK3" "$target/compiled_stprk3.f90"
  (cd "$target" && sha256sum -c deck_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged deck differs in %s\n' "$target" >&2
    exit 66
  }
  (cd "$target" && sha256sum -c input_files.sha256 >/dev/null) || {
    printf 'REFUSE: staged inputs differ in %s\n' "$target" >&2
    exit 66
  }
  sha256sum "$target/nemo" >"$target/binary.sha256"
  sha256sum "$target/compiled_iceistate.f90" "$target/compiled_stprk3.f90" \
    >"$target/compiled_source.sha256"
}

run_one() {
  local run=$1 mpi_rc tee_rc
  (
    cd "$run"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
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
    if [[ "$mpi_rc" -ne 0 || "$tee_rc" -ne 0 ]]; then
      printf 'RUN FAILED\n' >>run.user.time.log
      printf 'REFUSE: NEMO twin failed in %s (mpi=%s tee=%s)\n' \
        "$run" "$mpi_rc" "$tee_rc" >&2
      exit 68
    fi
    printf 'RUN DONE\n' >>run.user.time.log
  )
}

preflight
if [[ "$MODE" == --preflight-only ]]; then
  exit 0
fi
if [[ "$MODE" == --finalize ]]; then
  finalize
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$RUN_A" && ! -e "$RUN_B" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2
  exit 64
}
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: operator must create the non-symlink evidence directory first: %s\n' \
    "$EVIDENCE" >&2
  exit 64
}
for mount in "$NEMO_ROOT" "$EVIDENCE"; do
  [[ "$(df -Pk "$mount" | awk 'NR==2 {print $4}')" -ge 8388608 ]] || {
    printf 'REFUSE: %s has under 8 GiB free\n' "$mount" >&2
    exit 67
  }
done

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not create the isolated target\n' >&2
  exit 68
}
while IFS= read -r -d '' source; do
  name=$(basename "$source")
  # Phase-2z icesbc is outside this record.  Phase-2x icethd and dynamics are
  # restored from frozen acquisition sources below.
  [[ "$name" == icesbc.F90 ]] && continue
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$name"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 -type f -print0 | sort -z)
cp "$FROZEN_ICE/icedyn_adv_pra.F90" "$TARGET_ROOT/MY_SRC/icedyn_adv_pra.F90"
cp "$FROZEN_ICE/icedyn_rhg_evp.F90" "$TARGET_ROOT/MY_SRC/icedyn_rhg_evp.F90"
cp "$FROZEN_ICE/icethd.F90" "$TARGET_ROOT/MY_SRC/icethd.F90"
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
(cd "$TARGET_ROOT/MY_SRC" && patch -p0 <"$PATCH") || {
  printf 'REFUSE: write-only surface patch failed\n' >&2
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
[[ -f "$COMPILED_ICEISTATE" ]] || {
  printf 'REFUSE: target build produced no compiled iceistate branch\n' >&2
  exit 68
}
for marker in \
  'CALL l4_dump_ocean_surface_input( kstp, Nbb )' \
  'WRITE(cl_surface_file' \
  'WRITE(cl_traj' \
  'oracle_step_entry_rank' \
  'mpprank, kstp' \
  "STATUS='NEW'"; do
  grep -Fq "$marker" "$COMPILED_STPRK3" || {
    printf 'REFUSE: compiled source lacks marker: %s\n' "$marker" >&2
    exit 68
  }
done
if grep -Fq 'IF( lwp .AND. kstp >= nit000' "$COMPILED_STPRK3"; then
  printf 'REFUSE: compiled source retained the root-only step-entry writer\n' >&2
  exit 68
fi
if sed -n '/SUBROUTINE l4_dump_ocean_surface_input/,/END SUBROUTINE l4_dump_ocean_surface_input/p' \
    "$COMPILED_STPRK3" | grep -Fq 'IF( .NOT.lwp ) RETURN' || \
    grep -Fq 'IF( kstp == nit000 )   CALL l4_dump_ocean_surface_input' "$COMPILED_STPRK3"; then
  printf 'REFUSE: compiled source retained a root-only or kt=1-only surface call\n' >&2
  exit 68
fi
for marker in \
  'snwice_mass  (:,:) = tmask(:,:,1) * SUM' \
  'ssh(:,:,Kbb) = ssh(:,:,Kbb) - zsshadj'; do
  grep -Fq "$marker" "$COMPILED_ICEISTATE" || {
    printf 'REFUSE: compiled iceistate source lacks marker: %s\n' "$marker" >&2
    exit 68
  }
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in the acquisition binary\n' >&2
  exit 68
fi

stage_run "$RUN_A"
stage_run "$RUN_B"
sha256sum "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$PATCH" \
  "$FROZEN_ICE/icedyn_adv_pra.F90" "$FROZEN_ICE/icedyn_rhg_evp.F90" \
  "$FROZEN_ICE/icethd.F90" >"$RUN_A/acquisition_sources.sha256"
cp "$RUN_A/acquisition_sources.sha256" "$RUN_B/acquisition_sources.sha256"
run_one "$RUN_A"
run_one "$RUN_B"
finalize
