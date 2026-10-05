#!/usr/bin/env bash
# ORCA2 round-79b acquisition: NEMO's own per-step vertical-mixing chain.
#
# It records, for the first ten steps and on both ranks, the diffusivities at
# the five boundaries of the compiled zdf_phy in NEMO's execution order --
# after the turbulent closure, after the river-mouth enhancement, after the
# enhanced-diffusion convection, after the double-diffusive split and after
# the internal-wave arm -- together with the closure internals (turbulent
# energy, the two mixing lengths, the dissipation length scale).  The operator
# runs this file; the agent does not invoke mpirun in the sandbox.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-79b vertical-mixing acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R79BZDF
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round79b/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_zdf_vmix_10step_np2
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PHY_PATCH=$here/zdfphy_round79b.patch
readonly TKE_PATCH=$here/zdftke_round79b.patch
readonly WRITER=$here/zdfphy_round79b_writer.F90
readonly GATE=$here/../nemo_testcase_l4_orca2_round79b_vertical_mixing_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round79b.md
readonly SOURCE_PHY=$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90
readonly SOURCE_TKE=$NEMO_ROOT/src/OCE/ZDF/zdftke.F90

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
# The repo root itself must be on PYTHONPATH: the gate imports through the
# scripts package (round-67 defect).
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

for path in "$PHY_PATCH" "$TKE_PATCH" "$WRITER" "$GATE" "$PREREG" \
  "$SOURCE_PHY" "$SOURCE_TKE" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing input %s\n' "$path" >&2; exit 64; }
done
pin 76eeec2a280d56c55bec9e59b579e5290623c408c03962d04652e406334b45fd "$SOURCE_PHY" 'source zdfphy'
pin 2e78c1a215e4a7ee51dab4b3b19ef446cceb1ef8e5ad9380dd7c482d37e10397 "$SOURCE_TKE" 'source zdftke'
pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$SOURCE_RUN/nemo" 'source run binary'
pin 51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9 "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin 3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5 "$SOURCE_RUN/input_files.sha256" 'input manifest'
grep -Eq 'number of the last time step.*nn_itend *= *10' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run is not ten steps\n' >&2; exit 65;
}
grep -Eq 'internal wave \(de Lavergne et al 2017\) *ln_zdfiwm *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run does not execute internal-wave mixing\n' >&2; exit 65;
}
grep -Eq 'double diffusive mixing *ln_zdfddm *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run does not execute double-diffusive mixing\n' >&2; exit 65;
}
grep -Eq 'force +rn_emin = 1.e-10 and rmxl_min = 1.e-3' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run did not take the forced mixing-length floor arm\n' >&2; exit 65;
}
grep -Eq 'Tiling \(T\) or not \(F\).*ln_tile *= *F' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run enables unsupported tiling\n' >&2; exit 65;
}

for patch in "$PHY_PATCH" "$TKE_PATCH"; do
  removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$patch")
  [[ "$removed" -eq 0 ]] || { printf 'REFUSE: additions-only patch removes %s lines: %s\n' "$removed" "$patch" >&2; exit 66; }
done
dry=$(mktemp -d /tmp/orca2-r79b-zdf.XXXXXX)
cp "$SOURCE_PHY" "$dry/zdfphy.F90"
cp "$SOURCE_TKE" "$dry/zdftke.F90"
cp "$WRITER" "$dry/zdfphy_round79b_writer.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$PHY_PATCH"
patch -s --fuzz=0 -p0 -d "$dry" <"$TKE_PATCH"
[[ "$(grep -Fc 'CALL zdf_iwm(' "$dry/zdfphy.F90")" -eq 1 ]] || { printf 'REFUSE: internal-wave call count changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'CALL zdf_ddm(' "$dry/zdfphy.F90")" -eq 1 ]] || { printf 'REFUSE: double-diffusion call count changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'CALL zdf_evd(' "$dry/zdfphy.F90")" -eq 1 ]] || { printf 'REFUSE: enhanced-diffusion call count changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'CALL tke_avn(' "$dry/zdftke.F90")" -eq 1 ]] || { printf 'REFUSE: closure call count changed\n' >&2; exit 66; }
[[ "$(grep -Ec 'R79_3D\(|R79_2D\(|R79_1D\(' "$dry/zdfphy_round79b_writer.F90")" -eq 27 ]] || {
  printf 'REFUSE: writer field census changed\n' >&2; exit 66;
}

mkdir -p "$EVIDENCE"
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 8388608 ]] || { printf 'REFUSE: %s has under 8 GB free\n' "$mount" >&2; exit 67; }
done

syntax=$(mktemp -d /tmp/orca2-r79b-syntax.XXXXXX)
cp "$SOURCE_ROOT"/BLD/inc/*.mod "$syntax/"
for unit in zdfphy_round79b_writer zdftke zdfphy; do
  cpp -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
    -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$dry/$unit.F90" -o "$syntax/$unit.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" -J "$syntax" "$syntax/$unit.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$unit"
done
"$PY" -m pytest -q -p no:cacheprovider \
  "$REPO/tests/ocean/fidelity/test_orca2_round79b_vertical_mixing_gate.py"
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND79B_ZDF_VMIX_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant
  [[ -f "$TARGET_RUN/producer_commit.txt" ]] || { printf 'REFUSE: missing producer commit\n' >&2; exit 70; }
  [[ "$(cat "$TARGET_RUN/producer_commit.txt")" == "$COMMIT" ]] || { printf 'REFUSE: producer commit changed\n' >&2; exit 70; }
  for plant in header field-order truncation stamp content; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
      --plant "$plant" >"$TARGET_RUN/round79b_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -q 'STATUS PLANT-FIRED' "$TARGET_RUN/round79b_${plant}_plant.log"
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
    --output "$TARGET_RUN/round79b_zdf_vmix_admission.json"
  ( cd "$TARGET_RUN" && sha256sum oracle_zdf_vmix_kt*.bin ORCA2_00000010_restart*.nc \
      round79b_zdf_vmix_admission.json round79b_*_plant.log >round79b_outputs.sha256 )
  printf 'ORCA2_ROUND79B_ZDF_VMIX_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || { printf 'REFUSE: staged binary differs\n' >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 64;
}
manifest=$(mktemp -d /tmp/orca2-r79b-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PHY_PATCH" "$TKE_PATCH" "$WRITER" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfphy.f90" "$SOURCE_ROOT/BLD/ppsrc/nemo/zdftke.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfiwm.f90" "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfddm.f90" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$SOURCE_PHY" "$TARGET_ROOT/MY_SRC/zdfphy.F90"
cp "$SOURCE_TKE" "$TARGET_ROOT/MY_SRC/zdftke.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/zdfphy_round79b_writer.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PHY_PATCH"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$TKE_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 68; }
grep -q 'NEMO_L4_ZDFV_1' "$TARGET_ROOT/BLD/ppsrc/nemo/zdfphy_round79b_writer.f90" || {
  printf 'REFUSE: compiled writer is absent\n' >&2; exit 68;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector math symbol present\n' >&2; exit 68; fi

mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
( cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipe_rc[0]}" -eq 0 ]] || { printf 'REFUSE: mpirun exited %s\n' "${pipe_rc[0]}" >&2; exit 69; }
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep -q 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || { printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 69; }
# Note-AS admission: the write-only patch must leave the trajectory alone, so
# every restart file is byte-identical to the pinned ten-step record.
for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \
  ORCA2_00000010_restart_ice_0000.nc ORCA2_00000010_restart_ice_0001.nc; do
  cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name" || { printf 'REFUSE: calibration changed: %s\n' "$name" >&2; exit 69; }
done
admit
