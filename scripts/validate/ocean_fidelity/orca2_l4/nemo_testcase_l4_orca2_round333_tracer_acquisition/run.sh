#!/usr/bin/env bash
# Operator-only extension of the reviewed round-328/332 acquisition driver.
# The default is a read-only preflight. Codex never invokes --run.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run]\n' "$0" >&2; exit 64 ;;
esac

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PARENT_DIR=$here/../nemo_testcase_l4_orca2_round328_dynzdf_acquisition
readonly PARENT=$PARENT_DIR/run.sh
readonly PATCH=$here/stprk3_stg_round333.patch
readonly GATE=$here/../nemo_testcase_l4_orca2_round333_tracer_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round333_tracer_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_tracer_stage3_kt2_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

for path in "$PARENT" "$PATCH" "$GATE" "$TEST"; do
  [[ -f "$path" ]] || refuse "missing acquisition input $path"
done
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] \
  || refuse 'acquisition requires a clean committed tree'

# Import every reviewed helper, plant, syntax proof, same-program check and
# admission step from the round-328/332 driver. Five pinned substitutions give
# that driver a fresh config/run and this additions-only stage observer. The
# generated driver is evidence-local, never /tmp, and its diff is audited.
mkdir -p "$EVIDENCE"
driver_dir=$(mktemp -d "$EVIDENCE/.r333-driver.XXXXXX")
cleanup() {
  local rc=$?
  trap - EXIT
  rm -rf -- "$driver_dir"
  exit "$rc"
}
trap cleanup EXIT
readonly driver_dir
readonly ADAPTED=$driver_dir/run.sh
sed \
  -e "s|^here=\$(CDPATH= cd -- \"\$(dirname -- \"\$0\")\" && pwd -P)$|here=$PARENT_DIR|" \
  -e 's|TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R332RAW|TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R333TR|' \
  -e 's|EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition|EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition|' \
  -e 's|DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_raw_postfold_v_kt1_np2|DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tracer_stage3_kt2_np2|' \
  -e "s|readonly R332_STAGE_PATCH=\$R332_DIR/stprk3_stg_round332.patch|readonly R332_STAGE_PATCH=$PATCH|" \
  "$PARENT" >"$ADAPTED"
chmod 700 "$ADAPTED"

if diff -u "$PARENT" "$ADAPTED" >"$driver_dir/import.diff"; then
  refuse 'round-333 driver import changed no pinned line'
else
  [[ $? -eq 1 ]] || refuse 'could not audit the imported driver'
fi
changed=$(awk '
  /^--- |^\+\+\+ / { next }
  /^[-+]/ { count++ }
  END { print count+0 }
' "$driver_dir/import.diff")
[[ "$changed" -eq 10 ]] || refuse "imported driver changed $changed lines, expected 10"
for token in \
  "here=$PARENT_DIR" \
  'TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R333TR' \
  "EVIDENCE=$EVIDENCE" \
  'DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tracer_stage3_kt2_np2' \
  "readonly R332_STAGE_PATCH=$PATCH"; do
  [[ $(grep -Fxc "$token" "$ADAPTED") -eq 1 ]] \
    || refuse "adapted driver lacks unique substitution: $token"
done

export R328_ACQ_TARGET=tke332
"$ADAPTED" "$MODE"

export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
"$PY" -m pytest -q -p no:xdist "$TEST"
for plant in header truncation; do
  log=$driver_dir/$plant.log
  if "$PY" "$GATE" --self-test --plant "$plant" >"$log" 2>&1; then
    refuse "round-333 parser plant stayed green: $plant"
  fi
  grep -F "STATUS PLANT-FIRED" "$log" >/dev/null \
    || refuse "round-333 parser plant missed its refusal: $plant"
  printf 'PREFLIGHT_PLANT_FIRED r333-%s\n' "$plant"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND333_TRACER_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
for plant in header truncation stamp zero kmm; do
  log=$driver_dir/admit-$plant.log
  if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" \
      --plant "$plant" >"$log" 2>&1; then
    refuse "round-333 admission plant stayed green: $plant"
  fi
  grep -F "STATUS PLANT-FIRED" "$log" >/dev/null \
    || refuse "round-333 admission plant missed its refusal: $plant"
  printf 'ACQUISITION_PLANT_FIRED r333-%s\n' "$plant"
done
"$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" \
  --output "$TARGET_RUN/round333_tracer_admission.json"
printf 'ORCA2_ROUND333_TRACER_ACQUISITION_PASS %s\n' "$TARGET_RUN"
