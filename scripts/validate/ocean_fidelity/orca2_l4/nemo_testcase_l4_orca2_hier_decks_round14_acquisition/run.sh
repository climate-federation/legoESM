#!/usr/bin/env bash
# ORCA2-DECKS round 14: Decision-83 uniform-background rung-5 reacquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 Decision-83 rung-5 acquisition failed at line %s (exit %s)\n' \
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly UPPER=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6/record
readonly UPPER_ADMISSION=$UPPER/../rung6_admission.json
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5
readonly OLD_RECORD=$EVIDENCE/record_absent_v2
readonly SUPERSEDED_RECORD=$EVIDENCE/record_havtb1_superseded
readonly OLD_DECK=$EVIDENCE/deck
readonly PENDING_DECK=$EVIDENCE/deck_havtb0_pending
readonly SUPERSEDED_DECK=$EVIDENCE/deck_havtb1_superseded
readonly OLD_ADMISSION=$EVIDENCE/rung5_round8_admission.json
readonly SUPERSEDED_ADMISSION=$EVIDENCE/rung5_admission_havtb1_superseded.json
readonly FAILED_RECORD=$EVIDENCE/record
readonly FAILED_PRESERVED=$EVIDENCE/record_pre_absent_repair_failed
readonly RUN=$EVIDENCE/record
readonly CURRENT_ADMISSION=$EVIDENCE/rung5_admission.json
readonly UPPER_ADMISSION_SHA=0cc13243c3fb100af19e048e4890b07138081d4d690477a44b828e8580cd34f2
readonly OLD_ADMISSION_SHA=9f5de467d92d2b2b7514f8907604bf008e61606cbf69ab16a61e94e0b8289b70
readonly OLD_INVENTORY_SHA=879b8181feff012017ef217ac3c3b93056c2d54135a7d42a5566044f19aa8c7a
readonly OLD_CFG_SHA=f537e3d29a6e2472f89cc9a8d23ec70d18756e3ad112088e7256804698bc1d59
readonly OLD_EXEC_SHA=53b45d1c98af7554e254e25e5b07acbab56cb09ed22896cf090415b371f293b8
readonly NEW_CFG_SHA=9bb44379e04861da2c22d76e07afc2ac2a8a1ce82b3dcc262583b841d2e71ead
readonly NEW_EXEC_SHA=aa1c49f2c46181674b68ae3eb3aabd71b5590d973d785335cfdb1236869e7e70
readonly BINARY_SHA=cee66aec4a9a9c2b85e755af0250c9b27cccef6ed7333b454a742af0692bc9ec
readonly WRITER_SHA=81304be81526980fb9f80549fdfd02c20df293b7a1a0725e94ef9b7e25c88c72
readonly STPRK3_SHA=b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422
readonly MANIFEST_SHA=1b2cc07405f784afa91465f14626c88c6bb571991adcbde19b758c542e427f6e
readonly FAILED_CFG_SHA=f537e3d29a6e2472f89cc9a8d23ec70d18756e3ad112088e7256804698bc1d59
readonly FAILED_LOG_SHA=58340d707ce7aea147f2bfe4f05f6f68fa4df70891bf32ba1ebd96a7bdb165bd

if [[ -d "$SUPERSEDED_RECORD" ]]; then
  readonly PINNED_OLD_RECORD=$SUPERSEDED_RECORD
  readonly PINNED_OLD_DECK=$SUPERSEDED_DECK
  readonly PINNED_OLD_ADMISSION=$SUPERSEDED_ADMISSION
else
  readonly PINNED_OLD_RECORD=$OLD_RECORD
  readonly PINNED_OLD_DECK=$OLD_DECK
  readonly PINNED_OLD_ADMISSION=$OLD_ADMISSION
fi
if [[ -d "$FAILED_PRESERVED" ]]; then
  readonly PINNED_FAILED_RECORD=$FAILED_PRESERVED
else
  readonly PINNED_FAILED_RECORD=$FAILED_RECORD
fi

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly GATE=$here/../nemo_testcase_l4_orca2_hier_decks_round14_gate.py
readonly MANIFEST=$here/rung5_havtb0_manifest.json
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_hier_decks_round14.md

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

for path in "$0" "$GATE" "$MANIFEST" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 65;
  }
done
pin "$UPPER_ADMISSION_SHA" "$UPPER_ADMISSION" 'replacement rung-6 admission'
pin "$OLD_ADMISSION_SHA" "$PINNED_OLD_ADMISSION" 'superseded rung-5 admission'
pin "$OLD_INVENTORY_SHA" "$PINNED_OLD_RECORD/SHA256SUMS" 'superseded rung-5 inventory'
pin "$OLD_CFG_SHA" "$PINNED_OLD_RECORD/namelist_cfg.deck" 'superseded exact deck'
pin "$OLD_EXEC_SHA" "$PINNED_OLD_RECORD/namelist_cfg" 'superseded execution deck'
pin "$BINARY_SHA" "$PINNED_OLD_RECORD/nemo" 'repaired recorder binary'
pin "$WRITER_SHA" "$PINNED_OLD_RECORD/compiled_l4_r69_surface.f90" 'compiled recorder'
pin "$STPRK3_SHA" "$PINNED_OLD_RECORD/compiled_stprk3.f90" 'compiled step program'
pin "$MANIFEST_SHA" "$MANIFEST" 'replacement manifest'
pin "$FAILED_CFG_SHA" "$PINNED_FAILED_RECORD/namelist_cfg.deck" 'failed pre-repair deck'
pin "$FAILED_LOG_SHA" "$PINNED_FAILED_RECORD/run.user.stdout.log" 'failed pre-repair log'

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory\n' >&2; exit 65;
}
bash -n "$0"
"$PY" -m py_compile "$GATE"
if [[ "$MODE" == --admit-existing ]]; then
  "$PY" "$GATE" --preflight-only \
    --output "$EVIDENCE/round14_preflight.json" >/dev/null
else
  "$PY" "$GATE" --preflight-only --stage-deck "$PENDING_DECK" \
    --output "$EVIDENCE/round14_preflight.json" >/dev/null
fi
for plant in deck-extra missing-havtb retained-coefficient source-pin superseded-pin upper-pin; do
  if "$PY" "$GATE" --preflight-only --plant "$plant" \
    >"$EVIDENCE/round14_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s preflight plant stayed green\n' "$plant" >&2; exit 72;
  fi
  grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round14_${plant}_plant.log"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_HIERARCHY_RUNG5_HAVTB0_PREFLIGHT_READY %s\n' "$RUN"
  exit 0
fi

admit() {
  local expected_commit plant
  expected_commit=$(<"$RUN/producer_commit.txt")
  for plant in field-name truncated frame-nonfinite absent-as-zero owner-on missing-frame \
    terminal-nonfinite terminal-step ice-sentinel-read tke-sentinel-read \
    resolved-consequence resolved-havtb runoff-group-unread active-runoff-print \
    sha-inventory; do
    if "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
      --plant "$plant" >"$EVIDENCE/round14_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s admission plant stayed green\n' "$plant" >&2; exit 72;
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round14_${plant}_plant.log"
  done
  "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
    --output "$CURRENT_ADMISSION"
  (cd "$EVIDENCE" && sha256sum deck/SHA256SUMS round14_preflight.json \
    rung5_admission.json round14_*_plant.log >round14_SHA256SUMS)
  printf 'ORCA2_HIERARCHY_RUNG5_HAVTB0_ACQUISITION_PASS %s\n' "$RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$RUN" && -d "$SUPERSEDED_RECORD" && -d "$FAILED_PRESERVED" ]] || {
    printf 'REFUSE: replacement, superseded, or failed-record preservation is absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$SUPERSEDED_RECORD" && ! -e "$SUPERSEDED_DECK" \
   && ! -e "$SUPERSEDED_ADMISSION" && ! -e "$FAILED_PRESERVED" ]] || {
  printf 'REFUSE: a preservation target already exists; use --admit-existing only after a completed run\n' >&2; exit 68;
}
[[ -d "$OLD_RECORD" && -d "$OLD_DECK" && -d "$FAILED_RECORD" && -d "$PENDING_DECK" ]] || {
  printf 'REFUSE: source evidence or staged replacement deck is absent\n' >&2; exit 68;
}
(cd "$OLD_RECORD" && sha256sum -c SHA256SUMS >/dev/null)
for mount in "$EVIDENCE" /tmp; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

mv "$FAILED_RECORD" "$FAILED_PRESERVED"
mv "$OLD_RECORD" "$SUPERSEDED_RECORD"
mv "$OLD_DECK" "$SUPERSEDED_DECK"
mv "$OLD_ADMISSION" "$SUPERSEDED_ADMISSION"
mv "$PENDING_DECK" "$OLD_DECK"

mkdir "$RUN"
while read -r digest name; do
  case "$name" in
    namelist_cfg)
      cp -a "$SUPERSEDED_RECORD/namelist_cfg" "$RUN/namelist_cfg"
      sed -i 's/^   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)$/   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)/' "$RUN/namelist_cfg"
      ;;
    *) cp -a "$SUPERSEDED_RECORD/$name" "$RUN/$name" ;;
  esac
done <"$SUPERSEDED_RECORD/deck_files.sha256"
while read -r digest name; do cp -a "$SUPERSEDED_RECORD/$name" "$RUN/$name"; done \
  <"$SUPERSEDED_RECORD/input_files.sha256"
cp -a "$SUPERSEDED_RECORD/deck_files.sha256" "$SUPERSEDED_RECORD/input_files.sha256" "$RUN/"
cp -a "$OLD_DECK/namelist_cfg" "$RUN/namelist_cfg.deck"
cp -a "$MANIFEST" "$RUN/hierarchy_manifest.json"
cp -a "$SUPERSEDED_RECORD/recorder_repair_manifest.json" "$RUN/"
cp -a "$SUPERSEDED_RECORD/nemo" "$RUN/nemo"
cp -a "$SUPERSEDED_RECORD/compiled_l4_r69_surface.f90" "$RUN/"
cp -a "$SUPERSEDED_RECORD/compiled_stprk3.f90" "$RUN/"
printf '%s\n' "$COMMIT" >"$RUN/producer_commit.txt"
pin "$NEW_CFG_SHA" "$RUN/namelist_cfg.deck" 'replacement exact deck'
pin "$NEW_EXEC_SHA" "$RUN/namelist_cfg" 'replacement execution deck'
(cd "$RUN" && sha256sum -c input_files.sha256 >/dev/null)

(
  cd "$RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipe_rc[0]}" -eq 0 ]] || { printf 'REFUSE: mpirun exited %s\n' "${pipe_rc[0]}" >&2; exit 70; }
  [[ "${pipe_rc[1]:-0}" -eq 0 ]] || { printf 'REFUSE: tee exited %s\n' "${pipe_rc[1]}" >&2; exit 70; }
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  grep -q 'STOP 0' run.user.stdout.log || { printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 70; }
  [[ "$(tr -d '[:space:]' <time.step)" == 240 ]] || { printf 'REFUSE: NEMO did not reach step 240\n' >&2; exit 70; }
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' | sort -z | xargs -0 sha256sum >SHA256SUMS
)
admit
