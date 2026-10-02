#!/usr/bin/env bash
# ORCA2-DECKS round 16: Decision-83 uniform-background rung-3 reacquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 Decision-83 rung-3 acquisition failed at line %s (exit %s)\n' \
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
readonly UPPER=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung4/record
readonly UPPER_ADMISSION=$UPPER/../rung4_admission.json
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung3
readonly OLD_RECORD=$EVIDENCE/record
readonly SUPERSEDED_RECORD=$EVIDENCE/record_havtb1_superseded
readonly OLD_DECK=$EVIDENCE/deck
readonly PENDING_DECK=$EVIDENCE/deck_havtb0_pending
readonly SUPERSEDED_DECK=$EVIDENCE/deck_havtb1_superseded
readonly OLD_ADMISSION=$EVIDENCE/rung3_admission.json
readonly SUPERSEDED_ADMISSION=$EVIDENCE/rung3_admission_havtb1_superseded.json
readonly RUN=$EVIDENCE/record
readonly CURRENT_ADMISSION=$EVIDENCE/rung3_admission.json
readonly UPPER_ADMISSION_SHA=b337142222c0a0d9fc44539abc55c25c8df09ee5e2155022cbfc14676bb88089
readonly OLD_ADMISSION_SHA=5e4ce3e944b8561e46f472b0109006c435f5dfa36552bce1668c8e1eef39238e
readonly OLD_INVENTORY_SHA=0209b7c848d7ffc31df7d289c08e7accf483101b2b8dc66c11f54b2dbf20d2eb
readonly OLD_CFG_SHA=6dbb38d77721e49dec63e452aca2c6e58cd159028a23eb5643016c7b40f6d6ab
readonly OLD_EXEC_SHA=ef1894f7a55f4ac753523287daf9bd1d2c9910390e5e6169dbb9f75f320ca391
readonly NEW_CFG_SHA=0362b8cb572758ed20b7fceba5f151ae5dd4d8cb6b4352eb12e5cc21e7836d74
readonly NEW_EXEC_SHA=20866bcbfe851191ae91728f5e3f364d87563f7781cbfaff81bfa5d457f27469
readonly BINARY_SHA=cee66aec4a9a9c2b85e755af0250c9b27cccef6ed7333b454a742af0692bc9ec
readonly WRITER_SHA=81304be81526980fb9f80549fdfd02c20df293b7a1a0725e94ef9b7e25c88c72
readonly STPRK3_SHA=b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422
readonly MANIFEST_SHA=737810f46c93585f39cbf10dee132a81d4bca40a21fb9e50aa5c437730caef97

if [[ -d "$SUPERSEDED_RECORD" ]]; then
  readonly PINNED_OLD_RECORD=$SUPERSEDED_RECORD
  readonly PINNED_OLD_DECK=$SUPERSEDED_DECK
  readonly PINNED_OLD_ADMISSION=$SUPERSEDED_ADMISSION
else
  readonly PINNED_OLD_RECORD=$OLD_RECORD
  readonly PINNED_OLD_DECK=$OLD_DECK
  readonly PINNED_OLD_ADMISSION=$OLD_ADMISSION
fi

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly GATE=$here/../nemo_testcase_l4_orca2_hier_decks_round16_gate.py
readonly MANIFEST=$here/rung3_havtb0_manifest.json
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_hier_decks_round16.md

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
pin "$UPPER_ADMISSION_SHA" "$UPPER_ADMISSION" 'replacement rung-4 admission'
pin "$OLD_ADMISSION_SHA" "$PINNED_OLD_ADMISSION" 'superseded rung-3 admission'
pin "$OLD_INVENTORY_SHA" "$PINNED_OLD_RECORD/SHA256SUMS" 'superseded rung-3 inventory'
pin "$OLD_CFG_SHA" "$PINNED_OLD_RECORD/namelist_cfg.deck" 'superseded exact deck'
pin "$OLD_EXEC_SHA" "$PINNED_OLD_RECORD/namelist_cfg" 'superseded execution deck'
pin "$BINARY_SHA" "$PINNED_OLD_RECORD/nemo" 'repaired recorder binary'
pin "$WRITER_SHA" "$PINNED_OLD_RECORD/compiled_l4_r69_surface.f90" 'compiled recorder'
pin "$STPRK3_SHA" "$PINNED_OLD_RECORD/compiled_stprk3.f90" 'compiled step program'
pin "$MANIFEST_SHA" "$MANIFEST" 'replacement manifest'

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory\n' >&2; exit 65;
}
bash -n "$0"
"$PY" -m py_compile "$GATE"
if [[ "$MODE" == --admit-existing ]]; then
  "$PY" "$GATE" --preflight-only --output "$EVIDENCE/round16_preflight.json" >/dev/null
else
  "$PY" "$GATE" --preflight-only --stage-deck "$PENDING_DECK" \
    --output "$EVIDENCE/round16_preflight.json" >/dev/null
fi
for plant in deck-extra missing-havtb retained-coefficient source-pin superseded-pin upper-pin; do
  if "$PY" "$GATE" --preflight-only --plant "$plant" \
    >"$EVIDENCE/round16_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s preflight plant stayed green\n' "$plant" >&2; exit 72;
  fi
  grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round16_${plant}_plant.log"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_HIERARCHY_RUNG3_HAVTB0_PREFLIGHT_READY %s\n' "$RUN"
  exit 0
fi

admit() {
  local expected_commit plant
  expected_commit=$(<"$RUN/producer_commit.txt")
  for plant in field-name truncated frame-nonfinite absent-as-zero owner-on missing-frame \
    terminal-nonfinite terminal-step ice-sentinel-read tke-sentinel-read \
    resolved-consequence runoff-group-unread active-runoff-print shortwave-consequence \
    sha-inventory zero-flux-nonzero zero-flux-missing surface-consequence resolved-havtb; do
    if "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
      --plant "$plant" >"$EVIDENCE/round16_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s admission plant stayed green\n' "$plant" >&2; exit 72;
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round16_${plant}_plant.log"
  done
  "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
    --output "$CURRENT_ADMISSION"
  (cd "$EVIDENCE" && sha256sum deck/SHA256SUMS round16_preflight.json \
    rung3_admission.json round16_*_plant.log >round16_SHA256SUMS)
  printf 'ORCA2_HIERARCHY_RUNG3_HAVTB0_ACQUISITION_PASS %s\n' "$RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$RUN" && -d "$SUPERSEDED_RECORD" ]] || {
    printf 'REFUSE: replacement or superseded record is absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$SUPERSEDED_RECORD" && ! -e "$SUPERSEDED_DECK" \
   && ! -e "$SUPERSEDED_ADMISSION" ]] || {
  printf 'REFUSE: a preservation target already exists; use --admit-existing only after a completed run\n' >&2; exit 68;
}
[[ -d "$OLD_RECORD" && -d "$OLD_DECK" && -d "$PENDING_DECK" ]] || {
  printf 'REFUSE: source evidence or staged replacement deck is absent\n' >&2; exit 68;
}
(cd "$OLD_RECORD" && sha256sum -c SHA256SUMS >/dev/null)
for mount in "$EVIDENCE" /tmp; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

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
while IFS= read -r -d '' source; do cp -a "$source" "$RUN/"; done \
  < <(find "$SUPERSEDED_RECORD" -maxdepth 1 -type f -name 'compiled_*.f90' -print0)
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
  [[ "$(tr -d '[:space:]' <time.step)" == 240 ]] || {
    printf 'REFUSE: NEMO did not reach step 240\n' >&2; exit 70;
  }
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' | sort -z | xargs -0 sha256sum >SHA256SUMS
)
admit
