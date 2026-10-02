#!/usr/bin/env bash
# ORCA2-DECKS round 17: Decision-83 uniform-background rung-2 reacquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 Decision-83 rung-2 acquisition failed at line %s (exit %s)\n' \
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
readonly UPPER=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung3/record
readonly UPPER_ADMISSION=$UPPER/../rung3_admission.json
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung2
readonly OLD_RECORD=$EVIDENCE/record
readonly SUPERSEDED_RECORD=$EVIDENCE/record_havtb1_superseded
readonly OLD_DECK=$EVIDENCE/deck
readonly PENDING_DECK=$EVIDENCE/deck_havtb0_pending
readonly SUPERSEDED_DECK=$EVIDENCE/deck_havtb1_superseded
readonly OLD_ADMISSION=$EVIDENCE/rung2_admission.json
readonly SUPERSEDED_ADMISSION=$EVIDENCE/rung2_admission_havtb1_superseded.json
readonly RUN=$EVIDENCE/record
readonly CURRENT_ADMISSION=$EVIDENCE/rung2_admission.json
readonly UPPER_ADMISSION_SHA=591bce28aea5626ea009c674271061ae5f0d4c7959df2ef879bd9b36134fb4b0
readonly OLD_ADMISSION_SHA=b1553d791db7df6ade5d9f3da06790e789a21b0be2ab4758b88e27f21509003e
readonly OLD_INVENTORY_SHA=72b65f0fcb930fb928ad97f3cbc490bd18f66fc6640970e59e0ee61d8c21607b
readonly OLD_CFG_SHA=fa8d0a34d6f3bce8cce4c98ae56fefa9d2a0f5e51e4ba582aa695362154d75b7
readonly OLD_EXEC_SHA=cf42f174857eef17f3654518357e28f4c0f73f8e25e7c0a4b07e77813bfbf39a
readonly NEW_CFG_SHA=b4b8cb87249bf41ef41ea13637273ef097f6632ae4879dca883feebdd9011e38
readonly NEW_EXEC_SHA=d691aec39430a952bc4cff73e3af2487fe10a4cf728111fe3ec8fc66609c0416
readonly BINARY_SHA=cee66aec4a9a9c2b85e755af0250c9b27cccef6ed7333b454a742af0692bc9ec
readonly WRITER_SHA=81304be81526980fb9f80549fdfd02c20df293b7a1a0725e94ef9b7e25c88c72
readonly STPRK3_SHA=b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422
readonly MANIFEST_SHA=28017071068403a9d5f30914d31f15618a1bd7ea4411d9d79a53dae158cefc68

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
readonly GATE=$here/../nemo_testcase_l4_orca2_hier_decks_round17_gate.py
readonly MANIFEST=$here/rung2_havtb0_manifest.json
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_hier_decks_round17.md

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
pin "$UPPER_ADMISSION_SHA" "$UPPER_ADMISSION" 'replacement rung-3 admission'
pin "$OLD_ADMISSION_SHA" "$PINNED_OLD_ADMISSION" 'superseded rung-2 admission'
pin "$OLD_INVENTORY_SHA" "$PINNED_OLD_RECORD/SHA256SUMS" 'superseded rung-2 inventory'
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
  "$PY" "$GATE" --preflight-only --output "$EVIDENCE/round17_preflight.json" >/dev/null
else
  "$PY" "$GATE" --preflight-only --stage-deck "$PENDING_DECK" \
    --output "$EVIDENCE/round17_preflight.json" >/dev/null
fi
for plant in deck-extra missing-havtb retained-coefficient source-pin superseded-pin upper-pin; do
  if "$PY" "$GATE" --preflight-only --plant "$plant" \
    >"$EVIDENCE/round17_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s preflight plant stayed green\n' "$plant" >&2; exit 72;
  fi
  grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round17_${plant}_plant.log"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_HIERARCHY_RUNG2_HAVTB0_PREFLIGHT_READY %s\n' "$RUN"
  exit 0
fi

admit() {
  local expected_commit plant
  expected_commit=$(<"$RUN/producer_commit.txt")
  for plant in field-name truncated frame-nonfinite absent-as-zero owner-on missing-frame \
    terminal-nonfinite terminal-step ice-sentinel-read tke-sentinel-read \
    resolved-consequence runoff-group-unread active-runoff-print shortwave-consequence \
    sha-inventory zero-flux-nonzero zero-flux-missing surface-consequence \
    gm-mle-consequence resolved-havtb; do
    if "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
      --plant "$plant" >"$EVIDENCE/round17_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s admission plant stayed green\n' "$plant" >&2; exit 72;
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round17_${plant}_plant.log"
  done
  "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
    --output "$CURRENT_ADMISSION"
  (cd "$EVIDENCE" && sha256sum deck/SHA256SUMS round17_preflight.json \
    rung2_admission.json round17_*_plant.log >round17_SHA256SUMS)
  printf 'ORCA2_HIERARCHY_RUNG2_HAVTB0_ACQUISITION_PASS %s\n' "$RUN"
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
