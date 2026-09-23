# Current-default climate re-battery: exact SLOT handoff

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. **PREREGISTERED, NOT RUN.** These
blocks implement `PREREG_climate_rebattery_round94.md`. They launch no legacy
selector arm: each pair is an independent duplicate of the same current
faithful default.

Every block changes into its own checkout, uses absolute repository paths,
uses `grep`, exports checkout-local `PYTHONPATH`, pins the producer by SHA, and
uses a tracked-only porcelain gate. Block 1 refuses a pre-existing root,
requires 15 GiB free, and caps the checkout at 5 GiB; it never copies a NEMO
source or run archive.

## Block 1 — fresh root, pinned producer, and CPU admission

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
run_root=/tmp/dino-climate-rebattery-01a04e34
checkout="$run_root/producer-checkout"
producer=9548be86181a1f362355e0cdc30ce4ac98d14e3a
nemo_wall=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$repo"
test ! -e "$run_root"
free_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$free_kb" -ge 15728640 || {
  echo "STOP: climate battery requires at least 15 GiB free in /tmp" >&2
  exit 1
}
git cat-file -e "$producer^{commit}"
mkdir -p "$run_root/arms" "$run_root/logs"
git worktree add --detach "$checkout" "$producer"
test "$(git -C "$checkout" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=no)"
checkout_kb=$(du -sk "$checkout" | awk '{print $1}')
test "$checkout_kb" -le 5242880 || {
  echo "STOP: producer checkout exceeds 5 GiB" >&2
  exit 1
}
test "$(sha256sum "$nemo_wall" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
test -d /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
test -d /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
cd "$checkout"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  /home/dbalwada/legoESM/.venv/bin/python -m pytest -q \
  tests/ocean/unit/test_climate_rebattery_score.py \
  tests/ocean/unit/test_barotropic_continuity_and_drag.py \
  -k 'climate_rebattery or kamm_recipes_select_nemo_ssh_avg_seed'
printf '%s\n' "$producer" > "$run_root/producer_commit.txt"
printf 'producer=%s checkout_kb=%s free_tmp_kb=%s session=%s\n' \
  "$producer" "$checkout_kb" "$free_kb" "$CODEX_SESSION_ID"
```

## Block 2 — fresh 360-day MLD/basin duplicate pair

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-climate-rebattery-01a04e34
checkout="$run_root/producer-checkout"
producer=9548be86181a1f362355e0cdc30ce4ac98d14e3a
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
for arm in climate_a climate_b; do
  test ! -e "$run_root/arms/$arm.npz"
  test ! -e "$run_root/logs/$arm.log"
done

run_climate() {
  gpu=$1
  arm=$2
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/arms/$arm.npz" \
    --days 360 --save-3d --snap-days 0,90,360 --fp64-3d \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    > "$run_root/logs/$arm.log" 2>&1
}

run_climate 0 climate_a & p0=$!
run_climate 1 climate_b & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0
for arm in climate_a climate_b; do
  grep -F "SAVED $run_root/arms/$arm.npz  stable=True" \
    "$run_root/logs/$arm.log"
  printf 'SLOT __MEASURED_%s_SHA256__ VALUE=%s\n' \
    "$(printf '%s' "$arm" | tr '[:lower:]' '[:upper:]')" \
    "$(sha256sum "$run_root/arms/$arm.npz" | awk '{print $1}')"
done
```

## Block 3 — fresh five-day wall duplicate pair

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-climate-rebattery-01a04e34
checkout="$run_root/producer-checkout"
producer=9548be86181a1f362355e0cdc30ce4ac98d14e3a
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
for arm in wall_a wall_b; do
  test ! -e "$run_root/arms/$arm.npz"
  test ! -e "$run_root/logs/$arm.log"
done

run_wall() {
  gpu=$1
  arm=$2
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/arms/$arm.npz" \
    --days 5 --save-step-eta \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    > "$run_root/logs/$arm.log" 2>&1
}

run_wall 0 wall_a & p0=$!
run_wall 1 wall_b & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0
for arm in wall_a wall_b; do
  grep -F "SAVED $run_root/arms/$arm.npz  stable=True" \
    "$run_root/logs/$arm.log"
  printf 'SLOT __MEASURED_%s_SHA256__ VALUE=%s\n' \
    "$(printf '%s' "$arm" | tr '[:lower:]' '[:upper:]')" \
    "$(sha256sum "$run_root/arms/$arm.npz" | awk '{print $1}')"
done
```

## Block 4 — bracket, offline CPU/fp64 score, and receipts

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-climate-rebattery-01a04e34
checkout="$run_root/producer-checkout"
producer=9548be86181a1f362355e0cdc30ce4ac98d14e3a
nemo_wall=/tmp/dino_eta_waves/nemo_5d_eta.npz
score="$run_root/dino_climate_rebattery_score.json"

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(sha256sum "$nemo_wall" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
for arm in climate_a climate_b wall_a wall_b; do
  test -s "$run_root/arms/$arm.npz"
  grep -F "SAVED $run_root/arms/$arm.npz  stable=True" \
    "$run_root/logs/$arm.log"
done
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  /home/dbalwada/legoESM/.venv/bin/python \
  "$checkout/scripts/validate/ocean_fidelity/dino_1226/climate_rebattery_score.py" \
  --climate-a "$run_root/arms/climate_a.npz" \
  --climate-b "$run_root/arms/climate_b.npz" \
  --wall-a "$run_root/arms/wall_a.npz" \
  --wall-b "$run_root/arms/wall_b.npz" \
  --nemo-wall "$nemo_wall" \
  --producer-commit "$producer" \
  --session-id "$CODEX_SESSION_ID" \
  --output "$score" | tee "$run_root/logs/climate_rebattery_score.log"
grep -E '"verdict"|"epoch_duplicate_identity"' "$score"
sha256sum "$run_root"/arms/*.npz "$score" \
  "$run_root/dino_climate_rebattery_score_wall_detail.json" \
  "$run_root"/logs/*.log
printf 'SLOT __MEASURED_CLIMATE_REBATTERY_SCORE_SHA256__ VALUE=%s\n' \
  "$(sha256sum "$score" | awk '{print $1}')"
printf 'producer=%s session=%s\n' "$producer" "$CODEX_SESSION_ID"
```
