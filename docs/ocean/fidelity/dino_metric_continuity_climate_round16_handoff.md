# Handoff: V-face metric x QCO association climate factorial

These blocks are frozen by
`PREREG_metric_continuity_climate_factorial_round16.md` and its dated receipt
amendment. They run no NEMO process and apply no patch. Every block changes to
its own checkout, uses absolute paths, uses `grep` rather than `rg`, and exports
the session ID. After Blocks 1--2 produced artifacts at `e013e95ca`, the
corrected scoring blocks deliberately use a second clean adjudication checkout;
the original producer checkout and completed artifacts remain untouched.

## Block 1 — clean detached checkout and disk guard

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
branch=fidelity/dino-zdf-sweep-codex
run_root=/tmp/dino-metric-continuity-factorial-01a04e34
checkout="$run_root/checkout"

cd "$repo"
producer=$(git rev-parse "$branch")
test -n "$producer"
test ! -e "$run_root"
available_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kb" -ge 10485760 || {
  echo "STOP: less than 10 GiB free in /tmp" >&2
  exit 1
}
mkdir -p "$run_root/basin" "$run_root/wall" "$run_root/logs"
git worktree add --detach "$checkout" "$producer"
cd "$checkout"
test -z "$(git status --porcelain)"
size_kb=$(du -sk "$checkout" | awk '{print $1}')
test "$size_kb" -le 5242880 || {
  echo "STOP: checkout exceeds 5 GiB; no run archive should be copied" >&2
  exit 1
}
printf '%s\n' "$producer" > "$run_root/producer_commit.txt"
grep -n "V-face metric x QCO" \
  "$checkout/docs/ocean/fidelity/PREREG_metric_continuity_climate_factorial_round16.md"
sha256sum \
  "$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py" \
  "$checkout/scripts/validate/ocean_fidelity/dino_1226/metric_continuity_factorial_climate_score.py" \
  "$checkout/scripts/validate/ocean_fidelity/dino_1226/metric_continuity_factorial_wall_score.py"
printf 'producer_commit=%s session_id=%s checkout_kb=%s free_tmp_kb=%s\n' \
  "$producer" "$CODEX_SESSION_ID" "$size_kb" "$available_kb"
```

This checkout is a Git worktree, not a copied oracle tree. The 10-GiB free
space and 5-GiB checkout guards are the standard protection against a future
base/path regression.

## Block 2 — four day-360 basin arms (two GPUs, two waves)

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1
export JAX_ENABLE_X64=1
export LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-metric-continuity-factorial-01a04e34
checkout="$run_root/checkout"
cd "$checkout"
export PYTHONPATH="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP

run_basin() {
  gpu=$1
  arm=$2
  metric=$3
  association=$4
  CUDA_VISIBLE_DEVICES="$gpu" python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/basin/$arm.npz" \
    --days 360 --save-3d --snap-days 0,360 \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-before --bridge-before-stress-tpoint \
    --vface-zonal-metric-evaluation "$metric" \
    --barotropic-continuity-evaluation "$association" \
    > "$run_root/logs/basin_$arm.log" 2>&1
}

run_basin 0 legacy_generic legacy_tracer_midpoint generic & p0=$!
run_basin 1 legacy_literal legacy_tracer_midpoint nemo_literal & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_basin 0 nemo_generic nemo_vpoint generic & p0=$!
run_basin 1 nemo_literal nemo_vpoint nemo_literal & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

for arm in legacy_generic legacy_literal nemo_generic nemo_literal; do
  grep -q '^SAVED ' "$run_root/logs/basin_$arm.log"
  grep '^SAVED ' "$run_root/logs/basin_$arm.log"
done
sha256sum "$run_root"/basin/*.npz
```

## Block 3 — frozen basin bracket and score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
branch=fidelity/dino-zdf-sweep-codex
run_root=/tmp/dino-metric-continuity-factorial-01a04e34
scorer_checkout="$run_root/scorer-checkout"
producer=$(cat "$run_root/producer_commit.txt")
test "$producer" = e013e95ca54957a4454878ed7118e623da0a19ba
cd "$repo"
scorer_commit=$(git rev-parse "$branch")
test -n "$scorer_commit"
test ! -e "$scorer_checkout" || {
  echo "STOP: scorer checkout already exists; inspect rather than overwrite" >&2
  exit 1
}
available_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kb" -ge 10485760 || {
  echo "STOP: less than 10 GiB free in /tmp" >&2
  exit 1
}
git worktree add --detach "$scorer_checkout" "$scorer_commit"
cd "$scorer_checkout"
test -z "$(git status --porcelain)"
size_kb=$(du -sk "$scorer_checkout" | awk '{print $1}')
test "$size_kb" -le 5242880 || {
  echo "STOP: scorer checkout exceeds 5 GiB" >&2
  exit 1
}
printf '%s\n' "$scorer_commit" > "$run_root/scorer_commit.txt"
export PYTHONPATH="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
grep -n "receipt-only amendment" \
  "$scorer_checkout/docs/ocean/fidelity/PREREG_metric_continuity_climate_factorial_round16.md"
python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/metric_continuity_factorial_climate_score.py" \
  --legacy-generic "$run_root/basin/legacy_generic.npz" \
  --legacy-literal "$run_root/basin/legacy_literal.npz" \
  --nemo-generic "$run_root/basin/nemo_generic.npz" \
  --nemo-literal "$run_root/basin/nemo_literal.npz" \
  --producer-commit "$producer" \
  --out "$run_root/metric_continuity_basin_factorial.json" \
  | tee "$run_root/logs/basin_score.log"
grep '"headline_verdict"' "$run_root/metric_continuity_basin_factorial.json"
sha256sum "$run_root/metric_continuity_basin_factorial.json"
printf 'producer_commit=%s scorer_commit=%s session_id=%s\n' \
  "$producer" "$scorer_commit" "$CODEX_SESSION_ID"
```

## Block 4 — four five-day wall arms

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1
export JAX_ENABLE_X64=1
export LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-metric-continuity-factorial-01a04e34
checkout="$run_root/checkout"
cd "$checkout"
export PYTHONPATH="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP

run_wall() {
  gpu=$1
  arm=$2
  metric=$3
  association=$4
  CUDA_VISIBLE_DEVICES="$gpu" python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/wall/$arm.npz" \
    --days 5 --save-step-eta \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-before --bridge-before-stress-tpoint \
    --vface-zonal-metric-evaluation "$metric" \
    --barotropic-continuity-evaluation "$association" \
    > "$run_root/logs/wall_$arm.log" 2>&1
}

run_wall 0 legacy_generic legacy_tracer_midpoint generic & p0=$!
run_wall 1 legacy_literal legacy_tracer_midpoint nemo_literal & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_wall 0 nemo_generic nemo_vpoint generic & p0=$!
run_wall 1 nemo_literal nemo_vpoint nemo_literal & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

for arm in legacy_generic legacy_literal nemo_generic nemo_literal; do
  grep -q '^SAVED ' "$run_root/logs/wall_$arm.log"
  grep '^SAVED ' "$run_root/logs/wall_$arm.log"
done
sha256sum "$run_root"/wall/*.npz
```

## Block 5 — frozen wall bracket and score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-metric-continuity-factorial-01a04e34
scorer_checkout="$run_root/scorer-checkout"
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz
test -e "$scorer_checkout/.git" || {
  echo "STOP: run corrected Block 3 to create the scorer checkout" >&2
  exit 1
}
cd "$scorer_checkout"
test -z "$(git status --porcelain)"
scorer_commit=$(cat "$run_root/scorer_commit.txt")
test "$(git rev-parse HEAD)" = "$scorer_commit"
export PYTHONPATH="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
producer=$(cat "$run_root/producer_commit.txt")
test "$(sha256sum "$nemo" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/metric_continuity_factorial_wall_score.py" \
  --nemo "$nemo" \
  --legacy-generic "$run_root/wall/legacy_generic.npz" \
  --legacy-literal "$run_root/wall/legacy_literal.npz" \
  --nemo-generic "$run_root/wall/nemo_generic.npz" \
  --nemo-literal "$run_root/wall/nemo_literal.npz" \
  --producer-commit "$producer" \
  --out "$run_root/metric_continuity_wall_factorial.json" \
  | tee "$run_root/logs/wall_score.log"
grep '"headline_verdict"' "$run_root/metric_continuity_wall_factorial.json"
sha256sum "$run_root/metric_continuity_wall_factorial"*.json
printf 'session_id=%s producer_commit=%s scorer_commit=%s\n' \
  "$CODEX_SESSION_ID" "$producer" "$scorer_commit"
```

Do not remove the checkout or artifacts until both scorers have exited zero
and their SHA-256 lines have been retained.
