# Handoff: four-group ZDF wall-epoch attribution

Blocks use a lean Git worktree, absolute paths, `grep`, an exported session ID,
and the standard 10-GiB/5-GiB disk guards. They run no NEMO process and apply
no patch. The existing current control and historical receipt are read-only.

## Block 1 — clean checkout and frozen-input bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
branch=fidelity/dino-zdf-sweep-codex
run_root=/tmp/dino-zdf-wall-epoch-bisect-01a04e34
checkout="$run_root/checkout"
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$repo"
producer=$(git rev-parse "$branch")
test -n "$producer"
test ! -e "$run_root"
available_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kb" -ge 10485760 || {
  echo "STOP: less than 10 GiB free in /tmp" >&2
  exit 1
}
mkdir -p "$run_root/arms" "$run_root/logs"
git worktree add --detach "$checkout" "$producer"
cd "$checkout"
test -z "$(git status --porcelain)"
size_kb=$(du -sk "$checkout" | awk '{print $1}')
test "$size_kb" -le 5242880 || {
  echo "STOP: checkout exceeds 5 GiB" >&2
  exit 1
}
test "$(sha256sum "$current" | awk '{print $1}')" = \
  c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a
test "$(sha256sum "$historical" | awk '{print $1}')" = \
  5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0
test "$(sha256sum "$nemo" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
git diff --quiet e013e95ca54957a4454878ed7118e623da0a19ba.."$producer" -- \
  packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py
printf '%s\n' "$producer" > "$run_root/producer_commit.txt"
grep -n "Four group reversions" \
  "$checkout/docs/ocean/fidelity/PREREG_zdf_wall_epoch_group_bisect.md"
python "$checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py" \
  --self-test
printf 'producer=%s checkout_kb=%s free_tmp_kb=%s session=%s\n' \
  "$producer" "$size_kb" "$available_kb" "$CODEX_SESSION_ID"
```

## Block 2 — four five-day arms, two GPUs in two waves

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1
export JAX_ENABLE_X64=1
export LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-zdf-wall-epoch-bisect-01a04e34
checkout="$run_root/checkout"
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP
cd "$checkout"
export PYTHONPATH="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"

run_arm() {
  gpu=$1
  arm=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/arms/$arm.npz" \
    --days 5 --save-step-eta \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-before --bridge-before-stress-tpoint \
    --vface-zonal-metric-evaluation legacy_tracer_midpoint \
    --barotropic-continuity-evaluation generic \
    "$@" > "$run_root/logs/$arm.log" 2>&1
}

run_arm 0 entry \
  --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state \
  --tke-shear-metric-source tpoint_jacobian \
  --tke-n2-evaluation-stage implicit_solve_state & p0=$!
run_arm 1 tke_core \
  --tke-matrix-evaluation factored \
  --tke-solver-evaluation shared_thomas \
  --tke-langmuir-evaluation vectorized & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_arm 0 mxl_zdf \
  --tke-etau-exponential-evaluation jax_expression \
  --tke-htau-evaluation jax_expression \
  --tke-mxl-raw-evaluation factored \
  --zdf-implicit-solver-evaluation shared_thomas & p0=$!
run_arm 1 slopes \
  --gm-redi-slope-n2-evaluation recompute \
  --gm-redi-slope-prd-evaluation density_roundtrip \
  --gm-redi-slope-metric-evaluation division \
  --gm-redi-slope-face-thickness-evaluation static_face \
  --gm-redi-slope-depth-evaluation legacy_jacobian_t_surface & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0
```

## Block 3 — completion and freshness bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-zdf-wall-epoch-bisect-01a04e34
checkout="$run_root/checkout"
cd "$checkout"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(cat "$run_root/producer_commit.txt")"
for arm in entry tke_core mxl_zdf slopes; do
  test -s "$run_root/arms/$arm.npz"
  grep -q '^SAVED ' "$run_root/logs/$arm.log"
  grep '^SAVED ' "$run_root/logs/$arm.log"
done
sha256sum "$run_root"/arms/*.npz "$run_root"/logs/*.log
```

## Block 4 — offline score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
run_root=/tmp/dino-zdf-wall-epoch-bisect-01a04e34
checkout="$run_root/checkout"
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz
cd "$checkout"
export PYTHONPATH="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout/scripts/validate/ocean_fidelity/dino_1226"
producer=$(cat "$run_root/producer_commit.txt")
python "$checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py" \
  --nemo "$nemo" --current "$current" \
  --entry "$run_root/arms/entry.npz" \
  --tke-core "$run_root/arms/tke_core.npz" \
  --mxl-zdf "$run_root/arms/mxl_zdf.npz" \
  --slopes "$run_root/arms/slopes.npz" \
  --historical-receipt "$historical" \
  --producer-commit "$producer" \
  --out "$run_root/zdf_wall_epoch_group_score.json" \
  | tee "$run_root/logs/group_score.log"
grep '"disposition"' "$run_root/zdf_wall_epoch_group_score.json"
sha256sum "$run_root/zdf_wall_epoch_group_score"*.json \
  "$run_root/logs/group_score.log"
printf 'producer=%s session=%s\n' "$producer" "$CODEX_SESSION_ID"
```

Do not run an individual-selector peel until Block 4 names one group or calls
for the registered all-16/complement escalation.
