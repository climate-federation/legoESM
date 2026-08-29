# FULL-FRESH blocks: dependency-coherent ZDF wall-epoch attribution

## Block 1 — fresh pinned producer and scorer checkouts

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
run_root=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34
producer_checkout="$run_root/producer-checkout"
scorer_checkout="$run_root/scorer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
scorer_commit=482593ff9c90a40c9541f4aab85c403319dab544
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$repo"
test ! -e "$run_root"
available_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kb" -ge 10485760 || {
  echo "STOP: less than 10 GiB free in /tmp" >&2
  exit 1
}
mkdir -p "$run_root/arms" "$run_root/logs"
git worktree add --detach "$producer_checkout" "$producer"
git worktree add --detach "$scorer_checkout" "$scorer_commit"
test -z "$(git -C "$producer_checkout" status --porcelain)"
test -z "$(git -C "$scorer_checkout" status --porcelain)"
test "$(git -C "$producer_checkout" rev-parse HEAD)" = "$producer"
test "$(git -C "$scorer_checkout" rev-parse HEAD)" = "$scorer_commit"
producer_kb=$(du -sk "$producer_checkout" | awk '{print $1}')
scorer_kb=$(du -sk "$scorer_checkout" | awk '{print $1}')
test "$producer_kb" -le 5242880
test "$scorer_kb" -le 5242880
test "$(sha256sum "$current" | awk '{print $1}')" = \
  c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a
test "$(sha256sum "$historical" | awk '{print $1}')" = \
  5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0
test "$(sha256sum "$nemo" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
git diff --quiet e013e95ca54957a4454878ed7118e623da0a19ba.."$producer" -- \
  packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py
scorer_pythonpath="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
cd "$scorer_checkout"
grep -n "dependency correction before scoring" \
  "$scorer_checkout/docs/ocean/fidelity/PREREG_zdf_wall_epoch_group_bisect.md"
PYTHONPATH="$scorer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py" \
  --self-test
printf '%s\n' "$producer" > "$run_root/producer_commit.txt"
printf '%s\n' "$scorer_commit" > "$run_root/scorer_commit.txt"
printf 'producer=%s scorer=%s producer_kb=%s scorer_kb=%s free_tmp_kb=%s session=%s\n' \
  "$producer" "$scorer_commit" "$producer_kb" "$scorer_kb" \
  "$available_kb" "$CODEX_SESSION_ID"
```

## Block 2 — all five fresh arms, two GPUs in three waves

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1
export JAX_ENABLE_X64=1
export LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34
checkout="$run_root/producer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain)"
producer_pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
for arm in entry_dep slope_n2_only tke_core mxl_zdf slopes; do
  test ! -e "$run_root/arms/$arm.npz"
  test ! -e "$run_root/logs/$arm.log"
done

run_arm() {
  gpu=$1
  arm=$2
  shift 2
  PYTHONPATH="$producer_pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/arms/$arm.npz" \
    --days 5 --save-step-eta \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-before --bridge-before-stress-tpoint \
    --vface-zonal-metric-evaluation legacy_tracer_midpoint \
    --barotropic-continuity-evaluation generic \
    "$@" > "$run_root/logs/$arm.log" 2>&1
}

run_arm 0 entry_dep \
  --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state \
  --tke-shear-metric-source tpoint_jacobian \
  --tke-n2-evaluation-stage implicit_solve_state \
  --gm-redi-slope-n2-evaluation recompute & p0=$!
run_arm 1 slope_n2_only \
  --gm-redi-slope-n2-evaluation recompute & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_arm 0 tke_core \
  --tke-matrix-evaluation factored \
  --tke-solver-evaluation shared_thomas \
  --tke-langmuir-evaluation vectorized & p0=$!
run_arm 1 mxl_zdf \
  --tke-etau-exponential-evaluation jax_expression \
  --tke-htau-evaluation jax_expression \
  --tke-mxl-raw-evaluation factored \
  --zdf-implicit-solver-evaluation shared_thomas & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_arm 0 slopes \
  --gm-redi-slope-n2-evaluation recompute \
  --gm-redi-slope-prd-evaluation density_roundtrip \
  --gm-redi-slope-metric-evaluation division \
  --gm-redi-slope-face-thickness-evaluation static_face \
  --gm-redi-slope-depth-evaluation legacy_jacobian_t_surface
```

## Block 3 — five-arm completion bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34
checkout="$run_root/producer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
cd "$checkout"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
for arm in entry_dep slope_n2_only tke_core mxl_zdf slopes; do
  test -s "$run_root/arms/$arm.npz"
  grep -q "^SAVED .*${arm}.npz  stable=True" "$run_root/logs/$arm.log"
  grep "^SAVED .*${arm}.npz  stable=True" "$run_root/logs/$arm.log"
done
sha256sum "$run_root"/arms/*.npz "$run_root"/logs/*.log
```

## Block 4 — offline dependency-conditioned score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34
scorer_checkout="$run_root/scorer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
scorer_commit=482593ff9c90a40c9541f4aab85c403319dab544
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz
cd "$scorer_checkout"
test "$(git rev-parse HEAD)" = "$scorer_commit"
test "$(cat "$run_root/scorer_commit.txt")" = "$scorer_commit"
test -z "$(git status --porcelain)"
scorer_pythonpath="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$scorer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py" \
  --nemo "$nemo" --current "$current" \
  --entry-dep "$run_root/arms/entry_dep.npz" \
  --slope-n2-only "$run_root/arms/slope_n2_only.npz" \
  --tke-core "$run_root/arms/tke_core.npz" \
  --mxl-zdf "$run_root/arms/mxl_zdf.npz" \
  --slopes "$run_root/arms/slopes.npz" \
  --historical-receipt "$historical" \
  --producer-commit "$producer" \
  --out "$run_root/zdf_wall_epoch_group_score.json" \
  | tee "$run_root/logs/group_score.log"
grep '"disposition"' "$run_root/zdf_wall_epoch_group_score.json"
grep '"interaction_status"' "$run_root/zdf_wall_epoch_group_score.json"
sha256sum "$run_root/zdf_wall_epoch_group_score"*.json \
  "$run_root/logs/group_score.log"
printf 'producer=%s scorer=%s session=%s\n' \
  "$producer" "$scorer_commit" "$CODEX_SESSION_ID"
```
