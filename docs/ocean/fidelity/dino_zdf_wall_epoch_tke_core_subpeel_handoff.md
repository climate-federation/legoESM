# SLOT blocks: TKE-core wall-epoch sub-peel

These blocks run three new legal selector arms and retain the faithful control
and full-core endpoints by hash plus model/harness-diff-zero admission. They do
not copy any oracle tree or run archive. Every Python invocation uses the
checkout-local `PYTHONPATH`; every checkout and model producer is SHA-pinned.

## Block 1 — fresh guarded SLOT root and pinned checkouts

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
run_root=/tmp/dino-zdf-wall-epoch-tke-core-peel-01a04e34
producer_checkout="$run_root/producer-checkout"
scorer_checkout="$run_root/scorer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
scorer_commit=74c442dd8166be4d88de8ad60f7d2eab3d576f7b
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
full_core=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34/arms/tke_core.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$repo"
test ! -e "$run_root"
available_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kb" -ge 10485760 || {
  echo "STOP: less than 10 GiB free in /tmp" >&2
  exit 1
}
git cat-file -e "$producer^{commit}"
git cat-file -e "$scorer_commit^{commit}"
mkdir -p "$run_root/arms" "$run_root/logs"
git worktree add --detach "$producer_checkout" "$producer"
git worktree add --detach "$scorer_checkout" "$scorer_commit"
test "$(git -C "$producer_checkout" rev-parse HEAD)" = "$producer"
test "$(git -C "$scorer_checkout" rev-parse HEAD)" = "$scorer_commit"
test -z "$(git -C "$producer_checkout" status --porcelain)"
test -z "$(git -C "$scorer_checkout" status --porcelain)"
producer_kb=$(du -sk "$producer_checkout" | awk '{print $1}')
scorer_kb=$(du -sk "$scorer_checkout" | awk '{print $1}')
test "$producer_kb" -le 5242880 || {
  echo "STOP: producer checkout exceeds 5 GiB" >&2
  exit 1
}
test "$scorer_kb" -le 5242880 || {
  echo "STOP: scorer checkout exceeds 5 GiB" >&2
  exit 1
}
test "$(sha256sum "$current" | awk '{print $1}')" = \
  c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a
test "$(sha256sum "$full_core" | awk '{print $1}')" = \
  1be9010230834eca71f349aa672c20c5704c2f6b0887ed19d0b4eff01e38ee3a
test "$(sha256sum "$historical" | awk '{print $1}')" = \
  5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0
test "$(sha256sum "$nemo" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
producer_pythonpath="$producer_checkout/packages/core:$producer_checkout/packages/ocean:$producer_checkout/packages/atmosphere:$producer_checkout/packages/coupler:$producer_checkout/packages/ice:$producer_checkout/packages/land:$producer_checkout/packages/ml:$producer_checkout/packages/tools:$producer_checkout/scripts/validate/ocean_fidelity/dino_1226"
retained_producers=$( \
  PYTHONPATH="$producer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python - "$current" "$full_core" "$CODEX_SESSION_ID" <<'PY'
import sys
import numpy as np

session = sys.argv[3]
for path in sys.argv[1:3]:
    with np.load(path) as z:
        assert int(np.asarray(z["producer_dirty_tracked_files"]).item()) == 0
        assert str(np.asarray(z["codex_session_id"]).item()) == session
        assert bool(np.asarray(z["stable"]).item())
        assert np.asarray(z["eta"]).shape == (160, 199, 52)
        assert np.asarray(z["eta"]).dtype == np.float64
        print(str(np.asarray(z["producer_git_sha"]).item()))
PY
)
while IFS= read -r retained_producer; do
  test -n "$retained_producer"
  git cat-file -e "$retained_producer^{commit}"
  git diff --quiet "$retained_producer".."$producer" -- \
    packages/core packages/ocean \
    scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py
done <<< "$retained_producers"
scorer_pythonpath="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
cd "$scorer_checkout"
grep -n "Frozen decomposition and bars" \
  "$scorer_checkout/docs/ocean/fidelity/PREREG_zdf_wall_epoch_tke_core_subpeel.md"
PYTHONPATH="$scorer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_tke_core_score.py" \
  --self-test
printf '%s\n' "$producer" > "$run_root/producer_commit.txt"
printf '%s\n' "$scorer_commit" > "$run_root/scorer_commit.txt"
printf 'producer=%s scorer=%s retained_producers=%s producer_kb=%s scorer_kb=%s free_tmp_kb=%s session=%s\n' \
  "$producer" "$scorer_commit" "$(printf '%s' "$retained_producers" | tr '\n' ',')" \
  "$producer_kb" "$scorer_kb" "$available_kb" "$CODEX_SESSION_ID"
```

## Block 2 — three fresh arms in two SLOT waves

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1
export JAX_ENABLE_X64=1
export LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-zdf-wall-epoch-tke-core-peel-01a04e34
checkout="$run_root/producer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain)"
producer_pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
for arm in solver_only langmuir_only solver_langmuir; do
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

run_arm 0 solver_only \
  --tke-solver-evaluation shared_thomas & p0=$!
run_arm 1 langmuir_only \
  --tke-langmuir-evaluation vectorized & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_arm 0 solver_langmuir \
  --tke-solver-evaluation shared_thomas \
  --tke-langmuir-evaluation vectorized
```

## Block 3 — retained plus new completion bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-zdf-wall-epoch-tke-core-peel-01a04e34
checkout="$run_root/producer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
full_core=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34/arms/tke_core.npz

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain)"
test "$(sha256sum "$current" | awk '{print $1}')" = \
  c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a
test "$(sha256sum "$full_core" | awk '{print $1}')" = \
  1be9010230834eca71f349aa672c20c5704c2f6b0887ed19d0b4eff01e38ee3a
for arm in solver_only langmuir_only solver_langmuir; do
  test -s "$run_root/arms/$arm.npz"
  grep -q "^SAVED .*${arm}.npz  stable=True" "$run_root/logs/$arm.log"
  grep "^SAVED .*${arm}.npz  stable=True" "$run_root/logs/$arm.log"
done
sha256sum \
  "$current" "$full_core" \
  "$run_root/arms/solver_only.npz" \
  "$run_root/arms/langmuir_only.npz" \
  "$run_root/arms/solver_langmuir.npz" \
  "$run_root/logs/solver_only.log" \
  "$run_root/logs/langmuir_only.log" \
  "$run_root/logs/solver_langmuir.log"
```

## Block 4 — offline component score and receipt

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-zdf-wall-epoch-tke-core-peel-01a04e34
scorer_checkout="$run_root/scorer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
scorer_commit=74c442dd8166be4d88de8ad60f7d2eab3d576f7b
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
full_core=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34/arms/tke_core.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$scorer_checkout"
test "$(git rev-parse HEAD)" = "$scorer_commit"
test "$(cat "$run_root/scorer_commit.txt")" = "$scorer_commit"
test -z "$(git status --porcelain)"
scorer_pythonpath="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$scorer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_tke_core_score.py" \
  --nemo "$nemo" --current "$current" --full-core "$full_core" \
  --solver-only "$run_root/arms/solver_only.npz" \
  --langmuir-only "$run_root/arms/langmuir_only.npz" \
  --solver-langmuir "$run_root/arms/solver_langmuir.npz" \
  --historical-receipt "$historical" \
  --producer-commit "$producer" \
  --out "$run_root/zdf_wall_epoch_tke_core_score.json" \
  | tee "$run_root/logs/tke_core_score.log"
grep '"disposition"' "$run_root/zdf_wall_epoch_tke_core_score.json"
grep '"component_classification"' \
  "$run_root/zdf_wall_epoch_tke_core_score.json"
grep '"full_core_reconstruction_pass": true' \
  "$run_root/zdf_wall_epoch_tke_core_score.json"
sha256sum "$run_root"/zdf_wall_epoch_tke_core_score*.json \
  "$run_root/logs/tke_core_score.log"
printf 'producer=%s scorer=%s session=%s\n' \
  "$producer" "$scorer_commit" "$CODEX_SESSION_ID"
```
