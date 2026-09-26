# Blocks: legal-lattice ZDF wall-epoch attribution

The retained `slope_n2_only` artifact is admitted by content hash plus a
model/harness-diff-zero receipt against the pinned producer. The five new arms
run at pinned producer `9ac2550d...`; every Python invocation resolves imports
from its pinned checkout. Twin SSH artifacts contain the full haloed
`(time,199,52)` field; reduction probes, not the harness, strip it to
`(197,50)`.

## Block 1 — admit retained arm and create pinned scorer checkout

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
run_root=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34
producer_checkout="$run_root/producer-checkout"
scorer_checkout="$run_root/lattice-scorer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
scorer_commit=96b3858f29334f983f33a3a22961387c416ba667
retained="$run_root/arms/slope_n2_only.npz"
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$repo"
test -d "$run_root/arms"
test -d "$run_root/logs"
available_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kb" -ge 10485760 || {
  echo "STOP: less than 10 GiB free in /tmp" >&2
  exit 1
}
test "$(git -C "$producer_checkout" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$producer_checkout" status --porcelain)"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test "$(sha256sum "$retained" | awk '{print $1}')" = \
  1395b84ec8e9f482c17bde34ebb59d273f794b7e5901ac6c3ea02c0e67d43083
grep -q '^SAVED .*slope_n2_only.npz  stable=True' \
  "$run_root/logs/slope_n2_only.log"
test "$(sha256sum "$current" | awk '{print $1}')" = \
  c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a
test "$(sha256sum "$historical" | awk '{print $1}')" = \
  5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0
test "$(sha256sum "$nemo" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
git cat-file -e "$scorer_commit^{commit}"
test ! -e "$scorer_checkout"
git worktree add --detach "$scorer_checkout" "$scorer_commit"
test -z "$(git -C "$scorer_checkout" status --porcelain)"
test "$(git -C "$scorer_checkout" rev-parse HEAD)" = "$scorer_commit"
producer_kb=$(du -sk "$producer_checkout" | awk '{print $1}')
scorer_kb=$(du -sk "$scorer_checkout" | awk '{print $1}')
test "$producer_kb" -le 5242880
test "$scorer_kb" -le 5242880
git diff --quiet e013e95ca54957a4454878ed7118e623da0a19ba.."$producer" -- \
  packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py
producer_pythonpath="$producer_checkout/packages/core:$producer_checkout/packages/ocean:$producer_checkout/packages/atmosphere:$producer_checkout/packages/coupler:$producer_checkout/packages/ice:$producer_checkout/packages/land:$producer_checkout/packages/ml:$producer_checkout/packages/tools:$producer_checkout/scripts/validate/ocean_fidelity/dino_1226"
retained_producer=$( \
  PYTHONPATH="$producer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python - "$retained" "$CODEX_SESSION_ID" <<'PY'
import json
import sys
import numpy as np

path, session = sys.argv[1:]
with np.load(path) as z:
    cfg = json.loads(str(np.asarray(z["run_config"]).item()))
    assert int(np.asarray(z["producer_dirty_tracked_files"]).item()) == 0
    assert str(np.asarray(z["codex_session_id"]).item()) == session
    assert bool(np.asarray(z["stable"]).item())
    assert np.asarray(z["eta"]).shape == (160, 199, 52)
    assert np.asarray(z["eta"]).dtype == np.float64
    assert cfg["gm_redi_slope_n2_evaluation"] == "recompute"
    print(str(np.asarray(z["producer_git_sha"]).item()))
PY
)
git cat-file -e "$retained_producer^{commit}"
git diff --quiet "$retained_producer".."$producer" -- \
  packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py
scorer_pythonpath="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
cd "$scorer_checkout"
grep -n "Complete executable dependency graph" \
  "$scorer_checkout/docs/ocean/fidelity/PREREG_zdf_wall_epoch_group_bisect.md"
PYTHONPATH="$scorer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py" \
  --self-test
printf '%s\n' "$scorer_commit" > "$run_root/lattice_scorer_commit.txt"
printf 'producer=%s retained_producer=%s scorer=%s producer_kb=%s scorer_kb=%s free_tmp_kb=%s session=%s\n' \
  "$producer" "$retained_producer" "$scorer_commit" "$producer_kb" \
  "$scorer_kb" "$available_kb" "$CODEX_SESSION_ID"
```

## Block 2 — five new legal arms, two GPUs in three waves

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
for arm in tke_core core_slope_n2 entry_closed mxl_zdf slope_rest; do
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

run_arm 0 tke_core \
  --tke-matrix-evaluation factored \
  --tke-solver-evaluation shared_thomas \
  --tke-langmuir-evaluation vectorized & p0=$!
run_arm 1 core_slope_n2 \
  --tke-matrix-evaluation factored \
  --tke-solver-evaluation shared_thomas \
  --tke-langmuir-evaluation vectorized \
  --gm-redi-slope-n2-evaluation recompute & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_arm 0 entry_closed \
  --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state \
  --tke-shear-metric-source tpoint_jacobian \
  --tke-n2-evaluation-stage implicit_solve_state \
  --tke-matrix-evaluation factored \
  --tke-solver-evaluation shared_thomas \
  --tke-langmuir-evaluation vectorized \
  --gm-redi-slope-n2-evaluation recompute & p0=$!
run_arm 1 mxl_zdf \
  --tke-etau-exponential-evaluation jax_expression \
  --tke-htau-evaluation jax_expression \
  --tke-mxl-raw-evaluation factored \
  --zdf-implicit-solver-evaluation shared_thomas & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
test "$rc" -eq 0

run_arm 0 slope_rest \
  --gm-redi-slope-prd-evaluation density_roundtrip \
  --gm-redi-slope-metric-evaluation division \
  --gm-redi-slope-face-thickness-evaluation static_face \
  --gm-redi-slope-depth-evaluation legacy_jacobian_t_surface
```

## Block 3 — six-arm completion bracket

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
test "$(sha256sum "$run_root/arms/slope_n2_only.npz" | awk '{print $1}')" = \
  1395b84ec8e9f482c17bde34ebb59d273f794b7e5901ac6c3ea02c0e67d43083
for arm in slope_n2_only tke_core core_slope_n2 entry_closed mxl_zdf slope_rest; do
  test -s "$run_root/arms/$arm.npz"
  grep -q "^SAVED .*${arm}.npz  stable=True" "$run_root/logs/$arm.log"
  grep "^SAVED .*${arm}.npz  stable=True" "$run_root/logs/$arm.log"
done
sha256sum \
  "$run_root/arms/slope_n2_only.npz" \
  "$run_root/arms/tke_core.npz" \
  "$run_root/arms/core_slope_n2.npz" \
  "$run_root/arms/entry_closed.npz" \
  "$run_root/arms/mxl_zdf.npz" \
  "$run_root/arms/slope_rest.npz" \
  "$run_root/logs/slope_n2_only.log" \
  "$run_root/logs/tke_core.log" \
  "$run_root/logs/core_slope_n2.log" \
  "$run_root/logs/entry_closed.log" \
  "$run_root/logs/mxl_zdf.log" \
  "$run_root/logs/slope_rest.log"
```

## Block 4 — offline lattice score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34
scorer_checkout="$run_root/lattice-scorer-checkout"
producer=9ac2550d4f559b1c73c2b65174fc2e436025176a
scorer_commit=96b3858f29334f983f33a3a22961387c416ba667
current=/tmp/dino-metric-continuity-factorial-01a04e34/wall/legacy_generic.npz
historical=/tmp/codex-basin-rect/results/dino_1455/wind_place_tcarry_explicit_flicker.json
nemo=/tmp/dino_eta_waves/nemo_5d_eta.npz

cd "$scorer_checkout"
test "$(git rev-parse HEAD)" = "$scorer_commit"
test "$(cat "$run_root/lattice_scorer_commit.txt")" = "$scorer_commit"
test -z "$(git status --porcelain)"
scorer_pythonpath="$scorer_checkout/packages/core:$scorer_checkout/packages/ocean:$scorer_checkout/packages/atmosphere:$scorer_checkout/packages/coupler:$scorer_checkout/packages/ice:$scorer_checkout/packages/land:$scorer_checkout/packages/ml:$scorer_checkout/packages/tools:$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$scorer_pythonpath" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python "$scorer_checkout/scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py" \
  --nemo "$nemo" --current "$current" \
  --slope-n2-only "$run_root/arms/slope_n2_only.npz" \
  --tke-core "$run_root/arms/tke_core.npz" \
  --core-slope-n2 "$run_root/arms/core_slope_n2.npz" \
  --entry-closed "$run_root/arms/entry_closed.npz" \
  --mxl-zdf "$run_root/arms/mxl_zdf.npz" \
  --slope-rest "$run_root/arms/slope_rest.npz" \
  --historical-receipt "$historical" \
  --producer-commit "$producer" \
  --out "$run_root/zdf_wall_epoch_lattice_score.json" \
  | tee "$run_root/logs/lattice_score.log"
grep '"disposition"' "$run_root/zdf_wall_epoch_lattice_score.json"
grep '"structurally_unreachable_contrasts"' \
  "$run_root/zdf_wall_epoch_lattice_score.json"
sha256sum "$run_root"/zdf_wall_epoch_lattice_score*.json \
  "$run_root/logs/lattice_score.log"
printf 'producer=%s scorer=%s session=%s\n' \
  "$producer" "$scorer_commit" "$CODEX_SESSION_ID"
```
