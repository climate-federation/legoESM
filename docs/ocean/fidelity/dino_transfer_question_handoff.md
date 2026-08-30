# DINO transfer question: exact SLOT handoff

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`. **PREREGISTERED, NOT RUN.** This
handoff implements `PREREG_dino_transfer_question.md` in T3, T2, T1 order.
GPU arms and any NEMO regeneration belong to the user. The author ran no GPU,
`mpirun`, or NEMO command.

Replace every `__SLOT_*__` token before executing a block. A block containing
an unresolved token must stop. Producer code, new instruments, and catalog
wiring must be committed before a science arm is launched; citable results may
not come from inline or throwaway probes.

## Receipt slots

```text
SLOT __SLOT_TRANSFER_PRODUCER_COMMIT__ VALUE=
SLOT __SLOT_T3_CATALOG_RECIPE__ VALUE=nemo_dino_kamm_mlf_v1
SLOT __SLOT_T3_IDENTITY_PROBE_RELATIVE_PATH__ VALUE=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py
SLOT __SLOT_T3_TWIN_RELATIVE_PATH__ VALUE=
SLOT __SLOT_T3_IDENTITY_ARTIFACT_SHA256__ VALUE=
SLOT __SLOT_T3_ORACLE_ARTIFACT_SHA256__ VALUE=
SLOT __SLOT_T3_CATALOG_ARTIFACT_SHA256__ VALUE=
SLOT __SLOT_T2_FE_CLIMATE_A_SHA256__ VALUE=
SLOT __SLOT_T2_FE_CLIMATE_B_SHA256__ VALUE=
SLOT __SLOT_T2_FE_WALL_A_SHA256__ VALUE=
SLOT __SLOT_T2_FE_WALL_B_SHA256__ VALUE=
SLOT __SLOT_T2_SCORE_SHA256__ VALUE=
SLOT __SLOT_T1_STANDALONE_RUNNER_RELATIVE_PATH__ VALUE=
SLOT __SLOT_T1_SCORE_RELATIVE_PATH__ VALUE=
SLOT __SLOT_T1_NEMO_STANDALONE_ROOT__ VALUE=
SLOT __SLOT_T1_NEMO_ENSEMBLE_MANIFEST_SHA256__ VALUE=
SLOT __SLOT_T1_LEGO_ENSEMBLE_MANIFEST_SHA256__ VALUE=
SLOT __SLOT_T1_SCORE_SHA256__ VALUE=
SLOT __SLOT_T1_MEASURED_GPU_HOURS__ VALUE=
SLOT __SLOT_T1_MEASURED_NEMO_HOURS__ VALUE=
```

## Block 0 — fresh root, producer pin, and CPU admission

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
repo=/tmp/codex-transfer
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
nemo_wall=/tmp/dino_eta_waves/nemo_5d_eta.npz

case "$producer" in __SLOT_*__|'') echo 'STOP: fill producer slot' >&2; exit 1;; esac
cd "$repo"
test ! -e "$run_root"
test -z "$(git status --porcelain --untracked-files=no)"
git cat-file -e "$producer^{commit}"
free_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$free_kb" -ge 15728640 || {
  echo 'STOP: at least 15 GiB free in /tmp required' >&2
  exit 1
}
mkdir -p "$run_root"/{arms,logs,receipts,t1,t2,t3}
git worktree add --detach "$checkout" "$producer"
test "$(git -C "$checkout" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=no)"
test -d "$nemo/RUN_TRAJ"
test -d "$nemo/RUN_STEPDUMP"
test "$(sha256sum "$nemo_wall" | awk '{print $1}')" = \
  52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a
test "$(sha256sum "$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py" | awk '{print $1}')" = \
  4be31374f2858a477f13f90c688c278feb28861bb12e175fab7e68f2e56b1448
test "$(sha256sum "$checkout/scripts/validate/ocean_fidelity/dino_1226/climate_rebattery_score.py" | awk '{print $1}')" = \
  cf85b65302ed65d2a5d8b7ea7a77fb6fa35abd55b5fde8d6e541db9f266996db
test "$(sha256sum "$checkout/scripts/validate/ocean_fidelity/dino_1226/mld_climate_audit.py" | awk '{print $1}')" = \
  cf1bcffb4ee7994bd4eb433f6b3fa3ba5ac0f2610463b9bb3133ea0c1762dc14
test "$(sha256sum "$checkout/scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py" | awk '{print $1}')" = \
  5d952fbee44f4ed9297be2b94750b52ef846afec259bc01259d2a00e99ec181c
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
cd "$checkout"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  /home/dbalwada/legoESM/.venv/bin/python -m pytest -q \
  tests/ocean/unit/test_recipes.py \
  tests/ocean/unit/test_recipe_snapshots.py \
  tests/ocean/unit/test_dino_recipe_transfer_identity.py \
  tests/ocean/unit/test_climate_rebattery_score.py
printf '%s\n' "$producer" > "$run_root/receipts/producer_commit.txt"
printf 'producer=%s session=%s free_tmp_kb=%s\n' \
  "$producer" "$CODEX_SESSION_ID" "$free_kb"
```

## T3.1 finding and resolved implementation choice

The preregistration-parent catalog had no faithful-versioned MLF entry. That is
a T3 finding: the faithful configuration was not catalog-reachable. Per the
user's requested completion, this commit makes the following **ASKED**
user-visible catalog choice:

- public recipe name: `nemo_dino_kamm_mlf_v1`;
- identity probe:
  `scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py`;
- probe SHA-256:
  `5d952fbee44f4ed9297be2b94750b52ef846afec259bc01259d2a00e99ec181c`.

The entry owns the frozen structural fields. The probe's disjoint setup
allowlist owns coefficients, constants, grid-dependent values, and nested
physics/setup objects. It fails on an ownership collision, missing/extra
ownership row, or any recursive resolved-config leaf difference. Its
one-ULP `asselin_gamma` plant and `nemo_dino_v1` negative control run in the
same invocation. **UNASKED changes: none.** The catalog-aware behavioral twin
wrapper remains the separately declared T3.2 prerequisite and is not invented
by this T3.1 repair.

## Block T3.1 — CPU field-by-field identity gate

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
catalog_recipe=nemo_dino_kamm_mlf_v1
probe_rel=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py

for value in "$producer" "$catalog_recipe" "$probe_rel"; do
  case "$value" in __SLOT_*__|'') echo 'STOP: fill every T3.1 slot' >&2; exit 1;; esac
done
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/receipts/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
test -f "$checkout/$probe_rel"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  /home/dbalwada/legoESM/.venv/bin/python "$checkout/$probe_rel" \
  --oracle-card nemo_dino_kamm_mlf \
  --catalog-recipe "$catalog_recipe" \
  --negative-recipe nemo_dino_v1 \
  --producer-commit "$producer" \
  --session-id "$CODEX_SESSION_ID" \
  --output "$run_root/t3/config_identity.json" \
  | tee "$run_root/logs/t3_config_identity.log"
grep -F 'CONFIG_IDENTITY=PASS' "$run_root/logs/t3_config_identity.log"
grep -F 'PLANT=FIRED' "$run_root/logs/t3_config_identity.log"
grep -F 'OWNERSHIP_COLLISION=FIRED' "$run_root/logs/t3_config_identity.log"
grep -F 'NEGATIVE_CONTROL=DIFF' "$run_root/logs/t3_config_identity.log"
printf 'SLOT __SLOT_T3_IDENTITY_ARTIFACT_SHA256__ VALUE=%s\n' \
  "$(sha256sum "$run_root/t3/config_identity.json" | awk '{print $1}')"
```

If this block does not pass, T3 stops without GPU use.

## Block T3.2 — user GPU behavioral identity pair

The committed twin adapter must expose the catalog construction path without a
physics override. Replace only the adapter's documented construction-path
argument below if its reviewed CLI spelling differs; the two commands must be
identical except `--config-source` and output/log path.

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
catalog_recipe=nemo_dino_kamm_mlf_v1
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
twin_rel=__SLOT_T3_TWIN_RELATIVE_PATH__
probe_rel=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py

for value in "$producer" "$catalog_recipe" "$twin_rel" "$probe_rel"; do
  case "$value" in __SLOT_*__|'') echo 'STOP: fill every T3.2 slot' >&2; exit 1;; esac
done
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
grep -F 'CONFIG_IDENTITY=PASS' "$run_root/logs/t3_config_identity.log"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/$twin_rel"
test -f "$twin"
test -f "$checkout/$probe_rel"

run_t3() {
  gpu=$1
  source=$2
  arm=$3
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$run_root/t3/$arm.npz" \
    --days 5 --save-3d --snap-days 0,5 --fp64-3d \
    --run-traj "$nemo/RUN_TRAJ" --run-stepdump "$nemo/RUN_STEPDUMP" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    --config-source "$source" --catalog-recipe "$catalog_recipe" \
    > "$run_root/logs/$arm.log" 2>&1
}
run_t3 0 oracle t3_oracle & p0=$!
run_t3 1 catalog t3_catalog & p1=$!
rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; test "$rc" -eq 0
for arm in t3_oracle t3_catalog; do
  grep -F "SAVED $run_root/t3/$arm.npz  stable=True" "$run_root/logs/$arm.log"
done

# The committed T3 probe must also support artifact comparison.
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python "$checkout/$probe_rel" \
  --compare-oracle "$run_root/t3/t3_oracle.npz" \
  --compare-catalog "$run_root/t3/t3_catalog.npz" \
  --output "$run_root/t3/behavior_identity.json" \
  | tee "$run_root/logs/t3_behavior_identity.log"
grep -F 'BEHAVIOR_IDENTITY=PASS' "$run_root/logs/t3_behavior_identity.log"
sha256sum "$run_root/t3/"*.npz "$run_root/t3/"*.json
```

## Block T2.1 — user GPU FE climate duplicate

This is round-94 Block 2 with exactly one scientific edit:
`nemo_dino_kamm_mlf` becomes `nemo_dino_kamm`.

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
run_fe_climate() {
  gpu=$1; arm=$2
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$run_fp64" "$twin" \
    nemo_dino_kamm "$run_root/t2/$arm.npz" \
    --days 360 --save-3d --snap-days 0,90,360 --fp64-3d \
    --run-traj "$nemo/RUN_TRAJ" --run-stepdump "$nemo/RUN_STEPDUMP" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    > "$run_root/logs/$arm.log" 2>&1
}
run_fe_climate 0 fe_climate_a & p0=$!
run_fe_climate 1 fe_climate_b & p1=$!
rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; test "$rc" -eq 0
for arm in fe_climate_a fe_climate_b; do
  grep -F "SAVED $run_root/t2/$arm.npz  stable=True" "$run_root/logs/$arm.log"
  sha256sum "$run_root/t2/$arm.npz"
done
```

## Block T2.2 — user GPU FE wall duplicate

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
run_fe_wall() {
  gpu=$1; arm=$2
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$run_fp64" "$twin" \
    nemo_dino_kamm "$run_root/t2/$arm.npz" \
    --days 5 --save-step-eta \
    --run-traj "$nemo/RUN_TRAJ" --run-stepdump "$nemo/RUN_STEPDUMP" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    > "$run_root/logs/$arm.log" 2>&1
}
run_fe_wall 0 fe_wall_a & p0=$!
run_fe_wall 1 fe_wall_b & p1=$!
rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; test "$rc" -eq 0
for arm in fe_wall_a fe_wall_b; do
  grep -F "SAVED $run_root/t2/$arm.npz  stable=True" "$run_root/logs/$arm.log"
  sha256sum "$run_root/t2/$arm.npz"
done
```

## Block T2.3 — unmodified offline scorer

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
nemo_wall=/tmp/dino_eta_waves/nemo_5d_eta.npz
score="$run_root/t2/fe_climate_rebattery_score.json"
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  /home/dbalwada/legoESM/.venv/bin/python \
  "$checkout/scripts/validate/ocean_fidelity/dino_1226/climate_rebattery_score.py" \
  --climate-a "$run_root/t2/fe_climate_a.npz" \
  --climate-b "$run_root/t2/fe_climate_b.npz" \
  --wall-a "$run_root/t2/fe_wall_a.npz" \
  --wall-b "$run_root/t2/fe_wall_b.npz" \
  --nemo-wall "$nemo_wall" \
  --producer-commit "$producer" \
  --session-id "$CODEX_SESSION_ID" \
  --output "$score" | tee "$run_root/logs/t2_score.log"
sha256sum "$run_root/t2/"*.npz "$score" \
  "$run_root/t2/fe_climate_rebattery_score_wall_detail.json"
```

## T1 checkpoint — fresh NEMO ensemble and missing standalone machinery

No recorded NEMO ensemble is admissible for T1. The user must produce six
fresh, from-rest, 20-year NEMO members (control plus temperature-kick seeds
1–5) with the 75 registered output dates. The NEMO manifest must hash each
member's restart-quality states, perturbation receipt, namelists,
executable/source receipt, grid/mask, logs, and `STOP 0`. `RUN_20Y_REBUILD` and
`RUN_40Y_REBUILD` are reducer controls only. No NEMO launch command is supplied
or run here; the user owns ranks, filesystem placement, and execution.

NEMO must first capture its own analytic DINO initialization as a pre-step
`t=0` restart. Member 0 uses it unchanged; members 1–5 change only now-level
temperature with relative kick `1e-14` and seeds 1–5. The setup receipt must
prove every non-temperature restart variable is byte-identical. This needs a
NEMO-side t=0 capture/perturbation setup step but no legoESM state or bridge.

The repo also needs a committed standalone legoESM runner that saves the same
75 fp64 snapshots and reductions without accepting any bridge argument, plus a
committed scorer importing the held 20-year battery's reducers and plants.
Record their paths in the SLOTs and restart from Block 0. Do not adapt
`kamm_twin_90d.py` by merely omitting filenames: its bridge semantics are the
experimental variable.

## Block T1.1 — CPU instrument and fresh-NEMO manifest admission

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
runner_rel=__SLOT_T1_STANDALONE_RUNNER_RELATIVE_PATH__
score_rel=__SLOT_T1_SCORE_RELATIVE_PATH__
nemo_root=__SLOT_T1_NEMO_STANDALONE_ROOT__
manifest="$run_root/t1/nemo_standalone_manifest.sha256"
for value in "$producer" "$runner_rel" "$score_rel" "$nemo_root"; do
  case "$value" in __SLOT_*__|'') echo 'STOP: fill every T1.1 slot' >&2; exit 1;; esac
done
test -f "$checkout/$runner_rel"
test -f "$checkout/$score_rel"
for member in 0 1 2 3 4 5; do
  test -d "$nemo_root/m$member"
  grep -Fx 'STOP 0' "$nemo_root/m$member/run.log"
done
find "$nemo_root" -type f -print0 | sort -z | xargs -0 sha256sum > "$manifest"
sha256sum "$manifest"
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
```

## Block T1.2 — user GPU six standalone 20-year members

The committed runner owns exact CLI spelling. Its reviewed interface must
implement these semantics, use the model's own initializer, and reject every
bridge/resume option. Member 0 is unperturbed; members 1–5 use the same-numbered
`1e-14` temperature-kick seed at initialized time zero.

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
runner_rel=__SLOT_T1_STANDALONE_RUNNER_RELATIVE_PATH__
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout"
run_member() {
  gpu=$1; member=$2
  args=''
  if test "$member" -ne 0; then args="--temperature-kick-seed $member --temperature-kick-relative 1e-14"; fi
  # shellcheck disable=SC2086 -- args is the frozen optional two-flag member arm.
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$checkout/$runner_rel" \
    --recipe nemo_dino_kamm_mlf --start standalone --years 20 \
    --annual-years 1:15 --monthly-years 16:20 --fp64-snapshots \
    --member "$member" $args \
    --output "$run_root/t1/lego_m${member}_20y.npz" \
    > "$run_root/logs/t1_lego_m${member}.log" 2>&1
}
for wave in 0 2 4; do
  run_member 0 "$wave" & p0=$!
  run_member 1 "$((wave + 1))" & p1=$!
  rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; test "$rc" -eq 0
done
for member in 0 1 2 3 4 5; do
  grep -F 'STANDALONE_20Y_STABLE' "$run_root/logs/t1_lego_m${member}.log"
done
sha256sum "$run_root/t1/lego_m"*_20y.npz > "$run_root/t1/lego_manifest.sha256"
sha256sum "$run_root/t1/lego_manifest.sha256"
```

## Block T1.3 — CPU 20-year distribution score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
run_root=/tmp/dino-transfer-01a053d4
checkout="$run_root/producer-checkout"
producer=__SLOT_TRANSFER_PRODUCER_COMMIT__
score_rel=__SLOT_T1_SCORE_RELATIVE_PATH__
nemo_root=__SLOT_T1_NEMO_STANDALONE_ROOT__
nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout"
cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python \
  "$checkout/$score_rel" \
  --lego-members "$run_root/t1/lego_m0_20y.npz" \
                 "$run_root/t1/lego_m1_20y.npz" \
                 "$run_root/t1/lego_m2_20y.npz" \
                 "$run_root/t1/lego_m3_20y.npz" \
                 "$run_root/t1/lego_m4_20y.npz" \
                 "$run_root/t1/lego_m5_20y.npz" \
  --nemo-members "$nemo_root/m0" "$nemo_root/m1" "$nemo_root/m2" \
                 "$nemo_root/m3" "$nemo_root/m4" "$nemo_root/m5" \
  --nemo-manifest "$run_root/t1/nemo_standalone_manifest.sha256" \
  --lego-manifest "$run_root/t1/lego_manifest.sha256" \
  --from-rest-control-y1-y20 "$nemo/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc" \
  --from-rest-control-y21-y40 "$nemo/RUN_40Y_REBUILD/DINO_1y_00210101_00401230_grid_U.nc" \
  --bootstrap-draws 20000 --bootstrap-seed 1455 \
  --output "$run_root/t1/standalone_transfer_score.json" \
  | tee "$run_root/logs/t1_score.log"
grep -E 'STANDALONE_|CONFIRM|REFUTE|UNRESOLVED|ONE_SIDED' \
  "$run_root/logs/t1_score.log"
sha256sum "$run_root/t1/"*.npz "$run_root/t1/"*.sha256 \
  "$run_root/t1/standalone_transfer_score.json"
```

## Completion receipt

The result commit must fill every applicable SLOT, preserve the complete
row-wise JSON artifacts, state NEMO compute actually used, and quote only the
claim sentences licensed by the preregistration. Post all findings, including
refutations and unavailable archives, to the live #1455 evidence trail before
citing them elsewhere. Do not edit an open PR branch after its finished unit.
