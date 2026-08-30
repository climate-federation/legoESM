# DINO transfer question: T1 standalone-arm handoff

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 BLOCKED-NEEDS-FE-STABILIZATION;
T1 LEGOESM ARMS READY / SCIENCE UNMEASURED.** No GPU, NEMO, `mpirun`, push, or
FE bisect was performed in this build round. The previous T2 GPU blocks remain
withdrawn. The executable blocks below are the six registered T1 legoESM GPU
arms for the human to run.

## Receipt slots

```text
SLOT TRANSFER_PRODUCER_REF VALUE=refs/heads/fidelity/dino-transfer-codex
SLOT TRANSFER_SESSION_ID VALUE=01a053d4-8e9f-7212-bbdb-19ba2d64e140
SLOT T3_STATUS VALUE=TRANSFER_CONFIRMED
SLOT T3_CATALOG_RECIPE VALUE=nemo_dino_kamm_mlf_v1
SLOT T3_IDENTITY_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py
SLOT T3_TWIN_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py
SLOT T3_IDENTITY_ARTIFACT_SHA256 VALUE=cae74e17dd2e3c60f592c203b398a9d1d15790532be1387090ad9dc35fdd27fe
SLOT T3_BEHAVIOR_IDENTITY_ARTIFACT_SHA256 VALUE=f52990b76afe44cc5a045fd3d2d8e7c10c311a79838bc83296fd6b873cf28169
SLOT T3_ORACLE_ARTIFACT_SHA256 VALUE=d94fd2370413c151723a9986df046d9a5d1051202802b34de2f21b7f9e725f52
SLOT T3_CATALOG_ARTIFACT_SHA256 VALUE=fa778f3ed225955cf0f2ae9ac523d48b07b5763840d05fb7d7387726b65c8983
SLOT T2_STATUS VALUE=BLOCKED-NEEDS-FE-STABILIZATION
SLOT T2_FE_CLIMATE_R2_A_ARTIFACT VALUE=BLOCKED_NO_ARTIFACT
SLOT T2_FE_CLIMATE_R2_A_LOG_SHA256 VALUE=e07de38ab808ce86d46ba813b488b3c4661c9bed9b5627eeb5479685971db664
SLOT T2_FE_CLIMATE_R2_B_ARTIFACT VALUE=BLOCKED_NO_ARTIFACT
SLOT T2_FE_CLIMATE_R2_B_LOG_SHA256 VALUE=637b769cba5c81dd6f3080d22ad3322e64d29bb65be6bc986258a23a80ce4452
SLOT T2_FE_WALL_ARTIFACTS VALUE=BLOCKED_NOT_RUN
SLOT T2_SCORE_ARTIFACT VALUE=BLOCKED_NOT_RUN
SLOT T2_CPU_REPRO_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/fe_stability_repro.py
SLOT T2_CPU_REPRO_SCRIPT_SHA256 VALUE=d8d8be9e0d2b057260dcebdd9a3c77c370d16005a26cf3b0f26864ee68289fae
SLOT T2_CPU_REPRO_RECEIPT_SHA256 VALUE=3adc8ba20f26dec0631b7118c5f881abd3f630391c808067ccf7725435a5281e
SLOT T2_CPU_REPRO_LOG_SHA256 VALUE=24d0c63f2208a476d70e011587a341de00e6964a57b638a84d332042b0d0489b
SLOT T2_SCOPE_BISECT_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_fe_default_scope_bisect.md
SLOT T2_ALL_RESTORED_RECEIPT_SHA256 VALUE=6392216940040b02184f1c2d4a3da99adab1dfda55a7f833f7bc3a0318cbe4d1
SLOT T2_ALL_RESTORED_LOG_SHA256 VALUE=a70c489bae288b657113401ee974928d66a6fd78e7bfaf0c320b07e5c9fa6fed
SLOT T2_PRE1696_COMMIT VALUE=b794c0618e287ebf1d364a8713c3c504ac2eb01c
SLOT T2_PRE1696_ARTIFACT_SHA256 VALUE=9f97a8ba82800672116a7f60c1cb1043dfc1aab7b51ab230311bced1284f2839
SLOT T2_PRE1696_LOG_SHA256 VALUE=4f5f128c7ea83ebfe3d81a41da4b29146b4d744e0862a573b3c9f04605218464
SLOT T2_PR1696_ADDENDUM_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_pr1696_t2_addendum.md
SLOT T1_STATUS VALUE=LEGO_ARMS_READY_SCIENCE_UNMEASURED
SLOT T1_COLD_START_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_standalone_cold_start_corrector.md
SLOT T1_VALIDATED_BUILD_PRODUCER_SHA VALUE=28be310c6ab487c3fa685063af6554abea5938f2
SLOT T1_COLD_START_CORE_SHA256 VALUE=ddb0c414a3d9f44073fed646516b9dbe463a4a9cac42b721b54aff94d26ef8cf
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=de0d17b3651da380356ca8f89ad0b7e8166a24d8f15a23049c4557fe1d2bf763
SLOT T1_SCORE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y_score.py
SLOT T1_SCORE_SCRIPT_SHA256 VALUE=de121565dc4ecdd9b2543886f2f240ed53f994e1c0f1622e5b62aaffe71d32f9
SLOT T1_CONFIG_IDENTITY_R2_SHA256 VALUE=c8f885a669734ed03f9bfd769eca5e367488bb34dcc21a1b52f07f676de73fe1
SLOT T1_CPU_SMOKE_MANIFEST_SHA256 VALUE=41c044ad8e13e1ce9745a94b2d0724451b32d113f4ed1c74f74434d28217ebc1
SLOT T1_CPU_SMOKE_INITIAL_RECEIPT_SHA256 VALUE=a2d69016458af5b11258b5975841246bbaa9d2404624050c06f19db55fef7951
SLOT T1_LEGO_STANDALONE_ROOT VALUE=HANDOFF_TO_CREATE
SLOT T1_NEMO_STANDALONE_ROOT VALUE=HUMAN_TO_CREATE_FROM_REST
SLOT T1_NORMALIZED_STATISTICS_PRODUCER VALUE=BUILD_IN_NEMO_AND_NORMALIZER_ROUND
SLOT T1_NEMO_ENSEMBLE_MANIFEST_SHA256 VALUE=NOT_RUN
SLOT T1_LEGO_ENSEMBLE_MANIFEST_SHA256 VALUE=NOT_RUN
SLOT T1_SCORE_SHA256 VALUE=NOT_RUN
SLOT T1_MEASURED_GPU_HOURS VALUE=0
SLOT T1_MEASURED_NEMO_HOURS VALUE=0
```

There are no empty slots. Every T3/T2 path above is committed, every completed
artifact or log has a concrete hash, and every unavailable science artifact is
explicitly marked blocked rather than represented by an empty value.

## T3 bound verdict

The public `nemo_dino_kamm_mlf_v1` recipe is catalog-reachable. The resolved
oracle/catalog config gate reports zero diff rows, and the five-day behavioral
gate reports zero bit differences. The catalog-built model is bit-identical to
the oracle-card model within the registered T3 scope.

`OWNERSHIP_COLLISION=FIRED` is the planted negative control. The identity JSON
records `ownership_collision_plant.field=outer_integrator` and
`ownership_collision_plant.fired=true`; it contains no live ownership
collision, and `verdict=PASS`. Do not interpret that control receipt as a T3
defect.

No T3 rerun is required. The prior executable T3 blocks and their implemented
`--config-source {oracle,catalog}` / `--catalog-recipe` selectors remain in git
history and in the certified producer; this handoff binds their completed
artifacts instead of asking the user to spend more GPU time.

## T2 stop receipt

Both corrected FE climate arms used `nemo_dino_kamm`, alpha `0.01`, generic
Nbb/Kaa selectors, the NEMO day-180 restart, ladder `both`, fp64, TKE bridge,
BEFORE bridge, T-point-stress bridge, and the NEMO seasonal epoch. Both become
unphysical by the day-1 checkpoint (`|eta|max=525.7945 m`) and then surface the
same checked error:

```text
raw-mesh e3w_int must contain only finite values > 0
```

The committed CPU-only reproducer uses the same state construction and forcing
loop with `JAX_DISABLE_JIT=1`. It completes steps 1--35, records the accelerating
SSH and velocity mode, and raises the unwrapped `EquinoxTracetimeError` at step
36 from `compute_buoyancy_frequency_nemo_bn2`, called by the GM/Redi native
slope path. The geometry check is the immediate raise site but not the
upstream instability: by step 35 `|u|max=214.312787 m s-1`, after SSH has
already reached hundreds of metres.

The five-step alpha control was too short. Alpha `0.01` delays the failure but
does not stabilize this admitted FE bundle. The exact remaining composition
owner within the FE/split-explicit barotropic path is unresolved. No additional
selector change is authorized by this evidence.

The selector bisection does not change that disposition. Restoring every
sweep-promoted evaluation selector still fails at step 37. The clean pre-#1696
target itself is fully nonfinite at its step-32/day-1 checkpoint. Thus the
premise that #1696 introduced this FE explosion is refuted, no additional
selector passed the frozen causal gate, and no new default was scoped. The
ready-to-paste addendum records that correction without weakening #1696's MLF
climate result.

Consequences:

- do not run the withdrawn 360-day FE climate pair;
- do not run the five-day FE wall pair, because its trajectory is already
  unstable;
- do not invoke the offline scorer without admitted artifacts;
- open a distinct, preregistered FE-stabilization round before reissuing T2.

T2 therefore has no statistical `CONFIRMED`/`REFUTED` verdict. Its result is a
model-scope block: the MLF bridge certification has not transferred to the FE
sibling card.

## T1 cold-start closure receipt

The executed NEMO ownership is now bound, not inferred. A from-rest start sets
`l_1st_euler=.true.` and copies `Kbb` to `Kmm`
(`src/OCE/DOM/istate.F90:107-137`). The built DINO source selects
`rDt=rn_Dt` at `MY_SRC/stpmlf.F90:134-137`, calls `mlf_baro_corr` at line 578,
and clears `l_1st_euler` only at lines 685--688. legoESM now captures the
bootstrap's pre-mixing barotropic target and raw Kaa SSH, runs implicit mixing,
then calls the existing shared corrector before the conservation fixer.

`barotropic_cold_start_after_reconcile="nemo_mlf_baro_corr"` is pinned on the
oracle and catalog DINO MLF cards. It must equal the regular-step selector;
unknown, mismatched, or non-leapfrog-family configurations raise during model
construction. The former warning implementation is removed.

The clean one-step receipt has zero admission blockers, is finite, and stamps
standalone start plus empty bridge/restart paths. It says
`claim_admissible=false` only because `steps_completed=1`, not because T1 still
has a physics blocker. Config identity remains zero-diff. Focused validation is
`213/213` plus `47/47` config/partial-cell tests.

## Frozen T1 bars

Do not substitute the historical 0.091 Sv or other round-94 floors. For every
registered scalar `s`, the only legal T1 floor is computed from the fresh six
standalone members on each side:

```text
gap_s   = mean(lego_s) - mean(nemo_s)
floor_s = sqrt(sample_sd(lego_s)^2 + sample_sd(nemo_s)^2)
R_s     = abs(gap_s) / floor_s
```

With 20,000 independent six-member bootstrap resamples at seed 1455:

- `CONFIRM` iff `R_hi <= 2.0`;
- `REFUTE` iff `R_lo > 2.0`;
- otherwise `UNRESOLVED`;
- any quantization admission failure is `UNRESOLVED_QUANTIZED`.

Each family scores its bootstrap maximum R. All six families must confirm for
`STANDALONE_STATISTICALLY_INDISTINGUISHABLE_AT_20Y`. This is a years-16--20
from-rest horizon claim, not equilibrium.

## T1 setup block — run once

This creates a clean detached worktree from the delivered branch ref and runs
only CPU admission. The bundle is the transfer artifact; the auxiliary git dir
is the current workspace's branch store.

```bash
set -euo pipefail
T1_GIT_DIR=/tmp/codex-transfer-aux.git
T1_REF=refs/heads/fidelity/dino-transfer-codex
T1_WORKTREE=/tmp/dino-transfer-t1-01a053d4-producer
T1_ROOT=/tmp/dino-transfer-01a053d4-t1
T1_MESH=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc
T1_PYTHON=/home/dbalwada/miniconda3/bin/python
T1_SHA=$(git --git-dir="$T1_GIT_DIR" rev-parse "$T1_REF")
test ! -e "$T1_WORKTREE"
test -f "$T1_MESH"
git --git-dir="$T1_GIT_DIR" worktree add --detach "$T1_WORKTREE" "$T1_SHA"
test -z "$(git -C "$T1_WORKTREE" status --porcelain --untracked-files=no)"
mkdir -p "$T1_ROOT/logs" "$T1_ROOT/lego"
T1_PYTHONPATH="$T1_WORKTREE/packages/atmosphere:$T1_WORKTREE/packages/core:$T1_WORKTREE/packages/coupler:$T1_WORKTREE/packages/ice:$T1_WORKTREE/packages/land:$T1_WORKTREE/packages/ml:$T1_WORKTREE/packages/ocean:$T1_WORKTREE/packages/tools"
PYTHONPATH="$T1_PYTHONPATH" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$T1_PYTHON" -m pytest \
  "$T1_WORKTREE/tests/ocean/unit/test_barotropic_after_reconcile.py" \
  "$T1_WORKTREE/tests/ocean/unit/test_dino_standalone_20y.py" -q
PYTHONPATH="$T1_PYTHONPATH:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226" \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$T1_PYTHON" "$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/standalone_20y_score.py" \
  --lego-root /tmp/T1_SELF_TEST_UNUSED_LEGO \
  --nemo-root /tmp/T1_SELF_TEST_UNUSED_NEMO \
  --output "$T1_ROOT/classifier_self_test.json" --self-test
```

## T1 legoESM GPU arms — three two-GPU waves

The function uses only implemented flags: `--member`, `--output-dir`,
`--reducer-mesh`, and `--steps`. `RUN_TRAJ/mesh_mask.nc` is diagnostic-only;
the runner refuses restart/bridge inputs and stamps that it did not use them.

```bash
set -euo pipefail
T1_RUNNER="$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py"
t1_run_member () {
  local T1_MEMBER="$1"
  local T1_GPU="$2"
  CUDA_VISIBLE_DEVICES="$T1_GPU" \
  PYTHONPATH="$T1_PYTHONPATH:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226" \
  JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$T1_PYTHON" "$T1_RUNNER" \
    --member "$T1_MEMBER" \
    --output-dir "$T1_ROOT/lego/m$T1_MEMBER" \
    --reducer-mesh "$T1_MESH" --steps 230400 \
    >"$T1_ROOT/logs/lego_m$T1_MEMBER.log" 2>&1
}
```

Wave 1:

```bash
t1_run_member 0 0 & T1_PID_A=$!
t1_run_member 1 1 & T1_PID_B=$!
wait "$T1_PID_A"
wait "$T1_PID_B"
```

Wave 2:

```bash
t1_run_member 2 0 & T1_PID_A=$!
t1_run_member 3 1 & T1_PID_B=$!
wait "$T1_PID_A"
wait "$T1_PID_B"
```

Wave 3:

```bash
t1_run_member 4 0 & T1_PID_A=$!
t1_run_member 5 1 & T1_PID_B=$!
wait "$T1_PID_A"
wait "$T1_PID_B"
```

Admission audit after all six finish:

```bash
PYTHONPATH="$T1_PYTHONPATH" "$T1_PYTHON" - "$T1_ROOT/lego" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for member in range(6):
    manifest = json.loads((root / f"m{member}" / "manifest.json").read_text())
    assert manifest["member"] == member
    assert manifest["steps_completed"] == 230400
    assert manifest["claim_admissible"] is True
    assert manifest["claim_admission_blockers"] == []
    assert manifest["twin_start_mode"] == "standalone"
    assert manifest["bridge_paths"] == [] and manifest["restart_paths"] == []
    assert manifest["storage_dtype"] == manifest["compute_dtype"] == "float64"
    assert manifest["barotropic_cold_start_after_reconcile"] == "nemo_mlf_baro_corr"
    assert len(manifest["sample_days"]) == 75
print("T1_LEGO_ADMISSION=PASS members=6 dates=75")
PY
find "$T1_ROOT/lego" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$T1_ROOT/lego_files.sha256"
sha256sum "$T1_ROOT/lego_files.sha256"
```

## NEMO-side need and scoring stop

T1 still needs six **fresh NEMO-owned from-rest** 20-year members: control plus
`1e-14` relative now-temperature perturbations at seeds 1--5, with every other
restart byte identical, the same 75 dates, fp64 scored storage, and the six
registered family instruments. No NEMO command is emitted here: its safe
parallel launch/initial-checkpoint machinery and the cross-model normalized
statistics producer are explicitly `BUILD_IN_NEMO_AND_NORMALIZER_ROUND`.

Recorded `RUN_20Y_REBUILD`/`RUN_40Y_REBUILD` U archives are legal only for
deterministic ACC/reducer context. Their T/S history requires a committed
validity audit. `RUN_ENS_M*`, `RUN_VERDICT360_M*`, day-180 bridge products, the
held bridge battery, and every historical floor are illegal as T1 arms or
denominators.

The committed scorer flags all exist, but do not run it until every
`lego/m0..m5` and `nemo/m0..m5` directory has a validated
`statistics.json` with schema `dino_standalone_20y_statistics_v1`. At that
point the command is:

```bash
for T1_SIDE in lego nemo; do
  for T1_MEMBER in 0 1 2 3 4 5; do
    test -f "$T1_ROOT/$T1_SIDE/m$T1_MEMBER/statistics.json"
  done
done
PYTHONPATH="$T1_PYTHONPATH:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226" \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$T1_PYTHON" "$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/standalone_20y_score.py" \
  --lego-root "$T1_ROOT/lego" --nemo-root "$T1_ROOT/nemo" \
  --output "$T1_ROOT/standalone_20y_score.json"
```

## Self-audit

- The six GPU commands above were emitted for the human and were not executed
  in this round. No NEMO, `mpirun`, push, or FE-bisect command was run.
- Every emitted runner/scorer flag is implemented by the committed parser.
  There are no empty SLOT values. Missing future artifacts say `NOT_RUN`, and
  the absent normalizer is explicitly `BUILD_IN_NEMO_AND_NORMALIZER_ROUND`.
- The T1 runner exposes no restart, bridge, `--run-traj`, or `--run-stepdump`
  flag. All six member output paths are distinct and fail closed if preexisting.
- T3 remains certified. T2 remains blocked and its withdrawn arms are not
  re-emitted. The FE registered bisect is future work, as requested.

Post future T1/T2 results to GitHub issue #1455 when the human-owned evidence
trail is updated. No remote action was taken in this round.
