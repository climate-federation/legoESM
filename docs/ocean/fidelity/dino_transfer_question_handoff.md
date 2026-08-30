# DINO transfer question: scoped T2 probe and repaired T1 rerun handoff

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 PERPETUAL FE OUT OF MODEL CLAIM BY USER;
T1 REPAIRED ARMS READY FOR RERUN / SCIENCE UNMEASURED.** The AB3/AM4 fix
remains because it matches NEMO's actual one Euler bootstrap. No further
perpetual-FE stabilization is in scope. The T2 handoff starts with one cheap
five-day probe and gates any optional climate diagnostic on it. The T1 blocks
use fresh r2 paths and a runner that executes its live reducer at day 0.

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
SLOT T2_STATUS VALUE=PERPETUAL_FE_OUT_OF_MODEL_CLAIM_BY_USER
SLOT T2_FE_CLIMATE_R2_A_ARTIFACT VALUE=BLOCKED_NO_ARTIFACT
SLOT T2_FE_CLIMATE_R2_A_LOG_SHA256 VALUE=e07de38ab808ce86d46ba813b488b3c4661c9bed9b5627eeb5479685971db664
SLOT T2_FE_CLIMATE_R2_B_ARTIFACT VALUE=BLOCKED_NO_ARTIFACT
SLOT T2_FE_CLIMATE_R2_B_LOG_SHA256 VALUE=637b769cba5c81dd6f3080d22ad3322e64d29bb65be6bc986258a23a80ce4452
SLOT T2_R2_FE_WALL_ARTIFACTS VALUE=HISTORICAL_BLOCKED_NOT_RUN
SLOT T2_R2_SCORE_ARTIFACT VALUE=HISTORICAL_BLOCKED_NOT_RUN
SLOT T2_CPU_REPRO_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/fe_stability_repro.py
SLOT T2_CPU_REPRO_SCRIPT_SHA256 VALUE=0ea386a3248706a8f40dfd74226a0f3eb1ce905a6a7bf72751ef16bd339718d8
SLOT T2_CPU_REPRO_RECEIPT_SHA256 VALUE=3adc8ba20f26dec0631b7118c5f881abd3f630391c808067ccf7725435a5281e
SLOT T2_CPU_REPRO_LOG_SHA256 VALUE=24d0c63f2208a476d70e011587a341de00e6964a57b638a84d332042b0d0489b
SLOT T2_SCOPE_BISECT_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_fe_default_scope_bisect.md
SLOT T2_ALL_RESTORED_RECEIPT_SHA256 VALUE=6392216940040b02184f1c2d4a3da99adab1dfda55a7f833f7bc3a0318cbe4d1
SLOT T2_ALL_RESTORED_LOG_SHA256 VALUE=a70c489bae288b657113401ee974928d66a6fd78e7bfaf0c320b07e5c9fa6fed
SLOT T2_PRE1696_COMMIT VALUE=b794c0618e287ebf1d364a8713c3c504ac2eb01c
SLOT T2_PRE1696_ARTIFACT_SHA256 VALUE=9f97a8ba82800672116a7f60c1cb1043dfc1aab7b51ab230311bced1284f2839
SLOT T2_PRE1696_LOG_SHA256 VALUE=4f5f128c7ea83ebfe3d81a41da4b29146b4d744e0862a573b3c9f04605218464
SLOT T2_PR1696_ADDENDUM_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_pr1696_t2_addendum.md
SLOT T2_PLAIN_FILTER_FASTTERM_RECEIPT_SHA256 VALUE=c2623575caf21283f0abe820873f94611f5d401cd3f32e611fd2cad967dcf26c
SLOT T2_PLAIN_FILTER_FASTTERM_LOG_SHA256 VALUE=15936968c8cb9197b6ec8568d369c64bbb8bf4d004c425c05096292274790a38
SLOT T2_REFUTED_CORRECTOR_RECEIPT_SHA256 VALUE=5ec1765659fc68d3f56f3cd0255b027b162734cc9e66eeedde82f90b8f84a91f
SLOT T2_REFUTED_CORRECTOR_LOG_SHA256 VALUE=e6a8394dbcc69547e27f78c001b19b38ed156cf29610a86bc1aec3d371be7223
SLOT T2_AB3AM4_ARM40_RECEIPT_SHA256 VALUE=eb7a378a8eb5c85294e0591db17927bfc1f52fb12c6e7fbf3700aebef794c92e
SLOT T2_AB3AM4_ARM40_LOG_SHA256 VALUE=7c91c5c77732b0f979a05de477f3ff744aaecab03226187c9a8e0902c90c7a2d
SLOT T2_DEFAULT64_RECEIPT_SHA256 VALUE=1d2da9994bc18246a05de5ac20ad5d333ee1e42b83447e2cbac18afdb735ef38
SLOT T2_DEFAULT64_LOG_SHA256 VALUE=1d5649ffe67344ace9125be863e5baf2f8447b21bb866d8c562a9fe2cbd5e724
SLOT T2_SCOPED_RUN_ROOT VALUE=/tmp/dino-transfer-01a053d4-t2scope
SLOT T2_R3_5DAY_ADMISSION_SHA256 VALUE=MEASURED_IF_RUN
SLOT T2_R3_CLIMATE_A_SHA256 VALUE=CONDITIONAL_DIAGNOSTIC_ONLY
SLOT T2_R3_CLIMATE_B_SHA256 VALUE=CONDITIONAL_DIAGNOSTIC_ONLY
SLOT T2_R3_SCORE VALUE=WITHDRAWN_NO_PERPETUAL_FE_ORACLE_CLAIM
SLOT T2_MEASURED_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_STATUS VALUE=REPAIRED_ARMS_READY_FOR_FRESH_RERUN_SCIENCE_UNMEASURED
SLOT T1_COLD_START_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_standalone_cold_start_corrector.md
SLOT T1_COLD_START_VALIDATED_BUILD_PRODUCER_SHA VALUE=28be310c6ab487c3fa685063af6554abea5938f2
SLOT T1_COLD_START_CORE_SHA256 VALUE=ddb0c414a3d9f44073fed646516b9dbe463a4a9cac42b721b54aff94d26ef8cf
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=fefb22f0356711bf86304ba17cf44a07ef72a4bb085881fb8a63f9d568496988
SLOT T1_REDUCER_FIX_COMMIT VALUE=8209506a38c91aff9e293192eb6c18ff37118ff4
SLOT T1_REDUCER_CONVENTION_SCHEMA VALUE=dino_standalone_reducer_frame_v1
SLOT T1_REDUCER_DAY0_RECEIPT_SHA256 VALUE=afcbaf82948e50747f2cf66911d758bfdcfba23cf60062e4380356542d4adcfc
SLOT T1_REDUCER_DAY0_MANIFEST_SHA256 VALUE=917a9020e8b008244dea8b371da941fded9ceff8a18ec6b638a9bbec4cfd7ecf
SLOT T1_REDUCER_DAY0_LOG_SHA256 VALUE=7839ef464a7b96d734efb9fdd8c9f73e2c2788b64aa2ebf2a9a678f7eedb9992
SLOT T1_ABORTED_R1_360D_STATE VALUE=USER_REPORTED_CLEAN_FINITE_NO_ARTIFACT_HASH
SLOT T1_ABORTED_R1_FAILURE VALUE=DAY360_REDUCER_FRAME_MISMATCH
SLOT T1_SCORE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y_score.py
SLOT T1_SCORE_SCRIPT_SHA256 VALUE=de121565dc4ecdd9b2543886f2f240ed53f994e1c0f1622e5b62aaffe71d32f9
SLOT T1_CONFIG_IDENTITY_R2_SHA256 VALUE=c8f885a669734ed03f9bfd769eca5e367488bb34dcc21a1b52f07f676de73fe1
SLOT T1_PRE_REDUCER_CPU_SMOKE_MANIFEST_SHA256 VALUE=41c044ad8e13e1ce9745a94b2d0724451b32d113f4ed1c74f74434d28217ebc1
SLOT T1_PRE_REDUCER_CPU_SMOKE_INITIAL_RECEIPT_SHA256 VALUE=a2d69016458af5b11258b5975841246bbaa9d2404624050c06f19db55fef7951
SLOT T1_LEGO_STANDALONE_ROOT VALUE=/tmp/dino-transfer-01a053d4-t1r2/lego
SLOT T1_NEMO_STANDALONE_ROOT VALUE=HUMAN_TO_CREATE_FROM_REST
SLOT T1_NORMALIZED_STATISTICS_PRODUCER VALUE=BUILD_IN_NEMO_AND_NORMALIZER_ROUND
SLOT T1_NEMO_ENSEMBLE_MANIFEST_SHA256 VALUE=NOT_RUN
SLOT T1_LEGO_ENSEMBLE_MANIFEST_SHA256 VALUE=NOT_RUN
SLOT T1_SCORE_SHA256 VALUE=NOT_RUN
SLOT T1_AGENT_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_ABORTED_R1_GPU_COST VALUE=USER_REPORTED_APPROX_ONE_HOUR
SLOT T1_MEASURED_NEMO_HOURS VALUE=0
```

There are no empty slots. Every completed artifact or log available to this
workspace has a concrete hash. The unavailable aborted-T1 receipt is explicitly
user-reported, while optional T2 outputs say `MEASURED_IF_RUN` or
`CONDITIONAL_DIAGNOSTIC_ONLY` rather than using empty values.

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

## T2 CPU admission receipt and scope

The r2 arms used `nemo_dino_kamm`, alpha `0.01`, generic Nbb/Kaa selectors, the
NEMO day-180 restart, ladder `both`, fp64, TKE bridge, BEFORE bridge,
T-point-stress bridge, and the NEMO seasonal epoch. Both became unphysical by
the day-1 checkpoint (`|eta|max=525.7945 m`) and surfaced:

```text
raw-mesh e3w_int must contain only finite values > 0
```

The extended CPU probe localizes the upstream injection to the live
continuity/surface-PGF fast pair. At step 25, surface PGF is
`0.90269 m s-2`, versus slow forcing `3.3084e-5 m s-2` and EEN Coriolis
`2.308e-4 m s-2`; continuity divergence is `22.1698 m s-1`. Repeating the
after-level corrector is refuted by the same step-37 failure and is not shipped.

NEMO's one Euler bootstrap still executes nn_bt_flt=2's AB3 velocity predictor
and `ts_bck_interp` SSH interpolation. Changing only the FE card's temporal
filter from the plain boxcar to `nemo_boxcar_ab3` completes 40 steps, and the
promoted clean card completes 64. The old selector is the red step-36 control.
NEMO switches from Euler to MLF after that first step, so this admits a
perpetual-Euler stabilization but not an oracle-faithful trajectory.

The user's final decision limits the model claim to NEMO-covered use: one Euler
bootstrap followed by leapfrog. Perpetual-Euler longevity is out of scope. The
64-step receipt is retained as engineering evidence, but T2 cannot issue a
perpetual-FE oracle-fidelity verdict. Run only the five-day admission first.
The climate duplicates below are optional diagnostics and are hard-gated on
that artifact; the offline fidelity scorer is withdrawn.

## T2 scoped setup and CPU admission — run once

Every flag below was checked against the committed parser. The setup resolves
the delivered branch from its actual auxiliary Git directory, creates a clean
detached producer, and pins the unchanged round-94 instruments by hash.

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a053d4-8e9f-7212-bbdb-19ba2d64e140
T2_GIT_DIR=/tmp/codex-transfer-aux.git
T2_REF=refs/heads/fidelity/dino-transfer-codex
T2_ROOT=/tmp/dino-transfer-01a053d4-t2scope
T2_WORKTREE=/tmp/dino-transfer-t2scope-01a053d4-producer
T2_NEMO=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
T2_PYTHON=/home/dbalwada/legoESM/.venv/bin/python
T2_SHA=$(git --git-dir="$T2_GIT_DIR" rev-parse "$T2_REF")
test ! -e "$T2_ROOT"
test ! -e "$T2_WORKTREE"
test -d "$T2_NEMO/RUN_TRAJ"
test -d "$T2_NEMO/RUN_STEPDUMP"
git --git-dir="$T2_GIT_DIR" worktree add --detach "$T2_WORKTREE" "$T2_SHA"
test "$(git -C "$T2_WORKTREE" rev-parse HEAD)" = "$T2_SHA"
test -z "$(git -C "$T2_WORKTREE" status --porcelain --untracked-files=no)"
mkdir -p "$T2_ROOT/arms" "$T2_ROOT/logs" "$T2_ROOT/receipts"
printf '%s\n' "$T2_SHA" > "$T2_ROOT/receipts/producer_commit.txt"
T2_PYTHONPATH="$T2_WORKTREE/packages/atmosphere:$T2_WORKTREE/packages/core:$T2_WORKTREE/packages/coupler:$T2_WORKTREE/packages/ice:$T2_WORKTREE/packages/land:$T2_WORKTREE/packages/ml:$T2_WORKTREE/packages/ocean:$T2_WORKTREE/packages/tools:$T2_WORKTREE/scripts/validate/ocean_fidelity/dino_1226"
test "$(sha256sum "$T2_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py" | awk '{print $1}')" = \
  7eb37dec1ebee1179b1dc7fe79f68325196f15ee84b538b489cfd2c348fb079f
test "$(sha256sum "$T2_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/fe_stability_repro.py" | awk '{print $1}')" = \
  0ea386a3248706a8f40dfd74226a0f3eb1ce905a6a7bf72751ef16bd339718d8
PYTHONPATH="$T2_PYTHONPATH" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 "$T2_PYTHON" -m pytest -q \
  "$T2_WORKTREE/tests/ocean/unit/test_fe_stability_repro.py" \
  "$T2_WORKTREE/tests/ocean/unit/test_nemo_ab3am4_filter.py" \
  "$T2_WORKTREE/tests/ocean/unit/test_dino_experiment.py"
```

## T2.1 — one cheap user GPU FE admission probe

This probe resolves the AB3/AM4 fix from the shipped card and spends only five
model days. Stop T2 if it does not write the exact stable artifact line.

```bash
set -euo pipefail
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
T2_TWIN="$T2_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
T2_RUN_FP64="$T2_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
T2_ADMISSION=fe_admission_5d
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$T2_PYTHONPATH" \
  "$T2_PYTHON" "$T2_RUN_FP64" "$T2_TWIN" \
  nemo_dino_kamm "$T2_ROOT/arms/$T2_ADMISSION.npz" \
  --days 5 --save-step-eta --save-3d --snap-days 0,5 --fp64-3d \
  --run-traj "$T2_NEMO/RUN_TRAJ" \
  --run-stepdump "$T2_NEMO/RUN_STEPDUMP" \
  --bridge-tke --bridge-before --bridge-before-stress-tpoint \
  >"$T2_ROOT/logs/$T2_ADMISSION.log" 2>&1
grep -F "SAVED $T2_ROOT/arms/$T2_ADMISSION.npz  stable=True" \
  "$T2_ROOT/logs/$T2_ADMISSION.log"
sha256sum "$T2_ROOT/arms/$T2_ADMISSION.npz" \
  "$T2_ROOT/logs/$T2_ADMISSION.log"
```

## T2.2 — conditional user GPU FE climate diagnostics

These perpetual-FE arms have no NEMO trajectory analogue and cannot support a
transfer verdict. Run them only if T2.1 passed and the diagnostic is still
worth its cost.

```bash
set -euo pipefail
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
test -f "$T2_ROOT/arms/fe_admission_5d.npz"
grep -F "SAVED $T2_ROOT/arms/fe_admission_5d.npz  stable=True" \
  "$T2_ROOT/logs/fe_admission_5d.log"
t2_run_climate_diagnostic () {
  local T2_GPU="$1"
  local T2_ARM="$2"
  CUDA_VISIBLE_DEVICES="$T2_GPU" PYTHONPATH="$T2_PYTHONPATH" \
    "$T2_PYTHON" "$T2_RUN_FP64" "$T2_TWIN" \
    nemo_dino_kamm "$T2_ROOT/arms/$T2_ARM.npz" \
    --days 360 --save-3d --snap-days 0,90,360 --fp64-3d \
    --run-traj "$T2_NEMO/RUN_TRAJ" \
    --run-stepdump "$T2_NEMO/RUN_STEPDUMP" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    >"$T2_ROOT/logs/$T2_ARM.log" 2>&1
}
t2_run_climate_diagnostic 0 fe_climate_r3_a & T2_PID_A=$!
t2_run_climate_diagnostic 1 fe_climate_r3_b & T2_PID_B=$!
T2_RC=0
wait "$T2_PID_A" || T2_RC=1
wait "$T2_PID_B" || T2_RC=1
test "$T2_RC" -eq 0
for T2_ARM in fe_climate_r3_a fe_climate_r3_b; do
  grep -F "SAVED $T2_ROOT/arms/$T2_ARM.npz  stable=True" \
    "$T2_ROOT/logs/$T2_ARM.log"
  sha256sum "$T2_ROOT/arms/$T2_ARM.npz" "$T2_ROOT/logs/$T2_ARM.log"
done
```

There is intentionally no T2.3 scorer block. A scorer verdict on a perpetual-
Euler trajectory would exceed the user's NEMO-covered model scope.

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
`196/196` across the repaired runner, cold-start corrector, FE filter, and DINO
recipe suites.

The first human GPU attempt then reached its first scheduled day-360 capture
with a user-reported clean, finite cold-start trajectory, but the live reducer
failed on `(199,52,36)` NEMO weights versus `(195,48,1)` standalone fields.
The abort preceded snapshot/manifest output, so it is a runnability receipt,
not a citable science artifact. The repaired runner now stamps
`dino_standalone_reducer_frame_v1`, asserts T/S/eta/U/V/land-mask and every
mesh-weight shape at day 0, fires a mismatch plant, and executes the real
reducer before step 1. The clean CPU receipt reports `REDUCER_DAY0=PASS` and
all 12 reduction keys.

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
T1_WORKTREE=/tmp/dino-transfer-t1r2-01a053d4-producer
T1_ROOT=/tmp/dino-transfer-01a053d4-t1r2
T1_MESH=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc
T1_PYTHON=/home/dbalwada/miniconda3/bin/python
T1_SHA=$(git --git-dir="$T1_GIT_DIR" rev-parse "$T1_REF")
test ! -e "$T1_WORKTREE"
test ! -e "$T1_ROOT"
test -f "$T1_MESH"
git --git-dir="$T1_GIT_DIR" worktree add --detach "$T1_WORKTREE" "$T1_SHA"
test -z "$(git -C "$T1_WORKTREE" status --porcelain --untracked-files=no)"
mkdir -p "$T1_ROOT/logs" "$T1_ROOT/lego"
T1_PYTHONPATH="$T1_WORKTREE/packages/atmosphere:$T1_WORKTREE/packages/core:$T1_WORKTREE/packages/coupler:$T1_WORKTREE/packages/ice:$T1_WORKTREE/packages/land:$T1_WORKTREE/packages/ml:$T1_WORKTREE/packages/ocean:$T1_WORKTREE/packages/tools"
T1_RUNNER="$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py"
test "$(sha256sum "$T1_RUNNER" | awk '{print $1}')" = \
  fefb22f0356711bf86304ba17cf44a07ef72a4bb085881fb8a63f9d568496988
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
PYTHONPATH="$T1_PYTHONPATH:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226" \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$T1_PYTHON" "$T1_RUNNER" \
  --member 0 --output-dir "$T1_ROOT/smoke_m0" \
  --reducer-mesh "$T1_MESH" --steps 1 \
  >"$T1_ROOT/logs/smoke_m0.log" 2>&1
grep -F 'REDUCER_DAY0=PASS convention=dino_standalone_reducer_frame_v1 plant=FIRED' \
  "$T1_ROOT/logs/smoke_m0.log"
PYTHONPATH="$T1_PYTHONPATH" "$T1_PYTHON" - "$T1_ROOT/smoke_m0" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipt = root / "reducer_convention_receipt.json"
manifest = json.loads((root / "manifest.json").read_text())
data = json.loads(receipt.read_text())
assert data["schema"] == "dino_standalone_reducer_frame_v1"
assert data["day0_reduction_status"] == "PASS"
assert data["shape_mismatch_plant"] == "FIRED"
assert data["asserted_input_names"] == ["S", "T", "eta", "land_mask", "u", "v"]
assert manifest["steps_completed"] == 1 and manifest["claim_admissible"] is False
assert manifest["reducer_convention_schema"] == data["schema"]
assert manifest["reducer_convention_receipt_sha256"] == hashlib.sha256(
    receipt.read_bytes()).hexdigest()
print("T1_REDUCER_DAY0_ADMISSION=PASS")
PY
```

## T1 legoESM GPU arms — three two-GPU waves

The function uses only implemented flags: `--member`, `--output-dir`,
`--reducer-mesh`, and `--steps`. `RUN_TRAJ/mesh_mask.nc` is diagnostic-only;
the runner refuses restart/bridge inputs and stamps that it did not use them.

```bash
set -euo pipefail
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
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for member in range(6):
    directory = root / f"m{member}"
    manifest = json.loads((directory / "manifest.json").read_text())
    receipt_path = directory / "reducer_convention_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert manifest["member"] == member
    assert manifest["steps_completed"] == 230400
    assert manifest["claim_admissible"] is True
    assert manifest["claim_admission_blockers"] == []
    assert manifest["twin_start_mode"] == "standalone"
    assert manifest["bridge_paths"] == [] and manifest["restart_paths"] == []
    assert manifest["storage_dtype"] == manifest["compute_dtype"] == "float64"
    assert manifest["barotropic_cold_start_after_reconcile"] == "nemo_mlf_baro_corr"
    assert manifest["reducer_convention_schema"] == "dino_standalone_reducer_frame_v1"
    assert manifest["reducer_convention_receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()).hexdigest()
    assert receipt["day0_reduction_status"] == "PASS"
    assert receipt["shape_mismatch_plant"] == "FIRED"
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

- The one T2 five-day admission and conditional climate diagnostics were
  emitted for the human and were not executed here. The earlier T1 wave was
  stopped by the day-360 reducer failure; no old output is reused. No GPU,
  NEMO, `mpirun`, push, or remote command was run by this round.
- Every emitted T2 and T1 runner/scorer flag is implemented by the committed
  parser. There are no empty SLOT values. Pending T2 products say
  `MEASURED_IF_RUN` or `CONDITIONAL_DIAGNOSTIC_ONLY`; the absent T1 normalizer remains explicitly
  `BUILD_IN_NEMO_AND_NORMALIZER_ROUND`.
- The T1 runner exposes no restart, bridge, `--run-traj`, or `--run-stepdump`
  flag. All six member output paths are distinct and fail closed if preexisting.
- The T1 setup uses fresh `t1r2` worktree/output paths, pins the repaired runner
  hash, and executes the live reducer before any GPU command. Every full member
  must carry the matching convention receipt.
- T3 remains certified. Perpetual FE is out of the model claim by user choice;
  the offline T2 scorer is intentionally absent. No further FE stabilization
  is authorized.

Post future T1/T2 results to GitHub issue #1455 when the human-owned evidence
trail is updated. No remote action was taken in this round.
