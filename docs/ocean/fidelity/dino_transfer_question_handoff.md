# DINO transfer question: T3 bound; T1 IC-geometry stop

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 PERPETUAL FE OUT OF MODEL CLAIM BY USER;
T1 BLOCKED AT IC-GEOMETRY FIRST DIVERGENCE.** The step-2 ladder shows that a
359-day legoESM run started from NEMO's post-step-1 state remains twin-class.
The committed peel stops earlier on the fully independent path: wet topology,
latitude evaluation, and T-depth already differ before the Euler bootstrap.

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
SLOT T2_STATUS VALUE=PERPETUAL_FE_OUT_OF_MODEL_CLAIM_BY_USER
SLOT T2_R3_SCORE VALUE=WITHDRAWN_NO_PERPETUAL_FE_ORACLE_CLAIM
SLOT T2_MEASURED_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_STATUS VALUE=BLOCKED_AT_IC_GEOMETRY_FIRST_DIVERGENCE
SLOT T1_LADDER_SCOPE VALUE=TWIN_CLASS_DAILY_ENDPOINT_WITH_OFFSET
SLOT T1_KT2_FREE_DAYS VALUE=359
SLOT T1_KT2_ENDPOINT_OFFSET_DAYS VALUE=0.94
SLOT T1_KT2_SST_RMS_DEGC VALUE=0.0104
SLOT T1_KT2_SSH_RMS_MM VALUE=0.29
SLOT T1_INDEPENDENT_SST_RMS_DEGC VALUE=0.39
SLOT T1_KT2_ARTIFACT_SHA256 VALUE=3b9bec883ad79ad5e7dc43e95118c0f7ca903b2f39d8374eee0a74b2131661d9
SLOT T1_D10_ARTIFACT_SHA256 VALUE=759abcc2e86346abf1e25ee49d1c83535e54789cc73ecab1a7686a604e8d7062
SLOT T1_BRIDGE_PRODUCER VALUE=7c0f121baed6bc57243e849127040865dda2d597
SLOT T1_BRIDGE_DIRTY_TRACKED_FILES VALUE=1
SLOT T1_BRIDGE_DIRTY_OVERRIDE VALUE=LEGOESM_ALLOW_DIRTY=1
SLOT T1_D10_FIRST_REFUSAL VALUE=USER_REPORTED_NO_HASH
SLOT T1_IC_EULER_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_ic_euler_peel.md
SLOT T1_IC_EULER_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py
SLOT T1_IC_EULER_PROBE_SHA256 VALUE=c07a4c071f75f91270bc684dd30697036c63d38de6602872a1726281c1c502ab
SLOT T1_IC_EULER_ARTIFACT_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_ic_euler_peel_artifact.json
SLOT T1_IC_EULER_ARTIFACT_SHA256 VALUE=d7e10ed32d39182167dd3ebbcb1944965b6a6d8b8847c23fab7fc6e6b889e21f
SLOT T1_IC_EULER_FIRST_OVER_BAR VALUE=input_wet_mask
SLOT T1_IC_EULER_RUNG2_SCOPE VALUE=CONDITIONAL_AFTER_RUNG1_FAILURE
SLOT T1_NEMO_WET_LEGO_DRY_CELLS VALUE=6830
SLOT T1_NEMO_DRY_LEGO_WET_CELLS VALUE=218
SLOT T1_MAX_LATITUDE_DIFF_DEG VALUE=5.684341886080802e-14
SLOT T1_MAX_T_DEPTH_DIFF_M VALUE=104.96931566119792
SLOT T1_MAX_IC_T_DIFF_COMMON_WET_DEGC VALUE=0.0763656229991243
SLOT T1_MAX_IC_S_DIFF_COMMON_WET_PSU VALUE=0.003602246810750387
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=dbde7c322014915e6c771cfdfac5d6f2aec9df4ababaf8deeb7c4b215c8f6268
SLOT T1_CURRENT_20Y_ARMS VALUE=DIAGNOSTIC_ONLY_OLD_PRODUCER_DO_NOT_SCIENCE_SCORE
SLOT T1_NEXT_BUILD VALUE=BUILD_IN_IC_GEOMETRY_CLOSURE_ROUND
SLOT TWIN_HARNESS_SHA256 VALUE=53e5243a55c565e5c12ba4bec38246dbeb69745d622a701e947ee2a0211367e6
SLOT TWIN_RESTART_GUARD_DEFAULT VALUE=EXACT_EQUALITY_AND_RESTART_NONZERO
SLOT TWIN_FINAL_SNAPSHOT_FLAG VALUE=--snap-final
SLOT T1_AGENT_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_MEASURED_NEMO_HOURS_THIS_ROUND VALUE=0
```

There are no empty slots. `BUILD_IN_IC_GEOMETRY_CLOSURE_ROUND` is an explicit
future implementation marker, not an unimplemented flag in an executable
block.

## Bound ladder and scope

The user-supplied ladder compares three starts against the same NEMO day-360
from-rest target: day-180 twin SST RMS 0.005 degC; step-2 bridge SST RMS 0.0104
degC and SSH RMS 0.29 mm after 359 free days; independent SST RMS 0.39 degC.
The step-2 row is a daily-series endpoint with a 0.94-day offset because no
final 3-D snapshot was scheduled.

Both bridge artifacts were produced from one-dirty-file commit
`7c0f121baed6bc57243e849127040865dda2d597` with
`LEGOESM_ALLOW_DIRTY=1`. The temporary `DINO_TWIN_MIN_SPEED` override and the
first ten-day attempt's mid-run-edit refusal are recorded; the latter has no
artifact and is explicitly `USER_REPORTED_NO_HASH`.

## IC/Euler peel verdict

The clean CPU/fp64 probe passes the real step-2 bridge equality/nonzero gate
and both planted controls. Its first over-bar row is the wet mask. The
standalone and NEMO cores differ in 7,048 mask cells, latitude by up to
`5.68e-14` degree, and T-depth by up to 104.97 m. On common wet cells, the
resolved T/S IC differs by 0.0764 degC / 0.00360 PSU. Common-input profile
evaluation differs only at `4.44e-15` degC / `1.42e-14` PSU.

The Euler and filtered-carry rows are conditional diagnostics only. They cannot
own the independent gap while earlier geometry rows fail. The next build must
make the standalone analytic wet topology, T-point latitude operand, and
three-dimensional T-depth match the NEMO source path, then rerun this same
probe. Only a Rung-1 pass licenses Euler attribution or the old 20-year science
arms.

Any already-running 20-year arm from a producer before this handoff may finish
for engineering stability, but its output is not legal for the T1 model-level
score. The current runner now refuses a 230,400-step claim launch with the
measured IC-geometry blocker.

## CPU reproduction after the geometry fix

This is the only re-emitted T1 execution block. It uses committed flags and
paths and remains CPU-only. Run it after `BUILD_IN_IC_GEOMETRY_CLOSURE_ROUND`
has been replaced by an implemented, reviewed geometry closure.

```bash
set -euo pipefail
T1_WORKTREE=/tmp/dino-transfer-t1-ic-geometry-producer
T1_ROOT=/tmp/dino-transfer-01a053d4-t1-ic-geometry
T1_GIT_DIR=/tmp/codex-transfer-aux.git
T1_REF=refs/heads/fidelity/dino-transfer-codex
T1_SHA=$(git --git-dir="$T1_GIT_DIR" rev-parse "$T1_REF")
test ! -e "$T1_WORKTREE"
test ! -e "$T1_ROOT"
git --git-dir="$T1_GIT_DIR" worktree add --detach "$T1_WORKTREE" "$T1_SHA"
test -z "$(git -C "$T1_WORKTREE" status --porcelain --untracked-files=no)"
mkdir -p "$T1_ROOT"
T1_PYTHON=/home/dbalwada/legoESM/.venv/bin/python
T1_PYTHONPATH="$T1_WORKTREE/src:$T1_WORKTREE/packages/atmosphere:$T1_WORKTREE/packages/core:$T1_WORKTREE/packages/coupler:$T1_WORKTREE/packages/ice:$T1_WORKTREE/packages/land:$T1_WORKTREE/packages/ml:$T1_WORKTREE/packages/ocean:$T1_WORKTREE/packages/tools:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$T1_PYTHONPATH" JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu \
  CUDA_VISIBLE_DEVICES='' "$T1_PYTHON" \
  "$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py" \
  --run-kt2 /data/abyssal/dbalwada/RUN_KT2 \
  --run-traj-y1 /data/abyssal/dbalwada/RUN_TRAJ_Y1 \
  --lego-kt2 /data/abyssal/dbalwada/lego_bridged_kt2.npz \
  --lego-d10 /data/abyssal/dbalwada/lego_bridged_d10.npz \
  --output "$T1_ROOT/ic_euler_peel.json"
```

The output is admissible only if `first_over_bar` moves past every Rung-1 row.
Do not substitute a looser bar or mask away non-common topology; topology is
itself the current finding.

## Exact final-day 3-D capture

The twin harness now implements `--snap-final`. A future human-owned 359-day
repeat adds it beside `--save-3d`:

```text
--days 359 --save-3d --snap-final
```

The parser hard-fails `--snap-final` without `--save-3d`. The resolved snapshot
list and restart-speed threshold/source are stamped in the artifact. The
restart guard always checks state equality independently and rejects an
equal-but-zero restart.

## Self-audit

- Every slot has a nonempty value; pending work is explicitly
  `BUILD_IN_IC_GEOMETRY_CLOSURE_ROUND`.
- Every flag in the executable block is implemented by the committed parser.
- Every referenced committed relative path exists. External ladder inputs are
  SHA-256 stamped by the peel artifact.
- No T1 GPU launch block is emitted while claim admission is red. No GPU,
  NEMO, `mpirun`, push, or remote action ran in this round.
- T3 remains certified. Perpetual FE remains outside the user-selected model
  claim; no T2 scorer is reintroduced.
