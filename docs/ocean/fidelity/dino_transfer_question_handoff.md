# DINO transfer question: T1 geometry confirmed; initialization gate stops next

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 PERPETUAL FE OUT OF MODEL CLAIM BY USER;
T1 INIT GEOMETRY CONFIRMED, SOURCE-ORDER PROFILE ROW OVER BAR, EULER
WITHHELD.** No new standalone-year arm is emitted.

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
SLOT T1_STATUS VALUE=BLOCKED_AT_SOURCE_ORDER_PROFILE_AFTER_GEOMETRY_CONFIRMED
SLOT T1_GEOMETRY_STATUS VALUE=CONFIRMED_ZERO_DIFF_ALL_REGISTERED_ROWS
SLOT T1_INITIALIZATION_STATUS VALUE=INIT_GEOMETRY_PARTIAL_common_depth_T_profile
SLOT T1_EULER_STATUS VALUE=EULER_WITHHELD
SLOT T1_GEOMETRY_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_standalone_init_geometry_repair.md
SLOT T1_IC_EULER_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_ic_euler_peel.md
SLOT T1_IC_EULER_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py
SLOT T1_IC_EULER_PROBE_SHA256 VALUE=fc74401e2234ec1b3ab5586b8a3748860398067a3f9779a490ef21e65f1763a7
SLOT T1_IC_EULER_ARTIFACT_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_ic_euler_peel_artifact.json
SLOT T1_IC_EULER_ARTIFACT_SHA256 VALUE=448696a845a6f751e3acf68dd8bb31ee4956bfbb3dca70a7538a628bfc1792ec
SLOT T1_IC_EULER_PRODUCER VALUE=17a03abde73248d95feb23f184a198fb39ec9f3a
SLOT T1_IC_EULER_FIRST_OVER_BAR VALUE=common_depth_T_profile
SLOT T1_WET_MASK_MISMATCH_CELLS VALUE=0
SLOT T1_LATITUDE_EXACT_MISMATCH_VALUES VALUE=0
SLOT T1_T_DEPTH_EXACT_MISMATCH_VALUES VALUE=0
SLOT T1_MAX_PROFILE_T_DIFF_DEGC VALUE=4.440892098500626e-15
SLOT T1_MAX_PROFILE_S_DIFF_PSU VALUE=1.4210854715202004e-14
SLOT T1_MAX_RESOLVED_IC_T_DIFF_DEGC VALUE=0.04079998207163005
SLOT T1_MAX_RESOLVED_IC_S_DIFF_PSU VALUE=0.003602379509992204
SLOT T1_FINAL_DEPTH_MASK_PLANT_MISMATCH_CELLS VALUE=929
SLOT T1_NEXT_REPAIR VALUE=SOURCE_ORDER_PROFILE_AND_LIVE_ANCHOR_EVALUATION
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=dbde7c322014915e6c771cfdfac5d6f2aec9df4ababaf8deeb7c4b215c8f6268
SLOT T1_CURRENT_20Y_ARMS VALUE=NO_NEW_ARM_OLD_PRODUCERS_DIAGNOSTIC_ONLY
SLOT T1_LADDER_SCOPE VALUE=TWIN_CLASS_DAILY_ENDPOINT_WITH_OFFSET
SLOT T1_KT2_FREE_DAYS VALUE=359
SLOT T1_KT2_ENDPOINT_OFFSET_DAYS VALUE=0.94
SLOT T1_KT2_SST_RMS_DEGC VALUE=0.0104
SLOT T1_KT2_SSH_RMS_MM VALUE=0.29
SLOT T1_INDEPENDENT_SST_RMS_DEGC VALUE=0.39
SLOT T1_KT2_ARTIFACT_SHA256 VALUE=3b9bec883ad79ad5e7dc43e95118c0f7ca903b2f39d8374eee0a74b2131661d9
SLOT T1_D10_ARTIFACT_SHA256 VALUE=759abcc2e86346abf1e25ee49d1c83535e54789cc73ecab1a7686a604e8d7062
SLOT TWIN_HARNESS_SHA256 VALUE=53e5243a55c565e5c12ba4bec38246dbeb69745d622a701e947ee2a0211367e6
SLOT TWIN_FINAL_SNAPSHOT_FLAG VALUE=--snap-final
SLOT T1_AGENT_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_MEASURED_NEMO_HOURS_THIS_ROUND VALUE=0
```

There are no empty slots or build markers.

## Bound result

The initialization geometry now reproduces the NEMO physical core exactly:
zero wet-mask differences across 327,600 cells, zero bit differences across
9,360 T latitudes, and zero bit differences for `gdept_0` at every NEMO-wet T
cell. The source owners are:

- `usrdef_hgr.F90:96,106` and legoESM
  `packages/core/legoesm/grids/latlon.py:723-736` for Mercator latitude;
- `zgr_lib.F90:161-189` and legoESM's extended
  `create_levy_stretched_z_star` for the transitioned 3-D T depths; and
- `usrdef_zgr.F90:112-120` and legoESM
  `packages/ocean/legoesm/ocean/experiments/dino.py:2763-2777` for the
  separate one-dimensional `pdept_1d` wet-mask operand.

The old final-`gdept` mask choice is a live red control: it changes exactly
929 cells. The old seam wall, JAX latitude, first-pass-only final-depth, field,
shape, and initialization-admission plants all fire.

The next registered row fails. Common-depth profile evaluation differs by at
most `4.44e-15 degC` and `1.42e-14 PSU`, above the frozen `1e-15` pointwise
bar. Resolved fields differ by `0.0408000 degC` and `0.00360238 PSU` because
the standalone analytic initializer still uses its nominal latitude/bottom
anchors; supplying NEMO's live anchors on the repaired geometry produces exact
T/S. This is the next owner.

The probe is fail-closed: because initialization admission is false, it does
not invoke `_step_rows`. No Euler/carry values are present in the v3 artifact.
The year criterion in the preregistration is therefore false, so there is no
GPU arm to run with `--snap-final` yet.

## CPU reproduction

This block uses only implemented flags and committed paths. It regenerates the
same clean CPU artifact; it is not a GPU/NEMO/MPI arm.

```bash
set -euo pipefail
T1_WORKTREE=/tmp/dino-transfer-t1-init-score
T1_ROOT=/tmp/dino-transfer-01a053d4-t1-init-score
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
cd "$T1_WORKTREE"
PYTHONPATH="$T1_PYTHONPATH" JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$T1_PYTHON" \
  "$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py" \
  --run-kt2 /data/abyssal/dbalwada/RUN_KT2 \
  --run-traj-y1 /data/abyssal/dbalwada/RUN_TRAJ_Y1 \
  --lego-kt2 /data/abyssal/dbalwada/lego_bridged_kt2.npz \
  --lego-d10 /data/abyssal/dbalwada/lego_bridged_d10.npz \
  --output "$T1_ROOT/ic_euler_peel.json"
```

The regenerated artifact must report the three geometry rows `PASS`,
`first_over_bar=common_depth_T_profile`, and `euler_outcome=EULER_WITHHELD`.

## Self-audit

- Every receipt slot is nonempty, every relative path exists, and every flag
  in the executable block is implemented by the committed parser.
- No GPU launch block is emitted because the registered year-admission rule is
  false. `--snap-final` remains implemented on the twin harness for the next
  legally admitted final-day 3-D run.
- T3 remains certified and the user-selected T2 scope remains unchanged.
- No GPU, NEMO execution, `mpirun`, push, or remote action ran in this round.
