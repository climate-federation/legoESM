# DINO transfer question: initialization exact; Euler debt is launch blocker

Date: 2026-08-31. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 PERPETUAL FE OUT OF MODEL CLAIM BY USER;
T1 INIT_CONFIRMED, EULER_DEBT, STANDALONE-YEAR ARM WITHHELD.**

## Receipt slots

```text
SLOT TRANSFER_PRODUCER_REF VALUE=refs/heads/fidelity/dino-transfer-codex
SLOT TRANSFER_SESSION_ID VALUE=01a053d4-8e9f-7212-bbdb-19ba2d64e140
SLOT T3_STATUS VALUE=TRANSFER_CONFIRMED
SLOT T3_CATALOG_RECIPE VALUE=nemo_dino_kamm_mlf_v1
SLOT T3_IDENTITY_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py
SLOT T3_IDENTITY_ARTIFACT_SHA256 VALUE=cae74e17dd2e3c60f592c203b398a9d1d15790532be1387090ad9dc35fdd27fe
SLOT T3_BEHAVIOR_IDENTITY_ARTIFACT_SHA256 VALUE=f52990b76afe44cc5a045fd3d2d8e7c10c311a79838bc83296fd6b873cf28169
SLOT T2_STATUS VALUE=PERPETUAL_FE_OUT_OF_MODEL_CLAIM_BY_USER
SLOT T2_MEASURED_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_STATUS VALUE=BLOCKED_AT_COUPLED_TRAADV_PAIR_INIT_EXACT
SLOT T1_INITIALIZATION_STATUS VALUE=INIT_CONFIRMED
SLOT T1_EULER_STATUS VALUE=EULER_DEBT_T_after_traadv
SLOT T1_RULE_1B_STATUS VALUE=INELIGIBLE_COUPLED_OPERATOR_RESIDUAL
SLOT T1_IC_EULER_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_cold_euler_operator_peel.md
SLOT T1_IC_EULER_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py
SLOT T1_IC_EULER_PROBE_SHA256 VALUE=042a366e49e90787b782ba3f4c7111530df3116beabb18fab29c7fff0805077f
SLOT T1_IC_EULER_ARTIFACT_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_ic_euler_peel_artifact.json
SLOT T1_IC_EULER_ARTIFACT_SHA256 VALUE=92df136d97c24f4bbf9333db3907b446dd824c35964d1e880c29ca6f0851575b
SLOT T1_IC_EULER_PRODUCER VALUE=7c6952b5f2eef3ac52e4a75f296f1c4072071ebb
SLOT T1_IC_EULER_FIRST_OVER_BAR VALUE=T_after_traadv
SLOT T1_WET_MASK_MAX_DIFF VALUE=0
SLOT T1_LATITUDE_MAX_DIFF_DEG VALUE=0
SLOT T1_T_DEPTH_MAX_DIFF_M VALUE=0
SLOT T1_MAX_PROFILE_T_DIFF_DEGC VALUE=0
SLOT T1_MAX_PROFILE_S_DIFF_PSU VALUE=0
SLOT T1_MAX_RESOLVED_IC_T_DIFF_DEGC VALUE=0
SLOT T1_MAX_RESOLVED_IC_S_DIFF_PSU VALUE=0
SLOT T1_MAX_TRAADV_T_DIFF_KPS VALUE=1.935793899665525e-9
SLOT T1_MAX_TRAADV_S_DIFF_PSUPS VALUE=1.6939489048408599e-10
SLOT T1_MAX_EULER_T_DIFF_DEGC VALUE=4.132883526075659e-5
SLOT T1_MAX_EULER_S_DIFF_PSU VALUE=4.855222066169063e-6
SLOT T1_MAX_EULER_U_DIFF_MPS VALUE=6.939598505437128e-6
SLOT T1_MAX_EULER_V_DIFF_MPS VALUE=3.193385225393544e-8
SLOT T1_MAX_EULER_SSH_DIFF_M VALUE=1.7069679003611782e-15
SLOT T1_FSLOW_U_OWNER VALUE=WIND_STRESS_PROJECTION_CLOSED
SLOT T1_FSLOW_U_MAX_DIFF_MPS2 VALUE=1.1712877473750872e-21
SLOT T1_FSLOW_V_OWNER VALUE=HPG_ACCUMULATION_CLOSED
SLOT T1_FSLOW_V_MAX_DIFF_MPS2 VALUE=4.129285617864714e-20
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=b7f4ad05cf823ae4c127b5bb5770b75bcd6e74cb48c9ccb62d3c1e24949b5a2a
SLOT T1_STANDALONE_FINAL_SNAPSHOT_FLAG VALUE=--snap-final
SLOT T1_STANDALONE_YEAR_ARM VALUE=WITHHELD_FROZEN_EULER_GATE_FALSE
SLOT T1_PREDICTION VALUE=INDEPENDENT_YEAR_SST_RMS_COLLAPSES_0.39C_TOWARD_0.01C_AFTER_ADMISSION
SLOT T1_CONFIRM_BAR_DEGC VALUE=0.02
SLOT T1_REFUTE_BAR_DEGC VALUE=0.10
SLOT T1_KT2_SST_RMS_DEGC VALUE=0.0104
SLOT T1_KT2_SSH_RMS_MM VALUE=0.29
SLOT T1_AGENT_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_MEASURED_NEMO_HOURS_THIS_ROUND VALUE=0
```

There are no empty slots or `BUILD_IN_THIS_ROUND` markers.

## Bound outcome

The v7 artifact admits initialization with exact zero differences on all seven
registered rows. The cold Euler RHS then passes `tra_sbc` and `tra_qsr`; its
first failure is `T_after_traadv` at `1.935793899665525e-9 K/s`. NEMO's
recorded FCT face fluxes localize this to a cancelling horizontal/vertical
pair, so the standing pair rule forbids a one-component patch. U wind and V
HPG slow-forcing rows are now independently at bar (`1.17e-21` and
`4.13e-20 m/s2`) and are exonerated.

Consequently no GPU arm is emitted. The runner does implement and stamp
`--snap-final`, but its claim-length guard now refuses this exact Euler debt.
The frozen prediction remains registered for the first legally admitted run:
day-360 SST RMS should collapse from `0.39 degC` toward the `~0.01 degC`
step-2 bridge class; `<=0.02 degC` confirms and `>=0.10 degC` refutes.

## CPU reproduction

Every flag below is implemented by the committed parser and every input path
is named above. This block runs no GPU, NEMO executable, or MPI process.

```bash
set -euo pipefail
T1_WORKTREE=/tmp/dino-transfer-t1-euler-score
T1_ROOT=/tmp/dino-transfer-01a053d4-t1-euler-score
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
  scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py \
  --run-kt2 /data/abyssal/dbalwada/RUN_KT2 \
  --run-traj-y1 /data/abyssal/dbalwada/RUN_TRAJ_Y1 \
  --lego-kt2 /data/abyssal/dbalwada/lego_bridged_kt2.npz \
  --lego-d10 /data/abyssal/dbalwada/lego_bridged_d10.npz \
  --output "$T1_ROOT/ic_euler_peel.json"
```

Expected receipts are `INIT_CONFIRMED`, `EULER_DEBT_T_after_traadv`, and
`FIRST_OVER_BAR=T_after_traadv`.

## Self-audit

- Every slot is nonempty; every relative path exists.
- `ic_euler_peel.py --help` implements all five flags used above.
- `standalone_20y.py --help` implements `--member`, `--output-dir`,
  `--reducer-mesh`, `--steps`, and `--snap-final`; no unimplemented standalone
  launch block is present because the gate is false.
- Planted field, shape, initialization-admission, legacy geometry/profile, and
  reducer-shape controls fire.
- CPU admission suites: the DINO experiment plus leapfrog files completed
  their first 177 tests green in the combined run; the documented per-process
  compiler limit then aborted in the following file. Fresh processes pass
  `test_barotropic_after_reconcile.py` 41/41 and
  `test_dino_standalone_20y.py` 20/20. The four initially exposed stale
  synthetic fixtures were corrected and their focused rerun passes 4/4.
- No GPU, NEMO execution, `mpirun`, push, or remote action ran in this round.
