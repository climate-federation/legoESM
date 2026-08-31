# DINO transfer question: T1 Euler admitted; standalone-year arm released

Date: 2026-08-31. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 PERPETUAL FE OUT OF MODEL CLAIM BY USER;
T1 INIT_CONFIRMED + EULER_AT_BAR; STANDALONE-YEAR ARM RELEASED.**

## Receipt slots

```text
SLOT TRANSFER_PRODUCER_REF VALUE=refs/heads/fidelity/dino-transfer-codex
SLOT TRANSFER_SESSION_ID VALUE=01a053d4-8e9f-7212-bbdb-19ba2d64e140
SLOT T3_STATUS VALUE=TRANSFER_CONFIRMED
SLOT T3_CATALOG_RECIPE VALUE=nemo_dino_kamm_mlf_v1
SLOT T3_IDENTITY_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py
SLOT T2_STATUS VALUE=PERPETUAL_FE_OUT_OF_MODEL_CLAIM_BY_USER
SLOT T1_STATUS VALUE=STANDALONE_YEAR_ARM_RELEASED
SLOT T1_INITIALIZATION_STATUS VALUE=INIT_CONFIRMED
SLOT T1_EULER_STATUS VALUE=EULER_AT_BAR
SLOT T1_RULE_1B_STATUS VALUE=NOT_USED
SLOT T1_IC_EULER_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_cold_euler_operator_peel.md
SLOT T1_IC_EULER_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py
SLOT T1_IC_EULER_PROBE_SHA256 VALUE=27bbbdef45e0090aa09d568fdf89133b843f76f9fd61ce2084744eb5b9e064c6
SLOT T1_IC_EULER_ARTIFACT_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_ic_euler_peel_artifact.json
SLOT T1_IC_EULER_ARTIFACT_SHA256 VALUE=595ca607c53a85276f96a031ef3825c52049f078d9d734b02389cca9ad85310d
SLOT T1_IC_EULER_PRODUCER VALUE=019bc38beaa08c7da8750ab2b9f7f9147f2ee3ca
SLOT T1_IC_EULER_FIRST_OVER_BAR VALUE=NONE
SLOT T1_MAX_TRAADV_T_DIFF_KPS VALUE=7.757167435624285e-19
SLOT T1_MAX_TRAADV_S_DIFF_PSUPS VALUE=6.204261567009084e-18
SLOT T1_MAX_LITERAL_TRAZDF_T_DIFF_DEGC VALUE=0
SLOT T1_MAX_LITERAL_TRAZDF_S_DIFF_PSU VALUE=0
SLOT T1_FSLOW_U_MAX_DIFF_MPS2 VALUE=1.1712877473750872e-21
SLOT T1_FSLOW_V_MAX_DIFF_MPS2 VALUE=4.129285617864714e-20
SLOT T1_PUBLIC_ENDPOINT_SCOPE VALUE=POST_HOC_NON_GATING_AND_NOT_BIT_EXACT
SLOT T1_PUBLIC_T_DIFF_DEGC VALUE=6.750155989720952e-14
SLOT T1_PUBLIC_S_DIFF_PSU VALUE=4.192202140984591e-13
SLOT T1_PUBLIC_U_DIFF_MPS VALUE=1.3363623935745694e-8
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=eb4310aaea0a5b7a57f67f325a877db7b4d7bcd59bf2a9cb64c46d03dba32086
SLOT T1_STANDALONE_FINAL_SNAPSHOT_FLAG VALUE=--snap-final
SLOT T1_STANDALONE_YEAR_STEPS VALUE=11520
SLOT T1_STANDALONE_YEAR_MEMBER VALUE=0
SLOT T1_STANDALONE_YEAR_OUTPUT_DIR VALUE=/data/abyssal/dbalwada/LEGO_STANDALONE_Y1_COLD_EULER_CLOSURE_M0
SLOT T1_REDUCER_MESH VALUE=/data/abyssal/dbalwada/RUN_KT2/mesh_mask.nc
SLOT T1_PREDICTION VALUE=INDEPENDENT_YEAR_SST_RMS_0.39C_COLLAPSES_TOWARD_0.01C
SLOT T1_CONFIRM_BAR_DEGC VALUE=0.02
SLOT T1_REFUTE_BAR_DEGC VALUE=0.10
SLOT T1_AGENT_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_MEASURED_NEMO_HOURS_THIS_ROUND VALUE=0
```

There are no empty slots or unimplemented flags.

## Human-owned standalone-year GPU arm

This is the single frozen member-0 arm. It is 11,520 fp64 steps, constructs
the public `nemo_dino_kamm_mlf` card from rest, uses the NEMO mesh only for
diagnostic reduction, and schedules the exact final-day 3-D snapshot.

```bash
set -euo pipefail
T1_WORKTREE=/tmp/dino-transfer-t1-year
T1_GIT_DIR=/tmp/codex-transfer-aux.git
T1_REF=refs/heads/fidelity/dino-transfer-codex
T1_SHA=$(git --git-dir="$T1_GIT_DIR" rev-parse "$T1_REF")
test ! -e "$T1_WORKTREE"
test ! -e /data/abyssal/dbalwada/LEGO_STANDALONE_Y1_COLD_EULER_CLOSURE_M0
git --git-dir="$T1_GIT_DIR" worktree add --detach "$T1_WORKTREE" "$T1_SHA"
test -z "$(git -C "$T1_WORKTREE" status --porcelain --untracked-files=no)"
T1_PYTHON=/home/dbalwada/legoESM/.venv/bin/python
T1_PYTHONPATH="$T1_WORKTREE/src:$T1_WORKTREE/packages/atmosphere:$T1_WORKTREE/packages/core:$T1_WORKTREE/packages/coupler:$T1_WORKTREE/packages/ice:$T1_WORKTREE/packages/land:$T1_WORKTREE/packages/ml:$T1_WORKTREE/packages/ocean:$T1_WORKTREE/packages/tools:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226"
cd "$T1_WORKTREE"
PYTHONPATH="$T1_PYTHONPATH" JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \
  "$T1_PYTHON" scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py \
  --member 0 \
  --output-dir /data/abyssal/dbalwada/LEGO_STANDALONE_Y1_COLD_EULER_CLOSURE_M0 \
  --reducer-mesh /data/abyssal/dbalwada/RUN_KT2/mesh_mask.nc \
  --steps 11520 \
  --snap-final
```

Frozen interpretation: day-360 SST RMS `<=0.02 degC` confirms, `>=0.10 degC`
refutes, and the interval is inconclusive. The prediction remains `0.39 degC`
toward the step-2 bridge class near `0.01 degC`.

## Self-audit

- `standalone_20y.py --help` implements every flag in the arm.
- The output directory was absent and reducer mesh present at handoff time.
- The committed artifact reports `EULER_AT_BAR`, `FIRST_OVER_BAR=None`, and
  all frozen rows passing; public endpoint residuals remain separately stamped.
- No GPU, NEMO execution, `mpirun`, push, or remote action ran in this round.
