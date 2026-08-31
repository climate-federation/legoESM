# DINO transfer question: standalone-year prediction confirmed

Date: 2026-08-31. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 PERPETUAL FE OUT OF MODEL CLAIM BY USER;
T1 COLD-START TRANSFER CONFIRMED; TRANSFER LANE COMPLETE PENDING REVIEW.**

## Receipt slots

```text
SLOT TRANSFER_PRODUCER_REF VALUE=refs/heads/fidelity/dino-transfer-codex
SLOT TRANSFER_SESSION_ID VALUE=01a053d4-8e9f-7212-bbdb-19ba2d64e140
SLOT T3_STATUS VALUE=TRANSFER_CONFIRMED
SLOT T3_CATALOG_RECIPE VALUE=nemo_dino_kamm_mlf_v1
SLOT T3_IDENTITY_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/recipe_transfer_identity.py
SLOT T2_STATUS VALUE=PERPETUAL_FE_OUT_OF_MODEL_CLAIM_BY_USER
SLOT TRANSFER_LANE_PR_READINESS VALUE=COMPLETE_PENDING_REVIEW
SLOT T1_STATUS VALUE=COLD_START_TRANSFER_CONFIRMED
SLOT T1_INITIALIZATION_STATUS VALUE=INIT_CONFIRMED
SLOT T1_EULER_STATUS VALUE=EULER_AT_BAR
SLOT T1_RULE_1B_STATUS VALUE=NOT_USED
SLOT T1_IC_EULER_PREREG_RELATIVE_PATH VALUE=docs/ocean/fidelity/PREREG_dino_cold_euler_operator_peel.md
SLOT T1_IC_EULER_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/ic_euler_peel.py
SLOT T1_IC_EULER_PROBE_SHA256 VALUE=cc46a5715aa2657e024a7675717e95b0312e68a75c563164094a90860dbfd1f1
SLOT T1_IC_EULER_ARTIFACT_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_ic_euler_peel_artifact.json
SLOT T1_IC_EULER_ARTIFACT_SHA256 VALUE=98738a8ea067e2c97f3adc8fa83462e4e8d621a297e422ca9558164c6b241980
SLOT T1_IC_EULER_PRODUCER VALUE=c5b1c5a9e363a951aa422f811d7f6dfe7bc88c94
SLOT T1_IC_EULER_FIRST_OVER_BAR VALUE=NONE
SLOT T1_MAX_TRAADV_T_DIFF_KPS VALUE=7.757167435624285e-19
SLOT T1_MAX_TRAADV_S_DIFF_PSUPS VALUE=6.204261567009084e-18
SLOT T1_FCT_COUPLED_PATH_DIVERGENCE_AT_BAR VALUE=TRUE
SLOT T1_FCT_INDIVIDUAL_FACE_BIT_IDENTITY_CLAIMED VALUE=FALSE
SLOT T1_MAX_LITERAL_TRAZDF_T_DIFF_DEGC VALUE=0
SLOT T1_MAX_LITERAL_TRAZDF_S_DIFF_PSU VALUE=0
SLOT T1_FSLOW_U_MAX_DIFF_MPS2 VALUE=1.1712877473750872e-21
SLOT T1_FSLOW_V_MAX_DIFF_MPS2 VALUE=4.129285617864714e-20
SLOT T1_PUBLIC_ENDPOINT_SCOPE VALUE=POST_HOC_NON_GATING_AND_NOT_BIT_EXACT
SLOT T1_PUBLIC_T_DIFF_DEGC VALUE=6.750155989720952e-14
SLOT T1_PUBLIC_S_DIFF_PSU VALUE=4.192202140984591e-13
SLOT T1_PUBLIC_U_DIFF_MPS VALUE=1.3363623935745694e-8
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=7fe417178bb1827f467d3bd513f6e96c202d438054b3cb4b0be4e748b9bb8cc6
SLOT T1_YEAR_SCORE_PROBE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_year_transfer_receipt.py
SLOT T1_YEAR_SCORE_PROBE_SHA256 VALUE=f00183e0f3dee19114d6a3c7f36ba2a2db0085f8aa3d120f0da453338cf0c7f4
SLOT T1_YEAR_SCORE_ARTIFACT_RELATIVE_PATH VALUE=docs/ocean/fidelity/dino_standalone_year_transfer_artifact.json
SLOT T1_YEAR_SCORE_ARTIFACT_SHA256 VALUE=50b6952e46deb15296b8aeb8149caad27122b81a4f39e59e01b2cdbca3d82ca0
SLOT T1_YEAR_SCORE_PRODUCER VALUE=079432b47fc94336dd69d81aa23cc1deed1deb9c
SLOT T1_STANDALONE_FINAL_SNAPSHOT_FLAG VALUE=--snap-final
SLOT T1_STANDALONE_YEAR_STEPS VALUE=11520
SLOT T1_STANDALONE_YEAR_MEMBER VALUE=0
SLOT T1_STANDALONE_YEAR_OUTPUT_DIR VALUE=/data/abyssal/dbalwada/dino-standalone-y1-fixed
SLOT T1_REDUCER_MESH VALUE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc
SLOT T1_PREDICTION VALUE=INDEPENDENT_YEAR_SST_RMS_0.39C_COLLAPSES_TOWARD_0.01C
SLOT T1_CONFIRM_BAR_DEGC VALUE=0.02
SLOT T1_REFUTE_BAR_DEGC VALUE=0.10
SLOT T1_PREDICTION_OUTCOME VALUE=CONFIRM
SLOT T1_SST_RMS_DEGC VALUE=0.007586936458923034
SLOT T1_SST_MAX_ABS_DEGC VALUE=0.2637286932718048
SLOT T1_SSH_RMS_MM VALUE=0.3070795636118303
SLOT T1_T3_RMS_DEGC VALUE=0.0056596643780580415
SLOT T1_T3_BIAS_DEGC VALUE=-5.292327156826694e-5
SLOT T1_S3_RMS_PSU VALUE=5.671816962734792e-4
SLOT T1_FINAL_SNAPSHOT_ALIGNMENT VALUE=FULL_NEMO_CONSTRUCTION_FRAME_DROP_TERMINAL_JPK
SLOT T1_AGENT_GPU_HOURS_THIS_ROUND VALUE=0
SLOT T1_MEASURED_NEMO_HOURS_THIS_ROUND VALUE=0
```

There are no empty slots or unimplemented flags.

## Completed human-owned standalone-year GPU arm

The human completed the single frozen member-0 arm at producer
`bd84872874a342d10d4c6459fb4cd2fc6a33907a`. It ran 11,520 fp64 steps,
constructed the public `nemo_dino_kamm_mlf` card from rest, used the canonical
NEMO mesh only for diagnostic reduction, and wrote the exact final-day 3-D
snapshot. This command is a provenance receipt, not an instruction to overwrite
the existing output directory.

```bash
set -euo pipefail
T1_WORKTREE=/tmp/dino-transfer-t1-year
T1_PYTHON=/home/dbalwada/legoESM/.venv/bin/python
T1_PYTHONPATH="$T1_WORKTREE/src:$T1_WORKTREE/packages/atmosphere:$T1_WORKTREE/packages/core:$T1_WORKTREE/packages/coupler:$T1_WORKTREE/packages/ice:$T1_WORKTREE/packages/land:$T1_WORKTREE/packages/ml:$T1_WORKTREE/packages/ocean:$T1_WORKTREE/packages/tools:$T1_WORKTREE/scripts/validate/ocean_fidelity/dino_1226"
cd "$T1_WORKTREE"
PYTHONPATH="$T1_PYTHONPATH" JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \
  "$T1_PYTHON" scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py \
  --member 0 \
  --output-dir /data/abyssal/dbalwada/dino-standalone-y1-fixed \
  --reducer-mesh /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc \
  --steps 11520 \
  --snap-final
```

Frozen interpretation: day-360 SST RMS `<=0.02 degC` confirms, `>=0.10 degC`
refutes, and the interval is inconclusive. The measured `0.007586936458923034
degC` confirms the prediction that `0.39 degC` would collapse toward the
step-2 bridge class near `0.01 degC`.

## Self-audit

- `standalone_20y.py --help` implements every flag in the arm.
- The completed manifest records member 0, 11,520 fp64 steps, standalone start,
  `--snap-final`, and a present day-360 fp64 snapshot. The year scorer verifies
  its captured hash before scoring.
- The prior `RUN_KT2/mesh_mask.nc` handoff path is corrected. Although its
  bytes match, the reducer imports and pathname-checks the canonical oracle
  `RUN_TRAJ/mesh_mask.nc`; the runner now rejects aliases at day 0.
- Frame-adaptive scoring selected `full_nemo_construction_frame` for the
  `(199,52,35)` final snapshot and dropped only NEMO's dry terminal `jpk`.
  The unsupported-frame plant fires.
- The committed artifact reports `EULER_AT_BAR`, `FIRST_OVER_BAR=None`, and
  all frozen rows passing. Both horizontal/vertical FCT divergence paths pass
  independently. Individual stored-face bit identity is explicitly not claimed:
  NEMO dumps metric-complete products whereas legoESM applies the metrics in its
  divergence; both the normalized and rejected native-frame comparisons remain
  stamped. Public endpoint residuals remain separately stamped.
- T3 is certified, T2 is scoped by user decision, and T1's cold-start transfer
  discriminator confirms. PR disposition: `COMPLETE_PENDING_REVIEW`.
- The agent ran no GPU, NEMO execution, `mpirun`, push, or remote action in the
  binding round.
