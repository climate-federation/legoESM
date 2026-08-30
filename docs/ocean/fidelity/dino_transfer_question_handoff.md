# DINO transfer question: certified/blocked handoff

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 BLOCKED-NEEDS-FE-STABILIZATION;
T1 BUILD_IN_T1_ROUND.** No GPU, NEMO, `mpirun`, or push was performed in the
diagnosis/package round. The T2 GPU blocks from the previous handoff are
withdrawn and must not be run from this producer.

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
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=BUILD_IN_T1_ROUND
SLOT T1_SCORE_RELATIVE_PATH VALUE=BUILD_IN_T1_ROUND
SLOT T1_NEMO_STANDALONE_ROOT VALUE=PROVIDE_IN_T1_ROUND
SLOT T1_NEMO_ENSEMBLE_MANIFEST_SHA256 VALUE=MEASURED_IN_T1_ROUND
SLOT T1_LEGO_ENSEMBLE_MANIFEST_SHA256 VALUE=MEASURED_IN_T1_ROUND
SLOT T1_SCORE_SHA256 VALUE=MEASURED_IN_T1_ROUND
SLOT T1_MEASURED_GPU_HOURS VALUE=MEASURED_IN_T1_ROUND
SLOT T1_MEASURED_NEMO_HOURS VALUE=MEASURED_IN_T1_ROUND
```

There are no empty slots. Every T3/T2 path above is committed, every completed
artifact or log has a concrete hash, every unavailable T2 artifact is marked
blocked, and every not-yet-built T1 item is explicitly assigned to the T1
build round.

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

Consequences:

- do not run the withdrawn 360-day FE climate pair;
- do not run the five-day FE wall pair, because its trajectory is already
  unstable;
- do not invoke the offline scorer without admitted artifacts;
- open a distinct, preregistered FE-stabilization round before reissuing T2.

T2 therefore has no statistical `CONFIRMED`/`REFUTED` verdict. Its result is a
model-scope block: the MLF bridge certification has not transferred to the FE
sibling card.

## T1 checkpoint — proceed with the build round

T1 is independent of the T2 FE block and may proceed. There is intentionally no
executable T1 shell block yet: the committed standalone legoESM runner, scorer,
fresh from-rest NEMO ensemble root, and their parsers do not exist. Adding a
command now would recreate the missing-selector failure class.

The T1 build round must provide:

- a standalone `nemo_dino_kamm_mlf` legoESM runner that accepts no bridge or
  NEMO-restart argument and saves the registered 75 fp64 windows;
- a scorer importing the held 20-year reducers, ensemble floors, frozen bars,
  boundary tests, and planted violation;
- six fresh from-rest NEMO members and six fresh from-rest legoESM members
  (control plus relative-temperature kick seeds 1--5);
- a NEMO t=0 capture/perturbation receipt proving all non-temperature restart
  values are byte-identical within the NEMO ensemble;
- manifests hashing all states, grids, configs, source/executable receipts,
  logs, perturbations, and successful completion receipts.

No recorded NEMO ensemble is legal as T1 science data or an ensemble floor.
`RUN_20Y_REBUILD` and `RUN_40Y_REBUILD` remain reducer/control context only;
the bridge-started ensembles remain ineligible because their start and horizon
do not match T1.

## Self-audit

- No GPU, NEMO, `mpirun`, or push command appears in this handoff.
- No T2 arm or scorer command remains.
- All completed T3/T2 paths named above exist in the committed tree.
- The T2 reproducer's parser implements `--recipe`, `--run-traj`,
  `--run-stepdump`, `--max-steps`, and `--output`; no other probe flag is
  claimed here.
- Every T1 non-path is explicitly `BUILD_IN_T1_ROUND`,
  `PROVIDE_IN_T1_ROUND`, or `MEASURED_IN_T1_ROUND`.

Post the T2 block and any future T1/T2 result to GitHub issue #1455 when the
human-owned evidence trail is updated. No remote action was taken in this
round.
