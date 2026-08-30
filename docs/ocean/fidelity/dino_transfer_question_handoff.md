# DINO transfer question: certified/blocked handoff

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **T3 TRANSFER_CONFIRMED; T2 BLOCKED-NEEDS-FE-STABILIZATION;
T1 BUILD-BLOCKED-FIRST-STEP-RECONCILIATION.** No GPU, NEMO, `mpirun`, or push
was performed. The previous T2 GPU blocks remain withdrawn, and the T1 runner
refuses a claim-length arm until its recorded first-step debt is fixed.

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
SLOT T1_STATUS VALUE=BUILD-BLOCKED-FIRST-STEP-RECONCILIATION
SLOT T1_STANDALONE_RUNNER_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y.py
SLOT T1_STANDALONE_RUNNER_SHA256 VALUE=45c161f11b7055badb786dbd5974f6f8b4ea06ff56c72e12db008bd91c0a414b
SLOT T1_SCORE_RELATIVE_PATH VALUE=scripts/validate/ocean_fidelity/dino_1226/standalone_20y_score.py
SLOT T1_SCORE_SCRIPT_SHA256 VALUE=de121565dc4ecdd9b2543886f2f240ed53f994e1c0f1622e5b62aaffe71d32f9
SLOT T1_CPU_SMOKE_MANIFEST_SHA256 VALUE=3fba61701e206019a95234fbc587ab3d72d126c257394b35d89dfb4c74004bcc
SLOT T1_CPU_SMOKE_INITIAL_RECEIPT_SHA256 VALUE=3ebe7648f14250e4587338a8027fe16278ffa6908e4f979edc5a72c22bfdb3ea
SLOT T1_NEMO_STANDALONE_ROOT VALUE=BLOCKED_NOT_CREATED
SLOT T1_NEMO_ENSEMBLE_MANIFEST_SHA256 VALUE=BLOCKED_NOT_RUN
SLOT T1_LEGO_ENSEMBLE_MANIFEST_SHA256 VALUE=BLOCKED_NOT_RUN
SLOT T1_SCORE_SHA256 VALUE=BLOCKED_NOT_RUN
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

## T1 build receipt and stop

The runner and scorer now exist at the paths in the SLOT block. The runner has
no restart, bridge, `--run-traj`, or `--run-stepdump` selector. It constructs
the exact-card analytic grid, vertical coordinate and from-rest state; builds
literal raw vertical/QCO/EEN operands from those public constructions; applies
only the registered member-temperature perturbation; streams the 75 fp64
snapshots and live reductions; and emits hashes/manifests. The score program
owns the frozen six-member floor, 20,000 independent resamples at seed 1455,
quantization rules, scalar/family/capstone labels and planted controls. It
expects each model's recorded family instruments to emit the normalized
`dino_standalone_20y_statistics_v1` contract; it does not retype reducers.

A CPU/no-JIT one-step run now completes with finite prognostic fields and no
NEMO state input. It is intentionally `claim_admissible=false`. The model's
own warning proves why the 20-year arms are not emitted: on the no-history
Euler bootstrap, legoESM skips `nemo_mlf_baro_corr`; NEMO runs that row on
`l_1st_euler`. The runner's `--steps 230400` path refuses before integration
while this blocker is registered. Fixing it requires surfacing the barotropic
target and raw Kaa SSH from `_step_impl`'s implicit-vmix path into the cold-
start reconciliation site, then a red first-step comparison. A bridge is not
an admissible workaround.

No recorded NEMO ensemble is legal as T1 science data or an ensemble floor.
`RUN_20Y_REBUILD` and `RUN_40Y_REBUILD` remain reducer/control context only;
the bridge-started ensembles remain ineligible because their start and horizon
do not match T1.

## Self-audit

- No GPU, NEMO, `mpirun`, or push command appears in this handoff.
- No T2 or claim-length T1 arm command is emitted.
- All completed T3/T2 paths named above exist in the committed tree.
- The T2 reproducer's parser implements `--recipe`, `--run-traj`,
  `--run-stepdump`, `--max-steps`, and `--output`; no other probe flag is
  claimed here.
- The T1 runner implements `--member`, `--output-dir`, `--reducer-mesh`, and
  `--steps`; the score implements `--lego-root`, `--nemo-root`, `--output`, and
  `--self-test`. Every named flag and committed path exists.
- The final one-step smoke manifest stamps producer `c981468d05`, fp64,
  standalone start, empty bridge/restart paths, the exact first-step blocker,
  and `claim_admissible=false`.

Post the T2 block and any future T1/T2 result to GitHub issue #1455 when the
human-owned evidence trail is updated. No remote action was taken in this
round.
