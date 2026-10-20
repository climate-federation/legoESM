# NEMO testcase L2 GYRE higher-resolution round 26 closing receipt

Date: 2026-10-20. Branch: `fidelity/gyre-high-resolution`.

## Decision

**HUNT CLOSED -- RECURRING ROUNDING CHECK PARTIAL; NO PHYSICS CHANGE.** The
operator-acquired 4,320-application experiment reproduces between `0.61x` and
`2.51x` of the measured legoESM--NEMO day-180 separation across the six
registered fields. It is larger than the independent scalar/vector NEMO
rounding envelope in every field, by `2.05x` to `15.32x`, but it does not
reproduce the full legoESM--NEMO separation in every field. The terminal
verdict fixed before acquisition is therefore `PARTIAL_OR_INCONCLUSIVE`.

This result is **CONFIRMED as a controlled sensitivity measurement**: a fixed
residue pattern of the measured round-22 size, injected at every step, can
produce separation of the observed order. It is only **PLAUSIBLE as an
explanation of the production legoESM--NEMO gap**: the intervention repeats a
frozen pattern rather than measuring the live compiler-composition difference
at every evolving state, and the preregistered full-explanation rule does not
pass. No production model, card, comparison bar, NEMO source, or stored run is
changed.

## Finding -> disposition -> evidence

### 1. Recurring experiment -> registered partial verdict

The admitted run is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round25/recurring_rounding_180d/`.
Its complete run manifest has SHA-256
`1c35b70ffca0c9a04837450a293d0c9dd41fd4a88d0d39404adaab5a4084eaed`.
It records clean producer `6ece9285a1a0ec1c790752cd5d067f83888dc10a`,
physical GPU 0, the native-fp64 non-certified policy, the registered common
restart, 180 complete daily files, and exactly 4,320 seed applications at steps
4,321--8,640.

Re-running only the committed scorer against those stored daily files exits
zero with `PARTIAL_OR_INCONCLUSIVE`. It reproduces the acquired curves byte for
byte: SHA-256
`51a625f4baf70a9c1a6018ebe5159d0d9e8bacf643eca482355c0e9af9e15219`.
The acquired JSON report has SHA-256
`65792a420368ae2cea42cdcb33e0c91f9483718482b9e11159b0bf57927df1bf`.

| daily-mean field | recurring/control NRMS | scalar/vector NEMO NRMS | recurring / twin | legoESM/NEMO NRMS | recurring / legoESM--NEMO |
|---|---:|---:|---:|---:|---:|
| SSH | `0.001627858` | `0.000295252` | `5.513` | `0.002422111` | `0.672` |
| SST | `0.010838505` | `0.000707661` | `15.316` | `0.004313538` | `2.513` |
| surface U | `0.020658086` | `0.008231653` | `2.510` | `0.027607691` | `0.748` |
| surface V | `0.024225955` | `0.011828076` | `2.048` | `0.029898957` | `0.810` |
| depth-mean U | `0.006614641` | `0.001180081` | `5.605` | `0.008454192` | `0.782` |
| depth-mean V | `0.008017108` | `0.002326126` | `3.447` | `0.013121674` | `0.611` |

The exact-self normalized RMS is zero in all six fields. The active one-ULP
plants are nonzero in all six, spanning `1.3963994263216962e-19` to
`1.2637816333904328e-17`. The first-step identity transform compares 2,818,060
state words and moves zero. Thus the scorer resolves its registered floor and
the intervention seam is passive when given the identity transform.

The result satisfies neither terminal extreme. Five fields remain below the
registered legoESM--NEMO row, so `CONFIRM_FULL` is false. All six fields exceed
the scalar/vector NEMO row, so `REFUTE_ROUNDING_EXPLANATION` is false. The
pre-acquisition catch-all `PARTIAL_OR_INCONCLUSIVE` applies without a post-hoc
threshold.

### 2. One-time seed survival -> sparse, not zero

The round-24 one-time run manifest, SHA-256
`947fc0c3496f00b1f0e4d7839e2348a2a4d36c6fe03f6f160497e2fe3f901d5f`,
already contains the requested post-step word comparison. Of 2,818,060 state
words, 29 move: 28 in the first pytree leaf and one in the second. The state
definition orders `u` and `v` first and each `Field` contributes its data as
one leaf; direct instantiation confirms those leaves are native prognostic U
and V. The changed fraction is `1.029076740736535e-5`, or `0.001029%`.

Therefore the literal explanation that a one-time `~3.4e-21 m s-2` right-hand-
side residue changes no prognostic word is **REFUTED**. The measured statement
is narrower: `99.998971%` of state words remain byte-identical after the seeded
step, and only 29 velocity words preserve a change. That extreme sparsity is a
**PLAUSIBLE**, not uniquely proved, explanation for why the one-time arm remains
only `1.39e-5`--`4.31e-5` of the NEMO twin envelope at day 180. The recurring
experiment shows that applying the same small pattern 4,320 times changes the
outcome qualitatively.

### 3. Operator walk -> exact given measured inputs; no transcription owner

Rounds 18--22 close the examined production operators at their registered
boundaries:

| boundary | exact result | disposition |
|---|---|---|
| completed stage-3 vorticity/Coriolis V stencil | scalar-NEMO q plus `F_u` gives 0/1,896 unequal accumulator words; `F_v` negative control moves none | operator exonerated given its two inputs |
| live q and `F_u` replay | legoESM source builders reproduce their exposed arrays exactly; production ENE replay reproduces the captured term exactly | exposure/replay calibrated; no self-comparison |
| six q/`F_u` source factors | NEMO Coriolis and `e2u` are exact; no single velocity, metric, Coriolis, or thickness substitution makes the NEMO accumulator exact | coupled formation boundary; no single transcription owner |
| stage-1 Smagorinsky coefficients | `ahmt` 0/300,120 unequal and `ahmf` 0/306,270 unequal for legoESM/scalar/vector NEMO | every examined coefficient factor exonerated |
| coefficient/operator composition | materializing either exact coefficient makes post-LDF U and V exact; live composition leaves maxima `1.85e-21` and `3.39e-21` | measured JAX composition debt, not a physics defect |

These are exact-given-input claims, not trajectory identity claims. Round 18's
Claude review accepted its exoneration with the recorded caveat that the NEMO
before-accumulator was held fixed and upstream stage state remained in scope.
Rounds 18--22 and this closing synthesis remain **UNREVIEWED-BY-GLM** because
that service is unavailable. No physics change depends on an unavailable
review.

## Certification carried forward

The certified shipped `GYRE-zco` card remains 31 levels, WS-RK3,
split-explicit free surface, TKE/EVD, Redi tracers, and scalar-libm CPU policy,
with admitted digest `dca02dffba3eb14f`. That digest replaced
`56c5171d43504b54` only after the resolved-config audit found 33 additions,
zero removals, and zero changes to old values, and the merged and pre-merge
cards produced identical bytes for all state leaves after each of ten steps.

The registered five-year `r4_smag` climate run at commit `219db484b` remains
PASS against both NEMO builds. The vector-math NEMO build is the certified
climate reference; the scalar-math build is explicitly labelled a
non-certified alternative reference. The year-5 scalar comparison passes all
four registered rows: KE differs by `2.15%`, eddy fraction by `15.12%`, the
absolute-U jet row is exact, and the SSH zero crossing differs by `0.755` row.
The climate PASS transfers across the later digest admission only because the
ten-step whole-state comparison is byte-identical; it was not re-earned by a
new five-year run and is not a trajectory-match claim.

## Closing boundary / OPEN

The fidelity hunt stops here by user decision. No shared-path change is
warranted: the five-year statistical target passes, every examined operator is
exact given its recorded inputs, the remaining stage-1 residue is a measured
composition effect, the one-time sensitivity refutes sufficiency, and the
recurring frozen-pattern sensitivity is partial rather than a full causal
proof.

OPEN, without a new work request:

1. The evolving, live XLA/JAX composition residue at every production state
   has not been recorded; the closing intervention repeats one frozen day-180
   pattern.
2. The exact fraction of the real legoESM--NEMO separation caused by recurring
   composition rounding is not identified. The controlled experiment supports
   a substantial contribution as **PLAUSIBLE**, not **CONFIRMED**.
3. GLM review remains unavailable. The round-18 Claude caveats and round-19
   review gap remain attached to those localizations.
4. The closing finding has not been posted to issue #1455 from this sandbox.

No further acquisition, GPU run, NEMO build, or user decision is requested.

## Tests, controls, and choice audit

The stored recurring score was independently replayed under CPU/x64 from all
180 daily files; its curve artifact is byte-identical to the operator's score,
all six exact-self controls are zero, and all six active one-ULP plants fire.
The existing direct instrument tests cover the exhaustive terminal rules,
pinned preregistrations/comparison artifact, moved-seed refusal, explicit GPU
selection, recurring-arm routing, and strict NumPy-scalar JSON serialization.

The complete high-resolution split reports `207 passed, 2 deselected`. The
time-level/card-reconciliation/phase-3 split reports `62 passed, 2
deselected`. Fresh compiler processes report nine generic gradient cases
passed, the registered six-step GYRE case skipped, and the SWG card case
passed. The isolated one-step GYRE gradient case emitted no verdict during the
bounded wait and was terminated; it is **INCOMPLETE (compiler stall)**, not a
failed gradient measurement. Aggregate completed results are
`FULL_LANE_BATTERY_SPLIT_PASS: 279 passed, 4 deselected, 1 skipped; one GYRE gradient case incomplete`.

The citation gate run explicitly from this receipt's compiled-source heading
reports `PASS` with four citations, zero failures, zero unmapped entries, an
empty map audit, and all nine controls firing. Shifting the mode-32 coefficient
citation reports `SYMBOL-NOT-AT-LINE` and exits nonzero. `git diff --check`
passes.

Choice audit -- **ASKED**: verify the acquired report and controls, measure the
one-time post-step word count, record the one-time and recurring verdicts at
their evidence level, carry forward only the exact-audited certification,
write the closing receipt, make no new run or physics change, and stop.
**UNASKED**: none. No configuration, threshold, metric, mask, cadence, data
source, output, model state, NEMO source, GPU job, or external publication was
changed.

## Compiled-source citations

The selected total-vorticity routine forms Coriolis plus relative curl and the
live-thickness q at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:552-572`, then forms and
consumes `F_u` in the V statement at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:576-591`.
The mode-32 branch forms, bounds, converts, masks, and exchanges the
Smagorinsky coefficients at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`.
The selected level-wise rotated operator consumes those square-root
coefficients in both bilaplacian passes at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/dynldf_lev.f90:249-330`.
