# NEMO testcase L2 GYRE higher-resolution round 25 receipt

Date: 2026-10-20. Branch: `fidelity/gyre-high-resolution`.

## Decision

**ONE-TIME ROUNDING EXPLANATION REFUTED; RECURRING CHECK PREREGISTERED AND
AWAITING GPU ACQUISITION.** The repaired round-24 scorer reads the existing
180-day two-arm run and applies the already-registered rules without changing
the run.  All six day-180 self-divergences are below the scalar/vector NEMO
envelope, so the terminal verdict is `REFUTE_ROUNDING_EXPLANATION`.  The user
then selected one final every-step experiment.  Its intervention, window,
fields, masks, controls, and terminal rules were committed before its runner;
this sandbox has no CUDA device, so that result remains `UNMEASURED`.

No model physics, card configuration, comparison bar, NEMO source, or stored
run was changed in this round.

## Finding -> disposition -> evidence

### 1. Missing report -> serializer fixed without weakening strict JSON

The round-24 score completed every numerical reduction and wrote its NPZ, then
failed because a NumPy integer in the one-ULP control location was not accepted
by Python's JSON encoder.  The writer now converts only NumPy scalar objects to
their Python scalar value.  Unsupported objects still raise, and non-finite
numbers remain forbidden by `allow_nan=False`.

The direct test covers NumPy integer, floating, and boolean scalars and an
unsupported-object refusal.  With the conversion removed, that test fails at
the old serialization boundary.  Re-running only the score against the stored
run exited zero and wrote:

- report:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round25/round24_rounding_rescore/round24_rounding_report.json`,
  SHA-256 `fcd800299025d38116f3cfe20377bdff931691da5752827494c4b0d877945645`;
- arrays: the sibling `round24_rounding_curves.npz`, SHA-256
  `81a881d12ff559946b790e9b486c4cbe51edffb9fadccb6653b4a72605e152e6`;
- source run manifest: SHA-256
  `947fc0c3496f00b1f0e4d7839e2348a2a4d36c6fe03f6f160497e2fe3f901d5f`.

The report is stamped clean at commit `62f527a1f7eb56672d6acd3d015afe2740c80f8e`.
The generalized scorer at `34cd86f346859888bdf894fb27fb21fbb18a18ae`
reproduces the same verdict, day-180 rows, and controls on the same run.

### 2. One-time residue -> registered REFUTE verdict

| daily-mean field | one-time legoESM self divergence | scalar/vector NEMO envelope | self / envelope |
|---|---:|---:|---:|
| SSH | 6.992126104502291e-09 | 2.952524211198950e-04 | 2.368185865498104e-05 |
| SST | 2.269618554547576e-08 | 7.076608127647132e-04 | 3.207212429469640e-05 |
| surface U | 3.544397268662500e-07 | 8.231652544169346e-03 | 4.305814961994566e-05 |
| surface V | 1.643272326177393e-07 | 1.182807647245313e-02 | 1.389298023228438e-05 |
| depth-mean U | 3.389225315016372e-08 | 1.180081422270612e-03 | 2.872026667867640e-05 |
| depth-mean V | 3.828291791965347e-08 | 2.326125995158220e-03 | 1.645780065195889e-05 |

The exact-self normalized RMS is zero in all six fields.  The six active
one-ULP plants are nonzero, spanning `1.3963994263216962e-19` to
`1.2637816333904328e-17`.  The run manifest also records zero unequal words for
the identity transform and 29 moved state words for the one-time intervention.
Thus both controls fire, and every terminal REFUTE inequality is satisfied.

This result refutes only a **one-time** round-22 residue as the explanation of
the days 30--180 gap.  It does not answer whether a similarly sized difference
recurring at every step can accumulate to the measured separation.

### 3. Recurring residue -> fixed before acquisition

The round-25 preregistration is commit
`17957ee9252e9fb6eeb1223c66d86cf2de28ee28`, SHA-256
`8d555a6a9c1bdf2bedd75775fe85b9ce75eaef6424560b46c569fba42d7f1624`.
It holds the common day-180 restart, the exact admitted U/V residue arrays,
the 180-day window, six fields, masks, checkpoint days, controls, and the three
round-24 terminal rules fixed.  The only variable changed from round 24 is
cadence: the same fixed pattern is added to the intervention arm's complete
stage-1 momentum right-hand side at every step from 4,321 through 8,640,
inclusive, for 4,320 applications.  It is not recomputed, rescaled, moved, or
applied to the control arm.

The runner builds separate compiled tail/day steppers for the production and
intervention models.  Its direct routing test proves both intervention
steppers receive the transformed model; replacing either with the production
model makes the test fail.  The run refuses a dirty producer, an implicit GPU,
a moved seed, a moved preregistration, a non-descendant producer, a reused
output path, or a non-GPU backend.  The score refuses an unknown schema, an
incomplete or non-finite run, a moved daily file, a wrong step cadence/count,
or a moved round-11 comparison artifact.

Use the already-admitted round-24 seed; do not rebuild or alter it.  When
physical GPU 0 is idle, the exact acquisition is:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-96610919
scripts/run/run_gyre_hires_round25_recurring_rounding_gpu.sh \
  --physical-gpu 0 \
  --seed-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round24/rounding_closing_180d/seed \
  --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round25/recurring_rounding_180d
```

The expected terminal evidence is `run/manifest.json`,
`score/round25_recurring_rounding_report.json`, and
`score/round25_recurring_rounding_curves.npz` below that output root.

## Inherited pins / OPEN

Inherited unchanged: the certified `GYRE-zco` scalar-libm CPU card at digest
`dca02dffba3eb14f`; its complete ten-step state-byte identity across the ORCA2
merge; the common restart SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`;
the admitted round-22 coefficient-composition residue; and the five-year
`r4_smag` climate PASS against both NEMO builds earned at `219db484b` and
transferred through the round-24 byte-identity audit.

OPEN: one operator-run recurring-residue acquisition and its registered score.
The hunt's closing receipt is deferred until that result, exactly as the user
directed.  No shared-path change is warranted by the one-time result.

Rounds 18--22 remain `HELD`.  The round-25 read-only Claude review emitted only
the known workspace-trust warning and no verdict before its bounded
termination.  GLM remains unavailable.  This round is therefore
**UNREVIEWED-BY-CLAUDE/GLM**; no physics fix depends on it.

## Tests, controls, and choice audit

The complete high-resolution split reports `207 passed, 2 deselected`.  The
time-level/card-reconciliation/phase-3 split reports `62 passed, 2 deselected`.
Fresh compiler processes report nine generic gradient cases passed, the
registered six-step long case skipped, and the SWG card case passed.  The
one-step GYRE gradient case emitted no verdict and was terminated after the
extended bounded wait; it is **INCOMPLETE (compiler stall)**, not passed.
Aggregate completed results are `279 passed, 4 deselected, 1 skipped`; one
gradient case is incomplete.  The certified GYRE digest and ten-step
trajectory tests are in the completed high-resolution and reconciliation
splits and passed.

The citation gate run explicitly from this receipt's compiled-source heading
reports `PASS` with one citation.  Its shifted-line plant reports
`SYMBOL-NOT-AT-LINE` and exits nonzero; the repository citation-gate suite
reports `17 passed` with this receipt present.

The serializer test fails with the conversion removed.  The recurring-routing
test fails when the transformed tail/day steppers are replaced by production
steppers.  Those are the two changed behaviors, and both controls are
non-vacuous.

Choice audit -- **ASKED**: repair the serializer, rescore only the stored run,
retain exact-self and one-ULP controls, preregister and build one recurring
4,320-step experiment with the same metrics and verdicts, require an explicit
physical GPU, request the acquisition, and defer the closing receipt.
**UNASKED**: none.  No physics, deck, threshold, data source, output cadence,
NEMO build/run, GPU launch, issue comment, or external publication was changed.

## Compiled-source citation

The fixed residue is the already-measured difference at NEMO mode-32's
Smagorinsky coefficient boundary: the compiled routine constructs strain,
bounds, and the bilaplacian square-root coefficient, then performs its halo
exchange (`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`).
Round 25 changes neither that oracle program nor legoESM production physics; it
only repeats the previously admitted boundary-residue array as a controlled
sensitivity intervention.
