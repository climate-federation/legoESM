# ORCA2 round 80 — Decision-77 month refusal and stale ladder-control refusal

Base: `16e99b7eb694`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round80.md`.  No package file, selector,
configuration value, carried state, threshold, NEMO source, or sea-ice field
was changed.

## Verdict

**HELD.**  Both requested measurements refused mechanically, for different
reasons.  The independent 240-step candidate reaches a non-positive or
non-finite raw-mesh `e3w_int` before the first step-40 progress marker.  The
given-entry kt=10 walk cannot pass its baseline control because the pinned
round-79b ladder artifact was produced at commit `9b27d1b3ad198a275b639018ea4ce007f786e9eb`,
before the later model commits present at the round-80 base.  Consequently no
terminal month score, basin ranking, current-tip extreme location, or
`avt`/`avm` causal arm is admitted.

## Independent month — before to after

The BEFORE row is round 79a and is retained exactly as preregistered.  AFTER
means the round-80 current-tip run from the card's own initial state.  A
runtime refusal is not converted into a partial terminal metric.

| field | round-79a BEFORE max_abs | round-79a BEFORE rms | round-80 AFTER |
|---|---:|---:|---|
| T | 21.16204086038964 degC | 0.18871605115627163 degC | **UNMEASURED — runtime refusal** |
| S | 4.2942151451623225 g/kg | 0.04399530164405456 g/kg | **UNMEASURED — runtime refusal** |
| u | 1.7552134244778128 m/s | 0.019175453374117628 m/s | **UNMEASURED — runtime refusal** |
| v | 1.2125945863179692 m/s | 0.012956413517199093 m/s | **UNMEASURED — runtime refusal** |
| ssh | 5.578935655726782 m | 0.06850832563417603 m | **UNMEASURED — runtime refusal** |

The unmodified round-71 gate failed with:

```text
equinox._errors._EquinoxRuntimeError: raw-mesh e3w_int must contain only finite values > 0
STATUS FAIL: INTERNAL: CpuCallback error calling callback
```

It emitted no `MONTH_PROGRESS step=40/240`, so the failure is before that
checkpoint; the exact step is **UNMEASURED**.  A diagnostic rerun proved steps
1 and 2 complete, then was stopped before step 3 completed to reserve the
round's CPU budget for the ordered kt=10 walk.  The failed output is preserved
under `orca2_rounds/round80/independent_month.log`; the stopped diagnostic is
preserved separately as `independent_month_stepwise.log`.

Because no terminal candidate exists, the Atlantic/Pacific/Indian/complement
wet-volume ranking and the month T/S argmax regions are **UNMEASURED**.  This
does not reuse or mix the superseded round-79a regional magnitudes.

## Given NEMO's entry — baseline gate

The SHA-pinned reference
`orca2_rounds/round79b/ladder_after.json` has SHA-256
`8347540880af2aac58ff6d3f0af240d676c7f9f36dda55867e4c4d35c9780749` and
stamps its producer as clean commit
`9b27d1b3ad198a275b639018ea4ce007f786e9eb`.  The round-80 base is
`16e99b7eb694`, after model commits `c83c18b25a` and `e48530dc02` and the
round-79b receipt commit.  Thus the artifact is a pre-review-change ladder,
not a certification of the current tip.

Four attempts are retained, with none promoted:

| attempt | exact control | disposition |
|---|---|---|
| no-hook ordinary plus four arms | ordinary kt10 comparison differs from pinned artifact | **REFUSED** |
| observer-aligned five-arm walk | stopped after kt=2 when projected CPU exceeded the round bound | **STOPPED, no score** |
| observer-aligned all-false baseline + `avt` + `avm` | all-false baseline differs from pinned artifact | **REFUSED; arms inadmissible** |
| ordinary observer-aligned locator, no substitution seam | ordinary baseline differs from pinned artifact | **REFUSED** |

The last row is the discriminator: even with no consumer seam, the current-tip
ordinary path is not the artifact's path.  Predictions 5 through 8 therefore
remain mechanically gated: baseline prediction 5 is **REFUTED**; extreme
geography, process ranking, and consumer-arm directions are **UNMEASURED**.
No argmax cell or arm magnitude is quoted from a refused run.

## Oracle source citations

The admitted vertical record remains structurally valid.  Its compiled NEMO
branch copies the closure output, then applies river-mouth, convection,
double-diffusive, and internal-wave contributions in source order
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381`).
Temperature consumes `avt` while salinity consumes `avs`
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:178-215`).
Momentum constructs its implicit matrix from adjacent T-point `avm`
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:191-205`).
Those statements name the intended walk; the failed current-tip baseline means
the round does not assign them a causal magnitude.

## Prediction ledger

| prediction | status | evidence |
|---|---|---|
| month completes and ten-step calibration remains exact | **REFUTED** | raw-mesh `e3w_int` runtime refusal before step 40 |
| all five terminal fields remain non-bit; T leads | **UNMEASURED** | no terminal candidate |
| Decision-77 pair reduces independent-month T/S rms | **UNMEASURED** | no terminal candidate |
| Pacific retains largest weighted T SSE | **UNMEASURED** | no terminal candidate |
| landed kt10 baseline reproduces round 79b | **REFUTED** | ordinary current-tip locator differs from the SHA-pinned artifact |
| T/S extrema are not fold or river-mouth cells | **UNMEASURED** | baseline gate refused before location admission |
| recorded `avt` improves; `avm` is smaller | **UNMEASURED** | exact control refused, so directed arms are inadmissible |
| at least one extreme touches convection | **UNMEASURED** | no admitted location |
| disposition HELD absent a fully gated statement | **CONFIRMED** | no statement landed |

No prediction was rewritten after measurement.

## Gates, review, and tests

The round adds only committed diagnostic gates and tests.  The month progress
option defaults to the original 40-step cadence.  The consumer gate admits the
self-describing round-79b record and refuses a changed baseline, a non-passive
seam, inert `avt`, inert `avm`, and a false claim label.  The baseline-location
gate refuses a changed baseline, false claim label, and invalid location.
Their runtime refusals above are evidence, not test failures.

Validation at the committed scripts/receipt tip:

* focused round-71/79b/80 battery: **31 passed**;
* round receipt citation gate: **PASS**, 3 citations, zero failures and zero
  unmapped; campaign-default citation gate: **PASS**, zero failures and zero
  unmapped; the shifted `zdfphy` citation plant exits 1 with
  `SYMBOL-NOT-AT-LINE`;
* wide `tests/ocean/fidelity -n 12`: 2,177 collected; the xdist wrapper was
  stopped after a repeat of the standing 99% no-summary stall, with **2,154
  passed, 6 known failures, 7 skipped, and 10 tests not reported**.  The six
  reds are the standing round-129 record certification, round-35 stamp scope,
  worktree-stamp emitter, missing case-board row, SI3 scalar-math provenance,
  and round-51 private trace registry tests.  Every round-80 test passed in the
  wide run.

The required separate `codex exec --sandbox read-only` review was attempted on
the committed diff and failed before reading it with `failed to initialize
in-process app-server client: Read-only file system`.  Verdict:
**independent review unavailable in-sandbox**.

No `packages/` file changed.  GYRE therefore cannot move from its certified
day-30 `2.3440e-06` K, day-240 `6.5826e-05` K, and day-360 `5.4085e-05` K
trajectory on this branch; DINO and the tank cards likewise cannot move from
this round's script/document changes.

## OPEN

1. Re-certify the current tip's given-entry kt=1..10 ladder.  Bisect the first
   current-tip row against producer commit `9b27d1b3ad`: the existing
   `ladder_after.json` does not cover the later round-79b review fixes.
2. Instrument the independent run at the raw-mesh thickness assertion and
   name the first failing step/cell and upstream statement.  No stabilizer is
   authorized.
3. Only after item 1 passes, rerun the `avt` and `avm` consumer arms with an
   exact passive control and locate the T/S extrema.
4. The distinct salt/heat double-diffusive coefficient and river-mouth
   diffusivity remain declared unbuilt; neither was approximated here.
