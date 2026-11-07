# NEMO-testcases L2 GYRE round 111 receipt: FCT first-writer boundary

Date: 2026-09-18

Incoming tip: `3081ecacc81f6d07b342f91ba8fbbe32efebdab2`

Preregistration commit: `e67c70c689739a124052d6ef029b67984f11a0fa`

Diagnostic commits: `89428545838b0c4662883b96b759689f42efbe44`,
`a5b89279328c7ce07db1452e4e309f757d543513`, and
`e76b721c404b4f4eea5246070a910541b250ba25`

Acquisition-card commits: `706a2115691a03b1c3807fde2f472cb4b85ab423`
and final fuzz-free patch hardening
`42d0bb6cc59d95b81588b5c0a8386b6ec062d4b7`

Round status: **STOPPED_FOR_RECORD — the first directly scored FCT upstream
writer is non-bit under both production JIT and production eager, so the
preregistered two-write candidate is ineligible and no physics lands**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round111/`

## Outcome first

The residual trajectory and its magnitude localization reproduce.  The
certified immutable arm remains
`phase3/round110/candidate/{ladder.json,day_gap.json}`: kt2 U/V are
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S are
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T RMS is
`6.890484901489568e-5 K`.

The new production capture then reaches the first admitted NEMO FCT output,
`adv_up1`.  Temperature is unequal in 18,000/18,000 wet cells with maximum
`6.545655547452618e-11 K s-1`; salinity is unequal in 18,000/18,000 with
maximum `5.680531218351438e-12 psu s-1`.  The later `after_adv` output is also
unequal in every wet cell, at T/S maxima
`5.977741183599542e-11` / `4.957434813195115e-12` per second.  Production
eager and production JIT dictionaries are byte-for-byte equal.  This is not
an XLA-only discrepancy.

Therefore NEMO's two-write accumulation is **not** a candidate.  The
preregistered rule expressly forbids deriving the missing anti-diffusive
operand by subtracting two oracle outputs.  No ladder or month candidate run
was performed, no row moved, and the immutable before arm does not change.

The round ships a new-target, write-only acquisition card at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round111_fct_writers/run.sh`.
It records the compiled upstream input transports/base, first upwind faces,
first divergence and midpoint, averaged faces, final divergence, and direct
`Krhs` output for T/S at kt2 stage 3.  The operator must run it before round
112.

## Prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round111.md`.

| prediction | disposition | evidence |
|---|---|---|
| P1, residual calibration | **CONFIRMED except for one over-precise local/trajectory number, retained as REFUTED** | The self-check passed; kt1 is rounding scale; the material gap is first present after step 2; every immutable ladder/day-30 field reproduced.  The 60-step trajectory scorer reports step-3-entry T RMS `8.802191255844365e-8`, not the local-closure value `8.80218929304543e-8` frozen in P1 (difference `1.9627989356339543e-14 K`).  The mismatch is the already documented local-versus-chained final rounding, and the failed literal prediction is not relabeled. |
| P2, day-23 localization | **CONFIRMED** | Day-23 T RMS is `5.283867389538261e-4 K`; 99.7721% of squared T error is in the west third, 72.4611% is at 100--1000 m, and the peak is `6.016909572113249e-2 K` at `k=8`, 111.224 m. |
| P3, production FCT split | **CONFIRMED** | Both first and final recorded boundaries are non-bit; frozen local content and kt3 maxima reproduce exactly.  `fct_split/jit.json` and `eager.json` agree exactly for all T/S boundary and local dictionaries. |
| P4, eligible source candidate | **REFUTED** | Eligibility required both direct boundaries to be BIT in production JIT and eager.  Each is unequal in every wet cell.  The one-ULP production plant prints `STATUS PLANT-FIRED` and exits 1, but a functioning scorer does not make a non-bit candidate eligible. |
| P5, Decision-43 trajectory | **NOT REACHED by preregistered stopping rule** | No numerical candidate exists, so a scratch ladder/month run would not establish source exactness and was not run. |

## Magnitude ranking

The owners self-check passed all seven controls, including the beyond-step-60
registry plant.  The residual current-tip replay has kt2-entry T RMS
`2.0293251845119845e-15 K` and kt3-entry T RMS
`8.802191255844365e-8 K`, so the remaining material T gap is still born during
step 2.  Day 30 retains the west/upper-ocean structure: 54.0502% of squared T
error is at 100--1000 m, 54.9790% is in the west third, and 83.9340% is in the
wind band.  The day-23 spike is a different spatial regime, overwhelmingly
west and at 100--1000 m; it must not be treated as smooth day-30 growth.

“Carries” below is a measured intervention effect.  It is not an assertion
that maxima or RMS components add linearly.  The FCT row is ranked by its
measured kt3 discriminator; its month effect remains unmeasured because the
source-proof prerequisite failed.

| rank / owner or bounded family | kt3 T carried or remaining | day-30 T RMS carried or remaining | where born | evidence row |
|---|---:|---:|---|---|
| 1. Residual aggregate, causal owner not yet closed | `8.600419718618468e-7 K` remains | `6.890484901489568e-5 K` remains | Step 2; day 30 is 54.0502% at 100--1000 m and 54.9790% west | `baseline_rank/step60/step_gap.json`, `day_gap.json`, `decompose_day30.json` |
| 2. Stage-3 FCT/advection-content family, bounded discriminator rather than assigned owner | Preserved oracle-output intervention removes `8.024323285837909e-7 K` of the local maximum (`8.600420500215478e-7` -> `5.760972143775689e-8`), 93.3015% | **UNMEASURED**; ineligible for a month run | Inside step-2 stage-3 advection; first directly scored output is already non-bit | `fct_split/jit.json`; preserved round-70 endpoint |
| 3. Day-23 transient regime, not a separate additive owner | Not a kt3 quantity | Peaks at `5.283867389538261e-4 K`, then returns to `6.890484901489568e-5 K` at day 30 | 99.7721% west; 72.4611% at 100--1000 m; peak `k=8` | `baseline_rank/decompose_day23.json`, `day_gap.json` |

The direct upstream T discrepancy times the resolved `14400 s` step is
`9.425743988331769e-7 K`, the scale of the kt3 residual.  This is a magnitude
check only: maxima need not occur in the same cell, and ZDF subsequently
multiplies by thickness and solves a column.  It is not used as source proof.

## Compiled-order boundary and first non-bit statement

The compiled stage clears tracer `Krhs` and calls advection before the later
stage-3 sources at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`.
Inside the active two-step FCT routine, NEMO computes the first upwind faces at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-510`, computes
the midpoint tracer at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:532-540`, and
replaces the faces by their first/midpoint averages at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:570-580`.

The first directly scored non-bit compiled statement is the upstream divided
write to `pt_rhs` in
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:602-611` (the write
itself is at `:607`).  “Directly scored” matters: the existing admitted record
contains this output, not the internal faces, midpoint, divergence, or divisor
operands.  This receipt therefore does **not** claim line 607 creates the
error; one of the preceding operands/statements may already differ.  That is
the precise question the new record answers.

After the limiter, NEMO computes another divergence, divides it separately,
and adds it to the existing `Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:318-329`.
The later ZDF program consumes the accumulated `Krhs` in its content build at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.
legoesm's diagnostic exposes an algebraic split of its own low and limited
anti-diffusive flux divergences through the production step.  It does not
substitute any oracle value into the implementation.

## Production JIT/eager table and plant

| tracer / direct boundary | production JIT unequal / max abs | production eager unequal / max abs | verdict |
|---|---:|---:|---|
| T `adv_up1` | 18,000 / `6.545655547452618e-11` | 18,000 / `6.545655547452618e-11` | first directly scored output non-bit |
| S `adv_up1` | 18,000 / `5.680531218351438e-12` | 18,000 / `5.680531218351438e-12` | first directly scored output non-bit |
| T split vs `after_adv` | 18,000 / `5.977741183599542e-11` | 18,000 / `5.977741183599542e-11` | non-bit |
| S split vs `after_adv` | 18,000 / `4.957434813195115e-12` | 18,000 / `4.957434813195115e-12` | non-bit |
| T combined vs `after_adv` | 18,000 / `5.977741183583660e-11` | 18,000 / `5.977741183583660e-11` | non-bit |
| S combined vs `after_adv` | 18,000 / `4.957434814075235e-12` | 18,000 / `4.957434814075235e-12` | non-bit |

The split and combined forms differ from one another at last-bit scale
(T `6.114519087991981e-21`, S `8.746607797135715e-21`), roughly ten orders
smaller than their common discrepancy from NEMO.  Reassociating
only the two writes therefore cannot close the stage.

The production plant changes the first wet upstream operand by one ULP,
changes the captured bit word, prints

```text
STATUS PLANT-FIRED: upstream=6.545655547453e-11 final=5.977741183600e-11
```

and exits 1.  Evidence is `fct_split/plant.{json,log}`.

## Decision-43 trajectory disposition

| required headline | immutable before | round-111 after | disposition |
|---|---:|---:|---|
| kt2 T max | `1.4210854715202004e-14` | no candidate; unchanged | AT-BAR |
| kt2 S max | `2.1316282072803006e-14` | no candidate; unchanged | AT-BAR |
| kt2 U max | `2.7377110452773967e-12` | no candidate; unchanged | first-over-bar DEBT |
| kt2 V max | `3.2849219221489645e-12` | no candidate; unchanged | first-over-bar DEBT |
| kt3 T max | `8.600419718618468e-7` | no candidate; unchanged | DEBT |
| kt3 S max | `6.979441735666114e-8` | no candidate; unchanged | DEBT |
| day-30 T RMS | `6.890484901489568e-5` | no candidate; unchanged | DEBT |

There is no moved-row registry because there is no numerical candidate and no
after trajectory.  This is a stopping-rule outcome, not a Decision-43 pass or
failure.  First-over-bar remains kt2 U/V and no kt1 row changes.

## Passive acquisition card

The new card creates target `GYRE_OMIP_L2_P3_SM_R111FCTW` from the exact R64
configuration.  It changes no namelist and applies only additive writer calls.
Its local proof dry-applies the patch with zero fuzz and compiles both
preprocessed Fortran units with `gfortran -fsyntax-only`;
`acquisition_card/syntax.log` says `STATUS PASS`.  The fixed record is
self-describing (34 ordered fields with
rank, extent, and origin), so its gate parses to exact EOF instead of trusting
hand-written byte arithmetic.

The user-executed script additionally:

- refuses a dirty or incomplete source tree and prints a named `REFUSE` from
  its global error trap before any unexpected nonzero exit;
- checks the resolved ten-step GYRE/FCT card and keeps `namelist_cfg`
  byte-identical;
- compiles under a new target name, rejects vector-math symbols, hashes the
  binary and source card, and stamps the record with the producing commit;
- requires three controls (`stamp`, `truncation`, and `rhs-entry-ulp`) to print
  `STATUS PLANT-FIRED` and exit nonzero; and
- admits the final restart, mesh, and inherited round-64 records bit-for-bit
  against the recorded R64 run before printing READY.

`acquisition_card/focused_tests.log` records `31 passed in 25.44s`, including
all three gate plants and the default-false diagnostic API's JIT bit-identity
test.  The acquisition itself was not run: this agent did not invoke
`makenemo` or `mpirun`.

## Shared-card and configuration disposition

No numerical implementation, default, card, coefficient, stabilizer, carried
state, or restart schema changes.  The diagnostic return is default-false and
its test proves ordinary JIT outputs bit-identical with and without capture.
Consequently GYRE, DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 production bits do
not move in this round; no cross-card trajectory claim is inferred from the
diagnostic.

ORCA2 remains **UNMEASURED-WITH-SPEC**: resolve its compiled tracer integrator
and FCT selection, record both FCT writes plus the first/averaged face,
midpoint, divergence, thickness, mask, and transport operands at kt1--10,
replay the production fp64 JIT closure from those inputs, and score the next
consumed T/S state.

## Independent review

The required command was attempted against clean receipt commit
`cb79add69e8f0f602a7187257fd2a85b368c82ef` with `--sandbox read-only` and an
adversarial prompt covering the production capture, stopping rule, citations,
unchanged trajectory table, and acquisition passivity.  It exited 1 before a
review agent started.  The complete output, quoted verbatim, is:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus **independent review unavailable in-sandbox**; there is no independent
verdict to quote and, in particular, no `DO NOT SHIP` verdict was produced.
The attempt is preserved at `codex_review.log` rather than silently omitted.

## Citation control and tests

The receipt citation gate finds eight citations, maps all eight, audits the
entire registry and reports `status: PASS`.  Its shifted-line plant moves the
direct-writer citation at line 607 to line 609, reports `SYMBOL-NOT-AT-LINE`, and
exits 1.  Evidence is `citation_gate.{json,log}` and
`citation_gate_plant.{json,log}`.

Every suite/command run is reported below.  The combined-tree controller did
not complete: after xdist worker restarts it raised an internal `MemoryError`
while formatting a failure at 75% and exited 3.  Its terminal summary line is
quoted exactly; this receipt does not call that run green.

| command/suite | terminal result |
|---|---|
| New gate plus complete changed advection test file, before the broad run | `31 passed in 25.44s` |
| `tests/ocean/fidelity tests/ocean/unit -n 12` | `164 failed, 5840 passed, 116 skipped, 2 xfailed, 69 warnings, 36 errors in 876.95s (0:14:36)`; then xdist internal error / exit 3 |
| Same focused changed paths after all broad-run workers exited | `31 passed in 26.19s` |
| Complete fidelity subtree, `-n 12` | no terminal summary: reached 98%, then was interrupted with exit 130 after the last active end-to-end stage-sweep prediction plant emitted no completion for more than 15 minutes |

The literal node-ID diff against
`phase3/merge_main_2026-09-17/tests/preexisting_red_ids_on_both_trees.txt`
is in `full_ocean_failure_diff.log`.  Before the controller abort, the log has
164 FAILED-event IDs, 36 ERROR-event IDs, and one separately named crash item:
201 unique IDs.  Sixty-one intersect the frozen 87-ID list, 140 do not, and 26
frozen IDs had not been observed before the abort.  This large non-baseline
set comes from a resource-contaminated, aborted run and cannot serve as a clean
regression diff; it is not suppressed or mislabeled as the known-red set.

No new round-111 acquisition-gate test ID is red.  One test in the changed
advection file appears in the broad-run non-baseline set,
`TestFct2Centred::test_fct2_conserves_on_periodic_channel`; the complete file
passes in both isolated 31-test runs, including after the MemoryError workers
exit.  The evidence therefore supports no reproducible changed-path failure,
but it does **not** support an all-green 8,173-test claim.  Logs are
`acquisition_card/focused_tests.log`, `full_ocean_tests.log`,
`full_ocean_failure_diff.log`, `focused_tests_after_full.log`, and
`full_fidelity_tests.log`.

## ASKED / UNASKED

ASKED and completed: residual owners self-check, step/day/decomposition replay,
day-23 localization, compiled-order production JIT/eager discriminator, a
nonzero-exit production plant, first directly scored non-bit statement,
new-target passive source card, Fortran syntax proof, and acquisition controls.

UNASKED and not done: NEMO source was not modified; `makenemo` and `mpirun`
were not run; no oracle result entered the independent implementation; no
physics/configuration/carried state landed; the year harness, reconciliation
gate, freshwater pair, and #1484 guard were not touched; and no ineligible
candidate was run on the ladder or month.

## OPEN for round 112

1. Run the acquisition card reported in `ACQUISITION_NEEDED`.  Do not start a
   physics candidate until its normal gate and twin admission both pass.
2. From the admitted record, drive the production-JIT FCT closure with NEMO's
   exact recorded inputs and score in compiled order: first u/v/w upwind
   faces, first divergence, midpoint tracer, averaged u/v/w faces, final
   divergence, then the divided `Krhs` write.  The first non-bit row owns the
   next walk; do not start from `after_adv` subtraction.
3. Preserve the current immutable trajectory arm
   `phase3/round110/candidate/{ladder.json,day_gap.json}`.  If and only if a
   source-exact production-JIT candidate closes the direct boundary, apply the
   Decision-43 ladder/month criterion and register every moved row.
4. Keep the day-23 west/100--1000-m transient separate from the day-30
   localization when ranking the next magnitude owner.
5. Keep ORCA2 UNMEASURED until the explicit operand-and-consumer spec above is
   executed.  The remaining GYRE residual is not zero and no complete-identity
   claim is made.
