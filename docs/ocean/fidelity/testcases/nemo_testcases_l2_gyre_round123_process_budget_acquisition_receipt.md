# NEMO testcase L2 GYRE phase 3 — round 123 process-budget acquisition receipt

Date: 2026-09-19

Incoming tip: `d6765dd6fd740b6e105a990c491cee44e450b093`

Status: **STOPPED_FOR_RECORD — no admitted record contains the day-180-to-240
tracer-process boundaries, so no process owner is named; a passive,
fail-closed NEMO acquisition and its reader are committed and await the
operator.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123/`

## Outcome first

Round 122's ranked target remains the southern/western upper-ocean temperature
mode that appears between days 180 and 240.  Round 123 does **not** infer that
surface forcing, advection, lateral diffusion, vertical diffusion, W, ZAD or
any other operator owns it.  The existing one-year records contain daily
prognostic state but not the accumulated tracer RHS at individual process
boundaries.  The committed process-record gate found zero Round-123 frames,
printed

```text
GATE FAILED: process-record set is not exactly steps 1081..1440
exit_code=1
```

and stopped before a magnitude table could be fabricated.  The process owner,
stage owner and first non-bit statement are therefore **UNMEASURED**.

The round instead finishes the prerequisite acquisition unit:

- an additive writer for a new NEMO target,
  `GYRE_OMIP_L2_P3_SM_R123PROC`;
- an exact 360-frame parser in the existing year-owner instrument;
- passive admission against the source run's day-180 and day-240 restarts;
- exact layout, producer-stamp, truncation, raw-ULP, effect-propagation and
  step-chain controls; and
- a user-run script at
  `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round123_process_budget/run.sh`.

The agent did not run `makenemo`, `mpirun` or NEMO.  Nothing lands in the
shared ocean implementation, no card/default/carried state changes, and the
immutable before arm remains the Round-110 landing's admitted year trajectory.

## Frozen preregistration and prediction ledger

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round123.md`, committed
before instrument work as full commit
`b54e6cdd1db72a385d947d29fa1a6a0cc0d667ac`.

- **P0 CONFIRMED.** The campaign-root census contains no
  `oracle_process_budget_kt*.bin` frames.  The new gate independently refuses
  because the file set is not exactly steps 1081--1440.  No process is ranked.
- **P1 PARTLY CONFIRMED, MEASUREMENT PENDING.** The source binary and the
  step-1080/step-1440 restart hashes reproduce the frozen values.  Whether the
  new writer is passive is deliberately unclaimed until the operator run
  reproduces both restart files byte for byte.
- **P2 STATIC CONTRACT CONFIRMED, RECORD PENDING.** The additive patch applies
  without fuzz, its preprocessed translation unit passes
  `gfortran -fsyntax-only`, and a synthetic record parses at the frozen byte
  count.  No acquired header or payload is claimed.
- **P2 RAW-ULP PROPAGATION PREDICTION REFUTED.** The preregistration required a
  one-ULP post-SBC RHS perturbation to move both its decoded temperature
  contribution and interval closure.  The committed synthetic counterexample
  moves the raw RHS increment by one ULP but moves **zero** decoded temperature
  rows: the increment is rounded away when accumulated with the order-one
  tracer content.  The gate keeps that honest raw-parser plant and adds a
  separate, self-calibrating effect-scale plant which moves the decoded
  `surface_boundary` row.  It never prints success for the raw ULP as if it
  propagated.
- **P3 UNMEASURED.** The signed whole-domain day-240 projection and ranked
  process table require both independent trajectories.  Neither a synthetic
  check nor this NEMO-only acquisition card is promoted into a causal owner.
- **P4 CONFIRMED.** Only a diagnostic reader, source-card patch, controls,
  tests, citation registry, preregistration and receipt changed.  Production
  physics and all protected harnesses are untouched.

## Compiled program and the recorded boundaries

This acquisition instruments the exact branch that produced the admitted
NEMO year.  The compiled stage program clears tracer `Krhs`, calls advection,
then calls the RK3 surface boundary condition at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:812,843,849`.
At stage 3 it next calls penetrative shortwave, lateral diffusion and implicit
vertical diffusion, in that order, at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:915,930,944`.

The intervening optional call sites are mechanically enumerated at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:902,933,934,935,937,939,953`.
The admitted source run's resolved `ocean.output` proves that boundary damping,
bottom heat, bottom boundary layer, internal damping, mass-flux convection,
OSMOSIS and non-penetrative convection are all disabled; the new run refuses
if any of those resolved rows changes.  Thus the five active operator rows are
advection, surface boundary, shortwave, lateral diffusion and vertical
diffusion, with free-surface geometry retained as a separate budget row.

The implicit tracer routine constructs its RHS from before tracer content and
the accumulated `Krhs` at the middle free-surface ratio at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:546-560`, then performs
the forward/back vertical solve at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:563-578`.
That compiled statement, rather than a guessed tendency partition, is why the
record carries `Tbb`, all three `r3t` slots, every accumulated RHS boundary and
the post-`tra_zdf` `Taa`.

The completed stage-3 slot is swapped into the next step's `Nbb` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:215-222`.
Consequently the gate also requires every wet interior `Taa(kstp)` to be
bit-identical to `Tbb(kstp+1)`.  The separate one-ULP trajectory plant must
break exactly one cell in that production chain and exit nonzero.

## Frozen frame and admission contract

The source is the admitted seed-0 run
`phase3/year_fromrest/nemo_seed0` and its compiled
`GYRE_OMIP_L2_P3_SM_YRPERT` card.  Read-only re-hashing reproduced:

| frozen input | SHA-256 |
|---|---|
| source NEMO binary | `578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a` |
| step-1080 restart | `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976` |
| step-1440 restart | `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a` |

The target run copies that run's prepared seed-0 inputs file by file.  The only
runtime-namelist change is `nn_itend: 2160 -> 1440`; `nn_stock=180`,
`nn_write=2160`, `nn_pert_seed=0`, `rn_Dt=14400 s` and every physical choice
remain fixed.  This is a shorter observation of the same trajectory, not a
new physics choice.

Each step from 1081 through 1440 writes one stream frame in this exact order:

| order | field | extent/type |
|---:|---|---|
| 1 | magic `NEMO_L2_R123PROC` | 16 bytes |
| 2 | version, step, stage, four slots, `jpi/jpj/jpk`, storage bits | 11 native 32-bit integers |
| 3 | stage timestep `rDt` | one binary64 |
| 4 | `Tbb` | `36 x 26 x 31` binary64 |
| 5 | `r3t(Kbb)`, `r3t(Kmm)`, `r3t(Kaa)` | three `36 x 26` binary64 arrays |
| 6 | `Krhs` after advection | `36 x 26 x 31` binary64 |
| 7 | `Krhs` after surface boundary | same |
| 8 | `Krhs` after shortwave | same |
| 9 | `Krhs` after lateral diffusion | same |
| 10 | `Taa` after vertical diffusion | same |

The byte identity is fixed before acquisition:

```text
16 + 11*4 + 8 + (6*36*26*31 + 3*36*26)*8 = 1,415,300 bytes/frame
360 frames = 509,508,000 bytes
```

The reader rejects a wrong magic, version, step/stage, slot range, dimension,
storage width, timestep, byte count, EOF, record set, sequence, checksum,
producer commit, binary checksum, resolved card, nonfinite payload, inactive
active-process row, broken `Taa -> Tbb` chain or either restart hash.  The run
script prints a named `REFUSE` before every explicit nonzero exit and also has
an `ERR` trap, so an operator never receives another silent size refusal.

The writer patch adds source lines only.  The one namelist patch replaces only
the terminal step.  The local compiled-syntax artifact ends literally:

```text
SYNTAX_PROOF_PASS
expected_bytes=1415300
expected_records=360
expected_total_bytes=509508000
```

## Budget metric frozen for the acquired records

For each independently evolving model and each wet interior cell, the reader
forms cumulative explicit-temperature boundaries from the compiled qco
content statement:

```text
B0   = qbb*Tbb/qaa
Badv = (qbb*Tbb + rDt*qmm*Radv)/qaa
Bsbc = (qbb*Tbb + rDt*qmm*Rsbc)/qaa
Bqsr = (qbb*Tbb + rDt*qmm*Rqsr)/qaa
Bldf = (qbb*Tbb + rDt*qmm*Rldf)/qaa
```

Adjacent differences assign geometry, advection, surface boundary, shortwave
and lateral diffusion; `Taa-Bldf` assigns vertical diffusion.  An explicit
rounding-closure row remains visible.  Summed over steps 1081--1440 on NEMO's
and legoESM's own trajectories, each process component is projected onto the
measured day-240 temperature-error vector.  Signed carries, the incoming
day-180 gap and closure must reconstruct the day-240 RMS; process rows are
ranked by absolute signed carry.  The round-122 west/surface/wind intersection
is diagnostic only.  This metric is frozen here but is **not evaluated** until
the NEMO record is admitted and legoESM's production-JIT trace exists.

## Controls and verification

The committed self-check closes a synthetic process budget exactly, verifies
the `1,415,300`-byte layout, proves the raw ULP counterexample, and proves the
effect-scale plant reaches the decoded temperature budget.  Its terminal
lines are:

```text
process layout 1415300 bytes and synthetic endpoint closure -- OK
raw RHS ULP moves its decoded RHS increment but 0 temperature rows; effect-scale plant moves the temperature budget -- OK
self-check: all checks passed
```

The acquisition itself will run five nonzero plants: wrong producer stamp,
truncated frame, raw post-SBC ULP, effect-scale post-SBC propagation and a
one-ULP break in the `Taa -> next Tbb` chain.  `run.sh` accepts a plant only if
its log contains its exact `STATUS PLANT-FIRED` marker.

The final focused test, citation-gate and shifted-citation results are recorded
in the final verification amendment below.

## Landing, blast radius and required headlines

No candidate exists and nothing lands.  The Decision-43 ladder/month gate and
Decision-45 year gate are not invoked.  The unchanged admitted headlines are:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

The diagnostic patch exists only in a new NEMO acquisition target, not the
shared legoESM implementation.  GYRE production, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE and OVERFLOW execute no changed statement.  ORCA2 remains
**UNMEASURED-WITH-SPEC**: use its native compiled process order, dimensions,
mask and passive restart hashes before applying the same independent-
trajectory projection.  No `DECISION_NEEDED` arises from requesting a passive
record.

## Independent review

The required command was run as `codex exec --sandbox read-only` against the
complete incoming-tip-to-receipt diff with an adversarial prompt covering the
stop verdict, existing-record audit, field list, byte arithmetic, passive
admission, projection algebra, ULP-control retraction, citations and the lack
of a landing table.  Its verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
codex_exit_code=1
```

Independent review unavailable in-sandbox.  No `SHIP` or `DO NOT SHIP`
verdict is fabricated; the complete log is `round123/codex_review.log`.

## Final verification amendment

The clean-tree receipt citation gate found all six compiled-source citations,
no unmapped citations, no failures and no failing map entries.  Its literal
verdict was `status=PASS`.  The shifted comma-citation plant moved all three
line numbers by two, found the real first symbol at line 812, emitted
`SYMBOL-NOT-AT-LINE`, printed `status=FAIL`, and exited 1.  Thus the new
compiled-order citations are plant-controlled rather than merely present in a
map.

The final focused owner, citation and worktree-stamp suites ended literally:

```text
============================= 39 passed in 32.38s ==============================
```

The earlier failed focused attempt is retained in
`focused_tests_clean_pre_amendment.log`: it caught a test that applied the
swallowed-ULP assertion to the 14,400-second layout fixture instead of the
preregistered unit-timestep counterexample.  Commit `026bffea7e75` corrected
the test input; no gate or scientific criterion was weakened.  The final
green rerun is `focused_tests_final.log`.

The whole ocean test trees were not spent: no production implementation,
recipe or trajectory changed, and the focused set directly covers every
changed executable Python path, both synthetic record plants, source-card
static assertions, compiled-citation mapping and clean-tree stamping.

## OPEN — exact handoff to round 124

1. The operator runs exactly
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round123_process_budget/run.sh`.
   It must finish with `ROUND123_PROCESS_RECORD_READY` and `RUN_DONE`.  It
   creates the new target `GYRE_OMIP_L2_P3_SM_R123PROC`; it never modifies the
   source card.
2. Round 124 first reads the acquisition log and
   `round123_process_record_validation.json`.  Re-run the normal gate and all
   five plants.  Verify both restart hashes and byte comparisons before using
   a payload.  Read and cite the **target's compiled** `stprk3_stg.f90` writer
   statements.  If the target/run directory exists without full admission,
   diagnose it in place; never rebuild an unchanged binary or silently accept
   a partial record.  A source-card correction requires a new target name.
3. Preregister and add the legoESM side to this existing owner instrument.  It
   must run seed 0 independently from rest through step 1440 on CPU/fp64 via
   the production `_step_jitted` path, record the same accumulated stage-3
   boundaries for steps 1081--1440, and carry an effect-scale production
   plant.  No NEMO state substitution, isolated closure, or eager-only proof.
4. Require the NEMO and lego traces separately to close from day 180 to day
   240 and reproduce the immutable day-240 T3D RMS exactly.  Then emit the one
   ranked table: incoming gap, geometry, advection, surface boundary,
   shortwave, lateral diffusion, vertical diffusion and rounding closure;
   signed global projection and component RMS; day-30 comparator where
   available; birth interval and localized diagnostic.
5. Only the largest admitted process projection becomes the next candidate.
   If it requires a card or carried-state choice, stop with
   `DECISION_NEEDED` and a pick.  Otherwise any production change still needs
   every Decision-43 ladder/month row, all Decision-45 year rows, DINO and the
   recipe-derived executing-card set before landing.
