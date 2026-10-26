# NEMO testcase L2 GYRE Round 155 receipt — developed transport walk

Date: 2026-09-22  
Status: **HELD** — no physics or configuration changed.  Given NEMO's own
recorded operands the two written stage-3 U-transport statements rebuild
NEMO's transport BIT, so no statement inside the stage transport is an owner.
The first unequal INPUT in compiled order is the external time-mean transport
`un_adv`; the operand that CARRIES the difference is the stage-2 velocity
`uu(Kmm)`, which alone removes 99.33% of it.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round155.md`, committed as
`06ebee850`; its directed-attribution addendum was frozen separately as
`49530f7e8`, before the substitution arms were run.  The authoritative
measurement commit is `2c3817e9e`, on a clean tree.  It reproduces the
compiled-order table value for value at each of the three commits that
measured it (`c29c3e995`, `6c4c2a2c0`, `f9422daed`) and carries the
attribution measured through the model's own helpers.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round155/`; the final
measurement is `developed_transport_walk_final.json`.  Commits after it are
this receipt and a rigid citation re-anchor, neither of which moves a number.

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair, or #1484 guard changed.
No configuration choice was made.  Without a source-exact candidate there is
no ladder, month, year, DINO, LOCK_EXCHANGE, or OVERFLOW candidate arm.

## Compiled program

This receipt cites only the compiled branch that produced the acquired
record.  Stage 3 binds Kmm to N+1/2 and restores the stage fields at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:261-277`.
The active barotropic-update dispatch forms `zub`/`zvb` from `un_adv`/`vn_adv`,
the live inverse depth, and `uu_b`/`vv_b(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:300-309`.
The next loop writes `zFu`/`zFv` from metric, reference thickness, live QCO
ratio, mask, Kmm velocity, and that correction at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:313-315`.
The additive writer reads those operands only after both production loops at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:317-321`.

## Admission and method

The Round-154 record's normal gate reports `STATUS PASS`: 20 finite registered
fields and all seven restarts plus `mesh_mask.nc` bit-identical to the
Round-153 baseline.  Its stamp, truncation, restart-byte, and operand-ULP
plants each print their named `STATUS PLANT-FIRED` marker and exit 1; all five
runs were repeated during this round
(`round154_admission_recheck_6c4c2a2c0.json`,
`round154_plants_6c4c2a2c0.status`).  The daily record audit at the
measurement commit admits 360/360 boundaries and 12/12 monthly overlaps.

The existing year-owner developed-step harness loads NEMO's exact step-1080
restart and executes `LatLonCGridOceanModel.step` through production JIT.  It
observes the shared QCO geometry, corrected-velocity, and metric-transport
helpers without changing their return values.  The observer's full returned
state is bit-identical to an ordinary step.  Production eager and isolated
JIT are reported separately; production-step JIT is authoritative.

The source-ordered production-JIT U table is below.  `unequal` counts every
registered face; `active unequal` counts only faces on which NEMO's 3-D U mask
is one.  Two-dimensional rows use a column with any active U level.

| row | unequal / scored | max abs | active unequal / scored | active max abs |
|---|---:|---:|---:|---:|
| `un_adv` | 580 / 726 | 1.3413024362307624e-6 | 580 / 580 | 1.3413024362307624e-6 |
| `r1_hu_0` | 0 / 726 | 0 | 0 / 580 | 0 |
| `1+r3u(Kmm)` | 580 / 726 | 6.522560269672795e-13 | 580 / 580 | 6.522560269672795e-13 |
| live inverse depth | 580 / 726 | 1.5165277887640993e-16 | 580 / 580 | 1.5165277887640993e-16 |
| `uu_b(Kmm)` | 617 / 726 | 6.134203041352482e-10 | 580 / 580 | 6.134203041352482e-10 |
| written `zub` | 580 / 726 | 3.0153192095982995e-10 | 580 / 580 | 3.0153192095982995e-10 |
| `e2u` | 0 / 726 | 0 | 0 / 580 | 0 |
| reference `e3u_0` | 4,380 / 21,780 | 3.007100172156124e2 | 0 / 17,400 | 0 |
| `umask` | 0 / 21,780 | 0 | 0 / 17,400 | 0 |
| live `e3u(Kmm)` | 21,780 / 21,780 | 3.007100172156124e2 | 17,400 / 17,400 | 1.9616663848864846e-10 |
| `uu(Kmm)` | 17,693 / 21,780 | 1.30926020461275e-6 | 17,400 / 17,400 | 1.30926020461275e-6 |
| corrected U | 17,400 / 21,780 | 1.3093558326737753e-6 | 17,400 / 17,400 | 1.3093558326737753e-6 |
| completed `zFu` | 17,400 / 21,780 | 1.4246544619672932 | 17,400 / 17,400 | 1.4246544619672932 |

All three execution modes have 3/13 bit-exact rows and name `un_adv` first.
Production eager has the same `un_adv` value; its later `uu_b`, `zub`, Kmm
velocity, corrected velocity and completed transport differ in last-bit
rounding from production JIT.  Isolated JIT reproduces the captured production
JIT correction and product exactly, but is not relabelled production.

## Instrument correction and retraction

The first committed instrument reconstructed zero-ssh reference thickness
from the bridge mesh instead of capturing the masked reference operand passed
to the production stage.  That made the live-thickness row appear unequal in
all 21,780 cells with a 300.710017 m maximum.  The internal reconstruction
control exposed 4,380 dry-cell failures.  That row is **RETRACTED**.

The corrected instrument captures the exact production reference operand.
Its model-side reconstruction of live `e3u(Kmm)` from reference thickness,
`1+r3u`, and mask is bit-exact in both production JIT and eager execution.
The remaining 4,380 reference-thickness differences are all dry and the
active reference row is bit-exact.  The corrected active live-thickness
difference is 17,400 cells, maximum 1.9616663848864846e-10 m, inherited from
the already non-bit live ratio.

## Directed operand attribution (Decision 43, magnitude)

Compiled order names the first unequal input; it does not say which operand
CARRIES the difference.  The two written statements were re-evaluated in one
isolated JIT closure that calls the SAME two shared helpers the production
stage calls, with ONE operand replaced by NEMO's recorded value, and the
product scored against NEMO's own recorded transport over the 17,400 active U
faces.  Arms are isolated-closure JIT and are not production.

Two anchor arms calibrate the instrument.  With nothing substituted the
closure is byte-identical to the production transport row, which the gate
requires.  With every operand substituted it rebuilds NEMO's transport
**BIT — 0 of 17,400 active cells and 0 of all 21,780 cells unequal, maximum
0** — so legoESM's transcription of the compiled barotropic correction and of
the compiled metric-transport product is exact, and no statement inside the
stage-3 transport is an admissible owner.

The ranking uses the root mean square over active faces, because the argmax
of a maximum is free to move between arms; the maximum is reported beside it.

| substituted operand | active max abs | max removed | active rms | rms removed |
|---|---:|---:|---:|---:|
| none (production) | 1.4246544619672932 | 0 | 4.119014779519268e-2 | 0 |
| `uu(Kmm)` | 9.611541259801015e-3 | 0.9932534228359285 | 1.971739597490058e-3 | 0.9521307957598496 |
| `un_adv` | 1.4247952934965724 | -9.885311353657392e-5 | 4.0906314330495554e-2 | 6.890809571949426e-3 |
| live inverse depth | 1.42465446217102 | -1.43000859952954e-10 | 4.1190142497562796e-2 | 1.2861400521610125e-7 |
| live `e3u(Kmm)` | 1.424654508344247 | -3.255312433357603e-8 | 4.119014533873493e-2 | 5.963702188595683e-8 |
| `e2u` | 1.4246544619672932 | 0 | 4.119014779519268e-2 | 0 |
| `umask` | 1.4246544619672932 | 0 | 4.119014779519268e-2 | 0 |
| `uu_b(Kmm)` | 1.424409581362852 | 1.7188771802470907e-4 | 4.1951556442092586e-2 | -1.8485212791316354e-2 |
| `uu(Kmm)` and `un_adv` | 1.9552775600459427e-2 | 0.986275425991044 | 3.7540190212523483e-3 | 0.9088612393449463 |
| all (calibration) | 0 | 1 | 0 | 1 |

Every arm leaves all 17,400 active faces unequal except the calibration arm,
which leaves none.

`uu(Kmm)` is the magnitude owner: it alone removes 95.2% of the transport
difference by rms and 99.3% by maximum.  `un_adv`, the first unequal input in
compiled order, removes 0.69% by rms and none at all by maximum, and adding it
on top of `uu(Kmm)` raises the residual rms from 1.97e-3 to 3.75e-3, so at
this magnitude the two partly compensate.  `uu_b(Kmm)` is negative on rms:
substituting NEMO's value alone makes the transport slightly worse.

This is a one-step magnitude at the developed day-180 entry, not a day-240
carry: no candidate exists, so no year arm was run and no day-240 number is
claimed for any operand.

## Predictions and verdict

1. Record admission and all four acquisition plants: **CONFIRMED**.
2. Reference geometry bit-exact on every registered cell: **REFUTED**.  The
   reference thickness differs in 4,380 dry cells; its 17,400 active cells,
   plus `e2u`, `umask`, and `r1_hu_0`, are bit-exact.
3. First production-JIT non-bit U row is `un_adv`: **CONFIRMED**, 580/580
   active columns, maximum 1.3413024362307624e-6 m2/s.
4. At least one later Kmm state operand is non-bit: **CONFIRMED**; all three
   preregistered state rows are non-bit.
5. Production `un_adv` ULP plant: **CONFIRMED**.  One ULP at active index
   `[1,2]` moves 16 completed `zFu` levels, maximum
   7.275957614183426e-12 m3/s, prints `STATUS PLANT-FIRED`, and exits 1; it
   was re-run at the final measurement commit with the same values.

Addendum predictions, frozen at `49530f7e8` before the substitution arms ran:

A1. NEMO's own operands rebuild NEMO's transport bit for bit: **CONFIRMED**,
   0 of 17,400 active cells and 0 of all 21,780 cells unequal.
A2. `uu(Kmm)` alone removes more than 90%: **CONFIRMED**, 95.21% by rms and
   99.33% by maximum.
A3. `un_adv` alone removes less than 1%: **CONFIRMED**, 0.69% by rms.
A4. The magnitude owner is `uu(Kmm)`, not `un_adv`: **CONFIRMED** on both
   metrics.

**First non-bit statement:** none inside the measured transport block, and
the calibration arm proves it: NEMO's own operands rebuild NEMO's transport
bit for bit through legoESM's two statements.  The first unequal input in
compiled order is the external time-mean transport `un_adv`; the operand that
carries the difference is the stage-2 velocity `uu(Kmm)`.  The stage transport
inherits both and nothing lands.  Round 152 measured the completed FCT
contribution at only 1.0974591404090626e-8 K one-step RMS; this walk localises
that measured row but claims no day-240 carry for either operand.

The campaign headline rows remain inherited and unmodified: kt2 T/S at the
bar, kt2 U/V approximately 2.7377e-12 / 3.2849e-12, kt3 T approximately
8.60e-7 K, day-30 6.888194e-5 K, day-240 1.644671864e-2 K, and day-360
1.122345086e-2 K.  ORCA2 is **UNMEASURED-WITH-SPEC**: repeat this developed
stage-3 transport and upstream external-average walk on its ocean-only card
before transferring the verdict.

## Controls, review, and tests

The synthetic first-operand registry test passes and fails when the registered
`zFu` row is removed.  The final production-path ULP plant is described above.

The attribution carries its own two controls inside the run, both required by
the gate: the unsubstituted arm must reproduce the production transport row
cell for cell, and the fully substituted arm is the calibration that either
rebuilds NEMO's transport bit for bit or names the statement association as
the candidate.  It rebuilt it bit for bit.

The required read-only Codex review was attempted after the implementation.
It did not start a reviewer and emitted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Its exit was 1, so an independent review was obtained from a separate
reviewer instead.  That reviewer returned **DO NOT SHIP** on the first
attribution, with this blocker verbatim:

> The closure hand-copies the arithmetic of `_nemo_stage_corrected_velocity`
> and `_nemo_metric_stage_transport` instead of calling them. Every
> substitution arm, including the "all" calibration, therefore exercises the
> *instrument's* transcription, not legoESM's.

It was right, and it is fixed: the closure now calls both shared helpers, and
the corrected-velocity helper returns the written correction itself on a
write-only keyword so the walk reports the executed value rather than a
recomputation.  Re-measured, every compiled-order row is unchanged value for
value and the calibration arm is still bit — which also REFUTES the same
review's hypothesis that the recomputed correction differed from the executed
one under production JIT: both are 3.0153192095982995e-10.

Its other findings are closed as follows.  The baseline anchor now requires
byte identity with the production row instead of an equal count of unequal
cells.  The ranking moved to the root mean square, whose value is a whole-field
quantity, with the maximum reported beside it; `uu(Kmm)` is first on both.  The
small-effect predicate now takes an absolute value, so a large negative
fraction can no longer pass it.  The fifth observed call is pinned to the
zonal metric, so a U/V ordering change cannot silently score the wrong side.
The two static reference rows, `e3u_0` and `r1_hu_0`, record in the report
that they are an EAGER re-evaluation of the shared builder at zero free
surface from the production call's own inputs, not values sunk from the jitted
stage; the retraction paragraph above should be read with that provenance.
Registered and not closed: the new registry test proves row ordering and
registry completeness only, not the observation, the index mapping, or the
attribution — those rest on the in-run controls.

This round lands no physics.

The citation gate on this receipt reported `PASS` with 4 citations, zero
failures, zero unmapped citations and zero map entries failing audit; its
shifted-line plant exited 1 with `SYMBOL-NOT-AT-LINE`.  Exposing the written
correction added ten lines to the C-grid ocean model, which moved every
citation pinned below it, so thirty-five map entries and the four receipts
that quote them were re-anchored by a rigid ten-line shift: both endpoints
moved, no pinned extent changed and no anchor symbol was weakened.  The gate
on the cumulative receipt is green again at 274 citations, zero unmapped.

The focused campaign and blast-radius set reported exactly:

> 199 passed in 747.59s (0:12:27)

An earlier run of the same set reported `1 failed, 198 passed` on the
cumulative-receipt citation check; that failure is what showed the re-anchor
had to reach the receipts as well as the map, and it is green once it did.
The set covers the six-file push gate, the year-owner harness tests, and the
RK3 tracer tests that exercise the corrected-velocity helper whose signature
gained the write-only keyword.

No unasked scientific or configuration choice was made.  No production
default, configuration, carried state, restart schema or physics changed: the
only production edit is a write-only keyword that returns an already-computed
value and is set by no production caller.

## OPEN — Round 156

The magnitude owner is `uu(Kmm)`, the stage-2 velocity that stage 3 consumes,
not the compiled-order first input `un_adv`.  Round 156 walks it.

1. From the same admitted day-180 entry, walk the stage-2 momentum program in
   compiled order under production JIT until the first non-bit operand of the
   velocity that stage 3 reads as `uu(Kmm)`.  Reuse the existing developed-step
   harness and the round-140/146 RHS family records rather than writing another
   implementation; round 149 already made the developed momentum LDF bit, so
   start from the terms that walk ranked below it.
2. Keep the calibration discipline this round established: before naming any
   statement, show that legoESM's transcription of it rebuilds NEMO's output
   bit for bit from NEMO's own recorded operands.  A statement that passes that
   test is exonerated and the walk moves upstream.
3. `un_adv` stays registered as the first unequal input but is NOT the round-156
   candidate: it removes none of the transport difference, and the external-step
   walk should wait until a magnitude justifies it.
4. Only a single source-exact production-JIT statement becomes a candidate, and
   it must pass the full Decision-43/45 month, day-240, day-360, ladder,
   moved-row and DINO gates, plus the generic NEMO-GYRE card and the
   LOCK_EXCHANGE/OVERFLOW tanks, before landing.
5. If the stage-2 record lacks the operands, write a fail-closed operator-run
   script under a new target name rather than inferring them.
