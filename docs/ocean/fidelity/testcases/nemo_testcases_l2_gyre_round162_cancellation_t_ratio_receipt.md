# NEMO testcase L2 GYRE Round 162 receipt — the cancellation, refuted

Date: 2026-09-23
Status: **HELD** — nothing landed, and no production line changed: the round's
whole range touches no file under `packages/`, so there is nothing to restore.
Three of the four measurements it was ordered to make are in and each answers
its question, and the round also found a MISREAD in the campaign's own summary
of where the stage-2 residual lives.

**Order A, first, because Decision 55 is waiting on it.  THE CANCELLATION IS
REFUTED.**  The one-step tracer advection error and the vertical-diffusion
error do not cancel; at the developed day-180 entry they are essentially
INDEPENDENT.  Their cell-by-cell correlation over the 18,000 wet cells is
`+0.0027` — positive, not negative — the two carry opposite signs on `48.6%`
of cells, which is a coin flip, and the combined one-step temperature
increment of the pair moves the WRONG way for a cancellation: it is
`2.1834125817722328e-05` K in production, ABOVE the vertical-diffusion row
alone at `2.1834089000437955e-05` K, and removing the advection error makes it
BETTER, `2.1834094298886284e-05` K in the corrected arm.  So round 160's
second continuity solve makes this pair of tracer rows MORE exact at the
developed state, not less, and the day-240 worsening it was refused for has no
measured mechanism in this pair.

**Order B.  The T-point ratio is innocent too.**  Installing the ratio NEMO's
own recorded stage height implies, in the one place the ratio at the now level
enters the velocity-indicator continuity solve, moves the developed stage-2
residual from `2.334682468902387e-13` to `2.334682468820876e-13` m/s — it
removes `3.49e-11` of it, and its whole influence on the scored field is
`3.228417925758222e-22` m/s, which is `1.4e-09` of the residual.  Five
operands of that producer have now been substituted one at a time and none of
them owns it.

**Order C.  The stage sea surface height is measured, and round 161's
arithmetic was right.**  legoESM's stage-2 `Kmm` height differs from the
oracle's recorded one on all 600 active columns, at
`6.279490676432346e-10` m root mean square — round 161 predicted "of order
`6e-10` m" from the ratio statement alone — and dividing that difference by
the reference depth reproduces round 161's measured face-ratio difference to
`0.6%` (`1.4601055758923643e-13` against `1.4516405645036015e-13`).  But the
height is NOT the residual's owner, and that follows from Orders A and B
rather than from a new measurement: the height's only two consumers in this
producer are the face ratio and the T-point ratio, and both are now measured
inert.

**The misread, corrected loudly.**  Round 161's OPEN section, and the campaign
state line that quotes it, say the oracle's own stage-2 entry velocity "removes
nothing (`-1.5e-06`)".  That number is from round 159's TRANSPORT-form arm,
where the residual is `1.23e-08` and a `1e-13`-scale operand cannot show.  In
the VELOCITY-form arm — the arm rounds 160, 161 and 162 have all been working
in — round 159's own published table says the oracle's entry velocity takes the
residual from `2.3346825e-13` to `1.0492082e-13` m/s.  **It removes `55.06%`
of it.**  The residual is therefore NOT unowned: its largest measured owner has
been sitting in a committed receipt for three rounds while three more operands
were substituted at the `1e-06`-and-below level.  That is round 163's target.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round162.md`, committed as
`10de147f7` before any round-162 measurement ran.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round162/`: `rank/` for
Order A, `walk/` for Orders B and C, `plants/` for the plant runs.  Entry for
every developed-state row: NEMO's admitted day-180 daily restart, step 1080,
the entry to step 1081.  Every arm is one production step through
`LatLonCGridOceanModel.step` under production just-in-time compilation, never
an isolated closure.  Oracle stage-2 record sha256
`004f8493a91fbb5a5fd69227e198bea531c6e945bc84f0a7dc8eb4ec3c9563f9`, read
through the reader rounds 158-161 already share.

## Order A — the discriminating measurement

Each row is legoESM-minus-NEMO for one isolated one-step temperature
contribution at the same developed entry.  The production arm is required to
reproduce round 152's published table to one part in a million before anything
is reported, and it does.

| quantity | value |
|---|---:|
| advection row, production | `1.0974591404090626e-08` K |
| vertical diffusion row, production | `2.1834089000437955e-05` K |
| **combined row, production** | **`2.1834125817722328e-05` K** |
| **combined row, corrected** | **`2.1834094298886284e-05` K** |
| combined if the two were independent | `2.1834091758548193e-05` K |
| cross term | `+1.4873034263716965e-15` K² |
| Pearson correlation of the two error fields | `+0.0026539162702729348` |
| uncentred cosine | `+0.0031034599597189708` |
| fraction of cells carrying opposite signs | `0.4862222222222222` |
| combined row's change between arms | `-3.1518836043735454e-11` K |

Read three ways, all agreeing:

1. **The correlation is positive and negligible.**  `+0.0027` against a
   preregistered threshold of `-0.1`.  Cancellation needs a NEGATIVE
   correlation; this is the opposite sign and two orders too small.
2. **The combined row is ABOVE the quadrature reference, not below it.**  If
   the two errors were independent the combined row would be
   `2.1834091758548193e-05` K; it is `2.758e-12` K above that, which is the
   positive cross term.  A cancelling pair would put the combined row BELOW
   both that reference and the vertical-diffusion row alone.  It is
   `3.68e-11` K above the vertical-diffusion row instead.
3. **Removing the advection error IMPROVES the pair.**  The corrected arm's
   combined row is `3.15e-11` K smaller.  Under a cancellation it would have
   grown.

The sum rule the reading rests on is checked, not assumed: the combined row's
square minus the two squares minus the cross term is
`2.0679515313825692e-25` K², against terms of order `4.8e-10` K² — fifteen
orders down, so the decomposition is exact to the precision of the claim.

**VERDICT: CANCELLATION REFUTED.**  All three preregistered tests fail in the
same direction.

### What this says about Decision 55, which is NOT reopened here

Round 160's second continuity solve was refused by the year gate (day 240
`1.6446718648e-02` -> `1.6448360701e-02` K, day 360 `1.1223450850e-02` ->
`1.1225660019e-02` K).  Round 161 found the split makes the tracer advection
row 1,776 times more exact and the vertical-diffusion row marginally worse,
and offered the cancellation as the PLAUSIBLE story that would have made the
split innocent and named a cancelling partner as the real defect.  **That story
is now measured and it is false.**  At the developed one-step level the split
improves the pair of rows that carries the year, so nothing in this
measurement excuses the year's refusal, and there is no named partner to
blame.  The input to Decision 55 is therefore: the split is better locally on
every row measured and still worse over the year, with no mechanism; keep
holding it rather than landing it on a story that has now been refuted.

Per the preregistration, the conditional naming of a first non-bit statement
in the vertical-diffusion path does NOT trigger, and nothing there is named.

### Controls

| control | result |
|---|---|
| production arm reproduces round 152's six published rows | PASS, to 1e-6 relative, enforced before anything is reported |
| the combined row decomposes into the two rows the ranking scored | residual `0.0` on every one of the 18,000 cells, against an allowance of `8.67e-19` K |
| that control can fail | its plant is below |

## Order B — the T-point ratio

### What the compiled program does with it, and where the order's premise was wrong

The ratio at the now level enters the velocity arm of the divergence exactly
twice, and both occurrences are the same live thickness: the whole horizontal
divergence is divided by it at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130` and the same
live thickness multiplies the divergence back at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153`.  legoESM forms the
ratio at `ocean_pe_latlon_cgrid.py:1674`, forms the thickness at
`ocean_pe_latlon_cgrid.py:1675` and hands it to the divergence block at
`ocean_pe_latlon_cgrid.py:1699`, so replacing that one operand replaces both
compiled occurrences and nothing else.

**The order for this round named a second home for it and the compiled source
refutes that.**  The stretching term at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298` reads the ratio
at the AFTER and BEFORE levels, not the now level; legoESM's answering pair is
`ocean_pe_latlon_cgrid.py:1717-1718`.  Round 160 already measured the
after-level pair inert, and the before level is the entry state's own sea
surface height, which this round's entry bridge loads from NEMO's restart with
zero mismatches.  So the stretching term is accounted for and needs no
substitution, and this round's Order B is the divisor alone.  This was
registered in the preregistration, before measuring.

### The operand, and why the transcription is the statement

The record carries the stage sea surface height, not the T-point ratio, so
what is installed is NEMO's own ratio statement
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:257`, one product of the
height and the reciprocal reference depth) evaluated on the oracle's recorded
`ssh(Kmm)`.  That is said rather than left implied.  It is calibrated before
it is used: fed legoESM's OWN height, the same transcription reproduces the
production step BIT FOR BIT, 0 of 18,000 cells, so it is the same statement
and not a second copy of it.

### The scores

Over the 18,000 wet cell interfaces, against the oracle's own recorded stage-2
vertical velocity, whose root mean square is `2.3968803420671446e-06` m/s.

| arm | vertical velocity (rms, m/s) | as a fraction of the field | residual removed |
|---|---:|---:|---:|
| corrected (round 160's second solve) | `2.334682468902387e-13` | `9.740504888487189e-08` | — |
| corrected + the oracle's T-point ratio | `2.334682468820876e-13` | `9.740504888147118e-08` | **`3.4913062409732493e-11`** |
| shared velocity form (round 159's arm) | `2.334682468902387e-13` | `9.740504888487189e-08` | — |
| shared + the oracle's T-point ratio | `2.334682468820876e-13` | `9.740504888147118e-08` | **`3.4913062409732493e-11`** |

The two arms agree to every digit printed, which is what the one-producer claim
predicts.

The magnitude bound is again the cleaner statement.  The substitution's WHOLE
effect on the scored field is `3.228417925758222e-22` m/s root mean square,
maximum `4.235164736271502e-21`, on 3,927 of the 18,000 interfaces.  That is
`1.4e-09` of the residual: even a substitution that happened to point the
right way could not have removed more than a billionth of it.

**Why it had to be that small, and the mechanism is mechanical rather than
argued.**  The divergence block divides by the live thickness and multiplies
the same live thickness back, so a RELATIVE change in that operand leaves the
transport term unchanged except for rounding.  A committed test perturbs the
operand by a part in a million and requires the answer to move by less than a
part in a million million.  The one term where the cancellation does not hold
is the river mass flux, which is divided by the thickness and not multiplied
back; on the GYRE card the freshwater forcing the gate builds sets the runoff
to an array of zeros, so that term contributes nothing here, and the test
covers the case anyway so the caveat is not carried as prose.

## Order C — the stage sea surface height

The height legoESM hands the continuity producer at the stage-2
velocity-indicator call, sunk in the harness by the same observer and scored
against the oracle's recorded `ssh(Kmm)` on the same T window.

| row | value |
|---|---:|
| active columns unequal | 600 of 600 |
| difference, root mean square | `6.279490676432346e-10` m |
| difference, maximum | `1.875196477263419e-09` m |
| the oracle's own height, root mean square | `0.12221990649579682` m |
| relative | `5.137862445221475e-09` |
| the T-point ratio it implies | `1.4601055758923643e-13` |
| round 161's measured face ratio difference (u) | `1.4516405645036015e-13` |
| ratio of the two | `1.0058313411706412` |

**C1 CONFIRMED**: `6.28e-10` m, inside the preregistered `1e-10` to `5e-09`
band and on round 161's predicted order.  **C2 CONFIRMED**: the height
difference divided by the reference depth reproduces round 161's measured face
ratio difference to `0.6%`, far inside the factor of two the preregistration
allowed.  Round 161's arithmetic off the ratio statement was right, measured
from the other end.

**C3 is answered differently from the way round 161 expected, and the
difference matters.**  Round 161 said that if C1 held, the walk should move to
the barotropic step that produces the after height.  It should not, for this
residual.  The stage height's only consumers inside the velocity-indicator
producer are the two ratios it feeds — the face ratio at
`ocean_pe_latlon_cgrid.py:1674` through the face geometry, and the T-point
ratio at the same line into `ocean_pe_latlon_cgrid.py:1675` — and both are now
measured inert, at `1.5e-05` and `1.4e-09` of the residual respectively.  A
`6.28e-10` m height error is a real difference in the model's own stage state
and it will matter wherever the height is consumed directly, but it cannot be
the owner of the stage-2 vertical-velocity residual, because everything it
touches there has been substituted and moves nothing.

## The residual, and what actually owns it

Five operands of the velocity-indicator continuity producer have now been
substituted from the oracle's own record, one at a time:

| operand | round | residual removed |
|---|---|---:|
| the stage-2 entry velocity, in the TRANSPORT-form arm | 159 | `-1.5e-06` |
| **the stage-2 entry velocity, in the VELOCITY-form arm** | **159** | **`55.06%`** |
| the stage clock and after-level pair | 160 | `-5.70e-11` |
| the face free-surface ratio | 161 | `-6.34e-07` |
| the T-point free-surface ratio | 162 | `3.49e-11` |

**The second row is the correction this round makes to the campaign's own
record.**  Round 159's published arm table reads: velocity-indicator call form
alone `2.3346825e-13` m/s, "both together" — the call form plus the oracle's
own stage-2 entry velocity — `1.0492082e-13` m/s.  That is a removal of
`55.06%`, and it is the largest measured owner of the residual by nine orders
of magnitude over anything rounds 160 to 162 have substituted.  Round 159's
prose concluded "the difference is made at this statement, not inherited from
stage 1", which was true of the `1.23e-08` the transport form carries and is
NOT true of the `2.33e-13` that survives the form fix; round 161's OPEN
section carried the transport-form number forward as if it were the
velocity-form one, and this round's first three orders were spent on operands
a hundred million times smaller.  **Retracted here, in the same place it was
asserted.**

What is left after the entry velocity is `1.0492082e-13` m/s, `4.377e-08` of
the field's own size, and that is still four orders above the compiled
rounding floor this campaign has measured.  So the walk is not finished after
round 163 either, and the receipt says so now rather than later.

## Controls, before any attribution

| control | result |
|---|---|
| observer passivity, substitution OFF | **0 bytes** moved on every state leaf — a hard zero, not a bound |
| height-sink passivity | **0 bytes** moved on every state leaf, so reading the height out did not change the step |
| statement calibration, legoESM's own height through the substituted statement | **BIT**, 0 of 18,000 cells |
| substitution liveness | 3,927 of 18,000 interfaces move, at `3.228417925758222e-22` m/s |
| tracer identity | 0 of 18,000 on the tracer's own vertical velocity |
| call ledger, corrected arm | 6 producer calls, 2 velocity-indicator, exactly 1 substituted, exactly 30 levels inside it |
| call ledger, shared arm | 4 producer calls, 1 velocity-indicator, exactly 1 substituted |
| authority | both baselines reproduce round 160's residual to `5e-22` |
| combined-row decomposition | residual `0.0` on 18,000 cells against `8.67e-19` K allowed |

The statement calibration is the control this round adds and it is the one that
makes Order B a substitution rather than a re-implementation.  The walk cannot
install "the oracle's recorded `r3t`" because the record does not carry one; it
installs the compiled ratio statement on the oracle's recorded height.  Run
with legoESM's OWN height in place of the oracle's, that statement has to
reproduce the production step exactly, and it does, on every cell.  Anything
less would have meant the arm was measuring the transcription rather than the
operand.

## Predictions and verdict

1. **A1, the correlation is NEGATIVE** — **REFUTED.**  It is
   `+0.0026539162702729348`, the opposite sign and two orders below the
   preregistered `-0.1` threshold.
2. **A2, the production combined row is below the vertical-diffusion row and
   grows in the corrected arm** — **REFUTED both ways.**  It is `3.68e-11` K
   ABOVE the vertical-diffusion row, and it SHRINKS by `3.15e-11` K in the
   corrected arm.
3. **A3, the decomposition control holds** — **CONFIRMED**, with a residual of
   exactly `0.0` on all 18,000 cells against an allowance of `8.67e-19` K.
4. **A4, the production arm reproduces round 152's table to 1e-6** —
   **CONFIRMED**, enforced by the walk before anything is reported.
5. **B1, the T-point ratio does not own the residual** — **CONFIRMED.**  It
   removes `3.49e-11` against a 10% threshold, and its whole influence on the
   scored field is `1.4e-09` of the residual.
6. **B2, the two arms agree** — **CONFIRMED**, to every digit printed.
7. **B3, the observer is passive and live** — **CONFIRMED** both ways: 0 bytes
   with the substitution off (and 0 bytes with the height sink on), 3,927 of
   18,000 cells with it on.
8. **B4, the substitution does not reach the tracers** — **CONFIRMED**,
   0 of 18,000.
9. **C1, the height difference is of order 6e-10 m** — **CONFIRMED**,
   `6.279490676432346e-10` m, inside the preregistered band.
10. **C2, it reproduces round 161's face ratio difference** — **CONFIRMED**,
    to `0.6%` against a factor-of-two threshold.
11. **C3, the walk moves to the barotropic after height** — **NOT ADOPTED**,
    and the reason is given above: the height's two consumers in this producer
    are both measured inert, so it cannot own this residual.  Round 163's
    target is the stage-2 entry velocity instead, which round 159 already
    measured at `55.06%`.
12. **D1/D2, the plants** — see below.

**A prediction the round did not make and should have.**  Nothing in the
preregistration said "check whether an earlier round already measured a larger
owner before substituting a new operand".  Three rounds of operand
substitution at the `1e-06`-and-below level ran while a `55%` row sat in round
159's own published table.  The cheap check that would have caught it is
reading the previous walk's arm table rather than its OPEN section, and it is
written into round 163's order below.
