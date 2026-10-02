# NEMO testcase L2 GYRE Round 162 receipt — the cancellation, refuted

Date: 2026-09-23
Status: **HELD** — nothing landed, and no production line changed: the round's
whole range touches no file under `packages/`, so there is nothing to restore.
All four measurements it was ordered to make are in and each answers its
question, the round also found a MISREAD in the campaign's own summary of
where the stage-2 residual lives, and it found and disclosed two defects in
its own instrument, each caught by one of its own controls.

Decision 55 was answered by the user while this round was measuring — LAND
round 160's split, under an amended year gate — so Order A is reported below
as what it measured rather than as an input to a pending decision, and what it
does and does not support in that decision's stated rationale is said plainly
in the Order A section.

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
residual from `2.334682468902387e-13` to `2.3346824689582235e-13` m/s — it
makes it `2.39e-11` WORSE rather than better — and its whole influence on the
scored field is `3.362782104179888e-22` m/s, which is `1.4e-09` of the
residual.  Five operands of that producer have now been substituted one at a
time and none of them owns it.

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

### What this says about Decision 55, WHICH WAS ANSWERED WHILE THIS ROUND RAN

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

**Decision 55 was answered by the user on 2026-09-24, while this round was
measuring: LAND round 160's second continuity solve, under an amended year
gate.**  A statement that is NEMO's own and cited, proven one-variable, and
takes at least one certified trajectory row from debt to the bar now lands
even if the day-240 or day-360 temperature root-mean-square moves by less than
one part in a thousand.  Round 160's split qualifies on the numbers already
measured: day 240 moves `9.98e-05` relative and day 360 `1.97e-04` relative,
both inside that bar.

**What this round's measurement does and does not support in that decision's
stated rationale, said plainly because the decision cites this round.**  The
rationale reads "the year cost is the vertical-diffusion term's OWN error being
exposed, not a defect in the split".  The first half of that — no defect in the
split — is what this round's measurement supports: at the developed state the
split leaves the pair of tracer rows that carries day 240 MORE exact, by
`3.15e-11` K.  The second half is NOT supported by this measurement and is not
refuted by it either: at the one-step level the vertical-diffusion row grows by
only `5.36e-12` K, a thousandth of what would be needed to account for
`1.64e-06` K at day 240, and the combined row moves the other way.  So the
one-step state carries no mechanism for the year cost in EITHER direction, and
the year cost remains unexplained.  That is a statement about what has been
measured, not an argument against the decision, which rests on the relative
year bar and not on a mechanism.

### Controls

| control | result |
|---|---|
| production arm reproduces round 152's six published rows | PASS, to 1e-6 relative, enforced before anything is reported |
| the combined row decomposes into the two rows the ranking scored | residual `0.0` on every one of the 18,000 cells, against an allowance of `8.67e-19` K |
| that control can fail | its plant is below: mis-wired, the residual is `0.0398` K on all 18,000 |

**Why that residual is a HARD zero rather than a few last places, explained
rather than quoted.**  Every row is one temperature boundary minus another at
the same few tens of kelvin, so each row is an exact integer multiple of that
temperature's own last place — about `3.55e-15` K at 20 K — and so is every
sum and difference of them, because none of them leaves that grid.  The
identity the control checks is then exact by construction.  A committed test
builds four rows the same way and finds the residual zero on essentially every
cell, and builds four independent numbers of the same magnitude and finds it
non-zero on most of them, so the zero is a property of how the rows are made
and not of the check.

## Order B — the T-point ratio

### What the compiled program does with it, and where the order's premise was wrong

The ratio at the now level enters the velocity arm of the divergence exactly
twice, and both occurrences are the same live thickness: the whole horizontal
divergence is divided by it at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130` and the same
live thickness multiplies the divergence back at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153`.  legoESM forms the
ratio at `ocean_pe_latlon_cgrid.py:1739`, forms the thickness at
`ocean_pe_latlon_cgrid.py:1740` and hands it to the divergence block at
`ocean_pe_latlon_cgrid.py:1764`, so replacing that one operand replaces both
compiled occurrences and nothing else.

**The order for this round named a second home for it and the compiled source
refutes that.**  The stretching term at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298` reads the ratio
at the AFTER and BEFORE levels, not the now level; legoESM's answering pair is
`ocean_pe_latlon_cgrid.py:1835-1836`.  Round 160 already measured the
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
| corrected, legoESM's own height through the same statement | `2.334682468902387e-13` | `9.740504888487189e-08` | **BIT**, 0 of 18,000 |
| corrected + the oracle's T-point ratio | `2.3346824689582235e-13` | `9.740504888720145e-08` | **`-2.3916190563377753e-11`** |
| shared velocity form (round 159's arm) | `2.334682468902387e-13` | `9.740504888487189e-08` | — |
| shared + the oracle's T-point ratio | `2.3346824689582235e-13` | `9.740504888720145e-08` | **`-2.3916190563377753e-11`** |

A negative number is a worsening.  The two arms agree to every digit printed,
which is what the one-producer claim predicts.

The magnitude bound is again the cleaner statement.  The substitution's WHOLE
effect on the scored field is `3.362782104179888e-22` m/s root mean square,
maximum `6.829203137237796e-21`, on 3,679 of the 18,000 interfaces.  That is
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
| the T-point ratio the height difference implies | `1.4601055758923643e-13` |
| round 161's measured face ratio difference (u) | `1.4516405645036015e-13` |
| ratio of the two | `1.0058313411706412` |

The last three rows compare a T-point quantity with a u-point one, and that is
why they agree to `0.6%` rather than exactly: the face ratio is a
surface-weighted average of two neighbouring cells' heights divided by a face
reference depth, so it is the same difference seen through one averaging
step.  Agreement to half a percent is what that predicts, and it is what the
preregistration's factor-of-two threshold was set for.

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
`ocean_pe_latlon_cgrid.py:1739` through the face geometry, and the T-point
ratio at the same line into `ocean_pe_latlon_cgrid.py:1740` — and both are now
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

## The defect this round's own plant found, and disclosed

**The round's first Order B measurement was WRONG, and the walk's inert plant
is what refused it.**  The substitution hands the divergence block one level
of the substituted thickness per call, selected by a counter.  The counter was
carried across model runs instead of being reset per call, so from the SECOND
substituted arm onward every level was handed the index of a level that does
not exist.  The array library CLAMPS an out-of-range integer index instead of
raising, so what came back was the deepest level's thickness on every level —
a plausible small number rather than an error.

The first scored arms of this round were therefore measuring a thickness
column that no stage ever builds, and they reported a residual removal of
`+3.49e-11` and a liveness of 3,927 cells, both meaningless — the re-run gives
`-2.39e-11` and 3,679 cells.  What caught it:
the `t-r3-inert` plant installs legoESM's OWN height and claims it is the
oracle's, and the liveness control must see zero cells move.  It saw 3,831
cells move at `4.24e-21`, so the plant was NOT caught, the walk raised
`PLANT-BLIND` and exited 2 — the fail-closed path Order D was going to
demonstrate deliberately, fired for real first.

Three things changed and all three are in the diff: the level counter is per
call, it is checked against the number of levels rather than trusted, and the
calibration arm is now repeated AFTER every other substituted arm, because a
calibration that runs only first was bit-exact and still missed this.  A
committed test pins the clamping behaviour that made the defect silent.  Every
Order B number in this receipt is from the re-run.  Orders A and C are
unaffected: neither touches this code path, and the height was sunk in a run
with no substitution at all.

## Controls, before any attribution

| control | result |
|---|---|
| observer passivity, substitution OFF | **0 bytes** moved on every state leaf — a hard zero, not a bound |
| height-sink passivity | **0 bytes** moved on every state leaf, so reading the height out did not change the step |
| statement calibration, legoESM's own height through the substituted statement | **BIT**, 0 of 18,000 cells |
| the same calibration REPEATED after every other substituted arm | **BIT**, 0 of 18,000 cells |
| substitution liveness | 3,679 of 18,000 interfaces move, at `3.362782104179888e-22` m/s |
| tracer identity | 0 of 18,000 on the tracer's own vertical velocity |
| call ledger, corrected arm | 6 producer calls, 2 velocity-indicator, exactly 1 substituted, exactly 30 levels inside it |
| the height sink | fired exactly once, on the stage-2 velocity indicator |
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
   removes `-2.39e-11` — it is marginally worse — against a 10% threshold, and
   its whole influence on the scored field is `1.4e-09` of the residual.
6. **B2, the two arms agree** — **CONFIRMED**, to every digit printed.
7. **B3, the observer is passive and live** — **CONFIRMED** both ways: 0 bytes
   with the substitution off (and 0 bytes with the height sink on), 3,679 of
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

## Plants

A control that CATCHES its plant returns and the walk exits 1 with
`STATUS PLANT-FIRED`; a control that does NOT catch it raises with its own
marker and exits 2 with `STATUS PLANT-BLIND`.

* `t-r3-inert` claims the oracle's height is installed while handing the ratio
  statement legoESM's OWN height.  The liveness control refuses it: 0 of
  18,000 interfaces move, the walk prints `STATUS PLANT-FIRED: t-r3-inert` and
  exits 1.

  > STATUS PLANT-FIRED: t-r3-inert: {'cells_scored': 21120,
  > 'cells_unequal': 0, 'active_cells_scored': 18000,
  > 'active_cells_unequal': 0, 'active_max_abs': 0.0, 'active_rms': 0.0}

* `t-r3-tracer` wires the substitution into the SHARED producer that the
  tracers also read.  The tracer-identity control refuses it: the tracer field
  moves, the walk prints `STATUS PLANT-FIRED: t-r3-tracer` and exits 1.

  > STATUS PLANT-FIRED: t-r3-tracer: {'cells_scored': 21120,
  > 'cells_unequal': 3679, 'active_cells_scored': 18000,
  > 'active_cells_unequal': 3679, 'active_max_abs': 6.829203137237796e-21,
  > 'active_rms': 3.362782104179888e-22}

* `rank-combined-mispair` wires the combined row to the wrong recorded trend
  while the two single rows keep the right one.  The decomposition control
  refuses it, and its numbers also show the control is not perturbing a zero:
  the residual goes from `0.0` on 0 cells to `0.0398` K on all 18,000, against
  an allowance of `5.55e-17` K.

  > STATUS PLANT-FIRED: rank-combined-mispair: {'max_abs_residual_K':
  > 0.03979497507367036, 'cells_with_a_residual': 18000,
  > 'combined_error_max_abs_K': 0.03979746876204082, 'allowed_K':
  > 5.551115123125783e-17, ..., 'holds': False}

### Order D — the blind path, demonstrated at last

Rounds 159, 160 and 161 each registered the `STATUS PLANT-BLIND` path as
structurally distinct and each said out loud that it had not been exercised.
It has now been exercised TWICE, once by accident and once on purpose, and
both are reported.

**By accident, and it caught a real defect.**  The first Order B run's
`t-r3-inert` plant was NOT caught, because the level-index defect described
above made the "inert" substitution move 3,831 cells.  The walk raised, printed
`STATUS PLANT-BLIND: t-r3-inert` and exited 2, on the committed tree at
`28dfe92ff`.  That is the strongest form of the demonstration: the path fired
on a violation nobody planted.

**On purpose.**  A deliberately blinded plant was committed at `d6711bfae`,
run, and removed again at `136f10d35`, and the removal is an exact revert (the
walk's source at `136f10d35` is byte-identical to its source at `a9dab4f96`).
The plant installs the INERT substitution and hands it to the TRACER-IDENTITY
control, which is scoped to the tracer's own vertical velocity and cannot see
a violation on the momentum path.  The control passed, the plant was not
caught, and the walk raised, printed its own marker and exited 2:

> STATUS PLANT-BLIND: t-r3-blind: PLANT-BLIND: the tracer-identity control did
> not refuse an inert substitution on the momentum path: {'cells_scored':
> 21120, 'cells_unequal': 0, 'active_cells_scored': 18000,
> 'active_cells_unequal': 0, 'active_max_abs': 0.0, 'active_rms': 0.0}

The marker a caught plant prints was never printed, and the exit code was 2,
not 1.  **This item is now CLOSED and no longer carried.**

## Cards

Nothing in the model changed, so no card's numbers can move and none was
re-measured for a change.  The census is reported for the STATEMENT under
test, which is the velocity-indicator continuity solve: GYRE-zco resolves it,
both tanks take the flux-form momentum advection and the generic continuity
solve, the generic NEMO-GYRE recipe takes the generic solve, and both DINO
cards take the Euler momentum integrator and never enter the RK3 stage
program.

**That census is round 160's, RESTATED here and not re-measured, because no
production line changed.**  Round 161's receipt said the same census was
"re-asserted by this round's own tests"; its test file contains no card or
census assertion, so this receipt does not repeat that claim.  The census
stands on round 160's own measurement and is labelled as such.

**ORCA2 stays UNMEASURED-WITH-SPEC**: repeat this walk on the ocean-only ORCA2
card before transferring any verdict.

**DINO** is not executed by the statement under test — both DINO cards take
the Euler momentum integrator and never enter the RK3 stage program — and
nothing changed in the model, so there is nothing to measure on it.

## The independent review

Codex is paused, so a Claude reviewer was run instead, on the whole round's
diff, the receipt, the preregistration, the evidence and the compiled sources,
told to try to break the claims.  Its verdict was **SHIP WITH CHANGES**,
verbatim:

> SHIP WITH CHANGES

It ran the test file rather than reading it, pulled round 159's evidence file
itself to check the 55% correction, recomputed Order A's quadrature identity
independently and matched it to the last digit, spot-checked more than five
receipt numbers against the evidence, opened the cited compiled lines, and
confirmed from the diff that no file under `packages/` is touched.  Its three
findings and what was done with each:

1. **BLOCKER — a new test could not reach its own assertion.**  CLOSED by a
   fix.  The guard that pins the array library's silent index clamping read a
   value off the trailing axis and converted it to a number; that leaves a
   two-dimensional array, and the conversion raises before the comparison
   runs.  The reviewer found it by RUNNING the file, which reading it would
   not have done.  The guard now asserts on a true scalar and additionally
   asserts the same silence in the trailing-axis form the walk itself writes.
   This is the second instrument defect this round found and disclosed rather
   than shipped.
2. **MEDIUM — "the ratio enters twice" allegedly overstates the scored path.**
   **REFUTED, measured against the compiled source rather than argued.**  The
   reviewer read the multiply-back at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153` as a separate
   downstream consumer that the vertical-velocity path never reaches, and read
   legoESM's divergence helper as never re-multiplying.  Both are wrong, and
   the source settles it: the array that statement writes is
   `pe3divUh`, which is the routine's OWN output argument
   (`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:101`, declared
   `INTENT(out)` and documented `e3t*div[Uh]`), and the vertical-velocity
   routine receives exactly that array as `ze3div`
   (`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:278`) before
   consuming it at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298`.  legoESM's
   answering helper likewise ends by multiplying the live thickness back, at
   `ocean_pe_latlon_cgrid.py:1593`.  The framing stands and the finding is
   registered as refuted, with the lines, rather than quietly dropped.
3. **LOW — the ordinal convention that selects which continuity call is
   substituted is inherited, not re-verified.**  REGISTERED, not closed.  The
   walk substitutes the FIRST velocity-indicator call of the step, which is
   round 161's convention taken unchanged.  What stands behind it here is
   indirect: both baselines are required to reproduce round 160's residual to
   `5e-22` before anything is reported, so the scored field is the one those
   rounds scored.  A future round should tag the call explicitly rather than
   count it, and this is written into round 163's order.

## The gates and the tests

The citation gate on this receipt reported `PASS` with 12 citations, zero
failures, zero unmapped citations, zero map entries failing audit, and all
nine self-tests fired.  Its shifted-line plant, on `ocean_pe_latlon_cgrid.py:1764`, exited 1
with `SYMBOL-NOT-AT-LINE`.  The same gate on the preregistration reported
`PASS` with 8 citations and its own plant, on
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:257`, exited 1 with
`SYMBOL-NOT-AT-LINE`.

The six-file push gate together with this round's own test file and round
161's ran as ONE battery, on the committed tree at `036392846`, and reported
exactly:

> 147 passed in 1014.52s (0:16:54)

The commits after that battery change this receipt and the citation map and
nothing else.  The two battery members that read a receipt — the citation
gate's own test file and this round's — were re-run at the final tip together
with round 161's, and the citation gate itself was re-run there and reported
`PASS` with 12 citations and its plant exiting 1.  That focused run reported
exactly:

> 28 passed in 4.04s

The three plant runs are quoted verbatim in the Plants section above, each
with its own exit code: `t-r3-inert` and `t-r3-tracer` exit 1 with
`STATUS PLANT-FIRED`, `rank-combined-mispair` exits 1 with
`STATUS PLANT-FIRED`, and the deliberately blinded `t-r3-blind` exits 2 with
`STATUS PLANT-BLIND`.

**Nothing in the model changed, and that is checked rather than asserted.**
The round's whole range touches five files: the preregistration, this receipt,
the walk, the citation gate and one test file.  No file under `packages/`
appears in it, so there is no production path to restore and no card whose
numbers could have moved.

## OPEN — Round 163

1. **THE RESIDUAL'S OWNER IS ALREADY MEASURED AND IT IS THE STAGE-2 ENTRY
   VELOCITY.**  Round 159's committed evidence file, not its prose, says:
   the velocity-indicator call form alone leaves `2.334682468902387e-13` m/s,
   and the call form WITH the oracle's own recorded `uu(Kmm)`/`vv(Kmm)` leaves
   `1.0492082366460718e-13` m/s.  That is `55.06%` of the residual by root
   mean square and `67.33%` by maximum, against `1.4e-09` for the T-point
   ratio and `1.5e-05` for the face ratio.  Round 163 walks that velocity:
   it is the stage-1 OUTPUT velocity, which the stage program hands the
   momentum continuity solve at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`.
   **Run the missing null first.**  Round 159's passivity control for the
   entry-override mechanism was run in the TRANSPORT-form arm only, where it
   moves `2.6e-13` of a `1.2326857e-08` m/s residual — `3.2e-21` m/s absolute,
   eight orders below the `1.2855e-13` m/s the entry velocity removes in the
   velocity-form arm.  That makes the `55.06%` robust, but the velocity-form
   null itself has never been run, so round 163 runs it before anything else
   and reports it.
2. **READ THE PREVIOUS WALK'S ARM TABLE, NOT ITS OPEN SECTION.**  Rounds 160,
   161 and 162 each substituted an operand at the `1e-06`-and-below level
   while a `55%` row sat in a committed evidence file, because round 161's
   OPEN section quoted the entry-velocity number from the wrong arm and the
   two rounds after it inherited that summary.  From now on, before choosing
   an operand to substitute, the round opens the previous walk's own JSON and
   ranks every arm in it.  That is a one-command check and it is the cheapest
   thing in this receipt.
3. **What is left after the entry velocity, so the walk is not declared
   finished early.**  `1.0492082366460718e-13` m/s is `4.377e-08` of the
   field's own root mean square, four orders above the compiled-rounding floor
   this campaign has measured many times.  So round 163 ends with a residual
   too, and the rounds after it should expect one.
4. **The stage sea surface height is measured and is NOT this residual's
   owner.**  It differs from the oracle's by `6.279490676432346e-10` m and it
   owns the ratio difference to nine digits, but both of its consumers inside
   this producer are now measured inert.  It stays on the board for round
   138's barotropic boundary, where the height is consumed directly, and it is
   NOT round 163's target.
5. **Decision 55 is ANSWERED and round 163's first job is the operator's, not
   this receipt's.**  The user decided on 2026-09-24 to LAND round 160's
   second continuity solve under an amended year gate, and the order for round
   163 is to flip that default, register the day-240 and day-360 change with
   the `2e-10` K floor quoted, and then walk the vertical-diffusion chain as
   the day-240 owner.  That comes first.  Items 1 to 4 and 6 above are the
   CONTINUATION of the residual walk and are for after it — except item 2,
   which costs one command and should be done before any operand is chosen.
   One thing to carry into the flip: this round measured that the one-step
   state shows NO mechanism for the year cost in either direction, so the
   day-240 re-ranking the amended gate asks for should not be expected to
   reproduce the "exposure" reading without its own measurement.
6. **Tag the substituted continuity call instead of counting it.**  The
   walk picks the FIRST velocity-indicator call of the step, a convention
   inherited unchanged from round 161.  What stands behind it is indirect —
   both baselines must reproduce round 160's residual to `5e-22` — and the
   round's reviewer registered that as worth making explicit rather than
   inheriting silently.  Round 163 tags the call.
7. **ORCA2 stays UNMEASURED-WITH-SPEC**: repeat this walk on the ocean-only
   ORCA2 card before transferring any verdict.
