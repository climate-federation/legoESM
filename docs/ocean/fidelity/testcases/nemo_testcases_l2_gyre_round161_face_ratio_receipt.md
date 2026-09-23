# NEMO testcase L2 GYRE Round 161 receipt — the face ratio, exonerated

Date: 2026-09-23
Status: **HELD** — nothing landed, nothing in the model changed, and production
is untouched by construction: this round added no production line at all.  The
two measurements it was ordered to make are both in, and both answer their
question.

**Order A.**  The live free-surface ratio at the u and v points does NOT own
the vertical-velocity residual round 160 left.  Installing the oracle's own
recorded `r3u(Kmm)`/`r3v(Kmm)` in the velocity-indicator continuity solve moves
the residual from `2.334682468902387e-13` to `2.334683948520e-13` m/s — it
makes it `0.00006%` WORSE, not better — and it does the same thing to every
digit in both of the arms that take that indicator.  The ratio's entire
influence on that field is `3.461798022705186e-18` m/s, which is 1.5 parts in
100,000 of the residual.  The operand round 160 named as "the remaining
candidate" is measured, and it is exonerated.

**Order A, the producer.**  The ratio difference itself is an OPERAND, not a
statement.  legoESM's ratio form, evaluated on the ORACLE'S OWN recorded
`ssh(Kmm)`, reproduces the oracle's recorded ratio to `4.04e-21` root mean
square — about ONE unit in the last place of a ratio whose size is `2.5e-05`,
i.e. the compiled-rounding floor — so the statement carries `2.8e-08` of the
live difference and the stage sea surface height legoESM feeds it carries all
the rest.  The first non-bit statement upstream of the ratio is not in the
ratio.

**Order B.**  The corrected velocity makes the term it feeds nearly exact and
makes the term that carries the year slightly worse.  Run at the same
developed day-180 entry, round 152's one-step magnitude ranking says the
second continuity solve takes the tracer ADVECTION row from
`1.097459140409e-08` K to `6.181193168124e-12` K — it removes `99.94%` of it —
and the only row that grows by more than a rounding is VERTICAL DIFFUSION,
from `2.183408900044e-05` K to `2.183409436263e-05` K, `+5.3622e-12` K.
Vertical diffusion is the process rounds 123-128 measured as the day-240 owner
and round 152 ranked first.  That the two rows move in OPPOSITE directions is
**CONFIRMED**.  That this is a cancellation, and that it is what carries round
160's `+1.64e-06` K day-240 worsening, is **PLAUSIBLE ONLY**; the measurement
that would decide it is named in OPEN, and the two changes differ in size by a
factor of about two thousand, which that measurement has to account for.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round161.md`, committed as
`dd66afa79` before any round-161 measurement ran.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round161/`: `walk/` for
Order A, `rank/` for Order B.  Entry for every developed-state row: NEMO's
admitted day-180 daily restart, step 1080, the entry to step 1081.  Every arm
is one production step through `LatLonCGridOceanModel.step` under production
just-in-time compilation, never an isolated closure.  Oracle record sha256
`004f8493a91fbb5a5fd69227e198bea531c6e945bc84f0a7dc8eb4ec3c9563f9`, read
through the reader rounds 158-160 already share.

**Registered, an amendment to a frozen document.**  One reference in the
preregistration named the stage file without the compiled branch in front of
it, so it resolved against no build and the citation gate refused the whole
document.  It was respelled with the build that owns it, in its own commit,
after the measurements were taken.  The line number, the statement and the
claim are unchanged; only the build the citation binds to is now explicit.
Nothing else in the preregistration was touched.

## The compiled program

The stage program takes the vector-invariant arm
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356`), skips the
momentum continuity solve at stage 1
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:358`) and at every
later stage solves it on the raw stage velocity
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`), through the
velocity indicator
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`), which is
where each face transport is rebuilt from `e3u_0*(1+r3u(Kmm))`.  That is the
statement this round substitutes into.  The tracer transport re-solves
continuity in the transport form
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`) and does not read
the ratio at all.

**Where the ratio comes from, which is the part round 160 had not read.**  The
oracle does NOT rebuild it from the stage sea surface height.  It calls the
ratio routine ONCE per step, on the after height
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:185`), whose own
statement is
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:266-267`, and each stage
then COMBINES two ratios: stage 1 sets the level this walk scores at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:211`, and stage 2
sets its own at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:255`, the hybrid arm
this deck selects at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:52`.  The heights are
blended by the same weights one block earlier
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:172`).

legoESM builds ONE ratio from the already-blended height: `vertical.py:238-239`
is the surface-weighted numerator, `vertical.py:247` is the ratio, and
`vertical.py:251` is the thickness statement the substitution re-executes with
the oracle's ratio in place of legoESM's.  Because the ratio statement is
LINEAR in the height, combining ratios and taking the ratio of the combined
height are the same number, and can differ only in rounding — which is what
makes a difference of forty million units in the last place an operand
difference rather than a transcription one.  That linearity is asserted
mechanically, not assumed:
`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round161_face_r3.py`.

## How the substitution is made, and why nothing in the model moved

An observer wraps the continuity producer and, for exactly ONE
velocity-indicator call, replaces the face thickness with NEMO's own
thickness statement carrying the oracle's ratio.  It restores the producer
immediately.  This is the technique this instrument's tracer and transport
observers already use, and it means the round changed no production line, so
no card can reach the arm and there is nothing to restore.

The reciprocals the producer's geometry helper also returns are NOT
substituted; the velocity indicator does not consume them (its caller takes
only the two thicknesses), and that is said here rather than left implied.

## Controls, before any attribution

| control | result |
|---|---|
| observer passivity, substitution OFF | **0 bytes** moved on every state leaf — a hard zero, not a bound |
| substitution liveness | 17,997 of 18,000 cells move, at `3.461798022705186e-18` m/s |
| tracer identity | 0 of 18,000 on the tracer's own vertical velocity |
| call ledger, corrected arm | 6 producer calls, 2 velocity-indicator, exactly 1 substituted |
| call ledger, shared arm | 4 producer calls, 1 velocity-indicator, exactly 1 substituted |
| mesh source | the card's rebuilt mesh operands and NEMO's raw ones give **BIT-IDENTICAL** ratios on the same height, 0 of 580 u and 0 of 570 v |
| live-ratio readout floor | `1.109918092764145e-16` maximum, against half a unit in the last place of 1.0 at `1.1102230246251565e-16` |
| authority | both baselines reproduce round 160's residual to `5e-22` |

The mesh-source control is the one that decides whether the decomposition
below is about the sea surface height or about the mesh.  legoESM's LIVE stage
ratio is built by the stage helper from the card's rebuilt operands while the
continuity producer resolves NEMO's own raw ones; on this card those two
operand sets produce the same ratio bit for bit, so the decomposition is about
the height.

**The readout floor is a defect this round found in its OWN instrument, and it
is written down because the first draft of this receipt reported it as
physics.**  The model can only hand the live ratio back as the stretching
factor `1 + ratio`.  The ratio is of order `2.5e-05` and the factor is of
order one, so recovering the ratio by subtracting one quantises it at half a
unit in the last place of 1.0 — about `1.1e-16`, which is FOUR ORDERS above
the ratio's own last place.  The first version of the offline reconstruction
did that subtraction, and its statement row came out at `5.34e-17`, which was
reported as a real difference that nothing explained.  It was the arithmetic
of the instrument.  The reconstruction now reads the ratio off the native
helper, which returns it directly, and the round trip is measured once, above,
where its maximum reproduces half a unit in the last place of 1.0 to 0.06%.
The rows that must use the live readout — every row containing legoESM's live
ratio — carry that floor, and at `5.3e-17` it is a thousandth of the
`1.45e-13` they report, so it changes nothing there.  The guard that caught it
is a committed test, not a second look.

## Order A — the substitution

Scores over the 18,000 wet cell interfaces, against the oracle's own recorded
stage-2 vertical velocity, whose root mean square is `2.3968803420671446e-06`
m/s.

| arm | vertical velocity (rms, m/s) | as a fraction of the field | residual removed |
|---|---:|---:|---:|
| corrected (round 160's second solve) | `2.334682468902387e-13` | `9.740505e-08` | — |
| corrected + the oracle's own ratio | `2.334683948520e-13` | `9.740511e-08` | **`-6.337555e-07`** |
| shared velocity form (round 159's arm) | `2.334682468902387e-13` | `9.740505e-08` | — |
| shared + the oracle's own ratio | `2.334683948520e-13` | `9.740511e-08` | **`-6.337555e-07`** |

A negative number is a worsening.  The two arms agree to every digit printed,
which is what the one-producer claim predicts.

The magnitude bound is the cleaner statement: the substitution's WHOLE effect
on the scored field is `3.461798022705186e-18` m/s root mean square, maximum
`3.912257247932441e-17`.  That is `1.48e-05` of the residual.  Even a
substitution that happened to point the right way could not have removed more
than a hundred-thousandth of it.

## Order A — the producer decomposition, offline from the record alone

The oracle's ratio has a root mean square of `2.5360349337097374e-05` (u) and
`2.506891783775622e-05` (v), so the units in the last place of the quantity
being decomposed are about `3.4e-21`.

| row | u (rms) | v (rms) | share of the live difference |
|---|---:|---:|---:|
| legoESM LIVE against the oracle | `1.4516405645036015e-13` | `1.3880628732222712e-13` | 100% |
| legoESM's STATEMENT on the oracle's own height | `4.036794459191678e-21` | `4.126336854973703e-21` | **`2.78e-08` / `2.97e-08`** |
| legoESM's live height against the oracle's | `1.4516405653154865e-13` | `1.3880628737935194e-13` | **`1.0000000006` / `1.0000000004`** |

Every row is 580 of 580 active u faces and 570 of 570 active v faces unequal.
The account closes to nine digits: the height row IS the live row, and the
statement row is a hundred-millionth of it.

The statement row's own scale settles it.  The ratio's root mean square is
`2.5360349337097374e-05` (u), so one unit in its last place is about
`3.4e-21`.  The statement row is `4.04e-21` root mean square — 1.2 units in
the last place — with a worst face at `2.0328790734103208e-20`, 6 units.  That
is the compiled-rounding floor of a re-association, which is exactly what the
compiled program and legoESM's transcription differ by here: the oracle builds
one ratio per step and combines ratios per stage, legoESM builds one ratio from
the combined height, and those are the same number up to rounding because the
statement is linear in the height.

**CONFIRMED**: the stage sea surface height legoESM feeds the ratio statement
owns the ratio difference, to nine digits.  The ratio statement itself is at
the rounding floor.

What this makes the next candidate, and it is the only place left upstream of
the ratio: legoESM's stage-2 `Kmm` sea surface height differs from the
oracle's.  The size follows from the ratio's own definition — a ratio
difference of `1.45e-13` divided by the reciprocal reference depth is a height
difference of order `6e-10` m at the u points, which at the one-third stage
weight is of order `2e-09` m in the after height the barotropic step produces.
That is arithmetic off the statement, not a measurement, and it is labelled
PLAUSIBLE; the measurement is named in OPEN.

## Order B — the same ranking in both stage programs

Round 152's one-step magnitude ranking, at the same developed entry, with the
production arm required to reproduce round 152's own published table to one
part in a million before anything is reported.  It does.

| contribution | production (K) | corrected (K) | change (K) | relative |
|---|---:|---:|---:|---:|
| advection | `1.097459140409e-08` | `6.181193168124e-12` | **`-1.0968e-08`** | `-99.94%` |
| vertical diffusion | `2.183408900044e-05` | `2.183409436263e-05` | **`+5.3622e-12`** | `+2.456e-07` |
| shortwave | `2.513690760498e-07` | `2.513690760779e-07` | `+2.8021e-17` | `+1.115e-10` |
| surface boundary | `4.657741019603e-15` | `4.680755513457e-15` | `+2.3014e-17` | `+4.941e-03` |
| lateral diffusion | `1.065521736676e-05` | `1.065521736676e-05` | `+7.5064e-18` | `+7.045e-13` |
| geometry | `6.281725340556e-12` | `6.281725340556e-12` | `0` | `0` |

The largest owner is vertical diffusion in BOTH arms, so the ranking's top is
unchanged.  What changed is that the row the corrected velocity feeds directly
— the tracer advection — collapses by a factor of 1,776, and the row that
carries day 240 grows.

**The answer to "which day-240 owner does the corrected velocity feed" is:
vertical diffusion, and it feeds it by REMOVING an error that was cancelling
against it.**  That is consistent with everything the campaign already
measured and did not fit together: rounds 124-128 found the day-240
attribution circling between temperature, the convective-adjustment trigger
and vertical diffusion; round 129 found the model deterministic, so the year
gap is a fixed difference amplified through the spin-up; round 134 found the
tracer pair carrying 99.7% of day 240.  A one-step advection error of
`1.1e-08` K that partly cancels a vertical-diffusion error of `2.2e-05` K is
exactly the shape that makes a locally exact fix worse over a year, and it is
the shape rounds 91 and 109 measured without being able to name the two terms.

**PLAUSIBLE, not confirmed**: that the cancellation is what carries the
`+1.64e-06` K day-240 worsening round 160 measured.  What is CONFIRMED is that
the corrected velocity moves those two rows in opposite directions at the
developed state.  The discriminating measurement is named in OPEN.

## Predictions and verdict

1. **The live face ratio does not own the residual** — **CONFIRMED**.  It
   removes `-6.34e-07` against a 10% threshold, and its whole influence is
   `1.5e-05` of the residual.
2. **The two arms agree** — **CONFIRMED**, to every digit printed, far inside
   the factor of two the preregistration allowed.
3. **The ratio difference is an operand, not the composition** —
   **CONFIRMED.**  The statement row carries `2.8e-08` of the live difference
   against a 10% threshold.  The preregistration also predicted that row would
   sit within about 4 units in the last place of the ratio, roughly
   `1.4e-20`: its root mean square is `4.04e-21`, 1.2 units, inside that; its
   WORST face is `2.03e-20`, 6 units, outside it by half again.  Registered as
   a partial miss rather than claimed as a clean hit.
   **Registered against this prediction:** it was REFUTED by the round's first
   measurement, at `5.34e-17`, and that refutation was written into a draft of
   this receipt as an unexplained physical result before the round's own guard
   showed it was the instrument's readout arithmetic.  The sequence is
   disclosed above rather than replaced by the corrected number alone.
4. **The observer is passive and live** — **CONFIRMED** both ways: 0 bytes
   with the substitution off, 17,997 of 18,000 cells with it on.
5. **The substitution does not touch the tracer path** — **CONFIRMED**, 0 of
   18,000.
6. **The grown process row is advection** — **REFUTED, and the refutation is
   the finding.**  Advection did not grow; it SHRANK by 99.94%.  The row that
   grew is vertical diffusion, which is the day-240 owner, and that is the
   answer Order B was asked for.  Vertical diffusion staying the largest was
   predicted and is confirmed.
7. **The compiled-scheduling floor is not a general floor** — **CONFIRMED**.
   Both unrelated exposure arms — the stage-1 output velocity and the stage-3
   tracer exposure, neither of which this round or round 160 wrote — move
   **ZERO bytes** against an ordinary production step.  So round 160's single
   one-unit-in-the-last-place row is specific to its own exposure, and the
   zero-byte passivity control is restored as the default for future arms.
   This closes round 160's registered review item 3.
8. **Both plants fire** — see below.

## Plants

A control that CATCHES its plant returns and the walk exits 1 with
`STATUS PLANT-FIRED`; a control that does NOT catch it raises with its own
marker and exits 2 with `STATUS PLANT-BLIND`.

* `face-r3-inert` claims the oracle's ratio is installed while handing the
  producer legoESM's OWN live ratio.  The liveness control refuses it: 0 of
  18,000 interfaces move, and the walk prints
  `STATUS PLANT-FIRED: face-r3-inert` and exits 1.
* `face-r3-tracer` wires the substitution into the SHARED producer that the
  tracers also read.  The tracer-identity control refuses it: the tracer field
  moves, and the walk prints `STATUS PLANT-FIRED: face-r3-tracer` and exits 1.

**Registered, not demonstrated:** the `STATUS PLANT-BLIND` path is
structurally distinct — a raise with its own marker and its own exit code —
but exercising it needs a deliberately blinded plant committed to the tree,
which the walk's fail-closed commit stamp would then be measuring.  This round
did NOT run it, said out loud, exactly as rounds 159 and 160 recorded the same
gap.

## Cards

Nothing in the model changed, so no card's numbers can move and none was
re-measured for a change.  The census is reported for the STATEMENT under
test, which is the velocity-indicator continuity solve: GYRE-zco resolves it,
both tanks take the flux-form momentum advection and the generic continuity
solve, the generic NEMO-GYRE recipe takes the generic solve, and both DINO
cards take the Euler momentum integrator and never enter the RK3 stage
program.  That census is round 160's, re-asserted by this round's own tests
rather than restated from its receipt.

**ORCA2 stays UNMEASURED-WITH-SPEC.**

## The independent review

Codex is paused, so a Claude reviewer was run instead, on the whole round's
diff, the receipt, the preregistration, the evidence and the compiled sources,
told to try to break the claims.  Its verdict was **SHIP WITH CHANGES**,
verbatim:

> Core mechanism is sound. One real defect in the receipt's own honesty
> discipline, one weak test.

It opened the cited compiled lines itself, traced which continuity call the
substitution reaches, checked the native/redundant face map against the shared
helper, confirmed that the producer's caller consumes only the two thicknesses
so the unsubstituted reciprocals are irrelevant, and confirmed from the diff
that no file under `packages/` is touched.  Its findings and what was done
with each:

1. **The headline stated a PLAUSIBLE causal story as fact.**  CLOSED.  The
   Order B headline said "exactness at stage 2 makes the year worse because it
   removes a CANCELLATION" with no hedge, while the body correctly labelled
   that PLAUSIBLE.  The headline now carries the CONFIRMED/PLAUSIBLE split
   itself and names the factor-of-two-thousand asymmetry the story has to
   account for.
2. **One new test was a source grep, not an executing check.**  CLOSED by
   DELETION.  It asserted that three literal strings were present in the walk;
   the walk already fails closed on the thing it was watching, so the test was
   removed rather than rewritten.
3. Substitution mechanics, face-index inversion, citations, the preregistration
   amendment, and the no-production-code claim: **no findings**.

**Found after the review, by the round's own push gate, and disclosed here:**
the face-map test the reviewer had just called non-vacuous FAILED, because it
asserted the map on the recovered ratio rather than on the stretching factor,
and the round trip between those two is not exact.  Chasing that failure is
what exposed the readout floor described above and turned the statement row
from an unexplained `5.34e-17` into a `4.04e-21` rounding floor.  The gate
found what the reviewer did not; both are reported.

## The gates and the tests

The citation gate on this receipt reported `PASS` with 14 citations, zero
failures, zero unmapped citations, zero map entries failing audit, and all nine
self-tests fired.  Its shifted-line plant, on
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:266-267`, exited 1 with
`SYMBOL-NOT-AT-LINE`.  The same gate on the preregistration reported `PASS`
with 9 citations, and its own plant on
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:211` exited 1 with
`SYMBOL-NOT-AT-LINE`.

The six-file push gate together with this round's own test file and round
160's ran as ONE battery, on the committed tree, and reported exactly:

> 167 passed in 972.57s (0:16:12)

An earlier run of that same battery, before the readout floor was found,
reported `1 failed, 166 passed in 970.86s` — the failure was this round's own
face-map guard, and chasing it is what produced the corrected statement row.
Both runs are quoted so that the green one cannot be read as the only one.

The green battery ran at `5bb50561a`; the commits after it change this receipt
and nothing else.  The one battery member that reads a receipt — the citation
gate's own test file — was re-run at the final tip together with this round's
test file and reported `21 passed in 3.51s`, and the citation gate itself was
re-run there and reported `PASS` with 15 citations and its plant exiting 1.

The walk's two plants were re-run on the final committed tree, and both still
fire, each exiting 1:

> STATUS PLANT-FIRED: face-r3-inert: {'cells_scored': 21120,
> 'cells_unequal': 0, 'active_cells_scored': 18000, 'active_cells_unequal': 0,
> 'active_max_abs': 0.0, 'active_rms': 0.0}

> STATUS PLANT-FIRED: face-r3-tracer: {'cells_scored': 21120,
> 'cells_unequal': 17997, 'active_cells_scored': 18000,
> 'active_cells_unequal': 17997, 'active_max_abs': 3.912257247932441e-17,
> 'active_rms': 3.461798022705186e-18}

**Nothing in the model changed, and that is checked rather than asserted.**
The round's whole range touches five files: the preregistration, this receipt,
the instrument, the citation gate and one test file.  No file under
`packages/` appears in it, so there is no production path to restore and no
card whose numbers could have moved.

## OPEN — Round 162

1. **The residual is UNOWNED, and the shelf of named candidates is empty.**
   Four operands of the velocity-indicator producer have now been substituted
   from the oracle's own record, one at a time, and every one is inert: the
   oracle's stage-2 entry velocity (round 159, removes `-1.5e-06`), the stage
   clock and after-level pair (round 160, `-5.70e-11`), and the face
   free-surface ratio (this round, `-6.34e-07`, in both arms).  What has NOT
   been substituted, and what the record still carries, is the T-point ratio
   `r3t(Kmm)` — it divides the whole horizontal divergence through the live
   `e3t` in
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130` and it
   appears again in the stretching term at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298`.  That is
   round 162's first arm, and it is the same one-call observer this round
   already has.  The static metric operands are NOT candidates: this round's
   mesh-source control showed the producer already resolves NEMO's own raw
   mesh.
2. **The stage sea surface height is the ratio's owner and has never been
   scored directly.**  The record carries `ssh(Kmm)` and `ssh(Kaa)`; legoESM's
   stage heights are not exposed.  Round 162 should score legoESM's stage-2
   `Kmm` height against the oracle's recorded one and check the arithmetic in
   this receipt — a height difference of order `6e-10` m at the u points.  If
   it holds, the walk moves to the barotropic step that produces the after
   height, which is where round 138 already found the first non-bit boundary.
3. **The discriminating measurement for Order B**, which this round did NOT
   run and names instead, as ordered.  The claim to test is that the tracer
   advection error and the vertical-diffusion error CANCEL in production.  The
   measurement is their COMBINED one-step temperature increment, scored
   against NEMO's combined trend, in both arms: if the two errors cancel, the
   combined row is smaller than either alone in production and grows in the
   corrected arm, even though the advection row alone collapses.  It costs one
   more scored row in the ranking mode this round already built, no new
   record, and it is the thing that would turn this round's PLAUSIBLE into a
   verdict.  A cell-by-cell sign correlation between the two rows in the
   production arm is the same measurement seen the other way and is equally
   cheap.
4. **What this says about Decision 55**, which is pending with the user and is
   NOT reopened here: round 160's split was refused by the year.  This round
   measured that the split makes the term it feeds 1,776 times more exact and
   nudges the term that carries the year.  If item 3 confirms the
   cancellation, the split is not the defect and the cancelling partner is,
   which is a different decision from the one currently on the table.  Said as
   an input to that decision, not as a proposal to change it.
5. **The `STATUS PLANT-BLIND` path is still not demonstrated**, for the third
   round running.  It needs a deliberately blinded plant committed to the
   tree, which the walk's fail-closed commit stamp would then measure.
   Carried.
6. **ORCA2 stays UNMEASURED-WITH-SPEC**: repeat this walk on the ocean-only
   ORCA2 card before transferring any verdict.

