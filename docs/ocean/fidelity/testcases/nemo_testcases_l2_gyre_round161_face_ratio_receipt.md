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
`ssh(Kmm)`, reproduces the oracle's recorded ratio to `5.34e-17`, which is
`0.037%` of the live difference; the other `99.96%` is the stage sea surface
height legoESM feeds that statement.  So the first non-bit statement upstream
of the ratio is not in the ratio.

**Order B, and it is the round's real finding.**  Exactness at stage 2 makes
the year worse because it removes a CANCELLATION.  Run at the same developed
day-180 entry, round 152's one-step magnitude ranking says the second
continuity solve takes the tracer ADVECTION row from `1.097459140409e-08` K to
`6.181193168124e-12` K — it removes `99.94%` of it — and the only row that
grows by more than a rounding is VERTICAL DIFFUSION, from
`2.183408900044e-05` K to `2.183409436263e-05` K, `+5.3622e-12` K.  Vertical
diffusion is the process rounds 123-128 measured as the day-240 owner and
round 152 ranked first.  The corrected velocity makes the term it feeds nearly
exact and makes the term that carries the year slightly worse.

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
| authority | both baselines reproduce round 160's residual to `5e-22` |

The mesh-source control is the one that decides whether the decomposition
below is about the sea surface height or about the mesh.  legoESM's LIVE stage
ratio is built by the stage helper from the card's rebuilt operands while the
continuity producer resolves NEMO's own raw ones; on this card those two
operand sets produce the same ratio bit for bit, so the decomposition is about
the height.

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
| legoESM's STATEMENT on the oracle's own height | `5.344637401239055e-17` | `5.213641248509828e-17` | **0.0368% / 0.0376%** |
| legoESM's live height against the oracle's | `1.4516389246421464e-13` | `1.3880815316769283e-13` | **99.99988% / 100.0013%** |

Every row is 580 of 580 active u faces and 570 of 570 active v faces unequal.
The account closes: the statement row and the height row sum to the live row
to better than one part in a thousand, which is what a decomposition of two
independent small differences should do.

**CONFIRMED**: the stage sea surface height legoESM feeds the ratio statement
owns the ratio difference.
**REGISTERED, unexplained**: the statement row is `5.34e-17`, which is about
`1.6e+04` units in the last place of the ratio, not the handful the
preregistration predicted for a pure re-association.  It is three orders below
the row it is being compared with, so it cannot change this round's verdict,
but it is not the compiled-rounding floor either and nothing here explains it.
It is named in OPEN rather than explained away.

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
3. **The ratio difference is an operand, not the composition** — **CONFIRMED
   in substance, REFUTED in its number.**  The statement row carries 0.037% of
   the live difference against a 10% threshold, so the operand conclusion
   stands; but the preregistration also said that row would be within about 4
   units in the last place of the ratio, and it is about 16,000.  That
   quantitative prediction is REFUTED and kept here rather than edited away,
   and the unexplained residual is carried into OPEN.
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

