# NEMO testcase L2 GYRE Round 160 receipt — the second continuity solve

Date: 2026-09-23
Status: **HELD** — the build is committed behind this receipt and is NOT the
production path.  No configuration, no carried state, no card changed.

legoESM now has both of the oracle's per-stage continuity solves.  The
momentum vertical advection reads a vertical velocity built from the raw stage
velocity; the tracer transport keeps the one it always had, byte for byte.
With it on, the developed stage-2 vertical velocity drops by five orders of
magnitude, the stage-2 momentum right-hand side lands on round 158's ceiling,
the step-2 entry velocity goes ON THE BAR for the first time in this campaign,
the first row over the bar moves from step two to step three, and the day-30
temperature gap falls 4.8%.

**And the year refuses it.**  Day 240 worsens from 1.6446718648e-02 K to
1.6448360701e-02 K and day 360 from 1.1223450850e-02 K to 1.1225660019e-02 K,
eight thousand and eleven thousand times the harness's own run-to-run floor, so
neither is noise.  Under Decision 45 that is a HOLD, and the round holds: the
tip's trajectory is bit-identical to its base commit on all 70 certified rows,
proven, and the second solve is selected only by a private arm no card can
construct.

The residual round 159 registered is now separated.  The stage clock and its
after-level pair are MEASURABLY INERT.  What is left is the live thickness
ratio, and this round measures the operand difference directly.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round160.md`, committed as
`ba8e56dc0fee` before any round-160 measurement ran.  The developed-state
measurement is `split_walk.json` under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round160/walk/`.  The
trajectory, month and year arms are under the same root: the before arm is the
landing's OWN base commit `de4e0cfe35da`, measured from a pristine clone of it
(operator note W, item 2), and the candidate arm is `d255ff61d0dd`, the commit
that carries the model change, measured from its own pristine clone.

## The compiled program

The record's namelist sets `ln_dynadv_vec = .true.`, so the stage program takes
the vector-invariant arm at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356`, skips the solve
at stage 1 (`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:358`) and
at every later stage solves continuity on the RAW stage velocity at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`, through the
velocity indicator at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`.  The tracer
transport then re-solves it in the transport form at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`, entered because
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:195` sets the
vertical-transport flag, overwriting the same array before forming the tracer's
own vertical transport at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:280`; its indicator
branch is `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:134-138`.
Both branches divide by the live thickness and multiply it back at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153`, and both feed the
same bottom-up recurrence at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298`.

legoESM's side of the change is one function and two call sites:
`ocean_model_latlon_cgrid.py:1681-1692` is the predicate that decides which
cards are in this program at all,
`ocean_model_latlon_cgrid.py:1816-1821` is the second solve itself, and
`ocean_model_latlon_cgrid.py:6956-6962` is the helper the two momentum
consumers read it through.  Stage 1 gets no momentum solve.

**The adaptive-implicit pair is refused, not run once.**  Where
`ln_zad_Aimp` is on, the oracle partitions the momentum pair at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:362` and the tracer
pair at `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:277` separately.
Only the tracer partition is transcribed, so selecting both raises with a named
message.  No certified card reaches it.

## Controls, before any attribution

* **Face window.** The oracle's recorded step-entry velocity is BIT on both
  components against the level this walk loads from the same restart, 0 of
  21,780 and 0 of 22,080 cells.
* **Cell window.** The oracle's 104 dry columns and legoESM's 104 agree
  exactly, 0 disagreeing, which is what pins a cell-centre window.
* **Mask.** The oracle's vertical velocity is identically 0.0 on every cell the
  card calls dry.
* **Split liveness, both directions.** With the arm OFF the two consumers read
  ONE field: 0 of 18,000 interfaces differ.  With it ON they read two: all
  18,000 differ, at 1.2326874e-08.  An inert arm would otherwise report "the
  tracer field did not move" while proving nothing.
* **Authority.** The walk refuses to report unless its before arm reproduces
  round 159's production vertical velocity to 5e-22 and round 158's
  after-advection rows to 5e-18.  It does.
* **Passivity, as a magnitude.** Fourteen of the fifteen exposure arms move no
  other field at all.  One — the stage-2 right-hand-side exposure with the
  split on — moves three bytes of one number, 3.552713678800501e-15, which is
  1.0 unit in the last place of that field's own largest value and inside the
  campaign's own bound of 2.  That is the compiled-scheduling floor the
  campaign has measured before (operator notes L and AL); it is REGISTERED
  here, not excused.

## The developed stage-2 vertical velocity

One production step through `LatLonCGridOceanModel.step` under production
just-in-time compilation, from the oracle's admitted day-180 entry (step 1080,
the entry to step 1081).  Scores over the 18,000 wet cell interfaces the
exposure carries.  The oracle's own field has a root mean square of
2.3968803e-06 m/s.

| field | cells unequal | difference (rms) | as a fraction of the field | removed |
|---|---:|---:|---:|---:|
| before, the one shared field | 18000/18000 | 1.2326857e-08 | 5.143e-03 | — |
| after, the MOMENTUM field | 18000/18000 | **2.3346825e-13** | **9.741e-08** | **99.998%** |
| after, the TRACER field | 18000/18000 | 1.2326857e-08 | 5.143e-03 | 0 (by design) |

The tracer row is scored against the same oracle array, which the record
captures on the MOMENTUM side of the second solve, so it is expected to sit
where it was: the oracle's tracer field is a different array this record does
not carry.  The momentum row reproduces round 159's velocity-form arm to every
digit printed, 2.334682468902387e-13, from the production split instead of from
a private one-variable override.

## What it carries at the consumer

| arm | u rms | v rms | as a multiple of round 158's ceiling |
|---|---:|---:|---:|
| before | 3.8441664e-12 | 6.4285456e-12 | 342.8x / 491.5x |
| after | 1.1213546e-14 | 1.3075745e-14 | 0.99981x / 0.99964x |

Round 158's ceiling is what installing the oracle's OWN recorded vertical
velocity left.  The split reaches it from legoESM's own arithmetic.

## What stage 3 inherits

The stage-2 OUTPUT velocity is the state the stage-3 transport reads, and its
maximum difference against the oracle's recorded row is round 157's
1.30926020461275e-06 m/s, reproduced here to every digit.

| arm | u rms | u max | v rms | v max | rms removed |
|---|---:|---:|---:|---:|---:|
| before | 2.7410518e-08 | 1.3092602e-06 | 4.5815029e-08 | 2.1662164e-06 | — |
| after | 1.9012425e-10 | 6.1361964e-10 | 1.6744103e-10 | 5.8487051e-10 | 99.31% / 99.63% |

The split also fires at stage 3: the two fields there differ on all 18,000
interfaces at 1.2326873e-08.

## The tracer path does not move

| field | cells moved |
|---|---:|
| stage-2 tracer vertical velocity | 0/18000 |
| stage-2 tracer temperature | 0/18000 |
| stage-2 tracer salinity | 0/18000 |

The FINAL one-step tracer state does move, because the corrected stage-2
velocity feeds the stage-3 transport: temperature by 1.0762036e-08 K, which is
8.9% of the 1.2034924e-07 K that round 159 measured for changing the SHARED
field.  That is the difference between a change that reaches the tracers
through the physics and one that reaches them through a defect.

## The residual round 159 registered, separated

Round 159 left the corrected arm at 9.74e-08 of the field's own size, four to
ten times the floor it declared, and named two candidates.

| candidate | how it was measured | result |
|---|---|---|
| the stretching term's clock and after-level pair | substituted, both together | **INERT**: removes −5.70e-11 of the residual |
| the live thickness ratio | the operand scored directly against the oracle's own recorded one | **a real difference**: 580/580 u faces and 570/570 v faces, 5.724e-09 and 5.537e-09 of the operand's own size |

The clock arm is live, not a control that perturbs a zero: it moves the scored
field from 2.334682468902387e-13 to 2.3346824690355334e-13, i.e. it changes the
answer only in its own last bits.  That is what the source says it should do —
at stage 2 the oracle's clock is `rn_Dt/2`
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:221-222`) and its
after-level is the HALF level
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:254`, the np_HYB arm
this deck selects at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:52`), while legoESM
uses the full step with the full after level, and the two products are the same
number algebraically.  At stage 3 the oracle's clock and after-level are
already legoESM's
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:265-266`,
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:268`), so this
candidate could never have owned a residual that is present at both stages.

**CONFIRMED**: the clock and after-level pair is not the owner.
**PLAUSIBLE, not confirmed**: the live thickness ratio is.  What is measured is
that the operand itself differs on every active face at 5.7e-09 relative, four
orders above the compiled-rounding floor; what is NOT measured is that
substituting the oracle's own ratio closes the 9.74e-08.  Round 161 runs that
substitution.

## The full Decision 43/45 gate

Before arm: `de4e0cfe35da`, the landing's own base commit, measured.  Candidate:
`d255ff61d0dd`.  Report: `decision43_45.json`.

| criterion | verdict |
|---|---|
| day-30 temperature root-mean-square decreases | **PASS** 6.888193513796918e-05 -> 6.572574374770603e-05 K, a factor of 1.0480 |
| first row over the bar is not earlier | **PASS** step 2 (u, v) -> step 3 (T, S, u, v, ssh) |
| no kt=1 row at the bar leaves it | **PASS** every kt=1 row is unchanged, 0 cells moved |
| every moved row registered | **PASS** 60 rows, listed in `moved_rows.tsv` |
| DINO measured if the statement is shared | **PASS** not executed; see the card census |
| day 240 not worse | **FAIL** 1.6446718648e-02 -> 1.6448360701e-02 K |
| day 360 not worse | **FAIL** 1.1223450850e-02 -> 1.1225660019e-02 K |

Two criteria fail, so the round HOLDS.

### The year, all eight registered rows

The harness's own run-to-run floor is about 2e-10 K (round 129), quoted next to
every row as required.

| day | before (K) | after (K) | change | relative | multiples of the floor |
|---:|---:|---:|---:|---:|---:|
| 30 | 6.8881935138e-05 | 6.5725743748e-05 | −3.1562e-06 | −4.58e-02 | 15,781 |
| 60 | 1.9325579917e-04 | 2.0808020004e-04 | +1.4824e-05 | +7.67e-02 | 74,122 |
| 90 | 1.8644545401e-03 | 1.8643018774e-03 | −1.5266e-07 | −8.19e-05 | 763 |
| 120 | 1.0500500326e-03 | 1.0484738105e-03 | −1.5762e-06 | −1.50e-03 | 7,881 |
| 180 | 3.5805134081e-03 | 3.5838854686e-03 | +3.3721e-06 | +9.42e-04 | 16,860 |
| 240 | 1.6446718648e-02 | 1.6448360701e-02 | +1.6421e-06 | +9.98e-05 | 8,210 |
| 300 | 1.3597381746e-02 | 1.3602990041e-02 | +5.6083e-06 | +4.13e-04 | 28,041 |
| 360 | 1.1223450850e-02 | 1.1225660019e-02 | +2.2092e-06 | +1.97e-04 | 11,046 |

Three days improve and five worsen.  Every worsening is thousands of times the
floor, so the verdict is not a noise call.

### The certified trajectory, the rows that changed status

60 of the 70 certified rows moved; 494,431 cells improved and 114,308 worsened.
Two rows changed status, both from DEBT to AT-BAR, and none went the other way.

| row | cells improved | cells worsened | largest worsening | the difference it carried before |
|---|---:|---:|---:|---:|
| GYRE-zco.kt2.before.u | 17397 | 2 | 4.1633e-17 (0.19 ulp) | 2.7377e-12 |
| GYRE-zco.kt2.before.v | 17100 | 0 | 0 | 3.2849e-12 |

Those two numbers are the campaign's own long-standing step-2 velocity debt,
and the second continuity solve takes both to the bar.  The kt=2 temperature
and salinity rows do not move at all, so they stay at the bar.  The largest
worsening anywhere is GYRE-zco.kt6.before.u at 4.9400e-06, where 13,496 cells
improve and 3,904 worsen.

### The card census, from the code's own predicates

The admission gate imports the model's two predicates rather than restating
either.  The middle column is what the gate scores: which cards would run the
CANDIDATE, i.e. with its arm selected, which is the blast radius of the
statement under test.  The last column is what runs at this tip, and it is no
everywhere because the round is HELD.  The census in `decision43_45.json` is
stamped at the round's final commit and says exactly this.

| card | momentum integrator | momentum advection | continuity solve | runs the candidate | runs at this tip |
|---|---|---|---|---|---|
| GYRE-zco | rk3_ws | vector_invariant | nemo_literal | **yes** | no |
| LOCK_EXCHANGE-zco | rk3_ws | flux_form | generic | no | no |
| OVERFLOW-zps | rk3_ws | flux_form | generic | no | no |
| NEMO-GYRE recipe | rk3_ws | vector_invariant | generic | no | no |
| DINO nemo_dino_kamm | euler | vector_invariant | nemo_literal | no | no |
| DINO nemo_dino_kamm_mlf | euler | vector_invariant | nemo_literal | no | no |

DINO is out on the momentum integrator: its lane never enters the RK3 stage
program, so the statement is not shared with it.  Both tanks are out on the
momentum-advection form, which is the oracle's own `ELSE` arm.  This is READ
OFF EACH CARD'S RESOLVED CONFIGURATION and confirmed by running each card's own
tests, not off a deck comment.

## Production is restored, and that is measured

**Registered, and found by the reviewer, not by me: for five of this round's
commits the split WAS the production default.**  The build commit selected it
from the card's resolved configuration with no arm to turn it off, and it
stayed that way until the final commit moved it behind its private selector.
That window is what the candidate arm was measured at, which is what the
controlled comparison needs, but it is also exactly the "default that nobody
chose" shape the campaign's own rules exist to catch, so it is written down
here rather than left in the history for a bisect to find.  Nothing was pushed
during it, and the tip is proven restored below.

At the held tip the certified trajectory is BIT-IDENTICAL to the base commit:
70 rows compared, 0 rows moved, 0 status changes, largest worsening 0 units in
the last place, first row over the bar unchanged at step 2.  Evidence:
`held_comparison.json`.

## Predictions and verdict

1. The tracer field does not move at the stage the split is made:
   **CONFIRMED**, 0 of 18,000 on the vertical velocity and on both tracers; and
   the final one-step temperature move is 8.9% of round 159's shared-field
   move, as predicted smaller.
2. The split reproduces round 159's corrected arm at stage 2: **CONFIRMED**,
   2.334682468902387e-13 against 2.334682468902387e-13, identical.
3. The stage-2 right-hand side reaches round 158's ceiling: **CONFIRMED**,
   0.99981x and 0.99964x.
4. Stage 3 inherits the correction: **CONFIRMED**, 99.31% and 99.63% of the
   stage-2 output velocity difference removed against a 50% threshold, and the
   before arm reproduces round 157's 1.30926020461275e-06 m/s exactly.
5. Neither named candidate owns the remaining 9.74e-08: **HALF CONFIRMED, HALF
   OPEN, and the open half is REGISTERED.**  The clock and after-level pair is
   inert as predicted (−5.7e-11 against a 10% threshold).  The live thickness
   ratio was NOT substituted, so the prediction about it is neither confirmed
   nor refuted; what is measured is that the operand differs at 5.7e-09 on
   every face, which makes it the only remaining named candidate.
6. The year verdict was preregistered as a coin flip whose sign was not
   predicted: the sign came out NEGATIVE on five of eight days including both
   the day-240 and day-360 rows, and the round HOLDS, exactly as the
   preregistration said it would in that case.  The preregistration also said
   the year rows would move by less than 1e-05 relative; that is **REFUTED**,
   they move by up to 7.7e-02 (day 60) and 2.0e-04 at day 360, and the
   refutation is kept here rather than edited away.
7. Both plants fire: **CONFIRMED**, see below.

## Plants, review and tests

Two plants.  A control that CATCHES its plant returns and the walk exits 1 with
`STATUS PLANT-FIRED`; a control that does NOT catch it raises with its own
marker and exits 2 with `STATUS PLANT-BLIND`.

* `wzv-split-shared` claims the split is on while the arm that moved is the
  SHARED field (round 159's arm).  The tracer-identity control refuses it: all
  18,000 interfaces of the tracer field move, at 1.2326874e-08, and the walk
  prints `STATUS PLANT-FIRED: wzv-split-shared` and exits 1.
* `wzv-split-inert` claims the split is on while both consumers still read one
  field.  The liveness control refuses it: 0 of 18,000 interfaces separate the
  two consumers, and the walk prints `STATUS PLANT-FIRED: wzv-split-inert` and
  exits 1.

**Registered, not demonstrated:** the `STATUS PLANT-BLIND` path is structurally
distinct — a raise with its own marker and its own exit code — but exercising
it needs a deliberately blinded plant committed to the tree, which the walk's
fail-closed commit stamp would then be measuring.  This round did NOT run it,
said out loud, exactly as round 159 recorded the same gap.

Adding the second solve lengthened the model file, so 178 citations across the
citation map and every fidelity document were re-anchored by RIGID shift: both
endpoints of each range moved by the same delta, every pinned extent unchanged,
every endpoint symbol re-resolved.  The pass rewrites each occurrence once and
never re-scans its own output, which is the self-collision that had to be
repaired by hand in round 159.  Sixteen prose citations could NOT be shifted
rigidly, because the block each names genuinely grew; they are reported rather
than widened, and none of them is in the citation map.  They are fenced here so
this receipt does not make them as citations:

```
ocean_model_latlon_cgrid.py:1274-1352   (round-8 and round-159 receipts)
ocean_model_latlon_cgrid.py:1287-1332   (nemo_branch_isomorphism_map)
ocean_model_latlon_cgrid.py:1888-1938   (round-66, round-67, round-68 receipts)
ocean_model_latlon_cgrid.py:6671-6734   (card reconciliation receipt)
ocean_model_latlon_cgrid.py:6686-6751   (decision-35 preregistration + receipt)
ocean_model_latlon_cgrid.py:6940-7426   (mlf step transcription spec)
ocean_model_latlon_cgrid.py:7132-7151   (round-66 ldf preregistration)
ocean_model_latlon_cgrid.py:7132-7571   (round-66 and round-67 receipts)
ocean_model_latlon_cgrid.py:9041-9121   (split-explicit rounds 2 and 3)
ocean_model_latlon_cgrid.py:9046-9119   (decision-36 receipt)
```

## The independent review

Codex is paused, so a Claude reviewer was run instead, on the whole round's
diff and the compiled sources, told to try to break the claims.  Its verdict
was **SHIP WITH CHANGES**, verbatim:

> the physics transcription and the held-tip safety are real and well-verified.

It opened every cited compiled line itself and confirmed each one says what
this receipt says it says, including the stage-2 half clock and half
after-level and the stage-3 full pair the residual argument rests on.  Its
findings and what was done with each:

1. **The admission gate's card census contradicted this receipt.**  CLOSED.
   The census was stamped at the round's second-to-last commit, where the
   predicate could not express a held candidate, and it read "GYRE executes".
   The census now answers two separate questions from the two predicates — what
   the candidate would reach, and what this tip reaches — and the gate was
   re-run at the final commit.
2. **For five commits the split was the production default.**  CLOSED as a
   REGISTERED DISCLOSURE, in its own paragraph above.  The reviewer found it; I
   did not.
3. **The passivity control was loosened in the round that needed it.**
   REGISTERED, not argued away.  It went from a hard zero-byte requirement to a
   bound of two units in the last place at row scale.  The bound and its unit
   are the repo's own shared constant, and the floor it admits is documented in
   the two previous rounds' receipts, not invented here; the one arm that used
   it sits at 1.0 of the 2.  The reviewer's caveat stands and is carried into
   OPEN: the floor should be reproduced on an arm that does not touch this
   round's code before it is relied on again.
4. Transcription, held-tip safety, test non-vacuity and the residual
   discriminator: no findings.  It checked that no production driver anywhere
   passes the private hooks, so no card can reach the split today.

## The six-file push gate and this round's own tests

The six-file push gate together with this round's own three test files, on the
committed tree, reported exactly:

> 182 passed in 1001.79s (0:16:41)

The other cards' own gates — both DINO recipes, both tanks and the tank
zero-diffusion removal — were run as a separate battery, because two full
batteries at once collapse this box's compiler:

> 170 passed, 9 warnings in 339.86s (0:05:39)

The generic NEMO-GYRE recipe's own file is inside the first battery.  Nothing
on any of those cards moved, which is what the census predicts: none of them
runs the statement, before or after.

The citation gate on this receipt reported `PASS` with 20 citations, zero
failures, zero unmapped citations, zero map entries failing audit, and all nine
self-tests fired.  Its shifted-line plant, on the named first non-bit statement
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`, exited 1 with
`SYMBOL-NOT-AT-LINE`.

The walk's two plants were re-run on the committed tree after the hold and both
still fire, each exiting 1.

Three earlier runs of the walk refused themselves before any of this was
reported, and each refusal was a control doing its job rather than a nuisance:
a passivity control that had not been told which slots an exposure writes, and
then the same control counting bytes where the honest quantity is a magnitude.
Both are fixed in the tree and the second is registered above.

## One more measurement this round did NOT make

The stage-3 momentum vertical velocity is scored only against the stage-3
TRACER field, not against the oracle, because no developed stage-3
vertical-velocity record exists.  Everything said about stage 3 here is either
that the split fires there (measured, 18,000 interfaces) or that the stage-2
output velocity it inherits improves (measured against the oracle).  The
stage-3 producer itself is **UNMEASURED**.

## OPEN — Round 161

1. **Substitute the live thickness ratio.**  Install the oracle's OWN recorded
   `r3u(Kmm)`/`r3v(Kmm)` into the velocity-form producer's face transports with
   the split on, and report how much of the 2.334682e-13 m/s residual it
   removes.  The record carries them; this round measured the operand
   difference (5.7e-09 relative, every face) but not its ownership.
2. **The year is the open question, not the statement.**  This is the first
   candidate in the campaign to take a certified trajectory row TO the bar and
   move the first row over the bar LATER, and the year still worsens.  That
   pattern — exactness up, year down — is the thing to explain before the next
   landing attempt: name which day-240 owner the corrected velocity feeds.
3. **Do not re-walk the clock and after-level pair.**  It is measurably inert
   here, and at stage 3 the oracle's pair is already legoESM's.
4. **Do not add a momentum solve at stage 1**, and check the adaptive-implicit
   partition before transferring this anywhere: where it is on there are two
   partitions, and only the tracer one is transcribed.
5. **Reproduce the compiled-scheduling floor on an unrelated arm** before the
   passivity bound is relied on again; the reviewer's caveat, carried.
6. **ORCA2 stays UNMEASURED-WITH-SPEC**: repeat this walk on the ocean-only
   ORCA2 card before transferring the verdict.
