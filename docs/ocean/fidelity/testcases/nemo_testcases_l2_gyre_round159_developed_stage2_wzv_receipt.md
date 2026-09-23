# NEMO testcase L2 GYRE Round 159 receipt — the developed stage-2 vertical-velocity producer

Date: 2026-09-23
Status: **HELD** — no physics, no configuration, no carried state changed.

The oracle solves continuity TWICE per stage on this deck, and legoESM solves
it once.  The momentum vertical advection is handed a vertical velocity built
from the RAW stage velocity; the tracer transport is handed a different one,
rebuilt from the barotropically corrected transports by a second call that
overwrites the same array.  legoESM builds only the second of those two and
gives it to both consumers.  Selecting the first at the stage-2 solve, one
variable, removes 99.998% of the stage-2 vertical-velocity difference and
drops the stage-2 momentum right-hand side onto round 158's ceiling to four
decimal places.  Substituting the oracle's own stage-2 entry velocity instead
removes nothing at all, so the difference is not inherited from stage 1: it is
this statement.

Nothing landed.  The faithful fix has to produce two vertical velocities per
stage rather than change the shared one, and the measurement of what changing
the shared one costs the tracers is in this receipt, as the reason.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round159.md`, committed as
`6e40cca43` before any round-159 measurement ran.  The authoritative
measurement is `wzv_walk.json` under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round159/walk/`, produced on
a clean tree at `ebcee8a8d`.  No card, no scheme selection, no tunable and no
carried field changed; the one model edit is a private diagnostic arm that no
public configuration can construct and no production path reads.

## The compiled program

The record's own namelist sets `ln_dynadv_vec = .true.`, so the stage program
takes the vector-invariant branch at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356` and, for every
stage after the first
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:358`), solves
continuity on the RAW stage velocity at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`.  The
flux-form call on the already-built transports,
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:367`, is in the
`ELSE` and does not execute on this deck.

The two solves differ in what they are handed, and that is not a rounding
difference.  The transports carry a barotropic velocity correction, added at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:309` and multiplied
into the face transports at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:315-316`; the raw
stage velocity does not.  Inside the solve the two branches are
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`, which rebuilds
each face transport from the velocity it was handed, and
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:134-138`, which simply
differences the corrected pair; both then divide by the live thickness and
multiply it back at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153`, and both feed the
same quasi-Eulerian bottom-up recurrence at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298`.

The second solve is the tracer transport's own.  The vector-invariant flag
switches it on at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:195`; it re-solves
continuity in the transport form at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`, overwriting the
same shared array, and only then forms the tracer's vertical transport from it
at `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:280`.  The momentum
consumer reads the array before that happens, at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:112-116`, and the record
captures it in between, at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:495`.

**So the premise this round was handed — that the stage vertical velocity the
momentum advection reads is also the tracer transport's vertical operand — is
TRUE OF legoESM AND FALSE OF THE ORACLE.**  That is registered here as a
correction, because it changes what a fix may do: correcting the momentum
field must leave the tracer field alone.

Two citation corrections against the preregistration, which quoted enclosing
branches rather than statements.  They are fenced so the gate reads the
corrected citations above rather than the superseded ones:

```
preregistered  divhor.f90:123-130   ->  divhor.f90:126-130  (the assignment,
                                        not the CASE and the DO around it)
this receipt   stprk3_stg.f90:310   ->  stprk3_stg.f90:309  (the correction is
                                        the first line of the pair, not the
                                        second)
```

The statements cited are the same ones.

## Controls, before any attribution

* **Face window.** The oracle's recorded step-entry velocity is the level this
  walk loads from the same restart.  It is BIT on both components, 0 of 21,780
  and 0 of 22,080 cells.  A wrong window or transpose cannot survive it.
* **Cell window.** The vertical velocity is a cell-centre field, so the face
  control does not cover it.  What pins the cell window is the dry-column
  pattern round 158 established: the oracle's density anomaly is identically
  zero on every dry column, the oracle and legoESM each have 104 of them, and
  0 disagree.
* **Mask.** The scored mask is the card's own wet-cell mask, so it has to be
  the oracle's: the oracle's vertical velocity is required to be identically
  zero on every cell the card calls dry, and it is, at 0.0.
* **Liveness.** The arm has to change the solve.  It moves all 18,000 scored
  interfaces, at a maximum of 4.896573e-08.
* **Authority.** The walk refuses to report unless its own production stage-2
  right-hand-side row reproduces the row round 158 split.  It does, at
  3.844166e-12 (u) and 6.428546e-12 (v).
* **Passivity.** The three exposure runs that are WRITE-only are byte-identical
  to an ordinary step on every other field, 0 on all three.  The arms that
  substitute an operand change the step itself and say so rather than being
  excused.

## The vertical velocity, scored against the oracle's own

Every arm is one production step through `LatLonCGridOceanModel.step` under
production just-in-time compilation from the oracle's admitted day-180 entry,
never an isolated closure.  Scores are over the 18,000 wet cell interfaces the
exposure carries.  The oracle's own field has a root mean square of
2.3968803e-06 m/s.

| arm | cells unequal | difference (rms) | as a fraction of the field | removed |
|---|---:|---:|---:|---:|
| production | 18000/18000 | 1.2326857e-08 | 5.143e-03 | — |
| null (legoESM's own entry velocity through the mechanism) | 18000/18000 | 1.2326857e-08 | 5.143e-03 | −2.6e-13 |
| the oracle's own stage-2 entry velocity | 18000/18000 | 1.2326876e-08 | 5.143e-03 | −1.5e-06 |
| **the velocity-indicator call form** | 18000/18000 | **2.3346825e-13** | **9.741e-08** | **99.998%** |
| both together | 18000/18000 | 1.0492082e-13 | 4.377e-08 | 99.999% |

legoESM's stage-2 vertical velocity is wrong by half a percent of its own
size.  That is four and a half orders of magnitude above the compiled-rounding
floor this campaign has measured many times, so it is not a floor.

The entry-velocity arm is the one that could have moved the walk upstream and
did not: the oracle's own stage-2 entry velocity removes nothing, and is
marginally worse than legoESM's own.  The difference is made at this statement,
not inherited from stage 1.

## What it carries at the consumer

| arm | u rms | v rms | removed | as a multiple of round 158's ceiling |
|---|---:|---:|---:|---:|
| production | 3.8441664e-12 | 6.4285456e-12 | — | 342.8x / 491.5x |
| the velocity-indicator call form | 1.1213546e-14 | 1.3075745e-14 | 99.708% / 99.797% | 0.99981x / 0.99964x |

Round 158's ceiling is what installing the oracle's OWN recorded vertical
velocity left, 1.121563e-14 and 1.308047e-14.  Correcting the call form
reaches it to within 1.9e-04 and 3.6e-04 — the same number, from legoESM's own
arithmetic instead of from the oracle's recorded array.  That is the closure
that identifies the producer's defect with the operand round 158 named.

## What changing the SHARED field costs the tracers

The oracle re-solves continuity in the transport form for the tracers, so
correcting the momentum field must not touch them.  legoESM shares one field,
so changing it does.  Measured, one production step from the same day-180
entry, over the same 18,000 wet cells:

| field | cells moved | one-step rms | one-step max |
|---|---:|---:|---:|
| temperature | 18000/18000 | 1.2034924e-07 K | 4.4462169e-06 K |
| salinity | 17998/18000 | 1.9511111e-07 | 6.8409722e-06 |

For scale, round 152 ranked the day-180 one-step tracer terms and found
vertical diffusion at 2.1834e-05 K and lateral diffusion at 1.0655e-05 K.  The
one-step temperature move here is 0.55% of the first of those.  It is small,
and it is not zero, and it is in the WRONG DIRECTION for a fix: the oracle's
tracer path does not see this correction at all.  A landing that changes the
shared field would therefore introduce a new tracer difference while removing a
momentum one.  **That is the measured reason round 160's candidate is two
vertical velocities per stage, not a one-line change.**

## Predictions and verdict

1. The producer's output is not at the floor: **CONFIRMED**, at 5.1e-03 of the
   field's own size against a 1e-06 threshold, with every scored cell unequal.
2. The owner is the call form: **CONFIRMED**, 99.998% removed against a 90%
   threshold.
3. The difference is not inherited from stage 1: **CONFIRMED**, the oracle's
   own stage-2 entry velocity removes −1.5e-06 against a 20% threshold.
4. The round-158 ceiling is reached: **CONFIRMED, more tightly than
   predicted** — the prediction allowed a factor of two and the measurement
   lands within 4e-04.
5. The tracer premise is false for the oracle and changing the shared field
   moves the tracers: **CONFIRMED**, 1.2034924e-07 K per step.
6. Both plants fire: **CONFIRMED**, see below.

## First non-bit statement

`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360` — the stage
continuity solve on the raw stage velocity, selected at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356` and evaluated
through `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`.
legoESM executes the other branch,
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:134-138`, on the
corrected transports.

What it carries at day 180: 99.998% of the stage-2 vertical-velocity
difference, hence 99.708% (u) and 99.797% (v) of the stage-2 momentum
right-hand-side difference, hence essentially all of the 1.309260e-06 m/s
stage-2 velocity difference round 157 measured and the 95.2% of the stage-3
transport difference round 155 attributed to it.

What it carries at DAY 240 is **NOT MEASURED**, because there is no candidate
to score: the faithful change is a second continuity solve at stages 2 and 3,
which does not exist yet.  The honest bound from what is already measured:
round 134 found velocity resets remove 6.5% of the day-240 gap, and round 149's
momentum landing moved day 240 by 2.2e-08 K on 1.6447e-02 K, within 1e-06
relative and near the harness's own run-to-run floor of about 2e-10 K (round
129).  That bound is read off earlier measurements, labelled **PLAUSIBLE**, not
a measurement of this statement.  Note also that the tracer number above is
2.4e-08 of the day-30 gap per step, so this statement's year carry cannot be
ruled out from the momentum family alone.

## Landing

Nothing landed, and the full Decision 43/45 gate (the month, day 240, day 360,
the kt2 temperature and salinity bar, first-over-bar, every moved row with the
noise floor quoted, DINO, the generic NEMO-GYRE card, the LOCK_EXCHANGE and
OVERFLOW tanks) was NOT run, because there is no candidate arm to run it on.
The arm this round used is a measurement arm: it changes the shared field, and
the tracer measurement above shows that is not the fix.

The campaign headline rows are inherited and unmodified: kt2 temperature and
salinity at the bar, kt2 U/V about 2.7377e-12 / 3.2849e-12, kt3 temperature
about 8.60e-07 K, day-30 6.888194e-05 K, day-240 1.644671864e-02 K, day-360
1.122345086e-02 K.  ORCA2 is **UNMEASURED-WITH-SPEC**: repeat this walk on the
ocean-only ORCA2 card before transferring the verdict.

## Plants, review and tests

Two plants, each printing its own marker and exiting nonzero:

* `wzv-cell-window` shifts the oracle's dry-column pattern by one column and
  requires the cell-window control to refuse it.  It does: 40 columns
  disagree, and the walk prints `STATUS PLANT-FIRED: wzv-cell-window` and
  exits 1.  Forty is the same count round 158 measured for the nearest
  candidate window, which is independent evidence that the control
  discriminates rather than merely fires.
* `wzv-form-inert` claims the arm is on while leaving it off, and requires the
  liveness control to refuse a run that never selected the other call form.
  It does: 0 of 18,000 interfaces move, and the walk prints
  `STATUS PLANT-FIRED: wzv-form-inert` and exits 1.

Three new tests, each shown to FAIL when the guard it checks is removed: a
continuity-form arm that is not a plain boolean is refused at construction,
both call forms are selectable, and the production default is the transport
form.  With the guard reverted the file reports `6 failed, 3 passed`; with it
in place, `9 passed`.

Adding the arm lengthened the model file by eighteen lines in three places, so
thirty-six map entries and twenty-four prose citations across four earlier
receipts were re-anchored by RIGID shift: both endpoints of each citation
moved by one delta, every pinned extent unchanged, every endpoint symbol
re-resolved in the current file.  No citation was weakened, widened or
removed.  **One citation is registered rather than moved.**  It is fenced
because it is a report of a stale citation, not a citation this receipt makes:

```
ocean_model_latlon_cgrid.py:1273-1325   (round-8 receipt, NOT re-anchored)
```

That range straddles this round's insertion point, so the block it names
genuinely grew and no rigid shift exists for it; widening its extent is what
the gate's own plant does, so it is reported here instead of papered over.  It
is not in the gate's map and was already carrying an older drift before this
round.

Independent review by codex not run (codex is paused).  A Claude reviewer was
run instead, on the whole round's diff and the compiled sources.

