# NEMO testcase L2 GYRE Round 158 receipt — developed stage-2 advection split

Date: 2026-09-23
Status: **HELD** — no physics, no configuration, no carried state changed.

The momentum advection round 157 named is two compiled routines, and they do
not share the blame.  The kinetic-energy gradient is right to five parts in a
billion of its own size.  The vertical advection carries the whole difference,
and inside it the owner is not a statement at all: it is the VERTICAL VELOCITY
the routine is handed.  Installing NEMO's own recorded vertical velocity at
that one call, with every other stage input left at legoESM's value, removes
99.71% of the eastward and 99.80% of the northward stage-2 right-hand-side
difference and leaves exactly the vorticity floor round 157 measured.  So the
advection is exonerated and the walk moves upstream, to where legoESM builds
that vertical velocity.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round158.md`, committed as
`beeac4f76` before any legoESM stage-2 advection measurement ran.  The
authoritative measurement is `adv_split.json` under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round158/walk/`, produced on
a clean tree at `c950f028b`, the tip after the independent review's findings
were closed; the same rows were produced identically at `a6cff1c3a` before
them.  No card, no scheme selection, no
tunable and no carried field changed; the two model edits are private
diagnostic seams that no public configuration can construct and no production
path reads.

## The compiled program

The record's own configuration namelist sets `ln_dynadv_vec = .true.` with
`nn_dynkeg = 0`, and the run's own `ocean.output` reports `ln_zad_Aimp = F`
(both under `round157/oracle_developed_stage2/`), so the compiled selector at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:153` takes the centred
vector case and runs exactly two routines into one accumulator.

| half | call | statements | what it reads |
|---|---|---|---|
| kinetic-energy gradient | `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:171` | `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90:121-125` builds the T-point energy; `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90:129-130` subtracts its gradient | the stage velocity and two static metrics |
| vertical advection | `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:176` | `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:112-116` forms the transport, `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:119` the shear product, `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:123-126` the update and `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:134-137` the bottom level, with the surface term zeroed at `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:102` | the VERTICAL VELOCITY, the stage velocity, the live thickness and the static metrics |

The vertical velocity the record carries is the one this routine consumes:
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:495` records it
after the stage's continuity solve and before the pressure gradient, and
nothing between there and the advection call writes it.

## Controls, before any attribution

* **Face windows.** NEMO's recorded before-level velocity is the step-entry
  level, which this walk loads from the same restart.  It is BIT on both
  components, 0 of 21,780 and 0 of 22,080 cells.
* **Cell window.** The vertical velocity is a cell-centre field, so the face
  control does not cover it.  The property that pins the cell window is the
  DRY-COLUMN PATTERN: NEMO's density anomaly is identically zero on every dry
  column, NEMO and legoESM each have 104 of them — the basin's boundary ring
  — and 0 disagree.  The review enumerated all 25 candidate windows of this
  shape and exactly one satisfies that; its nearest neighbours disagree on
  40, 60 and 98 columns.  **The first cell window this walk tried was one of
  those neighbours and this control caught it, at 98.  It is registered
  rather than quietly fixed, because it is the reason the control exists.**
  The companion row, that the vertical velocity is zero at the bottom
  interface, reads zero for EVERY candidate window — the oracle zeroes it
  architecturally — so it pins nothing, and it is labelled
  non-discriminating in the report rather than counted as a control.
* **Passivity.** Each exposure substitutes its own slots only after the
  ordinary step completes; every other prognostic field is byte-identical to
  an ordinary step, 0 bytes on all five passive runs.
* **Authority.** The walk refuses to report unless its own production
  after-advection row reproduces the round-157 row it is splitting.  It does,
  at 3.844166e-12 (u) and 6.428546e-12 (v).
* **The split adds no term.** legoESM's two halves sum to the advection
  bucket they came out of with 0 of 17,400 and 0 of 17,100 active faces
  unequal.
* **The substitution reaches only the routine it names.** The kinetic-energy
  gradient reads no vertical velocity, so its row under the substitution has
  to be byte-identical to production: 0 of 21,780 and 0 of 22,080 cells.
* **The substituted operand is live.** NEMO's vertical velocity is not
  legoESM's, so a substitution that reached the routine must have moved its
  output; every active face of the vertical-advection term moves.

## The split

Every arm is one production step from NEMO's admitted day-180 entry through
`LatLonCGridOceanModel.step` under production just-in-time compilation, never
an isolated closure.  Scores are over NEMO's own active faces.

NEMO's own advection increment is the difference of its two recorded
cumulative snapshots and carries one rounding on the oracle side; it is
context for the split, never a scored production row.  The kinetic-energy
half is evaluated on NEMO's own operands by driving the production stage with
NEMO's recorded entry velocity, which is the only dynamic operand that half
reads; the vertical half is then NEMO's increment minus it.

**What the first row pair of the table below is, exactly.**  The record has
no boundary between the two halves, so NEMO's own kinetic-energy output does
not exist as a measured array.  The kinetic-energy row is therefore legoESM's
OWN operator evaluated twice, once on its own operands and once on NEMO's; it
is an OPERAND SENSITIVITY, not a NEMO-versus-legoESM fidelity number.  The
review named the consequence and it is registered here rather than argued
away: if legoESM's kinetic-energy gradient carried a systematic transcription
error, that error would be invisible in this row and would land entirely on
the vertical row — a 4.5e-04 relative error in that half would reproduce this
table exactly.  What rules that out is the LAST arm of the next table, which
does not rest on this construction at all: substituting only the vertical
velocity removes 99.7% of the difference, which no error in the
kinetic-energy gradient can produce.

| half | its own size (rms) | difference (rms) | relative | difference (max) |
|---|---:|---:|---:|---:|
| kinetic-energy gradient, u | 8.593696e-09 | 4.207387e-17 | 4.896e-09 | 8.299566e-16 |
| kinetic-energy gradient, v | 5.275828e-09 | 3.778976e-17 | 7.163e-09 | 6.108160e-16 |
| vertical advection, u | 1.533552e-09 | 3.844051e-12 | 2.507e-03 | 1.828764e-10 |
| vertical advection, v | 1.198423e-09 | 6.429683e-12 | 5.365e-03 | 3.026392e-10 |

The vertical half carries 91,364 times the eastward difference of the
kinetic-energy half and 170,144 times the northward one.  The two add back to
the total the walk is splitting, to 3.1e-05 (u) and 1.8e-04 (v) relative,
which is the after-vorticity rounding the total also carries.

The kinetic-energy gradient's 5e-09 relative is the compiled-rounding floor
this campaign has measured many times (operator notes L and AL); the vertical
advection's 2.5e-03 is not a floor.

## Which operand, measured

Four arms against NEMO's own recorded stage-2 right-hand-side total.  The
null is legoESM's own entry bundle fed back through the substitution
mechanism, so the mechanism's cost is in both arms and the comparison stays
one variable; its measured offset on the kinetic-energy half is exactly zero.

| arm | u rms | removed | v rms | removed |
|---|---:|---:|---:|---:|
| production | 3.844166e-12 | — | 6.428546e-12 | — |
| null (legoESM's own entry) | 3.844166e-12 | 0.000% | 6.428546e-12 | 0.000% |
| NEMO's entry velocity | 3.844036e-12 | +0.0034% | 6.429669e-12 | −0.017% |
| **NEMO's vertical velocity** | **1.121563e-14** | **99.708%** | **1.308047e-14** | **99.797%** |

The residue the vertical-velocity substitution leaves, 1.121563e-14 (u) and
1.308047e-14 (v), is the after-vorticity row round 157 measured on the same
state, 1.120967e-14 and 1.307121e-14, to within 0.05% and 0.07% — at that
floor, not exactly on it.  Once the routine is handed NEMO's own vertical
velocity, the whole stage-2 right-hand side sits at the floor and the
advection contributes nothing measurable.

**The closure that makes the split non-circular**, the one measurement the
review asked for: the vertical half evaluated on NEMO's OWN vertical velocity,
scored face by face against the vertical increment the split implies, which is
NEMO's total advection increment minus the kinetic-energy half.

| component | closure rms | closure max | as a fraction of the difference being split |
|---|---:|---:|---:|
| u | 3.858218e-22 | 1.537894e-20 | 1.0e-10 |
| v | 2.084387e-22 | 7.404921e-21 | 3.2e-11 |

That closes it.  Any systematic error in legoESM's kinetic-energy gradient
would appear here, because the implied vertical increment is built from that
half; the closure bounds such an error at about 4e-22, ten orders below the
3.844051e-12 the split attributes.  So the kinetic-energy half is not hiding
the difference, and neither is the vertical half's arithmetic: what the
vertical half is HANDED is.

## Predictions and verdict

1. The transcription is exact given NEMO's operands: **CONFIRMED in the form
   that could be measured**.  The kinetic-energy half sits at 5e-09 relative
   on NEMO's own entry velocity, and the vertical half is exact to the floor
   once its vertical velocity is NEMO's.  Registered as a weakening of the
   prediction: the record has no boundary between the two halves, so the
   rebuilt SUM the preregistration named was not directly scorable; the two
   directed substitutions measure the same thing one operand at a time.
2. The vertical advection owns the magnitude, by at least a factor of two:
   **CONFIRMED**, by a factor of 91,364 (u) and 170,144 (v).
3. The split accounts for the whole difference to better than one percent:
   **CONFIRMED**, to 3.1e-05 (u) and 1.8e-04 (v).
4. The two halves sum to the bucket: **CONFIRMED**, 0 active faces unequal.
5. A single operand removes at least half of the named half's difference:
   **CONFIRMED**, one operand removes 99.7% and 99.8%.
6. The plants fire: **CONFIRMED**, both, see below.

## First non-bit operand that carries magnitude

The vertical velocity read at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:112-116`, entered
through `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:176` under the
vector case selected at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:153`.

What it carries at day 180: 99.71% (u) and 99.80% (v) of the 3.844166e-12 /
6.428546e-12 m/s^2 stage-2 right-hand-side difference, hence essentially all
of the 1.309260e-06 m/s stage-2 velocity difference round 157 measured and the
95.2% of the stage-3 transport difference round 155 attributed to it.

What it carries at day 240 is NOT measured, because there is no candidate to
score: the fix is not a statement inside the advection but whatever legoESM
does differently when it BUILDS that vertical velocity, and that has its own
walk.  The honest bound from what is already measured: round 134 found
velocity resets remove 6.5% of the day-240 gap, and round 149's momentum
landing moved day 240 by 2.2e-08 K on 1.6447e-02 K, which is within 1e-06
relative and near the harness's own run-to-run floor of about 2e-10 K (round
129).  That bound is read off earlier measurements, labelled **PLAUSIBLE**,
not a measurement of this operand.

## Landing

Nothing landed.  No single source-exact statement was proven — the measurement
NAMES AN OPERAND, not a statement — so the full Decision 43/45 gate (the
month, day 240, day 360, the kt2 T/S bar, first-over-bar, every moved row with
the noise floor quoted, DINO, the generic NEMO-GYRE card, the LOCK_EXCHANGE
and OVERFLOW tanks) has no candidate arm to run on and was not run.  The
campaign headline rows are inherited and unmodified: kt2 T/S at the bar, kt2
U/V about 2.7377e-12 / 3.2849e-12, kt3 T about 8.60e-07 K, day-30 6.888194e-05
K, day-240 1.644671864e-02 K, day-360 1.122345086e-02 K.  ORCA2 is
**UNMEASURED-WITH-SPEC**: repeat this split on the ocean-only ORCA2 card
before transferring the verdict.

## Plants, review and tests

Two plants, each printing its own marker and exiting nonzero:

* `keg-zad-sum` scales legoESM's two halves, after they are summed, by one
  part in a million million.  The split control has to catch a sum that is
  not the bucket it claims to be, and it does: 17,400 of 17,400 eastward and
  17,100 of 17,100 northward active faces go unequal, at a maximum of
  `2.441572e-19`.  It prints `STATUS PLANT-FIRED: keg-zad-sum` and exits 1.
* `nemo-ww-ulp` installs NEMO's recorded vertical velocity a second time with
  its largest interior cell scaled by one part in a million million, and
  requires the stage's vertical advection to move with it.  It does: the moved
  cell is row 16, column 30, interface 13, and 2 eastward and 4 northward
  active faces change, at `2.444319e-22` and `2.786565e-22`.  It prints
  `STATUS PLANT-FIRED: nemo-ww-ulp` and exits 1.

  Registered rather than quietly fixed, TWICE.  The FIRST version moved the
  first non-zero cell of the recorded vertical velocity, which is the SURFACE
  interface; the routine reads only the interior ones, so nothing consumed the
  moved value and the plant announced itself for no reason at all.  The SECOND
  moved one interior cell by a single unit in the last place and reached
  exactly one eastward face at `5.17e-26` — the review called that no margin,
  and the scaling above is the answer.  The plant's claim is the narrow one it
  measures: the operand is read.  The broad one is the ordinary run's own
  refusal, which requires every one of the 17,400 eastward and 17,100
  northward active faces to move when the vertical velocity is substituted.

Four new tests, each shown to FAIL when the guard it checks is removed: the
two advection halves are selectable by name, an unknown operator name is
refused, a vertical-operand substitution that is not a three-slot tuple is
refused, and an all-live triple is accepted.  With both guards reverted the
file reports `7 failed, 2 passed`; with them in place, `9 passed`.

The citation gate on this receipt reported `PASS` with 11 citations, zero
failures, zero unmapped citations, zero map entries failing audit and every
self-test fired.  Its shifted-line plant on the vertical-transport citation
exited 1 with `SYMBOL-NOT-AT-LINE`.

Adding the two diagnostic seams lengthened the model file and moved every
citation below them, so thirty-six map entries and twenty prose citations
across four earlier receipts were re-anchored by RIGID shift: both endpoints
of each citation moved by one delta, every pinned extent unchanged, every
endpoint symbol re-resolved in the current file.  No citation was weakened,
widened or removed.

The six-file push gate plus this round's own tests were run together on the
committed tree and reported exactly:

> 144 passed in 522.38s (0:08:42)

Independent review by codex not run (codex paused).  A Claude reviewer was
run instead, on the whole diff and the compiled sources, and its verdict was
**SHIP WITH CHANGES**, verbatim:

> I could not break the headline conclusion. The transcription is exact, the
> record point is right, and the cell window is unique. Three receipt claims
> overstate what was measured and two controls are not fail-closed.

Its findings and what was done with each:

1. HIGH, the kinetic-energy row is circular — a systematic error in
   legoESM's own kinetic-energy gradient would be invisible in it and would
   land on the vertical row.  CLOSED BY MEASUREMENT: the closure above bounds
   such an error at 4e-22, and the receipt now says plainly what that row is.
2. MEDIUM, two checks the receipt called controls were never asserted.
   FIXED: both are refusals now.
3. MEDIUM, the bottom-interface half of the cell-window control reads zero
   for every candidate window and pins nothing.  FIXED: it is labelled
   non-discriminating and the refusal rests on the dry-column pattern, which
   the review verified is satisfied by exactly one window of 25.
4. MEDIUM, the vertical-velocity plant had no margin.  FIXED, see above.
5. LOW, the exposure of a half would silently omit a flux-form card's own
   vertical term.  FIXED: it is refused outright on such a card.
6. LOW, "exactly the vorticity floor" overstates 1.0005x.  FIXED in the
   prose.
7. LOW, the construction guard checks the substitution triple's arity but not
   its order.  NOT FIXED and registered: the three slots have different
   shapes, so a transposed triple fails downstream rather than at
   construction.

The review also independently reproduced legoESM's vertical-advection kernel
against a literal Fortran-order transcription of the compiled routine and
found a maximum relative difference of 1.4e-16, one unit in the last place,
covering the loop bounds, the separate bottom statement, the interface
alignment, the shear sign and the sum-versus-twice-the-mean factor.  That is
independent support for this round's conclusion that the statement is not the
owner.

## OPEN — Round 159

1. **Walk legoESM's stage-2 vertical velocity, not the advection.**  The
   advection is exonerated by measurement: given NEMO's own vertical velocity
   the whole stage-2 right-hand side sits at the vorticity floor.  The owner
   is upstream, in the stage's continuity solve — the routine that produces
   the vertical velocity from the stage transport and the free-surface
   tendency.  Score legoESM's stage-2 vertical velocity directly against
   NEMO's recorded one, cell by cell and level by level, and then walk that
   producer's compiled statements in order.
2. **Calibrate before naming, as here.**  The record carries NEMO's vertical
   velocity on the cell grid, its free-surface pair and its thickness ratios
   at this stage, so each statement of the producer can be driven from NEMO's
   own operands before any candidate is proposed.
3. **Expect an operand, not a statement.**  Two rounds running, the named
   routine turned out to be exact on NEMO's operands and the difference came
   in through what it was handed.  Round 159 should measure the producer's
   INPUTS first — the stage transport and the free-surface tendency — before
   walking its arithmetic.
4. **Do not walk the floor.**  The kinetic-energy gradient at 5e-09 relative,
   the vorticity at 1e-08 and the pressure gradient at 1e-11 are the
   compiled-rounding floor (notes L and AL).
5. **Rank at day 240, quote the floor.**  If round 159 names a statement whose
   day-240 carry is within the harness's own run-to-run floor of about 2e-10
   K, say so and put the question to the user rather than landing it as an
   improvement.
