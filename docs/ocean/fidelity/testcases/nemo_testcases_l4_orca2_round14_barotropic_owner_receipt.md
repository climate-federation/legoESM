# NEMO testcase Lane 4 — ORCA2 card round 14 attribution receipt

Date: 2026-09-24

Starting tip: `0501c06f425aba4a5a4f3628e9e6f380d52f5470`.

Preregistration: `d453727c3`

Status: **HELD — nothing landed, and the record says nothing should have.**

Round 13's largest open row was the end-of-step sea surface, 0.2448 m at
kt=1, with its owner labelled PLAUSIBLE rather than named.  Round 14 ran the
discriminating substitution round 13 specified and the answer is clean:

**Feeding legoESM's barotropic solver NEMO's OWN recorded slow forcing, over
the 90 longitude columns the record covers, moves the end-of-step sea surface
by at most 2.7614e-07 m.  The disagreement it would have to close is
0.2448 m.  That is ONE PART IN 886,000.  The momentum forcing the solver
RECEIVES is exonerated; what is left is the solver itself and the two of its
inputs that this round could not substitute.**

The forcing side was walked anyway, because the same record carries its
ordered intermediates, and it produced a second, smaller, genuinely new
finding: **every structural disagreement in the meridional forcing operands
sits on ONE row — the last one, which on this tripolar grid is the fold.**
legoESM's three-dimensional meridional face mask disagrees with NEMO's
`vmask` on 668 cells, all of them on that row and on no other; the meridional
wind stress disagrees on 35 cells, all on that row.  Away from the fold the
only differences are a thickness CONVENTION that provably cancels, and
rounding.

The sea-surface disagreement is **not** a fold artefact and this receipt says
so with the measurement rather than the impression: it stands on 8,794 of
13,320 cells across 147 of 148 rows, and with the fold row excluded entirely
it is still 0.2126 m.  **Its single worst cell is nevertheless ON the fold
row** — row 147, column 38 — and the review was right to insist that be said
out loud rather than left in a JSON field.  The honest statement is therefore
BOTH: the disagreement is basin-wide, and its peak sits on the one row that
also carries this round's mask defect.

Two things this round found in its OWN instrument and corrected before
reporting anything are in section 3.  One of them would have produced the same
headline for the wrong reason.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  **No file under `packages/` was touched at all**, so GYRE cannot
have moved (section 7).  Sea ice remains out of scope and the six-entry
`unmeasured_features` tuple is unchanged.

Independent review **not run (codex is paused)**; a fresh adversarial reviewer
was run in its place (section 8).

## 1. What the record fixes, and what nothing here may choose

Read from the card's RESOLVED configuration and from the compiled build that
produced the record, not from a deck comment.

| resolved setting | value | where it is resolved |
|---|---|---|
| barotropic solver | split-explicit sub-stepping, 65 sub-steps | card `barotropic_solver = explicit_substep`, `n_barotropic_substeps = 65` |
| barotropic time filter | `nemo_ab3am4` | card `barotropic_time_filter` |
| barotropic Coriolis | `een_metric`, split `live` | card `barotropic_coriolis`, `barotropic_coriolis_split` |
| planetary Coriolis in the 3-D right-hand side | `explicit_ab2` | card `coriolis_scheme` |
| barotropic bottom-drag correction | ON | card `barotropic_drag_substep` |
| surface stress into the barotropic forcing | ON | card `surface_stress_implicit` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output` line 217 |
| the record's slow forcing | **RANK 0 ONLY** | record census: no `oracle_slow_forcing_rank0001_*` file exists |

The last row was preregistered as a hard constraint, before measuring, so it
could not be discovered afterwards and quietly absorbed.  Every substituted
number below is reported over the whole rank-0 half AND over its interior,
with the gravity-wave contamination margin measured on the card's own metric.
**No new NEMO run was made or needed.**

## 2. Statement A — the substitution, and the number it produces

### 2a. What NEMO does, in the order the record writes it

| # | what NEMO does | compiled owner |
|---|---|---|
| A1 | evaluates the 3-D right-hand side ONCE at the before level — pressure gradient, lateral viscosity, then vorticity | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:139-147` |
| A2 | adds the kinetic-energy gradient and the vertical advection | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:162-166` |
| A3 | depth-averages it with the REFERENCE-thickness reciprocal `r1_hu_0`, not a live one | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:196-199` |
| A4 | adds `dyn_drg_init`'s baroclinic-residual bottom drag, and records the drag coefficient with it | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:218-221` |
| A5 | adds the wind, `r1_rho0 * utauU * r1_hu_0/(1+r3u)` | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:229-230` |
| A6 | hands the result to the external solver | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:302-303` |
| A7 | the solver copies it, and the sea-surface forcing and drag coefficients with it | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:287-291` |
| A8 | removes the 2-D Coriolis trend from it | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:320-324` |
| A9 | adds it to every one of the 65 sub-steps, in the vector-form arm | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:668-671` |

The record's write statements sit BETWEEN those steps, which is what makes it
an ordered ladder rather than a single snapshot.  The value after A5 is the
one substituted here, and A8 is where legoESM's substitution hook lands — the
same boundary, in the same order.

### 2b. Rule 4 — what was searched before anything was written

Searched `slow_forcing`, `oracle_slow_forcing`, `slow_forcing_incoming_override`,
`barotropic_slow_forcing_override`, `post_wind`, `post_drag` and `Ue_rhs`
across `packages/`, `scripts/validate/ocean_fidelity/` and `tests/`.  Five
things already existed and were REUSED rather than rebuilt:

| for | what already existed |
|---|---|
| reading the record | `read_slow_forcing` in the GYRE round-16 script, which already parses this exact layout with explicit size and EOF checks |
| comparing candidate to oracle | `compare` in the same module |
| the substitution point | `_NEMOWSRK3TestHooks.slow_forcing_incoming_override`, which already lands exactly where
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:320-324` removes the 2-D Coriolis |
| legoESM's own ordered intermediates | the production step already exposes them through `expose_barotropic_substeps` and `expose_live_stage_operands` |
| the card, entry state, recorded surface operands and stage frames | the round-1 ORCA2 ladder gate |

**Statement A owed NO new model statement and no new operator.**  It owed one
new gate and ONE extension to an existing reader: a grid-extent argument,
because the reader's only GYRE-specific part was a hard-coded extent.  Every
existing caller keeps the old default, and a test requires the default to
REFUSE the ORCA2 record, so the extension is not vacuous.

### 2c. The measurement

Label: **given NEMO's entry** — the kt=1 recorded state, the recorded surface
frames and the recorded slow forcing.

| row | result |
|---|---|
| baseline end-of-step sea surface, against the record | **0.2448430937728719 m**, 8,794 / 13,320 cells |
| round 13's published value | 0.24484 m — **reproduced**, relative departure 1.26e-05 |
| **the substitution moves the end-of-step sea surface by** | **2.7614e-07 m**, on 8,794 of 13,320 cells |
| as a share of the disagreement it would have to close | **1.13e-06** |
| verdict | `THE_FORCING_IS_NOT_THE_OWNER` |

Preregistered threshold: a movement above 2.4e-02 m would have made the
forcing a real contributor.  **R14-P3 is REFUTED**, and that refutation is the
round's result.

### 2d. Four controls, each of which EXECUTED and PRINTED its own line

| control | what it printed |
|---|---|
| is the substitution a ONE-variable arm? | "entry operands other than the sea surface are already bit-identical: T True, S True, u True, v True" — so seeding them changes nothing and only the Decision-52 sea surface is bridged |
| is the hook inert on legoESM's own operand? | "the substitution hook fed legoESM's own forcing leaves the end-of-step sea surface bit-identical: True" |
| did the injection land where it was aimed? | "the substitution landed where it was aimed: on the window True, off the window legoESM's own True" |
| can the reader and the transcription be trusted at all? | "the record's own operands replay its own depth mean: u true, v true" — NEMO's written sum, replayed from NEMO's written operands, is bit-exact on 8,568 and 8,589 faces |
| **the plant** | "**PLANT FIRED**: the gate refuses a one-representable-value move in the injected slow forcing — it changes the end-of-step sea surface on 84 of 13320 cells, max 1.665335e-16 m" |

The plant is the one that makes 2c a finding rather than an artefact.  "The
forcing barely moves the sea surface" and "the channel is deaf" produce the
same headline, and only the plant separates them: one unit in the last place,
on one injected value, reaches the end-of-step sea surface.

### 2e. Half a domain, said out loud

The record's slow forcing exists for rank 0 only, so 90 of 180 longitude
columns keep legoESM's own forcing and the substituted half is contaminated
inward from both of its periodic edges.

| row | value |
|---|---|
| external gravity-wave speed on this bathymetry | 232.24 m/s |
| how far it travels in one baroclinic step | 2,508 km |
| that distance in columns, at the row carrying the maximum | 29 |
| interior columns after the margin | 29 to 61 |
| baseline maximum over that interior | 0.2448430937728719 m |
| substituted maximum over that interior | 0.24484308140827876 m |

**R14-P7 is REFUTED, and it does not matter.**  The prediction was that the
residual would be concentrated near the rank boundaries so the interior number
would be smaller.  It is not: the interior maximum equals the whole-half
maximum to nine figures, because the worst cell is interior.  The contamination
argument was a precaution against a movement that never happened — the
substitution moves nothing anywhere, so there is no contaminated number to
discount.

## 3. What this round found wrong in its own instrument

Both were found by running the gate and reading the result, not by reasoning,
and both are recorded because either one alone would have produced a confident
wrong receipt.

**The first version measured the difference of two MAXIMA.**  It reported that
the substitution changed the sea surface by 1.2e-08 m.  That number is the
difference between the two arms' maxima against the oracle, and a difference
of maxima can sit still while the field moves somewhere else entirely.  The
measure is now the maximum of the DIFFERENCE between the two arms' own sea
surfaces, which is what "the substitution moved it" means.  The verdict is
read from that.  It happens to point the same way — 2.76e-07 m rather than
1.24e-08 m, both negligible — but that was luck, not method.

**The first plant could not fail.**  It put one unit in the last place into
the candidate face thickness and required that to become the ladder's first
non-bit boundary.  The face thickness already WAS the first non-bit boundary,
so the plant fired by construction.  It is replaced by the plant in section 2d,
which is the one this round's conclusion actually needs.

A third correction, smaller: the first run refused on a shape mismatch because
the gate reached for the model's two-dimensional surface face mask where
NEMO's vertical sum multiplies a three-dimensional one.  The refusal was
correct behaviour and is recorded as such.

## 4. Statement B — the forcing's own boundary ladder

NEMO applies the bottom drag BEFORE the wind; legoESM applies the wind before
the drag.  The two are additive on the same depth mean, so the gate compares
the INCREMENTS, not the intermediates.  That was preregistered: a boundary
compared across two different orders is a confound, not a result.

All rows on rank 0's own columns, on the record's own mask.  Label: **given
NEMO's entry**.

| boundary | U face | V face |
|---|---|---|
| face thickness `e3` | 3.6310e-02 m on 226,236 / 226,236 | **437.20 m** on 226,637 / 226,637 |
| 3-D right-hand side `Krhs` | 5.9307e-17 on 226,187 / 226,236 | 2.3620e-06 on 226,564 / 226,637 |
| face mask | **bit-exact**, 0 / 399,600 | **668 / 399,600** |
| depth reciprocal `r1_h0` | 1.2143e-04 on 8,568 / 8,568 | 2.9711e-03 on 8,589 / 8,589 |
| **depth mean** | **2.5135e-17** on 8,568 / 8,568 | 1.6553e-06 on 8,589 / 8,589 |
| drag increment | **bit-exact**, 0 / 8,568 | **bit-exact**, 0 / 8,589 |
| wind stress `utauU`/`vtauV` | **bit-exact**, 0 / 8,568 | 0.3203 Pa on 35 / 8,589 |
| wind increment | 5.3701e-12 on 2,571 / 8,568 | 5.4887e-07 on 8,589 / 8,589 |
| **final value handed to the solver** | 5.3701e-12 on 8,568 / 8,568 | 1.6561e-06 on 8,589 / 8,589 |

### 4a. The thickness convention is real, and it CANCELS

`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:196-199`
weights the sum with `e3u_3d`, the REFERENCE u-face scale
factor — the same array
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domain.f90:199` sums into
`hu_0` — and divides by
`r1_hu_0`, the reference depth reciprocal.  legoESM weights with its LIVE
face thickness and divides by the live column depth.  Both operands differ:
the thickness by 3.63e-02 m on every one of 226,236 U faces, the reciprocal by
1.21e-04 on every one of 8,568 U columns.

**And the depth mean they produce agrees to 2.51e-17.**  Under the quasi-
Eulerian coordinate the live thickness is the reference thickness times a
column factor that does not depend on depth, so that factor divides out
between the weights and the divisor.  **R14-P6 is CONFIRMED and VACUOUS**: the
two spellings are genuinely different operands and produce the same number.
It is reported as vacuous rather than as a finding.

### 4b. The meridional face on the tripolar fold row, which is NOT vacuous

Every structural V-face row above is ONE row, and it is the last one.

| row | differing cells | rows carrying any difference | where its maximum is |
|---|---|---|---|
| V face mask | 668 / 399,600 | **1 of 148** | row 147, column 29, level 0 |
| V wind stress | 35 / 8,589 | **1 of 148** | row 147, column 41 |
| V face thickness, its maximum | — | 147 of 148 | **row 147**, column 51, level 24 |
| V right-hand side, its maximum | — | 147 of 148 | **row 147**, column 55, level 23 |
| V depth mean / final, their maxima | — | 147 of 148 | **row 147** |

The U face has no such row: its mask and its wind stress are bit-exact
everywhere, and its own maxima are in the interior.

So: **legoESM's three-dimensional meridional face mask disagrees with NEMO's
`vmask` on 668 cells, all of them on the northern fold row and on no other
row, and the meridional wind stress disagrees on 35 cells of that same row.**
The face thickness gap of 437 m at row 147 is the visible consequence: on a
face NEMO calls wet and legoESM calls dry, the whole of NEMO's thickness shows
up as the difference.

One qualification the gate measures rather than assumes: the mask that the
vertical sum EFFECTIVELY applies is the support of legoESM's own face
thickness, not a separately stored array, and that support differs from NEMO's
`vmask` on **15** cells, not 668.  So the 668-cell mask row overstates what
this particular sum sees; the 15 is the number that bounds THIS statement,
and the 668 is the number that matters for any consumer that reads the face
mask directly.  Both are reported.

**And the fold row is also where the sea surface is worst, which is a
coincidence this receipt is not allowed to leave unremarked.**  The end-of-step
disagreement's own maximum is at row 147, column 38.  The fold-row mask cannot
OWN that peak through the forcing channel, because the substitution in section
2c bounds the entire momentum forcing — mask defect included — at 2.76e-07 m.
But the same three-dimensional face mask is read by consumers other than this
vertical sum, and nothing in this round measured those.  That is stated as a
limit on the attribution, not waved away, and it is round 15's first stop
after the solver's two unchecked inputs.

### 4c. Which half of the depth mean owns its disagreement

Three one-variable arms, each replaying NEMO's own written statement with
exactly ONE operand family swapped for legoESM's.  No model run; no other
operand moves.

| arm | U face | V face |
|---|---|---|
| NEMO's own operands, replayed (the instrument control) | **bit-exact** | **bit-exact** |
| only the 3-D right-hand side is legoESM's | 2.5135e-17 | 1.6553e-06 |
| only the metric is legoESM's | **6.7763e-21** | 1.6553e-06 |
| both are legoESM's | 2.5135e-17 | 1.6553e-06 |

On the U face the metric contributes four orders of magnitude less than the
right-hand side, which is the same statement as 4a from the other side.  On
the V face both arms return the whole disagreement, because on the fold row
the mask and the right-hand side are wrong together and neither can be
isolated from the other by substitution.

### 4d. What kt=1 can and cannot test, stated rather than assumed

**The kt=1 entry velocity is exactly zero on every cell, both components.**
The gate prints it.  So at this step the vorticity term, the lateral
viscosity, the kinetic-energy gradient and the vertical advection are all
identically zero, and the right-hand side `Krhs` is the pressure gradient
alone.  The U-face `Krhs` row above, 5.93e-17, therefore certifies the
PRESSURE GRADIENT and says NOTHING about the four velocity-dependent
operators — including the lateral viscosity whose scoping is Decision 54.
Any attempt to read this round's `Krhs` row as evidence about that operator is
a category error, and it is flagged here so that no later round does it.

## 5. What the ladder says, unchanged

Still `LADDER_MEASURED`, kt = 1 to 10, exit 0.  Nothing in this round touched
a model file, so no row could move, and none did.

| row | round 13 | round 14 |
|---|---|---|
| first non-bit statement | kt=1 stage 1, temperature, UNATTRIBUTED | **identical** |
| kt=10 entry temperature | 3.943079 on 430,552 | **3.9430791763114783 on 430,552** |
| kt=1 stage-1 sea surface | 8.1614e-02 m | **8.1614e-02 m** |
| kt=1 end-of-step sea surface | 2.4484e-01 m | **2.4484e-01 m** |

Round 13's own stage-1 owner probe, re-run at this round's tip, reproduces
every published number EXACTLY: stage maxima 8.1614e-02 / 1.2242e-01 /
2.4484e-01 m, median spread over column mean 1.699e-10 (zonal) and 1.649e-10
(meridional), 6,897 and 7,120 columns under one percent.  That is the
instrument control for the comparison in section 2c.

## 6. Frozen predictions, resolved

| ID | outcome |
|---|---|
| R14-P1 | **CONFIRMED** — 0.2448430937728719 m against round 13's 0.24484, relative departure 1.26e-05. |
| R14-P2 | **CONFIRMED** — the hook fed legoESM's own forcing leaves the end-of-step sea surface bit-identical. |
| R14-P3 | **REFUTED, and this is the round's result** — the substitution moves the sea surface by 2.7614e-07 m, not the 2.4e-02 m threshold. The forcing is not the owner. |
| R14-P4 | **CONFIRMED** — legoESM's final slow forcing is not bit-exact: 5.37e-12 on 8,568 U columns, 1.66e-06 on 8,589 V columns. It differs, and the difference does not matter. |
| R14-P5 | **CONFIRMED** — the first non-bit boundary is the face thickness, at or before the depth mean; neither increment created it. The drag increment is bit-exact on both faces. |
| R14-P6 | **CONFIRMED but VACUOUS** — the two divisors differ on every column, and the depth mean they produce agrees to 2.51e-17 (section 4a). |
| R14-P7 | **REFUTED and immaterial** — the residual is not concentrated at the rank boundaries; the interior maximum equals the whole-half maximum, because nothing moved anywhere (section 2e). |
| R14-P8 | **CONFIRMED** — `LADDER_MEASURED` kt=1..10, same first statement, same kt=10 magnitude. |
| R14-P9 | **CONFIRMED** — no file under `packages/` changed (section 7). |

## 7. GYRE cannot have moved, and here is why that is not a promise

`git diff --name-only 0501c06f4..HEAD -- packages/` lists **nothing**.  Every
commit in this round is a preregistration, a gate, a test, a citation-map
entry or this receipt.  The one edit to shared code is in a GYRE VALIDATION
SCRIPT, not in the model: `read_slow_forcing` gained a grid-extent argument
whose default is GYRE's own extent, so its three existing callers are
character-for-character unchanged in behaviour, and a test asserts both that
the default is still GYRE's and that it REFUSES the ORCA2 record.

No trajectory comparison is quoted, because running one would measure the
absence of a diff rather than prove anything about it.

**Which cards execute the changed code.**  None: no card executes a validation
script.  The ORCA2, GYRE, DINO, LOCK_EXCHANGE and OVERFLOW cards all run
exactly the code they ran at `0501c06f4`.

## 8. Independent review

Codex is paused, so `codex exec` was NOT run and this round claims no
independent codex verdict.  A fresh adversarial reviewer was run in its place.
It read the compiled Fortran itself, traced the substitution hook to its
landing point in the model source, checked the array slicing against the
existing ladder gate's own convention, and re-derived the thickness
cancellation rather than accepting it.

### 8a. Its verdict, verbatim

> SHIP WITH FIXES

### 8b. What it found, and what was done

| finding | what it was | what was done |
|---|---|---|
| 1 (MEDIUM) | The GATE SCRIPT's own citation dictionary — which is copied verbatim into every evidence JSON it writes — named lines 296 to 300 of the solver's `dynspg_ts.f90` as the statement that removes the 2-D Coriolis trend.  Those five lines are a comment banner and the coefficient-setup call; the removal is thirty lines further down, at `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:320-324`, which is what the receipt's own prose already said.  Those five wrong lines are written here WITHOUT backticks around a line range on purpose, so that naming a defect does not create a citation to it.  Nothing checked the dictionary, because the citation test read only the strings the RECEIPT renders. | **VERIFIED IN THE COMPILED SOURCE AND FIXED.**  The wrong range is corrected, three other entries are narrowed to the audited ranges, three statements the receipt cites are added, and the citation test now checks EVERY value in the gate's dictionary the same way it checks the receipt: it must be a map key, it must identify its line, and it must FAIL under a two-line shift.  50 tests pass |
| 2 (MEDIUM) | "The sea-surface disagreement is not a fold artefact" is supported by the row coverage, but the receipt never said that its own headline maximum sits on the fold row — the same row it separately flags as carrying a live mask defect. | **ACCEPTED IN FULL.**  The headline paragraph and section 4b now say both things: the disagreement is basin-wide (147 of 148 rows, 0.2126 m with the fold row removed) AND its peak is at row 147, column 38.  The limit on the attribution is stated with it |
| 3 (LOW) | The gravity-wave contamination margin computes its column reach from the grid spacing at row 147, whose metric is atypical on a tripolar grid. | Carried into OPEN.  It does not move any number here — the prediction it serves was refuted and the interior maximum equals the whole-half maximum to nine figures — but an interior row is the right choice for any round that needs the margin to bind |

What it verified INDEPENDENTLY and found correct: that the gate's array
slicing is character-for-character the existing ladder gate's convention, so
the rank-0 window is right; that the substitution hook lands at the same
boundary as NEMO's copy-then-remove, so the arm is not a confound; that the
replacement plant is non-vacuous and targets the channel the conclusion rests
on; that the kt=1 rest-step reasoning is sound and consistently scoped; that
the thickness cancellation is mathematically correct and backed by the
measured residual rather than asserted; that both new test files would fail if
what they test were broken; and that no file under `packages/` changed.

## 9. Gate and test results at the round's final tip

Every measured row below was produced at `fc4a4c147`, the tip after the
review fixes, on a clean worktree.  The two citation rows were re-run once
more after this section was written, because editing a receipt changes what
its own citation gate reads — the numbers they report are from the final tip.

| check | result |
|---|---|
| round-14 barotropic owner gate | **exit 0** |
| its one-representable-value plant | **FIRES** — exit 1; the quoted line is in section 2d |
| ORCA2 ladder, kt = 1 to 10 | **LADDER_MEASURED**, exit 0 |
| round-13 stage-1 owner probe, re-run at this tip | exit 0, every published number reproduced |
| GYRE round-8 receipt citation gate (the campaign gate) | **PASS**, 274 citations, 0 unmapped, 0 failures, 0 map-audit failures |
| this receipt's own citation gate | **PASS**, 10 citations, 0 unmapped, 0 failures, 0 map-audit failures |
| its plant, on a REAL key this receipt renders | **FIRES** — exit 1, status FAIL, `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:196-199`, `SYMBOL-NOT-AT-LINE`, "that symbol identifies line 196", checked against line 198 |

### 9a. The gate re-run after the review, and what it reproduced

The review's first finding changed a citation that the gate stamps into every
evidence file, so both arms were re-run rather than edited in place.  **Every
number reproduces EXACTLY**: the forcing ladder, the instrument controls, the
arm-to-arm movement, all six sea-surface stage rows, the round-13
reproduction, the entry velocity, the contamination margin, the substitution
controls and the first non-bit boundary are byte-identical between the two
runs, and only the citation strings moved.  That is the check that the review
fix touched metadata and not a measurement.

Three earlier runs are kept in the evidence directory on purpose, and named
for what they are: `owner_firstdraft_defective.json` (the difference-of-maxima
measure), `owner_prelocalize.json` (before the fold-row localization) and
`owner_prereview.json`.  A superseded number this campaign deleted would be a
number nobody could check.

### 9b. Batteries

| battery | result |
|---|---|
| the operator push list plus round 13's and round 14's citation tests and this round's gate tests | **191 passed in 362.44 s** |
| DINO, lock exchange and overflow | **169 passed in 509.91 s** |

Both exited 0.  The card battery's 169 is the same count round 13 recorded,
which is expected: no card executes a validation script, and no file under
`packages/` changed.

**Non-vacuity, since no model statement was landed and so none could be
reverted.**  Four separate checks are shown to FAIL when what they test is
broken: the substitution plant (one unit in the last place reaches the sea
surface, section 2d); each of the 10 receipt citations and each of the 11 the
gate stamps (a two-line shift makes every one of them fail); the record
reader's default (it REFUSES the ORCA2 extent, so the new argument is not
decorative); and the injection window (a window of the wrong extent is
refused).

## 10. OPEN — round 15's order

1. **THE SOLVER OWNS THE SEA SURFACE, AND ITS OTHER TWO INPUTS ARE STILL
   UNCHECKED.**  This round substituted the MOMENTUM forcing only.  The solver
   also receives the sea-surface freshwater forcing and the barotropic drag
   coefficients in the same copy (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:287-291`),
   and the record carries both (`cd_u`, `cd_v` in the slow-forcing frame, and
   `oracle_bt_drag_operands_kt00000001.bin`).  Substitute those two FIRST —
   it is cheap and it is the same one-variable method — and only then walk the
   65 sub-steps against `oracle_bt_substeps_kt00000001.bin`,
   `oracle_bt_frames_kt00000001.bin` and
   `oracle_bt_ordered_operands_kt00000001.bin`, in the compiled order at
   `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:668-671`.
   Until those two are substituted, "the solver owns it" means "not the
   momentum forcing", which is narrower.
2. **The meridional face mask on the tripolar fold row**: 668 cells, all on
   row 147, against NEMO's `vmask`; the meridional wind stress on 35 cells of
   the same row; the effective weight support on 15.  REPORTED, not landed:
   the mask builder is shared with every lat-lon card and nothing yet says
   what NEMO's fold rule is.  It is worth at most 1.7e-06 m/s² of forcing and
   2.8e-07 m of sea surface, so it is not urgent — but it is structural, and
   a consumer that reads the face mask directly sees all 668.
3. **The thickness convention** (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:196-199`
   weights with the
   reference thickness and divides by the reference depth; legoESM uses the
   live pair).  Measured, CANCELS in the depth mean, reported as vacuous
   HERE — but it is only vacuous where the two appear as a ratio, and any
   future consumer of one without the other inherits it.
4. **Decision 54 is still three-part and still pending** — and note section
   4d: kt=1 starts from rest, so nothing in this round tested the lateral
   viscosity at all.
5. **Decision 57, the density-reciprocal spelling**, still reported and not
   landed.
6. The vertex-cell-area docstring still estimates the two conventions' gap at
   4.1e-05, three orders of magnitude low.  Round 12's item, uncorrected.
7. The runoff depth operand is cross-checked against NEMO at stage 1 only.
8. Rounds 1 through 12's own compiled citations are still hand-checked; round
   13's seven and round 14's ten are in the map.
9. Round 11's carried items: the tripolar `0.0 / 0.0` in the meridional
   partial-cell pressure gradient; the 1-D reference ladder's
   one-representable-value disagreement at levels 28-29; the barotropic vertex
   thickness; the independent sea-surface height's 1.55 cm on 16,433 cells,
   owned by the initial sea-ice category configuration and out of scope.
10. `emp`'s freedom from runoff is still verified on the CITED paths and the
    `nn_fwb = 2` arm only.
11. **The contamination margin is computed on the fold row's grid spacing.**
    Round 14's review item 3.  Immaterial here — the prediction it served was
    refuted and the interior maximum equals the whole-half maximum to nine
    figures — but any round that needs the margin to BIND should compute it
    on an interior row, because the fold row's metric is atypical.
12. The wide ocean-fidelity battery has not been run at this round's final
    tip.  Operator action.

## Choices

ASKED: the round's order asked for the substitution, for both numbers with
Decision 52's labels, for a walk of whichever half owns it, and for a fix only
where the record proves one.  All four were followed; the record proves no fix
that is landable without a pending decision, so nothing landed.

UNASKED: none STANDING.

No scheme selection, selector default, tunable, threshold, cadence,
resolution, timestep, carried state, data source or previously-tolerated
condition moved.  The lateral-viscosity operator was not touched (Decision 54)
and the density-reciprocal spelling was not landed (Decision 57).
