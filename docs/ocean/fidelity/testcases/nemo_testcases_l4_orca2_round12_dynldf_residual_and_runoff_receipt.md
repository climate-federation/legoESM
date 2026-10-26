# NEMO testcase Lane 4 — ORCA2 card round 12 transcription receipt

Date: 2026-09-23

Starting tip: `4dfdd5a1cebc080e23eb38ff1ad31cbd57c19264`, rebased at the
end onto `2c832edb2` — the lane moved by one receipt-only commit while
this round ran (round 11's wide battery finished green and its receipt
says so).  Nothing in this round's measurements depends on that commit;
the citation gate was re-run after the rebase and still passes.  The
commit each saved gate JSON stamps is a PRE-REBASE sha and no longer
resolves on this branch; every one of them stamped a CLEAN worktree,
and the rebase changed no file any of them read.

Preregistration: `ac8b2539f`

Status: **LANDED-READY.**

Two things happened, and the second is not the one the round's order expected.

**A — the lateral-viscosity operator is now FULLY explained.**  Round 11 left
2 to 4 percent of the disagreement with no owner.  It has one: legoESM's
stored VERTEX-CELL AREA, which is the exact spherical-cap area while NEMO
stores `e1f*e2f`, and on ORCA2 the two differ by a MEDIAN of 5.4 percent.
With that and one smaller statement added to round 11's two, the compiled
loops reproduce the production operator to **3.1e-16 (u) and 2.7e-16 (v)** in
L2 — machine precision.  Nothing from statement A is landed: Decision 54 is
pending, and the decision is now a THREE-way one.

**B — the river-runoff tracer source is transcribed, gated bit-exact, and
LANDED — and it is NOT the owner of the ladder's first disagreement.**  The
ladder's attribution of its whole stage-1 temperature row to this statement
was a hard-coded mapping, and it is too broad: 191,282 of the 233,341
disagreeing cells lie off every runoff column, and adding the channel leaves
that count unchanged.  Round 13's order follows from that.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  Sea ice remains out of scope and the six-entry `unmeasured_features`
tuple is unchanged.

Independent review **not run (codex is paused)**; a fresh adversarial reviewer
was run in its place (section 8).

## 1. What the record fixes, and what nothing here may choose

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity operator family | div-rot, `nn_dynldf_typ = 0`, laplacian, iso-level | run `ocean.output:1176-1180,1193,1195` |
| viscosity coefficient source | `ahmt_3d`/`ahmf_3d` read whole from `eddy_viscosity_3D.nc` | run `ocean.output:1184,1197,1200-1201` |
| lateral momentum boundary condition | no-slip, `rn_shlat = 2.0` | run `ocean.output:339` |
| river runoff | ON | run `ocean.output:534` |
| runoff river-mouth treatment | `ln_rnf_mouth = T`, `rn_hrnf = 15 m`, `rn_avt_rnf = 1e-3` | run `ocean.output:648-650` |
| runoff multiplier | `rn_rfact = 1.0` | run `ocean.output:651` |
| runoff source | `runoff_core_monthly`, variable `sorunoff`, monthly climatology | run `ocean.output:657-660` |
| runoff DEPTH spreading | NOT selected | neither `ln_rnf_depth`'s nor `ln_rnf_depth_ini`'s banner appears in the run's `ocean.output` |
| runoff temperature / salinity from a file | NOT selected | neither banner appears |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

The record is the pinned ORCA1-ice reference run
`orca1ice_surface_entry_every_step_a_np2`.  **No new NEMO run was made or
needed**: `rnf`, `rnf_b`, `rnf_tsc` and `rnf_tsc_b` are recorded in every
surface frame the round-5 acquisition wrote.

A correction to the round's own order: there is **no `dynldf_lap_blp.f90`** in
this build.  The compiled owner of both the laplacian and the bilaplacian
level operators is `dynldf_lev.f90`.

## 2. Statement A — the last few percent, named

### 2a. The three candidates, and why only three

Reading the two sides statement by statement, exactly three legoESM behaviours
were left unsubstituted by round 11.  A fourth candidate round 11 listed —
NEMO's stage-three `Kbb`/`Kmm` split — is **excluded by construction and not
measured**: the gate drives both sides with ONE recorded sea surface, so no
such difference can exist inside it.

| ID | statement | compiled owner | result |
|---|---|---|---|
| A-e3t | the OUTER DIVISOR of the divergence term | `dynldf_lev.f90:127` | **VACUOUS on the scored set** |
| A-curl-edge | the zonal edge length of the circulation loop | `dynldf_lev.f90:123-125` | **small** — 0.2 percent of the residual |
| A-metric | the stored horizontal metrics | `domain.f90:197-205,212-215` | **THE OWNER** — 93 percent of it |
| A-slope | the slope-foot factor on the operator's output | (no NEMO statement) | **VACUOUS** — the factor is the scalar 1.0 |

### 2b. The numbers

Every row is scored against the PRODUCTION operator, on ONE cell set, at kt=2
(kt=1 is rest and cannot score a momentum operator — round 11's finding).
Round 11's rows were re-run with the new rows added and **reproduce exactly**,
so the extension did not change the instrument.

| row | u, L2 | v, L2 |
|---|---|---|
| round 11: the production operator against the compiled loops | 0.7681 | 0.6136 |
| round 11 closure — the extra vertex mask and the min-rule thicknesses | 0.02669 | 0.01904 |
| + A-e3t (legoESM's live cell thickness as the divisor) | 0.02669 | 0.01904 |
| + A-curl-edge (the cell metric on the circulation's zonal edges) | 0.02665 | 0.01902 |
| **+ A-metric (legoESM's stored horizontal metrics)** | **0.001850** | **0.001405** |
| **all of them together** | **3.06e-16** | **2.72e-16** |

**The operator's disagreement is therefore 100 percent attributed.**  That
also certifies, in one measurement, the transcription, the index map, the
coefficient read and all four named statements together: nothing is left over.

### 2c. Which metric, and by how much

Five of the six metrics the compiled statements read are **bitwise** the
record's own mesh-mask arrays.  Exactly one is not.

| metric role | cells unequal | max relative |
|---|---|---|
| cell area `e1t*e2t` | 0 / 26,640 | 0 |
| u-face zonal length `e1u` | 0 / 26,640 | 0 |
| u-face meridional length `e2u` | 0 / 26,640 | 0 |
| v-face zonal length `e1v` | 0 / 26,640 | 0 |
| v-face meridional length `e2v` | 0 / 26,640 | 0 |
| **vertex-cell area `e1f*e2f`** | **26,456 / 26,640** | **14,007** |

The vorticity bracket divides by that area, so the error passes straight into
the tendency.  It is not a corner case: the **median** relative difference is
**5.4 percent**, 26,376 cells exceed 0.1 percent (16,269 of them on a wet
column) and 3,335 exceed 10 percent.  legoESM builds the vertex area as the
EXACT spherical cap
(`packages/core/legoesm/grids/operators_latlon_cgrid.py` `_vertex_dual_area_interior`);
NEMO stores the product of two midpoint metrics.  The module's own docstring
already recorded that the two conventions differ — it estimated the gap at
4.1e-05, which is right for a regular lat-lon grid and wrong by three orders
of magnitude for this one.

### 2d. The two that are vacuous, and what that means

**A-e3t.** legoESM's live cell thickness differs from NEMO's
`e3t_0*(1+r3t*tmask)` on 368,648 of 799,200 cells, by up to 500 m — and
substituting it changes the operator on NOT ONE cell.  Every cell where the
two disagree carries a viscosity coefficient of exactly zero, so the divisor
never multiplies anything.  Reported as vacuous ON THE SCORED SET, not as
agreement.

**A-slope.** `slope_foot_alpha` is 0.0 on this card, so both factors are the
scalar 1.0 and the operator's output is returned unmultiplied.  Settled by
reading the value, not by a substitution, because at alpha = 0 there is no
array to substitute.

### 2e. Nothing from statement A lands, and the decision has grown

Round 11 raised Decision 54 as mask-only versus both halves.  It is now a
THREE-part fix, and the third part is the largest of the three that can be
scoped:

| part | can it be scoped to ORCA2? | why |
|---|---|---|
| drop the second zero/one vertex mask | yes | only the file-sourced coefficient already carries NEMO's mask, and only the ORCA2 card selects that source |
| use NEMO's stored face and vertex thicknesses | no | GYRE selects the same `nemo_e3` weighting through the same operator |
| use NEMO's `e1f*e2f` as the vertex area | no | every lat-lon card's vorticity divides by the stored vertex area |

## 3. Statement B — the river-runoff tracer source

### 3a. What NEMO executes

| statement | compiled owner | what it says |
|---|---|---|
| runoff is on | `sbcrnf.f90:184` | `IF( ln_rnf ) THEN` |
| the temperature content | `sbcrnf.f90:221` | the runoff enters at the sea surface temperature, floored at 0 |
| the salinity content | `sbcrnf.f90:227` | `zrnf_sal * rnf * r1_rho0`, and `zrnf_sal` is 0 (`sbcrnf.f90:175`) |
| how deep it goes | `sbcrnf.f90:487-488` | the surface arm: one level, over the LIVE top-cell thickness |
| the deposit | `trasbc.f90:318-326` | the reciprocal of the depth is formed FIRST, then multiplied |
| which stages | `trasbc.f90:278` | the block sits OUTSIDE the stage switch, so all THREE stages |

### 3b. What legoESM already had, before anything was written (Rule 4)

Searched `runoff` and `rnf` across `packages/ocean/legoesm/ocean/`.  Found:
the runoff MASS channel into the horizontal divergence
(`ocean_pe_latlon_cgrid.py:1563-1565`, cited to `sbcrnf.F90:253-260`), the
NEMO depth-spreading virtual-salt helper and its argument resolver
(`freshwater.py:485` and `:442`), and the MPAS and lat-lon callers of both.
NOT found: any channel carrying the runoff's TRACER content.  The deposit is
therefore an extension of the existing surface-forcing channel set, not a
second runoff path.  **The mass side is deliberately NOT switched on**: it
forces the sea surface and the horizontal divergence and is a separate
statement, recorded in OPEN.

### 3c. The gate, and why it is not circular

The gate runs the PRODUCTION step twice on the record's own kt=1 state — once
with the recorded runoff content supplied and once with it withheld — and
reads the per-stage tracer source rates the production stage helper consumes
out of the live-operand trace, which now carries them.  The compiled statement
says the first must equal the second PLUS `rnf_tsc * (1/h_rnf)`, with
`rnf_tsc` read from the RECORD and `h_rnf` the live top thickness that same
stage divided by.  The only legoESM-supplied operand is that thickness, and it
is gated separately **at stage 1** — the reviewer's point, recorded rather
than glossed: stages 2 and 3 divide by legoESM's own intermediate
thicknesses, which this gate does not cross-check against NEMO, so a
thickness wrong the SAME way at every stage would not be caught here.

| row | result |
|---|---|
| stage 1, temperature and salinity | **0 / 799,200 unequal each** |
| stage 2, temperature and salinity | **0 / 799,200 unequal each** |
| stage 3, temperature and salinity | **0 / 799,200 unequal each** |
| the runoff depth operand — legoESM's live top thickness against NEMO's own, at stage 1 | **0 / 16,433 wet cells unequal**, max 0.0 |
| can the two spellings of the divisor be told apart on this card at all? | **yes, on 749 cells** — so the bitwise rows above ARE a reciprocal-first control and not a vacuous one |
| control — the stage-2 thickness fed to the stage-1 row | **REFUSES**, 3,400 cells |
| the gate at the round's BASE commit | **REFUSES** — "the ladder's surface forcing carries no runoff tracer content" |
| one-representable-value plant | **FIRES** — "PLANT FIRED: the gate refuses a one-representable-value move" |
| gate exit at the tip | **0 — AT BAR** |

Label: **given NEMO's entry** (the kt=1 recorded state and the recorded
runoff frames).

### 3d. What the record says about the statement itself

| quantity | value |
|---|---|
| surface cells carrying a non-zero runoff | 4,649 of 26,640 |
| of those, cells whose runoff carries HEAT | 3,400 — the other 1,249 sit where the sea surface temperature is at or below 0 degC, which the compiled `MAX(sst_m, 0)` floors away |
| largest runoff temperature content | 1.5424e-05 K m/s |
| largest runoff SALINITY content | **exactly 0.0** — `zrnf_sal = 0` |

So the runoff adds heat at 3,400 surface cells and **no salt anywhere**.

## 4. The landing makes the cells it touches WORSE, and that is the round's finding

The deposit is bit-exact against NEMO's own runoff content.  It nevertheless
moves the ORCA2 trajectory AWAY from NEMO on every cell it reaches.

| stage-1 temperature, against NEMO's recorded stage-1 frame | whole field, max | on the top cell of a runoff column, max |
|---|---|---|
| WITHOUT the runoff channel (round 11's state) | 1.4764e-03 degC | **1.4655e-04 degC** |
| WITH it (this landing) | 1.5287e-03 degC | **1.5287e-03 degC** — about **10x worse** |
| with it, and the runoff's MASS paired in as well (measurement arm, nothing landed) | 1.4990e-03 degC | 1.4990e-03 degC |

The deposit itself is the right size: it moves the stage-1 temperature on
exactly **3,400 cells**, all of them top cells, by up to **5.5692e-03 degC**,
which is `rnf_tsc / h_rnf` times the stage's own time increment.  So NEMO's
stage-1 temperature at those river mouths behaves as if it had NOT received a
5.6e-03 degC runoff kick, while the compiled source says it should have.

**The obvious explanation was tested and REFUTED.**  NEMO pairs this heat with
the runoff's MASS: the same runoff enters the sea-surface forcing and the
horizontal divergence, thickening the top cell and diluting exactly what the
heat adds.  Supplying the recorded runoff through legoESM's freshwater channel
barely moves the row (1.5287e-03 -> 1.4990e-03).  **That arm is CONFOUNDED and
is reported as confounded, not as a refutation of the mechanism**: legoESM's
freshwater channel feeds BOTH the sea surface AND the stage dilution term
(`_emp_stage_rate`), and NEMO's dilution term reads `emp`, which EXCLUDES the
runoff — so adding the runoff there deposits a SECOND copy of the same heat
instead of the compensating thickness.  Pairing the mass correctly needs the
divergence source, not the freshwater channel, and that is round 13's order.

So: the statement is transcribed and gated; the pairing it belongs to is not,
and until it is, this channel costs about a factor of ten on 3,400 cells.
**DECISION_NEEDED**: keep the landing, or hold it until its partner lands.

## 5. What the ladder now says

| row | before | after |
|---|---|---|
| ORCA2 ocean ladder, kt = 1 to 10 | `LADDER_MEASURED`, exit 0 | unchanged — `LADDER_MEASURED`, exit 0 |
| first non-bit statement | kt=1 stage 1 temperature, cited to the runoff source | kt=1 stage 1 temperature, **UNATTRIBUTED** |
| what is ruled out, and how | (nothing — the citation was a hard-coded map) | the runoff source, on **231,291 of 233,341** disagreeing cells it cannot reach |
| kt=10, same field, step entry | 430,552 / 799,200, max 3.9436 degC | 430,552 / 799,200, max **3.9435** degC |
| kt=10, stage 3 | 233,341 / 399,600, max 0.7713 degC | 233,341 / 399,600, max **0.7713** degC |

Label: `INDEPENDENT_WITH_DECISION52_SSH`.

**The ladder's attribution rule was a claim, and it is now a measurement.**  It
bound every stage-1 temperature or salinity row to the runoff source.  With
that source satisfied bit-exactly the row does not move at all, and 191,282 of
its cells lie off every runoff column — places the statement cannot reach.
A citation that survives its own statement being satisfied is not an
attribution, so the ladder now names the runoff only when the disagreement is
confined to where it can act, and otherwise reports the row as unattributed
with what was ruled out and on how many cells.

Ranked by magnitude the stage-1 kt=1 rows are: sea surface 8.17e-02 m on
8,794 cells, zonal velocity 6.41e-02 m/s on 247,035, meridional velocity
3.40e-02 m/s on 237,822, salinity 1.85e-03 on 233,341, temperature 1.53e-03
degC on 233,341.  The temperature row is the SMALLEST of the five and is
reported first only because the ladder walks the fields in a fixed order.

## 6. Frozen predictions, resolved

| ID | outcome |
|---|---|
| R12-P1 | **REFUTED** — the divisor is not the largest candidate; it is VACUOUS on the scored set (section 2d). |
| R12-P2 | **CONFIRMED** — the slope-foot factor is the scalar 1.0. |
| R12-P3 | **CONFIRMED, and more strongly than predicted** — the residual does not merely fall, it reaches machine precision (3.06e-16 / 2.72e-16). |
| R12-P4 | **CONFIRMED, and sharpened** — exactly ONE of the six metrics differs, and it differs by a median of 5.4 percent. |
| R12-P5 | **CONFIRMED** — 4,649 of 26,640 surface cells carry runoff and 3,400 carry runoff heat, against 233,341 disagreeing cells; 191,282 of them lie off every runoff column. |
| R12-P6 | **CONFIRMED** — the recorded runoff salinity content is exactly 0.0. |
| R12-P7 | **CONFIRMED** — 0 of 799,200 unequal at every stage, for both tracers, with a gate that refuses at the base commit and whose plant fires. |
| R12-P8 | **HALF REFUTED** — the ladder still runs kt=1..10, but its first disagreement does NOT move.  That is what exposed the attribution defect. |
| R12-P9 | **CONFIRMED** — GYRE is bit-identical (section 7). |

## 7. GYRE is unchanged

Proven at the round's base tip and at its final tip with the evaluation
protocol byte-identical.

| row | result |
|---|---|
| ten-step ladder, offline oracle-relative compare | **PASS** — 70 certified rows, 0 status changes, 0 violations, first-over-bar unmoved (`u`,`v` at kt=2) |
| largest oracle-residual worsening | **0.0 ULP** |
| residual arrays, elementwise equal | **210 / 210** |
| thirty-day member, byte-identical daily snapshots | **30 / 30** |
| day-30 digest | `14a7e64b4512860e...` — the same as rounds 7, 8, 9, 10 and 11 |

**CORRECTION to round 11's receipt.**  It reported "max worsening 0 ULPs"
against the comparator's `max_ulp_worsening` field.  That field reads **2** in
round 11's own saved comparison and **2** here, on runs both proven
byte-identical; the field that is zero, and the one that means what round 11
said, is `largest_oracle_residual_worsening_ulps`.  Quoted correctly above.

## 8. Independent review

Codex is paused, so `codex exec` was NOT run and this round claims no
independent codex verdict.  A fresh adversarial reviewer was run in its place.
It read the compiled Fortran itself, traced the call chain into
`sbc_rnf_div`, diffed every edited line, RAN the new tests, and spot-checked
nine citations and the JSON numbers against the receipt.

### 8a. Its verdict, verbatim

> SHIP WITH FIXES — 2 reproducible test failures block merge as-is; the
> physics transcription and GYRE isolation are otherwise sound.

### 8b. What it found, and what was done

| finding | what it was | what was done |
|---|---|---|
| B1 (blocking) | the gate cited the runoff switch to `sbcrnf.f90:184`, which is a blank comment line; the guard it meant is in `trasbc.f90` | re-pointed at `trasbc.f90:318`, the RK3 guard actually transcribed.  **The gate's own citation test had already caught this** — that is what that test is for |
| B2 (blocking) | three of the five new controls never EXECUTED: they read the per-stage rates out of the live operand trace, which refuses on the lock-exchange card because it populates no turbulence operands | rewritten to read the per-stage tracer state through the stage exposure hook, which the card does support; all now run, and four of the six FAIL when the statement is reverted |
| N1 | the runoff-depth operand is cross-checked against NEMO at stage 1 only, while the receipt read as full coverage | scope stated explicitly in section 3c and carried into OPEN |
| N2 | the GYRE snapshot numbers were quoted but their output was never saved | re-run and saved (`gyre_snapshot_compare.log`); numbers reproduce exactly |

What it verified INDEPENDENTLY and found correct: that `tra_sbc_RK3` and not
the leapfrog `tra_sbc` is the right subroutine; that the runoff block is
outside the stage switch and therefore runs at all three stages; the
reciprocal-first spelling; `nk_rnf = 1` and the live top-cell depth, traced
through `div_hor` to `sbc_rnf_div` to confirm it is recomputed per call at the
live time level; the identically-zero salinity content; that a card supplying
no runoff content skips the arithmetic rather than adding a zero, which is the
signed-zero concern; that the two vacuity claims are MEASURED rather than
inferred; and nine legoESM citations including the one whose extent changed.

Two further things this round did on its own after the review: it added the
row that shows the two spellings of the divisor ARE distinguishable on this
card (749 cells), so the bitwise rows cannot be a vacuous reciprocal-first
control; and it found that the top-cell control is only about this statement
at stage 1, because by stage 2 the deposit has already been advected out of
the top cell (1.3e-12 on the lock card).

## 9. Gate and test results at the round's final tip

| check | result |
|---|---|
| ORCA2 ladder, kt = 1 to 10 | **LADDER_MEASURED**, exit 0 |
| round-12 runoff gate | **AT BAR**, exit 0 — 0 / 799,200 unequal on all six rows |
| its base-commit control | **REFUSES** — "the ladder's surface forcing carries no runoff tracer content" |
| its one-representable-value plant | **FIRES** — "PLANT FIRED: the gate refuses a one-representable-value move" |
| its wrong-stage-thickness control | **FIRES** — 3,400 cells |
| lateral-viscosity operator gate | **REFUSES**, exit 2 (nothing from statement A landed); round 11's rows reproduce exactly; the all-substitutions closure reaches 3.06e-16 / 2.72e-16 |
| GYRE identity, base vs tip | section 7 |
| receipt citation gate | **PASS**, 274 citations, 0 failures, 0 map-audit failures |
| citation gate with a rigid two-line plant on a REAL receipt key | **FIRES** — exit 1, status FAIL, `dynldf_lev_rot_scheme.h90:24-25`, `SYMBOL-NOT-AT-LINE`, "that symbol identifies line 24", found at line 26 |
| the operator push battery plus this round's tests | **156 passed in 584.05 s** |
| this round's own tests | **13 passed in 213.45 s** |
| the same tests with the landed statement reverted | **4 failed, 2 passed** — the controls are not vacuous; the model file was then restored and `git status --porcelain` printed nothing |

**A BLIND SPOT IN THE CITATION GATE, found here and recorded.**  Planting a
citation that exists only as a map key and is not rendered in the receipt's
prose does NOT fire: the gate reports `status: PASS` and exits 2.  A plant
must therefore name a citation the receipt itself renders, or it proves
nothing — the same class of error round 10 withdrew two plant rows for.

## Choices

ASKED: the round's order asked for the runoff statement to be transcribed,
gated and landed, and for statement A to be measured and NOT landed.  Both
were followed.

UNASKED: none STANDING.  Two decisions are RAISED rather than taken:

1. The lateral-viscosity operator fix is now a THREE-part scoping question,
   not round 11's two-part one (section 2e).
2. The runoff landing degrades the cells it touches by about a factor of ten
   until its partner statement lands (section 4), and is offered for revert.

No scheme selection, selector default, tunable, threshold, cadence,
resolution, timestep, carried state, data source or previously-tolerated
condition moved.  The runoff MASS channel was deliberately NOT switched on.

## 10. OPEN — round 13's order

1. **Pair the runoff's MASS with its heat.**  NEMO feeds the same runoff into
   the horizontal divergence and the sea-surface forcing; legoESM's freshwater
   channel is the WRONG place to put it, because that channel also feeds the
   stage dilution term and would deposit the heat twice.  Until this lands the
   heat-only landing costs a factor of ten on 3,400 cells (section 4).
2. **Name what owns the stage-1 disagreement.**  It is not the runoff.  It is
   on 233,341 cells at every depth, and the three largest stage-1 rows are the
   sea surface (8.2 cm) and the two velocities (6.4 and 3.4 cm/s) — an
   upstream, momentum-or-barotropic owner, not a tracer source.
3. **Decision 54 is now three-part** (section 2e): the extra vertex mask
   (scopeable to ORCA2), the face and vertex thicknesses (not scopeable), and
   the vertex CELL AREA (not scopeable, and the largest of the three by
   residual).  Nothing lands until it is settled.
4. **legoESM's vertex cell area is not NEMO's `e1f*e2f` on this grid**, and the
   module docstring that records the two conventions estimates the gap at
   4.1e-05 — right for a regular lat-lon grid, wrong by three orders of
   magnitude for ORCA2 (median 5.4 percent).  That estimate should be
   corrected wherever it is repeated.
5. **The runoff depth operand is cross-checked against NEMO at stage 1 only**;
   stages 2 and 3 divide by legoESM's own intermediate thicknesses, unchecked.
6. **A citation planted only in the gate's map does not fire.**  Any future
   plant must name a citation the receipt renders.
7. Round 11's items that remain open: the tripolar `0.0 / 0.0` in the
   meridional partial-cell pressure gradient (unreachable here, live for other
   tripolar configurations); the 1-D reference ladder's one-representable-value
   disagreement at levels 28-29; the barotropic vertex thickness, still
   unmeasured; the independent sea-surface height's 1.55 cm on 16,433 cells,
   owned by the initial sea-ice category configuration and out of scope.
8. The wide ocean-fidelity battery has not been run at this round's final tip.
   Operator action.
