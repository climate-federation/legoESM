# NEMO testcase Lane 4 — ORCA2 card round 11 transcription receipt

Date: 2026-09-23

Starting tip: `c28ac176c51ea090913bcabac7517fd8dad1bda8`

Preregistration: `d88f9bd6a781`

Status: **LANDED-READY.**  Round 10's OPEN item 1 is discharged and, with it,
the whole ladder: the quantity that was non-finite inside the implicit
vertical mixing at kt=1 is named, walked to its FIRST PRODUCER STATEMENT, and
fixed, and the ORCA2 ocean ladder now runs **kt = 1 to 10** for the first time
in this lane.  `kt=10` is therefore **MEASURED**, not UNMEASURED.

Round 9's OPEN item 3 — the second masking — is now gated rather than only
transcribed, together with the rest of the operator.  The gate BINDS and
REFUSES: legoESM's production lateral viscosity disagrees with NEMO's compiled
loops on **every scored cell**.  It is **NOT LANDED**, because the larger half
of the fix cannot be scoped to this card; section 7 states the decision.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  Sea ice remains out of scope and the six-entry `unmeasured_features`
tuple is unchanged.

Independent review **not run (codex is paused)**; a fresh adversarial reviewer
was run in its place (section 9).

## 1. What the record fixes, and what nothing here may choose

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity operator family | div-rot, `nn_dynldf_typ = 0`, laplacian, iso-level | run `ocean.output:1176-1180,1193,1195` |
| viscosity coefficient source | `ahmt_3d`/`ahmf_3d` read whole from `eddy_viscosity_3D.nc` (`nn_ahm_ijk_t = -30`) | run `ocean.output:1184,1197,1200-1201` |
| lateral momentum boundary condition | no-slip, `rn_shlat = 2.0` | run `ocean.output:339` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

The record is the pinned ORCA1-ice reference run
`orca1ice_surface_entry_every_step_a_np2`.  No new NEMO run was made or needed.

## 2. Statement B — the non-finite sea surface, named and walked to its producer

Round 10 measured the refusal and left its cause **PLAUSIBLE**.  It is now
**CONFIRMED**, and the producer is not where the refusal is.

### 2a. The discriminator (R11-P7, R11-P8): it IS the sea surface

The production step's own buoyancy-divisor builder was instrumented at every
call.  The three calls report, on the step's own operands:

| call | sea surface non-finite | column depth non-finite | raw mesh spacing non-finite |
|---|---|---|---|
| step entry | **0 of 26,640** | 0 of 26,640 | 0 of 799,200 |
| implicit vertical mixing (post-Runge-Kutta) | **26,640 of 26,640** | 0 | 0 |
| its companion call | **26,640 of 26,640** | 0 | 0 |

Both probes were re-run against a clean clone at the round's BASE commit and
their output saved, so these are reproducible pre-fix numbers rather than a
transcript quotation (`base_probe_ssh_discriminator.log`,
`base_probe_baro_substeps.log` in the round's evidence directory).

So the sea surface the implicit vertical mixing receives is **entirely NaN**,
the column depth and the raw mesh spacing are clean, and the entry sea surface
— the record's own, through Decision 52's bridge — is clean.  Both
predictions hold.  Label: **given NEMO's entry**.

### 2b. The first producer statement

With the guard shimmed off IN THE PROBE ONLY, the step completes and its own
traces can be read.  Walking them:

| trace | first non-finite |
|---|---|
| stage-1 right-hand side, stage-1 raw velocities, the whole slow-forcing producer | none |
| stage-1 state | none |
| the barotropic sub-step trace, **substep 0** | `drag_v`, **exactly 180 cells, all on v-row 0** |
| the same substep's `v_exit` | the same 180 cells |
| two substeps later | 17,263 cells; then the sea surface, both velocities and every downstream divisor |

`drag_v` is therefore the FIRST non-finite array anywhere in the ORCA2 step.
Its only operand that the trace did not already show finite is the entry
inverse face depth, and the trace masks that one before printing it — which is
why round 10 could not see it.

Measured directly on the card: the southernmost v-face row has
`e1v = e2v = 0` **exactly**, on all 180 longitudes (180 of 26,820 v faces;
every other row is positive, and the u faces have no zero at all).  It is a
closed wall of zero extent.  The loop-invariant prep built the inverse face
area as a plain reciprocal
(`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:513` at the
base commit), so that row was `+inf`; the stretch statement a few lines later
multiplies it by the dry face's exactly-zero reference reciprocal — `0 * inf`
— and the entry inverse face depth came out NaN on the whole row.

**The same function already knew that row is degenerate**: its sibling
statement zeroes the ssh-average on both polar rows, which is why the face
DEPTH was finite (exactly 0.0) while the inverse face depth was NaN.

NEMO has no statement to match here: its own `r1_e1e2u`/`r1_e1e2v`
(`domain.f90:212-215`) are reciprocals of metrics that are strictly positive
everywhere in its domain.  This is legoESM's own representation of a closed
wall, so the fix is legoESM-side and adds no stabiliser NEMO lacks.

### 2c. What landed

A zero-area face now gets a reciprocal of **exactly zero**, and wherever the
face area is positive the value is **bitwise** the plain reciprocal.  One
helper, two call sites (the u statement gets it too, because it is the same
statement and today's card only happens to have no zero there).

Two tests ship with it, and both were shown to **FAIL** when the guard is
reverted to a plain reciprocal — measured, then the model file restored and
`git status --porcelain` confirmed clean.

### 2d. The ladder, which is the point

| row | before this round | after |
|---|---|---|
| ORCA2 ocean ladder, kt = 1 to 10 | refused at kt=1 by the raw-mesh guard | **runs to kt=10**, status `LADDER_MEASURED`, exit 0 |
| first non-bit statement | not reachable | **kt=1, stage 1, temperature** (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:314-328` — the runoff tracer source, a channel the production step does not carry; round 10's receipt already listed it as unsupported) |
| kt=10 magnitude, same field, step entry | **UNMEASURED** | 430,552 of 799,200 cells unequal, max 3.9436 °C, mean over unequal 0.005030 °C |
| kt=10, stage 3 | UNMEASURED | 233,341 of 399,600 unequal, max 0.7713 °C |

Label: `INDEPENDENT_WITH_DECISION52_SSH` (the ladder's own existing label — the
initial temperature and salinity are legoESM's own, the entry sea surface is
NEMO's).

## 3. Statement A — the whole lateral viscosity operator, gated

### 3a. A correction round 11 had to make first

Round 10 section 6 described legoESM's operator as applying NEITHER the
thickness weighting nor the outer face-thickness division, citing the docstring
of `nemo_ldf_lap_viscosity_cgrid`.  **The ORCA2 card does not call that
function.**  The shared NEMO-identity base selects both
`lateral_viscosity_operator="nemo_div_curl"` and
`lateral_viscosity_e3_weighting="nemo_e3"`
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:308-309`), so
ORCA2 — and GYRE — run `nemo_ldf_lap_viscosity_e3_cgrid`
(`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3440`), which
DOES carry a thickness weighting.  R11-P1 **CONFIRMED**, including its second
half: that function also has a literal compiled-statement branch, but it is
reachable only when a caller supplies `thickness_operands`, and **no production
call site does** — the only callers that pass them are gates.  Production
therefore takes the algebraic branch and builds every face and vertex thickness
by the min rule from ONE time level.

### 3b. What NEMO actually executes

`dyn_ldf` is dispatched to `dynldf_lev_lap`'s `np_typ_rot` arm
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf.f90:85`) and is
called from Runge-Kutta **stage three** with `Kbb = Nbb`, `Kmm = Nnn`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:493`).
Four statements:

| statement | compiled owner |
|---|---|
| `zwf` = stored `ahmf` x live vertex thickness x inverse vertex area x the circulation bracket, annotated "ahmf already * by fmask" | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf_lev.f90:123` |
| `zwt` = stored `ahmt` x inverse cell area / live T thickness x the thickness-weighted flux sum, annotated "ahmt already * by tmask" | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf_lev.f90:127-129` |
| the u right-hand side: curl branch divided by the live u thickness at `Kmm`, div branch not | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf_lev.f90:133-136` |
| the v right-hand side, its mirror | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf_lev.f90:137-140` |

with the stretches from `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domqco.f90:257`
(T), `:266-269` (u and v) and `:281-285` (the four-cell vertex average), the
reference column thicknesses and their reciprocals from
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domain.f90:197-205` and
`:212-215`, and the thickness mask frozen from the FREE-SLIP mask at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dommsk.f90:258` — taken, with
its column maximum at `:247-250`, BEFORE `rn_shlat` rewrites the vorticity mask
at `:269-277`.  The vertex thickness itself is read whole from the domain file
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domzgr.f90:188`), and the
coefficient from the viscosity file
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:348-353`).

### 3c. kt=1 cannot score this operator, and that is a finding

ORCA2 starts from **rest**: the recorded kt=1 step-entry velocity is
identically zero on all 799,200 cells.  At kt=1 this operator is therefore
exactly zero on BOTH sides, and every comparison and every ablation is a
zero against a zero — the gate's first run said `0 unequal` on every row,
including rows that CANNOT agree.  The gate now REFUSES a step whose recorded
velocity is identically zero, and scores the earliest step that carries one,
kt=2 (peak recorded speed 0.540 m/s).  The preregistration's "kt=1 velocities"
is **REFUTED as a scoreable choice** and is kept here rather than dropped.

### 3d. The gate, its rows and its controls

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round11_dynldf_operator_gate.py`
recomputes the compiled loops from the record's own inputs and scores them
against legoESM's PRODUCTION lateral-viscosity tendency, reached through the
production momentum-tendency path.  Both sides are driven by the SAME recorded
step-entry sea surface, so the comparison is controlled; NEMO's stage-three
`Kbb`/`Kmm` split is a separate statement and is NOT scored (section 8).

| row | result |
|---|---|
| **u momentum, production vs the compiled loops** | **411,736 of 411,736 scored wet cells unequal**, max 2.209e-05 m/s2, residual 0.768 of the oracle in L2 |
| **v momentum** | **412,537 of 412,537 unequal**, max 1.798e-05 m/s2, residual 0.614 in L2 |
| ablation — legoESM's extra zero/one vertex mask alone | moves 31,399 u and 34,166 v cells; 0.769 / 0.612 in L2 |
| ablation — legoESM's min-rule live thicknesses alone | moves every cell; 0.768 / 0.612 in L2 |
| **closure — BOTH ablations applied to the compiled loops** | residual falls to **0.027 (u)** and **0.019 (v)** in L2: a 29x and 32x reduction |
| coefficient cells the extra mask zeroes | **48,287 of 799,200** |
| gate exit | **2 — REFUSES** |

Every row above is scored on ONE cell set — wet faces where the compiled
transcription is defined — so the rows are comparable.  The reviewer found the
closure row originally scored 1,319 more v cells than the row it was compared
against (an ablation can define a cell the reference leaves undefined); with
that closed the v closure improves from 0.039 to 0.019.

**The gate binds.**  Its binding rows run `LatLonCGridOceanModel.
tendencies_with_diagnostics` — the production momentum-tendency path — on the
card's own configuration, and the gate refuses unless that card still selects
the div-curl operator, the `nemo_e3` weighting and the file coefficient.

**The instrument was validated before its disagreement was quoted**, by the
closure row above and by four unit controls that ship with it: the index shifts
are checked against what their names say (periodic in longitude, no wrap in
latitude), the reciprocal is exact where the metric is positive, the
transcription returns **exactly zero** on a field whose divergence and
vorticity both vanish, and each ablation is shown to move the answer.  The
preregistration's "agree to 1e-12 on a uniform case" control was replaced by
these.  They are NOT strictly stronger — they catch a different failure class
— and none of them independently reproduces the compiled arithmetic; what
establishes that is the closure row and the independent reviewer's own
line-by-line reading of the Fortran (section 9).

The 48,287 count is an **independent reproduction of round 9's**, from a
different direction — it falls out of the production mask construction, not out
of round 9's gate.

### 3e. What the numbers say, and what they do not

**R11-P2 CONFIRMED**: the production operator differs on every scored cell.

**R11-P3 CONFIRMED in its count** (48,287 reproduces) and in its shape (the
mask moves the F cells and their two right-hand-side neighbours, 31,399 u and
34,166 v).

**R11-P4 REFUTED**: the thickness term is NOT measurably larger than the mask
term.  In this norm they are the same size (0.768 vs 0.769 for u; 0.612 vs
0.612 for v) and they **OVERLAP almost completely** — neither is separable
from the other, because both alter the same F-point contribution in the same
places.  The honest statement is the closure: TOGETHER they account for about
97 percent of the u disagreement and about 98 percent of the v disagreement,
and neither alone accounts for its own share.

**R11-P9 REFUTED**: the ladder does reach kt=10 (section 2d).

## 4. GYRE is unchanged

Proven, not argued, at the round's base tip and at its final tip with the
evaluation protocol byte-identical:

| row | result |
|---|---|
| ten-step ladder, offline oracle-relative compare | **PASS** — 70 certified rows, 0 status changes, 0 violations, max worsening 0 ULPs, first-over-bar unmoved (`u`,`v` at kt=2) |
| residual arrays, elementwise equal | **210 / 210** |
| thirty-day member, byte-identical daily snapshots | **30 / 30** |
| day-30 digest | `14a7e64b4512860e...` — the same as rounds 7, 8, 9 and 10 |

## 5. The other cards that execute the changed statement

The changed statement is in the SHARED barotropic solver, so every card with a
split-explicit external mode executes it.  Their cheapest gates:

| card | result |
|---|---|
| DINO, lock exchange and overflow, together | **169 passed** |
| every barotropic unit battery | **26 passed** in isolation for the continuity-and-drag file, **90 passed, 1 skipped, 1 xfailed** for the rest |
| the operator push battery plus this round's new tests | **132 passed in 363.02 s** |

One battery that mixed all of them in a single pytest process aborted mid-run
with a JAX compile abort and reported six failures in the continuity-and-drag
file; re-run in isolation that file is **26 passed**, so those six were the
abort, not this change.  Recorded rather than dropped.

## 6. Frozen predictions, resolved

| ID | outcome |
|---|---|
| R11-P1 | **CONFIRMED** — the card runs the e3-weighted operator, and no production call site supplies the literal compiled operands (section 3a). |
| R11-P2 | **CONFIRMED** — every scored cell differs (section 3d). |
| R11-P3 | **CONFIRMED** — 48,287 reproduces independently. |
| R11-P4 | **REFUTED** — the two terms are the same size and overlap; neither is separable (section 3e). |
| R11-P5 | **NOT TESTED** — the mask fix is not landed (section 7), so its scoping was not measured.  It is not counted as confirmed. |
| R11-P6 | **CONFIRMED** — GYRE selects the same `nemo_e3` weighting through the same operator, so the thickness half cannot be scoped to ORCA2. |
| R11-P7 | **CONFIRMED** — the sea surface is the non-finite quantity, 26,640 of 26,640 (section 2a). |
| R11-P8 | **CONFIRMED** — the entry sea surface is clean; the NaN is made inside the step (section 2b). |
| R11-P9 | **REFUTED** — the ladder reaches kt=10. |

## 7. The decision the operator fix needs

Both halves of the operator fix change a statement GYRE also executes, but not
in the same way.

| half | can it be scoped to ORCA2? | why |
|---|---|---|
| drop the second zero/one vertex mask | **yes** | only the file-sourced coefficient already carries NEMO's mask, and only the ORCA2 card selects that source |
| use NEMO's own stored face and vertex thicknesses instead of the min rule | **no** | GYRE selects the same `nemo_e3` weighting through the same operator and would move |

Landing only the first would move the ORCA2 trajectory while leaving the
operator refusing, and the closure shows the two are not separable — fixing one
leaves the disagreement at about the same size.  So this round lands neither
and raises it.  DECISION_NEEDED is in the final report.

## 8. OPEN — round 12's order

1. **The operator fix, both halves, under the decision in section 7.**  It is
   transcribed, cited and gated; only the landing is pending.
2. **A third, unattributed difference of about 3 to 4 percent** remains after
   both known statements are applied (section 3d closure).  Candidates, none
   measured: NEMO's stage-three `Kbb`/`Kmm` split, which legoESM does not carry
   at all in this call; the slope-foot adjustment the production path applies
   after the operator; legoESM's own stretch construction.  Name it before
   landing anything, or the landing will be scored against a moving target.
3. **The first non-bit statement is now the runoff tracer source**
   (`trasbc.f90:314-328`), at kt=1 stage 1, carrying 3.94 degrees C by kt=10.
   That is the largest measured term on this lane and it owns the next walk.
4. **kt=1 cannot score any momentum operator** (the card starts from rest).
   Any future momentum gate on this lane must refuse a rest step rather than
   report zero unequal.
5. **The barotropic trace masks its own inverse face depth before printing it**,
   which is why round 10 could not see the NaN there.  The trace is a
   diagnostic, not the model, but it hid the defect for a round.
6. **A SIBLING of the landed bug exists elsewhere and is NOT fixed**, found by
   the independent reviewer: the meridional partial-cell pressure-gradient
   correction zeroes its polar rows and then divides by the v-face width, which
   on a tripolar grid is the same zero-extent wall row — that is `0.0 / 0.0`,
   not the inert `0 / x` the neighbouring comment claims.  It is UNREACHABLE on
   this lane (every NEMO-fidelity card is refused unless it selects the
   `nemo_sco` pressure gradient, which takes the other branch), so it does not
   touch this round's numbers; it is a live hazard for any OTHER tripolar
   configuration on the default pressure gradient.  Verified unreachable, not
   fixed.
7. The card's 1-D reference ladder still disagrees with NEMO by one
   representable value at levels 28-29 (round 10's OPEN item 3).
8. The independent sea-surface height still differs by up to 1.55 cm on 16,433
   of 26,640 surface cells, owned by the initial sea-ice category configuration,
   out of scope on this lane.
9. The barotropic vertex thickness is still unmeasured (round 8's OPEN item 6).
10. Two of the three sites round 10 changed still carry no gate of their own.
11. **The wide ocean-fidelity battery spans two commits** — it ran across the
   review fixes rather than at a single tip (section 10).  It is green apart
   from the listed pre-existing sea-ice red, so nothing is hidden behind this;
   a single-tip re-run is tidiness, not a blocker.
12. GitHub issue 1455 remains an operator-post action.

## Choices

ASKED: none were needed for what landed.  The landed fix is a NaN, not a
configuration value, and it adds no stabiliser NEMO lacks: it gives a face of
zero area an inverse area of zero, and is bitwise the old value wherever the
area is positive.

UNASKED: none STANDING.  One decision is RAISED rather than taken, in section 7
and in the final report: how the lateral-viscosity operator fix should be
scoped, given that its larger half is a GYRE landing.

No scheme selection, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source or previously-tolerated condition moved.

## 9. Independent review

Codex is paused, so `codex exec` was NOT run and this round claims no
independent codex verdict.  A fresh adversarial reviewer was run in its place.

### 9a. Its verdict, verbatim

> VERDICT: SHIP WITH FIXES (no blocking defects found in either landed
> statement; two non-blocking items should be closed before the round is
> called fully done)

It did not take the receipt's word for anything: it read the compiled Fortran
line by line against the transcription, traced the barotropic root cause
through the real consumer code, re-ran the GYRE byte-identity comparison and
one geometry probe live against the repo, and cross-checked the numbers
against the saved artifacts.

### 9b. What it found, and what was done

| finding | what it was | what was done |
|---|---|---|
| R1 | the closure row was scored on 1,319 MORE v cells than the row it is compared against, because an ablation can define a cell the reference transcription leaves undefined | every gate row now shares ONE cell set; the v closure improves from 0.039 to **0.019** and section 3d is re-stated on the new numbers |
| R2 | a SIBLING of the landed bug exists elsewhere — the meridional partial-cell pressure-gradient correction zeroes its polar rows and then divides by the v-face width, which on a tripolar grid is `0.0 / 0.0` | verified UNREACHABLE on this lane (every NEMO-fidelity card is refused unless it selects the `nemo_sco` pressure gradient) and registered as OPEN item 6 rather than fixed out of scope |
| R3 | the gate read the record's `fmask` and never used it | removed |
| R4 | the receipt called the replacement controls "stronger" than the preregistered uniform-case one | weakened: they catch a DIFFERENT failure class, and what establishes the arithmetic is the closure row plus this reviewer's own reading of the Fortran |
| R5 | the pre-fix numbers (the all-NaN sea surface, `drag_v` at 180 cells) could not be verified because the probe output was never saved | both probes re-run against a clean clone at the BASE commit and their output saved to the evidence directory; both numbers reproduce exactly |
| R6 | this section promised the review's findings and did not carry them | it carries them now |

What it verified INDEPENDENTLY and found correct: the whole compiled
transcription term by term, including the two subtle points — NEMO's `hf_0`
uses the product of a v mask with its eastern neighbour, and the thickness
mask freezes from the FREE-SLIP mask before `rn_shlat` rewrites it (it checked
the card's own namelist to confirm `rn_shlat = 2.0`, so that distinction is
material and not moot here); that the stretches come from the Runge-Kutta
variant of the stretch routine, matching the stage-three call; the
non-vacuity of all six new tests; that the two other unguarded reciprocals in
the changed module are genuinely safe; and every number in sections 2d, 3d, 4
and 5 against the saved artifacts, re-deriving the GYRE digest itself.

## 10. Gate and test results at the round's final tip

| check | result |
|---|---|
| ORCA2 ladder, kt = 1 to 10 | **LADDER_MEASURED**, exit 0 |
| round-11 lateral-viscosity operator gate | **REFUSES**, exit 2, every scored cell unequal |
| its ablations and its closure row | section 3d |
| GYRE identity, base vs tip | section 4 |
| DINO, lock exchange, overflow | 169 passed |
| barotropic batteries | 26 passed (isolated) plus 90 passed, 1 skipped, 1 xfailed |
| the named push gates plus this round's new tests, at the final clean tip | **132 passed in 363.02 s** |
| this round's own new tests, re-run after the review fixes | **6 passed** |
| receipt citation gate, at the final clean tip | **PASS**, 274 citations, 0 failures, 0 map-audit failures, 0 unmapped |
| citation gate with a rigid two-line plant on a REAL key | **FIRES** — exit 1, status FAIL, the planted citation named with its own line: `dynldf_lev_rot_scheme.h90:24-25`, `SYMBOL-NOT-AT-LINE`, "that symbol identifies line 24", found at line 26 |
| the wide ocean-fidelity battery, once, with twelve workers | **1 failed, 1,539 passed, 7 skipped in 2,391.69 s** — the one failure is the listed pre-existing sea-ice scalar-math provenance red, the same one rounds 9 and 10 reported, and the counts sit beside round 10's 1 failed / 1,535 passed / 7 skipped in 2,372.00 s.  **RETRACTION, in the same table row rather than quietly:** an earlier version of this receipt recorded this battery as DISCARDED on the prediction that the worktree going dirty under it mid-run (the review fixes landed while it ran) would make every worktree-stamping gate refuse, as happened to round 10's discarded run.  That prediction is REFUTED — no stamp refusal occurred.  The remaining, smaller caveat stands and is why this is not quite a final-tip number: the run spans commits `2a50d47a4` to `ac5d6008a`, so some of it executed before the review fixes and some after.  It flags nothing new either way. |
