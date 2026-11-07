# NEMO testcase Lane 4 — ORCA2 card round 10 transcription receipt

Date: 2026-09-23

Starting tip: `84ba7f9cd04d4c2b0e5aca0f81f2b73086ca2c1c`

Preregistration: `5191f0be6c48`

Status: **HELD.**  Round 9's OPEN item 1 is discharged: the three-band
chlorophyll shortwave penetration that ORCA2's namelist selects is now the
deposit the shared physics pipeline applies, routed through the RGB kernel
legoESM already carried and gated cell by cell against the record's own `qsr`
right-hand-side increment with the card's own configuration and operands.  The
ladder leaves that stop and reaches the next one, which is NOT an unbuilt
statement: the production step's post-Runge-Kutta sea-surface height is
non-finite at kt=1, so `kt=10` stays **UNMEASURED** — not zero, not
extrapolated.

Round 9's OPEN item 3 (the second masking of an already-masked viscosity
coefficient) is TRANSCRIBED and CITED here but deliberately **NOT LANDED**;
section 6 says exactly why and what decision it needs.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  Sea ice remains out of scope and the six-entry `unmeasured_features`
tuple is unchanged.

Independent review **not run (codex paused)**; a separate fresh adversarial
reviewer was run in its place (section 8).

## 1. What NEMO actually does, and what nothing here may choose

Every setting below was read from the record's own resolved output, not from a
deck comment.  The record is the pinned ORCA1-ice reference run
`orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| shortwave family | `ln_qsr_rgb = T` | `ocean.output:1207` |
| chlorophyll source | data file, `nn_chldta = 1` | `ocean.output:1211` |
| chlorophyll vertical profile | Morel-Berthon analytical, `nn_chlprfl = 1` | `ocean.output:1212` |
| infrared fraction | `rn_abs = 0.58` | `ocean.output:1213` |
| infrared attenuation length | `rn_si0 = 0.35` m | `ocean.output:1214` |
| ocean time step | `rn_Dt = 10800` s | `ocean.output:217` |
| reference density and heat capacity | `rho0 = 1026`, `rcp = 3991.8679571196299` | `ocean.output:172,174` |
| lateral momentum boundary condition | **no-slip**, `rn_shlat = 2.0` | `ocean.output:339` |

With those, `tra_qsr` dispatches `qsr_RGBc`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:213`).  Five
statements, in this order:

| statement | compiled owner |
|---|---|
| the look-up-table index per level from the Morel-Berthon profile evaluated on the LIVE interface depth `gdepw_1d(jk+1)*(1+r3t(Kmm))` | `traqsr.f90:349` |
| the surface partition: infrared `rn_abs`, three equal red/green/blue bands `(1-rn_abs)/3`, and the total | `traqsr.f90:368-374` |
| four depth ranges bounded by `nk0`, `nkR`, `nkG`, `nkB`, one band dropped at each boundary, the summed flux multiplied by the w-level mask | `traqsr.f90:379-463` |
| the deposit into the temperature right-hand side, `r1_rho0_rcp*(zeT - zzeT)/ze3t` with `ze3t = e3t_3d*(1+r3t(Kmm)*tmask)` | `traqsr.f90:388,394` |
| the deepest level light reaches, `nksr = nkV = nkB` | `traqsr.f90:1193-1195,1308` |

`tra_qsr` is called ONCE per step, at Runge-Kutta stage three, with `Kmm`.
That is the whole reason the routing needed a second site: legoESM's shared
pipeline runs before the stages.

## 2. What landed, and what it deliberately does not add

**No second RGB kernel.**  legoESM already carried `qsr_RGBc` literally, behind
`apply_shortwave_penetration(scheme="nemo_qsr_rgb")` in
`packages/ocean/legoesm/ocean/physics/shortwave_penetration.py`, used by the
external-forcing stage and gated against this record by the pre-existing
`nemo_testcase_l4_orca2_rgb_gate.py`.  Searched first (Rule 4): `grep -rn
"nemo_qsr_rgb\|apply_shortwave_penetration"` over `packages/`, `src/`,
`scripts/` and `tests/` returned that kernel, its dispatcher, the external
stage's call and the existing gate — so this round routes, it does not write.

Three sites changed, each mirroring the two-band arm that was already there:

1. The shared pipeline dispatches the RGB schemes through
   `apply_shortwave_penetration` with NEMO's live `key_qco` operands; the
   two-band call beside it is untouched and bit-identical.
2. The Runge-Kutta stage-three seam gains the RGB twin of the existing
   `qsr_2BD` substitution: the pipeline's step-entry evaluation is removed and
   the `Kmm` evaluation added.  The removal arm reuses the pipeline's OWN
   jacobian expression.  The two arms are NOT the same call — the pipeline's
   live thickness is built with no minimum-water-column floor and the seam's
   with a 0.5 m one — so the exactness of the removal is a MEASURED claim, not
   a structural one: on the ORCA2 card the two thickness fields are
   **0 of 799,200 bitwise unequal and their wet masks identical**, because
   every one of the 10,207 columns where the floor binds is dry and the
   shallowest WET column is 29.9 m.  It would stop being exact the day a wet
   column is shallower than the floor; that is registered in OPEN.
3. The external surface-forcing stage stops ALSO depositing under the NEMO RGB
   selector, exactly as it already declined to under the two-band one.  Without
   that, `qsr` would be counted twice while the column-integrated heat budget
   stayed misleadingly close through the `qns` subtraction.  Every non-NEMO
   caller keeps its generic `rgb_chl` deposit unchanged.

## 3. The gate, and the four controls that each fire

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round10_qsr_routing_gate.py`
takes nothing by hand.  The scheme, infrared fraction and length, chlorophyll
profile and time step come from the ORCA2 CARD and each is required to equal
the value section 1 reads from the record.  The reference depth ladder comes
from the card's own vertical coordinate.

| row | result |
|---|---|
| **the SHARED PHYSICS PIPELINE's own deposit** vs the record's `qsr` increment, every owned wet cell | **AT_BAR**, 0 of 233,341 unequal |
| the same gate run at the round's BASE commit | **REFUSES**, exit 2, with the pipeline's own two-band refusal text |
| the operand-level deposit vs the same record | **AT_BAR**, 0 of 233,341 unequal |
| control: the card's two-band sibling through the same dispatcher | fires, 149,842 unequal, max 4.81e-07 K/s |
| control: the reference ladder instead of the live one | fires, 148,122 unequal, max 2.29e-08 K/s |
| control: an unresolved chlorophyll profile | fires — REFUSED at the kernel, never substituted |
| control: one representable value moved on one wet cell, operand row | fires, 1 unequal |
| control: the same, on the SHARED PIPELINE row | fires, 1 unequal |

The first row is the one that BINDS on the diff.  It calls
`make_ocean_physics` — the production factory — with the card's own shortwave
configuration and every other module switched off, so the only tendency it can
return is the block this round changed.  The pipeline derives its own stretch
from the sea surface, so the state is given the sea surface that carries the
record's own `r3t`; that round trip is exact and the gate refuses if it ever
stops being (measured: 0 of 16,433 wet surface columns unequal).

**The gate was PROVEN to bind, not assumed to.**  A clean clone at the round's
base commit, with this gate dropped in unchanged, refuses: the pipeline's own
`shortwave_penetration_tendency is the two-band Jerlov kernel but got
scheme='nemo_qsr_rgb'`.  An earlier draft of this gate did NOT bind — it
re-implemented the operands and passed with the whole model diff absent — and
the independent reviewer measured that.  See section 8a.

**WHAT THIS GATE DOES NOT COVER, stated rather than implied.**  Round 10
changed THREE sites and this gate binds on ONE of them: the shared physics
pipeline.  The Runge-Kutta stage-three seam
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`) and the
external-stage suppression
(`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py`) carry NO
gate of their own — the gate never builds a model — and the seam is exactly
where section 2's exactness claim lives.  A second blind spot: the pipeline row
feeds in a sea surface constructed so the pipeline's own jacobian reproduces
the record's `r3t`, so the row cannot detect an error in how that stretch is
DERIVED, only in what is done with it.  Both are in OPEN.

Label: **given NEMO's entry** (the `qsr`, chlorophyll and `r3t` operands are
the record's own recorded frames).

### 3a. A second, independent statement the gate MEASURED rather than assumed

The card's 1-D reference ladder is not bit-identical to NEMO's:

| row | disagreeing levels | magnitude |
|---|---|---|
| `gdepw_1d` vs the record's own dumped ladder | level 29 | 9.094947e-13 m (one representable value at 4500 m) |
| `e3t_1d` vs the mesh mask | levels 28 and 29 | 9.094947e-13 m |
| `e3t_0`, the 3-D partial-cell thickness that actually enters the deposit | none | 0 of the owned wet cells |

Those two levels sit far below the blue extinction level, where NEMO's deposit
and legoESM's are BOTH exactly zero — so the AT_BAR result at those levels is a
zero-against-zero comparison.  It shows the disagreement is inert **for this
record at this step**; it is NOT a structural proof that a deeper-reaching
band could never see it.  Carried into OPEN as its own statement.

## 4. Where the ladder stops now, and the control that says it is not this round's change

With the shortwave routed, the ORCA2 ladder no longer returns
`STOP_PRODUCTION_QSR_RGB_PIPELINE_GAP`.  It reaches the next condition at
kt=1 and that condition is **not** a deliberate refusal: legoESM's own
raw-mesh guard (`packages/ocean/legoesm/ocean/eos.py:738-742`) refuses a
buoyancy-frequency divisor that is not finite and positive.

The call chain, read off a trace-time marker inserted temporarily and reverted
(`git status --porcelain` clean afterwards):

```
_step_jitted -> _step_impl -> _apply_implicit_vertical_mixing
  -> vertical_mixing/k_profiles.py:426,1260 -> convection/enhanced_diffusion.py:181
  -> eos.py:738-742
```

The divisor is the raw mesh `e3w_0` times the stretch `max(1+r3t, 1e-6)`.  The
raw field is measured clean (0 of 799,200 non-positive or non-finite) and the
floor cannot produce a non-positive value, so the guard can only fire on a
**non-finite divisor**.  That the sea-surface height itself is non-finite is
**PLAUSIBLE, not CONFIRMED**: an overflow of `e3w_0 * stretch` from a
finite-but-enormous sea surface fires the same guard, and the guard is a global
`jnp.all` so it names no cell.  The discriminating measurement is one line —
print the sea surface the implicit vertical mixing receives — and it was NOT
run this round.

**CONTROL, one variable:** the same step, same compiled graph, with the
shortwave flux set identically to zero, fires the SAME guard.  With `qsr = 0`
the RGB kernel still runs and returns exactly zero, so the step is numerically
the step without any shortwave at all.  This round's deposit is therefore NOT
what makes the guard fire; the statement was simply unreachable while the
pipeline refused earlier.  The pipeline's own RGB field is separately measured
finite, 0 NaN, 0 Inf, peak 6.874e-06 K/s (about 0.6 K/day).

That is where round 11 starts.  No magnitude is registered for kt=10 and none
is guessed.

## 5. GYRE is unchanged

The GYRE card selects `nemo_qsr_2bd` and carries no chlorophyll field, so it
takes none of the new branches.  Proven, not argued, at the round's base tip
and at its final tip with the evaluation protocol byte-identical:

| row | result |
|---|---|
| ten-step ladder, offline oracle-relative compare | **PASS** — 70 certified rows, 0 status changes, 0 violations, max worsening 0 ULPs, first-over-bar unmoved (`u`,`v` at kt=2) |
| residual arrays `np.array_equal` | 210 / 210 |
| thirty-day member, byte-identical daily snapshots | 30 / 30 |
| day-30 digest | `14a7e64b4512860e...` — the same as rounds 7, 8 and 9 |

**CORRECTION.**  An earlier draft of this receipt said DINO executes the same
RGB kernel through its own `rgb_chl` selector.  It does not: DINO's shortwave
is `ShortwavePenetrationConfig(water_type=...)`, whose scheme defaults to
`jerlov_2band`, and neither `dino.py` nor its test mentions a chlorophyll
scheme.  DINO takes the unchanged two-band branch.
`tests/ocean/unit/test_dino_experiment.py` — **128 passed** — is therefore a
REGRESSION check that the branch DINO does take is untouched, not coverage of
the RGB arm.  The RGB arm's coverage is this round's own gate and unit tests.

## 6. Round 9's OPEN item 3, transcribed and NOT landed

NEMO stores the F-point viscosity with `fmask` already folded in, and the
compiled operator says so in its own comment.  The shearing term of the
resolved divergence-vorticity laplacian is

```
zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk)
   * ( e3f_3d(ji-1,jj-1,jk) * (1._wp + r3f(ji-1,jj-1)*fe3mask(ji-1,jj-1,jk)) )
   * r1_e1e2f(ji-1,jj-1) * ( d(e2v*v) - d(e1u*u) )
```

(`fe3mask` there is the key_qco THICKNESS substitution — it gates the stretch,
not the viscosity — and is quoted in full so the sentence below is about the
viscosity's mask and nothing else.)

annotated `! ahmf already * by fmask`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf_lev.f90:123`); the
divergence twin carries `! ahmt already * by tmask` at `:127`.  **No further
mask multiplies either term.**  With `rn_shlat = 2` the stored `fmask` is not a
zero/one field: a coastal F point whose free-slip value was zero is reset to
`rn_shlat` times the neighbouring velocity-mask maximum
(`dommsk.f90:269-277`), and the strait overrides may replace it again
(`dommsk.f90:283-303`).  That non-zero coastal value IS ORCA2's no-slip lateral
boundary condition.

legoESM's shared div-curl multiplies the F-point vorticity by its own zero/one
vertex mask before applying the coefficient
(`packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py:1484-1489`),
which discards exactly those values.  Round 9's own gate already measures the
size: **48,287 of 799,200** F cells where NEMO's coefficient is non-zero and
legoESM's vertex mask zeroes it.

**The mask is not the only difference in that operator, and landing it alone
would be a partial fix.**  NEMO's `zwf` also carries the live vertex thickness
and the inverse cell area, `ahmf * e3f*(1+r3f*fe3mask) * r1_e1e2f`
(`dynldf_lev.f90:123`), and `zwt` divides by the live T thickness
(`dynldf_lev.f90:127`).  legoESM's shared div-curl applies NEITHER — its own
docstring says so, as a documented deviation
(`packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py:1426-1431`).
So the ORCA2 lateral-viscosity operator has at least TWO unmatched statements,
and the one round 9 measured is the smaller of them to name.  Any round that
lands the mask must say which of the two it closed.

It is NOT landed here, for two stated reasons:

1. **No record can gate it.**  The admitted 10-step records carry no isolated
   lateral-viscosity tendency stream — the instrumented writer list is
   `l1_dump_bt_frames`, `l1_dump_stage`, `l1_dump_rhs`, `l2_dump_zdf` and
   `l4_dump_ocean_surface_input`, and none of them emits the `dyn_ldf`
   increment (R10-P7 **CONFIRMED**).
2. **No trajectory movement could be registered either**, because the ladder
   stops at kt=1 on section 4's statement.

The fix's shape is settled and scoped: the second masking must be dropped ONLY
where NEMO's mask is already inside the coefficient, which today is the single
card selecting the file source.  Every other card, GYRE included, builds the
coefficient from the metric formula, which carries no mask, and must stay
bit-identical.  That is a decision, and it is in the final report rather than
in this diff.

## 7. Frozen predictions, resolved

| ID | outcome |
|---|---|
| R10-P1 | **CONFIRMED** — all five statements read as preregistered, and the existing kernel gate is unchanged. |
| R10-P2 | **CONFIRMED** — the refusal is gone and the ladder advances past it. |
| R10-P3 | **CONFIRMED** — 0 of 233,341 owned wet cells unequal, with the card's own configuration and the production operand expressions. |
| R10-P4 | **CONFIRMED** — section 5. |
| R10-P5 | **CONFIRMED** — `dynldf_lev.f90:123` carries `ahmf` and no mask factor, and round 9's count of 48,287 reproduces. |
| R10-P6 | **NOT TESTED** — the fix is not landed (section 6), so neither half of the prediction was measured.  It is not counted as confirmed. |
| R10-P7 | **CONFIRMED** — no lateral-viscosity tendency stream exists in the admitted records. |

## 7a. CORRECTION to rounds 8 and 9: their citation-gate plant never fired

Both receipts record a row reading "citation gate with a rigid two-line plant:
refuses, exit 2".  That is a misreading of the tool.  The gate's `--plant`
argument takes an EXACT citation key and shifts that citation's lines by two;
its exit codes are **1 when the plant correctly made the gate fail** and **2
when the plant did NOT fire**, which the source says in as many words.

Round 8 passed `--plant 2` and round 9 passed `--plant shift`.  Neither string
is a citation key, so nothing was planted, the gate passed unchanged, and exit
2 was recorded as success.  Their plant rows therefore prove NOTHING and are
withdrawn.  (Round 7's plant used a real key and did fire, status FAIL.)

This round's plant uses a real key, `BLD/ppsrc/nemo/dynhpg.f90:378,397`, and
fires: exit 1, status FAIL, the planted citation named.  The gate's own nine
built-in self-test controls fire as well.

## 8. Independent review

Codex is paused, so `codex exec` was NOT run and this round does not claim an
independent codex verdict.  A fresh adversarial reviewer was run in its place.

### 8a. Its verdict, verbatim, and what was done about it

> VERDICT: BLOCK — the round's one new gate passes with the round's code
> reverted (measured), the pipeline branch was widened to an unpreregistered
> scheme that can now double-count qsr, and the only coverage claimed for it
> does not exist.

Nine findings, every one of them this round's own.  All are fixed or recorded:

| finding | what it was | what was done |
|---|---|---|
| D1 (blocking) | the new gate passed with the whole model diff ABSENT, because it re-implemented the operands instead of running the pipeline; and its unit test asserted that copy against an inline retyping of itself | the gate now runs `make_ocean_physics` — the production factory — and the same gate at the base commit REFUSES (section 3).  The tautological test is replaced by one that asserts the gate against `compute_layer_thickness` / `compute_ocean_jacobian`, the helpers the pipeline calls |
| D2 (blocking) | the pipeline branch had been widened to the generic `rgb_chl` scheme as well, so a previously-refused configuration would deposit — and double-count with the external stage, which still deposits that scheme | narrowed to the NEMO identity selector alone; a new test pins that the generic scheme still refuses |
| D3 | the claim that DINO covers the new arm is false | corrected in section 5 |
| D4 | `traqsr.f90:386` is a bare comment; the cited statement is `:388` | fixed at all three sites, including inside the gate's own JSON |
| D5 | the removal arm's exactness was asserted, not measured | measured: 0 of 799,200, with the condition under which it would stop holding (section 2) |
| D6 | "the sea-surface height is non-finite" was stated as measured | relabelled PLAUSIBLE, with the discriminator named (section 4) |
| D7 | a `PUSH_BATTERY` placeholder, a missing section 8a, a stale JSON artifact | all three fixed; the gate artifact re-run at the final tip |
| D8 | section 3a's proof is a zero-against-zero comparison | weakened to what it actually shows |
| D9 | section 6 elided the `fe3mask` factor from the quoted line | the line is now quoted in full |

One further defect surfaced while fixing D1 and is fixed here: with no physics
module enabled at all, the pipeline seeded EVERY zero tendency from `state.u`,
which on a C-grid is one column wider than the tracer field.  Each slot now
takes its own field's shape.  Production never reached it — the lat-lon driver
hands the pipeline a cell-centred proxy in which velocity and tracer fields
share a shape — so it was reachable only from a caller passing a raw C-grid
state, which is what the new gate does.  The reviewer established that
reachability boundary; an earlier wording of this receipt overstated it.

### 8b. Second review pass, on the fixes

> VERDICT: SHIP WITH FIXES (was BLOCK).

Two residual defects, both acted on:

| finding | what it was | what was done |
|---|---|---|
| F1 | the new "the pipeline still refuses the generic scheme" test called the KERNEL, not the pipeline, so re-widening the branch left it green — the same class of defect as D1 | rewritten to call `make_ocean_physics`, and PROVEN non-vacuous: with the branch re-widened it FAILS (`DID NOT RAISE`), and the model file was restored (`git status --porcelain` shows only the test moved).  A companion test pins that the NEMO selector's branch is reachable and deposits |
| F2 | the gate binds on one of the three sites the round changed, and the receipt implied more | stated explicitly in section 3, and in OPEN |

Two smaller ones, also acted on: section 3's "the production entry point"
sentence (the framing that produced D1) is deleted, and the pipeline row now
has its own one-representable-value plant control, which fires.

The reviewer also verified independently that D1's fix binds (base commit
refuses, tip is AT_BAR), that the `eta = r3t*H` round trip is a two-rounding
round trip rather than an identity, that no second widened path was left, and
that no wrong `traqsr` citation remains.

## 9. OPEN

1. **The ladder now stops on a non-finite buoyancy-frequency divisor at kt=1**
   (section 4), inside the implicit vertical mixing.  The refusal and its call
   chain are MEASURED, and a one-variable control shows it is not this round's
   shortwave.  That the SEA-SURFACE HEIGHT is the non-finite quantity is
   PLAUSIBLE only; the discriminator (print the sea surface the implicit
   vertical mixing receives, and split wet from dry) is round 11's first job
   and needs no new record.
2. **Round 9's OPEN item 3 — the second masking — is transcribed and cited but
   not landed** (section 6).  It needs either a new instrumented NEMO
   acquisition of the `dyn_ldf` increment, or a decision that an
   operator-level bit gate against the compiled loop is the right bar here.
3. **The card's 1-D reference ladder disagrees with NEMO by one representable
   value at levels 28-29** (section 3a).  Measured, proven unable to reach the
   shortwave deposit, owner unassigned.
4. The independent sea-surface height still differs by up to 1.55 cm on 16,433
   of 26,640 surface cells, owned by the initial sea-ice category
   configuration, which is out of scope on this lane.
5. The recorded runoff tracer-source operands remain an explicit later
   boundary; this round did not reach them.
6. **The barotropic vertex thickness is still unmeasured** (round 8's OPEN item
   6, untouched here).
7. **The two-band seam's step-entry arm is not exact.**  For `nemo_qsr_2bd` the
   pipeline evaluates on the reference ladder while the stage-three
   substitution removes a live-ladder reconstruction, so the two do not cancel
   exactly and the residual is reweighted into the surface term.  The RGB arm
   landed here does not have that property (it reuses the pipeline's own
   jacobian expression).  This is a PRE-EXISTING GYRE statement, unmeasured,
   reported because it was read while transcribing beside it — it is NOT
   touched by this diff.
8. **The stage-three seam's exact removal is conditional on no WET column
   being shallower than the minimum water column** (section 2).  Measured true
   on this card today; nothing enforces it.
9. **Two of the three sites this round changed carry no gate**: the
   Runge-Kutta stage-three seam and the external-stage suppression (section
   3).  The seam owns section 2's exactness claim.
10. **The pipeline row cannot detect an error in how the stretch is DERIVED**,
   because its input sea surface is constructed to reproduce the record's
   `r3t` through that same derivation (section 3).
11. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Choices

ASKED: none were needed for what landed.  Every setting this round touched is
fixed by the record's resolved configuration and was read from it (section 1).
One decision is RAISED rather than taken, in the final report: how round 9's
OPEN item 3 should be gated before it lands (section 6).

UNASKED: none STANDING.  One was made and then REVERTED before landing, and it
is recorded rather than dropped: an earlier commit widened the pipeline's new
branch to the generic `rgb_chl` scheme as well, which would have turned a
previously-refused configuration into a silent (and double-counted) deposit.
The independent reviewer caught it; the branch now admits only the NEMO
identity selector, which is what the preregistration scoped.

Two changes that are NOT configuration and are declared here anyway:
- the pipeline's empty-module path now seeds each zero tendency from its own
  field instead of from `state.u` — a shape fix with no effect on any card
  that enables any other module, needed because the binding gate runs the
  pipeline with shortwave alone;
- the receipts of rounds 8 and 9 have their citation-gate plant rows withdrawn
  (section 7a).

No scheme selection, selector default, tunable, threshold, cadence,
resolution, timestep, carried state, data source or previously-tolerated
condition moved, and no stabiliser NEMO lacks was added.  The external stage's
non-NEMO RGB callers keep the behaviour they had.

## 10. Gate and test results at the round's final tip

| check | result |
|---|---|
| round-10 shortwave routing gate | **AT_BAR**, 0 of 233,341 unequal |
| its five controls | all five fire |
| ORCA2 ladder gate at the base tip | exit 4, `STOP_PRODUCTION_QSR_RGB_PIPELINE_GAP` at kt=1 |
| ORCA2 ladder gate at the final tip | past that stop; refused by the raw-mesh guard of section 4, no magnitude registered |
| DINO card, same kernel | 128 passed |
| GYRE identity, base vs tip | section 5 |
| receipt citation gate | **PASS**, 274 citations, 0 failures, 0 map-audit failures, 0 unmapped |
| citation gate with a rigid two-line plant | **fires**, exit 1, status FAIL, the planted citation named |
| the named push gates plus this round's new tests, at the final tip | **177 passed in 577.69 s** |
| the ocean-fidelity battery, once, with twelve workers, at the final tip | **1 failed, 1,535 passed, 7 skipped in 2,372.00 s** -- the one failure is the listed pre-existing sea-ice scalar-math provenance red |
| a first run of that battery, discarded | started against a DIRTY tree (an uncommitted receipt edit), so every gate that stamps the worktree refused and reported 24 failures; re-run clean, the number is the row above.  Recorded rather than dropped |
| the new refusal test with the pipeline branch re-widened | **fails** (`DID NOT RAISE`), then the model file restored and the tree confirmed clean |
| the same gate at the round's BASE commit, in a clean clone | **refuses**, exit 2 |
