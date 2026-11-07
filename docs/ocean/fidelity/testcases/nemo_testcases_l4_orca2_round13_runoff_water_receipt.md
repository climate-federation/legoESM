# NEMO testcase Lane 4 — ORCA2 card round 13 transcription receipt

Date: 2026-09-23

Starting tip: `638a18efbc1843690a816db8beb0efbc0a880f1d`.

Preregistration: `4fa143867`

Status: **LANDED-READY.**

Round 12's finding was that NEMO's river-runoff heat, landed bit-exactly, moved
the river-mouth cells TEN TIMES further from NEMO, and that the one arm which
tested the obvious explanation was confounded.  **The explanation was right.**
Pairing the runoff's WATER with its heat returns those cells to where they were
before the heat landed — **1.5287e-03 degC back to 1.4823e-04 degC** — and the
salinity row of the whole ladder improves by a factor of **5.7**.

The water needed no new operator.  legoESM already carried both of NEMO's
owners for it, and exactly ONE model statement was wrong: the tracer
concentration/dilution term was using legoESM's whole net freshwater flux,
which includes the runoff, where NEMO's `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:282-286` reads `emp`, which
never does.  That single wrong operand is why round 12's pairing arm barely
moved: it deposited a second, nearly identical copy of the runoff's own heat
instead of the compensating volume.

Statement B does NOT name an owner, and the first draft of this receipt said it
did — **retracted here** (section 5e).  What it establishes is narrower and
still useful: three downstream statements are bit-exact against this record,
the ladder's sea-surface rows carry no stage-specific content, and the stage-1
velocity disagreement is a DEPTH-UNIFORM offset (median spread 1.7e-10 of the
column mean), which is the signature of the two-dimensional barotropic
correction and not of any three-dimensional momentum operator.  The barotropic
external mode is the leading candidate, labelled PLAUSIBLE, with the
discriminating substitution named for round 14.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  Sea ice remains out of scope and the six-entry `unmeasured_features`
tuple is unchanged.

Independent review **not run (codex is paused)**; a fresh adversarial reviewer
was run in its place (section 8).

## 1. What the record fixes, and what nothing here may choose

| resolved setting | value | where it is printed |
|---|---|---|
| river runoff | ON, `ln_rnf = T` | run `ocean.output:534` |
| runoff river-mouth treatment | `ln_rnf_mouth = T` | run `ocean.output:648` |
| runoff multiplier | `rn_rfact = 1.0` | run `ocean.output:651` |
| runoff DEPTH spreading | NOT selected — neither `ln_rnf_depth`'s nor `ln_rnf_depth_ini`'s banner appears | run `ocean.output:646-667` |
| closed seas | OFF, `ln_closea = F` | run `ocean.output:118` |
| freshwater-budget control | `nn_fwb = 2`, "volume adjusted from previous year budget" | run `ocean.output:532,1652` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

Two of those rows exist to close doors rather than open them.  `ln_closea = F`
removes the only compiled path that would move runoff between cells, and
`nn_fwb = 2` adds a single SCALAR correction to `emp` while never touching
`rnf`, so the record's own `emp` frames already carry it and no new statement
is owed for it.  **No new NEMO run was made or needed.**

## 2. Statement A — the runoff's water, as the compiled source spells it

### 2a. Five statements, and which of them legoESM already had

| # | what NEMO does | compiled owner | legoESM owner |
|---|---|---|---|
| A1 | adds the runoff to the three-dimensional horizontal divergence, once per `div_hor` call | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/divhor.f90:142` | ALREADY THERE |
| A2 | the surface arm: `h_rnf` becomes the live top-cell thickness and `phdivn(:,:,1)` is DECREASED by `rnf * r1_rho0 /` that same thickness | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcrnf.f90:279-282` | ALREADY THERE |
| A3 | the barotropic sea-surface forcing is `r1_rho0 * ( emp - rnf )` | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:278-281` | ALREADY THERE |
| A4 | the stage-1/2 concentration/dilution term reads `emp` **ALONE** | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:282-286` | **WRONG — this is the round's landing** |
| A5 | the tracer content the runoff carries, at all three stages | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:318-324` | round 12's landing |

`sshwzv.f90` also spells a surface freshwater term and it is **NOT cited as
executing**: `grep` over the whole compiled branch finds no `CALL ssh_nxt`, so
that routine is dead in this RK3 build.

### 2b. Rule 4 — what was searched before anything was written

Searched `runoff`, `rnf`, `runoff_mass_flux`, `emp` and `FreshwaterForcing`
across `packages/ocean/legoesm/ocean/` and
`scripts/validate/ocean_fidelity/`.  Five things already existed and are
REUSED rather than rebuilt:

| for | what already existed |
|---|---|
| A1 and A2 | the `runoff_mass_flux` branch of `nemo_transport_wzv_divergence_level` (`ocean_pe_latlon_cgrid.py:1577-1585`), already cited to `sbcrnf.F90:253-260` |
| its production wiring | `ocean_model_latlon_cgrid.py:6330-6331`, which already passes `freshwater.runoff` into the stage transport |
| A3 | `ocean_model_latlon_cgrid.py:5725-5727`, whose net is `precip - evap + runoff + ice_fw` — algebraically `-(emp - rnf)` on this card |
| a bit-exact gate for A1/A2 against NEMO's own `ww` and `pFw` | `nemo_testcase_l4_orca2_wzv_gate.py` |
| the production-binding stage-source harness and its water arm | `nemo_testcase_l4_orca2_round12_runoff_gate.py` |

So the card supplies the recorded `rnf` as `FreshwaterForcing.runoff` and both
of NEMO's water owners are reached.  **No new operator was written for the
water.**

### 2c. The one statement that is landed, and the spelling that failed first

legoESM's dilution operand was `net_freshwater_flux(freshwater)`, which carries
the runoff.  With it in, the dilution deposits `rnf*T_top/rho0/h` while the
runoff source deposits `MAX(sst,0)*rnf/rho0/h` — the same quantity twice, since
`sst_m` and the top-cell temperature agree to within the step.  The net
freshwater helper now takes the runoff OUT OF THE SUM when the caller asks for
NEMO's `emp`, and the dilution call site asks for it unconditionally.

**The first spelling of that fix was wrong and the gate caught it.**
Subtracting the runoff from the assembled sum is not the same as never adding
it: `(x + r) - r` is not bitwise `x`, and the stage-1 rate still moved on 157
temperature cells and 328 salinity cells of 799,200, by about 2e-22.  Nothing
physically, everything to a bitwise bar.  Recorded here because it is exactly
the class of defect this campaign's bars exist to catch.

The default keeps the runoff, and that is not a default preserving a bug: the
sea-surface and volume channel genuinely needs it (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:278-281` is
`emp - rnf`).  Only the dilution asks for it to be left out.

### 2d. The gate

| row | result |
|---|---|
| stage-1 tracer source rate, unmoved by the water — temperature | **0 / 799,200 unequal**, max 0.0 |
| stage-1 tracer source rate, unmoved by the water — salinity | **0 / 799,200 unequal**, max 0.0 |
| the gate at the round's BASE commit | **REFUSES** — 4,649 cells each, max 1.5480e-06 (T) and 1.8826e-06 (S), and 4,649 is exactly the number of surface cells carrying a non-zero runoff |
| one-representable-value plant | **FIRES** — "PLANT FIRED: the gate refuses a one-representable-value move", 1 / 799,200, max 2.1176e-22 |
| control — is the stage-1 comparison controlled? | **EXECUTED and printed**: "CONTROL stage-1 thickness identical in both arms: True", max difference 0.0 |
| control — is the water inert? | **EXECUTED and printed**: "CONTROL the water moves the stage-2 thickness on 15588 cells", max 5.0226e-04 m |
| stages 2 and 3 | reported as LEGALLY MOVED (15,585 and 261,829 cells), never scored — the water changed the sea surface, so those stages divide by a different thickness |

The base-commit refusal is the measurement that names the mechanism, not just a
gate control: the double deposit is on EXACTLY the runoff cells and its size,
1.5e-06 K/s, is the runoff heat source's own size.

Label: **given NEMO's entry** (the kt=1 recorded state and the recorded
`rnf`, `rnf_tsc` and `emp` frames).

### 2e. The water's divergence half, bit-exact against NEMO's own record

The existing WZV gate scores legoESM's divergence, its runoff term and the
resulting vertical velocity against the record's own `ww` and `pFw`.

| row | result |
|---|---|
| divergence | **AT BAR** — 0 / 233,341, max ULP 0 |
| runoff | **AT BAR** — 0 / 233,341, max ULP 0 |
| `ww` against NEMO's own | **AT BAR** — 0 / 233,341, max ULP 0 |
| `pFw` against NEMO's own | **AT BAR** — 0 / 233,341, max ULP 0 |
| ablation — omit the runoff from the divergence | **REFUSES** — 58,534 / 233,341 unequal, max 1.6993e-07 m/s |

The ablation is what makes the four bars non-vacuous: NEMO's own vertical
velocity moves on a quarter of its cells when the runoff is left out.

## 3. The discriminator, and round 12's hypothesis

Stage-1 temperature against NEMO's recorded stage-1 frame, on the top cell of
the 2,050 river-mouth columns the rank-0 half carries.

| arm | river-mouth top cell, max | whole field, max |
|---|---|---|
| neither the heat nor the water (round 11) | 1.4655e-04 degC | 1.4764e-03 degC |
| the heat alone (round 12) | **1.5287e-03 degC** | 1.5287e-03 degC |
| the heat AND the water (this round) | **1.4823e-04 degC** | 1.4770e-03 degC |

Preregistered threshold: below 3.0e-04 degC confirms, at or above 1.0e-03
refutes.  **CONFIRMED.**

The physics is why it lands there rather than at zero: NEMO adds river water at
the sea surface temperature and adds its volume, so the two cancel in the
temperature almost exactly.  Round 11 had NEITHER and was accidentally close;
round 12 had one half and was ten times worse; this round has both and is back,
with the statement now transcribed rather than absent.

**The salinity is where the water pays.**  NEMO's runoff carries no salt at all
(`zrnf_sal = 0`), so its whole effect on salinity is dilution by volume — a
statement legoESM simply did not have.  The kt=1 stage-1 salinity row improves
by a factor of **5.7**, from 1.8538e-03 to 3.2773e-04.

## 4. What the ladder now says

Still `LADDER_MEASURED`, kt = 1 to 10, exit 0.  Every moved row is registered.

kt = 1, stage 1, ranked by magnitude:

| field | round 12 | round 13 | unequal / scored |
|---|---|---|---|
| sea surface | 8.1656e-02 m | **8.1614e-02 m** | 8,794 / 13,320 |
| zonal velocity | 6.4143e-02 m/s | 6.4192e-02 m/s | 247,035 / 399,600 |
| meridional velocity | 3.3963e-02 m/s | 3.4006e-02 m/s | 237,822 / 399,600 |
| temperature | 1.5287e-03 degC | **1.4770e-03 degC** | 233,341 / 399,600 |
| salinity | 1.8538e-03 | **3.2773e-04** | 233,341 / 399,600 |

kt = 10, the same fields:

| row | round 12 | round 13 |
|---|---|---|
| entry temperature | 3.943541 degC on 430,552 | **3.943079** on 430,552 |
| entry salinity | 1.438947 on 430,552 | **1.436946** on 430,552 |
| entry sea surface | 5.3373e-01 m on 16,433 | **5.3043e-01** on 16,433 |
| stage-3 temperature | 0.771276 degC on 233,341 | 0.772665 on 233,341 |

Registered: the two velocity rows and the kt=10 stage rows worsen by about a
tenth of a percent, while the sea surface, both tracers at kt=1 and both
tracers at the kt=10 entry improve.  Nothing leaves the bar and the
first-over-bar statement does not move earlier.

First non-bit statement: kt=1 stage 1, still **UNATTRIBUTED** by the ladder's
own rule — the runoff cannot reach 231,291 of the row's 233,341 cells, which is
what round 12 measured and this round does not change.  Statement B names the
owner the ladder's field order hides.

Label: `INDEPENDENT_WITH_DECISION52_SSH`.

## 5. Statement B — one owner for the three largest rows

### 5a. What is already bit-exact against this record

| gate | result |
|---|---|
| stage-1 transport (`zFu`, `zFv`) | **AT BAR** — U 0 / 226,236, V 0 / 226,637, max ULP 0 |
| divergence, runoff, `ww`, `pFw` | **AT BAR** — 0 / 233,341 on each |
| hydrostatic pressure gradient (`zhpi`, `zuap`, their sum, both faces) | **AT BAR** — 0 unequal on all six rows |

So the disagreement is owned UPSTREAM of the stage program, not inside it.

### 5b. The sea-surface rows carry no stage-specific content — and that is WEAKER than it looks

NEMO's stage sea surfaces are the step's end-of-step value interpolated at 1/3,
1/2 and 1 (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:137,152-154`),
and so are legoESM's.  If the three stage disagreements stand in the ratio
1 : 1.5 : 3, the whole disagreement is inherited from the end-of-step value.

| stage | max abs |
|---|---|
| 1 | 8.1614e-02 m |
| 2 | 1.2242e-01 m |
| 3 | 2.4484e-01 m |

Measured ratios 1.0, 1.4999999999999996, 2.9999999999999996 — worst relative
departure from the prediction **3.0e-16**, machine precision.

**WHAT THAT DOES AND DOES NOT SHOW, and the first draft of this receipt got it
wrong.**  Once the step-entry sea surface agrees bitwise — it does at kt=1 —
and both sides interpolate the same weights, the stage difference equals
`w_i` times the end-of-step difference ALGEBRAICALLY, for any cause whatever.
So the ratio CONFIRMS only that legoESM adds no per-stage sea-surface source of
its own and that the two interpolations agree; it attributes nothing.  The
useful number it carries is the size: legoESM's end-of-step sea surface differs
from NEMO's by **0.2448 m** at kt=1 on the rank-0 half, and that single number
is what the three stage rows are.

### 5c. The velocities are the same owner

Each stage velocity carries the barotropic correction
`un_adv/hu(Kmm) - uu_b(Kmm)` (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:270-277`), ONE two-dimensional
field added to every level of a column.  A three-dimensional momentum operator
has no reason to be constant down a column; this correction has no choice.

| field | columns scored | median spread / column mean | columns under 1 percent | columns where the spread exceeds the mean |
|---|---|---|---|---|
| zonal velocity | 8,568 | **1.70e-10** | 6,897 | 34 |
| meridional velocity | 8,589 | **1.65e-10** | 7,120 | 50 |

**CONFIRMED: the stage-1 velocity disagreement is a depth-uniform offset** on
the overwhelming majority of columns.  The tail is the expected artefact of
dividing by a near-zero column mean and is reported rather than trimmed.

### 5d. The verdict, and what round 14 must run

**CONFIRMED**: the stage sea-surface rows carry no per-stage source and reduce
to one end-of-step number, 0.2448 m; the stage-1 velocity disagreement is
depth-uniform on the great majority of columns; the stage transport, the
divergence/WZV recurrence and the pressure gradient are each bit-exact against
this record.

**PLAUSIBLE**: the barotropic external mode owns all three of the largest kt=1
rows.  Two reasons it is not CONFIRMED.  "Depth-uniform" is necessary for the
barotropic correction and not sufficient — a depth-uniform forcing error would
look the same.  And the sea-surface ratio, which the first draft of this
receipt read as an attribution, is algebraic rather than physical (section 5b),
so it cannot separate the barotropic solver from anything else upstream that
reaches only the end-of-step sea surface.

**The discriminating measurement, named for round 14**: the record carries
`oracle_slow_forcing_kt00000001.bin`, `oracle_bt_frames_kt00000001.bin`,
`oracle_bt_substeps_kt00000001.bin` and `oracle_bt_ordered_operands_kt00000001.bin`.
Substitute NEMO's own slow forcing into legoESM's barotropic solve and
re-measure the end-of-step sea surface.  If it closes, the owner is the
depth-mean momentum tendency the solver RECEIVES; if it does not, the owner is
the solver itself.  No new NEMO run is needed for either arm.

## 6. Frozen predictions, resolved

| ID | outcome |
|---|---|
| R13-P1 | **CONFIRMED** — both legoESM owners of the water existed at the base tip, and supplying the recorded runoff moves the stage-2 live thickness on 15,588 cells. |
| R13-P2 | **CONFIRMED** — the base commit refuses on 4,649 cells, exactly the runoff cells; the tip is 0 / 799,200. |
| R13-P3 | **CONFIRMED** — 1.4823e-04 degC, below the preregistered 3.0e-04 threshold and a factor of 10.3 below the heat-only arm. |
| R13-P4 | **CONFIRMED** — the sea-surface row moved by 0.05 percent (8.1656e-02 to 8.1614e-02 m); the runoff does not own it. |
| R13-P5 | **CONFIRMED** — the two spellings of the density reciprocal differ on 5,981 of 26,640 cells, max 4.2352e-22. Reported, not landed (section 9). |
| R13-P6 | **CONFIRMED** — GYRE is bit-identical (section 7). |
| R13-P7 | **CONFIRMED** — `LADDER_MEASURED` kt=1..10, first statement still UNATTRIBUTED at kt=1 stage 1. |
| R13-P8 | **CONFIRMED** — the stage-1 transport, the WZV recurrence and the pressure gradient are each AT BAR, so the owner is upstream of them. |

### 5e. RETRACTION

The first draft of this receipt wrote that statement B "narrowed the ladder's
first disagreement to ONE owner: the barotropic external mode's end-of-step sea
surface", and labelled the sea-surface ratio CONFIRMED evidence for it.  **That
is withdrawn.**  The ratio is algebraic, not physical: with the step-entry sea
surface bitwise equal and both sides interpolating the same weights it holds
for ANY end-of-step difference.  The corrected reading is in sections 5b and
5d — the ratio rules a per-stage source OUT, the depth-uniformity is real
evidence about the velocities, and the barotropic owner is PLAUSIBLE and
unmeasured until round 14 runs the substitution in section 5d.  The
`nemo_testcase_l4_orca2_round13_stage1_owner_probe.py` instrument was changed
to say so where it computes the number and to stop printing "ONE_QUANTITY" as
its verdict, because a probe that prints its own conclusion gets that
conclusion quoted back as evidence.


## 7. GYRE is unchanged

Proven at the round's base tip (`638a18efb`) and at its final tip, with the
evaluation protocol byte-identical.

| row | result |
|---|---|
| ten-step ladder, offline oracle-relative compare | **PASS** — 70 certified rows, first-over-bar unmoved (`u`, `v` at kt=2) |
| largest oracle-residual worsening | **0.0 ULP** |
| residual arrays, elementwise equal | **210 / 210** |
| thirty-day member, byte-identical daily snapshots | **30 / 30** |
| day-30 digest | `14a7e64b4512860e...` — the same as rounds 7 through 12 |

As round 12 recorded, the comparator's `max_ulp_worsening` field reads 2 on
runs proven byte-identical; the field that means what this table says is
`largest_oracle_residual_worsening_ulps`, quoted above.

**Which tip GYRE was proven at, stated rather than implied.**  The candidate
arm ran at `00e5201ce`, which already carries BOTH model commits of this round.
`git diff --name-only 00e5201ce..HEAD` lists no file under `packages/`: every
later commit is a gate, a probe, a test, a citation map or this receipt.  So
the GYRE proof covers the landed model statement in full, and nothing after it
could move a trajectory.

**Which cards execute the changed code.**  The changed statement is inside the
WS-RK3 branch of the shared lat-lon ocean model, which GYRE, DINO,
LOCK_EXCHANGE, OVERFLOW and ORCA2 all select, so all five execute it.  Only
ORCA2 supplies a non-zero runoff, so only ORCA2's numbers move; the other four
subtract nothing and are bitwise unchanged, which is what the GYRE identity
above and the 169-test card battery measure.

## 8. Independent review

Codex is paused, so `codex exec` was NOT run and this round claims no
independent codex verdict.  A fresh adversarial reviewer was run in its place.
It read the compiled Fortran itself, traced `freshwater.runoff` to both of its
consumers, enumerated every caller of the changed helper, and checked the
plant, the revert evidence and a sample of the citation shifts.

### 8a. Its verdict, verbatim

> SHIP WITH FIXES

### 8b. What it found, and what was done

| finding | what it was | what was done |
|---|---|---|
| 1 (HIGH) | the saved gate JSON did not contain the reachability control, so the published discriminator numbers came from a run that predated it — "controls that never ran on the cited artifact", a failure this campaign has hit before | the gate was re-run at the control's own commit; the JSON now carries the row (`wzv_call2_evaluation: nemo_literal`, `freshwater_closure: real_freshwater`, both owners reachable) and the discriminator numbers are UNCHANGED at 1.5287e-03 and 1.4823e-04.  **CLOSED by re-measurement** |
| 2 (HIGH) | the sea-surface ratio test is close to true BY CONSTRUCTION, not an independent discriminator, and the receipt's "narrowed to ONE owner" overclaims it | **accepted in full and RETRACTED** (section 5e).  The receipt's headline, section 5b and section 5d now say the ratio is algebraic, rules a per-stage source out and attributes nothing; the probe says so where it computes the number and no longer prints "ONE_QUANTITY" as a verdict |
| 3 (LOW) | `emp` was verified free of runoff on the cited paths and in the `nn_fwb = 2` arm, but `sbcmod`/`sbcblk`/`sbcice_*` were not read | scope stated: the receipt claims the cited paths and the freshwater-budget arm, nothing wider.  Carried into OPEN |

What it verified INDEPENDENTLY and found correct: that the surface arm of
`sbc_rnf_div` is the one that runs and that legoESM's divergence branch matches
its multiply-then-divide order; that `zfact = 0.5*r1_rho0` is assigned and
never read in this build, so the receipt is right to ignore it; that the
`nn_fwb = 2` correction is a spatially uniform scalar that never touches `rnf`;
that every other caller of the changed helper wants the runoff included, so the
default preserves no bug; that the plant fires and the revert makes two real
tests fail; and that a sample of the twenty-four citation shifts is a uniform
+20 with preserved block lengths.

## 9. Gate and test results at the round's final tip

| check | result |
|---|---|
| ORCA2 ladder, kt = 1 to 10 | **LADDER_MEASURED**, exit 0 |
| round-13 runoff-water gate | **AT BAR**, exit 0 |
| its base-commit control | **REFUSES**, exit 2 — 4,649 cells each |
| its one-representable-value plant | **FIRES**, exit 1 |
| round-12 runoff gate, re-run at this tip | **AT BAR**, exit 0 — 13 of its 14 rows reproduce EXACTLY (section 9a) |
| WZV / stage-1 transport / HPG gates | **AT BAR**, exit 0 each |
| stage-1 owner probe | exit 0 — one quantity, depth-uniform |
| GYRE identity, base vs tip | section 7 |
| receipt citation gate | **PASS**, 274 citations, 0 failures, 0 unmapped, 0 map-audit failures |
| citation gate plant on a REAL key the receipt RENDERS | **FIRES** — exit 1, status FAIL, `ocean_model_latlon_cgrid.py:6924`, `SYMBOL-NOT-AT-LINE`, "that symbol identifies line 6924", checked against line 6926 |

**The citation map now covers this round's compiled statements.**  Round 12
recorded that a citation planted only in the map does not fire, and the
converse blind spot is worse: the ORCA2 receipts' own compiled citations were
never in the map at all, so nothing checked them.  All seven of this round's
are now map entries against the record's OWN compiled branch, and `audit_map`
verifies every one of them on every invocation — that is what the "0
map-audit failures" line above is worth.

The model statement added twenty lines to the lat-lon ocean model, which moved
twenty-four legoESM citations.  Every one was re-anchored RIGIDLY at the same
measured offset of +20 with its extent unchanged, and the six that the GYRE
round-8 receipt renders in prose were moved with them.

A second half of that coverage is closed here too: `audit_map` checks that a
map KEY resolves, not that the receipt RENDERS that key, so a receipt could
cite `:279-283` while the map pinned `:279-282` and nothing would notice.  A
new test extracts every compiled citation this receipt renders, requires each
to be a map key, re-runs the gate's own `check` on it, and requires a two-line
shift to make each one FAIL.  14 passed.

### 9a. Round 12's gate, re-run at this tip

The preregistration's stop rule: round 12's existing rows must reproduce
EXACTLY, or the change altered the instrument.  They do — 13 of its 14 rows are
byte-identical to round 12's saved JSON, including all six bitwise stage rows,
the runoff depth operand, the reciprocal-first discriminability row and the
wrong-stage-thickness control.

The fourteenth row is the one that MUST move, and it is round 12's own
confounded pairing arm: `stage1_temperature_row_with_the_mass_paired`, whose
river-mouth maximum goes from **1.4990e-03 degC to 1.4823e-04 degC**.  That is
this round's discriminator reproduced through round 12's instrument rather than
this round's, which is the strongest form the check can take.

### 9b. Batteries

| battery | result |
|---|---|
| the operator push list plus this round's two model-statement files | **139 passed in 657.06 s** |
| DINO, lock exchange and overflow | **169 passed in 490.77 s** |
| the freshwater helper's own suite plus the receipt-citation test | **66 passed in 68.57 s** |
| this round's tests with the landed statement REVERTED | **2 failed, 5 passed** — both model-level tests fail with "max 0", the bitwise-identical arms the reverted spelling predicts; the tree was then restored and `git status --porcelain` printed nothing |

A first attempt at the push battery aborted mid-run inside
`test_freshwater.py` with a JAX compile abort (`Fatal Python error: Aborted`,
exit 134) after every operator file had already passed.  Re-run in isolation it
is green, which is the same class round 11 recorded.  The numbers above are
from the clean re-runs, and the abort is reported rather than hidden because a
non-zero exit that is NOT a test failure is exactly the thing this campaign
refuses to read as success.

## Choices

ASKED: the round's order asked for the runoff's water to be transcribed and
gated, for the river-mouth cells to be re-measured with the confound removed,
and for the kt=1 rows to be ranked and attributed.  All three were followed.

UNASKED: none STANDING.  Two decisions are RAISED rather than taken:

1. **Decision 56 now has a measurement.**  Round 12 asked whether to keep the
   heat landing or hold it until its partner landed.  The partner has landed
   and the cells are back: keeping it is the answer the measurement gives.
2. **The density reciprocal's spelling** (section 6, R13-P5) is a real
   difference on 5,981 cells at 4e-22.  Changing it would move every card that
   carries a surface freshwater flux, GYRE included, so it is REPORTED and NOT
   LANDED.

No scheme selection, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source or previously-tolerated condition moved.
The runoff temperature and salinity FILES, the depth spreading and the
freshwater-budget correction were NOT switched on.

## 10. OPEN — round 14's order

1. **Split the barotropic owner — and note it is PLAUSIBLE, not named.**  Substitute the record's own
   `oracle_slow_forcing_kt00000001.bin` into legoESM's barotropic solve and
   re-measure the end-of-step sea surface (section 5d).  That one substitution
   separates "the solver" from "what the solver is handed", and it is the
   largest row on the ladder.
2. **The density reciprocal spelling** (`stp2d.f90:281` multiplies by
   `r1_rho0`; legoESM divides by `rho_0`), 5,981 cells at 4e-22.  Not landed
   because it would move GYRE.
3. **Decision 54 is still three-part and still pending**: the extra vertex
   mask (scopeable to ORCA2), the face and vertex thicknesses (not scopeable),
   and the vertex CELL AREA (not scopeable, largest by residual).  Nothing
   lands until it is settled.
4. **The vertex-cell-area docstring** still estimates the two conventions'
   gap at 4.1e-05, three orders of magnitude low for ORCA2 (median 5.4
   percent).  Round 12's item, uncorrected.
5. **The runoff depth operand is cross-checked against NEMO at stage 1 only**;
   stages 2 and 3 divide by legoESM's own intermediate thicknesses, unchecked.
6. **A citation planted only in the gate's map does not fire**, and until this
   round the ORCA2 receipts' compiled citations were not in the map at all.
   Seven are now; the rest of rounds 1 through 12 are still hand-checked only.
7. Round 11's carried items: the tripolar `0.0 / 0.0` in the meridional
   partial-cell pressure gradient (unreachable here, live for other tripolar
   configurations); the 1-D reference ladder's one-representable-value
   disagreement at levels 28-29; the barotropic vertex thickness, still
   unmeasured; the independent sea-surface height's 1.55 cm on 16,433 cells,
   owned by the initial sea-ice category configuration and out of scope.
8. **`emp`'s freedom from runoff is verified on the CITED paths only** — the
   barotropic forcing, the dilution term and the `nn_fwb = 2` arm.  The
   modules that BUILD `emp` (`sbcmod`, `sbcblk`, `sbcice_*`) were not read.
   Nothing in this round depends on them, because the card is driven by the
   record's own `emp` frames, but an independent surface-forcing lane would.
9. The wide ocean-fidelity battery has not been run at this round's final tip.
   Operator action.
