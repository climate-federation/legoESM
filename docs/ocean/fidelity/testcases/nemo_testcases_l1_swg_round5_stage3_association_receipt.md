# NEMO SWG level-1 round 5 — stage-3 source association

Date: 2026-10-11. Base: `3757d3eedcf3b61ab9fc758702f701f3908aab61`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round5`.

## Preregistration

This section is frozen before any round-5 implementation or measurement. The
operator's independent round-4 review reproduced the science rows and returned
DO NOT SHIP on two controls. This round first repairs those controls, then walks
the remaining `kt=2, stage=3` U debt in the operator-prescribed source order.

Frozen predictions and falsifiers:

1. **Round-4 one-ULP control.** One admitted, nonzero, wet `u_k` value from the
   recorded `kt=2, stage=1` entry will be copied unchanged except for exactly
   one `numpy.nextafter(value, +inf)` step. The same JIT production stage will
   be run for the baseline and planted entries, and a scored output must change
   bit-for-bit, report DEBT, and make the plant command exit nonzero. More than
   one input ULP, a synthetic replacement value, an unchanged output, or a
   green exit **REFUTES** the control. The former exact-zero and exact-one arms
   are withdrawn: the former cannot bind after subnormal flushing and the
   latter did not perturb the admitted value.
2. **Round-4 fp64 dispatch control.** The viscosity-source test will set and
   verify the campaign fp64 precision policy itself and compare the dispatched
   coefficient arrays, or use a manufactured nonuniform-metric case whose two
   coefficients differ. It must pass both alone and after the ladder has set
   fp64. A test whose verdict changes with order or precision **REFUTES** the
   control.
3. **Arm A — stage-3 RHS association.** NEMO accumulates `dyn_adv`, `dyn_vor`,
   and `dyn_ldf` into `uu/vv(Nrhs)` at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:312-317`; forms
   `z1_2rho0 = 0.5*r1_rho0`, pressure, centred wind, and before-velocity drag
   in the operand order at `:319-328`; adds that scalar once to `Nrhs` at
   `:329-331`; and only then updates from `Nbb` at `:342-345`. legoESM will
   reproduce that grouping and operand order, including `(utau_b+utauU)`
   before division. Prediction: `given.kt2.s3.u_k` becomes bit-identical and
   AT-BAR. Any unequal wet U value **REFUTES** Arm A and triggers Arm B.
4. **Arm B — wind operands, conditional on Arm A failing.** The admitted P3
   records' `utau_before/utau_now` and `vtau_before/vtau_now` will be compared
   directly at `kt=2, stage=3`, and the production replay will be shown to read
   those two distinct fields. NEMO forms the U/V stresses at compiled
   `sbcmod.f90:553-560`, initializes the before fields from the current fields
   on a non-restart deck at `:563-585`, and consumes both at compiled
   `stprk3.f90:325-328`. Prediction: each recorded before/current pair is
   bit-identical on this steady deck and legoESM reads each once. A differing
   pair or a duplicated operand **CONFIRMS** the wind arm; equality and correct
   dataflow **REFUTE** wind as the first owner and trigger Arm C.
5. **Arm C — drag operand, conditional on Arms A-B failing.** The production
   drag input will be compared bit-for-bit with the whole-step `Nbb` velocity
   recorded at `kt=2, stage=1`, not the stage-3 entry velocity. Compiled
   `stprk3.f90:326,328` multiplies `rn_rfr` by `uu/vv(Nbb)`, and the final
   update also starts from `Nbb` at `:344-345`. A stage-entry drag operand
   **CONFIRMS** this arm; the recorded whole-step entry **REFUTES** it and
   triggers Arm D.
6. **Arm D — viscosity body, conditional on Arms A-C failing.** The shared
   operator will be aligned statement-by-statement with compiled
   `dynldf.f90:81-92` and `dynldf_lev.f90:130-150`: curl and divergence read
   `Kbb`; curl thickness reads `r3f`; divergence thickness reads `r3t/u/v` at
   `Kbb`; output curl denominators read `r3u/v` at `Kmm`; and the final U/V
   mask is applied before accumulation. Any differing time level, mask,
   metric, or association **CONFIRMS** the arm. A clean alignment **REFUTES**
   it as an attribution.
7. **Term-record stop condition.** If Arms A-D leave `given.kt2.s3.u_k` in
   DEBT, this round will add a passive, admitted P3T acquisition under a new
   `SWG_OMIP_L1_P3T` target. It will record `uu/vv(Nrhs)` after `dyn_adv`,
   `dyn_vor`, `dyn_ldf`, and the final `zrhs` accumulation, plus `zrhs_u/v`
   before accumulation, at `kt=2, stage=3`. Admission must reject malformed or
   nonfinite records and require the instrumented run's own NEMO outputs to be
   bit-identical to the admitted P3 reference. The runner will print a named
   `REFUSE` line before every nonzero exit. No NEMO run will be attempted here.
8. **Rule 12.** Given-entry, chained, and independent ladders will be scored
   with the unchanged fp64, wet/staggered-mask, bit-for-bit gate. No AT-BAR row
   may leave the bar, neither first-over-bar boundary may move earlier, and all
   moved rows will be registered. TSUNAMI, VORTEX, VORTEX_VEC, GYRE,
   LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 will be remeasured where they can
   execute the changed shared statement or shown not to select `swe_rk3`.
   Any regression **REFUTES** landing.
9. The receipt-citation gate must map every compiled-source citation, and its
   shifted-citation plant must exit nonzero. Focused card, stage, dispatch,
   differentiability, geometry, and control tests must pass. A citation plant
   that exits zero, an ineffective numerical plant, or a moved certified card
   hash **REFUTES** landing.

No configuration or carried-state choice is introduced. The stage program,
source order, stress fields, drag, viscosity, masks, gravity, and timestep are
fixed by the shipped deck and the already resolved SWG card.

## Independent review

Independent review run by the operator.

## Measurement

Preregistration commit: `cb5d77be30754261980e0b6f8c0e0ea593ba5aa1`.
Clean measured implementation and tests: `184595fd729f6fc48eb03ab70fc567840038825c`.

1. **CONFIRMED.** The only numerical plant now starts from the admitted
   `kt=2, stage=1` entry, selects its smallest-magnitude nonzero wet U value
   (`-2.755605959386421e-05` at native cell `[40,1]`), and changes it to
   `-2.7556059593864208e-05` with exactly one `nextafter` step. The JIT
   production stage changes two scored SSH cells by at most
   `1.091393642127514e-19`; the row reports DEBT and the command exits 1.
   Artifact `swg_recorded_ulp_plant.json`, SHA-256
   `37aecbe0e4b3c383792641278225f0ef1d0d0017c8f722c2f89ca1150ebb0e48`.
   The old exact-zero and synthetic-one controls are withdrawn.
2. **CONFIRMED.** The viscosity dispatch test sets and checks fp64 itself and
   compares coefficient arrays on a manufactured anisotropic grid. It passes
   alone, after the ladder, and inside the 223-test SWG/shared battery. The
   former order-dependent tendency comparison is withdrawn.
3. **REFUTED.** Arm A's exact scalar grouping and `0.5*r1_rho0` transcription
   left the complete given-entry payload byte-unchanged: 306/338 AT-BAR and
   32/338 DEBT, first at `given.kt2.s3.u_k`, 6,320 unequal wet U faces and
   maximum absolute difference `1.7267697688394534e-13`. Artifact
   `swg_stages_given_arm_a.json`, SHA-256
   `dd2a56a5265cc80c8d22ebfce92255dcc26af60542ffb825d6aa605f50e59fbd`.
4. **REFUTED.** Arm B found both admitted stage-3 stress pairs bit-identical:
   zero unequal wet U or V faces and zero maximum difference. A manufactured
   production test with distinct before/current stresses verifies that each
   field is read once. The source forms those U/V stresses at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/sbcmod.f90:553-560`, initializes their
   before values on this non-restart deck at `:563-585`, and consumes both at
   compiled `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:325-328`.
5. **REFUTED.** Arm C's admitted whole-step and stage-3 velocities differ on
   all 6,320 wet U and V faces, so the operand choice is discriminating. The
   production manufactured test proves drag reads the whole-step entry. This
   matches NEMO's `Nbb` drag at compiled `stprk3.f90:326,328` and its `Nbb`
   final update at `:344-345`.
6. **CONFIRMED.** Arm D found the owner. legoESM supplied whole-step-entry
   `r3f` to the viscosity curl thickness. NEMO's `r3f` is unindexed: stage 2
   overwrites it through `dom_qco_r3c` at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:255-257`, and the stage-3
   viscosity body consumes that current value at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:130-150`. The admitted
   whole-step and current F ratios differ on 6,241 wet F points, with maximum
   absolute difference `2.7708960042224185e-05`. Routing the current ratio,
   while retaining the `Kbb` divergence and velocity operands, makes all
   338/338 given-entry rows and all 338/338 chained rows bit-identical.
   Artifacts `swg_stages_given.json` and `swg_stages_chained.json`, SHA-256
   `20fc0f100926f4f0aaf3a2640a93006de21d9edf0b29e09eec3acac6d4bd5a78`
   and `cd052640f497b4e3a1772764b54797188c053d13adbc3156ec527da358868508`.
7. **NOT NEEDED.** The P3T term record was conditional on Arm D leaving the
   stage row in DEBT. Because every given and chained stage row is now exact,
   no NEMO acquisition runner was added.
8. **CONFIRMED.** No AT-BAR row left the bar. The independently initialized
   ladder improves from 24/50 to 37/50 bit-identical rows, and its frontier
   moves later from `kt=2` to `kt=6`. The first remaining row is
   `independent.kt6.un`: one wet U face differs by
   `4.336808689942018e-19`; the ten-step worst difference is
   `1.1102230246251565e-16` in `kt=10` SSH. Artifact
   `swg_independent_kt10.json`, SHA-256
   `b6fc82a154553aed9967f8bd5cd72e64b14e26d0b0f41b7ce1b25e6f99b5b7de`.

## First non-bit statement

There is no non-bit statement in any admitted given-entry or chained stage:
all 26 transitions and all 338 rows are bit-identical. The first remaining
independent difference appears only at the after-state of step 6. Because the
same production stages are exact when seeded with NEMO's complete native
entry, the present record localizes the next debt to state carried between
whole steps (including redundant face storage), not to another stage physics
statement. Attribution below that boundary remains **UNMEASURED**.

## Round-4 review findings and fixes

The operator's independent round-4 verdict was:

> DO NOT SHIP — Add a genuine one-ULP recorded-entry plant and make the production-dispatch test pass under the fp64 campaign policy.

Both blockers are fixed. The control now changes one recorded nonzero input by
one ULP, and the viscosity dispatch discriminator is explicitly fp64 and
order-independent.

## Rule 12 disposition

The changed time-level statement is inside the `swe_rk3` production program.
A repository-wide execution search finds that only `build_swg_card()` selects
that program; TSUNAMI, VORTEX, VORTEX_VEC, GYRE, LOCK_EXCHANGE, OVERFLOW, DINO,
and ORCA2 do not execute it. Their focused card suites pass 273 tests. The
VORTEX suite additionally passes 93/96; its three failures are the already
documented stale expected digest pins for GYRE, LOCK_EXCHANGE, and OVERFLOW,
whose actual hashes remain the round-4 values `f02d2f42a570f413`,
`d5dfdf83b3a1e0ab`, and `83de0010e1170032`. No shared default or card field
moved.

## Gates

The focused SWG, production-stage, shared-viscosity, record-reader, geometry,
and ladder battery passes 223/223. The geometry gate remains 34/34
bit-identical, SHA-256
`98ffd560e6c12a242c7a11c799cb18724626e9fcb4634c1c7411b8cd4a1d6671`;
its signed-y Coriolis, wet-cell, and layer-thickness plants each exit 1.

The given and chained ladders exit 0. The independent ladder exits 1 on its
honestly registered step-6 debt, and the admitted-entry one-ULP plant exits 1
on its changed output. All comparisons use fp64 on CPU with the card's T/U/V/F
masks.

The receipt-citation gate result and its shifted-citation plant are recorded
after this receipt is committed.

## OPEN

1. Localize the first independent carried-state difference between the exact
   step-5 after-state and step-6 entry, including redundant west/south face
   storage and every unscored internal field. Do not begin the 30-day rung
   until the independent ten-step ladder is bit-identical.
2. After the ten-step ladder is exact, run the 30-day rung, then the shipped
   172800-step statistics.

No acquisition or configuration/carried-state decision is requested.
