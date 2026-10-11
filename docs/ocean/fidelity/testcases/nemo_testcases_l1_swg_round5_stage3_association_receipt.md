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

## OPEN

Frozen until measurement.
