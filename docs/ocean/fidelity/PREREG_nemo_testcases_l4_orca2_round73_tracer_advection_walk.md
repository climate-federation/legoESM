# ORCA2 round 73 preregistration — independent tracer-advection walk

Date frozen: 2026-09-29

Base: `7f7f91cbe1d123b5afec7e6e1972109b4ce98431`

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and consumes the admitted exact kt=1 surface operands. No NEMO entry
field or completed transport is substituted into the production arm.

Sea ice remains out of scope. The card's six-entry `unmeasured_features`
tuple, selectors, 10,800 s step, carried state, and all thresholds are frozen.

## Existing evidence, not predictions

Round 72 established that independent kt=1 rank-0 entry temperature is
raw-bit exact on 228,641 wet cells and that the first non-bit temperature
boundary is immediately after `CALL tra_adv`: 228,641 / 228,641 unequal,
maximum `3.0869700763080185e-07 K`. The admitted record contains the exact
rank-0 Kmm tracer and metric stage transports `zFu`, `zFv`, and `zFw`.

The search-before-build audit found and will reuse the round-43 production
transport exposure, the round-72 source-boundary runner, the phase-2l tracer
and transport readers, the source-rounded `_nemo_cen2_tracer_rhs` production
helper, and the round-61 card-scope pattern. No second model trajectory or
advection implementation is authorized.

## Compiled path and ordered statements

The compiled dispatcher sets `ll_dofct = .FALSE.` when the optional RK3 stage
is not 3, then routes an FCT card to `tra_adv_cen` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv.f90:491-535`.
The round therefore walks the live stage-1 CEN2 path, not the stage-3 FCT
limiter.

For temperature, the ordered arithmetic boundaries are:

1. U face flux, `0.5 * pU * (T + T_east)`,
   `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:149-155`;
2. V face flux, `0.5 * pV * (T + T_north)`, same span;
3. parenthesized horizontal divergence and live-thickness division,
   `traadv_cen.f90:157-161`;
4. centered interior vertical flux and its sequential divergence update,
   `traadv_cen.f90:202-228`.

The gate first prints which of ORCA2, GYRE, OVERFLOW, and LOCK resolves this
tracer program from the instantiated cards. It then scores the production
stage-1 metric transports against the admitted record, constructs each
compiled arithmetic boundary from the corresponding live operands and from
NEMO's recorded operands, and stops attribution at the first non-bit row.
Every comparison is raw-bit fp64 on the same rank-0 wet support.

## Frozen predictions and falsifiers

1. **Card execution census.** All four instantiated cards predict
   `tracer_advection="fct2"` and `tracer_time_integrator="rk3_ws"`; therefore
   all four execute CEN2 at stages 1 and 2 and FCT2 at stage 3. Any different
   resolved pair is **REFUTED** and the gate refuses that card's claimed scope.
2. **Entry calibration.** Independent kt=1 Kmm temperature predicts 0 / 228,641
   unequal on the recorded wet support. Any mismatch refuses the arithmetic
   walk.
3. **First operand.** Production stage-1 metric `pU` predicts the first non-bit
   transport operand in the compiled CEN2 order. If `pU` is exact, this is
   **REFUTED** and the walk proceeds through `pV` then `pW` without relabeling.
4. **First arithmetic statement.** The U face-flux assignment at
   `traadv_cen.f90:153` predicts the first non-bit arithmetic row. If it is
   exact, this is **REFUTED** and the first observed non-bit row in the frozen
   order becomes the owner.
5. **Recorded-operand control.** Replaying the same ordered source statements
   from NEMO's recorded Kmm T and metric transports predicts a raw-bit-exact
   after-advection accumulator. A mismatch makes the instrument or schema
   uncalibrated and no production attribution is reported.
6. **Boundary reproduction.** The production replay predicts round 72's
   after-advection tuple exactly: 228,641 unequal and maximum
   `3.0869700763080185e-07 K`. Any change refuses the measurement.
7. **Disposition.** This diagnostic round predicts **HELD**. No model,
   configuration, selector, threshold, stabiliser, carried-state, or sea-ice
   change is authorized.

Failed predictions remain **REFUTED** in the receipt. Operand, arithmetic,
control, and final-boundary predicates are separate and cannot waive one
another.

## Controls and validation

- The instrument must refuse a one-ULP mutation in the first exact recorded
  input and a one-ULP mutation in the first exact recorded-operand replay row.
- It must also refuse a false card-scope selector and a changed round-72
  boundary tuple.
- Run focused tests, the 170-test shared-card/tank battery, both citation gates
  with a real firing citation plant, the wide ocean-fidelity battery once, and
  the required read-only Codex review.
- No `packages/` change is authorized, so GYRE trajectory landing gates do not
  apply.

ASKED: walk inside independent ORCA2's selected tracer-advection call and name
the first non-bit compiled arithmetic statement after printing card scope.

UNASKED: model arithmetic, configuration, new NEMO output, selectors, sea ice,
and the held shared tracer QCO/RK candidate.

## Addendum after the first fail-closed run

The first committed instrument run refused at prediction 5 before writing a
report: recorded Kmm T plus recorded metric transports alone did **not** make
the after-advection accumulator exact. Prediction 5 is therefore retained as
**REFUTED**, not rewritten. Source inspection shows why the control was
incomplete: the horizontal and vertical divergence statements divide by live
`e3t_3d(Kmm)` at `traadv_cen.f90:157-161,214-228`; independent ORCA2's entry
SSH is already non-bit, so the recorded external endpoint/live thickness is a
distinct required operand.

Before rerunning, the calibration is tightened to report both arms separately:

- recorded transports with independent live thickness predict non-bit and
  retain the failed prediction;
- recorded transports plus the admitted recorded external endpoint/live
  thickness predict raw-bit equality. Failure of this complete-operand arm
  invalidates the arithmetic walk.

The production arm and its frozen first-operand/first-statement predictions
are unchanged and remain independent.
