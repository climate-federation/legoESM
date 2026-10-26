# Preregistration — round 176, developed accumulated-content process walk

Committed before running the Round-176 production-step comparison.  Round 175
showed that NEMO's final temperature-content expression is bit exact when fed
NEMO operands and that inherited `T(Krhs)` carries essentially all of the
`7.046353169355859e-04 K m` production-JIT accumulated-content RMS.  This
round walks the directly observed producers of that operand at developed step
1081.  It changes no production physics, configuration, carried state,
restart schema, card default, or immutable Round-163 before arm.

## Compiled order and registered rows

The admitted Round-123 build calls advection and writes its temperature
`Krhs`, then calls the RK3 surface boundary and writes again at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`.
At stage 3 it calls and records penetrative shortwave and lateral diffusion,
then calls the implicit vertical solve, at the same compiled file's
`:930-970`.  The Round-125 build consumes the resulting `Krhs` in the literal
content expression at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567`.

The existing year-owner harness will be extended, not duplicated.  From the
admitted Round-132 step-1080 restart it drives the complete production step in
two modes: `LatLonCGridOceanModel.step -> self._step_jitted`, which is
authoritative, and the same complete step with JIT disabled.  It reads the
existing `tracer_process_trace` boundaries and `vertical_solve_trace` content;
an independently compiled ordinary step must have zero returned-state byte
differences in each mode.

For both NEMO and legoESM, the registered content-space sequence is:
`after_advection`, `after_surface_boundary`, `after_shortwave`,
`after_lateral_diffusion`, and `complete_accumulated_content`.  The four
isolated process components are the first cumulative row followed by adjacent
differences.  A fifth `rounding_closure` component is always retained.  Every
row reports unequal wet cells, maximum absolute difference, RMS, and its
signed projection onto the complete accumulated-content error.  The signed
components plus closure must reconstruct the complete error field.

## Frozen predictions and falsifiers

1. Round-123 admission and the Round-125 literal content reconstruction remain
   bit exact.  Any failed calibration refuses the walk.
2. Production-JIT and complete eager observers each move zero returned-state
   bytes relative to their ordinary step.  Any movement invalidates that mode.
3. `after_advection` is the first non-bit directly observed cumulative row in
   both modes and moves at least 1,000 wet cells.  An earlier unregistered row,
   a bit-exact advection row, or fewer cells REFUTES the prediction.
4. Lateral diffusion has the largest isolated content-component RMS and is at
   least `1e-4 K m`.  A different largest row or smaller scale REFUTES this
   magnitude prediction; the measured ranking is retained.
5. The authoritative complete accumulated-content row reproduces Round 175's
   `7.046353169355859e-04 K m` RMS within `1e-15 K m`.  A larger difference
   means the walk did not score the same boundary and refuses interpretation.
6. Production eager and JIT name the same first non-bit cumulative boundary.
   A difference adopts production JIT and records the eager divergence.
7. A fixed nonzero perturbation to one wet production surface-rate cell moves
   its surface, shortwave, lateral-diffusion, and complete-content rows, leaves
   the upstream advection row unchanged, prints `STATUS PLANT-FIRED`, and
   exits nonzero.  A zero exit, upstream movement, or missing downstream
   movement invalidates the instrument.

The first non-bit boundary is not automatically an internal operator owner.
An internal statement is named only when its direct operands are captured.
If the boundary inherits a non-bit input, the receipt names that input and the
next upstream walk.  The full Decision 43/45/55/59 trajectory gate runs only
if this round finds a one-variable source-exact production candidate.  No NEMO
acquisition is preregistered because every required record already exists.
