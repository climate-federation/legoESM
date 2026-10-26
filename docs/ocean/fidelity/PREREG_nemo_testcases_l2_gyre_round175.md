# Preregistration — round 175, developed temperature-content producer walk

Committed before running the Round-175 production-step comparison.  Round 174
ranked the temperature-content family above live `e3t(Kaa)` by a nonlinear
NEMO-side forced-input experiment.  This round walks the content producer at
step 1081 from NEMO's admitted day-180 entry.  It changes no production
physics, configuration, carried state, restart schema, card default, or
immutable Round-163 before arm.

## Compiled statement and registered rows

The producing Round-125 build forms temperature content at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567` as two
compiled-order families:

1. before content, `e3t(Kbb) * T(Kbb)`; and
2. accumulated explicit content, `rDt * e3t(Kmm) * T(Krhs)`.

Their sum is the recorded `rhs_T`.  The existing year-owner harness will be
extended, not duplicated.  It admits the Round-125 step-1081 record, rebuilds
the literal NEMO statement from its recorded operands, then drives the real
legoESM production step from the admitted Round-132 step-1080 restart.  The
authoritative mode is `LatLonCGridOceanModel.step -> self._step_jitted`; the
same rows are also reported with JIT disabled around the complete production
step.  An observer-passivity comparison binds each observed result to the
corresponding unobserved result.

The registered order is operand `T(Kbb)`, operand `e3t(Kbb)`, before-content
product, operand `T(Krhs)`, operand `e3t(Kmm)`, accumulated-`Krhs` content, and
complete content.  Every row reports cells unequal, maximum absolute
difference, RMS, and the first unequal cell over the 18,000 wet active cells.

## Frozen predictions and falsifiers

1. The existing Round-125 admission and its literal `rhs_T` reconstruction
   remain bit exact at step 1081.  Any unequal calibration cell refuses the
   walk.
2. The production-JIT observer changes zero returned-state bytes relative to
   the ordinary production-JIT step; the eager observer likewise changes zero
   bytes relative to the ordinary eager step.  Any movement invalidates that
   execution mode.
3. `T(Kbb)`, `e3t(Kbb)`, and their before-content product are bit exact in the
   authoritative production-JIT walk.  The accumulated `Krhs` family is the
   first non-bit family and moves at least 1,000 wet cells.  A non-bit earlier
   row or fewer than 1,000 unequal accumulated-content cells REFUTES this
   prediction and promotes the measured first row instead.
4. The complete model content remains non-bit with maximum absolute
   difference at least `1e-6 K m`, consistent with content being a magnitude
   owner rather than a last-bit floor.  A smaller maximum REFUTES the predicted
   scale.
5. Production eager and production JIT name the same first non-bit family.
   Different first rows adopt the compiled production-JIT result and register
   the eager-only divergence as a fusion-context finding.
6. A one-ULP change to NEMO's first wet `T_Krhs_in` operand changes the literal
   accumulated-content row and complete RHS, prints `STATUS PLANT-FIRED`, and
   exits nonzero.  A zero exit, unchanged registered row, or movement outside
   the selected cell invalidates the instrument.

If the content association is locally exact given NEMO's own operands, it is
not a candidate; the walk names the first non-bit input and moves upstream.
The full Decision 43/45/55/59 trajectory gate runs only if this round finds a
one-variable source-exact production candidate.  No NEMO acquisition is
preregistered because every required operand is already in the admitted
Round-125 record.
