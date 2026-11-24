# ORCA2 round 292 preregistration — external-mode V target producer

Date: 2026-10-16

Round 291 proved that the independent OMT-4 kt=1 stage-1 barotropic
correction first differs in its carried external-mode V target: 70/180 faces,
maximum 0.01620736958946065 m/s.  This round walks that target through the
existing admitted split-explicit substep stream, in NEMO's compiled order,
without adding an executable observer.

The compiled ORCA2 record build forms the midpoint velocity at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:460-485`, the live
V-face depth at `:487-520`, the V transport at `:530-536`, sea level at
`:546-558`, pressure/Coriolis/drag at `:601-652`, the vector-form V update at
`:655-679`, the in-loop association at `:721-739`, and the final carried
target at `:857-860`.  The deck executes the vector branch
(`ln_dynadv_vec=.true.`), but NEMO's face-depth, transport, and association
statements are outside that branch.

Every value is **independent OMT-4, kt=1 stage 1**.  The candidate instrument
reuses the existing passive barotropic substep trace; the oracle side reuses
the admitted round-222 `oracle_bt_substeps` and `oracle_bt_advmean_operands`
streams.  The current card is instantiated and must print its resolved
vector-form and held fold-unit selectors.  No NEMO run, executable observer,
model edit, card change, configuration choice, carried-state change, or
landing is authorised.

## Frozen predictions and falsifiers

1. **R292-P1 — admission and passivity.**  Every inherited stream is
   twin-exact on its defined payload; the traced and ordinary completed states
   are array-identical; CPU production JIT and fp64/libm are printed.  Any
   failure refuses all numbers.
2. **R292-P2 — target calibration.**  The final candidate `va_e` on the
   admitted rank slab reproduces the corresponding half of round 291's carried
   target debt: 35/90 fold-row faces and maximum
   0.01620736958946065 m/s.  A changed count or maximum refutes the program-
   point alignment.
3. **R292-P3 — source-ordered first boundary.**  At substep 1, slow forcing,
   entry V, midpoint V, and midpoint SSH remain within the 2e-10 floor.  The
   first valued intermediate is the live V-face depth on NEMO's fold-image
   row, because the production ORCA2 vector path still leaves the literal
   fold-depth/transport unit held.  An earlier valued row names that row as
   owner; a depth row at the floor refutes the prediction and the walk
   continues through V transport, sea level, pressure/Coriolis/drag, update,
   and association.
4. **R292-P4 — locality.**  The first valued row is fold-local: the interior
   (all rows south of the final three) remains at the floor at that boundary.
   An interior valued difference refutes the fold-unit attribution.
5. **R292-P5 — endpoint causality.**  The first valued boundary occurs no
   later than the first valued V exit and precedes the final carried target.
   A clean complete substep table alongside a non-bit target refutes the
   instrument.
6. **R292-P6 — non-vacuity.**  Passivity, source order, target calibration,
   first-boundary, locality, endpoint-causality, and one-bit plants must each
   make the gate refuse.  A green plant refutes the measurement.

The outcome is measurement-only and **HELD**.  Decision 125's completed-stage
top-V exchange, the resolved stage mask, and the vector-form fold unit remain
off.  Decision 126 remains pending and unchanged.  Sea ice and the shipped
card's `unmeasured_features` tuple remain unchanged.

Independent review: operator, as required for this lane.
