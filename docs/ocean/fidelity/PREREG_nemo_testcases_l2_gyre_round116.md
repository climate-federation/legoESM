# Preregistration — NEMO testcase L2 GYRE round 116

Date: 2026-09-19

Incoming lane tip: `bb9c586891df9cb089be3675b3a1de38e160ee2f`

This document is frozen before any Round-116 scientific measurement.  The
production baseline is unchanged from Round 115: kt2 T/S/U/V maximum error
`1.4210854715202004e-14` / `2.1316282072803006e-14` /
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S maximum error
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T rms
`6.890484901489568e-5 K`.  These are frozen comparison anchors, not new
measurements.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round116/`.

## P1 — reuse, admission, and final-output alignment

Round 116 extends the admitted Round-81 external-step reader and Round-82
compiled-order walk.  It does not create a second barotropic harness.  The
Round-81 stream is admitted only with its registered producer, SHA-256, byte
size, physical EOF, duplicate-record census, and Round-77 U-midpoint identity.
The Round-75 next-step entry independently supplies the final filtered N+1 SSH
consumed by the RK3 stages.

The record build writes each substep immediately after exchange and before the
history swap at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:797-850`.
The current compiled build then rotates the histories, accumulates the
weighted SSH, and divides the completed average at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:749-804`.
Therefore `swap_eta` is a per-substep boundary, not the final N+1 SSH.  The
gate will compare the production trace's returned weighted SSH directly with
the independent next-step entry; it will not relabel substep 50 as the final
output.

Prediction: the final production-JIT row reproduces Round 115 exactly:
600/600 wet cells unequal, maximum `7.072560112143626e-7 m`.  Production
eager may differ in its last bits but must remain 600/600 and within
`1e-12 m` of that maximum.  A different count, a maximum outside
`[7.0e-7, 7.2e-7] m`, or failure to align the weighted output with the N+1
entry refutes the alignment.

## P2 — current-tip external-step walk

The walk scores all 50 substeps in compiled order: extrapolation coefficients;
current, one-back, and two-back U/V/SSH; midpoint U/V/SSH; face depths and
transports; forcing/divergence/continuity SSH; backward interpolation;
pressure, Coriolis, drag and inverse depth; frozen slow forcing; velocity
update; exchanged exit; and history swap.  The executing statements are the
current compiled `dynspg_ts` midpoint and transport block at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:456-532`,
continuity at `:542-554`, backward interpolation and pressure at `:593-608`,
Coriolis/drag/update at `:610-668`, exchange at `:701-730`, and swap at
`:749-761`.  A source diff must show that the Round-81 build differs only by
write-only instrumentation over these executing statements.

Production-step JIT governs.  The complete production closure with JIT
disabled is reported as `production eager`.  The record's existing arithmetic
replays are reported as `isolated scalar replay`; they are admission controls,
not production evidence.

The histories/drag/inverse/ZAD bundle landed after Round 82, so its old owner
label is not reused as evidence.  Frozen prediction: at substep 1 every row
through `trd_u` and `trd_v` is BIT; the first non-BIT boundary is the imported
frozen `slow_u` or `slow_v`, with U maximum in `[1e-12, 2e-11] m s-2`.
Compiled NEMO constructs that field by copying `Ue_rhs`/`Ve_rhs`, computing
the Kmm barotropic Coriolis trend, and subtracting it at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:286-322`.
If an earlier row is non-BIT, that row owns the walk.  If every operand through
slow forcing is BIT and the first result is non-BIT, that exact result is a
candidate statement.  An imported non-BIT slow-forcing row moves the next
walk upstream and is not itself an external-loop arithmetic candidate.

The production plant advances one finite, exact `u_b` record word by one ULP.
It must become the first non-BIT row under production JIT, print
`STATUS PLANT-FIRED`, and exit nonzero.  A plant observed only by the scalar
replay refuses the instrument.

## P3 — magnitude of the full-SSH boundary

A second full production-JIT step differs in one field at the RK3 handoff:
the external solver's live weighted SSH is replaced by the independently
recorded N+1 SSH, while the live U/V external means and live U/V transport
averages are retained byte-for-byte.  This is an oracle-directed diagnostic,
not a production candidate and not independent model execution.

For ordinary and SSH-directed arms, the gate registers complete kt2/stage-3
`zFu`, `zFv`, and `zFw` against their direct NEMO records and local kt3 T/S
against the independent step-3 entry.  It also registers the direct delta
between arms so an unchanged maximum cannot hide moved cells.  Frozen
prediction: all three transport families move; none becomes BIT; local kt3 T
remains non-BIT and changes by less than 50% from the Round-114 local value
`8.600420500215478e-7 K`; local kt3 S remains non-BIT and changes by less
than 50% from `6.979443156751586e-8`.  Falsifiers are an inert transport
family, a BIT transport family, either tracer becoming BIT, or a local kt3
change of 50% or more.  The Round-114 single-family cancellation warning
remains binding: an improved or worsened diagnostic arm is magnitude evidence,
not ownership or a landing.

A separate handoff control advances one finite recorded SSH word by one ULP.
It must flip the final handoff row and its directly derived QCO ratio under the
production JIT.  Whether that last bit survives the later thickness,
transport, and tracer roundings is recorded, not assumed; requiring a
downstream tracer bit to move would make the control depend on an unrelated
rounding cancellation.  The control exits nonzero even if aggregate maxima do
not move.

## P4 — candidate and landing boundary

The predicted result is diagnostic: the external loop inherits a non-BIT slow
forcing input, so no production physics changes and no ladder/month arm is
spent.  If the walk instead identifies a first non-BIT result with all direct
inputs BIT, a committed addendum must freeze that precise candidate and its
same-base kt1..10/month/card predictions before changing production.

Any candidate remains subject to Decision 43: day-30 T rms must decrease
against a measured same-base arm; first-over-bar cannot move earlier; no kt1
AT-BAR row may leave the bar; every moved row, including worsened rows, must
be registered; and every executing card derived from resolved recipes must be
measured.  Without a candidate, the incoming ladder and month numbers above
remain the production values and are not remeasured.

DINO shares the external solver but uses a different time-integration card;
an oracle handoff substitution cannot move DINO and makes no neutrality
claim.  LOCK_EXCHANGE and OVERFLOW likewise remain unchanged because no
shared production statement is edited.  ORCA2 remains
`UNMEASURED-WITH-SPEC`: acquire its per-substep U/V/SSH histories, forcing,
transport, continuity, pressure, Coriolis, drag, update, swap, weighted final
output, and next-step entry on native masks; then run this production-JIT walk
and its certified trajectory.
