# Preregistration — NEMO testcase L2 GYRE round 117

Date: 2026-09-19

Incoming lane tip: `475ca7396`

This document is frozen before any Round-117 scientific measurement.  The
unchanged production anchors are kt2 T/S/U/V maximum error
`1.4210854715202004e-14` / `2.1316282072803006e-14` /
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S maximum error
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T rms
`6.890484901489568e-5 K`.  They are inherited comparison anchors, not new
measurements.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round117/`.

## P1 — extend the admitted producer gate under the production step

Round 117 extends
`nemo_testcase_l2_gyre_round83_slow_forcing_walk.py`; it does not create a
second producer harness.  The Round-64 momentum-stage and slow-forcing records
and the Round-81 external-step record retain their existing admission,
producer-commit, hash, byte-size, physical-EOF and duplicate-boundary checks.
The complete production step supplies a WRITE-only trace with observer
non-interference over every returned state leaf.  Rows are labelled
`production step JIT`, `production eager`, or `isolated-closure JIT`; only the
first may certify the production owner.

The executing NEMO branch copies `Ue_rhs`/`Ve_rhs`, initializes the Kmm 2-D
Coriolis coefficients, evaluates `dyn_cor_2D`, then subtracts and masks the
trend at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:291-327`.
The gate exposes exactly three live pairs in that order: incoming forcing,
Coriolis result, and final forcing handed to the external step.  The final pair
and Coriolis pair have direct Round-81 records.  The incoming pair does not:
the gate reports both (a) the Round-64 compiled producer replay and (b) the
algebraic preimage `recorded_final + recorded_coriolis * mask`.  It must call
the preimage `RECONSTRUCTED`, never a directly recorded NEMO field.  A
source-exact candidate is forbidden unless an admitted direct record, or an
exact forward replay from admitted direct operands, certifies that input.

Frozen production-JIT prediction: the Coriolis result is BIT for U and V; the
incoming and final pairs each differ on all 580/570 wet U/V faces; final maxima
reproduce Round 116 at `1.0529650291768787e-11` and
`1.0765559917925099e-11 m s-2` within `1e-22`.  The incoming maxima are in
`[1.0e-11, 1.2e-11] m s-2`.  Thus the first non-BIT boundary is imported
`Ue_rhs`, not `dyn_cor_2D` or the final subtract.  An inexact Coriolis row, a
different cell count, a final maximum outside `[1.0e-11, 1.2e-11]`, or a first
non-BIT final result with both inputs BIT refutes that prediction and redirects
the walk to the measured boundary.

The production plant advances one finite incoming-U trace word by one ULP.  It
must move the incoming row and the final subtract row under the complete
production JIT, print `STATUS PLANT-FIRED`, and exit nonzero.  A plant visible
only in an isolated replay is inert.  A separate observer-control test removes
the full-pytree identity guard and must fail.

## P2 — current-tip compiled-order `Ue_rhs`/`Ve_rhs` walk

If P1 confirms imported `Ue_rhs`/`Ve_rhs`, the same admitted gate reruns the
stage-1 cumulative momentum RHS in compiled order.  The record build calls HPG,
LDF, VOR, KEG and ZAD and snapshots the shared accumulator at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`; it then performs
the written vertical sums at `:202-208`, drag at `:225-227`, and wind addition
at `:229-239`.  The current production-step JIT is scored at `after_hpg`,
`after_ldf`, `after_vor`, `after_keg`, `after_zad`, and `after_adv`, with U and
V separate and the exact final-RHS-to-live-total closure retained.

Frozen prediction: HPG stays BIT.  LDF is the first non-BIT cumulative
boundary, reproducing the inherited maxima within a factor of four:
U `[6.3e-15, 1.02e-13]`, V `[8.7e-15, 1.41e-13] m s-2`.  The largest
incremental residual remains at ZAD.  `after_adv` equals `after_zad` BIT in the
NEMO record, and the live cumulative `after_adv` closes BIT to the production
`du_dt`/`dv_dt`.  Any earlier non-BIT row, LDF outside those intervals, a
different largest-increment boundary, or a non-BIT live-total closure refutes
the corresponding prediction.  In particular, a failed closure withholds an
operator owner rather than assigning the residual post hoc.

No historic Round-84 owner label is reused without these current-tip,
production-JIT rows.  If LDF is first, the first statement is NEMO's
`CALL dyn_ldf(...,Krhs)` accumulation at compiled `stp2d.f90:147-149`; that
names the next operand walk but is not itself a landing candidate unless its
direct inputs and output are proven source-exact under the production step.

The cumulative-walk plant advances one finite exact HPG word by one ULP.  HPG
must become first non-BIT, the log must print `STATUS PLANT-FIRED`, and the
process must exit nonzero.

## P3 — magnitude through the external output and local kt3 tracers

One diagnostic production-JIT arm substitutes only the direct recorded final
slow-U/slow-V pair at the external solver call, retaining the ordinary state,
histories, coefficients, pressure, Coriolis, drag and every other input.  The
wrapper must prove that its ordinary arm is bit-identical to the unwrapped
production step.  The directed pair must equal the Round-81 record on native
wet faces; all other registered call inputs must remain BIT.  This is an
oracle-directed magnitude arm, not an independently running model and not a
candidate.

The gate registers weighted final SSH and local kt3 T/S for ordinary and
directed arms, plus every directed-minus-ordinary moved-cell count and maximum.
Frozen prediction: all three fields move but stay non-BIT.  The final SSH
maximum changes by less than 10% from `7.072560112143626e-7 m`; local kt3 T
changes by less than 10% from `8.600420500215478e-7 K`; local kt3 S changes by
less than 50% from `6.979443156751586e-8`.  The predicted direction is an
improvement for SSH and T.  An inert field, a BIT field, a change outside those
bands, or either predicted improvement becoming a worsening is a retained
refutation.  Because the arm replaces the final pair rather than one proved
producer statement, even an improvement measures carried magnitude only.

## P4 — landing, cards, and stop conditions

The expected outcome is diagnostic: P1 moves the walk to the current-tip LDF
boundary, while P2 names an operator whose exact direct owner is not yet
proved.  No production physics changes and no kt1..10 or month arm is spent in
that outcome.  If instead this round proves a precise source-exact production
statement from direct operands, an addendum must freeze that candidate and its
same-base trajectory/card predictions before editing production.

Any candidate remains subject to Decision 43: day-30 T rms must decrease
against a measured arm from this exact base commit; first-over-bar cannot move
earlier; no kt1 AT-BAR row may leave the bar; every moved row, including
worsened rows, must be registered; and every executing card derived from its
resolved recipe must be measured.  A shared candidate measures DINO before
landing.  With no candidate, GYRE, the generic NEMO-GYRE recipe, DINO,
LOCK_EXCHANGE and OVERFLOW cannot move.  ORCA2 remains
`UNMEASURED-WITH-SPEC`: independently record and admit its direct producer
RHS, Kmm Coriolis result, final frozen forcing, cumulative 3-D RHS boundaries,
external-step outputs, next-step entry and local tracer outputs; run the
production-JIT/eager/isolated controls and plants; then run its certified
trajectory for a shared candidate.

If the Round-64/Round-81 join cannot certify the directly consumed incoming
pair, the receipt says so explicitly.  A new NEMO record is requested only if
that absence blocks the next exact operand walk; it is never replaced with an
uncertified inverse reconstruction.
