# Preregistration — NEMO testcase L2 GYRE round 115

Date: 2026-09-19

Incoming lane tip: `da07184ff`

This document is frozen before any Round-115 scientific measurement.  The
production baseline is unchanged from Round 114: kt2 T/S/U/V maximum error
`1.4210854715202004e-14` / `2.1316282072803006e-14` /
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S maximum error
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T rms
`6.890484901489568e-5 K`.  These are frozen comparison anchors, not new
measurements.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round115/`.

## P1 — reuse and admission

Round 115 extends
`nemo_testcase_l2_gyre_round67_ldf_order.py`, the production-step instrument
used by Rounds 111--114.  It does not add another stage harness or duplicate
the shared QCO geometry implementation.  The pre-implementation search found
the existing source-associated implementation in
`ocean/vertical.py::nemo_qco_live_face_geometry_from_operands` and the existing
Round-46 stage reader and records.

The Round-46 kt2 stage-3 record supplies the directly recorded full-step
`ssh_Kaa` input, full-step `r3u_Kaa` result, step-entry `ssh_Kbb/r3u_Kbb`,
half-step `ssh_Kmm/r3u_Kmm`, and static `e1e2t`, `r1_e1e2u`, `e3u_0`, and
`umask`.  It is admitted only with its registered producer and SHA-256.  The
NEMO depth reciprocal is source-replayed from the same record, not inferred
from a downstream transport: compiled `domain.f90:193-200` accumulates
`hu_0 = SUM(e3u_3d*umask)` in ascending level order and
`domain.f90:212-215` writes the masked reciprocal.  The face-area reciprocal
is directly recorded after compiled `domhgr.f90:170-171` constructs it.

Prediction: the ordinary production-step `1+r3u(Kmm)` row reproduces Round
114 exactly: 580/580 wet columns unequal with maximum
`7.552691805301492e-11`.  Any mismatch in record producer, digest, header,
shape, dtype, or physical EOF refuses the run.

## P2 — compiled-order U geometry walk

The active compiled statement is
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90:256-270`.
For U it evaluates, in written order,

`0.5 * (e1e2t(i,j)*pssh(i,j) + e1e2t(i+1,j)*pssh(i+1,j))`
`* r1_hu_0(i,j) * r1_e1e2u(i,j)`.

The walk registers input SSH and static operands, then the west product, east
product, parenthesized sum, multiplication by 0.5, multiplication by
`r1_hu_0`, final multiplication by `r1_e1e2u`, and `1+r3u`.  Intermediate
references are labelled source replays; the final `r3u_Kaa` and `r3u_Kmm`
references are directly recorded NEMO outputs.  Production-step JIT governs;
production eager and isolated-closure JIT are reported separately and cannot
replace it.

NEMO calls this geometry statement once from the saved full-step `ssha` at
compiled `stprk3_stg.f90:140-180`.  Its hybrid branch then forms the one-third
ratio at `:184-193`, the half-step SSH and ratios at `:228-238`, and exposes
stage 3 with `Kmm=N+1/2` at `:241-258`.  The gate therefore also scores:

1. model full-step and half-step SSH against the two directly recorded fields;
2. geometry from NEMO's recorded full-step SSH;
3. NEMO's recorded half-step ratio against
   `0.5*(r3u_Kbb + r3u_full)` in the compiled association; and
4. the model's direct geometry of its half-step SSH.

Frozen prediction: the model's own full-step or half-step SSH is the first
non-BIT live input, with a maximum SSH error between `1e-8` and `1e-5 m`.
Given NEMO's recorded full-step SSH and the recorded/source-replayed static
operands, the full-step `r3u_Kaa` result and the compiled half-step
interpolation are predicted BIT (0/580 unequal).  Thus the ordinary
580-column / `7.552691805301492e-11` row is predicted inherited from the
external-mode SSH chain, not owned by the QCO arithmetic.

Falsifiers are: failure to reproduce the ordinary endpoint; any earlier
non-BIT static operand; non-BIT full-step `r3u` given NEMO SSH; non-BIT
compiled half-step interpolation given the recorded full-step result; or BIT
model SSH.  Every falsifier remains in the receipt and changes the owner rather
than being explained away.

The production plant advances one finite, nonzero recorded full-step SSH word
by exactly one ULP inside the production-step trace.  It must change a written
product, the final full-step ratio, and the half-step ratio.  The process must
print `STATUS PLANT-FIRED` and exit nonzero.  An isolated-only plant refuses
the instrument.

## P3 — candidate and landing boundary

The predicted outcome is diagnostic and changes no production physics.  If a
non-BIT statement is instead owned by the shared geometry implementation or
by the same-stage ratio interpolation, an addendum will freeze that exact
candidate and its ladder/month predictions before the candidate is measured.
No downstream U/V/W or held Round-112 FCT patch may be revived from this walk.

Any implementable candidate remains subject to Decision 43: day-30 T rms must
decrease against a same-base measured arm; first-over-bar cannot move earlier;
no kt1 AT-BAR row may leave the bar; every moved row, including worsened rows,
must be registered; and the executing-card set is derived from resolved
recipes and measured.  Without such a candidate, all baseline trajectory
numbers above remain unchanged and no ladder/month run is spent.

GYRE-zco and the generic NEMO-GYRE recipe share the WS QCO implementation.
DINO uses the shared geometry code through a different tracer lane, so any
production edit requires an executed-path DINO measurement; a diagnostic
record substitution cannot move it.  LOCK_EXCHANGE and OVERFLOW must be
classified from their resolved recipes before any production edit.  ORCA2
remains `UNMEASURED-WITH-SPEC`: admit its full/half-step SSH and U-face QCO
outputs, run this production-JIT table, then run its certified trajectory.
