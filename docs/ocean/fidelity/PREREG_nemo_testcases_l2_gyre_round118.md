# Preregistration — NEMO testcase L2 GYRE round 118

Date: 2026-09-19

Incoming lane tip: `c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd`

This document is frozen before any Round-118 record parsing or scientific
measurement.  The unchanged production anchors are kt2 T/S/U/V maximum error
`1.4210854715202004e-14` / `2.1316282072803006e-14` /
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S maximum error
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T rms
`6.890484901489568e-5 K`.  They are inherited comparison anchors, not new
measurements.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round118/`.

## P1 — reconcile the Round-117 record from the compiled writer

The operator's completed NEMO run emitted a 112,560-byte pre-loop record and
then the acquisition script refused it because the registered arithmetic said
127,408 bytes.  The compiled writer, not either arithmetic, governs the
layout.  In the exact build that ran, the eight ENE coefficient arrays are
allocated on `Nis0:Nie0,Njs0:Nje0` at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:83-86` and
`:124-125`; `zu_frc`/`zv_frc` have that same owned extent while
`zu_trd`/`zv_trd` are `jpi,jpj` at `:180-182`.  The header and all 18 fields
are written at `:319-343`.

Frozen layout prediction: the record has a 48-byte header, six full 36x26
arrays (`puu_b`, `pvv_b`, two masks, `zu_trd`, `zv_trd`) and twelve owned
32x22 arrays (incoming U/V, eight ENE coefficients, final U/V):

`16 + 8*4 + (6*36*26 + 12*32*22)*8 = 112560` bytes.

The header must be exactly `(1, 2, 3, 36, 26, 64, 18, 704)`, every field must
parse at its compiled extent, and physical EOF must follow final V.  A
different header, field count, extent, byte size, non-finite field, or trailing
byte refutes this diagnosis.  The retained 127,408-byte expectation is
retracted if and only if those checks pass; it is not adjusted merely to equal
the observed file size.  A planted eight-byte size shift and a compiled-layout
plant must each print a named `STATUS PLANT-FIRED` refusal and exit nonzero.

## P2 — recover and admit the existing run without NEMO execution

The existing target and run directory are reused.  The recovery path must be
an explicit `--admit-existing` mode in the existing Round-117 source card; it
must execute neither `makenemo` nor `mpirun`.  Before creating the missing
pre-loop stamp or appending `RUN_DONE`, it must require the recorded producer
commit `c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd`, verify the source-card and
toolchain manifests, require `STOP 0`, and prove both the built and copied
executables equal the registered binary digest.  The original run log is
preserved and recovery only appends a named recovery marker.

Frozen admission prediction: the binary identity checks pass; the slow record
is 1,486,548 bytes; the repaired pre-loop reader accepts 112,560 bytes; the
same-run slow post-wind U/V equals the pre-loop incoming U/V bit-for-bit; the
compiled subtract replay equals final U/V bit-for-bit; and final U/V equals
the admitted Round-81 substep-1 frozen forcing bit-for-bit.  The clean gate,
the inherited-record/restart/mesh admission, and every stamp, truncation,
header, input-ULP, reference-ULP and consumed-field plant must run.  Every
plant exits nonzero with its own named firing.  Any duplicate row that is not
BIT, any binary/source/tool mismatch, or any plant that stays green refuses
the record and stops the scientific walk.  No new acquisition is requested
unless the compiled layout or identity checks fail.

## P3 — direct production pre-loop boundary

Only after P1-P2 pass, the existing Round-83 producer gate replaces the two
Round-117 proxies with the admitted direct pre-loop fields.  It scores, in
compiled order, incoming U/V, the Kmm `dyn_cor_2D` result, and final U/V under
the complete production step JIT.  Production eager and isolated-closure JIT
remain separately labelled and cannot certify the owner.  The executing copy,
call and subtract are the compiled statements at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:289-301` and
`:335-343`.

Frozen production-JIT prediction: Kmm Coriolis is BIT for U and V; incoming
and final differ on all 580/570 wet U/V faces; their maxima reproduce
`1.0529650291768787e-11` / `1.0765559917925099e-11 m s-2` within `1e-22`.
Thus incoming U is the first direct non-BIT row and there is no incoming/
Coriolis cancellation.  A non-BIT Coriolis row, different cell count, maximum
outside the stated tolerance, or final result that does not close from the
two direct operands refutes the corresponding prediction and moves the walk
to the first measured boundary.  The production incoming-ULP plant must move
one incoming word and the downstream final word while leaving Coriolis BIT,
print `STATUS PLANT-FIRED`, and exit nonzero.

## P4 — direct source chain and landing stop

If incoming is first, the same-run kt2 slow record is joined to the admitted
Round-46/64 cumulative stage-1 record.  The registered order remains HPG ->
LDF -> VOR -> KEG -> ZAD -> ADV -> depth average -> drag -> wind.  The gate
must first prove the inherited NEMO `after_adv` arrays equal the same-run slow
record's direct `Krhs` arrays on every consumed cell, and that the slow
post-wind result equals the direct pre-loop incoming pair.  It then reports
the complete production-JIT cumulative table and live-total closure.

Frozen prediction: HPG remains BIT; LDF is the first non-BIT cumulative row at
`2.5292467120726215e-14` U and `3.502735092670824e-14` V within `1e-26`; ZAD
is the largest incremental residual; and the isolated cumulative-to-live
total closure retains the Round-117 last-bit residual (U 6,882/17,400 and V
6,566/17,100 at `8.470329472543003e-22`).  That result withholds an LDF owner
and forbids a candidate: the first unclosed production association, not LDF by
historic label, becomes the OPEN walk.  If the live-total closure is instead
BIT and every direct operand closes, the first source-owning statement is
reported and a separate committed addendum must preregister any candidate
before production is edited.

With no source-exact candidate, no physics, configuration, ladder, month arm,
or certified card can move.  GYRE, the generic NEMO-GYRE recipe, DINO,
LOCK_EXCHANGE and OVERFLOW remain unchanged.  ORCA2 remains
`UNMEASURED-WITH-SPEC`: independently record and admit this direct pre-loop
layout and its cumulative producer chain on native extents, run the production
JIT/eager/isolated controls and plants, then run its certified trajectory for
any shared candidate.  No stabilizer, default, coefficient, carried state,
restart representation, year harness, reconciliation gate, freshwater pair,
#1484 guard, or held patch changes in this round.
