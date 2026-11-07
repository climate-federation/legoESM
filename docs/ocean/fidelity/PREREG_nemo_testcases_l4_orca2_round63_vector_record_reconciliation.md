# ORCA2 round 63 preregistration — vector record reconciliation and walk

Date: 2026-09-28

Base: `990a34547bc1ca1f074a580f5ff9801324400a0d`

Claim label: **given NEMO's recorded operands**.  Independent-start and
month-scale claims are out of scope for this record-reconciliation round.  Sea
ice, its six selectors, and the ORCA2 card's `unmeasured_features` tuple remain
frozen.

## Source-first scope

The operator ran the committed round-62 additions-only acquisition to `STOP 0`,
but its admission refused the first rank stream while expecting `e3v_Kmm`:
the next sixteen bytes were all zero.  The compiled writer declares and emits
the fields in source order at
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynadv_round62_writer.f90:79-101`.
The committed parser consumes the same declared schema at
`nemo_testcase_l4_orca2_round62_vector_split_admission.py:23-33,83-100`.
No scientific statement may be scored until those two descriptions are
reconciled against the physical stream through EOF.

The executing ORCA2 branch remains vector-invariant.  Its compiled dispatcher
calls KEG before ZAD at
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynadv.f90:134-138`;
KEG's arithmetic is
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynkeg.f90:117-130`,
and ZAD's is
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynzad.f90:102-137`.

## Frozen predictions and falsifiers

1. **Physical stream reconciliation.** Number: parsed field names, shapes,
   payload lengths, and physical EOF for both rank streams, derived from their
   self-describing headers rather than a predicted whole-file byte count.
   Prediction: all named records through `e3u_Kmm` are intact, and the refusal
   begins because the compiled write statements for `e3u_Kmm` and `e3v_Kmm`
   emitted no payload while still emitting their headers.  CONFIRM only if a
   header-driven offset walk shows exactly that on both ranks.  REFUTE if the
   first structural discrepancy is anywhere else, if ranks differ, or if any
   later record cannot be recovered through physical EOF.
2. **Cause.** Number: the compiled declarations, actual bounds/extents, and
   call-time `Kmm` for the two live thickness arrays.  Prediction: the writer's
   array sections are the defect, not the parser's generic length arithmetic.
   CONFIRM only if the compiled source and record together explain the two
   zero-length payloads.  REFUTE if those sections are valid and full-sized;
   in that case do not guess and request a fresh self-describing acquisition.
3. **Existing-record admission.** Prediction: the existing record is not
   admissible for a KEG/ZAD replay if either live thickness payload is absent.
   A parser may diagnose it but must not invent or reconstruct missing carried
   state.  CONFIRM by a fail-closed missing-field status.  REFUTE only if both
   payloads are physically present under a schema the parser misread.
4. **Acquisition repair.** If the writer is defective, change only the two
   WRITE operands needed to make their payloads explicit, keep all NEMO
   arithmetic unchanged, use a new target name, and preserve header-derived
   admission plus restart identity and firing controls.  Prediction: the
   repaired preflight and Fortran syntax proof pass.  Any removed arithmetic,
   reused target, uncommitted dependency, or non-firing plant REFUTES it.
5. **KEG then ZAD walk.** This is conditional on a fully admitted record.
   Score the recorded before/after KEG boundary first, then ZAD.  Prediction
   retained from round 62: KEG is bit-exact given NEMO operands (0 unequal U,
   0 unequal V); any unequal bit REFUTES it and names KEG as the first non-bit
   child.  ZAD remains UNMEASURED_WITH_SPEC if a valid record is unavailable.
6. **Landing.** No model statement lands unless the first unequal statement is
   isolated, source-cited, and passes the complete ORCA2/shared-card gates.
   A repaired acquisition instrument alone yields `STOPPED_FOR_RECORD` with
   `ACQUISITION_NEEDED` naming its committed launcher.

## Controls and choices

The parser must refuse a field-order plant, a header/shape plant, a physical
truncation plant, a restart-identity plant, and a producer-stamp plant.  Any
new record checker parses magic, header integers, per-field
`(name, rank, n1, n2, n3, payload)`, and physical EOF; it may not accept a
hand-predicted file size.

ASKED: reconcile the operator's round-62 refusal and continue the ORCA2 KEG/ZAD
walk if the record is complete.

UNASKED: none.  No configuration, selector, threshold, state, forcing,
stabiliser, sea-ice field, or NEMO arithmetic statement may change.
