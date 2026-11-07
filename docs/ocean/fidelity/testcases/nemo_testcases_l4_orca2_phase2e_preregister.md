# NEMO testcase Lane 4 — ORCA2 Phase-2e O1 schema correction preregistration

Date: 2026-09-06

Parent: `b35e96792cbb415f9d4f3b3c16e17e7330e14ea8`

## Observed stop, before interpretation

The user-shell run
`variant_icebergs_off_o1_instrumented_10step_np2` completed ten steps with
`MPIRUN_RC=0`, `RUN DONE`, and 92 `oracle_*.bin` files.  It is **RETRACTED AS
AN O1 ORACLE** because its new record does not satisfy its declared schema.

The first frame is valid: header `(1,1,0,90,148,9,0,64)` followed by exactly
`9*90*148` f64 values.  The second frame header is
`(1,1,1,90,148,20,0,64)`, but the record is 3,113,568 bytes rather than the
derived 3,090,336 bytes.  Source inspection identifies the excess exactly:
`sbc_oce.F90:184-216` allocates `emp`, `utau`, and `vtau` on the full
`jpi*jpj=94*152` local domain while the other frame-2 fields and header use
`A2D(0)=90*148`.  The WRITE list at copied-config
`MY_SRC/sbcblk.F90:710-713` omitted `A2D(0)` slices for those three fields.
The excess is therefore
`3 * (94*152 - 90*148) * 8 = 23,232` bytes, exactly the observed size gap.

## Instrument-inertness result and unresolved raw-byte question

The frozen ordinary-output identity check passes: all four restart shards are
exact bytes, all eight history payloads are exact apart from the registered
global timestamp, and the remaining ordinary outputs are exact under their
registered rules.  Thus the added writer did not change model arithmetic.

For the 91 inherited oracle records, 84 are raw-byte identical to the accepted
VARIANT instrumented run and seven differ.  In the post-`sbc` record, all 24
different f64 values are the first four elements of `utauU`, `vtauV`,
`utau_b`, `vtau_b`, `rnf_b`, and `rnf_tsc_b`; their magnitudes are pointer-like
`4.6e-310..6.9e-310`.  The other six differing streams repeat the same byte
pattern at corresponding carried-array positions.  They are not finite model
results initialized by NEMO and change across executable layouts while the
ordinary state remains exact.

No raw-byte waiver is registered here.  The replacement gate will continue
to report the exact inherited-record count and will fail a strict accepted-run
digest comparison.  Review must choose one of two non-Frankenstein remedies:

1. compare only source-defined cells/fields and explicitly register the
   uninitialized slots as outside the oracle payload; or
2. regenerate the accepted VARIANT record set after all affected writers
   canonicalize writer-local copies of undefined slots.

The lane will not score O1 or enter the NCAR bulk operator until that choice is
made.  This is the first boundary needing a decision.

## Preregistered schema-only replacement

One change is permitted before the decision because it is required under
either remedy: change only the O1 frame-2 WRITE list to
`emp(A2D(0)), utau(A2D(0)), vtau(A2D(0))`.  These are I/O-list array sections;
no model field or arithmetic is assigned.  Rebuild with the same scalar-math
architecture, require zero `_ZGV*`, and prepare but do not execute:

`variant_icebergs_off_o1_schemafix_instrumented_10step_np2`

Acceptance for the new O1 record is:

- exactly two `NEMO_L4_BLKIO_1` frames;
- headers `(1,1,0,90,148,9,0,64)` and `(1,1,1,90,148,20,0,64)`;
- exact size 3,090,336 bytes, finite payload, and EOF;
- header-count and digest-bound one-ULP plants exit nonzero;
- ordinary-output identity against the accepted icebergs-off uninstrumented
  control remains exact.

The prior malformed run and all its outputs remain preserved and flagged.
No legoESM operator, SI3 operator, shipped NEMO file, or comparison bar changes.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| validate O1 schema and both frames | ASKED | frame 1 PASS; frame 2 FAIL; run retracted as O1 oracle |
| confirm instrument inertness | ASKED | ordinary outputs and four restarts PASS; inherited raw records 84/91 exact |
| continue ladder and stop at first decision | ASKED | stopped before O1 scoring |
| prepare any required NEMO rerun | ASKED | one schema-only replacement run to prepare |
| slice three full arrays to the declared reduced domain | UNASKED necessary writer repair | WRITE-only, source-derived, no model assignment |
| waive or normalize undefined inherited bytes | UNASKED and decision-requiring | not done |
