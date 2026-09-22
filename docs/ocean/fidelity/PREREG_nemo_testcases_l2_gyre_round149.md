# Preregistration — NEMO testcase L2 GYRE round 149

Date: 2026-09-22

Incoming lane tip: `903625dd150777539a84e06e2e176f25256e8dca`.
Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round149/`.  This document is
frozen after the operator reported the Round-148 acquisition refusal and before
decoding or comparing the refused inherited record.

## Reported fact and source boundary

The operator ran the committed Round-148 acquisition.  NEMO completed with
`STOP 0`, the direct 4,717,612-byte LDF record exists, and the script then
exited 71 at `REFUSE: passive instrument moved inherited
oracle_developed_rhs_kt00001081.bin`.  The record is not admitted and no LDF
operand may be interpreted until that movement is closed.

The compiled Round-148 writer allocates and zeroes two diagnostic arrays,
writes the LDF inputs before the operator, copies the assigned `zwf`/`zwt`
extents, and writes the post-LDF accumulators at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:111-140` and
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:142-197`.
The inherited Round-140 record writes the full completed U/V RHS arrays and
their native masks before depth averaging at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:204-217`.

## Frozen discrimination

The primary prediction is that the refusal is the already registered
whole-array ownership trap: the two restarts and all other inherited records
are byte-identical, the Round-140 header/registry/geometry/masks are BIT, and
any changed completed-RHS cells have zero native U/V mask.  This is confirmed
only by a closed decoder that reports every field, the owned/excluded census,
unequal cells, and maximum absolute difference.

The falsifier is any changed restart byte, any changed inherited record other
than the reported Round-140 record, any changed header/registry/geometry/mask,
or any completed-RHS change where the corresponding native mask is nonzero.
That result proves the new writer is not passive; the Round-148 record stays
refused and a new target name is required.

The secondary prediction is that the new record's pre-LDF accumulators are BIT
to the same-run Round-146 HPG boundary and its post-LDF accumulators are BIT to
the same-run Round-146 LDF boundary on all native wet faces.  Failure on an
owned face also refuses the record.

## Allowed repair and measurement

If and only if movement is confined to registered unowned cells, extend the
existing Round-148 admission path to use the Round-146 ownership-aware parent
comparison while retaining byte identity for restarts and every fully owned
inherited record.  Add an `--admit-existing` path which verifies the stored
binary hash and source manifest and never reruns `makenemo` or `mpirun`.  Wet
RHS and restart-byte plants must print `STATUS PLANT-FIRED` and exit nonzero;
an excluded-cell plant must prove the exclusion without allowing a wet change.

After admission, extend the existing Round-50 walker only.  Reproduce Round
147's LDF-family gap, score recorded inputs and compiled `zcur`/`zdiv` in source
order under the production step, and distinguish the isolated LDF result from
the in-place accumulator.  The frozen scientific prediction remains that the
first non-bit operand is the live F-point thickness.  It is CONFIRMED only if
all earlier operands are BIT and that field is non-bit; it is REFUTED if the
field is BIT or an earlier operand differs.  The production plant must move a
registered consumed row.

Only a source-exact candidate may proceed through the full Decision-43/45
ladder, month, year, and DINO gates.  No configuration, default, carried state,
scheme, stabilizer, canonical NEMO source, or immutable before arm changes.

