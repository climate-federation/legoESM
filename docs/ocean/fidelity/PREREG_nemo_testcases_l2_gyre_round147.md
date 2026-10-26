# Preregistration — NEMO testcase L2 GYRE round 147

Date: 2026-09-21

Incoming lane tip: `8f3f3565d3f7c8f9ab819fd2fdfb79a3ca3210c6`.
Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round147/`.
This document is frozen after the operator reported the Round-146 acquisition's
single decisive refusal line and before inspecting or decoding the produced
artifacts.

## Reported fact and question

The operator ran the committed Round-146 acquisition.  NEMO completed, but the
script exited 71 at `REFUSE: passive instrument moved inherited
oracle_developed_rhs_kt00001081.bin`.  Therefore the Round-146 family record is
not admitted and none of its family values may be interpreted yet.

The Round-146 compiled writer records every full `uu(:,:,:,Krhs)` and
`vv(:,:,:,Krhs)` array between the five dynamics calls at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192`.
The inherited Round-140 writer records those full arrays plus their masks after
ZAD at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:180-192`.
The discrimination is whether the extra writes changed a model-owned value or
whether the byte comparison included dry, halo, or other unowned array cells.

## Frozen predictions and falsifiers

The primary prediction is an admission false positive: both restart files and
all other inherited records remain byte-identical; the moved Round-140 record
has identical header, geometry, masks, and wet owned RHS cells, with differences
confined to cells excluded by its native U/V masks.  This is CONFIRMED only if a
closed field-by-field decoder accounts for every changed byte and reports zero
unequal values wherever the corresponding native mask is nonzero.

The falsifier is any changed restart byte, any changed inherited record other
than the reported file, any changed record header/shape/mask/geometry field, or
any unequal U/V RHS value on a native-mask-nonzero cell.  Such a result proves
the interleaved writer is not passive; the family record stays refused and a
new acquisition under a new target name is required.

The secondary prediction is that the new family's final ZAD boundary is bit
identical to the new run's own inherited Round-140 completed-RHS boundary on
all cells and bit identical to the admitted Round-140 boundary on all wet owned
cells.  Either comparison failing on an owned cell also refuses the record.

## Allowed repair and controls

If and only if all movement is confined to native-mask-zero cells, amend the
existing Round-146 admission gate to compare every scientific field by its
registered ownership mask while retaining byte-identical restart admission and
byte-identical comparison for headers, geometry, masks, and fully owned
records.  Register the excluded-cell census and maximum difference; do not
silently drop cells.  A one-ULP change to a wet RHS cell and a one-byte restart
change must each print `STATUS PLANT-FIRED` and exit nonzero.  A dry-cell plant
must demonstrate the registered exclusion without allowing a wet change.

If an owned cell moved, do not weaken the gate.  Read the compiled source and
build flags to identify the execution-context change, then write a new
fail-closed acquisition script with a new target name.  Do not run `mpirun` in
the sandbox; report that script as `ACQUISITION_NEEDED`.

Only after admission may the frozen Round-146 directed family ranking proceed.
If admission cannot be repaired without a new NEMO record, stop this round for
that record.  No physics, configuration, default, carried state, scheme,
stabilizer, canonical NEMO source, or immutable before arm changes.  DINO,
LOCK_EXCHANGE, OVERFLOW, and ORCA2 have no executing production change.
