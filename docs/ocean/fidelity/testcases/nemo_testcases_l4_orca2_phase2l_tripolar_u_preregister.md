# ORCA2 Phase-2l tripolar U-face layout preregistration

Date: 2026-09-06

The compiled stage-transport walk exposed a card-owned boundary before any
shared transport arithmetic: `create_tripole_grid` stores NEMO's native
east-face `e1u/e2u` array at redundant U-face index 0, while every C-grid
consumer treats redundant index 0 as the periodic west face and compares
native NEMO U faces with `grid.{dx,dy}_u[:,1:]`.  On ORCA2, the current
`dy_u[:,1:]` differs from the recorded NEMO `e2u` in 1,961 / 8,568 wet faces;
`dy_u[:,:-1]` is exact in 0 / 8,568.

Preregistered single change: place the periodic last native U face at redundant
index 0 and the complete native U array at indices `1:`.  Apply the same
staggering map to `e1u`, `e2u`, and native U-point rotation angles.  Do not
change V-face metrics, `f_v`, `ff_f`, T fields, fold permutations, masks,
physics schemes, or any ocean operator.

Acceptance:

- `grid.dx_u[:,1:]` and `grid.dy_u[:,1:]` are bit-exact to NEMO `e1u/e2u`;
- redundant U endpoints both equal the native periodic last face;
- ORCA2 entry T/S/u/v/SSH remains bit-exact;
- the existing GYRE card is unchanged (it does not use `create_tripole_grid`);
- a synthetic tripole unit test pins the stagger map and separately pins V
  metrics and `f_v` unchanged; and
- the compiled production stage-transport/W gate is rerun.  Any remaining
  first mismatch is assigned only after its operand row is measured.

The owner is `ORCA2_OWNER_TRIPOLAR_U_FACE_LAYOUT`.  This is domain-file
staggering, not a shared EOS/HPG/transport/ZDF numerical repair.
