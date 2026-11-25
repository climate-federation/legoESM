# Preregistration — ORCA2 round 148 V-face north-neighbour association

Date: 2026-10-05. Frozen base: `ca49e5e393`.
Claim class: rung-0 results are **independent**. No configuration, initial
state, forcing, sea-ice, stabiliser, carried-state, threshold, or
`unmeasured_features` choice is authorised.

## Source statement and scope

The rung-0 build extrapolates `zsshp2_e`, then constructs the V-face depth
from its local and north-neighbour T cells at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545` before
forming `zhV` and the south difference at `dynspg_ts.f90:568-585`.
`zsshp2_e` inherits the prior substep's associated `ssha_e`; the executed
T-pivot T arm fills the full northern halo from the row below at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:584-617`, and the
seven-field call carrying `ssha_e` is at `dynspg_ts.f90:770-779`.

Round 147 proved all seven compact post-call arrays exact and closed the U
difference. Its substep-2 full-domain registry leaves the first prerequisite
differences at the northern V-face depth (30 cells, maximum 899 m), followed
by V transport (68 cells, maximum 155776.5627856178 transport units), V
difference (68 cells, same maximum), and SSH (68 cells, maximum
0.003203816535399729 m). Active-only scoring hid the first two boundary rows;
this round scores both active and full-domain operands in source order.

The one-variable arm changes only the compact north neighbour supplied to the
existing NEMO SSH-average V-depth formula: on a stored T pivot it uses the
already-shared `fold_ghost_source_T` row-below source before the existing T
permutation. The stored pivot-row source is the frozen control. No transport,
difference, or SSH statement may be changed in the arm.

## Frozen protocol

1. Reuse the admitted round-95/97 substep record and the round-146/147 passive
   trace on CPU, production JIT, fp64/libm, x64.
2. Extend the existing private complete-association arm; do not add a second
   solver, callback, or public selector. Ordinary production remains unchanged.
3. Require round 147's seven post-call rows and observer passivity to remain
   bit-exact before interpreting the V-depth arm.
4. Score substeps 1 and 2 in the frozen round-129 source order, including the
   full-domain operand rows. Plant the old stored-pivot north source and one ULP
   in the repaired V-depth boundary; both must refuse.
5. Only if the arm closes V depth, V transport, V difference, and SSH without
   moving an earlier exact row may the two ORCA2 ladders be run. A production
   landing additionally requires no exact/AT-BAR loss, unchanged first-over-bar,
   no return of the registered ~31 PSU salinity exposure, and all shared-card
   gates.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R148-P1 | The ordinary observer and seven-field association remain exact. | All returned state arrays and all seven post-call arrays are bit-exact. | Any bit moves. |
| R148-P2 | The stored-pivot-row source is the first wrong compact operand. | Frozen control reproduces 30 unequal northern V depths and 68 unequal V transports at substep 2. | Either census differs or an earlier full-domain operand is non-bit. |
| R148-P3 | NEMO's row-below T-halo source closes the face-depth boundary. | Substep-2 `mid_depth_v` has zero full-domain unequal cells. | Any bit remains or an earlier exact row moves. |
| R148-P4 | With that depth exact, the unchanged downstream statements close in order. | `transport_v`, `continuity_dv`, and `after_ssh` each have zero full-domain unequal cells. | Any row remains non-bit or another statement was needed. |
| R148-P5 | The source-exact arm does not revive the held tracer compensation. | Both ladders lose no exact/AT-BAR row, first-over-bar is unchanged, and kt=10 stage-3 S stays below the registered ~31 PSU exposure. | Any landing veto fires. |
| R148-P6 | A failed arm leaves production unchanged. | The arm stays private/default-off and verdict is HELD. | Any production/card/deck diff remains. |

## Controls and terminal rule

The gate must refuse the stored-pivot north-source plant, a V-depth ULP, the
existing post-call ULP, passivity bit, U-fold sign, and registry reorder. The
citation gate must pass and its rigid-shift plant must fire. A landing requires
the full standing gates; otherwise this is a measurement-only HELD round.

Pre-implementation search: the repository already provides
`fold_ghost_source_T` in `packages/core/legoesm/grids/operators_latlon_cgrid.py`;
this round reuses it and extends the existing round-146 gate/helper.

ASKED choices: none. UNASKED choices: empty.
