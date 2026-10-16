# PREREG — grid ladder G6, round 3 (gridladder_r38)

Frozen before any card change is measured.

## Offline attribution already run (evidence `gridladder_rounds/round38/`)

The card's stage-2 `pv_flux_al81_partial_cell` operands at kt 1 were captured
(debug callback, passive) and the captured output reproduces the round-37
dumped card accumulator (`after_hpg + out*mask == after_vor`, every cell).
Replaying that SAME operator with ONE operand changed — the relative
vorticity recomputed from the V field whose pivot row is NEMO's fold image
(`lbc_nfd_dp` 'V', `lbcnfd.f90`, pivot row ipj-ihls overwritten from the row
below with sign -1) — gives 0 unequal U and 0 unequal V cells against NEMO's
`after_vorticity` record at every level (was 1292 U / 2630 V).  The captured
zeta equals the zeta of a V field with the pivot row ZERO: the stage state
handed to the tendency (`_mom_pert_ws`) multiplies V by the compact
`v_mask_3d`, which is dry on the pivot row, while NEMO's `vor_een` np_CRV
(`dynvor.f90:762-766`) reads `pv = vv(Kmm)` there.

## Candidate (G6-card-selected through `rk3_vector_raw_vmask`)

Under `rk3_vector_raw_vmask` the WS-RK3 stage state handed to the momentum
tendency keeps V on NEMO's raw support (`_ws_stage_v_mask`, the mask the
stage update already uses) instead of the compact mask.  Every other card
unchanged (flag default False; only the G6 card selects it).

## Predictions

P1  kt-1 stage-2 `after_vor`: 0 unequal U, 0 unequal V, every level.
P2  `after_hpg` stays 0/0 every level.
P3  `after_zad` unequal count falls (round 37: U 84 / V 197 surface); it may
    not reach 0 (KEG/ZAD fold operands not yet walked).
P4  G1, G1b, G2, G4, G5 cards: configuration and focused tests unchanged.

## Falsifiers

Any unequal VOR cell at kt 1 stage 2; any HPG cell moving; any non-G6 card
selecting the change.  If P1 fails the stage-state mask is not the owner and
the narrower consumer (zeta reads the imaged V only) is measured instead.
