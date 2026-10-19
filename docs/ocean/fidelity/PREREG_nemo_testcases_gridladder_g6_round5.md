# PREREG — grid ladder G6, round 5 (gridladder_r40)

Frozen before the landed card is re-scored against NEMO.  Stated honestly:
the OFFLINE replay below was run before this file (it named the operand);
the predictions concern the CARD after landing, which has not been run.

## Instrument (already run, evidence `gridladder_rounds/round40/`)

Literal numpy replay of `div_hor_RK3_t` np_velocity (`divhor.f90:125-130`,
SUBROUTINE :85) feeding `wzv_RK3_t`'s qco recurrence (`sshwzv.f90:298-300`,
SUBROUTINE :236), on the card's own captured stage-2 operands (velocities
equal NEMO's recorded `uu/vv(Kmm)` at every cell, 0 unequal).  Calibration:
reproduces the card's per-level divergence to 2.2e-16 relative on every wet
cell below level 1 off the pivot row (level 1 carries the card's runoff term,
which the replay leaves out).  Scored by the round-39 inversion of NEMO's
recorded stage-2 ZAD increment for the U-face pair `e1e2t*ww`: card pivot T
row median 4.3e-4 (reproduces round 39), interior control rows ~1e-12.

## Finding being landed

The card's pivot V-row sea-level ratio `r3v` is formed by its stencil over the
row "north" of the pivot T row, which wraps to row 0; NEMO's `r3v(Kmm)` there
is the `lbc_lnk` sign +1 image of the row below (`stprk3_stg.f90:169`, after
`dom_qco_r3c_RK3` :167).  Card / fold-image ratio median 0.496 on all 1319
wet pivot-V cells.  Replacing only that row by its fold image takes the pivot
T row pair to median 1.3e-12 (p99 1.0e-9), the interior control level.

## Predictions (card after landing, G6 flag `rk3_vector_raw_vmask` selects it)

P1  kt 1 stage 2 after-ZAD: the fold-row residual (U pivot T row ~5.5e-12,
    V rows <=4.9e-12) is removed; remaining after-ZAD unequal cells only the
    interior last-place set (U ~314 / V ~600, <=8.5e-22 absolute) or fewer.
P2  kt 1 step-end fields unchanged or closer (u 4.6e-9/135, v 2.6e-9/202).
P3  kt 2..6 whole-domain debt essentially unchanged (owner = Decision 132).
P4  No card other than G6 changes (flag default False; caller grep).

## Falsifiers

After-ZAD fold rows still unequal at >1e-13 -> a second pivot-row operand;
any field at kt 1 worse than round 39 -> regression, not landed.
