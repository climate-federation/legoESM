# PREREG — grid ladder G6, round 6 (gridladder_r42)

Frozen before the committed card is run.  Decision 132 (user, 2026-10-16):
the G6 card selects `nemo_first_wzv_after_ssh='rk3_extrapolated_carried'`.

## Statement selected (cited, G5 round-31 build = ORCA2_OMIP_L4 unless noted)

- written: `ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)`, `stprk3.f90:241`
  (G5 r31) / `:237` (ORCA2_OMIP_L4), SUBROUTINE `stp_RK3` :62.
- read: `r3t(:,:,Kaa) = ssh(:,:,Kaa)*r1_ht_0` `stp2d.f90:152`, then the
  first vector-form `CALL wzv` `:156` (SUBROUTINE `stp_2D`).
- kt-1 fill (from rest, `ln_rstart=.false.` in the G6 deck):
  `ssh(:,:,Kaa) = ssh(:,:,Kbb)` `restart.f90:406` (G5 r31) / `:405`
  (ORCA2_OMIP_L4), SUBROUTINE `rst_read_ssh` :321.  The shared model's
  unseeded slot (`None`) reads the step-entry height: the same value.
- restart entry (given-NEMO-entry label): `CALL iom_get(...,'ssha',
  ssh(:,:,Kaa))` `restart.f90:367` (both builds), written by `rst_write`
  `:184` after the :241 extrapolation.  The ladder's NEMO entry seeds the
  slot from the restart's `ssha`.

## Predictions

P1  kt 1: every field identical to round 40 (independent = given).
P2  kt 2 independent sea level 4.461e-3 -> ~4.394e-8 m on ~2,396 cells
    (round-39 scratch arm); u ~8.4e-7.
P3  given-entry kt 3/5 rows improve by a similar factor (>=1e4) over round 40.
P4  No other card's config changes (the option is set on G6 only).

## Falsifiers

kt 1 moves -> the kt-1 fill is wrong.  kt 2 not within 10% of 4.394e-8 ->
the committed card differs from the scratch arm.  Given-entry kt 3 not
improved -> the `ssha` seed is not reaching the step.
