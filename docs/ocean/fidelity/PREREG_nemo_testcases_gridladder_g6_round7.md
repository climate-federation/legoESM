# PREREG — grid ladder G6, round 7 (gridladder_r43)

Frozen before the landed statement is scored.  DEVIATION, stated: the
stp_2D walk below (kt 1-3, record round 42 `oracle_g6_r42/twin_s`) ran
before this file; only the landing and the kt 1..6 re-score follow it.

## What the walk measured (card at `6f01650a57`, uncommitted walk arm)

- kt 1 (independent = given): every stp_2D boundary bit-exact (3-D RHS after
  HPG, VOR, KEG, ZAD; depth mean; after drag; after wind; dyn_spg_ts exit).
- kt 3 given-NEMO-entry (restart 2 incl. `ssha`): HPG, VOR, KEG exact; the
  first non-bit operand in compiled order is the first wzv's `ww`
  (levels 2..30: 90,618 values on 5,745 columns, ulp-scale), then dyn_zad.
- The card's `wzv` run EAGERLY on the same entry reproduces NEMO's `ww`
  exactly (levels 2..30: 0 unequal); under `jax.jit` it differs on 5,744
  columns.  Every recurrence input (per-level divergence, r3 after/before)
  is equal eager vs jit when exposed; materialising `r3_after` and
  `r3_before` with the shared `nemo_source_round` makes the jitted `ww` exact
  (0 unequal).  Mechanism (PLAUSIBLE, read from the HLO rule, not dumped):
  XLA strips `optimization_barrier` and factors
  `ssh_a*r1_h0 - ssh_b*r1_h0` into `(ssh_a - ssh_b)*r1_h0`.

## Statement landed (SUBROUTINE; G5 round-31 build = ORCA2_OMIP_L4 lines)

NEMO stores both ratios as arrays and the recurrence reads the stored values:
- `r3t(ji,jj,Kaa) = ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)` — `stp2d.f90:152`
  (SUBROUTINE `stp_2D`), before the first `CALL wzv` `:156`.
- `r3t(Kbb)`: `pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)` —
  `domqco.f90:257` (SUBROUTINE `dom_qco_r3c_RK3`) at the previous step's end.
- consumer: `pww(jk) = pww(jk+1) - ( ze3div + r1_Dt*e3t_3d*( r3t(Kaa) -
  r3t(Kbb) ) )*tmask` — `sshwzv.f90:297-298` (SUBROUTINE `wzv_RK3_t`).

Landing: a G6-selected config bool materialises both ratios before the
difference in the ONE shared `nemo_qco_wzv_operands` (every model call site
of that compiled line gets the card's value).  Default False; every other
card's config and program unchanged.

## Predictions (kt 1..6, both labels)

P1  kt 1 unchanged (r3 difference is 0 at kt 1).
P2  kt 3 given entry: first wzv `ww` levels 2..30 exact; the stp_2D exit
    (uu_b 1,997 / vv_b 1,980 cells) shrinks; the first non-bit boundary moves
    to dyn_zad or later.
P3  no row of either label gets worse than round 42 by more than last-place
    lottery (D96 class); the two given-entry kt-4 all-level DEBT rows
    (u 1.86e-15, v 1.38e-15) move toward the bar.
P4  no other card's jaxpr changes (default False).

## Falsifiers

kt 1 moves -> the guard changed something other than the ratio difference.
`ww` not exact at kt 3 in the model step -> another contraction inside the
step's fusion, the eager/jit probe does not transfer.  stp_2D exit not
improved -> the residual sits in dyn_zad or dyn_spg_ts, walk continues.
