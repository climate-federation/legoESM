# NEMO testcase lane 2 GYRE — phase-3 round-11 preregistration

Date: 2026-09-04
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`
Reviewed baseline: `d124c77c94c9307cec8a74b564b0ad2c0829b990`

## Frozen boundary and prediction

The production-JIT/fp64 source-term rows at stage 2 are already AT-BAR: HPG
u/v `1.07e-16`/`1.22e-16`, vorticity about `2.5e-21`, and advection about
`1.7e-24`.  The completed Kaa state is DEBT at
`4.67e-13`/`5.19e-13` u/v.  The preregistered prediction is that the first
owner-capable departure appears at the source-ordered, unprojected full Krhs
as it enters the stage update: an O(`6e-17`) raw association difference is
AT-BAR as a tendency but is amplified by stage-2 `rDt=7200 s` to the observed
O(`5e-13`) Kaa debt.  Replacing only the complete Krhs with NEMO's dumped
operand is predicted to clear both raw and mean-replaced Kaa to `1e-15`.

CONFIRM this owner only if the oracle-Krhs arm moves corrected Kaa at the
faithful residual scale (movement/residual in `[0.9,1.1]`) and leaves it
AT-BAR.  REFUTE it if movement is below 0.1 residual or corrected Kaa remains
DEBT.  If Krhs injection clears raw Kaa but not post-replacement Kaa, ownership
moves to `_replace_stage_mean`; if Krhs matches but raw Kaa first exceeds the
bar, ownership moves to the update statement itself.  Scaling is printed
before every label.

## Source-ordered oracle record

The config-local WRITE-only `MY_SRC/stprk3_stg.F90` record will be regenerated
and accepted only if the ordinary stage/restart hashes remain identical.  At
stage 2 of the first whole step it records, in order:

1. `uu/vv(:,:,:,Krhs)` at entry and after HPG, vorticity, and advection;
2. the accumulated full Krhs immediately before the stage update;
3. `uu/vv(:,:,:,Kaa)` immediately after the update;
4. `uu_b/vv_b(:,:,Kaa)`, diagnosed `zub/zvb`, and Kaa after replacement.

NEMO's resolved order is EOS/HPG, vorticity, then advection
(`stprk3_stg.F90:338-383`).  Stage 3 alone later adds `dyn_ldf`
(`:447-455`); `dyn_spg` has already placed its barotropic contribution into
Krhs before this stage block.  Resolved GYRE has `ln_dynadv_vec=.true.`, so
stage 2 executes the velocity recurrence
`(Kbb + rDt*Krhs)*mask` at `:414-417`.  The QCO thickness-weighted recurrence
using `(1+r3u/r3v)` at Kbb/Kmm/Kaa (`:419-435`) is source-cited and tested as
the non-selected NEMO alternative; it must not be applied to GYRE.  The
reference-thickness mean replacement is `:482-510`, with
`zub = uu_b(Kaa) - SUM(e3u_0*uu(Kaa))*r1_hu_0` before addition.

## One-variable arms and controls

The ordered arms are: (a) NEMO call-order RHS addition with optimization
barriers versus the current shared association; (b) exact selected update
association versus its private legacy alternative; (c) oracle operands for
the reference-depth mean reduction/addition versus production.  Each hook is
private and cannot be selected by a card.  A planted Krhs cell and a planted
post-replacement cell must each exit nonzero.  Dtype, header, dimensions,
time-level registry, finite owned values, and instrumentation bit identity are
fail-closed.

Only if the corrected stage-2 Kaa clears will the walk proceed to stage-3
`zFu/zFv/zFw`.  Only if both boundaries clear will any `tra_zdf/dyn_zdf`
matrix work begin.  The round-10 compare-to gate and EOS are immutable in this
round.  No external review has yet occurred.

## Registered redirect after the composition arms

The immediate pre-update dump refutes the original accumulation-association
prediction: changing only that association produces zero move, while replacing
the complete RHS with the oracle makes raw `Kaa` bit-identical.  The observed
scale instead localizes the amplification to HPG: its u/v residuals multiplied
by `rDt=7200 s` equal the raw-Kaa residuals.  Before opening that operator, the
next prediction is that the first differing literal operand is the horizontal
metric reciprocal or the `zhpi + zuap` association at the deepest wet level,
where both maxima occur.  A WRITE-only `dynhpg` record will separate cumulative
`zhpi`, local `zuap`, and their stored sum.  CONFIRM metric/association ownership
only if one of those operands first exceeds the effective stage-output bar
`1e-15/7200`; REFUTE it if all three meet that tighter bound or an earlier
density/e3w/depth operand differs.  No diagnostic value may feed either model.
