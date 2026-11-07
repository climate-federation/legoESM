# NEMO testcase lane 2 GYRE — substep-2 ENE operand preregistration

Date: 2026-09-01  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Parent evidence commit: `efdef59b28afd13bba028487dc7989df992370ca`

This freezes the operand walk before the WRITE-only oracle dump or legoESM
coefficient path changes.  The registered target is the first over-bar frame:
substep-2 `trd_u`, absolute error `8.602761661688124e-12`, which is `1.5904%`
of the `5.409032045634095e-10` oracle magnitude.  No owner follows from that
small absolute number alone.

## Reviewed residuals closed before new measurement

The receipt prints the relative ENE error beside the absolute value.  Its EMP
retraction now rests on both the seven-digit SSH runtime discriminator and the
executing reduction source: `lib_fortran_generic.h90:92,144-148` applies
`smask0_i`; `dommsk.F90:200-205` constructs that unique interior-domain mask.
Future reversals of reviewed findings will return to external review with the
discriminating evidence before landing.

## Source read and pre-implementation search

The executing GYRE source is
`cfgs/GYRE_OMIP_L2_P3/MY_SRC/dynspg_ts.F90`:

- `:552-560` selects `(za1,za2,za3)=(1,0,0)` while `jn<3` and `ll_init`, so
  substeps 1 and 2 use forward prediction, not AB3 extrapolation;
- `:563-572` forms the live ENE inputs `ua_e` and `va_e`; at substep 2 these
  are exactly the substep-1 exits `un_e` and `vn_e` after history rotation at
  `:836-844`;
- `:1418-1446` freezes the eight ENE coefficients from Kmm thicknesses,
  masks, inverse F-point thicknesses, metrics, depths, and `ff_f`;
- `:1528-1540` multiplies those coefficients by four neighboring live
  velocity inputs in two source-ordered pairs.

The existing trace already records `ua_e`, `va_e`, `zu_trd`, and `zv_trd` for
every substep.  The repository search found the canonical literal EEN builder
`_nemo_literal_een_coefficients`, its source-ordered four-corner application,
the existing trace reader, native-mask scorer, and planted-control framework.
It found no literal ENE coefficient sibling.  Any implementation therefore
extends those shared mechanisms; it must not add a second barotropic operator,
trace format, or scorer.

## Frozen operand walk

CPU fp64 is pinned.  The pointwise normalized maximum bar remains `1e-15` on
native owned masks.  The WRITE-only extension will dump the eight frozen ENE
coefficient arrays after `dyn_cor_2D_init`; the seven existing oracle artifacts
must retain their registered SHA256 values bit-for-bit.

The gate will classify the first divergence in this order:

1. predictor weights and the substep-2 `ua_e`/`va_e` state;
2. all eight frozen `ffu_*`/`ffv_*` coefficient arrays;
3. each of the eight coefficient-times-neighbor products;
4. the two pair sums and final source-ordered sum for each component;
5. the recorded `zu_trd`/`zv_trd` totals.

The direct-product rows use the already dumped substep-2 state and the newly
dumped coefficients.  A nonzero coefficient control will perturb the largest
finite owned coefficient by a registered amount and must make the coefficient
gate fail.  Corrupt magic/header controls must also fail closed.

## Scaling before owner and causal arm

For each operand row the report records absolute error, oracle magnitude, and
relative error.  The causal arm replaces only legoESM's eight ENE coefficients
with the dumped oracle arrays while leaving predictor state, forcing, pressure,
depths, timestep, and 50-substep recurrence unchanged.

- If state is AT-BAR, a coefficient first divergence exists, and oracle
  coefficient injection clears the substep-2 term at `1e-15`, the coefficient
  builder is `CONFIRMED_CAUSAL_OWNER_AT_SUBSTEP2`.
- If coefficients are AT-BAR but a product or pair sum first differs, the
  arithmetic/application boundary is the measured owner; no builder change is
  authorized.
- If injection moves the residual but does not clear it, the label is
  `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER` with movement/residual scaling printed.
- Movement below one tenth of the faithful residual is
  `NEAR_NULL_NO_DISCRIMINATING_POWER`, never exoneration.

Only a clearing causal arm authorizes selecting the existing literal
coefficient option in the pinned GYRE card.  That selection is the sole run
variable; no default changes.  The kt=1 exact gate is rerun first, then kt=2
and the same CPU/fp64 kt=10 sweep.  Any surviving trajectory row remains DEBT
and every uninstrumented internal operand remains UNMEASURED.
