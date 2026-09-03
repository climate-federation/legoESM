# NEMO testcase lane 2 GYRE — tracer-stage operand preregistration

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Parent geometry landing: `35cb4458e`.

The accepted first open boundary is stage-1 temperature/salinity, respectively
`0.3241418107674683` and `1.8542789845810148e-6` absolute.  The full oracle
stage-1 T/S/SSH bundle removes `0.9999992257753059` of the downstream stage-2
momentum residual, but that causal result does not assign an operator owner
inside the tracer stage.

## Source order and pre-implementation search

The source walk found that each NEMO RK stage first zeros tracer `Krhs`, then
calls `tra_adv`, then `tra_sbc_RK3` (`stprk3_stg.F90:510-527`).  Stages 1 and 2
then restart from `Kbb` and apply the QCO content recurrence using the live
`Kmm` RHS thickness and `Kaa` output thickness (`:540-562`).  On this cold-start
Euler step, `trasbc.F90:121-137` sets the previous and current surface-content
fields and `:150-155` deposits the SBC rate into the top cell.

The repository search found the canonical Kmm transport geometry, two-step FCT
predictor, QCO stage combine, and the three-stage WS helper.  It also found a
structural difference: the production step first applies the complete frozen
Euler tracer tendency and calls the WS helper from that `T_mid/S_mid`
(`ocean_model_latlon_cgrid.py:5149-5150,5652-5668`), while the helper itself
contains advection only (`:1035-1132`).  That is the registered hypothesis, not
an owner verdict.

## Frozen measurements

A WRITE-only NEMO stream at `kt=1` will record, for stages 1 and 2:

1. zeroed `Krhs` immediately before `tra_adv`;
2. `Krhs` immediately after `tra_adv`;
3. `Krhs` immediately after `tra_sbc_RK3`;
4. `Kbb`, `Kmm`, and the integrated `Kaa` tracer fields plus the QCO `r3t`
   operands used by the stage combine.

The candidate will expose the matching advection rate, SBC rate, base content,
and integrated stage through a private diagnostic return.  All comparisons use
fp64 native active cells.  The reader must fail closed on magic, header, stage
registry, dtype, and payload size.  A planted `+1` in the oracle post-SBC
temperature RHS must fire at at least `1.0`; all prior dumps remain SHA-256
bit-identity controls.

## Confirm/refute and landing rules

- The first source-ordered operand over `1e-15` is the boundary; scaling is
  printed before any label.
- If the post-advection row first fails, walk the FCT reconstruction inputs
  before changing SBC or integration.
- If advection clears and post-SBC first fails, compare the literal surface
  content/rate, including the cold-start `zfact=1`, before changing the stage
  combine.
- If both rates clear but `Kaa` fails, walk the QCO thickness/content recurrence.
- An arm moving less than 10% of the faithful residual is
  `NEAR_NULL_NO_DISCRIMINATING_POWER`, never exoneration.
- A production fix may land only as the collapsed NEMO WS-RK3 identity.  NEMO
  has no switch for pre-applying all tracer physics before its stages, so no
  public micro-selector or mixed-stage Frankenstein configuration is allowed.

After a supported landing the same fp64 CPU gate reruns kt=1, then kt=2-10.
The walk continues at the next first-over-bar boundary until kt=2 clears or the
registered operands are exhausted.  Until then trajectory matching is DEBT and
the later boundary is UNMEASURED.
