# Preregistration: GYRE LDF/ENE literal arithmetic follow-up, round 49

Date: 2026-09-11. Frozen after the first registered discriminator returned
`REFUTED` at clean producer `81985ff7d27e`. Evidence:
`round49/round49_ldf_ene_ladder_refuted.json`.

## Rule-11 disposition

The complete recorded LDF thickness family reduced stage-1 post-LDF from
17,400/17,100 unequal cells at maxima 2.529e-14/3.503e-14 to
10,626/10,411 at 4.798e-22/5.294e-22, and stage 3 to 5,008/4,487 at
8.470e-22/6.353e-22. It did not make either row exact, so the original frozen
prediction is REFUTED and the operand-only candidate is not a landed fix.
Recorded ENE reciprocals also left 4/2, 1/0, and 2/1 post-VOR unequal cells at
stages 1/2/3. The original ENE prediction is REFUTED too.

## Next source statement

The remaining scale is one final-operation ulp. The current LDF path composes
generic divergence, curl, and gradient helpers; those helpers divide by live
metrics and regroup fluxes. The compiled statement instead materializes each
metric-thickness-velocity product, two differences, their sum, and the
`ahm*r1_metric/e3` factors in the written order
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`). The
current ENE path likewise receives a generic curl, while compiled ENE forms
`((e2v*v east-e2v*v west)-(e1u*u north-e1u*u south))*r1_e1e2f` before adding
`ff_f` (`.../BLD/ppsrc/nemo/dynvor.f90:536-540`), then materializes the
transport and pair-sum order at `:555-573`.

Freeze one new cumulative arm: the already-tested exact operand families plus
literal, optimization-barriered evaluation of those cited statements. No
stencil, sign, mask, metric, or time-level choice changes. **CONFIRM** requires
zero unequal wet cells at post-LDF stages 1/3 and post-VOR stages 1/2/3.
**REFUTE** is any nonzero row; stop without landing. A one-nextafter LDF flux
product and ENE curl product must each make the gate exit nonzero. Default
non-NEMO routes remain byte-pinned; eager/JIT/grad must execute.
