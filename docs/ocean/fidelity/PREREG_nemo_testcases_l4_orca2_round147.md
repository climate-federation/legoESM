# Preregistration — ORCA2 round 147 T-pivot U association

Date: 2026-10-04. Frozen base: `04abf66161`.
Claim class: rung-0 results are **independent**. No configuration, initial
state, forcing, sea-ice, stabiliser, carried-state, threshold, or
`unmeasured_features` choice is authorised.

## Source statement and scope

The rung-0 build executes the U-point arm of NEMO's T-pivot association at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:639-683`. With
`nn_hls=2`, its pivot-row loop leaves native U columns 0..89 as computed and
writes columns 90..179 as the sign-minus reverse of columns 89..0. The one
seven-field caller is
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.

Round 146 proved that its passive observer moves no returned-state bit, that
the other six post-call arrays are exact, and that U differs in 56 pivot-row
signed zeros only. This round changes only the existing private seven-array
association helper and its controls. No public selector or production card
changes.

## Frozen protocol

1. Extend the existing helper; do not add a second association routine. Apply
   the compiled U pivot-row half-copy after the compact periodic closure.
2. Prove with a synthetic row that the right native half is the sign-minus
   reverse of the left half, the left half is untouched, and a wrong sign
   changes bits even when every magnitude is zero.
3. Re-run the round-146 gate on CPU, production JIT, fp64/libm. Require all
   seven post-call arrays bit-exact before interpreting the causal arm.
4. If the prerequisite passes, score substeps 1 and 2 in the frozen round-129
   source order. This round may name the next statement but may not introduce
   a downstream divergence or transport rewrite.
5. A production landing remains ineligible unless the complete call closes
   both divergence components and SSH, both ORCA2 ladders avoid the registered
   ~31 PSU salinity exposure, and every shared-card gate passes.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R147-P1 | The compiled half-row copy changes only the 56 U sign bits left by round 146. | U becomes bit-exact, the other six rows stay exact, and no interior or unregistered boundary bit moves. | U retains any unequal bit, any magnitude changes, or another row leaves exactness. |
| R147-P2 | With all seven post arrays exact, the complete-call arm still closes substep-2 `continuity_du`. | `continuity_du` has zero unequal active and full-domain cells. | Any `continuity_du` bit remains unequal or an earlier row moves. |
| R147-P3 | The next source-ordered debt is the V transport-difference statement already exposed by round 146's incomplete diagnostic arm. | First non-bit is substep-2 `continuity_dv`, 68 active cells, maximum 155776.5627856178 transport units. | A different boundary is first, or either registered number differs. |
| R147-P4 | The exact call does not yet make SSH exact. | Substep-2 `after_ssh` remains 68 unequal active cells with maximum 0.003203816535399729 m. | `after_ssh` is exact or has a different registered census/magnitude. |
| R147-P5 | The private-hook-only change leaves every ordinary production trajectory unchanged. | Ordinary ORCA2 passivity is exact and the required GYRE base/tip trajectory and month comparisons are byte-identical. | Any ordinary production bit moves. |

## Controls and terminal rule

The gate must refuse a wrong U-fold sign, a post-association ULP, a passivity
ULP, and a reordered seven-field registry. The citation gate must pass and its
rigid-shift plant must fire. If R147-P1 passes but R147-P3 confirms, the round
is HELD at the V transport-difference statement with no public production
landing.

ASKED choices: none. UNASKED choices: empty.
