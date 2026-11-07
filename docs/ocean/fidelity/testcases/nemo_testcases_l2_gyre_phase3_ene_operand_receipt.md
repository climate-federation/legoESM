# NEMO testcase lane 2 — GYRE ENE operand and kt=10 receipt

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Preregistration commit: `fab727768b6ad7a256ee216badc66cf582799a65`
Parent barotropic commit: `efdef59b28afd13bba028487dc7989df992370ca`

## Verdict

**DEBT, with the isolated live ENE component now exact and the first remaining
combined-tendency boundary reduced to the bottom-drag composition.**  The
historical generic ENE path's substep-2 `trd_u` error was
`8.602761661688124e-12`, or `1.5904%` of the
`5.409032045634095e-10` oracle magnitude.  Literal ENE reduces that combined
error to `6.731208045740773e-14` (`0.012444%` relative), while the isolated ENE
term itself agrees to `2.0679515313825692e-25` absolute.  Whole-step kt=2 and
the kt=10 trajectory remain DEBT; no trajectory is called matched.

## Review residuals and process

The prior receipt now prints the relative ENE figure beside the absolute one.
The corrected EMP interpretation rests on both the seven-digit SSH runtime
discriminator and the executing reduction source:
`lib_fortran_generic.h90:92,144-148` applies `smask0_i`, and
`dommsk.F90:200-205` constructs that unique interior-domain mask.

Claude's preceding review verdict was SHIP.  Its EMP objection was explicitly
re-reviewed and retracted before this round.  Codex claim review was attempted
before implementation but both CLI transports were network-blocked; GLM is
unavailable.  This round is therefore **UNREVIEWED**, not dual-reviewed.  The
standing process rule is recorded: any future reversal returns through review
with its evidence before landing.

## Source walk and time level

The executed `MY_SRC/dynspg_ts.F90` establishes the operand order:

- `:297` freezes the eight coefficients once from Kmm, and the WRITE-only dump
  follows immediately at `:300-312`;
- `:570-588` uses forward weights `(1,0,0)` for cold-start substeps 1 and 2, so
  substep 2 consumes the rotated substep-1 exit rather than an AB3 mixture;
- `:1440-1466` is the live `np_ENE` 1/4 coefficient recurrence;
- `:1551-1560` applies four neighbor products in two source-ordered pairs;
- `:744-750` adds bottom stress only after the pure ENE term.

The pre-implementation search found and extended the existing literal EEN
coefficient builder, its source-ordered four-corner application, the lane-2
substep trace, native-mask scorer, and lane-1 growth instrument.  No second
operator, trace format family, or scorer was introduced.  The GYRE card carries
the already certified analytic native-A2D operands and selects the literal ENE
sibling; defaults outside this card are unchanged.

## WRITE-only instrument and controls

The v2 trace adds the pure Coriolis arrays before bottom stress, and a separate
record contains all eight frozen coefficients.  Neither block assigns model
state.  All seven pre-existing step-entry, stage, RHS, and barotropic-frame
artifacts retain their registered SHA256 values bit-for-bit.

The synthetic 1/12-for-1/4 ENE violation made the literal-recurrence unit test
fail on all 28 nonzero cells with a `0.66666667` maximum relative error; restoring
1/4 made it pass.  The coefficient reader's planted `+1` control reports exactly
`1.0` and DEBT.  Corrupt magic/header checks remain fail-closed.

## First divergence within ENE

Scaling is recorded before labels:

| registered row | absolute max | oracle max | relative | status |
|---|---:|---:|---:|---|
| substep-2 U predictor state | `2.541098841762901e-21` | `5.790933390465428e-6` | `4.38893e-16` | AT-BAR |
| substep-2 V predictor state | `3.3881317890172014e-21` | `5.811892717783324e-6` | `5.83086e-16` | AT-BAR |
| worst of eight ENE coefficients | `6.776263578034403e-21` | `2.7733010124804627e-5` | `2.44339e-16` | AT-BAR |
| worst product/pair/final sum | — | — | `8.72720e-16` | AT-BAR |
| pure ENE `trd_u` reconstruction | `2.0679515313825692e-25` | `5.40845892639824e-10` | `3.82355e-16` | AT-BAR |
| historical generic combined `trd_u` | `8.602761661688124e-12` | `5.409032045634095e-10` | `1.5904%` | DEBT |
| literal combined `trd_u` | `6.731208045740773e-14` | `5.409032045634095e-10` | `0.012444%` | DEBT |

The literal selector moves `8.66995158306303e-12`, or `1.0078102734932355`
times the historical residual.  Its honest labels are
**CONFIRMED_CAUSAL_OWNER_OF_ENE_COMPONENT** and
**CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER** for the combined `trd` frame.  The first
remaining boundary is the post-ENE bottom-stress addition, whose coefficient
and depth operands were not dumped this round and remain **UNMEASURED**.

## kt sweep after the ENE selection

kt=1 remains exact/uninformative at rest.  The same CPU/fp64 gate then walks to
kt=10:

| kt | T max error | S max error | u max error | v max error | SSH max error |
|---:|---:|---:|---:|---:|---:|
| 2 | `5.162449518e-2` | `4.035174214e-3` | `2.525607809e-2` | `2.525679455e-2` | `1.535304069e-7` |
| 10 | `5.563805236e-1` | `9.806306920e-3` | `4.217947804e-2` | `9.093126494e-2` | `6.569664998e-4` |

The lane-1 log-log instrument classifies T and SSH as
`POLYNOMIAL_FIT_PREFERRED`, with tail exponents `0.94163` and `1.04704` over
kt=7-10.  U is `BOUNDED_OR_DECAYING_NO_AMPLIFYING_MODE` over the same tail;
this is characterization, not certification.  The whole-step TKE/stage-3 debt
still dominates kt=2 u/v, so the ENE correction does not clear the trajectory.

## Artifacts and stopping boundary

| artifact | SHA256 |
|---|---|
| final kt=10 gate | `85ac8f6936f568ddb7f577113acb8f1892f98089b94f4f7ed49e2c6e239002cb` |
| pre-selection causal gate | `dc65258b474a52cd09c675c5236e5ec5069cb1d54af9f49bba107810378e9a28` |
| ENE coefficient dump | `44cc555660f424bb0ae471a756d2e0b614eb9e7b65aa130e1239241f99fad052` |
| v2 50-substep trace | `efbb703dcc3546fb47fcfc83e592eff46ff933d19570273ceb7af1dec5d76c21` |
| instrumented `nemo.exe` | `9b2066bf700beebc54f81a3ba73cd02f984692d62bbdf8490348e087d5028afd` |
| `MY_SRC/dynspg_ts.F90` | `debf4adfeffb7ba4327734e77a6fe4611724f5eae3111d0525e7c68ee403849a` |

The open fronts are the bottom-drag operands of the first remaining combined
substep frame and the previously measured TKE/stage-3 whole-step debt.  They
are **UNMEASURED** here.  This round stops without a trajectory match claim.
