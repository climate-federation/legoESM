# ORCA2 round 264 preregistration — stage-ratio placement reconciliation

Date: 2026-10-12. Frozen base: `f73a2e6b1`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. This round reconciles the
round-260 and round-263 stage-2 tracer measurements before making another
owner claim. No deck, card selector, carried state, stabiliser, sea-ice field,
or `unmeasured_features` entry may change.

## Compiled order and controlled comparison

NEMO builds the stage-after ratios with `dom_qco_r3c_RK3` and immediately
associates `r3ua/r3va/r3fa` before storing them in the RK3 carried ratio at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:163-180`.
`dom_qco_r3c_RK3` writes the raw V ratio at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217`; the compiled
scalar V association uses sign `+1` through
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`. Stage 2 later
uses the stored ratio in `zvb` and `zFv` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285`.

The measurement restores round 260's private resolved stage-mask route, which
was present when the historical `1.3842782919937235e-8` PSU result was
measured and absent when round 263 produced `1.8200702777361712` PSU. On one
clean committed tree it runs three static private modes with every other hook,
input, record, dtype, JIT setting and matched program point identical:

1. `off`: no stage-ratio association;
2. `round262`: the retired placement, which associates both the stage-1
   transport geometry and the carried stage-after ratio;
3. `nemo`: association only after the stage-after ratio is built and before it
   is stored, matching the compiled call site.

The stage-1 completed U/V/T/S are associated before comparison with NEMO's
post-`lbc_lnk` frame. Measurements are CPU fp64/libm production JIT. The live
observer must reproduce the corresponding ordinary run array-exactly.

## Frozen predictions and falsifiers

1. **R264-P1 — controlled reproduction.** `off` reproduces round 260's
   matched stage-2 fold-band T/S residuals: approximately
   `6.435500132795369e-9` K and `1.3842782919937235e-8` PSU under both labels.
   Failure invalidates the A/B and triggers a commit/config bisect before any
   placement claim.
2. **R264-P2 — retired placement regression.** `round262` reproduces the
   current-tree regression class: stage-2 fold-band salinity is O(1) PSU,
   approximately `5.037097791516988` PSU rather than O(1e-8). If it instead
   stays near `off`, the placement hypothesis is refuted and the commits from
   the round-260 measurement tree to the round-262 arm are bisected on this
   single number.
3. **R264-P3 — compiled placement.** `nemo` removes the O(1) regression and is
   no worse than `off` at the matched stage-2 T/S boundary. It associates only
   the carried `r3va`; it must not alter stage-1 transport geometry. Remaining
   valued residual is debt, not discharged.
4. **R264-P4 — prefix consequence.** If P3 confirms, the complete private unit
   with `nemo` placement completes kt=2 under both labels without the
   round-262 live-thickness refusal. A refusal or a new earlier valued boundary
   holds the unit and names that boundary.
5. **R264-P5 — controls.** Both-label coverage, clean-tree provenance, exact
   record identity, mode isolation, observer passivity, round-260
   reproduction, retired-placement regression, compiled-placement recovery,
   matched program points, source-order citations and one-bit sensitivity each
   have a planted refusal.

## Stop and landing conditions

This is a measurement/reconciliation round. No package candidate lands unless
the complete private unit reaches the Decision-96 full-gate predicate; the
expected outcome is **HELD** with the first matched stage-2 boundary named.
No NEMO acquisition is planned. Missing admitted operands are
`STOPPED_FOR_RECORD`. No floor, clip, damping, alternative mask, or
configuration choice is permitted.
