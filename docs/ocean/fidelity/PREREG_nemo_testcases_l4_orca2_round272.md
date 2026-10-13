# ORCA2 round 272 preregistration — combined EEN statement split

Date: 2026-10-13. Base: `b5374a49a2`. Labels: **independent** and
**given NEMO's entry**, measured and reported separately.

This file is frozen before capturing or classifying the matched kt=1 stage-2
EEN residual. The round is measurement-only unless one cited, source-exact
statement closes the surviving boundary and passes the standing gates.

## Executed compiled program

The ORCA2 record build selects `np_CRV`. It forms the reciprocal live F-point
thickness at `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:737-739`,
the combined planetary-plus-relative vorticity at `:762-770`, the live U/V
transports at `:782-786`, the four triads at `:788-794`, and the final trend
and accumulator addition at `:796-803`.

The deck resolves `ln_dynvor_msk = F` (`ocean.output:761`). Therefore the
`fmask` multiplication at `dynvor.f90:768-771` is a dead arm on this card,
even though `rn_shlat = 2` changes the stored mask in `dommsk.f90:267-277`.
A coastal residual cannot be attributed to that inactive multiplication; it
would instead select the executed circulation/boundary operands for the next
split.

## Frozen predictions and falsifiers

- R272-P1 — passive capture. The existing production-JIT observer reproduces
  round 271 under both labels: matched entry below `2e-10 m/s2`, accumulator
  self-replay below that floor, after-VOR maximum
  `1.2863307536453391e-8 m/s2`, and argmax `[19,33,3]`. Any movement makes the
  new operand capture inadmissible.
- R272-P2 — top-100 coastal discriminator. Classify the 100 largest absolute
  after-VOR residual U faces as coastal when any of the four U/V faces around
  either adjacent EEN vertex is masked at that level. If all 100 are coastal,
  the next split is the executed relative-circulation boundary operands. If
  any are open, the coastal-owner hypothesis is refuted and the full compiled
  order (`ff_f`, circulation, `r1_e1e2f`, live F thickness, transports,
  triads, metric, accumulator) continues. The classification is a result, not
  a landing.
- R272-P3 — source replay. A replay from the exact captured production inputs
  must reproduce legoESM's live vorticity output before any NEMO substitution.
  Failure invalidates every leaf score. A one-bit input plant must make the
  replay check refuse.
- R272-P4 — first leaf. Under both labels the first non-floor leaf in compiled
  order is the same. A single source-exact substitution may be named only if
  it reduces the complete after-VOR maximum to the campaign floor. Otherwise
  the round remains HELD and reports the surviving boundary without a model
  change.
- R272-P5 — scope. No deck, card, carried state, initial state, sea-ice
  selector, or `unmeasured_features` entry changes. No NEMO acquisition is
  needed. Any such movement refutes the round.

## Measurement order

1. Capture the exact executing EEN inputs/output from the matched stage-2
   production-JIT call, and prove the capture reproduces the unobserved round-271
   boundary under both labels.
2. Run the top-100 coastal/open discriminator.
3. Calibrate the candidate-side source replay, then split the combined
   statement in NEMO's compiled order. Do not name a leaf from an uncalibrated
   replay and do not use the dead `ln_dynvor_msk` arm.
4. If no single cited statement closes the row, gate the HELD result, run the
   citation and focused tests, attempt the separate read-only review, write the
   receipt, and stop. A closing statement would require a fresh Decision-96
   trajectory census before landing.

No configuration choice is made.

## Post-measurement disposition (predictions above remain frozen)

R272-P1 used round 271's **private-arm** maximum rather than this round's
landed-tree baseline. It is **REFUTED**: the landed value is
`1.2863307536456779e-8 m/s2` (the argmax is unchanged). R272-P2 measured
78 coastal and 22 open points, refuting the coast-only hypothesis. R272-P3 is
**REFUTED**: the passive-operand offline replay differs from its own live
output by `2.5331686726199837e-9 m/s2`, above the floor. R272-P4 is therefore
not reached. R272-P5 is confirmed.
