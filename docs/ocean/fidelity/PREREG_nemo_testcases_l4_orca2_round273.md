# ORCA2 round 273 preregistration — source-complete relative-EEN replay

Date: 2026-10-13. Base: `26ec7877f`. Labels: **independent** and
**given NEMO's entry**, measured and reported separately.

This file is frozen before changing or rerunning the round-272 offline replay.
The round is measurement-only unless one cited, source-exact statement closes
the matched kt=1 stage-2 after-VOR boundary and passes the standing landing
gates.

## Executed compiled program

The admitted ORCA2 record executes `np_CRV`. The compiled build forms the live
F-thickness reciprocal at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:737-739`, the combined
planetary-plus-relative circulation at `:762-770`, the live face transports at
`:782-786`, the four triads at `:788-794`, and the trend plus accumulator at
`:796-803`. The deck has `ln_dynvor_msk = F`, so the `fmask` branch is dead.

Round 272's replay materialised the relative vorticity in one JIT and invoked
the EEN operator in a second JIT. The executing model has one fused source
graph. That extra replay boundary is therefore the first instrument difference
to remove before any leaf is scored.

## Frozen predictions and falsifiers

- R273-P1 — passive boundary. The existing round-266 trace remains passive
  and reproduces the landed after-VOR maximum
  `1.2863307536456779e-8 m/s2` at `[19,33,3]` under both labels. Any completed
  state movement or baseline movement above one ULP refuses the instrument.
- R273-P2 — candidate calibration. A single JIT wrapper that evaluates the
  executing relative circulation, fold-image transport row, and EEN operator
  without an intermediate host/device materialisation reproduces the captured
  candidate vorticity term and accumulator to the `2e-10 m/s2` floor under
  both labels. Failure means no leaf may be named; the round writes a
  fail-closed acquisition for NEMO's per-leaf stage-2 EEN stream.
- R273-P3 — relative branch. Once P2 passes, `ff_f` remains the identical
  production operand and the first non-floor difference lies in the executed
  relative-circulation half of `np_CRV`, not the G1-certified planetary half.
  The split follows NEMO's written order: `e2v*pv`, `e1u*pu`, their two
  differences, `r1_e1e2f`, live `e3f_0vor*(1+r3f)`, transports, triads, final
  metric, accumulator. A leaf is named only when its source-exact substitution
  closes the complete after-VOR row to the floor under both labels.
- R273-P4 — controls. A one-bit candidate-output plant and a relative-branch
  bypass plant each make the gate refuse. A plant that perturbs an inactive or
  zero operand is invalid and is not accepted as evidence.
- R273-P5 — scope. No deck, card, initial/carried state, sea-ice selector, or
  `unmeasured_features` entry changes. No package statement lands unless P2
  calibrates, one cited leaf closes the row, and the full Decision-96 gates are
  green.

## Measurement order

1. Reuse the admitted round-266 passive stage-2 handoff and round-272 oracle
   row; validate passivity and the landed baseline first.
2. Calibrate the single-graph candidate replay before reading any leaf result.
3. If calibrated, replay the compiled `np_CRV` expression in source order and
   test one operand boundary at a time. If not calibrated, stop the split and
   write the per-leaf NEMO acquisition using a previously admitted
   self-describing ORCA2 recorder pattern.
4. Run the plants, citation gate, focused tests, one fidelity battery, and a
   separate read-only review. Record failed predictions without rewriting
   them.

No configuration choice is made.
