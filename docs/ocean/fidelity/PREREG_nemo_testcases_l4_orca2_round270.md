# ORCA2 round 270 preregistration — G1 transfer Decision-96 landing

Date: 2026-10-13. Base: `42b598288c69d771e80d1218334a5e98d9263de2`.
This file is frozen before restoring any G1 selector or measuring round 270.

## Authorized unit

Restore the six compiled G1 statements measured in rounds 268-269, without
re-derivation, from the pinned grid-ladder tip
`b0fa6a3f5cc2b957900dacd1a72fcbbd70552d9c`.  Restore only the selectors on
the ORCA2 and OMT cards whose resolved branches execute them.  OMT-0 selects
the two flux-form-only statements; vector-form OMT-1..4, rung 0 and rung 10
leave those two off and select the four shared statements.  Defaults remain
off.  No deck, initial state, carried state, stabiliser, sea-ice selector or
`unmeasured_features` entry changes.

Decision 96 is the landing predicate as ruled by the operator after round 269:
a strict majority of RMS-moving rows moves toward NEMO, no exact row leaves
the bar, and the first over-bar row moves toward NEMO or is unchanged.  The
kt=10 stage-3 SSH maximum remains a reported diagnostic, not a veto.

## Frozen predictions and falsifiers

- R270-P1 — restored source identity.  The package/card diff equals the
  previously measured round-269 candidate, and all direct G1 plants fire.
  Any semantic drift or selector executing on an unselected branch refutes.
- R270-P2 — Decision-96 census.  Rung 0 reproduces 159 RMS rows toward, 25
  away and 16 score-equal, with zero exact-row losses and its first over-bar
  row unchanged.  The kt=10 stage-3 SSH maximum reproduces
  `0.3840334615160934 -> 0.3840334668286476 m`, a registered `5.3125542e-9 m`
  (`1.3833519997e-8` relative) movement, but does not veto.  OMT-1..4 retain
  their round-269 majorities and zero exact-row losses.  A majority failure,
  exact-row loss, or first-row movement away refutes the landing.
- R270-P3 — rung 10 and shared-card isolation.  Rung 10, GYRE, DINO, lock
  exchange, overflow and generic-card rows remain byte-identical or retain
  their frozen passing scores because their resolved cards do not execute the
  selected G1 branches.  Any unexplained movement is red.
- R270-P4 — stage-2 vorticity.  The selected fold-image statement preserves
  round 269's after-VOR improvement from `1.7082229745349855e-6` to
  `1.2863307536456779e-8 m/s2` under both labels; failure to close the
  interior residual is registered debt, not a landing veto.
- R270-P5 — gate correction.  A planted exact-row loss and a planted first-row
  movement away both refuse.  A planted SSH-maximum increase alone remains
  reported and passes, proving the retired veto is absent.
- R270-P6 — EEN leaf, only after landing.  Candidate-side replay first
  reproduces the live completed accumulator.  In compiled order `ff_f`,
  relative circulation and `r1_e1e2f` are at the floor; the first valued
  operand is live `e3f_0vor * (1 + r3f)`.  A prior valued operand refutes and
  becomes OPEN.  No new leaf statement lands in this round without its own
  frozen full-gate census.

## Measurement order

1. Restore the exact measured implementations and selectors; print the
   resolved execution census and run the direct controls.
2. Correct the Decision-96 classifier so the SSH maximum is reported rather
   than required; prove the exact-loss, first-row and SSH-reporting controls.
3. Reuse the admitted round-269 OMT/rung-0 measurements only after verifying
   their hashes and current-tree identity; run rung 10, GYRE, DINO, tanks,
   citation gates, focused tests and the required fidelity battery.
4. Land the restored G1 unit if every required predicate is green.  Then, only
   if time remains, resume the interior EEN operand replay and leave its first
   valued leaf OPEN unless it independently earns the full landing gate.

No NEMO acquisition or configuration decision is required.
