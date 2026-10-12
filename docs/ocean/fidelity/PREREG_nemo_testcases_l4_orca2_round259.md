# ORCA2 round 259 preregistration — integrate and score the complete fold unit

Date: 2026-10-12. Frozen base: `588a24344`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**, followed by the standing
Decision-96 landing gates only if the complete private unit qualifies. No deck,
card selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled statements and existing seams

NEMO forms the stage V transport from `e1v`, live `e3v(Kmm)`, `vv(Kmm)`,
`zvb`, and the three-dimensional `vmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285`. Its
split-explicit loop associates velocity, face depth, inverse depth, and SSH in
one call at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`, and the
completed RK3 stage associates U, V, T, and S at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.

The round reuses the committed round-217 atomic unit, round-245 raw-V-surface-
mask/T-pivot pair, round-247 resolved-stage-mask seam, round-256 passive stage
trace, and round-258 five-operand replay. Before any production landing, the
resolved stage V mask is integrated only into that indivisible private unit.
No in-executable observer is added.

## Frozen predictions and falsifiers

1. **R259-P1 — provenance and passivity.** Both labels run CPU fp64/libm
   production JIT from one clean committed tree, use the production D114
   resolved-mesh route, and preserve traced/untraced completed-state bit
   identity. A dirty tree, wrong route, missing label, or observer movement
   invalidates every numerical result.
2. **R259-P2 — kt=1 closure.** With the complete unit plus the resolved stage
   mask, the 668 pivot-row stage-mask and `zFv` debts remain zero. At kt=1
   stage 1 the fold-band T and S maxima are at the `2e-10` campaign floor
   under both labels. T or S returning to the round-258 baseline
   (`0.17735225873138316 K`, `3.283688696376231 PSU`) refutes the integration.
3. **R259-P3 — growth table.** The receipt prints every kt=1..7 completed
   stage boundary before any landing claim. No fold-band or interior boundary
   grows by more than 10x from a value above the `2e-10` floor. A greater-than-
   10x boundary names the first remaining owner and makes the round HELD.
4. **R259-P4 — terminal OMT-4 predicate.** The complete unit finishes kt=8,
   kt=9, and kt=10 under both labels; the historical `raw-mesh e3w_int must
   contain only finite values > 0` refusal is absent. Any refusal or non-finite
   field makes the round HELD at its first passive boundary.
5. **R259-P5 — atomicity and Decision 96.** Only if P1-P4 pass, score the
   complete indivisible unit: round-217 parts, raw `ssvmask`/T-pivot pair,
   resolved 3-D stage mask, and both external-mode association sites. It may
   land only if the first-over-bar row moves toward NEMO or is unchanged, a
   majority of RMS-moved rows move toward NEMO, no exact row is lost beyond
   the floor, and SSH maximum is not worse. Bit-moved/RMS-equal rows are
   registered separately.
6. **R259-P6 — full landing gates.** A qualifying package change must pass
   OMT-1..4, independent rung 0, given-entry rung 10, the certified GYRE
   ladder and year, DINO month, DINO/lock-exchange/overflow/generic-card gates,
   tank residual comparisons, citation gates and plant, focused tests, the
   once-only fidelity battery, and separate read-only review. Every moved row
   is registered. A scientific red outside Decision 96 holds the landing.
7. **R259-P7 — controls.** Exact worktree, both-label coverage, production
   route, passivity, mask/transport closure, kt=1 floor, growth selection,
   kt=10 completion, Decision-96 direction, exact-row retention, and one-bit
   sensitivity each have a planted violation that must refuse.

## Stop conditions

No NEMO acquisition is planned. A missing passive state or replay operand is
`STOPPED_FOR_RECORD`; a surviving numerical boundary is `HELD`. No partial
mask, association, floor, clip, damping, or alternative configuration lands.
The shipped ORCA2 card and its SI3 `unmeasured_features` tuple remain unchanged.
