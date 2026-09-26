# Preregistration: first-substep continuity operand, round 8

Date: 2026-08-29. Frozen before measurement.

## Question

Round 7 refuted the SSH-source candidate without ambiguity:
`spg_dump_ssh_frc.bin` is identically zero on all 9,920 wet T cells, and its
held arm changes no row-1.3 or row-1.4 score.  With U/V slow forcing and the
BEFORE seed held exactly, the first divergent output remains first-substep SSH
at `E=6.2048763005e-7`.  This round localizes the corresponding continuity
operand using only existing outputs.

At active NEMO `dynspg_ts.F90:651-725`, DINO's `key_qco` variable-volume arm
first constructs eta-dependent `zhup2_e/zhvp2_e` (`:651-688`), then assembles
`zhU/zhV` from those face depths, the half-step velocity, and face metric
(`:698-704`) before applying their divergence with zero `ssh_frc` (`:718-725`).
At substep 1 the
AB3/AM4 extrapolation coefficients reduce to the current loop seed.  Existing
`cor2d_dump_{ua,va}_e_in_substep1.bin` independently record those same
half-step velocity operands at `dynspg_ts.F90:784-805` before the velocity
update.

## Registered calculations and bars

1. Compare the exact held U/V seed to the existing NEMO `ua_e/va_e` substep-1
   dumps on 9,758/9,868 wet faces.  POINTWISE `1e-15`; all campaign aggregate
   axes apply.
2. Recover the continuity divergence from the already captured seed/output:
   `div = (ssh_seed - ssh_substep1)/rDt_e - ssh_frc` in both models.  This is an
   algebraic inversion of the cited executed equation, not a second operator
   implementation.  Compare on 9,920 wet T cells at ACCUMULATING `1e-12`.
3. Require the runtime-captured `rDt_e`, exact forcing/seed receipts, finite
   populations, clean CPU/fp64 execution, and planted controls already used by
   rounds 6--7.

**CONFIRM continuity-flux ownership:** held seed velocities are `AT BAR` while
the implied divergence is `DEBT`, and its forward application reproduces the
first-substep SSH difference to roundoff.  **REFUTE:** half-step velocity is
already `DEBT`, or the implied divergence fails to reproduce the SSH residual.
If confirmed, ownership is bounded to the `zhU/zhV -> zhdiv` composition
(`:698-724`); separating face-depth/metric multiplication from differencing
requires a direct `zhU/zhV` operand dump and is a designed SLOT-protocol follow-
up, not a guessed fix.

Rows 1.4 and 2--6 remain ordered-blocked behind the literal row-1.2 and held
row-1.3 stops.  Their existing probes may be inventoried or run as targeting-
only evidence, but cannot be promoted.  No new writer is used in this round.

## Provenance and admission retraction after adversarial review

The first nominally accepted round-8 receipt is **WITHDRAWN** because its clean
harness imported production modules from another editable-install worktree.
The unchanged calculations and bars must be rerun with the shared under-
checkout import guard and resolved-path stamps.

Review also found that the first instrument compared the reconstructed held
seed rather than the runtime-captured seed and treated the inverted-divergence
forward replay as a control even though that replay is an algebraic identity.
The rerun must compare the runtime-captured seed to `ua_e/va_e`, require all
three runtime seed fields `AT BAR`, and label the replay
`ALGEBRAIC_CLOSURE_ONLY_NOT_CONTROL`.  Confirmation depends only on the
independent runtime seed/half-step receipts plus the divergence gate.  No prior
round-8 score or ownership claim is admissible.
