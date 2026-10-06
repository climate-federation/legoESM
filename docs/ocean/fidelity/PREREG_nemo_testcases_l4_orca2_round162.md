# Preregistration — ORCA2 round 162 final external-mode association

Date: 2026-10-06. Frozen base: `afa4d0bbd`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round162/`.

Every science number is **independent**: hierarchy rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. Decision 52's
given-NEMO-entry bridge is not used. The round-159 U-cyclic/V-fold statement
remains a private arm. No configuration, initial state, forcing, carried-state
policy, stabiliser, sea-ice selector, or `unmeasured_features` entry may change.

## Frozen source order and protocol

NEMO first associates the seven live external-mode fields at each substep in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`. It then
rotates the substep state at `:838-848`, accumulates the associated U/V
transports at `:850-868`, divides the completed means at `:888-896`, converts
the transport means back to velocities at `:924-935`, and performs a second
U/V boundary association at `:937`. Rounds 156-159 measured only the first
association; the second association is therefore the next compiled statement
in the held arm's consumer chain.

1. Search the existing barotropic implementation and tests before adding any
   helper. Reuse the established compact NEMO association routine; do not
   reimplement cyclic or fold arithmetic. Add only a private test hook after
   the primary-mean transport division. Its default is false and no card can
   select it.
2. Using the admitted independent step-1..10 growth record, compare NEMO's
   returned `uu_b`/`vv_b` against four otherwise-identical arms: production,
   final-mean association only, per-substep association only, and the pair.
   Score full arrays bitwise and retain signed-zero differences.
3. Run the canonical independent rung-0 ten-step ladder for production, the
   per-substep association arm, and the pair. Require no exact-row loss and no
   earlier first debt. Apply the existing kt=10 stage-3 salinity-maximum veto.
4. Only if the pair clears the ten-step veto, run the unchanged independent
   month scorer and report whether its first non-finite boundary moves beyond
   step 36. This is a discriminator, not permission to add a stabiliser.
5. Because the private hook touches shared model files, prove the default GYRE
   ten-step residual archive array-identical and its 30-day snapshots
   byte-identical at base and tip. Run the prescribed review, citation gate and
   plant, focused tests, and one ocean-fidelity battery.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R162-P1 | The missing final U/V association is non-vacuous but the default hook is passive. | At least one `uu_b` or `vv_b` bit moves in the final-only arm, while unselected production is array-identical to the frozen base. | The final-only arm moves no bit, or its false default moves any production field. |
| R162-P2 | On top of the source-exact per-substep association, the final association reduces the NEMO `uu_b`/`vv_b` disagreement at kt=1. | The pair has fewer unequal full-domain barotropic-velocity cells than the per-substep-only arm, with no new non-finite value. | The unequal-cell count is unchanged or larger, or the pair is non-finite. |
| R162-P3 | The paired source statements remove the round-161 salinity veto. | The pair's kt=10 stage-3 S maximum is no larger than production's `0.4156673855238111`, with no exact-row loss or earlier first debt. | S maximum is larger, an exact row is lost, the first debt is earlier, or the arm refuses. |
| R162-P4 | If R162-P3 confirms, the pair advances the independent month beyond step 36. | First non-finite step is greater than 36 or all 240 steps complete. | First non-finite remains at or before step 36. If P3 refutes, this row is `UNMEASURED_BY_PROTOCOL`. |
| R162-P5 | The shared default path remains unchanged. | GYRE's 70 rows, all residual arrays, and all 30 daily snapshots are identical to the frozen base. | Any registered row, residual array, or snapshot moves. |

## Terminal rule

This round may retain measurement-only hook and gate code. A production
landing is eligible only if the pair makes the locally cited statements exact,
clears the rung-0 predicate, and completes every rung-7/shared-card/DINO/tank/
GYRE/citation/push gate. Otherwise production stays unchanged and the receipt
names the first failing row. The held halo arm is never folded into a different
statement to hide a red row.

ASKED choices: none. UNASKED choices: empty.
