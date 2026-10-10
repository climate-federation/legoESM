# Preregistration — GYRE lane round 247: pinned-main merge and PR preparation

Frozen before merging or running any trajectory on the merged tree. Lane base:
`03d9cd620`; pinned second parent: `origin/main-pin` at `471bee222`; merge base:
`cffa2ab79`. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round247/`.

## Scope and controlled comparison

Decision 98 authorises exactly one integration action: merge the pinned main
reference into the lane, audit the automatic composition, prove every certified
number inert, and write the pull-request summary. No model option, scientific
parameter, card, oracle record, score definition, acceptance threshold, or
carried state may change.

The pre-merge file-set audit finds 232 main-only commits and 142 lane-only
commits after the merge base. Main changes ten files under `packages/ocean` or
`src/legoesm`; the lane/main intersection in those trees is empty. The merge is
therefore expected to have zero textual conflicts. This file-set result does
not establish numerical inertness; the certified gates below do.

Pre-implementation search found the prior pinned-main merge receipt, the
existing GYRE ladder/year gates, the VORTEX/SMT trajectory and daily scorers,
the tank/card comparators, the DINO month gate, and the citation-gate plants.
This round reuses those instruments and adds no numerical implementation or
parallel harness.

## Frozen predictions and falsifiers

1. **R247-P1 — merge composition.** `git merge --no-ff origin/main-pin`
   completes with zero conflicted paths. The resulting tree equals Git's
   automatic merge tree. Every one of the ten main-changed ocean/source files
   is inspected in the merged tree, with the main change and its certified-card
   reachability recorded. Any conflict, dropped lane hunk, or unexplained
   executable composition stops the round.
2. **R247-P2 — citations.** The default citation receipt and every audited
   campaign receipt have zero stale endpoints and zero unmapped citations after
   symbol-based re-anchoring. The shifted-citation plant exits nonzero. A stale
   citation stops the round; no citation is weakened or deleted to pass.
3. **R247-P3 — GYRE short trajectory.** The 954-row `kt=1..10` registry has
   zero moved aggregate values, zero status changes, and the same kt=3
   first-over-bar row. Any movement stops the round.
4. **R247-P4 — GYRE year.** A fresh 360-day member reproduces the round-237
   certified values and snapshot digests exactly. In particular day 30, 240,
   and 360 T3D RMS remain `2.3432419318363155e-06`,
   `6.5816987106668941e-05`, and `5.4077212586815052e-05 K`. Any changed score
   or digest stops the round; values are not re-pinned.
5. **R247-P5 — seamount mini-ladder.** SMT-1, SMT-2, SMT-3, and SMT-4 each
   reproduce their certified 50-row registry with zero movement. Fresh
   100-day runs reproduce day-100 T3D RMS
   `4.3321114781972461e-05`, `8.1037591477894766e-06`,
   `1.7729713625071864e-04`, and `2.5527080520554426e-04 K`, respectively.
   Any movement stops the round.
6. **R247-P6 — controls.** The six flat VORTEX cards, LOCK_EXCHANGE, and
   OVERFLOW reproduce their certified 50-row registries with zero movement;
   the private DINO month gate remains at its certified value
   `2.056821682e-03 K` and below its fixed bar. Any moved registry row or DINO
   value stops the round.
7. **R247-P7 — test surface.** The prescribed push-gate battery has no new
   failures. Any failure introduced by the merge stops the round. Pre-existing
   failures are reported only after a same-test pre-merge comparison proves
   them unchanged.
8. **R247-P8 — disposition.** If P1-P7 pass, the merge and documentation land;
   production numerics remain unchanged. The PR summary records rounds 218-246,
   the four seamount statements, the four-rung ladder and movies, GYRE year
   pins, ORCA2 pointers, near-zero ratchet counts, and SMT-4's non-unique owner
   debt. The operator, not this round, opens the pull request.

## Review and controls

After the merge, measurements, citation re-anchors, receipt, and PR summary are
committed, a separate read-only `codex exec` pass attempts to refute the merge
composition, every numerical claim, and the PR summary. A `DO NOT SHIP` verdict
blocks landing. If the sandbox prevents the reviewer from starting, the receipt
quotes the failure verbatim and labels the review unavailable.

No hidden option, threshold, timestep, cadence, source, stabiliser, or model
state is introduced. **UNASKED list: EMPTY.**
