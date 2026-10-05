# ORCA2 round 69 preregistration — independent-month surface inputs

Date frozen: 2026-09-28

Base: `a7af362112aaf6aa72d1956768ff97f417057c47`

Claim label: **independent**.  This round acquires the exact NEMO ocean surface
operands needed to advance legoESM from its own ORCA2 state through 240 steps.
It does not run or score the legoESM month.

Sea ice remains out of scope.  The card's six-entry `unmeasured_features`
tuple, all selectors, and the admitted one-category external-forcing deck stay
frozen.

## Record boundary and source-first call site

Round 68 admitted the uninstrumented pinned-ORCA1ICE from-rest terminal month.
It contains no per-step ocean surface operand records.  The admitted
instrumented record supplies both MPI slabs only for steps 1 through 10: 20 of
the 480 rank-step frames needed through step 240.  The missing 460 frames are a
record gap, not permission to repeat step 10 or reconstruct another forcing.

NEMO completes its coupled sea boundary condition before the ocean consumes
the resulting fields at
`src/OCE/stprk3.F90:135-139`.  The acquisition adds one WRITE-only call
immediately after that compiled statement.  A separate module writes only the
ten operands already consumed by the production ladder: `utau`, `vtau`,
`taum`, `qsr`, `qns`, `emp`, `sfx`, `rnf`, `fr_i`, and `rnf_tsc`.

## Frozen predictions and falsifiers

1. **Complete self-describing inventory.** One 240-step two-rank run predicts
   exactly 480 surface files, each with magic/version/step/level/rank/field
   count and ten per-field `(name, rank, n1, n2, n3, payload)` records.  Any
   missing, duplicate, reordered, truncated, trailing, non-finite, or wrongly
   shaped field refuses.  No predicted whole-file byte count is used.
2. **Ten-step calibration.** Decoding the new format predicts bit identity for
   all ten fields, both ranks, and every step 1 through 10 against the admitted
   round-5 surface record.  One changed payload bit refuses.
3. **Write-only passivity.** Every variable payload in the four terminal ocean
   and ice restart shards predicts bit identity against round 68's admitted
   uninstrumented 240-step record.  Any missing variable, dtype/shape change,
   signed-zero change, or unequal bit refuses.
4. **Surface-only scope.** The run predicts no other `oracle_*.bin` files.  The
   module and two-line call-site patch add no assignment to a NEMO model array;
   source preflight refuses any removed line or assignment to a listed operand.
5. **Disposition.** The round predicts `STOPPED_FOR_RECORD`: PMIx remains
   unavailable in the sandbox, so the operator must build and run the committed
   acquisition.  The next round admits the frames and runs the independent
   legoESM month ranking.

Failed predictions remain **REFUTED** in the receipt.  Surface calibration and
terminal restart passivity are separate predicates; neither can waive the
other.

## Required controls and validation

- Commit every patch/module/schema artifact and reference it repo-relatively.
- Parse the self-describing header through physical EOF; do not predict file
  byte counts or positional payload tuples.
- Wrong field name, truncated payload, one-ULP calibration mutation, missing
  frame, extra oracle stream, and terminal restart mutation must each refuse.
- The launcher is fail-closed, creates the evidence directory before `df`,
  names a fresh target, never uses `/usr/bin/time`, and never invokes MPI in
  preflight mode.
- Shell syntax, patch replay, focused tests, 170-test card battery, citation
  audit with a firing plant, default citation audit, and separate read-only
  Codex review.  No `packages/` change is authorized.

ASKED: obtain the exact matched surface forcing required for Decision 52's
independent ORCA2 month-scale ranking.

UNASKED: configuration, selector, threshold, forcing reconstruction,
stabiliser, carried state, model arithmetic, sea-ice implementation, and the
held QCO/RK change.
