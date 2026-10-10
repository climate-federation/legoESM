# ORCA2 round 230 — fold-transport acquisition build repair

Date: 2026-10-10. Frozen base: `ebdafe2ef`. Preregistration commit:
`f923638ec`. Instrument repair commit: `07f212df5`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round230/`.

No model statement, card selector, deck physics, carried state, stabiliser,
sea-ice selector, or `unmeasured_features` entry changed. No NEMO trajectory
was run or scored in this round. Round 229's independent and given-NEMO-entry
claims therefore remain separately labelled and UNMEASURED at P2/P3.

## Failed acquisition diagnosis

The operator did run round 229's launcher. It reached the fresh NEMO build and
failed in the inserted call at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:558-558` with
`Error: Inconsistent ranks for operator`. The call had passed the whole-array
spelling `e3t(:,:,:,Kmm)`. NEMO's preprocessor expanded the scalar macro at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/inc/domzgr_substitute.h90:126-126` into a
product between 3-D reference thickness/mask and a time-level ratio. The
backing `e3t_3d` and `r3t` declarations are both rank 3, at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90:170-170` and
`:180-180`, but selecting `Kmm` makes `r3t(:,:,Kmm)` rank 2. Thus the
whole-array macro expansion was invalid Fortran. This is an instrument defect;
it is not an ocean-model result.

The full failed-build log is retained as `round229_failed_acquisition.log`
(SHA-256 `94c40bc58f9f134a2de8166c437c94bd90d10e350c1dc8addd20d166d4b7817f`).

## Repair and frozen predictions

The existing round-229 acquisition was reused rather than replaced. The
patched call now passes `e3t_3d`, `r3t(:,:,Kmm)`, and `tmask` as separate
operands. The write-only module constructs one vertical level at a time in the
same multiplication order used by the active FCT consumer, for example
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/traadv_fct.f90:538-538`.
The failed target is preserved; the launcher now names fresh configuration
`ORCA2_OMIP_L4_R230FOLDTRP` and round-230 evidence/run directories.

R230-P1 through R230-P4 remain UNMEASURED until the operator build/run. The
committed preflight reports
`ORCA2_ROUND230_FOLD_TRANSPORT_PREFLIGHT_READY` and names
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round230/acquisition/orca2_omt4_fold_transport_10step_np2`.
It preserves the additions-only patch check, fresh-target refusal, source and
deck hashes, self-describing record parser, three firing plants, and rank-0/1
terminal-restart byte-identity gate. The old `r229` record magic and checker
schema are intentionally retained as the same requested measurement contract;
only the build-safe operand spelling and fresh target changed.

R230-P5 is CONFIRMED: no transport/tracer attribution was made from the failed
build. OMT-5 remains blocked, the round-217 unit remains private, and the
round-229 P2/P3 alternatives remain open.

## Validation and review

The focused record/writer tests pass 2/2, including a static control that
refuses the whole-array `e3t` macro and requires the per-level expression.
`bash -n`, the committed acquisition preflight, and `git diff --check` pass.
No `packages/` file changed, so the shared GYRE, DINO, tank, rung-0, rung-10,
and OMT-4 trajectory gates cannot execute changed model code.

The receipt citation gate, rigid-shift plant, cumulative citation gate, and
separate read-only review are recorded in the final validation commit.

## OPEN

The operator runs the committed launcher. If it passes, the next round admits
the rank-complete post-consumer record and resumes round 229's frozen P2/P3
test: exact `zFv` with stale tracer operands names the missing tracer-fold
exchange; non-bit `zFv` resumes the source-ordered transport walk. No package
change precedes that discrimination.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
