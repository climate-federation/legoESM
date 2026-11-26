# ORCA2 round 146 — external-mode boundary association

Date: 2026-10-04. Base `23d4980b8`; clean measurement tip `8e6101636`.
Rung-0 results are **independent**. No configuration, forcing, initial state,
stabiliser, carried state, sea-ice selector, or `unmeasured_features` entry
changed.

## Verdict

**HELD.** The first non-bit statement inside the seven-array boundary call is
the T-pivot U association: 56 fold-row cells differ only in the sign of zero
(first native index `[147, 90]`, maximum magnitude difference `0.0`). NEMO
updates the four live-depth arrays and calls one `lbc_lnk` over `ua_e`, `va_e`,
`hu_e`, `hv_e`, `hur_e`, `hvr_e`, and `ssha_e` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`; its executed
T-pivot branches are
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:639-683` for U and
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:684-721` for V.

The observer is passive: T, S, u, v, ssh, `uu_b`, and `vv_b` are each
array-identical to an independently compiled ordinary step. Boundary scope is
also exact: 64 U-velocity, 180 V-velocity, 30 V-depth, and 68 V-reciprocal
stored cells move; no interior cell moves. U depth, U reciprocal, and SSH do
not need a compact-storage boundary rewrite.

Six post-call arrays reproduce NEMO bit-for-bit. U velocity does not, so the
preregistered complete-call prerequisite is false. The causal arm is therefore
diagnostic only: it closes substep-2 `continuity_du` from 64 unequal cells
(maximum `205276.59075050754`) to zero, but `continuity_dv` is then first,
with 68 unequal active cells and maximum `155776.5627856178`; `after_ssh`
remains unequal on 68 cells with maximum `0.003203816535399729 m`. Because the
arm has not reproduced the complete compiled call, this does **not** refute the
call. The ladders and shared-card landing gates were not run.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R146-P1 observer passive | **CONFIRMED**: all seven returned state arrays are bit-identical. |
| R146-P2 only boundary images move | **CONFIRMED**: 342 stored cells move, zero outside the registered boundary storage. |
| R146-P3 seven post arrays equal NEMO | **REFUTED**: U has 56 signed-zero differences; the other six are exact. |
| R146-P4 complete call closes the residual | **UNMEASURED**: P3 is a prerequisite; the incomplete diagnostic arm closes U divergence but not V divergence/SSH. |
| R146-P5 no ladder regression | **UNMEASURED**: stopped at P3 as preregistered. |
| R146-P6 shared gates pass | **UNMEASURED**: no production landing is eligible. |
| R146-P7 failed prediction leaves production unchanged | **CONFIRMED**: the arm remains private and default-off; no card/deck/public selector changed. |

## Controls, tests, and review

The passivity-ULP, post-association-ULP, and registry-reorder controls each
exit 2 with `STATUS PLANT-FIRED`. Focused round-129/146, literal-continuity,
and prognostic-barotropic tests passed 22/22. The required 12-worker fidelity
battery collected 2,591 tests and reached 99%, then reproduced the registered
xdist tail stall; it is **incomplete, not PASS**. Its displayed pre-existing
reds included SI3 scalar provenance, worktree/stamp ratchets, and the round-129
spread-record stamp. It also exposed two VORTEX observer-fusion regressions
from the first trace implementation; the corrected round-146-only trace type
passes both isolated controls, 2/2. The citation failures shown by the battery
were repaired by a full SequenceMatcher re-anchor and are re-run below.

Separate read-only Codex review was attempted and returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`.

## OPEN

1. Transcribe the compiled T-pivot U branch, including its signed-zero image,
   into the compact seven-array association and re-run R146-P3.
2. Only after all seven post arrays are exact, re-run the causal arm and decide
   R146-P4. If V divergence remains, continue in source order to the unmasked
   metric-transport statements already isolated by round 129.
3. Score both ten-step ladders only after P3 and P4 pass; the registered
   ~31 PSU salinity exposure remains a landing veto.

## UNVERIFIED

- The complete-call effect on substep-2 V divergence and SSH.
- Both ORCA2 ten-step ladders and the independent rung-0 month under a complete
  U+V association.

## Choices

ASKED: Decisions 52, 80, 83, 84, and 94 remain unchanged. UNASKED: none.
