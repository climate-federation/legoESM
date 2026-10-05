# Preregistration — ORCA2 round 155 V-transport production landing

Date: 2026-10-05. Frozen base: `361ff8bda`.
Claim labels: hierarchy rung 0 is **independent**; the shipped rung-7 card is
**given NEMO's recorded entry**. These labels remain separate in every table.
No configuration, initial-state, forcing, carried-state, stabiliser, sea-ice,
or `unmeasured_features` choice is authorised.

## Source statement and candidate

The compiled rung-0 oracle builds mid-step face depth from raw `hu_0/hv_0` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:538-545`, stores the
unmasked V metric transport in its own loop at `dynspg_ts.f90:568-570`, then
reads the stored array in the separate continuity loop at
`dynspg_ts.f90:584-591`. At the end of each external substep it associates the
velocity, depth, inverse depth, and sea surface in one `lbc_lnk` call at
`dynspg_ts.f90:761-779`.

Rounds 146-154 proved the four prerequisites independently and together on the
production CPU JIT: the seven-array boundary association, raw reference face
depth, omission of legoESM's extra compact V mask, and materialisation of the
completed V metric transport. With all four active, the recorded substep-2 V
transport, V difference, complete divergence, and sea surface are bit-exact.

This round transcribes that complete source unit onto the existing
`nemo_literal` barotropic path. Raw face depths are taken from the bridge-carried
NEMO mesh when present; a literal card missing either raw operand must refuse.
Generic barotropic paths remain unchanged. No public selector is added.

## Frozen protocol

1. Promote all four proved prerequisites atomically on the NEMO-literal path,
   retaining the private arms only as controls proving that explicitly enabling
   the old candidate no longer moves the production result.
2. Run the round-146 causal gate. Require the production path and the former
   four-arm candidate to be array-identical, and require the admitted substep-2
   transport/difference/divergence/SSH boundary to remain bit-exact.
3. Run the independent rung-0 and given-entry rung-7 kt=1..10 ladders against
   the frozen-base artifacts. Register all 200 rows per ladder, exact-row
   losses, first-debt movement, and kt=10 stage-3 salinity maxima.
4. Run the complete GYRE Decision 43/45/55/59 ladder/year gate, DINO, VORTEX,
   LOCK_EXCHANGE, OVERFLOW, generic-card, citation, focused-test, and push-gate
   coverage. Run the rung-0 independent month only after the landing predicate
   is green.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R155-P1 | Production equals the proved four-arm candidate at the named substep boundary. | `transport_v`, `continuity_dv`, complete divergence, and `after_ssh` remain bit-exact; explicitly enabling every retained arm changes no ordinary or exposed array. | Any named boundary is non-bit or an explicit arm moves a bit. |
| R155-P2 | Neither ORCA2 ladder loses an exact row or advances its first debt. | Both 200-row ladders complete with zero exact-row losses and unchanged-or-later first debt. | Refusal, any exact-row loss, or earlier first debt. |
| R155-P3 | The landing is salinity-safe. | Each kt=10 stage-3 salinity maximum is no greater than base: independent rung 0 `0.4156673855360964`; given-entry rung 7 `0.28803500879531185`. | Either maximum increases; the registered approximately 31 PSU exposure is an immediate veto. |
| R155-P4 | Shared-card gates admit the production statement. | GYRE, DINO, VORTEX, tanks, generic cards, citations, and tests satisfy every registered predicate. | Any unregistered or over-floor regression. |
| R155-P5 | The independent rung-0 month advances beyond its current step-36 failure once the exact chain lands. | Its first non-finite step is later than 36 or the 240-step run completes. | First non-finite remains step 36 or moves earlier. |

## Controls and terminal rule

The operand-registry, one-ULP exact-cell, boundary-association, raw-depth,
unmasked-transport, and materialisation controls must each fire. A synthetic
literal card missing raw `hu_0` or `hv_0` must refuse rather than reconstruct a
plausible substitute. Any failure through R155-P4 leaves the production change
unlanded and the round **HELD** with the exact red row. R155-P5 may be refuted
after a successful landing; it names the next independent-month boundary and
does not by itself revert a bit-exact source statement.

ASKED choices: none. UNASKED choices: empty.
