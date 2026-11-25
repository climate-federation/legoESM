# ORCA2 round 206 preregistration — atomic OMT-0 fold-unit census

Date: 2026-10-09. Frozen base: `0f069e01c`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round206/`.
Every trajectory number is labelled either **independent OMT-0** or **given
NEMO's entry**. The shipped rung-10 card, sea ice, its six selectors and its
`unmeasured_features` tuple remain unchanged.

## Admitted evidence and controlled candidate

Round 204 admits both 200-row OMT-0 ladders. Their first numerical debt is
kt=1 stage-1 SSH (`0.006243153742634906 m` RMS,
`0.13136125371061705 m` maximum); kt=10 stage-3 SSH is
`0.0288100436468264 / 0.5086605459790218 m` RMS/maximum. Round 205 admits the
65-substep stream and proves the source-ordered fold unit: the seven-array
association (`dynspg_ts.f90:712-741`), raw reference face depths (`:509-520`),
unmasked V metric transport (`:530-536`), and separately materialised V
transport. Given those statements, substituting the 35 complete-recorded
slow-V halo values changes no substep state score; therefore no unavailable
per-step forcing stream is inferred or fabricated in this census.

The candidate differs from the baseline only by those four already-existing
private controls. The round-103 ladder runner is the reused execution path;
the round-184 comparator is the reused Decision-96 classifier. The candidate
must name all four controls in its JSON, and the baseline must name none.
Both use the same OMT-0 card, forcing, entry state, record, CPU fp64/libm
policy, checkpoints, field masks and score definitions.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R206-P1 | The private candidate is one controlled atomic unit. | Its manifest names association, raw face depth, unmasked V transport and materialised V transport; no other hook or config field differs. | Any fifth difference or missing member: **REFUTED**; read no trajectory direction. |
| R206-P2 | Both candidate ladders remain finite through kt=10. | Each emits 40 checkpoints and 200 finite field rows. | First non-finite checkpoint: **REFUTED** and dispositive HOLD; report it and do not run downstream shared gates. |
| R206-P3 | The exact unit is a Decision-96 net improvement on each OMT-0 label. | Among RMS-moved rows, toward is a strict majority; the first over-bar row moves toward or stays equal; no exact row is lost; kt=10 stage-3 SSH maximum does not increase. | Any failed predicate: **REFUTED**, retain the unit private and HOLD. Score-equal bit moves are registered but are not votes. |
| R206-P4 | The unit reduces the headline SSH debt. | kt=1 stage-1 SSH RMS and maximum are below `0.006243153742634906 / 0.13136125371061705 m`, and kt=10 stage-3 SSH maximum is at most `0.5086605459790218 m`. | An equal or larger kt=1 score, or larger kt=10 maximum: **REFUTED**; no landing. |
| R206-P5 | Independent and given-entry censuses agree because their active entries are identical. | Their per-row toward/away/equal classifications and exact-row-loss sets are identical. | Any disagreement is retained and localized before interpreting either label. |
| R206-P6 | If the atomic unit is ineligible, the existing kt=1 stream can name its next partner. | The first post-unit row above the `2e-10` floor is printed in source order and a recorded-operand replay either closes it or names the missing stream. | No closure and no recorded operand: mark **UNMEASURED_WITH_SPEC** and request only that stream; do not guess. |
| R206-P7 | The census gate binds. | Missing-arm, exact-row-loss, false-majority and score-equal-as-vote plants each refuse at their named predicate. | Any plant stays green: the census is invalid. |

## Landing predicate

An ineligible or non-finite atomic candidate is HELD without a model change.
An eligible candidate may land only after the four cited statements are made
the shared production behavior and the OMT-0, hierarchy rung-0, shipped ORCA2,
GYRE, DINO and tank gates all satisfy Decision 96 with every moved row
registered. No half of the unit lands. OMT-1 does not begin in this round.

ASKED choices: Decision 103's OMT-0 unit and Decision 96's census.
UNASKED choices: empty.
