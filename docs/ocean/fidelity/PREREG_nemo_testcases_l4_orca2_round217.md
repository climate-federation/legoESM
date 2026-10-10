# Preregistration — ORCA2 round 217 complete vector transport unit

Date: 2026-10-09. Frozen base: `ac1e05f26`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round217/`.
Every OMT-1 result is labelled either **independent OMT-1** or **given NEMO's
entry OMT-1**. Rung-0 results are **independent** and rung-10 results are
**given NEMO's entry**. No sea-ice selector or `unmeasured_features` entry may
change.

## Frozen source unit

Rounds 215-216 established an indivisible source unit on NEMO's executed
vector-form split-explicit branch:

1. NEMO copies `Ve_rhs`, then subtracts the Coriolis trend through raw
   `ssvmask` at compiled `ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/
   dynspg_ts.f90:289-328`.
2. The vector update uses the same raw `ssvmask` at `dynspg_ts.f90:669-682`.
3. The complete seven-array boundary association follows at
   `dynspg_ts.f90:747-756`, with the active T-pivot V mapping at compiled
   `lbcnfd.f90:684-718`.
4. The next substep forms the unmasked, materialised
   `zhV = e1v * va_e * zhvp2_e` and consumes it in continuity at
   `dynspg_ts.f90:533-560`.

The rank-complete replay proves the first pair closes pre-LBC `va_e` on all
cells only when both operands move together. The association then closes its
post-LBC V target on all cells. The unmasked, materialised transport closes
transport, continuity and after-SSH on all cells. NEMO exposes no selector
between these statements. Round 217 therefore scores them as one production
candidate; no constituent may land alone.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R217-P1 | Production selects the complete source unit exactly on vector-form RK3 cards carrying NEMO's raw mesh, while flux-form cards retain their already-landed literal unit and unrelated cards are inert. | Direct unit tests show the resolved vector predicate, raw mask shape, association, unmasked transport and materialisation together; each partial-unit plant refuses. | Any silent fallback, half-carried mask/depth pair, unrelated-card movement, or partial arm sufficient: restore production and **HELD**. |
| R217-P2 | The complete unit is Decision-96 eligible on both OMT-1 labels. | In each 160-row ladder, a strict majority of RMS-moved rows goes toward NEMO, first-over-bar is toward/unchanged, no exact row is lost, and kt=8 stage-3 SSH maximum is not worse. | Any predicate fails: restore production and **HELD** at the exact row. |
| R217-P3 | Adding the downstream compensating statements removes round 215's independent-rung-0 final-SSH veto. | In the 200-row ladder, a strict majority of RMS-moved rows goes toward NEMO, first-over-bar is toward/unchanged, no exact row is lost, and kt=10 stage-3 SSH maximum is no worse than `0.42832517646246693 m`. | Any predicate fails: restore production and **HELD**; name the next downstream statement from the admitted substep stream. |
| R217-P4 | The complete unit is admissible on the shipped rung-10 card and all shared cards. | Rung 10 meets the same ladder predicate; GYRE ladder/year, DINO month, DINO/lock-exchange/overflow and generic-card gates meet their standing bars; every moved row is registered. | Any non-Decision-96 red: restore production and **HELD** at the exact gate row. |
| R217-P5 | The implementation and receipt remain mechanically auditable. | Focused tests pass; the cumulative and round receipt citation gates have zero unmapped citations; rigid-shift and decision plants fire; independent read-only review returns no blocking finding or is recorded unavailable. | Any plant stays green, citation is unmapped, or blocking review finding survives: no landing. |

## Landing predicate

The full slow-V/raw-mask pair plus raw reference face depths, complete
seven-array association, and unmasked/materialised V transport lands only if
R217-P1 through R217-P5 pass. A faithful statement that exposes another
compensating error remains private and the next source boundary is named.
OMT-2 waits for this disposition.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
