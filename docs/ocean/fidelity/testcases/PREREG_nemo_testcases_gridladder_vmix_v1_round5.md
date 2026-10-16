# PREREG — vertical-mixing lane V1 round 5 (vmix_r5)

Frozen before measuring. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1`
(admitted round 3). Deck: as built (Decision 129 pending; NEMO's own floors).

Question: round 4 found 118 kt-1 wet W cells where the card's published `avm`
disagrees (over the 1e-15 bar, max 0.341) with NEMO's `tke_avn` coefficient
statement replayed from the card's OWN published `en`
(`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90:706-709`).

Measurement: one kt-1 step with a write-only capture of every operand of the
closure's FINAL mixing-length and coefficient calls (en, rn2, e3t incl.
terminal row, surface anchor, lengths, K_M), compared cellwise with (a) the
published state and (b) the round-4 replay operands.

- **P1.** The captured final K_M equals the published `avm` bit-for-bit, and
  the captured final en equals the published `en`. Falsifier: any wet cell
  unequal (then the owner is the publish path, not tke_avn).
- **P2.** At the 118 cells, exactly one operand class of the final length
  call differs over the bar between closure and replay; that operand names
  the owner. Falsifier: all operands equal to AT-BAR at those cells while
  the lengths differ (owner inside the length routine), or K_M differs with
  equal lengths (owner inside the coefficient routine).
- **P3.** Any landed fix is V1-selected only (G1-G5, full ORCA2, library
  defaults unchanged; config-delta test).
