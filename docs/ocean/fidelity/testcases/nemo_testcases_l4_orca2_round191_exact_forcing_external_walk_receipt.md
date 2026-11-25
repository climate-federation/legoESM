# ORCA2 round 191 — exact-forcing external-mode walk

Date: 2026-10-09. Base `7b7c8371b`. Status: **HELD**.

Every number is **independent hierarchy rung 0**. The card starts from its
corrected initial state, not NEMO's recorded entry. The shipped rung-10 card,
sea ice, configuration, stabilisers, carried state and `unmeasured_features`
tuple are unchanged.

## Result

Round 190's proposed downstream `ssh_after` boundary is **REFUTED**. With the
completed U/V slow forcing replaced by NEMO's bit-exact values, the first
over-floor row is earlier: substep-1 midpoint V-face depth.

| row | complete-domain unequal | RMS | maximum | argmax |
|---|---:|---:|---:|---:|
| midpoint U depth | 0 / 26,640 | 0 m | 0 m | — |
| midpoint V depth | 30 / 26,640 | `12.091634833626163 m` | `899 m` | `[j=147,i=132]` |

Every preceding field is bit-exact over the complete recorded domain: SSH/U/V
forcing, U/V/SSH entry, all six carried before/before-before histories,
substep-1 midpoint U/V/SSH and midpoint U depth. The six-history arm is
bitwise null: exact-forcing-only and exact-forcing-plus-history have identical
trace digest
`a7dd4ffb1ed174519f59b319eeadfa92badbf4fd04da64a4506cdc5e4473cbe4`.

NEMO's first non-bit statement is therefore the midpoint V-depth expression
that adds frozen `hv_0` to the area-weighted north/local SSH average at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:542-545`. This is the same
30-cell/899-m raw-reference-depth owner measured in round 150, now reproduced
on the current tree downstream of exact slow forcing. The current replay does
not reach NEMO's later continuity update at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-591` because the
source-order gate stops at the first debt.

No physics lands. The raw-reference-depth statement is already a measured
member of the held fold/transport cancelling unit; landing it partially would
repeat the round-129/150 failure and cannot satisfy Decision 96. The production
package tree is identical to the round base, so GYRE, DINO, tanks, rung 0,
rung 7 and the independent month cannot move.

Machine evidence:

- `round191/exact_forcing_external_walk_final.json`, SHA-256
  `5e5eff4f98e184c35f44364306f00aef165f7ca1040dea489af9a1829afd1bfa`;
- two self-describing rank records, exactly-once 148x180 coverage, 65 external
  substeps, SHA-256 `d24710b9…b7e7cf` and `fe8a930a…3cadb1`;
- traced/untraced pure-solver SSH, U/V, external U/V and transport U/V are all
  array-identical.

## Frozen prediction ledger

| prediction | result |
|---|---|
| R191-P1 record and instrument remain admissible | **CONFIRMED**. |
| R191-P2 first exact-forcing debt is `ssh_after` | **REFUTED**: midpoint V depth is earlier. |
| R191-P3 histories remain a null arm | **CONFIRMED**: identical trace digest and rows. |
| R191-P4 continuity owns the first downstream debt | **REFUTED**: the walk stops at frozen `hv_0`. |
| R191-P5 no standalone statement lands | **CONFIRMED**. |
| R191-P6 controls are non-vacuous | **CONFIRMED**. |

## Controls, validation and review

Rank-placement, record-bit, source-order, arm-identity, endpoint-ULP,
exact-arm-order and exact-arm-selector plants each exit 2 with their named
`STATUS PLANT-FIRED` line. The round-178/191 focused tests pass 14/14 before
measurement. The final focused round-178/191 and citation suite passes 31/31
in 4.09 s.

The citation gate passes its default receipt and this receipt with no unmapped
spans; the planted rigid line shift exits 1 with one failed citation.

The required single `tests/ocean/fidelity -n 12` invocation is **INCOMPLETE,
not PASS**. It collected 2,900 tests, recorded 1,970 passes and three registered
pre-existing failures, then ended without a terminal summary or live pytest
process. The three visible failures are the stale GYRE round-129 spread-floor
record, SI3 scalar-math provenance, and round-35 escape-scope ratchet. No
round-191 test failed. The transcript is `round191/fidelity_battery.log`.

The required separate `codex exec --sandbox read-only` review was attempted.
It returned `failed to initialize in-process app-server client: Read-only file
system`; verdict: **independent review unavailable in-sandbox**. The transcript
is `round191/independent_review.log`.

No configuration choice was made. ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Re-test one private atomic unit containing the held exact slow-depth
   association, raw `hu_0/hv_0`, no extra compact V mask, the seven-array
   boundary association and materialised V transport on the corrected entry.
2. Walk from the now-exact midpoint depth through metric V transport and
   continuity in compiled order. Do not land a partial member of this known
   cancelling unit.
3. Apply Decision 96 to both ORCA2 ladders, the independent month's next
   boundary, GYRE, DINO and tanks only if the complete unit is a net
   improvement. Otherwise keep it private and name the next partner.
