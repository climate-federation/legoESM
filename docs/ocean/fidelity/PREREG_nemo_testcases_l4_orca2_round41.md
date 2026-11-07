# NEMO testcase Lane 4 — ORCA2 card round 41 preregistration

Date: 2026-09-27

Parent: `512c25ee0e467a0e59d6f7e534212cdf51fd69b6`

Status: **PREREGISTERED BEFORE ROUND-41 SCIENTIFIC SCORING.**

All scientific numbers will be **given NEMO's entry**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain frozen.

Round 40 acquired one passive per-rank stream containing the stage-1 momentum
accumulator after HPG, LDF, VOR, KEG, and ZAD.  Its admission has already
proved restart identity, inherited-stream identity, exact post-ZAD closure,
and firing corruption plants.  Round 41 scores those five boundaries against
the production operator components in compiled order.  No configuration,
state, physics, or threshold changes.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R41-P1 | The admitted Round-40 stream remains observationally passive and complete. | Both rank records retain their admitted digests; inherited ranked streams, four restarts, and post-ZAD U/V remain bit-exact. | Any moved byte, missing field, or failed admission control; stop for record drift. |
| R41-P2 | Round 39 reproduces exactly at this parent. | The product, first partial sum, completed sum, and final depth-mean counts and maxima equal the Round-39 receipt. | Any movement; stop for instrument drift. |
| R41-P3 | HPG is the first non-bit stage-1 momentum operator on the 64 disputed rank-1 U rows. | Post-HPG is unequal; no earlier registered boundary exists. | Post-HPG is exact, or the record cannot be aligned with the candidate's HPG boundary. |
| R41-P4 | LDF, VOR, KEG, and ZAD do not add a new residual on those 64 rows. | The bitwise residual after each later boundary equals the post-HPG residual cell by cell. | Any later boundary changes the residual; that first boundary becomes the named next owner. |
| R41-P5 | No production statement lands unless one isolated NEMO-recorded operator substitution closes all 64 rows and the full ORCA2/GYRE gates pass. | One operator-only arm reaches 0/64 and all required trajectory gates pass. | Any nonzero residual or multi-operator requirement; hold and register the first boundary. |

The scoring gate must refuse missing or non-finite fields, compare only active
owned cells, prove its first-boundary selector with a one-ULP plant, and prove
the 64-row residual-equality check can fail.

## Choices

ASKED: score the acquired five compiled momentum boundaries in order.

UNASKED: none.
