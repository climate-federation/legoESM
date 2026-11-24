# PREREG — ORCA2 round 307: V1+V2 vertical-mixing statements under Decision 122

Frozen before any measurement. Round prompt: claude_orca2/orca2_r307.prompt.md
(addenda 83, 97, 101, 103; D96, D122, D130).

## Which cards run TKE (read from round 305's resolved-config dump, re-dumped this round)

- rung 10 (shipped ORCA2 card): vertical mixing = TKE (+ IWM; EVD tracer-only). The ONLY ORCA2 card that runs TKE.
- rung 0, OMT-1..4: constant mixing (ln_zdfcst). They carry the shipped card's TKE block inertly.
  Under D130 the statements stay OFF there; the selection must leave their resolved configs unchanged.

## Statements (gridladder-vmix, certifying receipts V1 rounds 6-11, V2 round 12) and arms on rung 10

| arm | selector(s) | V1 receipt |
|---|---|---|
| N2 | nemo_bn2_source_rounded | round 6 |
| SH2E | nemo_sh2_step_entry_ssh | round 6 |
| SOLVE | nemo_tke_solve_source_rounded | round 7 |
| FOLD | nemo_sh2_fold_top_v | round 8 |
| RHS | nemo_tke_matrix_rhs_source_rounded | round 9 |
| R3 | nemo_sh2_r3_source_rounded + nemo_sh2_r3_stored_reciprocal (pair; either alone raises) | rounds 10-11 |
| ALL | all of the above | — |

htau native latitude and EXP (round 7) are not given an arm: rung 10 resolves `etau_mode='none'`, so the
penetration code they change does not run on the card (read off the code; NEMO's deck sets nn_etau=1 — a
card/NEMO gap reported, not changed). EVD: already the card's path (round 302).

Each arm: rung 10 independent and given-NEMO-entry labels, kt 1..10, vs a control = the import tree with all
flags off (= the shipped card). Non-TKE cards (rung 0, OMT-1..4): the import tree's shipped card vs round 306's
rerun ladders (row-for-row).

## Predictions

- P1: every arm, both labels: 0 exact rows lost (D122 c).
- P2: every arm, both labels: first over-bar row unchanged or toward (D122 d).
- P3: N2, SH2E, SOLVE, RHS, R3 and ALL each move at least one row on both labels (the code runs on rung 10).
- P4: FOLD moves at least one row (rung 10 has the T-pivot fold).
- P5: rung 0, OMT-1..4 ladders on the import tree equal round 306's row for row; their resolved configs
  unchanged after selection.
- P6: GYRE ten-step residual archive and day-30 snapshots byte-identical base vs tip.
- P7: DINO: resolved card configs unchanged and the flag-off DINO step jaxpr identical base vs tip.

Falsifiers: any exact row lost or first over-bar row away on any arm/label → that statement stays OFF with its
numbers. A group statement (R3) is measured only together (the pairing guard refuses either half alone).
