# ORCA2 round 307 — V1+V2 vertical-mixing statements under Decision 122

Date: 2026-10-17

Status: **LANDED.** The six V1 statements are ON in the shipped ORCA2 card
(rung 10), the only ORCA2 card that runs TKE. All seven Decision-122 pairs
pass (0 exact rows lost; first over-bar row unchanged). Rung 0 and OMT-1..4
run constant mixing and state the flags off (Decision 130). GYRE and DINO
unchanged.

Preregistration `PREREG_nemo_testcases_l4_orca2_round307.md` (`229df89cd`,
frozen before measurement). Import `50b6c269f` (code) + driver `0564a7cbb`;
selection `b609fc798`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round307/`.
codex review: paused (quota). Independent review: operator (Claude).

## Which cards run TKE

| card | vertical mixing (resolved) |
|---|---|
| rung 10 (shipped) | TKE + internal-wave mixing; EVD tracer-only |
| rung 0, OMT-0..4 | constant (ln_zdfcst); TKE block carried inertly |

## Decision-122 rows

Code: gridladder-vmix `e1b56d600`. Six files are byte-identical to its tip;
the C-grid model's shear/N2 wiring and the `hm_i` forcing field were applied
as its diff. (b): every cited routine is byte-identical between the V1 build
(`ORCA2V1_OMIP_L4_R1FULLENTRY`) and `ORCA2_OMIP_L4` (zdftke, zdfsh2, eosbn2,
domqco, zdfphy, lbcnfd, zdfevd); `stprk3.f90` differs only in instrumentation.
Rung 10 runs TKE because its deck sets `ln_zdftke` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:266`), which
calls the shear at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:318` and the closure at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:336`.

| statement | (a) V1 receipt | (b) ORCA2_OMIP_L4 line |
|---|---|---|
| N2 per-operation rounding | round 6 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1281-1285` (`rab_3d_t`), `ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1631-1637` (`bn2_t`) |
| shear at the step-entry level | round 6 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179` (`zdf_phy(kstp, Nbb, Nbb, Nrhs)`) |
| TKE solve rounding | round 7 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:466-472` |
| pivot-row shear fold | round 8 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:98-108` |
| TKE diagonal/RHS rounding | round 9 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:420-426` |
| r3u/r3v rounding + stored reciprocal (pair) | rounds 10-11 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-269` |

The r3 pair is one group: the imported pairing guard refuses either half
alone, so it was measured only together.

Not imported as selections: htau native latitude and EXP (round 7). The card
resolves `etau_mode='none'`, so the penetration code they change does not run
on it (read off the code). NEMO's deck runs it (`nn_etau = 1`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:498-500`, htau at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:864`) — a card/NEMO gap, see
OPEN. EVD: already the card's path since round 302.

## (c)+(d) per statement, rung 10, vs control (all V1 flags off)

Counts are RMS toward / away / equal over 200 rows (early = kt 1-3). Census
registered, not a veto.

| arm | label | all | early | exact lost | rows moved | first over-bar | kt-10 stage-3 ssh max (m) |
|---|---|---:|---:|---:|---:|---|---|
| N2 | independent | 84/88/9 | 8/24/9 | 0 | 181 | equal | 0.27722250001091014 -> 0.27722250001084214 |
| N2 | given entry | 115/58/10 | 25/9/9 | 0 | 183 | equal | 0.26170334350008734 -> 0.2617033435000624 |
| SH2E | independent | 67/96/0 | 7/16/0 | 0 | 163 | equal | 0.27722250001091014 -> 0.2772225025603945 |
| SH2E | given entry | 63/100/0 | 7/16/0 | 0 | 163 | equal | 0.26170334350008734 -> 0.2617033460435661 |
| SOLVE | independent | 66/94/14 | 5/16/14 | 0 | 174 | equal | 0.27722250001091014 -> 0.2772225000109271 |
| SOLVE | given entry | 94/75/10 | 12/19/8 | 0 | 179 | equal | 0.26170334350008734 -> 0.26170334350014113 |
| FOLD | independent | 21/142/0 | 3/20/0 | 0 | 163 | equal | 0.27722250001091014 -> 0.2772228116108405 |
| FOLD | given entry | 21/142/0 | 3/20/0 | 0 | 163 | equal | 0.26170334350008734 -> 0.26170365524748873 |
| RHS | independent | 49/105/7 | 4/12/5 | 0 | 161 | equal | 0.27722250001091014 -> 0.27722250001085896 |
| RHS | given entry | 90/64/4 | 4/11/4 | 0 | 158 | equal | 0.26170334350008734 -> 0.26170334350007196 |
| R3 pair | independent | 30/35/10 | 0/0/0 | 0 | 75 | equal | 0.27722250001091014 -> 0.27722250001091675 |
| R3 pair | given entry | 45/49/8 | 0/1/0 | 0 | 102 | equal | 0.26170334350008734 -> 0.26170334350008695 |
| ALL | independent | 36/138/8 | 6/28/8 | 0 | 182 | equal | 0.27722250001091014 -> 0.27722281413075195 |
| ALL | given entry | 43/132/8 | 14/21/8 | 0 | 183 | equal | 0.26170334350008734 -> 0.2617036577612212 |

First over-bar row: kt 1 entry ssh (independent), kt 1 stage-1 T (given
entry) — unchanged in every arm.

Size: kt-10 stage-3 RMS error, control -> ALL (independent): T 9.224760e-3 ->
9.224696e-3 K, S 2.690776e-3 -> 2.690910e-3, u 4.738180e-3 -> 4.738213e-3,
v 4.472555e-3 -> 4.472544e-3 m/s, ssh 2.913132e-2 -> 2.913129e-2 m. The
statements move rung 10 in the 5th-6th digit; its error is owned elsewhere.
The away-leaning census comes mostly from the fold statement (21/142).

## Seven pairs on the selection tree (`b609fc798`)

Control = the shipped card with every V1 flag off; D122 = the shipped card.

| card | label | all | exact lost | first over-bar |
|---|---|---:|---:|---|
| rung 0 | independent | 0/0/0 | 0 | equal (no row moves) |
| OMT-1 | independent | 0/0/0 | 0 | equal |
| OMT-2 | independent | 0/0/0 | 0 | equal |
| OMT-3 | independent | 0/0/0 | 0 | equal |
| OMT-4 | independent | 0/0/0 | 0 | equal |
| rung 10 | independent | 36/138/8 | 0 | equal |
| rung 10 | given NEMO's entry | 43/132/8 | 0 | equal |

Decision 122 (c)+(d): **PASS** (`final/decide.json`, `PASS_R278_D122_DECISION`).

Instrument checks (ladder dicts compared whole):
- rung-10 controls on the import tree equal round 306's ladders (both labels);
- rung-10 controls and shipped arms on the selection tree equal the import
  tree's control and ALL arms (both labels);
- rung 0, OMT-1..4 shipped ladders on the import and selection trees equal
  round 306's (200/160 rows each).

Resolved configs: the import adds only new default-off fields on every card;
the selection changes 7 rows on rung 10 (the seven flags) and none on rung 0
or OMT-0..4 (`config_import_vs_selected.txt`).

## GYRE and DINO

- GYRE: ten-step residual archive identical base (`64b28e144`) vs import
  (210 arrays, sha256 `3b3d865cf5c82d38…` both); 30/30 day-30 snapshots
  byte-identical (only the run manifest differs). The ladder gate exits 1 at
  both, as before.
- DINO: the flag-off DINO step jaxpr is identical base vs import (123,235
  lines, round-11 vmix probe). Its card configs gain only default-off fields.

## Predictions

- P1 CONFIRMED: 0 exact rows lost, every arm, both labels.
- P2 CONFIRMED: first over-bar row unchanged, every arm.
- P3 CONFIRMED: every arm moves rows (75 to 183).
- P4 CONFIRMED: FOLD moves 163 rows.
- P5 CONFIRMED: non-TKE ladders equal round 306; their configs unchanged.
- P6 CONFIRMED. P7 CONFIRMED.

## Gates and tests

- Ported vmix unit tests: 155 passed.
- New `test_orca2_round307_vmix_selection.py` (8 tests): reverting the rung-10
  selection fails 2; reverting the rung-0 off-statement fails 6 (tree
  restored, `git status` clean).
- Two independent-label arms aborted once (`Fatal Python error:
  PyThreadState_Get`, exit 134, interpreter crash under 21 parallel runs);
  rerun clean, kept in `arms/aborted/`.
- Citation gate, default gate, ratchet, battery: see the commit that adds
  this receipt's gate numbers.

## Choices

ASKED (round prompt): import V1+V2 statements; select those passing D122 on
the TKE card; leave htau/EXP off where the code does not run; no change to
after-SSH (D132), runoff depth (D126), drag (D133) or the fold unit.

UNASKED, offered for revert: rung 0's builder states the seven flags off
(keeps rung 0 and OMT-0..4 configs unchanged instead of inheriting them).

## OPEN

1. FINDING / DECISION: rung 10 does not run five TKE forms NEMO's deck
   uses and V1 certified: the `nn_etau = 1` penetration (card: off) with its
   mask and native htau/EXP; the dummy-bottom mixing-length step and the
   literal raw length; the literal Langmuir source; and `nn_mxlice = 2`
   (card: 0; needs SI3's `hm_i`). Pick: import and measure the first three
   under D122 next round; `nn_mxlice = 2` waits until the card computes
   `hm_i` itself (same rule as the top drag).
2. Unchanged: D126, D132; top drag off; fold unit + transport mask + top-V.
