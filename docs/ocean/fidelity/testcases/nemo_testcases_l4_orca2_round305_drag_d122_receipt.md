# ORCA2 round 305 — G5 top drag + G7 bottom drag under Decision 122

Date: 2026-10-17

Status: **STOPPED_FOR_DECISION.** The drag code is imported, default off. No
card selects it. The shipped card does not run NEMO's linear drag law, so the
certified statements cannot run on it without a scheme change, and that change
is the user's to make.

Preregistration `PREREG_nemo_testcases_l4_orca2_round305.md` (frozen before
measurement). Import commit `398e217ae` (measured clean). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round305/`.
codex review: paused (quota). Independent review: operator (Claude).

## Finding: the cards do not run NEMO's drag law

| card | NEMO deck | card today |
|---|---|---|
| rung 0 | linear drag, no ice | legacy explicit drag, 4e-4 m/s over a 100 m layer, no implicit term |
| OMT-1 | drag off | drag off |
| OMT-2..4 | linear drag, no ice | NEMO's linear implicit bundle |
| rung 10 (shipped) | linear drag + ice-ocean drag | quadratic law (GYRE's non-linear drag, inherited) |

Deck citations: `ORCA2_OMIP_L4/EXP00/namelist_cfg:251` (`ln_lin`),
`ORCA2_OMIP_L4/EXP00/namelist_ref:817` (`ln_drgimp`), `:818`
(`ln_drgice_imp`), `ORCA2_OMIP_L4/EXP00/namelist_cfg:85` (`nn_ice = 2`). The
rung-10 record's run log prints the same switches. Both cards already list
`linear_implicit_bottom_drag` as unmeasured; no decision ever chose either law.

## Decision-122 rows (statements imported from gridladder `9aad93036`)

| statement | (a) certifying rung | (b) ORCA2_OMIP_L4 line, routine, why it runs | runs on |
|---|---|---|---|
| combined top+bottom coefficient | G5 round 3/4 (top pair); G7 round 1 resume (with bottom pair) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1431-1432` (`dyn_drg_init`, `:1404`), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:219` (`stp_2D`, `:63`); needs the ice drag on | rung 10 |
| top residual, before the wind | G5 round 3/4 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1487-1488` | rung 10 |
| dyn_zdf top RHS | G5 round 3/4 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:175-178` (`dyn_zdf`, `:67`) | rung 10 |
| bottom residual | G7 round 1 resume §2 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1447-1448`, `:1462-1463` | rung 0, OMT-2..4, rung 10 |
| bottom coefficient | G7 round 1 resume §2 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:276` (`zdf_drg_lin`, `:258`), masked at `:525`, `:546` | rung 0, OMT-2..4, rung 10 |
| dyn_zdf bottom RHS | G7 round 1 resume §3 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:166-169` | rung 0, OMT-2..4, rung 10 |

The top coefficient is SI3's recorded field: with no ice-shelf cavity NEMO sets
it equal to the ice coefficient (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:187`).
On no-ice decks NEMO turns the ice drag off (`:380`), so only the three bottom
rows run there. The import adds one selector for that case (the bottom rows
without the top drag); it refuses any card that is not on the linear bundle.

## (c)+(d) and the census, every arm vs the shipped card

Arms: B = bottom statements; L = linear law only; LB = L + B; LBT = L + B + top
drag (G7's composition). Counts are RMS toward / away / equal. Census registered,
not a veto. First over-bar row: the classifier's direction at that row.

| card | label | arm | toward/away/equal | early kt1-3 | exact lost | first over-bar | kt-10 stage-3 ssh max (m) |
|---|---|---|---:|---:|---:|---|---|
| OMT-2 | independent | B | 105/70/4 | 20/15/4 | 0 | kt1 stage1 T, equal | 0.34887168162722165 -> 0.3488772734711646 |
| OMT-3 | independent | B | 101/74/4 | 21/14/4 | 0 | kt1 stage1 T, equal | 0.363396286656436 -> 0.363401237319785 |
| OMT-4 | independent | B | 82/93/4 | 22/13/4 | 0 | kt1 stage1 T, equal | 0.3941246940211905 -> 0.3941268969551739 |
| rung 0 | independent | L | 194/1/0 | 54/1/0 | 0 | kt1 stage1 T, toward | 0.38835166609370336 -> 0.39429341213102476 |
| rung 0 | independent | LB | 194/1/0 | 54/1/0 | 0 | kt1 stage1 T, toward | 0.38835166609370336 -> 0.3942956150650081 |
| rung 10 | independent | L | 188/7/0 | 48/7/0 | 0 | kt1 entry ssh, equal | 0.2933479046003188 -> 0.27721870774904545 |
| rung 10 | independent | LB | 188/7/0 | 48/7/0 | 0 | kt1 entry ssh, equal | 0.2933479046003188 -> 0.27722250001091014 |
| rung 10 | independent | LBT | 189/6/0 | 49/6/0 | 0 | kt1 entry ssh, equal | 0.2933479046003188 -> 0.2620064432944157 |
| rung 10 | given NEMO's entry | L | 194/1/0 | 54/1/0 | 0 | kt1 stage1 T, toward | 0.2855333068505171 -> 0.2616995907581532 |
| rung 10 | given NEMO's entry | LB | 194/1/0 | 54/1/0 | 0 | kt1 stage1 T, toward | 0.2855333068505171 -> 0.26170334350008734 |
| rung 10 | given NEMO's entry | LBT | 194/1/0 | 54/1/0 | 0 | kt1 stage1 T, toward | 0.2855333068505171 -> 0.2465012368414565 |

OMT-1 runs no drag (deck drag off); no arm applies and its card is unchanged.

Together and apart (one variable per step, same tree):

| card | label | step | toward/away/equal | early | exact lost | first over-bar |
|---|---|---|---:|---:|---:|---|
| rung 0 | independent | L -> LB (bottom statements) | 95/80/4 | 13/22/4 | 0 | equal |
| rung 10 | independent | L -> LB (bottom statements) | 147/28/0 | 15/20/0 | 0 | equal |
| rung 10 | independent | LB -> LBT (top drag) | 191/4/0 | 51/4/0 | 0 | equal |
| rung 10 | given NEMO's entry | L -> LB | 148/27/0 | 18/17/0 | 0 | equal |
| rung 10 | given NEMO's entry | LB -> LBT | 194/1/0 | 54/1/0 | 0 | toward |

The bottom statements and the top drag each work apart. Neither is
expressible on rung 0 or rung 10 without L, because the imported code admits
the top drag only with the linear bundle and the bottom statements are
written for the linear coefficient. The top drag cannot be measured without
the bottom statements: in the imported code the top-drag selector also selects
them, as G7 certified them.

Size of the effect, kt-10 stage-3 RMS error (m/s), u / v:

| card, label | control | L | LB | LBT |
|---|---|---|---|---|
| rung 10, independent | 5.2247e-3 / 5.3267e-3 | 4.7384e-3 / 4.4727e-3 | 4.7382e-3 / 4.4726e-3 | 3.9208e-3 / 3.7178e-3 |
| rung 10, given entry | 5.2256e-3 / 5.3266e-3 | 4.7387e-3 / 4.4728e-3 | 4.7385e-3 / 4.4726e-3 | 3.9206e-3 / 3.7176e-3 |
| rung 0, independent | 4.3211e-3 / 4.2572e-3 | 3.0090e-3 / 2.7072e-3 | 3.0088e-3 / 2.7072e-3 | n/a |

## Predictions

- R305-P1 **CONFIRMED**: B on OMT-2..4 loses no exact row; first over-bar row equal.
- R305-P2 **CONFIRMED**: B's first moved rows are kt-1 stage-3 u and v on all three; no stage-1/2 row moves.
- R305-P3 **CONFIRMED**: L on rung 10 moves 195/200 rows, 188/7 (independent) and 194/1 (given entry) toward.
- R305-P4 **CONFIRMED**: LB -> LBT moves 195 rows, 191/4 and 194/1 toward, no exact row lost.
- R305-P5 **CONFIRMED**: GYRE ten-step residual archive byte-identical to the base
  (sha256 `3b3d865cf5c82d38…`, 210 arrays equal); 30/30 day-30 snapshots
  byte-identical. The base is round 304's measured tip: packages are unchanged
  from it to this round's base. The ladder gate exits 1 at both, as before.

## Gates and tests

- Import: drag hunks of gridladder `9c51e9f00`, `eb3012db7`, `568ea3fde`,
  `0a81f804f`, `cba2e91be`, `9276a78c9`; every drag hunk equals the gridladder
  tip. One state-file conflict (two appended fields) resolved by keeping both.
- New test file (18 tests): 12 gridladder statement tests ported unchanged,
  plus the bottom-only selector's guard and a behaviour test. That test goes red
  when either of its two gate sites is disabled (checked, tree restored).
- Plants: the arm driver exits 2 on a fired plant and 3 on a green one.
- Citation gate on THIS receipt: PASS (20 citations, 0 failures); a planted
  2-line shift of the bottom-residual citation exits 1. Default gate: PASS
  (274 citations, map audit 0 failures). The import shifted 88 map entries
  citing our own model files: re-anchored mechanically (same tool as round
  304); three spans the import restructured were re-pinned by hand, one grew
  7 -> 43 lines (the G7 dyn_zdf branch now sits inside it).
- Tests: citation gate + private-import ratchet + rounds 304/305 import tests
  43 passed; TKE, recipe, freshwater 109 passed; stage face mask, prognostic
  barotropic, ORCA2 parallel receipt gate 12 passed. Round 304's test asserted
  the top drag was absent; it now asserts the default is off.

## Choices

ASKED (round prompt): import G5 top drag and G7 bottom drag; measure together
and apart; no change to after-SSH selection (D132), runoff depth (D126) or the
private fold unit.

UNASKED, offered for revert:
1. The bottom-only selector for no-ice decks (default off).
2. The rung-10 forcing passes the recorded ice coefficient (inert unless selected).

## OPEN

1. DECISION: move rung 0 and rung 10 to NEMO's linear drag law and select the
   imported drag set (rung 10: L+B+T; rung 0: L+B; OMT-2..4: B)? Pick: yes.
2. Unchanged: fold unit + transport mask + top-V atomic promotion (round 303);
   D126 and D132 pending.
