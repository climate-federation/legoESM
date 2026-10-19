# ORCA2 round 318 — the north-fold set under Decision 122

Date: 2026-10-19 (Claude round; codex review: operator).

Status: **HELD** — gate fixed after the Codex HOLD; lands after the user rules
on the "later first row" reading (operator review + push gate).

## Headline

- The ORCA2 cards' dynamics-only sea-level error was the missing north fold.
  With the whole fold set, the kt-10 maximum sea-level error falls from
  0.35-0.75 m to 3e-5 .. 7e-3 m on rung 0 and OMT-1..4 (table below), and
  kt-1 stage-1 T and S become bit-exact on OMT-1..4.
- The fold set passes only as a whole.  Each member alone refuses, fails
  D122 (d), or leans away (FCT fold images alone push the kt-10 sea-level
  error 0.39 -> 6.24 m; top-V alone -> 7.25 m).  Together, on all seven
  pairs: 0 exact rows lost, first over-bar row toward or equal, and every
  moved row toward NEMO (0 away).
- Landed: G9's seven fold/transport statements, G6's raw pivot V mask,
  vector fold unit and top-V exchange (held since round 308 / Decision 125),
  and G8's fold rows (Decision 137 re-measure).
- Stays off: G9's stage-3 Krhs order (lane A rounds 49-51: Krhs order,
  literal FCT, stored stage-3 divisor, one switch).  It refuses on every
  ORCA2 card: rung 0/OMT keep the forced stage-3 heat/salt forms off (no
  forcing module), and rung 10 runs the diffusive bottom boundary layer,
  which G9 never had.  Running it needs new code, not an import.

## Answer to the user's question (kt-10 stage-3 max |card - NEMO| sea level, m)

| card | before (shipped) | after (shipped) |
|---|---:|---:|
| rung 0 | 0.394173 | 0.00701981 |
| OMT-1 (kt 8, its last step) | 0.74699 | 2.90886e-05 |
| OMT-2 | 0.348812 | 4.30139e-05 |
| OMT-3 | 0.363341 | 7.46458e-13 |
| OMT-4 | 0.394004 | 0.000189112 |
| rung 10 independent | 0.277208 | 0.183209 |
| rung 10 given entry | 0.26169 | 0.168432 |

kt-10 stage-3 RMS (same rows): OMT-3 T 2.75e-4 -> 8.7e-15 K, u 3.0e-3 ->
2.8e-15 m/s, ssh 2.4e-2 -> 4.7e-15 m; OMT-1/2 T ~3e-4 -> ~9e-9; OMT-4 T
1.24e-3 -> 4.95e-6; rung 0 T 5.49e-3 -> 5.36e-3 (its remaining T error is
not the fold); rung 10 ssh 2.9e-2 -> 1.36e-2.  Exact rows: OMT-1/2 4 -> 48,
OMT-3 4 -> 36, OMT-4 4 -> 12, rung 0 5 -> 11, rung 10 4/5 unchanged.

## Provenance

- Base `3e9bb7dea`.  Import `7ccd8a1a1` = package hunks of gridladder
  `2b75714433..7a31b67ef` (rung G9 rounds 1-8 = lane A rounds 45-52,
  `nemo_testcases_gridladder_g9_round8_receipt`: G9 CERTIFIED kt 1..10 both
  labels), `git apply --3way`: one conflict in ocean_model_latlon_cgrid.py,
  ordering only (both helpers kept); advection.py byte-identical to the
  gridladder tip.  Not imported: G-card recipe hunks (no G9 rung table here).
  All eight switches import default-off.
- Preregistration + gate `d566bd09a`; addendum `bbb4cee48` (before arms2);
  addendum 2 `e3eea05eb` (before arms3).  Selection `5a0c45540`; test
  `5256f50cf`; citation re-anchor `f3b13de8c`, `506c0c850`.
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round318_fold_d122_gate.py`
  (round-285 ladder, round-276 classifier, round-278 decision).  CPU, fp64,
  libm, production JIT.  Evidence `orca2_rounds/round318/` (`arms/`,
  `arms2/`, `arms3/`, `final/`, `jaxpr/`).
- Classifier extension (this round's gate only): the frozen comparison
  refused any move of the first non-bit row.  The full set makes kt-1
  stage-1 T and S exact on OMT-1..4, so their first non-bit row moves later
  (T -> u).  The gate recomputes both first rows from the row metrics (the
  ladder's own rule: first non-exact row in ladder order, entry rows only on
  rung 10; it reproduces the stored marker on all 194 round-318 records) and
  refuses a record whose marker disagrees.  A later first row is admitted
  only if the old first row is now exact and no row before the new first
  row got worse on any metric; exact-row losses stay refused and the first
  over-bar direction is scored at the new row (toward or equal required).
  An earlier move is refused.

## Map: statement -> setting -> compiled line (ORCA2_OMIP_L4 ppsrc)

| arm | setting | NEMO line |
|---|---|---|
| VMASK | `rk3_stage_transport_raw_vmask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283` (vmask after `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:232`) |
| FOLDX | `rk3_stage_transport_fold_exchanges` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:844`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:169` |
| FCTF | `fct_fold_exchange` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:311` |
| PIVOT | `rk3_end_of_stage_pivot_fold` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796` |
| CLOCKP | `rk3_stage_wzv_nemo_clock_pair` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221` |
| TWZV | `rk3_stage_tracer_wzv_stored_kmm_ratios` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223` |
| R3UV | `rk3_stage_transport_stored_r3uv` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282-283` |
| KRHS3 | `rk3_stage3_tracer_krhs_nemo_order` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:284-290` (not landed) |
| FOLDUNIT, RAWV, TOPV | G6 (round 308 map) | — |
| FOLDR | `nemo_ldf_fold_rows` (G8, round 310) | — |

(b): the stage transport and its wzv run unconditionally in stp_RK3_stg;
`traadv_fct` runs only where FCT is selected (rung 0, OMT-4, rung 10), so
FCTF is stated off on OMT-1..3 (it moved 0 rows there); fold rows follow
the lateral diffusion (off on OMT-1/2, where the set with them moved the
same rows as without).

## D122 table (control = the shipped card at the same tree)

Census = rows toward / away / equal; lost = exact rows lost; first =
direction of the first over-bar row (rung 10 independent: kt-1 entry ssh,
equal by construction).  "—" = refused or non-finite.

| arm | rung 0 | OMT-1 | OMT-2 | OMT-3 | OMT-4 | r10 indep | r10 given | lost | 7-pair D122 |
|---|---|---|---|---|---|---|---|---:|---|
| VMASK | 0 moved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS (inert) |
| FOLDX | 116/70 | 63/77 | 41/139 | 37/143 | 115/71 | 113/66 | 112/67 | 0 | PASS |
| FCTF | 0/179, ssh 6.24 m | 0 | 0 | 0 | 0/179, 6.24 m | 2/177, 6.15 m | 2/177, 6.16 m | 0 | PASS, leans away |
| PIVOT | 131/59 first away | 48/102 away | 91/99 away | 90/100 away | 131/59 away | 116/65 | 117/64 | 0 | FAIL (d) |
| CLOCKP | 64/77/35 | 0 | 0 | 0 | 53/78/55 | 92/77/14 | 84/76/28 | 0 | PASS |
| TWZV | 11/25/45 | 0 | 0 | 0 | 2/1/85 | 0 | 2/4/11 | 0 | PASS |
| R3UV | 42/84/16 | 0 | 0/0/35 | 0 | 36/77/41 | 37/44/19 | 57/49/29 | 0 | PASS |
| KRHS3 (and G9, G9FOLD, G9TOPV, G9FOLDR, ALL) | — | — | — | — | — | — | — | — | refuses (above) |
| FOLDUNIT | — | runs | runs | runs | — | — | — | — | refuses |
| RAWV | — | non-finite u kt 8 | — | — | — | — | — | — | refuses |
| TOPV | 23/165, ssh 7.25 m | 100/42 | 111/71 | 119/63 | 23/165, 7.25 m | (r308) | (r308) | 0 | PASS, leans away |
| FOLDR | 23/152 | 0 | 0 | 10/165 (r310) | 23/152 | (r310) | (r310) | 0 | PASS, leans away |
| FP (FCTF+PIVOT) | 12/178 first away, 2.12 m | | | | 12/178 away, 2.12 m | | 4/179, 2.09 m | 0 | FAIL (d) |
| VT (VMASK+TOPV) | 17/171, 0.77 m | | | | 17/171, 0.77 m | | | 0 | PASS, leans away |
| VTX (VT+FOLDX) | 14/176 away, 1.55 m | 100/42 | 111/71 | 119/63 | 14/176 away, 1.55 m | 14/176, 1.37 m | 14/176, 1.39 m | 0 | FAIL (d) |
| G9N (G9 - KRHS3) | 6/184 away, 2.10 m | 48/102 away | 91/99 away | 90/100 away | 6/184 away, 2.10 m | 9/180, 2.03 m | 10/179, 2.01 m | 0 | FAIL (d) |
| G9NTOPV | 25/166 away, 1.34 m | 55/96 away | 98/93 away | 97/94 away | 25/166 away, 1.34 m | 26/164, 1.42 m | 26/164, 1.44 m | 0 | FAIL (d) |
| G9NFOLDR | 5/185 away, 2.05 m | 48/102 away | 91/99 away | 88/102 away | 5/185 away, 2.05 m | 4/185, 2.00 m | 4/185, 1.98 m | 0 | FAIL (d) |
| **G9NFOLD** (G9N + FOLDUNIT, RAWV, TOPV) | **195/0** toward | **155/0** toward | **195/0** toward | **195/0** toward | **195/0** toward | **195/0** equal | **195/0** toward | 0 | **PASS** |
| **ALLN** (G9NFOLD + FOLDR) | **195/0** | **155/0** | **195/0** | **195/0** | **195/0** | **195/0** | **195/0** | 0 | **PASS** |

kt-10 max |ssh err| (m), control -> arm, for the two passing sets:
G9NFOLD rung 0 0.394 -> 0.00702, OMT-1 0.747 -> 2.91e-5, OMT-2 0.349 ->
4.30e-5, OMT-3 0.363 -> 0.00635, OMT-4 0.394 -> 0.00621, rung 10 0.277 ->
0.183 / 0.262 -> 0.168.  ALLN differs only where the lateral diffusion runs:
OMT-3 7.46e-13, OMT-4 1.89e-4 (rung 0 and rung 10 at the printed digits
unchanged; rows differ).  On OMT-1..4 the first non-bit row moves from kt-1
stage-1 T to kt-1 stage-1 u (toward).

## Which members the set needs (arms3, diagnostic; rung 0, OMT-4, rung 10 indep)

| arm | rung 0 | OMT-4 | rung 10 |
|---|---|---|---|
| without VMASK | 117/78, first away, 0.0370 m | 117/78 away, 0.0370 m | 117/78, 0.183 m |
| without FOLDX | 63/132 away, 1.51 m | 63/132 away, 1.51 m | 62/133, 1.48 m |
| without FOLDUNIT | 25/166 away, 1.34 m | 25/166 away, 1.34 m | 26/164, 1.42 m |
| without TOPV | refuses | refuses | refuses |
| without FCTF | 195/0, 0.00702 m | 195/0, 0.00621 m | 195/0, 0.183 m |
| without PIVOT | 195/0, 0.00702 m | 195/0, 0.00623 m | 195/0, 0.183 m |
| without RAWV | 195/0, 0.00702 m | 195/0, 0.00620 m | 195/0, 0.183 m |
| FOLDSET (no CLOCKP/TWZV/R3UV) | 195/0, 0.00702 m | 195/0, 0.00621 m | 195/0, 0.183 m |

The core is VMASK + FOLDX + FOLDUNIT + TOPV.  FCTF, PIVOT and RAWV are
net-toward increments inside the set (rung 0: 123/56, 127/41, 107/78; OMT-4:
139/40, 113/55, 125/60; rung 10: FCTF 139/40, PIVOT 115/62; 0 rows lost,
first row unchanged) and land with it.  CLOCKP, TWZV and R3UV are not fold
partners (FOLDSET gives the same sea-level error); each passes D122 alone
and lands as certified.

## Shipped cards

- Selection: rung 0, OMT-4, rung 10 run ALLN; OMT-3 ALLN without FCTF;
  OMT-1/2 ALLN without FCTF and fold rows; OMT-0 (flux form, outside the
  seven pairs) states the whole set off.
- Shipped cards at `5a0c45540` reproduce the ALLN arm row for row on all
  seven pairs (1,360 rows); the two pre-selection controls (`d566bd09a`,
  `bbb4cee48`) agree row for row (`final/instrument_check.txt`).
- The landing decision is the fold-set run: ALLN (and G9NFOLD) against the
  pre-selection control, seven pairs, PASS (`arms2/decide7_ALLN.json`),
  re-run with the fixed classifier: PASS, identical decision document
  (`fixed_classifier/decide7_ALLN.json`, `fixed_classifier/decide7_G9NFOLD.json`).
  The `final/decide7_D122.json` run is NOT a decision: its control did not
  strip the selection, so it compared the shipped card with itself (0 rows
  moved); it only shows the shipped card is deterministic.

## GYRE, DINO, VORTEX, OMT-0

Step jaxprs identical to base `3e9bb7dea`: after the default-off import, all
11 (rung 0, OMT-0..4, rung 10, GYRE, VORTEX flux and vector, DINO); after
the selection, GYRE, both VORTEX, DINO and OMT-0.  The GYRE ten-step archive
and day-30 run were not rerun (step program identical).

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R318-P1 | **CONFIRMED** | FCTF and KRHS3 move 0 rows on OMT-1..3 (KRHS3 refuses). |
| R318-P2 | **FALSIFIED** | KRHS3 refuses on all seven pairs; the other seven G9 singles complete. |
| R318-P3 | **FALSIFIED** | G9 refuses (KRHS3); G9N fails (d). |
| R318-P4 | superseded by P10 (G9 refuses). |
| R318-P5 | superseded by P11. |
| R318-P6 | **CONFIRMED** | ALLN: rung 0 0.394 -> 0.0070 m (56x), OMT-4 0.394 -> 1.9e-4 m. |
| R318-P7 | **FALSIFIED** | FP fails (d) on rung 0/OMT-4 and grows ssh to 2.1 m. |
| R318-P8 | **FALSIFIED** | VT passes (c)+(d) but doubles ssh error (0.77 m). |
| R318-P9 | **FALSIFIED** | G9N fails (d) on rung 0 and OMT-1..4. |
| R318-P10 | **CONFIRMED** | G9NFOLD completes on every pair. |
| R318-P11 | **FALSIFIED** | G9NTOPV 1.34 m on rung 0 and OMT-4. |
| R318-P12 | **FALSIFIED** | dropping FCTF, PIVOT or RAWV stays within 2x. |
| R318-P13 | **CONFIRMED** | FOLDSET = G9NFOLD's kt-10 ssh error on all three pairs. |

## Gates and tests

- New `tests/ocean/fidelity/test_orca2_round318_selection.py` (15 tests):
  per-card selection, OMT validators, each landed switch changes the traced
  rung-0 step, plant, and the classifier on the real OMT-1 control/ALLN
  records: the real T -> u move admitted; old first row worsened (marker
  kept) REFUSED; an earlier row worsened (kt-1 entry T, no exact row lost)
  REFUSED; new first row away -> not eligible; earlier move REFUSED.  With
  the pre-fix classifier 3 of these 5 fail (the two worsened plants were
  admitted eligible).  Reverting the recipe selection fails 6 (OMT-0
  unaffected, as intended); restored, clean.
- All 140 round-318 census records re-classified with the fixed classifier:
  0 verdicts changed.
- Seven stale pins that asserted the fold switches off now follow the
  selection (rounds 282, 308, 310, 311 selection/import tests, the G1
  transfer test, the OMT-1 and OMT-4 builder tests); the round-310 and
  round-311 tests now check their own gates on cards without the fold rows,
  the round-310 plant on OMT-1.  Neighbour batteries: 60 + 71 passed.
- Citation map: 148 lines re-anchored by the repo tool; nine entries whose
  ranges grew around imported lines aligned (the four advection.py entries
  = the gridladder's own); four new entries for this receipt.
- Receipt citation gate on this receipt: PASS, 12 citations, 0 failures,
  0 unmapped, 0 map entries failing.
- Default gate (land.sh set, JAX_ENABLE_X64=1, CPU, in three pieces under
  the 10-minute cap): 118 + 18 + 8 = 144 passed, 0 failed; private-import
  ratchet 6 passed within it; final re-run of citation gate + ratchet 23
  passed.

## Not done / OPEN

- OMT-0 under the fold set (needs its own flux-form ladder; G9 itself is
  flux form).
- KRHS3 on ORCA2: needs a stage-3 path that carries the diffusive BBL and a
  forcing-free form — new code, so a rung must certify it first.
- Rung 0's remaining T error (kt-10 RMS 5.4e-3 K, max 0.77 K at kt 6) and
  rung 10's remaining 0.18 m are not the fold.
- One rung-10 ALLN run aborted at interpreter exit after writing its
  complete record (exit 134, STATUS line printed); its rows equal the
  shipped card's.

ASKED: import the fold set from lane A; D122 on seven pairs, alone and
together; report the kt-10 sea-level error; keep GYRE/DINO/VORTEX.
UNASKED, offered for revert: (1) OMT-0 states the set off (unmeasured
there); (2) the classifier admits a later first non-bit row (gate-local;
awaiting the user's ruling on this reading before landing).

## Review (round 318 resume)

- Claude review: LAND (import = lane A's G9 commits, rows reproduced).
- Codex review: HOLD — the classifier trusted the candidate's first-row
  marker, so a record whose old first row worsened returned eligible; the
  mock test could not see it; the final decision's control was the shipped
  card itself.  Fixed above (recomputed first rows, real-record plants,
  decision cited from the fold-set run).  Gate and docs only; no model change.
