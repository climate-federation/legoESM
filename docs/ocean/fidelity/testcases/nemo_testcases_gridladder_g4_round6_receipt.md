# Grid ladder — G1b repair, G3 re-certification, G4 round 6 (gridladder_r23)

Status: **LANDED.** G3 is CERTIFIED on the Decision-120 record. G4 is
NOT certified: three cited NEMO statements landed (G4 card only); the
first remaining non-bit statement is named below. G1b's acquisition is
repaired and returned to the operator.
Code: 11f3310be (G1b), a23a7217d (G4), a95f245a3 (review fixes).
Records: gridladder_rounds/round21/oracle_g3_r21, oracle_g4_r21 (run_r21.sh,
ACQUISITION_DONE, every check PASS). Evidence: gridladder_rounds/round23/.

## 0. G1b repair

- Round 22's run finished the 2-step smoke and the 100-step reference
  (RUN_DONE, time.step 100), then refused twin r01's deck before NEMO ran:
  at N = 100 only nn_stocklist moves, and the check wanted 0 or 3 lines.
- Fixed: any changed line that is not one of the three run-length lines
  refuses. Plant (rn_Dt moved) refuses in preflight.
- `run.sh --resume` admits the smoke and reference in place (pins, deck,
  log), checks all ten twin directories first, then runs the twins.
  Preflight: `RESUME_READY smoke and 100-step reference admissible in place`.

## 1. G3 re-certification (Decision-120 record, hand-edited cells included)

| label | steps | rows | unequal cells |
|---|---|---|---|
| INDEPENDENT | kt 1..10 | ssh, uu_b, vv_b, u, v every step; T, S level 1 kt 1..9; u v T S all levels kt 2..10 (full and partial); six barotropic histories kt 10 | 0 |
| GIVEN-NEMO-ENTRY | entries 1, 3..9 | 2-D and level 1 at the entry step; all levels at the next step | 0 |

Every region (fold, seam, south, coast, open, the three fold rows) and
both depth classes: 0. The hand-edited cells are inside these rows.
G3 is CERTIFIED on NEMO's exact initial state.

## 2. G4: the walk (kt = 1, NEMO's own surface fields, Decision 123)

Instrument: the round-21 route, now on the Decision-120 record (the card's
initial state is NEMO's, so no as-built swap), plus an offline probe that
traces the card's solver against NEMO's substep record. Before landing,
stage 1 reproduced round 21 exactly (ssh 2,244 cells / 4.2e-10).

Three first-unequal statements, in compiled order, each landed and re-measured:

1. **Barotropic wind divisor.** First unequal operand: the solver's forcing
   at entry, 226 U faces, all where the entry sea level is nonzero. NEMO
   divides by its stored `r1_hu_0/(1+r3u(Kbb))`
   (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:198`); the card by its live
   depth, 4.7e-6 relative apart. The stored ratio needs the fold exchange
   the card already has.
2. **Pivot-row V stress.** Then 68 V faces on the fold pivot row. NEMO folds
   the T stress with sign −1 (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcmod.f90:548`)
   and averages it (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcmod.f90:559`); the
   card took the scalar mirror, then doubled it (its pivot row is stored as
   wall). Rebuilt from NEMO's own halo: bit-equal on all 68.
3. **dyn_zdf stress deposit.** Stage 3, u off by up to 2.7e-3. NEMO divides
   by `e3u_3d(1)*(1+r3u(Kaa))` and adds it after the barotropic removal and
   the bottom drag (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynzdf.f90:338`).

| stage | ssh | uu_b / vv_b | u | v | T | S |
|---|---|---|---|---|---|---|
| 1 before | 2,244 / 4.2e-10 | 2,223 / 2,222 | 2,272 | 2,198 | 1,490 / 8.0e-12 | 1,478 / 1.0e-11 |
| 1 after | 0 | 0 / 0 | 56 (±0 only) | 0 | 58 / 3.6e-15 | 45 / 7.1e-15 |
| 2 after | 0 | 0 / 0 | 73 / 2.6e-16 | 17 / 5.7e-16 | 58 | 43 |
| 3 before | 2,255 / 1.3e-9 | 2,223 / 2,222 | 15,796 / 2.7e-3 | 15,943 / 7.1e-4 | 2,065 | 1,562 |
| 3 after | 0 | 0 / 0 | 90 / 2.6e-16 | 28 / 1.0e-15 | 489 / 3.6e-15 | 0 |

(unequal cells / max absolute difference.)

**First non-bit statement now (OPEN, PLAUSIBLE):** stage 1's mass-flux
dilution of T and S (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:284`),
whose divisor is `e3t_3d(1)*(1+r3t(Kmm))`. All 58 T and 45 S cells lie on
the 201 bump cells; the card's entry top thickness differs from NEMO's
form on 37 of them (≤4.5e-14 relative). That explains part, not all, of the
58 — not landed.

G4 ladders (kt 1..10, both labels) not run: stage 1 is not yet NEMO's.

## 3. Shared code and selection

| change | where | who selects |
|---|---|---|
| wind divisor = stored, fold-exchanged `r1_hu_0/(1+r3u(Kbb))` | model, barotropic forcing | G4 card only |
| pivot-row V stress from the vector fold halo and folded masks | shared stress helper (new keyword) | G4 card only |
| dyn_zdf stress: NEMO's face thickness, added after removal and drag | model, implicit solve | G4 card only |

Three flags, default False. Test pins: G4 the only card selecting them;
G1–G3 configs unchanged. Each refuses where it would do nothing (no RK3,
no fold, no NEMO masks, no NEMO divisors). Searched before building: the
shared face-ratio builder (vertical.py, reused), the fold-exchange helper
(reused), the drag divisor (reused).

**D122 pointer, same lines in ORCA2's build:**
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:229`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:557`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:568`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:338`. Unmeasured on ORCA2's card.

## 4. Runoff depth (verified)

- NEMO: neither deck sets `ln_rnf_depth`/`ln_rnf_depth_ini`, so runoff goes
  to the top cell (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcrnf.f90:279`);
  `rn_dep_max` is read only under `ln_rnf_depth_ini`.
- G4 card: 0.0 m, no map — set by the grid-ladder builder. Not a defect.
- **Finding (ORCA2 lane's card, not changed):** the shipped ORCA2 card
  spreads runoff over 150 m, inherited from the shared recipe default;
  NEMO's deck puts it at the surface. Its rung-0 gate pins 0.0.

## 5. Tests (synchronous)

- G4 fast + stress unit tests: 37 passed. Slow route test (stage 1 and 3
  bars, plus the deselected arm reproducing round 21's gap): 1 passed (205 s).
- Card tests (G1, G1b, G3): 73 passed, 1 failed — the stand-in G1 card
  "disagrees with its deck: initial_T", failing identically at the round's
  start commit (not this round's).
- Not run: the G1 and G2 ladder fast suites (over the 10-minute limit).
- Non-vacuity: deselecting the three flags restores round 21's 2,244-cell gap.

## 6. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r23.log).
1. Medium, confirmed, fixed: `--resume` now checks every twin directory
   before running any.
2. High, measured: "the pivot T row is not folded". NEMO's recorded
   `utau`/`vtau` pivot rows are already the sign −1 fold, and U stress is
   0 unequal. Applies only to a card-computed stress (the registered bulk
   debt); scope stated in the helper.

## 7. Choices

ASKED (round brief): resume path; the three statements on the G4 card;
the runoff report.
UNASKED, each offered for revert:
1. Three separate flags, one per statement (not one switch).
2. The round-19 test now runs the deselected card (keeps round 21's finding).

No gate exists for: dual review (single, lane rule 6), claim review before code.

UNVERIFIED:
- the stage-1 T/S cause (37 of 58 cells explained);
- that `--resume` runs end to end (preflight only).

## 8. Evidence hashes

SHA-256 prefixes in round23/ are listed in `round23/manifest.sha256`.
