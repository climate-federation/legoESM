# Grid ladder — G3 re-certification setup + G4 round 5 receipt (gridladder_r21)

Status: **STOPPED_FOR_RECORD.** Decision 120 is in the decks and cards and
proven offline; the one re-acquisition script is written and preflighted;
NEMO's per-step forcing is wired into the G4 card (Decision 123) and its
stage-1 test is committed. G3 is NOT certified on the new deck until the
record exists; G4 is NOT certified.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g4_round5.md (31cb65872).
Code: ea8c607b0, bee614813, 90ad59713, a1ea50949. Evidence: gridladder_rounds/round21/.
Acquisition: scripts/validate/ocean_fidelity/testcases/nemo_testcase_g4_orca2sbc/run_r21.sh.

## Take-aways

1. **Decision 120 holds offline, bit for bit.** Both cards' initial T and S
   equal NEMO's on all 430,552 wet cells, all levels; NEMO's branch is
   `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dtatsd.f90:219`. Without the edits
   the card differs on exactly the 1,283 T / 720 S edited cells (masks, not
   counts). Restoring stays off: the only per-step caller is
   `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/tradmp.f90:181`.
2. **One script re-acquires both records** on the new decks, three admitted
   binaries, no build, smokes first, --admit-existing.
3. **No step-1 restart can exist on this deck.** A listed step opens one step
   early (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:119`), never at step 0;
   every-step restarts and 1-step runs are stopped by NEMO because nn_fsbc = 2
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/sbcmod.f90:346`). So G3 keeps
   round 15's restarts 2..10.
4. **The forcing route is consumed, but stage 1 is not yet NEMO's.** P1 REFUTED.

## 1. Decision 120 (decks and cards)

- Decks: one line each, `ln_tsd_dmp = .true.`; `ln_tradmp = .false.` unchanged.
- Card: the damping switch moved from the shared required list to a
  per-rung table (G1, G2 off; G3, G4 on); G3/G4 build ORCA2's reader with
  the hand edits. G1, G2, the ORCA2 card and its ablation control unchanged.
- Offline check (d120_initial_state.py): NEMO side = the ORCA2 lane's kt = 1
  entry (same domain and climatology files by SHA-256; calendar, time step
  and namtsd read from NEMO's resolved namelist). Controls: as-built G3
  record level 1 equal (no edit reaches level 1); a planted 1e-15 exits 1.
- The as-built records (rounds 14, 15, 17, 19) are now refused by the card;
  the two slow G3 certification tests on them are skipped with that reason,
  and a fast test asserts the refusal.

## 2. run_r21.sh

| rung | runs | admission |
|---|---|---|
| G3 | smoke (reference binary), 100-step reference, twins a/b/c (P5) | round 15's: checker, 7 plants, writer fields, restarts, twin frames |
| G4 | P5 set (smoke, reference, p5_a/b/c) + P6 set (smoke6, p6_a/b/c) | round 17's on P5; round 19's forcing checks and consumed-support passivity (P6 vs P5) on P6 |

Preflight: each deck is the as-built deck (ad1ba285e) with only
ln_tsd_dmp moved; namtsd is ORCA2's block; binaries and every copied input
pinned. PREFLIGHT_OK at a1ea50949; two planted decks refused (exit 68).
Record length is NEMO's per rung (min(10, completed), even).

## 3. Decision 123: NEMO-supplied forcing in the G4 card

- **Reused:** the ORCA2 lane's `_surface_forcings` (round-1 ladder gate),
  unchanged, fed from the round-19 `s` frame. No new evaluator, no bulk or
  model change. The ORCA2 and G4 cards' configurations differ in no
  forcing-consumption setting (measured by diffing both cards).
- **Read sites** (P5 build): stress `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:198-199`;
  emp, rnf `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:242-245` and
  `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:285`; qns, sfx `:307-308`;
  runoff content `:323`; solar `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/traqsr.f90:371-373`.

Test, kt = 1, GIVEN-NEMO-ENTRY (as-built initial state), round-19 record:

| stage | ssh | uu_b / vv_b | u / v level 1 | T / S level 1 |
|---|---|---|---|---|
| zero forcing, 1 | 16,433 / 0.10 m | 15,853 / 15,943 | 0.19 m/s | 16,433 / 16,433 |
| NEMO forcing, 1 | 2,244 / 4.2e-10 | 2,223 / 2,222, 9.8e-10 | 2,272 / 2,198 | 1,490 / 1,478, 1e-11 |
| NEMO forcing, 2 | 2,255 / 6.3e-10 | same | 2,302 / 2,215 | 1,515 / 1,500 |
| NEMO forcing, 3 | 2,255 / 1.3e-9 | same | 15,799 / 15,943, 2.7e-3 | 2,065 / 1,562 |

(unequal wet cells / max absolute difference.)
- Stage 1: every unequal cell lies in one patch around the sea-level bump
  (rows 42-106, columns 28-75); 26 coastal cells, few near river mouths.
- Stage 3: u, v differ wherever the wind blows; median 1.1e-6 relative,
  largest 1.1 % at (j, i) = (115, 143). The stress increment itself agrees
  with NEMO's to 4e-4 relative at worst. PLAUSIBLE only: the surface-stress
  term divides by a thickness at a different time level.
- Predictions: P1 REFUTED, P2 REFUTED, P3 (stage 3 first) REFUTED.
  The preregistration also misplaced the runoff content: NEMO adds it at
  every stage (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:323` lies
  outside the stage switch), found by the review.

## 4. Tests (synchronous)

Fast focused (G1 tools, G2, G2 ladder, G3, G4, hand-alteration unit, ORCA2
rung-0 card): 99 passed, 10 deselected. Slow route test: 1 passed (158 s).
D120 offline check: exit 0; plant exits 1.

## 5. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r21.log). All four confirmed, acted on:
inputs not pinned (pinned); route always exit 0 and an aggregate-only test
(nonzero exit on any unequal row; every wet column checked); D120 compared
counts not cells and did not gate the level-1 control (masks; gated);
preregistered runoff timing wrong (recorded above, preregistration kept).

## 6. Choices

ASKED (round brief): the deck lines; offline proof; one script; binary
reuse; ORCA2 lane's route; round-19 record test.
UNASKED, each offered for revert:
1. G3's boundary rule is now G4's (accept any stp_ctl stop, record = min(10, completed)); round 15 accepted only a kt = 11 |V| stop.
2. The G3 twins stay a/b/c (no step-1 restart exists, item 3 above).
3. The two slow G3 certification tests on the as-built records are skipped, not deleted.

No gate exists for: dual review (single, lane rule 6), claim review before code.

## OPEN

1. Run `run_r21.sh --run`; then re-certify G3 (kt 1..10) on the new record.
2. G4 stage 1: the bump-patch difference (≤1e-9) — first unequal statement unnamed.
3. G4 stage 3: the stress response, ~1e-6 relative everywhere.
4. One-line finding, not worked: the ORCA2 card spreads runoff over 150 m; NEMO's deck leaves ln_rnf_depth off (top level only).

UNVERIFIED:
- the stage-3 cause (not replayed);
- that run_r21.sh runs end to end (preflight only; NEMO not run).

## 7. Evidence hashes

SHA-256 prefixes, round21/: d120_initial_state.json=f727bcc4d8340bcf, d120_plant.log=89f20976577aa51e, preflight_final.log=c8ec5523ec43fd67, route_kt1.json=0cb9695a26cf962f, route_kt1_dump.json=d1b55e43d96f1cd7, route_kt1_zero.json=c14f0d203dfbf29c, tests_focused_final.log=eca2266b1af470de, tests_route_slow.log=d8658f959ff64082, codex_review_r21.log=f03725eec25b2952, scripts/tmp/_probe_r21_route.py=9705dbe785a17ea5.
