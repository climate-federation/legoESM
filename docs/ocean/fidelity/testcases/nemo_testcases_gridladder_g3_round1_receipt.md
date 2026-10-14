# Grid ladder rung G3 — round 1 receipt (gridladder_r14)

Status: **STOPPED_FOR_RECORD**, with one decision for the user. The G3 deck,
the acquisition script, the G3 card and the preregistration exist and are
committed. NEMO has not been run on G3; nothing here is a comparison with NEMO.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g3_round1.md
(commit a1c1547cf, before the card).
Code: 4c7c6cc8b (deck, script, card, tests), 69e20880f, 67a68f023, 8dca821b1
(citation map), the review fix after it, and this receipt's commit.
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round14/.
**Acquisition needed:**
scripts/validate/ocean_fidelity/testcases/nemo_testcase_g3_orca2ts/run.sh
(operator: `run.sh --run`; it writes .../round14/oracle_g3_r14).

## Take-aways

1. **One deck switch, as expected: `&namtsd`.** `ln_tsd_init` turns on and
   ORCA2's three data lines appear; nothing outside that block moves.
2. **Damping off also drops ORCA2's hand edits of the initial state.** NEMO
   edits the Alboran and Red Sea T/S only when the damping switch is on, so
   G3 starts from ORCA2's state except on 1,283 T and 720 S cells (Red Sea
   deep water up to 20.5 °C apart). Decision below.
3. **No build.** G2's two binaries are reused (same keys; the T/S reader is
   compiled in every build and byte-identical to ORCA2's own build).
4. **The card is G2's, plus the initial state.** Its model configuration
   equals G2's field for field; the initial T/S comes from the ORCA2 card's
   existing reader.
5. **G1 and G2 are unchanged.** G1's 100-step certification and G2's step-1
   bit-identity test pass on this tree.

## 1. What exists

- **Deck:** nemo_testcase_g3_orca2ts/namelist_cfg. Outside `&namtsd` its
  assignments are G2's (which are G1's); the script and a test refuse
  otherwise. `ln_eos80`, `ln_hpg_sco`, damping off, everything else: G2's.
- **Acquisition script:** run.sh. Pins the two binaries, the shipped domain
  file and the two climatology files by SHA-256. Smoke (2 steps) first, then
  the 100-step reference, the 100-step record (twin a), twin b (restarts 2,
  4, 6, 8, 10) and twin c (restarts 3, 5, 7, 9). G2 round 3 learned that a
  list of consecutive steps never opens; even and odd lists each do.
  Admission: G1's record checker, G2's judgments, NEMO's log stating the T/S
  read with damping off, each run's whole deck at its own length and list,
  every listed restart present, and twins b and c writing step frames
  byte-identical to twin a's at steps 1 to 10.
- **Card:** ORCA2G3-ts, through the shared builder table. Declared not
  execution-ready (no record; density step unmeasured).
- **Tests:** tests/ocean/fidelity/test_nemo_testcase_g3_orca2ts.py.

## 2. Changes to shared code

| code | change | effect on G1, G2 |
|---|---|---|
| initial T/S | one rung-keyed helper: uniform 20/30, or ORCA2's reader | none: the same product, `20 * mask` |
| deck switch list | `ln_tsd_init` moved to the rung table | none: G1 and G2 still must state false |
| validator rows | `uniform_T/S` renamed `initial_T/S`, compared with the helper | none |
| rung tables | a G3 row in each; the three G2 statements selected for G3 | none |
| ORCA2 reader docstring | the no-hand-edits branch is NEMO's when damping is off | text only |

- The helper refuses a G3 deck whose data lines, start date or time step
  differ: the reader's time weight (249/496) is transcribed for this
  calendar only.
- Searched before building: the recipe file for an initial T/S reader
  (found `build_orca2_initial_ts` and `apply_orca2_hand_alterations`, the
  ORCA2 card's, reused) and the acquisition directories (G2's script,
  copied and extended). New: the helper, the G3 rows, a two-line card
  function.

## 3. Initial state, offline (the card, not NEMO)

g3_initial_state_facts.json (run on the uncommitted tree, before 4c7c6cc8b;
the committed tests reassert the counts): 430,552 wet cells; T from -2.09
to 30.09 °C, S from 4.51 to 40.58; at the bump centre T is 28.37 °C at the
top and 0.87 °C at the bottom. The card equals the unedited climatology
bit for bit; the edited one differs on 1,283 T and 720 S cells, the ORCA2
lane's counts against NEMO.

## 4. Gates

| claim | gate | result |
|---|---|---|
| deck = G2 outside namtsd; namtsd = ORCA2's lines | test; script preflight | pass; four planted decks refused, exit 68 |
| binaries, domain and T/S files pinned | preflight | PREFLIGHT_OK (preflight_g3_final.log) |
| admission refuses a missing listed restart, a wrong list | fake evidence, no NEMO run | exit 73, three plants |
| card = G2's configuration; unedited climatology | test | pass; planting the edits makes it fail (0 ≠ 1283) |
| card refuses wrong damping, file, time step, date, init | test | pass |
| G1 100-step certification | G1 test | 4 passed |
| G2 step 1 bit-identical to NEMO | G2 slow test | 1 passed |
| citations | gate on the preregistration | PASS, 11 citations, 0 unmapped, audit clean; plant exits 1 |
| the script's run and admission on a real record | NO GATE | unexecuted |
| dual review, claim review before code | NO GATE | single review (lane rule 6); claim review NOT done |

Focused tests, synchronous, committed tree before the review fix: 45
passed, 7 slow deselected (G3, G2, G2 ladder, hand-edits unit, citation
gate files); after the fix, the G3 file: 4 passed. NOT RUN: the rest of
the G1 ladder file; ratchets; closed cards.

## 5. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r14.log,
scope f6488baa7..8dca821b1). Both acted on, not re-reviewed:

- "CONFIRMED: run.sh lines 319-320 requires a step-10 restart from twin C, but
  its 3,5,7,9,0... list leaves `nitrst=0` after step 9; NEMO cannot create
  that file, so `--run` always aborts." Fixed: the runner checks each run's
  last listed step.
- "CONFIRMED: run.sh lines 192-194 and 206-208 excludes restart schedules from deck
  comparison and checks only steps 10/100; missing or altered advertised
  20–90 restarts can be admitted." Fixed: admission compares each run's
  whole deck and requires every listed restart (plants above). G2's script,
  which has the same gap, is NOT changed.

## 6. Choices

ASKED (round brief): namtsd as ORCA2's deck; EOS and pressure gradient as
ORCA2's (unchanged); damping off; everything else G2's; binary reuse; smoke
first; twins and 100-step record; G2 card machinery; preregistration;
focused tests; one review.

UNASKED, each offered for revert:
1. Twin c (10 steps, odd restarts) added to the restart set.
2. The card's case name ORCA2G3-ts; the experiment name stays ORCA2G1.
3. The shared-code changes of section 2, including the renamed validator rows.
4. The card refuses decks off this calendar instead of deriving the weight.
5. The citation map gained five source files and ten pins; one older pin
   re-anchored to the rewritten docstring line.

Compliance, stated because no gate enforces it: search-before-build
(section 2); no NEMO number quoted from an unvalidated instrument (none
quoted); non-vacuity (plants in section 4).

UNVERIFIED:
- that NEMO accepts the deck and reads the files (G3-P1; read off the
  source, not run);
- that the record binary's writers behave as on G2 with real T/S;
- the admission branch on a real record.

## 7. DECISION for the user

ORCA2's real initial state includes the hand edits; damping off removes
them. Options:
1. Keep damping off: G3 starts without the edits (built this way).
2. **Damping switch on, restoring off** (`ln_tsd_dmp = .true.`,
   `ln_tradmp = .false.`): ORCA2's exact initial state, still no restoring;
   the switch reaches nothing else when restoring is off. **My pick.**

Option 2 changes one deck line and one card row; the acquisition would be
rerun.

## 8. OPEN for round 2 (after the operator's acquisition)

1. Admit the record; read NEMO's log for G3-P1.
2. G3-P2: level-1 entry T/S against the card.
3. Walk step 1 in NEMO's order (density, pressure gradient, depth mean);
   score with G2's level-aware ladder and its depth-class and region splits.
4. Levels 2 to 30 have no witness before step 2; if the walk needs them
   inside step 1, a stage writer for all levels is a new build (named, not
   built).

SHA-256 prefixes, round14/: citation_prereg.json=f02326b2d09cfa9c,
citation_prereg_plant.json=4ff947542d86cff8,
g3_initial_state_facts.json=6da00fd15069661c,
codex_review_r14.log=dd4ae0bf82367f93,
preflight_g3_final.log=aff3ab7f1d361bc5,
deck_plant_outside.log=fbe8b38efb648f0a,
deck_plant_sn_sal.log=ec28ad6459e94bbf (dmp, init: same text),
admit_plant_missing_restart.log=775151af67c0a726,
admit_plant_wrong_list.log=ae808e5b5284bbb3,
plant_hand_alterations_on.log=79bdd8d5407935ff,
tests_focused_committed.log=7f845b56da79d00f,
tests_g3_final.log=66de5e4c9a1083ba,
tests_g2_step1_slow.log=2c5913435c557ad5,
tests_g2_step1_g1_100step_dirty.log=2123078d69937b4c,
tests_citation_gate.log=2d3a75258e9588b7.
Deck: G3 namelist_cfg=5b2699e8e20c49bd. Climatology: T ca00905c27078305,
S ad648d972f0631bd. Reused binaries: reference 5ed4b6e543f32b82, record
9b96f1427503da27.
