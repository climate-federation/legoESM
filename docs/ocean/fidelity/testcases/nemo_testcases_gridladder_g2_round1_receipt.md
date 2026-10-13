# Grid ladder rung G2 — round 1 receipt (gridladder_r10)

Status: **STOPPED_FOR_RECORD**. The G2 deck, the acquisition script, the G2
card and the preregistration exist and are committed. NEMO has not been run
on G2; nothing here is a comparison with NEMO.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g2_round1.md
(commit b785de2f4, before the card and before any record).
Code: 5189dc9bf (card, citation map, tests), 98ef167cd and 1aa08f52f (review
fixes), and this receipt's commit.
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/.
**Acquisition needed:**
scripts/validate/ocean_fidelity/testcases/nemo_testcase_g2_orca2bathy/run.sh
(operator: `run.sh --run`; it writes .../round10/oracle_g2_r10).

## Take-aways

1. **G2 needs no deck switch besides the domain.** The G2 deck's 129
   assignments equal the certified G1 deck's; the script refuses otherwise.
2. **No build.** NEMO sizes the domain from the file, and the compiled keys
   are G1's, so the two round-4 G1 binaries are reused, pinned by SHA-256.
3. **Three things change with the file without being deck switches:** 65
   automatic barotropic substeps instead of 13; the file's strait mask at
   Gibraltar and Bab el Mandeb; the file's own Coriolis fields.
4. **The record binary writes level 1 only.** Levels 2 to 30 come from NEMO's
   restart files: steps 10, 20, .., 100, and steps 2, 4, 6, 8, 10 in the short
   twin (one added output, offered for revert).
5. **The G2 card is the G1 builder on the G2 deck root.** Against the G1 card
   one configuration field differs: the resolved substep count.
6. **G1 is unchanged.** Its certification test (100 steps, no row over the
   bar, no unequal cell after step 11) passes on this tree.
7. **Under this build partial cells enter only through thicknesses and
   masks.** Depths and the interface thickness stay on the 1-D ladder
   (preregistration, section 2, with NEMO's lines).

## 1. Survey of ORCA2's shipped domain file

Path: /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
SHA-256: 7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839

Read offline from the file (not NEMO output, except where said):

| fact | value |
|---|---|
| levels | 31 (30 wet at most; 3 at least) |
| wet columns | 16,433; coast equal to G1's |
| columns ending in a partial cell | 15,554 |
| thinnest bottom cell | 1.31 m |
| depth at the bump centre (51,74) | 5,312 m |
| substeps, Courant number | 65, 0.7980181110475545 |
| the same two numbers in NEMO's own ORCA2 log | 65, 0.79801811104755460 |

The NEMO log is the ORCA2 lane's round-1 run (same file, same time step).

## 2. What exists

- **Deck:** nemo_testcase_g2_orca2bathy/namelist_cfg. Header and three
  comments differ from G1's deck; no assignment does. The experiment name
  stays ORCA2G1 (it names files; the reused binaries and checker rely on it).
- **Acquisition script:** nemo_testcase_g2_orca2bathy/run.sh. No build, no
  domain tool. A 2-step smoke of the reference binary first; then the
  100-step reference, the 100-step record (twin a) and a 10-step twin b.
  Admission is G1's record checker plus four G2 judgments: provenance of
  every run directory, the frames' field lists against the committed writer
  patch, twin b's step-10 restart equal to twin a's, and 31 levels in NEMO's
  log.
- **Card:** ORCA2G2-zps, dispatched by the shared builder table. Declared not
  execution-ready (no record; multi-level partial-cell step unmeasured).
- **Tests:** tests/ocean/fidelity/test_nemo_testcase_g2_orca2bathy.py.

## 3. Changes to shared G1 code

| code | change | G1 effect |
|---|---|---|
| substep helper | column depth summed over wet levels only | none |
| validator | T, S, cell thickness checked on every level | none |
| domain reader | level count is the rung's, not the constant 2 | none |
| strait mask | read from the file when present | none (G1's file has none) |
| dispatch | both rungs through one branch | none |

- The first row is a real defect for more than one level: the file holds
  nonzero thicknesses below the bottom. Its plant test: the unmasked sum does
  not give 65.
- Every edit inside the G1 block keeps its line number (earlier receipts cite
  them); new code is appended at the end of the file.
- Searched before building (grep of the recipe file and the acquisition
  directories): the builder, deck reader, bump, mask builder, substep helper
  and record checker existed and are reused. New: one small reader for the
  strait mask, a two-entry rung table, a three-line card function.

## 4. Card execution smoke (not a comparison)

Two steps of the G2 card from rest (card_smoke_2step_dirty.json, run on the
uncommitted tree before the card commit): all fields finite; velocity nonzero
on all 30 levels; T leaves 20 by up to 9.2e-4 after step 1. About 54 s per
step on this machine, so a 100-step chain is about 90 minutes.

## 5. Gates

| claim | gate | result |
|---|---|---|
| deck assignments equal G1's | script preflight; committed test | pass; a flipped switch is refused (deck_gate_plant.log) |
| binaries are the round-4 ones, G1's program, writer gates as stated | script preflight (SHA-256, compiled sources) | PREFLIGHT_OK at 98ef167cd |
| admission refuses a wrong domain file / a wrong binary | fake evidence, no NEMO run | exit 73, twice |
| frame field lists parse from the writer patch | against the certified G1 record's admission | 60 and 6 fields, equal; a dropped field differs |
| G2 card builds; selectors; 65 substeps; planted disagreements refused | pytest, on the shipped file | pass |
| G2 against G1: one configuration field | pytest, both real domain files | pass |
| G1 certification holds | the G1 100-step test, card arm | 1 passed |
| citations | citation gate on the preregistration | PASS: 11 citations, 0 unmapped, audit clean; shifted-citation plant exits 1 |
| the script's run and admission branches on a real record | NO GATE | unexecuted |
| NEMO runs this deck on 31 levels | NO GATE this round | prediction G-P3 |

Focused tests, synchronous, committed tree (tests_focused_committed.log):
108 passed (G2 file, G1 card file, G1 tools file, citation gate file).
G1 ladder file: 22 of 81 passed when its 590 s limit ended the run
(tests_g1_ladder.log); the other 59 were NOT run, except the 100-step card
arm, run alone afterwards (tests_g1_100step_card.log, 1 passed).
NOT RUN: the ratchet files; any closed-card ladder.

## 6. Review

Single review (codex): **findings**. Log: codex_review_r10.log (its last
lines). Scope: the two commits b785de2f4 and 5189dc9bf.

- "CONFIRMED: `--admit-existing` can certify evidence without the required
  smoke run or provenance from the pinned binaries/domain." Fixed: admission
  now judges every run directory (98ef167cd); two planted refusals shown.
- "CONFIRMED: G1's checker can admit an incomplete stage record." Fixed for
  G2: field lists are compared with the writer patch. G1's checker itself is
  NOT changed.
- "CONFIRMED: normal CI can skip every test that constructs or validates the
  G2 card because they require an absolute external domain path." True. Not
  fixed: the card is defined on that file. Registered.
- "CONFIRMED: the "no build" test rejects only the literal text". Fixed.
- "No refutation found for G1 value preservation, axis conversion,
  masks/strait handling, 65 substeps, one-dimensional depth/interface ladder,
  face-bottom minima, deck legality, restart list legality, or `mpirun`
  failure propagation."
- "VERDICT: findings".

A single review, not a dual one (the lane's rule 6). The fixes after the
review were not re-reviewed. Claim review before code: NOT done.

## 7. Corrections to my own work, and choices

Corrections:
1. Before reading NEMO's pressure-gradient routine I expected a spurious
   bottom pressure gradient at partial cells. Wrong for this build: depths
   are on the 1-D ladder. It never reached the preregistration.
2. The first draft said three straits; the file's history names two with a
   mask value. Corrected before the commit.
3. Three of my own test expectations were wrong (an assignment count, a
   refusal message, a word match on a file extension). The third reached a
   commit (98ef167cd) with the test red, because a piped test run hid its
   exit status; fixed in 1aa08f52f.

ASKED (round brief): shipped file directly; G1 deck otherwise; binary reuse;
smoke first; twins and 100-step record; G1 card machinery; preregistration;
focused tests; one review.

UNASKED, each offered for revert:
1. Twin b writes restarts at steps 2, 4, 6, 8 as well as 10.
2. The experiment name stays ORCA2G1.
3. The card's case name, ORCA2G2-zps.
4. The shared-code changes of section 3.
5. Two older line pins of the citation gate re-pinned to the edited lines.
6. Admission's four extra judgments (section 2).
7. The third review finding registered, not coded.

Compliance, stated because no gate enforces it: search-before-build
(section 3); no NEMO number quoted from an unvalidated instrument (none
quoted beyond NEMO's own log); non-vacuity (plants in section 5).

UNVERIFIED:
- that NEMO runs this deck on 31 levels without a further switch (read off
  the source and off ORCA2's own runs, not run);
- that no reader of the strait mask exists in this program;
- that the reused record binary's substep writer behaves at 65 substeps as
  at 13 (the checker reads the count from the stream's own header);
- the 59 G1 ladder tests not run; closed cards (default paths read as
  unchanged, not run).

## OPEN for round 2 (after the operator's acquisition)

1. Admit the record; read NEMO's log for G-P2; geometry rows (G-P1).
2. The G1 ladder script is written for one level (level-1 field names, one
   trailing level). It needs a level-aware reader before any G2 arm runs:
   level 1 from the step frames, levels 2 to 30 from the restarts.
3. A 100-step chain costs about 90 minutes: it must be run in chunks that
   each fit one foreground call, carrying the state between them.
4. Walk step 1 in NEMO's order; G-P7 names the first candidate (the solver's
   entry forcing in the substep record).
5. If the walk needs levels 2 to 30 inside a step, a stage writer for all
   levels is a new build (named in the preregistration, not built).
6. Round 9's two open readers (stage transport geometry; the vertical
   solve's interface thickness) now have levels to act on.

SHA-256 prefixes, round10/: card_smoke_2step_dirty.json=87581911cd943183,
citation_prereg.json=c8c5d604e62f5da1, citation_prereg_plant.json=5d8ec832f2dbe268,
codex_review_r10.log=cb1f941e471cdbc7, preflight_g2_final.log=86a0b0cd84344fd7,
deck_gate_plant.log=2e22725cbb216b1b, admit_plant_wrong_domain.log=062a022bcb1fbf6f,
admit_plant_wrong_binary.log=bedcbf1b20767de0, tests_focused_committed.log=ddcda833f95e4d77,
tests_g1_100step_card.log=e3cb54576dad0f8b, tests_g1_ladder.log=7a9a743558ed9716.
Decks: G2 namelist_cfg=0dff86e332e6f06d, G1 namelist_cfg=73015c81e3b4b5bb.
Reused binaries: reference 5ed4b6e543f32b82, record 9b96f1427503da27.
