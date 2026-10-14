# Grid ladder rung G1b — round 1 receipt (gridladder_r22)

Status: **STOPPED_FOR_RECORD**. Deck, card, acquisition script and
preregistration are committed; NEMO has not run. The operator runs
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1b_orca2vec/run.sh --run`.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round1.md
(47d2ee50f; two citations and one count corrected after review in e5089d91c,
before any record). Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round22/.

## Take-aways

1. **G1b = G1's deck with ORCA2's `&namdyn_adv`, nothing else**:
   ln_dynadv_vec = T and nn_dynkeg = 0 (ORCA2_OMIP_L4/EXP00/namelist_cfg:346-347),
   ln_dynadv_OFF = F. Every neighbouring choice on this path is already
   ORCA2's resolved value in G1's deck. No DECISION_NEEDED.
2. **NEMO derives more than the vorticity from the switch**: total vorticity
   (np_CRV), the KE gradient, and velocity-form time stepping in the stage
   updates, the stage-3 solve and the external mode (prereg section 2).
3. **The card is G1's with seven fields moved, measured**: four
   momentum selectors take the ORCA2 identity's own vector-form values, and
   three G1-certified flux-form-only options are not selected, because
   NEMO's vector branch never runs them
   (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:533`).
4. **Prediction: NEMO completes 100 steps.** G1's record peaks at 0.018 m/s,
   500 times under stp_ctl's 10 m/s; the vector terms move U by 1.1e-6 m/s at
   step 1 on the card. The script still handles an early stop.
5. **Predicted first non-bit statement: the velocity-form external update**
   at kt = 1 (PLAUSIBLE; the model's velocity update was never compared with
   NEMO statement by statement on any ladder).
6. One model docstring is wrong as worded: `nemo_flux_form_update_active`
   says NEMO's RK3 external mode "exposes no velocity-form arm"; it does,
   for vector decks. Not edited (reported).

## 1. Survey (read off the compiled source, R4REF build)

| reader | G1 (np_LIN_dyn) | G1b (np_VEC_c2) |
|---|---|---|
| dyn_vor's triad (dynvor.f90:744-748) | planetary only | planetary + relative, unmasked |
| stp_2D stage-0 RHS (stp2d.f90:144-163) | HPG + VOR | + wzv (velocity) + KEG + ZAD |
| stages 2-3 (stprk3_stg.f90:287-291, :331-332) | no advection | wzv on velocities, KEG + ZAD |
| stage 1-2 update (stprk3_stg.f90:366-370) | thickness-weighted | velocity |
| stage-3 update (dynzdf.f90:138-141) | thickness-weighted | velocity |
| external update (dynspg_ts.f90:533-545) | flux form | velocity form |
| external sum (dynspg_ts.f90:627-629) | transports | velocities |

With one wet level dyn_zad's loop is empty (dynzad.f90:105) and its bottom
statement adds a zero. At kt = 1 stp_2D reads the rest state, so the
relative vorticity and KE gradient first act at kt = 1 stage 2.

## 2. Deck, card, script

- Deck: nemo_testcase_g1b_orca2vec/namelist_cfg. Test pins: the only
  assignments differing from G1's are the three `&namdyn_adv` lines, and
  they equal ORCA2's.
- Card: `build_orca2g1b_vec_card` / case `ORCA2G1b-vec`, one rung table
  `_ORCA2G_DYNADV_VEC`, one switch-set reader. G1's option line is
  byte-identical; G1b's three deselections are one override after it.
  Each rung refuses the other rung's deck (tested). The reader refuses
  nn_dynkeg on a non-vector deck and a moved or missing nn_dynkeg on G1b.
- Searched before building: the G1 builder and its rung tables, the ORCA2
  identity's vector selectors, the VORTEX_VEC arm, the G3 acquisition
  pattern. Nothing new was written for the physics.
- Card smoke, 4 steps, both cards finite (card_smoke.log; taken before the
  inert stage-1 option was deselected, which the model ignores under
  vector form): G1b minus G1, max |du| 1.1e-6, 4.9e-6, 3.1e-6, 6.5e-6 m/s;
  max |dssh| 1.8e-6, 2.4e-5, 5.9e-5, 1.9e-5 m.
- Script: no build; G1's pinned round-4 binaries (R4REF, P5) and G1's
  domain file (SHA-256 73801c6f...). Smoke 2 steps, 100-step reference,
  then ten record twins with stride-10 restart lists: a restart at every
  step 1..N. N = 100, or the largest even step before an stp_ctl stop
  (a stop before kt 10 is refused). Admission: G1's check_records.py
  unchanged, writer field set, twin frames byte-identical, every run's
  namelist_ref equal to NEMO's, NEMO's log stating the vector form.
- Preflight: PREFLIGHT_OK at the final commit. Two deck plants (rn_shlat,
  nn_dynkeg) each refuse, exit 68 (preflight_plants.log).

## 3. Predictions

Frozen in the preregistration, section 5: S-P1 no stop in 100 steps; S-P2
|U|,|V| < 0.05 m/s; V-P1 kt-1 U differs from G1's record by > 1e-8;
F-P1 first non-bit row at kt = 1; F-P2 the external update
(dynspg_ts.f90:533-545); F-P3 fallbacks in order: velocity sum, stage
update, np_CRV vorticity; F-P4 no relative vorticity or KEG before kt 1
stage 2.

## 4. ORCA2 pointer (D122)

Every G1b statement runs on ORCA2's own build under its vector deck:
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:666` (external update),
`:799` (velocity sum), `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467`
(stage update), `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:138`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762` (np_CRV). (Unmapped in the
citation gate: that build carries ORCA2-lane writer lines.)

## 5. Review and tests

Single review (codex): **findings**, four CONFIRMED, all acted on
(codex_review_r22.log, last lines):
- "rk3_stage1_rhs_nemo_literal=True remains selected on G1b ... flux-form-only
  ... ignored by the vector branch": verified (the model's stage-1 arm takes
  the whole RHS under vector form regardless). Deselected on G1b; test now
  pins seven fields.
- "Admission never verifies each run's copied namelist_ref": fixed (cmp per
  run, and the reference's copy hashed).
- "Preregistration line 17 incorrectly calls G1's nn_dynkeg unread": fixed
  (read and range-checked, inert).
- "The stp_ctl citation ... is incomplete": fixed (predicate :243-250, stop
  :310).

Focused tests, synchronous:
- G1b card file: 11 passed. With the G1b deselection reverted (plant), 2
  fail (test_plant_revert.log).
- G1 card + G1b card + G2/G3/G4 rung files + G1 tools: 142 passed, 1 failed
  (tests_final.log). The failure is pre-existing: the G1 stand-in test
  plants a check named `uniform_T`, renamed `initial_T` by the G3 round
  (4c7c6cc8b).
- G1 ladder file, the eight "only the G1 card selects" tests: 4 failed,
  4 passed, identically at the base commit 0650720cf (field-position pins
  that later rounds' appended fields broke). The rest of the ladder file
  was not re-run in full (over 10 minutes); a partial earlier run is
  tests_g1_ladder.log.
- Citation gate on the preregistration: PASS, 20 citations, 0 unmapped,
  whole map clean (five shifted card anchors re-anchored with the lane's
  tool); the shifted-citation plant exits 1.

Compliance without a gate: search-before-build (section 2); one variable
(the deck test and the seven-field card test); non-vacuity (plants above).
Dual review: NOT done (rule 6, single review). The two deck plants used a
throwaway commit with hooks skipped in a temporary worktree, removed after.

## 6. Choices

ASKED: the round brief's items.
UNASKED, each offered for revert:
1. Three flux-form-only options deselected on G1b (NEMO's vector branch;
   two are refused by the model with vector form, one is ignored).
2. Restarts at every step as ten record twins (about 13 GB).
3. The early-stop rule for N.
4. The G1 ladder is not yet pointed at G1b (next round, with the record).

## OPEN

1. Next round: admit the record; point the ladder at `ORCA2G1b-vec`; walk
   kt = 1 in compiled order (substeps, then stage frames); land cited
   statements G1b-card-selected.
2. Whether G1's other certified stage options (written-order update, stage
   ratios) reach anything under the velocity-form update was not audited
   option by option; the walk will show it.

UNVERIFIED:
- that NEMO completes 100 steps (prediction only);
- the ORCA2 build citations in section 4 (read, not gated);
- that the stand-in and position-pin failures have no other cause than the
  renames named (base-commit comparison for the latter only).

SHA-256 prefixes (round22/): card_smoke.log=f091a5d4aeac49bf,
citation_prereg.json=2fa9b30e102e1b96, citation_prereg_plant.json=be2e40aac219c69b,
codex_review_r22.log=eac357d0b65c7e37, preflight.log=4c37d61095d00157,
preflight_plants.log=6afd414008d5c3c8, tests_final.log=7c907f45f4775eff,
tests_rungs.log=fb3fbcb03aaa6c13, tests_only_g1_card_head_vs_base.log=594be73fb084383d,
test_plant_revert.log=30e5388ba511d1bb, tests_g1_ladder.log=e0483bbc0a6bef1b.
