# Grid ladder rung G4 — round 3 receipt (gridladder_r19)

Status: **STOPPED_FOR_RECORD.** Acquisition written and preflighted; NEMO
not run (operator only). Nothing landed in the model; forcing NOT wired
into the card (Decision 123 pending). G4 is NOT certified.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g4_round3.md
(committed with the tools, before any run).
Acquisition: scripts/validate/ocean_fidelity/testcases/nemo_testcase_g4_orca2sbc/run_r19.sh
(evidence will land in gridladder_rounds/round19/oracle_g4_r19).
Decisions 120, 121, 123 pending; nothing here depends on their answers.

## Take-aways

1. **One new record build, P6** = round 4's record build (P5) plus four
   additive patches; every added line is a writer call, a USE or a comment.
   The writer is the existing G1 step writer (new frames, no new module).
2. **What it records, kt = 1..10:** every surface field the ocean reads,
   the two composites the barotropic solver reads, and on bulk steps the
   NCAR bulk's entry, all five iterations and its ocean result.
3. **Passivity is a hard gate:** admission refuses unless every word of
   round 17's admitted record (stage frames, substep streams, restarts of
   twins a, b, c and the smoke) is reproduced byte for byte.
4. **Citation housekeeping done:** round 18's 12 citations are mapped and
   pass the gate; two were off by one and are corrected in that receipt.

## 1. The record, field by field

Program points (NEMO 5.0.2, P5 compiled source; P6 adds only writer lines):

| frame | where written | fields | NEMO readers |
|---|---|---|---|
| `s`, every kt | after `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stprk3.f90:142` | utau, vtau (T), utauU, vtauV (U, V), taum, wndm, qns, qsr, emp, sfx, rnf, erp, qrp, rnf_tsc T and S, fr_i, sst/sss/ssu/ssv/ssh/e3t/frq means | listed below |
| `c`, every kt | inside dyn_spg_ts after its copies | ssh_frc, zu_frc, zv_frc | the solver |
| `n`, kt odd | turb_ncar entry | inputs, bounded wind, virtual-temperature difference, initial stability flag, neutral and initial coefficients, t_zu, q_zu | — |
| `A`..`E`, kt odd | each NCAR iteration | 1/L, clamped zeta, psi_m, floored U_N10, CdN, Cd, their roots, psi_h term, Cd ratio, stability flag, ChN, CeN, Ch, Ce | — |
| `k`, kt odd | ocean bulk after the stress | SST (potential, K), air T and q, ssq, wind, slp, Cd/Ch/Ce, theta_zu, q_zu, bulk wind, air density, taum, sensible, latent, evaporation, utau, vtau | — |

Read sites the `s` frame stands for (nothing writes these fields between
`sbc` and the readers; the one candidate, `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:269-270`,
runs only with `ln_traqsr = F`, and the deck sets T):

- stress into the barotropic forcing `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:198-199`;
  into the surface velocity `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynzdf.f90:338` and `:506`;
- emp, rnf into the sea-level forcing `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:242-245`
  (that array is freed at `:270`, so the composites are recorded where
  the solver copies them, `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynspg_ts.f90:274-276`);
- tracers: emp at stages 1-2 `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:285`, qns and sfx
  at stage 3 `:307-308`, runoff content `:323`;
- solar `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/traqsr.f90:371-373` (RGB with chlorophyll data).

Bulk cadence `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk.f90:554` (kt = 1, 3, 5, 7, 9);
iterations `cfgs/SHARED/namelist_ref:241` (deck silent); the round-18
operand candidates sit inside the recorded iteration,
`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:174-178` (1/L and the clamp),
`:207` (U_N10 floor), `:223` (stability flag). With 10 m for both
reference heights the height-shift block is skipped, so t_zu, q_zu stay at
their entry values (recorded at entry).

## 2. The acquisition (run_r19.sh)

- Preflight (run here, committed tree): shipped source + round-4 patches
  reproduce P5's MY_SRC byte for byte; the new patches apply unfuzzed and
  add only writer lines; round 17's decks for smoke and twins a, b, c
  equal the rendered deck; record kt = 1..10 from round 17's reference
  (NEMO stopped at kt = 11 in stp_ctl, same rule as round 17).
  `PREFLIGHT_OK`.
- Build: makenemo, new target ORCA2G1_OMIP_L1_P6 (refuses if it exists).
  One build judge, run after the build AND on `--admit-existing`: P6's
  MY_SRC is the committed sources byte for byte; the compiled file set is
  P5's and each file is P5's plus writer lines; the evidence's binary hash
  is the built binary's.
- Runs: smoke (2 steps, checked before twins), twins a/b/c (10 steps,
  round 17's restart lists), inputs copied from round 17's admitted run
  directories and compared byte for byte.
- Admission: 9 plants must fire (5 existing, 4 new forcing plants); record
  a vs b (with restart and substep checks), a vs c, the smoke; then
  passivity against round 17. `--admit-existing` re-admits completed runs.
- Syntax: the four patched sources pass `gfortran -fsyntax-only` against
  P5's module files; the unpatched copies preprocess to P5's compiled
  sources exactly.
- Dry runs here on round 17's data: the edited checker re-admits round 17
  with an identical JSON; the passivity block passes round 17 against
  itself and refuses a one-bit change (exit 1).

## 3. Checker forcing mode (check_records.py)

Off by default (round 17's admission reproduces unchanged). On: inner-domain
arrays allowed; forcing frames required exactly per step, each holding
exactly the fields the patches write (a test ties the two); judged finite and
twin-equal on the consumed support (halo words counted); stage frames stay
byte-exact. Non-vacuity: three mutations (stage frames not compared,
inner arrays always allowed, one iteration short) each turn tests red, and
so does disabling the field-schema check.

## 4. Predictions

R3-P1 held for the preflight half (the compiled-source half needs the
build). R3-P2..P5 need the record.

## 5. Citation housekeeping

Round 18's receipt, from its section 1: **PASS**, 12 citations, 0
failures, 0 unmapped; planted shift exits 1. Corrections made in that
receipt: the drag block is lines 1237-1248 (was 1238-1247), the bulk
formula call 788-792 (was 787-791).

## 6. Tests (synchronous)

Focused battery (G1 tools, G2, G2 ladder, G3, G4): 80 passed, 9
deselected (68 before + 12 new), after the review fixes. Citation gate
tests: 17 passed.

## 7. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r19.log).
Each checked before acting:
- HIGH, the smoke would be refused: CONFIRMED, the runoff depth is
  allocated with a one-cell halo (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcrnf.f90:151`),
  a shape the checker rejects. Fixed by not recording it: it is set once at
  start-up, not per-step forcing. Every other field's shape re-read: full
  or inner domain.
- HIGH, no field schema (dropping the solar write would pass): CONFIRMED.
  Fixed: per-frame schema in the checker, a test that the patches write
  exactly it, a fourth plant.
- HIGH, `--admit-existing` trusted an evidence-local binary hash:
  CONFIRMED. Fixed by the build judge above.
- MEDIUM, compiled-source gate saw only P5's file names: CONFIRMED. Fixed
  (file sets must be equal).
- LOW, the iteration count is namelist_ref's, not the deck's: kept. The
  deck must stay round 17's (changing it changes the program); the check now
  reads the namelist_ref copy the runs actually use, and the record's five
  iteration frames are its own witness.

## 8. Choices

ASKED (round brief): new record build; reuse round 17's reference and
inputs; twins as G3/G4; no card wiring.
UNASKED, each offered for revert: (1) recording through the G1 step writer
rather than a new module; (2) the `s` frame is ungated (every kt), so the
P6 binary is fit for runs up to 50 steps only under this checker;
(3) carrying round 4's substep writer into P6 (kt 1, 2; ~0.4 GB per twin)
to keep P6 = P5 + writer lines.
(4) not recording the runoff depth (time-invariant, review finding 1).

No gate exists for: dual review (single, lane rule 6), claim review before
code, non-vacuity (shown above).

UNVERIFIED:
- that P6 compiles and runs (no makenemo here);
- the instrument identities R3-P5 (need the record).

## OPEN (next round)

Admit the record (`run_r19.sh --admit-existing` after the operator's
`--run`), check R3-P5, then under Decision 123's answer: either feed the
`s`/`c` frames to the card (option 1) or walk the NCAR iterations for the
first non-bit operand (option 2).

## 9. Review verdict and evidence hashes

Citation gate on this receipt (from section 1): **PASS**, 18 citations, 0
failures, 0 unmapped, map audit clean; planted shift exits 1. Round 18's
receipt: PASS, 12 citations.

SHA-256 prefixes, round19/: preflight.log=c677b539646abe14,
citation_receipt.json=96e135ba4bd2ef71, citation_plant_r19.log=232ccb74c47dead4,
citation_round18_receipt.json=b97f28f850ad049c, citation_plant.log=f364a7f1a1dbb7e4,
codex_review_r19.log=b0f03de941316936, tests_focused_final.log=857961725c58ea85,
tests_citation_gate.log=09982498970dbb8e, r17_readmit.json=6d05567bc024e364
(= round 17's admission, re-admitted by the edited checker).
