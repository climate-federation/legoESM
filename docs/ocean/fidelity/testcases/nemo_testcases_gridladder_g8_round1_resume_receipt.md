# Grid ladder rung G8 — round 1 resume receipt (gridladder_r45, scoring)

Status: **LANDED. G8 CERTIFIED.** Every field is bit-identical to NEMO's
admitted record under both labels at kt 1..10: surface rows and all-level
rows, 0 unequal cells, INDEPENDENT and GIVEN-NEMO-ENTRY.  Worst row per label:
none above 0 (max abs 0.0; Decision 127 flushed subnormals 0 in every row).

- Record: `gridladder_rounds/round45/oracle_g8_r45`, run by the operator at
  `50a49037f` exactly as committed: `G8_PREFLIGHT_OK`,
  `PASS_G5_ADMISSION unequal_rows=None`,
  `G8_ACQUISITION_PASS steps=1..10 restarts=2..10 reference_completed=100`.
- Preregistration: `PREREG_nemo_testcases_gridladder_g8_round1.md` (`2fcbad6c0`).
- Landing commits: `81b28f995` (three dyn_ldf_lev statements, measured on this
  record by the first resume session), `2b7571443` (stage-3 F ratio, this
  session), `8e0edc1a7` (declared gaps; G2 ladder test's config tail).
- Final ladders stamped clean at `2b7571443` (packages tree `55d58e824dbf`).
  `8e0edc1a7` changes only the card's declared-gap list, a test and a
  citation line, not a model code path.
- Evidence: `gridladder_rounds/round45/score/` (`head/` = scores at
  `81b28f995`, `r3f/` = the fix measured, `final/` = clean stamp).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

## 1. Score at `81b28f995` (clean re-score of the first resume's dirty-tree numbers)

The first resume session scored on a dirty tree.  Re-scored clean at
`81b28f995`, the numbers reproduce it exactly.

| label | kt 1..5 | kt 6 | kt 7 | kt 8 | kt 9 | kt 10 |
|---|---|---|---|---|---|---|
| independent, worst normalised | 0 | 3.04e-16 ssh | **1.21e-14 v** | 1.14e-14 u | 1.91e-14 v | 2.91e-14 v |
| given-NEMO-entry, worst normalised | 0 | 1.6e-18 v | 6.3e-18 u | 3.2e-17 ssh | 9.6e-19 v | 1.85e-16 ssh |

Given-NEMO-entry was at the bar at every kt; independent was DEBT from kt 7
(all-level u 1.07e-14 at kt 7).  The seed was not at the surface:

- kt 3: one V cell at level 10 differs (all-level row; surface rows exact);
- kt 4-5: 2-25 deep U/V cells, each one or two ulps;
- kt 6: 688 sea-level cells; kt 7: over the bar.

## 2. The first non-bit statement

The kt-3 step under the stages arm (NEMO's restart of kt 2 as entry): stage 1
and stage 2 bit-identical at every level; stage 3 one V cell, card interior
`(j, i, k) = (125, 98, 9)`, NEMO `1.18264551468145289e-02`, card
`1.18264551468145306e-02` (one ulp, 1.7e-18).

Stage 3 adds `dyn_ldf(kstp, Kbb, Kmm)` after HPG and vorticity
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:493`, SUBROUTINE `stp_RK3_stg`
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:78`).  The card's order is the
same.  Its curl term multiplies `e3f` by `1 + r3f*fe3mask`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:123`, SUBROUTINE
`dynldf_lev_lap` at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:64`).  At
stage 3 NEMO's `r3f` is the mean of the two stored F ratios,
`r3f = r1_2 * ( r3fb + r3fa )`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:245`; the same line in the G5
round-31 record build, `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:245`,
SUBROUTINE at `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:78`,
call at `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:493`).
The card rebuilt the ratio from the stage sea level: equal as real numbers,
one ulp apart on rare points.

Searched for an existing implementation: `r3f`, `r3fb`, `r3f_native`,
`rk3_stage_r3f_nemo_literal`.  Found it: the vorticity path already selects
NEMO's interpolated ratio (`rk3_stage_r3f_nemo_literal`, true on every G
card) and the shared F-thickness builder takes it as `r3f_native`.  Only the
stage-3 diffusion did not pass it.  Nothing new was written: the stage-3
diffusion now hands the existing ratio to the existing builder.

## 3. What changed (`2b7571443`)

- New `nemo_ldf_stage_r3f` (default False).  When True, the stage-3 lateral
  diffusion's F thickness reads the existing stage ratio.
- Only the G8 card selects it (rung table `_ORCA2G_LDF`), with a selector row
  `stage-3 ldf r3f`.  Caller grep: the field is read in one place and set in
  one place.
- Other cards: G7's traced step jaxpr, addresses masked, is identical at
  `50a49037f`, `81b28f995` and this tree (`121e48a9c43cb193`).  G1-G7 and the
  full ORCA2 card leave the field False.

## 4. Score with the statement

Measured on the dirty tree (`r3f/`), then re-stamped clean at `2b7571443`
(`final/`).  Both give the same result:

| label | steps | rows | bit-identical | flushed subnormal | all-level unequal cells |
|---|---|---|---|---|---|
| independent | kt 1..10 | 70 | 70 | 0 | 0 |
| given-NEMO-entry (entries 1,3,5,7,9 + two-step) | kt 1..10 | 70 | 70 | 0 | 0 |

kt 1 has surface rows only: the record has no kt-1 restart for the all-level
comparison.  kt 3 stage 3 is bit-identical at every level.

Score SHA-256 (first 16): `indep_kt1_5.json` `ba86cc3288580841`,
`indep_kt6_10.json` `4325eba19614b643`, `given_e135.json` `7e5edc4b8a597f77`,
`given_e79.json` `e471e035ad08a2c0`.

Non-vacuity: without the statement (`81b28f995`) the kt-3 stage-3 cell is one
ulp off and the independent label is DEBT at kt 7..10 (section 1); with it,
0 unequal cells.

## 5. Gates (CPU, fp64, `JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH)

| gate | result |
|---|---|
| default gate, part 1 (citation gate, private-import ratchet, TKE, recipe, freshwater) | 132 passed |
| default gate, face-mask rank (`-n 2`) | 2 passed |
| default gate, prognostic barotropic state | 8 passed |
| G1 card files (`-n 16`) | 117 passed |
| G2-G8 card files (`-n 6`) | 125 passed, 1 failed (the G2 ladder test's config-tail positions, stale since `81b28f995`) |
| after the fix: G8 file + G2 ladder file | 24 passed, 9 deselected |
| plant: G8 card drops `nemo_ldf_stage_r3f` | G8 file 2 failed, restored, `git status` clean |
| plants: scorer `--plant score` / `--plant level` | both exit 1, DEBT |
| this receipt's citation gate | PASS, 0 failures (section 7) |

The face-mask and G1 runs first hit the 10-minute cap under machine load
(load average 43); they were re-run with workers.  No test failed in the
capped runs.

Re-anchor: the receipt citation map was red at `81b28f995` (63 code
citations shifted).  The existing re-anchor tool moved them mechanically
(80 lines, map and one GYRE round-8 receipt).  One cited range contains the
edit and grows 64 -> 66 lines.  A second pass moved 1 recipe citation.

## 6. Reports only

**Longer record: worth defining.**  G8 is the first rung where NEMO's own
writer-free run completed all 100 steps (G5 and G7 stopped at kt 11-13 on
NEMO's stability check).  A 100-step record with restarts every 10 steps
would test the independent label's last-bit growth well past kt 10.  Cost: the
card steps at ~80 s on CPU, so about 2.2 h per independent 100-step chain.

**Decision 122.**  The same compiled lines run on the full ORCA2 card
(ORCA2_OMIP_L4 line 245 feeding line 123 of the diffusion routine).  The four
G8 statements are import candidates.  The ORCA2 card is unchanged.

**The ORCA2 lane's stage-1 V-transport mask note** is not exercised: G8 has
no tracer advection.

## 7. Receipt citation gate

Run on this receipt before the final commit:
`nemo_testcase_receipt_citation_gate.py --receipt <this file> --from-heading
"# Grid ladder rung G8 — round 1 resume receipt (gridladder_r45, scoring)"`.
Result in `gridladder_rounds/round45/score/final/receipt_citation_gate.json`.

- At `4c616df63`: `PASS`, 8 citations, 0 failures, 0 unmapped, 0 map entries
  failing the audit.
- Its shift plant on the stage-3 F-ratio line exits 1.
- Default citation-gate test + private-import ratchet + G8 file: 36 passed
  (`final/final_gates.log`).

## Choices

ASKED: G8's definition and record (the brief); the bar and labels (D127).

UNASKED, offered for revert:
- the statement's switch defaults False and only G8 selects it (the campaign's
  no-silent-default rule); moving the default, or importing it to ORCA2, is a
  Decision 122 question;
- G8's declared gaps shrink to G5's inherited input route, as G7's did;
- the G2 ladder test's config-tail positions were updated for the four G8 fields;
- the re-anchor touched a GYRE round-8 receipt's line numbers (mechanical).

UNVERIFIED: the independent label past kt 10 (no record); the full ORCA2
card's response to the four statements.
