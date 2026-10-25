# ORCA2 round 334 receipt — CI ratchet cleanup (DECISION 166)

Round: 334 (Claude). Parent `072e4aa1b1c8`, fix commit `11943cc1a076`.
Disposition: **LANDED-pending** (operator review + push gate + DINO month gate).
codex review: operator.

## Ratchet suite, before

Ten ratchet files, fresh detached worktrees, `JAX_ENABLE_X64=1`, CPU.
Lane tip `072e4aa1b1c8`: 8 failed, 12275 passed. Main `b090be7b18d9`: 4 failed, 11412 passed.

Measured sets (logs `ratchet_tip.log`, `ratchet_main.log`). Main and the fixed
lane each fail four tests, but not the same four: three are shared, the lane
alone fails the slab-land dispatch test, and main alone fails private imports.

| failing test ID | lane tip | main | measured set | origin |
|---|---|---|---|---|
| hardcoded constants `[tests/unit/test_jra55_do.py]` | F | F | both | main code |
| inline coeffs `[packages/land/legoesm/land/restart.py]` | F | F | both | main code |
| param specs `[.../physics/shortwave_penetration.py]` | F | F | both | main code (lane lists 3 of main's 4 bad names) |
| no private cross imports | pass | F | **main only** | main code after the fork |
| dispatch `test_no_dispatch_guard_removed` (slab land `step_land`) | F | pass | **lane only** | main commit before the fork; main's baseline follow-up after it (below) |
| inline coeffs `[.../physics/geothermal.py]` | F | pass | lane only | lane (round 320 import), fixed |
| inline coeffs `[.../lateral_mixing/mle.py]` | F | pass | lane only | lane (round 323 import), fixed |
| inline coeffs `[.../lateral_mixing/mle_latlon_cgrid.py]` | F | pass | lane only | lane (round 323 import), fixed |
| param specs `[.../lateral_mixing/mle.py]` | F | pass | lane only | lane (round 323 import), fixed |

The dispatch failure is measured lane-only, but no lane commit caused it.
`git log -S'Unknown bulk_scheme'` and `git log cffa2ab79..HEAD` on
`slab_land.py` find no lane commit touching the file. The only lane commit
touching the dispatch test (round 309, `c7b3a91a95`) adds one unrelated ocean
entry. Main commit `642481601` (2026-10-03, an ancestor of the fork point
`cffa2ab79`) deliberately moved step_land's body, including its unknown-scheme
raises, into `step_land_with_diagnostics`. Main commit `f9cde047d`
(2026-10-05, not in the lane) then moved the baseline entry to follow the code;
its message says "the guards are unchanged". The lane carries the code move
without the baseline follow-up.

## Fixes (lane items only)

- Geothermal NEMO path: the inline `1.0e-3` is NEMO's mW to W conversion
  (`trabbc.F90:179`). It is now the module constant `_MW_TO_W`, with its
  provenance written next to it.
- MLE NEMO coefficient: the inline `3.141592653589793 / 180.0` is NEMO's `rad`
  (`phycst.F90:26`). It is now the module constant `_NEMO_RAD`.
- Lat-lon MLE NEMO branch: the code re-derived `5.0 / 21.0` (NEMO
  `tramle.F90:51`) even though `mle.py` already defines it. It now imports that
  constant, made public as `R5_21`. One test assertion follows the rename.
- MLE spec: the spec listed the boolean `nemo_association` under `excluded`.
  Only float fields can carry a spec, so the entry was removed. No float field
  changed tier or bounds.

No ratchet was loosened, no baseline entry was added, and no escape comment was
used. No configuration value or default changed.

## Ratchet suite, after

Fix tree: 4 failed, 12279 passed. Three are the shared rows; the fourth is
the lane-only dispatch row. Main's private-import failure does not occur on
the lane. Every item a lane commit introduced is green.

## Bit-identity (parent `072e4aa1b1c8` vs candidate `11943cc1a076`)

The shipped rung-10 card selects both touched NEMO paths: L8 geothermal and
L5 MLE with `nemo_association=True`.

- **D122 ladders, all seven card/label pairs, CONFIRMED.** These are rung 0,
  OMT-1..4, and rung 10 under both the independent and given-NEMO-entry labels.
  They were run with round 330's measurement recipe (round-329 gate `--measure
  recordice`, same deck roots), on clean worktrees at the two commits. On every
  pair the ladder row objects are byte-identical between parent and candidate,
  and no other key differs apart from the worktree stamp. Rows: rung 0 200,
  OMT-1 160, OMT-2..4 200 each, rung 10 200 per label.
  The registered D122 classifier refuses same-arm pairs ("control is not
  admitted"). Byte-identical rows make every one of its counts zero anyway.
- **DINO faithful grid, three production steps, CONFIRMED.** The round-330
  probe in `values` mode gives 66/66 leaf-step SHA-256 equal on both trees, and
  the whole file is byte-identical (`5f977ac14552…`). It also equals round 332
  resume 9's hashes.
- **DINO from-rest month gate (GPU), CONFIRMED.** Day-30 wet 3-D T rms is
  2.056821683e-03 K on both trees (bar 2.244317642e-03 K), PASS.
- **Touched routines on fixed inputs, CONFIRMED.** The routines are the MLE
  coefficient (both branches, three latitudes), the MLE vertical structure, and
  the NEMO geothermal tendency. Output hashes and float hex values are
  identical on both trees.

Other callers of the touched code are the MPAS MLE and the generic lat-lon MLE.
They reach it only through the renamed `R5_21`, which has the same Python float
(`0x1.e79e79e79e79ep-3`), and the routine-level check covers that.

Evidence: `orca2_rounds/round334/` (`d122/`, `dino/`, `unit/`, `ratchet_*.log`).

## Gates

- Ratchet suite: lane items green (above).
- Default gate battery (land.sh set) on the candidate: green. The fast files
  passed in one run. The three slow files had 12 passed under `-n 12`, because
  they exceed 10 minutes serially.
- Focused MLE/geothermal unit tests: 114 passed, 5 failed. The same 5 fail on
  the parent: two from SPEC_MODULES registry drift, and three because ORCA1
  geothermal input files are missing from the archive. These are pre-existing
  and out of scope.
- Per-receipt citation gate: PASS, 3 citations, 0 failures, 0 unmapped (three NEMO
  entries added to `CITATION_MAP`); default citation-gate test 17 passed.

## CHOICES

- Removing `nemo_association` from the MLE spec instead of adding a float spec
  for it — ASKED (task text; the rule says bools are not spec-eligible; the task orders
  honest specs).
- Making `_R5_21` public instead of copying it — ASKED (task text: no
  re-derivation; repo rule: promote, never import privately).
- Not fixing the inherited slab-land dispatch baseline — ASKED (task text says
  fix only lane-introduced items).
- Running the DINO month gate myself on idle GPUs 0/1 — UNASKED. The task named
  the month gate, but the standing note says CPU only. Offered for re-run by
  the operator.
- Running the three slow default-gate files with `-n 12` — UNASKED (to fit the
  10-minute foreground limit; it does not change the results).
- Re-exporting `R5_21` from the package — ASKED (resume task, item 2).
- Not restoring a guard in `step_land` — ASKED (resume task: stop if the move
  was deliberate and citable).
- Starting the ladders with nohup, recording their pids, and waiting on them in
  the foreground — UNASKED (each rung-10 arm takes about 14 minutes, longer than
  the foreground limit).

## Resume (review HOLD fixes)

1. Restoring a raising guard in `step_land` itself: STOPPED and reported.
   The move was main's and deliberate, with a citable reason (`642481601`,
   followed up by `f9cde047d`). No lane commit moved it, so there is no lane
   commit to undo. Decision put to the operator.
2. `R5_21` re-exported from `lateral_mixing/__init__.py`, as the promotion
   rule requires (commit `6729d95bd`). Pure addition: the package attribute is
   the same float object (`0x1.e79e79e79e79ep-3`).
3. Classification table above corrected to the measured lane-vs-main sets.

Re-measured on `6729d95bd`:
- Ratchet suite (`ratchet_resume.log`): 4 failed, 12279 passed. Three are the
  shared main rows; the fourth is the lane-only dispatch row, open on item 1.
- Bit-identity: the touched-routine probe (`unit/fn_hash.py`) output is
  byte-identical to the parent and candidate runs (`unit/resume.json`). No
  executed code changed, so the D122 ladders and DINO checks above stand.
- Default gate (land.sh set) on the committed tree: green. Fast files plus
  the MLE unit tests: 154 passed under `-n 12`; the only 2 failures were the
  citation-gate stamp refusing the then-dirty tree, and that file then gave
  17 passed on the clean tree. Slow files, split to fit the 10-minute limit:
  parallel receipt gate 2 passed, prognostic barotropic state 8 passed, stage
  face-mask rank 2 passed (`-n 2`, 564 s). Logs `default_gate_resume_*.log`.
- Per-receipt citation gate (`receipt_citations_resume.log`): PASS,
  3 citations, 0 failures, 0 unmapped.

## Resume 2 (DECISION 169: main's dispatch-baseline follow-up)

- Cherry-picked main's `f9cde047d` as `6b3683e06` (`-x`). Main = `origin`'s
  `origin/main` `b090be7b18d9`. Diff stat: 1 file, `tests/test_dispatch_hardening.py`,
  4 insertions, 1 deletion. Test baseline only; no model code changed.
- Conflict resolution: the pick's hunk context carried a second main baseline
  entry, the multilayer-land snowpack dispatcher, added on main by a different
  commit (`7b15e013e1`) that is not in the lane. Dropped it, so the pick carries
  only `f9cde047d`'s own change. Checked: the added and removed lines are
  identical to `f9cde047d`'s.
- Ratchet suite, same ten files (`ratchet_r2_head.log`, `ratchet_r2_main.log`):
  before (`6729d95bd`) the lane had 4 failed; after (`6b3683e06`) it has 3 failed, 12280 passed;
  main `b090be7b18d9` has 4 failed, 11412 passed. Lane-only failures: none (`comm`).
  The three lane failures all fail on main too: hardcoded constants
  `[tests/unit/test_jra55_do.py]`, inline coeffs
  `[packages/land/legoesm/land/restart.py]`, param specs
  `[.../physics/shortwave_penetration.py]`. Main also fails private imports.
- Bit-identity controls: unchanged. The change is test-only (diff stat
  above), so no executed model code moved. The D122, DINO and routine
  results above stand.
- Default gate (land.sh set) on `6b3683e06`: fast files 140 passed; parallel
  receipt gate plus prognostic barotropic state 10 passed (`-n 2`); stage
  face-mask rank: SLOW2_RESULT.
- CHOICES: dropping the multilayer baseline entry from the conflict — ASKED
  (DECISION 169 names `f9cde047d` only). Running the slow files with `-n 2` —
  UNASKED, same as resume 1.

## OPEN

- Main-side failures (jra55 literal, land restart coefficients, shortwave spec,
  private imports) belong to main, not this lane.
